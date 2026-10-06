"""Refinement of an EclipseNET from trajectory data (Section 4 of the paper).

The weights of the network only enter the dynamics through the times at which
the eclipse factor switches. Each switch, at the time t* where the event
function e vanishes, makes the sensitivity S = dx/dtheta of the state jump
(with - and + the two sides of the switch and f the dynamics)::

    dt*/dtheta = -(de/dtheta + de/dx S-) / (de/dt along the trajectory)
    S+ = S- + (f- - f+) dt*/dtheta

Between two switches the dynamics do not depend on the weights, and S is
carried forward by the state transition matrix. This gives the same
sensitivities as the variational equations with respect to the weights (Eq. 7
of the paper), at the cost of propagating 36 variational equations instead of
6 for each weight.

The network has to be smooth (a SIREN, or a feed-forward network with tanh):
with ReLU the times of the switches found by the integrator are only
approximate, and so are the sensitivities.
"""
import copy
import math

import numpy as np
import torch

from .dynamics import sun_direction_at
from .utils import cart2spherical


def event_function(model, position, t, sun_az, sun_el, sun_rate, kind="silhouette", r_gate=None):
    """
    Event functions of :class:`eclipsenets.EclipseNetPropagator`, in torch.

    Args:
        model: the EclipseNET, in double precision.
        position (torch.Tensor of shape (3,)): position of the spacecraft.
        t (torch.Tensor): time.
        sun_az, sun_el (float): azimuth at t = 0 and elevation of the Sun.
        sun_rate (float): rate of change of the azimuth of the Sun.
        kind (str): "silhouette" (the network), "terminator" or "gate".
        r_gate (float): gate distance, only for kind="gate".

    Returns:
        torch.Tensor: value of the event function.
    """
    az = sun_az + sun_rate * t
    cos_az, sin_az = torch.cos(az), torch.sin(az)
    cos_el, sin_el = math.cos(sun_el), math.sin(sun_el)
    x, y, z = position
    if kind == "terminator":
        return sin_el * (cos_az * x + sin_az * y) + cos_el * z
    X = -sin_az * x + cos_az * y
    Y = cos_el * (cos_az * x + sin_az * y) - sin_el * z
    if kind == "gate":
        return X * X + Y * Y - r_gate**2
    constant = torch.ones_like(az)
    return model(torch.stack([cos_az, sin_az, cos_el * constant, sin_el * constant, X, Y]))[0]


def state_sensitivities(model, propagation, sun_direction, srp, sun_rate, r_gate):
    """
    Sensitivities of the propagated states with respect to the weights of the network.

    Args:
        model: the EclipseNET, in double precision.
        propagation (Propagation): result of a variational propagator.
        sun_direction, srp, sun_rate: the ones used for the propagation.
        r_gate (float): the gate distance of the propagator.

    Returns:
        np.array of shape (K, 6, m): derivatives of the states at the K times
        of the propagation with respect to the m weights, ordered as in
        `model.parameters()`.
    """
    if propagation.stms is None:
        raise ValueError("The propagation has no state transition matrices: "
                         "build the propagator with variational=True")
    _, sun_az, sun_el = cart2spherical(sun_direction)[0, :]
    parameters = list(model.parameters())
    n_weights = sum(p.numel() for p in parameters)
    times = propagation.times

    sensitivities = np.zeros((len(times), 6, n_weights))
    S = np.zeros((6, n_weights))  # sensitivity right after the latest switch
    latest_stm = np.eye(6)        # state transition matrix at the latest switch
    k = 0
    for event in propagation.events:
        while k < len(times) and times[k] <= event.time:
            sensitivities[k] = propagation.stms[k] @ np.linalg.solve(latest_stm, S)
            k += 1
        S = event.stm @ np.linalg.solve(latest_stm, S)  # right before the switch

        # Derivatives of the event function that triggered the switch
        position = torch.tensor(event.state[:3], dtype=torch.float64, requires_grad=True)
        t = torch.tensor(event.time, dtype=torch.float64, requires_grad=True)
        e = event_function(model, position, t, sun_az, sun_el, sun_rate, event.kind, r_gate)
        of_network = event.kind == "silhouette"
        e_r, e_t, *e_theta = torch.autograd.grad(e, [position, t] + (parameters if of_network else []))
        e_r = e_r.numpy()
        e_theta = torch.cat([g.ravel() for g in e_theta]).numpy() if of_network else np.zeros(n_weights)
        e_dot = e_r @ event.state[3:] + e_t.item()

        dt_dtheta = -(e_theta + e_r @ S[:3]) / e_dot
        # The dynamics lose the radiation pressure when entering an eclipse
        # and get it back when exiting.
        sun = sun_direction_at(event.time, sun_direction, sun_rate)
        jump = np.zeros(6)
        jump[3:] = -srp * sun if event.entering else srp * sun
        S = S + np.outer(jump, dt_dtheta)
        latest_stm = event.stm

    sensitivities[k:] = propagation.stms[k:] @ np.linalg.solve(latest_stm, S)
    return sensitivities


def refine(propagator, model, ic, times, observations, sun_direction, srp=1e-3, sun_rate=None,
           n_iterations=100, learning_rate=1e-4, verbose=True):
    """
    Refines an EclipseNET so that the trajectory it produces matches observed states.

    The loss (Eq. 4 of the paper) is the sum of the squared differences between
    the propagated and the observed states. It is minimized with Adam, using
    the gradient given by :func:`state_sensitivities`.

    Args:
        propagator (EclipseNetPropagator): built with learnable=True and variational=True.
        model: the EclipseNET to refine. It is converted to double precision
            and modified in place: at the end it holds the weights with the
            lowest loss found.
        ic (np.array of shape (6,)): initial position and velocity.
        times (np.array of shape (K,)): initial time followed by the times of
            the observations.
        observations (np.array of shape (K - 1, 6)): observed states.
        sun_direction, srp, sun_rate: see :meth:`EclipseNetPropagator.propagate`.
        n_iterations (int): number of optimization steps.
        learning_rate (float): learning rate of Adam.
        verbose (bool): whether to print the loss at each iteration.

    Returns:
        list with the loss at each iteration, starting with the one of the initial weights.
    """
    sun_rate = propagator.body.omega if sun_rate is None else sun_rate
    observations = np.asarray(observations, dtype=float).reshape(len(times) - 1, 6)
    model.double()
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    losses = []
    best_state = None
    for iteration in range(n_iterations + 1):
        propagator.set_weights(model)
        propagation = propagator.propagate(ic, times, sun_direction, srp=srp, sun_rate=sun_rate)
        residuals = propagation.states[1:] - observations
        losses.append(float(np.sum(residuals**2)))
        if losses[-1] == min(losses):
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"Iteration {iteration}/{n_iterations} - Loss: {losses[-1]:.6e}")
        if iteration == n_iterations:
            break

        sensitivities = state_sensitivities(model, propagation, sun_direction, srp, sun_rate, propagator.r_gate)
        gradient = torch.from_numpy(2 * np.einsum("ki,kim->m", residuals, sensitivities[1:]))
        start = 0
        for parameter in parameters:
            parameter.grad = gradient[start:start + parameter.numel()].reshape(parameter.shape)
            start += parameter.numel()
        optimizer.step()

    model.load_state_dict(best_state)
    propagator.set_weights(model)
    return losses
