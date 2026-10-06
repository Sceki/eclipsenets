"""Motion of a spacecraft around a rotating small body under its gravity and
the solar radiation pressure, which switches off during eclipses.

The equations of motion (Eq. 1 of the paper) are written in the frame that
rotates with the body, in the non-dimensional units of
:class:`eclipsenets.Body`. The direction of the Sun keeps its elevation and
changes its azimuth at a constant rate::

    s(t) = Rz(sun_rate * t) s(0)

The paper uses sun_rate = omega, the default here. 

Two propagators are available: :class:`EclipseNetPropagator` detects the
eclipses with an EclipseNET used as event function of a Taylor integrator,
while :func:`propagate_ray_tracing` switches the radiation pressure with a
ray-tracing test at every evaluation of the dynamics, and is the ground truth.

An EclipseNET only knows the silhouette of the body, not its depth: together
with the test on the side of the body the spacecraft is on, it is exact for a
spacecraft outside the sphere that contains the body.
"""
import math
import warnings
from dataclasses import dataclass
from typing import List, Optional

import heyoka as hk
import numpy as np
from scipy.integrate import solve_ivp

from .nn import FFNN, heyoka_model, weights_and_biases_heyoka
from .ray_tracing import is_eclipsed
from .utils import cart2spherical

# Runtime parameters of the Taylor integrator.
PAR_LIGHT = 0     # eclipse factor: 1 in sunlight, 0 in eclipse
PAR_BEHIND = 1    # 1 if the spacecraft is on the night side of the terminator plane
PAR_NEAR = 2      # 1 if the spacecraft is closer than r_gate to the Sun-body line
PAR_OMEGA = 3     # angular velocity of the body
PAR_SUN_RATE = 4  # rate of change of the azimuth of the Sun
PAR_SUN_AZ = 5    # azimuth of the Sun at t = 0
PAR_SUN_EL = 6    # elevation of the Sun
PAR_SRP = 7       # magnitude of the acceleration due to the solar radiation pressure
PAR_GATE = 8      # r_gate squared
N_PARS = 9        # the weights of the network follow, when they are runtime parameters


def initial_state(radius: float, inclination: float, omega: float) -> np.ndarray:
    """
    State, in the rotating frame, of a spacecraft that starts on the -x axis
    with the velocity of a circular Keplerian orbit.

    The orbit is retrograde: the spacecraft moves against the rotation of the
    body, on a plane tilted by `inclination` with respect to the equator.

    Args:
        radius: distance from the center of the body, in units of L.
        inclination: angle between the orbital plane and the equator, in radians.
        omega: angular velocity of the body.

    Returns:
        np.array of shape (6,): position and velocity.
    """
    v = 1 / np.sqrt(radius)
    return np.array([-radius, 0, 0, 0, v * np.cos(inclination) + omega * radius, v * np.sin(inclination)])


def sun_direction_at(t, sun_direction, sun_rate):
    """
    Direction of the Sun at time t, given the one at t = 0.

    Args:
        t (float or np.array of shape (N,)): time.
        sun_direction (np.array of shape (3,)): direction of the Sun at t = 0.
        sun_rate (float): rate of change of the azimuth of the Sun.

    Returns:
        np.array of shape (3,) or (N, 3): unit vector(s).
    """
    _, az, el = cart2spherical(sun_direction)[0, :]
    az = az + sun_rate * np.asarray(t, dtype=float)
    return np.stack([np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el) * np.ones_like(az)], axis=-1)


def _misses_sphere(position, sun_direction, radius):
    """True if the ray from `position` towards the Sun misses the sphere of the given radius."""
    along = np.dot(position, sun_direction)
    distance2 = np.dot(position, position)
    return distance2 - along**2 > radius**2 or (along > 0 and distance2 > radius**2)


def propagate_ray_tracing(body, ic, tgrid, sun_direction, srp=1e-3, sun_rate=None,
                          rtol=1e-13, atol=1e-13):
    """
    Ground-truth propagation: the radiation pressure is switched off whenever
    the ray from the spacecraft to the Sun hits the mesh of the body
    (Möller-Trumbore test). The discontinuity is left to the step size control
    of the integrator (DOP853), hence the tight default tolerances.

    Args:
        body (Body): the body.
        ic (np.array of shape (6,)): initial position and velocity.
        tgrid (np.array of shape (K,)): times at which the state is returned,
            starting from the initial time.
        sun_direction (np.array of shape (3,)): direction of the Sun at t = 0.
        srp (float): magnitude of the acceleration due to the radiation pressure.
        sun_rate (float): rate of change of the azimuth of the Sun (body.omega if None).
        rtol, atol (float): tolerances of the integrator.

    Returns:
        np.array of shape (K, 6): the states at the times of tgrid.
    """
    sun_rate = body.omega if sun_rate is None else sun_rate
    omega = body.omega
    radius = body.radius
    v0, v1, v2 = body.triangle_vertices
    mascon_points, mascon_masses = body.mascon_points, body.mascon_masses

    def dynamics(t, state):
        r, v = state[:3], state[3:]
        diff = r - mascon_points
        acceleration = -(mascon_masses / np.linalg.norm(diff, axis=1) ** 3) @ diff
        acceleration[0] += omega**2 * r[0] + 2 * omega * v[1]
        acceleration[1] += omega**2 * r[1] - 2 * omega * v[0]
        sun = sun_direction_at(t, sun_direction, sun_rate)
        # the ray-tracing test is only needed if the ray can reach the body at all
        if _misses_sphere(r, sun, radius) or not is_eclipsed(r, sun, v0, v1, v2):
            acceleration -= srp * sun
        return np.concatenate([v, acceleration])

    tgrid = np.asarray(tgrid, dtype=float)
    solution = solve_ivp(dynamics, (tgrid[0], tgrid[-1]), ic, method="DOP853", t_eval=tgrid,
                         rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(f"The integration failed: {solution.message}")
    return solution.y.T


def compile_eclipsenet(model):
    """
    Compiles an EclipseNET with heyoka, for fast evaluations in double precision.

    Args:
        model: a SIREN or FFNN model with six inputs.

    Returns:
        heyoka compiled function of [cos(az), sin(az), cos(el), sin(el), X, Y].
    """
    inputs = hk.make_vars("cos_az", "sin_az", "cos_el", "sin_el", "x_coord", "y_coord")
    return hk.cfunc(heyoka_model(model, inputs), inputs)


def is_eclipsed_net(net, position, sun_direction, r_gate) -> bool:
    """
    Eclipse test based on an EclipseNET.

    Args:
        net: compiled network, from :func:`compile_eclipsenet`.
        position (np.array of shape (3,)): position of the spacecraft.
        sun_direction (np.array of shape (3,)): direction of the Sun (unit vector).
        r_gate (float): distance from the Sun-body line beyond which there is
            no eclipse and the network, never trained there, is not evaluated.

    Returns:
        True if the spacecraft is in eclipse.
    """
    # Plain floats: for a single point numpy would cost more than the network.
    x, y, z = position
    sx, sy, sz = sun_direction
    if x * sx + y * sy + z * sz > 0:
        return False
    sin_el = math.hypot(sx, sy)
    cos_az, sin_az = (sx / sin_el, sy / sin_el) if sin_el > 0 else (1., 0.)
    X = -sin_az * x + cos_az * y
    Y = sz * (cos_az * x + sin_az * y) - sin_el * z
    if X * X + Y * Y > r_gate**2:
        return False
    return bool(net([cos_az, sin_az, sz, sin_el, X, Y])[0] < 0)


@dataclass
class EclipseEvent:
    """A change of the eclipse factor during a propagation."""
    time: float
    state: np.ndarray           # position and velocity, shape (6,)
    entering: bool              # True when entering the eclipse, False when exiting
    # Event function that triggered the change: "silhouette" is the regular
    # case, the zero of the network. The eclipse factor can also change when
    # the network gets switched on or off, while it is negative, at the
    # "terminator" plane or at the "gate" distance from the Sun-body line.
    kind: str
    # Solution of the variational equations from the initial time, shape
    # (6, 6): the state transition matrix of the dynamics with the switches
    # frozen in time. The jumps due to the switches moving are not included.
    stm: Optional[np.ndarray]


@dataclass
class Propagation:
    """Result of :meth:`EclipseNetPropagator.propagate`."""
    times: np.ndarray           # (K,)
    states: np.ndarray          # (K, 6)
    events: List[EclipseEvent]
    starts_in_eclipse: bool
    stms: Optional[np.ndarray]  # (K, 6, 6), for a variational propagator (see EclipseEvent.stm)

    @property
    def entries(self) -> List[EclipseEvent]:
        return [event for event in self.events if event.entering]

    @property
    def exits(self) -> List[EclipseEvent]:
        return [event for event in self.events if not event.entering]

    @property
    def eclipses(self) -> np.ndarray:
        """Start and end times of the eclipses, shape (E, 2)."""
        times = [event.time for event in self.events]
        if self.starts_in_eclipse:
            times.insert(0, self.times[0])
        if len(times) % 2 == 1:  # the propagation ends in eclipse
            times.append(self.times[-1])
        return np.array(times).reshape(-1, 2)


class EclipseNetPropagator:
    """
    Taylor integrator of the spacecraft dynamics, in which an EclipseNET is the
    event function that detects the eclipses.

    The network is the event function only where it is meaningful: behind the
    body with respect to the Sun, and closer than `r_gate` to the Sun-body
    line. Two more events switch it on and off at these boundaries.

    The event detection of a Taylor integrator relies on the event function
    being smooth. This holds for SIREN and tanh networks; with a ReLU network
    the times of the eclipses are only approximate.

    Args:
        body (Body): the body, which provides the mascon model and the angular velocity.
        model: a SIREN or FFNN model with six inputs.
        r_gate (float): distance from the Sun-body line within which the
            network is evaluated. It has to contain the silhouette, without
            reaching too far from it: a network is not trained there, and may
            well be negative. The default is 1.25 times the radius of the body.
        learnable (bool): if True, the weights of the network are runtime
            parameters of the integrator, which :meth:`set_weights` can change
            without building a new integrator.
        variational (bool): if True, the state transition matrix is propagated
            along with the state.
        **kwargs: passed to heyoka's `taylor_adaptive`.
    """

    def __init__(self, body, model, r_gate=None, learnable=False, variational=False, **kwargs):
        self.body = body
        self.r_gate = 1.25 * body.radius if r_gate is None else r_gate
        self.learnable = learnable
        self.variational = variational
        self._events: List[EclipseEvent] = []
        if isinstance(model, FFNN) and model.activation == "relu":
            warnings.warn("A ReLU network is not smooth: the times of the eclipses found by the "
                          "Taylor integrator are only approximate.")

        x, y, z, vx, vy, vz = hk.make_vars("x", "y", "z", "vx", "vy", "vz")
        omega = hk.par[PAR_OMEGA]

        # Sun direction and coordinates on the plane orthogonal to it
        az = hk.par[PAR_SUN_AZ] + hk.par[PAR_SUN_RATE] * hk.time
        cos_az, sin_az = hk.cos(az), hk.sin(az)
        cos_el, sin_el = hk.cos(hk.par[PAR_SUN_EL]), hk.sin(hk.par[PAR_SUN_EL])
        sun = [sin_el * cos_az, sin_el * sin_az, cos_el]
        X = -sin_az * x + cos_az * y
        Y = cos_el * (cos_az * x + sin_az * y) - sin_el * z

        # Gravity of the mascons
        gravity = [[], [], []]
        for point, mass in zip(body.mascon_points, body.mascon_masses):
            diff = [x - point[0], y - point[1], z - point[2]]
            r_m3 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) ** hk.expression(-3. / 2)
            for i in range(3):
                gravity[i].append(-mass * r_m3 * diff[i])

        # Equations of motion
        srp = hk.par[PAR_SRP] * hk.par[PAR_LIGHT]
        dvx = hk.sum(gravity[0]) + omega**2 * x + 2 * omega * vy - srp * sun[0]
        dvy = hk.sum(gravity[1]) + omega**2 * y - 2 * omega * vx - srp * sun[1]
        dvz = hk.sum(gravity[2]) - srp * sun[2]
        sys = [(x, vx), (y, vy), (z, vz), (vx, dvx), (vy, dvy), (vz, dvz)]
        if variational:
            sys = hk.var_ode_sys(sys, hk.var_args.vars, order=1)

        # Event functions
        self.n_weights = len(weights_and_biases_heyoka(model))
        weights = [hk.par[N_PARS + i] for i in range(self.n_weights)] if learnable else None
        net = heyoka_model(model, [cos_az, sin_az, cos_el, sin_el, X, Y], weights)[0]
        terminator = sun[0] * x + sun[1] * y + sun[2] * z
        gate = X * X + Y * Y - hk.par[PAR_GATE]
        armed = hk.par[PAR_BEHIND] * hk.par[PAR_NEAR]
        self._event_functions = hk.cfunc([net, terminator, gate], vars=[x, y, z])

        # The callbacks are closures rather than methods: heyoka copies the
        # callbacks, and the copy of a method would write to a copy of self.
        def on_silhouette(ta, d_sgn):
            eclipse = d_sgn < 0
            # With a network that is not smooth the same crossing can be
            # reported twice: only a change of the eclipse factor is a switch.
            if eclipse != (ta.pars[PAR_LIGHT] == 0.):
                self._switch(ta, eclipse, kind="silhouette")
            return True

        def on_terminator(ta, d_sgn):
            ta.pars[PAR_BEHIND] = float(d_sgn < 0)
            self._refresh(ta, kind="terminator")
            return True

        def on_gate(ta, d_sgn):
            ta.pars[PAR_NEAR] = float(d_sgn < 0)
            self._refresh(ta, kind="gate")
            return True

        self.ta = hk.taylor_adaptive(
            sys, [0.] * 6,
            t_events=[
                # where the network is switched off, the event function is the constant 1
                hk.t_event(armed * net + (1. - armed), callback=on_silhouette, cooldown=1e-8),
                hk.t_event(terminator, callback=on_terminator, cooldown=1e-8),
                hk.t_event(gate, callback=on_gate, cooldown=1e-8),
            ],
            compact_mode=True, **kwargs)
        if learnable:
            self.set_weights(model)

    def set_weights(self, model):
        """Sets the weights of the network to the ones of `model` (for a learnable propagator)."""
        if not self.learnable:
            raise RuntimeError("The weights can only be changed in a propagator built with learnable=True")
        self.ta.pars[N_PARS:] = weights_and_biases_heyoka(model)

    def _stm(self, ta):
        return ta.state[6:].reshape(6, 6).copy() if self.variational else None

    def _switch(self, ta, eclipse, kind):
        """Sets the eclipse factor and records the event."""
        ta.pars[PAR_LIGHT] = 0. if eclipse else 1.
        self._events.append(EclipseEvent(ta.time, ta.state[:6].copy(), eclipse, kind, self._stm(ta)))

    def _in_eclipse(self, ta):
        """Eclipse state according to the network, where it is switched on."""
        if ta.pars[PAR_BEHIND] * ta.pars[PAR_NEAR] == 0.:
            return False
        return self._event_functions(ta.state[:3], pars=ta.pars, time=ta.time)[0] < 0

    def _refresh(self, ta, kind):
        """Keeps the eclipse factor consistent when the network is switched on or off."""
        eclipse = self._in_eclipse(ta)
        if eclipse != (ta.pars[PAR_LIGHT] == 0.):
            self._switch(ta, eclipse, kind)

    def propagate(self, ic, tgrid, sun_direction, srp=1e-3, sun_rate=None) -> Propagation:
        """
        Propagates the motion of the spacecraft.

        Args:
            ic (np.array of shape (6,)): initial position and velocity.
            tgrid (np.array of shape (K,)): increasing times at which the
                state is returned, starting from the initial time.
            sun_direction (np.array of shape (3,)): direction of the Sun at t = 0.
            srp (float): magnitude of the acceleration due to the radiation pressure.
            sun_rate (float): rate of change of the azimuth of the Sun
                (the angular velocity of the body if None).

        Returns:
            Propagation
        """
        ta = self.ta
        tgrid = np.asarray(tgrid, dtype=float)
        if tgrid[-1] < tgrid[0]:
            # the callbacks tell entries from exits assuming that time increases
            raise ValueError("tgrid must be increasing: backward propagations are not supported")
        _, az, el = cart2spherical(sun_direction)[0, :]

        ta.time = tgrid[0]
        ta.state[:6] = ic
        if self.variational:
            ta.state[6:] = np.eye(6).ravel()
        ta.pars[PAR_OMEGA] = self.body.omega
        ta.pars[PAR_SUN_RATE] = self.body.omega if sun_rate is None else sun_rate
        ta.pars[PAR_SUN_AZ] = az
        ta.pars[PAR_SUN_EL] = el
        ta.pars[PAR_SRP] = srp
        ta.pars[PAR_GATE] = self.r_gate**2
        ta.reset_cooldowns()

        # Initial values of the flags
        _, terminator, gate = self._event_functions(ta.state[:3], pars=ta.pars, time=ta.time)
        ta.pars[PAR_BEHIND] = float(terminator < 0)
        ta.pars[PAR_NEAR] = float(gate < 0)
        starts_in_eclipse = bool(self._in_eclipse(ta))
        ta.pars[PAR_LIGHT] = 0. if starts_in_eclipse else 1.

        self._events.clear()
        outcome, *_, states = ta.propagate_grid(tgrid)
        if outcome != hk.taylor_outcome.time_limit:
            raise RuntimeError(f"The integration stopped early: {outcome}")
        stms = states[:, 6:].reshape(-1, 6, 6) if self.variational else None
        return Propagation(tgrid, states[:, :6], list(self._events), starts_in_eclipse, stms)
