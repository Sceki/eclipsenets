import dataclasses

import numpy as np
import pytest
import torch

import eclipsenets as en

SUN = np.array([-1., -1., 0.]) / np.sqrt(2)
SRP = 1e-3


@pytest.fixture(scope="module")
def body():
    # A handful of mascons keeps the integrators quick to build; the shape is the real one.
    body = en.load_body("Bennu")
    keep = np.random.default_rng(0).choice(len(body.mascon_masses), 50, replace=False)
    masses = body.mascon_masses[keep]
    return dataclasses.replace(body, mascon_points=body.mascon_points[keep], mascon_masses=masses / masses.sum())


@pytest.fixture(scope="module")
def model():
    return en.load_pretrained("Bennu", "siren").double()


def test_bodies():
    for name in en.BODIES:
        body = en.load_body(name)
        assert np.ptp(body.mesh_points[:, 0]) == pytest.approx(1.0)  # L is the extent along x
        assert body.mascon_masses.sum() == pytest.approx(1.0)
        # the mascons sit inside the mesh, in the same units
        assert np.all(body.mascon_points.min(axis=0) > body.mesh_points.min(axis=0))
        assert np.all(body.mascon_points.max(axis=0) < body.mesh_points.max(axis=0))
        assert np.ptp(body.mascon_points[:, 0]) > 0.95
        low_poly = en.load_body(name, low_poly_mesh=True, low_poly_mascons=False)
        assert low_poly.L == body.L and len(low_poly.mesh_points) < len(body.mesh_points)
        assert len(low_poly.mascon_masses) > len(body.mascon_masses)
    assert en.load_body("67P").name == en.load_body("chu-ger").name == "Churyumov-Gerasimenko"
    # Table 1 of the paper
    assert en.load_body("Bennu").L == pytest.approx(563.4, abs=0.1)
    assert en.load_body("Eros").L == pytest.approx(32662.2, abs=0.1)


def test_sun_direction():
    assert en.sun_direction_at(0., 3 * SUN, 0.7) == pytest.approx(SUN)
    quarter_turn = en.sun_direction_at(np.pi / 2, [1., 0., 1.], 1.0)
    assert quarter_turn == pytest.approx(np.array([0., 1., 1.]) / np.sqrt(2))
    assert en.sun_direction_at(np.linspace(0, 1, 5), SUN, 0.7).shape == (5, 3)


def test_eclipse_tests_agree(body, model):
    v0, v1, v2 = body.triangle_vertices
    net = en.compile_eclipsenet(model)
    rng = np.random.default_rng(0)
    agree = 0
    eclipsed = 0
    for _ in range(500):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        # points behind the body, around the edge of the shadow
        position = en.plane_to_3D(*rng.uniform(-0.6, 0.6, 2), direction, -1.5)[0]
        truth = en.is_eclipsed(position, direction, v0, v1, v2)
        eclipsed += truth
        agree += truth == en.is_eclipsed_net(net, position, direction, 1.25 * body.radius)
        # in front of the body there is no eclipse
        assert not en.is_eclipsed_net(net, position + 3 * direction, direction, 1.25 * body.radius)
        assert not en.is_eclipsed(position + 3 * direction, direction, v0, v1, v2)
    assert 100 < eclipsed < 400
    assert agree >= 490


def test_propagation_against_ray_tracing(body, model):
    ic = en.initial_state(1.5, np.radians(11.), body.omega)
    tgrid = np.linspace(0, 2.0, 41)
    propagator = en.EclipseNetPropagator(body, model)
    propagation = propagator.propagate(ic, tgrid, SUN, srp=SRP)

    # two eclipses, entered and left through the silhouette
    assert [event.entering for event in propagation.events] == [True, False, True, False]
    assert {event.kind for event in propagation.events} == {"silhouette"}
    assert not propagation.starts_in_eclipse
    assert propagation.eclipses.shape == (2, 2)

    # the eclipse factor found by ray tracing flips at the events
    v0, v1, v2 = body.triangle_vertices
    for event in propagation.events:
        sun = en.sun_direction_at(event.time, SUN, body.omega)
        velocity = event.state[3:]
        before, after = (en.is_eclipsed(event.state[:3] + dt * velocity, sun, v0, v1, v2) for dt in (-0.01, 0.01))
        assert (before, after) == (not event.entering, event.entering)

    truth = en.propagate_ray_tracing(body, ic, tgrid, SUN, srp=SRP)
    assert np.abs(propagation.states - truth).max() < 1e-5
    # the eclipses matter: without them the trajectory is further off
    no_eclipses = en.EclipseNetPropagator(body, model, r_gate=0.).propagate(ic, tgrid, SUN, srp=SRP)
    assert len(no_eclipses.events) == 0
    assert np.abs(no_eclipses.states - truth).max() > 10 * np.abs(propagation.states - truth).max()

    with pytest.raises(ValueError, match="increasing"):
        propagator.propagate(ic, tgrid[::-1], SUN, srp=SRP)

    # a propagator can be reused, also from a state that is already in eclipse
    again = propagator.propagate(ic, tgrid, SUN, srp=SRP)
    assert np.array_equal(again.states, propagation.states)
    start, end = propagation.eclipses[0]
    middle = propagator.propagate(ic, [0., (start + end) / 2], SUN, srp=SRP).states[-1]
    resumed = propagator.propagate(middle, np.linspace((start + end) / 2, 2.0, 5), SUN, srp=SRP)
    assert resumed.starts_in_eclipse
    assert resumed.eclipses[0] == pytest.approx([(start + end) / 2, end], abs=1e-9)
    assert resumed.states[-1] == pytest.approx(propagation.states[-1], abs=1e-9)


def test_relu_network(body):
    # A ReLU network is not smooth, and the integrator can report a crossing
    # of its zero twice: the eclipses must still come out as entry-exit pairs.
    with pytest.warns(UserWarning, match="not smooth"):
        propagator = en.EclipseNetPropagator(body, en.load_pretrained("Bennu", "ffnn"))
    sun = np.array([1., 0.1, 0.25]) / np.linalg.norm([1., 0.1, 0.25])
    ic = en.initial_state(1.4, np.radians(25.), body.omega)
    propagation = propagator.propagate(ic, np.linspace(0, 0.2, 2001), sun, srp=2e-3, sun_rate=-body.omega)
    assert propagation.starts_in_eclipse
    assert [event.entering for event in propagation.events] == [False]
    assert propagation.eclipses.shape == (1, 2)


def test_sun_rate(body, model):
    # with a Sun fixed in the rotating frame, a spacecraft behind the body is still in eclipse shortly after
    propagator = en.EclipseNetPropagator(body, model)
    ic = en.initial_state(1.5, 0., body.omega)
    propagation = propagator.propagate(ic, [0., 0.05], [1., 0., 0.], srp=SRP, sun_rate=0.)
    assert propagation.starts_in_eclipse and len(propagation.events) == 0
    assert propagator.propagate(ic, [0., 0.05], [-1., 0., 0.], srp=SRP, sun_rate=0.).eclipses.size == 0


def test_state_sensitivities(body):
    model = en.load_pretrained("67P", "ffnn", epoch=0).double()  # smooth activations
    ic = en.initial_state(1.5, np.radians(11.), body.omega)
    times = np.array([0., 1.0, 2.0])
    propagator = en.EclipseNetPropagator(body, model, learnable=True, variational=True)
    propagation = propagator.propagate(ic, times, SUN, srp=SRP)
    assert len(propagation.events) == 4
    assert propagation.stms[0] == pytest.approx(np.eye(6))

    sensitivities = en.state_sensitivities(model, propagation, SUN, SRP, body.omega, propagator.r_gate)
    assert sensitivities.shape == (3, 6, en.count_parameters(model))
    assert np.all(sensitivities[0] == 0)

    # directional derivatives against central finite differences
    parameters = list(model.parameters())
    theta = torch.nn.utils.parameters_to_vector(parameters).detach().clone()
    direction = torch.tensor(np.random.default_rng(0).normal(size=len(theta)))
    direction /= direction.norm()
    states = []
    for step in (1e-4, -1e-4):
        torch.nn.utils.vector_to_parameters(theta + step * direction, parameters)
        propagator.set_weights(model)
        states.append(propagator.propagate(ic, times, SUN, srp=SRP).states)
    finite_differences = (states[0] - states[1]) / 2e-4
    analytic = sensitivities @ direction.numpy()
    assert np.abs(analytic[1:]).max() > 1e-6
    assert analytic == pytest.approx(finite_differences, rel=1e-4, abs=1e-9)


def test_refine(body):
    # trajectory data from the trained network, starting point from the first epoch
    target = en.load_pretrained("67P", "ffnn").double()
    model = en.load_pretrained("67P", "ffnn", epoch=0)
    ic = en.initial_state(1.5, np.radians(11.), body.omega)
    times = np.array([0., 2.0])
    observations = en.EclipseNetPropagator(body, target).propagate(ic, times, SUN, srp=SRP).states[1:]

    propagator = en.EclipseNetPropagator(body, model, learnable=True, variational=True)
    losses = en.refine(propagator, model, ic, times, observations, SUN, srp=SRP, n_iterations=15,
                       learning_rate=1e-4, verbose=False)
    assert len(losses) == 16
    assert min(losses) < 0.5 * losses[0]
    # the model is left with the best weights found
    final = propagator.propagate(ic, times, SUN, srp=SRP).states[1:]
    assert np.sum((final - observations) ** 2) == pytest.approx(min(losses))
