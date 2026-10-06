import numpy as np
import pytest
from shapely.geometry import Polygon

import eclipsenets as en

# Unit cube with outward-facing triangles
CUBE_POINTS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
CUBE_TRIANGLES = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                           [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
Z = np.array([0., 0., 1.])


def test_ray_triangle_intersection():
    v0, v1, v2 = np.array([0., 0, 1]), np.array([1., 0, 1]), np.array([0., 1, 1])
    origins = np.array([[0.2, 0.2, 0.], [5., 5., 0.], [0.2, 0.2, 2.]])  # hit, miss, triangle behind

    assert en.is_intersecting_v1(origins, Z, v0, v1, v2).tolist() == [True, False, False]
    for origin, expected in zip(origins, [True, False, False]):
        assert en.is_intersecting_v2(origin, Z, v0[None], v1[None], v2[None]).tolist() == [expected]

    t, point = en.ray_triangle_intersect(origins[0], Z, v0, v1, v2)
    assert t == pytest.approx(1.0)
    assert point == pytest.approx([0.2, 0.2, 1.0])
    with pytest.raises(ValueError):
        en.ray_triangle_intersect(origins[1], Z, v0, v1, v2)


def test_ray_triangle_intersection_does_not_depend_on_units():
    triangles = CUBE_POINTS[CUBE_TRIANGLES]
    origin = np.array([0.3, 0.4, -1.0])
    for scale in [1e-6, 1.0, 1e6]:
        v0, v1, v2 = (scale * triangles[:, i] for i in range(3))
        assert en.is_intersecting_v2(scale * origin, Z, v0, v1, v2).sum() == 2


def test_point_in_mesh():
    # the vertical through the second point runs along the diagonal shared by two triangles
    points = np.array([[0.3, 0.4, 0.5], [0.5, 0.5, 0.5], [2., 2., 2.], [0.3, 0.4, -0.5]])
    assert en.is_outside_v1(points, CUBE_POINTS, CUBE_TRIANGLES).tolist() == [False, False, True, True]
    assert [en.is_outside_v2(p, CUBE_POINTS, CUBE_TRIANGLES) for p in points] == [False, False, True, True]


def test_eclipse_function_sums_the_chords_inside_the_mesh():
    # two unit cubes, two units apart along the ray: the ray crosses 1 + 1 of material
    points = np.vstack([CUBE_POINTS, CUBE_POINTS + 3 * Z])
    triangles = np.vstack([CUBE_TRIANGLES, CUBE_TRIANGLES + 8])
    assert en.eclipse_function(points, triangles, np.array([0.3, 0.4, -1.0]), Z) == pytest.approx(-2.0)
    # the same through the diagonals shared by the triangles of the faces
    assert en.eclipse_function(points, triangles, np.array([0.5, 0.5, -1.0]), Z) == pytest.approx(-2.0)
    # a ray that misses: distance from the projected vertices
    assert en.eclipse_function(points, triangles, np.array([1.5, 0.0, -1.0]), Z) == pytest.approx(0.5)
    # a ray that starts inside the mesh
    assert np.isnan(en.eclipse_function(points, triangles, np.array([0.3, 0.4, 0.5]), Z))


@pytest.mark.parametrize("direction", [[0.3, -0.5, 0.6], [0., 0., 1.], [0., 0., -1.], [1., 0., 0.]])
def test_projection_on_the_plane(direction):
    direction = np.array(direction) / np.linalg.norm(direction)
    U, V = en.plane_basis(direction)
    assert np.allclose([U @ V, U @ direction, V @ direction, U @ U, V @ V], [0, 0, 0, 1, 1])

    points = np.random.default_rng(0).normal(size=(5, 3))
    X, Y = en.project_on_plane(points, direction)
    assert np.allclose(en.project_to_2d(points, direction), np.stack([X, Y], axis=1))
    # back to 3D: the component along the direction is the offset
    for offset in [0.7, np.float64(0.7), 1]:
        back = en.plane_to_3D(X, Y, direction, offset)
        assert np.allclose(back, points - np.outer(points @ direction - offset, direction))
        assert np.allclose(en.plane_to_3D(X, Y, 3 * direction, offset), back)  # no normalization needed


def test_network_inputs():
    direction = en.sph2cart(0.4, 1.1, rad=True)
    positions = np.random.default_rng(0).normal(size=(4, 3))
    inputs = en.network_inputs(positions, 3 * direction)  # the direction needs no normalization
    assert inputs.shape == (4, 6)
    assert np.allclose(inputs[:, :4], [np.cos(0.4), np.sin(0.4), np.cos(1.1), np.sin(1.1)])
    assert np.allclose(inputs[:, 4:], np.stack(en.project_on_plane(positions, direction), axis=1))
    assert np.allclose(en.network_inputs(positions[0], direction), inputs[0])


def test_fibonacci_sphere():
    points, az, el = en.fibonacci_sphere(200)
    assert np.allclose(np.linalg.norm(points, axis=1), 1)
    assert np.allclose(points, en.sph2cart(az, el, rad=True).T)
    assert np.abs(points.mean(axis=0)).max() < 0.01  # evenly spread
    assert en.fibonacci_sphere(1)[0].shape == (1, 3)


def test_sample_points():
    centers = np.array([[0., 0.], [5., 5.]])
    samples = en.sample_points(centers, r=0.1, num_samples_per_point=50, seed=0)
    assert samples.shape == (100, 2)
    assert np.all(np.linalg.norm(samples[:50] - centers[0], axis=1) <= 0.1)
    assert np.all(np.linalg.norm(samples[50:] - centers[1], axis=1) <= 0.1)
    assert np.array_equal(samples, en.sample_points(centers, r=0.1, num_samples_per_point=50, seed=0))

    # the global random state is left alone
    np.random.seed(0)
    expected = np.random.rand()
    np.random.seed(0)
    en.sample_points(centers)
    assert np.random.rand() == expected


def test_boundary_ordering_heuristics():
    angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    shuffled = np.random.default_rng(0).permutation(circle)
    for ordering in [en.sort_points_along_boundary, en.nearest_neighbor]:
        ordered = np.array(ordering(shuffled))
        assert sorted(map(tuple, ordered)) == sorted(map(tuple, circle))  # a permutation of the points
        assert Polygon(ordered).area == pytest.approx(Polygon(circle).area)


def test_alpha_shape():
    side = np.arange(6.)
    points = np.array([(x, y) for x in side for y in side])
    border = en.alpha_shape(points, alpha=1.0)
    assert len(border) == 20  # the perimeter of the 5x5 square, in unit edges
    assert all(points[[i, j]].min() == 0 or points[[i, j]].max() == 5 for i, j in border)
    # the edges form one closed loop: each point is the start of one edge and the end of another
    assert sorted(i for i, _ in border) == sorted(j for _, j in border)
    assert len(en.alpha_shape(points, alpha=1.0, only_outer=False)) == 3 * 25 + 10


@pytest.mark.parametrize("cull_backfaces", [False, True])
def test_silhouette_of_a_cube(cull_backfaces):
    square = en.silhouette_polygon(CUBE_POINTS, CUBE_TRIANGLES, Z, cull_backfaces=cull_backfaces)
    assert square.area == pytest.approx(1.0)
    assert len(en.silhouette_border(square)) >= 4
    on_border = en.sample_border(square, 200, seed=0)
    assert en.compute_eclipse_function(on_border, surface=square) == pytest.approx(0, abs=1e-12)
    # spread over the four sides
    assert all(np.sum(np.isclose(on_border[:, axis], value)) > 25 for axis in (0, 1) for value in square.bounds[axis::2])
    hexagon = en.silhouette_polygon(CUBE_POINTS, CUBE_TRIANGLES, np.ones(3), cull_backfaces=cull_backfaces)
    assert hexagon.area == pytest.approx(np.sqrt(3))


def test_silhouette_of_a_body():
    points, triangles, _ = en.load_mesh("67P", low_poly=True)
    direction = en.fibonacci_sphere(10)[0][3]
    exact = en.silhouette_polygon(points, triangles, direction)
    assert exact.geom_type == "Polygon" and len(exact.interiors) == 0
    culled = en.silhouette_polygon(points, triangles, direction, cull_backfaces=True)
    assert exact.symmetric_difference(culled).area < 1e-9

    # every projected vertex lies in the silhouette, and the border is made of projected vertices
    projected = en.project_to_2d(points, direction)
    assert np.all(en.compute_eclipse_function(projected, surface=exact) < 1e-7)

    # the alpha shape of the projected vertices is an approximation of it
    alpha = en.silhouette_polygon(points, triangles, direction, method="alpha", alpha=0.1)
    assert alpha.symmetric_difference(exact).area < 0.1 * exact.area


def test_compute_eclipse_function():
    # C-shaped region: not star-shaped with respect to its centroid
    ring = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 2), (3, 2), (3, 3), (0, 3)]
    samples = np.array([[2.0, 1.5], [0.5, 1.5], [2.0, 0.5], [4.0, 0.5], [3.0, 0.5]])
    expected = [0.5, -0.5, -0.5, 1.0, 0.0]  # in the notch, in the spine, in the lower arm, outside, on the border
    assert en.compute_eclipse_function(samples, border_points=ring) == pytest.approx(expected)
    assert en.compute_eclipse_function(samples, surface=Polygon(ring)) == pytest.approx(expected)

    # distance from the vertices instead of the border, as in the datasets of the paper
    approximate = en.compute_eclipse_function(samples, surface=Polygon(ring), exact=False)
    assert approximate == pytest.approx([np.hypot(1, 0.5), -np.hypot(0.5, 0.5), -np.hypot(1, 0.5), np.hypot(1, 0.5), 0.5])

    with pytest.raises(ValueError):  # points that are not ordered along the border
        en.compute_eclipse_function(samples, border_points=[(0, 0), (1, 1), (1, 0), (0, 1)])


def test_sample_view_covers_a_square_around_the_silhouette():
    square = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])  # away from the origin
    samples, values = en.sample_view(square, n_uniform=2000, n_border=10, n_per_border=5, n_on_border=20,
                                     border_radius=0.01, margin=0.5, seed=0)
    assert samples.shape == (2070, 2) and values.shape == (2070,)
    uniform = samples[50:2050]
    assert uniform.min() == pytest.approx(1.75, abs=0.01) and uniform.max() == pytest.approx(3.25, abs=0.01)
    assert np.all(np.abs(values[:50]) <= 0.01) and np.all(values[2050:] == 0)


def test_generate_dataset():
    points, triangles, _ = en.load_mesh("Bennu", low_poly=True)
    sampling = dict(n_uniform=50, n_border=20, n_per_border=5, n_on_border=30)
    data = en.generate_dataset(points, triangles, n_views=3, seed=0, progress=False, **sampling)
    assert data.shape == (3 * (50 + 20 * 5 + 30), 7)
    assert np.sum(data[:, 6] == 0) == 3 * 30
    assert np.array_equal(data, en.generate_dataset(points, triangles, n_views=3, seed=0, progress=False, **sampling))

    directions, az, el = en.fibonacci_sphere(3)
    assert np.allclose(np.unique(data[:, :4], axis=0), np.unique(en.encode_direction(az, el), axis=0))

    # the labels agree with ray tracing: a point is in eclipse if the ray towards the Sun hits the mesh
    v0, v1, v2 = (points[triangles][:, i] for i in range(3))
    first = data[np.all(data[:, :4] == en.encode_direction(az[0], el[0]), axis=1)]
    clear = np.abs(first[:, 6]) > 1e-3  # leave out the samples sitting on the border
    positions = en.plane_to_3D(first[clear, 4], first[clear, 5], directions[0], -2.0)
    eclipsed = [en.is_eclipsed(p, directions[0], v0, v1, v2) for p in positions]
    assert np.array_equal(eclipsed, first[clear, 6] < 0)
