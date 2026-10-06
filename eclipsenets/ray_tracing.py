"""Möller-Trumbore ray/triangle intersection, used as ground truth for eclipses."""
import numpy as np
from typing import Tuple


EPS = 1e-8

# Direction of the rays cast to tell inside from outside. A ray that runs
# through an edge shared by two triangles hits both, which spoils the count of
# the crossings: a direction that is not aligned with anything makes that
# unlikely for meshes and points with round coordinates.
_CASTING_DIRECTION = np.array([0.0541, 0.0912, 0.9944])


def _dot(a, b):
    return np.einsum("...i,...i->...", a, b)


def _moller_trumbore(ray_o, ray_d, v0, v1, v2):
    """Broadcasting Möller-Trumbore test. Returns the hit mask and the ray parameter t."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_d, edge2)
    a = _dot(edge1, h)
    # The determinant is a triple product of the two edges and the direction:
    # compare it against their magnitudes so the test is independent of the
    # units of the mesh.
    parallel = a * a <= EPS**2 * _dot(edge1, edge1) * _dot(edge2, edge2) * _dot(ray_d, ray_d)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 1.0 / a
        s = ray_o - v0
        u = _dot(s, h) * f
        q = np.cross(s, edge1)
        v = _dot(q, ray_d) * f
        t = _dot(q, edge2) * f
        hit = (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 0)
    return hit & ~parallel, t


def is_intersecting_v1(ray_o: np.ndarray, ray_d: np.ndarray, v0: np.ndarray,
                       v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Möller-Trumbore intersection test, vectorized over rays.

    Args:
        ray_o: Ray origins, shape (N, 3).
        ray_d: Ray direction, shape (3,).
        v0, v1, v2: Triangle vertices, each shape (3,).

    Returns:
        Boolean array (N,) indicating intersection (edge-inclusive).
    """
    ray_o = np.asarray(ray_o, dtype=float)
    if ray_o.ndim != 2 or ray_o.shape[1] != 3:
        raise ValueError("ray_o must have shape (N, 3)")
    hit, _ = _moller_trumbore(ray_o, np.asarray(ray_d, dtype=float), v0, v1, v2)
    return hit


def is_intersecting_v2(ray_o: np.ndarray, ray_d: np.ndarray, v0: np.ndarray,
                       v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Möller-Trumbore intersection test, vectorized over triangles.

    Args:
        ray_o: Ray origin, shape (3,).
        ray_d: Ray direction, shape (3,).
        v0, v1, v2: Triangle vertices, each shape (N, 3).

    Returns:
        Boolean array (N,) indicating intersection (edge-inclusive).
    """
    if v0.ndim != 2 or v0.shape[1] != 3:
        raise ValueError("Triangle vertices must have shape (N, 3)")
    hit, _ = _moller_trumbore(np.asarray(ray_o, dtype=float), np.asarray(ray_d, dtype=float), v0, v1, v2)
    return hit


def ray_triangle_intersect(ray_o: np.ndarray, ray_d: np.ndarray,
                           v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Möller-Trumbore intersection, scalar single ray-triangle.

    Args:
        ray_o: Ray origin, shape (3,).
        ray_d: Ray direction, shape (3,).
        v0, v1, v2: Triangle vertices, shape (3,).

    Returns:
        Tuple (t, intersection_point).

    Raises:
        ValueError: No intersection or degenerate cases.
    """
    ray_o = np.asarray(ray_o, dtype=float)
    ray_d = np.asarray(ray_d, dtype=float)
    if ray_o.shape != (3,):
        raise ValueError("ray_o must have shape (3,)")
    hit, t = _moller_trumbore(ray_o, ray_d, v0, v1, v2)
    if not hit:
        raise ValueError("The ray does not intersect the triangle")
    return float(t), ray_o + t * ray_d


def is_eclipsed(position: np.ndarray, sun_direction: np.ndarray, v0: np.ndarray,
                v1: np.ndarray, v2: np.ndarray) -> bool:
    """
    Ground-truth eclipse test: is the Sun hidden by the mesh as seen from `position`?

    Args:
        position: Spacecraft position (outside the body), shape (3,).
        sun_direction: Direction of the Sun, shape (3,).
        v0, v1, v2: Triangle vertices, each shape (N, 3).

    Returns:
        True if the ray from the spacecraft to the Sun hits the mesh.
    """
    return bool(np.any(is_intersecting_v2(position, sun_direction, v0, v1, v2)))


def is_outside_v1(points: np.ndarray, mesh_points: np.ndarray,
                  mesh_triangles: np.ndarray) -> np.ndarray:
    """
    Point-in-mesh test via ray casting, vectorized over points.

    Args:
        points: Test points, shape (N, 3).
        mesh_points: Mesh vertices, shape (M, 3).
        mesh_triangles: Triangle indices, shape (T, 3).

    Returns:
        Boolean (N,) True if outside (even crossings).
    """
    counter = np.zeros(len(points), dtype=int)
    for tri_idx in mesh_triangles:
        counter += is_intersecting_v1(points, _CASTING_DIRECTION,
                                      mesh_points[tri_idx[0]],
                                      mesh_points[tri_idx[1]],
                                      mesh_points[tri_idx[2]])
    return (counter % 2) == 0


def is_outside_v2(point: np.ndarray, mesh_points: np.ndarray,
                  mesh_triangles: np.ndarray) -> bool:
    """
    Point-in-mesh test via ray casting, vectorized over mesh.

    Args:
        point: Test point, shape (3,).
        mesh_points: Mesh vertices, shape (M, 3).
        mesh_triangles: Triangle indices, shape (T, 3).

    Returns:
        True if outside (even crossings).
    """
    tris = mesh_points[mesh_triangles]
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    intersects = is_intersecting_v2(point, _CASTING_DIRECTION, v0, v1, v2)
    return bool(np.sum(intersects) % 2 == 0)
