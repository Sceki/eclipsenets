import numpy as np
import warnings
from typing import Tuple


EPS = 1e-8


def is_intersecting_v1(ray_o: np.ndarray, ray_d: np.ndarray, v0: np.ndarray, 
                       v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Möller-Trumbore intersection test, vectorized over rays [web:7].

    Args:
        ray_o: Ray origins, shape (N, 3).
        ray_d: Ray direction, shape (3,).
        v0, v1, v2: Triangle vertices, each shape (3,).

    Returns:
        Boolean array (N,) indicating intersection (edge-inclusive).
    """
    if ray_o.shape[1] != 3:
        raise ValueError("ray_o must have shape (N, 3)")
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_d, edge2)
    a = np.dot(edge1, h)
    
    parallel = np.abs(a) < EPS
    if np.all(parallel):
        return np.zeros(len(ray_o), dtype=bool)
    
    f = 1.0 / a[~parallel]
    s = ray_o[~parallel] - v0
    u = np.dot(s, h) * f
    
    crit1 = (u >= 0) & (u <= 1)
    q = np.cross(s, edge1)
    v = np.dot(q, ray_d) * f
    crit2 = (v >= 0) & (u + v <= 1)
    t = np.dot(q, edge2) * f
    crit3 = t > 0
    
    result = np.zeros(len(ray_o), dtype=bool)
    mask = ~parallel & crit1 & crit2 & crit3
    result[~parallel][mask] = True  # Note: mask already sliced
    
    return result


def is_intersecting_v2(ray_o: np.ndarray, ray_d: np.ndarray, v0: np.ndarray, 
                       v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Möller-Trumbore intersection test, vectorized over triangles [web:7].

    Args:
        ray_o: Ray origin, shape (3,).
        ray_d: Ray direction, shape (3,).
        v0, v1, v2: Triangle vertices, each shape (N, 3).

    Returns:
        Boolean array (N,) indicating intersection (edge-inclusive).
    """
    if v0.shape[1] != 3:
        raise ValueError("Triangle vertices must have shape (N, 3)")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(ray_d, edge2)
        a = np.sum(edge1 * h, axis=1)
        
        f = 1.0 / a
        s = ray_o - v0
        u = np.sum(s * h, axis=1) * f
        
        crit1 = (u >= 0) & (u <= 1)
        q = np.cross(s, edge1)
        v = np.sum(q * ray_d, axis=1) * f
        crit2 = (v >= 0) & (u + v <= 1)
        t = np.sum(q * edge2, axis=1) * f
        crit3 = t > 0
        
        result = crit1 & crit2 & crit3
        result[np.abs(a) < EPS] = False
        return result


def ray_triangle_intersect(ray_o: np.ndarray, ray_d: np.ndarray, 
                           v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Möller-Trumbore intersection, scalar single ray-triangle [web:7].

    Args:
        ray_o: Ray origin, shape (3,).
        ray_d: Ray direction, shape (3,).
        v0, v1, v2: Triangle vertices, shape (3,).

    Returns:
        Tuple (t, intersection_point). Raises ValueError on no intersection.

    Raises:
        ValueError: No intersection or degenerate cases.
    """
    if ray_o.shape != (3,):
        raise ValueError("ray_o must have shape (3,)")

    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_d, edge2)
    a = np.dot(edge1, h)
    
    if np.abs(a) < EPS:
        raise ValueError("Ray parallel to triangle (no unique intersection)")
    
    f = 1.0 / a
    s = ray_o - v0
    u = np.dot(s, h) * f
    
    if u < 0 or u > 1:
        raise ValueError("No intersection (u out of [0,1])")
    
    q = np.cross(s, edge1)
    v = np.dot(ray_d, q) * f
    
    if v < 0 or u + v > 1:
        raise ValueError("No intersection (v out of bounds)")
    
    t = f * np.dot(edge2, q)
    if t <= 0:
        raise ValueError("No intersection (behind ray origin)")
    
    return t, ray_o + t * ray_d


def is_outside_v1(points: np.ndarray, mesh_points: np.ndarray, 
                  mesh_triangles: np.ndarray) -> np.ndarray:
    """
    Point-in-mesh test via ray casting, vectorized over points [web:7].

    Args:
        points: Test points, shape (N, 3).
        mesh_points: Mesh vertices, shape (M, 3).
        mesh_triangles: Triangle indices, shape (T, 3).

    Returns:
        Boolean (N,) True if outside (even crossings).
    """
    direction = np.array([0., 0., 1.])
    counter = np.zeros(len(points), dtype=int)
    for tri_idx in mesh_triangles:
        counter += is_intersecting_v1(points, direction, 
                                     mesh_points[tri_idx[0]], 
                                     mesh_points[tri_idx[1]], 
                                     mesh_points[tri_idx[2]])
    return (counter % 2) == 0


def is_outside_v2(point: np.ndarray, mesh_points: np.ndarray, 
                  mesh_triangles: np.ndarray) -> bool:
    """
    Point-in-mesh test via ray casting, vectorized over mesh [web:7].

    Args:
        point: Test point, shape (3,).
        mesh_points: Mesh vertices, shape (M, 3).
        mesh_triangles: Triangle indices, shape (T, 3).

    Returns:
        True if outside (even crossings).
    """
    ray_d = np.array([0., 0., 1.])
    tris = mesh_points[mesh_triangles]
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    intersects = is_intersecting_v2(point, ray_d, v0, v1, v2)
    return np.sum(intersects) % 2 == 0
