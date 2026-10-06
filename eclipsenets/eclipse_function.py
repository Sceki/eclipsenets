import numpy as np
import shapely
from shapely.geometry import Polygon
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from typing import Optional

from .ray_tracing import is_intersecting_v2, ray_triangle_intersect
from .silhouette import silhouette_border


def compute_eclipse_function(
    samples: np.ndarray,
    border_points: Optional[np.ndarray] = None,
    surface=None,
    exact: bool = True
) -> np.ndarray:
    """
    Computes the eclipse function: the signed distance from 2D samples to the
    border of the eclipsed region.

    - Negative: inside the region (eclipsed)
    - Positive: outside the region (illuminated)
    - Magnitude: distance to the border

    Args:
        samples: (N, 2) array of 2D query points
        border_points: (M, 2) array of border points. They are needed if
            `surface` is not given, in which case they must be ordered along
            the border, or if `exact` is False.
        surface: shapely Polygon (or MultiPolygon) of the eclipsed region
        exact: if True, the distance is the one to the border of `surface`.
            If False, it is the distance to the nearest of `border_points`,
            as in the datasets of the paper: this overestimates the distance
            next to the border, by up to half the spacing of the points.

    Returns:
        (N,) array of signed distances
    """
    samples = np.asarray(samples, dtype=float).reshape(-1, 2)
    if surface is None:
        if border_points is None:
            raise ValueError("Either border_points or surface must be given")
        surface = Polygon(border_points)
        if not surface.is_valid:
            raise ValueError(
                "border_points do not describe a simple polygon: they must be ordered "
                "along the border. Pass the eclipsed region as `surface` otherwise.")

    inside = shapely.contains_xy(surface, samples[:, 0], samples[:, 1])
    if exact:
        distances = shapely.distance(surface.boundary, shapely.points(samples))
    else:
        if border_points is None:
            border_points = silhouette_border(surface)
        distances, _ = cKDTree(np.asarray(border_points, dtype=float)).query(samples)
    return np.where(inside, -distances, distances)


def eclipse_function(
    mesh_points: np.ndarray,
    mesh_triangles: np.ndarray,
    ray_o: np.ndarray,
    ray_d: np.ndarray
) -> float:
    """
    Computes continuous eclipse metric along a ray through the mesh silhouette.

    - 0: on silhouette edge
    - >0: fully illuminated (min distance to projected mesh)
    - <0: eclipsed (sum of entry-exit chord lengths)
    - NaN: ray intersects odd number of faces (inside mesh)

    Args:
        mesh_points: (N, 3) vertex positions
        mesh_triangles: (M, 3) triangle vertex indices
        ray_o: (3,) ray origin
        ray_d: (3,) normalized ray direction

    Returns:
        float: eclipse metric (>0 illuminated, <0 eclipsed)
    """
    # Extract triangle vertices: (M, 3, 3)
    triangles = mesh_points[mesh_triangles]
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    # Find intersecting triangles
    intersecting_mask = is_intersecting_v2(ray_o, ray_d, v0, v1, v2)
    intersecting_id = np.flatnonzero(intersecting_mask)
    sorted_t = np.sort([
        ray_triangle_intersect(ray_o, ray_d, v0[idx], v1[idx], v2[idx])[0]
        for idx in intersecting_id
    ])
    # A ray through an edge or a vertex hits all the triangles that share it:
    # these are one crossing of the surface.
    sorted_t = sorted_t[np.diff(sorted_t, prepend=-np.inf) > 1e-9 * np.ptp(mesh_points)]
    n_intersections = len(sorted_t)

    if n_intersections == 0:
        # No intersections → illuminated → min distance to projected silhouette
        proj_axis = np.dot(mesh_points, ray_d)
        mesh_proj = mesh_points - ray_d * proj_axis[:, None]
        ray_proj = ray_o - np.dot(ray_o, ray_d) * ray_d

        dists = cdist([ray_proj], mesh_proj)[0]
        return np.min(dists)

    elif n_intersections % 2 == 0:
        # Even intersections → eclipsed → sum entry-exit chord lengths.
        # Sorted along the ray, the intersections alternate entry, exit, entry, ...
        chords = sorted_t[1::2] - sorted_t[0::2]
        return -np.sum(chords) * np.linalg.norm(ray_d)

    else:
        # Odd intersections → ray terminates inside mesh → invalid
        return np.nan
