import numpy as np
import shapely
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
from typing import Union, Optional, Tuple

from . import is_intersecting_v2, ray_triangle_intersect
from ._utils import sort_points_along_boundary


def compute_eclipse_function(
    samples: np.ndarray, 
    border_points: np.ndarray, 
    surface: Optional[Polygon] = None
) -> np.ndarray:
    """
    Computes signed minimum distance from 2D samples to a polygon border.
    
    - Negative: inside polygon (eclipsed)
    - Positive: outside polygon (illuminated)  
    - Magnitude: distance to nearest border point
    
    Args:
        samples: (N, 2) array of 2D query points
        border_points: (M, 2) array of polygon border points
        surface: Optional pre-computed Shapely Polygon
        
    Returns:
        (N,) array of signed distances
    """
    if surface is None:
        sorted_points = np.asarray(sort_points_along_boundary(border_points))
        surface = Polygon(sorted_points)
    
    # Vectorized point-in-polygon test (faster than list comprehension)
    bool_inside = np.array([surface.contains(Point(x, y)) for x, y in samples])
    
    # Convert True -> -1 (inside), False -> +1 (outside)
    sign = -2 * bool_inside + 1  # or: np.where(bool_inside, -1.0, 1.0)
    
    # Minimum distance to any border point
    min_distances = cdist(samples, border_points).min(axis=1)
    
    return min_distances * sign


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
    - <0: partially eclipsed (sum of entry-exit chord lengths)
    - NaN: ray intersects odd number of faces (inside mesh)
    
    Args:
        mesh_points: (N, 3) vertex positions
        mesh_triangles: (M, 3) triangle vertex indices
        ray_o: (3,) ray origin
        ray_d: (3,) normalized ray direction
        
    Returns:
        float: eclipse metric (>0 illuminated, <0 eclipsed)
    
    Raises:
        ValueError: Invalid ray intersections per triangle
    """
    # Extract triangle vertices: (M, 3, 3)
    triangles = mesh_points[mesh_triangles]
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    
    # Find intersecting triangles
    intersecting_mask = is_intersecting_v2(ray_o, ray_d, v0, v1, v2)
    intersecting_id = np.flatnonzero(intersecting_mask)
    n_intersections = len(intersecting_id)
    
    if n_intersections == 0:
        # No intersections → illuminated → min distance to projected silhouette
        proj_axis = np.dot(mesh_points, ray_d)
        mesh_proj = mesh_points - ray_d * proj_axis[:, None]
        ray_proj = ray_o - np.dot(ray_o, ray_d) * ray_d
        
        dists = cdist([ray_proj], mesh_proj)[0]
        return np.min(dists)
    
    elif n_intersections % 2 == 0:
        # Even intersections → eclipsed → sum entry-exit chord lengths
        all_t, all_p = [], []
        for idx in intersecting_id:
            t, p = ray_triangle_intersect(ray_o, ray_d, v0[idx], v1[idx], v2[idx])
            all_t.append(t)
            all_p.append(p)
        
        all_t = np.array(all_t)
        all_p = np.array(all_p)
        
        # Sort intersections along ray
        sort_idx = np.argsort(all_t)
        sorted_p = all_p[sort_idx]
        
        # Sum distances between consecutive pairs (entry→exit chords)
        chord_dists = np.linalg.norm(np.diff(sorted_p.reshape(-1, 3), axis=0))
        return -np.sum(chord_dists)
    
    else:
        # Odd intersections → ray terminates inside mesh → invalid
        return np.nan
