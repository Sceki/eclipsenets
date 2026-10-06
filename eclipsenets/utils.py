"""Coordinate conventions and projections shared by the whole library.

A Sun direction ``s`` is described by its azimuth ``az`` and its polar angle
``el`` (measured from +z, so it is a colatitude; it is called "elevation" in
the rest of the code and in the paper)::

    s = [sin(el) cos(az), sin(el) sin(az), cos(el)]

The plane orthogonal to ``s`` is spanned by::

    U = [-sin(az), cos(az), 0]
    V = [cos(az) cos(el), sin(az) cos(el), -sin(el)]

and an EclipseNET takes as input ``[cos(az), sin(az), cos(el), sin(el), X, Y]``
with ``X = r . U`` and ``Y = r . V``.

The azimuth is undefined for a Sun direction along the z axis (it is taken as
zero). The orientation of the plane changes abruptly around that direction,
and the networks are less accurate there.
"""
import numpy as np
from scipy.spatial import cKDTree


def cart2spherical(points):
    """Convert from cartesian to spherical coordinates. Convention used is
    x = r cos(t) sin(p)
    y = r sin(t) sin(p)
    z = r cos(p)

    Args:
        points ((., 3) np.array): cartesian points

    Returns:
        (., 3) np.array: array with columns containing r, theta, phi
    """
    points = np.array(points, dtype=float).reshape(-1, 3)
    r = np.linalg.norm(points, axis=1)
    # the clip guards against |z / r| exceeding 1 by a rounding error
    phi = np.arccos(np.clip(points[:, 2] / r, -1.0, 1.0))
    theta = np.arctan2(points[:, 1], points[:, 0])
    return np.array([r, theta, phi]).transpose()


def sph2cart(az, el, rad=False):
    """
    This converts angles in azimuth (between 0-360) and elevation (between 0-180) to a 3D vector.

    Args:
        az (float): azimuth angle in degrees
        el (float): elevation angle in degrees (polar angle, measured from +z)
        rad (bool): if True, the angles are in radians

    Returns:
        (3,) np.array: 3D vector
    """
    if not rad:
        az = np.radians(az)
        el = np.radians(el)
    x = np.sin(el) * np.cos(az)
    y = np.sin(el) * np.sin(az)
    z = np.cos(el)
    return np.array([x, y, z])


def plane_basis(ray_d):
    """Orthonormal basis (U, V) of the plane orthogonal to a direction.

    Args:
        ray_d (3D np.array): direction of the ray (does not need to be normalized).

    Returns:
        (np.array, np.array): the unit vectors U and V, each of shape (3,)
    """
    _, theta, phi = cart2spherical(ray_d)[0, :]
    U = np.array([-np.sin(theta), np.cos(theta), 0.0])
    V = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), -np.sin(phi)])
    return U, V


def project_on_plane(points, ray_d):
    """Projects points on the 2D UV plane

    Args:
        points ((., 3) np.array): cartesian points
        ray_d (3D np.array): direction of the ray.

    Returns:
        (np.array, np.array): X an Y coordinates on the plane
    """
    U, V = plane_basis(ray_d)
    X = np.dot(points, U)
    Y = np.dot(points, V)
    return X, Y


def project_to_2d(points, view_dir):
    """Same projection as :func:`project_on_plane`, returned as one (., 2) array."""
    return np.stack(project_on_plane(points, view_dir), axis=-1)


def plane_to_3D(X, Y, ray_d, offset):
    """From points on the 2D UV plane to 3D points

    Args:
        X (np.array): Coordinates on the U direction
        Y (np.array):  Coordinates on the V direction
        ray_d (3D np.array): direction of the ray.
        offset (float): the plane distance from the center

    Returns:
        (., 3) np.array: cartesian points
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    ray_d = np.asarray(ray_d, dtype=float)
    U, V = plane_basis(ray_d)
    return X[:, None] * U + Y[:, None] * V + float(offset) * ray_d / np.linalg.norm(ray_d)


def encode_direction(az, el):
    """Sun-direction part of the network input: [cos(az), sin(az), cos(el), sin(el)].

    Args:
        az (float or np.array): azimuth, in radians
        el (float or np.array): elevation (polar angle from +z), in radians

    Returns:
        (..., 4) np.array
    """
    az, el = np.broadcast_arrays(np.asarray(az, dtype=float), np.asarray(el, dtype=float))
    return np.stack([np.cos(az), np.sin(az), np.cos(el), np.sin(el)], axis=-1)


def network_inputs(positions, sun_direction):
    """Network inputs for spacecraft positions and a Sun direction.

    Args:
        positions ((3,) or (N, 3) np.array): spacecraft positions, in units of L
        sun_direction ((3,) np.array): direction of the Sun

    Returns:
        (6,) or (N, 6) np.array: [cos(az), sin(az), cos(el), sin(el), X, Y]
    """
    positions = np.asarray(positions, dtype=float)
    _, az, el = cart2spherical(sun_direction)[0, :]
    X, Y = project_on_plane(positions, sun_direction)
    angles = np.broadcast_to(encode_direction(az, el), np.shape(X) + (4,))
    return np.concatenate([angles, np.stack([X, Y], axis=-1)], axis=-1)


def sort_points_along_boundary(points):
    """Sorts 2D points by their polar angle around the centroid.

    This orders a boundary correctly only if the region is star-shaped with
    respect to its centroid: it is wrong for concave silhouettes such as the
    ones of 67P. See :func:`eclipsenets.silhouette_polygon` for a robust way of
    getting an ordered silhouette.
    """
    points = np.asarray(points)
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    return list(points[np.argsort(angles)])


def nearest_neighbor(points):
    """Greedy nearest-neighbour ordering of points, starting from the first one.

    This is a heuristic: it can jump across narrow necks and leave points
    behind, so the resulting polygon is not guaranteed to be simple. See
    :func:`eclipsenets.silhouette_polygon` for a robust way of getting an
    ordered silhouette.
    """
    points = np.asarray(points)
    n = len(points)
    tree = cKDTree(points)
    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=int)
    current = 0
    for i in range(n):
        order[i] = current
        visited[current] = True
        if i == n - 1:
            break
        # Query a growing neighbourhood until it contains an unvisited point.
        k = 8
        while True:
            _, idx = tree.query(points[current], k=min(k, n))
            idx = np.atleast_1d(idx)
            candidates = idx[~visited[idx]]
            if len(candidates) > 0:
                current = candidates[0]
                break
            if k >= n:
                raise RuntimeError("no unvisited point left")
            k *= 4
    return list(points[order])
