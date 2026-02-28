import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean


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
    points = np.array(points).reshape(-1, 3)
    r = np.linalg.norm(points, axis=1)
    phi = np.arccos(points[:, 2] / r)
    # phi=np.arctan2(points[:,2],np.sqrt(points[:,0]**2+points[:,1]**2))
    theta = np.arctan2(points[:, 1], points[:, 0])
    return (np.array([r, theta, phi]).transpose())


def project_on_plane(points, ray_d):
    """Projects points on the 2D UV plane

    Args:
        points ((., 3) np.array): cartesian points
        ray_d (3D np.array): direction of the ray.

    Returns:
        (np.array, np.array): X an Y coordinates on the plane
    """
    # Convert direction to spherical
    r, theta, phi = cart2spherical(ray_d)[0, :]
    # Define the basis on the sphere surface
    U = np.array([-np.sin(theta), np.cos(theta), 0])
    V = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), -np.sin(phi)])
    # Compute the projection
    X = np.dot(points, U)
    Y = np.dot(points, V)
    return X, Y


def sph2cart(az, el, rad=False):
    """
    This converts angles in azimuth (between 0-360) and elevation (between 0-180) to a 3D vector.

    Args:
        azimuth (float): azimuth angle in degrees
        elevation (float): elevation angle in degrees
        rad (bool): if True, the angles are in radians 

    Returns:
        (3,) np.array: 3D vector
    """
    if rad != True:
        az = np.radians(az)
        el = np.radians(el)
    x = np.sin(el) * np.cos(az)
    y = np.sin(el) * np.sin(az)
    z = np.cos(el)
    return np.array([x, y, z])


def project_to_2d(points, view_dir):
    # Calculate the orthonormal basis vectors
    z_axis = view_dir / np.linalg.norm(view_dir)
    x_axis = np.array([-z_axis[1], z_axis[0], 0])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Projection matrix
    projection_matrix = np.array([x_axis, y_axis])

    # Project points
    points_2d = points @ projection_matrix.T
    return points_2d


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
    assert(type(offset) is float)
    n = len(X)
    # Convert direction to spherical
    r, theta, phi = cart2spherical(ray_d)[0, :]
    # Define the basis on the sphere surface
    U = np.array([-np.sin(theta), np.cos(theta), 0])
    V = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), -np.sin(phi)])
    # Compute the 3D points
    points = (np.tile(X, (3, 1)) * U.reshape(3, 1)).transpose()
    points += (np.tile(Y, (3, 1)) * V.reshape(3, 1)).transpose()
    points += np.tile(ray_d, (n, 1)) * offset
    return points


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.

    Args:
        points (np.array of shape (n,2)): points.
        alpha (float): alpha value.
        only_outer (bool): specifies if we keep only the outer border
            or also inner edges.

    Returns:
        set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
        the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"
    assert points.shape[1] == 2, "Need two dimensional points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # [www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle]
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def calculate_centroid(points):
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    centroid_x = sum(x_coords) / len(points)
    centroid_y = sum(y_coords) / len(points)
    return (centroid_x, centroid_y)


def calculate_angle(point, centroid):
    dx = point[0] - centroid[0]
    dy = point[1] - centroid[1]
    return np.arctan2(dy, dx)


def sort_points_along_boundary(points):
    centroid = calculate_centroid(points)
    angles = [calculate_angle(point, centroid) for point in points]
    sorted_indices = np.argsort(angles)
    sorted_points = [points[i] for i in sorted_indices]
    return sorted_points


def nearest_neighbor(points):
    # Initialize variables
    n = len(points)
    visited = [False] * n
    sorted_points = []

    # Start from the first point
    current_index = 0

    # Iterate until all points are visited
    while len(sorted_points) < n:
        # Add current point to the sorted list
        sorted_points.append(points[current_index])
        visited[current_index] = True

        # Find the nearest neighbor to the current point
        nearest_dist = float('inf')
        nearest_index = -1
        for i, point in enumerate(points):
            if not visited[i]:
                dist = euclidean(points[current_index], point)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_index = i

        # Move to the nearest neighbor
        current_index = nearest_index

    return sorted_points
