import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from . import is_intersecting_v2

def plot_ray(ray_o, ray_d, ax, length=30, *args, **kwargs):
    """
    Plots a 3D ray on the given axis.

    Args:
        ray_o (np.ndarray, shape (3,)): Origin of the ray.
        ray_d (np.ndarray, shape (3,)): Direction vector of the ray.
        ax (mpl_toolkits.mplot3d.Axes3D): 3D axis to plot on.
        length (float, optional): Length of the ray. Default is 30.
        *args, **kwargs: Additional arguments passed to `ax.plot`.

    Returns:
        mpl_toolkits.mplot3d.Axes3D: The axis with the plotted ray.
    """
    ray_o = np.array(ray_o)
    ray_d = np.array(ray_d)
    end_point = ray_o + ray_d * length
    ax.plot(
        [ray_o[0], end_point[0]],
        [ray_o[1], end_point[1]],
        [ray_o[2], end_point[2]],
        *args,
        **kwargs
    )
    return ax


def plot_mesh_vertices(mesh_points, D, ax=None, plot_ref=True, axis_off=True, *args, **kwargs):
    """
    Plots the vertices of a 3D mesh.

    Args:
        mesh_points (np.ndarray, shape (N,3)): Vertex coordinates of the mesh.
        D (float): Axis limits for plotting (±D in all axes).
        ax (mpl_toolkits.mplot3d.Axes3D, optional): Axis to plot on. Creates a new one if None.
        plot_ref (bool, optional): Whether to plot reference axes at origin. Default True.
        axis_off (bool, optional): Whether to hide axis lines. Default True.
        *args, **kwargs: Additional arguments passed to `ax.scatter`.

    Returns:
        mpl_toolkits.mplot3d.Axes3D: The axis with the plotted mesh.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    mesh_points = np.array(mesh_points)
    ax.scatter(mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2], *args, **kwargs)

    # Set view and axis limits
    ax.view_init(elev=60., azim=45.)
    ax.set_xlim([-D, D])
    ax.set_ylim([-D, D])
    ax.set_zlim([-D, D])

    if axis_off:
        ax.set_axis_off()

    # Plot reference axes at origin
    if plot_ref:
        ref_length = D * 0.1
        ax.quiver(0, 0, 0, 1, 0, 0, length=ref_length, normalize=True, arrow_length_ratio=0.1, color='brown')
        ax.quiver(0, 0, 0, 0, 1, 0, length=ref_length, normalize=True, arrow_length_ratio=0.1, color='cornflowerblue')
        ax.quiver(0, 0, 0, 0, 0, 1, length=ref_length, normalize=True, arrow_length_ratio=0.1, color='gray')

    return ax


def plot_intersecting_triangles(mesh_points, mesh_triangles, ray_o, ray_d, length=30, c='r', s=1, ax=None, *args, **kwargs):
    """
    Plots the vertices of triangles from a mesh that intersect a given ray.

    Args:
        mesh_points (np.ndarray, shape (N,3)): Vertex coordinates of the mesh.
        mesh_triangles (np.ndarray, shape (M,3)): Indices of vertices forming triangles.
        ray_o (np.ndarray, shape (3,)): Origin of the ray.
        ray_d (np.ndarray, shape (3,)): Direction vector of the ray.
        length (float, optional): Ray length. Default 30.
        c (str, optional): Color for intersection points. Default 'r'.
        s (float, optional): Marker size for intersection points. Default 1.
        ax (mpl_toolkits.mplot3d.Axes3D, optional): Axis to plot on. Creates a new one if None.
        *args, **kwargs: Additional arguments passed to `ax.scatter`.

    Returns:
        mpl_toolkits.mplot3d.Axes3D: Axis with plotted intersections and ray.
    """
    mesh_points = np.array(mesh_points)
    mesh_triangles = np.array(mesh_triangles)
    ray_o = np.array(ray_o)
    ray_d = np.array(ray_d)

    # Create axis if not provided
    if ax is None:
        D = (np.max(mesh_points) - np.min(mesh_points)) / 3
        ax = plot_mesh_vertices(mesh_points, D, s=1, c='k', alpha=0.1, *args, **kwargs)

    # Plot the ray
    plot_ray(ray_o, ray_d, ax, length=length, alpha=0.3)

    # Extract triangle vertices
    v0 = mesh_points[mesh_triangles[:, 0]]
    v1 = mesh_points[mesh_triangles[:, 1]]
    v2 = mesh_points[mesh_triangles[:, 2]]

    # Check for intersections
    intersects = is_intersecting_v2(ray_o, ray_d, v0, v1, v2)
    intersect_indices = np.where(intersects)[0]

    # Plot vertices of intersecting triangles
    for i in intersect_indices:
        ax.scatter(v0[i, 0], v0[i, 1], v0[i, 2], c=c, s=s, *args, **kwargs)
        ax.scatter(v1[i, 0], v1[i, 1], v1[i, 2], c=c, s=s, *args, **kwargs)
        ax.scatter(v2[i, 0], v2[i, 1], v2[i, 2], c=c, s=s, *args, **kwargs)

    return ax