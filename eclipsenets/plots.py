import matplotlib.pyplot as plt
import numpy as np
import torch

from .ray_tracing import is_intersecting_v2
from .utils import cart2spherical, encode_direction

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
        ax = plot_mesh_vertices(mesh_points, D, s=1, c='k', alpha=0.1)

    # Plot the ray
    plot_ray(ray_o, ray_d, ax, length=length, alpha=0.3)

    # Extract triangle vertices
    v0 = mesh_points[mesh_triangles[:, 0]]
    v1 = mesh_points[mesh_triangles[:, 1]]
    v2 = mesh_points[mesh_triangles[:, 2]]

    # Plot vertices of intersecting triangles
    intersects = is_intersecting_v2(ray_o, ray_d, v0, v1, v2)
    vertices = np.concatenate([v0[intersects], v1[intersects], v2[intersects]])
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=c, s=s, *args, **kwargs)

    return ax


def plot_silhouette(model, sun_direction, ax=None, extent=0.7, resolution=300, **kwargs):
    """
    Plots the silhouette predicted by an EclipseNET: the zero level of the network.

    Args:
        model (torch.nn.Module): The EclipseNET.
        sun_direction (np.ndarray, shape (3,)): Direction of the Sun.
        ax (matplotlib Axes, optional): Axis to plot on. Creates a new one if None.
        extent (float, optional): The network is evaluated on [-extent, extent]^2. Default 0.7.
        resolution (int, optional): Number of grid points per side. Default 300.
        **kwargs: Additional arguments passed to `ax.contour`.

    Returns:
        matplotlib Axes: The axis with the plotted silhouette.
    """
    if ax is None:
        _, ax = plt.subplots()
    _, az, el = cart2spherical(sun_direction)[0, :]
    side = np.linspace(-extent, extent, resolution)
    X, Y = np.meshgrid(side, side)
    inputs = np.empty((X.size, 6))
    inputs[:, :4] = encode_direction(az, el)
    inputs[:, 4] = X.ravel()
    inputs[:, 5] = Y.ravel()
    parameter = next(model.parameters())
    with torch.no_grad():
        values = model(torch.as_tensor(inputs, dtype=parameter.dtype, device=parameter.device))
    ax.contour(X, Y, values.cpu().numpy().reshape(X.shape), levels=[0.], **kwargs)
    ax.set_aspect('equal')
    return ax


def plot_orbit(body, propagation, plane='xy', ax=None):
    """
    Plots the projection of an orbit on a coordinate plane, together with the
    vertices of the body and the points where the eclipses start and end.

    Args:
        body (Body): The body.
        propagation (Propagation): Result of `EclipseNetPropagator.propagate`.
        plane (str, optional): 'xy', 'xz' or 'yz'. Default 'xy'.
        ax (matplotlib Axes, optional): Axis to plot on. Creates a new one if None.

    Returns:
        matplotlib Axes: The axis with the plotted orbit.
    """
    if ax is None:
        _, ax = plt.subplots()
    i, j = ['xyz'.index(axis) for axis in plane]
    ax.plot(body.mesh_points[:, i], body.mesh_points[:, j], '.', markersize=1., alpha=0.2, color='black')
    ax.plot(propagation.states[:, i], propagation.states[:, j], alpha=0.5, color='coral')
    for events, color, label in [(propagation.entries, 'r', 'entry'), (propagation.exits, 'y', 'exit')]:
        points = np.array([event.state for event in events]).reshape(-1, 6)
        ax.scatter(points[:, i], points[:, j], s=10, c=color, label=label, zorder=3)
    ax.set_xlabel(f'{plane[0]} [-]')
    ax.set_ylabel(f'{plane[1]} [-]')
    ax.set_aspect('equal')
    ax.grid(True)
    return ax


def plot_position_errors(times, errors, eclipses=None, ax=None, **kwargs):
    """
    Plots the three components of a position error as a function of time.

    Args:
        times (np.ndarray, shape (K,)): Times.
        errors (np.ndarray, shape (K,3)): Position errors.
        eclipses (np.ndarray, shape (E,2), optional): Start and end times of
            the eclipses, shown as gray bands.
        ax (matplotlib Axes, optional): Axis to plot on. Creates a new one if None.
        **kwargs: Additional arguments passed to `ax.plot`.

    Returns:
        matplotlib Axes: The axis with the plotted errors.
    """
    if ax is None:
        _, ax = plt.subplots()
    for i, label in enumerate('xyz'):
        ax.plot(times, errors[:, i], label=label, **kwargs)
    if eclipses is not None:
        for n, (start, end) in enumerate(eclipses):
            ax.axvspan(start, end, alpha=0.9, color='gray', label='eclipse' if n == 0 else None)
    ax.set_xlabel('time [-]')
    ax.set_ylabel('error [m]')
    ax.grid(True)
    return ax
