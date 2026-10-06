import numpy as np

from .utils import cart2spherical


def fibonacci_sphere(samples=1000):
    """
    Generates points uniformly distributed on the surface of a sphere using the Fibonacci lattice method.
    Args:
        samples: Number of points to generate.
    Returns:
        points: Array of shape (samples, 3) containing the 3D coordinates of the points on the sphere.
        az: Array of shape (samples,) containing the azimuthal angles of the points.
        el: Array of shape (samples,) containing the elevation angles of the points
            (polar angles, measured from +z).
    """
    if samples < 1:
        raise ValueError("samples must be at least 1")
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    i = np.arange(samples)
    # y goes from 1 to -1 (a single sample sits at y = 1)
    y = 1 - (i / float(max(samples - 1, 1))) * 2
    radius = np.sqrt(1 - y * y)  # radius at y
    theta = phi * i  # golden angle increment

    points = np.stack([np.cos(theta) * radius, y, np.sin(theta) * radius], axis=1)
    spherical = cart2spherical(points)
    return points, spherical[:, 1], spherical[:, 2]


def sample_points(center_points, r=0.1, num_samples_per_point=1000, seed=None):
    """
    Sample points within a circle of radius r around each center point.

    The distance from the center is drawn uniformly in [0, r], so the samples
    are denser towards the center than a uniform distribution over the disc.

    Args:
        center_points: Array of shape (N, 2) containing the 2D coordinates of the center points.
        r: Radius of the circle around each center point.
        num_samples_per_point: Number of points to sample around each center point.
        seed: Random seed, or a numpy Generator, for reproducibility (optional).
    Returns:
        new_points: Array of shape (N * num_samples_per_point, 2) containing the sampled points.
    """
    rng = np.random.default_rng(seed)
    center_points = np.asarray(center_points, dtype=float).reshape(-1, 2)
    shape = (len(center_points), num_samples_per_point)
    theta = rng.uniform(0, 2 * np.pi, shape)
    r_ = rng.uniform(0, r, shape)
    offsets = np.stack([r_ * np.cos(theta), r_ * np.sin(theta)], axis=-1)
    return (center_points[:, None, :] + offsets).reshape(-1, 2)
