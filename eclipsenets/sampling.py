import numpy as np

def fibonacci_sphere(samples=1000):
    """
    Generates points uniformly distributed on the surface of a sphere using the Fibonacci lattice method.
    Args:
        samples: Number of points to generate.
    Returns:
        points: Array of shape (samples, 3) containing the 3D coordinates of the points on the sphere.
        az: Array of shape (samples,) containing the azimuthal angles of the points.
        el: Array of shape (samples,) containing the elevation angles of the points.
    """
    
    from .utils import cart2spherical
    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))
    az,el=cart2spherical(points)[:,1], cart2spherical(points)[:,2]

    return np.array(points),az,el
def sample_points(center_points, r=0.1, num_samples_per_point=1000,seed=None):
    """
    Sample points uniformly within a circle of radius r around each center point.
    Args:
        center_points: Array of shape (N, 2) containing the 2D coordinates of the center points.
        r: Radius of the circle around each center point.
        num_samples_per_point: Number of points to sample around each center point.
        seed: Random seed for reproducibility (optional).
    Returns:
        new_points: Array of shape (N * num_samples_per_point, 2) containing the sampled points.
    """
    np.random.seed(seed)
    new_points=np.zeros((int(len(center_points)*num_samples_per_point),2))
    k=0
    for center in center_points:
        theta = np.random.uniform(0, 2 * np.pi,num_samples_per_point)
        r_=np.random.uniform(0,r,num_samples_per_point)
        new_points[k:k+num_samples_per_point,0]=center[0]+r_*np.cos(theta)
        new_points[k:k+num_samples_per_point,1]=center[1]+r_*np.sin(theta)
        k+=num_samples_per_point
    return new_points


