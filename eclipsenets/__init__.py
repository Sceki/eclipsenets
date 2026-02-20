import os
import torch
import warnings

from .ray_tracing import (
    is_intersecting_v1, 
    is_intersecting_v2, 
    ray_triangle_intersect, 
    is_outside_v1, 
    is_outside_v2
)
# from ._plots import (
#     plot_mesh_vertices, 
#     plot_ray, 
#     plot_intersecting_triangles
# )
from .eclipse_function import (
    eclipse_function, 
    compute_eclipse_function
)
# from ._sampling import (
#     fibonacci_sphere, 
#     sample_points
# )
# from ._utils import (
#     cart2spherical, 
#     project_on_plane, 
#     plane_to_3D, 
#     sph2cart, 
#     project_to_2d, 
#     alpha_shape, 
#     sort_points_along_boundary, 
#     nearest_neighbor
# )
from .nn import (
    SIREN, 
    weights_and_biases_heyoka, 
    FFNN
)

# Device management (modern PyTorch approach)
def get_default_device(device_id: int = 0) -> torch.device:
    """
    Returns the preferred device (CUDA if available, else CPU).
    
    Args:
        device_id: CUDA device index to use if available.
    
    Returns:
        torch.device object.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")

def enable_gpu(device_id: int = 0) -> None:
    """
    Sets global default device to GPU if available.
    Use before creating any tensors/models for automatic allocation.
    
    Args:
        device_id: CUDA device index.
    
    Raises:
        RuntimeError: If CUDA requested but unavailable.
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        torch.set_default_device(device)
        print(f"GPU enabled: {torch.cuda.get_device_name(device_id)} "
              f"(device {device_id}, PyTorch {torch.__version__}, "
              f"cuDNN {torch.backends.cudnn.version()})")
    else:
        print("GPU unavailable, using CPU.")
