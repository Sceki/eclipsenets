import torch

from .ray_tracing import (
    is_intersecting_v1,
    is_intersecting_v2,
    ray_triangle_intersect,
    is_eclipsed,
    is_outside_v1,
    is_outside_v2
)
from .utils import (
    cart2spherical,
    project_on_plane,
    plane_basis,
    plane_to_3D,
    sph2cart,
    project_to_2d,
    encode_direction,
    network_inputs,
    sort_points_along_boundary,
    nearest_neighbor
)
from .silhouette import (
    alpha_shape,
    silhouette_polygon,
    silhouette_border,
    sample_border
)
from .eclipse_function import (
    eclipse_function,
    compute_eclipse_function
)
from .sampling import (
    fibonacci_sphere,
    sample_points
)
from .bodies import (
    BODIES,
    Body,
    load_body,
    load_mesh,
    load_mascons
)
from .dataset import (
    sample_view,
    generate_dataset,
    save_dataset,
    load_dataset,
    published_dataset_path
)
from .nn import (
    SIREN,
    FFNN,
    count_parameters,
    weights_and_biases_heyoka,
    heyoka_model,
    load_model,
    load_pretrained,
    pretrained_path
)
from .training import (
    TrainingConfig,
    build_model,
    evaluate_mse,
    train
)
from .dynamics import (
    EclipseNetPropagator,
    Propagation,
    EclipseEvent,
    propagate_ray_tracing,
    initial_state,
    sun_direction_at,
    compile_eclipsenet,
    is_eclipsed_net
)
from .neural_ode import (
    event_function,
    state_sensitivities,
    refine
)
from .plots import (
    plot_mesh_vertices,
    plot_ray,
    plot_intersecting_triangles,
    plot_silhouette,
    plot_orbit,
    plot_position_errors
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
    Sets global default device to GPU if available, and prints which device is used.
    Use before creating any tensors/models for automatic allocation.

    Args:
        device_id: CUDA device index.
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        torch.set_default_device(device)
        print(f"GPU enabled: {torch.cuda.get_device_name(device_id)} "
              f"(device {device_id}, PyTorch {torch.__version__}, "
              f"cuDNN {torch.backends.cudnn.version()})")
    else:
        print("GPU unavailable, using CPU.")
