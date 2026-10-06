"""Datasets of eclipse function samples.

A dataset is an (N, 7) array with columns
[cos(az), sin(az), cos(el), sin(el), X, Y, F]: the direction of the Sun, a
point on the plane orthogonal to it and the value of the eclipse function.
"""
import pickle
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .bodies import DATA_DIR, get_body_info
from .eclipse_function import compute_eclipse_function
from .sampling import fibonacci_sphere, sample_points
from .silhouette import sample_border, silhouette_polygon
from .utils import encode_direction

# Training and validation sets stored in datasets/ (with git lfs). On the
# ones of Bennu, 67P and Eros the networks in models/ reach the losses written
# in their file names. The ones of Itokawa stored before had a wrong
# silhouette in almost half of their views: they have been generated again, with
#     scripts/generate_dataset.py --body Itokawa --views 500 --seed 0
#     scripts/generate_dataset.py --body Itokawa --views 200 --seed 1
PUBLISHED_DATASETS = {
    "bennu": {"train": "bennu_db_2248220_L=0.5634379088878632_larger_train.pk",
              "valid": "bennu_db_899424_L=0.5634379088878632_valid.pk"},
    "itokawa": {"train": "itokawa_db_2250000_L=0.5607019066810608_larger_train.pk",
                "valid": "itokawa_db_900000_L=0.5607019066810608_valid.pk"},
    "chu-ger": {"train": "chu-ger_db_2349414_L=5002.5703125_larger_train.pk",
                "valid": "chu-ger_db_939678_L=5002.5703125_valid.pk"},
    "eros": {"train": "eros_db_2267612_L=32.66218376159668_larger_train.pk",
             "valid": "eros_db_907150_L=32.66218376159668_valid.pk"},
}


def sample_view(surface, n_uniform=1000, n_border=300, n_per_border=10, border_radius=0.009,
                n_on_border=500, margin=0.25, seed=None):
    """
    Samples the eclipse function of one silhouette.

    Three sets of points are drawn: clouds around points of the border, where
    the function has to be most accurate, points spread uniformly over a
    square containing the silhouette, and points on the border itself, where
    the function is zero.

    Args:
        surface (shapely Polygon): the silhouette.
        n_uniform (int): number of points spread over the square.
        n_border (int): number of points of the border around which a cloud is drawn.
        n_per_border (int): number of points of each cloud.
        border_radius (float): radius of each cloud.
        n_on_border (int): number of points on the border.
        margin (float): how far the square extends past the silhouette, as a
            fraction of half the side of the smallest square containing it.
        seed: random seed or numpy Generator.

    Returns:
        the points (N, 2) and the values of the eclipse function (N,).
    """
    rng = np.random.default_rng(seed)

    centers = sample_border(surface, n_border, seed=rng)
    clouds = sample_points(centers, r=border_radius, num_samples_per_point=n_per_border, seed=rng)

    low = min(surface.bounds[:2])
    high = max(surface.bounds[2:])
    pad = margin * (high - low) / 2
    uniform = rng.uniform(low - pad, high + pad, (n_uniform, 2))

    off_border = np.vstack([clouds, uniform])
    samples = np.vstack([off_border, sample_border(surface, n_on_border, seed=rng)])
    values = np.zeros(len(samples))
    values[:len(off_border)] = compute_eclipse_function(off_border, surface=surface)
    return samples, values


def generate_dataset(mesh_points, mesh_triangles, n_views=500, method="exact", alpha=None,
                     cull_backfaces=False, seed=None, progress=True, **sampling):
    """
    Generates a dataset of eclipse function samples for a body.

    The Sun directions are spread over the sphere with a Fibonacci lattice.
    The paper uses 500 directions for training and 200 for validation: the two
    lattices only share their first and last directions, [0, 1, 0] and [0, -1, 0].

    Args:
        mesh_points (np.array of shape (N,3)): mesh vertices, in units of L.
        mesh_triangles (np.array of shape (M,3)): indices of the triangle vertices.
        n_views (int): number of Sun directions.
        method, alpha, cull_backfaces: see :func:`eclipsenets.silhouette_polygon`.
        seed: random seed or numpy Generator.
        progress (bool): whether to show a progress bar.
        **sampling: passed to :func:`sample_view`.

    Returns:
        np.array of shape (N, 7), see the module docstring.
    """
    rng = np.random.default_rng(seed)
    directions, az, el = fibonacci_sphere(n_views)
    blocks = []
    for i in tqdm(range(n_views), disable=not progress):
        surface = silhouette_polygon(mesh_points, mesh_triangles, directions[i], method=method,
                                     alpha=alpha, cull_backfaces=cull_backfaces)
        samples, values = sample_view(surface, seed=rng, **sampling)
        block = np.empty((len(samples), 7))
        block[:, :4] = encode_direction(az[i], el[i])
        block[:, 4:6] = samples
        block[:, 6] = values
        blocks.append(block)
    return np.vstack(blocks)


def save_dataset(data, path):
    """Saves a dataset to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(np.asarray(data), f)


def load_dataset(path):
    """
    Loads a dataset from a pickle file.

    The datasets of the repository are stored with git lfs. If only the
    pointer of a file has been checked out, the file is looked up among the
    objects that git lfs has already downloaded; if it is not there either, an
    error explains how to get it.

    Args:
        path: path of the file.

    Returns:
        np.array of shape (N, 7), see the module docstring.
    """
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(200)
    if head.startswith(b"version https://git-lfs"):
        oid = re.search(rb"oid sha256:([0-9a-f]{64})", head).group(1).decode()
        for parent in path.resolve().parents:
            cached = parent / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
            if cached.exists():
                path = cached
                break
        else:
            raise FileNotFoundError(
                f"{path} is a git lfs pointer: the dataset has not been downloaded. "
                "Install git lfs and run `git lfs pull` in the repository.")
    with open(path, "rb") as f:
        return np.asarray(pickle.load(f))


def published_dataset_path(body, split="train"):
    """
    Path of the training or validation set of a body, stored in datasets/.

    Args:
        body: name of the body.
        split: "train" or "valid".
    """
    return DATA_DIR / "datasets" / PUBLISHED_DATASETS[get_body_info(body).key][split]
