"""The four small bodies of the paper: shape models, mascon models and physical constants."""
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

# Root of the repository, where the 3dmeshes/, mascons/, datasets/ and models/ folders live.
DATA_DIR = Path(__file__).resolve().parent.parent

G = 6.674e-11  # Cavendish constant [m^3 kg^-1 s^-2]


@dataclass(frozen=True)
class BodyInfo:
    name: str               # name used by the files in 3dmeshes/ and mascons/
    key: str                # name used by the files in datasets/ and models/
    mass: float             # [kg]
    rotation_period: float  # [h]
    mesh_unit: float        # length of one unit of the raw mesh [m]
    # Factor that brings the low-poly mascon model to the units of the mesh.
    lp_mascon_scale: float = 1.0


BODIES = {
    "Bennu": BodyInfo("Bennu", "bennu", 7.329e10, 4.296, 1e3),
    "Itokawa": BodyInfo("Itokawa", "itokawa", 3.51e10, 12.132, 1e3),
    "Churyumov-Gerasimenko": BodyInfo("Churyumov-Gerasimenko", "chu-ger", 9.982e12, 12.4043, 1.0),
    # mascons/Eros_lp.pk is the only mascon model that is not stored in units
    # of L: it is about 1.67 times larger than the mesh. The factor is the one
    # for which its gravity field best matches the one of mascons/Eros.pk:
    # the two then differ as little as the low-poly and full models of the
    # other bodies do.
    "Eros": BodyInfo("Eros", "eros", 6.687e15, 5.27, 1e3, lp_mascon_scale=1 / 1.6705),
}
_ALIASES = {"67p": "Churyumov-Gerasimenko"}


def get_body_info(name: str) -> BodyInfo:
    """Looks up a body by name ("Bennu"), by key ("chu-ger") or by alias ("67P")."""
    lowered = name.lower()
    for info in BODIES.values():
        if lowered in (info.name.lower(), info.key):
            return info
    if lowered in _ALIASES:
        return BODIES[_ALIASES[lowered]]
    raise ValueError(f"Unknown body {name!r}: available bodies are {list(BODIES)}")


class _PlainUnpickler(pickle.Unpickler):
    """The mascon files store pyvista arrays: read them as plain numpy arrays,
    so that pyvista does not have to be installed."""

    def find_class(self, module, name):
        if module.startswith("pyvista"):
            return np.ndarray
        return super().find_class(module, name)


def load_mesh(name: str, low_poly: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Loads the triangular mesh of a body.

    Args:
        name: name of the body.
        low_poly: if True, loads the mesh with ~1/10 of the triangles.

    Returns:
        the vertices (N, 3) in units of L, the triangles (M, 3) and L in meters.
        L, the extent of the full mesh along x, is the unit of length of the
        datasets, of the networks and of the mascon models.
    """
    info = get_body_info(name)

    def read(suffix):
        with open(DATA_DIR / "3dmeshes" / f"{info.name}_raw{suffix}.pk", "rb") as f:
            points, triangles = pickle.load(f)
        return np.array(points, dtype=float), np.array(triangles)

    points, triangles = read("")
    extent = np.ptp(points[:, 0])
    if low_poly:
        points, triangles = read("_lp")
    return points / extent, triangles, float(extent * info.mesh_unit)


def load_mascons(name: str, low_poly: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the mascon model of a body.

    Args:
        name: name of the body.
        low_poly: if True, loads the model with fewer mascons (3k-9k instead of 57k-100k).

    Returns:
        the mascon positions (N, 3) in units of L and the mascon masses (N,),
        which sum to one.
    """
    info = get_body_info(name)
    suffix = "_lp" if low_poly else ""
    with open(DATA_DIR / "mascons" / f"{info.name}{suffix}.pk", "rb") as f:
        points, masses, _ = _PlainUnpickler(f).load()
    points = np.array(points, dtype=float)
    if low_poly:
        points = points * info.lp_mascon_scale
    return points, np.array(masses, dtype=float)


@dataclass
class Body:
    """A small body, in the non-dimensional units used throughout the library:
    lengths in units of L, masses in units of the body mass and times in units
    of T, so that the gravitational parameter of the body is one."""
    name: str
    key: str
    mass: float                  # [kg]
    rotation_period: float       # [h]
    L: float                     # unit of length [m]
    mesh_points: np.ndarray      # (N, 3)
    mesh_triangles: np.ndarray   # (M, 3)
    mascon_points: np.ndarray    # (K, 3)
    mascon_masses: np.ndarray    # (K,)

    @property
    def T(self) -> float:
        """Unit of time [s]."""
        return float(np.sqrt(self.L**3 / (G * self.mass)))

    @property
    def omega(self) -> float:
        """Angular velocity of the body around its z axis, in units of 1/T."""
        return 2 * np.pi / (self.rotation_period * 3600) * self.T

    @property
    def radius(self) -> float:
        """Radius of the sphere centered in the origin that contains the body."""
        return float(np.linalg.norm(self.mesh_points, axis=1).max())

    @property
    def triangle_vertices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The vertices (v0, v1, v2) of the triangles, each of shape (M, 3)."""
        triangles = self.mesh_points[self.mesh_triangles]
        return triangles[:, 0], triangles[:, 1], triangles[:, 2]


def load_body(name: str, low_poly_mesh: bool = False, low_poly_mascons: bool = True) -> Body:
    """
    Loads the shape model, the mascon model and the physical constants of a body.

    Args:
        name: "Bennu", "Itokawa", "Churyumov-Gerasimenko" (or "67P") or "Eros".
        low_poly_mesh: if True, uses the mesh with ~1/10 of the triangles.
        low_poly_mascons: if True, uses the mascon model with fewer mascons.

    Returns:
        Body
    """
    info = get_body_info(name)
    mesh_points, mesh_triangles, L = load_mesh(name, low_poly=low_poly_mesh)
    mascon_points, mascon_masses = load_mascons(name, low_poly=low_poly_mascons)
    return Body(info.name, info.key, info.mass, info.rotation_period, L,
                mesh_points, mesh_triangles, mascon_points, mascon_masses)
