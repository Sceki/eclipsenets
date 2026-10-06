"""Silhouette (shadow outline) of a triangular mesh seen from a given direction."""
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon
from scipy.spatial import Delaunay

from .utils import project_to_2d


def _alpha_triangles(points, alpha):
    """Delaunay triangles of `points` whose circumradius is smaller than `alpha`."""
    points = np.asarray(points, dtype=float)
    assert points.shape[0] > 3, "Need at least four points"
    assert points.shape[1] == 2, "Need two dimensional points"
    simplices = Delaunay(points).simplices
    pa, pb, pc = points[simplices[:, 0]], points[simplices[:, 1]], points[simplices[:, 2]]
    a = np.linalg.norm(pa - pb, axis=1)
    b = np.linalg.norm(pb - pc, axis=1)
    c = np.linalg.norm(pc - pa, axis=1)
    s = (a + b + c) / 2.0
    # Heron's formula; the product can be slightly negative for flat triangles.
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        circum_r = a * b * c / (4.0 * area)
    return simplices[circum_r < alpha]


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
    simplices = _alpha_triangles(points, alpha)
    edges = np.concatenate([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]])
    # An edge shared by two triangles of the shape shows up twice (once per
    # direction): it is interior. Border edges show up once.
    _, first, inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_index=True, return_inverse=True, return_counts=True)
    if only_outer:
        edges = edges[counts[inverse.ravel()] == 1]
    else:
        edges = edges[first]
    return {(int(i), int(j)) for i, j in edges}


def _clean(geometry, min_area):
    """Drops the sliver parts and sliver holes left behind by a polygon union."""
    parts = list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]
    cleaned = [
        Polygon(part.exterior, [hole for hole in part.interiors if Polygon(hole).area > min_area])
        for part in parts if part.area > min_area
    ]
    if not cleaned:
        raise ValueError("The silhouette is empty: is the mesh degenerate?")
    return cleaned[0] if len(cleaned) == 1 else MultiPolygon(cleaned)


def silhouette_polygon(mesh_points, mesh_triangles, direction, method="exact", alpha=None,
                       cull_backfaces=False):
    """
    Region eclipsed by a mesh lit from `direction`, on the plane orthogonal to it.

    Coordinates on the plane are the ones of :func:`eclipsenets.project_on_plane`.

    Args:
        mesh_points (np.array of shape (N,3)): mesh vertices.
        mesh_triangles (np.array of shape (M,3)): indices of the triangle vertices.
        direction (np.array of shape (3,)): direction of the light source.
        method (str): "exact" takes the union of the projected triangles;
            "alpha" takes the alpha shape of the projected vertices, as done
            for the datasets of the paper (it needs an `alpha` tuned per body:
            too small and the outline breaks up, too large and concavities
            are filled in).
        alpha (float): alpha value, only for method="alpha".
        cull_backfaces (bool): only for method="exact": use only the triangles
            facing one way, which halves the cost but requires a closed and
            consistently oriented mesh.

    Returns:
        shapely Polygon, or a MultiPolygon if the shadow is made of several
        pieces (a bump joined to the rest through a single point is one).
    """
    mesh_points = np.asarray(mesh_points, dtype=float)
    mesh_triangles = np.asarray(mesh_triangles)
    direction = np.asarray(direction, dtype=float)
    points = project_to_2d(mesh_points, direction)
    extent = np.ptp(points, axis=0).max()

    if method == "alpha":
        if alpha is None:
            raise ValueError("method='alpha' needs an alpha value")
        triangles = points[_alpha_triangles(points, alpha)]
        if len(triangles) == 0:
            raise ValueError(f"alpha={alpha} is too small: no triangle is left")
        # Holes are artefacts here: they open up wherever the projected
        # vertices are sparser than alpha.
        union = shapely.union_all(shapely.polygons(triangles), grid_size=1e-9 * extent)
        parts = list(union.geoms) if isinstance(union, MultiPolygon) else [union]
        return Polygon(max(parts, key=lambda part: part.area).exterior)
    if method != "exact":
        raise ValueError(f"Unknown method {method!r}: use 'exact' or 'alpha'")

    triangles = points[mesh_triangles]
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    area2 = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    # triangles seen edge-on project to segments and do not contribute
    keep = np.abs(area2) > 1e-12 * extent**2
    if cull_backfaces:
        # The basis (U, V, -direction) is right-handed, so the sign of the
        # projected area tells which way a triangle faces. Either half of a
        # closed mesh covers the whole silhouette.
        keep &= area2 > 0
    # Snapping to a grid keeps the union robust against the nearly coincident
    # edges of neighbouring triangles.
    union = shapely.union_all(shapely.polygons(triangles[keep]), grid_size=1e-9 * extent)
    return _clean(union, min_area=1e-8 * extent**2)


def _rings(surface):
    """The closed lines that make up the outline of a silhouette."""
    parts = list(surface.geoms) if isinstance(surface, MultiPolygon) else [surface]
    return [ring for part in parts for ring in (part.exterior, *part.interiors)]


def silhouette_border(surface):
    """
    Vertices of the outline of a silhouette.

    Args:
        surface (shapely Polygon or MultiPolygon): the silhouette.

    Returns:
        np.array of shape (K,2): the vertices of all the rings of the outline.
    """
    # the last vertex of a ring repeats the first
    return np.vstack([np.asarray(ring.coords)[:-1] for ring in _rings(surface)])


def sample_border(surface, n, seed=None):
    """
    Draws points uniformly along the outline of a silhouette.

    Args:
        surface (shapely Polygon or MultiPolygon): the silhouette.
        n (int): number of points.
        seed: random seed or numpy Generator.

    Returns:
        np.array of shape (n,2)
    """
    rng = np.random.default_rng(seed)
    rings = np.array(_rings(surface))
    lengths = shapely.length(rings)
    ring = rng.choice(len(rings), size=n, p=lengths / lengths.sum())
    points = shapely.line_interpolate_point(rings[ring], rng.uniform(0, lengths[ring]))
    return shapely.get_coordinates(points)
