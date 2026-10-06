These pickles contain vertices and triangles of
the triangular mesh defining the asteroid surface.

Usage (example): 

```
import pickle as pk
with open("3dmeshes/Eros_raw.pk", "rb") as f:
    vertices, triangles = pk.load(f)
```

vertices is (N, 3) and contains the Cartesian coordinates of points.
triangles is (M, 3) and contains the indexes of three vertices belonging to a triangle of the mesh)

The files *_raw are direct exports from the original 3d model.
The files *_lp contain downsized models with ~1/10 of the triangles.

The units are km, except for Churyumov-Gerasimenko, which is in m.
`eclipsenets.load_mesh` returns the meshes normalized by their extent along x.
