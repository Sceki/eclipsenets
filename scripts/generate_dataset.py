#!/usr/bin/env python3
"""Generates a dataset of eclipse function samples for a body.

The paper uses 500 Sun directions for training and 200 for validation:

    python scripts/generate_dataset.py --body Itokawa --views 500 --output datasets/generated/itokawa_train.pk
    python scripts/generate_dataset.py --body Itokawa --views 200 --output datasets/generated/itokawa_valid.pk
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eclipsenets as en


def main():
    parser = argparse.ArgumentParser(description='Generate a dataset of eclipse function samples')
    parser.add_argument('--body', type=str, required=True)
    parser.add_argument('--views', type=int, default=500, help='number of Sun directions')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--method', type=str, default='exact', choices=['exact', 'alpha'],
                       help='how the silhouettes are extracted')
    parser.add_argument('--alpha', type=float, default=None, help='alpha value, for --method alpha')
    args = parser.parse_args()

    mesh_points, mesh_triangles, _ = en.load_mesh(args.body)
    # The meshes of the repository are closed and consistently oriented, so
    # half of the triangles are enough to get the silhouettes.
    data = en.generate_dataset(mesh_points, mesh_triangles, n_views=args.views, method=args.method,
                               alpha=args.alpha, cull_backfaces=True, seed=args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    en.save_dataset(data, output)
    print(f"Dataset with {len(data):,} samples saved to {output}")


if __name__ == '__main__':
    main()
