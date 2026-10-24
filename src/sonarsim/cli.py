"""Command line: ``sonarsim-generate`` writes a labelled dataset."""
from __future__ import annotations

import argparse

from sonarsim.autoplacer import PlacementBounds
from sonarsim.dataset import generate_dataset
from sonarsim.environment import Environment
from sonarsim.sonar import PRESETS, preset
from sonarsim.target import Mesh, Target


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="sonarsim-generate",
                                description="Generate a labelled active-sonar dataset.")
    p.add_argument("--mesh", required=True, help="target mesh (.obj; STL/GLTF need trimesh)")
    p.add_argument("--scale", type=float, default=1.0, help="mesh units to metres")
    p.add_argument("--up-axis", default="+y", help="mesh file's up axis (default +y)")
    p.add_argument("--front-axis", default="+z", help="mesh file's front axis (default +z)")
    p.add_argument("--preset", default="A_HF_imaging", choices=sorted(PRESETS))
    p.add_argument("-n", "--num", type=int, default=10, help="number of pings")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth", type=float, default=40.0, help="water depth (m)")
    p.add_argument("--max-bounces", type=int, default=None,
                   help="boundary reflections per one-way target path")
    p.add_argument("--save-raw", action="store_true", help="also store element signals")
    p.add_argument("--out", required=True, help="output directory")
    args = p.parse_args(argv)

    sonar = preset(args.preset)
    if args.max_bounces is not None:
        sonar = sonar.with_overrides(max_bounces=args.max_bounces)
    mesh = Mesh.from_file(args.mesh, args.scale, args.up_axis, args.front_axis)
    target = Target(mesh)
    print(f"{sonar.name}: {mesh.n_facets} facets, extent {mesh.extent.round(2)} m")
    generate_dataset(sonar, Environment(depth=args.depth), target, args.out, args.num,
                     seed=args.seed, bounds=PlacementBounds(), save_raw=args.save_raw)


if __name__ == "__main__":
    main()
