"""``python -m dmipy_sim.spectrum``: write the substrate spectrum (dmrai-lab/dmipy-sim#697) to a directory.

    python -m dmipy_sim.spectrum --class strands --edge 250e-6 --seed 1 --out DIR
    python -m dmipy_sim.spectrum --all --out DIR

``--all`` writes every class in :data:`~dmipy_sim.spectrum.GENERATORS` at every edge of
:data:`~dmipy_sim.spectrum.EDGES_M`, seed 1, one subdirectory per ``<class>/edge_<edge_um>um`` under ``--out``.
"""
import argparse
import json
import logging
import sys

from . import EDGES_M, GENERATORS
from .common import WALK_LENGTHS_S


def _fmt_edge(edge_m):
    if edge_m >= 1e-3:
        return f"{edge_m * 1e3:g}mm"
    return f"{edge_m * 1e6:g}um"


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m dmipy_sim.spectrum", description=__doc__.split("\n\n")[0])
    p.add_argument("--class", dest="cls", choices=sorted(GENERATORS), help="which generator (omit with --all)")
    p.add_argument("--edge", type=float, help="the domain edge, metres (e.g. 250e-6)")
    p.add_argument("--seed", type=int, default=1, help="the generator seed (default 1)")
    p.add_argument("--out", required=True, help="the output directory")
    p.add_argument("--all", action="store_true", help="every class at every edge of EDGES_M, seed 1")
    p.add_argument("--walk-lengths", type=float, nargs="+", default=list(WALK_LENGTHS_S),
                  help="walk lengths (s) to attempt per entry (default: the spectrum's own 20/100 ms)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    jobs = []
    if args.all:
        for cls in sorted(GENERATORS):
            for edge in EDGES_M:
                jobs.append((cls, edge, args.seed))
    else:
        if not args.cls or args.edge is None:
            p.error("pass --class and --edge, or --all")
        jobs.append((args.cls, args.edge, args.seed))

    results = []
    for cls, edge, seed in jobs:
        out_dir = f"{args.out}/{cls}/edge_{_fmt_edge(edge)}_seed{seed}"
        print(f"== {cls} edge={edge:g} m seed={seed} -> {out_dir}", file=sys.stderr)
        r = GENERATORS[cls](edge, seed, out_dir=out_dir, T_max_s=tuple(args.walk_lengths))
        print(json.dumps({k: v for k, v in r.items() if k != "index"}, default=str, indent=1))
        results.append(r)
    return results


if __name__ == "__main__":
    main()
