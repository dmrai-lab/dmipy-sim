"""Walk the two small cases of the walk's accumulation and write what the walk returns: a 10 um slab (positions only, K
= 60 over 100 ms, 2000 walkers, seed 7) and a one-cylinder periodic cell (every tier, so the surface channel is
recorded, K = 40 over 20 ms, 1000 walkers, seed 3, 50 raw walkers).

    python tools/accumulation_dtype.py --out DIR      # writes DIR/slab.npz and DIR/cell.npz and prints their dtypes

Each ``.npz`` holds every array of the :class:`~dmipy_sim.coefficient_walk.CoefficientWalk`; the printout states each
array's dtype and shape and the walk's ``block.precision``. Run from the repository root on the CPU
(``JAX_PLATFORMS=cpu``)."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

#: the arrays of a walk the cases write
ARRAYS = ("position_coeffs", "qv_positions", "boundary_endpoint", "boundary_bands", "qv_boundary", "compartment",
          "raw_positions", "raw_boundary", "raw_compartment")


def cases():
    """``{name: (walk keywords)}``: the slab of the accumulation's smoke run and the cell with a surface channel."""
    import dmipy_sim as d
    from dmipy_sim.geometry.base import Box1D
    return {
        "slab": dict(n_walkers=2000, diffusivity=2e-9, geometry=Box1D(length=10e-6), T_max=0.1, seed=7, K=60, tiers=()),
        "cell": dict(n_walkers=1000, diffusivity=2e-9, geometry=d.PackedCylinders([2e-6], [[0.0, 0.0]], 8e-6), T_max=0.02,
                     seed=3, K=40, tiers="all", raw_subset=50),
    }


def walk_case(kw):
    """The walk of one case, on the CPU."""
    from dmipy_sim.engine.core import _walk_geometry
    kw = dict(kw)
    return _walk_geometry(kw.pop("n_walkers"), kw.pop("diffusivity"), kw.pop("geometry"), kw.pop("T_max"),
                          require_gpu=False, **kw)


def arrays_of(walk):
    """``{name: array}`` of the walk's arrays that are present."""
    return {k: np.asarray(getattr(walk, k)) for k in ARRAYS if getattr(walk, k) is not None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, help="the directory the .npz files are written to")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for name, kw in cases().items():
        w = walk_case(kw)
        arrs = arrays_of(w)
        np.savez(os.path.join(a.out, f"{name}.npz"), **arrs)
        print(f"{name}: block.precision {w.block['precision']}")
        for k, v in arrs.items():
            print(f"  {k:18s} {str(v.dtype):8s} {v.shape}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
