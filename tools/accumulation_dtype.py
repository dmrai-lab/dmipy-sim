"""Walk the two small cases of the walk's accumulation in float32 and in float64 and measure the float32 accumulation's
rounding: a 10 um slab (positions only, K = 60 over 100 ms, 2000 walkers, seed 7) and a one-cylinder periodic cell
(every tier, so the surface channel is recorded, K = 40 over 20 ms, 1000 walkers, seed 3, 50 raw walkers).

    python tools/accumulation_dtype.py --out DIR                       # float32: DIR/float32/{slab,cell}.npz
    python tools/accumulation_dtype.py --out DIR --dtype both          # and float64, and the rounding of float32
    python tools/accumulation_dtype.py --out DIR --compare OLD         # and each float32 array against OLD/<case>.npz

Each ``.npz`` holds every array of the :class:`~dmipy_sim.coefficient_walk.CoefficientWalk`; the printout states each
array's dtype and shape and the walk's ``block.precision``. With ``--dtype both`` the script turns JAX's x64 mode on in
its own process (the float64 accumulation needs it; the float32 walk does not read it) and prints, per case, whether
the two calls walked the same walkers (the raw subset's positions and the starts, equal to the bit) and the rounding:
per band ``k`` of ``position_coeffs`` (``k = 0`` the start, ``1`` the displacement, ``2 ..`` the bands), over the
walkers and axes, the largest and the root-mean-square ``|float32 - float64|``, each also over the band's own
root-mean-square, and the same for ``qv_positions`` and, when recorded, the contact. ``--compare`` states, per array,
whether the float32 call returned the bytes of an earlier output. Run from the repository root on the CPU
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


def walk_case(kw, dtype):
    """The walk of one case accumulated in ``dtype``, on the CPU."""
    from dmipy_sim.engine.core import _walk_geometry
    kw = dict(kw)
    return _walk_geometry(kw.pop("n_walkers"), kw.pop("diffusivity"), kw.pop("geometry"), kw.pop("T_max"),
                          require_gpu=False, accumulation_dtype=dtype, **kw)


def arrays_of(walk):
    """``{name: array}`` of the walk's arrays that are present."""
    return {k: np.asarray(getattr(walk, k)) for k in ARRAYS if getattr(walk, k) is not None}


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def rounding(a32, a64):
    """The rounding of the float32 accumulation against the float64 one, as printed lines."""
    lines = []
    C32, C64 = np.asarray(a32["position_coeffs"], np.float64), a64["position_coeffs"]
    lines.append("  position_coeffs per band k (k = 0 start, 1 displacement, 2.. bands): "
                 "max |d| m, rms |d| m, max |d| / rms band, rms |d| / rms band")
    for k in range(C64.shape[2]):
        dk = (C32[:, :, k, :] - C64[:, :, k, :]).ravel()
        scale = rms(C64[:, :, k, :])
        lines.append(f"    k={k:3d}  {np.abs(dk).max():.3e}  {rms(dk):.3e}  {np.abs(dk).max() / scale:.3e}  {rms(dk) / scale:.3e}")
    d_all = (C32[:, :, 2:, :] - C64[:, :, 2:, :]).ravel()
    lines.append(f"  bands k >= 2 together: max |d| {np.abs(d_all).max():.3e} m, rms {rms(d_all):.3e} m, "
                 f"max |d| / max |band| {np.abs(d_all).max() / np.abs(C64[:, :, 2:, :]).max():.3e}")
    for name in ("qv_positions", "boundary_endpoint", "boundary_bands", "qv_boundary"):
        if name in a64:
            d = (np.asarray(a32[name], np.float64) - a64[name]).ravel()
            lines.append(f"  {name}: max |d| {np.abs(d).max():.3e}, rms {rms(d):.3e}, rms of the array {rms(a64[name]):.3e}")
    return lines


def same_walkers(a32, a64):
    """Whether the two calls walked the same walkers: the starts, and the raw subset's positions when kept."""
    same = {"starts": bool(np.array_equal(a32["position_coeffs"][:, :, 0, :], a64["position_coeffs"][:, :, 0, :]))}
    for name in ("raw_positions", "raw_boundary", "raw_compartment", "compartment"):
        if name in a32:
            same[name] = bool(np.array_equal(a32[name], a64[name]))
    return same


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, help="the directory the .npz files are written to")
    p.add_argument("--dtype", choices=("float32", "float64", "both"), default="float32")
    p.add_argument("--compare", help="a directory of <case>.npz the float32 arrays are compared with, byte for byte")
    a = p.parse_args(argv)
    dtypes = ("float32", "float64") if a.dtype == "both" else (a.dtype,)
    if "float64" in dtypes:
        import jax
        jax.config.update("jax_enable_x64", True)
    out = {}
    for name, kw in cases().items():
        for dt in dtypes:
            w = walk_case(kw, dt)
            arrs = arrays_of(w)
            os.makedirs(os.path.join(a.out, dt), exist_ok=True)
            np.savez(os.path.join(a.out, dt, f"{name}.npz"), **arrs)
            out[name, dt] = arrs
            print(f"{name} {dt}: block.precision {w.block['precision']}")
            for k, v in arrs.items():
                print(f"  {k:18s} {str(v.dtype):8s} {v.shape}")
            if a.compare and dt == "float32":
                with np.load(os.path.join(a.compare, f"{name}.npz")) as old:
                    for k in old.files:
                        print(f"  {k:18s} the bytes of {a.compare}: {old[k].dtype == arrs[k].dtype and np.array_equal(old[k], arrs[k])}")
        if len(dtypes) == 2:
            print(f"{name}: the same walkers: {same_walkers(out[name, 'float32'], out[name, 'float64'])}")
            print("\n".join(rounding(out[name, "float32"], out[name, "float64"])))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
