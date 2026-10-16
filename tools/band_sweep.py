"""The band-container sweep of #727: which bit-depth curve a pack's position bands need.

One cylinder walk is encoded under the two-level control container and under ``band_curve(K, b1)`` for each ``b1``.
Each pack is measured on three things: its bytes per walker, its certificate (``err_max`` against ``floor_max``, and
the container's own error against the float-container pack of the same walk), and the replayed rows of
``tools/gen_step_law.py`` against both the float pack's replay and the exact propagator of the pore
(``SteppedPore.continuous``). The candidate curve is the one whose container error sits at or under half the floor.

    JAX_PLATFORMS=cpu python tools/band_sweep.py --n 20000 --b1 16 14 13 12 11 10 --out sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

D0, T, DT, K = 2.0e-9, 0.100, 1.0e-4, 256
DIAM = 5.0e-6


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def bytes_per_walker(pack):
    """Bytes per walker of the position bands (every ``pos_*`` tensor but the exact endpoints) and of the whole pack."""
    n = pack.n_walkers
    pos = sum(np.asarray(v).nbytes for k, v in pack.arrays.items() if k.startswith("pos_") and not k.endswith("_ends"))
    tot = sum(np.asarray(v).nbytes for v in pack.arrays.values())
    return pos / n, tot / n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--b1", type=float, nargs="+", default=[16, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--blt-K", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    import dmipy_sim as d
    from dmipy_sim.math.step_law import SteppedPore
    from dmipy_sim.replay import compression as cx
    from dmipy_sim.replay.bank import build_replay_pack
    import gen_step_law as gl

    log(f"walk: {a.n} walkers, {DIAM * 1e6:.0f} um cylinder, D {D0}, T {T}, dt_save {DT}")
    geom = d.Cylinder(DIAM / 2, (0.0, 0.0, 1.0))
    walk = d.simulate_trajectories(a.n, D0, geom, T, DT, seed=a.seed, require_gpu=False, tiers="all")
    rows = gl.rows()
    exact = {k: SteppedPore("cylinder", DIAM, D0, geom, n_modes=(8, 8)).continuous(gl.g_axis(s), DT) for k, s in rows.items()}
    log("exact:", exact)

    def build(label, container):
        t0 = time.time()
        pk = build_replay_pack(walk, id=f"sweep/{label}", K=K, license="x", citation="x", blt_temporal_K=a.blt_K,
                               position_container=container, blt_container=("bands" if container is not None else None))
        pos_b, tot_b = bytes_per_walker(pk)
        sig = {k: float(np.abs(np.asarray(pk.replay(s))).ravel()[0]) for k, s in rows.items()}
        log(f"{label}: built + replayed in {time.time() - t0:.0f} s")
        return pk, pos_b, tot_b, sig

    ref, pos_f, tot_f, sig_f = build("float", None)
    n_t = int(ref.n_t)
    raw = cx.decoder(ref.arrays, ref.meta["compression"])
    results = [dict(label="float K=256", bits_mean=32.0, pos_B=pos_f, total_B=tot_f, fid=ref.meta["fidelity"], err_container=0.0,
                    floor=ref.meta["fidelity"]["floor_max"], rows=sig_f)]
    cases = [("two-level 16/8", cx.BAND_CONTAINER, (16 * 16 + 8 * (K - 16)) / K)]
    for b1 in a.b1:
        c = cx.band_curve(K, b1)
        mean = sum((r["bands"][1] - r["bands"][0]) * r["bits"] for r in c) / K
        cases.append((f"curve b1={b1:g}", c, mean))
    for label, container, mean_bits in cases:
        pk, pos_b, tot_b, sig = build(label, container)
        fid = pk.meta["fidelity"]
        mf = cx.measure_fidelity(np.asarray(raw(0, ref.n_walkers)), DT, cx.decoder(pk.arrays, pk.meta["compression"]))
        results.append(dict(label=label, bits_mean=mean_bits, pos_B=pos_b, total_B=tot_b, fid=fid, err_container=mf["err_max"],
                            floor=mf["floor_max"], rows=sig,
                            container=pk.meta["compression"]["container"]))
    out = dict(n=a.n, K=K, n_t=n_t, exact=exact, results=results)
    print()
    hdr = "| container | mean bits | pos B/walker | total B/walker | cert err_max | cert floor_max | container err | err/floor | " + \
          " | ".join(f"{k}: replay-float, replay-exact" for k in rows) + " |"
    print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
    for r in results:
        cells = [f"{abs(r['rows'][k] - sig_f[k]):.1e}, {abs(r['rows'][k] - exact[k]):.1e}" for k in rows]
        print(f"| {r['label']} | {r['bits_mean']:.1f} | {r['pos_B']:.0f} | {r['total_B']:.0f} | {r['fid']['err_max']:.2e} | "
              f"{r['fid']['floor_max']:.2e} | {r['err_container']:.2e} | {r['err_container'] / r['floor']:.1e} | " + " | ".join(cells) + " |")
    if a.out:
        a.out.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
