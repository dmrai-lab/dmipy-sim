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


def blt_bytes_per_walker(pack):
    """Bytes per walker of the contact channel's band tensors."""
    return sum(np.asarray(v).nbytes for k, v in pack.arrays.items() if k.startswith("blt_b")) / pack.n_walkers


def strand_walk(n, T_s, dt, seed=3):
    """A small strand substrate (three sheathed strands in a 20 um box) walked with its susceptibility field sampled in
    the walk, on the CPU: the walk the path channel (C3) is certified on."""
    import tempfile
    from dmipy_sim.io.strands import concat_centerlines, write_tck
    from dmipy_sim.spec import disco_spec, walk_spec
    tmp = Path(tempfile.mkdtemp())
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp / "t.tck"), str(tmp / "d.txt")
    write_tck(tck, *concat_centerlines(cls_), coordinate_unit_m=25e-6)
    np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    spec = disco_spec(tck, dia, side_m=20e-6)
    return walk_spec(spec, n, T_s, dt, seed=seed, n_probe=20_000, require_gpu=False, adaptive_steps=True, field_cutoff_max_m=25e-6)


def sweep_c3(a):
    """The path channel's container sweep: the same strand walk packed with the path modes (``Kp``) in native 16 and 8
    bit and in ``band_curve(Kp, b1)``, each pack's ``err_susc_path`` against ``floor_susc_path`` and the path bytes
    per walker."""
    from dmipy_sim.replay import compression as cx
    from dmipy_sim.replay.bank import build_replay_pack
    Kp, T_s, dt = a.path_K, a.c3_T, a.c3_dt
    log(f"C3 walk: {a.c3_n} walkers on 3 strands, T {T_s}, dt_save {dt}, path modes {Kp}")
    walk = strand_walk(a.c3_n, T_s, dt)
    cases = [("native 16", 16), ("native 8", 8)] + [(f"curve b1={b:g}", cx.band_curve(Kp, b)) for b in a.c3_b1]
    rows_ = []
    for label, knob in cases:
        t0 = time.time()
        pk = build_replay_pack(walk, id=f"c3/{label}", K=a.c3_K, license="x", citation="x", susc_path_K=Kp, susc_path_bits=knob, device="numpy")
        pm = pk.meta["compression"]["channels"]["susceptibility_path"]
        B = sum(np.asarray(v).nbytes for k, v in pk.arrays.items() if k.startswith("susc_path_dct")) / pk.n_walkers
        f = pk.meta["fidelity"]
        mean = (sum((r["modes"][1] - r["modes"][0]) * r["bits"] for r in pm["container"]) / Kp) if pm.get("container") else float(pm["bits"])
        rows_.append(dict(label=label, mean_bits=mean, path_B=B, err=f["err_susc_path"], floor=f["floor_susc_path"]))
        log(f"{label}: {time.time() - t0:.0f} s")
    print("\nC3 (path channel, %d modes, %d walkers)" % (Kp, a.c3_n))
    print("| container | mean bits | path B/walker | err_susc_path | floor_susc_path | err/floor |")
    print("|---|---|---|---|---|---|")
    for r in rows_:
        print(f"| {r['label']} | {r['mean_bits']:.1f} | {r['path_B']:.0f} | {r['err']:.2e} | {r['floor']:.2e} | {r['err'] / r['floor']:.2e} |")
    return rows_


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--b1", type=float, nargs="+", default=[16, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--blt-K", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--c3", action="store_true", help="sweep the path channel on a strand walk instead of the cylinder")
    ap.add_argument("--c3-n", type=int, default=5000)
    ap.add_argument("--c3-T", type=float, default=0.03)
    ap.add_argument("--c3-dt", type=float, default=1e-4)
    ap.add_argument("--c3-K", type=int, default=32, help="position bands of the strand packs")
    ap.add_argument("--path-K", type=int, default=32)
    ap.add_argument("--c3-b1", type=float, nargs="+", default=[14, 12, 10, 9, 8, 7, 6])
    a = ap.parse_args(argv)
    if a.c3:
        res = sweep_c3(a)
        if a.out:
            a.out.write_text(json.dumps(res, indent=1, default=float))
        return

    import dmipy_sim as d
    from dmipy_sim.math.step_law import SteppedPore
    from dmipy_sim.replay import compression as cx
    from dmipy_sim.replay.bank import build_replay_pack
    import gen_step_law as gl

    log(f"walk: {a.n} walkers, {DIAM * 1e6:.0f} um cylinder, D {D0}, T {T}, dt_save {DT}")
    geom = d.Cylinder(DIAM / 2, (0.0, 0.0, 1.0))
    walk = d.simulate_trajectories(a.n, D0, geom, T, DT, seed=a.seed, require_gpu=False, tiers="all", K=K, window_T=T)
    rows = gl.rows()
    exact = {k: SteppedPore("cylinder", DIAM, D0, geom, n_modes=(8, 8)).continuous(gl.g_axis(s), DT) for k, s in rows.items()}
    log("exact:", exact)

    def build(label, container, blt_container=None, **kw):
        t0 = time.time()
        pk = build_replay_pack(walk, id=f"sweep/{label}", K=K, license="x", citation="x", blt_temporal_K=a.blt_K,
                               position_container=container, blt_container=blt_container, **kw)
        pos_b, tot_b = bytes_per_walker(pk)
        sig = {k: float(np.abs(np.asarray(pk.replay(s))).ravel()[0]) for k, s in rows.items()}
        log(f"{label}: built + replayed in {time.time() - t0:.0f} s")
        return pk, pos_b, tot_b, sig

    master = walk._bank_dict()
    _cx_env = cx.default_envelope

    def surface_window(pk):
        """The contact channel's per-window surface term (windows of T/4) of a pack against the raw contact."""
        from dmipy_sim.replay.bank import _surface_fidelity
        return _surface_fidelity(master, pk.arrays, pk.meta["compression"]["channels"]["boundary_local_time"],
                                 _cx_env(), window_T=T / 4)

    ref, pos_f, tot_f, sig_f = build("float", None, None)
    n_t = int(ref.n_t)
    raw = cx.decoder(ref.arrays, ref.meta["compression"])
    results = [dict(label="float K=256", bits_mean=32.0, pos_B=pos_f, total_B=tot_f, fid=dict(ref.meta["fidelity"], **{"err_surface_window": surface_window(ref)["err_window"], "floor_surface_window": surface_window(ref)["floor_window"]}), err_container=0.0,
                    floor=ref.meta["fidelity"]["floor_max"], rows=sig_f, blt_B=blt_bytes_per_walker(ref))]
    cases = [("two-level 16/8", cx.TWO_LEVEL_CONTAINER, (16 * 16 + 8 * (K - 16)) / K, cx.TWO_LEVEL_CONTAINER),
             ("derived (default)", "derived", None, "derived")]
    for b1 in a.b1:
        c = cx.band_curve(K, b1)
        mean = sum((r["bands"][1] - r["bands"][0]) * r["bits"] for r in c) / K
        cases.append((f"curve b1={b1:g}", c, mean, cx.band_curve(a.blt_K, b1)))
    for label, container, mean_bits, blt_c in cases:
        pk, pos_b, tot_b, sig = build(label, container, blt_c)
        fid = dict(pk.meta["fidelity"])
        if mean_bits is None:
            c_ = pk.meta["compression"]["container"]
            mean_bits = sum((r["bands"][1] - r["bands"][0]) * r["bits"] for r in c_) / K
        sf = surface_window(pk)
        fid.update(err_surface_window=sf["err_window"], floor_surface_window=sf["floor_window"])
        mf = cx.measure_fidelity(np.asarray(raw(0, ref.n_walkers)), DT, cx.decoder(pk.arrays, pk.meta["compression"]))
        results.append(dict(label=label, bits_mean=mean_bits, pos_B=pos_b, total_B=tot_b, fid=fid, err_container=mf["err_max"],
                            floor=mf["floor_max"], rows=sig, blt_B=blt_bytes_per_walker(pk),
                            container=pk.meta["compression"]["container"]))
    print("\nBytes per walker of the pack ordered at sigma_star = 5e-3 (the default container is derived for that floor):")
    print("| container | pos B/walker | total B/walker | cert err_max | cert floor_max |"); print("|---|---|---|---|---|")
    for label, cont in (("two-level 16/8", cx.TWO_LEVEL_CONTAINER), ("derived (default)", "derived")):
        pk, pos_b, tot_b, _ = build(label, cont, cont, sigma_star=5e-3)
        print(f"| {label} | {pos_b:.0f} | {tot_b:.0f} | {pk.meta['fidelity']['err_max']:.2e} | {pk.meta['fidelity']['floor_max']:.2e} |")
    out = dict(n=a.n, K=K, n_t=n_t, exact=exact, results=results)
    print("\nC2 (contact channel, K=%d, same b1 curve; the whole-duration surface term is exact in the bridge's endpoints at any depth, "
          "so the band is read by the per-window term, windows of T/4)" % a.blt_K)
    print("| container | blt B/walker | err_surface | floor_surface | err_surface_window | floor_surface_window | err/floor (window) |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        f = r["fid"]; w_e, w_f = f.get("err_surface_window"), f.get("floor_surface_window")
        print(f"| {r['label']} | {r['blt_B']:.0f} | {f.get('err_surface', float('nan')):.2e} | {f.get('floor_surface', float('nan')):.2e} | "
              f"{w_e if w_e is None else format(w_e, '.2e')} | {w_f if w_f is None else format(w_f, '.2e')} | "
              f"{'' if not w_f else format(w_e / w_f, '.1e')} |")
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
