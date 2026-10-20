"""The measurements of the walk's band accumulation (``simulate_trajectories``, dmrai-lab/dmipy-sim#741), each a
function the tests read and a table this script prints:

* ``exactness``  -- at ``block_order=0`` the pack of the accumulated walk against the pack of the same walk's block
  ends (its raw subset, every walker; same seed, same steps), on a sphere, a cylinder, a packed cell, a mesh and a
  myelinated cylinder, and the bands against ``sqrt(h) sinc(pi k / 2N)`` times the DST-I of those block ends;
* ``quadrature`` -- the phase variance the block accumulation adds per measurement of the envelope's battery, against
  the same walk accumulated per step (``h = dt_sim``), beside the table ``rel(m, h)`` it is held to, absolutely;
* ``memory``     -- the walk's peak host bytes (VmHWM, read from ``/proc``) over three blocks spanning 16x at a fixed
  ``K``, accumulated alone and keeping every walker's block ends, each in a process of its own; on a GPU also the
  device's peak;
* ``oracle``     -- a sphere and a cylinder replayed from the accumulated pack against the matrix method;
* ``surface``    -- the contact tier (C2) at the signal: surface-relaxation rows at the envelope's largest ``rho2``
  replayed from the accumulated pack, from the pack of the same walk's block ends and by the fused engine.

    python tools/coefficient_walk_measure.py --what exactness quadrature memory oracle

CPU by default (``JAX_PLATFORMS=cpu`` is set unless the environment says otherwise).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def bands(cw):
    """``(n_w, K+2, 3)``: a one-window coefficient walk's coefficients, float64."""
    return np.asarray(cw.position_coeffs, np.float64)[:, 0]


# ----------------------------------------------------------------------------------------------- exactness at m = 0
def _geometries():
    import dmipy_sim as d
    from dmipy_sim.geometry import pack_cylinders
    out = {"sphere": lambda: d.Sphere(radius=4e-6), "cylinder": lambda: d.Cylinder(radius=3e-6, orientation=(0, 0, 1))}

    def packed():
        radii = np.full(4, 3e-6)
        centers, L, _ = pack_cylinders(radii, target_vf=0.30, seed=0)
        return d.PackedCylinders(radii=radii, centers=centers, L=L, orientation=(0.0, 0.0, 1.0))
    out["packed cell"] = packed

    def mesh():
        import trimesh
        m = trimesh.creation.icosphere(subdivisions=2, radius=4e-6)
        return d.Mesh(np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int32), feature_radius=1e-6)
    out["mesh"] = mesh
    out["myelinated cylinder"] = lambda: d.MyelinatedCylinder(2e-6, 3e-6, (0, 0, 1), 2e-9, 2e-9)
    return out


def exactness(names=None, *, n=400, T=0.01, dt_save=2e-4, K=24, sub_steps=4, seed=11):
    """Per geometry the largest differences between the pack of a walk's block ends (its raw subset, every walker,
    read at block order 0) and the pack of the same walk's accumulation at ``block_order=0`` (float containers on
    both): the start (m), the displacement and the bands relative to their largest value, the contact's endpoint and
    bands likewise, whether the occupancy is the same, the largest difference of a replayed PGSE, and the bands
    against the exact relation ``beta = sqrt(h) sinc(pi k / 2N) beta_DST-I`` with the orthonormal DST-I of the
    block ends' bridge."""
    import dmipy_sim as d
    from scipy.fft import dst
    from dmipy_sim.replay.bank import build_replay_pack
    from dmipy_sim.replay.compression import read_position_coeffs
    seq = d.pgse([[1, 0, 0], [0, 0, 1]], 2e-3, 5e-3, bvalues=[2e9, 2e9], TE=0.009, slew_rate=np.inf)
    out = {}
    import dataclasses
    for name, make in _geometries().items():
        if names and name not in names:
            continue
        kw = dict(T_max=T, dt_save=dt_save, seed=seed, require_gpu=False, sub_steps=sub_steps)
        pk = dict(license="x", citation="x", K=K, position_container=None, blt_container=None, blt_temporal_K=K,
                  blt_dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cw = d.simulate_trajectories(n, 2e-9, make(), K=K, block_order=0, raw_subset=n, **kw)
            walks = [cw.saves(), cw]
            spec = walks[0].spec
            if spec.field_source_pools:
                # the sheath's field is the C3 tier, which a coefficient walk does not carry yet (#740 stage 3): both
                # walks are packed with the sheath declared non-magnetic, the positions, contact and occupancy compared
                spec = spec.replace(susceptibility={p.name: None for p in spec.field_source_pools},
                                    validity=dataclasses.replace(spec.validity, tiers=[t for t in spec.validity.tiers if t != "field"]))
                walks = [dataclasses.replace(w, spec=spec) for w in walks]
            raw = build_replay_pack(walks[0], id="exact/raw", **pk)
            acc = build_replay_pack(walks[1], id="exact/acc", **pk)
        A, B = read_position_coeffs(raw.arrays), read_position_coeffs(acc.arrays)
        rel = lambda a, b: float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300))
        X = np.asarray(cw.raw_positions, np.float64)
        N = X.shape[1] - 1
        tau = np.arange(N + 1) / N
        u = X - (X[:, :1] + (X[:, -1:] - X[:, :1]) * tau[None, :, None])
        k = np.arange(1, K + 1)
        exact = np.sqrt(cw.h) * np.sinc(k / (2.0 * N))[None, :, None] * dst(u[:, 1:-1], type=1, norm="ortho", axis=1)[:, :K]
        row = dict(n_blocks=int(cw.n_blocks), start_m=float(np.abs(A[:, 0] - B[:, 0]).max()),
                   displacement_rel=rel(A[:, 1], B[:, 1]), bands_rel=rel(A[:, 2:], B[:, 2:]),
                   bands_vs_sinc_dst1_rel=rel(exact, B[:, 2:]),
                   replay_abs=float(np.abs(raw.replay(seq) - acc.replay(seq)).max()))
        if "blt_bridge_dst" in raw.arrays:
            row.update(contact_endpoint_rel=rel(raw.arrays["blt_endpoint"], acc.arrays["blt_endpoint"]),
                       contact_bands_rel=rel(raw.arrays["blt_bridge_dst"], acc.arrays["blt_bridge_dst"]))
        cr, ca = raw.meta["compression"]["channels"].get("compartment"), acc.meta["compression"]["channels"].get("compartment")
        if cr is not None:
            from dmipy_sim.replay.compression import decode_occupancy
            row["occupancy_equal"] = bool(np.array_equal(decode_occupancy(raw.arrays, cr)["comp"], decode_occupancy(acc.arrays, ca)["comp"]))
        out[name] = row
    return out


# ------------------------------------------------------------------------------------------------- the quadrature
def quadrature(geometry, *, D=2.95e-9, T=0.02, dt_sim=1e-6, K=64, cases=((0, 2e-5), (1, 2e-5), (2, 5e-5), (3, 1e-4)),
               n=2000, seed=5, env=None):
    """The phase variance the block accumulation at each ``(m, h)`` of ``cases`` adds per measurement of the battery,
    against the same walk accumulated per step (``h = dt_sim``; the same seed, starts and steps), all read on the
    continuous basis through one projection of the battery. Per case: the largest measured variance over the
    battery, the table's prediction for that measurement (``sum_k gamma^2 W_k^2 rel(m, h, omega_k) var_k`` at the free
    ``D``), the ratio of the two at the measurement the table predicts largest, the absolute bound ``rel(m, h,
    omega_K) 2 b D`` and whether every measurement is within it, the range of measured over predicted across the
    measurements, and the relative error of the top quarter of the modes against the table on the axis where it is
    largest (what must NOT be the criterion on a restricted walk)."""
    import dmipy_sim as d
    from dmipy_sim.constants import GAMMA
    from dmipy_sim.replay import compression as cx
    h_max = max(h for _, h in cases)
    n_t = int(round(T / h_max)) + 1
    G, meta = cx.acquisition_battery(n_t, h_max, env or cx.default_envelope())
    W = cx.band_projection(G, h_max, T, K)                                           # (n_meas, K+2, 3)
    b = np.array([x["b"] for x in meta])
    var = cx.band_variances(K, T, D)
    om = np.pi * np.arange(1, K + 1) / T
    per_band = GAMMA ** 2 * (W[:, 2:, :] ** 2).sum(axis=2) * var[None, :]

    def walk(h, m):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return d.simulate_trajectories(n, D, geometry, T_max=T, dt_save=h, seed=seed, require_gpu=False, tiers=(),
                                           sub_steps=int(round(h / dt_sim)), K=K, block_order=m, window_T=T)
    ref_walk = walk(dt_sim, 1)
    ref = bands(ref_walk)
    phi_ref = GAMMA * np.einsum("wkc,mkc->wm", ref, W)
    qv = np.asarray(ref_walk.qv_positions, np.float64)[:, 0].mean(axis=0)                # the measured QV per axis
    per_band_qv = GAMMA ** 2 * np.einsum("mkc,k,c->mk", W[:, 2:, :] ** 2, T / (np.pi * np.arange(1, K + 1)) ** 2, qv)
    top = slice(3 * K // 4, K)
    rows = []
    for m, h in cases:
        C = bands(walk(h, m))
        if not np.array_equal(C[:, :2], ref[:, :2]):
            raise AssertionError("the case and the reference did not walk the same paths")
        meas = np.var(GAMMA * np.einsum("wkc,mkc->wm", C, W) - phi_ref, axis=0)
        pred = (per_band * cx.quadrature_rel(m, h, om)[None, :]).sum(axis=1)
        pred_qv = (per_band_qv * cx.quadrature_rel(m, h, om)[None, :]).sum(axis=1)
        bound = cx.quadrature_rel(m, h, np.pi * K / T) * 2.0 * b * D
        i = int(np.argmax(pred))
        rel_meas = np.var(C[:, 2:, :] - ref[:, 2:, :], axis=0) / np.var(ref[:, 2:, :], axis=0)   # (K, 3)
        rel_tab = cx.quadrature_rel(m, h, om)[:, None]
        live = pred > 1e-6 * pred.max()
        rows.append(dict(m=int(m), h_us=float(h * 1e6), measured_max=float(meas.max()), predicted_at_max=float(pred[int(np.argmax(meas))]),
                         ratio_at_largest_prediction=float(meas[i] / pred[i]), measured_at_largest_prediction=float(meas[i]),
                         predicted_largest=float(pred[i]), bound_max=float(bound.max()), within_bound=bool(np.all(meas <= bound)),
                         worst_fraction_of_bound=float(np.max(meas / np.where(bound > 0, bound, np.inf))),
                         ratio_range=[float((meas[live] / pred[live]).min()), float((meas[live] / pred[live]).max())],
                         qv_over_2DT=[float(x) for x in qv / (2.0 * D * T)],
                         ratio_range_qv=[float((meas[live] / pred_qv[live]).min()), float((meas[live] / pred_qv[live]).max())],
                         relative_vs_table_top_modes=float((rel_meas[top] / rel_tab[top]).mean(axis=0).max())))
    return rows


# -------------------------------------------------------------------------------------------------------- memory
def _probe(mode, dt_save, n, T, dt_sim, K):
    """One walk in this process (``mode`` ``"accumulated"``: the bands alone; ``"saves"``: every walker's block ends
    kept beside them, ``raw_subset=n``): VmHWM (from /proc/self/status) over the walk's second call, the high-water
    mark reset after the first (which compiles), and the device's peak bytes when it reports them."""
    import dmipy_sim as d

    def vm(key):
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
    kw = dict(T_max=T, dt_save=dt_save, seed=1, require_gpu=False, sub_steps=int(round(dt_save / dt_sim)),
              walker_batch_size=n, K=K, raw_subset=(0 if mode == "accumulated" else n))
    geom = d.Cylinder(radius=4e-6, orientation=(0, 0, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = d.simulate_trajectories(n, 2e-9, geom, **kw)
        del w
        base = vm("VmRSS")
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")                                                          # the high-water mark := now
        w = d.simulate_trajectories(n, 2e-9, geom, **kw)
    out = dict(mode=mode, dt_save=dt_save, n_blocks=int(w.n_blocks), host_peak_over_base=vm("VmHWM") - base,
               walk_host_bytes=int(w.nbytes))
    try:
        import jax
        st = jax.devices()[0].memory_stats()
        if st and "peak_bytes_in_use" in st:
            out["device_peak_bytes"] = int(st["peak_bytes_in_use"])
    except Exception:
        pass
    print(json.dumps(out))


def memory(*, n=20000, T=0.01, dt_sim=3.125e-6, K=32, dt_saves=(5e-4, 1.25e-4, 3.125e-5)):
    """``memory_probe`` in a fresh process per (mode, block): the accumulated walk's peak host bytes against the same
    walk keeping every walker's block ends, over blocks spanning 16x at a fixed ``K`` and a fixed physics step (the
    walkers take the same steps at every block), a cylinder recording every tier."""
    rows = []
    for mode in ("accumulated", "saves"):
        for h in dt_saves:
            # one malloc arena: glibc's per-thread arenas hold freed blocks of XLA's worker threads and add tens of MB
            # of noise to the high-water mark, the same for both modes
            r = subprocess.run([sys.executable, __file__, "--memory-probe", mode, repr(h), str(n), repr(T), repr(dt_sim), str(K)],
                               capture_output=True, text=True, env=dict(os.environ, MALLOC_ARENA_MAX="1"), check=True)
            rows.append(json.loads(r.stdout.strip().splitlines()[-1]))
    return rows


# -------------------------------------------------------------------------------------------------------- the oracle
def oracle(*, n=20000, D=2e-9, T=0.04, K=80, f=400.0, dt_save=None, seed=1):
    """A sphere (radius 4 um) and a cylinder (radius 3 um) walked (the band's own block and order unless ``dt_save``
    gives the block), packed, and replayed under a cosine OGSE at ``f`` -- a band the pack's ``K`` only just holds --
    against the matrix method
    (``dmipy_sim.math.matrix_method``) on the same waveform. Per pore and measurement: both signals, the difference and
    the standard error of the replay's ensemble mean."""
    import dmipy_sim as d
    from dmipy_sim.math.matrix_method import MatrixPore
    from dmipy_sim.replay.bank import build_replay_pack
    seq = d.ogse([[1, 0, 0], [1, 0, 0]], f, 0.015, shape="cosine", bvalues=[1e9, 2e9], TE=T, n_t=4001, slew_rate=np.inf)
    out = {}
    for name, geom, shape, size in (("sphere", d.Sphere(radius=4e-6), "sphere", 8e-6),
                                    ("cylinder", d.Cylinder(radius=3e-6, orientation=(0, 0, 1)), "cylinder", 6e-6)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cw = d.simulate_trajectories(n, D, geom, T_max=T, dt_save=dt_save, seed=seed, require_gpu=False, K=K, tiers=(),
                                         window_T=T)
        pk = build_replay_pack(cw, id=f"oracle/{name}", license="x", citation="x")
        w, ew, E = pk.walker_signals(seq)
        S = np.abs((ew[:, None] * E).sum(0) / w.sum())
        se = np.std(np.real(E * np.conj(E.mean(0) / np.abs(E.mean(0)))), axis=0) / np.sqrt(E.shape[0])
        pore = MatrixPore(shape, size, D)
        Sm = np.array([abs(pore.signal(np.asarray(seq.G_eff)[i, :, 0], seq.dt)) for i in range(seq.G_eff.shape[0])])
        out[name] = dict(replay=S.tolist(), matrix_method=Sm.tolist(), diff=(S - Sm).tolist(), se=se.tolist(),
                         band_needed=pk.waveform_band(seq)[1], K=pk.K,
                         block=dict(m=cw.block["m"], h=cw.block["h"], rule=cw.block["rule"]), sub_steps=cw.sub_steps,
                         certificate=dict(err_max=pk.fidelity["err_max"], floor_max=pk.fidelity["floor_max"]))
    return out


def surface(*, n=20000, D=2e-9, T=0.02, K=64, R=4e-6, rho=1e-4, seed=3):
    """The contact tier at the signal (C2): a reflecting sphere of radius ``R`` walked once (the band's own block),
    packed from its accumulation and from its block ends (every walker kept), and walked again by the fused engine with
    ``surface_relaxivity_t2 = rho`` (the envelope's largest, rho R / D = 0.2 at the defaults). PGSE rows at b = 0, 1
    and 3 ms/um^2 with TE = T (the whole contact, the exact endpoint) and TE = T / 2 (the contact gated at the
    acquisition's end, read from the bands) replayed with ``rho2 = rho``. Per row the three signals, the replay's
    standard error, and the differences accumulated - saves (the codec) and accumulated - fused (two walks)."""
    import dmipy_sim as d
    from dmipy_sim.replay.bank import build_replay_pack
    from dmipy_sim.spec.tissue import Tissue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = d.Sphere(radius=R)
        cw = d.simulate_trajectories(n, D, g, T_max=T, seed=seed, require_gpu=False, K=K, raw_subset=n, window_T=T)
        raw = cw.saves()
    pa = build_replay_pack(cw, id="surface/acc", license="x", citation="x")
    pr = build_replay_pack(raw, id="surface/saves", license="x", citation="x", K=K)
    tissue = Tissue(rho2=rho)
    rows = []
    for TE in (T, T / 2):
        n_t = 401
        seq = d.pgse([[1, 0, 0]] * 3, 0.2 * TE, 0.5 * TE, bvalues=[0.0, 1e9, 3e9], TE=TE, n_t=n_t, slew_rate=np.inf)
        w, ew, E = pa.walker_signals(seq, tissue=tissue)
        S_acc = np.abs((ew[:, None] * E).sum(0) / w.sum())
        se = np.std(np.real((ew[:, None] / w.mean()) * E * np.conj(E.mean(0) / np.abs(E.mean(0)))), axis=0) / np.sqrt(E.shape[0])
        S_raw = pr.replay(seq, tissue=tissue)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S_fused = np.asarray(d.simulate(n, D, seq, d.Sphere(radius=R, surface_relaxivity_t2=rho), seed=seed + 100,
                                            require_gpu=False), np.float64).reshape(-1)
        rows.append(dict(TE_ms=TE * 1e3, b=[0.0, 1e9, 3e9], accumulated=S_acc.tolist(), saves=np.asarray(S_raw).tolist(),
                         fused=S_fused.tolist(), se=se.tolist(), acc_minus_saves=(S_acc - S_raw).tolist(),
                         acc_minus_fused=(S_acc - S_fused).tolist()))
    return dict(block=dict(m=cw.m, h=cw.h, n_blocks=cw.n_blocks, sub_steps=cw.sub_steps),
                qv_contact_mean=float(np.mean(cw.qv_boundary)), rho_over_D=rho / D,
                err_quadrature_surface=pa.fidelity.get("err_quadrature_surface"), rows=rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--what", nargs="+", default=["exactness", "quadrature", "memory", "oracle", "surface"])
    ap.add_argument("--memory-probe", nargs=6, metavar=("MODE", "DT_SAVE", "N", "T", "DT_SIM", "K"))
    a = ap.parse_args(argv)
    if a.memory_probe:
        mode, h, n, T, dt_sim, K = a.memory_probe
        _probe(mode, float(h), int(n), float(T), float(dt_sim), int(K))
        return
    import dmipy_sim as d
    res = {}
    if "exactness" in a.what:
        res["exactness"] = exactness()
    if "quadrature" in a.what:
        res["quadrature"] = {"free": quadrature(d.FreeDiffusion()), "slab 0.8 um": quadrature(d.Box1D(length=0.8e-6))}
    if "memory" in a.what:
        res["memory"] = memory()
    if "oracle" in a.what:
        res["oracle"] = oracle()
    if "surface" in a.what:
        res["surface"] = surface()
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
