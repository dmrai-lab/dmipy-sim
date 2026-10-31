"""The step law as evidence: regenerate ``docs/step-law.md`` and its figures from the current code.

Every number on the page comes from a measurement this script makes when it runs: the step length laws and
their bulk error, the stepped operator of the engine's own ``reflect`` on the three analytic pores and the
certificate on a packing's two pools, the fused engine against the operator, the two-walk certificate's
resolution against its walker count, the sub-step count each rule asks for and the walk's throughput per
substrate, and the phase rule the certificate does not see. The page carries the commit, the tier, the seeds,
the walker counts and the device it was built on.

Run::

    python tools/gen_step_law.py --tier small          # minutes on a CPU: what tests/test_step_law_report.py runs
    python tools/gen_step_law.py --tier full           # on a GPU: the committed page
    python tools/gen_step_law.py --only laws,operator  # a subset of the sections (the rest read from the json)

Set ``STEP_LAW_MESH`` to a PLY file in micrometres (a CACTUS pack, for instance) to include it in the cost
section; without it the mesh row is an icosphere.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from dmipy_sim.engine._fused import simulate_fused

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

D0 = 2.0e-9                 #: m^2/s, the free diffusivity of every measurement here
DT = 1.0e-4                 #: s, the rows' sample interval, which the operator steps the signal over
T_SEQ = 0.100               #: s, the rows' TE
N_T = 1001
SLEW, G_CAP = 200.0, 0.30   #: T/m/s and T/m, the scanner the rows are built for
F_BAND = 200.0              #: Hz, the clinical class band the cost section sizes steps for
CERT_T, CERT_F_BAND, CERT_K = 0.04, 1000.0, 80   #: the certificate's window, band and K = 2 T f_band: the walks
                                                 #: keep 80 bands in the band's own blocks of 127 us
EPS = 1e-3                  #: the tolerance the engine declares (physics.STEP_TOLERANCE)

TIERS = {
    "small": dict(levels=(0.5,), quad=dict(n_rad=60, n_th=16, n_a=32, n_b=64), op_modes=(6, 6),
                  fused_geoms=("sphere",), fused_rows=("OGSE 50 Hz",), fused_subs=(1, 2), fused_seeds=(11,),
                  fused_n=10_000, cert_walkers=(2_000, 12_000), cert_ratio=4, pack_walkers=4_000, pack_ratio=4,
                  cost_n=2_000, cost_T=0.02, mesh_full=False),
    "full": dict(levels=(0.1, 0.2, 0.3, 0.4, 0.5), quad=dict(n_rad=100, n_th=24, n_a=48, n_b=96), op_modes=(8, 8),
                 fused_geoms=("sphere", "cylinder"), fused_rows=("PGSE 10/40 b3000", "OGSE 50 Hz"),
                 fused_subs=(1, 2, 4), fused_seeds=(11, 12, 13, 14), fused_n=1_000_000,
                 cert_walkers=(4_000, 16_000, 64_000, 256_000), cert_ratio=4, pack_walkers=256_000, pack_ratio=1,
                 cost_n=100_000, cost_T=0.1, mesh_full=True),
}
SECTIONS = ("laws", "operator", "fused", "certificate", "cost", "outside")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------------- the rows
def rows():
    from dmipy_replay.sequences.builders import ogse, pgse
    axis = [[1.0, 0.0, 0.0]]
    out = [("PGSE 10/40 b3000", pgse(axis, 0.010, 0.040, bvalues=[3.0e9], TE=T_SEQ, n_t=N_T, slew_rate=SLEW))]
    f = 50.0
    g = min(G_CAP, SLEW / (2.0 * np.pi * f))
    out.append(("OGSE 50 Hz", ogse(axis, f, 0.4 * T_SEQ, shape="cosine", gradient_strengths=[g], TE=T_SEQ, n_t=N_T,
                                   slew_rate=SLEW)))
    return dict(out)


def g_axis(seq):
    G = np.asarray(seq.G_eff)[0]
    g = np.zeros(N_T)
    n = min(N_T, G.shape[0])
    g[:n] = G[:n, 0]
    return g


# ----------------------------------------------------------------------------------------------- the laws
def gauss_law(n):
    """The n-point Gauss-Laguerre (alpha = 1/2) quadrature of the chi^2_3 step length law: lengths in units of
    ``delta_0 = sqrt(6 D dt)`` and weights."""
    from scipy import special
    x, w = special.roots_genlaguerre(n, 0.5)
    w = w / w.sum()
    return tuple(float(v) for v in np.sqrt(4.0 * x / 6.0)), tuple(float(v) for v in w)


def laws():
    import dmipy_sim.math.step_law as sl
    t = math.sqrt(2.0 / 3.0)
    return {
        "fixed": (sl.FIXED_LENGTHS, sl.FIXED_WEIGHTS),
        "equal pair": ((math.sqrt(1 - t), math.sqrt(1 + t)), (0.5, 0.5)),
        "Gauss pair": (sl.PAIR_LENGTHS, sl.PAIR_WEIGHTS),
        "Gauss triple": gauss_law(3),
        "Gaussian (6 atoms)": gauss_law(6),
    }


def sec_laws(cfg, fig_dir):
    import dmipy_sim.math.step_law as sl
    import matplotlib.pyplot as plt
    out = {"laws": {}, "engine": {}}
    x = np.linspace(0.02, 3.0, 600)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, (L, W) in laws().items():
        m2, m4, m6 = sl.step_moments(L, W)
        m = sl.bulk_multiplier(x, L, W)
        ok = m > 0
        err = np.full_like(x, np.nan)
        err[ok] = np.abs(np.log(m[ok]) / (-x[ok] ** 2 / 6.0) - 1.0)
        out["laws"][name] = dict(lengths=[round(v, 5) for v in L], weights=[round(v, 5) for v in W],
                                 m2=m2, m4=m4, m6=m6, longest=max(L),
                                 reach_1e3=sl.band_reach(1e-3, tuple(L), tuple(W)),
                                 reach_1e4=sl.band_reach(1e-4, tuple(L), tuple(W)))
        ax.loglog(x, err, label=name)
    ax.axhline(1e-3, color="k", lw=0.6, ls=":")
    ax.set(xlabel=r"$k\,\delta_0$ of the bulk mode", ylabel="relative rate error of one step", ylim=(1e-7, 2),
           title="Bulk error of a step length law")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "laws.png", dpi=130)
    plt.close(fig)
    # the engine draws the law
    import jax
    from dmipy_sim.engine.physics import pair_step
    keys = jax.random.split(jax.random.PRNGKey(0), 200_000)
    s = np.asarray(jax.vmap(pair_step)(keys), np.float64)
    n2 = (s ** 2).sum(1)
    out["engine"] = dict(n=int(len(n2)), m2=float(n2.mean()), m4=float((n2 ** 2).mean() * 36 / 60),
                         m6=float((n2 ** 3).mean() * 216 / 840), longest=float(np.sqrt(n2).max()),
                         p_long=float(np.mean(np.sqrt(n2) > 1.2)),
                         lengths=[float(v) for v in np.unique(np.round(np.sqrt(n2), 4))])
    return out


# ----------------------------------------------------------------------------------------------- the operator
PORES = {"plane": ("Box1D", 8.5e-6), "cylinder": ("Cylinder", 10.4e-6), "sphere": ("Sphere", 5.6e-6)}


def make_pore(shape, diam):
    import dmipy_sim as d
    if shape == "plane":
        return d.Box1D(diam)
    if shape == "cylinder":
        return d.Cylinder(diam / 2, (0.0, 0.0, 1.0))
    return d.Sphere(diam / 2)


def stepped_pore(shape, diam, cfg):
    from dmipy_sim.math.step_law import SteppedPore
    nm = (24, 18) if shape == "plane" else cfg["op_modes"]
    return SteppedPore(shape, diam, D0, make_pore(shape, diam), n_modes=nm)


def sec_operator(cfg, fig_dir):
    import matplotlib.pyplot as plt
    out = {"pores": {}, "packing": {}}
    LAWS = laws()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, (shape, (cls, diam)) in zip(axes, PORES.items()):
        sp = stepped_pore(shape, diam, cfg)
        a = diam / 2
        lam = D0 * sp.lam
        order = np.argsort(lam)
        idx = [int(order[1]), int(order[2])]
        rec = dict(diameter_um=diam * 1e6, modes=[dict(rate_hz=float(lam[i]), angular=int(sp.ang[i])) for i in idx],
                   levels={})
        cache = {}

        def op(delta):
            key = round(delta, 15)
            if key not in cache:
                cache[key] = sp.operator(delta, **cfg["quad"])
            return cache[key]

        for lv in cfg["levels"]:
            d0 = lv * a
            dt_p = d0 ** 2 / (6 * D0)
            row = {}
            for name, (L, W) in LAWS.items():
                T = sum(w * op(d0 * l)[0] for l, w in zip(L, W))
                e = sp.rate_error(T, dt_p)
                row[name] = [float(e[i] / (lam[i] * dt_p / 5)) for i in idx]
            q = op(d0)[1]
            rec["levels"][str(lv)] = dict(dt_p_us=dt_p * 1e6, coefficient=row, asym=q["asym"], inv=q["inv"])
        out["pores"][shape] = rec
        for name in ("fixed", "equal pair", "Gauss pair", "Gaussian (6 atoms)"):
            ax.plot(cfg["levels"], [rec["levels"][str(lv)]["coefficient"][name][0] for lv in cfg["levels"]],
                    "o-", ms=4, label=name)
        ax.set(title=f"{shape} {diam*1e6:.1f} um, slowest mode", xlabel=r"$\delta_0 / a$")
        ax.axhline(0, color="k", lw=0.5)
    axes[0].set(ylabel=r"rate error / ($\lambda_1 dt_p / 5$)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "operator.png", dpi=130)
    plt.close(fig)
    out["packing"] = packing_certificate(cfg)
    return out


PACKING_VF = 0.30   #: the sphere packing's volume fraction (random sequential addition jams near 0.38)


def packing():
    """Twelve 2.5 um spheres in a periodic cell at ``PACKING_VF``: ``(radii, centers, L)``."""
    from dmipy_sim.geometry.packing import pack_spheres
    radii = np.full(12, 2.5e-6)
    centers, L, _ = pack_spheres(radii, target_vf=PACKING_VF, seed=3)
    return radii, centers, L


def packing_certificate(cfg):
    """The two-walk certificate on both pools of a packing of spheres: the envelope rate and coefficient size the
    step of a geometry that declares no pore, and the measurement says whether the convex (intra) and the concave
    (extra) side stay within what the envelope promises."""
    import dmipy_sim as d
    from dmipy_sim.engine.core import _walk_geometry
    from dmipy_sim.engine.accumulate import block_length
    from dmipy_sim.engine.physics import tolerance_sub_steps, regime_sub_steps
    from dmipy_sim.math.step_law import step_certificate, PILOT_STEP_RATIO
    radii, centers, L = packing()
    T, f_band, K = CERT_T, CERT_F_BAND, CERT_K
    h = block_length(T, K)
    out = {}
    for pool in ("intra", "extra"):
        g = d.PackedSpheres(radii, centers, L, pool=pool)
        n_rule = tolerance_sub_steps(g, D0, h, f_band=f_band)
        n_fine = max(2, n_rule // cfg["pack_ratio"])
        n_coarse = max(regime_sub_steps(g, D0, h), n_fine // PILOT_STEP_RATIO)
        eps = EPS * n_rule / n_fine
        n = cfg["pack_walkers"]
        t0 = time.time()
        kw = dict(K=K, require_gpu=False, window_T=T, tiers=())
        fine = _walk_geometry(n, D0, g, T, seed=21, sub_steps=n_fine, **kw)
        coarse = _walk_geometry(n, D0, g, T, seed=22, sub_steps=n_coarse, **kw)
        cert = step_certificate(fine.position_coeffs, coarse.position_coeffs, T, n_fine / n_coarse, f_band=f_band,
                                eps_declared=eps)
        out[pool] = dict(n_walkers=n, n_rule=n_rule, n_fine=n_fine, n_coarse=n_coarse, eps_declared=eps, h_s=h, K=K,
                         radius_um=2.5, volume_fraction=float(4 / 3 * np.pi * (radii ** 3).sum() / L ** 3),
                         min_gap_um=float(g.min_gap * 1e6), seconds=time.time() - t0,
                         **{k: cert[k] for k in ("eps_measured", "eps_se", "floor", "resolves", "n_bins")})
        log(f"packing {pool}: rule {n_rule} fine {n_fine} coarse {n_coarse} eps {eps:.1e} measured "
            f"{cert['eps_measured']:.2e} +- {cert['eps_se']:.1e} floor {cert['floor']:.1e}")
    return out


# ----------------------------------------------------------------------------------------------- the fused engine
def sec_fused(cfg, fig_dir):
    import dmipy_sim as d
    import matplotlib.pyplot as plt
    R = rows()
    out = {}
    geoms = {s: PORES[s] for s in cfg["fused_geoms"]}
    fig, axes = plt.subplots(len(geoms), len(cfg["fused_rows"]), figsize=(4.2 * len(cfg["fused_rows"]), 3.3 * len(geoms)),
                             squeeze=False)
    for gi, (shape, (cls, diam)) in enumerate(geoms.items()):
        sp = stepped_pore(shape, diam, cfg)
        geom = make_pore(shape, diam)
        for ri, label in enumerate(cfg["fused_rows"]):
            g = g_axis(R[label])
            ref = sp.continuous(g, DT)
            rec = dict(reference=ref, steps={})
            for sub in cfg["fused_subs"]:
                dt_p = DT / sub
                T_pair = sp.law_operator(dt_p, **cfg["quad"])
                T_fixed = sp.operator(np.sqrt(6 * D0 * dt_p), **cfg["quad"])[0]
                pred_pair = sp.signal(T_pair, g, DT, sub)
                pred_fixed = sp.signal(T_fixed, g, DT, sub)
                means, ses = [], []
                for seed in cfg["fused_seeds"]:
                    t0 = time.time()
                    res = simulate_fused(cfg["fused_n"], D0, waveform=R[label], geometry=geom, seed=seed, sub_steps=sub, return_walker_signals=True, require_gpu=False)
                    c = np.cos(np.asarray(res[2], np.float64).reshape(-1))
                    means.append(float(c.mean()))
                    ses.append(float(c.std() / np.sqrt(c.size)))
                    log(f"fused {shape} {label} sub {sub} seed {seed}: {means[-1]:.5f} +- {ses[-1]:.1e} "
                        f"(operator pair {pred_pair:.5f}) {time.time()-t0:.0f}s")
                m = float(np.mean(means))
                se = float(np.sqrt(np.sum(np.square(ses))) / len(ses))
                rec["steps"][str(sub)] = dict(dt_p_us=dt_p * 1e6, operator_pair=pred_pair, operator_fixed=pred_fixed,
                                              engine=m, engine_se=se, seeds=list(cfg["fused_seeds"]),
                                              n_walkers=cfg["fused_n"], pull=(m - pred_pair) / se)
            out[f"{shape}/{label}"] = rec
            ax = axes[gi, ri]
            subs = [str(s) for s in cfg["fused_subs"]]
            x = [rec["steps"][s]["dt_p_us"] for s in subs]
            ax.plot(x, [rec["steps"][s]["operator_fixed"] - ref for s in subs], "s--", color="gray", label="operator, fixed step")
            ax.plot(x, [rec["steps"][s]["operator_pair"] - ref for s in subs], "o-", color="C0", label="operator, Gauss pair")
            ax.errorbar(x, [rec["steps"][s]["engine"] - ref for s in subs], yerr=[rec["steps"][s]["engine_se"] for s in subs],
                        fmt="x", color="C3", capsize=3, label=f"fused engine, {cfg['fused_n']:,} walkers")
            ax.axhline(0, color="k", lw=0.5)
            ax.set(title=f"{shape} {diam*1e6:.1f} um, {label}", xlabel=r"$dt_p$ (us)", ylabel="S - continuous")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "fused.png", dpi=130)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------------------------- the certificate
def sec_certificate(cfg, fig_dir):
    import dmipy_sim as d
    from dmipy_sim.engine.core import _walk_geometry
    import matplotlib.pyplot as plt
    from dmipy_sim.engine.accumulate import block_length
    from dmipy_sim.engine.physics import tolerance_sub_steps, regime_sub_steps
    from dmipy_sim.math.step_law import step_certificate, PILOT_STEP_RATIO
    g = d.Sphere(2e-6)
    T, f_band, K = CERT_T, CERT_F_BAND, CERT_K
    h = block_length(T, K)
    n_rule = tolerance_sub_steps(g, D0, h, f_band=f_band)
    out = dict(radius_um=2.0, T=T, f_band_hz=f_band, K=K, h_s=h, n_rule=n_rule, ladder=[])
    configs = [(cfg["cert_ratio"], n) for n in cfg["cert_walkers"]]
    if cfg["cert_ratio"] != 1:
        configs.append((1, cfg["cert_walkers"][-1]))                 # the rule's own step at the top of the ladder
    for ratio, n in configs:
        n_fine = max(2, n_rule // ratio)
        n_coarse = max(regime_sub_steps(g, D0, h), n_fine // PILOT_STEP_RATIO)
        eps = EPS * n_rule / n_fine
        t0 = time.time()
        kw = dict(K=K, require_gpu=False, window_T=T, tiers=())
        fine = _walk_geometry(n, D0, g, T, seed=1, sub_steps=n_fine, **kw)
        coarse = _walk_geometry(n, D0, g, T, seed=2, sub_steps=n_coarse, **kw)
        cert = step_certificate(fine.position_coeffs, coarse.position_coeffs, T, n_fine / n_coarse, f_band=f_band,
                                eps_declared=eps)
        rec = dict(n_walkers=n, ratio=ratio, n_fine=n_fine, n_coarse=n_coarse, eps_declared=eps, seconds=time.time() - t0,
                   **{k: cert[k] for k in ("eps_measured", "eps_se", "floor", "resolves", "n_bins")})
        out["ladder"].append(rec)
        log(f"certificate n {n} fine {n_fine}/{n_coarse}: eps {eps:.1e} measured {cert['eps_measured']:.2e} "
            f"+- {cert['eps_se']:.1e} floor {cert['floor']:.1e} resolves {cert['resolves']}")
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    lad = [r for r in out["ladder"] if r["ratio"] == out["ladder"][0]["ratio"]]
    n = [r["n_walkers"] for r in lad]
    ax.errorbar(n, [r["eps_measured"] for r in lad], yerr=[r["eps_se"] for r in lad], fmt="o", capsize=3, label="measured eps")
    ax.plot(n, [r["floor"] for r in lad], "s--", color="gray", label="floor (noise)")
    ax.axhline(lad[0]["eps_declared"], color="C3", lw=0.8, label=f"declared {lad[0]['eps_declared']:.1e}")
    ax.set(xscale="log", yscale="log", xlabel="walkers in each pilot", ylabel="relative error of D(f) over the band",
           title="The two-walk certificate on a 2 um sphere")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "certificate.png", dpi=130)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------------------------- the cost
def cost_substrates(cfg):
    import dmipy_sim as d
    subs = {"sphere 2 um": d.Sphere(1e-6), "cylinder 20 um": d.Cylinder(10e-6, (0.0, 0.0, 1.0)),
            "plane 8.5 um": d.Box1D(8.5e-6)}
    radii, centers, L = packing()
    subs["packed spheres 2.5 um (both pools)"] = d.PackedSpheres(radii, centers, L)
    import trimesh
    ico = trimesh.creation.icosphere(subdivisions=3, radius=4e-6)
    subs["icosphere mesh 4 um"] = d.Mesh(np.asarray(ico.vertices), np.asarray(ico.faces))
    path = os.environ.get("STEP_LAW_MESH")
    if path and cfg["mesh_full"]:
        t0 = time.time()
        try:
            m = d.Mesh.from_ply(path, scale=1e-6)
        except Exception as e:                       # the row is reported as missing, the page still builds
            log(f"mesh {Path(path).name} not loaded: {e!r}")
        else:
            log(f"mesh {Path(path).name}: {len(m.faces):,} faces, built in {time.time()-t0:.0f}s")
            subs[f"mesh {Path(path).name[:40]}"] = m
    return subs


def sec_cost(cfg, fig_dir):
    import dmipy_sim as d
    from dmipy_sim.engine.core import _walk_geometry
    from dmipy_sim.engine.accumulate import block_length
    from dmipy_sim.engine.physics import sub_step_rules, collision_sub_steps, length_scales_of
    from dmipy_sim.math.step_law import LONGEST_STEP
    out = {}
    T = float(cfg["cost_T"])
    K = int(math.ceil(2.0 * T * F_BAND))                 # the band's own bands over the walk, and its block
    h = block_length(T, K)
    n_blocks = int(round(T / h))
    for name, g in cost_substrates(cfg).items():
        rules = sub_step_rules(g, D0, h, f_band=F_BAND)
        n = max(rules.values())
        ls = length_scales_of(g)
        rec = dict(rules={k: int(v) for k, v in rules.items()}, sub_steps=int(n), binds=max(rules, key=rules.get),
                   min_feature_um=None if ls.min_feature is None else ls.min_feature * 1e6,
                   lookup_cell_um=None if ls.lookup_cell is None else ls.lookup_cell * 1e6)
        if ls.min_feature:
            rec["retired_R6"] = int(math.ceil(h / ((ls.min_feature / 6) ** 2 / (6 * D0))))
        if ls.lookup_cell:
            cs = ls.lookup_cell
            rec["lookup_fixed_step"] = int(max(1, math.ceil((np.sqrt(6 * D0 * h) / (0.9 * cs)) ** 2)))
            rec["lookup_longest_atom"] = int(collision_sub_steps(g, D0, h))
            assert rec["lookup_longest_atom"] == rules["collision lookup"]
        try:
            _walk_geometry(min(2000, cfg["cost_n"]), D0, g, T, K=K, seed=5, require_gpu=False, sub_steps=n)
            t0 = time.time()
            _walk_geometry(cfg["cost_n"], D0, g, T, K=K, seed=5, require_gpu=False, sub_steps=n)
            sec = time.time() - t0
        except ValueError as e:                      # a surface the bare producer cannot seed (an open CACTUS box)
            rec["not_walked"] = str(e).splitlines()[0][:160]
            out[name] = rec
            log(f"cost {name}: rules {rules} -> {n} ({rec['binds']}); not walked: {rec['not_walked']}")
            continue
        rec.update(n_walkers=cfg["cost_n"], K=K, h_s=h, n_blocks=n_blocks, seconds=sec,
                   walker_steps_per_s=cfg["cost_n"] * n_blocks * n / sec,
                   walker_blocks_per_s=cfg["cost_n"] * n_blocks / sec)
        out[name] = rec
        log(f"cost {name}: rules {rules} -> {n} ({rec['binds']}); {rec['walker_steps_per_s']:.2e} walker-steps/s")
    return out


# ----------------------------------------------------------------------------------------------- outside
def sec_outside(cfg, fig_dir):
    R = rows()
    sp = stepped_pore("plane", PORES["plane"][1], cfg)
    g = {label: g_axis(seq) for label, seq in R.items()}
    out = {}
    for label in R:
        ref = sp.signal(None, g[label], DT, 16, phase="midpoint")
        rec = {}
        for sub in (1, 4):
            dt_p = DT / sub
            T_pair = sp.law_operator(dt_p, **cfg["quad"])
            both = sp.signal(T_pair, g[label], DT, sub)
            disp = sp.signal(T_pair, g[label], DT, sub, phase="midpoint")
            phase = sp.signal(None, g[label], DT, sub)
            rec[str(sub)] = dict(dt_p_us=dt_p * 1e6, total=both - ref, dispersion=disp - ref, phase_rule=phase - ref)
        out[label] = dict(reference=ref, steps=rec)
    return out


# ----------------------------------------------------------------------------------------------- the page
def fmt(v, digits=2):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    a = abs(v)
    if a == 0:
        return "0"
    if 1e-2 <= a < 1e4:
        return f"{v:.{digits + 1}g}" if a < 1 else f"{v:,.{digits}f}".rstrip("0").rstrip(".")
    return f"{v:.{digits - 1}e}"


def render(R, meta):
    L = []
    L.append("<!-- generated by tools/gen_step_law.py; edit the script, not this page -->")
    L.append("# The step law, shown")
    L.append("")
    L.append(f"Built from `dmipy-sim@{meta['commit']}` on {meta['date']} at tier `{meta['tier']}` on {meta['device']}; "
             f"the script is `tools/gen_step_law.py`, the numbers it measured are in `step-law/step_law.json`. "
             f"Every value on this page is one of those measurements.")
    L.append("")
    L.append("The walk's step is the **Gauss pair**: an isotropic direction times `sqrt(6 D dt_p)` times one of two "
             "lengths, so that the step's second, fourth and sixth moments are the Gaussian step's. The physics step "
             "`dt_p` is what holds the walk's diffusion spectrum within a declared tolerance `eps_D` up to the "
             "acquisition's band, and the build's pilot measures that error from two walks and records it. This page "
             "is the evidence; the invariants are in `tests/math/test_step_law.py`, `tests/test_step_rule.py` and "
             "`tests/math/test_step_certificate.py`.")
    L.append("")
    # 1 laws
    if "laws" in R:
        s = R["laws"]
        L.append("## 1. The laws")
        L.append("")
        L.append("A step of length `delta` multiplies a bulk mode `k` by `sinc(k delta)`; averaged over the Gaussian step's "
                 "length law (`|r|^2 / 2 D dt ~ chi^2_3`) that is `exp(-D k^2 dt)` exactly. A fixed length has a fourth "
                 "moment of 36 against the Gaussian's 60 (in `D^2 dt^2`), and that shortfall is the whole first-order "
                 "rate error `lambda dt_p / 5`. The n-point Gauss-Laguerre quadrature of the chi^2_3 law matches `2n - 1` "
                 "moments; the pair matches three. Moments are in units of the Gaussian step's; `reach` is the largest "
                 "`k delta_0` whose rate error stays within the tolerance.")
        L.append("")
        L.append("| law | lengths / delta_0 | weights | E[d^2] | E[d^4] | E[d^6] | longest | reach at 1e-3 | reach at 1e-4 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for name, r in s["laws"].items():
            L.append(f"| {name} | {', '.join(f'{v:.3f}' for v in r['lengths'])} | {', '.join(f'{v:.3f}' for v in r['weights'])} | "
                     f"{r['m2']:.3f} | {r['m4']:.3f} | {r['m6']:.3f} | {r['longest']:.2f} | {r['reach_1e3']:.2f} | {r['reach_1e4']:.2f} |")
        L.append("")
        L.append("![bulk error of each law](step-law/laws.png)")
        L.append("")
        e = s["engine"]
        L.append(f"The engine's `pair_step` over {e['n']:,} draws: E[d^2] {e['m2']:.4f}, E[d^4] {e['m4']:.4f}, "
                 f"E[d^6] {e['m6']:.4f}, longest step {e['longest']:.4f} delta_0, long-step frequency {e['p_long']:.4f} "
                 f"(the law's 0.18377), lengths drawn {', '.join(f'{v:.4f}' for v in e['lengths'])}.")
        L.append("")
    # 2 operator
    if "operator" in R:
        s = R["operator"]
        L.append("## 2. The operator on the three pores, and the envelope on a packing")
        L.append("")
        L.append("The one-step operator of the engine's own `reflect`, projected on the pore's eigenmodes by quadrature "
                 "(`dmipy_sim.math.step_law.SteppedPore`). Each entry is the rate error of the pore's slowest mode under "
                 "one step, in units of `lambda_1 dt_p / 5`, against the step `delta_0 / a` (`a` the radius or the slab's "
                 "half width). A value that stays constant as the step shrinks is a first-order term; the matched laws "
                 "share it, and it is zero on a flat wall.")
        L.append("")
        for shape, rec in s["pores"].items():
            lv = list(rec["levels"])
            L.append(f"**{shape} {rec['diameter_um']:.1f} um**, slowest mode of rate {rec['modes'][0]['rate_hz']:.0f} /s:")
            L.append("")
            L.append("| law | " + " | ".join(f"delta_0/a = {v}" for v in lv) + " |")
            L.append("|---|" + "---|" * len(lv))
            for name in rec["levels"][lv[0]]["coefficient"]:
                L.append(f"| {name} | " + " | ".join(f"{rec['levels'][v]['coefficient'][name][0]:+.2f}" for v in lv) + " |")
            L.append("")
        L.append("![the operator's first-order coefficient](step-law/operator.png)")
        L.append("")
        p = s["packing"]
        L.append("A geometry that declares no pore (a packing, a mesh) takes the envelope: a sphere of its smallest "
                 "feature with the sphere's coefficient. On a packing of 2.5 um spheres the two-walk certificate measures "
                 "each pool's real error at the step the envelope sizes; the extracellular pool sees the concave side of "
                 "every wall, which the sphere's coefficient does not describe, so the measurement is what certifies it:")
        L.append("")
        L.append("| pool | walkers | fine / coarse sub-steps | declared eps | measured eps | se | floor | resolves |")
        L.append("|---|---|---|---|---|---|---|---|")
        for pool, r in p.items():
            L.append(f"| {pool} | {r['n_walkers']:,} | {r['n_fine']} / {r['n_coarse']} | {r['eps_declared']:.1e} | "
                     f"{r['eps_measured']:.2e} | {r['eps_se']:.1e} | {r['floor']:.1e} | {fmt(r['resolves'])} |")
        L.append("")
    # 3 fused
    if "fused" in R:
        s = R["fused"]
        L.append("## 3. The fused engine against the operator")
        L.append("")
        L.append("The operator is one step of the engine's `reflect` in float64 quadrature. The fused engine runs the "
                 "whole walk in float32 with its bounce budget, nudges and sub-step loop, the gradient applied in the walk. "
                 "If the two agree to the standard error, the loop adds nothing the operator does not predict. `pull` is "
                 "the engine's distance from the pair operator in standard errors.")
        L.append("")
        L.append("| pore, row | dt_p (us) | continuous | operator, fixed | operator, pair | engine | se | pull | walkers x seeds |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for key, rec in s.items():
            for sub, r in rec["steps"].items():
                L.append(f"| {key} | {r['dt_p_us']:.0f} | {rec['reference']:.5f} | {r['operator_fixed']:.5f} | "
                         f"{r['operator_pair']:.5f} | {r['engine']:.5f} | {r['engine_se']:.1e} | {r['pull']:+.1f} | "
                         f"{r['n_walkers']:,} x {len(r['seeds'])} |")
        L.append("")
        L.append("![engine against operator](step-law/fused.png)")
        L.append("")
        if "operator" in R:
            asym = {shape: max(lv["asym"] for lv in rec["levels"].values()) for shape, rec in R["operator"]["pores"].items()}
            L.append("The operator's own precision is its symmetry check, the largest asymmetry of the one-step matrix over the "
                     "levels above: " + ", ".join(f"{k} {v:.0e}" for k, v in asym.items()) + ". A row whose standard error is "
                     "below that check reads the operator's quadrature, not the engine; a pull there is not a defect of the walk "
                     "until the operator is refined past it.")
            L.append("")
    # 4 certificate
    if "certificate" in R:
        s = R["certificate"]
        L.append("## 4. The certificate's resolution")
        L.append("")
        L.append(f"The two-walk certificate on a {s['radius_um']:.0f} um sphere read to {s['f_band_hz']:.0f} Hz over "
                 f"{s['T']*1e3:.0f} ms, both walks keeping K = {s['K']} bands in the band's own blocks of "
                 f"{s['h_s']*1e6:.0f} us (the rule's count at `eps_D = 1e-3`: {s['n_rule']} sub-steps per block). "
                 "Each walk's diffusion spectrum is read off its stored bands, `D(f_k) = (pi k)^2 <beta_k^2> / (2 T^2)` "
                 "at `f_k = k / (2 T)`; the statistic is the noise-corrected rms over the band's bands of the relative "
                 "difference between the fine and the coarse walk's spectra, and its floor is the noise it cannot see "
                 "under, falling as one over the square root of walkers times bands. A pilot resolves a tolerance when "
                 "its floor is under it; the record says so.")
        L.append("")
        L.append("| walkers | fine / coarse | declared | measured | se | floor | resolves | seconds |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in s["ladder"]:
            L.append(f"| {r['n_walkers']:,} | {r['n_fine']} / {r['n_coarse']} | {r['eps_declared']:.1e} | "
                     f"{r['eps_measured']:.2e} | {r['eps_se']:.1e} | {r['floor']:.1e} | {fmt(r['resolves'])} | {r['seconds']:.0f} |")
        L.append("")
        L.append("![certificate against walkers](step-law/certificate.png)")
        L.append("")
    # 5 cost
    if "cost" in R:
        s = R["cost"]
        L.append("## 5. The cost")
        L.append("")
        h_us = next(iter(s.values())).get("h_s", DT) * 1e6
        L.append(f"Sub-steps per block of the band's own {h_us:.0f} us that each rule asks for at `eps_D = 1e-3` and a "
                 f"{F_BAND:.0f} Hz band, "
                 "the rule that binds, and the walk's throughput on this device. `R/6` is the retired reflection rule "
                 "(`delta_0 = min_feature / 6`), shown for the pores it applied to. On a mesh the collision lookup bounds the "
                 "LONGEST step by the cell: `fixed` is that count for a fixed step, `atom` for the pair's long atom, the "
                 "price of the bounded law where this rule binds.")
        L.append("")
        L.append("| substrate | step law | regime | lookup (fixed -> atom) | crossing | binds | sub-steps | R/6 | walker-steps / s | walkers |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for name, r in s.items():
            ru = r["rules"]
            lk = f"{r['lookup_fixed_step']} -> {r['lookup_longest_atom']}" if "lookup_longest_atom" in r else "-"
            rate = f"{r['walker_steps_per_s']:.2e}" if "walker_steps_per_s" in r else f"not walked: {r['not_walked'][:110]}"
            walkers = f"{r['n_walkers']:,}" if "n_walkers" in r else "-"
            L.append(f"| {name} | {ru.get('step law', '-')} | {ru.get('regime', '-')} | {lk} | {ru.get('crossing', '-')} | "
                     f"{r['binds']} | {r['sub_steps']} | {r.get('retired_R6', '-')} | {rate} | {walkers} |")
        L.append("")
    # 6 outside
    if "outside" in R:
        s = R["outside"]
        L.append("## 6. What the certificate does not see")
        L.append("")
        L.append("The walk accrues each sub-step's phase from the position at the step's end, a rectangle rule for the "
                 "integral of `G . x`. On the slab the operator separates the two remaining terms: the step law's "
                 "dispersion (third order under the pair, and zero at first order on a flat wall) and the phase rule's, "
                 "whose error falls as the square of the step (the increment within a step has zero mean, so the rule's "
                 "first-order term averages out and its variance is what remains). The certificate reads trajectories, "
                 "so it measures the first and not the second; the phase rule is a kernel matter of its own.")
        L.append("")
        L.append("| row, plane 8.5 um | dt_p (us) | total error | dispersion (pair) | phase rule |")
        L.append("|---|---|---|---|---|")
        for label, rec in s.items():
            for sub, r in rec["steps"].items():
                L.append(f"| {label} | {r['dt_p_us']:.0f} | {r['total']:+.1e} | {r['dispersion']:+.1e} | {r['phase_rule']:+.1e} |")
        L.append("")
        L.append("The CUDA backend's kernels take the same two atoms and weight from the request and are admitted against "
                 "this operator (dmrai-lab/dmipy-sim-cuda#60).")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tier", choices=list(TIERS), default="small")
    ap.add_argument("--out", default=str(ROOT / "docs" / "step-law.md"))
    ap.add_argument("--only", default=",".join(SECTIONS), help="sections to rerun; the rest are read from the json (`none`: render only)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    fig_dir = out.parent / "step-law"
    fig_dir.mkdir(parents=True, exist_ok=True)
    json_path = fig_dir / "step_law.json"
    R = json.loads(json_path.read_text()) if json_path.exists() else {}
    kept = R.pop("meta", None)
    cfg = TIERS[a.tier]
    import matplotlib
    matplotlib.use("Agg")
    import jax
    dev = jax.devices()[0]
    device = f"{dev.device_kind}" if dev.platform == "gpu" else "a CPU"
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
    funcs = dict(laws=sec_laws, operator=sec_operator, fused=sec_fused, certificate=sec_certificate, cost=sec_cost, outside=sec_outside)
    for name in ([] if a.only in ("", "none") else a.only.split(",")):
        t0 = time.time()
        log(f"section {name} ...")
        R[name] = funcs[name](cfg, fig_dir)
        log(f"section {name} done in {time.time()-t0:.0f}s")
        json_path.write_text(json.dumps(R, indent=1, default=float))
    meta = dict(commit=commit, date=_dt.date.today().isoformat(), tier=a.tier, device=device, D0=D0, dt=DT)
    if a.only in ("", "none") and kept:
        meta = dict(kept, commit=commit)                  # a render-only pass keeps the measurements' tier and device
    json_path.write_text(json.dumps(dict(R, meta=meta), indent=1, default=float))
    out.write_text(render(R, meta), encoding="utf-8")
    log(f"wrote {out} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
