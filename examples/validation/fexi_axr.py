"""FEXI on permeable packed cylinders: the apparent exchange rate against the membrane permeability, and its replay.

One acquisition object throughout: the filter-exchange protocol of the acquisition layer (``dmipy_replay.fexi_protocol``:
the filter off and on, every mixing time, two detection b-values), walked by dmipy-sim's ``simulate`` and fitted by
dmipy-fit's AXR analysis (``FEXIApparentExchange``). Nothing here builds a waveform or fits a curve of its own.

* **(B) the walks.** Packed cylinders of radius ``R`` (``pack_cylinders``, area fraction ``f``), the gradient across
  them, a membrane permeability ``kappa`` swept so the two-site rate ``2 kappa / (R (1 - f))`` spans about 1 to 15 s^-1,
  and ``kappa = 0``, whose recovery is the geometry's alone (water moving between regions of the extra-cellular space
  that the filter weights differently). Each sequence of the protocol is one walk (``simulate``'s own: the coefficient
  walk of the sequence's span, contracted with it); its walkers give each signal's standard error and, through the
  fit's sensitivities, the AXR's.
* **The barrier-limited prediction.** With ``kappa R / D << 1`` the exchange is limited by the membrane, so the pools
  are the lumens (``f``) and the interstitium, exchanging at ``k_12 = 2 kappa / R`` (the surface over the volume of a
  lumen) and ``k_21 = k_12 f / (1 - f)``; their diffusivities are the ``kappa = 0`` walk's own, pool by pool, at this
  protocol's blocks. The two-site Karger signal of those pools on the same protocol, fitted by the same AXR analysis, is
  the prediction (it carries the analysis' own finite-b reading); the two-site rate ``k_12 + k_21`` is beside it.
* **(C) the replay.** One walk of the sweep is written as a replay pack (``.rpk``), read back and replayed on the same
  ``ScannerSequence``: it equals ``simulate``'s signal of that walk within the pack's certificate (``fidelity.err_max``).

Run on a GPU (``JAX_PLATFORMS=cuda``); ``--pilot`` sizes a short run. Writes a JSON record of every number.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import dmipy_replay as dr
from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
from dmipy_fit.signal_models.exchange_models import X0GeneralizedKarger
from dmipy_fit.signal_models.filter_exchange import FEXIApparentExchange, fexi_measurements
from dmipy_fit.signal_models.gaussian_models import G1Ball
from dmipy_sim.engine.core import _walk_band, _walk_geometry, _replay_signal
from dmipy_sim.geometry import PackedCylinders, pack_cylinders
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.compression import acquisition_envelope

D = 2e-9                     # m^2/s, both pools
R = 1.5e-6                   # m
N_CYL, VF, PACK_SEED = 16, 0.5, 0
DELTA, BIG_DELTA, DT = 4e-3, 10e-3, 2e-5
B_F, B_D = 1.2e9, 0.3e9      # s/m^2: the filter (b_f D ~ 1-2 for the interstitium) and the detection
T_M = (0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0)
AXR_TARGETS = (1.0, 2.0, 4.0, 8.0, 15.0)


def substrate(kappa):
    centers, L, vf = pack_cylinders(np.full(N_CYL, R), target_vf=VF, seed=PACK_SEED)
    return PackedCylinders(np.full(N_CYL, R), centers, L, orientation=(0.0, 0.0, 1.0),
                           permeability=float(kappa)), float(vf)


def protocol(t_m):
    return dr.fexi_protocol([1.0, 0.0, 0.0], DELTA, BIG_DELTA, t_m, b_f=B_F, bvalues=[0.0, B_D], dt=DT,
                            slew_rate=np.inf)


def walk_of(seq, geom, n, seed, tiers=()):
    """``simulate``'s walk of ``seq``: the coefficient walk over its span at the band it reads, and its envelope."""
    G = np.asarray(seq.G_eff, np.float64)
    T = seq.dt * (G.shape[1] - 1)
    K, f_band = _walk_band(G, seq.dt, T)
    walk = _walk_geometry(n, D, geom, T, seed=seed, tiers=tiers, K=K, f_band=f_band, window_T=T, require_gpu=None)
    envelope = acquisition_envelope(seq, band_hz=float(walk.K) / (2.0 * float(walk.T_window)))
    return walk, envelope


def walk_protocol(P, geom, n, seed0, compartments=False):
    """Every sequence of ``P`` walked on its own (seed ``seed0 + i``): the signal (``simulate``'s), its covariance from
    the walkers, and with ``compartments`` the signal of each pool (lumens 1, interstitium 0)."""
    S, C, pools = np.empty(P.n_meas), np.zeros((P.n_meas, P.n_meas)), {0: np.empty(P.n_meas), 1: np.empty(P.n_meas)}
    for i, (seq, rows) in enumerate(zip(P, P.rows)):
        walk, env = walk_of(seq, geom, n, seed0 + i, tiers=("all" if compartments else ()))
        pack = build_replay_pack(walk, id="fexi-axr", license="in-memory", citation="fexi-axr", envelope=env)
        S[rows] = _replay_signal(pack, seq, None, geom._orient_R)
        w, ew, E = pack.walker_signals(seq, orientation=geom._orient_R, tol=None)
        x = np.real(np.asarray(ew)[:, None] * np.asarray(E)) * (len(w) / float(np.sum(w)))
        C[np.ix_(rows, rows)] = np.cov(x, rowvar=False) / x.shape[0]
        if compartments:
            for c in (0, 1):
                pools[c][rows] = np.real(np.asarray(pack.replay(seq, orientation=geom._orient_R, compartment=c,
                                                                tol=None, complex_signal=True)))
    return S, C, pools


def fit_axr(scheme, S, C=None):
    """``(adc_eq, sigma, axr)`` and, with ``C``, their standard errors through the fit's sensitivities."""
    model = FEXIApparentExchange()
    p = model.fit(scheme, S).fitted_parameters
    theta = np.array([float(p[k]) for k in model.parameter_names])
    if C is None:
        return theta, None
    J = np.empty((3, S.size))
    for j in range(S.size):
        h = 1e-4 * max(abs(S[j]), 1e-3)
        y = S.copy()
        y[j] += h
        q = model.fit(scheme, y).fitted_parameters
        J[:, j] = (np.array([float(q[k]) for k in model.parameter_names]) - theta) / h
    return theta, np.sqrt(np.diag(J @ C @ J.T))


def pool_diffusivities(scheme, pools):
    """Each pool's ADC over the filter-off rows (the detection block at ``B_D``), ``(D_interstitium, D_lumen)``."""
    m = fexi_measurements(scheme)
    out = []
    for c in (0, 1):
        adc = [-np.log(pools[c][(m.series == s) & (m.b_d > 0)][0] / pools[c][(m.series == s) & (m.b_d == 0)][0]) /
               m.b_d[(m.series == s) & (m.b_d > 0)][0] for s in np.unique(m.series[~m.filtered])]
        out.append(float(np.mean(adc)))
    return tuple(out)


def two_site(scheme, f, D_lumen, D_inter, kappa):
    k12 = 2.0 * kappa / R
    E = X0GeneralizedKarger(G1Ball(), G1Ball())(scheme, G1Ball_1_lambda_iso=D_lumen, G1Ball_2_lambda_iso=D_inter,
                                                f=f, kappa=k12)
    return fit_axr(scheme, E)[0], k12 / (1.0 - f)


def replay_check(seq, geom, n, seed, path):
    """(C): one walk, ``simulate``'s signal of it and the signal its written pack replays, against the certificate."""
    walk, env = walk_of(seq, geom, n, seed)
    mem = build_replay_pack(walk, id="fexi-axr", license="in-memory", citation="fexi-axr", envelope=env)
    S_direct = np.asarray(_replay_signal(mem, seq, None, geom._orient_R), np.float64)
    build_replay_pack(walk, id="fexi-axr/replay-check", license="CC-BY-4.0", citation="fexi-axr", envelope=env,
                      out_path=str(path))
    pack = dr.read_rpk(path)
    S_full = np.real(np.asarray(pack.replay(seq, orientation=geom._orient_R, tol=None, complex_signal=True)))
    planned = pack.replay(seq, orientation=geom._orient_R, complex_signal=True)
    fid = pack.meta.get("fidelity", {})
    return dict(S_direct=S_direct.tolist(), S_replay=S_full.tolist(), S_replay_planned=np.real(np.asarray(planned)).tolist(),
                plan=str(planned.plan), err_max=fid.get("err_max"), floor_max=fid.get("floor_max"),
                max_abs_diff=float(np.max(np.abs(S_full - S_direct))),
                max_abs_diff_planned=float(np.max(np.abs(np.real(np.asarray(planned)) - S_direct))))


def verdict(runs, alpha=1e-3):
    """Monotone in kappa over the permeable runs (each step up by more than the family-wise band of its two standard
    errors; the ``kappa = 0`` recovery is the geometry's and is reported, not ranked) and agreeing with the
    two-site prediction where the exchange is barrier-limited (``kappa R / D <= 1e-2``): ``|AXR - AXR_pred| <= z SE`` with
    ``z`` the two-sided Bonferroni normal quantile of the comparisons at the family-wise false-alarm ``alpha``."""
    from scipy.stats import norm
    perm = [r for r in runs if r["kappa"] > 0]
    z_mono = float(norm.isf(alpha / (2.0 * max(1, len(perm)))))
    steps = [(b["axr"] - a["axr"], z_mono * np.hypot(a["axr_se"], b["axr_se"])) for a, b in zip(perm[:-1], perm[1:])]
    limited = [r for r in perm if r["kappa_R_over_D"] <= 1e-2]
    z = float(norm.isf(alpha / (2.0 * max(1, len(limited)))))
    agree = [dict(kappa=r["kappa"], axr=r["axr"], axr_predicted=r["axr_predicted"], two_site_rate=r["two_site_rate"],
                  z_score=(r["axr"] - r["axr_predicted"]) / r["axr_se"], within=bool(abs(r["axr"] - r["axr_predicted"])
                                                                                 <= z * r["axr_se"])) for r in limited]
    return dict(monotone=bool(all(d > band for d, band in steps)), steps=[dict(rise=d, band=band) for d, band in steps],
                z_family=z, agreement=agree, all_within=bool(all(a["within"] for a in agree)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-walkers", type=int, default=50_000)
    ap.add_argument("--out", default="fexi_axr.json")
    ap.add_argument("--pilot", action="store_true", help="two mixing times and two permeabilities")
    a = ap.parse_args(argv)
    t_m = (T_M[0], T_M[3]) if a.pilot else T_M
    n = a.n_walkers
    P = protocol(t_m)
    scheme = AcquisitionScheme(P)
    _, f = substrate(0.0)
    kappas = [0.0] + [axr * R * (1.0 - f) / 2.0 for axr in (AXR_TARGETS[:1] if a.pilot else AXR_TARGETS)]
    rec = dict(D=D, R=R, f=f, n_cylinders=N_CYL, delta=DELTA, Delta=BIG_DELTA, dt=DT, b_f=B_F, b_d=B_D, t_m=list(t_m),
               n_walkers=n, runs=[])
    t0 = time.time()
    pools = None
    for i, kappa in enumerate(kappas):
        geom, _ = substrate(kappa)
        t1 = time.time()
        S, C, pl = walk_protocol(P, geom, n, seed0=1000 * i, compartments=(kappa == 0.0))
        theta, se = fit_axr(scheme, S, C)
        run = dict(kappa=kappa, kappa_R_over_D=kappa * R / D, signal=S.tolist(), signal_se=np.sqrt(np.diag(C)).tolist(),
                   adc_eq=theta[0], sigma=theta[1], axr=theta[2], adc_eq_se=se[0], sigma_se=se[1], axr_se=se[2],
                   seconds=time.time() - t1)
        if kappa == 0.0:
            pools = pool_diffusivities(scheme, pl)
            rec["D_interstitium"], rec["D_lumen"] = pools
        else:
            pred, rate = two_site(scheme, f, pools[1], pools[0], kappa)
            run.update(axr_predicted=pred[2], sigma_predicted=pred[1], adc_eq_predicted=pred[0], two_site_rate=rate)
        rec["runs"].append(run)
        print(json.dumps({k: v for k, v in run.items() if not isinstance(v, list)}), flush=True)
    rec["verdict"] = verdict(rec["runs"])
    print(json.dumps(rec["verdict"]), flush=True)
    i_c = min(2, len(kappas) - 1)
    geom, _ = substrate(kappas[i_c])
    rec["replay"] = dict(kappa=kappas[i_c], t_m=t_m[-1],
                         **replay_check(P[-1], geom, n, 1000 * i_c + len(P) - 1, Path(a.out).with_suffix(".rpk")))
    print(json.dumps({k: v for k, v in rec["replay"].items() if not isinstance(v, list)}), flush=True)
    rec["seconds"] = time.time() - t0
    Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
