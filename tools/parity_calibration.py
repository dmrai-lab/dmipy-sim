"""Measure the pooled hit-count statistic of ``parity_report`` on real walks, on the CPU (a measurement, not a test).

    JAX_PLATFORMS=cpu python tools/parity_calibration.py --out new.json
    JAX_PLATFORMS=cpu python tools/parity_calibration.py --seed-rule legacy --out legacy.json

Per replicate (a report seed ``r = 0 .. R - 1``) and per default rate case of :func:`parity_cases`, the JAX kernels
walk the case's starts (stream 0), the sweep from those starts, and a stand-in backend: the reference itself at one
more independent stream from the same starts. The stand-in's hit count is scored against the sweep with the report's
own rule (:func:`_spread_p`), the ``t`` turned into normal scores (:func:`_normal_scores`) and summed over the cases
(Stouffer). A correct harness gives ``Z ~ N(0, 1)`` per replicate, so ``sum_r Z_r^2 ~ chi2(R)``.

Seed rules:

* ``new`` (the report's): ``case_seeds(r, name, sweep + 2)``; stream 0 the walk, streams ``1 .. sweep`` the sweep,
  the last the stand-in; the spread reads the sweep alone (``sweep - 1`` dof).
* ``legacy`` (the rule before the change): every case walks ``r``, the sweep ``r + 1 .. r + sweep``, the stand-in
  ``r + sweep + 1``; the spread reads stream 0 and the sweep (``sweep`` dof), as the report did.

One geometry object per case is reused over every walk, so each case compiles once. Output: one JSON with the
configuration, each case's resolved sub-steps, per replicate the per-case ``t`` and the pooled ``Z``, and over the
replicates the Pearson correlation matrix of ``t`` across cases, ``sum Z^2`` with its chi-square band at 1e-3
two-sided, ``Var(Z)``, the largest off-diagonal ``|rho|``, the pairs with ``rho > 0.9`` and the wall time.
"""
import argparse
import json
import logging
import time

import numpy as np

log = logging.getLogger("dmipy_sim.tools.parity_calibration")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the JSON written")
    ap.add_argument("--seed-rule", choices=("new", "legacy"), default="new")
    ap.add_argument("--replicates", type=int, default=40)
    ap.add_argument("--n-walkers", type=int, default=200)
    ap.add_argument("--T-max", type=float, default=2e-3)
    ap.add_argument("--block-h", type=float, default=0.5e-3)
    ap.add_argument("--sweep", type=int, default=4)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("dmipy_sim").setLevel(logging.WARNING)             # the walks' own records stay quiet
    log.setLevel(logging.INFO)

    from scipy.stats import chi2
    from dmipy_sim.engine.backends import _normal_scores, _spread_p, case_seeds, parity_cases
    from dmipy_sim.engine.core import _walk_geometry

    t0 = time.time()
    D, n, Tm, bh, m = 2e-9, int(args.n_walkers), float(args.T_max), float(args.block_h), int(args.sweep)
    cases = {name: c for name, c in parity_cases().items() if c.starts is None}
    names = list(cases)
    geoms = {}
    for name in names:
        g = cases[name](); g.count_walls = True
        geoms[name] = g

    def walk(name, seed, r0=None):
        kw = {} if r0 is None else dict(r0=r0)
        return _walk_geometry(n, D, geoms[name], Tm, bh, seed=int(seed), tiers="all", walker_batch_size=n, **kw)

    R = int(args.replicates)
    T = np.full((R, len(names)), np.nan)
    sub_steps, replicates = {}, []
    for r in range(R):
        row = {}
        for j, name in enumerate(names):
            if args.seed_rule == "new":
                s = case_seeds(r, name, m + 2)
                s0, sweep_seeds, standin = s[0], s[1:m + 1], s[-1]
            else:
                s0, sweep_seeds, standin = r, list(range(r + 1, r + 1 + m)), r + m + 1
            ref = walk(name, s0)
            sub_steps[name] = int(ref.sub_steps)
            x0 = np.asarray(ref.position_coeffs[:, 0, 0, :], np.float32)
            hits = [int(walk(name, sv, x0).work["n_hits"]) for sv in sweep_seeds]
            if args.seed_rule == "legacy":
                hits = [int(ref.work["n_hits"])] + hits                  # the old spread read stream 0 too
            b = int(walk(name, standin, x0).work["n_hits"])
            tv, _, sd = _spread_p(b, hits)
            T[r, j] = tv if sd > 0 else np.nan
            row[name] = dict(t=float(T[r, j]), backend=b, sweep=hits, seeds=dict(walk=int(s0), sweep=[int(v) for v in sweep_seeds], standin=int(standin)))
        dof = m - 1 if args.seed_rule == "new" else m
        ok = np.isfinite(T[r])
        Z = float(_normal_scores(T[r, ok], dof).sum() / np.sqrt(ok.sum()))
        replicates.append(dict(seed=r, Z=Z, cases=int(ok.sum()), per_case=row))
        log.info("replicate %d/%d: Z = %+.3f (%.0f s)", r + 1, R, Z, time.time() - t0)

    Zs = np.array([x["Z"] for x in replicates])
    rho = np.ma.corrcoef(np.ma.masked_invalid(T), rowvar=False).filled(np.nan)
    off = rho[~np.eye(len(names), dtype=bool)]
    i_max = np.nanargmax(np.abs(np.where(np.eye(len(names), dtype=bool), np.nan, rho)))
    pairs = [dict(a=names[i], b=names[k], rho=float(rho[i, k]))
             for i in range(len(names)) for k in range(i + 1, len(names)) if rho[i, k] > 0.9]
    band = [float(chi2.ppf(5e-4, R)), float(chi2.ppf(1 - 5e-4, R))]
    sz2 = float((Zs ** 2).sum())
    out = dict(
        seed_rule=args.seed_rule, replicates=R, n_walkers=n, T_max=Tm, block_h=bh, sweep=m, diffusivity=D,
        dof=m - 1 if args.seed_rule == "new" else m, cases=names, sub_steps=sub_steps, per_replicate=replicates,
        sum_Z2=sz2, sum_Z2_band=band, sum_Z2_within_band=bool(band[0] <= sz2 <= band[1]),
        var_Z=float(Zs.var(ddof=1)), mean_Z=float(Zs.mean()),
        correlation=[[None if not np.isfinite(v) else float(v) for v in row] for row in rho],
        max_abs_offdiag_rho=dict(rho=float(np.nanmax(np.abs(off))), a=names[i_max // len(names)], b=names[i_max % len(names)]),
        pairs_rho_above_0p9=pairs, wall_time_s=float(time.time() - t0),
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    log.info("sum Z^2 = %.2f (band %.1f .. %.1f), Var(Z) = %.3f, wall %.0f s", sz2, *band, out["var_Z"], out["wall_time_s"])


if __name__ == "__main__":
    main()
