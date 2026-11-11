"""Native permeable-shell full-grid diagnostic; see docs/permeable-shell-walk.md.

Run with python -m diagnostics.permeable_shell_walk. --n and --seeds are
required: the latter reads an ordered `seeds` list from a JSON object.
The checked-in companion JSON supplies the 90 archived batch seeds.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from dmipy_analytic import permeable_shell as reference_shell
from dmipy_sim.engine.physics import make_step_fn, seed_walkers
from dmipy_sim.geometry import PermeableShell

R, R_OUT, D = 5e-6, 8e-6, 2e-9
DELTA, BIG_DELTA, T = 0.005, 0.020, 0.025
DTS = (50e-6, 25e-6, 12.5e-6)
TIMES = (0.001, 0.005, 0.010, 0.025)
TARGETS = (0.8, 0.4, 0.1)
BATCH = 32768
Z_FAMILY = float(norm.isf(0.01 / (2 * 396)))
RATE = 2.0815759778181 ** 2 * D / R ** 2
MODAL_SLOPE = 1.18 * RATE / (5 * np.e)

REGISTERED_TOLERANCE = """Native permeable shell walk against the confirmed Bloch--Torrey reference (#48).

REGISTERED BEFORE ANY WALK (2026-11-11).

Prediction: both geometries, all three permeabilities and all final-pool signal
contributions agree with the intra-start reference within the following budget.
The pool signals are E[cos(phi) 1(final pool)], NOT conditional means or magnitudes;
the total is E[cos(phi)]. Imaginary means should be zero by radial symmetry.
The populations from the same paths are a zero-gradient exchange test: gradients
change only phase, never motion. All starts are the geometry's native placement.

R=5 um, R_out=8 um, D_in=D_out=2e-9 m^2/s; kappa=1e-5,1e-4,1e-3 m/s.
PGSE delta=5 ms, Delta=20 ms, gradient perpendicular to the cylinder. Three G
values are chosen BEFORE walking by the reference to give total E~0.8,0.4,0.1.
Physics dt=50,25,12.5 us (explicit sub_steps=1); N=262144 at EACH dt, with fresh,
nonzero, unrelated master seeds for every batch/run, distinct upper 16-bit seed
families. The three G values share paths, not independent walks; Bonferroni does
not assume their independence. The fast check uses N=65536 and only E~0.8.

Tolerance: |walk-ref| <= z_F SE + reference_error + B(dt), with empirical
SE=sd(per-walker REAL contribution, ddof=1)/sqrt(N). N=262144 guarantees
SE<=1/sqrt(N)=0.001954 (up to ddof); the fast low-G case must itself meet 0.002.
There are M=234 primary residuals (162 signal and 72 population), and also 162
imaginary-signal diagnostics, so z_F=norm.isf(0.01/(2*396)) (~4.21), a 1% family
error budget. The typical maximum is sqrt(2 ln 396)=3.46 SE, not 2 SE.

The CURRENT physics.py docstrings give a relative modal rate error c*lambda*dt/5,
with wall c=0.85 and membrane c=0.33, plus a one-sided-start transient bound
0.3*h^2/(D*t)=1.8*dt/t. Use the declared envelope lambda=(2.0815759778181)^2 D/R^2
for both geometries (the shell declares only min_feature). A rate perturbation
changes exp(-lambda*t) by at most its relative perturbation/e. Register the
absolute PGSE bias envelope B=(0.85+0.33)*lambda*dt/(5e)+1.8*dt/(Delta+delta).
For a population use the same modal envelope plus the transient relative error
on the exchanged fraction: B_p=(0.85+0.33)*lambda*dt/(5e)
+1.8*dt/t*(1-P_in). These are predictions to TEST, not a theorem extending the
slab calibration to curved membranes. They are not increased after observation.
Reference signals request <=2e-5 absolute error (reported separately), far below
MC SE. Exchange uses 160/320 intra cells, 96/192 extra cells, Richardson error.
Weighted fits E_walk-E_ref = intercept + slope*dt use all three independent dt
runs; slopes are per second, with standard errors, and no sign is assumed.

Prediction for placement: r^d/R^d is U(0,1), two-sided KS p>0.005 per geometry.
Also test six radius/direction features per geometry by Pearson correlation,
with 1% Bonferroni over 12. A passing finite sample cannot prove PRNG stream
independence. The inspected code reuses the parent key after splitting it;
this is a key-reuse hazard even if marginal and correlation tests pass.

Public API route: dmipy_sim.simulate(N,D,ScannerSequence,PermeableShell,...)
returns only the total real signal. dmipy_sim.walk(geometry,...,envelope=...)
returns CoefficientWalk.raw_positions/raw_compartment when raw_subset is given;
these permit per-pool phase sums, but retaining every step is expensive. Here
use the documented PUBLIC engine.physics.seed_walkers and make_step_fn, the
native scan body used by simulate, without reading any private attributes or
reimplementing motion/phase. Its carry contains phase and current compartment.
A separate fresh-seed simulate check verifies the ensemble entry point too.
No replay compression or geometry changes are involved.

RESOURCE AMENDMENT before the completed grid: the first 262144-walker coarse
run took 66.75 s. Extrapolation (six cases, cost ratio 1+2+4, ~1.8 CPU/wall)
slightly exceeds 90 CPU minutes. Reduce N to 163840 for EVERY completed-grid
resolution; cut no cases. The pilot's largest SE scales to 0.00174 at this N;
the test still REQUIRES measured SE<=0.002. The distribution-free guarantee
becomes 1/sqrt(N)=0.002471, while a cosine with sd~1/sqrt(2) has SE~0.001747.
No reference, target, tolerance, family count, or bias prediction changes.
The original timing data and the first KS outlier are retained in diagnostic
artifacts, not used to choose a physics expectation."""
TRANSIENT = """In the archived run, at kappa=1e-3 m/s, dt=50 µs and t=1 ms, the sphere intra population was 0.733014 versus 0.726684 (difference +0.006330, +5.79 SE); the cylinder was 0.818854 versus 0.815225 (+0.003628, +3.81 SE). These exceed a sampling-only interpretation but remain inside the registered transient budgets (0.030710 and 0.022144). At 25/12.5 µs the sphere differences were +0.001612/+0.001649; the cylinder differences were +0.002719/+0.000998. The envelope was not enlarged. All 162 signal and 72 population comparisons passed their fixed budgets; all 162 imaginary diagnostics passed. The largest measured full-grid signal SE was below 0.002. Slopes above include their uncertainty; the grid does not establish a universal slope sign."""
REPRODUCTION = """This report and its JSON were migrated from `.codex-out/permeable-shell-full-grid.json` without rerunning the grid. The JSON retains the full-grid rows and batch seeds verbatim; the earlier fast/placement checks are outside the diagnostic scope. Use the same N and ordered seeds to reproduce the numerical tables, up to backend floating-point differences; timings will differ. Batch size is 32768; loop order is kappa, geometry (sphere, cylinder), then dt. No seeds are generated, and all CLI inputs are required.

```sh
JAX_PLATFORMS=cpu python -m diagnostics.permeable_shell_walk --seeds docs/permeable-shell-walk.json --n 163840 --output /tmp/permeable-shell-walk.json --note /tmp/permeable-shell-walk.md
```

The CPU suite retains the low-G sphere and native placement checks. Archived placement (N=131072) had KS p=0.27922/0.21043 and minimum feature-correlation p=0.11071/0.05974 for sphere/cylinder. These checks cannot certify stream independence: init_positions splits the key then reuses the parent for radial U. A proposed fix is separate child keys for direction, radius and axial U; no geometry change is made here."""


def next_seed(study):
    """Consume the next explicitly supplied master seed."""
    return next(study["seed_iterator"])


def shell_spec(kind, kappa):
    return reference_shell.ShellSpec(kind, R, R_OUT, D, D, kappa)


def references(kind, kappa, targets):
    """Tune G with a modest reference grid, then call the error-certified API."""
    spec = shell_spec(kind, kappa)
    model = reference_shell.PermeableShell(spec, n_in=24, n_out=16, n_harmonics=10)
    waveform_dt = 0.0005

    def signal(G):
        g, rf = reference_shell.pgse_waveform(G, DELTA, BIG_DELTA, waveform_dt)
        return model.signal(g, waveform_dt, refocus=rf, start="intra", n_sub=2).total.real

    results = []
    for target in targets:
        G = brentq(lambda amplitude: signal(amplitude) - target, 0.0, 0.6, xtol=1e-5)
        g, rf = reference_shell.pgse_waveform(G, DELTA, BIG_DELTA, waveform_dt)
        ref = reference_shell.reference(spec, g, waveform_dt, refocus=rf, start="intra",
                                        tol=2e-5, cells=(10, 6), sub_steps=2,
                                        harmonics=8, max_levels=6)
        assert abs(ref.total.real - target) < 0.005
        results.append((G, ref))
    return results


def native_kernel(geometry, dt, gradients):
    """Drive the unmodified public native scan body, retaining only four labels."""
    step, _ = make_step_fn(geometry, D, dt, sub_steps=1, track_compartment=True)
    assert step.n_sub == 1
    n_steps = int(round(T / dt))
    waveform = np.zeros((n_steps, len(gradients), 3), np.float32)
    nd, nD = int(round(DELTA / dt)), int(round(BIG_DELTA / dt))
    waveform[:nd, :, 0] = gradients
    waveform[nD:nD + nd, :, 0] = -np.asarray(gradients)
    stops = [int(round(t / dt)) for t in TIMES]

    def one(position, key, compartment):
        carry = (position, jnp.zeros(len(gradients), jnp.float32),
                 jnp.float32(0), key, compartment)
        labels = []
        begin = 0
        for end in stops:
            gs = jnp.asarray(waveform[begin:end])
            carry, _ = jax.lax.scan(step, carry, (gs, jnp.ones(end - begin, jnp.float32),
                                                jnp.arange(begin, end, dtype=jnp.int32)))
            labels.append(carry[4])
            begin = end
        return carry[1], carry[2], jnp.stack(labels), carry[0]

    return jax.jit(jax.vmap(one))


def walk(study, kind, kappa, dt, gradients, n):
    geometry = PermeableShell(R, R_OUT, kappa, kind=kind)
    kernel = native_kernel(geometry, dt, gradients)
    contributions, populations, imaginary = [], [], []
    seeds = []
    started = time.perf_counter()
    for offset in range(0, n, BATCH):
        count = min(BATCH, n - offset)
        seed = next_seed(study)
        seeds.append(seed)
        _, positions, keys, _ = seed_walkers(geometry, count, seed)
        comp = geometry.classify_positions_exact(positions)
        phi, log_w, labels, final_positions = kernel(positions, keys, comp)
        phi, log_w, labels, final_positions = map(np.asarray, (phi, log_w, labels, final_positions))
        assert np.all(log_w == 0)
        radii = np.linalg.norm(final_positions[:, :3 if kind == "sphere" else 2], axis=1)
        assert np.all(radii <= R_OUT * (1 + 1e-6)), "outer wall leaked"
        assert np.all(np.isin(labels, (0, 1)))
        pool = labels[:, -1] == 1
        c, s = np.cos(phi.astype(np.float64)), np.sin(phi.astype(np.float64))
        contributions.append(np.stack((c * pool[:, None], c * ~pool[:, None], c), axis=2))
        imaginary.append(np.stack((s * pool[:, None], s * ~pool[:, None], s), axis=2))
        populations.append(labels == 1)
    values, imag = np.concatenate(contributions), np.concatenate(imaginary)
    pop = np.concatenate(populations)
    run = dict(kind=kind, kappa=kappa, dt=dt, n=n, seeds=seeds,
               seconds=time.perf_counter() - started)
    study["runs"].append(run)
    print("native run " + json.dumps(run), flush=True)
    del kernel, geometry
    gc.collect()
    return (values.mean(0), values.std(0, ddof=1) / np.sqrt(n),
            imag.mean(0), imag.std(0, ddof=1) / np.sqrt(n),
            pop.mean(0), pop.std(0, ddof=1) / np.sqrt(n))


def exchange_reference(kind, kappa):
    models = [reference_shell.PermeableShell(shell_spec(kind, kappa), n_in=ni, n_out=no,
                                             n_harmonics=1) for ni, no in ((160, 96), (320, 192))]
    p = [m.exchange(TIMES)[0, 0] for m in models]
    return p[1], np.abs(p[1] - p[0]) / 3


def record(study, kind, kappa, dt, refs, result, n, scope):
    mean, se, imag, imag_se, pop, pop_se = result
    failures = []
    bias = (MODAL_SLOPE + 1.8 / T) * dt
    for i, (G, ref) in enumerate(refs):
        for j, pool in enumerate(("intra", "extra", "total")):
            wanted, error = getattr(ref, pool).real, getattr(ref, "error_" + pool)
            diff = float(mean[i, j] - wanted)
            tol = Z_FAMILY * float(se[i, j]) + error + bias
            row = dict(scope=scope, kind=kind, kappa=kappa, dt=dt, n=n, G=G, pool=pool,
                       reference=wanted, error=error, walk=float(mean[i, j]), se=float(se[i, j]),
                       difference=diff, z=diff / float(se[i, j]), bias= bias, tolerance=tol,
                       imag=float(imag[i, j]), imag_se=float(imag_se[i, j]))
            study["signals"].append(row)
            if abs(diff) > tol:
                failures.append(row)
            if abs(imag[i, j]) > Z_FAMILY * imag_se[i, j]:
                failures.append(dict(imaginary=row))
    ref_pop, ref_err = exchange_reference(kind, kappa)
    for t, wanted, error, got, stderr in zip(TIMES, ref_pop, ref_err, pop, pop_se):
        b = MODAL_SLOPE * dt + 1.8 * dt / t * (1 - wanted)
        diff = float(got - wanted)
        tol = Z_FAMILY * stderr + error + b
        row = dict(scope=scope, kind=kind, kappa=kappa, dt=dt, time=t, reference=float(wanted),
                   error=float(error), walk=float(got), se=float(stderr), difference=diff,
                   z=diff / float(stderr), bias=float(b), tolerance=float(tol))
        study["exchange"].append(row)
        if abs(diff) > tol:
            failures.append(row)
    return failures


def fit(study, kind, kappa, refs):
    for G, _ in refs:
        for pool in ("intra", "extra", "total"):
            rows = [r for r in study["signals"] if r["scope"] == "full" and r["kind"] == kind
                    and r["kappa"] == kappa and r["G"] == G and r["pool"] == pool]
            x = np.array([r["dt"] for r in rows])
            y = np.array([r["difference"] for r in rows])
            se = np.array([r["se"] for r in rows])
            design = np.column_stack((np.ones(3), x))
            covariance = np.linalg.inv((design / se[:, None]).T @ (design / se[:, None]))
            beta = covariance @ ((design / se[:, None]).T @ (y / se))
            chi2 = np.sum(((y - design @ beta) / se) ** 2)
            study["slopes"].append(dict(kind=kind, kappa=kappa, G=G, pool=pool,
                                         intercept=float(beta[0]), intercept_se=float(np.sqrt(covariance[0, 0])),
                                         slope=float(beta[1]), slope_se=float(np.sqrt(covariance[1, 1])),
                                         chi2_one_dof=float(chi2)))


def write_note(study, path):
    lines = ["# Native permeable shell walk against the confirmed reference", "",
             REGISTERED_TOLERANCE,
             f"N={study['n']:,} per full resolution; dt=50/25/12.5 us. z_F={Z_FAMILY:.5f}; "
             f"B_signal(dt)={MODAL_SLOPE + 1.8 / T:.6f} dt (dt in seconds).",
             "Signals are real final-pool contributions, normalized by all starting walkers. "
             "Raw z=(walk-reference)/SE; the pass budget additionally includes reference error and registered step bias.",
             "All three G share paths at each resolution; different resolutions and batches have fresh master seeds. "
             "Full seeds, imaginary diagnostics and fit intercepts are in the companion JSON.", "",
             "## Signal comparison (all resolutions)", "",
             "|scope / case|kappa m/s|G T/m|dt us|pool|reference ± error|walk ± SE|difference|z|budget|pass|",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|"]
    for r in study["signals"]:
        lines.append(f'|{r["scope"]} {r["kind"]}|{r["kappa"]:.0e}|{r["G"]:.6f}|{r["dt"]*1e6:g}|{r["pool"]}|'
                     f'{r["reference"]:.6f} ± {r["error"]:.1e}|{r["walk"]:.6f} ± {r["se"]:.6f}|'
                     f'{r["difference"]:+.6f}|{r["z"]:+.2f}|{r["tolerance"]:.6f}|{abs(r["difference"]) <= r["tolerance"]}|')
    lines += ["", "## Fitted step-bias slopes", "", "Weighted residual = intercept + slope dt; one fit degree of freedom.", "",
              "|case|kappa|G T/m|pool|slope ± SE (1/s)|intercept ± SE|chi² (1 dof)|",
              "|---|---:|---:|---|---:|---:|---:|"]
    for r in study["slopes"]:
        lines.append(f'|{r["kind"]}|{r["kappa"]:.0e}|{r["G"]:.6f}|{r["pool"]}|{r["slope"]:+.2f} ± '
                     f'{r["slope_se"]:.2f}|{r["intercept"]:+.6f} ± {r["intercept_se"]:.6f}|{r["chi2_one_dof"]:.2f}|')
    lines += ["", "## Exchange check (start intra, no force from gradient)", "",
              "|scope / case|kappa|dt us|time ms|reference ± error|walk ± SE|difference|z|budget|pass|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in study["exchange"]:
        lines.append(f'|{r["scope"]} {r["kind"]}|{r["kappa"]:.0e}|{r["dt"]*1e6:g}|{r["time"]*1e3:g}|'
                     f'{r["reference"]:.6f} ± {r["error"]:.1e}|{r["walk"]:.6f} ± {r["se"]:.6f}|'
                     f'{r["difference"]:+.6f}|{r["z"]:+.2f}|{r["tolerance"]:.6f}|{abs(r["difference"]) <= r["tolerance"]}|')
    lines += ["", "## Early-time transient", "", TRANSIENT, "", "## Reproduction", "", REPRODUCTION,
              "", f'Measured native runs: {sum(r["seconds"] for r in study["runs"]):.1f} s. '
              "Archived study including fast and placement checks: 1740.1 s."]
    path.write_text("\n".join(lines) + "\n")


def run(*, seeds, n):
    """Measure the full grid using exactly the stated N and batch seeds."""
    if n < 2:
        raise ValueError("n must be at least two for empirical SE")
    expected = 18 * ((n + BATCH - 1) // BATCH)
    if len(seeds) != expected:
        raise ValueError(f"need exactly {expected} batch seeds for n={n}")
    if any(type(s) is not int or not 0 < s < 2**32 for s in seeds):
        raise ValueError("seeds must be nonzero uint32 integers")
    if len({s >> 16 for s in seeds}) != len(seeds) or any(s >> 16 == 0 for s in seeds):
        raise ValueError("seeds need distinct nonzero upper 16-bit families")
    study = dict(n=n, seeds=seeds, seed_iterator=iter(seeds), signals=[], exchange=[], slopes=[], runs=[])
    failures = []
    for kappa in (1e-5, 1e-4, 1e-3):
        for kind in ("sphere", "cylinder"):
            refs = references(kind, kappa, TARGETS)
            for dt in DTS:
                result = walk(study, kind, kappa, dt, [r[0] for r in refs], n)
                if np.max(result[1]) > 0.002:
                    failures.append(dict(kind=kind, kappa=kappa, dt=dt, reason="signal SE > 0.002"))
                failures += record(study, kind, kappa, dt, refs, result, n, "full")
            fit(study, kind, kappa, refs)
    del study["seed_iterator"]
    study["failures"] = failures
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--note", type=Path, required=True)
    args = parser.parse_args()
    study = run(seeds=json.loads(args.seeds.read_text())["seeds"], n=args.n)
    args.output.write_text(json.dumps(study, indent=2) + "\n")
    write_note(study, args.note)
    if study["failures"]:
        raise SystemExit(json.dumps(study["failures"], indent=2))
