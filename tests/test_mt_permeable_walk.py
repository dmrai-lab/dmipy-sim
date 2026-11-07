"""The MT walk on a permeable wall: cross first, then bind, against the exact 1-D answer (dmrai-lab/dmipy-sim#909).

At an encounter that touches a permeable wall the walk decides the crossing first (the transmission rule), then the
binding on the face the walker ends on, from the encounter's local time, whether it crossed or was reflected; a bound
walker freezes at the step's end, dwells for an exponential time and is released to the same face.

The substrate is a slab of two layers of ``R`` between reflecting walls with one membrane, a bound pool on each face of
it (:class:`~dmipy_sim.geometry.PermeableSlab1D`), walkers uniform over the slab, free water at no relaxation and no
off-resonance, the bound pool at ``R2_b`` and ``omega_b``. The reference is the 1-D Bloch-Torrey equation on the same
layers with the same pools (:class:`dmipy_analytic.bloch_torrey.BlochTorrey1D` with ``binding_reactivity``), exact up
to its cells. The step is ``h = 0.09 R`` (one step per block, the step law's regime of #915); the membrane is at
``kappa h / D`` of 0.5 and 1.0, the binding at ``kappa_MT h / D = 0.2`` with a dwell of 33 steps.

A walker's transverse magnetisation is ``exp(-(R2_b + i omega_b) t_b)`` with ``t_b`` its time bound, one step for each
block end it is found bound at. The dwell is a continuous exponential read at the block ends without bias
(``test_the_dwell_is_unbiased_at_the_block_ends``). What separates the walk from the reference is the step's own
first-order error: a walker is released where its step ended, not at the face, so it rebinds in fewer bursts than the
continuum's. The late signal sits below the reference by 0.008 (0.006 to 0.012 over dwells of 17 to 67 steps,
60 000 walkers each, 3 to 6 se) at ``kappa h / D = 0.5`` and ``h = 0.09 R``, by half of it at ``h = 0.045 R`` and
within noise at ``h = 0.0225 R``, whatever the dwell, and by more where the wall is nearly closed (0.017 at
``kappa h / D = 0.3``). The table allows ``STEP_ERROR`` for it on the signal, on top of the sampling tolerance and the
reference's cells.
"""
import numpy as np
import pytest

from tests.conftest import se_tol

bt = pytest.importorskip("dmipy_analytic.bloch_torrey")

D = 2e-9
R = 5e-6
H = 0.09                                      # the step, in R
DT = (H * R) ** 2 / (6.0 * D)
KAPPA_STAR = (0.5, 1.0)                  # kappa h / D
KMT_STAR = 0.2                                # kappa_MT h / D
K_ON = KMT_STAR * D / (H * R)                 # m/s
POOL = 0.5e-6                                 # k_on tau, m: each face's pool is 0.1 R of free water
TAU = POOL / K_ON                             # the dwell, s (33 steps)
R2B, OMEGA_B = 1500.0, 2500.0
TIMES = (0.1, 0.3)                            # in R^2 / D
STEPS = [int(round(t * R * R / D / DT)) for t in TIMES]
N = 10_000
STEP_ERROR = 0.012                            # the signal's first-order error in the step at h = 0.09 R


def _signals(kappa, step, breakpoints=()):
    """``((signal, error), (bound fraction, error))`` of the slab after ``step`` steps; ``kappa`` one value, or with
    ``breakpoints`` (s) one per interval of time."""
    perm = (kappa,) if not breakpoints else tuple((k,) for k in kappa)
    kw = dict(permeability=perm, permeability_breakpoints=breakpoints, density=(1.0, 1.0), binding_reactivity=K_ON,
              dwell_time=TAU)
    ms = [bt.BlochTorrey1D(np.array([0.0, 1.0, 2.0]) * R, D, n_cells=n, bound_R2=R2B, bound_omega=OMEGA_B, **kw)
          for n in (400, 800)]
    mb = [bt.BlochTorrey1D(np.array([0.0, 1.0, 2.0]) * R, D, n_cells=n, **kw) for n in (400, 800)]
    G = np.zeros(step)
    s = [complex(m.signal(G, DT)).real for m in ms]
    b = [complex(m.signal(G, DT, pool="bound")).real for m in mb]
    return (s[1], 2.0 * abs(s[0] - s[1]) * 4.0 / 3.0), (b[1], 2.0 * abs(b[0] - b[1]) * 4.0 / 3.0)


def _walk(kappa, seed):
    from dmipy_sim.engine.core import _walk_geometry
    from dmipy_sim.geometry import PermeableSlab1D
    r0 = np.zeros((N, 3), np.float32)
    r0[:, 0] = np.random.default_rng(seed).uniform(0.0, 2.0 * R, N)
    g = PermeableSlab1D(length=2 * R, permeability=kappa)    # kappa: a number or a PermeabilitySchedule
    return _walk_geometry(N, D, g, max(STEPS) * DT, DT, sub_steps=1, seed=seed, raw_subset=N, require_gpu=False,
                          r0=r0, kappa_MT=K_ON, dwell_time=TAU)


def test_the_permeable_mt_walk_is_the_continuum_slab_with_bound_pools():
    rows = []
    walks = {k: _walk(ks * D / (H * R), seed) for k, ks, seed in
             ((i, ks, 11 + i) for i, ks in enumerate(KAPPA_STAR))}
    comparisons = len(KAPPA_STAR) * len(STEPS) * 2
    for i, ks in enumerate(KAPPA_STAR):
        kappa = ks * D / (H * R)
        w = walks[i]
        assert w.work["exhausted_steps"] == 0
        assert w.work["n_crossings"] is None or w.work["n_crossings"] > 0
        bound = np.asarray(w.raw_bound, np.float64)
        for step in STEPS:
            t_b = bound[:, 1:step + 1].sum(1) * DT
            sig = np.exp(-R2B * t_b) * np.cos(OMEGA_B * t_b)
            frac = bound[:, step]
            (ref_s, e_s), (ref_b, e_b) = _signals(kappa, step)
            for name, x, ref, e in (("signal", sig, ref_s, e_s + STEP_ERROR), ("bound fraction", frac, ref_b, e_b)):
                tol = se_tol(N, x.std(), comparisons=comparisons) + e
                rows.append((ks, step, name, x.mean(), ref, (x.mean() - ref) / (x.std() / np.sqrt(N)), tol))
                assert abs(x.mean() - ref) < tol, (
                    f"kappa h/D = {ks}, step {step}: {name} {x.mean():.5f} against {ref:.5f} (tolerance {tol:.5f})")
    print("\nkappa* step quantity walk reference z tol\n" + "\n".join(
        f"{a} {b} {c} {d:.5f} {e:.5f} {z:+.1f} {t:.5f}" for a, b, c, d, e, z, t in rows))


# ── the order at one contact ────────────────────────────────────────────────────────────────────────────────────
def test_a_crossing_is_decided_before_the_binding_and_binding_never_vetoes_it():
    """Walkers one step from a strongly binding, strongly permeable membrane, bound pool empty (no burn-in). The
    fraction that cross is the same with and without binding (cross first: binding is decided after, on the face
    reached); bind-first would take from it the crossings that bind, ``P(cross and bound)``, which the same walk
    counts and which is many standard errors."""
    from dmipy_sim.engine.core import _walk_geometry
    from dmipy_sim.geometry import PermeableSlab1D
    n, kappa, kon, seed = 40_000, 1.5 * D / (H * R), 3.0 * D / (H * R), 5
    g = PermeableSlab1D(length=2 * R, permeability=kappa)
    r0 = np.zeros((n, 3), np.float32)
    r0[:, 0] = R - 0.3 * H * R

    def one(kmt, s):
        kw = dict(kappa_MT=kmt, dwell_time=10 * DT, equilibrate_binding="off") if kmt else {}
        w = _walk_geometry(n, D, g, DT, DT, sub_steps=1, seed=s, raw_subset=n, require_gpu=False, r0=r0, **kw)
        crossed = np.asarray(w.raw_positions)[:, 1, 0] > R
        bound = np.asarray(w.raw_bound)[:, 1] > 0 if kmt else np.zeros(n, bool)
        return crossed, bound

    c_mt, b_mt = one(kon, seed)
    c_0, _ = one(0.0, seed + 1)
    p_mt, p_0 = c_mt.mean(), c_0.mean()
    se = np.sqrt(p_mt * (1 - p_mt) / n + p_0 * (1 - p_0) / n)
    vetoed = (c_mt & b_mt).mean()                              # what bind-first would remove from the crossings
    assert vetoed > 10 * se, f"the control does not resolve the order: crossings that bind {vetoed:.4f}, se {se:.4f}"
    assert abs(p_mt - p_0) < se_tol(n, 1.0, runs=2) * np.sqrt(p_0 * (1 - p_0)), (
        f"crossing {p_mt:.4f} with binding against {p_0:.4f} without (se {se:.4f}); bind-first would be {p_0 - vetoed:.4f}")
    assert b_mt[c_mt].any() and b_mt[~c_mt].any()             # a walker binds on either side of a contact


def test_a_bound_walker_freezes_at_the_steps_end_not_at_the_face():
    """The walker that binds at the encounter that crossed rests where its step ended, on the face's far side."""
    from dmipy_sim.engine.core import _walk_geometry
    from dmipy_sim.geometry import PermeableSlab1D
    n = 20_000
    g = PermeableSlab1D(length=2 * R, permeability=2.0 * D / (H * R))
    r0 = np.zeros((n, 3), np.float32)
    r0[:, 0] = R - 0.3 * H * R
    w = _walk_geometry(n, D, g, 3 * DT, DT, sub_steps=1, seed=7, raw_subset=n, require_gpu=False, r0=r0,
                       kappa_MT=3.0 * D / (H * R), dwell_time=1e6 * DT, equilibrate_binding="off")
    x = np.asarray(w.raw_positions, np.float64)[:, :, 0]
    b = np.asarray(w.raw_bound)
    frozen = (b[:, 1] > 0) & (b[:, 2] > 0) & (b[:, 3] > 0)
    assert frozen.sum() > 1000
    assert np.all(x[frozen, 2] == x[frozen, 1]) and np.all(x[frozen, 3] == x[frozen, 1])   # it does not move
    crossed = frozen & (x[:, 1] > R)
    assert crossed.sum() > 100 and np.all(np.abs(x[crossed, 1] - R) > 0.0)                # past the face, not on it


def test_the_dwell_is_unbiased_at_the_block_ends():
    """Counted at the block ends, a dwell binds for its mean: the number of ends a walker is found bound at, after the
    step it binds in, averages the dwell in steps. The encounter falls uniformly in its step and the dwell left at the
    step's end is the draw less the time from the encounter to the end; a draw carried whole counted half a step too
    many (#918)."""
    from dmipy_sim.geometry._boundary import dwell_left
    rng = np.random.default_rng(0)
    n, mean = 400_000, 17.0
    draw = rng.exponential(mean, n)
    when = rng.uniform(size=n)
    left = np.asarray(dwell_left(draw, 1.0, when), np.float64)
    ends = np.ceil(left)                                      # ends found bound: one for each step it still has time in
    z = (ends.mean() - mean) / (ends.std() / np.sqrt(n))
    assert abs(z) < 4.0, f"{ends.mean():.3f} block ends bound per dwell of {mean} steps (z = {z:+.1f})"
    assert np.ceil(draw).mean() - mean > 0.4                  # the whole draw counted, the old rule


def test_a_permeability_over_time_is_the_continuum_slab_with_bound_pools():
    """The membrane at ``kappa h / D = 0.3`` for the first 111 steps (0.15 R^2/D) and 1.5 after, a breakpoint on a
    step boundary, against ``BlochTorrey1D`` with the same breakpoint. The bound pool is equilibrated at the first
    interval's permeability before t = 0 (it is independent of it)."""
    from dmipy_sim.geometry.base import PermeabilitySchedule
    k = (0.3 * D / (H * R), 1.5 * D / (H * R))
    cut = 111
    sched = PermeabilitySchedule([cut * DT], [k[0], k[1]])
    w = _walk(sched, 21)
    assert w.work["exhausted_steps"] == 0
    bound = np.asarray(w.raw_bound, np.float64)
    comparisons = 2 * len(STEPS)
    for step in STEPS:
        t_b = bound[:, 1:step + 1].sum(1) * DT
        sig = np.exp(-R2B * t_b) * np.cos(OMEGA_B * t_b)
        (ref_s, e_s), (ref_b, e_b) = _signals(k, step, breakpoints=(cut * DT,))
        for name, x, ref, e in (("signal", sig, ref_s, e_s + STEP_ERROR), ("bound fraction", bound[:, step], ref_b, e_b)):
            tol = se_tol(N, x.std(), comparisons=comparisons) + e
            assert abs(x.mean() - ref) < tol, f"step {step}: {name} {x.mean():.5f} against {ref:.5f} (tolerance {tol:.5f})"
    # the schedule is the walk's, not the first interval's: the membrane opens, so the layers' mixing differs
    (ref_first, _), _ = _signals(k[0], STEPS[1])
    (ref_sched, _), _ = _signals(k, STEPS[1], breakpoints=(cut * DT,))
    assert ref_first != ref_sched


def test_the_forward_vector_bloch_oracle_is_the_continuum_slab_with_bound_pools():
    """The forward walk of an excited ensemble (a 90 degree pulse, no gradient) on the permeable slab, its bound pool
    at ``R2_b`` and ``omega_b``, against ``BlochTorrey1D`` (total magnetisation, conjugated: the oracle's rotation sense):
    the complex signal and the bound fraction at the two times. A walker's magnetisation lies in the unit disc; its
    standard deviation is taken as the 0.5 measured on the walk of the first test."""
    from dataclasses import replace
    from dmipy_replay import ScannerSequence, RFEvent
    from dmipy_sim.engine.bloch import _simulate_bloch
    from dmipy_sim.geometry import PermeableSlab1D
    kappa, sd, n = 0.5 * D / (H * R), 0.5, 12_000
    g = PermeableSlab1D(length=2 * R, permeability=kappa)
    exc = [RFEvent(0.0, 90.0, axis_deg=90.0, duration_s=0.0, offset_hz=0.0)]
    comparisons = 3 * len(STEPS)
    for step in STEPS:
        wf = replace(ScannerSequence(G=np.zeros((1, step, 3)), dt=DT), rf=exc)
        r0 = np.zeros((n, 3), np.float32)
        r0[:, 0] = np.random.default_rng(31).uniform(0.0, 2.0 * R, n)
        sig, bf = _simulate_bloch(n, D, wf, g, seed=31, r0=r0, kappa_MT=K_ON, dwell_time=TAU, T2_bound=1.0 / R2B,
                                  T1_bound=1e3, off_resonance_bound=OMEGA_B / (2.0 * np.pi), sub_steps=1,
                                  return_bound_frac=True)
        sig = np.conj(complex(np.asarray(sig).ravel()[-1]))
        ms = [bt.BlochTorrey1D(np.array([0.0, 1.0, 2.0]) * R, D, permeability=(kappa,), density=(1.0, 1.0),
                               binding_reactivity=K_ON, dwell_time=TAU, bound_R2=R2B, bound_omega=OMEGA_B, n_cells=c)
              for c in (400, 800)]
        ref = [complex(m.signal(np.zeros(step), DT)) for m in ms]
        (_, _), (ref_b, e_b) = _signals(kappa, step)
        for name, got, want, e in (("Re", sig.real, ref[1].real, abs(ref[0].real - ref[1].real) * 8.0 / 3.0 + STEP_ERROR),
                                   ("Im", sig.imag, ref[1].imag, abs(ref[0].imag - ref[1].imag) * 8.0 / 3.0 + STEP_ERROR),
                                   ("bound fraction", float(np.asarray(bf)[-1]), ref_b, e_b)):
            tol = se_tol(n, 0.3 if name == "bound fraction" else sd, comparisons=comparisons) + e
            assert abs(got - want) < tol, f"step {step}: {name} {got:.5f} against {want:.5f} (tolerance {tol:.5f})"
