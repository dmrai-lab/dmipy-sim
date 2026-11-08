"""The MT walk on a permeable wall: cross first, then bind, against the exact 1-D answer (dmrai-lab/dmipy-sim#909).

At an encounter that touches a permeable wall the walk decides the crossing first (the transmission rule), then the
binding, from the encounter's local time, whether it crossed or was reflected: the binding clock ``1 - exp(-kappa_MT l
/ D)``. A reflected encounter binds on the walker's own face; a crossing one on the near or the far face, half each
(:data:`~dmipy_sim.geometry._boundary.NEAR_FACE_SHARE`, #926). A bound walker freezes at the step's end for an
exponential dwell that carries the encounter's geminate rebinds (mean ``tau exp(kappa_MT l / (2 D))``, #921), seen on
its face's side, and resumes from the step's end.

The substrate is a slab of two layers of ``R`` between reflecting walls with one membrane, a bound pool on each face of
it (:class:`~dmipy_sim.geometry.PermeableSlab1D`), walkers uniform over the slab (or, for each face's pool, free in one
layer), free water at no relaxation and no off-resonance, the bound pool at ``R2_b`` and ``omega_b``. The reference is
the 1-D Bloch-Torrey equation on the same layers with the same pools (:class:`dmipy_analytic.bloch_torrey.BlochTorrey1D`
with ``binding_reactivity``), exact up to its cells. The step is ``h = 0.09 R`` (one step per block, the step law's
regime of #915); the membrane is at ``kappa h / D`` of 0.5 and 1.0, the binding at ``kappa_MT h / D = 0.2`` with a dwell
of 33 steps.

A walker's transverse magnetisation is ``exp(-(R2_b + i omega_b) t_b)`` with ``t_b`` its time bound, one step for each
block end it is found bound at. The dwell is a continuous exponential read at the block ends without bias
(``test_the_dwell_is_unbiased_at_the_block_ends``). What separates the walk from the reference is the step's own
error, second order in it; the step law states it (:func:`~dmipy_sim.engine.physics.binding_error`: the pool's
binding reactivity and dwell each within ``MT_COEFFICIENT (kappa_MT h / D)^2``), and each comparison allows it through
the reference's own sensitivities to those two, on top of the sampling tolerance and the reference's cells.
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
SENS_STEP = 0.02                              # the relative step of the reference's sensitivities


def _binding_eps():
    """The step law's bound on the walk's binding at the test's step (:func:`~dmipy_sim.engine.physics.binding_error`)."""
    from dmipy_sim.engine.physics import binding_error
    return binding_error(D, K_ON, DT, R)                      # the pore: a layer of width R


def _reference(kappa, step, breakpoints=(), *, relax=True, pool="total", n_cells=800, k_on=K_ON, tau=TAU, face=None):
    """The slab's signal after ``step`` steps; ``face`` (0 the face of layer A, 1 of layer B): that face's pool, from
    free walkers in layer A alone and empty bound pools."""
    perm = (kappa,) if not breakpoints else tuple((k,) for k in kappa)
    kw = dict(permeability=perm, permeability_breakpoints=breakpoints, density=(1.0, 1.0), binding_reactivity=k_on,
              dwell_time=tau)
    if face is not None:
        kw.update(density=(1.0, 0.0), bound_start="empty")
    if relax:
        kw.update(bound_R2=R2B, bound_omega=OMEGA_B)
    m = bt.BlochTorrey1D(np.array([0.0, 1.0, 2.0]) * R, D, n_cells=n_cells, **kw)
    if face is None:
        return complex(m.signal(np.zeros(step), DT, pool=pool))
    return complex(np.asarray(m.signal(np.zeros(step), DT, pool=pool, by_layer=True))[face])


def _with_bound(value_of):
    """``(value, error)``: the reference at 800 cells, twice its Richardson error over 400 / 800 cells, plus the step
    law's binding bound, ``eps (|dX/d ln k_on| + |dX/d ln tau_b|)`` on the reference's own sensitivities."""
    v = [value_of(n_cells=n) for n in (400, 800)]
    sens = sum(abs(value_of(**{name: val * (1.0 + SENS_STEP)}) - v[1]) / SENS_STEP
               for name, val in (("k_on", K_ON), ("tau", TAU)))
    return v[1], 2.0 * abs(v[0] - v[1]) * 4.0 / 3.0 + _binding_eps() * sens


def _signals(kappa, step, breakpoints=()):
    """``((signal, error), (bound fraction, error))`` of the slab after ``step`` steps (:func:`_with_bound`); ``kappa``
    one value, or with ``breakpoints`` (s) one per interval of time. The signal's sensitivities are the complex
    signal's moduli, which bound its real part's."""
    s, e_s = _with_bound(lambda **kw: _reference(kappa, step, breakpoints, **kw))
    b, e_b = _with_bound(lambda **kw: _reference(kappa, step, breakpoints, relax=False, pool="bound", **kw).real)
    return (s.real, e_s), (b, e_b)


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
            for name, x, ref, e in (("signal", sig, ref_s, e_s), ("bound fraction", frac, ref_b, e_b)):
                tol = se_tol(N, x.std(), comparisons=comparisons) + e
                rows.append((ks, step, name, x.mean(), ref, (x.mean() - ref) / (x.std() / np.sqrt(N)), tol))
                assert abs(x.mean() - ref) < tol, (
                    f"kappa h/D = {ks}, step {step}: {name} {x.mean():.5f} against {ref:.5f} (tolerance {tol:.5f})")
    print("\nkappa* step quantity walk reference z tol\n" + "\n".join(
        f"{a} {b} {c} {d:.5f} {e:.5f} {z:+.1f} {t:.5f}" for a, b, c, d, e, z, t in rows))


N_FACE = 100_000                              # each face's tolerance at 0.1 R^2/D is about 0.0027: two thirds of the
                                              # far-face rule's 0.004


def test_each_faces_bound_pool_is_the_continuum_slabs_from_a_one_sided_start():
    """Free walkers in layer A alone and empty bound pools (no burn-in): the face of layer B fills only through the
    membrane, so each face's pool depends on the face a walker that crosses binds on (:data:`NEAR_FACE_SHARE`). Each
    face's bound fraction and its bound pool's signal (a walker bound on a face is seen on that face's side) at the two
    times against ``BlochTorrey1D``'s pool on that face, with the step law's binding bound on each. Every bind at a
    crossing on the far face, the walk's rule before #926, is first order: at ``t = 0.1 R^2 / D`` it puts about 0.004
    too much in the far face's pool and as much too little in the near face's, at both permeabilities -- about seven
    standard errors here, against a tolerance of four."""
    from dmipy_sim.engine.core import _walk_geometry
    from dmipy_sim.geometry import PermeableSlab1D
    comparisons = len(KAPPA_STAR) * len(STEPS) * 2 * 2
    rows = []
    for i, ks in enumerate(KAPPA_STAR):
        kappa = ks * D / (H * R)
        r0 = np.zeros((N_FACE, 3), np.float32)
        r0[:, 0] = np.random.default_rng(41 + i).uniform(0.0, R, N_FACE)
        w = _walk_geometry(N_FACE, D, PermeableSlab1D(length=2 * R, permeability=kappa), max(STEPS) * DT, DT,
                           sub_steps=1, seed=41 + i, raw_subset=N_FACE, require_gpu=False, r0=r0, kappa_MT=K_ON,
                           dwell_time=TAU, equilibrate_binding="off")
        assert w.work["exhausted_steps"] == 0
        bound = np.asarray(w.raw_bound, np.float64)
        x = np.asarray(w.raw_positions, np.float64)[:, :, 0]
        for step in STEPS:
            t_b = bound[:, 1:step + 1].sum(1) * DT
            sig = np.exp(-R2B * t_b) * np.cos(OMEGA_B * t_b)
            for face in (0, 1):
                on = bound[:, step] * ((x[:, step] > R) if face else (x[:, step] < R))
                ref_b, e_b = _with_bound(lambda **kw: _reference(kappa, step, relax=False, pool="bound", face=face,
                                                                 **kw).real)
                ref_s, e_s = _with_bound(lambda **kw: _reference(kappa, step, pool="bound", face=face, **kw))
                for name, v, ref, e in (("bound fraction", on, ref_b, e_b),
                                        ("bound signal", on * sig, ref_s.real, e_s)):
                    tol = se_tol(N_FACE, v.std(), comparisons=comparisons) + e
                    z = (v.mean() - ref) / (v.std() / np.sqrt(N_FACE))
                    rows.append((ks, step, "AB"[face], name, v.mean(), ref, z, tol))
                    assert abs(v.mean() - ref) < tol, (f"kappa h/D = {ks}, step {step}, face {'AB'[face]}: {name} "
                                                       f"{v.mean():.5f} against {ref:.5f} (tolerance {tol:.5f})")
    print("\nkappa* step face quantity walk reference z tol\n" + "\n".join(
        f"{a} {b} {c} {d} {e:.5f} {f:.5f} {z:+.1f} {t:.5f}" for a, b, c, d, e, f, z, t in rows))


# ── the order at one contact ────────────────────────────────────────────────────────────────────────────────────
def _one_step_from_the_membrane(n, kappa, kmt, dwell, seed, steps=1):
    """``n`` walkers ``0.3 h`` below a permeable membrane, bound pool empty (no burn-in), walked ``steps`` steps:
    ``(start, seen positions (n, steps + 1), bound (n, steps + 1))``."""
    from dmipy_sim.engine.core import _walk_geometry
    from dmipy_sim.geometry import PermeableSlab1D
    g = PermeableSlab1D(length=2 * R, permeability=kappa)
    r0 = np.zeros((n, 3), np.float32)
    r0[:, 0] = R - 0.3 * H * R
    kw = dict(kappa_MT=kmt, dwell_time=dwell, equilibrate_binding="off") if kmt else {}
    w = _walk_geometry(n, D, g, steps * DT, DT, sub_steps=1, seed=seed, raw_subset=n, require_gpu=False, r0=r0, **kw)
    x = np.asarray(w.raw_positions, np.float64)[:, :, 0]
    b = np.asarray(w.raw_bound) > 0 if kmt else np.zeros(x.shape, bool)
    return float(r0[0, 0]), x, b


def test_a_crossing_is_decided_before_the_binding_and_binding_never_vetoes_it():
    """Walkers one step from a strongly binding, strongly permeable membrane. The fraction that cross is the same with
    and without binding (cross first: binding is decided after, and never undoes the crossing); bind-first would take
    from it the crossings that bind, ``P(cross and bound)``, which the same walk counts and which is many standard
    errors. A walker that crossed and bound is seen on the face it bound on (:data:`NEAR_FACE_SHARE`): past the
    membrane on the far face, at its step's start on the near face."""
    from dmipy_sim.geometry._boundary import NEAR_FACE_SHARE
    n, kappa, kon, seed = 40_000, 1.5 * D / (H * R), 3.0 * D / (H * R), 5
    x0, x, b = _one_step_from_the_membrane(n, kappa, kon, 10 * DT, seed)
    _, x_0, _ = _one_step_from_the_membrane(n, kappa, 0.0, 0.0, seed + 1)
    far, near = b[:, 1] & (x[:, 1] > R), b[:, 1] & (x[:, 1] == x0)
    c_mt, c_0 = (x[:, 1] > R) | near, x_0[:, 1] > R
    p_mt, p_0 = c_mt.mean(), c_0.mean()
    se = np.sqrt(p_mt * (1 - p_mt) / n + p_0 * (1 - p_0) / n)
    vetoed = (c_mt & b[:, 1]).mean()                          # what bind-first would remove from the crossings
    assert vetoed > 10 * se, f"the control does not resolve the order: crossings that bind {vetoed:.4f}, se {se:.4f}"
    assert abs(p_mt - p_0) < se_tol(n, 1.0, runs=2) * np.sqrt(p_0 * (1 - p_0)), (
        f"crossing {p_mt:.4f} with binding against {p_0:.4f} without (se {se:.4f}); bind-first would be {p_0 - vetoed:.4f}")
    # the crossings that bind, shared between the faces
    k, m = int(near.sum()), int((near | far).sum())
    assert m > 1000 and abs(k / m - NEAR_FACE_SHARE) < se_tol(m, 0.5), f"{k} of {m} crossing binds on the near face"
    assert (b[:, 1] & (x[:, 1] < R) & ~near).any()            # and a reflected encounter binds on its own face


def test_a_bound_walker_is_seen_on_its_face_and_resumes_from_its_steps_end():
    """A walker that binds rests through its dwell: one bound on the far face of the encounter it crossed rests where
    its step ended, past the face (not on it); one bound on the near face is seen at the step's start, on the near
    side. Released, either resumes from where its step ended: the crossing stands, so a near-face walker released
    early in a step mostly ends it past the membrane, which from its step's start few walkers reach."""
    n = 20_000
    x0, x, b = _one_step_from_the_membrane(n, 2.0 * D / (H * R), 3.0 * D / (H * R), 1e6 * DT, 7, steps=3)
    frozen = b[:, 1] & b[:, 2] & b[:, 3]
    assert frozen.sum() > 1000
    assert np.all(x[frozen, 2] == x[frozen, 1]) and np.all(x[frozen, 3] == x[frozen, 1])   # it does not move
    far, near = frozen & (x[:, 1] > R), frozen & (x[:, 1] == x0)
    assert far.sum() > 100 and np.all(np.abs(x[far, 1] - R) > 0.0)                         # past the face, not on it
    assert near.sum() > 100
    # released within the next step: dwells of a step on average
    x0, x, b = _one_step_from_the_membrane(200_000, 2.0 * D / (H * R), 3.0 * D / (H * R), DT, 8, steps=2)
    near_out = b[:, 1] & (x[:, 1] == x0) & ~b[:, 2]
    free_past = (~b[:, 1]) & (x[:, 1] > R)
    assert near_out.sum() > 200
    past = (x[near_out, 2] > R).mean()
    assert past > 0.5 and past > 2.0 * free_past.mean(), (
        f"{past:.3f} of the released near-face walkers past the membrane; {free_past.mean():.3f} of all cross a step")


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
        for name, x, ref, e in (("signal", sig, ref_s, e_s), ("bound fraction", bound[:, step], ref_b, e_b)):
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
        ref, e_ref = _with_bound(lambda **kw: _reference(kappa, step, **kw))     # complex; e bounds Re and Im alike
        (_, _), (ref_b, e_b) = _signals(kappa, step)
        for name, got, want, e in (("Re", sig.real, ref.real, e_ref), ("Im", sig.imag, ref.imag, e_ref),
                                   ("bound fraction", float(np.asarray(bf)[-1]), ref_b, e_b)):
            tol = se_tol(n, 0.3 if name == "bound fraction" else sd, comparisons=comparisons) + e
            assert abs(got - want) < tol, f"step {step}: {name} {got:.5f} against {want:.5f} (tolerance {tol:.5f})"
