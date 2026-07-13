"""PGSTE stimulated echo: T1 relaxation gated by the transverse-coherence schedule.

A PGSTE stores the magnetisation along the longitudinal axis during the mixing
time TM.  While stored (chi_perp == 0) there is no transverse (T2) loss and no
surface-relaxivity loss — only T1 acts.  The stimulated echo stores half the
magnetisation, an idealized 0.5 amplitude factor.  For free diffusion the direct
simulation therefore yields

    S ≈ 0.5 · exp(-b·D) · exp(-TM/T1).

These tests exercise the direct simulate() path on FreeDiffusion (CPU-fast).
"""

import numpy as np
import numpy.testing as npt

from dmipy_sim import simulate, FreeDiffusion, set_b, calc_b
from dmipy_sim.waveforms import pgse, pgste
from .conftest import D, SEED

# Local walker count: FreeDiffusion is cheap per walker, so this test stays fast
# (it is not marked slow) while keeping the MC noise floor ~1/sqrt(N) small.
N = 40_000


def _pgste_at_b(delta, TM, b, n_t=600):
    """Single-direction PGSTE scaled to a target b-value (s/m²)."""
    bvecs = np.array([[1.0, 0.0, 0.0]])
    return set_b(pgste(delta=delta, TM=TM, G_magnitude=1.0, bvecs=bvecs, n_t=n_t),
                 np.array([b]))


def test_pgste_signal_equals_half_diffusion_times_t1():
    """PGSTE signal ≈ 0.5 · exp(-b·D) · exp(-TM/T1) for free diffusion."""
    TM = 40e-3
    T1 = 1.0
    b = 1.0e9
    wf = _pgste_at_b(delta=5e-3, TM=TM, b=b)
    b_eff = calc_b(wf)[0]

    S = simulate(N, D, wf, FreeDiffusion(), seed=SEED, T1=T1, require_gpu=False)
    expected = 0.5 * np.exp(-b_eff * D) * np.exp(-TM / T1)

    atol_mc = 1.0 / np.sqrt(N)
    npt.assert_allclose(
        S, expected, atol=3 * atol_mc,
        err_msg=f"PGSTE signal must equal 0.5·exp(-b·D)·exp(-TM/T1)={expected:.4f}")


def test_pgste_t1_decay_monotonic_in_mixing_time():
    """Longer mixing time → more T1 decay → lower signal (b held fixed)."""
    T1 = 800e-3
    b = 1.0e9
    signals = []
    for TM in (10e-3, 40e-3, 80e-3):
        wf = _pgste_at_b(delta=5e-3, TM=TM, b=b)
        signals.append(float(simulate(N, D, wf, FreeDiffusion(), seed=SEED,
                                      T1=T1, require_gpu=False)[0]))

    for lo, hi in zip(signals[1:], signals[:-1]):
        assert lo <= hi + 1e-3, (
            f"PGSTE signal must decrease with mixing time; got {signals}")


def test_pgste_without_t1_is_half_diffusion_attenuation():
    """With T1=None a PGSTE reduces to plain diffusion attenuation × 0.5.

    No T1 (and no T2) means the mixing-time storage interval contributes no
    relaxation, leaving only diffusion weighting and the 0.5 stimulated-echo
    factor.
    """
    b = 1.0e9
    wf = _pgste_at_b(delta=5e-3, TM=40e-3, b=b)
    b_eff = calc_b(wf)[0]

    S = simulate(N, D, wf, FreeDiffusion(), seed=SEED, require_gpu=False)
    expected = 0.5 * np.exp(-b_eff * D)

    atol_mc = 1.0 / np.sqrt(N)
    npt.assert_allclose(
        S, expected, atol=3 * atol_mc,
        err_msg=f"PGSTE (T1=None) must equal 0.5·exp(-b·D)={expected:.4f}")


def test_pgse_unaffected_by_t1():
    """A PGSE (all-transverse, chi_perp ≡ 1) is unchanged by adding a T1 value.

    With no longitudinal-storage interval the (1-chi)/T1 term is zero at every
    step, so T1 never acts and the signal is identical to the no-T1 run.
    """
    bvecs = np.array([[1.0, 0.0, 0.0]])
    wf = set_b(pgse(delta=5e-3, DELTA=45e-3, G_magnitude=1.0, bvecs=bvecs, n_t=600),
               np.array([1.0e9]))

    S_no_t1 = simulate(N, D, wf, FreeDiffusion(), seed=SEED, require_gpu=False)
    S_t1 = simulate(N, D, wf, FreeDiffusion(), seed=SEED, T1=1.0, require_gpu=False)

    npt.assert_allclose(
        S_t1, S_no_t1, atol=1e-5,
        err_msg="Adding T1 to an all-transverse PGSE must not change the signal")
