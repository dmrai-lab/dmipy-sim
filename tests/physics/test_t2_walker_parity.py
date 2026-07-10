"""Parity test: per-walker T2 accumulation vs. the analytical post-hoc formula.

Magnetisation is fully transverse throughout (ideal instantaneous pulses), so for a
spatially homogeneous T2 the per-walker log-weight approach and the post-hoc scalar
formula must agree (up to MC noise):

    S_T2 = exp(-bD) * exp(-tau_perp / T2),   tau_perp = n_t * dt.
"""

import numpy as np
import numpy.testing as npt

from dmipy_sim import simulate, FreeDiffusion, set_b
from dmipy_sim.waveforms import pgse


# ── Simulation constants ────────────────────────────────────────────────────
D         = 2e-9          # m²/s — standard test value
N_WALKERS = 100_000
SEED      = 42
T2        = 80e-3         # 80 ms — realistic white matter value

_ATOL = 1.0 / np.sqrt(N_WALKERS)  # MC noise floor, ~0.003 for N=100 000


def _make_waveform(b_values):
    bvecs = np.tile([1., 0., 0.], (len(b_values), 1))
    return set_b(pgse(delta=1e-3, DELTA=40e-3, G_magnitude=1.0, bvecs=bvecs, n_t=500),
                 b_values)


def test_t2_walker_matches_analytical():
    """Per-walker T2 gives exp(-bD) * exp(-tau_perp/T2) for all-transverse PGSE."""
    b_values = np.array([1e8, 5e8, 1e9, 2e9])
    wf = _make_waveform(b_values)
    # Fully transverse => tau_perp = n_t * dt (T2 accrues every step of the scan).
    tau_perp = wf.G.shape[1] * wf.dt

    signals = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED, T2=T2)
    expected = np.exp(-b_values * D) * np.exp(-tau_perp / T2)

    npt.assert_allclose(
        signals, expected, atol=3 * _ATOL,
        err_msg="Per-walker T2 must match exp(-bD)*exp(-tau_perp/T2)",
    )


def test_no_t2_unaffected():
    """Omitting T2 must give the same result as simulate() without the kwarg."""
    wf = _make_waveform(np.array([1e9]))
    S_plain = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED)
    S_none  = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED, T2=None)
    npt.assert_array_equal(
        S_plain, S_none,
        err_msg="simulate(..., T2=None) must equal simulate() without the kwarg",
    )
