"""The walk's MT binding rule and the ``Substrate`` MT configuration (kappa_MT/dwell_time <-> f_bound/k_forward
conversions, through dmipy-analytic's two-region exchange relations, whose own tests are dmipy-analytic's). Pure
numpy (no Monte-Carlo), so they run fast on CPU.
"""
import numpy as np
import pytest

from dmipy_sim.engine import mt
from dmipy_sim.substrate import Substrate

def test_stick_probability_form_and_saturation():
    D = 1.7e-9
    kappa_MT = 1e-6
    # saturation needs d_perp >= D/(2 kappa) = 8.5e-4 m; use a deep hit past that
    d_perp = np.array([0.0, 1e-8, 1e-2])
    p = mt.stick_probability(d_perp, kappa_MT, D)
    assert p[0] == 0.0
    assert p[1] == pytest.approx(2.0 * kappa_MT / D * 1e-8)
    assert p[2] == 1.0                      # saturates (clamped) at 1 for a deep hit


# ── transverse oracle: T2b -> 0 gives R2a + k_f ─────────────────────────────────
# ── longitudinal oracle: conservation + equilibrium ─────────────────────────────
# ── Z-spectrum: MT dip near resonance, broad bound saturation ──────────────────
# ── Substrate plumbing ─────────────────────────────────────────────────────────
def test_substrate_mt_off_by_default():
    s = Substrate()
    assert s.kappa_MT == 0.0
    assert s.mt_on is False


def test_substrate_mt_validators():
    with pytest.raises(ValueError):
        Substrate(kappa_MT=1e-6, dwell_time=0.0)      # reactivity without residence
    s = Substrate(kappa_MT=1e-6, dwell_time=5e-3)
    assert s.mt_on is True


def test_substrate_with_mt_conversion():
    S_over_V = 2.0 / 2e-6
    s = Substrate.with_mt(f_bound=0.1, k_forward=25.0, S_over_V=S_over_V)
    assert s.mt_on
    assert s.mt_forward_rate(S_over_V) == pytest.approx(25.0, rel=1e-6)
    assert s.mt_bound_fraction(S_over_V) == pytest.approx(0.1, rel=1e-6)
