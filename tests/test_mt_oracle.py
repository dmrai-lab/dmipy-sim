"""The walk's MT binding rule and the ``Substrate`` MT configuration (kappa_MT/dwell_time <-> f_bound/k_forward
conversions, through dmipy-analytic's two-region exchange relations, whose own tests are dmipy-analytic's). Pure
numpy (no Monte-Carlo), so they run fast on CPU.
"""
import numpy as np
import pytest

from dmipy_sim.engine import mt
from dmipy_sim.substrate import Substrate

def test_stick_probability_form_and_saturation():
    """The binding clock: the walk's own law, linear for a shallow hit, below 1 and tending to it for a deep one."""
    import jax.numpy as jnp
    from dmipy_sim.geometry._boundary import bind_probability
    D = 1.7e-9
    kappa_MT = 1e-6
    d_perp = np.array([0.0, 1e-8, 1e-3, 1e-2])
    p = mt.stick_probability(d_perp, kappa_MT, D)
    assert p[0] == 0.0
    assert p[1] == pytest.approx(2.0 * kappa_MT / D * 1e-8, rel=1e-5)
    assert p[2] == pytest.approx(1.0 - np.exp(-2.0 * kappa_MT / D * 1e-3))
    assert 1.0 - p[3] == pytest.approx(np.exp(-2.0 * kappa_MT / D * 1e-2), rel=1e-6)   # almost surely, never more
    walk = np.asarray(bind_probability(jnp.float32(kappa_MT / D), jnp.asarray(2.0 * d_perp, jnp.float32)))
    assert np.allclose(walk, p, rtol=1e-5, atol=1e-7)


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
