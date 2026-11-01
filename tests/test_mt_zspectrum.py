"""Emergent MT saturation / Z-spectrum end-to-end (Piece E of the MT staging ladder).

An off-resonance CW saturation pulse, propagated through the forward vector-Bloch
engine on an MT-binding substrate, saturates the broad (short-T2b) bound pool over a
wide offset range while sparing the narrow free-water line; the saturation transfers
to the free pool through the walk's emergent bind/release exchange -- the MT dip and
its Z-spectrum EMERGE (no super-Lorentzian lineshape is imposed).  Validated against
the analytic two-pool Z-spectrum oracle (dmipy_sim.engine.mt.mt_z_spectrum).

Fine dt so the carrier
2*pi*offset*dt does not alias.
"""

from dataclasses import replace
from dmipy_sim.engine.bloch import _simulate_bloch
import numpy as np
from dmipy_sim import ScannerSequence
from dmipy_sim.engine.pulse_sequence import saturation_pulse
from dmipy_sim import RFEvent
import pytest

from dmipy_sim import Sphere, bare_gradient_echo, prepend_mt_prep
from dmipy_sim.engine import mt

from tests.conftest import se_tol

# well-mixed sphere (mixing R^2/D = 2 ms << 1/k_f = 25 ms); bound pool broad (T2b ~ 10 us)
R, D = 2e-6, 2e-9
k_f, k_r = 40.0, 80.0
T2a, T1a, T2b, T1b = 80e-3, 1.0, 1e-5, 1.0
W1_HZ, T_SAT, DT = 200.0, 0.025, 2e-5
KAPPA_MT, DWELL = k_f * R / 3.0, 1.0 / k_r


SD_MZ = 0.3          # a walker's longitudinal magnetisation spreads by at most this about the mean (0.285 on resonance, measured)
N_Z = 3_000          # n_resolving(0.04, SD_MZ, comparisons=2) = 2,730: a 0.04 error fails the tolerance the noise passes (z = 3.48)


def _mz(offset_hz, *, with_mt, equilibrate='burnin', seed=3, n=N_Z):
    n_t = int(round(T_SAT / DT)) + 1
    wf = ScannerSequence(G=np.zeros((1, n_t, 3)), dt=DT)
    flip = 360.0 * W1_HZ * T_SAT                            # CW over the window
    rf = [RFEvent(T_SAT / 2, flip, axis_deg=0.0, duration_s=T_SAT, offset_hz=offset_hz)]
    kw = dict(T2=T2a, T1=T1a, return_mz=True, seed=seed)
    if with_mt:
        kw.update(kappa_MT=KAPPA_MT, dwell_time=DWELL, T2_bound=T2b, T1_bound=T1b,
                  equilibrate_binding=equilibrate)
    _, mz = _simulate_bloch(n, D, replace(wf, rf=rf), Sphere(radius=R), **kw)
    return float(mz[0])


def _oracle_total(offsets):
    """Two-pool oracle TOTAL longitudinal (free + bound), normalised to equilibrium -- the
    quantity the walker-mean ``return_mz`` reports.  NB: comparing the walker-mean (total) to
    the oracle FREE pool (read_pool='a') only agrees by coincidence when the bound pool is
    under-filled (all-free start); the honest comparison is total-to-total from a common
    thermal-equilibrium start (see test_equilibrate_binding_modes)."""
    offsets = np.atleast_1d(offsets)
    kw = dict(w1_hz=W1_HZ, t_sat=T_SAT, T1a=T1a, T2a=T2a, T1b=T1b, T2b=T2b, k_f=k_f, k_r=k_r)
    za = np.atleast_1d(mt.mt_z_spectrum(offsets, read_pool='a', **kw))
    zb = np.atleast_1d(mt.mt_z_spectrum(offsets, read_pool='b', **kw))
    M0b = k_f / k_r
    return (za + M0b * zb) / (1.0 + M0b)


@pytest.mark.gpu
def test_emergent_zspectrum_matches_oracle():
    """The walker-mean Z-spectrum (free + bound, thermal-equilibrium start) matches the two-pool oracle total on
    resonance and in the exchange-shaped wing. The bound pool is equilibrated (the default burn-in), so there is no
    initial-condition offset on resonance.

    Claim: the walk's Z is the oracle's to ``se_tol`` at two offsets, 0.019 at 3,000 walkers (z = 3.48): a 0.04 error, a
    twentieth of the 0.76 dip, fails with the margin the noise passes. Statistic: a walker mean of the longitudinal
    magnetisation, whose spread is at most 0.3 (0.285 on resonance, where the signal is 0.24). Far off resonance
    the signal is the free pool's and the third offset of the walk (8 kHz) is not run: it costs a third of the test."""
    offsets = np.array([0.0, 1000.0])
    mc = np.array([_mz(o, with_mt=True) for o in offsets])
    Z = _oracle_total(offsets)
    tol = se_tol(N_Z, SD_MZ, comparisons=len(offsets))
    assert np.max(np.abs(mc - Z)) < tol, f"MC {np.round(mc,3)} vs oracle-total {np.round(Z,3)} (tolerance {tol:.3f})"


@pytest.mark.gpu
def test_equilibrate_binding_modes():
    """Three-way check of the initial-condition handling at a wing offset (1/k_f = 25 ms is
    comparable to t_sat = 25 ms, so the start matters).  'burnin' (adaptive) and 'fast'
    (mid-air; the Z-spectrum has no gradient, so it is provably position-invariant) both reach
    the equilibrium bound pool and match the oracle TOTAL; the legacy 'off' (all-free) is
    biased high because the bound pool is still filling during the saturation.

    Four comparisons (z = 3.66) of runs of 3,000 walkers (spread 0.3): a run against the oracle to 0.020, two runs against
    each other to 0.028, and the legacy start above the oracle by more than a run's noise."""
    off = 3000.0
    Z_tot = float(_oracle_total(off)[0])
    m_burn = _mz(off, with_mt=True, equilibrate='burnin')
    m_fast = _mz(off, with_mt=True, equilibrate='fast')
    m_off = _mz(off, with_mt=True, equilibrate='off')
    tol, tol_pair = se_tol(N_Z, SD_MZ, comparisons=4), se_tol(N_Z, SD_MZ, comparisons=4, runs=2)
    assert abs(m_burn - Z_tot) < tol, f"burnin {m_burn:.3f} vs oracle-total {Z_tot:.3f}"
    assert abs(m_fast - Z_tot) < tol, f"fast {m_fast:.3f} vs oracle-total {Z_tot:.3f}"
    assert abs(m_burn - m_fast) < tol_pair, f"burnin {m_burn:.3f} vs fast {m_fast:.3f}"
    # the legacy all-free start under-fills the bound pool -> biased high (the artifact fixed here)
    assert m_off - Z_tot > tol, f"expected 'off' biased high: off {m_off:.3f} vs total {Z_tot:.3f}"



def test_equilibrate_binding_is_exactly_burnin_fast_or_off():
    """Three spellings and no other: 'auto', True, False and None are refused by name; 'fast' where it is unsafe (an
    unknown S/V, or a gradient with a bound pool that is not MR-dark) is a ValueError with the reason, never a warning
    and a demotion."""
    import warnings
    import pytest
    from dmipy_sim import Box1D, Cylinder
    from dmipy_sim.engine.mt import resolve_equilibrate_mode as resolve
    cyl = Cylinder(1e-6, (0, 0, 1))
    for mode in ("burnin", "fast", "off"):
        assert resolve(mode, cyl) == mode
    for bad in ("auto", True, False, None, "Burnin", 1):
        with pytest.raises(ValueError, match=r"exactly 'burnin' \| 'fast' \| 'off'"):
            resolve(bad, cyl)
    with pytest.raises(ValueError, match="surface-to-volume"):
        resolve("fast", Box1D(1e-6))
    G = np.ones((1, 5, 3))
    with pytest.raises(ValueError, match="not MR-dark"):
        resolve("fast", cyl, G=G, T2=0.05, T2_bound=0.04)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve("fast", cyl, G=G, T2=0.05, T2_bound=1e-4) == "fast"        # MR-dark bound pool
        assert resolve("fast", cyl, G=0 * G, T2=0.05, T2_bound=0.04) == "fast"    # no gradient
