"""Emergent MT saturation / Z-spectrum end-to-end (Piece E of the MT staging ladder).

An off-resonance CW saturation pulse, propagated through the forward vector-Bloch
engine on an MT-binding substrate, saturates the broad (short-T2b) bound pool over a
wide offset range while sparing the narrow free-water line; the saturation transfers
to the free pool through the walk's emergent bind/release exchange -- the MT dip and
its Z-spectrum EMERGE (no super-Lorentzian lineshape is imposed).  Validated against
the analytic two-pool Z-spectrum oracle (dmipy_sim.engine.relaxation.mt_z_spectrum).

Fine dt so the carrier
2*pi*offset*dt does not alias.
"""

from dataclasses import replace
from dmipy_sim.engine.bloch import _simulate_bloch
import numpy as np
from dmipy_replay import ScannerSequence
from dmipy_sim.engine.pulse_sequence import saturation_pulse
from dmipy_replay import RFEvent
import pytest

from dmipy_sim import Sphere, bare_gradient_echo, prepend_mt_prep
from dmipy_analytic import relaxation

from tests.conftest import se_tol

# well-mixed sphere (mixing R^2/D = 2 ms << 1/k_f = 25 ms); bound pool broad (T2b ~ 10 us)
R, D = 2e-6, 2e-9
k_f, k_r = 40.0, 80.0
T2a, T1a, T2b, T1b = 80e-3, 1.0, 1e-5, 1.0
W1_HZ, T_SAT, DT = 200.0, 0.025, 2e-5
KAPPA_MT, DWELL = k_f * R / 3.0, 1.0 / k_r


# The spread of one walker's longitudinal magnetisation about its mean was measured as the seed-to-seed scatter of the walker
# mean over five seeds: 0.0139 for 1,500 walkers on resonance (0.54 a walker), 0.0021 for 1,000 at 1 kHz (0.07) and 0.0022
# for 1,500 at 3 kHz (0.085). The walks sit on the oracle there (+0.003 at 1 kHz, +0.0004 at 3 kHz). The oracle test takes
# 2,300 walkers on resonance (``n_resolving(0.08, 0.55, comparisons=2)``) and 300 in the wing (``n_resolving(0.04, 0.10, ...)``).
ORACLE_POINTS = ((0.0, 2_300, 0.55), (1000.0, 300, 0.10))        # offset (Hz), walkers, spread of a walker's Mz
SD_MODES, N_MODES = 0.10, 600       # the modes test at 3 kHz: four comparisons (z = 3.66) at 0.015, 0.021 for two runs


def _mz(offset_hz, *, with_mt, equilibrate='burnin', seed=3, n):
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
    za = np.atleast_1d(relaxation.mt_z_spectrum(offsets, read_pool='a', **kw))
    zb = np.atleast_1d(relaxation.mt_z_spectrum(offsets, read_pool='b', **kw))
    M0b = k_f / k_r
    return (za + M0b * zb) / (1.0 + M0b)


@pytest.mark.gpu
def test_emergent_zspectrum_matches_oracle():
    """The walker-mean Z-spectrum (free + bound, thermal-equilibrium start) matches the two-pool oracle total on
    resonance and in the exchange-shaped wing. The bound pool is equilibrated (the default burn-in), so there is no
    initial-condition offset on resonance.

    Claim: the walk's Z is the oracle's to ``se_tol`` at two offsets: 0.040 on resonance (2,300 walkers; z = 3.48), where
    an error of 0.08, a tenth of the 0.76 dip, fails with the margin the noise passes, and 0.020 at 1 kHz (300 walkers).
    Statistic: a walker mean of the longitudinal magnetisation, of the spread measured above. Far off resonance the
    signal is the free pool's and the third offset of the walk (8 kHz) is not run."""
    Z = _oracle_total(np.array([p[0] for p in ORACLE_POINTS]))
    for (offset, n, sd), z in zip(ORACLE_POINTS, Z):
        mc = _mz(offset, with_mt=True, n=n)
        tol = se_tol(n, sd, comparisons=len(ORACLE_POINTS))
        assert abs(mc - z) < tol, f"{offset:g} Hz: MC {mc:.3f} vs oracle-total {z:.3f} (tolerance {tol:.3f}, {n} walkers)"


@pytest.mark.gpu
def test_equilibrate_binding_modes():
    """Three-way check of the initial-condition handling at a wing offset (1/k_f = 25 ms is
    comparable to t_sat = 25 ms, so the start matters).  'burnin' (adaptive) and 'fast'
    (mid-air; the Z-spectrum has no gradient, so it is provably position-invariant) both reach
    the equilibrium bound pool and match the oracle TOTAL; the legacy 'off' (all-free) is
    biased high because the bound pool is still filling during the saturation.

    Four comparisons (z = 3.66) of runs of 600 walkers (spread 0.1, measured 0.085): a run against the oracle to 0.015, two runs
    against each other to 0.021, and the legacy start (measured 0.033 to 0.037 above the oracle) above it by more than a run's noise."""
    off = 3000.0
    Z_tot = float(_oracle_total(off)[0])
    m_burn = _mz(off, with_mt=True, equilibrate='burnin', n=N_MODES)
    m_fast = _mz(off, with_mt=True, equilibrate='fast', n=N_MODES)
    m_off = _mz(off, with_mt=True, equilibrate='off', n=N_MODES)
    tol, tol_pair = se_tol(N_MODES, SD_MODES, comparisons=4), se_tol(N_MODES, SD_MODES, comparisons=4, runs=2)
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
