"""Tests for the continuous B1(t) RF representation (dmipy_sim.rf)."""
import numpy as np
import pytest

from dmipy_sim.rf import B1Pulse, bloch_simulate, slice_profile
from dmipy_sim.constants import GAMMA
from dmipy_sim.sequences import scanner_constants as scc


# ── basic quantities ──────────────────────────────────────────────────────────
def test_hard_pulse_area_gives_nominal_flip():
    p = B1Pulse.hard(flip_deg=90, duration=2.56e-3, dt=1e-5)
    assert p.nominal_flip_deg == pytest.approx(90.0, rel=1e-6)
    assert p.duration == pytest.approx(2.56e-3, rel=1e-9)
    assert p.peak_b1 > 0 and p.b1_rms > 0 and p.sar_proxy > 0


def test_sinc_area_normalised_to_flip():
    p = B1Pulse.windowed_sinc(flip_deg=90, duration=2.56e-3, dt=1e-5, time_bw=4)
    assert p.nominal_flip_deg == pytest.approx(90.0, rel=1e-6)


# ── Bloch forward: on-resonance flips ─────────────────────────────────────────
def test_hard_90_on_resonance_tips_to_transverse():
    p = B1Pulse.hard(flip_deg=90, duration=1e-3, dt=1e-6)
    Mxy, Mz = bloch_simulate(p, df_hz=0.0)
    assert abs(Mz) < 1e-3
    assert abs(Mxy) == pytest.approx(1.0, abs=1e-3)


def test_hard_180_on_resonance_inverts():
    p = B1Pulse.hard(flip_deg=180, duration=1e-3, dt=1e-6)
    Mxy, Mz = bloch_simulate(p, df_hz=0.0)
    assert Mz == pytest.approx(-1.0, abs=1e-3)
    assert abs(Mxy) < 1e-3


def test_phase_sets_rotation_axis():
    # 90 about y (phase 90 deg) tips +z -> +x ; about x (phase 0) tips +z -> -y
    px = B1Pulse.hard(90, 1e-3, 1e-6, phase_deg=0.0)
    py = B1Pulse.hard(90, 1e-3, 1e-6, phase_deg=90.0)
    Mxy_x, _ = bloch_simulate(px)
    Mxy_y, _ = bloch_simulate(py)
    # axis (cos,sin,0): phase 0 -> rotates z toward -y (Mxy ~ -i); phase 90 -> toward +x (Mxy ~ +1)
    assert Mxy_x.imag == pytest.approx(-1.0, abs=2e-3)
    assert Mxy_y.real == pytest.approx(1.0, abs=2e-3)


def test_zero_b1_scale_no_rotation():
    p = B1Pulse.hard(90, 1e-3, 1e-6)
    Mxy, Mz = bloch_simulate(p, b1_scale=0.0)
    assert Mz == pytest.approx(1.0, abs=1e-9)
    assert abs(Mxy) < 1e-9


# ── off-resonance behaviour ───────────────────────────────────────────────────
def test_large_off_resonance_reduces_excitation():
    # a long, low-bandwidth 90 loses excitation efficiency far off resonance
    p = B1Pulse.hard(90, 4e-3, 1e-5)
    Mxy0, _ = bloch_simulate(p, df_hz=0.0)
    Mxy_off, _ = bloch_simulate(p, df_hz=5000.0)        # 5 kHz >> bandwidth
    assert abs(Mxy_off) < abs(Mxy0)
    assert abs(Mxy0) == pytest.approx(1.0, abs=1e-3)


def test_ensemble_broadcasting():
    p = B1Pulse.hard(90, 1e-3, 1e-6)
    df = np.linspace(-2000, 2000, 7)
    b1 = np.linspace(0.8, 1.2, 7)
    Mxy, Mz = bloch_simulate(p, df_hz=df, b1_scale=b1)
    assert Mxy.shape == (7,) and Mz.shape == (7,)


def test_history_shape_and_endpoint():
    p = B1Pulse.hard(90, 1e-3, 1e-6)
    Mxy, Mz, hist = bloch_simulate(p, return_history=True)
    assert hist.shape == (p.n + 1, 3, 1)
    np.testing.assert_allclose(hist[0, 2, 0], 1.0)               # starts at +z
    np.testing.assert_allclose(hist[-1, 0, 0] + 1j * hist[-1, 1, 0], Mxy, atol=1e-9)


# ── slice selectivity ─────────────────────────────────────────────────────────
def test_sinc_slice_profile_passband_vs_stopband():
    # slice-selective 90: excited on-slice, suppressed far off-slice
    p = B1Pulse.windowed_sinc(90, 2.56e-3, 1e-5, time_bw=4)
    Gss = 20e-3                                                  # T/m
    z = np.linspace(-0.02, 0.02, 401)                           # m
    _, Mxy, _ = slice_profile(p, Gss, z)
    on = np.abs(Mxy)[np.abs(z) < 1e-3].mean()                   # near slice centre
    off = np.abs(Mxy)[np.abs(z) > 0.015].mean()                 # far edges
    assert on > 0.7
    assert off < 0.2
    assert on > 5 * off


# ── deliverability against the scanner_constants catalogue ────────────────────
def test_deliverability_reads_scanner_limits():
    model = "ge_signa_premier_3T"
    peak_lim = scc.get_limit(model, "rf", "peak_B1_body_coil", si=True)   # Tesla
    assert peak_lim is not None and 5e-6 < peak_lim < 5e-5                # ~ tens of uT

    soft = B1Pulse.hard(90, 2.56e-3, 1e-5)        # ~2-3 uT peak -> deliverable
    rep = soft.deliverability(model)
    assert rep["peak_b1_T"] < peak_lim
    assert rep["deliverable"] is True

    stiff = B1Pulse.hard(90, 1e-4, 1e-6)          # 0.1 ms 90 -> tens of uT peak -> not
    assert stiff.peak_b1 > peak_lim
    assert stiff.is_deliverable(model) is False
