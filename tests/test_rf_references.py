"""Closed-form RF reference cases -- the analogue of the LTE bang-bang / known-max-b
sanity anchors used to validate the gradient (NOW) waveforms.

These check the B1(t) Bloch forward (dmipy_sim.rf.bloch_simulate) against analytic
results that are *known* independently of the simulator:

  1. rectangular (hard) pulse: exact rotation about the tilted effective field, for
     arbitrary off-resonance (the bang-bang analogue -- a constant-amplitude pulse).
  2. small-tip-angle theorem (Pauly 1989): the spectral profile of a low-flip pulse is
     the Fourier transform of the B1 envelope.
"""
import numpy as np
import pytest

from dmipy_sim.rf import B1Pulse, bloch_simulate
from dmipy_sim.constants import GAMMA


def _rect_offres_analytic(B1_T, tau_s, df_hz):
    """Exact Bloch solution for a constant-B1 rectangle along x, M0=+z.

    Rotation of z-hat about the effective field axis (w1, 0, dw)/w_eff by angle w_eff*tau,
    with w1 = gamma*B1 (rad/s), dw = 2*pi*df (rad/s).  Returns (|Mxy|, Mz)."""
    w1 = GAMMA * B1_T
    dw = 2.0 * np.pi * np.asarray(df_hz, float)
    w_eff = np.sqrt(w1 ** 2 + dw ** 2)
    theta = w_eff * tau_s
    sa = w1 / w_eff                       # sin(tilt) = transverse fraction
    ca = dw / w_eff                       # cos(tilt) = longitudinal fraction
    Mx = sa * ca * (1.0 - np.cos(theta))
    My = -sa * np.sin(theta)
    Mz = np.cos(theta) + ca ** 2 * (1.0 - np.cos(theta))
    return np.hypot(Mx, My), Mz


def test_rect_pulse_exact_offresonance():
    """Hard pulse vs the exact tilted-effective-field solution across off-resonance."""
    B1, tau = 5e-6, 1e-3                                  # 5 uT, 1 ms
    p = B1Pulse.hard(flip_deg=np.degrees(GAMMA * B1 * tau), duration=tau, dt=1e-6,
                     phase_deg=0.0)
    df = np.linspace(-3000, 3000, 41)
    Mxy, Mz = bloch_simulate(p, df_hz=df)
    Mxy_a, Mz_a = _rect_offres_analytic(B1, tau, df)
    assert np.max(np.abs(np.abs(Mxy) - Mxy_a)) < 2e-3
    assert np.max(np.abs(Mz - Mz_a)) < 2e-3


def test_rect_pulse_on_resonance_exact_flip():
    """On resonance the rect pulse rotates by exactly gamma*B1*tau."""
    B1, tau = 10e-6, 0.5e-3
    flip = GAMMA * B1 * tau
    p = B1Pulse.hard(flip_deg=np.degrees(flip), duration=tau, dt=1e-6)
    Mxy, Mz = bloch_simulate(p, df_hz=0.0)
    assert abs(complex(Mxy[0])) == pytest.approx(np.sin(flip), abs=1e-3)
    assert float(Mz[0]) == pytest.approx(np.cos(flip), abs=1e-3)


def test_small_tip_fourier_theorem():
    """Pauly small-tip theorem: |Mxy(df)| ~= gamma * |FT{B1}(df)| for small flip.

    A low-flip windowed sinc therefore excites a near-rectangular spectral profile.
    """
    dt = 1e-5
    p = B1Pulse.windowed_sinc(flip_deg=8.0, duration=2.56e-3, dt=dt, time_bw=4)
    df = np.linspace(-4000, 4000, 161)

    Mxy, _ = bloch_simulate(p, df_hz=df)

    # analytic small-tip: Mxy(dw) = i*gamma * integral B1(t) e^{i dw (t - t_center)} dt
    t = p.times - 0.5 * p.duration
    dw = 2.0 * np.pi * df
    ft = (p.b1[None, :] * np.exp(1j * dw[:, None] * t[None, :])).sum(axis=1) * dt
    Mxy_a = GAMMA * np.abs(ft)

    # match in magnitude; small-tip error ~ flip^2/6 (~0.2% at 8 deg) plus discretisation
    peak = Mxy_a.max()
    assert np.max(np.abs(np.abs(Mxy) - Mxy_a)) < 0.03 * peak
    # and the profile is genuinely band-limited (rect-like): high in-band, low out-of-band
    inband = np.abs(df) < 600
    assert np.abs(Mxy)[inband].mean() > 5 * np.abs(Mxy)[~inband].mean()


def test_small_tip_hard_pulse_is_sinc_in_frequency():
    """A short rectangular low-flip pulse has a sinc spectral profile (FT of a box)."""
    B1, tau, dt = 1e-6, 1e-3, 1e-6
    p = B1Pulse.hard(flip_deg=np.degrees(GAMMA * B1 * tau), duration=tau, dt=dt)
    df = np.linspace(-4000, 4000, 161)
    Mxy, _ = bloch_simulate(p, df_hz=df)
    # FT of a box of width tau -> sinc(df*tau); first nulls at df = +-1/tau
    expected = GAMMA * B1 * tau * np.abs(np.sinc(df * tau))
    assert np.max(np.abs(np.abs(Mxy) - expected)) < 0.02 * expected.max()
