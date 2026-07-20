"""Forward vector-Bloch engine (Piece B of the MT staging ladder).

Pins ``dmipy_sim.bloch.simulate_bloch`` -- the single-pass engine that carries
M=(Mx,My,Mz) through real RF pulses + gradient + relaxation (no replay, no
susceptibility) -- against analytic Bloch results and against the scalar cos-phi
engine (``core.simulate``), which it must reproduce on the identical walk.

Free diffusion (no restriction) so a coarse dt already resolves the walk; small
walker counts suffice because with G=0 every walker is coherent.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from dmipy_sim import simulate, pgse, set_b, FreeDiffusion
from dmipy_sim.bloch import simulate_bloch

D = 2e-9


def _zero_waveform(n_t, dt):
    """A gradient-free timeline of ``n_t`` steps (pure RF + relaxation studies)."""
    return SimpleNamespace(G=np.zeros((1, n_t, 3), dtype=np.float64), dt=dt)


# ── 1. transverse relaxation: 90 then free decay -> exp(-t/T2) ──────────────────
def test_transverse_decays_at_T2():
    n_t, dt, T2 = 200, 2e-4, 0.05
    wf = _zero_waveform(n_t, dt)                       # no gradient
    exc = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 90.0}]   # 90_y -> Mx
    s = simulate_bloch(2000, D, wf, FreeDiffusion(), exc, T2=T2, seed=0)
    T = n_t * dt
    assert abs(s[0]) == pytest.approx(np.exp(-T / T2), rel=0.02)
    assert abs(np.angle(s[0])) < 1e-3                 # no gradient/offset -> no phase


# ── 2. inversion recovery: 180 then Mz -> M0(1 - 2 e^{-t/T1}) ────────────────────
def test_inversion_recovery_T1():
    n_t, dt, T1 = 300, 5e-4, 0.8
    wf = _zero_waveform(n_t, dt)
    inv = [{'t_s': 0.0, 'flip_deg': 180.0, 'axis_deg': 0.0}]    # 180_x inverts Mz
    _, mz = simulate_bloch(2000, D, wf, FreeDiffusion(), inv,
                           T1=T1, M0=1.0, seed=0, return_mz=True)
    T = n_t * dt
    assert mz[0] == pytest.approx(1.0 - 2.0 * np.exp(-T / T1), abs=0.01)


# ── 3. off-resonance free precession: phase = 2 pi f T, magnitude preserved ──────
def test_off_resonance_precession():
    n_t, dt, f = 200, 1e-5, 200.0                     # fine dt; f*T=0.4 turn (no wrap)
    wf = _zero_waveform(n_t, dt)
    exc = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 90.0}]
    s = simulate_bloch(2000, D, wf, FreeDiffusion(), exc,
                       off_resonance_hz=f, seed=0)     # no relaxation
    T = n_t * dt
    assert abs(s[0]) == pytest.approx(1.0, rel=1e-3)   # magnitude conserved
    assert np.angle(s[0]) == pytest.approx(2.0 * np.pi * f * T, rel=1e-3)


# ── 4. PGSE parity vs the scalar engine (identical walk) + analytic exp(-bD) ─────
def test_pgse_parity_with_scalar_engine():
    b = 1.0e9
    wf = set_b(pgse(delta=0.01, DELTA=0.03, G_magnitude=0.05,
                    bvecs=[[1., 0., 0.]], n_t=300), b)
    geom, N, seed = FreeDiffusion(), 16000, 7
    scalar = simulate(N, D, wf, geom, seed=seed, require_gpu=False)  # <cos phi>
    exc = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 90.0}]         # 90_y -> Mx=cos phi
    vec = simulate_bloch(N, D, wf, geom, exc, seed=seed)             # no relaxation
    tol = max(0.02, 1.0 / np.sqrt(N))
    # same seed + FreeDiffusion (identity reflect) => bit-identical walk
    assert np.real(vec[0]) == pytest.approx(float(scalar[0]), abs=tol)
    assert np.real(vec[0]) == pytest.approx(np.exp(-b * D), abs=tol)  # free-diffusion truth


# ── 5. emergent CPMG refocusing: a 180 train refocuses a static off-resonance ────
def test_cpmg_refocuses_static_offset():
    dt, TE, n_echo, T2, f = 1e-4, 4e-3, 4, 0.06, 100.0
    half = int(round((TE / 2) / dt))
    n_t = n_echo * int(round(TE / dt)) + 1
    wf = _zero_waveform(n_t, dt)
    echo_steps = [int(round((k + 1) * TE / dt)) for k in range(n_echo)]

    exc = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 90.0}]         # 90_y
    refocus = [{'t_s': (2 * k + 1) * (TE / 2), 'flip_deg': 180.0, 'axis_deg': 0.0}
               for k in range(n_echo)]                               # 180_x train

    ec = simulate_bloch(4000, D, wf, FreeDiffusion(), exc + refocus,
                        T2=T2, off_resonance_hz=f, seed=0, echo_steps=echo_steps)[0]
    # echoes track T2 and the static offset is REFOCUSED (phase ~0 at every echo)
    for k in range(n_echo):
        t = (k + 1) * TE
        assert abs(ec[k]) == pytest.approx(np.exp(-t / T2), rel=0.03)
        assert abs(np.angle(ec[k])) < 0.15

    # control: WITHOUT the 180 train the same static offset dephases the phase away
    free = simulate_bloch(4000, D, wf, FreeDiffusion(), exc,
                          T2=T2, off_resonance_hz=f, seed=0, echo_steps=echo_steps)[0]
    assert abs(np.angle(free[-1])) > 1.0                            # large unrefocused phase
