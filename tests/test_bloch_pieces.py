"""The Bloch replay over pieces (#147): the walk cut at every save and every RF instant, so the phase accrued before
a pulse is flipped by it exactly wherever the pulse falls and whatever gradient plays through it; and the coherence
gates read accumulated channels by interval average, sampled ones through the path interpolant."""
import numpy as np
from dmipy_sim import RFEvent
import pytest

from dmipy_sim.constants import GAMMA
from dmipy_sim.replay._replay_kernel import effective_gradient, gate_weights, bin_gate, piece_moments
from dmipy_sim.replay.trajectories import replay, replay_bloch, replay_bloch_jax

N_W, N_T, DT = 300, 80, 1e-4
T = (N_T - 1) * DT
D0 = 2e-9


def _walk(seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(scale=np.sqrt(2 * D0 * DT), size=(N_W, N_T, 3)), axis=1)


def _bloch_phase(traj, G, rf):
    """Per-walker transverse phase at the end of a 90y-180x Bloch replay (no relaxation)."""
    M, _ = replay_bloch(traj, DT, G, DT, rf, return_walker_signals=True)
    return M[0] + 1j * M[1]


def test_a_gradient_through_a_180_on_a_save_is_refocused_exactly():
    """A constant gradient over the whole echo with the 180 at a save: the Bloch route (physical G, emergent echo)
    and the scalar route (the 180-folded effective gradient) accumulate the same per-walker phase to rounding."""
    traj = _walk()
    G = np.zeros((1, N_T, 3)); G[0, :, 0] = 0.02                                   # on through everything
    k180 = 40
    rf = [RFEvent(0.0, 90.0, axis_deg=90.0), RFEvent(k180 * DT, 180.0, axis_deg=0.0)]
    mxy = _bloch_phase(traj, G, rf)
    assert np.allclose(np.abs(mxy), 1.0, atol=1e-12)
    G_eff = G.copy(); G_eff[0, k180:, 0] *= -1                                     # the folded gradient
    phi, _, _ = replay(traj, DT, G_eff, DT, return_walker_signals=True)
    d = np.angle(mxy * np.exp(1j * phi[0]))                                         # 90y then precession: Mxy ~ e^{-i phi}
    d2 = np.angle(mxy * np.exp(-1j * phi[0]))
    assert min(np.abs(d).max(), np.abs(d2).max()) < 1e-10


def test_a_180_between_saves_refocuses_a_static_gradient_to_the_analytic_phase():
    """Stationary spins in a constant gradient: the net phase is gamma G.r (2 t_180 - T); exactly zero for a 180 at
    T/2 wherever that falls between saves, and the analytic value for a 180 off centre (no snapping to a save)."""
    rng = np.random.default_rng(1)
    r = rng.uniform(-5e-6, 5e-6, (N_W, 3))
    traj = np.repeat(r[:, None, :], N_T, axis=1)
    G = np.zeros((1, N_T, 3)); G[0, :, 0] = 0.03
    for t180 in (T / 2, 0.37 * T + 0.3 * DT):
        rf = [RFEvent(0.0, 90.0, axis_deg=90.0), RFEvent(t180, 180.0, axis_deg=0.0)]
        mxy = _bloch_phase(traj, G, rf)
        expect = GAMMA * 0.03 * r[:, 0] * (2 * t180 - T)
        err = min(np.abs(np.angle(mxy * np.exp(1j * expect))).max(), np.abs(np.angle(mxy * np.exp(-1j * expect))).max())
        assert err < 1e-9, f"t180={t180}: phase error {err:.2e}"
    # numpy and JAX agree with the pulse between saves too
    rf = [RFEvent(0.0, 90.0, axis_deg=90.0), RFEvent(0.37 * T + 0.3 * DT, 180.0, axis_deg=0.0)]
    np.testing.assert_allclose(replay_bloch_jax(traj, DT, G, DT, rf), replay_bloch(traj, DT, G, DT, rf), atol=2e-5)


def test_pgse_with_the_pulses_clear_of_the_lobes_matches_the_scalar_replay():
    traj = _walk(2)
    G = np.zeros((2, N_T, 3)); G[0, 5:25, 0] = 0.04; G[0, 45:65, 0] = 0.04; G[1, 5:25, 2] = 0.06; G[1, 45:65, 2] = 0.06
    rf = [RFEvent(0.0, 90.0, axis_deg=90.0), RFEvent(35 * DT, 180.0, axis_deg=0.0)]
    S_bloch = np.abs(replay_bloch(traj, DT, G, DT, rf))
    G_eff = G.copy(); G_eff[:, 35:] *= -1
    phi, _, _ = replay(traj, DT, G_eff, DT, return_walker_signals=True)         # the scalar route's per-walker phase
    np.testing.assert_allclose(S_bloch, np.abs(np.exp(1j * phi).mean(1)), atol=1e-9)


def test_gates_read_accumulated_channels_by_interval_and_sampled_ones_by_interpolant():
    n_wf, dt_wf = 200, 4e-5                                                       # a gate on its own grid
    chi = np.ones(n_wf); chi[int(0.5 * T / dt_wf):] = 0.0                          # transverse until T/2
    b = bin_gate(chi, dt_wf, N_T, DT)[0]
    assert np.allclose(bin_gate(np.ones(n_wf), dt_wf, N_T, DT), 1.0, atol=1e-12)  # the whole step, every save
    k_half = int(round(0.5 * T / DT))
    assert np.allclose(b[:k_half], 1.0, atol=1e-12) and np.allclose(b[k_half + 1:], 0.0, atol=1e-12) and -1e-12 <= b[k_half] <= 1 + 1e-12
    assert b.sum() * DT == pytest.approx(0.5 * T + DT, abs=dt_wf)                  # the gate's on-time, plus the step before t = 0
    w = gate_weights(chi, dt_wf, N_T, DT)[0]
    assert w.sum() * DT == pytest.approx(int(0.5 * T / dt_wf) * dt_wf, rel=1e-12)  # the interpolant integral: exact on-time
    # the pieces of a cut: the moments add up to the save-interval moments whatever the cut
    rng = np.random.default_rng(3)
    G = rng.normal(size=(1, 37, 3)); saves = np.arange(N_T) * DT
    cut = np.sort(np.concatenate([saves, rng.uniform(0, T, 7)]))
    A0, A1, k = piece_moments(G, 1e-5, cut, N_T, DT)
    A0s, A1s, _ = piece_moments(G, 1e-5, saves, N_T, DT)
    for kk in range(N_T - 1):
        np.testing.assert_allclose(A0[:, k == kk].sum(1), A0s[:, kk], atol=1e-14)
        np.testing.assert_allclose(A1[:, k == kk].sum(1), A1s[:, kk], atol=1e-18)
    with pytest.raises(ValueError, match="outside the walk"):
        replay_bloch(_walk(), DT, np.zeros((1, N_T, 3)), DT, [RFEvent(T + DT, 90.0)])
