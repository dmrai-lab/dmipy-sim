"""Every replay term is a contraction in the space its channel is stored in (dmipy-sim#241).

The gradient row was the only one: the relaxation term decoded the occupancy runs to a track, the gated surface
term decoded the boundary bridge to a series, and the field term decoded the positions and the 64-mode path
channel to the save grid and integrated. Each is now a product with the stored coefficients -- runs, bridge
bands, cosine modes -- and equals the decoded computation to rounding; and no route through `walker_signals`
touches a decoded track or trajectory on a pack that carries the path channel.
"""
from __future__ import annotations

import dmipy_replay.replay.compression as _r_compression
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences as _seqmod
from dmipy_sim.replay import ReplayPack
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay import compression as _cx
from dmipy_replay.replay._replay_kernel import bin_gate
from dmipy_replay.acquisition.epg import pathway_weight
from dmipy_replay.spec.tissue import Tissue
from tests.replay_frames import field_along
from tests.conftest import dense_relaxation_logweight

D0 = 2e-9


def _series(pk, t, bands, start, end):
    """A channel's bridge (``start``, ``end`` and ``bands`` per walker) evaluated at the instants ``t`` of its window."""
    T = pk.T_window
    k = np.arange(1, bands.shape[1] + 1)
    return start[:, None] + (end - start)[:, None] * (t / T)[None, :] + bands @ (np.sqrt(2.0 / T) * np.sin(np.pi * np.outer(k, t) / T))


def _dense_logweights(pk, seq, T2, T1, rho2):
    """The decoded computation of the relaxation and surface log-weights: the oracle. The occupancy decoded on the
    blocks and relaxed through the gate binned there; the contact's series evaluated at the gate's own samples, each
    held over its interval, ``sum_i chi_i (B(t_i+1) - B(t_i))``."""
    ch = pk.meta["compression"]["channels"]
    n_t, dt = pk.n_t, pk.dt
    G = np.asarray(seq.G); chi_wf = np.ones(G.shape[1]) if seq.chi_perp is None else np.asarray(seq.chi_perp).reshape(-1)
    active = bin_gate(np.ones(chi_wf.shape[0]), seq.dt, n_t, dt)[0]; chi = bin_gate(chi_wf, seq.dt, n_t, dt)[0]
    comp = _r_compression.occupancy_at(pk.arrays, next(c for c in ch["compartment"]["columns"] if c["name"] == "comp"),
                            np.arange(n_t) * dt)                                    # the pool at every block end
    lw = dense_relaxation_logweight(comp, T2, T1, dt, chi, active)
    meta = dict(ch["boundary_local_time"])
    bands = _r_compression.bridge_bands(pk.arrays, meta)
    t = np.minimum(np.arange(chi_wf.shape[0]) * float(seq.dt), pk.T_window)
    B = _series(pk, t, bands, np.asarray(pk.arrays["blt_start"], np.float64), np.asarray(pk.arrays["blt_endpoint"], np.float64))
    return lw + (rho2 / D0) * (np.diff(B, axis=1) * chi_wf[None, :-1]).sum(1)


@pytest.mark.parametrize("make_seq", [
    lambda n_t: _seqmod.gre(4e-3, n_t=n_t),                                                   # an FID shorter than the walk
    lambda n_t: _seqmod.pgse([[1, 0, 0]], 1e-3, 3e-3, bvalues=[1e9], TE=6e-3, n_t=n_t, slew_rate=np.inf),
    lambda n_t: _seqmod.pgste([[1, 0, 0]], 1e-3, 3e-3, bvalues=[1e9], TE=8e-3, n_t=n_t, slew_rate=np.inf),   # a stored period
], ids=["fid", "pgse", "pgste"])
def test_relaxation_and_surface_weights_equal_the_decoded_ones(pack, make_seq):
    seq = make_seq(4 * pack.n_t + 1)
    T2, T1, rho2 = [0.08, 0.03], [1.0, 1.2], 1e-5                                   # by pool id, for the oracle
    w, ew, _ = pack.walker_signals(seq, tissue=Tissue(T2={"extra": 0.08, "intra": 0.03}, T1={"extra": 1.0, "intra": 1.2}, rho2=rho2, D=D0))
    # the route's weights also carry the amplitude of the pathway the readout is (1 for the fid and the spin
    # echo, 0.5 for the stimulated echo's store-and-recall), which is a property of the schedule and not of
    # the codec this oracle checks
    expect = pathway_weight(seq) * w * np.exp(_dense_logweights(pack, seq, T2, T1, rho2))
    np.testing.assert_allclose(ew, expect, rtol=1e-6, atol=1e-14)          # the oracle decodes the bridge in float32


def test_walker_phases_is_the_signal_before_the_exponential(pack, field_pack):
    """``walker_phases`` gives ``(w, ew, phi)`` with ``exp(1j * phi)`` the ``E`` of ``walker_signals`` and the same
    weights, on the gradient route and on the field route: a consumer that sums many walkers over its own groups
    forms the exponential where it accumulates."""
    seq = _seqmod.pgse([[1, 0, 0], [0, 1, 1]], 1e-3, 3e-3, bvalues=[1e9, 5e8], TE=6e-3, n_t=4 * pack.n_t + 1, slew_rate=np.inf)
    t = Tissue(T2={"extra": 0.08, "intra": 0.03}, rho2=1e-5, D=D0)
    w, ew, phi = pack.walker_phases(seq, tissue=t); w2, ew2, E = pack.walker_signals(seq, tissue=t)
    assert phi.shape == E.shape == (pack.n_walkers, 2) and np.isrealobj(phi)
    np.testing.assert_array_equal(w, w2); np.testing.assert_array_equal(ew, ew2); np.testing.assert_allclose(np.exp(1j * phi), E, rtol=0, atol=1e-12)
    pk = field_pack
    seq = _seqmod.gre(6e-4, gradient_directions=[[1, 0, 0]], bvalues=[5e8], delta=1e-4, Delta=3e-4, n_t=4 * pk.n_t + 1, slew_rate=np.inf)
    tf = Tissue(chi_iso=-1e-7, chi_aniso=-5e-8)
    _, _, phi = pk.walker_phases(seq, scanner=3.0, tissue=tf); _, _, E = pk.walker_signals(seq, scanner=3.0, tissue=tf)
    np.testing.assert_allclose(np.exp(1j * phi), E, rtol=0, atol=1e-12)


def test_the_field_phase_equals_the_decoded_path_integral(field_pack):
    """The path route: the per-channel path integrals are the cosine modes contracted with the gate's cosine
    projection; the dense route (the 13 channels' mode series evaluated finely, the held gate integrated against
    it) is the oracle."""
    from dmipy_replay.replay.field import susc_path_coeffs, susc_path_field
    from dmipy_sim.replay.replay import GAMMA
    pk = field_pack
    seq = _seqmod.gre(6e-4, gradient_directions=[[1, 0, 0]], bvalues=[5e8], delta=1e-4, Delta=3e-4, n_t=4 * pk.n_t + 1, slew_rate=np.inf)
    b0_dir = (0.6, 0.0, 0.8)
    seq_lab, R = field_along(seq, b0_dir)
    w, ew, E = pk.walker_signals(seq_lab, orientation=R, scanner=3.0, tissue=Tissue(chi_iso=-1e-7, chi_aniso=-5e-8), tol=None)
    # the oracle
    ch = pk.meta["compression"]["channels"]; gm = ch["susceptibility_grid"]
    Cf, _ = susc_path_coeffs(pk.arrays, ch["susc_field_basis"])                            # (n_w, 13, K_f) the stored modes
    T = pk.T_window; K_f = Cf.shape[2]
    sub = 256
    tf = (np.arange((seq.G.shape[1] - 1) * sub) + 0.5) * float(seq.dt) / sub              # the held samples' pieces, finely
    Em = np.sqrt(2.0 / T) * np.cos(np.pi * np.outer(np.arange(K_f), tf) / T); Em[0] = 1.0 / np.sqrt(T)
    gate_f = np.asarray(seq.effective_gate, np.float64)[np.minimum((tf / float(seq.dt)).astype(int), seq.G.shape[1] - 1)]
    b = np.einsum("wck,kt->wct", Cf, Em)                                                 # the field channels along the window
    dB = susc_path_field(b, np.asarray(b0_dir) / np.linalg.norm(b0_dir), B0=3.0, chi_iso=-1e-7, chi_aniso=-5e-8, has_aniso=bool(gm.get("has_aniso")))
    phi_x = GAMMA * (float(seq.dt) / sub) * (dB * gate_f[None, :]).sum(1)
    # the gradient's phase: the held waveform against the stored path (the chord plus the bands' series), finely
    C = _r_compression.read_position_coeffs(pk.arrays, dtype=np.float64)
    G = np.asarray(seq.G_eff, np.float64)
    t = tf
    k = np.arange(1, C.shape[1] - 1)
    r = (C[:, None, 0, :] + C[:, None, 1, :] * (t / T)[None, :, None]
         + np.einsum("wkc,kt->wtc", C[:, 2:, :], np.sqrt(2.0 / T) * np.sin(np.pi * np.outer(k, t) / T)))
    g = G[:, np.minimum((t / float(seq.dt)).astype(int), G.shape[1] - 1), :]
    phi_g = GAMMA * (float(seq.dt) / sub) * np.einsum("mtc,wtc->wm", g, r)
    expect = np.exp(1j * (phi_g + phi_x[:, None]))
    np.testing.assert_allclose(E, expect, rtol=1e-7, atol=1e-7)


def test_no_route_decodes_a_track_or_a_trajectory(field_pack, monkeypatch):
    """With the path channel present, T2, rho2 and the field replay without reading any stored function at instants."""
    pk = field_pack
    for name in ("decode_occupancy", "local_time_at", "positions_at"):
        monkeypatch.setattr(_r_compression, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError(f"{name} was called")))
    monkeypatch.setattr(ReplayPack, "positions", lambda self: (_ for _ in ()).throw(AssertionError("positions() was decoded")))
    seq = _seqmod.pgse([[1, 0, 0]], 1e-4, 3e-4, bvalues=[5e8], TE=6e-4, n_t=4 * pk.n_t + 1, slew_rate=np.inf)
    pk.walker_signals(seq, tissue=Tissue(T2={"extra": 0.08, "intra": 0.03, "myelin": 0.01}, T1={"extra": 1.0, "intra": 1.2, "myelin": 0.3}))
    pk.walker_signals(seq, tissue=Tissue(rho2=1e-5))                 # rho2 over the WALK's D; another D is a rescale (#289)
    pk.walker_signals(seq, scanner=3.0, tissue=Tissue(chi_iso=-1e-7, chi_aniso=-5e-8))
    pk.walker_signals(seq, tissue=pk.nominal, scanner=3.0)
