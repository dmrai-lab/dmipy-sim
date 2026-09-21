"""Every replay term is a contraction in the space its channel is stored in (dmipy-sim#241).

The gradient row was the only one: the relaxation term decoded the occupancy runs to a track, the gated surface
term decoded the boundary bridge to a series, and the field term decoded the positions and the 64-mode path
channel to the save grid and integrated. Each is now a product with the stored coefficients -- runs, bridge
bands, cosine modes -- and equals the decoded computation to rounding; and no route through `walker_signals`
touches a decoded track or trajectory on a pack that carries the path channel.
"""
from __future__ import annotations

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences as _seqmod
from dmipy_sim.replay import ReplayPack
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay import compression as _cx
from dmipy_sim.replay._replay_kernel import bin_gate
from dmipy_sim.acquisition.epg import pathway_weight
from dmipy_sim.spec.tissue import Tissue
from tests.replay_frames import field_along

D0 = 2e-9


@pytest.fixture(scope="module")
def pack():
    walk = d.simulate_trajectories(300, D0, d.Cylinder(2e-6, (0, 0, 1)), 0.01, 5e-4, seed=0, require_gpu=False)
    return build_replay_pack(walk, id="test/full", K=8, license="x", citation="x")


@pytest.fixture(scope="module")
def field_pack(tmp_path_factory):
    """A strand pack with the path channel (C3), from the three-strand fixture with a sheath."""
    from dmipy_sim.io.strands import write_tck
    from dmipy_sim.spec import disco_spec, walk_spec
    tmp = tmp_path_factory.mktemp("field")
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp / "t.tck"), str(tmp / "d.txt")
    write_tck(tck, cls_, coordinate_unit_m=25e-6); np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    spec = disco_spec(tck, dia, side_m=20e-6)
    w = walk_spec(spec, 90, 8e-4, 2e-4, seed=0, n_probe=20_000, field_res=0.5e-6, require_gpu=False)
    return build_replay_pack(w, id="test/field", license="x", citation="x", K=4, susc_path_K=4)


def _dense_logweights(pk, seq, T2, T1, rho):
    """The decoded computation of the relaxation and surface log-weights: the oracle."""
    ch = pk.meta["compression"]["channels"]
    n_t, dt = pk.n_t, pk.dt
    G = np.asarray(seq.G); chi = np.ones(G.shape[1]) if seq.chi_perp is None else np.asarray(seq.chi_perp).reshape(-1)
    active = bin_gate(np.ones(chi.shape[0]), seq.dt, n_t, dt)[0]; chi = bin_gate(chi, seq.dt, n_t, dt)[0]
    comp = _cx.decode_occupancy(pk.arrays, ch["compartment"])["comp"]
    lw = _cx.relaxation_logweight(comp, T2, T1, dt, chi, active)
    meta = dict(ch["boundary_local_time"]); meta.setdefault("n_t", n_t)
    ell = np.asarray(_cx.decode_boundary_bridge(pk.arrays, meta), np.float64)
    lw = lw + _cx.surface_logweight_series(ell, rho / D0, chi)
    return lw


@pytest.mark.parametrize("make_seq", [
    lambda n_t: _seqmod.gre(4e-3, n_t=n_t),                                                   # an FID shorter than the walk
    lambda n_t: _seqmod.pgse([[1, 0, 0]], 1e-3, 3e-3, bvalues=[1e9], TE=6e-3, n_t=n_t, slew_rate=np.inf),
    lambda n_t: _seqmod.pgste([[1, 0, 0]], 1e-3, 3e-3, bvalues=[1e9], TE=8e-3, n_t=n_t, slew_rate=np.inf),   # a stored period
], ids=["fid", "pgse", "pgste"])
def test_relaxation_and_surface_weights_equal_the_decoded_ones(pack, make_seq):
    seq = make_seq(4 * pack.n_t + 1)
    T2, T1, rho = [0.08, 0.03], [1.0, 1.2], 1e-5
    w, ew, _ = pack.walker_signals(seq, tissue=Tissue(T2=T2, T1=T1, rho=rho, D=D0))
    # the route's weights also carry the amplitude of the pathway the readout is (1 for the fid and the spin
    # echo, 0.5 for the stimulated echo's store-and-recall), which is a property of the schedule and not of
    # the codec this oracle checks
    expect = pathway_weight(seq) * w * np.exp(_dense_logweights(pack, seq, T2, T1, rho))
    np.testing.assert_allclose(ew, expect, rtol=1e-6, atol=1e-14)          # the oracle decodes the bridge in float32


def test_walker_phases_is_the_signal_before_the_exponential(pack, field_pack):
    """``walker_phases`` gives ``(w, ew, phi)`` with ``exp(1j * phi)`` the ``E`` of ``walker_signals`` and the same
    weights, on the gradient route and on the field route: a consumer that sums many walkers over its own groups
    forms the exponential where it accumulates."""
    seq = _seqmod.pgse([[1, 0, 0], [0, 1, 1]], 1e-3, 3e-3, bvalues=[1e9, 5e8], TE=6e-3, n_t=4 * pack.n_t + 1, slew_rate=np.inf)
    t = Tissue(T2=[0.08, 0.03], rho=1e-5, D=D0)
    w, ew, phi = pack.walker_phases(seq, tissue=t); w2, ew2, E = pack.walker_signals(seq, tissue=t)
    assert phi.shape == E.shape == (pack.n_walkers, 2) and np.isrealobj(phi)
    np.testing.assert_array_equal(w, w2); np.testing.assert_array_equal(ew, ew2); np.testing.assert_allclose(np.exp(1j * phi), E, rtol=0, atol=1e-12)
    pk = field_pack
    seq = _seqmod.gre(6e-4, gradient_directions=[[1, 0, 0]], bvalues=[5e8], delta=1e-4, Delta=3e-4, n_t=4 * pk.n_t + 1, slew_rate=np.inf)
    tf = Tissue(chi_iso=-1e-7, chi_aniso=-5e-8)
    _, _, phi = pk.walker_phases(seq, scanner=3.0, tissue=tf); _, _, E = pk.walker_signals(seq, scanner=3.0, tissue=tf)
    np.testing.assert_allclose(np.exp(1j * phi), E, rtol=0, atol=1e-12)


def test_the_field_phase_equals_the_decoded_path_integral(field_pack):
    """The path route: the per-channel path integrals are the cosine modes contracted with the gate's DCT; the
    decoded route (positions, the 13 channels on the save grid, the gate summed) is the oracle."""
    from dmipy_sim.replay.bank import susc_path_decode, susc_path_field
    from dmipy_sim.replay._replay_kernel import field_gate, gradient_phase, effective_gradient
    from dmipy_sim.replay.replay import GAMMA
    pk = field_pack
    seq = _seqmod.gre(6e-4, gradient_directions=[[1, 0, 0]], bvalues=[5e8], delta=1e-4, Delta=3e-4, n_t=4 * pk.n_t + 1, slew_rate=np.inf)
    b0_dir = (0.6, 0.0, 0.8)
    seq_lab, R = field_along(seq, b0_dir)
    w, ew, E = pk.walker_signals(seq_lab, orientation=R, scanner=3.0, tissue=Tissue(chi_iso=-1e-7, chi_aniso=-5e-8))
    # the oracle
    ch = pk.meta["compression"]["channels"]; gm = ch["susceptibility_grid"]
    b, _ = susc_path_decode(pk.arrays, ch["susceptibility_path"], n_w=pk.n_walkers)
    R = np.asarray(pk.pose_rotation(None), float) if False else np.eye(3)
    dB = susc_path_field(b, np.asarray(b0_dir) / np.linalg.norm(b0_dir), B0=3.0, chi_iso=-1e-7, chi_aniso=-5e-8, has_aniso=bool(gm.get("has_aniso")))
    phi_x = GAMMA * pk.dt * (dB * field_gate(seq, pk.n_t, pk.dt)[None, :]).sum(1)
    Geff = effective_gradient(np.asarray(seq.G_eff, np.float64), float(seq.dt), pk.n_t, pk.dt)
    phi_g = gradient_phase(Geff, pk.positions(), pk.dt).T
    expect = np.exp(1j * (phi_g + phi_x[:, None]))
    np.testing.assert_allclose(E, expect, rtol=1e-7, atol=1e-7)


def test_no_route_decodes_a_track_or_a_trajectory(field_pack, monkeypatch):
    """With the path channel present, T2, rho and the field replay without decoding anything to the save grid."""
    pk = field_pack
    for name in ("decode_occupancy", "decode_boundary_bridge"):
        monkeypatch.setattr(_cx, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError(f"{name} was called")))
    monkeypatch.setattr(ReplayPack, "positions", lambda self: (_ for _ in ()).throw(AssertionError("positions() was decoded")))
    seq = _seqmod.pgse([[1, 0, 0]], 1e-4, 3e-4, bvalues=[5e8], TE=6e-4, n_t=4 * pk.n_t + 1, slew_rate=np.inf)
    pk.walker_signals(seq, tissue=Tissue(T2={"extra": 0.08, "intra": 0.03, "myelin": 0.01}, T1={"extra": 1.0, "intra": 1.2, "myelin": 0.3}))
    pk.walker_signals(seq, tissue=Tissue(rho=1e-5))                 # rho over the WALK's D; another D is a rescale (#289)
    pk.walker_signals(seq, scanner=3.0, tissue=Tissue(chi_iso=-1e-7, chi_aniso=-5e-8))
    pk.walker_signals(seq, tissue=pk.nominal, scanner=3.0)
