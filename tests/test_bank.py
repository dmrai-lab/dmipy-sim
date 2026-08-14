"""Replay-pack assembler (dmipy_sim.bank): build_replay_pack + build_to_floor.

Produces a self-certifying .rpk from a master walk, on a SYNTHETIC master (reflecting-slab
random walk built in numpy — no simulator, fast tier). Checks: the pack compresses within the
MC floor, carries the requested tiers, round-trips through .rpk, is consumable by the lean
replay-signal forward (the fit/design path), and the floor-target policy converges. See the
end-to-end walk->pack->replay validation in test_replay_parity for the physics parity.
"""
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_sim import (build_replay_pack, build_to_floor, read_rpk,
                       compile_scheme, replay_signal)
from dmipy_sim import bank
from dmipy_sim.constants import GAMMA

N_W, N_T, DT, D0, L = 3000, 200, 5e-4, 2e-9, 6e-6


def _slab_master(n_w=N_W, seed=0):
    """A reflecting-slab (0<=x<=L) + free y,z random walk -> master-walk dict with a
    boundary-local-time channel (per-step wall contact), the shape build_replay_pack wants."""
    rng = np.random.default_rng(seed)
    step = np.sqrt(2 * D0 * DT)
    x = rng.uniform(0, L, n_w)
    traj = np.zeros((n_w, N_T, 3)); dlog = np.zeros((n_w, N_T))
    for t in range(N_T):
        x = x + rng.normal(0, step, n_w)
        hit_lo = x < 0; hit_hi = x > L
        x = np.where(hit_lo, -x, np.where(hit_hi, 2 * L - x, x))
        traj[:, t, 0] = x
        dlog[:, t] = (hit_lo | hit_hi) * step        # crude per-step contact (>=0, mostly zero)
    traj[:, :, 1:] = np.cumsum(rng.normal(0, step, (n_w, N_T, 2)), axis=1)
    return dict(traj=traj, dt_traj=DT, T_max=(N_T - 1) * DT,
                comp=np.zeros((n_w, N_T), np.int8), comp0=np.zeros(n_w, np.int64),
                w=np.ones(n_w), T2_per_comp=np.array([0.08]), T1_per_comp=np.array([1.0]),
                dlog_b=dlog, D_intra=D0, n_walkers=n_w, seed=seed)


def _lean_env():
    return dict(bvals=[0.0, 1e9, 3e9], dirs=[[1, 0, 0], [0, 0, 1]], ogse_periods=[2],
                shortd_b=1e9, shortd_deltas_frac=[0.05], B0_list=[], theta_deg=[0],
                delta_frac=0.2, Delta_frac=0.5, rho_list=[1e-5, 1e-4])


@pytest.fixture(scope="module")
def pack():
    return build_replay_pack(_slab_master(), id="test/slab", method="temporal_dct",
                             envelope=_lean_env(), K=64, surface_relaxivity=True,
                             license="CC-BY-4.0", citation="test")


def test_pack_compresses_within_floor_and_declares_tiers(pack):
    f = pack.fidelity
    assert f["err_max"] <= 2.0 * f["floor_max"]                 # codec loss below the MC floor
    assert f["within_2x_floor"] is True
    assert "err_surface" in f and f["err_surface"] <= 2.0 * f["floor_surface"] + 1e-9
    env = pack.replay_envelope
    assert env["gradient"] and env["bulk_relaxation"] and env["surface_relaxivity"]
    assert not env["field"] and not env["magnetization_transfer"]
    assert pack.method == "temporal_dct" and pack.license == "CC-BY-4.0"
    # a temporal_dct pack carries its positions as one tensor per spatial axis (pos_x/pos_y/pos_z), so
    # an axis subset is a contiguous read that composes with the walker-prefix read; `.dct_coeffs`
    # assembles them for consumers that want all three.
    from dmipy_sim.compression import POSITION_AXES, has_axis_layout
    assert has_axis_layout(pack.arrays) and all(k in pack.arrays for k in POSITION_AXES)
    assert np.asarray(pack.dct_coeffs).shape[2] == 3
    assert any(k.startswith("comp_rle") for k in pack.arrays)   # compartment tier
    assert any(k.startswith("blt_") for k in pack.arrays)       # surface tier


def test_rpk_roundtrip_and_lean_consumption(pack, tmp_path):
    out = tmp_path / "slab.rpk"
    build_replay_pack(_slab_master(), id="test/slab", method="temporal_dct", envelope=_lean_env(),
                      K=64, surface_relaxivity=True, license="CC-BY-4.0", citation="test",
                      out_path=str(out))
    p2 = read_rpk(out)
    npt.assert_allclose(p2.dct_coeffs, pack.dct_coeffs, rtol=0, atol=0)
    # consume through the lean compiled-scheme forward (the fit/design path)
    nt, dt = p2.n_t, p2.dt
    nd, ng = int(0.2 * (nt - 1)), int(0.5 * (nt - 1))
    bu = (GAMMA * nd * dt) ** 2 * ((ng - nd / 3) * dt)
    bs = np.array([0.0, 0.5e9, 1e9, 2e9])
    G = np.zeros((len(bs), nt, 3))
    for i, b in enumerate(bs):
        a = np.sqrt(b / bu); G[i, :nd, 0] = a; G[i, ng:ng + nd, 0] = -a   # perpendicular (restricted)
    E = replay_signal(p2, compile_scheme(G, dt, p2.K))
    assert abs(E[0] - 1.0) < 1e-6                                # b=0 -> 1
    assert np.all(np.diff(E) <= 1e-6)                            # monotone non-increasing


def test_susceptibility_master_is_rejected():
    m = _slab_master(); m["PhiC"] = np.zeros((5, 4, 4))
    with pytest.raises(NotImplementedError, match="susceptibility"):
        build_replay_pack(m, id="x", method="temporal_dct", K=8, license="x", citation="x")


def test_build_to_floor_converges_and_records_target():
    pk = build_to_floor(lambda n: _slab_master(n_w=n, seed=1), id="test/floor",
                        envelope=_lean_env(), method="temporal_dct", K=48, sigma_star=0.05,
                        pilot_n=800, max_n=6000, surface_relaxivity=False,
                        license="CC-BY-4.0", citation="test", verbose=False)
    assert pk.fidelity.get("target_floor") == 0.05
    assert "meets_target" in pk.fidelity


# ------------------------------------------------------------------ susceptibility path channel (C3)
def _field_basis_for_slab(shape=(24, 24, 24), seed=3):
    """A smooth synthetic field basis on a grid spanning the slab walk.

    iso_P is built so the exact trace identity P_xx+P_yy+P_zz == 3*iso_local holds (as it does for an
    un-apodised dipole solve), which is what lets the encoder drop iso_P_zz.
    """
    rng = np.random.default_rng(seed)
    lo, hi = -8e-6, 14e-6
    vs = np.array([(hi - lo) / (s - 1) for s in shape])
    g = lambda: np.cumsum(np.cumsum(rng.normal(0, 1, shape), axis=0), axis=1) * 1e-3
    iso_local = g()
    P = np.stack([g() for _ in range(6)])
    P[2] = 3.0 * iso_local - P[0] - P[1]                 # enforce the identity exactly
    return dict(iso_local=iso_local, iso_P=P, aniso_G=None, shape=tuple(shape),
                voxel_size=vs), np.array([lo, lo, lo])


def _susc_master(**kw):
    m = _slab_master(**kw)
    fb, origin = _field_basis_for_slab()
    m.update(susc_field_basis=fb, susc_grid_origin=origin, susc_chi_iso=1.06e-6, delta_chi_a=0.0)
    return m


def test_susc_path_channel_drops_zz_via_trace_identity_and_certifies_at_its_capability():
    K = 32
    env = dict(_lean_env(), B0_list=[7.0], theta_deg=[0, 90])
    pk = build_replay_pack(_susc_master(), id="test/slab-susc", method="temporal_dct",
                          envelope=env, K=64, susc_path_K=K,
                          license="CC-BY-4.0", citation="test")
    pm = pk.meta["compression"]["channels"]["susceptibility_path"]
    # the identity holds for this basis, so zz is implied -> 6 stored channels, not 7
    assert pm["iso_P_zz"] == "implied" and pm["n_ch"] == 6
    assert pm["max_refocus_pulses"] == K // 2
    assert pk.arrays["susc_path_dct"].dtype == np.float16
    # decode restores all 7 channels in canonical order
    b, names = bank.susc_path_decode(pk.arrays, pm)
    assert names[:4] == ["iso_local", "iso_P_xx", "iso_P_yy", "iso_P_zz"] and b.shape[1] == 7
    # certified at the CPMG train it advertises, not merely at spin echo
    f = pk.fidelity
    assert f["susc_path_pulses_certified"] == K // 2
    assert f["err_susc_path"] <= 2.0 * f["floor_susc_path"] + 1e-9


def test_susc_path_keeps_zz_when_trace_identity_is_broken():
    """A k-space window applied to iso_P but not iso_local breaks the identity; the encoder must
    detect that and store all six iso_P rather than reconstruct a wrong zz."""
    m = _susc_master()
    m["susc_field_basis"]["iso_P"] = m["susc_field_basis"]["iso_P"] * 1.05      # break the trace
    pk = build_replay_pack(m, id="test/slab-susc-broken", method="temporal_dct",
                           envelope=dict(_lean_env(), B0_list=[7.0], theta_deg=[0]),
                           K=64, susc_path_K=16, license="CC-BY-4.0", citation="test")
    pm = pk.meta["compression"]["channels"]["susceptibility_path"]
    assert pm["iso_P_zz"] == "stored" and pm["n_ch"] == 7
    assert pm["trace_residual"] > 1e-6
    b, names = bank.susc_path_decode(pk.arrays, pm)
    assert b.shape[1] == 7 and names.count("iso_P_zz") == 1


def test_susc_path_lossless_at_K_equals_nt():
    """At K=n_t the channel is an exact rewrite of the field along the path (no truncation)."""
    from dmipy_sim.susceptibility_field import assemble_field, sample_grid
    m = _susc_master()
    fb, origin = m["susc_field_basis"], m["susc_grid_origin"]
    traj = m["traj"][:200]
    a, pm = bank.susc_path_encode(fb, traj, origin, K=N_T, dtype=np.float64)
    b, _ = bank.susc_path_decode(a, pm)
    d = [1.0, 0.0, 0.0]
    ref = sample_grid(assemble_field(fb, d, B0=7.0, chi_iso=1.06e-6), traj, origin,
                      fb["voxel_size"], periodic=False)
    got = bank.susc_path_field(b, d, B0=7.0, chi_iso=1.06e-6)
    npt.assert_allclose(got, ref, rtol=0, atol=1e-12 * np.max(np.abs(ref)))


def test_path_pack_with_lossy_positions_omits_the_grid_arrays_and_replays_via_path():
    """The grid route samples DECODED positions, so it is unsound once the position codec is lossy.
    A path pack with lossy positions must therefore not ship grid arrays offering that route."""
    pk = build_replay_pack(_susc_master(), id="test/slab-path-only", method="temporal_dct",
                           envelope=dict(_lean_env(), B0_list=[7.0], theta_deg=[0]),
                           K=64, susc_path_K=32, license="CC-BY-4.0", citation="test")
    gm = pk.meta["compression"]["channels"]["susceptibility_grid"]
    assert gm["arrays_in_pack"] is False and gm["replay_route"] == "path"
    assert "susc_grid_iso_local" not in pk.arrays and "susc_grid_iso_P" not in pk.arrays
    # and it still replays: susceptibility off vs on must differ, and chi=0 must match the grid-free case
    class _W:
        G = np.zeros((1, N_T, 3)); dt = DT
    kw = dict(b0_dir=[1, 0, 0], B0=7.0, relaxation=False)   # relaxation off: isolate the field
    s0 = bank.replay_susc(pk, _W(), chi_iso=0.0, **kw)
    s1 = bank.replay_susc(pk, _W(), chi_iso=1.06e-6, **kw)
    npt.assert_allclose(s0, 1.0, atol=1e-12)          # no gradient, no field -> nothing to dephase
    assert s1 < s0                                     # field present -> dephasing


def test_lossless_positions_may_carry_both_routes():
    pk = build_replay_pack(_susc_master(), id="test/slab-both", method="temporal_dct",
                           envelope=dict(_lean_env(), B0_list=[7.0], theta_deg=[0]),
                           K=N_T, susc_path_K=32, license="CC-BY-4.0", citation="test")
    gm = pk.meta["compression"]["channels"]["susceptibility_grid"]
    assert gm["arrays_in_pack"] is True and gm["replay_route"] == "grid+path"


def test_c2_codec_is_chosen_by_cost_not_left_to_the_expensive_default():
    """Without an explicit blt_temporal_K the pack must still get a cheap boundary channel: the
    selector costs DCT candidates against the surface gate instead of falling back to sparse CSR,
    whose size grows with walk length."""
    pk = build_replay_pack(_slab_master(), id="test/slab-c2-auto", method="temporal_dct",
                           envelope=_lean_env(), K=64, surface_relaxivity=True,
                           license="CC-BY-4.0", citation="test")
    cm = pk.meta["compression"]["channels"]["boundary_local_time"]
    assert cm["mode"] == "dct" and cm["dtype"] == "float16"
    per_walker = sum(np.asarray(pk.arrays[k]).nbytes for k in ("blt_dct_coeffs", "blt_endpoint")) / N_W
    assert per_walker < 300                     # sparse CSR on this walk costs far more
    assert pk.fidelity["err_surface"] <= 2.0 * pk.fidelity["floor_surface"] + 1e-9


def test_precision_tiers_declare_shuffle_and_account_for_non_sliceable_arrays():
    m = _slab_master()
    m["walkers_shuffled"] = True
    pk = build_replay_pack(m, id="test/slab-tiers", method="temporal_dct", envelope=_lean_env(),
                           K=64, surface_relaxivity=True, license="CC-BY-4.0", citation="test")
    pt = pk.meta["compression"]["precision_tiers"]
    assert pt["usable"] is True and pt["walkers_shuffled"] is True
    assert pt["bytes_per_walker"] > 0
    # a tier must never be quoted as smaller than the arrays that cannot be prefix-sliced
    for t in pt["tiers"]:
        assert t["bytes"] >= pt["fixed_bytes"]
        assert 1 <= t["n_walkers"] <= N_W
    # coarser eps must need strictly fewer walkers than finer eps
    ns = [t["n_walkers"] for t in pt["tiers"]]
    assert ns == sorted(ns)


def test_precision_tiers_flag_unshuffled_packs_as_unusable():
    pk = build_replay_pack(_slab_master(), id="test/slab-noshuf", method="temporal_dct",
                           envelope=_lean_env(), K=64, license="CC-BY-4.0", citation="test")
    pt = pk.meta["compression"]["precision_tiers"]
    assert pt["usable"] is False and "NOT DECLARED SHUFFLED" in pt["note"]
