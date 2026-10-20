"""Replay-pack assembler (dmipy_sim.replay.bank): build_replay_pack + build_to_floor.

Produces a self-certifying .rpk from a coefficient walk of a reflecting slab (``Box1D``, walked by the engine on the
CPU at a few hundred walkers, fast tier). Checks: the pack carries the walk-free certificate and the requested
tiers, round-trips through .rpk, is consumable by the lean replay-signal forward (the fit/design path), and the
floor-target policy converges. See the end-to-end walk->pack->replay validation in test_replay_parity for the
physics parity.
"""
import dataclasses
import json
import os

import numpy as np
from dmipy_sim.fields.susceptibility_field import FieldGrid
import numpy.testing as npt
import pytest

from dmipy_sim import (build_replay_pack, build_to_floor, read_rpk,
                       compile_scheme, replay_signal)
from dmipy_sim.replay import bank
from dmipy_sim.constants import GAMMA
from dmipy_sim.spec.tissue import Tissue
from dmipy_sim.spec.substrate import Domain, Pool, Seeding, SubstrateSpec, Susceptibility, Validity
from tests.replay_frames import field_along

N_W, N_T, DT, D0, L = 600, 41, 5e-4, 2e-9, 6e-6
T_WALK = (N_T - 1) * DT                      # 20 ms of walk in blocks of 0.5 ms


def _slab_spec(magnetic=False, length=L, D=D0):
    """The slab walk's substrate: a reflecting slab 0 <= x <= ``length``, free in y and z, its walkers in pool 0 (a
    test relabels some into pool 1); pool 0 magnetic, with the field tier, for a walk that carries a field basis."""
    chi = Susceptibility(None, None, "none") if magnetic else None
    return SubstrateSpec("test/slab", Domain([0.0, -1e-3, -1e-3], [length, 1e-3, 1e-3], ["reflect", "open", "open"]),
                         [Pool(0, "extra", D, susceptibility=chi), Pool(1, "intra", D)], [], Seeding([0]),
                         Validity(length, ["gradient", "relaxation"] + (["field"] if magnetic else []))).validate().to_dict()


def _slab_walk(n_w=N_W, seed=0, *, T=T_WALK, block_h=DT, K=None, magnetic=False, field_basis=None, raw=True, **kw):
    """The reflecting slab (0 <= x <= L, free in y and z) walked by the engine: a coefficient walk with the contact
    channel (C2), the static occupancy (every walker in pool 0) and, with ``field_basis``, the field modes (C3);
    the raw subset is every walker (the dense oracle of a measured certificate). The spec is :func:`_slab_spec`
    (two named pools, so a test can relabel walkers into pool 1)."""
    import dmipy_sim as d
    from dmipy_sim.spec.substrate import SubstrateSpec
    w = d.simulate_trajectories(int(n_w), D0, d.Box1D(L), T, block_h, seed=int(seed), require_gpu=False, K=K,
                                raw_subset=(int(n_w) if raw else 0), field_basis=field_basis, window_T=T, **kw)
    return dataclasses.replace(w, spec=SubstrateSpec.from_dict(_slab_spec(magnetic=magnetic or field_basis is not None)))


def _lean_env():
    return dict(bvals=[0.0, 1e9, 3e9], dirs=[[1, 0, 0], [0, 0, 1]], ogse_periods=[2],
                shortd_b=1e9, shortd_deltas_frac=[0.05], B0_list=[], theta_deg=[0],
                delta_frac=0.2, Delta_frac=0.5, rho2_list=[1e-5, 1e-4])


@pytest.fixture(scope="module")
def pack():
    return build_replay_pack(_slab_walk(), id="test/slab", envelope=_lean_env(), K=16, fidelity="measured",
                             license="CC-BY-4.0", citation="test")


def test_pack_compresses_within_floor_and_declares_tiers(pack):
    f = pack.fidelity
    assert f["err_max"] <= 2.0 * f["floor_max"]                 # codec loss below the MC floor
    assert f["within_2x_floor"] is True
    assert "err_surface" in f and f["err_surface"] <= 2.0 * f["floor_surface"] + 1e-9
    assert f["certified"] == "measured" and set(f["terms"]) >= {"truncation", "quadrature", "rounding"}
    env = pack.replay_envelope
    assert env["gradient"] and env["bulk_relaxation"] and env["surface_relaxivity"]
    assert not env["field"] and not env["magnetization_transfer"]
    assert pack.method == "bridge_dst" and pack.license == "CC-BY-4.0"
    # a pack carries its positions as one tensor per spatial axis (pos_x/pos_y/pos_z), so
    # an axis subset is a contiguous read that composes with the walker-prefix read; `.position_coeffs`
    # assembles them for consumers that want all three.
    from dmipy_sim.replay.compression import POSITION_AXES, has_axis_layout
    assert has_axis_layout(pack.arrays) and (all(k in pack.arrays for k in POSITION_AXES) or "pos_x_ends" in pack.arrays)   # float or integer container
    assert np.asarray(pack.position_coeffs).shape[2] == 3
    from dmipy_sim.replay.compression import has_c1, has_c2
    assert has_c1(pack.arrays)                                  # compartment tier
    assert "comp_static" in pack.arrays                         # impermeable: one label per walker
    assert has_c2(pack.arrays)                                  # surface tier
    # no save grid anywhere in the header: the block and the bands are the walk
    wp = pack.meta["walk_params"]
    assert "n_t" not in wp and "dt_save" not in wp and "dt_traj" not in wp and wp["block"]["h"] == pytest.approx(DT)
    assert pack.n_t == N_T and pack.dt == pytest.approx(DT)      # derived from the block for a reader's sampling


def test_rpk_roundtrip_and_lean_consumption(pack, tmp_path):
    out = tmp_path / "slab.rpk"
    build_replay_pack(_slab_walk(), id="test/slab", envelope=_lean_env(), K=16, fidelity="measured",
                      license="CC-BY-4.0", citation="test", out_path=str(out))
    p2 = read_rpk(out)
    npt.assert_allclose(p2.position_coeffs, pack.position_coeffs, rtol=0, atol=0)
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


def test_build_to_floor_converges_and_records_target():
    pk = build_to_floor(lambda n: _slab_walk(n_w=n, seed=1, raw=False), id="test/floor",
                        envelope=_lean_env(), K=16, sigma_star=0.05,
                        pilot_n=300, max_n=2000,
                        license="CC-BY-4.0", citation="test", verbose=False)
    assert pk.fidelity.get("target_floor") == 0.05
    assert "meets_target" in pk.fidelity


def test_build_to_floor_keeps_the_pilot_walk_when_it_meets_the_floor():
    """A pilot whose floor is already under sigma* is the walk: the builder asks for no second one."""
    calls = []

    def make(n):
        calls.append(n)
        return _slab_walk(n_w=n, seed=1, raw=False)

    pk = build_to_floor(make, id="test/floor-pilot", envelope=_lean_env(), K=16, sigma_star=10.0,
                        pilot_n=300, max_n=2000, license="CC-BY-4.0", citation="test", verbose=False)
    assert calls == [300] and pk.n_walkers == 300


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


def _slab_grid(seed=3):
    """The synthetic basis as the :class:`FieldGrid` a walk reads (non-periodic, clamped at the edges)."""
    fb, origin = _field_basis_for_slab(seed=seed)
    return FieldGrid(fb, origin)


def _susc_walk(n_w=N_W, seed=0, **kw):
    """The slab walk reading the synthetic field basis in the walk (C3 on the cosine modes), its spec magnetic."""
    return _slab_walk(n_w, seed, field_basis=_slab_grid(), magnetic=True, **kw)


def _env7(**kw):
    env = dict(_lean_env(), B0_list=[7.0], theta_deg=[0]); env.update(kw)
    return env


def test_susc_path_channel_drops_zz_via_trace_identity_and_certifies_at_its_capability():
    """The pack of a walk that read the field: the modes accumulated IN the walk on the window's cosine basis, the
    band fixed by the producer's pilot, ``iso_P_zz`` implied by the trace identity, no save grid in the descriptor,
    and the measured certificate at the CPMG depth the band serves."""
    w = _susc_walk(300, seed=2)
    assert w.has_field and w.field_coeffs.shape == (300, 1, w.K_c3, 7) and w.block["field_band"]["rule"] == "pilot"
    pk = build_replay_pack(w, id="test/slab-susc", envelope=_env7(), K=16, fidelity="measured",
                           license="CC-BY-4.0", citation="test")
    pm = pk.meta["compression"]["channels"]["susc_field_basis"]
    K = pm["K"]
    assert K == w.K_c3 and pm["T"] == pytest.approx(T_WALK) and "n_t" not in pm and "dt" not in pm
    assert pm["iso_P_zz"] == "implied" and pm["n_ch"] == 6 and pm["sampling"] == "accumulated_in_walk"
    assert pm["max_refocus_pulses"] == K // 2 and pm["read_interval"] == pytest.approx(w.field_read_interval)
    # the pilot's depth (8 bits) over the modes above k = 0; the k = 0 mode (the window's integral) at 16 bits
    assert w.block["field_band"]["bits"] == 8 and pm["container"] == [{"modes": [0, 1], "bits": 16}, {"modes": [1, K], "bits": 8}]
    assert pk.arrays["susc_path_dct_b0"].dtype == np.int16 and pk.arrays["susc_path_dct_b1"].dtype == np.int8
    assert pk.arrays["susc_path_scale"].shape == (pm["n_ch"], K)
    assert pk.meta["walk_params"]["field_read_interval"] == pytest.approx(w.field_read_interval)
    assert "field" in pk.meta["walk_params"]["qv"] and "field" in pk.fidelity["terms"]["quadrature"] and "field_read" in pk.fidelity["terms"]
    # decode restores all 7 channels in canonical order, on a reader's sampling of the window
    b, names = bank.susc_path_decode(pk.arrays, pm, n_t=N_T)
    assert names[:4] == ["iso_local", "iso_P_xx", "iso_P_yy", "iso_P_zz"] and b.shape == (300, 7, N_T)
    f = pk.fidelity
    assert f["susc_path_pulses_certified"] == K // 2
    assert f["err_susc_path"] <= 2.0 * f["floor_susc_path"] + 1e-9
    assert f["err_susc_bloch"] <= 2.0 * f["floor_susc_bloch"] + 1e-9


def test_the_modes_are_the_block_means_cosine_transform():
    """The accumulated modes against the raw subset's block means: ``c_k = sqrt(h) X_k`` with ``X`` the orthonormal
    DCT-II of the means (the block-mean quadrature of the trapezoid reads), the k = 0 mode the window integral."""
    from scipy.fft import dct
    w = _susc_walk(120, seed=5)
    from dmipy_sim.replay.compression import quadrature_rel
    means = np.asarray(w.raw_field, np.float64)[:, 1:, :]                                  # (n_w, n_blocks, n_ch)
    X = dct(means, type=2, norm="ortho", axis=1)[:, :w.K_c3, :] * np.sqrt(w.h)
    c = np.asarray(w.field_coeffs, np.float64)[:, 0]
    scale = np.abs(X).max(axis=(0, 1))
    assert (np.abs(c[:, 0] - X[:, 0]).max(axis=0) <= 1e-6 * scale).all()                  # the k = 0 mode: the window integral
    # the DCT of the means is the ORDER-0 reading of the modes (a block's reads at its centre); the walk's order-m
    # moments are within that reading's own quadrature error of it, mode by mode (``quadrature_rel`` is a variance
    # ratio: the amplitude's bound is its square root)
    rel = np.abs(c - X).max(axis=(0, 2)) / scale.max()
    bound = np.sqrt(quadrature_rel(0, w.h, np.pi * np.arange(w.K_c3) / w.T))
    assert (rel[1:] <= bound[1:] + 1e-5).all(), (rel, bound)


def test_susc_path_keeps_zz_when_trace_identity_is_broken():
    """A k-space window applied to iso_P but not iso_local breaks the identity; the encoder must
    detect that and store all six iso_P rather than reconstruct a wrong zz."""
    fb, origin = _field_basis_for_slab()
    fb = dict(fb, iso_P=fb["iso_P"] * 1.05)                                 # break the trace
    w = _slab_walk(200, seed=3, field_basis=FieldGrid(fb, origin), magnetic=True)
    pk = build_replay_pack(w, id="test/slab-susc-broken", envelope=_env7(), K=16, license="CC-BY-4.0", citation="test")
    pm = pk.meta["compression"]["channels"]["susc_field_basis"]
    assert pm["iso_P_zz"] == "stored" and pm["n_ch"] == 7
    assert pm["trace_residual"] > 1e-6
    b, names = bank.susc_path_decode(pk.arrays, pm, n_t=N_T)
    assert b.shape[1] == 7 and names.count("iso_P_zz") == 1


def test_path_pack_with_lossy_positions_omits_the_grid_arrays_and_replays_via_path():
    """The grid route samples DECODED positions, so it is unsound once the position codec is lossy.
    A path pack with lossy positions must therefore not ship grid arrays offering that route."""
    pk = build_replay_pack(_susc_walk(200, seed=4), id="test/slab-path-only", envelope=_env7(), K=16,
                           license="CC-BY-4.0", citation="test")
    gm = pk.meta["compression"]["channels"]["susceptibility_grid"]
    assert gm["arrays_in_pack"] is False and gm["replay_route"] == "path"
    assert "susc_grid_iso_local" not in pk.arrays and "susc_grid_iso_P" not in pk.arrays
    # and it still replays: susceptibility off vs on must differ, and chi=0 must match the grid-free case
    from dmipy_sim import ScannerSequence
    W = ScannerSequence(G=np.zeros((1, N_T, 3)), dt=DT)     # a bare gradient echo: no pulse folded
    W_lab, R = field_along(W, (1, 0, 0))                    # no T2 given: isolate the field
    s0 = pk.replay(W_lab, orientation=R, scanner=7.0, tissue=Tissue(chi_iso=0.0))
    s1 = pk.replay(W_lab, orientation=R, scanner=7.0, tissue=Tissue(chi_iso=1.06e-6))
    npt.assert_allclose(s0, 1.0, atol=1e-12)          # no gradient, no field -> nothing to dephase
    assert s1 < s0                                     # field present -> dephasing


def test_precision_tiers_declare_shuffle_and_account_for_non_sliceable_arrays():
    pk = build_replay_pack(_slab_walk(), id="test/slab-tiers", envelope=_lean_env(), K=16, license="CC-BY-4.0", citation="test")
    pt = pk.meta["compression"]["precision_tiers"]
    assert pt["usable"] is True and pt["walkers_shuffled"] is True       # the producer's walkers are i.i.d.
    assert pt["bytes_per_walker"] > 0
    # a tier must never be quoted as smaller than the arrays that cannot be prefix-sliced
    for t in pt["tiers"]:
        assert t["bytes"] >= pt["fixed_bytes"]
        assert 1 <= t["n_walkers"] <= N_W
    # coarser eps must need strictly fewer walkers than finer eps
    ns = [t["n_walkers"] for t in pt["tiers"]]
    assert ns == sorted(ns)


def test_a_field_replay_can_restrict_to_one_compartment():
    """A per-pool observable must be reachable from the public API: the packs carry a compartment channel so
    intra-axonal signal can be asked for separately -- what a study reports and what a fit targets. Built on the
    suite's slab walk, relabelling half its walkers so there are two pools to separate."""
    w = _susc_walk(200, seed=6)
    comp = np.zeros(w.n_walkers, np.int8); comp[w.n_walkers // 2:] = 1            # two pools: ids 0 and 1
    w = dataclasses.replace(w, compartment=comp, raw_compartment=None)
    pk = build_replay_pack(w, id="test/slab-susc-comp", envelope=_env7(theta_deg=[0, 90]), K=16,
                           license="CC-BY-4.0", citation="test")
    nt, dt = pk.n_t, pk.dt
    from dmipy_sim import RFEvent, ScannerSequence
    W = ScannerSequence(G=np.zeros((1, nt, 3)), dt=dt,                   # no gradient; a spin echo's 180 at TE/2
                        rf=[RFEvent(0.0, 90), RFEvent((nt - 1) * dt / 2, 180)])
    kw = dict(scanner=7.0, tissue=Tissue(chi_iso=1.06e-6), complex_signal=True)
    everything = pk.replay(W, **kw)
    pool0 = pk.replay(W, compartment=0, **kw)
    pool1 = pk.replay(W, compartment=1, **kw)
    assert np.isfinite(complex(pool0[0])) and np.isfinite(complex(pool1[0]))
    assert complex(pool0[0]) != complex(pool1[0]), "the two pools must not give the same answer"
    assert complex(pool0[0]) != complex(everything[0]), "compartment= had no effect"
    # a boolean mask must select the same walkers as the id does
    mask = np.zeros(w.n_walkers, bool); mask[:w.n_walkers // 2] = True
    assert complex(pk.replay(W, compartment=mask, **kw)[0]) == pytest.approx(complex(pool0[0]), rel=1e-9)
    with pytest.raises(ValueError, match="matched no walkers"):
        pk.replay(W, compartment=99, **kw)


def test_susc_path_bit_depth_is_the_cheap_axis_not_K():
    """int8 at full K beats f16 at half K on the SIGNAL, at the same bytes, keeping the capability: the C3 spectrum
    is nearly white above DC, so truncating K is a brick-wall approximation to the gate-weighted bit allocation;
    keeping every mode at 8 bits is the graded version. Both occupy the same space, so this is a fair swap."""
    from dmipy_sim.replay import compression as cx
    w = _susc_walk(200, seed=7, K_c3=32)
    coeffs, names, T, K = np.asarray(w.field_coeffs, np.float64)[:, 0], list(w.field_channel_names), w.T, w.K_c3
    a8, pm8 = bank.susc_path_encode_modes(coeffs, names, T=T, bits=8)
    a16, pm16 = bank.susc_path_encode_modes(coeffs[:, :K // 2], names, T=T, bits=16)
    ref = np.transpose(coeffs, (0, 2, 1))                                                  # the walk's own float64 modes
    assert a8["susc_path_dct"].nbytes + a8["susc_path_scale"].nbytes < ref.astype(np.float64).nbytes
    assert a8["susc_path_dct"].nbytes <= a16["susc_path_dct"].nbytes    # same or less space

    def err(arr, pm, n_pulses):                     # worst signal error over a direction sweep
        C = bank.susc_path_coeffs(arr, pm)[0]
        g = cx.cosine_projection(bank._cpmg_gate(N_T, n_pulses).reshape(1, -1, 1), DT, T, pm["K"])[0, :, 0]
        gr = cx.cosine_projection(bank._cpmg_gate(N_T, n_pulses).reshape(1, -1, 1), DT, T, K)[0, :, 0]
        out = 0.0
        for d in ([0, 0, 1], [1, 0, 0], [0.3, 0.5, 0.8]):
            d = np.asarray(d, float) / np.linalg.norm(d)
            kw = dict(B0=7.0, chi_iso=1.06e-6)
            p_a = GAMMA * bank.susc_path_field(C, d, **kw) @ g
            p_r = GAMMA * bank.susc_path_field(ref, d, **kw) @ gr
            out = max(out, abs(abs(np.exp(1j * p_a).mean()) - abs(np.exp(1j * p_r).mean())))
        return out

    # at the pack's declared capability (K//2 refocusing pulses) the truncated store has no modes
    # left to represent the gate, while the quantised full-K store still does
    assert err(a8, pm8, K // 2) < err(a16, pm16, K // 2)
    assert err(a8, pm8, 1) < 1e-2
    with pytest.raises(ValueError, match="bits must be"):
        bank.susc_path_encode_modes(coeffs, names, T=T, bits=6)


# ------------------------------------------------------------------ the builder takes a CoefficientWalk
def test_builder_takes_a_coefficient_walk_and_assembles_the_tiers_it_carries():
    import dmipy_sim as d
    walk = d.simulate_trajectories(300, D0, d.Cylinder(2e-6, (0, 0, 1)), 0.01, 5e-4, seed=0, require_gpu=False, K=16)
    pk = build_replay_pack(walk, id="test/walk", K=8, envelope=_lean_env(), license="CC-BY-4.0", citation="test")
    env = pk.meta["replay_envelope"]
    assert env["gradient"] and env["bulk_relaxation"] and env["surface_relaxivity"] and not env["field"]
    assert "per_comp" not in pk.meta and pk.has_relaxation                   # the channel, never the values
    assert pk.meta["compression"]["precision_tiers"]["usable"]         # the producer's walkers are i.i.d.
    assert pk.n_walkers == 300 and pk.n_t == walk.n_blocks + 1 and pk.K == 8 < walk.K
    assert pk.fidelity["certified"] == "measured" and pk.fidelity["measured_by"] == "walk_free_terms"
    # positions only: gradient tier alone, nothing silently assembled
    plain = d.simulate_trajectories(300, D0, d.Cylinder(2e-6, (0, 0, 1)), 0.01, 5e-4, seed=0, require_gpu=False, tiers=(), K=16)
    pk0 = build_replay_pack(plain, id="test/plain", K=8, envelope=_lean_env(), license="CC-BY-4.0", citation="test")
    e0 = pk0.meta["replay_envelope"]
    assert e0["gradient"] and not (e0["bulk_relaxation"] or e0["surface_relaxivity"] or e0["field"])
    # the refusals
    with pytest.raises(ValueError, match="weights has"):
        build_replay_pack(walk, id="x", weights=np.ones(7), K=8, envelope=_lean_env(), license="x", citation="x")
    with pytest.raises(TypeError, match="field"):                 # build_replay_pack takes no field= at all (#539)
        build_replay_pack(walk, id="x", field={"iso_local": 1}, K=8, envelope=_lean_env(), license="x", citation="x")
    with pytest.raises(TypeError, match="CoefficientWalk"):       # positions at the block ends are not a pack's input
        build_replay_pack(np.zeros((50, 5, 3), np.float32), id="x", K=4, license="x", citation="x")
    with pytest.raises(ValueError, match="measured"):             # a measured certificate needs the raw subset
        build_replay_pack(walk, id="x", K=8, fidelity="measured", license="x", citation="x")


def test_the_field_tiers_band_is_derived_on_the_walk():
    """The band and the container are fixed BEFORE the walk by the producer's pilot: the first pair, in ascending
    band with the narrower container first, whose codec error on the certificate's battery is within the pilot's
    floor, and the record of every pair read travels with the channel; the pack takes the walk's band."""
    from dmipy_sim.replay.bank import SUSC_PATH_LADDER
    w = _susc_walk(300, seed=8, envelope=_env7(theta_deg=[0, 90]))
    band = w.block["field_band"]
    assert band["rule"] == "pilot" and band["pilot_walkers"] > 0
    pk = build_replay_pack(w, id="test/slab-susc-auto", envelope=_env7(theta_deg=[0, 90]), K=16, license="CC-BY-4.0", citation="test")
    pm = pk.meta["compression"]["channels"]["susc_field_basis"]
    assert pm["band"] == band and (pm["K"] in SUSC_PATH_LADDER or pm["K"] == N_T - 1) and band["bits"] in (8, 16)
    depth = pm["bits"] if pm.get("container") is None else pm["container"][-1]["bits"]       # the modes above k = 0
    assert depth == band["bits"]
    order = [(r["K"], r["bits"]) for r in band["ladder"]]
    rungs = {(r["K"], r["bits"]): r for r in band["ladder"]}
    chosen = rungs[(pm["K"], depth)]
    assert chosen["floor"] > 0 and order == sorted(order)                             # ascending band, narrower container first
    if band["within_floor"]:
        assert chosen["err"] <= chosen["floor"]
        before = order[:order.index((pm["K"], depth))]
        assert all(rungs[p]["err"] > rungs[p]["floor"] for p in before)                # the first pair within the floor
        assert order[-1] == (pm["K"], depth)                                          # and the ladder stopped there
    else:
        best = min(r["err"] for r in rungs.values())
        assert chosen["err"] <= 1.03 * best
    assert band["above_position_band"] == (pm["K"] > w.K)
    assert pm["max_refocus_pulses"] == pm["K"] // 2
    # the pilot's reading is the pack's certificate of the channel
    assert pk.fidelity["susc_path_measured_on"] == "pilot" and pk.fidelity["err_susc_path"] == pytest.approx(chosen["err"], rel=1e-5)


def test_a_declared_train_depth_puts_a_floor_under_the_ladder():
    """A pack built for a train of n pulses must serve n: the tier serves K / 2, so the envelope's
    ``max_refocus_pulses`` keeps every rung at or above twice it, whatever the floor would have allowed."""
    w = _susc_walk(200, seed=9, envelope=_env7(max_refocus_pulses=12))
    band = w.block["field_band"]
    assert band["min_K"] == 24 and w.K_c3 >= 24 and band["depth"] == 12 and all(r["K"] >= 24 for r in band["ladder"])
    pk = build_replay_pack(w, id="test/slab-susc-depth", envelope=_env7(max_refocus_pulses=12), K=16, license="CC-BY-4.0", citation="test")
    assert pk.meta["compression"]["channels"]["susc_field_basis"]["max_refocus_pulses"] == 12      # the declared depth, not K / 2


def test_a_given_field_band_is_taken_as_given():
    """``K_c3=`` skips the pilot; ``susc_path_K=`` beyond the walk's band is refused."""
    w = _susc_walk(100, seed=10, K_c3=12)
    assert w.K_c3 == 12 and w.block["field_band"]["rule"] == "given"
    pk = build_replay_pack(w, id="test/given", envelope=_env7(), K=16, susc_path_K=8, license="x", citation="x")
    assert pk.meta["compression"]["channels"]["susc_field_basis"]["K"] == 8
    with pytest.raises(ValueError, match="accumulated 12 field modes"):
        build_replay_pack(w, id="test/given", envelope=_env7(), K=16, susc_path_K=16, license="x", citation="x")


def test_build_to_floor_keeps_topping_up_until_the_floor_is_met(monkeypatch):
    """The floor of the pilot underestimates and the floor falls slower than N^-1/2, so every estimate
    undershoots: one top-up leaves the target unmet (#441). The policy walks again from the last
    measurement until the floor is met, and the pack records that it is."""
    from dmipy_sim.replay import compression as cx
    asked = []
    floor_of = lambda n: 0.01 if n == 8000 else 0.02 * (8000 / n) ** 0.25
    real = cx.measure_floor_coded
    monkeypatch.setattr(cx, "measure_floor_coded", lambda *a, **kw: dict(real(*a, **kw), floor_max=floor_of(asked[-1])))
    pk = build_to_floor(lambda n: (asked.append(int(n)), _slab_walk(n_w=64, seed=0, raw=False))[1], id="test/floor-loop", K=16,
                        sigma_star=5e-3, pilot_n=8000, max_n=10_000_000, verbose=False,
                        license="CC0-1.0", citation="test")
    assert len(asked) >= 4, asked                                   # pilot, N*, and at least two top-ups
    assert all(b > a for a, b in zip(asked, asked[1:])), asked        # the count only grows
    assert floor_of(asked[-1]) <= 5e-3                                # and the last walk meets the floor
    assert pk.fidelity.get("target_floor") == 5e-3


def test_build_to_floor_stops_at_the_cap_and_says_the_target_is_not_met(monkeypatch):
    from dmipy_sim.replay import compression as cx
    asked = []
    real = cx.measure_floor_coded
    monkeypatch.setattr(cx, "measure_floor_coded", lambda *a, **kw: dict(real(*a, **kw), floor_max=0.02))   # never met
    pk = build_to_floor(lambda n: (asked.append(int(n)), _slab_walk(n_w=64, seed=0, raw=False))[1], id="test/floor-cap", K=16,
                        sigma_star=5e-3, pilot_n=8000, max_n=50_000, verbose=False,
                        license="CC0-1.0", citation="test")
    assert asked[-1] == 50_000 and asked.count(50_000) == 1, asked   # walked the cap once, then stopped
    assert pk.fidelity.get("target_floor") == 5e-3


def test_build_replay_pack_is_byte_reproducible(tmp_path):
    """Two packs built from the same walk and seed, in the same process, are byte-identical files: the run
    record (ids, timestamps, wall time, pid, host, the record's absolute path) lives beside the pack, never
    inside it, and the certificate's floats are rounded before they are written, so nothing a run or a
    reduction's order varies between the two builds reaches the bytes (dmipy-sim#541). The sidecar beside each
    pack names that pack's own sha256."""
    import hashlib

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    out1, out2 = tmp_path / "a.rpk", tmp_path / "b.rpk"
    for out in (out1, out2):
        build_replay_pack(_slab_walk(seed=0), id="test/repro", envelope=_lean_env(), K=16, license="CC-BY-4.0", citation="test", out_path=str(out))
    assert sha(out1) == sha(out2)

    for out in (out1, out2):
        sidecar_path = str(out) + ".run.json"
        assert os.path.exists(sidecar_path)
        sidecar = json.load(open(sidecar_path))
        assert sidecar["pack_sha256"] == sha(out)
        assert sidecar["pack"]["producer"] == "build_replay_pack"
        assert "run" not in read_rpk(out).meta.get("provenance", {})     # the run record is not in the file


def test_certificate_rounding_absorbs_a_last_ulp_difference():
    """The two readings dmipy-sim#541 measured for one ``fidelity.per_family.OGSE3.err_max`` on two hosts (same
    tensors, same package versions, a last-ulp float64 reduction difference) round to the same 6-significant-
    digit value, so the written bytes agree."""
    a = bank._round_certificate(dict(err_max=0.0018340476685469705))
    b = bank._round_certificate(dict(err_max=0.00183404766854697))
    assert a == b


def test_a_pack_with_the_old_in_file_run_record_still_loads(tmp_path):
    """A pack built before dmipy-sim#541 carries its run record inside ``provenance.run``: that is just data the
    file carries, read as it is, never translated or stripped on load."""
    out = tmp_path / "old.rpk"
    build_replay_pack(_slab_walk(), id="test/old", envelope=_lean_env(), K=16, license="CC-BY-4.0", citation="test", out_path=str(out))
    pk = read_rpk(out)
    meta = dict(pk.meta)
    meta["provenance"] = dict(meta.get("provenance") or {},
                              run=dict(pack=dict(id="20250101T000000-build_replay_pack-1", host="old-host",
                                                 code=dict(version="0.0.0", commit="0" * 40), record="/old/path")))
    bank.write_rpk(out, pk.arrays, meta)
    reloaded = read_rpk(out)
    assert reloaded.meta["provenance"]["run"]["pack"]["id"] == "20250101T000000-build_replay_pack-1"
    assert reloaded.fidelity["err_max"] == pk.fidelity["err_max"]        # the rest of the pack is untouched
