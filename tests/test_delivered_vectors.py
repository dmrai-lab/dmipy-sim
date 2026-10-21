"""A machine's gradient side on the time-segment layout (the DiSCo page's route): every voxel's delivered effective
vector on every segment, the magnet's own gradient against the stored background vector, and the transmit scale on the
pathway amplitude, held against the reference route -- each voxel's acquisition rebuilt as the machine plays it
(``ScannerSequence.with_gradient_nonlinearity / with_background_gradient / with_concomitant``, through
``bore.encoding_classes`` with no binning) and replayed walker by walker on the merged pack -- for the three
catalogued machines with every term on, a spin echo, a stimulated echo and a double diffusion encoding of two levels."""
from __future__ import annotations

import os

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.acquisition.epg import transmit_amplitude
from dmipy_sim.acquisition.scanners import ScannerLimits
from dmipy_sim.phantom.bore import encoding_classes
from dmipy_sim.replay.bank import build_replay_pack, merge_packs
from dmipy_sim.replay.time_segments import TimeSegmentLayout, segment_rows, write_time_segment_layout
from dmipy_sim.spec import walk_spec, StratifiedByVoxel
from dmipy_sim.spec.tissue import Tissue

DELTA, BIG_DELTA = 0.2e-3, 0.5e-3
TISSUE = Tissue(T2={"intra": 0.03, "extra": 0.08, "myelin": 0.01}, T1={"intra": 0.9, "extra": 1.4, "myelin": 0.3},
                rho2=1e-5, chi_iso=-1e-7, chi_aniso=-1.5e-8)
# the phantom's centre 7.1 cm from isocentre, off every axis: inside the Swoop's 8 cm anchor, every term non-zero
OFFSET = np.array([0.05, 0.03, 0.04])
G_MAX = 60.0                                                     # T/m: the envelope of a 0.8 ms walk


@pytest.fixture(scope="module")
def layout(spec_grid):
    """Two blocks of the three-strand spec walked with the sheath's field, each packed and its segment rows written from
    the walk with the tiers and the background vectors; the layout of both and the merged pack."""
    spec, grid, tmp = spec_grid
    packs, dirs = [], []
    for b in (0, 1):
        want = np.zeros(grid.shape, np.int64); want[b] = 12
        w = walk_spec(spec, T_max=8e-4, block_h=2e-5, seed=31 + b, require_gpu=False, adaptive_steps=True, K=12,
                      seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}))
        kw = {} if not packs else dict(position_container=packs[0].meta["compression"]["container"],      # a fill's shards
                                       blt_container=packs[0].meta["compression"]["channels"]["boundary_local_time"]["container"])
        pk = build_replay_pack(w, id=f"t/dm{b}", license="x", citation="x", K=12, susc_path_K=4, voxel_grid=grid, **kw)
        packs.append(pk)
        n_t = pk.n_t
        dirs.append(str(tmp / f"dm_rows{b}"))
        segment_rows(w, _shapes(n_t), dirs[-1], grid=grid, pack=pk, g_max=G_MAX, tiers=True, background=True)
    write_time_segment_layout(dirs, str(tmp / "dm_layout"))
    return TimeSegmentLayout(str(tmp / "dm_layout")), merge_packs(packs, id="t/dm-merged"), grid, n_t, dirs, tmp


def _dde(n_t, a=1.0):
    """A double diffusion encoding refocused by the spin echo's one pulse: a pair along z and a pair along x at ``a``
    times its amplitude, each pair's lobes of one polarity on either side of the pulse (so the Maxwell term refocuses
    too, as it does not across a bipolar pair), four segments of two levels."""
    se = d.pgse([[0, 0, 1]], DELTA, BIG_DELTA, gradient_strengths=0.05, n_t=n_t, slew_rate=np.inf)
    G = np.zeros((1, n_t, 3), np.float32)
    for i0, v in ((1, [0, 0, 1.0]), (9, [a, 0, 0]), (22, [0, 0, 1.0]), (30, [a, 0, 0])):
        G[0, i0:i0 + 8, :] = v
    return se.with_gradient(G)


def _shapes(n_t):
    return {"se": d.pgse([[0, 0, 1]], DELTA, BIG_DELTA, gradient_strengths=0.05, n_t=n_t, slew_rate=np.inf),
            "ste": d.pgste([[0, 0, 1]], DELTA, BIG_DELTA - DELTA, gradient_strengths=0.05, n_t=n_t, slew_rate=np.inf,
                           ste_flip_angles=(90.0, 90.0, 90.0)),
            "dde": _dde(n_t)}


def _protocol(name, n_t):
    rng = np.random.default_rng(5)
    u = rng.normal(size=(4, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    b = np.array([1e9, 2e9, 3e9, 3e9, 0.0]); dirs = np.concatenate([u, [[0, 0, 1]]])
    if name == "se":
        return d.pgse(dirs.tolist(), DELTA, BIG_DELTA, bvalues=b, n_t=n_t, slew_rate=np.inf)
    if name == "ste":
        return d.pgste(dirs.tolist(), DELTA, BIG_DELTA - DELTA, bvalues=b, n_t=n_t, slew_rate=np.inf, ste_flip_angles=(90.0, 90.0, 90.0))
    rows = [np.asarray(_dde(n_t, a).G)[0] * g for g, a in ((20.0, 1.0), (40.0, 0.6), (50.0, 1.0), (30.0, -0.5), (0.0, 1.0))]
    se = _dde(n_t)
    return d.ScannerSequence(G=np.stack(rows), dt=se.dt, rf=se.rf, readout=se.readout)


def _term(sm, name, floor):
    """The rounding's term on a voxel: 3 sigma times its floor plus sigma^2 / 2, sigma^2 the partition's realised
    rounding variance."""
    s2 = sm.manifest["partitions"][name]["rounding"]["variance"]
    return 3.0 * np.sqrt(s2) * floor + 0.5 * s2


def _direct(merged, grid, placed, sc, seq, ijk):
    """The reference: every voxel's acquisition as the machine plays it (no binning), replayed on the merged pack
    walker by walker at the tissue and the machine's field, times the pathway amplitude at the voxel's transmit scale
    over the nominal one; ``(n_voxels, n_meas)``."""
    cls, played = encoding_classes(sc, placed, seq, ijk, tolerance=None)
    jk, _ = grid.bin(merged.r0); v = np.ravel_multi_index(jk.T, grid.shape)
    flat = np.ravel_multi_index(ijk.T, grid.shape)
    kappa = None
    if sc.has_transmit_profile:
        kappa = sc.b1_scale((placed.positions_m(ijk) - np.asarray(placed.isocenter_m)))
    out = np.zeros((len(ijk), seq.n_meas))
    for k in range(len(ijk)):
        w, ew, E = merged.walker_signals(played[cls[k]], tissue=TISSUE, scanner=sc)
        sel = v == flat[k]
        out[k] = np.abs((ew[sel, None] * E[sel]).sum(0) / w[sel].sum())
        if kappa is not None:
            out[k] *= transmit_amplitude(seq, [kappa[k]])[0] / transmit_amplitude(seq, [1.0])[0]
    return out


@pytest.mark.parametrize("scanner", ["swoop", "prisma", "terra"])
def test_the_delivered_layout_is_the_machines_own_replay(layout, scanner):
    """At 7.1 cm from isocentre, every catalogued term on: the layout's image under the delivery equals, voxel by
    voxel, the merged pack replayed on the acquisition the machine plays there, for a spin echo, a stimulated echo and a
    double diffusion encoding whose pairs play two levels (the Maxwell gradient is one vector per segment, whatever the
    levels), within the rounding's term; and the delivery moves the image off the field-only one by more than that."""
    sm, merged, grid, n_t, dirs, tmp = layout
    sc = ScannerLimits.of(scanner)
    placed = sm.grid.centred_at(OFFSET)
    live = sm.live_voxels
    ijk = np.stack(np.unravel_index(live, tuple(grid.shape)), axis=1)
    for name in ("se", "ste", "dde"):
        seq = _protocol(name, n_t)
        dl = sm.delivery(seq, sc, placed)
        assert dl.partition == name and dl.q.shape == (len(live), seq.n_meas, len(sm.partitions[name].segments), 3)
        assert (dl.g0 is not None) == sc.has_field_law and (dl.kappa is not None) == sc.has_transmit_profile
        S, floor = sm.image(seq, tissue=TISSUE, scanner=sc, b0_direction=sc.b0_axis, delivered=dl)
        S0, _ = sm.image(seq, tissue=TISSUE, scanner=sc, b0_direction=sc.b0_axis)
        ref = _direct(merged, grid, placed, sc, seq, ijk)
        got = S.reshape(-1, seq.n_meas)[live]
        err = np.abs(got - ref).max(); bound = _term(sm, name, float(np.nanmax(floor)))
        assert err < bound, (scanner, name, err, bound)
        moved = np.abs(got - S0.reshape(-1, seq.n_meas)[live]).max()
        assert moved > err, (scanner, name, moved, err)


def test_the_torch_backend_is_the_jax_image_under_a_delivery(layout):
    """The delivered kernel on torch (the shared GPU pool's backend) equals the JAX one, tiers and background on."""
    pytest.importorskip("torch")
    sm, merged, grid, n_t, dirs, tmp = layout
    sc = ScannerLimits.of("swoop")
    seq = _protocol("ste", n_t)
    dl = sm.delivery(seq, sc, sm.grid.centred_at(OFFSET))
    S, _ = sm.image(seq, tissue=TISSUE, scanner=sc, b0_direction=sc.b0_axis, delivered=dl)
    St, _ = sm.image(seq, tissue=TISSUE, scanner=sc, b0_direction=sc.b0_axis, delivered=dl, backend="torch", device="cpu")
    assert np.nanmax(np.abs(St - S)) < 5e-6


def test_each_term_reaches_the_image(layout):
    """On the Swoop at 7.1 cm the nonlinearity, the background and the Maxwell term each move the image alone, and the
    transmit scale multiplies it by the pathway amplitude's ratio exactly (sin(k 90) sin^2(k 90) for the spin echo)."""
    sm, merged, grid, n_t, dirs, tmp = layout
    sc = ScannerLimits.of("swoop")
    placed = sm.grid.centred_at(OFFSET)
    seq = _protocol("se", n_t)
    off = dict(nonlinearity=False, background=False, concomitant=False, transmit=False)
    base, _ = sm.image(seq, delivered=sm.delivery(seq, sc, placed, **off))
    S0, _ = sm.image(seq)
    assert np.nanmax(np.abs(base - S0)) < 1e-6
    for term in ("nonlinearity", "concomitant"):
        S, _ = sm.image(seq, delivered=sm.delivery(seq, sc, placed, **{**off, term: True}))
        assert np.nanmax(np.abs(S - S0)) > 1e-5, term
    # the magnet's own gradient over a 0.8 ms walk winds its walkers by less than half the background's derived step:
    # every background vector rounds to zero, no word is stored, and the term moves nothing it could resolve
    S, _ = sm.image(seq, delivered=sm.delivery(seq, sc, placed, **{**off, "background": True}))
    bg = sm.manifest["groups"][sm.manifest["partitions"]["se"]["group"]]["background"]
    if bg["n_words"]:
        assert np.nanmax(np.abs(S - S0)) > 1e-5
    else:
        import json
        rec = json.load(open(os.path.join(dirs[0], "rows.json")))
        amax = max(max(g["background_amax"]) for g in rec["groups"].values())
        assert amax <= 0.5 * bg["step"] and np.nanmax(np.abs(S - S0)) == 0.0
    dl = sm.delivery(seq, sc, placed, **{**off, "transmit": True})
    S, _ = sm.image(seq, delivered=dl)
    ratio = np.sin(np.pi / 2 * dl.kappa) ** 3
    live = sm.live_voxels
    np.testing.assert_allclose(S.reshape(-1, seq.n_meas)[live], S0.reshape(-1, seq.n_meas)[live] * ratio[:, None], rtol=1e-5)


def test_a_delivery_of_another_sequence_is_refused(layout):
    sm, merged, grid, n_t, dirs, tmp = layout
    sc = ScannerLimits.of("prisma")
    seq = _protocol("se", n_t)
    dl = sm.delivery(seq, sc, sm.grid.centred_at(OFFSET))
    with pytest.raises(ValueError, match="another sequence"):
        sm.image(d.pgse([[0, 0, 1]], DELTA, BIG_DELTA, bvalues=[1e9], n_t=n_t, slew_rate=np.inf), delivered=dl)


def test_a_layout_without_background_refuses_a_magnet_gradient(layout):
    """A layout written without the background words refuses the Swoop's own gradient by name; without the background
    term the same delivery replays."""
    sm, merged, grid, n_t, dirs, tmp = layout
    plain_rows = str(tmp / "dm_plain_rows")
    import json
    rec = json.load(open(os.path.join(dirs[0], "rows.json")))
    # the rows of block 0 written again without the background
    from dmipy_sim.replay.replay import ReplayPack
    pk = merged
    segment_rows(pk, _shapes(n_t), plain_rows, grid=grid, g_max=G_MAX, tiers=True, floor_target=rec["source"]["floor_target"])
    write_time_segment_layout([plain_rows], str(tmp / "dm_plain"))
    plain = TimeSegmentLayout(str(tmp / "dm_plain"))
    assert not any(grp.get("background") for grp in plain.manifest["groups"].values())
    sc = ScannerLimits.of("swoop")
    seq = _protocol("se", n_t)
    with pytest.raises(ValueError, match="background"):
        plain.image(seq, delivered=plain.delivery(seq, sc, plain.grid.centred_at(OFFSET)))
    plain.image(seq, delivered=plain.delivery(seq, sc, plain.grid.centred_at(OFFSET), background=False))


def test_a_magnets_gradient_through_a_stimulated_echo_accrues_nothing_while_stored():
    """The effective gradient is the coherence sign times the transverse gate: through the mixing time the
    magnetisation is along B0 and a magnet's constant gradient adds no phase, so the background of a stimulated echo
    is refocused like a spin echo's (to one sample per change of the gate) and not left as a winding."""
    from dmipy_sim.replay.time_segments import _background_sequence
    ste = d.pgste([[0, 0, 1]], 7.6e-3, 38.3e-3, gradient_strengths=0.1, n_t=1000, slew_rate=np.inf, ste_flip_angles=(90.0, 90.0, 90.0))
    bg = _background_sequence(ste)
    G = np.asarray(bg.G_eff)[0, :, 0]
    np.testing.assert_array_equal(G, np.asarray(ste.effective_gate))          # the background's time course IS the gate
    stored = np.asarray(ste.effective_gate) == 0.0
    assert stored.sum() > 600 and not bg.unbalanced
    assert set(np.unique(G[~stored])) == {-1.0, 1.0}
