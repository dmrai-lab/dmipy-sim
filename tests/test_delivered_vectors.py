"""A machine's gradient side on the time-segment layout (the DiSCo page's route): every voxel's delivered effective
vector on every segment, the magnet's own gradient against the stored background vector, and the transmit scale on the
pathway amplitude, held against the reference route -- each voxel's acquisition rebuilt as the machine plays it
(``ScannerSequence.with_gradient_nonlinearity / with_background_gradient / with_concomitant``, through
``bore.encoding_classes`` with no binning) and replayed walker by walker on the pack -- for the three
catalogued machines with every term on, a spin echo, a stimulated echo and a double diffusion encoding of two levels."""
from __future__ import annotations

from tests._envelope import stated, walk_envelope
import os

import numpy as np
import pytest

import dmipy_sim as d

import dmipy_replay as dr
from dmipy_replay.acquisition.epg import transmit_amplitude
from dmipy_replay.acquisition.scanners import ScannerLimits
from dmipy_replay.acquisition.scanner_sequence import concomitant_terms
from dmipy_replay.phantom.bore import encoding_classes, gradient_tensor_map
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_replay.replay.time_segments import TimeSegmentLayout
from dmipy_sim.replay.time_segments import segment_rows, write_time_segment_layout
from dmipy_sim.spec import StratifiedByVoxel
from dmipy_sim import walk
from dmipy_replay.spec.tissue import Tissue

# The machines play what their slew allows: every lobe is a trapezoid on a 0.6 ms grid, one ramp sample at half its
# plateau each side, at most 0.023 T/m (the Swoop's maximum), so every step is at most 19.2 T/m/s, under the Swoop's
# 22 T/m/s and the Prisma's and the Terra's 200 T/m/s; a lobe is three constant pieces of the layout's partition. Every
# encoding and the walk are 9.6 ms (16 steps).
DT, G_TOP = 0.6e-3, 0.023
TE_WALK, N_T = 9.6e-3, 17
TISSUE = Tissue(T2={"intra": 0.03, "extra": 0.08, "myelin": 0.01}, T1={"intra": 0.9, "extra": 1.4, "myelin": 0.3},
                rho2=1e-5, chi_iso=-1e-7, chi_aniso=-1.5e-8)
# the phantom's centre 7.1 cm from isocentre, off every axis: inside the Swoop's 8 cm anchor, every term non-zero
OFFSET = np.array([0.05, 0.03, 0.04])
G_MAX = 0.08                                                     # T/m: the largest amplitude the three machines play
# T/m: the least a machine's nonlinearity or Maxwell term moves a delivered vector of these encodings at OFFSET (the
# Terra's Maxwell term, 2.9e-6 T/m, is the smallest of the three machines')
TERM_MIN = 1e-6


@pytest.fixture(scope="module")
def layout(spec_grid):
    """The three-strand spec walked once with the sheath's field over every voxel of its grid, packed, and its segment
    rows written from the walk with the tiers and the background vectors; the layout and the pack."""
    spec, grid, tmp = spec_grid
    want = np.full(grid.shape, 6, np.int64)
    w = walk(spec, T_max=TE_WALK, block_h=DT, seed=31, require_gpu=False, adaptive_steps=True, K=24,
             seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}), envelope=walk_envelope(of=spec))
    pk = build_replay_pack(w, id="t/dm", license="x", citation="x", K=24, susc_path_K=4, voxel_grid=grid, **stated(of=w))
    rows = str(tmp / "dm_rows")
    segment_rows(w, _shapes(), rows, grid=grid, pack=pk, g_max=G_MAX, tiers=True, background=True)
    write_time_segment_layout([rows], str(tmp / "dm_layout"))
    return TimeSegmentLayout(str(tmp / "dm_layout")), pk, grid, pk.n_t, [rows], tmp


def _trapezoid(n):
    """A lobe of ``n`` samples on the grid: half its plateau on the first and the last, the plateau between."""
    return np.r_[0.5, np.ones(n - 2), 0.5]


def _lobed(base, lobes):
    """``base``'s grid, pulses and readout (its own lobes are not kept) playing ``lobes`` -- ``(first sample, samples,
    direction)`` -- at ``G_TOP``."""
    G = np.zeros((1, base.n_t, 3), np.float32)
    for i0, n, v in lobes:
        G[0, i0:i0 + n, :] = G_TOP * _trapezoid(n)[:, None] * np.asarray(v, np.float32)
    return base.with_gradient(G)


def _se():
    """The spin echo: a 4.2 ms trapezoid along z either side of the 180 at 4.8 ms."""
    base = dr.pgse([[0, 0, 1]], 3e-3, 5e-3, gradient_strengths=1e-3, TE=TE_WALK, n_t=N_T, slew_rate=22.0)
    return _lobed(base, [(0, 7, [0, 0, 1.0]), (9, 7, [0, 0, 1.0])])


def _ste():
    """The stimulated echo: a 3 ms trapezoid along z before the store at 3 ms and after the recall at 6.6 ms."""
    base = dr.pgste([[0, 0, 1]], 2.4e-3, 3.6e-3, gradient_strengths=1e-3, TE=TE_WALK, n_t=N_T, slew_rate=22.0,
                    ste_flip_angles=(90.0, 90.0, 90.0))
    return _lobed(base, [(0, 5, [0, 0, 1.0]), (11, 5, [0, 0, 1.0])])


def _dde(a=1.0):
    """A double diffusion encoding refocused by the spin echo's one pulse at 4.8 ms: a pair along z and a pair along x
    at ``a`` times its amplitude, each pair's 1.8 ms lobes of one polarity on either side of the pulse (so the Maxwell
    term refocuses too, as it does not across a bipolar pair), twelve constant pieces."""
    base = dr.pgse([[0, 0, 1]], 1.5e-3, 5e-3, gradient_strengths=1e-3, TE=TE_WALK, n_t=N_T, slew_rate=22.0)
    return _lobed(base, [(1, 3, [0, 0, 1.0]), (4, 3, [a, 0, 0]), (9, 3, [0, 0, 1.0]), (12, 3, [a, 0, 0])])


def _shapes():
    return {"se": _se(), "ste": _ste(), "dde": _dde()}


def _rows(template, dirs, amplitudes):
    """``template`` (one row along z) played along each of ``dirs`` at ``amplitudes`` times its own: the one trapezoid
    with its one-sample ramps at every amplitude, so no row steps faster than the template."""
    prof = np.asarray(template.G, np.float64)[0, :, 2]
    G = np.stack([prof[:, None] * (a * np.asarray(v, np.float64))[None, :] for a, v in zip(amplitudes, dirs)])
    return dr.ScannerSequence(G=G.astype(np.float32), dt=template.dt, rf=template.rf, readout=template.readout)


def _protocol(name):
    """Five measurements: four oblique directions at b in the ratio 1 : 2 : 3 : 3 (amplitudes up to ``G_TOP``) and a
    b = 0 row; the double diffusion encoding's rows play its two pairs at two levels."""
    rng = np.random.default_rng(5)
    u = rng.normal(size=(4, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    dirs = np.concatenate([u, [[0, 0, 1]]])
    amp = np.sqrt(np.array([1.0, 2.0, 3.0, 3.0, 0.0]) / 3.0)
    if name in ("se", "ste"):
        return _rows(_se() if name == "se" else _ste(), dirs, amp)
    rows = [np.asarray(_dde(a).G)[0] * g for g, a in ((0.4, 1.0), (0.8, 0.6), (1.0, 1.0), (0.6, -0.5), (0.0, 1.0))]
    se = _dde()
    return dr.ScannerSequence(G=np.stack(rows), dt=se.dt, rf=se.rf, readout=se.readout)


def _term(sm, name, floor):
    """The rounding's term on a voxel: 3 sigma times its floor plus sigma^2 / 2, sigma^2 the partition's realised
    rounding variance."""
    s2 = sm.manifest["partitions"][name]["rounding"]["variance"]
    return 3.0 * np.sqrt(s2) * floor + 0.5 * s2


def _direct(pack, grid, placed, sc, seq, ijk):
    """The reference: every voxel's acquisition as the machine plays it (no binning), replayed on the pack
    walker by walker at the tissue and the machine's field, times the pathway amplitude at the voxel's transmit scale
    over the nominal one; ``(n_voxels, n_meas)``."""
    cls, played = encoding_classes(sc, placed, seq, ijk, tolerance=None)
    jk, _ = grid.bin(pack.r0); v = np.ravel_multi_index(jk.T, grid.shape)
    flat = np.ravel_multi_index(ijk.T, grid.shape)
    kappa = None
    if sc.has_transmit_profile:
        kappa = sc.b1_scale((placed.positions_m(ijk) - np.asarray(placed.isocenter_m)))
    out = np.zeros((len(ijk), seq.n_meas))
    for k in range(len(ijk)):
        w, ew, E = pack.walker_signals(played[cls[k]], tissue=TISSUE, scanner=sc, tol=None)      # the layout is certified against the whole pack
        sel = v == flat[k]
        out[k] = np.abs((ew[sel, None] * E[sel]).sum(0) / w[sel].sum())
        if kappa is not None:
            out[k] *= transmit_amplitude(seq, [kappa[k]])[0] / transmit_amplitude(seq, [1.0])[0]
    return out


def _assert_vectors_are_the_machines_law(sm, sc, placed, seq):
    """The delivered vectors of ``seq`` are the machine's own law, term by term: with every term off they are the
    commanded levels through the gate; the nonlinearity alone is ``e L(r) G`` with the machine's tensor at each voxel;
    the Maxwell term alone is ``e (G + gc(G))`` with the machine's field; both are ``e (L G + gc(L G))``; and turning
    either term off moves some vector by more than ``TERM_MIN``."""
    off = dict(nonlinearity=False, background=False, concomitant=False, transmit=False)
    d0 = sm.delivery(seq, sc, placed, **off)
    ijk = np.stack(np.unravel_index(np.asarray(d0.voxels), tuple(placed.shape)), axis=1)
    e = np.asarray(sm.partitions[d0.partition].gate, np.float64)
    G = np.asarray(d0.levels, np.float64) * np.where(e == 0.0, 0.0, e)[None, :, None]           # the commanded plateaus
    r = placed.positions_m(ijk) - np.asarray(placed.isocenter_m, np.float64)
    axis = (0.0, 0.0, 1.0) if sc.b0_axis is None else sc.b0_axis
    LG = np.einsum("vij,msj->vmsi", gradient_tensor_map(sc, placed)(placed.positions_m(ijk)), G)
    gc = lambda g: concomitant_terms(g, r[:, None, None, :], float(sc.field_T), axis)[0]
    law = {(False, False): np.broadcast_to(G[None], LG.shape), (True, False): LG,
           (False, True): G[None] + gc(np.broadcast_to(G[None], LG.shape)), (True, True): LG + gc(LG)}
    q = {}
    for (nl, cc), want in law.items():
        q[nl, cc] = np.asarray(sm.delivery(seq, sc, placed, **{**off, "nonlinearity": nl, "concomitant": cc}).q)
        np.testing.assert_allclose(q[nl, cc], want * e[None, None, :, None], rtol=1e-12, atol=1e-15,
                                   err_msg=f"{sc.name} nonlinearity={nl} concomitant={cc}")
    assert np.abs(q[True, True] - q[False, True]).max() > TERM_MIN, (sc.name, "nonlinearity")
    assert np.abs(q[True, True] - q[True, False]).max() > TERM_MIN, (sc.name, "concomitant")


@pytest.mark.parametrize("scanner", ["swoop", "prisma", "terra"])
def test_the_delivered_layout_is_the_machines_own_replay(layout, scanner):
    """At 7.1 cm from isocentre, every catalogued term on, for a spin echo, a stimulated echo and a double diffusion
    encoding whose pairs play two levels (the Maxwell gradient is one vector per segment, whatever the levels):

    - the layout's image under the delivery equals, voxel by voxel, the pack replayed on the acquisition the
      machine plays there, within the rounding's term (every machine);
    - the delivered vectors are the machine's nonlinearity and Maxwell term at each voxel, and each term moves them by
      more than ``TERM_MIN`` (every machine: the test sees a term the delivery dropped);
    - on the Swoop, whose transmit profile, own gradient and 64 mT Maxwell term move the image by more than the
      rounding at the gradients it plays, the delivered image differs from the field-only one by more than its error
      to the reference (the Prisma's and the Terra's terms move these encodings' image by about 1e-6, below the
      rounding of this walk, so their guard is the vectors')."""
    sm, pack, grid, n_t, dirs, tmp = layout
    sc = ScannerLimits.of(scanner)
    placed = sm.grid.centred_at(OFFSET)
    live = sm.live_voxels
    ijk = np.stack(np.unravel_index(live, tuple(grid.shape)), axis=1)
    for name in ("se", "ste", "dde"):
        seq = _protocol(name)
        dl = sm.delivery(seq, sc, placed)
        assert dl.partition == name and dl.q.shape == (len(live), seq.n_meas, len(sm.partitions[name].segments), 3)
        assert (dl.g0 is not None) == sc.has_field_law and (dl.kappa is not None) == sc.has_transmit_profile
        S, floor = sm.image(seq, tissue=TISSUE, scanner=sc, b0_direction=sc.b0_axis, delivered=dl)
        S0, _ = sm.image(seq, tissue=TISSUE, scanner=sc, b0_direction=sc.b0_axis)
        ref = _direct(pack, grid, placed, sc, seq, ijk)
        got = S.reshape(-1, seq.n_meas)[live]
        err = np.abs(got - ref).max(); bound = _term(sm, name, float(np.nanmax(floor)))
        assert err < bound, (scanner, name, err, bound)
        _assert_vectors_are_the_machines_law(sm, sc, placed, seq)
        if scanner == "swoop":
            moved = np.abs(got - S0.reshape(-1, seq.n_meas)[live]).max()
            assert moved > err, (scanner, name, moved, err)


def test_the_torch_backend_is_the_jax_image_under_a_delivery(layout):
    """The delivered kernel on torch (the shared GPU pool's backend) equals the JAX one, tiers and background on."""
    pytest.importorskip("torch")
    sm, pack, grid, n_t, dirs, tmp = layout
    sc = ScannerLimits.of("swoop")
    seq = _protocol("ste")
    dl = sm.delivery(seq, sc, sm.grid.centred_at(OFFSET))
    S, _ = sm.image(seq, tissue=TISSUE, scanner=sc, b0_direction=sc.b0_axis, delivered=dl)
    St, _ = sm.image(seq, tissue=TISSUE, scanner=sc, b0_direction=sc.b0_axis, delivered=dl, backend="torch", device="cpu")
    assert np.nanmax(np.abs(St - S)) < 5e-6


def test_each_term_reaches_the_image(layout):
    """On the Swoop at 7.1 cm the nonlinearity, the background and the Maxwell term each move the image alone, and the
    transmit scale multiplies it by the pathway amplitude's ratio exactly (sin(k 90) sin^2(k 90) for the spin echo)."""
    sm, pack, grid, n_t, dirs, tmp = layout
    sc = ScannerLimits.of("swoop")
    placed = sm.grid.centred_at(OFFSET)
    seq = _protocol("se")
    off = dict(nonlinearity=False, background=False, concomitant=False, transmit=False)
    base, _ = sm.image(seq, delivered=sm.delivery(seq, sc, placed, **off))
    S0, _ = sm.image(seq)
    assert np.nanmax(np.abs(base - S0)) < 1e-6
    for term in ("nonlinearity", "concomitant"):
        S, _ = sm.image(seq, delivered=sm.delivery(seq, sc, placed, **{**off, term: True}))
        assert np.nanmax(np.abs(S - S0)) > 1e-5, term
    # the magnet's own gradient moves the image when the layout stores its background words; when it winds the walkers
    # by less than half the background's derived step, every background vector rounds to zero, no word is stored, and
    # the term moves nothing it could resolve
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
    sm, pack, grid, n_t, dirs, tmp = layout
    sc = ScannerLimits.of("prisma")
    seq = _protocol("se")
    dl = sm.delivery(seq, sc, sm.grid.centred_at(OFFSET))
    with pytest.raises(ValueError, match="another sequence"):
        sm.image(_se(), delivered=dl)


def test_a_layout_without_background_refuses_a_magnet_gradient(layout):
    """A layout written without the background words refuses the Swoop's own gradient by name; without the background
    term the same delivery replays."""
    sm, pack, grid, n_t, dirs, tmp = layout
    plain_rows = str(tmp / "dm_plain_rows")
    # the pack's rows written again without the background
    segment_rows(pack, _shapes(), plain_rows, grid=grid, g_max=G_MAX, tiers=True)     # the pack's own declared floor
    write_time_segment_layout([plain_rows], str(tmp / "dm_plain"))
    plain = TimeSegmentLayout(str(tmp / "dm_plain"))
    assert not any(grp.get("background") for grp in plain.manifest["groups"].values())
    sc = ScannerLimits.of("swoop")
    seq = _protocol("se")
    with pytest.raises(ValueError, match="background"):
        plain.image(seq, delivered=plain.delivery(seq, sc, plain.grid.centred_at(OFFSET)))
    plain.image(seq, delivered=plain.delivery(seq, sc, plain.grid.centred_at(OFFSET), background=False))


def test_a_magnets_gradient_through_a_stimulated_echo_accrues_nothing_while_stored():
    """The effective gradient is the coherence sign times the transverse gate: through the mixing time the
    magnetisation is along B0 and a magnet's constant gradient adds no phase, so the background of a stimulated echo
    is refocused like a spin echo's (to one sample per change of the gate) and not left as a winding."""
    from dmipy_sim.replay.time_segments import _background_sequence
    ste = dr.pgste([[0, 0, 1]], 7.6e-3, 38.3e-3, gradient_strengths=0.1, n_t=1000, slew_rate=np.inf, ste_flip_angles=(90.0, 90.0, 90.0))
    bg = _background_sequence(ste)
    G = np.asarray(bg.G_eff)[0, :, 0]
    np.testing.assert_array_equal(G, np.asarray(ste.effective_gate))          # the background's time course IS the gate
    stored = np.asarray(ste.effective_gate) == 0.0
    assert stored.sum() > 600 and not bg.unbalanced
    assert set(np.unique(G[~stored])) == {-1.0, 1.0}
