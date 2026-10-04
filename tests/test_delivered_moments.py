"""A machine's gradient side on the shape-moment layout (the DiSCo page's route): every voxel's delivered encoding
vector, the magnet's own gradient against the stored background moment, and the transmit scale on the pathway
amplitude, held against the reference route -- each voxel's acquisition rebuilt as the machine plays it
(``ScannerSequence.with_gradient_nonlinearity / with_background_gradient / with_concomitant``, through
``bore.encoding_classes`` with no binning) and replayed walker by walker on the merged pack -- for the three
catalogued machines with every term on."""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.acquisition.epg import transmit_amplitude
from dmipy_sim.acquisition.scanners import ScannerLimits
from dmipy_sim.phantom.bore import delivered_moments, encoding_classes
from dmipy_sim.replay.bank import build_replay_pack, merge_packs
from dmipy_sim.replay.columnar import ColumnarPack
from dmipy_sim.replay.shape_moments import ShapeMoments, stamp_background, write_shape_moments
from dmipy_sim.fill.consolidate import consolidate
from dmipy_sim.spec import walk_spec, fill_field, StratifiedByVoxel
from dmipy_sim.spec.tissue import Tissue

DELTA, BIG_DELTA = 0.2e-3, 0.5e-3
TISSUE = Tissue(T2={"intra": 0.03, "extra": 0.08, "myelin": 0.01}, T1={"intra": 0.9, "extra": 1.4, "myelin": 0.3},
                rho2=1e-5, chi_iso=-1e-7, chi_aniso=-1.5e-8)
# the phantom's centre 7.1 cm from isocentre, off every axis: inside the Swoop's 8 cm anchor, every term non-zero
OFFSET = np.array([0.05, 0.03, 0.04])


@pytest.fixture(scope="module")
def layout(spec_grid):
    """Two blocks of the three-strand spec walked with the sheath's field, consolidated into columns; the merged pack;
    a layout of a spin-echo and a stimulated-echo shape with the tiers and the background moments."""
    spec, grid, tmp = spec_grid
    os.makedirs(str(tmp / "dm_shards"), exist_ok=True)
    packs = []
    for b in (0, 1):
        want = np.zeros(grid.shape, np.int64); want[b] = 12
        w = walk_spec(spec, T_max=8e-4, dt_save=2e-5, seed=31 + b, require_gpu=False, field_res=0.5e-6,
                      seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}))
        w = fill_field(w, w.field_basis)
        packs.append(build_replay_pack(w, id=f"t/dm{b}", license="x", citation="x", K=3, susc_path_K=4, voxel_grid=grid,
                                       out_path=str(tmp / "dm_shards" / f"block-000{b}.p1.rpk")))
    consolidate(str(tmp / "dm_shards"), str(tmp / "dm_layout"), blocks=[0, 1], id="t/dm-columns")
    col = ColumnarPack(str(tmp / "dm_layout"))
    n_t = packs[0].n_t
    shapes = _shapes(n_t)
    out = str(tmp / "dm_moments")
    write_shape_moments(col, shapes, out, tol=1e-9, chunk_rows=7, tiers=True, background=True)
    return ShapeMoments(out), col, merge_packs(packs, id="t/dm-merged"), grid, n_t, shapes, tmp


def _shapes(n_t):
    return {"se": d.pgse([[0, 0, 1]], DELTA, BIG_DELTA, gradient_strengths=0.05, n_t=n_t, slew_rate=np.inf),
            "ste": d.pgste([[0, 0, 1]], DELTA, BIG_DELTA - DELTA, gradient_strengths=0.05, n_t=n_t, slew_rate=np.inf,
                           ste_flip_angles=(90.0, 90.0, 90.0))}


def _protocol(name, n_t):
    rng = np.random.default_rng(5)
    u = rng.normal(size=(4, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    b = np.array([1e9, 2e9, 3e9, 3e9, 0.0]); dirs = np.concatenate([u, [[0, 0, 1]]])
    if name == "se":
        seq = d.pgse(dirs.tolist(), DELTA, BIG_DELTA, bvalues=b, n_t=n_t, slew_rate=np.inf)
    else:
        seq = d.pgste(dirs.tolist(), DELTA, BIG_DELTA - DELTA, bvalues=b, n_t=n_t, slew_rate=np.inf, ste_flip_angles=(90.0, 90.0, 90.0))
    return b, dirs, seq


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
    voxel, the merged pack replayed on the acquisition the machine plays there, for a spin echo and a stimulated echo,
    to float32 arithmetic; and the delivery moves the image off the field-only one by far more than that."""
    sm, col, merged, grid, n_t, shapes, tmp = layout
    sc = ScannerLimits.of(scanner)
    placed = sm.grid.centred_at(OFFSET)
    live = sm.live_voxels
    ijk = np.stack(np.unravel_index(live, tuple(grid.shape)), axis=1)
    for name in ("se", "ste"):
        b, dirs, seq = _protocol(name, n_t)
        dl = sm.delivery(name, b, dirs, sc, placed)
        assert (dl.g0 is not None) == sc.has_field_law and (dl.kappa is not None) == sc.has_transmit_profile
        S, _ = sm.image(name, b, dirs, tissue=TISSUE, scanner=sc, b0_direction=sc.b0_axis, delivered=dl)
        S0, _ = sm.image(name, b, dirs, tissue=TISSUE, scanner=sc, b0_direction=sc.b0_axis)
        ref = _direct(merged, grid, placed, sc, seq, ijk)
        got = S.reshape(-1, len(b))[live]
        err = np.abs(got - ref).max()
        assert err < 5e-6, (scanner, name, err)                    # float32 against float64: measured 0.5-1.2e-7
        moved = np.abs(got - S0.reshape(-1, len(b))[live]).max()
        assert moved > 100 * err and moved > 1e-4, (scanner, name, moved, err)


def test_the_torch_backend_is_the_jax_image_under_a_delivery(layout):
    """The delivered kernel on torch (the shared GPU pool's backend) equals the JAX one, tiers and background on."""
    pytest.importorskip("torch")
    sm, col, merged, grid, n_t, shapes, tmp = layout
    sc = ScannerLimits.of("swoop")
    b, dirs, _ = _protocol("ste", n_t)
    dl = sm.delivery("ste", b, dirs, sc, sm.grid.centred_at(OFFSET))
    S, _ = sm.image("ste", b, dirs, tissue=TISSUE, scanner=sc, b0_direction=sc.b0_axis, delivered=dl)
    St, _ = sm.image("ste", b, dirs, tissue=TISSUE, scanner=sc, b0_direction=sc.b0_axis, delivered=dl, backend="torch", device="cpu")
    assert np.nanmax(np.abs(St - S)) < 5e-6


def test_each_term_reaches_the_image(layout):
    """On the Swoop at 7.1 cm the nonlinearity, the background and the Maxwell term each move the image alone, and the
    transmit scale multiplies it by the pathway amplitude's ratio exactly (sin(k 90) sin^2(k 90) for the spin echo)."""
    sm, col, merged, grid, n_t, shapes, tmp = layout
    sc = ScannerLimits.of("swoop")
    placed = sm.grid.centred_at(OFFSET)
    b, dirs, _ = _protocol("se", n_t)
    off = dict(nonlinearity=False, background=False, concomitant=False, transmit=False)
    base, _ = sm.image("se", b, dirs, delivered=sm.delivery("se", b, dirs, sc, placed, **off))
    S0, _ = sm.image("se", b, dirs)
    assert np.nanmax(np.abs(base - S0)) < 1e-6
    for term in ("nonlinearity", "background", "concomitant"):
        S, _ = sm.image("se", b, dirs, delivered=sm.delivery("se", b, dirs, sc, placed, **{**off, term: True}))
        assert np.nanmax(np.abs(S - S0)) > 1e-5, term
    dl = sm.delivery("se", b, dirs, sc, placed, **{**off, "transmit": True})
    S, _ = sm.image("se", b, dirs, delivered=dl)
    k = dl.kappa
    ratio = np.sin(np.pi / 2 * k) ** 3
    live = sm.live_voxels
    np.testing.assert_allclose(S.reshape(-1, len(b))[live], S0.reshape(-1, len(b))[live] * ratio[:, None], rtol=1e-6)


def test_a_shape_with_ramps_is_refused_for_the_maxwell_term(layout):
    """A trapezoid's Maxwell gradient goes as the square of its ramps, not as the shape: not one vector on the shape's
    moment, so it is refused by name; without the Maxwell term the same shape is accepted."""
    sm, col, merged, grid, n_t, shapes, tmp = layout
    sc = ScannerLimits.of("prisma")
    profile = np.array([0.0, 0.5, 1.0, 1.0, 0.5])
    with pytest.raises(ValueError, match="two-valued"):
        delivered_moments(sc, sm.grid.centred_at(OFFSET), profile, [0.05], [[0.0, 0.0, 1.0]], voxels=[[0, 0, 0]])
    delivered_moments(sc, sm.grid.centred_at(OFFSET), profile, [0.05], [[0.0, 0.0, 1.0]], voxels=[[0, 0, 0]], concomitant=False)


def test_a_layout_without_background_refuses_a_magnet_gradient_and_the_stamp_adds_it(layout):
    """A layout written without the background column refuses the Swoop's own gradient by name; ``stamp_background``
    adds it in one pass with every other column untouched, and the stamped layout's image is the one written with it."""
    sm, col, merged, grid, n_t, shapes, tmp = layout
    out = str(tmp / "dm_plain")
    write_shape_moments(col, shapes, out, tol=1e-9, chunk_rows=7, tiers=True)
    plain = ShapeMoments(out)
    assert plain.background is None
    sc = ScannerLimits.of("swoop")
    placed = plain.grid.centred_at(OFFSET)
    b, dirs, _ = _protocol("se", n_t)
    dl = plain.delivery("se", b, dirs, sc, placed)
    with pytest.raises(ValueError, match="background"):
        plain.image("se", b, dirs, delivered=dl)
    before = {f: open(os.path.join(out, f), "rb").read() for f in os.listdir(out) if f.endswith(".npy")}
    with pytest.raises(ValueError, match="not the one"):
        stamp_background(out, col, {**shapes, "se": d.pgse([[0, 0, 1]], DELTA, 0.4e-3, gradient_strengths=0.05, n_t=n_t, slew_rate=np.inf)})
    stamp_background(out, col, shapes)
    after = {f: open(os.path.join(out, f), "rb").read() for f in before}
    assert before == after                                          # nothing written but the new columns
    stamped = ShapeMoments(out)
    for name in ("se", "ste"):
        np.testing.assert_array_equal(np.asarray(stamped._column(f"bg_{stamped.background_group(name)}")),
                                      np.asarray(sm._column(f"bg_{sm.background_group(name)}")))
    S1, _ = stamped.image("se", b, dirs, delivered=dl)
    S2, _ = sm.image("se", b, dirs, delivered=sm.delivery("se", b, dirs, sc, placed))
    np.testing.assert_array_equal(S1, S2)
    assert stamped.sequence("ste").stimulated_echo and transmit_amplitude(stamped.sequence("ste"), [1.0])[0] == pytest.approx(0.5)


def test_a_magnets_gradient_through_a_stimulated_echo_accrues_nothing_while_stored():
    """The effective gradient is the coherence sign times the transverse gate: through the mixing time the
    magnetisation is along z and a magnet's constant gradient adds no phase, so the background of a stimulated echo
    is refocused like a spin echo's (to one sample per change of the gate) and not left as a winding."""
    from dmipy_sim.replay.shape_moments import _background_sequence
    ste = d.pgste([[0, 0, 1]], 7.6e-3, 38.3e-3, gradient_strengths=0.1, n_t=1000, slew_rate=np.inf, ste_flip_angles=(90.0, 90.0, 90.0))
    bg = _background_sequence(ste)
    G = np.asarray(bg.G_eff)[0, :, 0]
    np.testing.assert_array_equal(G, np.asarray(ste.effective_gate))          # the background's time course IS the gate
    stored = np.asarray(ste.effective_gate) == 0.0
    assert stored.sum() > 600 and not bg.unbalanced
    assert set(np.unique(G[~stored])) == {-1.0, 1.0}
