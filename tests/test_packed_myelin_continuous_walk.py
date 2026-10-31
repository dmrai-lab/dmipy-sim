"""A walk of a periodic packed substrate records the CONTINUOUS path (its raw subset's block ends, and the bands a pack
stores): the gradient phase of a replay is ``gamma * integral G . r dt`` on the lab position, and a path folded into
the cell adds a jump of one cell side to every walker that crosses a face. The field tier of such a walk reads its
periodic grid wrapped, so a walker outside the primary cell sees the field of its image."""
import dataclasses

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim import ScannerSequence
from dmipy_sim.fields.susceptibility_field import FieldGrid, field_grid_of
from dmipy_sim.replay.bank import build_replay_pack, grid_periodic_of
from dmipy_sim.replay.trajectories import replay
from dmipy_replay.spec.tissue import Tissue
from tests.conftest import spec_without_source
from tests.replay_frames import field_along
from tests.test_grid_field_in_walk import _modes_of

D0 = 2e-9
L = 10e-6
K_BANDS = 38                      # bands per window: two fewer than the 40 blocks of the walk
ENV = dict(bvals=[0.0, 1e9], dirs=[[1, 0, 0]], ogse_periods=[2], shortd_b=1e9, shortd_deltas_frac=[0.05],
           B0_list=[3.0], theta_deg=[90], delta_frac=0.2, Delta_frac=0.5, rho2_list=[1e-5])


def _cell():
    _, _, c = d.pack_myelinated_cylinders([1e-6] * 3, 0.7, None, cell_size=L, seed=0)
    return np.asarray(c)


def _fold(pos):
    """The positions folded into the cell ``[-L/2, L/2)`` in-plane: the in-cell convention."""
    out = np.array(pos, np.float32, copy=True)
    out[..., :2] -= np.float32(L) * np.floor(out[..., :2] / np.float32(L) + np.float32(0.5))
    return out


def _unwrap(folded):
    """The folded positions made continuous again in-plane: each block's displacement taken as its minimal image (a
    step is far shorter than half the cell) and re-integrated from the start."""
    out = np.array(folded, np.float64, copy=True)
    dx = np.diff(out[..., :2], axis=-2)
    dx -= L * np.round(dx / L)
    out[..., 1:, :2] = out[..., :1, :2] + np.cumsum(dx, axis=-2)
    return out


def _jumps(pos):
    return int((np.abs(np.diff(np.asarray(pos)[..., :2], axis=-2)) > L / 2).sum())


@pytest.fixture(scope="module")
def pm():
    return d.PackedMyelinatedCylinders(
        [1e-6] * 3,
        0.7,
        _cell(),
        L,
        N_max=4,
        compartments=d.Compartments(intra=d.Pool(D=D0), extra=d.Pool(D=D0), myelin=d.Pool(D=0.0)),
    )


@pytest.fixture(scope="module")
def cwalk(pm):
    w = _walk_geometry(300, D0, pm, 8e-3, 2e-4, seed=0, require_gpu=False, raw_subset=300, K=K_BANDS)
    # this fixture is about the continuous path, not the susceptibility tier: the geometry's spec declares one
    # (dmipy-sim#539 -- a walk without field_basis/field_coeffs of a source-declaring spec is refused a pack)
    return dataclasses.replace(w, spec=spec_without_source(w.spec))


def _assert_continuous(pos):
    pos = np.asarray(pos)
    folded = _fold(pos)
    assert _jumps(folded) > 50, "too few walkers crossed a cell face for the check to mean anything"
    assert _jumps(pos) == 0, "a saved in-plane step is longer than half the cell: the path is folded"
    ref = _unwrap(folded)                                     # the folded convention, unwrapped after the fact,
    ref = ref + (pos[:, :1, :] - ref[:, :1, :])               # from the walk's own start (a seed on a face folds)
    np.testing.assert_allclose(pos, ref, rtol=0, atol=1e-10)
    e2e = lambda p: np.var(p[:, -1, :] - p[:, 0, :], axis=0)
    np.testing.assert_allclose(e2e(pos), e2e(ref), rtol=1e-5)


def test_a_packed_myelinated_walk_records_the_continuous_path(cwalk):
    _assert_continuous(cwalk.raw_positions)


def test_a_packed_myelinated_walk_with_binding_records_the_continuous_path():
    pm = d.PackedMyelinatedCylinders(
        [1e-6] * 3,
        0.7,
        _cell(),
        L,
        N_max=4,
        compartments=d.Compartments(intra=d.Pool(D=D0), extra=d.Pool(D=D0), myelin=d.Pool(D=0.0)),
    )
    w = _walk_geometry(300, D0, pm, 8e-3, 2e-4, seed=1, require_gpu=False, kappa_MT=5e-5, dwell_time=1e-3,
                                equilibrate_binding="off", raw_subset=300)
    assert w.has_binding
    _assert_continuous(w.raw_positions)


def test_a_packed_cylinders_walk_records_the_continuous_path():
    pc = d.PackedCylinders(np.array([1e-6] * 3), _cell(), L)
    w = _walk_geometry(300, D0, pc, 8e-3, 2e-4, seed=0, require_gpu=False, tiers="all", raw_subset=300)
    _assert_continuous(w.raw_positions)


def test_a_pack_replays_the_in_plane_gradient_of_the_continuous_path(cwalk):
    """A pack of the walk (the bands of the continuous path) replays an in-plane PGSE as the direct replay of the
    continuous trajectory at the block ends does; the folded trajectory replays something else (the jumps act as
    displacements)."""
    n_t = cwalk.raw_positions.shape[1]
    pk = _float_pack(cwalk, id="test/pm-continuous", envelope=ENV, license="x", citation="x")
    assert pk.n_t == n_t and pk.dt == pytest.approx(cwalk.h)
    G = np.zeros((1, n_t, 3)); G[0, 2:12, 0] = 0.4; G[0, 22:32, 0] = -0.4
    S_pack = pk.replay(ScannerSequence(G=G, dt=cwalk.h), tol=None)[0]
    S_direct = replay(cwalk.raw_positions, cwalk.h, G, cwalk.h)[0]
    S_folded = replay(_fold(cwalk.raw_positions), cwalk.h, G, cwalk.h)[0]
    assert abs(S_pack - S_direct) < 5e-4          # the bands' series against the path through the saves (3.4e-4 measured)
    assert abs(S_folded - S_direct) > 20 * abs(S_pack - S_direct) and abs(S_folded - S_direct) > 0.01


def test_the_field_grid_of_a_packed_cell_is_read_periodically(pm, cwalk):
    """The grid is one period of the field: a point and its images read the same field, through the grid and
    through the walk's own read -- a walker outside the primary cell has the modes of its image's field, and its
    pack's field tier replays them; a single cylinder's grid is a box and declares no period."""
    fg = field_grid_of(pm, res=0.2e-6, include_aniso=False)
    assert fg.periodic == (True, True, True)
    assert field_grid_of(d.MyelinatedCylinder(1e-6, 1.4e-6, (0, 0, 1), compartments=d.Compartments(intra=d.Pool(D=D0), extra=d.Pool(D=D0), myelin=d.Pool(D=0.0))), res=0.2e-6).periodic == (False,) * 3
    p = _fold(cwalk.raw_positions[:, -1, :]).astype(float)
    base = fg.channels(p)
    for shift in ([L, 0, 0], [0, -L, 0], [-2 * L, 3 * L, 7e-6]):
        np.testing.assert_allclose(fg.channels(p + np.array(shift)), base, rtol=0, atol=1e-9 * np.abs(base).max())
    clamped = FieldGrid(fg.basis, fg.origin)                                   # the same grid read without its period
    assert np.abs(clamped.channels(p + np.array([L, 0, 0])) - base).max() > 0.1 * np.abs(base).max()

    # the walk reads the grid at every step (one step per block: the reads are the block ends of its raw subset); its
    # modes are the trapezoid projection of the reads at the FOLDED positions, which are the images' reads
    w = _walk_geometry(300, D0, pm, 8e-3, 2e-4, seed=0, require_gpu=False, raw_subset=300, K=K_BANDS,
                                sub_steps=1, field_basis=fg)
    pos = np.asarray(w.raw_positions, np.float64)
    assert _jumps(pos) == 0 and _jumps(_fold(pos)) > 50                         # the walkers do leave the cell
    modes = np.asarray(w.field_coeffs, np.float64)[:, 0]
    ref = _modes_of(fg, _fold(pos), w.dt_sim, w.T_window, w.K_c3)
    assert np.abs(modes - ref).max() <= 5e-3 * np.abs(ref).max()
    wrapped_wrong = _modes_of(clamped, pos, w.dt_sim, w.T_window, w.K_c3)    # the unwrapped path on a clamped grid: another field
    assert np.abs(wrapped_wrong - ref).max() > 0.01 * np.abs(ref).max()

    pk = _float_pack(w, id="test/pm-field", envelope=ENV, license="x", citation="x")
    gm = pk.meta["compression"]["channels"]["susceptibility_grid"]
    assert gm["periodic"] == [True, True, True] and grid_periodic_of(gm) == (True, True, True)
    G0 = ScannerSequence(G=np.zeros((1, pk.n_t, 3)), dt=pk.dt)
    G_lab, R = field_along(G0, (1, 0, 0))
    S = pk.replay(G_lab, orientation=R, scanner=3.0, tissue=Tissue(chi_iso=1e-6), tol=None)[0]
    assert S < 0.99                                                             # the field dephases


def _float_pack(*a, **k):
    """``build_replay_pack`` in the float containers: this module compares packs and replays to the bit, which the
    derived integer default does not promise."""
    from dmipy_sim.replay.bank import build_replay_pack
    k.setdefault("position_container", None); k.setdefault("blt_container", None); k.setdefault("field_container", None)
    return build_replay_pack(*a, **k)
