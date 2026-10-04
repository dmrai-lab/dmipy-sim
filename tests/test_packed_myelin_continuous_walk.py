"""A persistent walk of a periodic packed substrate records the CONTINUOUS path: the gradient phase of a replay is
``gamma * integral G . r dt`` on the lab position, and a path folded into the cell adds a jump of one cell side to
every walker that crosses a face. The field tier of such a pack reads its periodic grid wrapped, so a walker
outside the primary cell sees the field of its image."""
import dataclasses

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import ScannerSequence
from dmipy_sim.fields.susceptibility_field import FieldGrid, field_grid_of
from dmipy_sim.persistent_walk import PersistentWalk
from dmipy_sim.spec.walk import fill_field
from dmipy_sim.replay.bank import build_replay_pack, grid_periodic_of
from dmipy_sim.replay.trajectories import replay, unwrap_periodic
from dmipy_sim.spec.tissue import Tissue
from tests.conftest import spec_without_source
from tests.replay_frames import field_along

D0 = 2e-9
L = 10e-6
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


def _jumps(pos):
    return int((np.abs(np.diff(np.asarray(pos)[..., :2], axis=-2)) > L / 2).sum())


@pytest.fixture(scope="module")
def pm():
    return d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, _cell(), L, N_max=4, D_intra=D0, D_extra=D0)


@pytest.fixture(scope="module")
def walk(pm):
    import dataclasses
    w = d.simulate_trajectories(300, D0, pm, 8e-3, 2e-4, seed=0, require_gpu=False)
    # this fixture is about the continuous path, not the susceptibility tier: the geometry's spec declares one
    # (dmipy-sim#539 -- a walk without field_basis/field_samples of a source-declaring spec is refused a pack)
    return dataclasses.replace(w, spec=spec_without_source(w.spec))


def _assert_continuous(pos):
    pos = np.asarray(pos)
    folded = _fold(pos)
    assert _jumps(folded) > 50, "too few walkers crossed a cell face for the check to mean anything"
    assert _jumps(pos) == 0, "a saved in-plane step is longer than half the cell: the path is folded"
    ref = unwrap_periodic(folded, L)                          # the folded convention, unwrapped after the fact,
    ref = ref + (pos[:, :1, :] - ref[:, :1, :])               # from the walk's own start (a seed on a face folds)
    np.testing.assert_allclose(pos, ref, rtol=0, atol=1e-10)
    e2e = lambda p: np.var(p[:, -1, :] - p[:, 0, :], axis=0)
    np.testing.assert_allclose(e2e(pos), e2e(ref), rtol=1e-5)


def test_a_packed_myelinated_walk_records_the_continuous_path(walk):
    _assert_continuous(walk.positions)


def test_a_packed_myelinated_walk_with_binding_records_the_continuous_path():
    pm = d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, _cell(), L, N_max=4, D_intra=D0, D_extra=D0)
    w = d.simulate_trajectories(300, D0, pm, 8e-3, 2e-4, seed=1, require_gpu=False, kappa_MT=5e-5, dwell_time=1e-3,
                                equilibrate_binding="off")
    assert w.has_binding
    _assert_continuous(w.positions)


def test_a_packed_cylinders_walk_records_the_continuous_path():
    pc = d.PackedCylinders(np.array([1e-6] * 3), _cell(), L)
    w = d.simulate_trajectories(300, D0, pc, 8e-3, 2e-4, seed=0, require_gpu=False, tiers="all")
    _assert_continuous(w.positions)


def test_a_pack_replays_the_in_plane_gradient_of_the_continuous_path(walk):
    """A lossless pack of the walk replays an in-plane PGSE as the direct replay of the continuous trajectory does;
    the folded trajectory replays something else (the jumps act as displacements)."""
    n_t = walk.positions.shape[1]
    pk = build_replay_pack(walk, id="test/pm-continuous", K=n_t - 2, envelope=ENV, license="x",
                           citation="x")
    G = np.zeros((1, n_t, 3)); G[0, 2:12, 0] = 0.4; G[0, 22:32, 0] = -0.4
    S_pack = pk.replay(ScannerSequence(G=G, dt=walk.dt))[0]
    S_direct = replay(walk.positions, walk.dt, G, walk.dt)[0]
    S_folded = replay(_fold(walk.positions), walk.dt, G, walk.dt)[0]
    assert abs(S_pack - S_direct) < 1e-4
    assert abs(S_folded - S_direct) > 100 * abs(S_pack - S_direct) and abs(S_folded - S_direct) > 0.01


def test_the_field_grid_of_a_packed_cell_is_read_periodically(pm, walk):
    """The grid is one period of the field: a point and its images read the same field, through the grid and
    through a pack's field tier; a single cylinder's grid is a box and declares no period."""
    fg = field_grid_of(pm, res=0.2e-6, include_aniso=False)
    assert fg.periodic == (True, True, True)
    assert field_grid_of(d.MyelinatedCylinder(1e-6, 1.4e-6, (0, 0, 1), D0, D0), res=0.2e-6).periodic == (False,) * 3
    p = _fold(walk.positions[:, -1, :]).astype(float)
    base = fg.channels(p)
    for shift in ([L, 0, 0], [0, -L, 0], [-2 * L, 3 * L, 7e-6]):
        np.testing.assert_allclose(fg.channels(p + np.array(shift)), base, rtol=0, atol=1e-9 * np.abs(base).max())
    clamped = FieldGrid(fg.basis, fg.origin)                                   # the same grid read without its period
    assert np.abs(clamped.channels(p + np.array([L, 0, 0])) - base).max() > 0.1 * np.abs(base).max()

    n_t = walk.positions.shape[1]
    folded = PersistentWalk(_fold(walk.positions), walk.dt, walk.sub_steps, walk.dt_sim,
                            boundary_local_time=walk.boundary_local_time, compartment=walk.compartment,
                            seed=walk.seed, diffusivity=walk.diffusivity, geometry=pm)
    packs = [build_replay_pack(fill_field(w, fg), id="test/pm-field", K=n_t - 2, envelope=ENV, license="x", citation="x")
             for w in (dataclasses.replace(walk, spec=pm.spec), folded)]   # the field source's own spec
    gm = packs[0].meta["compression"]["channels"]["susceptibility_grid"]
    assert gm["periodic"] == [True, True, True] and grid_periodic_of(gm) == (True, True, True)
    G0 = ScannerSequence(G=np.zeros((1, n_t, 3)), dt=walk.dt)
    G_lab, R = field_along(G0, (1, 0, 0))
    S = [pk.replay(G_lab, orientation=R, scanner=3.0, tissue=Tissue(chi_iso=1e-6))[0] for pk in packs]
    assert S[0] < 0.99                                                          # the field dephases
    assert S[0] == pytest.approx(S[1], abs=1e-6)                                # every walker reads its image's field
