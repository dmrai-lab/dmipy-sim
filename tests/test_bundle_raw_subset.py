"""A bundle walk keeps a raw subset (the pools' joined walk, ``walk_spec(raw_subset=n)``): the first ``n`` walkers of
the joined order carry raw block ends, drawn from the pools in proportion to their walkers, and every raw array is the
walk's own for those walkers -- the diffusing pools' from their walks, a frozen pool's its seats held at every block
end. A pool walk that lacks a raw array it was asked for is refused by name."""
import numpy as np
import pytest
from scipy.fft import dst

import dmipy_sim as d
import dmipy_sim.engine.core as core
from dmipy_sim.replay import compression as cx
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec import SpecError, walk_spec
from dmipy_sim.spec.walk import _raw_allocation

D = 2e-9
T_MAX, H, K = 2e-3, 1e-4, 12                 # 20 blocks; K below the N - 1 = 19 bands the block ends hold
N, N_RAW = 600, 200
EXTRA, INTRA, MYELIN = 0, 1, 2          # the pool ids of the CACTUS spec


def _spec(tmp):
    """A CACTUS-shaped bundle: two capped myelinated tubes (inner and outer meshes) along z in a periodic 6 um cell;
    pools extra, intra and a frozen myelin sheath whose field is read in the walk."""
    import os
    import trimesh
    from dmipy_sim.spec import cactus_spec
    sim = tmp / "meshes" / "simulations"
    os.makedirs(sim)
    side = 6.0
    for i, (cx, cy) in enumerate(((-0.25 * side, -0.2 * side), (0.25 * side, 0.2 * side))):
        for kind, r in (("inner", 1.0), ("outer", 1.4)):
            m = trimesh.creation.cylinder(radius=r, height=1.6 * side, sections=48)
            m.apply_translation([cx, cy, 0.0])
            m.export(str(sim / f"strand_{i:05d}_{kind}_erode_0.ply"))
    return cactus_spec(str(tmp), side_um=side)


def _reference_coeffs(raw, h, K):
    """``(n_w, K + 2, 3)`` float64: start, displacement and the ``K`` bands of the paths through the block ends ``raw``
    ``(n_w, N + 1, 3)`` by the ``m = 0`` exactness relation (``tests/test_readers_against_the_walk.py``)."""
    x = np.asarray(raw, np.float64)
    n = x.shape[1] - 1
    tau = np.arange(n + 1) / n
    u = x - (x[:, :1] + (x[:, -1:] - x[:, :1]) * tau[None, :, None])
    k = np.arange(1, K + 1)
    factor = cx.cosine_band_scale(K, n * h) * 2.0 * np.sin(np.pi * k / (2.0 * n)) / np.sqrt(2.0 / n)
    beta = factor[None, :, None] * dst(u[:, 1:-1], type=1, norm="ortho", axis=1)[:, :K]
    return np.concatenate([x[:, :1], x[:, -1:] - x[:, :1], beta], axis=1)


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    return _spec(tmp_path_factory.mktemp("cactus"))


@pytest.fixture(scope="module")
def bundle(spec):
    """An order-0 walk of the bundle with ``N_RAW`` raw walkers."""
    return walk_spec(spec, N, T_max=T_MAX, block_h=H, K=K, block_order=0, seed=3, require_gpu=False, raw_subset=N_RAW)


def test_the_raw_arrays_have_the_walks_shapes_and_lead_the_order(bundle):
    w = bundle
    nb1 = w.n_blocks + 1
    assert w.position_coeffs.shape[0] == N and w.raw_positions.shape == (N_RAW, nb1, 3)
    assert w.raw_boundary.shape == (N_RAW, nb1) and w.raw_compartment.shape == (N_RAW, nb1) and w.raw_compartment.dtype == np.int8
    assert w.raw_bound is None
    assert w.raw_field.shape == (N_RAW, len(w.field_read_instants), w.field_coeffs.shape[3])
    np.testing.assert_array_equal(w.raw_positions[:, 0], w.position_coeffs[:N_RAW, 0, 0])
    np.testing.assert_allclose(w.raw_positions[:, -1], w.position_coeffs[:N_RAW, 0, 0] + w.position_coeffs[:N_RAW, 0, 1],
                               rtol=0, atol=1e-12)


def test_the_raw_walkers_coefficients_are_the_band_projection_of_their_raw_positions(bundle):
    """Diffusing and frozen walkers alike: at order 0 the walk's bands ARE the block ends' projection."""
    w = bundle
    ref = _reference_coeffs(w.raw_positions, w.h, K)
    got = np.asarray(w.position_coeffs[:N_RAW, 0], np.float64)
    assert np.abs(got[:, 2:] - ref[:, 2:]).max() <= 2e-6 * np.abs(ref[:, 2:]).max()
    np.testing.assert_allclose(got[:, :2], ref[:, :2], rtol=0, atol=1e-6 * np.abs(ref[:, 0]).max())


def test_the_raw_compartment_is_the_pool_id_of_the_joined_walkers(bundle):
    w = bundle
    np.testing.assert_array_equal(w.raw_compartment, np.repeat(np.asarray(w.compartment)[:N_RAW, None], w.n_blocks + 1, axis=1))
    assert set(np.unique(w.raw_compartment)) == {EXTRA, INTRA, MYELIN}


def test_the_raw_subset_is_drawn_from_the_pools_in_proportion_to_their_walkers(bundle):
    comp = np.asarray(bundle.compartment)
    for pid in (EXTRA, INTRA, MYELIN):
        share = (comp == pid).sum() * N_RAW / N
        assert abs((comp[:N_RAW] == pid).sum() - share) <= 1.0
    assert (comp[:N_RAW] == MYELIN).any()                       # the frozen pool is in the subset by the same rule


def test_a_frozen_walker_is_held_at_its_seat_in_every_raw_array(bundle):
    w = bundle
    i = np.flatnonzero(np.asarray(w.raw_compartment[:, 0]) == MYELIN)
    np.testing.assert_array_equal(w.raw_positions[i], np.broadcast_to(w.raw_positions[i, :1], w.raw_positions[i].shape))
    assert (w.raw_boundary[i] == 0).all()
    assert (w.raw_field[i] == w.raw_field[i, :1]).all()
    np.testing.assert_allclose(w.field_coeffs[i, 0, 0], np.sqrt(w.T_window) * w.raw_field[i, 0], rtol=1e-6)


def test_the_raw_contact_sums_to_the_walkers_boundary_endpoint(bundle):
    """The contact per block of a raw walker totals the window's boundary local time its coefficients carry (zero for a
    frozen walker)."""
    w = bundle
    total = np.asarray(w.raw_boundary, np.float64).sum(axis=1)
    np.testing.assert_allclose(total, np.asarray(w.boundary_endpoint[:N_RAW, 0], np.float64), rtol=1e-4, atol=1e-9)
    assert (total[np.asarray(w.raw_compartment[:, 0]) != MYELIN] != 0).any()


def test_the_raw_subset_is_clipped_to_the_walkers_and_absent_when_not_asked(spec):
    w = walk_spec(spec, 60, T_max=T_MAX, block_h=H, K=K, seed=3, require_gpu=False, raw_subset=1000)
    assert w.raw_positions.shape[0] == w.n_walkers and w.raw_field.shape[0] == w.n_walkers
    w = walk_spec(spec, 60, T_max=T_MAX, block_h=H, K=K, seed=3, require_gpu=False)
    assert w.raw_positions is None and w.raw_boundary is None and w.raw_compartment is None and w.raw_field is None


def test_the_allocation_is_proportional_exact_and_within_each_pool():
    assert _raw_allocation({0: 500, 1: 300, 2: 200}, 100) == {0: 50, 1: 30, 2: 20}
    a = _raw_allocation({0: 3, 1: 3, 2: 1}, 5)
    assert sum(a.values()) == 5 and all(a[i] <= n for i, n in ((0, 3), (1, 3), (2, 1)))
    assert _raw_allocation({0: 4, 1: 6}, 100) == {0: 4, 1: 6}
    assert _raw_allocation({0: 4, 1: 6}, 0) == {0: 0, 1: 0}


def test_the_joined_raw_subset_measures_a_pack(bundle):
    pk = build_replay_pack(bundle, id="t/bundle-raw", license="x", citation="x", K=K, fidelity="measured")
    assert pk.fidelity["certified"] == "measured" and pk.fidelity["measured_by"] == "raw_subset" and pk.fidelity["measured_on"] == N_RAW


def test_a_pool_walk_that_lost_its_raw_subset_is_refused_by_name(spec, monkeypatch):
    import dataclasses
    real = core.simulate_trajectories

    def stripped(*a, **k):
        return dataclasses.replace(real(*a, **k), raw_positions=None)
    monkeypatch.setattr(core, "simulate_trajectories", stripped)
    with pytest.raises(SpecError, match=r"pool '(intra|extra)'.*raw_positions"):
        walk_spec(spec, 60, T_max=T_MAX, block_h=H, K=K, seed=3, require_gpu=False, raw_subset=30)


def test_an_adaptive_strand_bundle_joins_its_raw_subset_too(tmp_path):
    """The adaptive producer's pools (strand field read in the walk) join on the same terms."""
    from tests.conftest import _three_strands
    w = walk_spec(_three_strands(tmp_path), 300, 8e-4, 2e-4, seed=0, n_probe=20_000, require_gpu=False, adaptive_steps=True,
                  K=4, block_order=0, raw_subset=100)
    comp = np.asarray(w.compartment)
    assert w.raw_positions.shape == (100, w.n_blocks + 1, 3) and w.raw_field.shape[:2] == (100, len(w.field_read_instants))
    np.testing.assert_array_equal(w.raw_compartment, np.repeat(comp[:100, None], w.n_blocks + 1, axis=1))
    np.testing.assert_array_equal(w.raw_positions[:, 0], w.position_coeffs[:100, 0, 0])
    for pid in np.unique(comp):
        assert abs((comp[:100] == pid).sum() - (comp == pid).sum() / 3) <= 1.0
