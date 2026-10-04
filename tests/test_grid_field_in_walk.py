"""A gridded susceptibility field (:class:`FieldGrid`: a mesh sheath, a myelinated cylinder, a packed cell) is sampled
BY THE WALK: :func:`simulate_trajectories` with ``field_basis=`` reads the grid's channels at every sub-step position
inside the step and keeps their mean over each save interval (sample 0 the start position's), on the field's own
save grid (``field_sample_every``), as the strand route does with ``adaptive_steps=True``. ``walk_spec`` samples
every gridded source this way, and the pack's path channel is encoded from those samples and certified against
them."""
import dataclasses

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.fields.susceptibility_field import FieldGrid, field_grid_of
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec import walk_spec
from tests.conftest import spec_without_source

D0 = 2e-9
UM = 1e-6
N_SUB = 4                                          # sub-steps per save of the walk that samples in the walk


def _dipole_grid(periodic=False):
    """Seven channels of a softened point dipole at the origin, rasterised on a 24^3 grid over [-3, 3] um: the
    ``iso_local`` channel ``1 / (r^2 + a^2)^(3/2)`` and the six ``(3 x_i x_j - r^2 delta_ij) / (r^2 + a^2)^(5/2)``."""
    n, half, a = 24, 3 * UM, 1 * UM
    vs = 2 * half / n
    x = -half + (np.arange(n) + 0.5) * vs
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)
    r2 = (X ** 2).sum(-1)
    s = (r2 + a * a)
    iso_local = a ** 3 / s ** 1.5
    pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    iso_P = np.stack([a ** 3 * (3 * X[..., i] * X[..., j] - (i == j) * r2) / s ** 2.5 for i, j in pairs])
    basis = dict(iso_local=iso_local, iso_P=iso_P, aniso_G=None, shape=(n, n, n), voxel_size=np.full(3, vs))
    return FieldGrid(basis, np.full(3, -half), periodic=(periodic,) * 3)


def _packed():
    _, _, c = d.pack_myelinated_cylinders([1e-6] * 3, 0.7, None, cell_size=10 * UM, seed=0)
    return d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, np.asarray(c), 10 * UM, N_max=4, D_intra=D0, D_extra=D0)


def _interval_means(fg, fine, n_sub):
    """The rule, read after the walk: the grid at every sub-step position of a walk that saved every sub-step
    (``fine``, ``dt_save = dt_sim``), averaged over each group of ``n_sub``; sample 0 the start position's."""
    n_w = fine.shape[0]
    ch = fg.channels(np.asarray(fine, np.float64).reshape(-1, 3)).reshape(n_w, fine.shape[1], -1)
    return np.concatenate([ch[:, :1], ch[:, 1:].reshape(n_w, -1, n_sub, ch.shape[-1]).mean(2)], axis=1)


CASES = {
    # (geometry, field grid, tiers): the three trajectory steps that read the grid
    "sphere-relax": lambda: (d.Sphere(2.5 * UM), _dipole_grid(), "all"),
    "sphere-positions": lambda: (d.Sphere(2.5 * UM), _dipole_grid(), ()),
    "packed-myelin-periodic": lambda: (_packed(), field_grid_of(_packed(), res=0.1 * UM), "all"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_in_walk_sample_is_the_interval_mean_of_the_grid_at_the_sub_steps(case):
    """(a) The walk at ``N_SUB`` sub-steps per save, sampling in the walk, against the same walk saved at every
    sub-step and the grid read afterwards: the positions are the same walk (to float32 rounding: the two are
    different programs, and XLA may round a fused step differently), and every sample is the mean of the grid over
    its interval's sub-step positions, to float32 rounding (the device reads in float32 at the position the step
    kernel holds -- the in-cell one of a packed cell, whose grid is periodic)."""
    geom, fg, tiers = CASES[case]()
    T, dt = 2e-3, 2e-4
    w = d.simulate_trajectories(120, D0, geom, T, dt, seed=4, sub_steps=N_SUB, tiers=tiers, require_gpu=False,
                                field_basis=fg)
    fine = d.simulate_trajectories(120, D0, geom, T, dt / N_SUB, seed=4, sub_steps=1, tiers=tiers, require_gpu=False)
    np.testing.assert_allclose(w.positions, fine.positions[:, ::N_SUB], rtol=0, atol=1e-11)   # measured 2.3e-12 m
    assert w.field_basis is fg and w.field_sample_every == 1
    assert w.field_samples.shape == (120, w.n_t, 7 if fg.basis.get("aniso_G") is None else 13)
    ref = _interval_means(fg, fine.positions, N_SUB)
    scale = np.abs(ref).max()
    # measured: 4.9e-7 (sphere, clamped dipole grid) and 6.4e-6 (packed cell, read in-cell on the device against the
    # unwrapped position wrapped on the host) of the channel scale, which changes by 0.34-0.74 of it between saves
    assert np.abs(w.field_samples - ref).max() <= 2e-5 * scale


def test_field_sample_every_reads_every_that_many_th_interval_only():
    """(b) ``field_sample_every=2`` is the strand route's rule: the samples are the every-save walk's at saves 0, 2,
    4, ... -- each the mean over ITS save interval -- ``ceil(n_t / 2)`` of them; the walk itself is unchanged,
    and an odd number of intervals leaves the last one unread."""
    geom, fg = _packed(), field_grid_of(_packed(), res=0.1 * UM)
    kw = dict(seed=5, sub_steps=N_SUB, require_gpu=False, field_basis=fg)
    w1 = d.simulate_trajectories(80, D0, geom, 1.8e-3, 2e-4, **kw)
    w2 = d.simulate_trajectories(80, D0, geom, 1.8e-3, 2e-4, field_sample_every=2, **kw)
    assert w1.n_t == 10 and w2.field_sample_every == 2 and w2.field_samples.shape[1] == 5
    np.testing.assert_array_equal(w2.positions, w1.positions)
    np.testing.assert_array_equal(w2.boundary_local_time, w1.boundary_local_time)
    np.testing.assert_array_equal(w2.field_samples, w1.field_samples[:, ::2])
    with pytest.raises(ValueError, match="goes with field_basis"):
        d.simulate_trajectories(8, D0, geom, 1.8e-3, 2e-4, seed=5, require_gpu=False, field_sample_every=2)


def test_a_strand_basis_is_not_read_as_a_grid():
    with pytest.raises(TypeError, match="FieldGrid"):
        d.simulate_trajectories(8, D0, d.Sphere(2 * UM), 4e-4, 2e-4, require_gpu=False, field_basis=object())


ENV = dict(bvals=[0.0, 1e9], dirs=[[1, 0, 0]], ogse_periods=[2], shortd_b=1e9, shortd_deltas_frac=[0.05],
           B0_list=[3.0, 7.0], theta_deg=[0, 90], delta_frac=0.2, Delta_frac=0.5, rho_2_list=[1e-5])


@pytest.mark.parametrize("family", ["packed", "single"])
def test_a_myelinated_walk_carries_its_samples_and_the_pack_certifies_the_path_channel(family):
    """(c) ``walk_spec`` of a myelinated cylinder or a packed cell samples its grid in the walk; the default pack
    encodes the path channel from those samples (no grid route: the positions are lossy) and its certificate is
    within twice its floor."""
    geom = _packed() if family == "packed" else d.MyelinatedCylinder(1.0 * UM, 1.5 * UM, (0, 0, 1), D0, D0)
    w = walk_spec(geom.spec, 600, 5e-3, seed=2, require_gpu=False)
    assert isinstance(w.field_basis, FieldGrid) and w.field_samples is not None and w.field_fill is None
    assert w.field_basis.periodic == ((True,) * 3 if family == "packed" else (False,) * 3)
    assert w.field_samples.shape == (600, w.n_t, 13)
    pk = build_replay_pack(w, id=f"test/{family}-in-walk", license="x", citation="x", envelope=ENV)
    ch = pk.meta["compression"]["channels"]
    assert pk.has_field and ch["susceptibility_path"]["sampling"] == "interval_mean_in_walk"
    assert ch["susceptibility_grid"]["replay_route"] == "path" and not ch["susceptibility_grid"]["arrays_in_pack"]
    assert "susc_grid_iso_local" not in pk.arrays and "err_susc_se" not in pk.meta["fidelity"]
    f = pk.meta["fidelity"]
    assert f["err_susc_path"] <= 2.0 * f["floor_susc_path"]
    assert f["within_2x_floor"]


def test_a_mesh_bundle_samples_its_grid_in_the_walk_and_a_frozen_pool_holds_its_seat(tmp_path):
    """The pool-by-pool walk of a meshed sheath samples the raster in every diffusing pool's walk; the frozen myelin
    pool's samples are the channels at its seat, at every save."""
    pytest.importorskip("trimesh")
    from tests.test_mesh_spec import _bundle_spec
    w = walk_spec(_bundle_spec(tmp_path), 120, 1e-3, 2.5e-4, seed=0, n_probe=20_000, field_res=0.4e-6, require_gpu=False)
    assert isinstance(w.field_basis, FieldGrid) and w.field_samples.shape == (120, w.n_t, 13)
    ids = np.asarray(w.compartment)[:, 0]
    frozen = w.field_samples[ids == 2]
    np.testing.assert_array_equal(frozen, np.repeat(frozen[:, :1], w.n_t, axis=1))
    seat = np.asarray(w.field_basis.channels_at_device()(np.asarray(w.positions[ids == 2, 0], np.float32)))
    np.testing.assert_array_equal(frozen[:, 0], seat)
    assert not np.allclose(w.field_samples[ids != 2][:, 1:], w.field_samples[ids != 2][:, :1])   # the walkers move
    pk = build_replay_pack(w, id="test/bundle-in-walk", license="x", citation="x", K=8, envelope=ENV)
    assert pk.meta["compression"]["channels"]["susceptibility_path"]["sampling"] == "interval_mean_in_walk"


def test_a_spec_without_a_source_carries_no_field():
    """(d) Unchanged: no source, no basis, no samples, no tier."""
    w = walk_spec(spec_without_source(_packed().spec), 100, 2e-3, seed=2, require_gpu=False)
    assert w.field_basis is None and w.field_samples is None
    pk = build_replay_pack(w, id="test/no-source", license="x", citation="x", K=8, envelope=ENV)
    assert not pk.has_field
