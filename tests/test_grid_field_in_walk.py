"""A gridded susceptibility field (:class:`FieldGrid`: a mesh sheath, a myelinated cylinder, a packed cell) is read
BY THE WALK: :func:`simulate_trajectories` with ``field_basis=`` reads the grid's channels at the kernel's position
every ``field_read_interval`` seconds (every physics step when the rule asks below the step), accumulates the
trapezoid of the reads as the field's cosine modes per window (and, on the raw subset, as block means), as the strand
route does with ``adaptive_steps=True``. ``walk_spec`` reads every gridded source this way, and the pack's path
channel is encoded from those modes."""
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


def _interval_means(fg, fine, n_sub, n_read=1):
    """The rule, read after the walk: the grid at every ``n_read``-th sub-step position of a walk that kept every
    sub-step (``fine``, a block of one step), each read interval's TRAPEZOID of its two reads averaged over a block
    of ``n_sub`` steps; sample 0 the start position's read."""
    n_w = fine.shape[0]
    ch = fg.channels(np.asarray(fine, np.float64).reshape(-1, 3)).reshape(n_w, fine.shape[1], -1)
    reads = ch[:, ::n_read]                                                        # the reads, every n_read-th step
    trap = 0.5 * (reads[:, :-1] + reads[:, 1:])                                    # one per read interval
    per_block = n_sub // n_read
    return np.concatenate([ch[:, :1], trap.reshape(n_w, -1, per_block, ch.shape[-1]).mean(2)], axis=1)


def _modes_of(fg, fine, dt_step, T, K, n_read=1):
    """The dense oracle of the field modes ``c_k = int e_k b dt`` (the window's orthonormal cosines): the grid read at
    every ``n_read``-th position of a walk that kept every step, each read interval carrying the trapezoid of its two
    reads, integrated against ``e_k`` exactly over the interval -- ``(n_w, K, n_ch)``."""
    n_w = fine.shape[0]
    ch = fg.channels(np.asarray(fine, np.float64).reshape(-1, 3)).reshape(n_w, fine.shape[1], -1)
    reads = ch[:, ::n_read]
    v = 0.5 * (reads[:, :-1] + reads[:, 1:])
    t = np.arange(v.shape[1] + 1) * n_read * dt_step
    a, b = t[:-1], t[1:]
    out = np.zeros((n_w, K, ch.shape[-1]))
    for k in range(K):
        if k == 0:
            e = (b - a) / np.sqrt(T)
        else:
            om = np.pi * k / T
            e = np.sqrt(2.0 / T) * (np.sin(om * b) - np.sin(om * a)) / om
        out[:, k] = np.einsum("wic,i->wc", v, e)
    return out


CASES = {
    # (geometry, field grid, tiers): the three trajectory steps that read the grid
    "sphere-relax": lambda: (d.Sphere(2.5 * UM), _dipole_grid(), "all"),
    "sphere-positions": lambda: (d.Sphere(2.5 * UM), _dipole_grid(), ()),
    "packed-myelin-periodic": lambda: (_packed(), field_grid_of(_packed(), res=0.1 * UM), "all"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_in_walk_sample_is_the_interval_mean_of_the_grid_at_the_sub_steps(case):
    """(a) The walk in blocks of ``N_SUB`` sub-steps, reading the field in the walk at every step (the rule asks below
    the step: the read IS the step), against the same walk kept at every sub-step (its raw subset, every walker) and
    the grid read afterwards: the positions are the same walk (to float32 rounding: the two are different programs,
    and XLA may round a fused step differently), and every block mean of the raw subset (``raw_field``) is the
    trapezoid mean of the grid over its block's sub-step positions, to float32 rounding (the device reads in float32
    at the position the step kernel holds -- the in-cell one of a packed cell, whose grid is periodic). The modes
    the walk accumulates are the block means' cosine projection (``sqrt(h)`` DCT-II, the k = 0 mode to 1e-6)."""
    geom, fg, tiers = CASES[case]()
    T, dt = 2e-3, 2e-4
    w = d.simulate_trajectories(120, D0, geom, T, dt, seed=4, sub_steps=N_SUB, tiers=tiers, require_gpu=False,
                                field_basis=fg, raw_subset=120)
    fine = d.simulate_trajectories(120, D0, geom, T, dt / N_SUB, seed=4, sub_steps=1, tiers=tiers, require_gpu=False,
                                   raw_subset=120)
    np.testing.assert_allclose(w.raw_positions, fine.raw_positions[:, ::N_SUB], rtol=0, atol=1e-11)   # measured 2.3e-12 m
    fr = w.block["field_read"]
    assert w.field_basis is fg and fr["n_read"] == 1 and fr["at_step"] and w.field_read_interval == pytest.approx(w.dt_sim)
    n_ch = 7 if fg.basis.get("aniso_G") is None else 13
    assert w.raw_field.shape == (120, w.n_blocks + 1, n_ch)
    assert w.field_coeffs.shape == (120, w.n_windows, w.K_c3, n_ch)
    ref = _interval_means(fg, fine.raw_positions, N_SUB)
    scale = np.abs(ref).max()
    # measured: 4.9e-7 (sphere, clamped dipole grid) and 6.4e-6 (packed cell, read in-cell on the device against the
    # unwrapped position wrapped on the host) of the channel scale, which changes by 0.34-0.74 of it between saves
    assert np.abs(w.raw_field - ref).max() <= 2e-5 * scale
    # the modes: c_k = int e_k b dt of the window, against the exact projection of the dense reads' trapezoids; the
    # k = 0 mode is sqrt(h) times the block means' k = 0 DCT-II coefficient
    assert w.n_windows == 1
    mode_ref = _modes_of(fg, fine.raw_positions, w.dt_sim, w.T_window, w.K_c3)
    modes = np.asarray(w.field_coeffs, np.float64)[:, 0]
    assert np.abs(modes - mode_ref).max() <= 5e-3 * np.abs(mode_ref).max()   # measured 4.5e-4 (packed cell), 2.4e-3 (sphere, positions only)
    k0 = np.sqrt(w.h) * np.asarray(w.raw_field, np.float64)[:, 1:].sum(1) / np.sqrt(w.n_blocks)
    np.testing.assert_allclose(modes[:, 0], k0, rtol=0, atol=1e-6 * np.abs(k0).max())


def test_a_coarser_field_read_interval_reads_every_that_many_th_step_only():
    """(b) A given ``field_read_interval`` of two steps reads the field every second step (the largest divisor of the
    block's steps within it): the walk itself is unchanged -- the same positions, bands and contact, only the field
    differs, each block mean the trapezoid of ITS reads -- and an interval below the step is the step."""
    geom, fg = _packed(), field_grid_of(_packed(), res=0.1 * UM)
    T, dt = 1.8e-3, 2e-4
    dt_sim = dt / N_SUB
    kw = dict(seed=5, sub_steps=N_SUB, require_gpu=False, field_basis=fg, raw_subset=80)
    w1 = d.simulate_trajectories(80, D0, geom, T, dt, field_read_interval=dt_sim, **kw)
    w2 = d.simulate_trajectories(80, D0, geom, T, dt, field_read_interval=2 * dt_sim, **kw)
    w0 = d.simulate_trajectories(80, D0, geom, T, dt, field_read_interval=0.1 * dt_sim, **kw)
    assert w1.n_blocks == 9 and w1.block["field_read"]["n_read"] == 1 and w2.block["field_read"]["n_read"] == 2
    assert w2.field_read_interval == pytest.approx(2 * dt_sim) and w1.field_read_interval == pytest.approx(dt_sim)
    assert w0.block["field_read"]["n_read"] == 1 and w0.block["field_read"]["at_step"] and not w1.block["field_read"]["at_step"]
    # the same walk, two programs (a read every other step is a branch): float32 rounding, 3.6e-12 m measured
    np.testing.assert_allclose(w2.raw_positions, w1.raw_positions, rtol=0, atol=1e-11)
    np.testing.assert_allclose(w2.position_coeffs, w1.position_coeffs, rtol=0, atol=1e-11)
    np.testing.assert_allclose(w2.boundary_bands, w1.boundary_bands, rtol=1e-4, atol=1e-12)
    fine = d.simulate_trajectories(80, D0, geom, T, dt_sim, seed=5, sub_steps=1, require_gpu=False, raw_subset=80)
    for w, n_read in ((w1, 1), (w2, 2)):
        ref = _interval_means(fg, fine.raw_positions, N_SUB, n_read)
        assert np.abs(w.raw_field - ref).max() <= 2e-5 * np.abs(ref).max()
    assert np.abs(w2.raw_field - w1.raw_field).max() > 1e-4 * np.abs(w1.raw_field).max()      # the coarser read is a different quadrature
    with pytest.raises(ValueError, match="go with field_basis"):
        d.simulate_trajectories(8, D0, geom, T, dt, seed=5, require_gpu=False, field_read_interval=2 * dt_sim)


def test_a_strand_basis_is_not_read_as_a_grid():
    from dmipy_sim.fields.strand_field import StrandFieldBasis
    strand = StrandFieldBasis([np.array([[0.0, 0.0, -5 * UM], [0.0, 0.0, 5 * UM]])], [1 * UM], [1.4 * UM], cutoff_m=10 * UM,
                              domain=(np.full(3, -5 * UM), np.full(3, 5 * UM)))
    with pytest.raises(TypeError, match="FieldGrid"):
        d.simulate_trajectories(8, D0, d.Sphere(2 * UM), 4e-4, 2e-4, require_gpu=False, field_basis=strand)


ENV = dict(bvals=[0.0, 1e9], dirs=[[1, 0, 0]], ogse_periods=[2], shortd_b=1e9, shortd_deltas_frac=[0.05],
           B0_list=[3.0, 7.0], theta_deg=[0, 90], delta_frac=0.2, Delta_frac=0.5, rho2_list=[1e-5])


@pytest.mark.parametrize("family", ["packed", "single"])
def test_a_myelinated_walk_carries_its_samples_and_the_pack_certifies_the_path_channel(family):
    """(c) ``walk_spec`` of a myelinated cylinder or a packed cell reads its grid in the walk; the default pack
    encodes the path channel from the modes the walk accumulated (no grid route: the positions are lossy) and its
    certificate is within twice its floor."""
    geom = _packed() if family == "packed" else d.MyelinatedCylinder(1.0 * UM, 1.5 * UM, (0, 0, 1), D0, D0)
    w = walk_spec(geom.spec, 600, 5e-3, seed=2, require_gpu=False)
    assert isinstance(w.field_basis, FieldGrid) and w.field_coeffs is not None
    assert w.field_basis.periodic == ((True,) * 3 if family == "packed" else (False,) * 3)
    assert w.field_coeffs.shape == (600, w.n_windows, w.K_c3, 13)
    pk = build_replay_pack(w, id=f"test/{family}-in-walk", license="x", citation="x", envelope=ENV)
    ch = pk.meta["compression"]["channels"]
    assert pk.has_field and ch["susc_field_basis"]["sampling"] == "accumulated_in_walk"
    assert ch["susceptibility_grid"]["replay_route"] == "path" and not ch["susceptibility_grid"]["arrays_in_pack"]
    assert "susc_grid_iso_local" not in pk.arrays and "err_susc_se" not in pk.meta["fidelity"]
    f = pk.meta["fidelity"]
    assert f["err_susc_path"] <= 2.0 * f["floor_susc_path"]
    assert f["within_2x_floor"]


def test_a_mesh_bundle_samples_its_grid_in_the_walk_and_a_frozen_pool_holds_its_seat(tmp_path):
    """The pool-by-pool walk of a meshed sheath reads the raster in every diffusing pool's walk; the frozen myelin
    pool's modes are the channels at its seat held over the window: the k = 0 mode alone, ``sqrt(T) b(seat)``."""
    pytest.importorskip("trimesh")
    from tests.test_mesh_spec import _bundle_spec
    w = walk_spec(_bundle_spec(tmp_path), 120, 1e-3, 2.5e-4, seed=0, n_probe=20_000, field_res=0.4e-6, require_gpu=False, K=8)
    assert isinstance(w.field_basis, FieldGrid) and w.field_coeffs.shape == (120, w.n_windows, w.K_c3, 13)
    ids = np.asarray(w.compartment)
    frozen = np.asarray(w.field_coeffs)[ids == 2]
    assert (frozen[:, :, 1:] == 0).all()                                           # held: no modulation
    seat = np.asarray(w.field_basis.channels_at_device()(np.asarray(w.position_coeffs[ids == 2, 0, 0], np.float32)))
    np.testing.assert_allclose(frozen[:, 0, 0], np.sqrt(w.T_window) * seat, rtol=1e-6, atol=1e-6 * np.abs(seat).max())
    moving = np.asarray(w.field_coeffs)[ids != 2]
    assert np.abs(moving[:, :, 1:]).max() > 0                                      # the walkers move: their field is modulated
    pk = build_replay_pack(w, id="test/bundle-in-walk", license="x", citation="x", K=8, envelope=ENV)
    assert pk.meta["compression"]["channels"]["susc_field_basis"]["sampling"] == "accumulated_in_walk"


def test_a_spec_without_a_source_carries_no_field():
    """(d) Unchanged: no source, no basis, no samples, no tier."""
    w = walk_spec(spec_without_source(_packed().spec), 100, 2e-3, seed=2, require_gpu=False, K=8)
    assert w.field_basis is None and w.field_coeffs is None
    pk = build_replay_pack(w, id="test/no-source", license="x", citation="x", K=8, envelope=ENV)
    assert not pk.has_field
