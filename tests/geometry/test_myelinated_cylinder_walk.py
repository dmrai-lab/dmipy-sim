"""The isolated `MyelinatedCylinder` has a persistent walk: one instance of the concentric-cylinder
kernel (`physics.make_myelin_substep`) in an open domain, stepped by `physics.make_myelin_traj_step_fn`
-- the same carry and the same recording branch in `engine.core._walk_geometry` as
`PackedMyelinatedCylinders`' own trajectory step, with no periodic cell to unwrap (`r_unwrapped`
repeats `r`). See `tests/geometry/test_trajectory_walk_gap.py` for the pre-flight check that used to
refuse it, and `tests/test_compartment_ids.py::test_simulate_and_simulate_trajectories_report_the_same_ids`
for the packed geometry's version of the fused/trajectory label check mirrored here."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.core import _walk_geometry
import dmipy_sim.engine.core as core
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim import walk
from dmipy_replay.spec.tissue import Tissue
from tests.conftest import spec_without_source
from dmipy_sim.engine._fused import simulate_fused

D = 2e-9
EXTRA, INTRA, MYELIN = 0, 1, 2


def _g():
    return d.MyelinatedCylinder(2e-6, 2.8e-6, (0, 0, 1),
                                compartments=d.Compartments(intra=d.Pool(D=D, T2=0.05), myelin=d.Pool(T2=0.01, D=0.0),
                                                            extra=d.Pool(D=D, T2=0.08)))


def test_walk_spec_no_longer_refuses_the_isolated_cylinder():
    g = _g()
    assert g.trajectory_walk_gap() is None
    w = walk(g.spec, 8, T_max=1e-3, require_gpu=False)   # must not raise
    assert w.n_walkers == 8 and w.position_coeffs.shape[-1] == 3


def test_pool_labels_agree_with_the_geometrys_own_classifier_at_every_save():
    g = _g()
    w = _walk_geometry(2000, D, g, T_max=1e-3, block_h=5e-5, seed=0, require_gpu=False, raw_subset=2000)
    pos = np.asarray(w.raw_positions)
    comp = np.asarray(w.raw_compartment)
    cls = np.asarray(jax.vmap(jax.vmap(g.classify_position))(jnp.asarray(pos)))
    assert (comp == cls).all(), "the carried label disagrees with re-classifying the saved position"

    r_xy = np.linalg.norm(pos[..., :2], axis=-1)
    assert (r_xy[comp == INTRA] < g.inner_radius).all(), "an intra-labelled save sits outside the axolemma"
    assert ((r_xy[comp == MYELIN] >= g.inner_radius) & (r_xy[comp == MYELIN] < g.outer_radius)).all(), \
        "a myelin-labelled save sits outside the sheath"
    assert (r_xy[comp == EXTRA] >= g.outer_radius).all(), "an extra-labelled save sits inside the sheath"
    # precondition: the walk actually visited all three pools, or the checks above are vacuous
    assert {EXTRA, INTRA, MYELIN} <= set(np.unique(comp))


def test_boundary_local_time_is_non_positive():
    g = _g()
    w = _walk_geometry(2000, D, g, T_max=1e-3, block_h=5e-5, seed=1, require_gpu=False, raw_subset=2000)
    blt = np.asarray(w.raw_boundary)
    assert blt.max() <= 0.0
    assert blt.min() < 0.0, "precondition: some walkers actually touched a wall"


def test_simulate_and_simulate_trajectories_report_the_same_compartment_at_t0():
    """`simulate`'s return_compartments='final' label and _walk_geometry' raw compartment at block end 0
    are the same classifier, collapsed the same way -- the isolated-cylinder analogue of
    test_compartment_ids.test_simulate_and_simulate_trajectories_report_the_same_ids."""
    wf = d.set_b(d.pgse([[1, 0, 0]], 2e-3, 4e-3, gradient_strengths=0.05, n_t=30, slew_rate=np.inf), 5e8)
    _, origin, _ = simulate_fused(300, None, wf, _g(), seed=0, return_compartments="final", require_gpu=False)
    out = _walk_geometry(300, D, _g(), T_max=6e-3, block_h=2e-3, seed=0, require_gpu=False, raw_subset=300)
    comp0 = np.asarray(out.raw_compartment)[:, 0]
    assert set(np.unique(origin)) <= {EXTRA, INTRA, MYELIN}
    assert (np.asarray(origin) == comp0).all()
    assert (comp0 == MYELIN).mean() > 0.05 and (comp0 == INTRA).mean() > 0.05


def test_a_replay_pack_of_the_walk_agrees_with_the_fused_signal():
    """build_replay_pack(walk).replay() of a bare PGSE agrees with the fused engine's own signal
    within the Monte-Carlo floor, at a few thousand walkers over a few ms. Measured: |dS| = 0.0076
    at N=4000, T_max=4 ms (floor max(0.02, 1/sqrt(N)) = 0.0158)."""
    N, T_max = 4000, 4e-3
    wf = d.set_b(d.pgse([[1, 0, 0]], 1e-3, 3e-3, gradient_strengths=0.08, n_t=60, slew_rate=np.inf), 5e8)
    s_fused = float(np.asarray(d.simulate(N, None, wf, _g(), seed=0, require_gpu=False)).ravel()[0])

    w = _walk_geometry(N, D, _g(), T_max=T_max, block_h=T_max / 60, K=59, seed=0, require_gpu=False)
    w = dataclasses.replace(w, spec=spec_without_source(w.spec))      # the bare signal: no field tier here
    pk = build_replay_pack(w, id="test/isolated-myelin-pgse", K=w.K, license="x", citation="x")
    tissue = Tissue(T2={"intra": 0.05, "myelin": 0.01, "extra": 0.08})
    s_replay = float(np.asarray(pk.replay(wf, tissue=tissue)).ravel()[0])

    assert abs(s_fused - s_replay) < max(0.02, 1.0 / np.sqrt(N)), \
        f"fused {s_fused:.4f} vs replay {s_replay:.4f}"


def test_kappa_mt_is_refused_for_the_isolated_cylinder_before_any_device_work(monkeypatch):
    """kappa_MT binds free water into a bound pool that only PackedMyelinatedCylinders'
    trajectory step carries; the isolated cylinder's step has none, so this must be refused
    by name rather than silently walked without it (#536)."""
    def _boom(*a, **k):
        raise AssertionError("seed_walkers was called: a walk was attempted")
    monkeypatch.setattr(core, "seed_walkers", _boom)
    with pytest.raises(NotImplementedError, match=r"MyelinatedCylinder.*PackedMyelinatedCylinders"):
        _walk_geometry(8, D, _g(), T_max=1e-3, block_h=1e-3, require_gpu=False,
                                kappa_MT=1e-3, dwell_time=1e-3)


def test_kappa_mt_still_walks_on_the_packed_class():
    """The refusal above is geometry-specific: PackedMyelinatedCylinders still walks with
    kappa_MT > 0 and records the bound-fraction channel."""
    L = float(np.sqrt(np.pi * 3 * (1e-6 / 0.7) ** 2 / 0.5))
    _, _, c = d.pack_myelinated_cylinders([1e-6] * 3, 0.7, None, cell_size=L, seed=0)
    pm = d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4, compartments=d.Compartments(intra=d.Pool(D=D), extra=d.Pool(D=D), myelin=d.Pool(D=0.0)))
    w = _walk_geometry(300, D, pm, T_max=8e-3, block_h=2e-4, seed=1, require_gpu=False,
                                kappa_MT=5e-5, dwell_time=1e-3, equilibrate_binding="off")
    assert w.has_binding


def test_walk_spec_builds_a_three_pool_pack():
    g = _g()
    w = walk(g.spec, 1500, T_max=1e-3, require_gpu=False)
    assert set(np.unique(np.asarray(w.compartment))) == {EXTRA, INTRA, MYELIN}
    pack = build_replay_pack(w, id="test/isolated-myelin-spec-walk", license="x", citation="x")   # the sheath's field sampled in the walk
    assert len(pack.substrate.pools) == 3
