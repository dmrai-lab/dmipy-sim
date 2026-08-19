"""Three-pool bundle master: per-compartment MT, exact seeding, and the replay-knob boundary.

Two defects these pin, both found by cross-checking a parametric MT prediction against an emergent binding
walk on real CACTUS geometry:

* the C4 tier averaged S/V over pools that never exchange, under-predicting the intra pool's bound fraction
  by ~8x (0.0042 against a measured 0.0388, where the pool's own S/V gives 0.0349);
* the extra pool was seeded from the RAW cell-gather classifier, which calls empty-gather points exterior --
  so deep intra-axonal points were seeded into the extra pool (49.3% of intra volume is beyond gather reach
  on a real bundle).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from dmipy_sim.io.cactus import load_cactus_bundle
from dmipy_sim.mesh_bundle import (bundle_mt_params, kappa_MT_for_voxel_f_bound, wm_mt_parameters,
                                   mesh_bundle_master, _exterior_seeds, _min_radius)

UM = 1e-6


def _bundle_dir(tmp_path, n_strands=2, side=20.0, g=0.7, r_out=2.0):
    """A CACTUS-shaped run directory: config header plus per-strand inner/outer PLYs."""
    run = tmp_path / "run"
    sim = run / "meshes" / "simulations"
    sim.mkdir(parents=True)
    for i in range(n_strands):
        for tag, radius in (("inner", g * r_out), ("outer", r_out)):
            m = trimesh.creation.cylinder(radius=radius, height=side, sections=48)
            m.apply_translation([5.0 * i - 2.5, 0.0, 0.0])
            m.export(sim / f"strand_{i:05d}_{tag}_erode_0.ply")
    lines = [f"{side}", f"{n_strands}", "2"]
    for i in range(n_strands):
        lines += ["2", f"{5.0*i-2.5} 0 {-side/2}", f"{5.0*i-2.5} 0 {side/2}"]
    (run / "optimized_final.txt").write_text("\n".join(lines) + "\n")
    return load_cactus_bundle(str(run))


# ---------------------------------------------------------------- C4: per compartment
def test_mt_params_are_per_compartment(tmp_path):
    """Impermeable pools do not exchange, so each has its own S/V and its own bound fraction."""
    b = _bundle_dir(tmp_path)
    p = bundle_mt_params(b, kappa_MT=1e-6, dwell_time=50e-3)

    assert set(p["f_bound"]) == {"intra", "extra"}, p["f_bound"]
    assert set(p["S_over_V"]) == {"intra", "extra"}
    # the two pools genuinely differ, so a single scalar could not describe both
    assert p["S_over_V"]["intra"] != pytest.approx(p["S_over_V"]["extra"], rel=0.02)
    assert p["f_bound"]["intra"] != pytest.approx(p["f_bound"]["extra"], rel=0.02)
    # and the voxel value is the volume-weighted mean of them, bracketed by the pools
    lo, hi = sorted(p["f_bound"].values())
    assert lo <= p["f_bound_voxel"] <= hi
    assert sum(p["volume_weights"].values()) == pytest.approx(1.0)


def test_the_bundle_average_would_disagree_with_the_pools(tmp_path):
    """The defect, made visible: one S/V over the pooled free volume is not either pool's rate.

    This is what shipped, and what an emergent walk contradicted by ~8x on the intra pool.
    """
    from dmipy_sim.mt import bound_fraction
    b = _bundle_dir(tmp_path)
    kappa, dwell = 1e-6, 50e-3
    p = bundle_mt_params(b, kappa, dwell)

    A_in = float(trimesh.Trimesh(*b.inner, process=False).area)
    A_out = float(trimesh.Trimesh(*b.outer, process=False).area)
    V_box = float(np.prod(b.box_side))
    sv_avg = (A_in + A_out) / ((b.f_intra + b.f_extra) * V_box)
    f_avg = float(bound_fraction(kappa, dwell, sv_avg))

    assert abs(f_avg - p["f_bound"]["intra"]) / p["f_bound"]["intra"] > 0.2, (
        f"geometry no longer exercises the averaging error (avg {f_avg:.4g} vs intra "
        f"{p['f_bound']['intra']:.4g})")


def test_kappa_solves_for_a_target_voxel_bound_fraction(tmp_path):
    b = _bundle_dir(tmp_path)
    dwell = 50e-3
    for target in (0.05, 0.139):
        kappa = kappa_MT_for_voxel_f_bound(b, target, dwell)
        assert bundle_mt_params(b, kappa, dwell)["f_bound_voxel"] == pytest.approx(target, rel=1e-4)


def test_literature_parameters_reproduce_the_catalogued_bound_fraction(tmp_path):
    """wm_mt_parameters must land the voxel bound fraction on the catalogue's M0B, and the dwell on 1/k_r."""
    from dmipy_sim.substrate.biophysical_constants import canonical_white_matter
    b = _bundle_dir(tmp_path)
    p = canonical_white_matter(3.0)
    M0B, R = p["mt_bound_pool_fraction"], p["mt_exchange_rate"]

    kappa, dwell = wm_mt_parameters(b)
    assert dwell == pytest.approx(1.0 / (R * (1.0 - M0B)), rel=1e-9)
    assert bundle_mt_params(b, kappa, dwell)["f_bound_voxel"] == pytest.approx(M0B, rel=1e-4)
    # sanity on the catalogued values themselves (Stanisz 2005 WM @3T)
    assert M0B == pytest.approx(0.139, abs=1e-3) and R == pytest.approx(23.0, abs=0.5)


# ---------------------------------------------------------------- seeding
def test_extra_pool_seeds_avoid_the_axon_interiors(tmp_path):
    """Seeded on the raw classifier, deep intra points land in the extra pool; the exact test excludes them.

    The oracle is analytic, not `trimesh.contains`: the fixture builds concentric cylinders at known axes, so
    "inside a fibre" is a radial distance. That matters because trimesh's ray-based containment is not
    deterministic for points essentially ON the surface -- the same 400 seeds came back 0 inside in one
    process and 1 inside in another. Tolerating the wall itself (a seed may legitimately sit a float away
    from it) while forbidding anything DEEP is the property actually worth asserting.
    """
    from dmipy_sim.mesh import Mesh
    b = _bundle_dir(tmp_path)
    Vo, Fo = b.outer
    mesh_out = Mesh(Vo, Fo, periodic=False, voxel_min=b.box_min, voxel_max=b.box_max,
                    feature_radius=_min_radius(Vo, Fo))
    pts = _exterior_seeds(mesh_out, b.box_min, b.box_max, 400, seed=0)

    r_out, axes_x, half_h = 2.0 * UM, (-2.5 * UM, 2.5 * UM), 10.0 * UM
    depth = np.full(len(pts), -np.inf)
    for ax in axes_x:                                        # radial depth inside each fibre
        radial = np.hypot(pts[:, 0] - ax, pts[:, 1])
        inside_len = np.abs(pts[:, 2]) < half_h
        depth = np.maximum(depth, np.where(inside_len, r_out - radial, -np.inf))
    tol = 1e-3 * r_out                                       # a seed may sit on the wall, not past it
    n_deep = int((depth > tol).sum())
    assert n_deep == 0, (
        f"{n_deep}/400 extra seeds lie more than {tol/UM:.4f} um inside a fibre; deepest "
        f"{depth.max()/UM:.4f} um")


# ---------------------------------------------------------------- the replay-knob boundary
@pytest.mark.slow
def test_master_omits_replay_knobs_unless_asked(tmp_path):
    """T2/T1/chi are replay knobs, so a master carries them only when a nominal value is supplied."""
    b = _bundle_dir(tmp_path)
    kw = dict(n_walkers=300, n_myelin=64, T_max=1e-3, n_t=10, seed=0, field=False,
              require_gpu=False, verbose=False)
    bare = mesh_bundle_master(b, **kw)
    assert "T2_per_comp" not in bare and "T1_per_comp" not in bare
    assert "traj" in bare and "dlog_b" in bare and "comp" in bare      # the walk's own output stands alone

    with_nominal = mesh_bundle_master(b, nominal_T2=(0.055, 0.05, 0.01), **kw)
    assert np.allclose(with_nominal["T2_per_comp"], [0.055, 0.05, 0.01])


@pytest.mark.slow
def test_emergent_mt_stores_a_per_walker_channel(tmp_path):
    """The emergent route bakes (kappa_MT, dwell_time) in, so it must also store what it produced."""
    b = _bundle_dir(tmp_path)
    kw = dict(n_walkers=300, n_myelin=64, T_max=1e-3, n_t=10, seed=0, field=False,
              require_gpu=False, verbose=False)
    m = mesh_bundle_master(b, mt="emergent", **kw)
    assert "bfrac" in m and m["bfrac"].shape == m["comp"].shape
    # a frozen myelin walker never moves, so it never strikes a wall and never binds
    assert np.all(m["bfrac"][m["comp0"] == 2] == 0.0)
    assert mesh_bundle_master(b, mt="parametric", **kw).get("bfrac") is None
