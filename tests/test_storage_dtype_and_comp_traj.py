"""Positions came back float16, and packed geometries reported no compartment at all.

Reported by Marco Pizzolato (issue #78): the intracellular count of an IMPERMEABLE packed
cylinder walk changed over the walk, which it cannot physically do. Two defects compounded.

`comp_traj` was built only for a *permeable* single Cylinder/Sphere; every packed geometry
fell through to a constant 0, so per-compartment occupancy had to be re-derived from the
stored positions. And those positions were cast to float16 IN METRES -- where a micron-scale
coordinate is subnormal, giving a flat ~6e-8 m quantum (12.5% of a 0.5 um coordinate). A
compartment is populated up to its wall and empty beyond, so the quantum can only EJECT
walkers: ~3% of a confined population at R=0.5 um, scaling as ~1/R. The signal never
noticed (|dS| <= 1.3e-5); compartment counting did.
"""
from __future__ import annotations

import numpy as np
import pytest

from dmipy_sim import simulate_trajectories
from dmipy_sim.geometries import Cylinder, PackedCylinders, pack_cylinders

UM = 1e-6
D = 0.6e-9


def _small_pack(R_um=0.5, n=8, seed=0, permeability=None):
    """A dense NON-OVERLAPPING pack of small cylinders -- the regime where f16 bites
    hardest. RSA (`pack_cylinders`) rather than random centres: overlapping cylinders make
    a walker's compartment genuinely ambiguous, which is a different problem from #78."""
    radii = np.full(n, R_um * UM)
    centers, L, _vf = pack_cylinders(radii, target_vf=0.35, seed=seed)
    return PackedCylinders(radii=radii, centers=centers, L=L,
                           permeability=permeability)


# ---------------------------------------------------------------- storage dtype

def test_positions_default_to_float32():
    """f32 is the default: the walk is f32, the pack is f32, and the spec permits only
    float32/float64 for `positions`."""
    geom = Cylinder(radius=2 * UM, orientation=(0., 0., 1.))
    tr, *_ = simulate_trajectories(n_walkers=64, diffusivity=D, geometry=geom,
                                   T_max=2e-3, dt_save=5e-4, seed=0, require_gpu=False)
    assert tr.dtype == np.float32


def test_float16_is_available_as_an_explicit_opt_in():
    geom = Cylinder(radius=2 * UM, orientation=(0., 0., 1.))
    kw = dict(n_walkers=64, diffusivity=D, geometry=geom, T_max=2e-3,
              dt_save=5e-4, seed=0, require_gpu=False)
    t16, *_ = simulate_trajectories(storage_dtype=np.float16, **kw)
    t32, *_ = simulate_trajectories(storage_dtype=np.float32, **kw)
    assert t16.dtype == np.float16 and t32.dtype == np.float32
    assert t16.nbytes * 2 == t32.nbytes                      # the point of the opt-in
    # same walk, so they agree to the f16 quantum and no further
    assert np.abs(t16.astype(np.float64) - t32).max() < 1e-7


def test_storage_dtype_rejects_a_non_float():
    with pytest.raises(ValueError, match="float16/32/64"):
        simulate_trajectories(n_walkers=8, diffusivity=D, geometry=Cylinder(radius=2 * UM, orientation=(0., 0., 1.)),
                              T_max=1e-3, dt_save=5e-4, require_gpu=False,
                              storage_dtype=np.int16)


def test_f16_in_metres_is_subnormal_and_ejects_confined_walkers_one_way():
    """The mechanism, isolated from the walk: quantization is unbiased in a filled box but
    strictly one-directional for a population confined by a wall."""
    rng = np.random.default_rng(0)
    R = 0.5 * UM
    rr = R * np.sqrt(rng.uniform(0, 1, 200_000))
    th = rng.uniform(0, 2 * np.pi, 200_000)
    P = np.stack([rr * np.cos(th), rr * np.sin(th), np.zeros_like(rr)], 1) + 12 * UM
    assert np.float16(R) < np.finfo(np.float16).tiny * 2      # subnormal territory
    d16 = np.linalg.norm((P.astype(np.float16).astype(np.float64) - 12 * UM)[:, :2], axis=1)
    ejected = (d16 >= R).mean()
    assert ejected > 0.02, "expected the f16 metre quantum to eject ~3% at R=0.5um"
    # and f32 does not
    d32 = np.linalg.norm((P.astype(np.float32).astype(np.float64) - 12 * UM)[:, :2], axis=1)
    assert (d32 >= R).mean() == 0.0


# ---------------------------------------------------------------- comp_traj

def _intra_r0(geom, n, seed=0):
    """Seed inside the pack's cylinders. `PackedCylinders.init_positions` seeds the
    extra-axonal space only, so a two-pool test has to place these itself."""
    rng = np.random.default_rng(seed)
    C = np.asarray(geom._centers_jax)
    Rk = np.asarray(geom._radii_jax)
    k = rng.integers(0, len(Rk), n)
    rr = Rk[k] * 0.5 * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    return np.stack([C[k, 0] + rr * np.cos(th),
                     C[k, 1] + rr * np.sin(th),
                     np.zeros(n)], 1).astype(np.float32)


def _mixed_r0(geom, n_intra, n_extra, seed=0):
    import jax
    extra = np.asarray(geom.init_positions(n_extra, jax.random.PRNGKey(seed)))
    return np.concatenate([_intra_r0(geom, n_intra, seed), extra]).astype(np.float32)


def test_impermeable_pack_is_extra_only_so_a_constant_zero_is_correct():
    """Guard against "fixing" this the wrong way. With `permeability=None` the reflection
    keeps walkers OUT of the cylinders -- intra-seeded walkers are ejected on the first
    step -- so the geometry represents extra-axonal water only and comp_traj == 0 is the
    complete, correct answer, not the #78 defect."""
    geom = _small_pack(permeability=None)
    out = simulate_trajectories(n_walkers=64, diffusivity=D, geometry=geom, T_max=2e-3,
                                dt_save=1e-3, seed=0, require_gpu=False,
                                r0=_intra_r0(geom, 64), save_relaxation_data=True)
    assert (out[5] == 0).all()


def test_permeable_pack_reports_a_real_compartment_history():
    """The #78 regression. `comp_traj` was built only for a *permeable single* Cylinder or
    Sphere -- both branches test `hasattr(geometry, 'radius')`, and a pack carries `_radii_np`
    instead -- so every packed geometry fell through to a constant 0 even when its walkers
    genuinely occupied both pools. A caller then had no option but to re-derive occupancy
    from the stored positions, which is the classification f16 positions get wrong."""
    geom = _small_pack(permeability=1e-6)
    out = simulate_trajectories(n_walkers=256, diffusivity=D, geometry=geom, T_max=2e-3,
                                dt_save=1e-3, seed=0, require_gpu=False,
                                r0=_mixed_r0(geom, 128, 128), save_relaxation_data=True)
    comp = out[5]
    assert comp.max() > 0.0, "comp_traj is still constant zero for a permeable pack"
    assert comp.min() >= 0.0 and comp.max() <= 1.0, "occupancy must lie in [0, 1]"
    # both pools present at t=0: seeded-intra label 1, seeded-extra label 0
    assert comp[:128, 0].mean() > 0.5 and comp[128:, 0].mean() < 0.5, (
        "spec convention is 0 = extra-cellular / free, positive = intra")


def test_zero_permeability_conserves_compartments():
    """Marco's check (#78). `permeability=0.0` is a genuine two-pool substrate whose
    membrane never opens, so no walker may change pool and the intra count must be exactly
    constant. A constant-0 comp_traj, or a label re-derived from f16 positions, both fail
    this."""
    geom = _small_pack(permeability=0.0)
    n_intra = 256
    out = simulate_trajectories(n_walkers=512, diffusivity=D, geometry=geom, T_max=1e-2,
                                dt_save=5e-4, seed=1, require_gpu=False,
                                r0=_mixed_r0(geom, n_intra, 256, seed=1),
                                save_relaxation_data=True)
    comp = out[5]
    intra = comp > 0.5
    per_save = intra.sum(0)
    assert per_save[0] == n_intra, f"expected {n_intra} intra at t=0, got {per_save[0]}"
    assert per_save.min() == per_save.max(), (
        f"intra count drifted {per_save.min()} -> {per_save.max()} at kappa=0")
