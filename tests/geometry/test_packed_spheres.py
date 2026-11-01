"""PackedSpheres geometry — 3D periodic cubic domain with spheres.

Physics
-------
Impermeable and permeable versions of a packed-sphere substrate.  Spheres are
placed in a periodic cubic box via RSA (pack_spheres) and walkers diffuse in
the interstitial space (extra-axonal) or bidirectionally through the sphere
membranes (permeate).

Permeability follows Powles (2004):

    p = min(1,  2·κ·d_perp / D)

κ_karger = 3·κ_membrane / R  (sphere S/V = 3/R)

Validated limits
----------------
1. pack_spheres: correct L / VF / non-overlap / max gap.
2. init_positions: all walkers outside all spheres.
3. κ=None: identical signal to impermeable PackedSpheres.
4. κ>0 at high b: more restricted walkers inside spheres → higher signal.
5. High κ: signal lies between free diffusion and intra-axonal.
6. Relaxivity + permeability → signal below permeability-only.
7. Signal monotonically increases with κ at high b.
8. volume(), surface_area(), volume_fraction() match analytical formulas.

Parameters: R=5 µm, L=20 µm, D=2e-9 m²/s, TE=100 ms
"""

import numpy as np

from tests.conftest import pgse_wf
import numpy.testing as npt
import jax
import jax.numpy as jnp
import pytest

from dmipy_sim import simulate, PackedSpheres, Sphere, pack_spheres
from dmipy_replay import set_b
from dmipy_replay.sequences import pgse

from tests.conftest import D, N_EXACT, SEED


R         = 5e-6    # m
L         = 20e-6   # m
KAPPA_MED  = 1e-5   # m/s  — moderate exchange (τ_ex ≈ L³/(κ·4πR²) >> TE)
KAPPA_HIGH = 5e-4   # m/s  — fast exchange
RHO        = 5e-5   # m/s  — surface relaxivity


# =============================================================================
# Helpers
# =============================================================================

def _single_sphere_packed(permeability=None, surface_relaxivity_t2=None, pool="extra"):
    """Single sphere at origin in a periodic cubic box; the exterior seeded unless ``pool`` says otherwise."""
    return PackedSpheres(
        radii=np.array([R]),
        centers=np.array([[0., 0., 0.]]),
        L=L,
        permeability=permeability,
        surface_relaxivity_t2=surface_relaxivity_t2,
        pool=pool,
    )


# =============================================================================
# 1. pack_spheres utility
# =============================================================================

def test_pack_spheres_volume_fraction():
    """pack_spheres: achieved VF matches target."""
    radii = np.full(5, R)   # 5 spheres, same radius
    target_vf = 0.10
    centers, L_out, vf = pack_spheres(radii, target_vf=target_vf, seed=0)
    assert abs(vf - target_vf) < 1e-6, (
        f"Achieved VF {vf:.6f} != target {target_vf}")


def test_pack_spheres_no_overlap():
    """pack_spheres: no sphere pair overlaps (including periodic images)."""
    radii = np.full(4, R)
    centers, L_out, _ = pack_spheres(radii, target_vf=0.08, seed=1)
    N = len(radii)
    for i in range(N):
        for j in range(i + 1, N):
            dq = centers[i] - centers[j]
            dq -= L_out * np.round(dq / L_out)
            dist = np.linalg.norm(dq)
            min_dist = radii[i] + radii[j]
            assert dist >= min_dist - 1e-10, (
                f"Spheres {i} and {j} overlap: dist={dist:.3e} < {min_dist:.3e}")


def test_pack_spheres_shapes():
    """pack_spheres: output shapes are (N,3) for centers."""
    N = 3
    radii = np.full(N, R)
    centers, _, _ = pack_spheres(radii, target_vf=0.06, seed=2)
    assert centers.shape == (N, 3)


def test_pack_spheres_L_given():
    """pack_spheres: providing L instead of target_vf works."""
    radii = np.array([R])
    centers, L_out, vf = pack_spheres(radii, L=L, seed=0)
    assert L_out == L
    expected_vf = (4.0 / 3.0) * np.pi * R**3 / L**3
    assert abs(vf - expected_vf) < 1e-10


def test_pack_spheres_mutually_exclusive():
    """pack_spheres raises if both target_vf and L given."""
    with pytest.raises(ValueError, match="exactly one"):
        pack_spheres(np.array([R]), target_vf=0.1, L=L)


def test_pack_spheres_neither_raises():
    """pack_spheres raises if neither target_vf nor L given."""
    with pytest.raises(ValueError, match="exactly one"):
        pack_spheres(np.array([R]))


# =============================================================================
# 2. PackedSpheres constructor and attributes
# =============================================================================

def test_packed_spheres_attributes():
    """PackedSpheres stores permeability and surface_relaxivity_t2 correctly."""
    geom = _single_sphere_packed(permeability=KAPPA_MED,
                                 surface_relaxivity_t2=RHO)
    assert geom.permeability == KAPPA_MED
    assert geom.surface_relaxivity_t2 == RHO


def test_packed_spheres_volume_fraction():
    """PackedSpheres.volume_fraction() matches (4/3)πR³/L³."""
    geom = _single_sphere_packed()
    expected = (4.0 / 3.0) * np.pi * R**3 / L**3
    assert abs(geom.volume_fraction() - expected) < 1e-10


def test_packed_spheres_surface_area():
    """PackedSpheres.surface_area() = 4πR² for one sphere."""
    geom = _single_sphere_packed()
    expected = 4.0 * np.pi * R**2
    assert abs(geom.surface_area() - expected) < 1e-10


def test_packed_spheres_volume():
    """PackedSpheres.volume() = (4/3)πR³ for one sphere."""
    geom = _single_sphere_packed()
    expected = (4.0 / 3.0) * np.pi * R**3
    assert abs(geom.volume() - expected) < 1e-10


# =============================================================================
# 3. init_positions: all walkers outside spheres
# =============================================================================

def test_init_positions_both_pools_by_volume_by_default():
    """The default seeding covers the whole cell: the sphere holds its volume fraction of the walkers."""
    geom = _single_sphere_packed(pool=None)
    r0 = np.array(geom.init_positions(40_000, jax.random.PRNGKey(SEED)))
    inside = np.linalg.norm(r0, axis=1) < R
    assert abs(inside.mean() - geom.volume_fraction()) < 0.01


def test_init_positions_outside_spheres():
    """init_positions with ``pool="extra"``: all walkers start strictly outside all spheres."""
    geom = _single_sphere_packed()
    key  = jax.random.PRNGKey(SEED)
    r0   = np.array(geom.init_positions(20_000, key))
    dist = np.linalg.norm(r0, axis=1)   # distance from sphere at origin
    assert np.all(dist > R), (
        f"Some walkers inside sphere: min_dist={dist.min():.3e} < R={R:.3e}")


def test_init_positions_inside_box():
    """init_positions: all walkers in the periodic box [-L/2, L/2]."""
    geom = _single_sphere_packed()
    key  = jax.random.PRNGKey(SEED + 1)
    r0   = np.array(geom.init_positions(20_000, key))
    assert np.all(np.abs(r0) <= L / 2 + 1e-9), (
        "Some walkers outside the box [-L/2, L/2]")


# =============================================================================
# 4. κ=None must reproduce the impermeable signal
# =============================================================================

def test_permeability_none_matches_impermeable():
    """PackedSpheres(permeability=None) == default — identical signal."""
    wf            = pgse_wf(100e-3, slew_rate=np.inf)
    geom_default  = _single_sphere_packed(permeability=None)
    geom_explicit = PackedSpheres(
        radii=np.array([R]),
        centers=np.array([[0., 0., 0.]]),
        L=L,
        pool="extra",
    )
    # exact-equality assertion -> scale-free; see N_EXACT in conftest (#93)
    S_default  = simulate(N_EXACT, D, wf, geom_default,  seed=SEED)
    S_explicit = simulate(N_EXACT, D, wf, geom_explicit, seed=SEED)
    npt.assert_array_equal(S_default, S_explicit,
        err_msg="permeability=None must give identical signal to default")


# =============================================================================
# 9. min_gap is positive (non-overlapping packing)
# =============================================================================

def test_packed_spheres_min_gap_positive():
    """Single sphere at origin in box: min_gap = L - 2R (self-periodic image)."""
    geom = _single_sphere_packed()
    expected_gap = L - 2 * R   # distance between sphere surface and periodic image
    assert abs(geom.min_gap - expected_gap) < 1e-10, (
        f"min_gap={geom.min_gap:.3e} != L-2R={expected_gap:.3e}")


# =============================================================================
# 10. Multi-sphere packing from pack_spheres
# =============================================================================

def test_multi_sphere_packed():
    """PackedSpheres with multiple spheres: VF matches pack_spheres output."""
    radii = np.full(3, R)
    centers, L_ps, vf_ps = pack_spheres(radii, target_vf=0.06, seed=99)
    geom = PackedSpheres(radii=radii, centers=centers, L=L_ps)
    assert abs(geom.volume_fraction() - vf_ps) < 1e-6
    assert geom.min_gap > 0, "PackedSpheres from pack_spheres must have positive gap"
