"""Tests for Item 4: surface_area() and volume() analytical properties.

Verifies exact formula values for known inputs across all geometry classes.
"""

import math
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_sim.geometries import (
    Box1D,
    Cylinder,
    Sphere,
    Ellipsoid,
    MyelinatedCylinder,
    PackedCylinders,
    pack_cylinders,
)


# ---------------------------------------------------------------------------
# Box1D
# ---------------------------------------------------------------------------

class TestBox1D:
    def test_volume_equals_length(self):
        """volume() should equal the slab thickness (length)."""
        d = 10e-6
        b = Box1D(length=d)
        assert b.volume() == pytest.approx(d)

    def test_surface_area_is_two(self):
        """surface_area() should always be 2.0 (two walls, per unit area)."""
        b = Box1D(length=5e-6)
        assert b.surface_area() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------

class TestSphere:
    def test_volume_r1(self):
        """Sphere(R=1): volume = 4π/3."""
        s = Sphere(radius=1.0)
        npt.assert_allclose(s.volume(), 4.0 * math.pi / 3.0, rtol=1e-12)

    def test_surface_area_r1(self):
        """Sphere(R=1): surface_area = 4π."""
        s = Sphere(radius=1.0)
        npt.assert_allclose(s.surface_area(), 4.0 * math.pi, rtol=1e-12)

    def test_volume_r5um(self):
        R = 5e-6
        s = Sphere(radius=R)
        npt.assert_allclose(s.volume(), (4.0 / 3.0) * math.pi * R ** 3, rtol=1e-12)

    def test_surface_area_r5um(self):
        R = 5e-6
        s = Sphere(radius=R)
        npt.assert_allclose(s.surface_area(), 4.0 * math.pi * R ** 2, rtol=1e-12)

    def test_sv_ratio(self):
        """S/V = 3/R for a sphere."""
        R = 3e-6
        s = Sphere(radius=R)
        sv = s.surface_area() / s.volume()
        npt.assert_allclose(sv, 3.0 / R, rtol=1e-12)


# ---------------------------------------------------------------------------
# Cylinder
# ---------------------------------------------------------------------------

class TestCylinder:
    def test_volume_per_unit_length(self):
        """volume(L=1) = π·R² (cross-sectional area)."""
        R = 5e-6
        c = Cylinder(radius=R, orientation=[0, 0, 1])
        npt.assert_allclose(c.volume(L=1.0), math.pi * R ** 2, rtol=1e-12)

    def test_volume_with_L(self):
        R = 5e-6
        L = 50e-6
        c = Cylinder(radius=R, orientation=[0, 0, 1])
        npt.assert_allclose(c.volume(L=L), math.pi * R ** 2 * L, rtol=1e-12)

    def test_surface_area_no_caps(self):
        """surface_area(L=1, include_caps=False) = 2π·R."""
        R = 5e-6
        c = Cylinder(radius=R, orientation=[0, 0, 1])
        npt.assert_allclose(c.surface_area(L=1.0, include_caps=False),
                            2.0 * math.pi * R, rtol=1e-12)

    def test_surface_area_with_caps(self):
        """surface_area(L=1, include_caps=True) = 2π·R + 2π·R²."""
        R = 5e-6
        c = Cylinder(radius=R, orientation=[0, 0, 1])
        expected = 2.0 * math.pi * R + 2.0 * math.pi * R ** 2
        npt.assert_allclose(c.surface_area(L=1.0, include_caps=True),
                            expected, rtol=1e-12)

    def test_sv_ratio_per_unit_length(self):
        """S/V = 2/R for an infinite cylinder (lateral only)."""
        R = 3e-6
        c = Cylinder(radius=R, orientation=[0, 0, 1])
        sv = c.surface_area(L=1.0) / c.volume(L=1.0)
        npt.assert_allclose(sv, 2.0 / R, rtol=1e-12)


# ---------------------------------------------------------------------------
# Ellipsoid
# ---------------------------------------------------------------------------

class TestEllipsoid:
    def test_volume_sphere_case(self):
        """When a=b=c=R, volume = (4/3)π·R³."""
        R = 2.0
        e = Ellipsoid(semiaxes=[R, R, R])
        npt.assert_allclose(e.volume(), (4.0 / 3.0) * math.pi * R ** 3, rtol=1e-12)

    def test_surface_area_sphere_case(self):
        """When a=b=c=R, Thomsen approx = 4π·R² (exact for sphere)."""
        R = 2.0
        e = Ellipsoid(semiaxes=[R, R, R])
        npt.assert_allclose(e.surface_area(), 4.0 * math.pi * R ** 2, rtol=1e-3)

    def test_volume_known(self):
        """Ellipsoid(a=1, b=2, c=3): volume = (4/3)π·6 = 8π."""
        e = Ellipsoid(semiaxes=[1.0, 2.0, 3.0])
        npt.assert_allclose(e.volume(), (4.0 / 3.0) * math.pi * 6.0, rtol=1e-12)

    def test_surface_area_thomsen_oblate(self):
        """Oblate spheroid (a=b>c): Thomsen approx is close to exact."""
        # Exact formula for oblate spheroid: S = 2πa²(1 + (1-e²)/e · arctanh(e))
        # with e = sqrt(1 - (c/a)²)
        a, b, c = 3.0, 3.0, 1.0
        e_ecc = math.sqrt(1.0 - (c / a) ** 2)
        exact = 2.0 * math.pi * a ** 2 * (
            1.0 + (1.0 - e_ecc ** 2) / e_ecc * math.atanh(e_ecc)
        )
        ell = Ellipsoid(semiaxes=[a, b, c])
        # Thomsen < 1.061% error
        npt.assert_allclose(ell.surface_area(), exact, rtol=0.011)

    def test_surface_area_thomsen_prolate(self):
        """Prolate spheroid (a=b<c): Thomsen approx is close to exact."""
        # Exact formula for prolate spheroid: S = 2πb²(1 + (c/b/e_ecc)·arcsin(e_ecc))
        a, b, c = 1.0, 1.0, 3.0
        e_ecc = math.sqrt(1.0 - (a / c) ** 2)
        exact = 2.0 * math.pi * a ** 2 * (
            1.0 + c / (a * e_ecc) * math.asin(e_ecc)
        )
        ell = Ellipsoid(semiaxes=[a, b, c])
        npt.assert_allclose(ell.surface_area(), exact, rtol=0.011)


# ---------------------------------------------------------------------------
# MyelinatedCylinder
# ---------------------------------------------------------------------------

class TestMyelinatedCylinder:
    @pytest.fixture
    def mc(self):
        return MyelinatedCylinder(
            inner_radius=3e-6,
            outer_radius=5e-6,
            orientation=[0, 0, 1],
            D_intra=1e-9,
            D_extra=2e-9,
        )

    def test_volume_intra(self, mc):
        npt.assert_allclose(mc.volume('intra'), math.pi * (3e-6) ** 2, rtol=1e-12)

    def test_volume_myelin(self, mc):
        npt.assert_allclose(mc.volume('myelin'),
                            math.pi * ((5e-6) ** 2 - (3e-6) ** 2), rtol=1e-12)

    def test_volume_extra(self, mc):
        R_extra = 2.0 * 5e-6
        npt.assert_allclose(mc.volume('extra'),
                            math.pi * (R_extra ** 2 - (5e-6) ** 2), rtol=1e-12)

    def test_surface_area_intra(self, mc):
        npt.assert_allclose(mc.surface_area('intra'),
                            2.0 * math.pi * 3e-6, rtol=1e-12)

    def test_surface_area_myelin(self, mc):
        npt.assert_allclose(mc.surface_area('myelin'),
                            2.0 * math.pi * (3e-6 + 5e-6), rtol=1e-12)

    def test_surface_area_extra(self, mc):
        npt.assert_allclose(mc.surface_area('extra'),
                            2.0 * math.pi * 5e-6, rtol=1e-12)

    def test_volume_fractions_sum_to_one(self, mc):
        """volume_fraction() over all three compartments must sum to 1."""
        total = (mc.volume_fraction('intra')
                 + mc.volume_fraction('myelin')
                 + mc.volume_fraction('extra'))
        npt.assert_allclose(total, 1.0, atol=1e-12)

    def test_volume_fraction_intra(self, mc):
        R_in, R_out = 3e-6, 5e-6
        R_total = 2.0 * R_out
        expected = R_in ** 2 / R_total ** 2
        npt.assert_allclose(mc.volume_fraction('intra'), expected, rtol=1e-12)

    def test_invalid_compartment_raises(self, mc):
        with pytest.raises(ValueError):
            mc.volume('invalid')
        with pytest.raises(ValueError):
            mc.surface_area('invalid')
        with pytest.raises(ValueError):
            mc.volume_fraction('invalid')

    def test_volume_with_L(self, mc):
        L = 100e-6
        npt.assert_allclose(mc.volume('intra', L=L),
                            math.pi * (3e-6) ** 2 * L, rtol=1e-12)


# ---------------------------------------------------------------------------
# PackedCylinders
# ---------------------------------------------------------------------------

class TestPackedCylinders:
    @pytest.fixture
    def pc(self):
        radii = np.array([1e-6, 1e-6, 2e-6])
        centers, L, vf = pack_cylinders(radii=radii, target_vf=0.3, seed=42)
        return PackedCylinders(radii=radii, centers=centers, L=L), L, vf

    def test_volume_fraction_matches_manual(self, pc):
        """volume_fraction() == Σπ·Rk² / L²."""
        geom, L, _ = pc
        radii = np.array([1e-6, 1e-6, 2e-6])
        expected = math.pi * np.sum(radii ** 2) / L ** 2
        npt.assert_allclose(geom.volume_fraction(), expected, rtol=1e-12)

    def test_volume_fraction_matches_pack_cylinders_output(self, pc):
        """volume_fraction() should match the achieved_vf from pack_cylinders."""
        geom, L, achieved_vf = pc
        npt.assert_allclose(geom.volume_fraction(), achieved_vf, rtol=1e-10)

    def test_volume_per_unit_length(self, pc):
        """volume(L=1) = Σ π·Rk²."""
        geom, L, _ = pc
        radii = np.array([1e-6, 1e-6, 2e-6])
        expected = math.pi * np.sum(radii ** 2)
        npt.assert_allclose(geom.volume(L=1.0), expected, rtol=1e-12)

    def test_surface_area_per_unit_length(self, pc):
        """surface_area(L=1) = Σ 2π·Rk."""
        geom, L, _ = pc
        radii = np.array([1e-6, 1e-6, 2e-6])
        expected = 2.0 * math.pi * np.sum(radii)
        npt.assert_allclose(geom.surface_area(L=1.0), expected, rtol=1e-12)

    def test_monodisperse_volume_fraction(self):
        """Single cylinder: volume_fraction = π·R² / L²."""
        R = 1e-6
        centers, L, vf = pack_cylinders(radii=[R], target_vf=0.2, seed=0)
        pc = PackedCylinders(radii=[R], centers=centers, L=L)
        expected = math.pi * R ** 2 / L ** 2
        npt.assert_allclose(pc.volume_fraction(), expected, rtol=1e-12)
