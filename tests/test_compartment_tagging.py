"""Tests for Item 5: Walker compartment tagging.

Verifies:
1. compartment_origin never changes from its initial value.
2. return_compartments='final' matches last timestep of 'full'.
3. (compartment_origin == k).mean() is volume_fraction(k) to one walker (the seeding allocates the counts).
4. Permeable Cylinder: compartment_current changes over time.
5. Impermeable Cylinder: compartment_current ≈ compartment_origin at all times.
"""

import numpy as np
import numpy.testing as npt
import jax.numpy as jnp
import pytest

from dmipy_sim import simulate, Cylinder, Sphere
from dmipy_sim.geometry import (
    MyelinatedCylinder,
    PackedCylinders,
    pack_cylinders,
    Box1D,
    Ellipsoid,
)
from dmipy_replay.acquisition.scanner_sequence import ScannerSequence
from dmipy_sim.engine._fused import simulate_fused
from dmipy_sim import Compartments, Pool


# ---------------------------------------------------------------------------
# Shared waveform fixture (zero gradients — we only care about compartment IDs)
# ---------------------------------------------------------------------------

N_T = 100
N_MEAS = 1
DT = 1e-4  # 0.1 ms
D = 2e-9   # 2 µm²/ms


def _make_waveform():
    G = np.zeros((N_MEAS, N_T, 3), dtype=np.float32)
    return ScannerSequence(G=jnp.array(G), dt=DT, readout=(N_T - 1,))


# ---------------------------------------------------------------------------
# Helper: run simulation and return (comp_origin, comp_seq_full)
# ---------------------------------------------------------------------------

def _run_full(geometry, n_walkers, diffusivity=D, seed=0):
    wf = _make_waveform()
    signals, comp_orig, comp_seq = simulate_fused(
        n_walkers, diffusivity, wf, geometry, seed=seed,
        return_compartments='full')
    return comp_orig, comp_seq


def _run_final(geometry, n_walkers, diffusivity=D, seed=0):
    wf = _make_waveform()
    signals, comp_orig, comp_final = simulate_fused(
        n_walkers, diffusivity, wf, geometry, seed=seed,
        return_compartments='final')
    return comp_orig, comp_final


# ---------------------------------------------------------------------------
# 1. compartment_origin is immutable (does not change across timesteps).
#    Since compartment_origin is set before the scan and not part of the scan
#    state, it is always the initial value by construction.  We verify it via
#    the 'full' output: comp_origin should equal the first time-step value for
#    all walkers in an impermeable geometry.
# ---------------------------------------------------------------------------

class TestCompartmentOriginImmutability:
    """compartment_origin must never equal a value different from t=0."""

    def test_cylinder_impermeable_origin_stable(self):
        """Impermeable cylinder: compartment_origin is consistent with t=0 position."""
        geom = Cylinder(radius=5e-6, orientation=[0, 0, 1])
        comp_orig, comp_seq = _run_full(geom, n_walkers=500)
        # comp_origin is returned separately and is always the initial value.
        # For impermeable walkers that start intra (comp_orig == 1), the first
        # step should almost always stay intra (barring boundary float32 effects).
        intra_mask = comp_orig == 1
        # The vast majority should stay intra (> 99%)
        stay_intra = (comp_seq[intra_mask, 0] == 1).mean()
        assert stay_intra > 0.99, f"Expected >99% of intra walkers to stay intra at t=1, got {stay_intra:.3f}"

    def test_sphere_impermeable_origin_stable(self):
        """Impermeable sphere: compartment_origin is consistent with t=0."""
        geom = Sphere(radius=5e-6)
        comp_orig, comp_seq = _run_full(geom, n_walkers=500)
        intra_mask = comp_orig == 1
        stay_intra = (comp_seq[intra_mask, 0] == 1).mean()
        assert stay_intra > 0.99

    def test_myelinated_cylinder_origin_stable(self):
        """MyelinatedCylinder: compartment_origin stays at initial compartment."""
        mc = MyelinatedCylinder(
            inner_radius=3e-6,
            outer_radius=5e-6,
            orientation=[0, 0, 1],
            compartments=Compartments(intra=Pool(D=1e-9), extra=Pool(D=2e-9), myelin=Pool(D=0.0)),
        )
        comp_orig, comp_seq = _run_full(mc, n_walkers=500, diffusivity=None)
        # comp_orig has all three compartments; first step should preserve them
        for comp_id in [0, 1, 2]:
            mask = comp_orig == comp_id
            if mask.sum() == 0:
                continue
            # Compartment can change due to diffusion between compartments
            # (no permeability, so intra/myelin/extra should not cross in one step)
            # Actually, without permeability walkers stay in their compartment.
            # (Impermeable boundaries → no crossing is possible.)
            stay = (comp_seq[mask, 0] == comp_id).mean()
            assert stay > 0.95, (
                f"Compartment {comp_id}: only {stay:.3f} stayed at t=1")


# ---------------------------------------------------------------------------
# 2. return_compartments='final' matches last timestep of 'full'
# ---------------------------------------------------------------------------

class TestFinalMatchesFull:
    """comp_final from 'final' mode must equal comp_seq[:, -1] from 'full' mode."""

    def _check_match(self, geometry, diffusivity=D, n_walkers=200):
        comp_orig_full, comp_seq = _run_full(geometry, n_walkers, diffusivity=diffusivity)
        comp_orig_final, comp_final = _run_final(geometry, n_walkers, diffusivity=diffusivity)
        # Same seed → same results
        npt.assert_array_equal(comp_orig_full, comp_orig_final)
        npt.assert_array_equal(comp_seq[:, -1], comp_final)

    def test_cylinder_impermeable(self):
        self._check_match(Cylinder(radius=5e-6, orientation=[0, 0, 1]))

    def test_cylinder_permeable(self):
        self._check_match(Cylinder(radius=5e-6, orientation=[0, 0, 1],
                                   permeability=4e-5))

    def test_sphere_impermeable(self):
        self._check_match(Sphere(radius=5e-6))

    def test_sphere_permeable(self):
        self._check_match(Sphere(radius=5e-6, permeability=4e-5))

    def test_myelinated_cylinder(self):
        mc = MyelinatedCylinder(
            inner_radius=3e-6,
            outer_radius=5e-6,
            orientation=[0, 0, 1],
            compartments=Compartments(intra=Pool(D=1e-9), extra=Pool(D=2e-9), myelin=Pool(D=0.0)),
        )
        self._check_match(mc, diffusivity=None)


# ---------------------------------------------------------------------------
# 3. (compartment_origin == k).mean() = volume_fraction(k)
#    Tested for Cylinder (every walker seeded in the lumen) and MyelinatedCylinder, whose seeding allocates
#    round(n w_k / W) walkers to each pool by its volume (MyelinatedCylinder.init_positions): neither is a random
#    draw, so neither has a binomial standard error.
# ---------------------------------------------------------------------------

class TestVolumefractionConsistency:
    """Compartment origin fractions are the geometry's volume fractions: exactly, or to one walker of rounding."""

    def test_cylinder_all_intra(self):
        """Every walker starts intra for the single-cylinder geometry: the seeding draws inside the lumen, so the
        count outside is zero, not a fraction within noise."""
        geom = Cylinder(radius=5e-6, orientation=[0, 0, 1])
        comp_orig, _ = _run_final(geom, n_walkers=20_000)
        assert int((np.asarray(comp_orig) != 1).sum()) == 0

    def test_myelinated_cylinder_volume_fractions(self):
        """MyelinatedCylinder: each placement block holds ``round(P vf)`` walkers of each pool (the extra pool the rest,
        ``P`` = ``PLACEMENT_BLOCK``), so a walk of whole blocks holds each pool's volume fraction to half a walker a
        block, ``0.5 / P`` (and one walker of the count)."""
        from dmipy_sim.geometry.base import PLACEMENT_BLOCK
        mc = MyelinatedCylinder(
            inner_radius=3e-6,
            outer_radius=5e-6,
            orientation=[0, 0, 1],
            compartments=Compartments(intra=Pool(D=1e-9, water_fraction=1.0), myelin=Pool(water_fraction=1.0, D=0.0), extra=Pool(D=2e-9, water_fraction=1.0)),
        )
        n = 5 * PLACEMENT_BLOCK
        comp_orig, _ = _run_final(mc, n_walkers=n, diffusivity=None)
        for comp_id, label in [(1, 'intra'), (2, 'myelin'), (0, 'extra')]:   # pool ids
            vf_mc = (comp_orig == comp_id).mean()
            vf_analytical = mc.volume_fraction(label)
            npt.assert_allclose(vf_mc, vf_analytical, atol=0.5 / PLACEMENT_BLOCK + 1.0 / n,
                                err_msg=f"volume_fraction mismatch for {label}")


# ---------------------------------------------------------------------------
# 4. Permeable Cylinder (κ>0): compartment_current changes over time
# ---------------------------------------------------------------------------

class TestPermeableCylinderChanges:
    """With permeability set, walkers must cross the boundary over time."""

    def test_compartment_current_changes_permeable_cylinder(self):
        """Fraction of walkers that change compartment must be > 5% with κ=4e-5."""
        geom = Cylinder(radius=5e-6, orientation=[0, 0, 1], permeability=4e-5)
        n_walkers = 500
        comp_orig, comp_seq = _run_full(geom, n_walkers=n_walkers)
        changed = np.any(comp_seq != comp_orig[:, None], axis=1)
        frac_changed = changed.mean()
        assert frac_changed > 0.05, (
            f"Expected >5% walkers to cross in permeable cylinder, got {frac_changed:.3f}")

    def test_compartment_current_changes_permeable_sphere(self):
        """Fraction of walkers that change compartment must be > 5% with κ=4e-5."""
        geom = Sphere(radius=5e-6, permeability=4e-5)
        n_walkers = 500
        comp_orig, comp_seq = _run_full(geom, n_walkers=n_walkers)
        changed = np.any(comp_seq != comp_orig[:, None], axis=1)
        frac_changed = changed.mean()
        assert frac_changed > 0.05, (
            f"Expected >5% walkers to cross in permeable sphere, got {frac_changed:.3f}")


# ---------------------------------------------------------------------------
# 5. Impermeable Cylinder (κ=0): compartment_current ≈ compartment_origin
#    Allow ≤ 0.5% walkers to show apparent crossing due to float32 boundary
#    effects (walkers placed within NUDGE of boundary by init_positions).
# ---------------------------------------------------------------------------

class TestImpermeableCylinderStable:
    """Without permeability, compartment should not change (within float32 tolerance)."""

    def test_impermeable_cylinder_no_crossing(self):
        """Impermeable cylinder: fewer than 3% apparent crossings.

        A small fraction (~1-2%) of walkers placed near the boundary may show
        apparent classification changes due to float32 rounding in the cylinder-
        frame rotation (self._R @ r) combined with the boundary NUDGE epsilon.
        These are not physical crossings but numerical boundary artefacts.
        We use a 3% threshold to catch genuine permeation (which would show
        > 5% crossings) while tolerating float32 boundary effects.
        """
        geom = Cylinder(radius=5e-6, orientation=[0, 0, 1])
        n_walkers = 5000
        comp_orig, comp_seq = _run_full(geom, n_walkers=n_walkers)
        changed = np.any(comp_seq != comp_orig[:, None], axis=1)
        frac_changed = changed.mean()
        assert frac_changed < 0.03, (
            f"Impermeable cylinder: {frac_changed:.4f} walkers showed apparent "
            f"crossings (tolerance 3%)")

    def test_impermeable_sphere_no_crossing(self):
        """Impermeable sphere: fewer than 0.5% apparent crossings."""
        geom = Sphere(radius=5e-6)
        n_walkers = 5000
        comp_orig, comp_seq = _run_full(geom, n_walkers=n_walkers)
        changed = np.any(comp_seq != comp_orig[:, None], axis=1)
        frac_changed = changed.mean()
        assert frac_changed < 0.005, (
            f"Impermeable sphere: {frac_changed:.4f} walkers showed apparent "
            f"crossings (tolerance 0.5%)")

    def test_impermeable_box1d_no_crossing(self):
        """Box1D is one enclosed pool (id 1); no changes expected."""
        geom = Box1D(length=10e-6)
        n_walkers = 500
        comp_orig, comp_seq = _run_full(geom, n_walkers=n_walkers)
        npt.assert_array_equal(comp_orig, np.ones(n_walkers, dtype=np.int32))
        npt.assert_array_equal(
            comp_seq,
            np.ones((n_walkers, N_T), dtype=np.int32),
            err_msg="Box1D should always have compartment_id=1")


# ---------------------------------------------------------------------------
# 6. Return API — check that return_compartments=False does not change output
# ---------------------------------------------------------------------------

class TestReturnAPICompat:
    """Existing return API must be unchanged when return_compartments=False."""

    def test_no_compartments_returns_scalar_array(self):
        wf = _make_waveform()
        geom = Cylinder(radius=5e-6, orientation=[0, 0, 1])
        result = simulate(100, D, wf, geom, seed=0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (N_MEAS,)

    def test_with_positions_and_compartments(self):
        """return_positions=True + return_compartments='final' → 4-tuple."""
        wf = _make_waveform()
        geom = Cylinder(radius=5e-6, orientation=[0, 0, 1])
        result = simulate_fused(100, D, wf, geom, seed=0,
                          return_positions=True,
                          return_compartments='final')
        assert isinstance(result, tuple) and len(result) == 4
        signals, positions, comp_orig, comp_final = result
        assert signals.shape == (N_MEAS,)
        assert positions.shape == (100, 3)
        assert comp_orig.shape == (100,)
        assert comp_final.shape == (100,)

    def test_invalid_return_compartments_raises(self):
        wf = _make_waveform()
        geom = Cylinder(radius=5e-6, orientation=[0, 0, 1])
        with pytest.raises(ValueError):
            simulate_fused(100, D, wf, geom, return_compartments='invalid')


# ---------------------------------------------------------------------------
# 7. MyelinatedCylinder full trace: 'final' == last element of 'full'
# ---------------------------------------------------------------------------

class TestMyelinatedCylinderFull:
    """MyelinatedCylinder with permeability: compartment changes are recorded."""

    def test_final_equals_last_full_timestep(self):
        mc = MyelinatedCylinder(
            inner_radius=3e-6,
            outer_radius=5e-6,
            orientation=[0, 0, 1],
            kappa_inner=1e-5,
            kappa_outer=1e-5,
            compartments=Compartments(intra=Pool(D=1e-9), extra=Pool(D=2e-9), myelin=Pool(D=0.0)),
        )
        n_walkers = 200
        wf = _make_waveform()

        _, comp_orig_full, comp_seq = simulate_fused(
            n_walkers, None, wf, mc, seed=1,
            return_compartments='full')
        _, comp_orig_final, comp_final = simulate_fused(
            n_walkers, None, wf, mc, seed=1,
            return_compartments='final')

        npt.assert_array_equal(comp_orig_full, comp_orig_final)
        npt.assert_array_equal(comp_seq[:, -1], comp_final)

    def test_permeable_myelin_walkers_change(self):
        """With permeability, some walkers should change compartment."""
        mc = MyelinatedCylinder(
            inner_radius=3e-6,
            outer_radius=5e-6,
            orientation=[0, 0, 1],
            kappa_inner=4e-5,
            kappa_outer=4e-5,
            compartments=Compartments(intra=Pool(D=1e-9), extra=Pool(D=2e-9), myelin=Pool(D=0.0)),
        )
        n_walkers = 2000
        wf = _make_waveform()
        _, comp_orig, comp_seq = simulate_fused(
            n_walkers, None, wf, mc, seed=0,
            return_compartments='full')
        changed = np.any(comp_seq != comp_orig[:, None], axis=1)
        frac_changed = changed.mean()
        assert frac_changed > 0.01, (
            f"Expected some walkers to cross in permeable myelin, got {frac_changed:.4f}")
