"""GreyMatterSubstrate builder — iron placement and mass conservation.

The cortical GM substrate packs somata and anchors the non-heme-iron susceptibility
field to a glial subpopulation.  The invariants the builder must satisfy (independent
of the cited numeric values):

  * perturbers are a subset of the packed somata (iron is IN glial cells, not free ECS);
  * volume-averaged susceptibility is conserved: Δχ_glia · f_glia_volume = Δχ_tissue;
  * the achieved packing fraction matches f_cell and the glia count matches the fraction;
  * the resulting geometry simulates to a finite, bounded signal.
"""
import numpy as np
import pytest

from dmipy_sim import simulate, set_b
from dmipy_sim.waveforms import pgse
from dmipy_sim.substrate import GreyMatterSubstrate


def _gm(**overrides):
    kw = dict(f_cell=0.18, f_ecs=0.20, soma_diameter_mean=12e-6, soma_diameter_cv=0.30,
              glia_number_fraction=0.5, D_intra=1e-9, kappa=2e-5, rho2=0.0,
              T2=0.06, T1=1.4, iron_concentration=40.0, chi_per_iron=1.0e-9, field_T=3.0)
    kw.update(overrides)
    return GreyMatterSubstrate(**kw)


def test_iron_perturbers_are_glial_somata():
    geom, info = _gm().build(n_cells=40, seed=0)
    cc = geom._centers_np
    pc = geom.off_resonance.centers
    assert geom.off_resonance.n_perturbers == info['n_glia'] == 20
    # every perturber coincides with a packed soma
    assert all(any(np.allclose(p, c) for c in cc) for p in pc)


def test_volume_averaged_susceptibility_conserved():
    gm = _gm()
    geom, info = gm.build(n_cells=40, seed=0)
    assert info['delta_chi_glia'] * info['f_glia_volume'] == pytest.approx(
        gm.delta_chi_tissue, rel=1e-6)
    assert gm.delta_chi_tissue == pytest.approx(40.0 * 1.0e-9)


def test_packing_and_glia_counts():
    geom, info = _gm(glia_number_fraction=0.4).build(n_cells=50, seed=1)
    assert info['achieved_vf'] == pytest.approx(0.18, rel=1e-6)
    assert info['n_glia'] == 20        # 0.4 * 50


def test_zero_iron_gives_no_field():
    geom, info = _gm(iron_concentration=0.0).build(n_cells=30, seed=0)
    assert geom.off_resonance is None


def test_builds_simulable_geometry():
    gm = _gm()
    geom, info = gm.build(n_cells=30, seed=0)
    wf = set_b(pgse(delta=5e-3, DELTA=40e-3, G_magnitude=0.1, bvecs=[[1, 0, 0]],
                    n_t=300), 1e9)
    sig = float(np.asarray(simulate(2000, gm.D_intra, wf, geom, seed=0, T2=gm.T2,
                                    require_gpu=False))[0])
    assert np.isfinite(sig) and 0.0 <= sig <= 1.0


def test_validation_rejects_impossible_fractions():
    with pytest.raises(ValueError):
        _gm(f_cell=0.9, f_ecs=0.3)


def test_canonical_calibrates_static_r2prime_to_cited_cortex():
    from dmipy_sim.substrate import get_value
    gm = GreyMatterSubstrate.canonical(field_T=3.0)          # calibrate_r2prime=True
    target = get_value('gm_R2prime_cortex', 3.0, allow_nearest=True)
    assert gm.static_r2prime == pytest.approx(target, rel=1e-6)
    assert 0.0 < gm.iron_clustered_fraction < 1.0           # cortex is mostly smooth
    assert gm.delta_chi_effective < gm.delta_chi_tissue


def test_static_r2prime_scales_with_iron():
    cortex = GreyMatterSubstrate.canonical(field_T=3.0)
    gp = GreyMatterSubstrate.canonical(field_T=3.0, iron_concentration=213.0)
    # same clustered fraction (calibrated at cortex) -> R2' scales with iron
    assert gp.static_r2prime == pytest.approx(cortex.static_r2prime * 213.0 / 30.0, rel=1e-6)


def test_uncalibrated_is_mass_conservation_upper_bound():
    gm = GreyMatterSubstrate.canonical(field_T=3.0, calibrate_r2prime=False)
    assert gm.iron_clustered_fraction is None
    assert gm.delta_chi_effective == pytest.approx(gm.delta_chi_tissue)


def test_mt_transverse_rate_is_kf():
    gm = GreyMatterSubstrate.canonical(field_T=3.0)
    assert gm.mt_transverse_rate == pytest.approx(gm.mt_exchange_rate * gm.mt_bound_fraction)
    # cited GM: R=40, f=0.05 -> k_f = 2.0/s
    assert gm.mt_transverse_rate == pytest.approx(2.0, rel=1e-6)


def test_mt_longitudinal_rate_reduced_by_back_exchange():
    # k_r >> R1b, so the stored-magnetization loss is far below the naive k_f
    gm = GreyMatterSubstrate.canonical(field_T=3.0, T1=1.820)   # Stanisz cross-check
    kf = gm.mt_transverse_rate
    r1_mt = gm.mt_longitudinal_rate(0.035)
    assert 0.0 < r1_mt < kf                       # reduced below k_f
    # research: ~3.8% loss over T_M=35 ms
    assert 1.0 - np.exp(-r1_mt * 0.035) == pytest.approx(0.038, abs=0.005)


def test_mt_longitudinal_zero_without_mt():
    gm = GreyMatterSubstrate.canonical(field_T=3.0, mt_bound_fraction=0.0)
    assert gm.mt_transverse_rate == 0.0
    assert gm.mt_longitudinal_rate(0.035) == 0.0
