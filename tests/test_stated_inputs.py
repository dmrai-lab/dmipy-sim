"""Physical and certified quantities come from stated inputs (dmipy-sim#847): a missing term, floor, z, diffusivity or
radius floor is refused by name, never read as a number nobody chose, and a check that cannot run raises instead of
being read as an answer."""
import dataclasses

import numpy as np
import pytest

from dmipy_sim import Compartments, Pool
from dmipy_sim.replay import bank


_TERMS = dict(truncation=1e-4, truncation_from="band_variance", quadrature=dict(positions=1e-5))
_WINDOW = dict(metric="max_abs_complex_signal_error", err_max=1e-3, floor_max=1e-3, terms=_TERMS)


def test_windows_of_one_walk_combine_their_terms_as_sums():
    out = bank.combine_window_fidelity([_WINDOW, _WINDOW])
    assert out["terms"]["truncation"] == pytest.approx(2e-4)
    assert out["terms"]["quadrature"] == {"positions": pytest.approx(2e-5)}


@pytest.mark.parametrize("key, window", [
    ("fidelity.terms.rounding", dict(_WINDOW, terms=dict(_TERMS, rounding=1e-6))),
    ("fidelity.terms.quadrature.field", dict(_WINDOW, terms=dict(_TERMS, quadrature=dict(positions=1e-5, field=1e-6)))),
    ("fidelity.noise_floor", dict(_WINDOW, noise_floor=1e-3)),
], ids=["rounding", "quadrature channel", "noise floor"])
def test_a_term_some_windows_lack_is_refused_not_summed_as_zero(key, window):
    with pytest.raises(ValueError, match=key.replace(".", r"\.")):
        bank.combine_window_fidelity([window, _WINDOW])


def test_a_certificate_without_its_bounded_terms_is_refused():
    with pytest.raises(KeyError, match="err_truncation"):
        bank._terms_of(dict(err_quadrature=1e-5))
    with pytest.raises(KeyError, match="truncation"):
        bank._cover_terms(dict(err_max=1e-3, terms=dict(quadrature=dict(positions=1e-5))))


def test_the_largest_b_is_the_class_or_the_battery_never_zero():
    assert bank._battery_b_max(dict(bvals=[0.0, 1e9], shortd_b=3e9), 0.1) == 3e9
    with pytest.raises(ValueError, match="no G_max"):
        bank._battery_b_max({}, 0.1)


def test_a_certificate_without_a_floor_has_no_precision_tiers():
    rec = bank._precision_tiers({"x": np.zeros((10, 3), np.float32)}, 10, None, True)
    assert rec["tiers"] == [] and rec["floor_at_full_n"] is None
    assert bank._precision_tiers({"x": np.zeros((10, 3), np.float32)}, 10, 1e-2, True)["tiers"]


def test_a_walk_clock_starts_at_its_first_step_unless_placed_later():
    from dmipy_sim.coefficient_walk import _t_start, _window_t0
    assert _t_start({}) == 0.0 and _window_t0({}) == 0.0
    assert _window_t0({"t_start": 0.2}) == 0.2
    assert _window_t0({"t_start": 0.2, "window": {"t0": 0.3}}) == 0.3


def test_a_calibration_step_without_its_z_is_refused():
    from tests.test_calibration_record import arguments
    from dmipy_sim.engine.calibration import FieldCalibration
    args = arguments()
    args["step"] = {k: v for k, v in args["step"].items() if k != "z"}
    with pytest.raises(ValueError, match="states no z"):
        FieldCalibration.create(**args)


def test_a_radius_law_states_its_floor():
    from dmipy_sim.spec.producers import _radii_from_law
    law = {"family": "gamma", "shape": 4.0, "scale": 0.5e-6}
    with pytest.raises(ValueError, match="r_min"):
        _radii_from_law(law, 10, 0, 3)
    assert _radii_from_law(dict(law, r_min=0.2e-6), 10, 0, 3).min() >= 0.2e-6


def test_the_white_matter_pools_carry_the_catalogue_T1():
    from dmipy_sim.spec.producers import wm_pools
    from dmipy_sim.substrate.biophysical_constants import canonical_white_matter
    c = canonical_white_matter(field_T=3.0)
    assert [p.T1 for p in wm_pools(3.0)] == [c["T1_extra"], c["T1_intra"], c["T1_myelin"]]


def test_a_myelinated_spec_pool_without_D_is_refused_never_frozen():
    from dmipy_sim.geometry import MyelinatedCylinder
    from dmipy_sim.spec import geometry_from_spec, spec_of
    g = MyelinatedCylinder(3e-6, 5e-6, (0, 0, 1),
                           compartments=Compartments(intra=Pool(D=2e-9), extra=Pool(D=2e-9), myelin=Pool(D=0.0)))
    spec = spec_of(g)
    no_D = dataclasses.replace(spec, pools=[dataclasses.replace(p, D=None) if p.name == "intra" else p for p in spec.pools])
    with pytest.raises(ValueError, match="intra"):
        geometry_from_spec(no_D)


def test_a_mesh_closedness_check_that_fails_raises_rather_than_reading_open(monkeypatch):
    trimesh = pytest.importorskip("trimesh")
    from dmipy_sim import Mesh
    m = trimesh.creation.icosphere(subdivisions=2, radius=5e-6)
    mesh = Mesh(np.asarray(m.vertices, float), np.asarray(m.faces, int))

    def broken(*a, **k):
        raise RuntimeError("the edge grouping failed")
    monkeypatch.setattr(trimesh.grouping, "group_rows", broken)
    mesh._closed = None
    with pytest.raises(RuntimeError, match="edge grouping failed"):
        mesh._surface_is_closed()


def test_a_geometry_whose_walls_cannot_be_read_raises_rather_than_reading_unconfined():
    from dmipy_sim.engine.physics import tolerance_sub_steps
    from dmipy_sim.geometry.base import FreeDiffusion

    class Walled(FreeDiffusion):
        def surface_area(self):
            raise RuntimeError("no surface area")

        def volume(self):
            return 1e-15

    assert tolerance_sub_steps(FreeDiffusion(), 2e-9, 1e-4) == 1
    with pytest.raises(RuntimeError, match="no surface area"):
        tolerance_sub_steps(Walled(), 2e-9, 1e-4)


def test_the_catalogue_states_chi_A_in_the_package_convention_with_its_source_value():
    """Wharton & Bowtell's tensor is chi_I I + diag(chi_A, -chi_A/2, -chi_A/2); the package's is
    chi_I I + chi_A (n n^T - I/3), so its chi_A is (3/2) of theirs. The entry carries both numbers."""
    from dmipy_sim.substrate.biophysical_constants import canonical_white_matter, get_constant
    default = get_constant("delta_chi_a_myelin")["default"]
    assert default["source_key"] == "wharton2012" and default["location"].startswith("Table 2")
    assert default["value"] == pytest.approx(1.5 * default["reported_value"], rel=1e-12)
    assert canonical_white_matter(field_T=3.0)["delta_chi_a"] == default["value"]
    assert canonical_white_matter(field_T=3.0)["chi_iso_myelin"] == -0.1e-6      # chi_I is the same in both conventions
