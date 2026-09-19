"""The Gamma diameter law describes the OUTER (fibre) diameter, including myelin.

This is the one convention in `Substrate` that cannot defend itself. Eight derived properties depend on it,
and reading the same `gamma_scale_diameter` as an AXON diameter instead leaves every one of them internally
consistent: nothing raises, no invariant breaks, and each formula still agrees with its neighbours. What
changes is the physical size of the fibre -- mean 0.608 um becomes 0.869 um -- and with it every
surface-to-volume and surface-relaxivity rate, by 1/g ~ 1.43x.

A defect that self-consistent is not caught by testing the code against itself, so these pin the convention
to its source instead: the cited Gamma fit (Aboitiz 1992, "Fiber composition of the human corpus callosum")
is a fit to FIBRE diameters, and the constants table must keep saying so.
"""
from __future__ import annotations

import math

import pytest

from dmipy_sim.substrate.substrate import Substrate
from dmipy_sim.substrate import biophysical_constants as bc


def _sub(**kw):
    return Substrate(**kw)


def test_the_gamma_law_is_the_outer_fibre_diameter():
    """mean outer radius = alpha*beta/2 -- the Gamma mean, halved, with NO g-ratio applied.

    This is the load-bearing assertion. Under the axon-diameter reading this expression would instead give
    the INNER radius, and the outer would be alpha*beta/(2g).
    """
    s = _sub(gamma_shape_diameter=2.0, gamma_scale_diameter=0.304e-6, g_ratio=0.7)
    assert s.mean_outer_radius == pytest.approx(0.5 * 2.0 * 0.304e-6)
    assert s.mean_outer_radius * 2 == pytest.approx(0.608e-6), "mean FIBRE diameter, per Aboitiz 1992"


def test_the_lumen_follows_from_the_fibre_by_the_g_ratio():
    """Inner = g * outer, in both the mean and the spread. The direction matters: the fibre is measured and
    the lumen derived, not the other way round."""
    s = _sub(gamma_shape_diameter=2.0, gamma_scale_diameter=0.304e-6, g_ratio=0.7)
    assert s.mean_inner_radius == pytest.approx(s.g_ratio * s.mean_outer_radius)
    assert s.std_inner_radius == pytest.approx(s.g_ratio * s.std_outer_radius)
    assert s.mean_inner_radius < s.mean_outer_radius


@pytest.mark.parametrize("alpha,beta,g", [(2.0, 0.304e-6, 0.7), (3.1, 0.21e-6, 0.62), (1.4, 0.55e-6, 0.8)])
def test_derived_geometry_matches_its_closed_form(alpha, beta, g):
    """Each derived quantity against the Gamma moments it is built from.

    Pinned per-property rather than as one aggregate: a sign or a stray g_ratio in any single formula is
    what a convention flip actually looks like, and an aggregate check can absorb it.
    """
    s = _sub(gamma_shape_diameter=alpha, gamma_scale_diameter=beta, g_ratio=g)
    # Gamma(alpha, beta): mean = alpha*beta, var = alpha*beta^2, E[d^2] = alpha(alpha+1)beta^2
    assert s.mean_outer_radius == pytest.approx(0.5 * alpha * beta)
    assert s.std_outer_radius == pytest.approx(0.5 * math.sqrt(alpha) * beta)
    assert s.mean_sq_outer_radius == pytest.approx(alpha * (alpha + 1.0) * beta ** 2 / 4.0)


@pytest.mark.parametrize("f_axon", [0.35, 0.55, 0.72])
def test_extra_axonal_surface_to_volume_is_built_on_the_outer_diameter(f_axon):
    """S/V of the extra-axonal space sees the OUTER surface -- what the extra-axonal water touches is the
    outside of the sheath, so no g-ratio enters. This is the quantity that sets surface relaxivity, and it
    is where the 1/g error would land if the convention flipped."""
    alpha, beta = 2.0, 0.304e-6
    s = _sub(gamma_shape_diameter=alpha, gamma_scale_diameter=beta, g_ratio=0.7, f_axon=f_axon)
    expected = 4.0 * f_axon / ((1.0 - f_axon) * (alpha + 1.0) * beta)
    assert s.S_ext_over_V_EA == pytest.approx(expected, rel=1e-12)


def test_intra_axonal_surface_rate_uses_the_lumen_not_the_fibre():
    """The intra-axonal side sees the INNER wall, so its rate must scale with g. Rising g (a thinner sheath,
    a wider lumen at fixed fibre size) must lower the intra surface-to-volume."""
    kw = dict(gamma_shape_diameter=2.0, gamma_scale_diameter=0.304e-6)
    thin, thick = _sub(g_ratio=0.85, **kw), _sub(g_ratio=0.55, **kw)
    assert thin.intra_surface_rate < thick.intra_surface_rate


def test_the_constants_table_still_describes_a_fibre_diameter_fit():
    """Code and provenance must not drift apart.

    The convention lives in two places -- the formulas above and the citation metadata -- and a flip in
    either alone is silent. Aboitiz 1992 fits FIBRE diameters, so the table must keep saying so.
    """
    entry = bc.BIOPHYSICAL_CONSTANTS["gamma_scale_diameter"]
    method = str(entry["default"]["method"]).lower()
    location = str(entry["default"].get("location", "")).lower()
    assert "fibre" in method or "fiber" in method, (
        f"gamma_scale_diameter is documented as {method!r}; it must describe a FIBRE (outer) diameter fit, "
        f"which is what Aboitiz 1992 reports and what Substrate's derived geometry assumes")
    assert "outer" in location or "fibre" in location or "fiber" in location
    assert entry["default"]["value"] == pytest.approx(0.304e-6)


def test_canonical_substrate_carries_a_physiological_fibre_size():
    """End-to-end sanity on the shipped defaults: a mean fibre diameter under a micron with a lumen a little
    smaller. Under the axon-diameter reading the fibre would come out at 0.869 um, outside the range the
    cited histogram describes."""
    s = Substrate.canonical()
    d_outer_um = 2e6 * s.mean_outer_radius
    assert 0.5 < d_outer_um < 0.75, f"mean fibre diameter {d_outer_um:.3f} um is not the Aboitiz value"
    assert s.mean_inner_radius < s.mean_outer_radius


def test_the_64_mT_rows_are_cited_and_field_matched():
    """A 64 mT replay reads its tissue from the table like any other field (dmipy-sim#285). The rows are
    Jordanova 2023's in vivo measurements on a Hyperfine Swoop, and they do not follow the 3 T values by any
    single rule: both tissues relax SLOWER at 64 mT than at 3 T, while the CSF figure is well under half of
    it -- which is why the table stores them and refuses to interpolate."""
    at64 = {n: bc.get_value(n, field_T=0.064) for n in ("T2_white_matter", "T2_grey_matter", "T2_csf")}
    assert at64 == {"T2_white_matter": 0.081, "T2_grey_matter": 0.108, "T2_csf": 1.166}
    assert at64["T2_white_matter"] > bc.get_value("T2_white_matter", field_T=3.0)      # 81 ms vs 69 ms
    assert at64["T2_grey_matter"] > bc.get_value("T2_grey_matter", field_T=3.0)        # 108 ms vs 99 ms
    assert at64["T2_csf"] < 0.6 * bc.get_value("T2_csf", field_T=3.0)                  # 1166 ms vs 2000 ms
    for name in at64:
        alt = [a for a in bc.get_constant(name)["alternatives"] if a["field_T"] == 0.064]
        assert len(alt) == 1 and alt[0]["source_key"] == "jordanova2023" and alt[0]["location"]


def test_the_64_mT_T1_rows_say_they_are_unverified():
    """T1 at 64 mT is entered so the Swoop experiment has a number, and flagged so nobody mistakes it for a
    checked one: the source is paywalled and the secondary figures disagreed. T2 at the same field was read
    from two independent reports and carries no such flag. A row that loses its note has been promoted by
    someone who must have verified it -- which is the point of asserting this."""
    for name in ("T1_white_matter", "T1_grey_matter"):
        alt = [a for a in bc.get_constant(name)["alternatives"] if a["field_T"] == 0.064]
        assert len(alt) == 1, f"{name} has no 0.064 T row"
        assert "UNVERIFIED" in alt[0]["note"] and alt[0]["source_key"] == "jordanova2023"
    for name in ("T2_white_matter", "T2_grey_matter", "T2_csf"):
        alt = [a for a in bc.get_constant(name)["alternatives"] if a["field_T"] == 0.064][0]
        assert "UNVERIFIED" not in alt.get("note", "")
    # T1 collapses at low field where T2 barely moves: the reason a field-matched read matters more for T1.
    assert bc.get_value("T1_white_matter", 0.064) < 0.4 * bc.get_value("T1_white_matter", 3.0)
    assert bc.get_value("T2_white_matter", 0.064) > 1.1 * bc.get_value("T2_white_matter", 3.0)


def test_whole_tissue_white_matter_is_not_the_three_pool_decomposition():
    """`T2_white_matter` is what a segmented image reports, one pool. The myelinated substrate's three pools
    are separate constants and must not be confused with it: the single pool sits between them."""
    one = bc.get_value("T2_white_matter", field_T=3.0)
    assert bc.get_value("T2_myelin") < one < bc.get_value("T2_extra_axonal")
    assert "not the three-pool decomposition" in bc.get_constant("T2_white_matter")["description"].lower()


def test_the_tissue_values_a_phantom_declares_are_catalogued_with_a_citation():
    """No magic numbers: a brain phantom's grey-matter T2, and every tissue's proton density, come from the
    table with a source key, a location and a citation; the field-dependent ones are field-matched."""
    for name in ("T2_grey_matter", "T1_grey_matter", "proton_density_white_matter", "proton_density_grey_matter",
                 "proton_density_csf"):
        e = bc.get_constant(name)
        assert e["citation"]["key"] == e["default"]["source_key"] and e["default"]["location"]
        assert e["citation"]["doi"]
    assert bc.get_value("T2_grey_matter", field_T=3.0) == 0.099 and bc.get_value("T2_grey_matter", field_T=1.5) == 0.095
    assert bc.get_value("proton_density_csf") == 1.0 > bc.get_value("proton_density_grey_matter") > bc.get_value("proton_density_white_matter")
