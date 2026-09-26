"""Cross-engine parity: this engine against MC/DC's released signals and against MISST, on their own geometry.

Two published references, each walked here on the SAME surface at the SAME acquisition and compared at signal
level. The reproduction is ``examples/validation/cross_engine_parity.py``; these are its assertions.

**MC/DC** (Rafael-Patino et al. 2020, Front. Neuroinform. 14:8, LGPL-2.1,
https://github.com/jonhrafe/Robust-Monte-Carlo-Simulations) released, for every undulating axon, the PLY, the
initial-walker list, the ``ActiveAxG140_PM.scheme`` ActiveAx protocol (372 measurements, TE 53.52 ms, four
shells at b = 1925 / 1932 / 3094 / 13190 s/mm^2) and the raw ``*_DWI.bfloat`` their engine produced. Their
signal is a Monte-Carlo estimate too, so the tolerance is BOTH floors: theirs is the standard error at the
walker count their b = 0 entry states (50,000) with the variance of ``cos(phi)`` measured on this walk, ours
is this walk's split half. The test asserts ``|dS| <= 3 sqrt(ours^2 + theirs^2)`` on every measurement.

**Disimpy** (Kerkelae et al. 2020, JOSS 5(52):2527, MIT) released ``tests/cylinder_mesh_closed.pkl`` and the
MISST signal it is asserted against. MISST is an exact eigenfunction solution, so the tolerance is our floor
alone. Both the analytic cylinder and the mesh meet it -- the mesh only since dmrai-lab/dmipy-sim#483, which
this fixture found (#479): 294 of the pickle's 588 faces are wound the other way, so the sidedness test called
half the lumen exterior and ``reject_escape`` discarded 15.6 % of steps, putting the walk 1.3-2.1e-2 below
MISST with a gap that did not close with the step.

Measured on an L40S at ``N_WALKERS`` = 8,000 -- **this test's own default** -- and recorded so the numbers can
be checked rather than trusted. Every fixture passes here, which is the point of a default: 0 of the 360 live
measurements outside their own 3-sigma band, against the 4 chance allows, and the worst well inside the
3.198-sigma family-wise band. The 12 zero-variance (b = 0) rows agree exactly.

| fixture | max\\|dS\\| | rms | our floor | their floor | worst | over 3 sigma |
|---|---|---|---|---|---|---|
| MC/DC amp 0.2 wL 32 um | 0.01185 | 0.00538 | 0.00792 | 0.00317 | 1.992 sigma | 0 of 360 |
| MC/DC amp 1.0 wL 12 um | 0.01124 | 0.00417 | 0.00807 | 0.00323 | 1.791 sigma | 0 of 360 |
| MC/DC amp 2.6 wL 4 um | 0.00811 | 0.00349 | 0.00681 | 0.00273 | 2.015 sigma | 0 of 360 |

The published packs walk 100,000 and reach 2.87 sigma at worst; a smaller count is not a weaker test, because
the band is the walk's own standard error and shrinks with it.

The step matters too, and at 8,000 walkers it is what the MC/DC fixture is really about: MC/DC walked these
axons in T = 5000 steps of 10.7 us -- a 196 nm step against a 500 nm lumen radius -- and this engine's rule
takes its own, which on this substrate is COARSER (268 nm). Walking at their step moved max|dS| on amp 0.2 /
wL 32 um from 0.01185 to 0.00819, so part of what is left between the engines is the step and not the engine,
which is what their paper is about.

The second row is the point of the MC/DC fixture. MC/DC walked these axons in T = 5000 steps of 10.7 us -- a
196 nm step against a 500 nm lumen radius -- and this engine's sub-step rule takes its own, which on this
substrate comes out COARSER (268 nm). Walking at their step moves our signal a third of the way to theirs, so
part of what is left between the two engines is the step and not the engine, which is what their paper is
about. At the walker count the packs are built with, the tolerance is set by THEIR 50,000 walkers rather than
by ours -- measured on amp 0.2 / wL 32 um at 100,000 walkers: max abs dS **0.00688** against a tolerance of
**0.01344** (our floor 0.00360 against their 0.00317), and the published pack reproduces the walk's own number
to 7e-6.

The data are not in the repository. Point ``DMIPY_SIM_MCDC_ROBUST_DIR`` at a checkout of
Robust-Monte-Carlo-Simulations and ``DMIPY_SIM_DISIMPY_DIR`` at a checkout of disimpy to run this.
"""
import importlib
import os

import numpy as np
import pytest

MCDC = os.environ.get("DMIPY_SIM_MCDC_ROBUST_DIR")
DISIMPY = os.environ.get("DMIPY_SIM_DISIMPY_DIR")
N_WALKERS = int(os.environ.get("DMIPY_SIM_PARITY_N", "8000"))


def fixtures():
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("examples.validation.cross_engine_parity")


def _thresholds(parity):
    """The derived thresholds for a parity record -- the same derivation the gate uses, from the same place, so
    a test cannot assert a looser rule than the family publishes under."""
    return fixtures().multiplicity_thresholds(parity["n_live"], parity["dof"], k_per=parity["k"])


needs_mcdc = pytest.mark.skipif(not MCDC, reason="set DMIPY_SIM_MCDC_ROBUST_DIR to the MC/DC data checkout")
needs_disimpy = pytest.mark.skipif(not DISIMPY, reason="set DMIPY_SIM_DISIMPY_DIR to a disimpy checkout")


@pytest.fixture(scope="module")
def mcdc_scheme():
    X = fixtures()
    from dmipy_sim.io import mcdc
    os.environ["DMIPY_SIM_SURFACE_DIR"] = os.path.abspath(
        os.path.join(MCDC, "Experiments-mesh -files", "Undulated-fibres", "d_1"))
    return mcdc.read_scheme(os.path.join(MCDC, "Simulator-Conf-files", X.MCDC_SCHEME), n_t=X.MCDC_N_T)


@needs_mcdc
@pytest.mark.parametrize("amp,wL", [(0.2, 32.0), (1.0, 12.0), (2.6, 4.0)])
def test_the_signal_reproduces_mcdcs_released_dwi_within_the_two_floors(mcdc_scheme, amp, wL):
    """Every one of the 372 ActiveAx measurements, against their released ``*_DWI.bfloat``."""
    X = fixtures()
    ref, n_theirs = X.mcdc_reference(MCDC, amp, wL, mcdc_scheme)
    assert n_theirs == 50_000
    walk = X.mcdc_walk(MCDC, amp, wL, N_WALKERS)
    assert walk.illegal_crossings == 0
    p = X.parity(X.walk_cos_phi(walk, mcdc_scheme), ref, n_theirs,
                 reference_quantisation=X.MCDC_QUANTISATION)
    mult = _thresholds(p)
    assert (p["n_meas"], p["n_live"], p["n_exact"]) == (372, 360, 12)
    assert p["max_abs_diff_exact"] == 0.0, "a b = 0 measurement must agree exactly: it has no band"
    assert p["worst_sigma"] <= mult["k_family_wise"], (
        f"amp {amp} wL {wL}: the worst measurement is at {p['worst_sigma']:.3f} sigma of its own standard "
        f"error (measurement {p['worst_sigma_at']}; max|dS| {p['max_abs_diff']:.5f}; our floor "
        f"{p['floor_ours_max']:.5f}, theirs {p['floor_theirs_max']:.5f}) against the family-wise band "
        f"{mult['k_family_wise']:.3f} sigma for {p['n_live']} measurements on {p['dof']} dof")
    assert p["n_over_k_sigma"] <= mult["max_exceedances"], (
        f"amp {amp} wL {wL}: {p['n_over_k_sigma']} of {p['n_live']} measurements outside their own "
        f"{p['k']:g}-sigma band, against the {mult['max_exceedances']} chance allows "
        f"(expected {mult['expected_exceedances']:.3f})")


@needs_mcdc
def test_the_walk_stays_inside_the_axon_and_never_reaches_the_domain(mcdc_scheme):
    """An intra-only walk in a closed tube: no walker leaves the lumen, and the domain faces are never met, so
    what MC/DC's ``<voxels>`` block said about them cannot matter. Measured, not assumed."""
    X = fixtures()
    spec = X.mcdc_spec(MCDC, 1.0, 12.0)
    walk = X.mcdc_walk(MCDC, 1.0, 12.0, 2000)
    r = np.asarray(walk.positions).reshape(-1, 3)
    lo, hi = np.asarray(spec.domain.box_min), np.asarray(spec.domain.box_max)
    assert walk.illegal_crossings == 0
    comp = np.asarray(walk.compartment)                      # the walk's own occupancy channel
    assert np.all(comp == comp[:, :1]), "a walker changed pool through a wall with no crossing granted"
    margin = np.minimum(r.min(0) - lo, hi - r.max(0))
    assert np.all(margin > 0), f"the walk reached the domain face: margin {margin} m"


@needs_mcdc
def test_the_uncapped_meshes_are_refused_as_substrates():
    """10 of the 31 released meshes are uncapped tubes -- 40 boundary edges each -- and enclose no volume. MC/DC
    walks them because its walkers never reach the rim; a spec refuses them, by name."""
    from dmipy_sim.spec import SpecError, mcdc_axon_spec
    ply = os.path.join(MCDC, "Experiments-mesh -files", "Undulated-fibres", "d_1",
                       "uAxon_d_1.0_amp_0.0_wL_4.0.ply")
    with pytest.raises(SpecError, match="closed surface"):
        mcdc_axon_spec(ply, scale=1e-6)


@needs_disimpy
def test_the_analytic_cylinder_of_the_same_radius_lands_on_misst():
    """The control. The same waveform, seed and walker count on ``Cylinder(radius=5e-6)`` instead of the mesh.

    MISST is an exact eigenfunction solution, so the tolerance here IS our split-half floor alone, and it is
    met: measured 1.18e-3 against a floor of 5.4e-4 at 8,000 walkers and sub_steps 2 (2.65e-3 at sub_steps 4).
    This is what makes the mesh's larger gap in the next test a property of the surface and not of the engine.
    """
    X = fixtures()
    walk = X.disimpy_analytic_walk(N_WALKERS, sub_steps=2)
    p = X.parity(X.walk_cos_phi(walk, X.disimpy_sequence()), X.disimpy_reference(DISIMPY),
                 reference_quantisation=X.DISIMPY_QUANTISATION)
    assert p["n_meas"] == 100
    assert p["max_abs_diff"] <= max(4e-3, 3 * p["floor_ours_max"]), (
        f"max|dS| {p['max_abs_diff']:.5f} at measurement {p['at_measurement']} (ours {p['ours_at']:.5f}, "
        f"MISST {p['theirs_at']:.5f}); our floor {p['floor_ours_max']:.5f}")


@needs_disimpy
def test_the_cylinder_mesh_signal_does_not_depend_on_the_step():
    r"""A converged walk's signal does not move with the step beyond its own noise -- the property that #479
    broke and #483 restored, kept as a regression guard.

    Before #483 the gap to MISST was 2.08e-2 at a 775 nm step, 1.58e-2 at 548, 1.48e-2 at 387 and 1.30e-2 at
    274: a step dependence that did not extrapolate to the floor, because ``reject_escape`` discarded the steps
    of the walkers the sidedness test put outside the lumen, and the fraction it discarded depends on the step.
    A sidedness or confinement defect shows itself exactly that way, so what is asserted is that refining the
    step by a factor of two changes nothing the floor cannot explain.
    """
    X = fixtures()
    lad = X.disimpy_step_ladder(DISIMPY, N_WALKERS, sub_steps=(4, 8))
    mult = fixtures().multiplicity_thresholds(lad["n_live"], lad["dof"])
    assert lad["coarsest_to_finest_sigma"] <= mult["k_family_wise"], (
        f"halving the step moved the signal by {lad['coarsest_to_finest_sigma']:.2f} sigma of our own floor "
        f"(gaps {[round(v, 5) for v in lad['max_abs_diff']]} at steps "
        f"{[round(h * 1e9) for h in lad['steps_m']]} nm), which a converged walk does not do")
    assert max(lad["worst_sigma"]) <= mult["k_family_wise"], (
        f"worst measurement per step: {[round(v, 2) for v in lad['worst_sigma']]} sigma, band "
        f"{mult['k_family_wise']:.3f}")


@needs_disimpy
def test_the_disimpy_pickle_needs_orienting_and_gets_it():
    """The fixture's stored winding, and that the walked geometry no longer has it.

    Of the merged pickle's 588 faces **294 point one way and 294 the other**, and 784 of its directed edges are
    unpaired, so its divergence-theorem volume is 0.332 of ``pi r^2 L`` while its AREA is 0.999 of the ideal --
    the signed contributions cancel. Reflection never cared (a facet normal is signed by the step), which is why
    the leak was nil and the defect silent; everything that reads the SIGN of a normal did, and
    ``_classify_arr`` called half the lumen exterior.

    ``orient_faces`` at ``Mesh.__init__`` (dmrai-lab/dmipy-sim#483) reorients 294 of them and leaves 0
    inconsistent edges, so the geometry the walk sees is consistent and its enclosed volume is 0.997 of the
    ideal. Both halves are asserted here: the input still needs the repair, and the geometry has had it.
    """
    X = fixtures()
    from dmipy_sim.geometry.mesh import orient_faces
    V, F, _ = X.disimpy_mesh(DISIMPY)
    _, winding = orient_faces(V, F)
    assert (winding["inconsistent_edges"], winding["reoriented"]) == (784, 294)
    assert winding["inconsistent_edges_after"] == 0
    got = X.disimpy_geometry(DISIMPY).winding
    assert got["inconsistent_edges_after"] == 0
    assert abs(got["volume_after"]) / (np.pi * X.DISIMPY_RADIUS ** 2 * 25e-6) == pytest.approx(0.997, abs=0.002)


@needs_disimpy
def test_the_cylinder_mesh_lands_on_misst_within_our_floor():
    """The fixture itself: Disimpy's cylinder mesh against the MISST signal it ships, at MISST's exact
    30 / 40 ms timing, on the ORIENTED surface.

    MISST is an exact eigenfunction solution, so the band here is OUR standard error alone -- there is no second
    Monte-Carlo run to widen it -- and the mesh meets it. Before #483 oriented the faces it did not: the walk sat
    1.3-2.1e-2 below MISST and the gap did not close with the step, because ``reject_escape`` discarded the steps
    of every walker the sidedness test put on the wrong side.

    The analytic cylinder of the same radius is the control
    (:func:`test_the_analytic_cylinder_of_the_same_radius_lands_on_misst`): whatever the two share is the
    engine's, and what differs is the surface's representation.
    """
    X = fixtures()
    walk = X.disimpy_walk(DISIMPY, N_WALKERS, sub_steps=8)
    # A capped cylinder's rim is a sharp edge where the wall meets a cap, and the confinement guard refuses the
    # occasional step there rather than letting it through; #483 made those refusals countable. Measured: 4 of
    # 8,000 x 1400 x 8 steps at sub_steps 8 and 1 of 8,000 x 1400 at sub_steps 1, i.e. a rate below 1e-7. What
    # must hold is that the guard is RARE, not that it never fires -- a guard that never fires on a rim is a
    # guard that is not looking.
    steps = N_WALKERS * (X.DISIMPY_N_T - 1) * int(walk.sub_steps)
    assert walk.illegal_crossings / steps <= 1e-6, (
        f"{walk.illegal_crossings} refused steps of {steps} is a rate of "
        f"{walk.illegal_crossings / steps:.2e}, which is not the rim")
    p = X.parity(X.walk_cos_phi(walk, X.disimpy_sequence()), X.disimpy_reference(DISIMPY),
                 reference_quantisation=X.DISIMPY_QUANTISATION)
    mult = _thresholds(p)
    assert p["n_meas"] == 100
    assert p["worst_sigma"] <= mult["k_family_wise"], (
        f"the worst measurement is at {p['worst_sigma']:.3f} sigma of our own standard error (measurement "
        f"{p['worst_sigma_at']}; max|dS| {p['max_abs_diff']:.5f}, our floor {p['floor_ours_max']:.5f}) against "
        f"the family-wise band {mult['k_family_wise']:.3f} sigma for {p['n_live']} measurements on "
        f"{p['dof']} dof")
    assert p["n_over_k_sigma"] <= mult["max_exceedances"], (
        f"{p['n_over_k_sigma']} of {p['n_live']} measurements outside their own {p['k']:g}-sigma band, against "
        f"the {mult['max_exceedances']} chance allows")


@needs_disimpy
def test_the_disimpy_cylinder_is_closed_once_its_duplicate_vertices_are_merged():
    """``cylinder_mesh_closed.pkl`` writes a coincident duplicate of every seam vertex, so as stored it reads
    as OPEN with 112 boundary edges. Merged it is 296 vertices, 588 faces, 0 boundary edges, and its area is
    0.999 of the ideal cylinder plus caps -- which is why the name is right and the raw read is not."""
    X = fixtures()
    V, F, rep = X.disimpy_mesh(DISIMPY)
    assert rep["boundary_edges_before"] == 112 and rep["boundary_edges_after"] == 0
    assert (rep["vertices_before"], rep["vertices_after"]) == (352, 296)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    ideal = 2 * np.pi * X.DISIMPY_RADIUS * 25e-6 + 2 * np.pi * X.DISIMPY_RADIUS ** 2
    assert area / ideal == pytest.approx(0.999, abs=0.002)
