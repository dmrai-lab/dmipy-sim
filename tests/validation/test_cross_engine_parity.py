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
alone -- and the ANALYTIC cylinder of the same radius meets it while the MESH of the same cylinder does not,
by 1.3-2.1e-2 at b = 3000 s/mm^2, a gap that does not close with the step. That is what this fixture found:
dmrai-lab/dmipy-sim#479.

Measured on an L40S at 8,000 walkers, recorded so the numbers can be checked rather than trusted:

| fixture | sub_steps | step | max\\|dS\\| | rms | our floor | their floor | x tolerance |
|---|---|---|---|---|---|---|---|
| MC/DC amp 0.2 wL 32 um | 1 (auto) | 268 nm | 0.01185 | 0.00538 | 0.01149 | 0.00317 | 1.41 |
| the same, at MC/DC's own step | 2 | 190 nm | 0.00819 | 0.00242 | 0.01149 | 0.00317 | 1.03 |

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
    p = X.parity(X.walk_cos_phi(walk, mcdc_scheme), ref, n_theirs)
    assert p["n_meas"] == 372
    assert p["worst_in_tolerance_units"] <= 1.0, (
        f"amp {amp} wL {wL}: max|dS| {p['max_abs_diff']:.5f} at measurement {p['at_measurement']} "
        f"(ours {p['ours_at']:.5f}, MC/DC {p['theirs_at']:.5f}); tolerance {p['tol_max']:.5f} "
        f"= 3 sqrt(our floor {p['floor_ours_max']:.5f}^2 + their floor {p['floor_theirs_max']:.5f}^2); "
        f"{p['n_over_tolerance']} of {p['n_meas']} measurements over it")


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
    p = X.parity(X.walk_cos_phi(walk, X.disimpy_sequence()), X.disimpy_reference(DISIMPY))
    assert p["n_meas"] == 100
    assert p["max_abs_diff"] <= max(4e-3, 3 * p["floor_ours_max"]), (
        f"max|dS| {p['max_abs_diff']:.5f} at measurement {p['at_measurement']} (ours {p['ours_at']:.5f}, "
        f"MISST {p['theirs_at']:.5f}); our floor {p['floor_ours_max']:.5f}")


@needs_disimpy
def test_the_cylinder_mesh_gap_to_misst_shrinks_with_the_step_but_not_to_the_floor():
    r"""The fixture itself: Disimpy's ``cylinder_mesh_closed.pkl`` against the MISST signal it ships.

    The mesh **under-restricts**. Measured at 8,000 walkers (the worst measurement is always the last,
    b = 3000 s/mm^2, where MISST gives 0.87630):

    | sub_steps | step | ours | max\|dS\| | our floor |
    |---|---|---|---|---|
    | 1 (auto) | 775 nm | 0.85554 | 0.02077 | 0.00086 |
    | 2 | 548 nm | 0.86051 | 0.01579 | 0.00145 |
    | 4 | 387 nm | 0.86147 | 0.01483 | 0.00207 |
    | 8 | 274 nm | 0.86332 | 0.01299 | 0.00118 |

    Refining the step by three closes a third of the gap, so extrapolating it does not reach the floor, and
    the analytic control above does. **That is dmrai-lab/dmipy-sim#479**, open, and this test asserts only
    what is true today: the gap shrinks monotonically with the step, and it stays inside 2.5e-2 -- the bound
    the repo's own ``tests/geometry/test_cylinder.py::test_cylinder_misst_config1`` already carries (0.02) for
    the same MISST configuration, with the measurement's own margin. The floor-level assertion that #479 is
    about is the strict xfail below.
    """
    X = fixtures()
    lad = X.disimpy_step_ladder(DISIMPY, N_WALKERS, sub_steps=(1, 2, 4))
    assert lad["monotone"], f"the gap does not shrink with the step: {lad['max_abs_diff']} at {lad['steps_m']}"
    assert lad["max_abs_diff"][-1] <= 2.5e-2, (
        f"gaps {[round(v, 5) for v in lad['max_abs_diff']]} at steps "
        f"{[round(h * 1e9) for h in lad['steps_m']]} nm")


@needs_disimpy
@pytest.mark.xfail(strict=True, reason="dmrai-lab/dmipy-sim#479: the mesh of a cylinder under-restricts by "
                                      "1.3-2.1e-2 at b = 3000 s/mm^2 and the gap does not close with the step")
def test_the_cylinder_mesh_lands_on_misst_within_our_floor():
    """What the fixture would assert if #479 were fixed: the mesh, like the analytic cylinder, within the floor."""
    X = fixtures()
    walk = X.disimpy_walk(DISIMPY, N_WALKERS, sub_steps=4)
    p = X.parity(X.walk_cos_phi(walk, X.disimpy_sequence()), X.disimpy_reference(DISIMPY))
    assert p["worst_in_tolerance_units"] <= 1.0


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
