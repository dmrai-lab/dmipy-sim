"""Shared test constants and fixture loader."""

import numpy as np
import pytest
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Standard simulation parameters matching disimpy validation suite
D = 2e-9          # m²/s
N_WALKERS = 100_000   # overridden to 1_000_000 when --heavy is passed
SEED = 123

def pytest_addoption(parser):
    parser.addoption(
        "--heavy",
        action="store_true",
        default=False,
        help=(
            "High-N mode: run all MC tests with 1 000 000 walkers (10×) "
            "to measure systematic bias rather than statistical noise. "
            "Slower (~10×), but tolerances can be tightened."
        ),
    )

def pytest_configure(config):
    """Scale N_WALKERS before test files are collected and imported."""
    global N_WALKERS
    if config.getoption("--heavy", default=False):
        N_WALKERS = 1_000_000

def load_fixture(name):
    """Load a .npy fixture file, skipping the test if it is missing."""
    path = FIXTURE_DIR / name
    if not path.exists():
        pytest.skip(f"Fixture '{name}' missing — run scripts/generate_fixtures.py")
    return np.load(path)


# Test modules dominated by heavy CPU Monte-Carlo (measured per file: tens of seconds to minutes
# each). Mark every test in them `slow` so the default CI selection (-m "not slow and not gpu")
# stays fast (~1 min): it keeps the primitive / geometry / waveform unit tests plus a packed-myelin
# MC smoke and the fast permeability checks, while the heavy statistical MC-validation runs in the
# nightly / offline `slow` job. (Analytical parity on the dmipy-fit side is covered separately by
# committed MC fixtures there, with no live Monte Carlo.)
_SLOW_MC_MODULES = {
    "test_cylinder", "test_ellipsoid", "test_sphere", "test_mixture", "test_myelin",
    "test_box_1d", "test_free_1d", "test_free_3d", "test_free_ogse", "test_general_waveform",
    "test_packed_cylinders", "test_packed_spheres",
    "test_packed_cylinders_permeability", "test_ellipsoid_permeability",
    "test_sphere_permeability", "test_permeability_crossing", "test_bloch_permeation",
    "test_compartment_tagging", "test_t2", "test_sh_convolution",
    "test_karger_mc_parity", "test_t2_walker_parity",
    "test_mesh_mc",
    # Six mesh MC walks (three grid resolutions x two assertions). Collision sub-stepping makes each
    # one several times what it would otherwise cost, which is the point of it -- but it belongs in the
    # nightly job, not the ~1 min lane.
    "test_mesh_acceleration_invariance",
    "test_replay_parity",
    "test_replay_fields_mt",
    "test_engine_dispatch",
}


def pytest_collection_modifyitems(config, items):
    slow = pytest.mark.slow
    for item in items:
        mod = getattr(item, "module", None)
        name = mod.__name__.rsplit(".", 1)[-1] if mod is not None else ""
        if name in _SLOW_MC_MODULES:
            item.add_marker(slow)

def assert_step_resolves_the_collision_lookup(geometry, step_length, *, bound=0.9):
    """Fail unless a hand-picked step is short enough for the mesh's collision lookup to cover it.

    The candidate lookup gathers only the 27 cells around a step's START, so a step longer than a cell crosses
    triangles that were never candidates and the wall is simply missed. Any mesh test that asserts a
    confinement, escape or exchange number is measuring the LOOKUP rather than the physics once that bound is
    broken -- and it breaks silently.

    Measured on a permeable mesh sphere (R=5 um, subdivisions=4) at a permeability where no walker may
    legitimately cross, per 1 ms: step/cell 0.25 -> 0.00% escaped, 0.89 -> 0.00%, 1.79 -> 0.05%,
    3.31 -> 4.85%, 6.63 -> 26.2%. A 200 nm step against a 30.2 nm cell (ratio 6.6) once read as a 90% walker
    loss and was filed as a permeability bug (dmrai-lab/dmipy-sim#65) before being traced to the step choice;
    the engine's own rule for that mesh is 7.55 nm, i.e. 0.25 cells, and leaks nothing.

    The trap is that `_geometry_radius` returns `feature_radius` for a Mesh -- a MESHING parameter -- so a step
    derived from the PORE size can be far coarser than anything the engine would pick. Hence: assert, do not
    assume.

    A geometry with no ``cell_size`` (analytic pores) has no lookup to outrun and passes trivially.
    """
    cell = getattr(geometry, "cell_size", None)
    if not cell:
        return
    ratio = float(step_length) / float(cell)
    assert ratio <= bound, (
        f"step {float(step_length):.3e} m is {ratio:.2f} x the collision-lookup cell "
        f"({float(cell):.3e} m); the bound is {bound}. Above it the walk misses walls and any confinement or "
        f"exchange number measured here is an artefact of the lookup, not physics. Use more sub-steps.")
