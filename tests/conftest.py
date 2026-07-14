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
    "test_sphere_permeability", "test_permeability_crossing",
    "test_compartment_tagging", "test_t2", "test_sh_convolution",
    "test_karger_mc_parity", "test_t2_walker_parity",
    "test_mesh_mc",
}


def pytest_collection_modifyitems(config, items):
    slow = pytest.mark.slow
    for item in items:
        mod = getattr(item, "module", None)
        name = mod.__name__.rsplit(".", 1)[-1] if mod is not None else ""
        if name in _SLOW_MC_MODULES:
            item.add_marker(slow)
