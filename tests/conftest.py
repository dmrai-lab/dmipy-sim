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
