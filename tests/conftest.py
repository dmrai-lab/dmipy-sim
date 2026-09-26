"""Shared test constants and fixture loader."""

import os
from pathlib import Path

import numpy as np
import pytest

# ── one dmipy_sim, this one ─────────────────────────────────────────────────────────────────
# A `pip install -e` of ANOTHER checkout leaves a setuptools editable finder on sys.meta_path that
# maps the name `dmipy_sim` to that checkout. It answers for any `dmipy_sim.<name>` the package
# imported here lacks, so a module deleted in this tree quietly imports from the other one (and its
# stale code with it). Drop every such finder that does not point at this tree; the API-surface
# test checks afterwards that every loaded dmipy_sim module lives under this package.
import pathlib as _pathlib
import sys as _sys
import dmipy_sim as _dmipy_sim
_ROOT = _pathlib.Path(_dmipy_sim.__file__).parent.resolve()
for _f in list(_sys.meta_path):
    _mapping = getattr(_sys.modules.get(getattr(_f, "__module__", ""), None), "MAPPING", None)
    if isinstance(_mapping, dict) and "dmipy_sim" in _mapping \
            and _pathlib.Path(_mapping["dmipy_sim"]).resolve() != _ROOT:
        _sys.meta_path.remove(_f)

# ── Persistent XLA compilation cache ────────────────────────────────────────────────────
# MUST be configured before anything triggers a JAX computation, hence the position here.
#
# Both test tiers are compile-bound, not compute-bound (#91, #93). Measured on one
# `simulate_bloch` call, the cost is independent of problem size -- 400 walkers x 201 steps
# takes 7.98 s while 25 walkers x 21 steps takes 5.09 s -- because every call re-traces and
# recompiles. The cause is that every `jax.jit` in the package is built INSIDE a function
# body over a fresh closure, so jit's in-memory cache keys on a new object each time and can
# never hit.
#
# The persistent cache keys on the HLO instead of the Python object, so it hits anyway:
#
#     repeat call, same process     7.9 s -> 1.6 s   (5x)
#     first call, NEW process      21.1 s -> 7.8 s   (2.7x, cache warm on disk)
#
# Set DMIPY_JAX_CACHE=0 to disable, or DMIPY_JAX_CACHE=<dir> to relocate it. CI can persist
# the directory between runs to get the cross-process win on the first test too.
_cache = os.environ.get("DMIPY_JAX_CACHE", "")
if _cache != "0":
    import jax
    _dir = _cache or str(Path(__file__).parent.parent / ".jax_cache")
    jax.config.update("jax_compilation_cache_dir", _dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Standard simulation parameters matching disimpy validation suite
D = 2e-9          # m²/s
N_WALKERS = 100_000   # overridden to 1_000_000 when --heavy is passed
SEED = 123

# For assertions that are EXACT equality (npt.assert_array_equal), not a statistical comparison.
# Those tests check that two code paths which must be identical -- `permeability=None` vs the
# default, `T2=None` vs omitting the kwarg -- produce bit-identical output. Given a fixed seed the
# walk is deterministic, so if the paths agree, every N passes; if they diverge, the divergence is
# deterministic too and shows in the walkers that reach the branch. There is no sampling error to
# average down, so a statistical N buys nothing and costs two full-size walks per test.
#
# 5,000 is not a new judgement: tests/test_permeability_crossing.py already runs exactly this
# assertion at N_ORDINAL = 5_000. This makes the other copies consistent with it. Deliberately NOT
# scaled by --heavy -- a bigger N cannot make an exact equality any more exact.
N_EXACT = 5_000

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
    "test_compartment_tagging", "test_t2",
    "test_karger_mc_parity", "test_t2_walker_parity",
    "test_mesh_mc",
    # Six mesh MC walks (three grid resolutions x two assertions). Collision sub-stepping makes each
    # one several times what it would otherwise cost, which is the point of it -- but it belongs in the
    # nightly job, not the ~1 min lane.
    "test_mesh_acceleration_invariance",
    "test_replay_parity",
    "test_replay_fields_mt",
    "test_engine_dispatch",
    # Crossing-RATE validation against the closed two-compartment exchange law: six walks of
    # 6000 walkers x 10000 sub-steps. The cheap sealed-wall detector that guards the same
    # property lives in the fast lane, in test_boundary_compartment_integrity.
    "test_permeable_crossings",
    # One 200,000-walker, 6000-step walk per rock on a 300^3 micro-CT image, against Talabi 2008
    # (needs DMIPY_SIM_IMPERIAL2007_DIR; skipped without it).
    "test_talabi_rocks",
}


def pytest_runtest_logreport(report):
    """Drop JAX's in-memory compile caches at every module boundary. Every jit in the package is built inside a
    function body over a fresh closure, so the caches only grow: measured over the fast tier on the CPU, the
    process went 3.0 GB after the first file to 6.2 GB by the p-files and the 7 GB hosted runner shut the
    3.11 job down around the r-files six times in a day; with the caches cleared per module the same files
    grew by a third of that."""
    if report.when != "teardown":
        return
    mod = report.nodeid.split("::")[0]
    if mod != _last_module[0]:
        if _last_module[0] is not None:
            import gc
            import jax
            jax.clear_caches()
            gc.collect()
        _last_module[0] = mod


_last_module = [None]


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


# ------------------------------------------------------------------ the packs and specs several test modules share
def pgse_wf(TE_s, n_t=500, slew_rate=None):
    """The permeability tests' PGSE along x at b = 0, 0.5, 1 and 2 ms/um^2, its lobes 5 % of ``TE_s`` (5 us at the
    least); ``slew_rate=np.inf`` for the square lobes the restricted-diffusion checks were validated against."""
    from dmipy_sim import set_b
    from dmipy_sim.sequences import pgse
    delta = max(TE_s * 0.05, 5e-6)
    kw = {} if slew_rate is None else {"slew_rate": slew_rate}
    return set_b(pgse(np.tile([1.0, 0.0, 0.0], (4, 1)), delta, TE_s - delta, gradient_strengths=1.0, n_t=n_t, **kw),
                 np.array([0.0, 500e6, 1000e6, 2000e6]))


@pytest.fixture(scope="module")
def pack():
    """A 300-walker cylinder walk packed at K = 8: the small pack of the replay tests."""
    import dmipy_sim as d
    from dmipy_sim.replay.bank import build_replay_pack
    walk = d.simulate_trajectories(300, 2e-9, d.Cylinder(2e-6, (0, 0, 1)), 0.01, 5e-4, seed=0, require_gpu=False)
    return build_replay_pack(walk, id="test/full", K=8, license="x", citation="x")


def _three_strands(tmp):
    """The three-strand DiSCo-format fixture (a sheath on each) written under ``tmp``: its spec."""
    from dmipy_sim.io.strands import write_tck
    from dmipy_sim.spec import disco_spec
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp / "t.tck"), str(tmp / "d.txt")
    write_tck(tck, cls_, coordinate_unit_m=25e-6)
    np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    return disco_spec(tck, dia, side_m=20e-6)


@pytest.fixture(scope="module")
def field_pack(tmp_path_factory):
    """A strand pack with the path channel (C3), from the three-strand fixture with a sheath."""
    from dmipy_sim.spec import walk_spec
    from dmipy_sim.replay.bank import build_replay_pack
    spec = _three_strands(tmp_path_factory.mktemp("field"))
    w = walk_spec(spec, 90, 8e-4, 2e-4, seed=0, n_probe=20_000, field_res=0.5e-6, require_gpu=False)
    return build_replay_pack(w, id="test/field", license="x", citation="x", K=4, susc_path_K=4)


@pytest.fixture(scope="module")
def spec_grid(tmp_path_factory):
    """The three-strand spec on a 2 x 2 x 2 grid of 10 um voxels: ``(spec, grid, tmp)``."""
    from dmipy_sim.phantom.grid import Grid
    tmp = tmp_path_factory.mktemp("strands")
    return _three_strands(tmp), Grid(shape=(2, 2, 2), voxel_size_m=(10e-6,) * 3, origin_m=(5e-6,) * 3), tmp
