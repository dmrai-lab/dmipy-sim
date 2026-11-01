"""The band of a pack is the scanner class's, never a search on the walk (#643): ``K = ceil(2 f T)`` with ``f`` the
class's certified band times the storage margin; a walk from ``walk`` carries its class; a walk with no class
and no stated band is refused by name; a class with no certified band is refused by name; the search is a
diagnostic no producer calls. And the codec's numpy pieces (#642): the phase GEMM equals the einsum it replaced."""
import pathlib
import re

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.core import _walk_geometry
from dmipy_replay.acquisition.scanners import ScannerLimits
from dmipy_sim.replay import bank
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_replay.replay.storage import mode_count, storage_band_hz, STORAGE_BAND_MARGIN
from dmipy_sim.replay import compression as cx
from dmipy_sim.spec.walk import walk

PAPER = {"prisma": 522.0, "connectom": 310.0, "magnus": 877.0, "connectome_2": 960.0, "bruker_bga_s": 3012.0, "micro_insert": 4095.0}


@pytest.mark.parametrize("scanner,hz", sorted(PAPER.items()))
def test_the_certified_band_is_the_papers_and_the_stored_band_is_it_times_the_margin(scanner, hz):
    assert ScannerLimits.of(scanner).certified_band_hz == hz
    assert storage_band_hz(scanner) == pytest.approx(STORAGE_BAND_MARGIN * hz)


@pytest.mark.parametrize("T", [0.02, 0.1, 0.3])
@pytest.mark.parametrize("scanner", ["prisma", "connectom", "connectome_2"])
def test_the_mode_count_is_two_f_t_rounded_up(scanner, T):
    assert mode_count(T, scanner) == max(2, int(np.ceil(2.0 * STORAGE_BAND_MARGIN * PAPER[scanner] * T)))


def test_a_stored_k_of_256_is_above_the_certified_band_of_every_human_class_and_below_the_rule_on_connectome_2():
    """The rule is a design rule (certified band x margin); a pack stored at K = 256 per 100 ms resolves 1280 Hz, above every
    human class's certified band (192 bands on Connectome 2.0) and 13 bands short of the rule's 269 there: whether it serves
    a waveform is the band check's, per waveform."""
    assert mode_count(0.1, "connectome_2") == 269
    for scanner in ("prisma", "connectom", "magnus", "connectome_2"):
        assert 2.0 * 0.1 * ScannerLimits.of(scanner).certified_band_hz <= 256
    assert 256 < mode_count(0.1, "connectome_2")


def test_a_class_without_a_certified_band_is_refused_by_name():
    with pytest.raises(ValueError, match="no certified band.*hyperfine_swoop_64mT"):
        storage_band_hz("low_field")


def _walk(**kw):
    g = d.Cylinder(2e-6, (0, 0, 1))
    return _walk_geometry(300, 2e-9, g, 4e-3, 5e-4, seed=0, require_gpu=False, **kw)


def test_a_walk_states_its_own_band():
    """A walk that accumulated its bands states its own K and packs at it, with no class and no band stated."""
    w = _walk()
    assert build_replay_pack(w, id="t", license="x", citation="x").K == w.K


def test_scanner_sets_the_band_and_the_pack_records_the_class():
    w = _walk()
    pk = build_replay_pack(w, id="t", license="x", citation="x", scanner="prisma")
    assert pk.K == mode_count(4e-3, "prisma") and pk.meta["compression"]["scanner"] == "prisma"
    assert pk.temporal_bandwidth_hz == pytest.approx(pk.K / (2 * 4e-3))
    k2 = build_replay_pack(w, id="t", license="x", citation="x", K=5)
    assert k2.K == 5 and "scanner" not in k2.meta["compression"]            # a stated K is the band, no class derived


def test_a_walk_from_walk_spec_carries_its_class_and_packs_without_an_argument():
    spec = d.Cylinder(2e-6, (0, 0, 1)).spec
    w = walk(spec, 300, 4e-3, 5e-4, scanner="prisma", seed=0, require_gpu=False, diffusivity=2e-9)
    assert w.scanner == "prisma"
    pk = build_replay_pack(w, id="t", license="x", citation="x")
    assert pk.K == mode_count(4e-3, "prisma") and pk.meta["compression"]["scanner"] == "prisma"


def test_no_producer_calls_the_mode_search():
    """``auto_select_modes`` is a diagnostic: the bank, the replay and the spec walker never call it."""
    root = pathlib.Path(bank.__file__).parent.parent
    for rel in ("replay/bank.py", "replay/replay.py", "replay/continuation.py", "spec/walk.py"):
        src = (root / rel).read_text()
        assert not re.search(r"auto_select_modes\(", src), f"{rel} calls auto_select_modes"


def test_the_phase_gemm_equals_the_einsum():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(50, 41, 3)) * 1e-6
    G = rng.normal(size=(7, 41, 3)) * 0.1
    from dmipy_replay.replay._replay_kernel import effective_gradient
    from dmipy_replay.constants import GAMMA
    dt = 5e-4
    ref = (GAMMA * dt) * np.einsum("mtd,ntd->nm", effective_gradient(G, dt, 41, dt), pos)
    np.testing.assert_allclose(cx._walker_phases(pos, dt, G), ref, rtol=1e-12, atol=1e-9)


def test_the_threaded_transforms_change_no_bit():
    from scipy.fft import dst
    x = np.random.default_rng(1).normal(size=(64, 99))
    np.testing.assert_array_equal(cx._dst(x, axis=1, type=1, norm="ortho"), dst(x, axis=1, type=1, norm="ortho"))
