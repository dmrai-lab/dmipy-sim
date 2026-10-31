"""The field tier's producer-level contract (dmipy-sim#539): the SPEC decides whether a susceptibility source
exists (``spec.field_source_pools``); every walk of such a spec carries its basis, and the only producer that reads a
strand field in its own stepping -- ``adaptive_steps=True`` -- reads it in the walk (the walk's modes are the pack's
path channel). ``build_replay_pack`` carries no ``field=`` at all: it encodes whatever the walk carries."""

import numpy as np
import pytest

from dmipy_sim.io.strands import write_tck, concat_centerlines
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec import disco_spec
from dmipy_sim import walk
from tests.conftest import spec_without_source


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    """Three strands with a sheath (a susceptibility source by default): disco_spec's own fixture shape."""
    tmp = tmp_path_factory.mktemp("source")
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp / "t.tck"), str(tmp / "d.txt")
    write_tck(tck, *concat_centerlines(cls_), coordinate_unit_m=25e-6)
    np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    return disco_spec(tck, dia, side_m=20e-6)


def _walk(spec, **kw):
    return walk(spec, 60, 8e-4, 5e-5, seed=3, n_probe=20_000, require_gpu=False, **kw)


def test_a_source_walks_with_its_field_by_default_and_the_pack_carries_the_tier(spec):
    """No ``field=`` toggle is needed, or exists: the basis is built because the spec declares a source, and
    ``adaptive_steps=True`` reads it in the walk -- the pack's path channel is the modes the walk accumulated."""
    w = _walk(spec, adaptive_steps=True, K=8, K_c3=4)
    assert w.field_basis is not None and w.field_coeffs is not None and w.has_field
    pk = build_replay_pack(w, id="t/default-field", license="x", citation="x", K=8, susc_path_K=4)
    assert pk.has_field
    pm = pk.meta["compression"]["channels"]["susc_field_basis"]
    assert pm["sampling"] == "accumulated_in_walk"


def test_a_spec_without_a_source_reads_no_field(spec):
    nf = spec_without_source(spec)
    w = _walk(nf, adaptive_steps=True, K=8)
    assert w.field_basis is None and w.field_coeffs is None and not w.has_field
    pk = build_replay_pack(w, id="t/no-source", license="x", citation="x", K=8)
    assert not pk.has_field


def test_build_replay_pack_refuses_a_field_argument(spec):
    """``build_replay_pack`` carries no ``field=`` of any kind -- the TypeError is Python's own, for a keyword
    the function no longer has."""
    w = _walk(spec, adaptive_steps=True, K=8)
    with pytest.raises(TypeError, match="field"):
        build_replay_pack(w, id="x", field=None, K=8, license="x", citation="x")
