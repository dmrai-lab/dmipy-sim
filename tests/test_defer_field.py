"""The field tier's producer-level contract (dmipy-sim#539): the SPEC decides whether a susceptibility source
exists (``spec.field_source_pools``); every walk of such a spec carries its basis, and the default -- the only
producer that reads a field in its own stepping -- samples it in the walk at the sub-step
(``adaptive_steps=True``). ``walk_spec``'s old ``field=`` boolean is gone: ``defer_field=True`` records the
obligation on the walk instead of building the basis now (refused without a ``run_dir=``, since the raw walk
must be kept for the step that fills it later), and ``fill_field(walk, basis)`` reads a basis -- given now or
after the walk was kept -- along the walk's own saved positions, a quadrature at save resolution rather than the
sub-step interval mean an in-walk sampling takes. ``build_replay_pack`` carries no ``field=`` at all: it encodes
whatever the walk carries and refuses by name a walk whose spec declares a source but has neither samples nor a
recorded deferral."""
import dataclasses

import numpy as np
import pytest

from dmipy_sim.io.strands import write_tck, concat_centerlines
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec import SpecError, disco_spec, fill_field, walk_spec
from dmipy_sim.spec.walk import field_grid_of_spec
from tests.conftest import spec_without_source


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    """Three strands with a sheath (a susceptibility source by default): disco_spec's own fixture shape."""
    tmp = tmp_path_factory.mktemp("defer")
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp / "t.tck"), str(tmp / "d.txt")
    write_tck(tck, *concat_centerlines(cls_), coordinate_unit_m=25e-6)
    np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    return disco_spec(tck, dia, side_m=20e-6)


def _walk(spec, **kw):
    return walk_spec(spec, 60, 8e-4, 5e-5, seed=3, n_probe=20_000, require_gpu=False, **kw)


def test_a_source_walks_with_field_samples_by_default_and_the_pack_carries_the_tier(spec):
    """No ``field=`` toggle is needed, or exists: the basis is built because the spec declares a source, and
    ``adaptive_steps=True`` samples it in the walk -- the pack's path channel is read from those samples."""
    w = _walk(spec, adaptive_steps=True)
    assert w.field_basis is not None and w.field_samples is not None and not w.field_deferred
    pk = build_replay_pack(w, id="t/default-field", license="x", citation="x", K=8, susc_path_K=4)
    assert pk.has_field
    pm = pk.meta["compression"]["channels"]["susceptibility_path"]
    assert pm["sampling"] == "interval_mean_in_walk"


def test_a_spec_without_a_source_neither_samples_nor_accepts_a_deferral(spec, tmp_path):
    nf = spec_without_source(spec)
    w = _walk(nf, adaptive_steps=True)
    assert w.field_basis is None and w.field_samples is None
    pk = build_replay_pack(w, id="t/no-source", license="x", citation="x", K=8)
    assert not pk.has_field
    with pytest.raises(SpecError, match="declares no susceptibility"):
        _walk(nf, defer_field=True, run_dir=str(tmp_path / "nf-run"))


def test_defer_field_needs_a_run_dir_and_build_replay_pack_refuses_the_unfilled_walk_until_fill_field(spec, tmp_path):
    """``defer_field=True`` is refused without ``run_dir=`` (the quadrature ``fill_field`` needs the raw walk
    kept); given one, the walk records the deferral and carries neither a basis nor samples; a pack of it is
    refused by name; ``fill_field`` fills it and the pack then carries the tier, reading at save resolution."""
    with pytest.raises(TypeError, match="run_dir"):
        _walk(spec, defer_field=True)
    w = _walk(spec, defer_field=True, run_dir=str(tmp_path / "run"))
    assert w.field_deferred and w.field_basis is None and w.field_samples is None
    with pytest.raises(ValueError, match="deferred field tier"):
        build_replay_pack(w, id="t/deferred", license="x", citation="x", K=8, susc_path_K=4)
    basis = field_grid_of_spec(spec, field_res=0.5e-6)        # given after the walk was kept: the CACTUS route
    filled = fill_field(w, basis)
    assert filled.field_samples is not None and filled.field_fill is not None
    pk = build_replay_pack(filled, id="t/filled", license="x", citation="x", K=8, susc_path_K=4)
    assert pk.has_field
    pm = pk.meta["compression"]["channels"]["susceptibility_path"]
    assert pm["sampling"] == "read_at_save_resolution"
    assert pm["field_fill"]["read"] == "save_resolution" and "basis_certificate" in pm["field_fill"]


def test_build_replay_pack_refuses_a_field_argument(spec):
    """``build_replay_pack`` carries no ``field=`` of any kind -- the TypeError is Python's own, for a keyword
    the function no longer has."""
    w = _walk(spec, adaptive_steps=True)
    with pytest.raises(TypeError, match="field"):
        build_replay_pack(w, id="x", field=None, K=8, license="x", citation="x")


def test_the_quadrature_at_save_resolution_agrees_with_the_sub_step_sample_within_the_save_grid_bound(spec):
    """``fill_field`` reads the SAME basis at the SAME saved positions the in-walk sampling used, but at a
    single point per save rather than the sub-step interval mean: the two must agree to the quadrature bound
    the save grid implies, not merely be of the same order."""
    w = _walk(spec, adaptive_steps=True)
    bare = dataclasses.replace(w, field_samples=None)
    filled = fill_field(bare, w.field_basis)
    sub_step = np.asarray(w.field_samples, np.float64)
    save_grid = np.asarray(filled.field_samples, np.float64)
    scale = np.abs(sub_step).max()
    rel = float(np.abs(save_grid - sub_step).max() / scale)
    # measured on this fixture (60 walkers, dt_save 50 us over 0.8 ms): 3.5e-7 relative to the channel's own
    # scale -- at this save grid the field barely moves within one interval, so the sub-step mean and the
    # save-point read agree to near machine precision; a coarser save grid (a longer walk, fewer saves) is
    # where fill_field's quadrature bias actually shows up (dmipy-sim#539's own PackedMyelinatedCylinders check)
    assert rel < 1e-4, rel
