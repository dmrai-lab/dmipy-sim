"""The field a spec walk reads for a myelinated cylinder and a packed myelinated cell is the closed form of their
sheaths (:meth:`HollowCylinderField.of`), and every other gridded field source keeps the spec's raster.

1. the field basis the spec route hands the walk of a packed cell (and of an isolated myelinated cylinder) IS
   ``HollowCylinderField.of`` of the geometry the spec builds: its type, sources and cell;
2. a field-on spec walk of the cell equals the direct walk with ``HollowCylinderField.of(g)`` to the bit (field modes,
   field quadratic variation, positions);
3. a spec whose magnetic pool is bounded by sphere unions keeps its raster (a :class:`FieldGrid`) in the walk;
4. ``field_res=`` is refused for a geometry read in closed form that has a field, and not for one without;
5. ``continue_walk`` extends a closed-form pack with the closed form, and refuses a pack whose recorded field is a
   raster or a closed form that differs from the geometry's."""
import numpy as np
import pytest

from dmipy_sim import Compartments, Pool
from dmipy_sim.fields.hollow_cylinder import HollowCylinderField
from dmipy_sim.fields.susceptibility_field import FieldGrid
from dmipy_sim.geometry import MyelinatedCylinder, PackedMyelinatedCylinders
from dmipy_sim.spec.build import geometry_from_spec
from dmipy_sim.spec.walk import SpecError, _single_geometry_field
from tests._envelope import walk_envelope

D = 2e-9
R = 2e-6
L = 4.4 * R
OUTER = np.array([R, 0.8 * R, R, 0.9 * R])
CENTRES = np.array([[0.0, 0.0], [2.2, 0.0], [0.0, 2.2], [2.2, 2.2]]) * R
G_RATIOS = np.array([0.7, 1.0, 0.7, 1.0])                  # cylinders 0 and 2 sheathed, 1 and 3 plain


def _cell():
    return PackedMyelinatedCylinders(OUTER * G_RATIOS, G_RATIOS, CENTRES, L, N_max=4,
                                     compartments=Compartments(intra=Pool(D=D), myelin=Pool(D=0.0), extra=Pool(D=D)))


def test_the_spec_route_reads_a_packed_cell_in_closed_form():
    """The basis of the cell's spec is the closed form of the sheathed cylinders in the cell's lattice: the same
    sources, radii and side as ``HollowCylinderField.of`` of the geometry the spec builds, and the same record."""
    spec = _cell().spec
    g = geometry_from_spec(spec)
    fb = _single_geometry_field(spec, g, None, None)
    assert type(fb) is HollowCylinderField
    want = HollowCylinderField.of(g)
    np.testing.assert_array_equal(fb.centers, want.centers)
    np.testing.assert_array_equal(fb.inner_radii, want.inner_radii)
    np.testing.assert_array_equal(fb.outer_radii, want.outer_radii)
    assert fb.L == want.L == pytest.approx(L) and fb.summation == "lattice" and fb.meta == want.meta
    np.testing.assert_allclose(fb.centers, CENTRES[[0, 2]])
    np.testing.assert_allclose(fb.outer_radii, OUTER[[0, 2]])
    np.testing.assert_allclose(fb.inner_radii, 0.7 * OUTER[[0, 2]])


def test_the_spec_route_reads_an_isolated_myelinated_cylinder_in_closed_form():
    g0 = MyelinatedCylinder(0.7e-6, 1e-6, (0, 0, 1), compartments=Compartments(intra=Pool(D=D), extra=Pool(D=D),
                                                                                 myelin=Pool(D=0.0)))
    spec = g0.spec
    g = geometry_from_spec(spec)
    fb = _single_geometry_field(spec, g, None, None)
    assert type(fb) is HollowCylinderField and fb.L is None and fb.summation == "none"
    assert fb.meta == HollowCylinderField.of(g).meta


def test_a_field_on_spec_walk_of_a_cell_is_the_direct_walk_with_the_closed_form_to_the_bit():
    """``walk(spec)`` of the cell with a stated band and read equals ``_walk_geometry`` of the geometry the spec builds
    with ``HollowCylinderField.of`` and the same arguments: field modes, field quadratic variation and positions."""
    from dmipy_sim import walk
    from dmipy_sim.engine.core import _walk_geometry
    from dmipy_sim.replay.compression import certificate_settings
    cell = _cell()
    spec = cell.spec
    env = walk_envelope(of=spec)                              # as test_grid_field_in_walk's myelinated-walk test builds it
    n, T_max, h, K, K_c3, dt_f = 48, 5e-4, 1.25e-4, 4, 4, 1.25e-4
    w = walk(spec, n, T_max, h, envelope=env, seed=3, require_gpu=False, K=K, K_c3=K_c3, field_read_interval=dt_f)
    g = geometry_from_spec(spec)
    direct = _walk_geometry(n, D, g, T_max=T_max, block_h=h, seed=3, tiers="all", first_walker=0,
                            field_basis=HollowCylinderField.of(g), K_c3=K_c3, field_read_interval=dt_f,
                            field_target_floor=None, K=K, block_order=None, window_T=None, raw_subset=0,
                            envelope=certificate_settings(env), bounce_budget=None, backend="jax", require_gpu=False,
                            walker_batch_size=50_000, f_band=float(env.band_hz), accumulation_dtype="float32")
    assert np.any(np.asarray(w.field_coeffs) != 0.0)
    np.testing.assert_array_equal(np.asarray(w.field_coeffs), np.asarray(direct.field_coeffs))
    np.testing.assert_array_equal(np.asarray(w.qv_field), np.asarray(direct.qv_field))
    np.testing.assert_array_equal(np.asarray(w.position_coeffs), np.asarray(direct.position_coeffs))
    assert type(w.field_basis) is HollowCylinderField and w.field_basis.meta == direct.field_basis.meta


def test_a_source_without_a_closed_form_keeps_the_spec_raster():
    """A magnetic shell between two sphere unions has no closed form: its spec walk reads the spec's raster."""
    from dmipy_sim import walk
    from tests.test_field_declaration import _shell_union_spec
    spec = _shell_union_spec(True)
    w = walk(spec, 24, 4e-4, 2e-4, seed=0, n_probe=5_000, require_gpu=False, K=3, K_c3=2, field_res=0.2e-6,
             envelope=walk_envelope(of=spec))
    assert type(w.field_basis) is FieldGrid and w.field_basis.certificate is not None and w.field_coeffs is not None


def _field_off(spec):
    """``spec`` with no magnetic pool and no field tier."""
    from dmipy_replay.spec.substrate import SubstrateSpec
    v = spec.to_dict()
    for p in v["pools"]:
        p["susceptibility"] = None
    v["validity"]["tiers"] = [t for t in v["validity"]["tiers"] if t != "field"]
    return SubstrateSpec.from_dict(v).validate()


def test_a_node_spacing_is_refused_by_a_field_on_cell_walk_and_not_by_a_field_off_one():
    """``field_res=`` is refused where the route builds a field; a spec without a magnetic pool builds none."""
    from dmipy_sim import walk
    spec = _cell().spec
    kw = dict(seed=0, require_gpu=False, field_res=0.25e-6)
    with pytest.raises(SpecError, match="field_res=.*closed form"):
        _single_geometry_field(spec, geometry_from_spec(spec), 0.25e-6, None)
    with pytest.raises(SpecError, match="field_res=.*closed form"):
        walk(spec, 8, 2.5e-4, 1.25e-4, envelope=walk_envelope(of=spec), **kw)
    off = _field_off(spec)
    w = walk(off, 8, 2.5e-4, 1.25e-4, envelope=walk_envelope(of=off), **kw)
    assert w.field_basis is None and w.field_coeffs is None


# -- continuation --------------------------------------------------------------------------------------------------

def _closed_form_pack():
    """A pack of the cell's field-on walk in windows of 2.5e-4 s: its field record states the closed form."""
    from dmipy_sim import build_replay_pack, walk
    from tests._envelope import stated
    from tests.test_grid_field_in_walk import ENV
    spec = _cell().spec
    w = walk(spec, 40, 5e-4, 1.25e-4, envelope=walk_envelope(of=spec), seed=1, require_gpu=False, K=2, K_c3=2,
             window_T=2.5e-4)
    return build_replay_pack(w, id="test/cell-closed-form", license="x", citation="x", K=2, window_T=2.5e-4,
                             **stated(ENV, of=w, G_max=0.5))


@pytest.fixture(scope="module")
def closed_form_pack():
    return _closed_form_pack()


def _continue_spying(pack, monkeypatch):
    """``continue_walk(pack)`` with the field basis handed to the continued walk recorded."""
    import dmipy_sim.engine.core as core
    from dmipy_sim.replay.continuation import continue_walk
    seen = []
    real = core._walk_geometry
    def spy(*a, **k):
        seen.append(k.get("field_basis"))
        return real(*a, **k)
    monkeypatch.setattr(core, "_walk_geometry", spy)
    return continue_walk(pack, 2.5e-4, seed=5, require_gpu=False), seen


def test_a_closed_form_pack_is_continued_with_the_closed_form_it_records(closed_form_pack, monkeypatch):
    rec = closed_form_pack.meta["compression"]["channels"]["susceptibility_grid"]
    assert "source" in rec and "shape" not in rec
    cont, seen = _continue_spying(closed_form_pack, monkeypatch)
    assert len(seen) == 1 and type(seen[0]) is HollowCylinderField
    import json
    assert json.loads(json.dumps(seen[0].meta)) == json.loads(json.dumps(rec["source"]))
    assert cont.n_walkers == closed_form_pack.n_walkers and cont.field_coeffs is not None


def test_a_pack_whose_recorded_field_is_a_raster_is_not_continued(closed_form_pack, monkeypatch):
    raster = dict(origin=[0.0, 0.0, 0.0], voxel_size=[1e-7] * 3, shape=[29, 29, 4], has_aniso=True, periodic=[True, True, False],
                  arrays_in_pack=False, replay_route="path")
    monkeypatch.setitem(closed_form_pack.meta["compression"]["channels"], "susceptibility_grid", raster)
    with pytest.raises(ValueError, match="raster of shape.*closed form.*Walk the pack again"):
        _continue_spying(closed_form_pack, monkeypatch)


def test_a_pack_whose_recorded_closed_form_differs_from_the_geometrys_is_not_continued(closed_form_pack, monkeypatch):
    import copy
    rec = copy.deepcopy(closed_form_pack.meta["compression"]["channels"]["susceptibility_grid"])
    rec["source"]["outer_radii"][0] *= 1.01                   # one radius other than the geometry's
    monkeypatch.setitem(closed_form_pack.meta["compression"]["channels"], "susceptibility_grid", rec)
    with pytest.raises(ValueError, match="field source.*closed form.*Walk the pack again"):
        _continue_spying(closed_form_pack, monkeypatch)
