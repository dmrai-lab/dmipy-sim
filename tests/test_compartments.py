"""`Compartments` is the one spelling of per-compartment properties: pool names resolve to the
compartment ids of every channel, a property is read in id order, and half-specified properties
are refused rather than defaulted silently."""
import pytest

from dmipy_sim.compartments import Compartments, Pool, pool_id


def test_pool_ids_are_the_channel_convention():
    assert (pool_id("extra"), pool_id("intra"), pool_id("myelin")) == (0, 1, 2)
    assert pool_id(1) == 1
    with pytest.raises(KeyError, match="unknown pool"):
        pool_id("csf")
    with pytest.raises(KeyError):
        pool_id(3)


def test_compartments_order_by_id_and_coerce_mappings():
    c = Compartments(intra={"T2": 0.05, "D": 1.7e-9}, extra=Pool(T2=0.08, D=1.7e-9))
    assert list(c) == ["extra", "intra"] and c.ids == (0, 1)
    assert c["intra"].T2 == 0.05 and c[1].T2 == 0.05 and c[0] is c["extra"]
    assert c.by_id("T2") == (0.08, 0.05) and c.by_id("D") == (1.7e-9, 1.7e-9)
    assert c.by_id("T1") is None
    assert Compartments.coerce({"extra": {"T2": 0.1}}) == Compartments(extra=Pool(T2=0.1))
    assert Compartments.coerce(None) == Compartments() and len(Compartments()) == 0
    assert c.replace(myelin=Pool(T2=0.01)).ids == (0, 1, 2)


def test_half_specified_properties_are_refused_unless_a_default_is_given():
    c = Compartments(extra=Pool(T2=0.08), intra=Pool())
    with pytest.raises(ValueError, match="not \\['intra'\\]"):
        c.by_id("T2")
    assert c.by_id("T2", default=0.05) == (0.08, 0.05)


def test_invalid_pools_and_names_fail_at_construction():
    with pytest.raises(KeyError, match="unknown pools"):
        Compartments(csf=Pool(T2=2.0))
    with pytest.raises(KeyError, match="unknown Pool properties"):
        Compartments(intra={"t2": 0.05})
    with pytest.raises(ValueError, match="positive"):
        Pool(T2=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        Pool(D=-1.0)
    with pytest.raises(TypeError):
        Compartments.coerce(3.0)


# ── the geometries take the one spelling, and the old spellings are the same walk ─────────────
import numpy as np

import dmipy_sim as d

D = 2e-9


def _wf():
    return d.set_b(d.pgse(delta=3e-3, DELTA=8e-3, G_magnitude=0.1, bvecs=[[1, 0, 0]], n_t=60,
                          slew_rate=np.inf), 1e9)


def test_mesh_takes_compartments_and_the_dict_spelling_is_the_same_walk():
    from dmipy_sim.geometry import mesh_shapes
    V, F = mesh_shapes.icosphere(2e-6, subdivisions=2)
    comps = Compartments(intra=Pool(T2=0.02, D=1e-9, surface_relaxivity_t2=2e-6),
                         extra=Pool(T2=0.2, D=1e-9))
    m = d.Mesh(V, F, feature_radius=1e-6, compartments=comps)
    assert m.compartments == comps and m._T2_comp == (0.2, 0.02) and m._D_comp == (1e-9, 1e-9)
    assert m.surface_relaxivity_t2 == 2e-6
    with pytest.warns(DeprecationWarning, match="compartments="):
        old = d.Mesh(V, F, feature_radius=1e-6, intra={"T2": 0.02, "D": 1e-9, "surface_relaxivity_t2": 2e-6},
                     extra={"T2": 0.2, "D": 1e-9})
    assert old.compartments == comps
    s_new = d.simulate(300, None, _wf(), m, seed=0, require_gpu=False, engine="fused")
    s_old = d.simulate(300, None, _wf(), old, seed=0, require_gpu=False, engine="fused")
    np.testing.assert_array_equal(np.asarray(s_new), np.asarray(s_old))
    with pytest.raises(ValueError, match="no myelin pool"):
        d.Mesh(V, F, compartments={"myelin": {"T2": 0.01}})
    with pytest.raises(ValueError, match="not both"):
        with pytest.warns(DeprecationWarning):
            d.Mesh(V, F, compartments=comps, intra={"T2": 0.02})
    with pytest.raises(NotImplementedError):
        with pytest.warns(DeprecationWarning):
            d.Mesh(V, F, intra={"kurtosis": 1.0})


def test_myelinated_cylinder_takes_compartments():
    kw = dict(orientation=(0, 0, 1), D_intra=D, D_extra=D)
    comps = Compartments(intra=Pool(T2=0.05), myelin=Pool(T2=0.01), extra=Pool(T2=0.08))
    g = d.MyelinatedCylinder(2e-6, 3e-6, compartments=comps, **kw)
    assert (g.T2_extra, g.T2_intra, g.T2_myelin) == (0.08, 0.05, 0.01) and g.compartments == comps
    with pytest.warns(DeprecationWarning, match="compartments="):
        old = d.MyelinatedCylinder(2e-6, 3e-6, T2_intra=0.05, T2_myelin=0.01, T2_extra=0.08, **kw)
    assert (old.T2_extra, old.T2_intra, old.T2_myelin) == (0.08, 0.05, 0.01)
    s_new = d.simulate(300, None, _wf(), g, seed=0, require_gpu=False, engine="fused")
    s_old = d.simulate(300, None, _wf(), old, seed=0, require_gpu=False, engine="fused")
    np.testing.assert_array_equal(np.asarray(s_new), np.asarray(s_old))
    with pytest.raises(ValueError, match="both"):
        with pytest.warns(DeprecationWarning):
            d.MyelinatedCylinder(2e-6, 3e-6, compartments=comps, T2_intra=0.05, **kw)
    with pytest.raises(ValueError, match="given twice"):
        d.MyelinatedCylinder(2e-6, 3e-6, compartments=Compartments(intra=Pool(D=1e-9)), **kw)
    # the Substrate's pools drive a geometry directly
    sub = d.Substrate() if hasattr(d, "Substrate") else __import__("dmipy_sim.substrate.substrate", fromlist=["Substrate"]).Substrate()
    c = sub.compartments
    assert c.ids == (0, 1, 2) and c["myelin"].water_fraction == sub.myelin_water_proton_density
    g2 = d.MyelinatedCylinder(2e-6, 3e-6, orientation=(0, 0, 1), D_intra=sub.D_intra, D_extra=sub.D_extra,
                              D_myelin=sub.D_myelin, compartments=c)
    assert g2.T2_myelin == sub.T2_myelin and g2.water_fractions == (1.0, sub.myelin_water_proton_density, 1.0)


def test_packed_myelinated_cylinders_take_compartments_and_keep_per_axon_arrays():
    L = float(np.sqrt(np.pi * 3 * (1e-6 / 0.7) ** 2 / 0.5))
    _, _, c = d.pack_myelinated_cylinders([1e-6] * 3, 0.7, None, cell_size=L, seed=0)
    comps = Compartments(intra=Pool(T2=0.05), myelin=Pool(T2=0.01), extra=Pool(T2=0.08))
    pm = d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4, compartments=comps)
    with pytest.warns(DeprecationWarning, match="compartments="):
        old = d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4, T2_intra=0.05, T2_myelin=0.01,
                                          T2_extra=0.08)
    for a in ("_T2_intra_jax", "_T2_myelin_jax", "_T2_extra_jax"):
        np.testing.assert_array_equal(np.asarray(getattr(pm, a)), np.asarray(getattr(old, a)))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                          # per-axon arrays are not the old spelling
        arr = d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4, T2_intra=[0.05, 0.06, 0.07],
                                          compartments=Compartments(myelin=Pool(T2=0.01)))
    assert np.asarray(arr._T2_intra_jax)[:3].tolist() == pytest.approx([0.05, 0.06, 0.07])
    with pytest.raises(ValueError, match="both"):
        d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4, T2_intra=[0.05] * 3, compartments=comps)
