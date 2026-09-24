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


def test_compartments_order_by_id():
    c = Compartments(intra=Pool(T2=0.05, D=1.7e-9), extra=Pool(T2=0.08, D=1.7e-9))
    assert list(c) == ["extra", "intra"] and c.ids == (0, 1)
    assert c["intra"].T2 == 0.05 and c[1].T2 == 0.05 and c[0] is c["extra"]
    assert c.by_id("T2") == (0.08, 0.05) and c.by_id("D") == (1.7e-9, 1.7e-9)
    assert c.by_id("T1") is None
    assert Compartments.coerce(None) == Compartments() and len(Compartments()) == 0
    assert Compartments.coerce(c) is c
    assert c.replace(myelin=Pool(T2=0.01)).ids == (0, 1, 2)


def test_half_specified_properties_are_refused_unless_a_default_is_given():
    c = Compartments(extra=Pool(T2=0.08), intra=Pool())
    with pytest.raises(ValueError, match="not \\['intra'\\]"):
        c.by_id("T2")
    assert c.by_id("T2", default=0.05) == (0.08, 0.05)


def test_invalid_pools_and_names_fail_at_construction():
    with pytest.raises(KeyError, match="unknown pools"):
        Compartments(csf=Pool(T2=2.0))
    with pytest.raises(TypeError, match="t2"):
        Pool(t2=0.05)
    with pytest.raises(ValueError, match="positive"):
        Pool(T2=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        Pool(D=-1.0)
    with pytest.raises(TypeError, match="is a Compartments"):
        Compartments.coerce(3.0)


def test_a_pool_and_the_pools_have_one_spelling():
    """A pool is a `Pool` and the pools are a `Compartments`: a mapping of properties is refused, naming the
    spelling."""
    with pytest.raises(TypeError, match=r"intra=Pool\("):
        Compartments(intra={"T2": 0.05})
    with pytest.raises(TypeError, match="is a Compartments"):
        Compartments.coerce({"extra": Pool(T2=0.1)})
    with pytest.raises(TypeError):
        Compartments({"extra": Pool(T2=0.1)})                          # keyword only
    with pytest.raises(TypeError, match="permeability"):
        Pool(permeability=1e-5)                                        # a membrane's, not a pool's


# ── the geometries take the one spelling ─────────────────────────────────────────────────────
import numpy as np

import dmipy_sim as d

D = 2e-9


def test_mesh_takes_compartments():
    from dmipy_sim.geometry import mesh_shapes
    V, F = mesh_shapes.icosphere(2e-6, subdivisions=2)
    comps = Compartments(intra=Pool(T2=0.02, D=1e-9, surface_relaxivity_t2=2e-6),
                         extra=Pool(T2=0.2, D=1e-9))
    m = d.Mesh(V, F, feature_radius=1e-6, compartments=comps)
    assert m.compartments == comps and m._T2_comp == (0.2, 0.02) and m._D_comp == (1e-9, 1e-9)
    assert m.surface_relaxivity_t2 == 2e-6
    with pytest.raises(ValueError, match="no myelin pool"):
        d.Mesh(V, F, compartments=Compartments(myelin=Pool(T2=0.01)))
    with pytest.raises(TypeError):                                     # the pools have one spelling
        d.Mesh(V, F, intra=Pool(T2=0.02))
    with pytest.raises(TypeError, match="is a Compartments"):
        d.Mesh(V, F, compartments={"intra": {"T2": 0.02}})


def test_myelinated_cylinder_takes_compartments():
    kw = dict(orientation=(0, 0, 1), D_intra=D, D_extra=D)
    comps = Compartments(intra=Pool(T2=0.05), myelin=Pool(T2=0.01), extra=Pool(T2=0.08))
    g = d.MyelinatedCylinder(2e-6, 3e-6, compartments=comps, **kw)
    assert (g.T2_extra, g.T2_intra, g.T2_myelin) == (0.08, 0.05, 0.01) and g.compartments == comps
    with pytest.raises(TypeError, match="is a Compartments"):
        d.MyelinatedCylinder(2e-6, 3e-6, compartments={"intra": {"T2": 0.05}}, **kw)
    with pytest.raises(TypeError):                                     # a pool's T2 has one spelling
        d.MyelinatedCylinder(2e-6, 3e-6, T2_intra=0.05, **kw)
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
    with pytest.raises(ValueError, match="per cylinder"):            # a pool-wide T2 is a pool property
        d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4, T2_intra=0.05, T2_myelin=0.01, T2_extra=0.08)
    with pytest.raises(ValueError, match="both"):
        d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4, T2_intra=[0.05, 0.06, 0.07],
                                    compartments=Compartments(intra=Pool(T2=0.05)))
    arr = d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4, T2_intra=[0.05, 0.06, 0.07],
                                      compartments=Compartments(myelin=Pool(T2=0.01)))
    assert np.asarray(arr._T2_intra_jax)[:3].tolist() == pytest.approx([0.05, 0.06, 0.07])
    with pytest.raises(ValueError, match="both"):
        d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4, T2_intra=[0.05] * 3, compartments=comps)
