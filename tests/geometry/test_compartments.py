"""`Compartments` is the one spelling of per-compartment properties: pool names resolve to the
compartment ids of every channel, a property is read in id order, and half-specified properties
are refused rather than defaulted silently."""
import pytest

from dmipy_sim.compartments import Compartments, Pool, pool_id


def test_pool_ids_are_the_channel_convention():
    assert (pool_id("extra"), pool_id("intra"), pool_id("myelin")) == (0, 1, 2)
    with pytest.raises(KeyError, match="unknown pool"):
        pool_id("csf")
    for an_id in (0, 1, 3, True, None):                                # pools are addressed by name; the id is internal
        with pytest.raises(TypeError, match="addressed by its name"):
            pool_id(an_id)


def test_compartments_order_by_id():
    c = Compartments(intra=Pool(T2=0.05, D=1.7e-9), extra=Pool(T2=0.08, D=1.7e-9))
    assert list(c) == ["extra", "intra"] and c.ids == (0, 1)
    assert c["intra"].T2 == 0.05 and c["extra"].D == 1.7e-9
    for an_id in (0, 1):
        with pytest.raises(TypeError, match="addressed by its name"):
            c[an_id]
    assert 1 not in c and "intra" in c and "myelin" not in c
    with pytest.raises(KeyError, match="no 'myelin' pool"):
        c["myelin"]
    assert c.by_id("T2") == (0.08, 0.05) and c.by_id("D") == (1.7e-9, 1.7e-9)
    assert c.by_id("T1") is None
    assert Compartments.coerce(c) is c
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
    with pytest.raises(TypeError, match="takes a Pool, got dict"):
        Compartments(intra={"T2": 0.05})
    with pytest.raises(TypeError):                                     # the positional mapping is not a spelling
        Compartments({"intra": Pool(T2=0.05)})
    assert not hasattr(Pool, "coerce")
    with pytest.raises(ValueError, match="positive"):
        Pool(T2=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        Pool(D=-1.0)
    with pytest.raises(TypeError, match="takes a Compartments or None, got float"):
        Compartments.coerce(3.0)
    with pytest.raises(TypeError, match="takes a Compartments or None, got dict"):
        Compartments.coerce({"extra": Pool(T2=0.1)})


# ── the geometries take the one spelling, and the old spellings are the same walk ─────────────
import numpy as np

import dmipy_sim as d
from dmipy_sim.engine._fused import simulate_fused

D = 2e-9


def _wf():
    return d.set_b(d.pgse([[1, 0, 0]], 3e-3, 8e-3, gradient_strengths=0.1, n_t=60, slew_rate=np.inf), 1e9)


def test_mesh_takes_compartments_and_refuses_dicts_of_pools():
    from dmipy_sim.geometry import mesh_shapes
    V, F = mesh_shapes.icosphere(2e-6, subdivisions=2)
    comps = Compartments(intra=Pool(T2=0.02, D=1e-9, surface_relaxivity_t2=2e-6),
                         extra=Pool(T2=0.2, D=1e-9))
    m = d.Mesh(V, F, feature_radius=1e-6, compartments=comps)
    assert m.compartments == comps and m._T2_comp == (0.2, 0.02) and m._D_comp == (1e-9, 1e-9)
    assert m.surface_relaxivity_t2 == 2e-6
    for spelling in ({"intra": Pool(T2=0.02), "extra": Pool(T2=0.2)}, {"intra": {"T2": 0.02}}):
        with pytest.raises(TypeError, match="takes a Compartments or None, got dict"):
            d.Mesh(V, F, feature_radius=1e-6, compartments=spelling)
    with pytest.raises(ValueError, match="no myelin pool"):
        d.Mesh(V, F, compartments=Compartments(intra=Pool(D=D), myelin=Pool(D=0.0, T2=0.01), extra=Pool(D=D)))
    with pytest.raises(TypeError):                                     # the pools have one spelling
        d.Mesh(V, F, intra={"T2": 0.02})
    with pytest.raises(TypeError):
        Pool(kurtosis=1.0)
    with pytest.raises(TypeError, match="permeability"):                # a membrane's kappa is the geometry's
        Pool(permeability=1e-5)


def test_myelinated_cylinder_takes_compartments_and_no_other_spelling_of_a_pool():
    kw = dict(orientation=(0, 0, 1))
    comps = Compartments(intra=Pool(D=D, T2=0.05), myelin=Pool(D=0.0, T2=0.01), extra=Pool(D=D, T2=0.08))
    g = d.MyelinatedCylinder(2e-6, 3e-6, compartments=comps, **kw)
    assert (g.T2_extra, g.T2_intra, g.T2_myelin) == (0.08, 0.05, 0.01) and g.compartments == comps
    assert (g.D_intra, g.D_myelin, g.D_extra) == (D, 0.0, D)
    with pytest.raises(TypeError, match="takes a Compartments or None, got dict"):
        d.MyelinatedCylinder(2e-6, 3e-6, compartments={"intra": {"T2": 0.05}}, **kw)
    simulate_fused(300, None, _wf(), g, seed=0, require_gpu=False)
    for spelling in (dict(D_intra=D), dict(D_extra=D), dict(D_myelin=0.0), dict(water_fractions=(1.0, 0.4, 1.0))):
        with pytest.raises(TypeError, match=next(iter(spelling))):     # a pool's physics is the Pool's, once
            d.MyelinatedCylinder(2e-6, 3e-6, compartments=comps, **kw, **spelling)
        with pytest.raises(TypeError, match=next(iter(spelling))):
            d.PackedMyelinatedCylinders([1e-6], 0.7, np.zeros((1, 2)), 1e-5, N_max=2, compartments=comps, **spelling)
    cyl_kw = [(d.MyelinatedCylinder, (2e-6, 3e-6)),
              (d.PackedMyelinatedCylinders, ([1e-6], 0.7, np.zeros((1, 2)), 1e-5))]
    for cls, args in cyl_kw:                                           # D is the Pool's, on every pool: no default
        kk = kw if cls is d.MyelinatedCylinder else dict(N_max=2)
        with pytest.raises(ValueError, match="diffusivity of its intra pool"):
            cls(*args, compartments=Compartments(extra=Pool(D=D), myelin=Pool(D=0.0)), **kk)
        with pytest.raises(ValueError, match="diffusivity of its extra pool"):
            cls(*args, **kk)
        with pytest.raises(ValueError, match="diffusivity of its myelin pool.*frozen myelin pool states D=0.0"):
            cls(*args, compartments=Compartments(intra=Pool(D=D), extra=Pool(D=D)), **kk)
    d.MyelinatedCylinder(2e-6, 3e-6, compartments=Compartments(          # a myelin pool holding no water needs no D
        intra=Pool(D=D, water_fraction=1.0), myelin=Pool(water_fraction=0.0), extra=Pool(D=D, water_fraction=1.0)), **kw)
    with pytest.raises(ValueError, match="all three pools or none"):
        d.MyelinatedCylinder(2e-6, 3e-6, compartments=Compartments(intra=Pool(D=D, water_fraction=1.0), myelin=Pool(D=0.0), extra=Pool(D=D)), **kw)
    with pytest.raises(TypeError):                                     # a pool's T2 has one spelling
        d.MyelinatedCylinder(2e-6, 3e-6, T2_intra=0.05, compartments=comps, **kw)
    # the Substrate's pools drive a geometry directly
    sub = d.Substrate() if hasattr(d, "Substrate") else __import__("dmipy_sim.substrate.substrate", fromlist=["Substrate"]).Substrate()
    c = sub.compartments
    assert c.ids == (0, 1, 2) and c["myelin"].water_fraction == sub.myelin_water_proton_density
    g2 = d.MyelinatedCylinder(2e-6, 3e-6, orientation=(0, 0, 1), compartments=c)
    assert (g2.D_intra, g2.D_extra, g2.D_myelin) == (sub.D_intra, sub.D_extra, sub.D_myelin)
    assert g2.T2_myelin == sub.T2_myelin and g2.water_fractions == (1.0, sub.myelin_water_proton_density, 1.0)


def test_packed_myelinated_cylinders_take_compartments_and_keep_per_axon_arrays():
    L = float(np.sqrt(np.pi * 3 * (1e-6 / 0.7) ** 2 / 0.5))
    _, _, c = d.pack_myelinated_cylinders([1e-6] * 3, 0.7, None, cell_size=L, seed=0)
    comps = Compartments(intra=Pool(D=D, T2=0.05), myelin=Pool(D=0.0, T2=0.01), extra=Pool(D=D, T2=0.08))
    pm = d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4, compartments=comps)
    with pytest.raises(ValueError, match="per cylinder"):            # a pool-wide T2 is a pool property
        d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4, T2_intra=0.05, T2_myelin=0.01, T2_extra=0.08)
    with pytest.raises(ValueError, match="both"):
        d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4, T2_intra=[0.05, 0.06, 0.07],
                                    compartments=Compartments(intra=Pool(D=D, T2=0.05), myelin=Pool(D=0.0), extra=Pool(D=D)))
    arr = d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4, T2_intra=[0.05, 0.06, 0.07],
                                      compartments=Compartments(intra=Pool(D=D), myelin=Pool(D=0.0, T2=0.01), extra=Pool(D=D)))
    assert np.asarray(arr._T2_intra_jax)[:3].tolist() == pytest.approx([0.05, 0.06, 0.07])
    with pytest.raises(ValueError, match="both"):
        d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4, T2_intra=[0.05] * 3, compartments=comps)
