"""A permeability per object of a packed cell, and over the walk's time: the trial of a wall encounter reads the
permeability of the object met, at the step's midpoint (:func:`~dmipy_sim.engine.physics.membrane_kappa`).

Consistency, exact rather than statistical:

* every object at one permeability is the scalar walk, bit for bit (the same draws, the same arithmetic);
* an object at zero is impermeable: no walker seeded in it ever leaves it and none outside ever enters it, while the
  other objects exchange (a trial that read any other object's permeability there would let walkers through);
* a schedule that seals every wall at ``t_b`` stops every pool change at ``t_b`` exactly (a trial that read the
  start's permeability throughout keeps exchanging), and one that never changes its value is the constant walk.

The permeability's record and the step the breakpoint is taken at are checked on the walk's work record.
"""
import numpy as np
import pytest

D = 2e-9


def _cylinders(permeability):
    from dmipy_sim.geometry import PackedCylinders, pack_cylinders
    radii = [1.0e-6, 0.8e-6, 1.2e-6, 0.9e-6]
    c, L, _ = pack_cylinders(radii, target_vf=0.45, seed=3)
    return PackedCylinders(radii, c, L, permeability=permeability)


def _spheres(permeability):
    from dmipy_sim.geometry import PackedSpheres, pack_spheres
    radii = [1.0e-6, 0.8e-6, 1.1e-6]
    c, L, _ = pack_spheres(radii, target_vf=0.3, seed=3)
    return PackedSpheres(radii, c, L, permeability=permeability)


KAPPA = 4e-3                 # m/s: kappa R / D = 2 at R = 1 um
DT = 2e-6                    # one step per block: h = sqrt(6 D dt) = 0.15 um
N_STEPS = 400


def _walk(g, n=512, seed=5, raw=0):
    from dmipy_sim.engine.core import _walk_geometry
    return _walk_geometry(n, D, g, N_STEPS * DT, DT, sub_steps=1, seed=seed, raw_subset=raw, require_gpu=False)


@pytest.mark.parametrize("make", [_cylinders, _spheres])
def test_every_object_at_one_permeability_is_the_scalar_walk_to_the_bit(make):
    a = _walk(make(KAPPA))
    b = _walk(make([KAPPA] * len(make(KAPPA)._radii_np)))
    np.testing.assert_array_equal(a.position_coeffs, b.position_coeffs)
    np.testing.assert_array_equal(a.boundary_bands, b.boundary_bands)
    np.testing.assert_array_equal(a.comp_events.t, b.comp_events.t)
    np.testing.assert_array_equal(a.comp_events.label, b.comp_events.label)
    assert a.comp_events.n_events > 0
    assert a.work["permeability"] is None and b.work["permeability"]["values_m_per_s"] == [[KAPPA] * len(make(KAPPA)._radii_np)]


@pytest.mark.parametrize("make", [_cylinders, _spheres])
def test_an_object_at_zero_permeability_is_impermeable_and_the_others_exchange(make):
    n_obj = len(make(KAPPA)._radii_np)
    sealed = 2
    kappa = [KAPPA] * n_obj
    kappa[sealed] = 0.0
    g = make(kappa)
    n = 2048
    w = _walk(g, n=n, raw=n)
    pos = np.asarray(w.raw_positions)                                   # (n, n_steps + 1, 3)
    obj = np.asarray(g.classify_positions_exact(pos.reshape(-1, 3))).reshape(pos.shape[:2])   # 0 outside, k + 1 in k
    in_sealed = obj == sealed + 1
    started = in_sealed[:, 0]
    assert started.sum() > 20
    assert in_sealed[started].all(), "a walker left the sealed object"
    assert not in_sealed[~started].any(), "a walker entered the sealed object"
    others = (obj != obj[:, :1]).any(axis=1) & ~started
    assert others.sum() > 50, "the other objects did not exchange"


def test_a_trial_reading_another_objects_permeability_would_open_the_sealed_one():
    """The comparison above resolves the object: with the sealed object's neighbours' permeability on every wall the
    same walk changes the sealed object's walkers at once (the variant the test fails on)."""
    g = _cylinders(KAPPA)
    n = 2048
    w = _walk(g, n=n, raw=n)
    pos = np.asarray(w.raw_positions)
    obj = np.asarray(g.classify_positions_exact(pos.reshape(-1, 3))).reshape(pos.shape[:2])
    started = obj[:, 0] == 3
    assert (obj[started] != 3).any(axis=1).mean() > 0.5


def test_a_schedule_that_seals_every_wall_stops_the_exchange_at_its_breakpoint():
    from dmipy_sim.geometry import PermeabilitySchedule
    n_obj = 4
    t_b = 150 * DT                                                      # on a step boundary
    per_object = [KAPPA, 0.5 * KAPPA, 2 * KAPPA, KAPPA]
    g = _cylinders(PermeabilitySchedule([t_b], [per_object, [0.0] * n_obj]))
    w = _walk(g, n=2048)
    t = w.comp_events.t
    assert (t <= t_b * (1 + 1e-9)).all(), f"{int((t > t_b * (1 + 1e-9)).sum())} pool changes after the walls sealed"
    assert (t > 0.5 * t_b).sum() > 20                                   # it was exchanging up to the breakpoint
    rec = w.work["permeability"]
    assert rec["rule"] == "midpoint" and rec["taken_at_s"] == pytest.approx([t_b], rel=1e-12)
    assert rec["values_m_per_s"] == [per_object, [0.0] * n_obj] and rec["breakpoints_s"] == [t_b]


def test_a_schedule_that_keeps_its_value_is_the_constant_walk_to_the_bit():
    from dmipy_sim.geometry import PermeabilitySchedule
    a = _walk(_cylinders(KAPPA))
    b = _walk(_cylinders(PermeabilitySchedule([100 * DT, 300 * DT], [KAPPA, KAPPA, KAPPA])))
    np.testing.assert_array_equal(a.position_coeffs, b.position_coeffs)
    np.testing.assert_array_equal(a.comp_events.t, b.comp_events.t)


def test_the_step_reads_the_permeability_at_its_midpoint():
    """A breakpoint on a step boundary is taken there; one inside a step at the boundary nearest to it, with a warning
    that states the shift (at most half a step) and the record of where it was taken."""
    import jax.numpy as jnp
    from dmipy_sim.engine.physics import membrane_kappa
    from dmipy_sim.geometry import PermeableSlab1D, PermeabilitySchedule
    dt = 1e-6
    on = PermeableSlab1D(1e-5, PermeabilitySchedule([7 * dt], [1e-5, 3e-5]))
    k_at, rec = membrane_kappa(on, D, dt)
    assert [float(k_at(jnp.int32(i))) * D for i in (6, 7)] == pytest.approx([1e-5, 3e-5], rel=1e-6)
    assert rec["taken_at_s"] == pytest.approx([7 * dt]) and abs(rec["shift_s"][0]) < 1e-9 * dt
    for frac, step in ((0.3, 7), (0.7, 8)):
        off = PermeableSlab1D(1e-5, PermeabilitySchedule([(7 + frac) * dt], [1e-5, 3e-5]))
        with pytest.warns(UserWarning, match="nearest step boundary"):
            k_at, rec = membrane_kappa(off, D, dt)
        assert float(k_at(jnp.int32(step - 1))) * D == pytest.approx(1e-5, rel=1e-6)
        assert float(k_at(jnp.int32(step))) * D == pytest.approx(3e-5, rel=1e-6)
        assert rec["taken_at_s"] == pytest.approx([step * dt]) and abs(rec["shift_s"][0]) <= 0.5 * dt
    with pytest.raises(ValueError, match="varies in time"):
        membrane_kappa(on, D, dt)[0](None)


@pytest.mark.parametrize("bad, match", [
    (dict(breakpoints=[0.002, 0.001], values=[1e-5, 2e-5, 3e-5]), "strictly increasing"),
    (dict(breakpoints=[0.001], values=[1e-5]), "one entry per interval"),
    (dict(breakpoints=[0.001], values=[1e-5, -2e-5]), "non-negative"),
    (dict(breakpoints=[0.001], values=[[1e-5, 2e-5], [1e-5, 2e-5, 3e-5]]), "different lengths"),
])
def test_an_ill_formed_schedule_is_refused(bad, match):
    from dmipy_sim.geometry import PermeabilitySchedule
    with pytest.raises(ValueError, match=match):
        PermeabilitySchedule(**bad)


def test_a_permeability_per_object_is_refused_where_there_is_one_object_or_no_label():
    from dmipy_sim import Compartments, Pool
    from dmipy_sim.geometry import Sphere, Mesh, PermeabilitySchedule, MyelinatedCylinder
    from dmipy_sim.geometry import mesh_shapes
    with pytest.raises(ValueError, match="one wall"):
        Sphere(3e-6, permeability=[1e-5, 2e-5])
    with pytest.raises(ValueError, match="4 objects"):
        _cylinders([1e-5, 2e-5])
    with pytest.raises(ValueError, match="4 objects"):
        _cylinders(PermeabilitySchedule([1e-3], [[1e-5] * 3, 0.0]))
    V, F = mesh_shapes.icosphere(5e-6, subdivisions=2)
    for k in ([1e-5, 2e-5], PermeabilitySchedule([1e-3], [1e-5, 0.0])):
        with pytest.raises(NotImplementedError, match="no label"):
            Mesh(V, F, feature_radius=1e-6, permeability=k)
    with pytest.raises(NotImplementedError, match="PermeabilitySchedule"):
        MyelinatedCylinder(2e-6, 3e-6, (0, 0, 1), kappa_inner=PermeabilitySchedule([1e-3], [1e-5, 0.0]),
                           compartments=Compartments(intra=Pool(D=D), extra=Pool(D=D), myelin=Pool(D=0.0)))


def test_a_backend_that_does_not_declare_a_permeability_table_is_refused_by_name():
    """The backend states which permeability inputs it walks: one that declares no capability (a duck-typed backend, or
    the protocol's default) is refused a permeability per object or over time before it is asked, one that declares
    :data:`~dmipy_sim.engine.backends.PERMEABILITY_TABLE` is handed the walk, and a walk at one number is no one's table."""
    from dmipy_sim.engine import backends
    from dmipy_sim.engine.core import _backend_refusal, _resolve_walk

    class Plain(backends.Backend):
        name = "plain"

    class Declares(Plain):
        name = "declares"
        capabilities = frozenset({backends.PERMEABILITY_TABLE})

    class Duck:
        name = "duck"

        def refuses(self, request):
            return None

    g = _cylinders([KAPPA, 0.0, KAPPA, KAPPA])
    res = _resolve_walk(g, D, N_STEPS * DT, DT, sub_steps=1)
    assert res.jax_only == []
    assert res.request.kappa_over_D == pytest.approx(KAPPA / D)          # the largest, which sizes the events
    for b in (Plain(), Duck()):
        assert "does not declare it walks (capability 'permeability_table')" in _backend_refusal(b, b.name, res, g)
        assert f"backend {b.name!r}: this walk uses a permeability per object or over time" in _backend_refusal(b, b.name, res, g)
    assert _backend_refusal(Declares(), "declares", res, g) is None
    assert backends.PERMEABILITY_TABLE in backends.JAX_CAPABILITIES
    one = _cylinders(KAPPA)
    assert _backend_refusal(Plain(), "plain", _resolve_walk(one, D, N_STEPS * DT, DT, sub_steps=1), one) is None


def test_a_backend_that_declares_a_table_still_refuses_what_it_refuses():
    from dmipy_sim.engine import backends
    from dmipy_sim.engine.core import _backend_refusal, _resolve_walk

    class Declares(backends.Backend):
        name = "declares"
        capabilities = frozenset({backends.PERMEABILITY_TABLE})

        def refuses(self, request):
            return "no tables on Tuesday"

    g = _cylinders([KAPPA, 0.0, KAPPA, KAPPA])
    res = _resolve_walk(g, D, N_STEPS * DT, DT, sub_steps=1)
    assert "backend 'declares' refuses this walk of PackedCylinders: no tables on Tuesday" == _backend_refusal(Declares(), "declares", res, g)
