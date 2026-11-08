"""The step law's transient term (dmrai-lab/dmipy-sim#930).

From walkers placed on one side of a membrane, a read at time ``t`` carries up to ``TRANSIENT_COEFFICIENT h^2 / (D t)``
beyond the law's other terms. A walk records whether its start is at spatial equilibrium, and a read of one whose start
is not is stated with that term and a flag."""
import numpy as np
import pytest

from dmipy_analytic.bloch_torrey import BlochTorrey1D
from dmipy_sim.engine import physics
from dmipy_sim.math.step_law import LayeredLattice

#: the reads, in units of ``l^2 / D`` (``l`` a layer's width)
TIMES = (0.01, 0.03, 0.1)


def _exchanged(kappa, h):
    """``(t, walk, continuum)``: the fraction on the far side of the membrane at :data:`TIMES` from walkers placed
    uniformly on one side, two layers of width 1 (``D = 1``), on the walk's exact chain and on the continuum."""
    lat = LayeredLattice([1.0, 1.0], h)
    n = [max(1, int(round(t / lat.dt))) for t in TIMES]
    walk = lat.occupancy([1.0, 0.0], lambda step: kappa, n)[:, 1]
    tt = np.asarray(n) * lat.dt
    edges = [0.0, lat.widths[0], lat.widths.sum()]
    ref = np.array([BlochTorrey1D(edges, 1.0, permeability=kappa, density=(1.0, 0.0), n_cells=800, gamma=1.0)
                    .signal(np.zeros(200), t / 200, sub_steps=4, by_layer=True).real[1] for t in tt])
    return lat, tt, walk, ref


@pytest.mark.parametrize("kappa", [0.5, 5.0])
def test_a_one_sided_starts_exchange_is_within_the_transient_term(kappa):
    """The exchanged fraction from walkers placed on one side of a membrane (``kappa l / D`` 0.5 and 5), on the walk's
    exact chain against the continuum: within ``TRANSIENT_COEFFICIENT h^2 / (D t)`` at every read, at ``h = 0.09 l`` and
    ``h / 2``, and second order: its error falls by about four when the step halves."""
    rel = {}
    for h in (0.09, 0.045):
        lat, tt, walk, ref = _exchanged(kappa, h)
        rel[h] = walk / ref - 1.0
        bound = np.array([physics.transient_error(1.0, lat.dt, t) for t in tt])
        assert np.all(np.abs(rel[h]) <= bound), (h, rel[h], bound)
    ratio = rel[0.045] / rel[0.09]
    assert np.all((ratio > 0.15) & (ratio < 0.4)), ratio


def test_a_walk_states_its_start_and_a_read_out_of_equilibrium_carries_the_term():
    """A walk records whether its walkers start at spatial equilibrium (``work["start"]``):
    - a permeable slab seeded on one side is out of it, said at the walk, and a read carries the transient term and a
      flag;
    - a walk handed its starts cannot tell (not established);
    - a packed cell seeded over both pools is at it;
    - an impermeable geometry needs no term.
    A walk of parts takes the least of its parts' starts."""
    from dmipy_sim.coefficient_walk import merge_work
    from dmipy_sim.engine.core import _walk_geometry
    from dmipy_sim.geometry.analytic import Cylinder, PermeableSlab1D
    from dmipy_sim.geometry.packed import PackedCylinders
    D, L = 2e-9, 10e-6
    slab = PermeableSlab1D(length=L, permeability=1e-5)
    with pytest.warns(UserWarning, match="spatial equilibrium"):
        w = _walk_geometry(32, D, slab, 2e-3, 5e-4, seed=1, require_gpu=False)
    start = w.work["start"]
    assert start["spatial_equilibrium"] is False and start["transient_coefficient"] == physics.TRANSIENT_COEFFICIENT
    assert start["kappa_over_D_max"] == pytest.approx(1e-5 / D)
    read = w.start_error(1e-3)
    assert read["flag"] == physics.START_OUT_OF_EQUILIBRIUM
    assert read["relative_error"] == pytest.approx(physics.TRANSIENT_COEFFICIENT * 6.0 * w.dt_sim / 1e-3)

    face_read = w.start_error(1e-3, per_face_local_time=True)
    assert face_read["flag"] == physics.PER_FACE_LOCAL_TIME_START
    assert face_read["relative_error"] == pytest.approx(0.5 * np.sqrt(6.0 * D * w.dt_sim) * (1 / np.sqrt(D * 1e-3) + 1e-5 / D))

    assert physics.start_record(slab, True, D, w.dt_sim)["spatial_equilibrium"] is None
    assert physics.start_error(physics.start_record(slab, True, D, w.dt_sim), D, w.dt_sim, 1e-3)["flag"] == physics.START_NOT_ESTABLISHED
    R = 2e-6
    centres = np.array([[0.0, 0.0], [2.2 * R, 0.0], [0.0, 2.2 * R], [2.2 * R, 2.2 * R]])
    both = physics.start_record(PackedCylinders([R, R, R, R], centres, 4.4 * R, permeability=1e-5), False, D, w.dt_sim)
    one = physics.start_record(PackedCylinders([R, R, R, R], centres, 4.4 * R, permeability=1e-5, pool="intra"), False, D, w.dt_sim)
    assert both["spatial_equilibrium"] is True and one["spatial_equilibrium"] is False
    assert physics.start_error(both, D, w.dt_sim, 1e-3) == dict(relative_error=0.0, flag=None)
    closed = physics.start_record(Cylinder(R, (0, 0, 1)), False, D, w.dt_sim)
    assert closed["spatial_equilibrium"] is True and closed["transient_coefficient"] is None
    assert physics.start_error(closed, D, w.dt_sim, 1e-3) == dict(relative_error=0.0, flag=None)

    merged = merge_work([dict(start=both), dict(start=start)])["start"]
    assert merged["spatial_equilibrium"] is False and merged["transient_coefficient"] == physics.TRANSIENT_COEFFICIENT
    assert merge_work([dict(start=both), dict(start=both)])["start"] == both


@pytest.mark.parametrize("h", [0.09, 0.045])
@pytest.mark.parametrize("near", [False, True])
def test_one_sided_face_local_time_is_within_first_order_term(h, near):
    """The exact walk's face local-time rates against continuum boundary densities, without crossing shares."""
    lat = LayeredLattice([1.0, 1.0], h)
    kappa = 10.0 if near else 0.5
    steps = round((0.01 if near else 0.1) / lat.dt)
    t = steps * lat.dt
    v = (np.array([1.0, 0.0]) / lat.cells)[lat.layer]
    P = lat.matrix(kappa).T
    for _ in range(steps):
        v = P @ v
    # Every encounter contributes twice its overshoot, including transmissions.
    i = np.arange(lat.N)[:, None]
    y = i + lat.k[None, :]
    edge = lat.edges[1]
    hit = (i < edge) & (y >= edge) if near else (i >= edge) & (y < edge)
    local_time_rate = v @ np.sum(lat.w * hit * 2.0 * np.abs(y + 0.5 - edge) * lat.dx, axis=1) / lat.dt

    ref = BlochTorrey1D([0.0, lat.widths[0], lat.widths.sum()], 1.0, permeability=kappa,
                       density=(1.0, 0.0), n_cells=800, gamma=1.0)
    # The zero-gradient Bloch-Torrey eigenmodes give cell densities at the read.
    density = (ref.U[0] @ (np.exp(ref.lam[0] * t) * (ref.U[0].T @ (ref.m0 * np.sqrt(ref.h))))) / np.sqrt(ref.h)
    j = ref.cells_per_layer[0]
    # Eliminate the half-cell resistances to recover the membrane's face density.
    flux = (density[j - 1] - density[j]) / (1.0 / kappa + 0.5 * (ref.h[j - 1] + ref.h[j]))
    face = density[j - 1] - flux * ref.h[j - 1] / 2.0 if near else density[j] + flux * ref.h[j] / 2.0
    exact_rate = face / np.sum(ref.m0 * ref.h)  # D = 1
    error = local_time_rate / exact_rate - 1.0
    bound = physics.per_face_local_time_error(1.0, lat.dt, t, kappa)
    if near:
        assert 0.18 * bound < error <= bound, (error, bound)
    else:
        assert -bound <= error < -0.5 * bound, (error, bound)


@pytest.mark.parametrize("equilibrium", [False, None, True])
def test_per_face_local_time_read_flag(equilibrium):
    start = dict(spatial_equilibrium=equilibrium, transient_coefficient=physics.TRANSIENT_COEFFICIENT)
    read = physics.start_error(start, 2e-9, 1e-5, 1e-3, per_face_local_time=True)
    if equilibrium is True:
        assert read == dict(relative_error=0.0, flag=None)
    else:
        assert read == dict(relative_error=pytest.approx(0.5 * np.sqrt(0.06)),
                            flag=physics.PER_FACE_LOCAL_TIME_START)
    assert physics.per_face_local_time_error(1.0, 0.01, 0.0) == float("inf")
    assert physics.per_face_local_time_error(1.0, 0.01, -1.0) == float("inf")


@pytest.mark.parametrize("permeability", [
    pytest.param(4e-5, id="scalar"),
    [1e-5, 4e-5],
    pytest.param("schedule", id="all-objects-and-intervals"),
])
@pytest.mark.parametrize("geometry_kind", ["packed", "slab"])
def test_start_records_largest_permeability_over_D(permeability, geometry_kind):
    from dmipy_sim.geometry.base import PermeabilitySchedule
    from dmipy_sim.geometry.packed import PackedCylinders
    from dmipy_sim.coefficient_walk import merge_starts

    if isinstance(permeability, str):
        permeability = PermeabilitySchedule([1e-3], [[1e-5, 2e-5], [4e-5, 0.0]])
    geometry = PackedCylinders([1e-6, 1e-6], [[0.0, 0.0], [3e-6, 0.0]], 6e-6,
                               permeability=permeability, pool="intra")
    if geometry_kind == "slab":
        from dmipy_sim.geometry.analytic import PermeableSlab1D
        geometry = PermeableSlab1D(6e-6, permeability, membranes=[2e-6, 4e-6])
    start = physics.start_record(geometry, False, 2e-9, 1e-5)
    assert start["kappa_over_D_max"] == pytest.approx(4e-5 / 2e-9)
    lesser = dict(start, kappa_over_D_max=1e-5 / 2e-9)
    assert merge_starts([lesser, start])["kappa_over_D_max"] == start["kappa_over_D_max"]
    read = physics.start_error(start, 2e-9, 1e-5, 1e-3, per_face_local_time=True)
    assert read["relative_error"] == pytest.approx(
        physics.per_face_local_time_error(2e-9, 1e-5, 1e-3, 4e-5 / 2e-9))
