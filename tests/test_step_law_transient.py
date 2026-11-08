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
    read = w.start_error(1e-3)
    assert read["flag"] == physics.START_OUT_OF_EQUILIBRIUM
    assert read["relative_error"] == pytest.approx(physics.TRANSIENT_COEFFICIENT * 6.0 * w.dt_sim / 1e-3)

    assert physics.start_record(slab, True)["spatial_equilibrium"] is None
    assert physics.start_error(physics.start_record(slab, True), D, w.dt_sim, 1e-3)["flag"] == physics.START_NOT_ESTABLISHED
    R = 2e-6
    centres = np.array([[0.0, 0.0], [2.2 * R, 0.0], [0.0, 2.2 * R], [2.2 * R, 2.2 * R]])
    both = physics.start_record(PackedCylinders([R, R, R, R], centres, 4.4 * R, permeability=1e-5), False)
    one = physics.start_record(PackedCylinders([R, R, R, R], centres, 4.4 * R, permeability=1e-5, pool="intra"), False)
    assert both["spatial_equilibrium"] is True and one["spatial_equilibrium"] is False
    assert physics.start_error(both, D, w.dt_sim, 1e-3) == dict(relative_error=0.0, flag=None)
    closed = physics.start_record(Cylinder(R, (0, 0, 1)), False)
    assert closed["spatial_equilibrium"] is True and closed["transient_coefficient"] is None
    assert physics.start_error(closed, D, w.dt_sim, 1e-3) == dict(relative_error=0.0, flag=None)

    merged = merge_work([dict(start=both), dict(start=start)])["start"]
    assert merged["spatial_equilibrium"] is False and merged["transient_coefficient"] == physics.TRANSIENT_COEFFICIENT
    assert merge_work([dict(start=both), dict(start=both)])["start"] == both
