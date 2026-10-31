"""The myelinated substrates' seeders pull every accepted point strictly onto its pool's side of both of that
pool's walls (dmipy-sim#659; the analytic family's counterpart is ``test_seed_wall_rounding.py``).

The rejection tests run in float64 and the accepted batch is cast to float32. The RNG is monkeypatched to plant, at
the head of every batch, draws ``1e-12`` relative off each wall: strictly on the pool's side in float64, yet their
float32 value is the wall itself (``test_seed_wall_rounding._round_onto_wall``). Every seed must then lie strictly on
its pool's side, in float32, of the inner and the outer wall.
"""
import jax
import numpy as np
import pytest

from dmipy_sim.geometry import myelin
from dmipy_sim.geometry.myelin import MyelinatedCylinder, PackedMyelinatedCylinders
from dmipy_sim import Compartments, Pool

_REAL_DEFAULT_RNG = np.random.default_rng
_OFF = 1e-12


class _PlantedRng:
    """Stand-in for ``np.random.Generator``: every ``uniform`` batch opens with ``rows``, then real draws."""

    def __init__(self, rows):
        self._rows = np.asarray(rows, np.float64)
        self._fallback = _REAL_DEFAULT_RNG(4321)

    def uniform(self, lo, hi, size):
        out = self._fallback.uniform(lo, hi, size)
        k = min(len(self._rows), len(out))
        out[:k] = self._rows[:k]
        return out


def _rows(R_in, R_out):
    """Draws within one float32 ulp of the inner wall (either side) and of the outer wall (either side)."""
    return [[R_in * (1 - _OFF), 0.0], [R_in * (1 + _OFF), 0.0], [R_out * (1 - _OFF), 0.0], [R_out * (1 + _OFF), 0.0]]


def _check_strict(xy, pool, R_in, R_out):
    """Every seed strictly on its pool's side of both walls, by the float32 squared distance the kernel tests."""
    f32 = np.float32
    d2 = (xy[:, 0].astype(f32) ** 2 + xy[:, 1].astype(f32) ** 2)
    Ri2, Ro2 = f32(R_in) ** 2, f32(R_out) ** 2
    for name, ok in (("lumen", (pool == 1, d2 < Ri2)),
                     ("sheath", (pool == 2, (d2 > Ri2) & (d2 < Ro2))),
                     ("extra", (pool == 0, d2 > Ro2))):
        sel, strict = ok
        assert sel.any(), f"{name}: no seed was drawn"
        assert strict[sel].all(), f"{name}: {int((~strict[sel]).sum())} seeds are not strictly on the pool's side"


def test_myelinated_cylinder_seeds_are_strictly_on_their_pools_side(monkeypatch):
    R_in, R_out = 1e-6, 1.43e-6
    monkeypatch.setattr(myelin.np.random, "default_rng", lambda seed: _PlantedRng(_rows(R_in, R_out)))
    g = MyelinatedCylinder(R_in, R_out, (0, 0, 1), compartments=Compartments(intra=Pool(D=2e-9), myelin=Pool(D=0.0), extra=Pool(D=2e-9)))
    pos = np.asarray(g.init_positions(400, jax.random.PRNGKey(0)))
    pool = np.asarray(g._init_compartments)
    _check_strict(pos[:, :2], pool, R_in, R_out)


def test_packed_myelinated_seeds_are_strictly_on_their_pools_side(monkeypatch):
    R_in, L = 1e-6, 20e-6
    g = PackedMyelinatedCylinders([R_in], 0.7, np.zeros((1, 2)), L, N_max=2)
    R_in, R_out = float(g._inner_radii_np[0]), float(g._outer_radii_np[0])
    monkeypatch.setattr(myelin.np.random, "default_rng", lambda seed: _PlantedRng(_rows(R_in, R_out)))
    pos = np.asarray(g.init_positions(600, jax.random.PRNGKey(0)))
    pool = np.asarray(g.pool_of(g._init_compartments))
    _check_strict(pos[:, :2], pool, R_in, R_out)
