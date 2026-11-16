"""A cell of one myelinated and one plain cylinder: each keeps its own behaviour when the two coexist.

The cell is the permeable packed cell of ``test_mt_permeable_packed`` (two cylinders of outer radius R in a 20 R
cell). The plain cylinder (g-ratio 1) is permeable at ``KAPPA``; the myelinated one (g-ratio 0.7) is sealed, with a
diffusing sheath (``D_myelin = D``) so that no pool is frozen by accident. The walkers start in the lumens only
(``seed_pools=("intra",)``, by area), and are told apart by the lumen they start in.

* The plain cylinder's walkers leave it as the isolated permeable :class:`~dmipy_sim.geometry.Cylinder`'s do: the
  fraction inside at the end agrees to ``se_tol`` over the comparisons of the file.
* The myelinated cylinder's walkers never leave its lumen, and no walker is ever in a sheath (every recorded block
  end, zero tolerance); with its walls opened at ``KAPPA_CONTROL`` most of them leave, so the check resolves the
  object.
* Swapping which cylinder is myelinated swaps the outcome: it follows the cylinder, not its index.
* The power of the exchange check: the same cell with the plain wall sealed fails it (an expected failure). A plain
  wall given the myelinated rule (one trial per step, at its first encounter) does not fail it here: on a single
  convex wall at this step a second encounter in one step is a grazing rarity; that rule is resolved by the paired
  single-step check of ``tests/geometry/test_mixed_myelinated_plain_cell.py``.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.compartments import Compartments, Pool
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.geometry import PackedMyelinatedCylinders
from tests.conftest import se_tol

D = 2e-9
R = 2e-6
L = 40e-6
T = 6e-3
DT = 5e-5
G = 0.7
CENTRES = np.array([[10e-6, 20e-6], [30e-6, 20e-6]])
N = 6000                                    # by area, 4027 start in the plain lumen and 1973 in the myelinated one
COMPARISONS = 3

#: the plain cylinder's permeability, the oracle comparison's (the isolated cylinder's)
KAPPA = 4e-5
#: the negative control's: the myelinated cylinder's two walls opened, so that its walkers do leave
KAPPA_CONTROL = 4e-3


def _cell(sheathed, kappa_plain=KAPPA, kappa_sheath=0.0):
    """The cell with cylinder ``sheathed`` myelinated and the other plain."""
    g = np.ones(2); g[sheathed] = G
    k_in = np.zeros(2); k_in[sheathed] = kappa_sheath
    k_out = np.full(2, kappa_plain); k_out[sheathed] = kappa_sheath
    return PackedMyelinatedCylinders(R * g, g, CENTRES, L, N_max=2,
                                     compartments=Compartments(intra=Pool(D=D), myelin=Pool(D=D), extra=Pool(D=D)),
                                     kappa_inner=k_in, kappa_outer=k_out, seed_pools=("intra",))


def _walk(geometry, seed, r0=None):
    n = N if r0 is None else len(r0)
    return _walk_geometry(n, D, geometry, T, DT, seed=seed, raw_subset=n, require_gpu=False, r0=r0)


def _inside(w, centre, periodic):
    x = np.asarray(w.raw_positions, np.float64)[:, -1, :2] - np.asarray(centre)
    if periodic:
        x = (x + L / 2.0) % L - L / 2.0
    return (np.hypot(x[:, 0], x[:, 1]) < R).astype(float)


def _start_cylinder(w):
    x = np.asarray(w.raw_positions, np.float64)[:, 0, None, :2] - CENTRES[None]
    return np.argmin(np.hypot(x[..., 0], x[..., 1]), axis=1)


def _agree(a, b, what):
    """Two samples of one statistic agree to ``se_tol`` of their pooled spread."""
    sd = np.sqrt(0.5 * (a.var() + b.var()))
    tol = se_tol(len(a), sd, comparisons=COMPARISONS, runs=2)
    assert abs(a.mean() - b.mean()) < tol, f"{what}: {a.mean():.4f} against {b.mean():.4f} (tolerance {tol:.4f})"


_ISOLATED = {}


def _isolated():
    """The fraction inside at the end, per walker, of the isolated permeable cylinder (the oracle)."""
    if "inside" not in _ISOLATED:
        rng = np.random.default_rng(1)
        rho = R * np.sqrt(rng.uniform(0.0, 1.0, 4000)); phi = rng.uniform(0.0, 2.0 * np.pi, 4000)
        r0 = np.zeros((4000, 3), np.float32); r0[:, 0] = rho * np.cos(phi); r0[:, 1] = rho * np.sin(phi)
        w = _walk(d.Cylinder(radius=R, orientation=(0, 0, 1), permeability=KAPPA), 3, r0=r0)
        _ISOLATED["inside"] = _inside(w, (0.0, 0.0), False)
    return _ISOLATED["inside"]


def _populations(sheathed, seed, **kw):
    """The walk of the cell; per population (plain, myelinated) the walkers that start in its lumen."""
    w = _walk(_cell(sheathed, **kw), seed)
    start = _start_cylinder(w)
    comp = np.asarray(w.raw_compartment)
    assert (comp[:, 0] == 1).all(), "every walker starts in a lumen"
    plain = start != sheathed
    return w, comp, plain


@pytest.mark.parametrize("sheathed", [0, 1], ids=["myelinated_first", "myelinated_second"])
def test_each_cylinder_keeps_its_own_behaviour_in_the_mixed_cell(sheathed):
    w, comp, plain = _populations(sheathed, seed=2)
    assert plain.sum() >= 4000
    ip = _inside(w, CENTRES[1 - sheathed], True)[plain]
    assert 0.2 < ip.mean() < 0.95                                       # it crosses: the comparison resolves it
    _agree(ip, _isolated(), f"the plain cylinder (myelinated cylinder {sheathed}): fraction inside")
    assert (comp != 2).all(), "a walker entered a sheath"
    assert (comp[~plain] == 1).all(), "a walker left the sealed myelinated cylinder's lumen"
    assert (_inside(w, CENTRES[sheathed], True)[~plain] == 1.0).all()
    assert w.work["exhausted_steps"] == 0


def test_the_myelinated_cylinder_opened_lets_its_walkers_out():
    """The negative control of the confinement: the same walkers leave a myelinated cylinder whose two walls are
    permeable (``KAPPA_CONTROL``)."""
    w, comp, plain = _populations(0, seed=2, kappa_sheath=KAPPA_CONTROL)
    left = (comp[~plain] != 1).any(axis=1)
    assert left.mean() > 0.5, f"only {left.mean():.3f} of the myelinated cylinder's walkers left its lumen"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="the power of the exchange check: a sealed plain wall must fail it")
def test_a_sealed_plain_wall_fails_the_exchange_check():
    w, comp, plain = _populations(0, seed=2, kappa_plain=0.0)
    ip = _inside(w, CENTRES[1], True)[plain]
    _agree(ip, _isolated(), "the plain cylinder, sealed: fraction inside")
