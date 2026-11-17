"""The equilibrium bound fraction of a cell of myelinated and plain cylinders, per water domain.

:func:`~dmipy_sim.engine.mt.equilibrium_domains` states, for a :class:`~dmipy_sim.geometry.PackedMyelinatedCylinders`
cell whose sheaths are unseeded and sealed from every seeded pool, each water domain's equilibrium ``P = x / (1 + x)``,
``x = kappa_MT dwell S / V``, and its seeded share. Exact, not statistical:

* the table of a mixed cell is the closed form evaluated here in float64 from the cell's own lengths, its plain
  cylinders, their permeability and the seeded pools; counting a face otherwise gives another value;
* a cell of plain cylinders only is the plain packing, permeable and sealed; one sealed myelinated cylinder's lumen is
  the isolated cylinder; one sealed plain cylinder is the sealed packing of one;
* a sheath that is seeded or reachable from a seeded pool through an open face (the reach of
  ``placement_is_stationary``) has no table, by name, and ``'fast'`` is refused there with the same reason;
* ``'fast'`` binds each walker by its own draw against its own start domain's equilibrium, walker for walker;
* the burn-in of the cell stops by the equilibrium and records the table; every other geometry's equilibrium, surface
  to volume and burn-in record are what they were.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.compartments import Compartments, Pool
from dmipy_sim.engine import mt
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.geometry import PackedCylinders, PackedMyelinatedCylinders, PermeableSlab1D, FreeDiffusion

D = 2e-9
R = 2e-6
OUTER = np.array([R, 0.8 * R, R, 0.9 * R])
L = 4.4 * R
CENTRES = np.array([[0.0, 0.0], [2.2, 0.0], [0.0, 2.2], [2.2, 2.2]]) * R
MIXED = (0.7, 1.0, 0.7, 1.0)
OPEN = (0.0, 4e-5, 0.0, 4e-5)
KAPPA_MT, TAU = 4e-5, 1e-3
DRY = Pool(D=0.0)                                          # the sheath's spelling: unseeded by seed_pools
SEEDED = ("extra", "intra")


def _cell(g=MIXED, kappa_outer=OPEN, kappa_inner=0.0, myelin=DRY, seed_pools=SEEDED, outer=OUTER, centres=CENTRES,
          cell=L):
    g = np.asarray(g, float)
    comps = Compartments(intra=Pool(D=D), myelin=myelin, extra=Pool(D=D))
    return PackedMyelinatedCylinders(np.asarray(outer) * g, g, centres, cell, N_max=len(g), compartments=comps,
                                     kappa_outer=list(kappa_outer), kappa_inner=kappa_inner, seed_pools=seed_pools)


def _formula(g, kappa_outer, seeded, *, outer=OUTER, cell=L, kappa=KAPPA_MT, tau=TAU, outer_faces=1.0,
             open_faces=2.0):
    """The closed form, per domain: ``[(pools, cylinders, V, S, weight, P)]`` in the table's order, and ``p_eq``.
    ``outer_faces`` / ``open_faces`` count the faces of a sheath's or sealed plain wall and of an open plain wall in the
    exterior (1 and 2: the walk binds at every face it meets)."""
    g = np.asarray(g, float)
    a = np.asarray(outer) * g                              # the lumen radii as given to the cell
    b = a / g                                              # and its outer radii as the cell forms them
    open_plain = (g == 1.0) & (np.asarray(kappa_outer, float) > 0.0)
    x = lambda S, V: kappa * tau * S / V
    P = lambda S, V: x(S, V) / (1.0 + x(S, V))
    rows = []
    if "intra" in seeded:
        for k in range(g.size):
            if not open_plain[k]:
                rows.append([["intra"], [k], np.pi * a[k] ** 2, 2.0 * np.pi * a[k], np.pi * a[k] ** 2])
    V = cell ** 2 - sum(np.pi * b[k] ** 2 for k in range(g.size) if not open_plain[k])
    S = (sum(outer_faces * 2.0 * np.pi * b[k] for k in range(g.size) if not open_plain[k])
         + sum(open_faces * 2.0 * np.pi * b[k] for k in range(g.size) if open_plain[k]))
    area = ((cell ** 2 - sum(np.pi * b ** 2) if "extra" in seeded else 0.0)
            + (sum(np.pi * b[open_plain] ** 2) if "intra" in seeded else 0.0))
    if area > 0.0:
        rows.append([["extra", "intra"] if open_plain.any() else ["extra"], np.flatnonzero(open_plain).tolist(),
                     V, S, area])
    total = sum(r[4] for r in rows)
    out = [(r[0], r[1], r[2], r[3], r[4] / total, P(r[3], r[2])) for r in rows]
    return out, sum(w * p for *_, w, p in out)


def _within(got, want, spacings):
    """``|got - want|`` within ``spacings`` float64 spacings of ``want``; a NaN fails."""
    return bool(np.abs(got - want) <= spacings * np.spacing(np.abs(want)))


def _assert_table(table, g, kappa_outer, seeded, **kw):
    rows, p_eq = _formula(g, kappa_outer, seeded, **kw)
    assert table.reason is None and len(table.domains) == len(rows)
    for dom, (pools, cyl, V, S, w, P) in zip(table.domains, rows):
        assert dom["pools"] == pools and dom["cylinders"] == cyl
        for key, want in (("volume", V), ("surface", S), ("surface_to_volume", S / V), ("weight", w), ("p_eq", P)):
            assert _within(dom[key], want, 8), f"{key}: {dom[key]!r} against the closed form {want!r}"
    assert _within(sum(dom["weight"] for dom in table.domains), 1.0, 4)
    assert _within(table.p_eq, p_eq, 8)
    assert table.p_eq == sum(dom["weight"] * dom["p_eq"] for dom in table.domains)
    return rows


# ── the table of a mixed cell is the closed form ───────────────────────────────────────────────────────────────────

def test_the_mixed_cell_is_two_lumina_and_the_exterior():
    """The two myelinated lumina are domains of their own; the open plain lumina join the extra pool."""
    rows = _assert_table(mt.equilibrium_domains(_cell(), KAPPA_MT, TAU), MIXED, OPEN, SEEDED)
    assert [r[1] for r in rows] == [[0], [2], [1, 3]]
    assert mt.equilibrium_bound_fraction(_cell(), KAPPA_MT, TAU) == mt.equilibrium_domains(_cell(), KAPPA_MT, TAU).p_eq


def test_sealed_plain_cylinders_are_domains_of_their_own():
    rows = _assert_table(mt.equilibrium_domains(_cell(kappa_outer=(0, 0, 0, 0)), KAPPA_MT, TAU), MIXED, (0, 0, 0, 0),
                         SEEDED)
    assert [r[1] for r in rows] == [[0], [1], [2], [3], []]


def test_the_faces_are_counted_once_per_side_a_walker_meets():
    """The engine's exterior is not the exterior with an open plain wall counted once, nor without the sheaths' outer
    faces: the face count is the formula's, not only its arithmetic."""
    ext = mt.equilibrium_domains(_cell(), KAPPA_MT, TAU).domains[-1]["p_eq"]
    for kw in (dict(open_faces=1.0), dict(outer_faces=0.0)):
        rows, _ = _formula(MIXED, OPEN, SEEDED, **kw)
        assert not _within(ext, rows[-1][5], 1e6), kw


@pytest.mark.parametrize("seeded", [("extra",), ("intra",)])
def test_a_pool_seeded_alone_weights_the_domains_it_seeds(seeded):
    table = mt.equilibrium_domains(_cell(seed_pools=seeded), KAPPA_MT, TAU)
    rows = _assert_table(table, MIXED, OPEN, seeded)
    if seeded == ("extra",):
        assert len(rows) == 1 and table.domains[0]["weight"] == 1.0


def test_the_spelling_of_the_dry_sheath_does_not_matter():
    """A sheath of no water, seeded by default, is the same cell as a sheath unseeded by name."""
    a = mt.equilibrium_domains(_cell(), KAPPA_MT, TAU)
    b = mt.equilibrium_domains(_cell(myelin=Pool(D=0.0, water_fraction=0.0), seed_pools=None), KAPPA_MT, TAU)
    assert a == b


def test_an_open_axolemma_out_of_reach_leaves_the_exterior_alone():
    """Every sheath's axolemma open, every outer wall sealed, the extra pool seeded alone: the lumina and sheaths are
    out of reach, and the exterior is the one domain."""
    geo = _cell(kappa_outer=(0, 0, 0, 0), kappa_inner=4e-5, seed_pools=("extra",))
    assert geo.reachable_pools == ("extra",)
    table = mt.equilibrium_domains(geo, KAPPA_MT, TAU)
    _assert_table(table, MIXED, (0, 0, 0, 0), ("extra",))
    assert len(table.domains) == 1 and table.domains[0]["weight"] == 1.0


# ── the exact identities ────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kappa", [4e-5, 0.0])
def test_a_cell_of_plain_cylinders_is_the_plain_packing(kappa):
    cell = _cell(g=(1.0, 1.0, 1.0, 1.0), kappa_outer=(kappa,) * 4)
    plain = PackedCylinders(OUTER, CENTRES, L, permeability=kappa or None)
    got = mt.equilibrium_bound_fraction(cell, KAPPA_MT, TAU)
    want = mt.equilibrium_bound_fraction(plain, KAPPA_MT, TAU)
    if kappa > 0.0:
        assert got == want                                    # one domain: the same sums in the same order
    else:
        assert _within(got, want, 4)                          # the pools' weights summed in another order


def test_one_sealed_myelinated_cylinder_is_the_isolated_cylinder():
    a = 1.4e-6
    cell = _cell(g=(0.7,), kappa_outer=(0.0,), outer=[a / 0.7], centres=np.zeros((1, 2)))
    lumen = mt.equilibrium_domains(cell, KAPPA_MT, TAU).domains[0]
    assert lumen["cylinders"] == [0]
    assert lumen["p_eq"] == mt.equilibrium_bound_fraction(d.Cylinder(radius=a, orientation=(0, 0, 1)), KAPPA_MT, TAU)


def test_one_sealed_plain_cylinder_is_the_sealed_packing_of_one():
    cell = _cell(g=(1.0,), kappa_outer=(0.0,), outer=[R], centres=np.zeros((1, 2)))
    want = mt.equilibrium_bound_fraction(PackedCylinders(np.array([R]), np.zeros((1, 2)), L), KAPPA_MT, TAU)
    assert _within(mt.equilibrium_bound_fraction(cell, KAPPA_MT, TAU), want, 4)


# ── refusals, each beside a neighbour that has a table ─────────────────────────────────────────────────────────────

REACHABLE = ("the myelin pool is reachable from a seeded pool through an open face: only a sheath that is unseeded and "
             "sealed from every seeded pool (the reach of placement_is_stationary) has a closed form")
SEEDED_SHEATH = ("the myelin pool is seeded: only a sheath that is unseeded and sealed from every seeded pool (the "
                 "reach of placement_is_stationary) has a closed form")
REFUSED = {
    "open axolemma, lumina seeded": (dict(kappa_inner=4e-5), REACHABLE),
    "open outer wall of a myelinated cylinder": (dict(kappa_outer=(4e-5,) * 4), REACHABLE),
    "sheath seeded": (dict(seed_pools=None), SEEDED_SHEATH),
    "open axolemma reached through an open plain wall": (dict(kappa_inner=4e-5, seed_pools=("extra",)), REACHABLE),
}


@pytest.mark.parametrize("case", list(REFUSED))
def test_a_sheath_in_reach_has_no_table_and_says_why(case):
    kw, reason = REFUSED[case]
    geo = _cell(**kw)
    assert "myelin" in geo.reachable_pools
    table = mt.equilibrium_domains(geo, KAPPA_MT, TAU)
    assert table.domains is None and table.p_eq is None and table.reason == reason
    assert mt.equilibrium_bound_fraction(geo, KAPPA_MT, TAU) is None
    crit = mt.burnin_criterion(geo, KAPPA_MT, TAU, D, 1e-3)
    assert crit.kind == "plateau" and crit.p_eq is None and crit.p_eq_reason == reason
    with pytest.raises(ValueError, match=r"\(the reach of placement_is_stationary\)") as err:
        mt.resolve_equilibrate_mode("fast", geo)
    assert reason in str(err.value)


def test_fast_is_accepted_where_the_table_exists():
    assert mt.resolve_equilibrate_mode("fast", _cell()) == "fast"
    assert mt.resolve_equilibrate_mode("fast", d.Cylinder(radius=R, orientation=(0, 0, 1))) == "fast"
    with pytest.raises(ValueError, match="needs a known surface-to-volume"):
        mt.resolve_equilibrate_mode("fast", PackedCylinders(OUTER, CENTRES, L))


def test_the_table_is_of_the_cell_alone():
    with pytest.raises(TypeError, match="takes a PackedMyelinatedCylinders cell"):
        mt.equilibrium_domains(PackedCylinders(OUTER, CENTRES, L), KAPPA_MT, TAU)


# ── 'fast': each walker bound against its own domain's equilibrium ─────────────────────────────────────────────────

@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_fast_binds_each_walker_by_its_own_draw_against_its_domains_equilibrium(precision, request):
    """The bound flag at the start is ``u_i < float32(P_d(i))`` for walker ``i``'s own draw and the closed form of the
    domain its start lies in, walker for walker. Against ``P_d`` in float64 the flag differs only for a draw within
    one float32 spacing of ``P_d``; those walkers are counted and left out of that comparison."""
    if precision == "float64":
        request.getfixturevalue("x64")
    n, seed = 512, 7
    geo = _cell()
    w = _walk_geometry(n, D, geo, 1e-4, 5e-5, tiers="all", seed=seed, require_gpu=False, K=4, raw_subset=n,
                       kappa_MT=KAPPA_MT, dwell_time=TAU, equilibrate_binding="fast", accumulation_dtype=precision)
    assert w.work["mt_equilibration"] == {"kind": "fast"}
    from dmipy_sim.engine.physics import walker_keys
    ek = walker_keys(jax.random.fold_in(jax.random.PRNGKey(seed), 0xB0117E), 0, n)
    u = np.asarray(jax.vmap(lambda k: jax.random.uniform(k, (2,), dtype=jnp.float32))(ek))[:, 0]
    rows, _ = _formula(MIXED, OPEN, SEEDED)
    table = mt.equilibrium_domains(geo, KAPPA_MT, TAU)
    for dom, row in zip(table.domains, rows):
        assert np.float32(dom["p_eq"]) == np.float32(row[5])
    enc = np.asarray(jax.vmap(geo.classify_position)(jnp.asarray(np.asarray(w.raw_positions)[:, 0])))
    P = np.full(enc.shape, np.nan)
    for pools, cyl, *_, p in rows:
        P[np.isin(enc, ([0] if "extra" in pools else []) + [k + 1 for k in cyl])] = p
    assert not np.isnan(P).any()
    bound = np.asarray(w.bound_start) == 1
    assert np.array_equal(bound, u < P.astype(np.float32))
    near = np.abs(u.astype(np.float64) - P) <= np.spacing(P.astype(np.float32)).astype(np.float64)
    assert np.array_equal(bound[~near], (u.astype(np.float64) < P)[~near])
    assert near.sum() == 0, f"{near.sum()} draws within a float32 spacing of their P"    # ~4e-9 per walker
    lumen = np.isin(enc, [1, 3])                              # both kinds of domain hold walkers
    assert 0 < lumen.sum() < n


def test_fast_refuses_a_walker_that_starts_in_a_sheath():
    geo = _cell()
    r0 = np.zeros((2, 3), np.float32)
    r0[0, 0] = 0.85 * R                                       # in cylinder 0's sheath (lumen 0.7 R, wall R)
    r0[1, 0] = 0.5 * R
    with pytest.raises(ValueError, match="not in a water domain of the seeded walkers"), \
            pytest.warns(UserWarning, match="do not start at spatial equilibrium"):
        _walk_geometry(2, D, geo, 1e-4, 5e-5, tiers="all", seed=1, require_gpu=False, K=4, r0=r0,
                       kappa_MT=KAPPA_MT, dwell_time=TAU, equilibrate_binding="fast")


# ── the burn-in ─────────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("g", [MIXED, (0.7, 0.7, 0.7, 0.7)])
def test_the_burn_in_of_a_cell_stops_at_its_equilibrium(g):
    geo = _cell(g=g, kappa_outer=OPEN if g == MIXED else (0, 0, 0, 0))
    table = mt.equilibrium_domains(geo, KAPPA_MT, TAU)
    chunk = 1e-3
    crit = mt.burnin_criterion(geo, KAPPA_MT, TAU, D, chunk)
    from dmipy_sim.engine.physics import length_scales_of
    ls = length_scales_of(geo)
    rate = ls.pore_rate if ls.pore_rate is not None else (2.0816 / ls.min_feature) ** 2
    n_min = max(1, int(np.ceil(mt.BURNIN_RELAXATIONS * max(TAU, 1.0 / (D * rate)) / chunk)))
    assert crit.kind == "equilibrium" and crit.p_eq == table.p_eq and crit.p_eq_reason is None
    assert crit.equilibrium_domains == table.domains
    assert (crit.min_chunks, crit.max_chunks) == (n_min, max(mt.BURNIN_MIN_CHUNKS, n_min + 20))


def test_the_walk_records_the_domains_it_equilibrated_to():
    w = _walk_geometry(64, D, _cell(), 1e-4, 5e-5, tiers="all", seed=3, require_gpu=False, K=4,
                       kappa_MT=KAPPA_MT, dwell_time=TAU)
    rec = w.work["mt_equilibration"]
    assert rec["kind"] == "equilibrium" and rec["converged"] and np.isfinite(rec["p_eq"])
    doms = rec["equilibrium_domains"]
    assert doms == list(mt.equilibrium_domains(_cell(), KAPPA_MT, TAU).domains)
    assert _within(sum(x["weight"] for x in doms), 1.0, 4)
    assert rec["p_eq"] == sum(x["weight"] * x["p_eq"] for x in doms)
    assert "p_eq_reason" not in rec


def test_the_walk_of_a_refused_cell_records_the_reason():
    with pytest.warns(UserWarning, match="do not start at spatial equilibrium"):    # a sheath fills behind its face
        w = _walk_geometry(64, D, _cell(kappa_inner=4e-5), 1e-4, 5e-5, tiers="all", seed=3, require_gpu=False, K=4,
                           kappa_MT=KAPPA_MT, dwell_time=TAU)
    rec = w.work["mt_equilibration"]
    assert rec["kind"] == "plateau" and rec["p_eq"] is None and rec["p_eq_reason"] == REACHABLE
    assert "equilibrium_domains" not in rec


def test_another_geometrys_burn_in_record_has_its_keys():
    w = _walk_geometry(64, D, d.Cylinder(radius=R, orientation=(0, 0, 1)), 1e-4, 5e-5, seed=1, raw_subset=64,
                       require_gpu=False, kappa_MT=KAPPA_MT, dwell_time=TAU)
    assert set(w.work["mt_equilibration"]) == {"kind", "p_eq", "tolerance", "window", "min_chunks", "max_chunks",
                                               "chunk_s", "chunks_run", "occupancy", "converged"}


# ── every other geometry is what it was ─────────────────────────────────────────────────────────────────────────────

_R, _L = 1.5e-6, 12e-6
_C2 = np.array([[3e-6, 6e-6], [9e-6, 6e-6]])
_C3 = np.array([[3e-6, 6e-6, 6e-6], [9e-6, 6e-6, 6e-6]])
#: ``(equilibrium_bound_fraction at kappa_MT 4e-5 m/s and dwell 1 ms, surface_to_volume)``, recorded on the engine
#: before the cell had a closed form.
RECORDED = {
    "sphere": (lambda: d.Sphere(radius=_R), 0.07407407407407407, 2000000.0),
    "cylinder": (lambda: d.Cylinder(radius=_R, orientation=(0, 0, 1)), 0.05063291139240506, 1333333.3333333333),
    "cylinder permeable": (lambda: d.Cylinder(radius=_R, orientation=(0, 0, 1), permeability=2e-5), None,
                           1333333.3333333333),
    "packed cylinders sealed": (lambda: PackedCylinders(np.array([_R, 1.2 * _R]), _C2, _L), 0.011215489915906972,
                                None),
    "packed cylinders permeable": (lambda: PackedCylinders(np.array([_R, 1.2 * _R]), _C2, _L, permeability=4e-5),
                                   0.01138799280322023, None),
    "packed cylinders, extra alone": (lambda: PackedCylinders(np.array([_R, 1.2 * _R]), _C2, _L, permeability=4e-5,
                                                              pool="extra"), None, None),
    "packed spheres sealed": (lambda: d.PackedSpheres(np.array([_R, 1.2 * _R]), _C3, _L), 0.003083962002835435, None),
    "packed spheres permeable": (lambda: d.PackedSpheres(np.array([_R, 1.2 * _R]), _C3, _L, permeability=4e-5),
                                 0.0031837836772151493, None),
    "slab": (lambda: PermeableSlab1D(10e-6, 4e-5), 0.007936507936507934, None),
    "slab, sealed membrane": (lambda: PermeableSlab1D(10e-6, 0.0), None, None),
    "free": (lambda: FreeDiffusion(), None, None),
}


@pytest.mark.parametrize("name", list(RECORDED))
def test_every_other_geometry_keeps_its_equilibrium(name):
    build, p_eq, sv = RECORDED[name]
    geo = build()
    assert mt.equilibrium_bound_fraction(geo, 4e-5, 1e-3) == p_eq
    assert mt.surface_to_volume(geo) == sv
    assert mt.burnin_criterion(geo, 4e-5, 1e-3, D, 1e-3)[7:] == (None, None)
