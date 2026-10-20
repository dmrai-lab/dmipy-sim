"""The canonical pores as a reference family, run through the reference-pack protocol (#482, #480).

Three shapes -- a cylinder, a sphere and a pair of parallel plates -- at every diameter from 0.1 to 20 um in
steps of 0.1 um, 600 packs in all, each walked to 100 ms at D = 2e-9 m^2/s and stored as one 100 ms window at
K = 256 bands with the contact and occupancy channels, published to ``SubstrateCommons/canonical-pores``. The
reference is the closed form these shapes have and no tissue substrate does: the eigenmode (matrix) solution of
the Bloch-Torrey equation for a reflecting planar, cylindrical or spherical pore under a generalised gradient
waveform, Codd & Callaghan 1999. Its grade is therefore ``analytic`` by the rule of #459.

**The declaration carries no list of 600 packs.** The family is its grid: :func:`grid_rows` gives every substrate,
pack id and dataset path from the three shapes and the diameters, and the walk stage walks each pack through the
producer at the design's band (``K`` bands over the 100 ms window, accumulated in the band's own blocks; no path
is kept), at its own seed.

::

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \\
      nice -n 10 python examples/substrate_bank/build_canonical_pores.py --work <the family dir> \\
      --stage source        # then reference, spec, design, walk, pack, gate, card, publish
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from dmipy_sim.replay.reference import (Build, Design, Direct, FreeParameter, Published, Reference, ReferenceFamily,
                                        pack_seed, ReferenceQuantity, Publication, Source, SourceFile, Systematic,
                                        Tier, Tolerance)

REPO = "SubstrateCommons/canonical-pores"
LICENCE = "CC-BY-4.0"
CITATION = "Fick RHJ (2026), the canonical pores family of replay packs, SubstrateCommons"

D0 = 2e-9           #: m^2/s, the walk's diffusivity, the same for all 600
T = 0.1             #: s, the storage window: one window per pack
K_BANDS = 256       #: bands per 100 ms window
SIGMA = 5e-3        #: the floor every pack of this family was built to (`build_to_floor`'s sigma_star)
BUDGET_BYTES = 60_000_000_000
PILOT_N = 4000
FALSE_FAILURE_RATE = 0.01

#: The closed form. Codd & Callaghan extend Callaghan's matrix formalism to exactly the three pores this family
#: is made of, with wall relaxivity -- the observable of the C2 tier -- so ONE document covers every pack.
REFERENCE_DOI = "10.1006/jmre.1998.1679"
REFERENCE_TITLE = ("Spin Echo Analysis of Restricted Diffusion under Generalized Gradient Waveforms: Planar, "
                   "Cylindrical, and Spherical Pores with Wall Relaxivity")
PUBMED = "10089170"
#: The line of that document which states the closed form, quoted from the abstract as PubMed distributes it.
VERBATIM = ("We have extended the method to the treatment of restricted diffusion in parallel plate, "
            "cylindrical, and spherical geometries, including the effects of fluid-surface interactions.")

#: The reference's OWN uncertainty: it is a series, and truncating it is the only error it has. Measured on this
#: family's grid by refining the mode counts (`ANALYTIC_MODES` against `ANALYTIC_MODES_FINE`), so the
#: `published-` comparison can fail -- a closed form that states no uncertainty cannot be failed on at all.
ANALYTIC_MODES = (24, 18)
ANALYTIC_MODES_FINE = (30, 26)

#: The one measurement of the battery the quantity is read at, and the sampling the direct walk is measured on.
#: `delta` 10 ms / `Delta` 40 ms / b = 1e9 s/m^2 is in the battery every pack was certified over; b = 1e9 rather
#: than the battery's 3e9 because over a 100 ms window the high-b end of a 20 um pore is a nearly dephased
#: signal whose modulus and real part part company (dmipy-sim#484), and that disagreement is not this family's.
QUANTITY_GRID = dict(delta_s=10e-3, Delta_s=40e-3, b_s_per_m2=1e9, direction=[1, 0, 0], n_t=800,
                     solver=f"the eigenmode (matrix) solution of Codd & Callaghan 1999 at "
                            f"{ANALYTIC_MODES} modes, on a piecewise-constant PGSE")

#: The substrates the reference records a QUANTITY for: both a closed form and a direct walk of our own, at the
#: two diameters whose R/6 sub-step rule a CPU walk affords. The other 594 packs are gated on everything the
#: gate can read about them -- their certificate against the design's targets, their served-vs-decoded
#: difference, their band, their spec's round trip -- and the card says which carry a comparison.
QUANTITY_DIAMETERS_UM = (10.0, 20.0)
DIRECT_N = 20_000

#: The family's grid: three shapes at every diameter from 0.1 to 20 um in steps of 0.1 um.
SHAPES = ("cylinder", "plane", "sphere")
DIAMETERS_UM = tuple(round(0.1 * i, 1) for i in range(1, 201))


def pack_id(shape, d_um):
    """The pack's id, ``canonical-pores/<shape>/d<NN.NN>um``: the name the manifest lists it under, and the one
    thing its walk seed (``pack_seed``) is derived from."""
    return f"canonical-pores/{shape}/d{d_um:05.2f}um"


def substrate_name(shape, d_um):
    return f"{shape}-d{d_um:05.2f}um"


def pack_path(shape, d_um):
    """The pack's path in the dataset, ``packs/<shape>/d<NN.NN>um.rpk``."""
    return f"packs/{shape}/d{d_um:05.2f}um.rpk"


def grid_rows(shapes=SHAPES, diameters_um=DIAMETERS_UM):
    """Every pack of this family as a row ``{id, path, shape, d_um}`` by substrate name: the grid IS the
    declaration, so the substrate list, the specs, the pack ids and the dataset paths all follow from it."""
    return {substrate_name(shape, d): dict(id=pack_id(shape, d), path=pack_path(shape, d), shape=shape, d_um=float(d))
            for shape in shapes for d in diameters_um}


# ----------------------------------------------------------------- the spec producer
def geometry(shape, d_um):
    """The pore, in its own frame: the cylinder along +z, the plates' normal along x."""
    import dmipy_sim as d
    r = 0.5 * float(d_um) * 1e-6
    if shape == "cylinder":
        return d.Cylinder(radius=r, orientation=(0.0, 0.0, 1.0))
    if shape == "sphere":
        return d.Sphere(radius=r)
    if shape == "plane":
        return d.Box1D(length=2.0 * r)
    raise ValueError(f"canonical-pores has three shapes, not {shape!r}")


def spec_of(shape, d_um):
    """The pore's substrate spec. An analytic geometry writes its own situation, so this family cites no released
    file and the spec stage's digest check has nothing to check -- which it says."""
    return geometry(shape, d_um).spec


def walk(spec, n, *, T, K, seed, sub_steps=None):
    """The walk the pilot runs and every pack is built from: no relaxivity, every tier stored, ``K`` bands over the
    window ``T`` in the band's own blocks, at the pack's own ``seed`` (``pack_seed`` of its id); ``sub_steps`` is the
    step pilot's coarse count, None the rule's. A :class:`~dmipy_sim.coefficient_walk.CoefficientWalk`."""
    import dmipy_sim as d
    from dmipy_sim.spec import geometry_from_spec
    return d.simulate_trajectories(int(n), D0, geometry_from_spec(spec), T_max=float(T), K=int(K), seed=int(seed),
                                   tiers="all", require_gpu=False, walker_batch_size=2000, sub_steps=sub_steps,
                                   window_T=float(T))


# ----------------------------------------------------------------- the envelope and the declared waveforms
#: What these packs say they are FOR: the PGSE / OGSE / short-delta battery the family's floor was measured over.
ENVELOPE = dict(bvals=[0.0, 0.5e9, 1e9, 2e9, 3e9], dirs=[[0, 0, 1], [1, 0, 0], [1, 0, 1]],
                ogse_periods=[1, 2, 3, 5], shortd_b=1e9, shortd_deltas_frac=[0.2, 0.1, 0.05, 0.025],
                delta_frac=0.2, Delta_frac=0.5)

#: The (delta, Delta) pairs the family DECLARES, in seconds. Every one fits inside the 100 ms window: the
#: battery's own (10, 100) ms pair does not -- its TE is 110 ms -- and is refused rather than truncated.
DECLARED_PAIRS = ((10e-3, 20e-3), (10e-3, 40e-3), (10e-3, 60e-3), (5e-3, 40e-3))


def sequence(delta, Delta, *, b=1e9, direction=(1, 0, 0), n_t=800):
    from dmipy_sim import pgse
    return pgse([list(direction)], float(delta), float(Delta), bvalues=[float(b)], n_t=int(n_t),
                slew_rate=np.inf)


def waveforms():
    """Every waveform the family declares, checked against ``waveform_band`` per pack by the pack stage."""
    return tuple((f"pgse-d{delta * 1e3:g}ms-D{Delta * 1e3:g}ms", sequence(delta, Delta))
                 for delta, Delta in DECLARED_PAIRS)


# ----------------------------------------------------------------- the quantity
def _modulus_and_se(e):
    """``(|<e>|, relative standard error of it)`` for per-walker complex contributions ``e``.

    The modulus is what a pack SERVES (dmipy-sim#297), so its standard error is the one of a derived statistic:
    the ensemble mean's error projected on the direction the modulus is taken along, by the delta method. No
    fold, no split half, no seed.
    """
    e = np.asarray(e, np.complex128)
    m = e.mean()
    if not abs(m) > 0.0:
        raise ValueError("the ensemble mean is zero, so the modulus has no gradient to project on")
    proj = (e * np.conj(m / abs(m))).real
    return float(abs(m)), float(np.std(proj, ddof=1) / np.sqrt(len(e)) / abs(m))


def analytic_signal(shape, d_um, grid, *, modes=ANALYTIC_MODES):
    """The closed form's signal at ``grid``: the eigenmode (matrix) solution of Codd & Callaghan 1999.

    Evaluated by :mod:`dmipy_sim.math.matrix_method`, the package's own implementation of the document's
    formalism -- the reference is the document, not our code, and the code's own error is the series
    truncation the reference record carries as the reference's uncertainty.
    """
    from dmipy_sim.math.matrix_method import MatrixPore
    seq = sequence(grid["delta_s"], grid["Delta_s"], b=grid["b_s_per_m2"], direction=grid["direction"],
                   n_t=grid["n_t"])
    g = np.asarray(seq.G_eff, np.float64)[0]
    axis = int(np.argmax(np.abs(np.asarray(grid["direction"], float))))
    # the plane's eigenfunctions carry one index: its count is the pair's product, the count every record of
    # this family was computed at when the solver took one pair for every shape
    n_modes = int(modes[0]) * int(modes[1]) if shape == "plane" else tuple(int(x) for x in modes)
    return float(abs(MatrixPore(shape, float(d_um) * 1e-6, D0, n_modes=n_modes).signal(g[:, axis], float(seq.dt))))


def direct_walk_signal(shape, d_um, grid, *, n=DIRECT_N):
    """OUR direct measurement of the same quantity: the fused engine's walk of the pore with the gradient applied
    in the walk (``dmipy_sim.engine.core.simulate(engine="fused")``, the oracle the replay is held to), each
    walker's own phase, and the analytic standard error over its walkers. No band and no codec stand between the
    walk and the number, so what the gate's ``reproduces-`` check compares is the pack's bands and the walker
    count -- and the ``published-`` check then compares both of them with the closed form."""
    from dmipy_sim.engine.core import simulate
    seq = sequence(grid["delta_s"], grid["Delta_s"], b=grid["b_s_per_m2"], direction=grid["direction"],
                   n_t=grid["n_t"])
    _S, _log_w, phi = simulate_fused(int(n), D0, waveform=seq, geometry=geometry(shape, d_um),
                               seed=pack_seed(pack_id(shape, d_um)), return_walker_signals=True,
                               require_gpu=False)
    return _modulus_and_se(np.exp(1j * np.asarray(phi, np.float64)[:, 0]))


def reproduce(pack, quantity, grid):
    """The signal this pack SERVES at the reference record's own grid, with the analytic standard error of the
    same estimator the direct measurement used."""
    seq = sequence(grid["delta_s"], grid["Delta_s"], b=grid["b_s_per_m2"], direction=grid["direction"],
                   n_t=grid["n_t"])
    phi = np.asarray(pack.walker_primitives(seq).phi, np.float64).ravel()
    value, se = _modulus_and_se(np.exp(1j * phi))
    return dict(value=value, se=se, se_kind="delta_method", solver=grid["solver"],
                se_derivation="the delta method on the modulus of the weighted ensemble mean: each walker's "
                              "own contribution projected on the direction the modulus is taken along, sd over "
                              "walkers / sqrt(N) / |S|",
                n_walkers=int(pack.n_walkers))


def served_vs_channel(pack):
    """How far the signal the pack SERVES (``ReplayPack.replay``, the consumer's route) is from the one its stored
    bands hold (``walker_primitives``: the bands contracted once per walker, the modulus of their mean), over every
    declared waveform. A pore of this family has one pool at water fraction 1, so its walkers weigh alike and the
    mean is the plain one. A family whose pools weigh differently reads ``walker_signals`` instead."""
    out = 0.0
    for _label, seq in waveforms():
        served = np.abs(np.asarray(pack.replay(seq), np.float64)).ravel()
        phi = np.asarray(pack.walker_primitives(seq).phi, np.float64)
        held = np.abs(np.exp(1j * phi).mean(axis=0)).ravel()
        out = max(out, float(np.max(np.abs(served - held))))
    return out


# ----------------------------------------------------------------- the "Use me" snippet
def snippet(uri):
    """One call that reproduces one signal of the reproduction table, at its own grid."""
    g = QUANTITY_GRID
    return f'''import numpy as np
from dmipy_sim import pgse
from dmipy_sim.replay import ReplayPack
from dmipy_sim.engine._fused import simulate_fused

pk = ReplayPack.load("{uri}")
seq = pgse([{g["direction"]!r}], {g["delta_s"]!r}, {g["Delta_s"]!r}, bvalues=[{g["b_s_per_m2"]!r}],
           n_t={g["n_t"]!r}, slew_rate=np.inf)
print("S = %.6f" % float(np.abs(np.asarray(pk.replay(seq))).ravel()[0]))
'''


# ----------------------------------------------------------------- the family
def licence_text(cache_dir):
    """The CC BY 4.0 legal code, fetched once and cached: the licence THIS family's packs carry, and the one
    dmipy-sim itself is the source under."""
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "cc-by-4.0-legalcode.txt")
    if not os.path.exists(cache):
        req = urllib.request.Request("https://creativecommons.org/licenses/by/4.0/legalcode.txt",
                                     headers={"User-Agent": "dmipy-sim (https://github.com/dmrai-lab/dmipy-sim)"})
        with urllib.request.urlopen(req, timeout=60) as fh, open(cache, "wb") as out:
            out.write(fh.read())
    return open(cache, encoding="utf-8").read()


def measured_quantities(work_dir, rows, grid):
    """The reference's quantities, every number in them MEASURED here at record time and cached on the inputs
    it was measured from: the closed form at two mode counts (its own truncation is the reference's
    uncertainty) and our direct walk with its delta-method standard error."""
    cache_path = os.path.join(work_dir, ".quantity-cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    out = []
    for shape in ("cylinder", "plane", "sphere"):
        for d_um in QUANTITY_DIAMETERS_UM:
            name = substrate_name(shape, d_um)
            if name not in rows:                       # a family of a sub-grid records the quantities it holds
                continue
            key = hashlib.sha256(json.dumps([shape, d_um, grid, ANALYTIC_MODES, ANALYTIC_MODES_FINE, DIRECT_N,
                                             "fused", pack_seed(pack_id(shape, d_um))], sort_keys=True).encode()).hexdigest()
            if cache.get(name, {}).get("key") != key:
                coarse = analytic_signal(shape, d_um, grid, modes=ANALYTIC_MODES)
                fine = analytic_signal(shape, d_um, grid, modes=ANALYTIC_MODES_FINE)
                value, se = direct_walk_signal(shape, d_um, grid)
                cache[name] = dict(key=key, value=dict(
                    analytic=coarse, analytic_fine=fine,
                    truncation=abs(fine - coarse) / abs(coarse), direct=value, direct_se=se))
                with open(cache_path, "w") as fh:
                    json.dump(cache, fh, indent=1, sort_keys=True)
            v = cache[name]["value"]
            out.append(ReferenceQuantity(
                substrate=name, name="signal_at_b1000",
                published=Published(
                    value=v["analytic"], unit="-",
                    uncertainty=max(v["truncation"], 1e-12),
                    uncertainty_is=(f"the series' OWN truncation, measured here: {ANALYTIC_MODES} modes against "
                                    f"{ANALYTIC_MODES_FINE} move the signal by {v['truncation']:.3g} of it. A "
                                    f"closed form's only error is where it is cut off, and stating it is what "
                                    f"lets this comparison fail"),
                    printed_in=f"Codd SL & Callaghan PT (1999), {REFERENCE_TITLE}, J. Magn. Reson. 137:358-372",
                    document=REFERENCE_DOI,
                    locator="the abstract's statement of the formalism (p. 358); the planar, cylindrical and "
                            "spherical matrix solutions of §2, evaluated on this family's own waveform",
                    verbatim=VERBATIM),
                direct=Direct(
                    value=v["direct"], unit="-", se=v["direct_se"], se_kind="delta_method",
                    se_derivation=("the delta method on the modulus of the ensemble mean over the walk's own "
                                   "walkers: each walker's contribution projected on the modulus' direction, "
                                   "sd / sqrt(N) / |S|. No fold, no split half, no seed spread"),
                    n_walkers=DIRECT_N, grid=grid, solver=grid["solver"],
                    source=(f"examples/substrate_bank/build_canonical_pores.py (direct_walk_signal): a "
                            f"{DIRECT_N:,}-walker walk of the same pore at the same D by the fused engine, the "
                            f"gradient applied in the walk, at the pack's own seed "
                            f"({pack_seed(pack_id(shape, d_um))}), with no band and no codec between the walk "
                            f"and the number"))))
    return tuple(out)


def family(work_dir, *, dry, create_dataset, rows=None):
    rows = rows if rows is not None else grid_rows()
    grid = dict(QUANTITY_GRID)
    text = licence_text(os.path.join(work_dir, ".licence-cache"))
    import dmipy_sim
    from dmipy_sim.replay.reference import code_commit
    sources = [Source(
        key="dmipy-sim", url="https://github.com/dmrai-lab/dmipy-sim",
        host_record=f"dmipy-sim at {code_commit(require_clean=False)[:8]}, the commit the packs are walked at",
        licence_id=LICENCE, licence_url="https://creativecommons.org/licenses/by/4.0/", licence_text=text,
        redistributes_bytes=False,
        files=(SourceFile(path=os.path.join(os.path.dirname(os.path.abspath(dmipy_sim.__file__)),
                                            "geometry", "analytic.py"),
                          cite_as="dmipy_sim/geometry/analytic.py",
                          role="the three pores themselves: Cylinder, Sphere and Box1D, whose wall interaction "
                               "and seeding ARE the released substrate -- this family has no released image or "
                               "mesh, its substrate is a radius and a shape"),
               SourceFile(path=os.path.join(os.path.dirname(os.path.abspath(dmipy_sim.__file__)),
                                            "geometry", "_boundary.py"),
                          cite_as="dmipy_sim/geometry/_boundary.py",
                          role="the one implementation of each boundary rule the three pores are walked with")))]

    reference = Reference(
        doi=REFERENCE_DOI, title=REFERENCE_TITLE, published_kind="analytic",
        sample=("the same material: a reflecting pore of one radius filled with a single fluid, which is what "
                "both the closed form and the walk are of -- there is no object, so the geometry cannot differ"),
        sample_relation="the same material",
        quantities=measured_quantities(work_dir, rows, grid),
        parameters=(
            FreeParameter(name="D0", value=D0, unit="m^2/s", whose="ours",
                          where="this family's own declaration", how="chosen once for all 600 packs: the "
                                "intra-axonal water value of Dhital 2019 / Veraart 2020, which is also the "
                                "test suite's. The closed form takes it as a parameter, so it is not a free "
                                "parameter OF the comparison -- both sides use this number"),
            FreeParameter(name="diameter", value="0.1 to 20.0 um in 0.1 um steps", unit="um", whose="ours",
                          where="the sweep the dataset is", how="the family's own grid; the closed form is "
                                "evaluated at each of them, so no interpolation enters the comparison"),
            FreeParameter(name="n_modes", value=str(ANALYTIC_MODES), unit="-", whose="ours",
                          where="dmipy_sim/math/matrix_method.py", how="the eigenmode truncation of the "
                                "reference's own series. Refining it is what measures the reference's "
                                "uncertainty, which is why it is recorded rather than assumed converged"),
            FreeParameter(name="surface relaxivity", value=0.0, unit="m/s", whose="ours",
                          where="this family's walk", how="none in the walk: the packs store the boundary "
                                "local time as a channel and rho2 is a replay knob, so the closed form's "
                                "wall-relaxivity case is a sweep over one pack rather than 600 more walks"),
            FreeParameter(name="block", value="the band's own, h = 0.8 T / (pi K): 99.5 us at K = 256 over 100 ms",
                          unit="s", whose="ours", where="dmipy_sim.engine.accumulate.block_length",
                          how="the walk keeps the bands, not a path: each pack accumulates its K bands in blocks "
                              "the band sets, the same for all 600, stated in every pack's walk_params.block"),
            FreeParameter(name="walk seed", value="one per pack, from its id", unit="-", whose="ours",
                          where="dmipy_sim.replay.reference.pack_seed",
                          how="the first four bytes of the SHA-256 of the pack's id, so no two packs share a "
                              "random stream whatever their walker counts; the seed is in each pack's "
                              "provenance (walk_params.seed)")),
        description=("Cylinders, spheres and parallel plates from 0.1 to 20 um, walked once each and stored so "
                     "that any waveform, any surface relaxivity and any per-pool relaxation can be replayed on "
                     "them. The three shapes whose restricted diffusion has a closed form, which is what makes "
                     "this family the ruler the others are read against."),
        source_note=("dmipy-sim itself: the substrate is a shape and a radius, so there is no released image or "
                     "mesh to cite. The reference is Codd & Callaghan 1999 (J. Magn. Reson. 137:358-372), the "
                     "matrix solution for exactly these three pores, after Callaghan 1997 and Caprihan 1996."),
        licence_note="CC-BY-4.0 packs; the substrate is dmipy-sim's own (CC-BY-4.0), and the reference paper's "
                     "numbers are computed from its formalism, never redistributed",
        caveats=dict(
            seed=("Every pack of this family is walked at one seed of its own, derived from its id "
                  "(dmipy_sim.replay.reference.pack_seed), so the residuals of the packs against the reference "
                  "are independent draws. The packs of the release published before this rule were all walked "
                  "at seed 0, and packs of equal walker count therefore shared their walkers' random streams: "
                  "their residuals against the reference are correlated across packs, a pattern over several "
                  "neighbouring diameters is one draw read many times, and the largest residual of the family "
                  "is not by itself evidence of a bias."),
            envelope=("The design's envelope is the default battery whole. Each pack was BUILT with that "
                      "battery's gradient directions restricted to the axes its pore confines -- a cylinder "
                      "across its section, the plates along their normal -- because a free axis' Monte-Carlo "
                      "floor would have sized every cylinder at the walker cap for a signal this family does "
                      "not certify. The floors on the card are therefore over the restricted battery, and the "
                      "declared waveforms are the ones the gate checks against each pack's own band."),
            quantities=(f"A closed form exists for all 600 packs, and the reference record carries a comparison "
                        f"for {3 * len(QUANTITY_DIAMETERS_UM)} of them: the three shapes at "
                        f"{', '.join('%g um' % d for d in QUANTITY_DIAMETERS_UM)}, where a direct walk of our "
                        f"own is affordable on the CPU this ran on (the R/6 sub-step rule is 1/R^2, so a "
                        f"0.1 um pore's direct walk costs four orders more than a 20 um one's). Every other "
                        f"pack is gated on its certificate, its served-vs-decoded difference, its band and its "
                        f"spec's round trip, and the card's reproduction table shows which carry a number."),
            window=("100 ms at K = 256 is a band of 1280 Hz, the same for all 600. The battery's (delta, "
                    "Delta) = (10, 100) ms pair does not fit in the window -- its TE is 110 ms -- so it is not "
                    "a declared waveform, and a consumer asking for it is refused by the pack's own length "
                    "rather than served a truncated answer.")))

    pilot = substrate_name("sphere", 20.0)
    design = Design(
        window_s=T, K=K_BANDS, envelope=ENVELOPE, waveforms=waveforms,
        tiers=(Tier(name="positions", floor_key="floor_max", err_key="err_max", target_floor=SIGMA),
               Tier(name="contact", floor_key="floor_surface", err_key="err_surface", target_floor=SIGMA)),
        memory_budget_bytes=BUDGET_BYTES, pilot_n=PILOT_N, safety=1.4, pilot_substrate=pilot,
        false_failure_rate=FALSE_FAILURE_RATE,
        tolerance=Tolerance(terms=("quantity.direct.se", "quantity.replay_se",
                                   "quantity.published.uncertainty", "pack.tiers.positions.err")),
        systematics=(Systematic(
            name="route_of_the_direct_walk", value=0.0,
            measured_on="not measured: the term is the pack's own certified error instead",
            evidence="the direct walk is the fused engine with the gradient applied at its own physics step and the "
                     "pack replays K = 256 stored bands, and the difference the two routes make -- the bands' "
                     "truncation, the block table's quadrature and the container's quantisation -- is what the pack "
                     "itself certifies over the battery, which the tolerance already carries as "
                     "pack.tiers.positions.err. A second term for the same effect would double-count it."),))

    build = Build(
        specs={name: (lambda r=r: spec_of(r["shape"], r["d_um"])) for name, r in rows.items()},
        pack_id={name: r["id"] for name, r in rows.items()},
        reproduce=reproduce, served_vs_channel=served_vs_channel, served_tier="positions", walk=walk)

    publication = Publication(repo=REPO, licence=LICENCE, citation=CITATION, snippet=snippet,
                              snippet_substrate=substrate_name("sphere", 10.0),
                              pack_path=lambda name, rows=rows: rows[name]["path"],
                              create_dataset=create_dataset, dry=dry)
    return ReferenceFamily("canonical-pores", work_dir, sources=sources, reference=reference, design=design,
                           build=build, publication=publication)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work", default=os.path.join(os.path.expanduser("~"), "dmrai-ws", "canonical-pores"),
                   help="the family directory (records/, previews/, README.md)")
    p.add_argument("--stage", default=None, help="run one stage (the protocol's order still holds)")
    p.add_argument("--dry", action="store_true", help="gate and render, upload nothing")
    p.add_argument("--create-dataset", action="store_true", help="create the dataset repository (explicit)")
    p.add_argument("--diameters", type=float, nargs="+", default=None,
                   help="a sub-grid of diameters (um), every shape at each: a family of those packs only")
    a = p.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("dmipy_sim").setLevel(logging.INFO)
    rows = grid_rows(diameters_um=tuple(a.diameters)) if a.diameters else None
    fam = family(os.path.abspath(a.work), dry=a.dry, create_dataset=a.create_dataset, rows=rows)
    fam.stage(a.stage) if a.stage else fam.run()
    print(f"records in {fam.records.dir}")


if __name__ == "__main__":
    main()
