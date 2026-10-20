"""The cross-engine parity fixtures as a reference family, run through the reference-pack protocol (#482, #480).

Four packs of two other simulators' own released substrates, each walked here on THEIR surface at THEIR
acquisition and published to ``SubstrateCommons/parity-fixtures``:

* three **MC/DC** undulating axons (Rafael-Patino et al. 2020, Front. Neuroinform. 14:8, LGPL-2.1), walked at
  their diffusivity and seeded from their own released initial-walker list, against their released
  ``*_DWI.bfloat`` over the ActiveAx protocol;
* the **Disimpy** closed cylinder mesh (Kerkelae et al. 2020, JOSS 5(52):2527, MIT), against the MISST
  reference signal distributed with it.

The fixtures, the specs, the walks, the references and the estimators are
``examples/validation/cross_engine_parity.py``; this file is the family's DECLARATION and nothing else. Every pack
is walked through the producer, ``K_BANDS`` bands over the fixture's own TE in the band's own blocks, and our
direct number is the fused engine's walk of the same surface at the same seeding.

::

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \\
      nice -n 10 python examples/substrate_bank/build_parity_fixtures.py --mcdc-data DIR --disimpy-data DIR \\
      --work <the family dir> --stage source      # then reference, spec, design, walk, pack, gate, card
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from dmipy_sim.replay.reference import (Build, Design, Direct, FreeParameter, Published, Reference,
                                        ReferenceFamily, ReferenceQuantity, Publication, Source, SourceFile,
                                        Systematic, Tier, Tolerance)

REPO = "SubstrateCommons/parity-fixtures"
SIGMA = 3e-3                #: the floor each pack was sized to
K_BANDS = 64                #: bands over the fixture's TE (cross_engine_parity.K_PARITY)
DIRECT_N = 100_000          #: walkers of the fused engine's direct walk
BUDGET_BYTES = 60_000_000_000
PILOT_N = 4000
FALSE_FAILURE_RATE = 0.01

#: The fixture names, the pack ids, and which released file each one's number is READ from. A released array IS
#: the measurement, so ``published_kind`` is ``data`` and each quantity records that file's URL and sha256.
FIXTURES = ("mcdc-0.2-32.0", "mcdc-1.0-12.0", "mcdc-2.6-4.0", "disimpy-cylinder")
#: The reservation the Disimpy pack ships WITH, written into its manifest row beside the verdict. It is not a
#: verdict: the gate passes the pack, and the row says so. The numbers here are the ones dmipy-sim#488 states.
HOLD_488 = ("dmipy-sim#488: this pack's MISST comparison is open. The gate compares the ONE measurement this "
            "family declares -- the last of the fixture's protocol, its highest b -- and there the pack agrees "
            "with the MISST reference to 1.8e-5, so it passes. #488's evidence is a VECTOR over the protocol's "
            "100 measurements, which no scalar quantity this protocol can express will see (dmipy-sim#493). Read "
            "the pack as a faithful record of a walk whose reference is in question.")

MCDC_URL = "https://github.com/jonhrafe/Robust-Monte-Carlo-Simulations"
DISIMPY_URL = "https://github.com/kerkelae/disimpy"


def fixtures_module():
    """``examples.validation.cross_engine_parity``: the one place this family's physics lives."""
    import importlib
    return importlib.import_module("examples.validation.cross_engine_parity")


def is_mcdc(name):
    return name.startswith("mcdc-")


def mcdc_params(name):
    """``(amp, wL)`` of an ``mcdc-<amp>-<wL>`` fixture name."""
    _tag, amp, wL = name.split("-")
    return float(amp), float(wL)


# ----------------------------------------------------------------- the spec producer and the walk
def spec_of(name, mcdc_data, disimpy_data):
    """The fixture's substrate spec, citing its surface by a path the source record digested.

    ``cross_engine_parity.disimpy_spec`` cites the repaired surface by BASENAME, which is what a dataset
    distributes it as; the source stage digests a file, so this family cites the resolved path of the same
    bytes -- and the round trip then rebuilds the same walls, which a basename cannot.
    """
    import dataclasses
    X = fixtures_module()
    if is_mcdc(name):
        spec = X.mcdc_spec(mcdc_data, *mcdc_params(name))
        path = X.mcdc_paths(mcdc_data, *mcdc_params(name))["ply"]
    else:
        from dmipy_sim.spec import build
        spec = X.disimpy_spec(disimpy_data)
        path = build.resolve_surface_file(spec.walls[0].surface.file)
    w = spec.walls[0]
    surface = dataclasses.replace(w.surface, file=path)
    return dataclasses.replace(spec, walls=[dataclasses.replace(w, surface=surface)])


def sequence_of(name, mcdc_data):
    X = fixtures_module()
    if is_mcdc(name):
        from dmipy_sim.io import mcdc
        return mcdc.read_scheme(os.path.join(mcdc_data, "Simulator-Conf-files", X.MCDC_SCHEME), n_t=X.MCDC_N_T)
    return X.disimpy_sequence()


def envelope_of(name, mcdc_data=None):
    """The envelope the pack is built to, with its b bound taken from the fixture's OWN scheme.

    ``mcdc_envelope`` states the four ActiveAx shells to four figures (1.319e10); ``read_scheme`` computes b from
    the scheme's own columns and lands a hair above it, so the declared waveform read as outside the envelope it
    IS. The envelope and the declared waveforms are one declaration, so the bound comes from the one sequence
    both of them are.
    """
    X = fixtures_module()
    env = X.mcdc_envelope() if is_mcdc(name) else X.disimpy_envelope()
    if mcdc_data is not None:
        b = float(max(sequence_of(name, mcdc_data).b()))
        env = dict(env, bvals=sorted(set(list(env["bvals"]) + [b])))
    return env


def name_of_spec(spec):
    """The fixture a spec is of, from its id: ``mcdc-robust/uAxon_d_1.0_amp_<amp>_wL_<wL>`` or the Disimpy mesh."""
    sid = str(spec.id)
    if sid.startswith("disimpy/"):
        return "disimpy-cylinder"
    stem = sid.split("/")[-1]
    return f"mcdc-{stem.split('amp_')[1].split('_')[0]}-{stem.split('wL_')[1].split('_')[0]}"


def walk_of(mcdc_data, disimpy_data):
    """The walk the pilot and the walk stage run: the fixture's own acquisition window, diffusivity and seeding, on
    the spec it is HANDED (the fixture read off the spec's id), ``K`` bands over the fixture's TE.

    The spec is the design stage's own, whose surface is cited by the resolved path the source record digested;
    rebuilding it from the producer here would walk a spec cited by basename and resolve it somewhere else. The
    window is the fixture's own TE: the design's ``T`` is its pilot fixture's, and a Disimpy pack's 70 ms is not an
    MC/DC pack's 53.52 ms.

    A fixture is a reproducibility test against a reference engine, so it walks at the fixture's own fixed
    ``X.SEED`` and the per-pack ``seed`` the framework offers is not used.
    """
    X = fixtures_module()

    def walk(spec, n, *, T, K, seed, sub_steps=None):
        import dmipy_sim as d
        from dmipy_sim.spec import geometry_from_spec
        geom = geometry_from_spec(spec)
        name = name_of_spec(spec)
        if is_mcdc(name):
            from dmipy_sim.io import mcdc
            p = X.mcdc_paths(mcdc_data, *mcdc_params(name))
            r0 = mcdc.seed_positions(mcdc.read_ini_walkers(p["ini"]), int(n))
            return d.simulate_trajectories(int(n), X.MCDC_D, geom, T_max=X.MCDC_TE, K=int(K), seed=X.SEED,
                                           tiers="all", r0=r0, require_gpu=False, walker_batch_size=2000,
                                           sub_steps=sub_steps, window_T=X.MCDC_TE)
        return d.simulate_trajectories(int(n), X.DISIMPY_D, geom, T_max=X.DISIMPY_TE, K=int(K), seed=X.SEED,
                                       tiers="all", require_gpu=False, walker_batch_size=2000, sub_steps=sub_steps,
                                       window_T=X.DISIMPY_TE)
    return walk


# ----------------------------------------------------------------- the quantity
#: The measurement each fixture's quantity is read at, DECLARED rather than chosen from the data: the last
#: measurement of the fixture's own protocol, which for both engines is the highest-b one and therefore the most
#: restriction-sensitive. Picking the measurement where the difference happens to be largest would be choosing
#: the comparison after seeing it.
def quantity_grid(name):
    X = fixtures_module()
    return dict(measurement="the last measurement of the fixture's own released protocol (its highest b)",
                index=-1, n_t=(X.MCDC_N_T if is_mcdc(name) else X.DISIMPY_N_T),
                solver=("the ensemble modulus |mean exp(i phi)| over the pack's walkers, the reduction "
                        "ReplayPack.replay returns (cross_engine_parity.ensemble_modulus)"))


def published_value(name, grid, mcdc_data, disimpy_data):
    """Their number at ``grid``, read from the released array itself."""
    X = fixtures_module()
    if is_mcdc(name):
        S, N = X.mcdc_reference(mcdc_data, *mcdc_params(name), sequence_of(name, mcdc_data))
        return float(np.asarray(S).ravel()[grid["index"]]), int(N)
    return float(np.asarray(X.disimpy_reference(disimpy_data)).ravel()[grid["index"]]), None


def _mean_and_se(z):
    """``(|mean z|, relative analytic standard error of it)`` over walkers, by the delta method on the modulus."""
    m = z.mean()
    proj = (z * np.conj(m / abs(m))).real
    return float(abs(m)), float(np.std(proj, ddof=1) / np.sqrt(len(z)) / abs(m))


def direct_at(name, grid, work_dir, mcdc_data, disimpy_data, *, n=DIRECT_N):
    """OUR number at ``grid``: the fused engine's walk of the fixture's surface at its diffusivity and seeding, the
    gradient applied in the walk (``dmipy_sim.engine.core.simulate(engine="fused")``, the oracle a replay is held
    to), with its analytic standard error and, for an MC/DC fixture, the per-walker spread at THEIR walker count
    (``cross_engine_parity.floors``: their released file is one realisation and states no spread).

    Measured once and cached on the fixture, the grid and the walker count, so every stage of a run does not walk
    it again. The measurement is the one the record DECLARES, not the one where a difference happens to be largest.
    """
    import hashlib
    from dmipy_sim.engine.core import simulate
    from dmipy_sim.spec import geometry_from_spec
    X = fixtures_module()
    cache_path = os.path.join(work_dir, ".direct-cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    key = hashlib.sha256(json.dumps([name, grid, int(n), X.SEED, "fused"], sort_keys=True).encode()).hexdigest()
    if cache.get(name, {}).get("key") != key:
        seq = sequence_of(name, mcdc_data)
        geom = geometry_from_spec(spec_of(name, mcdc_data, disimpy_data))
        r0, D, n_theirs = None, X.DISIMPY_D, None
        if is_mcdc(name):
            from dmipy_sim.io import mcdc
            r0 = mcdc.seed_positions(mcdc.read_ini_walkers(X.mcdc_paths(mcdc_data, *mcdc_params(name))["ini"]), int(n))
            D = X.MCDC_D
            _S, n_theirs = X.mcdc_reference(mcdc_data, *mcdc_params(name), seq)
        _sig, _log_w, phi = simulate(int(n), D, waveform=seq, geometry=geom, seed=X.SEED, r0=r0,
                                     return_walker_signals=True, engine="fused", require_gpu=False)
        phi = np.asarray(phi, np.float64)[:, grid["index"]]
        value, se = _mean_and_se(np.exp(1j * phi))
        _ours, theirs, _dof, _exact = X.floors(np.cos(phi)[None, :], n_theirs)
        cache[name] = dict(key=key, value=dict(value=value, se=se, n_walkers=int(n),
                                               floor_theirs=float(theirs[0]), n_theirs=n_theirs))
        os.makedirs(work_dir, exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump(cache, fh, indent=1, sort_keys=True)
    return cache[name]["value"]


def reproduce_of(mcdc_data, disimpy_data):
    def reproduce(pack, quantity, grid):
        name = quantity["substrate"]
        seq = sequence_of(name, mcdc_data)
        phi = np.asarray(pack.walker_primitives(seq).phi, np.float64)
        value, se = _mean_and_se(np.exp(1j * phi[:, grid["index"]]))
        return dict(value=value, se=se, se_kind="delta_method", solver=grid["solver"],
                    se_derivation="the delta method on the modulus of the ensemble mean: each walker's own "
                                  "phasor projected on the direction the modulus is taken along, sd over "
                                  "walkers / sqrt(N) / |S| (cross_engine_parity.floors' estimator, on the "
                                  "modulus rather than on cos phi)",
                    n_walkers=int(pack.n_walkers))
    return reproduce


def served_vs_channel_of(mcdc_data, disimpy_data):
    def served_vs_channel(pack):
        """The largest difference over the fixture's whole protocol between the signal the pack SERVES
        (``ReplayPack.replay``) and the one its stored bands hold (``walker_primitives``: the bands contracted
        once per walker), both reduced as the modulus of the ensemble mean.

        Reducing one side as ``mean(cos phi)`` and the other as the modulus is what made this check read
        3.1e-3 at b = 13190 s/mm^2 and filed dmipy-sim#484 against the engine; the per-walker phases are in
        fact identical across the routes.
        """
        X = fixtures_module()
        seq = sequence_of(_name_of(pack), mcdc_data)
        served = np.abs(np.asarray(pack.replay(seq), np.complex128)).ravel()
        phi = np.asarray(pack.walker_primitives(seq).phi, np.float64).T
        return float(np.max(np.abs(served - X.ensemble_modulus(phi))))
    return served_vs_channel


def _name_of(pack):
    """The fixture name of a pack of this family, from its id ``parity-fixtures/<name>``."""
    name = str(pack.meta["id"]).split("/", 1)[1]
    if name not in FIXTURES:
        raise ValueError(f"{pack.meta['id']!r} is not a pack of this family ({', '.join(FIXTURES)})")
    return name


# ----------------------------------------------------------------- the "Use me" snippet
def snippet(uri):
    return f'''import numpy as np
from dmipy_sim.replay import ReplayPack
from examples.validation.cross_engine_parity import MCDC_N_T, MCDC_TE
from dmipy_sim import pgse

pk = ReplayPack.load("{uri}")
# the ActiveAx shell this fixture's number is read at, as a single PGSE row
seq = pgse([[1.0, 0.0, 0.0]], 0.01015, 0.03578, bvalues=[1.319e10], TE=MCDC_TE, n_t=MCDC_N_T, slew_rate=np.inf)
print("S = %.6f" % float(np.abs(np.asarray(pk.replay(seq))).ravel()[-1]))
'''


# ----------------------------------------------------------------- the family
def _read(path):
    return open(path, encoding="utf-8", errors="replace").read()


def family(mcdc_data, disimpy_data, work_dir, *, names, dry, create_dataset):
    X = fixtures_module()
    grids = {n: quantity_grid(n) for n in names}

    sources = []
    if any(is_mcdc(n) for n in names):
        lic = next((os.path.join(mcdc_data, c) for c in ("LICENSE", "LICENSE.txt", "license.txt", "COPYING")
                    if os.path.exists(os.path.join(mcdc_data, c))), None)
        if lic is None:
            raise SystemExit(f"no licence file under {mcdc_data}; a source record carries the host's licence "
                             f"text verbatim and cannot be written without it")
        files = []
        for n in [x for x in names if is_mcdc(x)]:
            p = X.mcdc_paths(mcdc_data, *mcdc_params(n))
            files += [SourceFile(path=p["ply"], cite_as=os.path.basename(p["ply"]),
                                 role=f"{n}: their closed undulating-axon surface"),
                      SourceFile(path=p["ini"], cite_as=os.path.basename(p["ini"]),
                                 role=f"{n}: their released initial-walker list, which the walk seeds from"),
                      SourceFile(path=p["dwi"], cite_as=os.path.basename(p["dwi"]),
                                 role=f"{n}: their released signal, the number this family reproduces")]
        p0 = X.mcdc_paths(mcdc_data, *mcdc_params([x for x in names if is_mcdc(x)][0]))
        files += [SourceFile(path=p0["scheme"], cite_as=os.path.basename(p0["scheme"]),
                             role="their ActiveAx scheme file: the acquisition both engines play"),
                  SourceFile(path=p0["conf"], cite_as=os.path.basename(p0["conf"]),
                             role="their configuration: the diffusivity, the voxel and the PLY scale")]
        sources.append(Source(key="mcdc", url=MCDC_URL,
                              host_record="Robust-Monte-Carlo-Simulations, the released Experiments archive",
                              licence_id=X.MCDC_LICENSE, licence_url=MCDC_URL + "/blob/master/LICENSE",
                              licence_text=_read(lic), redistributes_bytes=False, files=tuple(files)))
    if any(not is_mcdc(n) for n in names):
        lic = next((c for c in ("license.txt", "LICENSE", "LICENSE.txt",
                                os.path.join("disimpy", "LICENSE"))
                    if os.path.exists(os.path.join(disimpy_data, c))), None)
        if lic is None:
            raise SystemExit(f"no licence file under {disimpy_data}; a source record carries the host's licence "
                             f"text verbatim and cannot be written without it")
        lic = os.path.join(disimpy_data, lic)
        d = os.path.join(disimpy_data, "disimpy", "tests")
        sources.append(Source(
            key="disimpy", url=DISIMPY_URL, host_record="disimpy, the repository's tests/ fixtures",
            licence_id=X.DISIMPY_LICENSE, licence_url=DISIMPY_URL + "/blob/master/LICENSE",
            licence_text=_read(lic), redistributes_bytes=False,
            files=(SourceFile(path=os.path.join(d, "cylinder_mesh_closed.pkl"),
                              cite_as="cylinder_mesh_closed.pkl", role="their closed cylinder mesh"),
                   SourceFile(path=os.path.join(
                       d, "misst_cylinder_signal_smalldelta_30ms_bigdelta_40ms_radius_5um.txt"),
                       cite_as="misst_cylinder_signal_smalldelta_30ms_bigdelta_40ms_radius_5um.txt",
                       role="the MISST reference signal distributed with it, the number this fixture "
                            "reproduces"),
                   SourceFile(path=_derived_surface(disimpy_data),
                              cite_as=os.path.basename(_derived_surface(disimpy_data)),
                              role="the closed surface DERIVED from cylinder_mesh_closed.pkl by "
                                   "cross_engine_parity.disimpy_mesh -- as stored the pickle reads as open "
                                   "with 112 boundary edges and closes with 0 once its duplicate vertices are "
                                   "merged (352 -> 296) -- written into the surface cache under its content "
                                   "hash. It is the file the spec cites, so it is the file whose digest the "
                                   "source record has to hold"))))

    direct = {n: direct_at(n, grids[n], work_dir, mcdc_data, disimpy_data) for n in names}
    quantities = []
    for n in names:
        grid = grids[n]
        their_value, their_n = published_value(n, grid, mcdc_data, disimpy_data)
        quantities.append(ReferenceQuantity(
            substrate=n, name="signal_at_top_b",
            published=Published(
                value=their_value, unit="-",
                uncertainty=(float(direct[n]["floor_theirs"]) / abs(their_value) if their_n else 0.0),
                uncertainty_is=(f"their released file is ONE Monte-Carlo realisation of {their_n:,} walkers and "
                                f"states no spread, so its standard error is the per-walker spread measured on "
                                f"OUR walkers at THEIR count (cross_engine_parity.floors), which is what makes "
                                f"this comparison failable" if their_n else
                                "none: MISST is an exact eigenfunction solution, not a Monte-Carlo estimate. "
                                "What the file does limit is its own resolution -- it is distributed to seven "
                                "decimal places -- and that quantisation is a term of the design's tolerance "
                                "rather than an uncertainty of the reference"),
                printed_in=(
                    "Robust Monte-Carlo Simulations in Diffusion-MRI: Effect of the Substrate Complexity and "
                    "Parameter Choice on the Reproducibility of Results (Rafael-Patino et al. 2020), whose "
                    "released Experiments-raw-signals archive the array is read from" if is_mcdc(n) else
                    "Disimpy: A massively parallel Monte Carlo simulator for generating diffusion-weighted MRI "
                    "data in Python (Kerkelae et al. 2020), whose tests/ fixtures the array is read from"),
                document=("10.3389/fninf.2020.00008" if is_mcdc(n) else "10.21105/joss.02527"),
                locator=("the last measurement of ActiveAxG140_PM.scheme in "
                         f"{os.path.basename(X.mcdc_paths(mcdc_data, *mcdc_params(n))['dwi'])}, an unnormalised "
                         f"sum over walkers divided by the count its b = 0 rows state" if is_mcdc(n) else
                         "the last of the 100 normalised values of "
                         "misst_cylinder_signal_smalldelta_30ms_bigdelta_40ms_radius_5um.txt"),
                data_url=(MCDC_URL if is_mcdc(n) else DISIMPY_URL),
                data_sha256=_sha(X.mcdc_paths(mcdc_data, *mcdc_params(n))["dwi"] if is_mcdc(n) else
                                 os.path.join(disimpy_data, "disimpy", "tests",
                                              "misst_cylinder_signal_smalldelta_30ms_bigdelta_40ms_"
                                              "radius_5um.txt"))),
            direct=Direct(
                value=float(direct[n]["value"]), unit="-", se=float(direct[n]["se"]),
                se_kind="delta_method",
                se_derivation="the delta method on the modulus of the ensemble mean over the fused walk's "
                              "own walkers: each walker's phasor projected on the direction the modulus is "
                              "taken along, sd / sqrt(N) / |S|. No fold, no split half -- the "
                              "one-permutation split half this family used before swung the worst "
                              "measurement between 2.09 and 3.51 sigma over 200 seeds, and the family-wise "
                              "check failed for 49.5 % of them",
                n_walkers=int(direct[n]["n_walkers"]), grid=grid, solver=grid["solver"],
                source="examples/substrate_bank/build_parity_fixtures.py (direct_at): the fused engine's walk "
                       "of the same surface at the fixture's diffusivity and seeding, the gradient applied in the "
                       "walk -- a walk of its own, independent of the pack's, with no band between the walk and "
                       "the number")))

    reference = Reference(
        doi="10.3389/fninf.2020.00008",
        title="Robust Monte-Carlo Simulations in Diffusion-MRI: Effect of the Substrate Complexity and "
              "Parameter Choice on the Reproducibility of Results",
        published_kind="data",
        sample=("the same objects: MC/DC's own released undulating-axon surfaces and Disimpy's own released "
                "cylinder mesh, each walked here on THEIR surface at THEIR acquisition and diffusivity, the "
                "MC/DC ones seeded from their own released initial-walker list"),
        sample_relation="the same object", quantities=tuple(quantities),
        parameters=(
            FreeParameter(name="TE / delta / Delta / G", value="ActiveAxG140_PM.scheme", unit="-",
                          whose="theirs", where="their scheme file",
                          how="read by io.mcdc.read_scheme, which puts TE, every Delta, every delta and every "
                              "pad on a sample and refuses what the format leaves ambiguous"),
            FreeParameter(name="diffusivity", value=0.6e-9, unit="m^2/s", whose="theirs",
                          where="their .conf", how="stated; 2e-9 for the Disimpy fixture, from their own test"),
            FreeParameter(name="seed positions", value="their released initial-walker list", unit="-",
                          whose="theirs", where="*_ini_points.txt",
                          how="read cyclically as MC/DC reads it, which is why the spec's seeding rule is "
                              "`explicit` and not a uniform draw over the lumen"),
            FreeParameter(name="sub-step rule", value="the engine's own", unit="-", whose="ours",
                          where="physics.resolve_sub_steps",
                          how="theirs is a fixed 5,000 steps over TE, recorded for comparison; a converged "
                              "walk's signal does not depend on either"),
            FreeParameter(name="surface relaxivity", value=0.0, unit="m/s", whose="ours",
                          where="this family's walk", how="none: both references' walls are purely reflecting")),
        description=("Two other simulators' own released substrates, walked here on their surfaces at their "
                     "acquisitions: the cross-ENGINE check that this engine is not being compared with itself."),
        source_note=("MC/DC (Rafael-Patino et al. 2020, Front. Neuroinform. 14:8, LGPL-2.1) and Disimpy "
                     "(Kerkelae et al. 2020, JOSS 5(52):2527, MIT), whose MISST reference is Drobnjak, Zhang, "
                     "Hall and Alexander's exact eigenfunction solution."),
        licence_note="the packs are the source's licence: LGPL-2.1 for the MC/DC fixtures, MIT for the "
                     "Disimpy one; neither source's bytes are redistributed",
        caveats={
            # keyed on the PACK's own name, so the card renders it beside that pack rather than three sections
            # away from it, and the organisation page inherits it with the rest of the reference record
            "disimpy-cylinder": (
                "**Published with a hold on "
                "[dmipy-sim#488](https://github.com/dmrai-lab/dmipy-sim/issues/488).** The bytes are public and "
                "this pack's manifest row carries the reservation beside its verdict; the verdict itself is what "
                "the gate gave, a pass.\n\n"
                "The gate compares ONE declared measurement -- the last of the fixture's own protocol, its "
                "highest b -- and there this pack agrees with the MISST reference to 1.8e-5. #488's evidence is "
                "a VECTOR over the protocol's 100 measurements, and a `ReferenceQuantity` is a scalar, so no "
                "scalar comparison this protocol can express will fail this pack; "
                "[dmipy-sim#493](https://github.com/dmrai-lab/dmipy-sim/issues/493) is the item that would let "
                "the gate carry a vector and therefore see it. What #488 weighs -- whether the residual is "
                "MISST's own truncation or ours -- and the numbers it weighs it with are stated there, not "
                "here: this card states only what its own records hold."),
            "direct": ("Our direct number is the fused engine's own walk of each fixture, the gradient applied "
                       "in the walk, at the fixture's seeding: an independent walk, so `reproduces-` compares the "
                       "pack's stored bands with the engine's signal, and `published-` both with the other "
                       "engine's released array."),
            "reduction": ("A pack's signal is the MODULUS of the weighted ensemble mean, and both references sum "
                       "cosines. Comparing one convention against the other read 1.2e-5 at b = 1925 s/mm^2 "
                       "and 3.1e-3 at 13190 and filed #484 against the engine; the per-walker phases are "
                       "identical across the routes. Every comparison on this card reduces both sides the "
                       "same way.")})

    design = Design(
        window_s=(X.MCDC_TE if is_mcdc(names[0]) else X.DISIMPY_TE), K=K_BANDS, envelope=envelope_of(names[0], mcdc_data),
        waveforms=lambda: _waveforms(names[0], mcdc_data),
        tiers=(Tier(name="positions", floor_key="floor_max", err_key="err_max", target_floor=SIGMA),
               Tier(name="contact", floor_key="floor_surface", err_key="err_surface", target_floor=SIGMA)),
        memory_budget_bytes=BUDGET_BYTES, pilot_n=PILOT_N, safety=1.4,
        pilot_substrate=names[0], false_failure_rate=FALSE_FAILURE_RATE,
        tolerance=Tolerance(terms=("quantity.direct.se", "quantity.replay_se",
                                   "quantity.published.uncertainty",
                                   "design.systematics.reference_quantisation.value")),
        systematics=(Systematic(
            name="reference_quantisation", value=float(X.MCDC_QUANTISATION),
            measured_on="the released files' own resolution",
            evidence="MC/DC's .bfloat is a float32 sum over 50,000 walkers, so half an ulp of the stored value "
                     "over the count is 3.906e-8 of the normalised signal; Disimpy's MISST file is "
                     "distributed to seven decimal places, a half-ulp of 5e-8. At b = 1 s/m^2 the walk's own "
                     "standard error is 1.6e-11, so without this term the comparison asks the file for digits "
                     "it does not have -- 68 sigma, measured."),))

    build = Build(
        specs={n: (lambda nn=n: spec_of(nn, mcdc_data, disimpy_data)) for n in names},
        pack_id={n: f"parity-fixtures/{n}" for n in names},
        reproduce=reproduce_of(mcdc_data, disimpy_data),
        served_vs_channel=served_vs_channel_of(mcdc_data, disimpy_data), served_tier="positions",
        walk=walk_of(mcdc_data, disimpy_data))

    publication = Publication(repo=REPO, licence="LGPL-2.1", citation=X.MCDC_CITATION, snippet=snippet,
                              snippet_substrate=names[0],
                              pack_path=lambda name: f"packs/{name}.rpk",
                              hold=({"disimpy-cylinder": HOLD_488} if "disimpy-cylinder" in names else {}),
                              create_dataset=create_dataset, dry=dry)
    return ReferenceFamily("parity-fixtures", work_dir, sources=sources, reference=reference, design=design,
                           build=build, publication=publication)


def _waveforms(name, mcdc_data):
    """The fixture's own protocol, as the one waveform the family declares: it is what these packs exist for,
    and ``waveform_band`` is checked against it rather than against a battery nobody will play."""
    return ((f"{name}-own-protocol", sequence_of(name, mcdc_data)),)


def _derived_surface(disimpy_data):
    """The path of the repaired Disimpy surface the spec cites: writing the spec writes the file."""
    X = fixtures_module()
    from dmipy_sim.spec import build
    spec = X.disimpy_spec(disimpy_data)
    return build.resolve_surface_file(spec.walls[0].surface.file)


def _sha(path):
    from dmipy_sim.fill.hub import sha256_of
    return sha256_of(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mcdc-data", required=True, help="the MC/DC Experiments archive")
    p.add_argument("--disimpy-data", required=True, help="the directory holding the disimpy checkout")
    p.add_argument("--work", default=os.path.join(os.path.expanduser("~"), "dmrai-ws", "parity-fixtures"))
    p.add_argument("--names", nargs="+", default=list(FIXTURES), choices=list(FIXTURES))
    p.add_argument("--stage", default=None)
    p.add_argument("--dry", action="store_true")
    p.add_argument("--create-dataset", action="store_true")
    a = p.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("dmipy_sim").setLevel(logging.INFO)
    os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/hf-session"))
    fam = family(a.mcdc_data, a.disimpy_data, os.path.abspath(a.work), names=list(a.names), dry=a.dry,
                 create_dataset=a.create_dataset)
    fam.stage(a.stage) if a.stage else fam.run()
    print(f"records in {fam.records.dir}")


if __name__ == "__main__":
    main()
