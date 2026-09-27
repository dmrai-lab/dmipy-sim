"""Ling et al. 2022's quartz/garnet sand packs as a reference family, run through the reference-pack protocol.

Five resin-set synthetic sediment packs whose CPMG was measured and whose pore space was THEN imaged by
micro-CT -- the same physical object, both halves released (figshare 10.6084/m9.figshare.17161730, CC BY 4.0) --
walked as a ``geometry.LabelVolume`` on the released 450^3 Avizo lattices and published to
``SubstrateCommons/ling-sand-packs``.

**Nothing about relaxation is in the walk**: each pack stores the boundary local time (C2) as a channel, so the
surface relaxivity ``rho`` and the bulk ``T2B`` are replay knobs and Ling's two relaxivities are a sweep over
one pack rather than a walk per number.

This file is the family's DECLARATION, re-expressed through
:class:`~dmipy_sim.replay.reference.ReferenceFamily` so that its records have the same shape as every other
family's. The packs already exist and are neither re-walked nor re-encoded; the walk stage reads the counters
the engine reported in the family's own pre-protocol records, which are kept beside the new ones under
``records/pre-protocol/``.

**The grade changes, by rule.** The pre-protocol record stated ``B`` as a FIELD, reasoning that the paper prints
no T2 value. :func:`~dmipy_sim.replay.reference.grade_of` assigns ``A``: the quantity is read from Ling's own
RELEASED echo trains, which are the measurement, on the same physical object, and ``published_kind = "data"`` is
graded like a number for exactly that reason. No record states its own grade any more.

::

    HF_HUB_CACHE=~/.cache/hf-session JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \\
      nice -n 10 python examples/substrate_bank/build_ling_sand_packs.py --data <the .am images> \\
      --work <the family dir> --stage source     # then reference, spec, design, walk, pack, gate, card
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from dmipy_sim.replay.reference import (Build, Design, Direct, FreeParameter, Published, Reference,
                                        ReferenceFamily, ReferenceQuantity, Publication, RecordedWalk, Source,
                                        SourceFile, Systematic, Tier, Tolerance)

REPO = "SubstrateCommons/ling-sand-packs"
LICENCE = "CC-BY-4.0"
CITATION = ("Ling N, Hussaini SR, Elsayed M, Connolly P, El-Husseiny A, Mahmoud M, May EF & Johns ML (2022), "
            "Model Synthetic Samples for Validation of NMR Signal Simulations, Transport in Porous Media "
            "142:623-639, and the micro-CT deposit figshare 10.6084/m9.figshare.17161730 (CC BY 4.0); "
            "Fick RHJ (2026), the replay packs, SubstrateCommons")

D0 = 2.3e-9         #: m^2/s, the brine's self-diffusion the walk uses -- theirs
T2B = 3.0           #: s, the brine's bulk T2 -- OURS, applied at replay (the paper states none)
TE_REF = 100e-6     #: s, the echo spacing whose train this family reproduces (their rho was fitted on it)
DT_SAVE = 1e-3      #: s, the save grid, which is also the grid the log-mean is inverted on
INVERT_MS = 1.0     #: ms, the sampling of the decay the inversion runs on -- part of the measurement
SIGMA = 5e-3        #: the floor per tier the family targets
K_BANDS = 128
BUDGET_BYTES = 60_000_000_000
PILOT_N = 4000
SEED = 0
FALSE_FAILURE_RATE = 0.01

PAPER_DOI = "10.1007/s11242-022-01764-w"
PAPER_TITLE = "Model Synthetic Samples for Validation of NMR Signal Simulations"
DATA_DOI = "10.6084/m9.figshare.17161730"
DATA_URL = "https://doi.org/" + DATA_DOI

#: Ling's fitted surface relaxivities, per mineral (section 3.2). A pack of ONE mineral declares its own as the
#: nominal; a MIXTURE declares none, because ``LabelVolume`` accumulates one boundary local time over every wall
#: and one scalar cannot be right on both surfaces.
RHO = {"quartz": 12.5e-6, "garnet": 98.5e-6}

#: The five packs, their pools in pool-id order, and the window each one's own released train spans.
SAMPLES = {
    "6_Q100": dict(pools={0: "free", 1: "quartz"}, composition="100 % quartz (v/v)", window_s=3.5),
    "5_Q75G25": dict(pools={0: "free", 1: "quartz", 2: "garnet"},
                     composition="75 % quartz / 25 % garnet (v/v)", window_s=3.5),
    "4_Q50G50": dict(pools={0: "free", 1: "quartz", 2: "garnet"},
                     composition="50 % quartz / 50 % garnet (v/v)", window_s=2.0),
    "2_Q25G75": dict(pools={0: "free", 1: "quartz", 2: "garnet"},
                     composition="25 % quartz / 75 % garnet (v/v)", window_s=2.0),
    "1_G100": dict(pools={0: "free", 1: "garnet"}, composition="100 % garnet (v/v)", window_s=2.0),
}

#: The packs whose reference quantity is recorded: the two PURE ones. A mixture's two mineral surfaces carry two
#: of Ling's relaxivities and the walk accumulates one contact channel over both, so no single ``rho`` describes
#: it; the three mixtures are built, certified, gated and published, and their reference row is BLOCKED
#: (dmipy-sim#491) rather than approximated with one of the two numbers.
PURE = ("1_G100", "6_Q100")

T2_GRID = dict(kind="log", n=60, from_s=1e-3, to_s=10.0)
SOLVER = "non-negative least squares with a second-difference penalty, lam = 0.1 (t2_distribution)"

#: What these packs say they are FOR: a relaxometry envelope. The acquisition this family reproduces plays no
#: gradient at all, so the declared waveforms are a clinical diffusion pair and the band is small and stated --
#: a consumer asking for more is refused by ``waveform_band`` rather than served a wrong number.
ENVELOPE = dict(bvals=[0.0, 0.5e9, 1e9], dirs=[[0, 0, 1], [1, 0, 0], [1, 0, 1]],
                ogse_periods=[1, 2, 3], shortd_b=1e9, delta_frac=40e-3 / 3.5,
                shortd_deltas_frac=[60e-3 / 3.5, 40e-3 / 3.5], Delta_frac=80e-3 / 3.5,
                rho_list=sorted(RHO.values()))

DECLARED_DELTAS = (40e-3, 60e-3)


def waveforms():
    """Every waveform the family declares, checked against ``waveform_band`` per pack by the pack stage."""
    from dmipy_sim import pgse
    return tuple((f"pgse-delta{delta * 1e3:.0f}ms",
                  pgse([[1, 0, 0]], delta, max(2 * delta, 80e-3), bvalues=[max(ENVELOPE["bvals"])], n_t=400))
                 for delta in DECLARED_DELTAS)


# ----------------------------------------------------------------- the pre-protocol records this family kept
def pre(work_dir, stage):
    """A record this family's own build script wrote before the protocol existed, kept under
    ``records/pre-protocol/``: the walk's counters and the reproduction's numbers were measured by it and are
    read, never re-measured."""
    return json.load(open(os.path.join(work_dir, "records", "pre-protocol", f"{stage}.json")))


def reproduction(work_dir):
    return json.load(open(os.path.join(work_dir, "reproduction.json")))


def direct_record(work_dir):
    return json.load(open(os.path.join(work_dir, "direct.json")))


def rho_of(sample):
    """The relaxivity a pack's spec declares as nominal: Ling's for one mineral, 0 for a mixture."""
    mins = {v for k, v in SAMPLES[sample]["pools"].items() if k}
    return RHO[mins.pop()] if len(mins) == 1 else 0.0


def image_path(sample, data_dir, work_dir):
    return os.path.join(data_dir, reproduction(work_dir)["samples"][sample]["image"]["file"])


# ----------------------------------------------------------------- the spec producer and the walk
def spec_of(sample, data_dir, work_dir):
    """The pack's substrate spec: the released lattice cited by the name the deposit distributes it at, with
    Ling's values as the NOMINAL ones a consumer overrides."""
    from dmipy_sim.spec import label_volume_spec
    s, rep = SAMPLES[sample], reproduction(work_dir)["samples"][sample]
    rho = rho_of(sample)
    return label_volume_spec(
        image_path(sample, data_dir, work_dir), pools=s["pools"], rho=rho, D=D0,
        T2_pools={name: T2B for name in s["pools"].values()},
        cite_image_as=rep["image"]["file"], id=f"ling2022/{sample.lower()}",
        source="Ling et al. 2022 model synthetic sediment packs, figshare "
               f"{DATA_DOI} (CC BY 4.0)",
        description=(f"{sample}: {s['composition']}, the released 450^3 micro-CT sub-volume; the brine walks "
                     f"between the voxel faces of the grains. Relaxation is not in the walk -- rho and T2 are "
                     f"replay knobs, and the nominal rho is "
                     + (f"Ling's {rho * 1e6:g} um/s for this mineral" if rho else
                        "NONE: two mineral surfaces carry two of Ling's relaxivities and one contact channel "
                        "cannot hold both, so a consumer states one")))


def walk_of(data_dir, work_dir):
    def walk(spec, n, *, n_t):
        import dmipy_sim as d
        from dmipy_sim.spec import geometry_from_spec
        return d.simulate_trajectories(int(n), D0, geometry_from_spec(spec), T_max=(n_t - 1) * DT_SAVE,
                                       dt_save=DT_SAVE, seed=SEED, tiers="all", require_gpu=False,
                                       walker_batch_size=2000)
    return walk


# ----------------------------------------------------------------- the quantity
def t2_grid(grid):
    return np.logspace(np.log10(grid["from_s"]), np.log10(grid["to_s"]), int(grid["n"]))


def decay_and_log_mean(pack, rho, grid, *, chunk=4000):
    """``(t, S, T2 log-mean, its relative delta-method standard error)`` from the pack's stored C2 channel.

    ``S(t) = <exp((rho/D) L(t) - t/T2B)>`` on the recorded sampling; the standard error is the log-mean's
    gradient with respect to the decay it is inverted from, projected on each walker's own decay -- the same
    estimator the Imperial rocks family uses, and the reason it is analytic rather than a fold spread.
    """
    from examples.validation.talabi_micro_ct_rocks import log_mean_gradient
    ell = pack.contact()
    if ell is None:
        raise ValueError(f"{pack.meta['id']} carries no C2 channel, so it cannot serve a relaxation decay")
    n_w, n_t = ell.shape
    every = max(1, int(round(grid["sample_ms"] * 1e-3 / pack.dt)))
    t = np.arange(n_t)[every::every] * pack.dt

    def per_walker(lo):
        L = np.cumsum(np.asarray(ell[lo:lo + chunk], np.float64), axis=1)[:, every::every]
        return np.exp(float(rho) / D0 * L - t / T2B)

    acc = np.zeros(len(t))
    for lo in range(0, n_w, chunk):
        acc += per_walker(lo).sum(axis=0)
    S = acc / n_w
    lm, g = log_mean_gradient(t, S, t2_grid(grid))
    proj = np.empty(n_w)
    for lo in range(0, n_w, chunk):
        s = per_walker(lo)
        proj[lo:lo + s.shape[0]] = s @ g
    return t, S, lm, float(np.std(proj, ddof=1) / np.sqrt(n_w) / lm)


def reproduce(pack, quantity, grid):
    """The T2 log-mean this pack serves, on the grid the reference record states."""
    rho = float(pack.nominal.rho)
    _t, _S, lm, se = decay_and_log_mean(pack, rho, grid)
    return dict(value=lm * 1e3, se=se, se_kind="delta_method", solver=grid["solver"],
                se_derivation="the delta method over walkers on the log-mean: its gradient with respect to the "
                              "decay it is inverted from (talabi_micro_ct_rocks.log_mean_gradient, the active "
                              "set's linear map) projected on each walker's own decay; sd / sqrt(N) / T2lm",
                rho_m_per_s=rho, n_points=int(len(_t)))


def served_vs_channel(pack, *, n_points=8):
    """How far the decay the pack SERVES a consumer is from the one its decoded contact channel holds.

    The consumer route for a multi-TE relaxation measurement is one gradient-free spin echo per echo time; the
    cheap route reads the same C2 channel. This is the check that they are one quantity.
    """
    from dmipy_sim.engine.pulse_sequence import bare_spin_echo
    rho = float(pack.nominal.rho)
    ell = pack.contact()
    n_w = ell.shape[0]
    step = max(1, pack.n_t // (n_points + 1))
    ks = np.asarray(list(range(step, pack.n_t, step))[:n_points])
    t = ks * pack.dt
    acc = np.zeros(len(ks))
    for lo in range(0, n_w, 4000):
        L = np.cumsum(np.asarray(ell[lo:lo + 4000], np.float64), axis=1)[:, ks]
        acc += np.exp(rho / D0 * L - t / T2B).sum(axis=0)
    served = [float(np.asarray(pack.replay(bare_spin_echo(float(te), dt=pack.dt),
                                          tissue=pack.nominal)).ravel()[0]) for te in t]
    return float(np.max(np.abs(np.asarray(served) - acc / n_w)))


# ----------------------------------------------------------------- the "Use me" snippet
def snippet(uri):
    return f'''import numpy as np
from dmipy_sim.replay import ReplayPack
from examples.validation.talabi_micro_ct_rocks import t2_distribution, log_mean_T2

pk = ReplayPack.load("{uri}")
ell = pk.contact()                                        # the stored wall-contact channel, per save
L = np.cumsum(np.asarray(ell, np.float64), axis=1)        # the boundary local time
t = np.arange(pk.n_t) * pk.dt
S = np.exp(pk.nominal.rho / {D0!r} * L - t / {T2B!r}).mean(axis=0)   # pk.nominal.rho IS Ling's for this mineral
every = round({INVERT_MS!r}e-3 / pk.dt)
g = np.logspace(-3, 1, 60)
print("T2 log-mean %.1f ms" % (1e3 * log_mean_T2(g, t2_distribution(t[every::every], S[every::every], g))))
'''


# ----------------------------------------------------------------- the family
def licence_text(cache_dir):
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "cc-by-4.0-legalcode.txt")
    if not os.path.exists(cache):
        req = urllib.request.Request("https://creativecommons.org/licenses/by/4.0/legalcode.txt",
                                     headers={"User-Agent": "dmipy-sim (https://github.com/dmrai-lab/dmipy-sim)"})
        with urllib.request.urlopen(req, timeout=60) as fh, open(cache, "wb") as out:
            out.write(fh.read())
    return open(cache, encoding="utf-8").read()


def measured_se(sample, work_dir, grid):
    """The DIRECT walk's relative delta-method standard error on the log-mean, measured on this family's own
    pack and scaled to the direct walk's count by 1/sqrt(N).

    The direct fused walk's per-walker decays were not retained, and the relative per-echo spread is a property
    of the substrate and the estimator rather than of the pack, which is what makes the substitution legitimate;
    the record states it and the numbers it rests on. Measured once and cached on the pack's bytes and the grid.
    """
    from dmipy_sim.fill.hub import sha256_of
    from dmipy_sim.replay import ReplayPack
    d = direct_record(work_dir)[f"{sample}@{rho_of(sample) * 1e6:g}"]
    path = os.path.join(work_dir, "packs", f"{sample.lower()}.rpk")
    key = hashlib.sha256(json.dumps([sha256_of(path), grid], sort_keys=True).encode()).hexdigest()
    cache_path = os.path.join(work_dir, ".direct-se-cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    if cache.get(sample, {}).get("key") != key:
        pack = ReplayPack.load(path)
        _t, _S, _lm, se_pack = decay_and_log_mean(pack, rho_of(sample), grid)
        n_pack = int(pack.n_walkers)
        scale = (n_pack / float(d["n_walkers"])) ** 0.5
        cache[sample] = dict(key=key, value=dict(se=se_pack * scale, se_pack=se_pack, n_pack=n_pack,
                                                 n_direct=int(d["n_walkers"]), scale=scale,
                                                 direct_ms=float(d["T2lm_ms"]),
                                                 measured_on=os.path.basename(path)))
        with open(cache_path, "w") as fh:
            json.dump(cache, fh, indent=1, sort_keys=True)
    return cache[sample]["value"]


def family(data_dir, work_dir, *, samples, dry, create_dataset):
    rep = reproduction(work_dir)
    walk_rec = pre(work_dir, "walk")["packs"]
    grid = dict(T2_GRID, sample_ms=INVERT_MS, solver=SOLVER)
    text = licence_text(os.path.join(work_dir, ".licence-cache"))

    sources = [Source(
        key="figshare-ling2022", url=DATA_URL,
        host_record=f"figshare record {DATA_DOI} ({rep['reference']['dataset']['year']}, "
                    f"{rep['reference']['dataset']['bytes']} bytes as the record states)",
        licence_id=LICENCE, licence_url="https://creativecommons.org/licenses/by/4.0/", licence_text=text,
        redistributes_bytes=False,
        files=tuple(SourceFile(path=image_path(s, data_dir, work_dir),
                               cite_as=rep["samples"][s]["image"]["file"],
                               role=f"{s} ({SAMPLES[s]['composition']}): the Avizo ASCII lattice of the "
                                    f"released 450^3 sub-volume, one label per voxel")
                    for s in samples))]

    se = {s: measured_se(s, work_dir, grid) for s in samples if s in PURE}
    quantities = tuple(ReferenceQuantity(
        substrate=s, name="T2_log_mean",
        published=Published(
            value=float(rep["samples"][s]["reference"]["measured"]["T2lm_ms"]), unit="ms",
            uncertainty=0.0,
            uncertainty_is=("none stated: the paper prints no T2 value anywhere -- its comparison is Fig. 5, two "
                            "distributions overlaid with no agreement metric -- and the released echo train "
                            "carries no repeat, so the gate's budget rests on our own standard errors alone and "
                            "the disagreement with them is recorded rather than failed on"),
            printed_in=f"{PAPER_TITLE} (Ling et al. 2022), whose released te = 100 us echo train the value is "
                       f"read from",
            document=PAPER_DOI,
            locator=f"the log-mean of the released te = 100 us CPMG train of sample {s}, inverted on the "
                    f"{INVERT_MS:g} ms grid with the same estimator as ours",
            data_url=DATA_URL, data_sha256=rep["samples"][s]["image"]["sha256"]),
        direct=Direct(
            value=float(se[s]["direct_ms"]), unit="ms", se=float(se[s]["se"]), se_kind="delta_method",
            se_derivation=(
                "the delta method over walkers on the log-mean (talabi_micro_ct_rocks.log_mean_gradient), "
                f"MEASURED at record time on this family's own {s.lower()}.rpk -- {se[s]['se_pack']:.6f} over "
                f"{se[s]['n_pack']:,} walkers at the same rho, channel and grid -- and scaled to the direct "
                f"walk's {se[s]['n_direct']:,} by 1/sqrt(N) (x {se[s]['scale']:.6f}). The direct walk's own "
                f"per-walker decays were not retained; the relative per-echo spread is a property of the "
                f"substrate and the estimator, not of the pack"),
            n_walkers=int(se[s]["n_direct"]), grid=grid, solver=SOLVER,
            source="examples/validation/ling_sand_packs.py, recorded in this family's direct.json by the "
                   "DIRECT fused-engine walk at the walker count and step stated with it; that walk is read, "
                   "never repeated")) for s in samples if s in PURE)

    reference = Reference(
        doi=PAPER_DOI, title=PAPER_TITLE, published_kind="data",
        sample=("the same physical object: the CPMG was measured on each pack, which was then resin-set and "
                "micro-CT imaged, and both halves are in the same deposit"),
        sample_relation="the same object", quantities=quantities,
        parameters=(
            FreeParameter(name="rho", value=RHO, unit="m/s", whose="theirs",
                          where="paper §3.2", how="fitted by them per mineral to their own te = 100 us trains; "
                                                 "an effective value tied to their 3.93 um voxel, because a "
                                                 "segmentation's surface is the Manhattan area at that "
                                                 "resolution, which is what both walks relax against"),
            FreeParameter(name="D0", value=D0, unit="m^2/s", whose="theirs",
                          where="paper §2.3", how="the brine's self-diffusion, stated"),
            FreeParameter(name="T2B", value=T2B, unit="s", whose="ours",
                          where="not in the paper", how="the paper states no bulk T2; 3.0 s is ours, applied at "
                                                        "replay and not in the walk, and the reproduction "
                                                        "reports the log-mean over 2.4-3.1 s as well"),
            FreeParameter(name="T2 inversion", value=SOLVER, unit="-", whose="ours",
                          where="the paper states its own only as a regularised inversion",
                          how="non-negative least squares with a second-difference penalty on 60 points from "
                              "1 ms, applied to THEIR released train and to ours identically, so the two sides "
                              "of the comparison share the estimator even though it is not theirs"),
            FreeParameter(name="surface rule", value="exp(-2 (rho/D) d_perp) per reflection", unit="-",
                          whose="ours", where="the walk",
                          how="the boundary local time is stored as a channel and weighted at replay; their "
                              "walk kills a walker at an attempted move into a grain voxel. The two agree in "
                              "the continuum limit")),
        description=("Resin-set quartz and garnet sand packs whose CPMG was measured and whose pore space was "
                     "then imaged: the relaxometry reference where the measurement and the geometry are the "
                     "SAME object. Relaxation is not in the walk, so one pack answers every relaxivity."),
        source_note=("Ling et al. 2022 (Transport in Porous Media 142:623-639) for the packs, the measured CPMG "
                     "and the fitted relaxivities; the micro-CT lattices and the echo trains are both in the "
                     "figshare deposit 10.6084/m9.figshare.17161730 (CC BY 4.0)."),
        licence_note="CC-BY-4.0 images and echo trains; the packs are CC-BY-4.0 and no byte of the deposit is "
                     "redistributed",
        caveats=dict(
            mixtures=("The three MIXTURES have no reference row. LabelVolume accumulates ONE boundary local "
                      "time over every wall and Tissue.rho is one scalar, so a single pack cannot carry Ling's "
                      "12.5 um/s on quartz and 98.5 um/s on garnet at once. They are built, certified, gated "
                      "and published -- the geometry, the positions and the contact channel are all sound -- "
                      "and their spec declares NO nominal relaxivity, so a consumer must state one. Recorded "
                      "as blocked (dmipy-sim#491), not approximated with one of the two numbers."),
            echo_spacing=("Only their te = 100 us train is reproduced: it is the one their relaxivities were "
                          "fitted on. The te-DEPENDENCE of their trains (100 us to 20 ms) IS the garnet's "
                          "internal-gradient effect, and a label-volume pack carries no C3 field channel, so a "
                          "longer-te train is refused by name rather than served as a decay without its "
                          "dephasing."),
            grade=("The pre-protocol record of this family stated grade B as a FIELD, reasoning that the paper "
                   "prints no T2 value. grade_of assigns A by rule: the quantity is read from their own "
                   "RELEASED echo trains, which ARE the measurement, on the same physical object. What the "
                   "absence of a printed number costs is the reference's uncertainty -- it is 0.0 with a "
                   "reason, so the `published-` comparison cannot be failed on and its disagreement is "
                   "recorded instead."),
            windows=("The five packs do not share a window: 3.5 s for the two quartz-rich packs (35,000 echoes) "
                     "and 2.0 s for the three garnet-rich ones (20,000 echoes), each the extent of ITS OWN "
                     "released train. The design's window is the longest and each pack's own is in the walk "
                     "record.")))

    design = Design(
        window_s=max(SAMPLES[s]["window_s"] for s in samples), dt_save_s=DT_SAVE,
        save_grid_why=("1 ms is the grid the log-mean is stable on (measured on the Imperial rocks: a 20 ms "
                       "sampling moved it by a factor of two) and it holds every 100 us echo of their train "
                       "within a save. Each PUBLISHED pack's own window is its released train's extent, which "
                       "is why the five do not share one and each is in the walk record"),
        K=K_BANDS, envelope=ENVELOPE, waveforms=waveforms,
        tiers=(Tier(name="contact", floor_key="floor_surface", err_key="err_surface", target_floor=SIGMA),
               Tier(name="positions", floor_key="floor_max", err_key="err_max", target_floor=SIGMA)),
        memory_budget_bytes=BUDGET_BYTES, pilot_n=PILOT_N, safety=1.4,
        pilot_substrate=("1_G100" if "1_G100" in samples else sorted(samples)[0]),
        false_failure_rate=FALSE_FAILURE_RATE,
        tolerance=Tolerance(terms=("quantity.direct.se", "quantity.replay_se",
                                   "quantity.published.uncertainty",
                                   "design.systematics.bulk_T2_is_ours.value")),
        systematics=(Systematic(
            name="bulk_T2_is_ours", value=float(abs(3.1 - 2.4) / 2.0 / T2B),
            measured_on="the reproduction's own T2B sweep",
            evidence="the paper states no bulk T2. The reproduction reports the log-mean over T2B in "
                     "[2.4, 3.1] s, and half that range over the 3.0 s used is the relative spread the "
                     "comparison inherits from a parameter that is ours and not theirs. It is a systematic of "
                     "the comparison, not Monte-Carlo noise more walkers would remove."),))

    build = Build(
        specs={s: (lambda ss=s: spec_of(ss, data_dir, work_dir)) for s in samples},
        pack_id={s: f"ling-sand-packs/{s.lower()}" for s in samples},
        reproduce=reproduce, served_vs_channel=served_vs_channel, served_tier="contact",
        walk=walk_of(data_dir, work_dir),
        recorded={s: RecordedWalk(
            pack_path=os.path.join(work_dir, "packs", f"{s.lower()}.rpk"),
            sub_steps=int(walk_rec[s]["sub_steps"]),
            illegal_crossings=int(walk_rec[s]["illegal_crossings"]),
            evidence=(f"records/pre-protocol/walk.json, packs.{s}: the walk that produced the pack reported "
                      f"sub_steps {walk_rec[s]['sub_steps']}, {walk_rec[s]['illegal_crossings']} refused steps, "
                      f"{walk_rec[s]['walk_peak_rss_gb']} GB walk peak and "
                      f"{walk_rec[s]['pack_peak_rss_gb']} GB pack peak over "
                      f"{walk_rec[s]['seconds']} + {walk_rec[s]['pack_seconds']} s, in "
                      f"{walk_rec[s]['log']}")) for s in samples})

    publication = Publication(repo=REPO, licence=LICENCE, citation=CITATION, snippet=snippet,
                              snippet_substrate=("1_G100" if "1_G100" in samples else sorted(samples)[0]),
                              pack_path=lambda name: f"packs/{name.lower()}.rpk",
                              create_dataset=create_dataset, dry=dry)
    return ReferenceFamily("ling-sand-packs", work_dir, sources=sources, reference=reference, design=design,
                           build=build, publication=publication)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="the directory holding the released .am lattices")
    p.add_argument("--work", default=os.path.join(os.path.expanduser("~"), "dmrai-ws", "ling-sand-packs"))
    p.add_argument("--samples", nargs="+", default=list(SAMPLES), choices=list(SAMPLES))
    p.add_argument("--stage", default=None)
    p.add_argument("--dry", action="store_true")
    p.add_argument("--create-dataset", action="store_true")
    a = p.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("dmipy_sim").setLevel(logging.INFO)
    os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/hf-session"))
    os.environ.setdefault("DMIPY_SIM_SURFACE_DIR", os.path.abspath(a.data))
    fam = family(os.path.abspath(a.data), os.path.abspath(a.work), samples=list(a.samples), dry=a.dry,
                 create_dataset=a.create_dataset)
    fam.stage(a.stage) if a.stage else fam.run()
    print(f"records in {fam.records.dir}")


if __name__ == "__main__":
    main()
