"""The Imperial micro-CT rocks as a reference family, run through the reference-pack protocol (#482, #480).

Two segmented micro-CT images of the Imperial College 2007 collection (Dong & Blunt 2009; Figshare, CC BY 4.0)
walked as a ``geometry.LabelVolume`` on Talabi's own crop and diffusivity, published to
``SubstrateCommons/imperial-rocks``. **Nothing about relaxation is in the walk**: the geometry is walked with no
surface relaxivity and no bulk T2, the pack stores the boundary local time (C2) as a channel, and ``rho2`` and
``T2B`` are replay knobs -- which is what makes Talabi's own open question (whether Berea's relaxivity is the
15 um/s of his §8.3 text or the 16 um/s of his Table 8-2) a sweep over one pack rather than a second walk.

This file is the family's DECLARATION plus the three callables the protocol asks of it (a spec producer, a
walk, and the reproduction of the quantity); every stage, rule and refusal is
:class:`dmipy_sim.replay.reference.ReferenceFamily`. The packs already exist and are **neither re-walked nor
re-encoded**: the walk and pack stages read them and their recorded numbers.

::

    HF_HUB_CACHE=~/.cache/hf-session DMIPY_SIM_SURFACE_DIR=<the images> \\
      python examples/substrate_bank/build_imperial_rocks.py --data <the images> --work <the family dir> [--dry]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))          # examples.validation is importable

from dmipy_sim.replay.reference import (Build, Design, Direct, FreeParameter, Published, Reference,
                                        ReferenceFamily, ReferenceQuantity, Publication, RecordedWalk, Source,
                                        SourceFile, Systematic, Tier, Tolerance)

REPO = "SubstrateCommons/imperial-rocks"
LICENCE = "CC-BY-4.0"
CITATION = ("Imperial College Consortium on Pore-scale Imaging and Modelling (2014), the micro-CT image; "
            "Dong H & Blunt MJ (2009) Phys Rev E 80:036307, the imaging and networks; "
            "Talabi O (2008) PhD thesis, the reference random walk and CPMG; "
            "Fick RHJ (2026), the replay pack, SubstrateCommons")

D0 = 2.07e-9        #: m^2/s, brine at 308 K (Talabi eq. 7.1)
T2B = 3.1           #: s, the brine's bulk T2 (eq. 7.2) -- NOMINAL, applied at replay
T = 3.0             #: s, the window of the decays the thesis plots (Figs 7-6..7-11, 8-6)
DT_SAVE = 200e-6    #: s, the MARAN2 train's inter-echo time (§7.3): every echo of that train falls on a save
CROP = 300          #: voxels, his central cubic section (App. A-1)
SIGMA = 5e-3        #: the split-half floor per tier this family targets
K_BANDS = 128       #: bands per 100 ms storage window, set by the declared gradient envelope
BUDGET_BYTES = 60_000_000_000
PILOT_N = 8000
SEED = 0
FALSE_FAILURE_RATE = 0.01   #: the probability this gate fails a CORRECT family, DECLARED before it runs; the
                            #: band per comparison follows from it and the comparison count (`reference.pass_band`)
INVERT_MS = 1.0     #: the sampling of the decay the log-mean is inverted on -- part of the measurement

#: The DOI of the DOCUMENT that prints the numbers: the thesis itself, in Imperial's Spiral (a DataCite DOI,
#: which Crossref answers 404 for). ``REFERENCE_DOI`` is the peer-reviewed publication of the same chapter --
#: both are resolved and title-compared, so the record never implies the paper prints a table it was not read for.
THESIS_DOI = "10.25560/4261"
REFERENCE_DOI = "10.1016/j.petrol.2009.05.013"
T2_GRID = dict(kind="log", n=60, from_s=1e-3, to_s=10.0)
SOLVER = "non-negative least squares with a second-difference penalty, lam = 0.1 (t2_distribution)"

#: Per rock: the Figshare record, its licence as the host states it, and Talabi's tabulated numbers.
ROCKS = {
    "LV60A": dict(doi="10.6084/m9.figshare.1153795.v2", version="v2", rho2=41e-6,
                  rho_note="fitted by him to the LV60Y CPMG (thesis §7.6.2)",
                  micro_ct_ms=512.0, measured_ms=496.0, porosity=0.377, s_over_v=57670,
                  direct_ms=487.6, direct_n=200_000, direct_dt=500e-6, sub_steps=1, illegal=37,
                  counters="imperial-rocks/family2.log:4 -- 'walked N=45,000 in 312 s; sub_steps 1; refused 37'",
                  verbatim="LV60A             496        512       565      "
                           "        32.2      35.3     27.2       4.8        4.9        3.8"),
    "F42A": dict(doi="10.6084/m9.figshare.1189259.v1", version="v1", rho2=41e-6,
                 rho_note="fitted by him to the F42Y CPMG (thesis §7.6.2)",
                 micro_ct_ms=677.0, measured_ms=668.0, porosity=0.330, s_over_v=43770,
                 direct_ms=679.0, direct_n=200_000, direct_dt=500e-6, sub_steps=1, illegal=40,
                 counters="imperial-rocks/f42a.log:4 -- 'walked N=45,000 in 311 s; sub_steps 1; refused 40'",
                 verbatim="F42A              668        677       756      "
                          "        42.0      59.0     61.5       5.2        5.8        3.6"),
}

#: The family's certified acquisition envelope -- what these packs say they are FOR, which is what
#: ``waveform_band`` then holds a consumer to. A CLINICAL diffusion envelope (b to 1000 s/mm^2, delta 40 ms,
#: Delta 80 ms, three directions, OGSE at 0.3-1 Hz) beside the contact tier at Talabi's two relaxivities, and
#: NOT the high-b, short-delta end of the default battery: the acquisition this family reproduces is 2 MHz
#: relaxometry, which plays no gradient at all, and over a 3 s window a b = 3000 s/mm^2 measurement is a fully
#: dephased signal whose Monte-Carlo value is noise.
ENVELOPE = dict(bvals=[0.0, 0.5e9, 1e9], dirs=[[0, 0, 1], [1, 0, 0], [1, 0, 1]],
                delta_frac=40e-3 / T, Delta_frac=80e-3 / T, ogse_periods=[1, 2, 3],
                shortd_b=1e9, shortd_deltas_frac=[60e-3 / T, 40e-3 / T], rho2_list=[15e-6, 41e-6])


#: The distinct deltas the envelope declares, as a set: `delta_frac` and the `shortd_deltas_frac` list name
#: 40 ms twice, and de-duplicating them inside `waveforms()` hid that the two fields overlap.
DECLARED_DELTA_FRACS = tuple(sorted({ENVELOPE["delta_frac"], *ENVELOPE["shortd_deltas_frac"]}))


def waveforms():
    """Every waveform the family DECLARES, which the pack stage checks against ``waveform_band``.

    The shortest delta is 40 ms because that is what the stored band serves: measured on this window at
    K = 128, delta 40 ms needs 17.2 Hz against the pack's 21.3 Hz, and 30 ms needs 22.8 and is refused.
    """
    from dmipy_sim import pgse
    out = []
    for frac in DECLARED_DELTA_FRACS:
        delta = frac * T
        Delta = max(2 * delta, ENVELOPE["Delta_frac"] * T)
        out.append((f"pgse-delta{delta * 1e3:.0f}ms",
                    pgse([[1, 0, 0]], delta, Delta, bvalues=[max(ENVELOPE["bvals"])], n_t=400)))
    return tuple(out)


# ----------------------------------------------------------------- the licence text, verbatim from the host
def licence_text(url, cache_dir):
    """The licence at ``url``, fetched once and cached: what the source record copies verbatim.

    Figshare states each record's licence as ``CC BY 4.0`` with this URL; the TEXT is what the record carries,
    because a title is not a licence.
    """
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "cc-by-4.0-legalcode.txt")
    if not os.path.exists(cache):
        req = urllib.request.Request(url, headers={"User-Agent": "dmipy-sim (https://github.com/dmrai-lab/dmipy-sim)"})
        with urllib.request.urlopen(req, timeout=60) as fh, open(cache, "wb") as out:
            out.write(fh.read())
    return open(cache, encoding="utf-8").read()


# ----------------------------------------------------------------- the spec producer
def central_crop(shape, n=CROP):
    lo = [(int(s) - int(n)) // 2 for s in shape]
    if any(v < 0 for v in lo):
        raise ValueError(f"an image of {shape} has no central {n}^3 cube")
    return tuple(lo) + tuple(v + int(n) for v in lo)


def spec_of(rock, data_dir):
    """The rock's substrate spec: the image cited by the name a dataset distributes it at, the crop, and
    Talabi's values as the NOMINAL ones a consumer overrides."""
    from dmipy_sim.io.label_volume import read_label_volume
    from dmipy_sim.spec import label_volume_spec
    ref = ROCKS[rock]
    path = os.path.join(data_dir, f"{rock}.nhdr")
    shape = read_label_volume(path).labels.shape
    return label_volume_spec(
        path, pools={0: "free", 1: "grain"}, crop=central_crop(shape), rho2=ref["rho2"], D=D0,
        T2_pools={"free": T2B, "grain": T2B}, cite_image_as=f"{rock}.nhdr", id=f"imperial2007/{rock.lower()}",
        source="Imperial College 2007 micro-CT collection (Dong & Blunt 2009), CC BY 4.0",
        description=(f"{rock}: the central {CROP}^3 crop of the released micro-CT image; the pore fluid walks "
                     f"between the voxel faces of the grain. Relaxation is not in the walk -- rho2 and T2 are "
                     f"replay knobs, and the nominal values are Talabi 2008's ({ref['rho_note']})."))


def walk(spec, n, *, n_t):
    """The walk the pilot runs and a fresh family would run: no relaxivity, C2 stored as a channel."""
    import dmipy_sim as d
    from dmipy_sim.spec import geometry_from_spec
    return d.simulate_trajectories(int(n), D0, geometry_from_spec(spec), T_max=(n_t - 1) * DT_SAVE,
                                   dt_save=DT_SAVE, seed=SEED, tiers="all", require_gpu=False,
                                   walker_batch_size=2000)


# ----------------------------------------------------------------- the quantity, replayed
def t2_grid(grid):
    return np.logspace(np.log10(grid["from_s"]), np.log10(grid["to_s"]), int(grid["n"]))


def decay_and_projection(pack, rho2, grid, *, chunk=4000):
    """``(t, S, projection)``: the decay ``S(t) = <exp((rho2/D) L(t) - t/T2B)>`` on the recorded sampling, and
    each walker's own decay projected on the log-mean's gradient -- what the delta method needs.

    ``pack.contact()`` is the pack's stored C2 channel as a per-save series; the walkers are read in chunks so
    no float64 copy of the whole walk is held.
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
        return np.exp(float(rho2) / D0 * L - t / T2B)

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
    """The T2 log-mean this pack serves, ON the grid the reference record states, with its analytic standard
    error by the delta method over walkers (no folds, no seed, no coverage factor)."""
    rho2 = float(pack.nominal.rho2)
    _t, _S, lm, se = decay_and_projection(pack, rho2, grid)
    return dict(value=lm * 1e3, se=se, se_kind="delta_method", solver=grid["solver"],
                se_derivation=("the delta method over walkers: the log-mean's gradient with respect to the "
                               "decay it is inverted from (examples.validation.talabi_micro_ct_rocks."
                               "log_mean_gradient, the active set's linear map), projected on each walker's "
                               "own decay; sd over walkers / sqrt(N) / T2lm"),
                rho_m_per_s=rho2, n_points=int(len(_t)))


def served_vs_channel(pack, *, n_points=8):
    """How far the signal the pack SERVES a consumer is from the one its decoded channel holds.

    The consumer route for a multi-TE relaxation measurement is one gradient-free spin echo per echo time
    (a single ``cpmg`` train is read at its last echo, not at every one); the cheap route above reads the same
    contact channel. This is the check that they are one quantity.
    """
    from dmipy_sim.engine.pulse_sequence import bare_spin_echo
    rho2 = float(pack.nominal.rho2)
    ell = pack.contact()
    n_w = ell.shape[0]
    step = max(1, pack.n_t // (n_points + 1))
    ks = np.asarray(list(range(step, pack.n_t, step))[:n_points])
    t = ks * pack.dt
    acc = np.zeros(len(ks))                                   # the channel's decay at those saves, chunked
    for lo in range(0, n_w, 4000):
        L = np.cumsum(np.asarray(ell[lo:lo + 4000], np.float64), axis=1)[:, ks]
        acc += np.exp(rho2 / D0 * L - t / T2B).sum(axis=0)
    S_ch = acc / n_w
    served = [float(np.asarray(pack.replay(bare_spin_echo(float(te), dt=pack.dt),
                                           tissue=pack.nominal)).ravel()[0]) for te in t]
    return float(np.max(np.abs(np.asarray(served) - S_ch)))


# ----------------------------------------------------------------- the "Use me" snippet
def snippet(uri):
    """One call that reproduces the T2 log-mean of the reproduction table, on its own grid."""
    return f'''import numpy as np
from dmipy_sim.replay import ReplayPack
from examples.validation.talabi_micro_ct_rocks import t2_distribution, log_mean_T2

pk = ReplayPack.load("{uri}")
ell = pk.contact()                                        # the stored wall-contact channel, per save
L = np.cumsum(np.asarray(ell, np.float64), axis=1)        # the boundary local time
t = np.arange(pk.n_t) * pk.dt
S = np.exp(pk.nominal.rho2 / {D0!r} * L - t / {T2B!r}).mean(axis=0)   # pk.nominal IS Talabi's rho2 and T2B
every = round({INVERT_MS!r}e-3 / pk.dt)                             # the 1 ms sampling the record states
g = np.logspace(-3, 1, 60)
print("T2 log-mean %.1f ms" % (1e3 * log_mean_T2(g, t2_distribution(t[every::every], S[every::every], g))))
'''


# ----------------------------------------------------------------- the family
def family(data_dir, work_dir, *, rocks, dry, create_dataset):
    cache = os.path.join(work_dir, ".licence-cache")
    text = licence_text("https://creativecommons.org/licenses/by/4.0/legalcode.txt", cache)
    sources = [Source(
        key=f"figshare-{rock.lower()}", url=f"https://doi.org/{ROCKS[rock]['doi']}",
        host_record=f"Figshare record {ROCKS[rock]['doi']} ({ROCKS[rock]['version']})",
        licence_id=LICENCE, licence_url="https://creativecommons.org/licenses/by/4.0/", licence_text=text,
        files=(SourceFile(path=os.path.join(data_dir, f"{rock}.nhdr"), cite_as=f"{rock}.nhdr",
                          role="the NRRD header of the segmented micro-CT image"),
               SourceFile(path=os.path.join(data_dir, f"{rock}.raw"), cite_as=f"{rock}.raw",
                          role="its payload: one byte per voxel, 0 pore and 1 grain"))) for rock in rocks]

    grid = dict(T2_GRID, sample_ms=INVERT_MS, solver=SOLVER)
    _se = {rock: direct_se(rock, work_dir, grid) for rock in rocks}      # measured, never typed
    quantities = tuple(ReferenceQuantity(
        substrate=rock, name="T2_log_mean",
        published=Published(
            value=ROCKS[rock]["micro_ct_ms"], unit="ms", uncertainty=0.0,
            uncertainty_is=("none stated: Table 7-2 prints one figure per sample with no error bar and the "
                            "thesis quotes none, so the gate's budget rests on our own standard error alone"),
            printed_in="Talabi O (2008), Pore Scale Simulation of NMR Response in Porous Media, Imperial "
                       "College London PhD thesis",
            document=THESIS_DOI,
            locator="Table 7-2, p. 63, the 'Micro-CT' column of Mean T2 (ms)",
            verbatim=ROCKS[rock]["verbatim"]),
        direct=Direct(
            value=ROCKS[rock]["direct_ms"], unit="ms",
            se=_se[rock]["se"], se_kind="delta_method",
            se_derivation=(
                "the delta method over walkers on the log-mean (talabi_micro_ct_rocks.log_mean_gradient), "
                f"MEASURED here at record time on this family's own {rock.lower()}.rpk -- "
                f"{_se[rock]['se_pack']:.6f} over {_se[rock]['n_pack']:,} walkers at the same rho2, channel and "
                f"grid -- and scaled to the direct walk's {_se[rock]['n_direct']:,} by 1/sqrt(N) "
                f"(x {_se[rock]['scale']:.6f}). The direct walk's own per-walker decays were not retained; the "
                "relative per-echo spread is a property of the substrate and the estimator, not of the pack."),
            n_walkers=ROCKS[rock]["direct_n"], grid=grid, solver=SOLVER,
            source="examples/validation/talabi_micro_ct_rocks.py (run_rock), asserted by "
                   "tests/validation/test_talabi_rocks.py")) for rock in rocks)

    reference = Reference(
        doi=REFERENCE_DOI, title="Pore-scale simulation of NMR response",
        published_kind="number",
        sample=("the same released micro-CT images: his 'Micro-CT' column is a random walk on these very "
                "volumes, on the central 300^3 section (App. A-1), and his 'Experiment' column is a CPMG "
                "measured on the sand the images are of"),
        sample_relation="the same object",
        quantities=quantities,
        parameters=(
            FreeParameter(name="rho2", value=41e-6, unit="m/s", whose="theirs",
                          where="thesis §7.6.2", how="fitted by him to his own LV60Y / F42Y CPMG; an effective "
                                                     "value tied to his ~10 um voxel, because a segmentation's "
                                                     "surface is the Manhattan area at that resolution, which is "
                                                     "what both walks relax against"),
            FreeParameter(name="D0", value=D0, unit="m^2/s", whose="theirs",
                          where="thesis eq. 7.1", how="the brine's self-diffusion at 308 K, stated"),
            FreeParameter(name="T2B", value=T2B, unit="s", whose="theirs",
                          where="thesis eq. 7.2", how="the brine's bulk T2, stated; applied at replay, not in "
                                                      "the walk"),
            FreeParameter(name="crop", value=f"central {CROP}^3", unit="voxels", whose="ours",
                          changes_geometry=True,
                          where="thesis App. A-1 states the size, not the offset",
                          how="a central cube: over all 3,442,951 possible 300^3 crops of LV60A the porosity "
                              "spans [0.3555, 0.3713] and his 0.377 is not reachable, so the offset is ours"),
            FreeParameter(name="T2 inversion", value=SOLVER, unit="-", whose="ours",
                          where="thesis App. A-3 (after Chen et al. 1999)",
                          how="his is an unconstrained normal-equation solve with a fourth-derivative penalty "
                              "on 100 points from 0.1 ms; ours is non-negative with a second-difference "
                              "penalty on 60 points from 1 ms, so the log-mean is comparable to his Table 7-2 "
                              "rather than computed the same way"),
            FreeParameter(name="surface rule", value="exp(-2 (rho2/D) d_perp) per reflection", unit="-",
                          whose="ours", where="thesis §3.5 for his",
                          how="he kills a walker at an attempted move into a grain voxel with probability "
                              "2 rho2 s / (3 D); this walk weights it per specular reflection. The two agree in "
                              "the continuum limit")),
        description=("Micro-CT sand packs and a sandstone walked on their own voxels through the label-volume "
                     "producer: the surface-relaxation reference outside the brain, replayed as a CPMG decay. "
                     "Relaxation is not in the walk -- one pack answers every relaxivity."),
        source_note=("Dong & Blunt 2009 (Phys Rev E 80:036307) images, Imperial College 2007 collection on "
                     "Figshare; Talabi 2008 (Imperial College PhD thesis) for the reference random walk and "
                     "the measured CPMG, published as Talabi et al., J. Pet. Sci. Eng. 2009."),
        licence_note="CC-BY-4.0 images; the thesis is CC-BY-ND and its numbers are cited, never redistributed; "
                     "the packs are CC-BY-4.0",
        caveats=dict(
            LV60A_crop="Talabi's 0.377 porosity is not reachable on the released LV60A image: over all "
                       "3,442,951 possible 300^3 crops the porosity spans [0.3555, 0.3713]. The offset he used "
                       "is not recorded in the thesis, and a central cube is what his other six samples' "
                       "porosities support. That is the whole of LV60A's porosity gap, and it is not the walk.",
            inversion="The log-mean is the weakest link in this comparison, and every number here is on a 1 ms "
                      "sampling of the decay. Measured on F42A, which still holds 3.3 % of its signal at 3 s: "
                      "674.4 ms at 1 ms sampling, 668.0 at 10 ms, 309.7 at 20 ms -- a truncated decay lets the "
                      "regularised solve split amplitude between a short and a very long component. LV60A, down "
                      "to 1.0 %, is 487.7 / 487.6 / 487.4 at the same three. Over lam in [0.003, 1] and five "
                      "T2 grids the log-mean spans 3.1 % on LV60A and 4.2 % on Berea.",
            band="This is a RELAXATION pack. Its 200 us save grid is 22x coarser than the #143 rule asks of a "
                 "Connectom, deliberately: the acquisition it reproduces carries no gradient at all. What keeps "
                 "that honest is the pack's own band -- 21.33 Hz over the walk at K = 128, small and stated -- "
                 "and a consumer replaying a gradient beyond it is refused by waveform_band rather than served "
                 "a wrong number."))

    design = Design(
        window_s=T, dt_save_s=DT_SAVE,
        save_grid_why=("the MARAN2 train's inter-echo time (thesis §7.3), so every echo of the train Talabi "
                       "compared against falls on a save. The window holds 15,000 echoes; a 32,000-echo "
                       "(6.4 s) train is beyond it and is refused rather than truncated"),
        K=K_BANDS, envelope=ENVELOPE, waveforms=waveforms,
        tiers=(Tier(name="contact", floor_key="floor_surface", err_key="err_surface", target_floor=SIGMA),
               Tier(name="positions", floor_key="floor_max", err_key="err_max", target_floor=SIGMA)),
        memory_budget_bytes=BUDGET_BYTES, pilot_n=PILOT_N, safety=1.4,
        false_failure_rate=FALSE_FAILURE_RATE,
        tolerance=Tolerance(terms=("quantity.direct.se", "quantity.replay_se",
                                   "quantity.published.uncertainty",
                                   "design.systematics.save_grid_step.value")),
        systematics=(Systematic(
            name="save_grid_step", value=3.1e-3,
            measured_on="LV60A, before any pack existed",
            evidence="the log-mean of the same walk at dt 500 / 250 / 125 us is 487.6 / 487.0 / 486.1 ms, a "
                     "relative shift of 3.1e-3 over a fourfold refinement. The direct walk steps at 500 us and "
                     "a pack's at 200 us, so this is part of the comparison's budget and is not Monte-Carlo "
                     "noise that more walkers would remove."),))

    build = Build(
        specs={rock: (lambda r=rock: spec_of(r, data_dir)) for rock in rocks},
        pack_id={rock: f"imperial-rocks/{rock.lower()}" for rock in rocks},
        reproduce=reproduce, served_vs_channel=served_vs_channel, served_tier="contact", walk=walk,
        recorded={rock: RecordedWalk(pack_path=os.path.join(work_dir, "packs", f"{rock.lower()}.rpk"),
                                     sub_steps=ROCKS[rock]["sub_steps"],
                                     illegal_crossings=ROCKS[rock]["illegal"],
                                     evidence=ROCKS[rock]["counters"]) for rock in rocks})

    publication = Publication(repo=REPO, licence=LICENCE, citation=CITATION, snippet=snippet,
                              snippet_substrate=rocks[0],
                              pack_path=lambda name: f"packs/{name.lower()}.rpk",
                              create_dataset=create_dataset, dry=dry)
    return ReferenceFamily("imperial-rocks", work_dir, sources=sources, reference=reference, design=design,
                           build=build, publication=publication)


def direct_se(rock, work_dir, grid):
    """The DIRECT walk's relative delta-method standard error on the log-mean, MEASURED rather than typed.

    It was a hand-written pair of numbers feeding the gate's own tolerance. It is now measured here, at record
    time, on this family's own pack -- the same substrate, the same rho2, the same channel and the same grid --
    and scaled to the direct walk's walker count by 1/sqrt(N), because the direct walk's per-walker decays were
    not retained. The relative per-echo spread is a property of the substrate and the estimator, not of the
    pack, which is what makes the substitution legitimate; the record states it and the numbers it rests on.
    """
    import hashlib
    import json
    from dmipy_sim.fill.hub import sha256_of
    from dmipy_sim.replay import ReplayPack
    ref = ROCKS[rock]
    path = os.path.join(work_dir, "packs", f"{rock.lower()}.rpk")
    # keyed on the pack's own bytes and the grid, so the measurement is made once and re-made the moment either
    # changes; every stage of a run would otherwise decode both contact channels again
    key = hashlib.sha256(json.dumps([sha256_of(path), grid], sort_keys=True).encode()).hexdigest()
    cache_path = os.path.join(work_dir, ".direct-se-cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    if cache.get(rock, {}).get("key") == key:
        return cache[rock]["value"]
    pack = ReplayPack.load(path)
    _t, _S, _lm, se_pack = decay_and_projection(pack, float(ref["rho2"]), grid)
    n_pack = int(pack.n_walkers)
    scale = (n_pack / float(ref["direct_n"])) ** 0.5
    value = dict(se=se_pack * scale, se_pack=se_pack, n_pack=n_pack, n_direct=int(ref["direct_n"]), scale=scale,
                 measured_on=os.path.basename(path), pack_sha256=sha256_of(path))
    cache[rock] = dict(key=key, value=value)
    with open(cache_path, "w") as fh:
        json.dump(cache, fh, indent=1, sort_keys=True)
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="the directory holding <rock>.nhdr + <rock>.raw")
    p.add_argument("--work", required=True, help="the family directory (records/, previews/, packs/, README.md)")
    p.add_argument("--rocks", nargs="+", default=["LV60A", "F42A"], choices=sorted(ROCKS))
    p.add_argument("--stage", default=None, help="run one stage (the protocol's order still holds)")
    p.add_argument("--dry", action="store_true", help="gate and render, upload nothing")
    p.add_argument("--create-dataset", action="store_true", help="create the dataset repository (explicit)")
    a = p.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("dmipy_sim").setLevel(logging.INFO)
    os.environ.setdefault("DMIPY_SIM_SURFACE_DIR", os.path.abspath(a.data))
    fam = family(a.data, os.path.abspath(a.work), rocks=list(a.rocks), dry=a.dry,
                 create_dataset=a.create_dataset)
    if a.stage:
        fam.stage(a.stage)
    else:
        fam.run()
    print(f"records in {fam.records.dir}")


if __name__ == "__main__":
    main()
