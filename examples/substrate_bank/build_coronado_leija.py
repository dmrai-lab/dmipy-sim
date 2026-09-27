"""Coronado-Leija et al. 2024's injured rat white matter as a reference family: 3d EM of the same five brains
the ex vivo dMRI was measured on, both CC BY 4.0, run through the reference-pack protocol (#482, #480).

Ten SBEM volumes of rat corpus callosum and cingulum, segmented into myelin, myelinated axons and cell
nuclei by DeepACSON and released on Fairdata, beside the 9.4 T ex vivo dMRI of the very brains they were cut
from -- the same-specimen, diffusion-weighted, released-geometry pair that dmipy-sim#476 §4.4 names as public
and never simulated. This file is the family's DECLARATION: its three sources, the intra-axonal water
fraction ``f`` per region of interest as the quantity it reproduces, and the design the packs will be built
to.

**Nothing is walked here and no pack exists.** Two things must land first, and each is a refusal this module
raises BY NAME rather than a note to a reader:

* the **walk** needs the adaptive surface step of dmipy-sim#478. The surface-local-time rule of
  :func:`~dmipy_sim.engine.physics.resolve_sub_steps` divides the 50 nm voxel by 8, and measured with that
  rule on this substrate one 625 us save costs 48,001 sub-steps, one 25 us save 1,921, and the 15 nm
  high-resolution tier is refused outright above a 100 us save (853,334 against the 100,000 cap). ``walk()``
  states those numbers and refuses.
* the **spec** needs a label-volume producer that composes several released files into one graded volume and
  that can hold water in two pools. :func:`~dmipy_sim.spec.label_volume_spec` reads ONE file, maps at most
  256 label values and gives ``water_fraction`` 1.0 to exactly one pool; this substrate is three files
  (``*_myelin.mat``, ``*_myelinated_axons.mat``, ``*_nucleus.mat``), its axon file is ``uint16`` instance
  labels, and its water is intra- AND extra-axonal with myelin invisible. ``spec_of()`` calls the producer,
  lets it refuse in its own words, and re-raises naming all three limits.

The source and reference stages are complete and run::

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 nice -n 10 \\
      python examples/substrate_bank/build_coronado_leija.py --data ~/data/coronado2024 \\
      --work ~/coronado-leija-2024 --stage source     # then reference
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from dmipy_sim.replay.reference import (Build, Design, Direct, FreeParameter, Published, Reference,
                                        ReferenceFamily, ReferenceRefusal, ReferenceQuantity, Publication,
                                        Source, SourceFile, Tier, Tolerance, crossref)

REPO = "SubstrateCommons/coronado-leija-2024"
LICENCE = "CC-BY-4.0"
CITATION = (
    "Coronado-Leija R, Abdollahzadeh A, Lee HH, Coelho S, Ades-Aron B, Liao Y, Salo RA, Tohka J, Sierra A, "
    "Novikov DS & Fieremans E (2024), Volume electron microscopy in injured rat brain validates white matter "
    "microstructure metrics from diffusion MRI, Imaging Neuroscience 2, doi:10.1162/imag_a_00212; "
    "Abdollahzadeh A, Belevich I, Jokitalo E, Tohka J & Sierra A (2021), the 3d EM segmentation, "
    "doi:10.23729/bad417ca-553f-4fa6-ae0a-22eddd29a230 (CC BY 4.0); Coronado-Leija R et al. (2024), the ex "
    "vivo dMRI, doi:10.23729/0f41bbec-78cc-43f0-b95a-9e9657cbd2c1 (CC BY 4.0); "
    "Fick RHJ (2026), the replay packs, SubstrateCommons")

PAPER_DOI = "10.1162/imag_a_00212"
PAPER_TITLE = ("Volume electron microscopy in injured rat brain validates white matter microstructure metrics "
               "from diffusion MRI")
EM_DOI = "10.23729/bad417ca-553f-4fa6-ae0a-22eddd29a230"
EM_URL = "https://etsin.fairdata.fi/dataset/f8ccc23a-1f1a-4c98-86b7-b63652a809c3"
EM_RECORD = "Fairdata Etsin f8ccc23a-1f1a-4c98-86b7-b63652a809c3, published 2021-01-08, modified 2022-04-21"
MRI_DOI = "10.23729/0f41bbec-78cc-43f0-b95a-9e9657cbd2c1"
MRI_URL = "https://etsin.fairdata.fi/dataset/29e410dc-5d58-45ac-a716-ce1a38b46037"
MRI_RECORD = "Fairdata Etsin 29e410dc-5d58-45ac-a716-ce1a38b46037, published 2024-01-31, modified 2024-02-09"
NYU_URL = "https://github.com/NYU-DiffusionMRI/Standard-Model-validation-using-3d-EM"
NYU_COMMIT = "479a580202b67357e5119bb667f10550aea8c4ae"

#: The ten released white-matter samples, in the order the authors' own scripts loop over them
#: (``script03_compute_volume_fractions.m``), which is the column order of the ROI-limits file.
SAMPLES = ("Sham_25_contra", "Sham_25_ipsi", "Sham_49_contra", "Sham_49_ipsi", "TBI_24_contra", "TBI_24_ipsi",
           "TBI_28_contra", "TBI_28_ipsi", "TBI_2_contra", "TBI_2_ipsi")

#: The five animals: three lateral-fluid-percussion injuries and two sham operations (paper §2.1), and the
#: specimen directory of the dMRI release that is the SAME brain.
ANIMALS = {"25": "sham-operated", "49": "sham-operated", "2": "traumatic brain injury",
           "24": "traumatic brain injury", "28": "traumatic brain injury"}

#: The label values ``RefineSeg.SetSubstrate`` writes, and the pool each one is. The insertion order is the
#: pool id, so the free (extra-axonal) pool is pool 0, which is the ``.rpk`` convention every channel is
#: indexed by. Myelin is water-free: its T2 is far below the 35 ms echo time of the acquisition this family
#: reproduces, which is the authors' own reason for leaving it out of ``f`` (paper §2.5.2).
POOLS = {1: "extra", 0: "myelin", 2: "intra", 3: "nucleus"}

#: m: the released low-resolution voxel, isotropic (README, paper §2.3).
VOXEL_LM = (50e-9, 50e-9, 50e-9)
#: m: the released high-resolution voxel, anisotropic in z (README, paper §2.3).
VOXEL_HM = (15e-9, 15e-9, 50e-9)

D0 = 2.0e-9         #: m^2/s: free water at the 21 C the brains were scanned at -- THEIR value (paper §2.5.3)
TE = 35e-3          #: s: the echo time of the released acquisition (paper §2.2), and this family's window
DELTA = 6e-3        #: s: the released pulse length (paper §2.2)
BIG_DELTA = 11.5e-3  #: s: the released inter-pulse duration (paper §2.2)
BVALS = (0.0, 2e9, 3e9, 4e9)   #: s/m^2: the three released shells, b = 2, 3, 4 ms/um^2, plus b0
N_DIRS = 43         #: directions per shell, uniform on the half sphere (paper §2.2)
DT_SAVE = 25e-6     #: s: the save grid (see ``SAVE_GRID_WHY``)
K_BANDS = 128       #: bands over the window
SIGMA = 5e-3        #: the floor per tier this family targets
BUDGET_BYTES = 60_000_000_000
PILOT_N = 4000
FALSE_FAILURE_RATE = 0.01

#: The direct measurement's own grid: how many uniform points the engine classifies, and how the volume is
#: read. The slab axis is the MATLAB ``y`` axis because that is the released files' HDF5 chunk axis
#: (``chunks=(nz, 1, 105)``), so a slab is whole chunks and the estimator is the same one it would be on the
#: whole volume: a point uniform in a region is a point uniform in one of its slabs, drawn in proportion to
#: the slab's volume.
N_POINTS = 4_000_000
SEED = 0
SLAB_Y = 64

#: The step counts of :func:`dmipy_sim.engine.physics.resolve_sub_steps` on this substrate, MEASURED with the
#: engine's own rule at ``D0`` and quoted by ``walk()``'s refusal. ``None`` is a save grid the rule refuses
#: outright (above ``MAX_SUB_STEPS`` = 100,000).
SUB_STEPS = {("lm", 25e-6): 1_921, ("lm", 100e-6): 7_681, ("lm", 625e-6): 48_001, ("lm", 1e-3): 76_801,
             ("hm", 25e-6): 21_334, ("hm", 100e-6): 85_334, ("hm", 625e-6): None, ("hm", 1e-3): None}

SAVE_GRID_WHY = (
    "the released acquisition's shortest gradient feature is the delta = 6 ms pulse, and 25 us samples it 240 "
    "times; over the TE = 35 ms window that is 1,401 saves. It is also the coarsest grid at which the "
    "high-resolution 15 nm tier is not refused by the sub-step rule (21,334 sub-steps per save against the "
    "100,000 cap, where a 625 us grid needs 533,334)")

#: The relaxivities the CONTACT tier's certificate is probed at. They are a probe of the channel and not a
#: physical claim: this family fits no surface relaxivity and no permeability -- it stores the boundary local
#: time so that both are replay knobs, which is the only way a white-matter pack can answer the axolemma
#: exchange question at all -- and the ladder is the bank's own default, which brackets the membrane
#: permeabilities the exchange literature reports (1e-6 to 1e-4 m/s) at the scale the channel has to be
#: accurate at. A ``rho_list`` of ``[0.0]`` would make the tier's floor and codec error identically zero and
#: its ``meets_target`` vacuously true, which is a certificate that certifies nothing.
RHO_PROBE = (1e-5, 3e-5, 1e-4)

ENVELOPE = dict(bvals=list(BVALS), dirs=[[0, 0, 1], [1, 0, 0], [1, 0, 1]],
                delta_frac=DELTA / TE, Delta_frac=BIG_DELTA / TE, ogse_periods=[1, 2, 3],
                shortd_b=max(BVALS), shortd_deltas_frac=[DELTA / TE], rho_list=list(RHO_PROBE))


# ----------------------------------------------------------------- the regions of interest, as released
def roi_limits(path):
    """``{substrate: {sample, roi, crop, shape, n_axons, n_nuclei}}`` from the authors' own ROI-limits file.

    ``000_animal_exp_axons_cells_nx_ny_nz_cclims_cglims.txt`` is 19 rows by 10 columns, one column per
    sample in the order of :data:`SAMPLES`: the animal id, the hemisphere (0 contra, 1 ipsi), the number of
    segmented axons and nuclei, the volume's ``nx, ny, nz``, then the corpus-callosum and cingulum limits as
    ``x0 x1 y0 y1 z0 z1``, one-based and inclusive. ``script03_compute_volume_fractions.m`` keeps a region
    only when its voxel count is above 10, which is how the ipsilateral corpus callosum of the sham #49 rat
    -- imaged as cingulum only, the limits written as ``1 2`` on every axis -- excludes itself.

    ``crop`` is returned zero-based and half-open, in the ``(nx, ny, nz)`` index order of the volume.
    """
    rows = np.loadtxt(path)
    if rows.shape != (19, len(SAMPLES)):
        raise ReferenceRefusal(f"source: {path} is {rows.shape}, not (19, {len(SAMPLES)}); the ROI limits are "
                               f"19 rows by one column per sample and the file is not the one this family read")
    out = {}
    for col, sample in enumerate(SAMPLES):
        c = rows[:, col]
        animal, ipsi = str(int(c[0])), int(c[1])
        if not sample.endswith("ipsi" if ipsi else "contra") or f"_{animal}_" not in sample:
            raise ReferenceRefusal(f"source: column {col} of {path} is animal {animal} "
                                   f"{'ipsi' if ipsi else 'contra'}, the sample order says {sample}")
        shape = tuple(int(v) for v in c[4:7])
        for roi, lims in (("cc", c[7:13]), ("cg", c[13:19])):
            x0, x1, y0, y1, z0, z1 = (int(v) for v in lims)
            crop = (x0 - 1, y0 - 1, z0 - 1, x1, y1, z1)
            if (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1) <= 10:     # their own rule, in their own words
                continue
            out[f"{sample.lower()}_{roi}"] = dict(
                sample=sample, animal=animal, condition=ANIMALS[animal],
                hemisphere="ipsilateral" if ipsi else "contralateral", roi=roi, crop=crop, shape=shape,
                n_axons=int(c[2]), n_nuclei=int(c[3]))
    return out


def em_files(data_dir, sample, resolution="LM"):
    """The released segmentation files of one sample at one resolution, by role.

    The low-resolution samples carry myelin, myelinated axons and cell nuclei; the high-resolution ones carry
    myelin and myelinated axons only (README). One file of the release is named ``mat_LM_28_ipsi_myelin.mat``
    where every other is ``LM_28_ipsi_myelin.mat``, which is why the myelin path is looked up rather than
    composed.
    """
    d = os.path.join(data_dir, "em", "White matter EM", sample)
    stem = f"{resolution}_{sample.split('_', 1)[1]}"
    out = {}
    for role, suffix in (("myelin", "_myelin.mat"), ("axons", "_myelinated_axons.mat"),
                         ("nucleus", "_nucleus.mat")):
        if role == "nucleus" and resolution == "HM":
            continue
        for name in (stem + suffix, "mat_" + stem + suffix):
            if os.path.exists(os.path.join(d, name)):
                out[role] = os.path.join(d, name)
                break
        else:
            raise ReferenceRefusal(f"source: neither {stem + suffix} nor mat_{stem + suffix} is in {d}; a "
                                   f"source file that is not on disk cannot be digested")
    return out


def mri_files(data_dir, animal):
    """The released ex vivo dMRI of one brain: the preprocessed volumes, their scheme, the masks and the ROIs.

    These are the MEASUREMENT half of the pair. No stage of this family reads them yet -- the signal the
    packs will be gated against is theirs, and the source record digests them now so that the quantity can be
    added without a second source record.
    """
    root = os.path.join(data_dir, "mri", "Ex vivo diffusion", "MRI_data")
    pre = os.path.join(root, "preproc", animal)
    out = {f"preproc/{n}": os.path.join(pre, n) for n in sorted(os.listdir(pre))} if os.path.isdir(pre) else {}
    rois = os.path.join(root, "rois", animal)
    out.update({f"rois/{n}": os.path.join(rois, n) for n in sorted(os.listdir(rois))} if os.path.isdir(rois)
               else {})
    if not out:
        raise ReferenceRefusal(f"source: neither {pre} nor {rois} holds a file; the dMRI half of the pair is "
                               f"not on disk and cannot be digested")
    return out


# ----------------------------------------------------------------- the licence text, verbatim from the host
def licence_text(url, cache_dir, name):
    """The licence at ``url``, fetched once and cached: what the source record copies verbatim.

    Both Fairdata records state ``Creative Commons Attribution 4.0 International (CC BY 4.0)`` with this URL,
    and the EM release's own README repeats the grant in its "Rights and permissions" paragraph. The TEXT is
    what the record carries, because a title is not a licence.
    """
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, name)
    if not os.path.exists(cache):
        req = urllib.request.Request(url, headers={"User-Agent": "dmipy-sim (https://github.com/dmrai-lab/dmipy-sim)"})
        with urllib.request.urlopen(req, timeout=60) as fh, open(cache, "wb") as out:
            out.write(fh.read())
    return open(cache, encoding="utf-8", errors="replace").read()


# ----------------------------------------------------------------- the spec producer, as far as it goes
def spec_of(substrate, data_dir, rois):
    """The substrate spec of one region of interest -- which the label-volume producer cannot write.

    It calls :func:`~dmipy_sim.spec.label_volume_spec` on the released axon file so that the producer states
    its own limit, and re-raises with all three: this substrate is THREE released files where the producer
    reads one, its axon file is ``uint16`` instance labels (up to 67,244 axons per sample) where a label
    volume is ``0..255``, and its water is the intra- AND extra-axonal pool with myelin invisible where the
    producer gives ``water_fraction`` 1.0 to exactly one pool.
    """
    from dmipy_sim.spec import label_volume_spec
    r = rois[substrate]
    files = em_files(data_dir, r["sample"])
    try:
        return label_volume_spec(files["axons"], format="hdf5", pools=POOLS, voxel_size=VOXEL_LM,
                                 crop=r["crop"], D=D0, id=f"coronado2024/{substrate}")
    except Exception as e:
        raise ReferenceRefusal(
            f"spec {substrate!r}: label_volume_spec cannot write this substrate -- {type(e).__name__}: "
            f"{str(e)[:200]}. Three limits, all of the producer and none of the release: (1) the segmentation "
            f"is {len(files)} released files ({', '.join(sorted(files))}) and the producer reads ONE, citing "
            f"one sha256; (2) {os.path.basename(files['axons'])} is uint16 INSTANCE labels, "
            f"{r['n_axons']:,} axons in this sample, where a label volume is 0..255, so the pools have to be "
            f"composed before the producer sees them; (3) the water of a white-matter substrate is the "
            f"intra-axonal AND the extra-axonal pool with myelin invisible, and the producer gives "
            f"water_fraction 1.0 to exactly one pool with one rho and one kappa on every face, so it cannot "
            f"hold a reflecting myelin sheath and a permeable axolemma at once. A producer that composes "
            f"several released label files into one graded volume with water in more than one pool is what "
            f"this family's spec stage needs") from e


def walk(spec, n, *, n_t):
    """The walk this family will run, and the refusal it is until dmipy-sim#478 lands.

    ``resolve_sub_steps``' surface-local-time rule divides the smallest feature -- here the voxel, since a
    segmentation can express no pore narrower than one -- by 8, and the whole walk is then stepped at that
    length whether or not a walker is anywhere near a wall. Measured with the engine's own rule at
    ``D0 = 2 um^2/ms`` on a 50 nm label volume: 48,001 sub-steps per 625 us save, 1,921 per 25 us save (this
    family's grid) and so 2,690,000 steps per walker over the TE = 35 ms window; on the 15 nm
    high-resolution tier, 21,334 per 25 us save and a refusal above 100 us. Without the surface tier the
    reflection rule asks 121 per 25 us save -- sixteen times less -- which is where the cost lives and what
    an adaptive rule has to recover.
    """
    lm, hm = SUB_STEPS[("lm", 625e-6)], SUB_STEPS[("hm", 25e-6)]
    raise ReferenceRefusal(
        f"walk: this family is not walked until dmipy-sim#478 (the adaptive surface step) lands. The "
        f"surface-local-time rule of resolve_sub_steps needs {lm:,} sub-steps per 625 us save on the "
        f"released 50 nm segmentation and {SUB_STEPS[('lm', DT_SAVE)]:,} on this family's {DT_SAVE * 1e6:g} us "
        f"grid -- {int(round(TE / DT_SAVE)) * SUB_STEPS[('lm', DT_SAVE)]:,} steps per walker over the "
        f"{TE * 1e3:g} ms window -- and {hm:,} per save on the 15 nm tier, which it refuses above a 100 us "
        f"save. The rule is uniform in space; the substrate is not, and #478 is where the fine step becomes "
        f"a near-wall step. Nothing about the geometry, the envelope or the walker count is waiting: this "
        f"refusal is the pilot's, and the design record cannot be written without it")


def reproduce(pack, quantity, grid):
    """The reproduction of ``f`` from a pack, which no pack of this family can serve yet.

    ``f`` is the intra-axonal share of the dMRI-visible water, so it is the weight with which this family's
    two packs of one region -- the intra-axonal walk and the extra-axonal one -- are summed into the region's
    signal. Reading it back needs both, and the second one needs a producer that can hold water in two pools
    (``spec_of``) as much as the first needs #478 (``walk``).
    """
    raise ReferenceRefusal(
        "pack: this family has no pack. `f` is the weight joining a region's intra-axonal pack to its "
        "extra-axonal one, so reproducing it needs the pair; the walk of either is refused until "
        "dmipy-sim#478, and the extra-axonal half also needs the multi-water-pool producer `spec_of` names. "
        "An estimator written against a pack that does not exist is the fourth copy of an estimator that was "
        "never executed (docs/reference-family.md)")


def served_vs_channel(pack, *, n_points=8):
    """How far the signal a pack serves is from its decoded channel -- undefined until a pack exists."""
    raise ReferenceRefusal("pack: this family has no pack, so there is no served signal to compare with a "
                           "decoded channel; the walk is refused until dmipy-sim#478")


def snippet(uri):
    """The card's "Use me" snippet, which is not written until there is a pack to read."""
    raise ReferenceRefusal(f"card: there is no pack at {uri} to teach a consumer to read; a snippet is "
                           f"written when the gate has passed a pack, and this family's walk is refused "
                           f"until dmipy-sim#478")


def waveforms():
    """The released acquisition itself: one PGSE at the published delta, Delta and highest shell."""
    from dmipy_sim import pgse
    return (("pgse-delta6ms-Delta11.5ms",
             pgse([[1, 0, 0]], DELTA, BIG_DELTA, bvalues=[max(BVALS)], n_t=400)),)


# ----------------------------------------------------------------- the quantity, measured
def _dataset(path):
    """The one array in a v7.3 MAT-file, by name: ``myelin`` for the myelin masks, ``final_lbl`` for the
    instance segmentations. A file with more than one array is refused rather than guessed at."""
    import h5py
    with h5py.File(path, "r") as fh:
        names = [n for n in fh if isinstance(fh[n], h5py.Dataset) and fh[n].ndim == 3]
    if len(names) != 1:
        raise ReferenceRefusal(f"{path}: holds {names} 3-D arrays; a released segmentation is one array")
    return names[0]


def _slab(path, name, crop, y0, y1):
    """One y-slab of a released segmentation, cropped and transposed into ``(x, y, z)`` index order.

    h5py sees a MATLAB array with its axes reversed, so the released ``(nx, ny, nz)`` volume is an HDF5
    ``(nz, ny, nx)`` dataset and the crop is applied in that order before the transpose.
    """
    import h5py
    x0, ys, z0, x1, ye, z1 = crop
    with h5py.File(path, "r") as fh:
        a = fh[name][z0:z1, ys + y0:ys + y1, x0:x1]
    return np.transpose(a, (2, 1, 0))


def measure_roi(substrate, data_dir, rois, *, n_points=N_POINTS, seed=SEED, slab_y=SLAB_Y):
    """``f`` of one region, twice: by the authors' voxel count and by this engine's own classifier.

    The **published** value is their recipe on their released arrays -- "the intra-axonal, extra-axonal, and
    myelin volumes were computed as the number of voxels covering each compartment [and] f was computed as
    the intra-axonal volume divided by the total volume of the sample, excluding the myelin compartment"
    (paper §2.5.2), the compartments composed in the order ``RefineSeg.SetSubstrate`` writes them (extra 1,
    myelin 0, nuclei 3, axons 2, each overwriting the last), on the region's own released limits.

    The **direct** value is ours: ``n_points`` points uniform in the region, classified by
    :meth:`~dmipy_sim.geometry.label_volume.LabelVolume.classify_positions_exact` -- the engine's own
    position-to-pool map, with the origin, the voxel size and the index arithmetic it walks with -- and
    ``f = n_intra / (n_points - n_myelin)``. Its standard error is the binomial one of that ratio,
    ``sqrt((1 - f) / (f N))`` relative, over the ``N`` points that are not myelin; the volume is read in
    slabs of ``slab_y`` along its chunk axis and a point is classified by the slab it falls in, which is the
    same estimator as on the whole volume and bounds the memory at a few hundred megabytes.

    The two agree only if the engine puts a coordinate in the voxel the count put it in. That is the whole of
    what this measurement is for, and it is the check that has failed before on a real substrate: a
    metre-scale mesh whose predicates were silently wrong (#213), a box mirror and an interpolated normal
    that leaked walkers out of a sealed geometry (#213's three defects), a float32 rotation that moved 48-91 %
    of walkers out of their cylinders.
    """
    from dmipy_sim.geometry.label_volume import LabelVolume
    r = rois[substrate]
    files = em_files(data_dir, r["sample"])
    names = {k: _dataset(p) for k, p in files.items()}
    x0, y0, z0, x1, y1, z1 = r["crop"]
    shape = (x1 - x0, y1 - y0, z1 - z0)
    vox = np.asarray(VOXEL_LM)
    origin = -0.5 * np.asarray(shape) * vox          # centred: a far-from-origin volume cannot be walked
    rng = np.random.default_rng(seed)
    pts = origin + rng.random((int(n_points), 3)) * (np.asarray(shape) * vox)
    jy = np.minimum(((pts[:, 1] - origin[1]) / vox[1]).astype(np.int64), shape[1] - 1)

    counts = np.zeros(len(POOLS), np.int64)          # the voxel census, by pool id
    classified = np.zeros(len(POOLS), np.int64)      # the engine's census of the points
    outside = 0
    t0 = time.time()
    ids = {int(v): i for i, v in enumerate(POOLS)}
    for ys in range(0, shape[1], int(slab_y)):
        ye = min(ys + int(slab_y), shape[1])
        axons = _slab(files["axons"], names["axons"], r["crop"], ys, ye) > 0
        lab = np.where(axons, np.uint8(2), np.uint8(1))
        nuc = _slab(files["nucleus"], names["nucleus"], r["crop"], ys, ye) > 0
        lab[nuc & ~axons] = 3
        del nuc
        mye = _slab(files["myelin"], names["myelin"], r["crop"], ys, ye) > 0
        lab[mye & ~axons & (lab != 3)] = 0
        del mye, axons
        for value, i in ids.items():
            counts[i] += int(np.count_nonzero(lab == value))
        sel = (jy >= ys) & (jy < ye)
        if sel.any():
            g = LabelVolume(lab, vox, pools=POOLS, pool="intra",
                            origin=origin + np.asarray([0.0, ys * vox[1], 0.0]))
            got = np.asarray(g.classify_positions_exact(pts[sel]))
            outside += int(np.count_nonzero(got < 0))
            for i in range(len(POOLS)):
                classified[i] += int(np.count_nonzero(got == i))
            del g
        del lab
    if int(counts.sum()) != int(np.prod(shape)):
        raise ReferenceRefusal(f"{substrate}: the voxel census is {counts.sum():,} of {np.prod(shape):,} "
                               f"voxels; every voxel of the region is one pool")
    if outside:
        raise ReferenceRefusal(f"{substrate}: the engine classified {outside:,} of {n_points:,} points as "
                               f"outside the volume they were drawn inside")
    p = {n: i for i, n in enumerate(POOLS.values())}
    n_my, n_ias = int(counts[p["myelin"]]), int(counts[p["intra"]])
    f_count = n_ias / float(int(counts.sum()) - n_my)
    m_my, m_ias = int(classified[p["myelin"]]), int(classified[p["intra"]])
    n_water = int(classified.sum()) - m_my
    f_direct = m_ias / float(n_water)
    return dict(substrate=substrate, shape=list(shape), crop=list(r["crop"]),
                voxels=int(counts.sum()), voxel_counts={n: int(counts[i]) for n, i in p.items()},
                f_voxel_count=f_count, classified={n: int(classified[i]) for n, i in p.items()},
                n_points=int(n_points), n_water=n_water, f_direct=f_direct,
                se_direct=float(((1.0 - f_direct) / (f_direct * n_water)) ** 0.5),
                seed=int(seed), slab_y=int(slab_y), seconds=round(time.time() - t0, 1),
                files={k: dict(path=os.path.basename(v), sha256=_sha256(v)) for k, v in sorted(files.items())})


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(files, crop):
    """The key a cached measurement is held under: the released files' own digests and the estimator's grid.

    The measurement is re-made the moment either changes and never re-made otherwise -- reading the ten
    low-resolution samples is some 400 GB of gzip through h5py, at about 30 s per gigavoxel.
    """
    return hashlib.sha256(json.dumps(
        [sorted((k, _sha256(v)) for k, v in files.items()), [int(v) for v in crop], N_POINTS, SEED, SLAB_Y],
        sort_keys=True).encode()).hexdigest()


def measurements(work_dir, data_dir, rois, substrates):
    """``{substrate: measure_roi(...)}``, measured once and cached on the bytes it was measured from."""
    path = os.path.join(work_dir, "f-measurements.json")
    os.makedirs(work_dir, exist_ok=True)
    cache = json.load(open(path)) if os.path.exists(path) else {}
    out, dirty = {}, False
    for s in substrates:
        files = em_files(data_dir, rois[s]["sample"])
        key = cache_key(files, rois[s]["crop"])
        held = cache.get(s)
        if held and held.get("key") == key:
            out[s] = held["value"]
            continue
        print(f"measuring {s} ...", flush=True)
        out[s] = measure_roi(s, data_dir, rois)
        cache[s], dirty = dict(key=key, value=out[s]), True
        with open(path, "w") as fh:
            json.dump(cache, fh, indent=1, sort_keys=True)
    if dirty:
        print(f"measurements in {path}", flush=True)
    return out


def _locator(roi):
    """Where the published value is read from: the section that states the recipe and the region it is read on,
    with the released limits written the way the authors write them (one-based, inclusive)."""
    x0, y0, z0, x1, y1, z1 = roi["crop"]
    return (f"\u00a72.5.2, applied to the released segmentation of {roi['sample']} on the {roi['roi']} limits "
            f"of EM_analysis/000_animal_exp_axons_cells_nx_ny_nz_cclims_cglims.txt "
            f"(x {x0 + 1}-{x1}, y {y0 + 1}-{y1}, z {z0 + 1}-{z1}, one-based inclusive)")


# ----------------------------------------------------------------- the family
def family(data_dir, work_dir, *, substrates=None, dry=True, create_dataset=False,
           resolver=crossref):
    rois = roi_limits(os.path.join(data_dir, "nyu-analysis", "EM_analysis",
                                   "000_animal_exp_axons_cells_nx_ny_nz_cclims_cglims.txt"))
    names = list(substrates) if substrates else sorted(rois)
    unknown = [n for n in names if n not in rois]
    if unknown:
        raise ReferenceRefusal(f"{unknown} is not a released region of this study; the regions are "
                               f"{sorted(rois)}")
    cache = os.path.join(work_dir, ".licence-cache")
    cc_by = licence_text("https://creativecommons.org/licenses/by/4.0/legalcode.txt", cache,
                         "cc-by-4.0-legalcode.txt")
    nyu = open(os.path.join(data_dir, "nyu-analysis", "LICENSE"), encoding="utf-8").read()

    em = [SourceFile(path=p, cite_as=f"White matter EM/{s}/{os.path.basename(p)}",
                     role=f"the DeepACSON {role} instance segmentation of {s} at {res} resolution")
          for s in SAMPLES for res in ("LM", "HM") for role, p in sorted(em_files(data_dir, s, res).items())]
    em.append(SourceFile(path=os.path.join(data_dir, "em", "White matter EM", "README.docx"),
                         cite_as="White matter EM/README.docx",
                         role="the release's own description of the samples, the voxel sizes, the label dtypes "
                              "and its Rights and permissions paragraph"))
    mri = [SourceFile(path=p, cite_as=f"Ex vivo diffusion/MRI_data/{animal}/{k}",
                      role=f"the released ex vivo dMRI of brain {animal}: {k}")
           for animal in sorted(ANIMALS) for k, p in sorted(mri_files(data_dir, animal).items())]
    mri.append(SourceFile(path=os.path.join(data_dir, "mri", "Ex vivo diffusion", "MRI_data", "README.docx"),
                          cite_as="Ex vivo diffusion/MRI_data/README.docx",
                          role="the dMRI release's own description of the acquisition and the ROIs"))
    nyu_dir = os.path.join(data_dir, "nyu-analysis")
    analysis = [
        SourceFile(path=os.path.join(nyu_dir, "EM_analysis",
                                     "000_animal_exp_axons_cells_nx_ny_nz_cclims_cglims.txt"),
                   cite_as="EM_analysis/000_animal_exp_axons_cells_nx_ny_nz_cclims_cglims.txt",
                   role="the corpus-callosum and cingulum limits of every released volume: the crop this "
                        "family's regions ARE, and theirs"),
        SourceFile(path=os.path.join(nyu_dir, "EM_analysis", "script03_compute_volume_fractions.m"),
                   cite_as="EM_analysis/script03_compute_volume_fractions.m",
                   role="the recipe for f: which compartment each label is and what the denominator excludes"),
        SourceFile(path=os.path.join(nyu_dir, "EM_analysis", "RefineSeg.m"),
                   cite_as="EM_analysis/RefineSeg.m",
                   role="SetSubstrate and SetCleanSubstrate: how the three released files become one graded "
                        "volume, and what the authors' own cleanup does to it"),
        SourceFile(path=os.path.join(nyu_dir, "LICENSE"), cite_as="LICENSE",
                   role="the analysis code's licence, which is not the data's")]

    sources = [
        Source(key="fairdata-em", url=f"https://doi.org/{EM_DOI}", host_record=EM_RECORD, licence_id=LICENCE,
               licence_url="https://creativecommons.org/licenses/by/4.0/", licence_text=cc_by,
               files=tuple(em)),
        Source(key="fairdata-dmri", url=f"https://doi.org/{MRI_DOI}", host_record=MRI_RECORD,
               licence_id=LICENCE, licence_url="https://creativecommons.org/licenses/by/4.0/",
               licence_text=cc_by, files=tuple(mri)),
        Source(key="nyu-analysis", url=NYU_URL, host_record=f"git {NYU_COMMIT}",
               licence_id="NYU non-commercial research licence",
               licence_url=f"{NYU_URL}/blob/{NYU_COMMIT}/LICENSE", licence_text=nyu, files=tuple(analysis),
               redistributes_bytes=False),
    ]

    m = measurements(work_dir, data_dir, rois, names)
    grid = dict(estimator="uniform points in the region classified by LabelVolume.classify_positions_exact",
                n_points=N_POINTS, seed=SEED, slab_axis="y", slab_voxels=SLAB_Y,
                pools={str(k): v for k, v in POOLS.items()}, voxel_size_m=list(VOXEL_LM))
    solver = "f = n_intra / (n_points - n_myelin); the compartments composed as RefineSeg.SetSubstrate writes them"
    quantities = tuple(ReferenceQuantity(
        substrate=s, name="f_intra_axonal",
        published=Published(
            value=m[s]["f_voxel_count"], unit="-", uncertainty=0.0,
            uncertainty_is=(
                "none stated, and none exists for the count: f is a deterministic voxel count on ONE released "
                "segmentation, and the paper prints it only as a point in the scatter plots of Figures 3 and 4 "
                "with no error bar on either axis. The gate's budget therefore rests on our own standard error "
                "alone, and any disagreement with their number is recorded rather than failed on"),
            printed_in=PAPER_TITLE,
            document=PAPER_DOI,
            locator=_locator(rois[s]),
            data_url=f"https://doi.org/{EM_DOI}", data_sha256=m[s]["files"]["axons"]["sha256"]),
        direct=Direct(
            value=m[s]["f_direct"], unit="-", se=m[s]["se_direct"], se_kind="analytic_mean",
            se_derivation=(
                f"the binomial standard error of a ratio of counts: of the {m[s]['n_points']:,} points drawn "
                f"uniform in this region and classified by the engine, {m[s]['n_water']:,} are not myelin, and "
                f"f is the intra-axonal share of those, so sd(f)/f = sqrt((1 - f) / (f N)) with N the "
                f"dMRI-visible count. It is the standard error of one ensemble mean, not a spread over seeds "
                f"or folds"),
            n_walkers=m[s]["n_water"], grid=grid, solver=solver,
            source=("examples/substrate_bank/build_coronado_leija.py (measure_roi), cached in the family's "
                    "f-measurements.json on the released files' own digests, asserted by "
                    "tests/validation/test_coronado_leija.py")))
        for s in names)

    reference = Reference(
        doi=PAPER_DOI, title=PAPER_TITLE, published_kind="data",
        sample=("the same five brains: the rats were perfusion-fixed, scanned ex vivo at 9.4 T, and only then "
                "sectioned -- \"After the ex vivo dMRI acquisitions, brains were placed in 0.9% NaCl for at "
                "least 4 h to remove excess perfluoropolyether and then sectioned into 1 mm-thick coronal "
                "sections\" -- so the released segmentation and the released dMRI are of one object. f is read "
                "from the released segmentation itself, on the released ROI limits"),
        sample_relation="the same object", quantities=quantities,
        parameters=(
            FreeParameter(name="D0", value=D0, unit="m^2/s", whose="theirs",
                          where="paper §2.5.3 and §2.7", how="\"assuming free diffusivity Dw = 2 um^2/ms (at "
                                                             "room temperature)\"; the brains were scanned at "
                                                             "21 C, stated in §2.2"),
            FreeParameter(name="ROI limits", value="EM_analysis/000_animal_exp_axons_cells_nx_ny_nz_cclims_"
                                                   "cglims.txt, rows 8-13 (cc) and 14-19 (cg)",
                          unit="voxels", whose="theirs", changes_geometry=True,
                          where=f"{NYU_URL} at {NYU_COMMIT[:8]}",
                          how="the corpus-callosum and cingulum crops of each released volume, released by the "
                              "authors with the analysis code and used verbatim, one-based inclusive limits "
                              "read as zero-based half-open. The crop is what makes a region a region and it "
                              "is not ours: over the whole volume f would mix the two tracts"),
            FreeParameter(name="label composition", value="extra 1, myelin 0, nuclei 3, axons 2, each "
                                                          "overwriting the last", unit="-", whose="theirs",
                          where="RefineSeg.SetSubstrate", changes_geometry=True,
                          how="their own order, so an axon voxel that is also myelin is intra-axonal and a "
                              "nucleus voxel that is also myelin is a nucleus; the three released instance "
                              "segmentations overlap and the order decides the overlap"),
            FreeParameter(name="myelin is dMRI-invisible", value=True, unit="-", whose="theirs",
                          where="paper §2.5.2",
                          how="\"excluding the myelin compartment due to its short T2 compared with the TE of "
                              "the dMRI acquisition (MacKay et al., 1994), which makes it dMRI invisible\"; it "
                              "is out of f's denominator and it is a water-free pool of the substrate"),
            FreeParameter(name="nuclei count as extra-axonal water", value=True, unit="-", whose="theirs",
                          where="script03_compute_volume_fractions.m",
                          how="their extra-axonal volume is total - intra-axonal - myelin, which leaves the "
                              "cell nuclei inside it, so f's denominator includes them. They are a pool of "
                              "their own in the substrate and water in it"),
            FreeParameter(name="their segmentation cleanup", value="NOT applied", unit="-", whose="ours",
                          where="RefineSeg.SetCleanSubstrate, called by script02_proofread_vols.m",
                          how="the paper's §2.5.2 recipe is a voxel count on the segmentation, and that is "
                              "what is counted here. Their released code additionally runs bwareaopen at 100 "
                              "voxels, keeps only myelin within 0.35 um of a dilated axon, DILATES the "
                              "intra-axonal space by a one-voxel sphere and closes holes, before counting. "
                              "That cleanup changes f and its size is not measured here; it is a morphology "
                              "pass over 3-7 G voxels per region and it is the first thing to measure when the "
                              "packs exist. It does not change WHICH object is walked -- the released "
                              "segmentation is -- so it is recorded as ours and not as a crop"),
            FreeParameter(name="the direct estimator's points", value=N_POINTS, unit="-", whose="ours",
                          where="this module (measure_roi)",
                          how="4 million points uniform in the region, seed 0, classified by the engine's own "
                              "position-to-pool map; the count sets the standard error and nothing else"),
        ),
        description=("Rat corpus callosum and cingulum from serial block-face electron microscopy, segmented "
                     "by DeepACSON, beside the 9.4 T ex vivo diffusion MRI of the same five brains -- the "
                     "same-specimen, diffusion-weighted, released-geometry pair, both halves CC BY 4.0, that "
                     "no simulation has been run on."),
        source_note=("Abdollahzadeh et al. 2021 / Sierra et al., the 3d EM and its segmentation "
                     f"(doi:{EM_DOI}); Coronado-Leija et al. 2024, the ex vivo dMRI (doi:{MRI_DOI}) and the "
                     f"paper that compares them (doi:{PAPER_DOI}); the ROI limits and the recipe for f from "
                     f"the authors' analysis code ({NYU_URL})."),
        licence_note=("CC-BY-4.0 on both Fairdata records and on the paper; the analysis code that supplies "
                      "the ROI limits is under a non-commercial NYU research licence and is read and cited, "
                      "never redistributed; the packs will be CC-BY-4.0"),
        caveats=dict(
            what_f_gates=(
                "f is a compartment quantity, not a diffusion one. It is the weight with which a region's "
                "intra-axonal pack and its extra-axonal pack sum into the region's signal, so a family that "
                "gets it wrong cannot get the signal right -- but a family that gets it right has only shown "
                "that the engine's position-to-pool map agrees with a voxel count. The diffusion-weighted "
                "comparison this family exists for is the ROI-mean signal of the released 43-direction, "
                "b = 2/3/4 ms/um^2 acquisition, whose volumes are already digested in the source record; it "
                "is a second quantity, and it needs a walk."),
            cleanup=(
                "Their f is computed after RefineSeg.SetCleanSubstrate, which dilates the intra-axonal space "
                "by one voxel among other things; ours is the plain voxel count of §2.5.2. The difference is "
                "unmeasured and is the largest known systematic of the comparison."),
            cingulum=(
                "The authors flag partial volume in their cingulum dMRI ROI -- it \"contained a small "
                "fraction of corpus callosum axons in the perpendicular direction\" -- so the cingulum "
                "regions' eventual signal comparison is against a mixed voxel. That is theirs, it is stated, "
                "and it is a reason to read the corpus-callosum regions first."),
            resolution=(
                "Two tiers are released per sample and both are declared: the 50 nm low-resolution volumes "
                "these regions are cropped from, and 15 x 15 x 50 nm high-resolution volumes of the corpus "
                "callosum whose whole extent IS the region, so they need no crop at all. The 15 nm tier is "
                "3.1e8 voxels against 3-7e9 and is the cheaper first walk; it is also the one the sub-step "
                "rule refuses hardest.")))

    # the pilot is named, and it is the CHEAPEST region: the pilot walks the real window, and on a substrate
    # of 3.3-7.4 G voxels an unnamed pilot is whichever region sorts first, which here is the largest of the
    # nineteen. The protocol scales the floor it measures, so the choice costs the design nothing but time
    pilot = min(names, key=lambda n: int(np.prod([rois[n]["crop"][i + 3] - rois[n]["crop"][i]
                                                 for i in range(3)])))
    design = Design(
        window_s=TE, dt_save_s=DT_SAVE, save_grid_why=SAVE_GRID_WHY, K=K_BANDS, envelope=ENVELOPE,
        waveforms=waveforms,
        # BOTH tiers, deliberately, and this is what makes #478 load-bearing rather than an optimisation: the
        # contact tier is the stored boundary local time, and declaring it puts the surface-local-time rule in
        # force (1,921 sub-steps per 25 us save against the reflection rule's 121). A positions-only pack of
        # this substrate is walkable today at a sixteenth of the cost -- and freezes rho and kappa into the
        # walk, which for a myelinated axon is the one thing a pack must not do
        tiers=(Tier(name="positions", floor_key="floor_max", err_key="err_max", target_floor=SIGMA),
               Tier(name="contact", floor_key="floor_surface", err_key="err_surface", target_floor=SIGMA)),
        memory_budget_bytes=BUDGET_BYTES, pilot_n=PILOT_N, safety=1.4,
        false_failure_rate=FALSE_FAILURE_RATE, pilot_substrate=pilot,
        tolerance=Tolerance(terms=("quantity.direct.se", "quantity.replay_se",
                                   "quantity.published.uncertainty")))

    build = Build(
        specs={s: (lambda n=s: spec_of(n, data_dir, rois)) for s in names},
        pack_id={s: f"coronado-leija-2024/{s}" for s in names},
        reproduce=reproduce, served_vs_channel=served_vs_channel, served_tier="positions", walk=walk)
    publication = Publication(repo=REPO, licence=LICENCE, citation=CITATION, snippet=snippet,
                              snippet_substrate=pilot, pack_path=lambda name: f"packs/{name}.rpk",
                              create_dataset=create_dataset, dry=dry)
    return ReferenceFamily("coronado-leija-2024", work_dir, sources=sources, reference=reference,
                           design=design, build=build, publication=publication, resolver=resolver)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="the directory holding em/, mri/ and nyu-analysis/")
    p.add_argument("--work", required=True, help="the family directory (records/, f-measurements.json)")
    p.add_argument("--substrates", nargs="+", default=None, help="the regions to declare (default: all)")
    p.add_argument("--stage", default=None, help="run one stage (the protocol's order still holds)")
    p.add_argument("--dry", action="store_true", help="gate and render, upload nothing")
    a = p.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("dmipy_sim").setLevel(logging.INFO)
    fam = family(os.path.abspath(a.data), os.path.abspath(a.work), substrates=a.substrates, dry=a.dry)
    if a.stage:
        fam.stage(a.stage)
    else:
        fam.run()
    print(f"records in {fam.records.dir}")


if __name__ == "__main__":
    main()
