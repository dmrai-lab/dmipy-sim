"""The reference-pack protocol: one deterministic pipeline every SubstrateCommons family runs (dmipy-sim#482),
with a card producer (#480) as its eighth stage and a gate that can fail.

A **reference family** is a set of replay packs of one released substrate that reproduces a published quantity.
Every family runs the same nine stages in order, each writing ONE record file into the family directory:

=============== ================================ ==========================================================
stage           record                           what it is
=============== ================================ ==========================================================
``source``      ``records/source.json``          every input file with its URL, the host's licence text
                                                 copied verbatim, its sha256 and size, the host record
``reference``   ``records/reference.json``        the published quantity as data: a DOI resolved through
                                                 Crossref with its title compared, the quantity, the
                                                 sample, our own recorded direct measurement of it with
                                                 its analytic standard error, and every free parameter
                                                 marked whose it is
``spec``        ``records/spec.json``             the substrate spec per pack, round-tripped through
                                                 ``spec_of(geometry_from_spec(spec))``, citing the source
                                                 files by their digests
``design``      ``records/design.json``           the declared envelope, the target floor per tier, the
                                                 memory budget, and a PILOT measured on the real window
                                                 from which the walker count follows
``walk``        ``records/walk.json``             the walk at the design with the engine's own counters
``pack``        ``records/pack.json``             the pack per substrate, its certificate per tier, the
                                                 served-vs-decoded check, the declared waveforms against
                                                 ``waveform_band``, and the reproduced quantity
``gate``        ``records/gate.json``             deterministic, reads only records, fails by name
``card``        ``records/card.json``             the card rendered from the records, its previews, and
                                                 its "Use me" snippet EXECUTED
``publish``     ``records/publish.json``          what went to the hub, and the organisation page
=============== ================================ ==========================================================

The rules are in the code, not in a convention:

* **Order.** A stage reads the previous record and nothing else, and refuses when it is missing or when its
  digest is not the one that record was written against. No flag skips a stage and no flag reorders them.
* **Write once.** A record is written by the stage that measured it. A re-run whose inputs are unchanged
  returns the record it already wrote; a re-run whose inputs changed is refused, naming the record to delete.
* **Every number in a card or a paper is read from a record.**
* **No tolerance constant lives in the gate.** :class:`Tolerance` names the record paths its terms come from
  and they are combined in quadrature. There is no coverage factor and no resampled (fold, split-half) error
  bar: a term is an ANALYTIC standard error -- of an ensemble mean, or of a derived statistic by the delta
  method -- or the reference's own stated uncertainty.
* **The gate never re-walks, re-measures or re-fits.** Every number it compares was recorded upstream.
* **The memory budget is a hard cap**, checked against the pilot's measured bytes per walker before the walk.
* **Publication is explicit**: the pack list comes from the gate's record, a stray pack is fatal, the code
  commit must be clean and pushed, and creating the dataset is its own logged step.

An agent producing a family fills the source and reference records by hand from primary sources and then
operates the pipeline; its judgement stops at the reference record. ``docs/reference-family.md`` is its brief.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

log = logging.getLogger("dmipy_sim.replay")

#: The nine stages, in the only order they run in.
STAGES = ("source", "reference", "spec", "design", "walk", "pack", "gate", "card", "publish")

#: The reference grades of dmipy-sim#459, assigned BY RULE from the reference record (:func:`grade_of`).
GRADES = ("A", "B", "analytic", "none")

#: Whose a free parameter is. A parameter without one of these is not a recorded parameter.
WHOSE = ("theirs", "literature", "ours")

#: The counters of a walk a :class:`RecordedWalk` may declare ABSENT, by name. ``not_recorded`` is a tuple drawn
#: from these and never a sentence: prose that happened to contain ``peak_rss_bytes`` because it said "for 118 of
#: the 600, peak_rss_bytes" turned the refusal off for all 600 packs of canonical-pores rather than the 118 it
#: described, which is what a substring test buys.
RECORDED_WALK_COUNTERS = ("sub_steps", "illegal_crossings", "peak_rss_bytes")

#: What the publication states about the quantity: a number in a document, released DATA (a signal, a set of
#: per-object signals, a volume), a figure, a closed form, or nothing. ``data`` is graded like a number, since a
#: released array IS the measurement and needs no printed line -- which is what parity-fixtures,
#: winther-g6-axons and disco-replay reproduce, and what the "verbatim line" rule forced them to fabricate.
PUBLISHED_KINDS = ("number", "data", "figure", "analytic", "none")

#: How a standard error was obtained. Both are analytic; neither resamples one realisation.
SE_KINDS = ("analytic_mean", "delta_method")

#: What the reference's sample is to ours.
SAMPLE_RELATIONS = ("the same object", "the same material", "a matched statistic")

#: What ``spec_of(geometry_from_spec(spec))`` cannot reproduce, and therefore what the spec stage's round trip
#: does not compare, as dotted paths (``*`` matches one list index or key).
#:
#: ``spec_of`` reconstructs a spec FROM a geometry. Two kinds of field are therefore beyond it, and both are
#: named here rather than left to make the rule pass or fail by accident:
#:
#: * the producer's own LABELLING -- ``id``, ``provenance``, ``description``, a wall's ``name``. "axolemma" and
#:   "surface" are the same wall; a citation is the producer's words about where the substrate came from.
#: * what the producer MEASURED off the released surface and ``spec_of`` reconstructs with a coarser rule.
#:   ``realisation`` is the producer's measurement and comes back ``None``; ``validity.mesh_edge_feature_ratio``
#:   is derived from a feature scale that ``mcdc_axon_spec`` measures as the tube's ``2 V / S`` where
#:   ``spec_of`` takes ``_surface_stats``' half-thinnest-extent, which for an undulating tube is the undulation
#:   envelope and three times too coarse (1.62 against 2.92 on the MC/DC axons). ``smallest_feature`` itself IS
#:   compared, and agrees.
#:
#: Everything that says what was WALKED -- the domain, the pools, every wall's surface and physics, the seeding,
#: the rest of the validity -- is compared, which is what the rule exists for: it caught an
#: ``extra.water_fraction`` of 1.0 for a pool the reference engine never seeded and a ``seeding.rule`` of
#: ``uniform_by_volume`` for one that was ``explicit``.
ROUND_TRIP_NOT_COMPARED = ("id", "provenance", "description", "realisation",
                           "validity.mesh_edge_feature_ratio", "walls.*.name")

#: A licence text is the licence, not its title: the shortest in use (the CC BY 4.0 deed) is some 1.5 kB and
#: an MIT licence 1.0 kB, so a record under this many characters is a name and not a text.
MIN_LICENCE_TEXT_CHARS = 400

#: The ``licence_id`` of a host that states NO licence. dmipy-sim#459's own survey records such hosts verbatim
#: ("unstated"), and the protocol has to be able to record one too: the Winther G6 morphology is served by
#: resources.drcmr.dk, whose pages carry a copyright line and no licence at all, and a rule that only a licence
#: TEXT may be recorded leaves an agent two choices, both wrong -- not record the family, or write down a licence
#: nobody granted (the packs on the hub today assert CC-BY-4.0 for that morphology). Such a source is READ and
#: CITED; not one of its bytes may be redistributed, which :class:`Source` declares and the source stage refuses.
LICENCE_NONE_STATED = "none stated"

#: How short the host's own words about rights may be when they are all there is. resources.drcmr.dk states
#: "Copyright (c) 2022. All Rights Reserved" and nothing else; the record carries that, plus where it was looked
#: for, so an auditor checks the absence the same way they would check a licence.
MIN_LICENCE_ABSENT_CHARS = 40

#: Names a tolerance term may not end in: each is a way of widening a threshold without measuring anything.
FORBIDDEN_TOLERANCE_KEYS = ("coverage", "coverage_factor", "k", "fold_spread", "folds", "split_half", "safety")

#: A card's "Use me" snippet must reproduce one recorded number within this, on the CPU, or the card is not
#: written. A snippet a reader will not wait for is not a snippet.
SNIPPET_CEILING_S = 60.0

__all__ = ["ReferenceFamily", "ReferenceRefusal", "Records", "STAGES", "GRADES", "LICENCE_NONE_STATED",
           "RECORDED_WALK_COUNTERS", "grade_of", "crossref",
           "code_commit", "tolerance_of", "gate_verdict", "pass_band", "grade_reason", "Source", "SourceFile", "Reference",
           "ReferenceQuantity", "Published", "Direct", "FreeParameter", "Design", "Tier", "Systematic",
           "Tolerance", "Build", "RecordedWalk", "Publication", "organisation_index", "organisation_card", "organisation_example",
           "ORGANISATION"]


class ReferenceRefusal(ValueError):
    """A family that does not meet the protocol. The message names the stage and the field."""


# --------------------------------------------------------------- what a family author writes, as data
@dataclass(frozen=True)
class SourceFile:
    """One input file: where it is now, the name a dataset cites it by, and what it is for."""
    path: str
    cite_as: str
    role: str


@dataclass(frozen=True)
class Source:
    """One host record and its files. ``licence_text`` is the host's licence VERBATIM, not its title, and
    ``host_record`` the commit or version of the record the files came from.

    ``licence_id`` is :data:`LICENCE_NONE_STATED` for a host that states no licence at all. ``licence_text`` is
    then the host's own words about rights (its copyright line, and where each page was looked at), which is
    what an auditor checks the absence against, and ``redistributes_bytes`` must be false: a source whose
    licence nobody granted may be read and cited, never republished.

    ``redistributes_bytes`` says whether this family puts the source's OWN bytes on the hub. A family that
    publishes only packs of its own walk does not.
    """
    key: str
    url: str
    host_record: str
    licence_id: str
    licence_url: str
    licence_text: str
    files: tuple
    redistributes_bytes: bool = False


@dataclass(frozen=True)
class FreeParameter:
    """A parameter the comparison rests on, and whose it is: fitted by them, a literature value, or ours.

    ``changes_geometry`` says whether the parameter decides WHICH object was walked -- a crop, an offset, a
    decimation, a synthetic sheath. One of those marked ``ours`` means the geometry is not the released one, and
    the grade follows (:func:`grade_of`): it is not a boolean anyone sets by hand.
    """
    name: str
    value: object
    unit: str
    whose: str
    where: str
    how: str
    changes_geometry: bool = False


@dataclass(frozen=True)
class Published:
    """Their number and where it comes from: the document that PRINTS it, quoted verbatim, or the released DATA
    it is read from, digested.

    ``uncertainty`` is RELATIVE (a fraction of ``value``). A publication that states none records ``0.0`` and
    says so in ``uncertainty_is``; it is never left unstated, since the gate reads it.

    ``document`` is the persistent identifier of the thing ``printed_in`` names, RESOLVED at record time with
    its title compared -- a DOI through Crossref or DataCite (a thesis's Spiral DOI is a DataCite one), never
    left as prose. ``data_url`` and ``data_sha256`` are the released array a ``data`` reference reads its value
    from; a ``data`` reference needs them and no verbatim line, and a ``number`` reference needs the line.
    """
    value: Optional[float]
    unit: str
    uncertainty: float
    uncertainty_is: str
    printed_in: str
    locator: str
    verbatim: str = ""
    document: Optional[str] = None
    data_url: Optional[str] = None
    data_sha256: Optional[str] = None


@dataclass(frozen=True)
class Direct:
    """OUR direct measurement of the same quantity -- the forward engine with the parameter baked in -- as
    recorded by the validation script that owns the estimator.

    ``se`` is RELATIVE and ANALYTIC: the standard error of an ensemble mean (``analytic_mean``) or of a
    derived statistic by the delta method over walkers (``delta_method``). A fold spread, a split half or any
    other resampling of one realisation is not a standard error: it is seed-dependent, and a threshold cannot
    be. ``grid`` and ``solver`` are part of the measurement -- the same inversion on a 20 ms sampling instead
    of a 1 ms one moved one of these numbers by a factor of two -- so the gate checks that the reproduction
    ran on this grid and refuses when it did not.
    """
    value: float
    unit: str
    se: float
    se_kind: str
    se_derivation: str
    n_walkers: int
    grid: dict
    solver: str
    source: str


@dataclass(frozen=True)
class ReferenceQuantity:
    """The quantity one substrate of the family reproduces: theirs, and ours measured directly."""
    substrate: str
    name: str
    published: Published
    direct: Direct


@dataclass(frozen=True)
class Reference:
    """The published reference, as data. ``doi`` is resolved through Crossref at record time and its title
    compared with ``title``; the grade of dmipy-sim#459 follows from ``published_kind`` and from whether the
    geometry is the released one, which the reference stage DERIVES from the parameters (no free parameter
    marked ``ours`` may change it) -- neither the grade nor that fact is a field anyone sets.

    ``description``, ``source_note`` and ``licence_note`` are the family's own row on the organisation page:
    the descriptions live with the family, not in the page's renderer.
    """
    doi: str
    title: str
    published_kind: str
    sample: str
    sample_relation: str
    quantities: tuple
    parameters: tuple
    description: str
    source_note: str
    licence_note: str
    caveats: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Tier:
    """A tier of the pack's certificate and the floor this family targets for it. ``floor_key`` / ``err_key``
    name the fields of the pack's own ``fidelity`` that carry it, so nothing is "not applicable"."""
    name: str
    floor_key: str
    err_key: str
    target_floor: float


@dataclass(frozen=True)
class Systematic:
    """A measured systematic of the comparison: not noise more walkers would remove. It carries its evidence."""
    name: str
    value: float
    measured_on: str
    evidence: str


@dataclass(frozen=True)
class Tolerance:
    """The gate's budget, as the record paths its terms come from -- never as numbers in the gate.

    Each term is a dotted path resolved against the records, ``quantity.*`` relative to the quantity being
    gated and ``design.*`` / ``pack.*`` / ``reference.*`` absolutely. The terms are combined in quadrature.
    There is no coverage factor: the tolerance is the standard uncertainty, and a term whose name is one of
    :data:`FORBIDDEN_TOLERANCE_KEYS` is refused.
    """
    terms: tuple
    combine: str = "quadrature"


@dataclass(frozen=True)
class Design:
    """What the packs serve and at what precision. The walker count is NOT here: the pilot measures it.

    ``waveforms`` is a callable returning ``((label, ScannerSequence), ...)`` -- every waveform the family
    DECLARES, which the pack stage checks against ``waveform_band`` and the gate then reads.

    ``false_failure_rate`` is the probability, DECLARED here and recorded before the gate runs, that the gate
    fails a correct family; :func:`pass_band` turns it and the number of comparisons into the multiplier the
    gate applies to its standard uncertainty. Without it the criterion is one standard error, which fails a
    correct pack a third of the time per comparison.

    ``pilot_substrate`` NAMES the substrate the pilot walks. Unnamed it is the first in sorted order, which is
    an arbitrary choice the moment a family has more than a handful: canonical-pores' first is the 0.1 um
    cylinder, whose sub-step rule is 1/R^2 and whose pilot would cost more than every pack it sizes. The name is
    refused when it is not one of this family's substrates, and the design record carries it either way.
    """
    window_s: float
    dt_save_s: float
    save_grid_why: str
    K: int
    envelope: dict
    waveforms: object
    tiers: tuple
    memory_budget_bytes: int
    pilot_n: int
    safety: float
    tolerance: Tolerance
    false_failure_rate: float = 0.01
    systematics: tuple = ()
    pilot_substrate: Optional[str] = None


@dataclass(frozen=True)
class RecordedWalk:
    """A walk that already happened, as its own record: the counters the engine reported, with the log they
    were reported in. The walk stage writes a record FROM this instead of walking again.

    A counter the producing run did not report is ``None``, and ``not_recorded`` then NAMES it and says why it
    is absent -- measured on this conversion: the canonical-pores build logged the floor and the bytes, and the
    Winther G6 rebuild set the ``dmipy_sim`` logger to ``WARNING``, so neither family's published walks reported
    ``sub_steps`` or ``illegal_crossings`` anywhere, and the packs' headers carry neither. The alternatives were
    to refuse the family or to let an agent type a plausible number into a measurement; recording the absence by
    name is neither, and the card prints "not recorded" where the number would be.

    ``peak_rss_bytes`` may be named in ``not_recorded`` too, for a pack whose walk left no resident-bytes
    measurement at all (118 of canonical-pores' 600). The budget is then not checked against a measurement for
    that pack and the gate SAYS so per pack, rather than the family being refused over a number about a walk
    that has already happened.

    ``not_recorded`` is a tuple of :data:`RECORDED_WALK_COUNTERS` names, PER PACK, and a string is refused. It
    was matched as a substring of prose, so one sentence set on every pack of a family -- "sub_steps,
    illegal_crossings and, for 118 of the 600, peak_rss_bytes" -- declared every counter absent on all 600
    instead of naming the 118 that lacked a peak, and the refusal it was meant to arm never fired.
    """
    pack_path: str
    sub_steps: Optional[int] = None
    illegal_crossings: Optional[int] = None
    evidence: str = ""
    not_recorded: tuple = ()


@dataclass(frozen=True)
class Build:
    """How stages 3, 5 and 6 are performed for this family.

    ``specs`` maps a substrate name to a zero-argument callable returning its
    :class:`~dmipy_sim.spec.SubstrateSpec`; ``pack_id`` names the pack built from it. ``walk`` walks a spec at
    a count (``walk(spec, n, *, n_t)``) and is what the pilot and the walk stage call; ``recorded`` gives a
    :class:`RecordedWalk` per substrate instead, for a family whose packs already exist and must not be
    re-walked or re-encoded.

    ``served_tier`` names the tier whose codec error bounds that difference -- the channel the served number
    is read from.

    ``reproduce(pack, quantity, grid)`` replays the pack and returns
    ``dict(value=, se=, se_kind=, se_derivation=, solver=)`` for that quantity, ON the grid it is handed --
    the reference record's own. ``served_vs_channel(pack)`` returns the largest difference between the signal
    the pack SERVES a consumer and the one its decoded channel holds.
    """
    specs: dict
    pack_id: dict
    reproduce: object
    served_vs_channel: object
    served_tier: str
    walk: object = None
    recorded: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Publication:
    """The dataset a family publishes to, and the snippet its card teaches.

    ``snippet(uri)`` returns the source of the "Use me" snippet reading the pack at ``uri``; the card renders
    it at the hub URI and EXECUTES it at the local path of the same bytes, recording both.

    ``hold`` is ``{substrate: why}`` -- a DECLARED reservation about a pack the gate passed, written into that
    pack's manifest row beside the verdict and onto the card. It is not a verdict and cannot become one: the
    ``gate`` field stays what the gate gave, and a hold on a pack the gate FAILED is refused, since that pack is
    withheld and needs no reservation. It exists because a consumer reads the row: the Disimpy fixture is
    published with a reservation on dmipy-sim#488, whose evidence is a vector the scalar gate cannot see, and a
    row reading ``gate: pass`` with nothing else said that the card's caveat was invisible to anyone reading
    the manifest.
    """
    repo: str
    licence: str
    citation: str
    snippet: object
    snippet_substrate: str
    pack_path: object = None
    create_dataset: bool = False
    dry: bool = False
    hold: dict = field(default_factory=dict)


def _pruned(node, paths=ROUND_TRIP_NOT_COMPARED, prefix=()):
    """``node`` with every path of :data:`ROUND_TRIP_NOT_COMPARED` removed, ``*`` matching one index or key.

    Pruning both sides of the round trip is what lets the rule compare the walls' surfaces and physics while
    not comparing a wall's NAME: dropping the whole of ``walls`` to avoid one label would drop the field the
    rule exists to check.
    """
    def hidden(path):
        for p in paths:
            parts = p.split(".")
            if len(parts) == len(path) and all(x == "*" or x == str(y) for x, y in zip(parts, path)):
                return True
        return False

    if isinstance(node, dict):
        return {k: _pruned(v, paths, prefix + (k,)) for k, v in node.items() if not hidden(prefix + (k,))}
    if isinstance(node, list):
        return [_pruned(v, paths, prefix + (str(i),)) for i, v in enumerate(node)]
    return node


# --------------------------------------------------------------- the grade rule
def _protocol_reference(reference_record, *, where="grade"):
    """``reference_record`` if the reference stage of this protocol wrote it, else a refusal naming what is
    missing.

    A pre-protocol family publishes a ``records/reference.json`` of its own shape -- ling-sand-packs' carries
    ``grade`` as a FIELD and no ``published_kind`` -- and reading a grade off it raised ``KeyError`` from inside
    the organisation page's renderer, after this family's own publish had already succeeded. A record the
    protocol did not write is refused by name; it is not graded, and it is not guessed at.
    """
    missing = [k for k in ("published_kind", "same_released_geometry") if k not in reference_record]
    if missing:
        raise ReferenceRefusal(
            f"{where}: this records/reference.json states {missing} nowhere, so the reference stage of the "
            f"reference-pack protocol (dmipy-sim#482) did not write it"
            + (f" (it states a grade of {reference_record['grade']!r} as a FIELD, which no protocol record does: "
               f"grade_of assigns it)" if "grade" in reference_record else "")
            + "; convert the family through the protocol rather than grading a record of another shape")
    return reference_record


def grade_of(reference_record):
    """The grade of dmipy-sim#459, from the reference record and nothing else.

    ``A`` a published NUMBER, or released DATA, on the same released geometry; ``B`` a published figure, or a
    number or data on something that is not the released geometry; ``analytic`` a closed form; ``none`` no
    reference at all. A grade is not a field a record states: it follows from what the publication gives and
    from whether the geometry is the released one, both of which the reference record carries. A record the
    protocol did not write is refused rather than graded (:func:`_protocol_reference`).
    """
    if not reference_record or reference_record.get("absent"):
        return "none"
    kind = _protocol_reference(reference_record)["published_kind"]
    if kind == "analytic":
        return "analytic"
    if kind == "none":
        return "none"
    if kind in ("number", "data") and reference_record["same_released_geometry"]:
        return "A"
    return "B"


def grade_reason(reference_record):
    """Why :func:`grade_of` gave that grade, as one sentence the card prints."""
    if not reference_record or reference_record.get("absent"):
        return "no published reference."
    kind, ours = _protocol_reference(reference_record)["published_kind"], reference_record.get("geometry_parameters_ours") or []
    what = {"number": "a published number", "data": "the released data itself",
            "figure": "a published figure", "analytic": "a closed form",
            "none": "no published reference"}[kind]
    if kind in ("analytic", "none"):
        return what + "."
    if reference_record["same_released_geometry"]:
        return (f"{what} on the same released geometry: no free parameter marked 'ours' changes it."
                if kind in ("number", "data") else f"{what}: a figure is read, not a number.")
    return (f"{what}, but NOT on the released geometry -- "
            f"{', '.join(ours)} {'is' if len(ours) == 1 else 'are'} ours and change"
            f"{'s' if len(ours) == 1 else ''} it, so the grade is B rather than A.")


# --------------------------------------------------------------- Crossref
def _normalise_title(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _get_json(url, *, timeout, accept=None):
    import urllib.request
    headers = {"User-Agent": "dmipy-sim (https://github.com/dmrai-lab/dmipy-sim)"}
    if accept:
        headers["Accept"] = accept
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as fh:
        return json.load(fh)


def crossref(doi, *, timeout=30.0):
    """Resolve ``doi`` and return ``dict(doi, title, type, container, agency, resolved)``.

    This is the default resolver of the reference stage. Crossref first, then DataCite: a journal article is
    registered with Crossref and a thesis or a dataset with DataCite (Imperial's Spiral gives Talabi's thesis
    the DOI ``10.25560/4261``, which Crossref answers 404 for), and a protocol that could only resolve one of
    them pushed a family into citing the paper for a number printed in the thesis. A DOI that resolves with
    neither is a refusal, not a warning: the record's whole point is that the citation was checked.
    """
    tried = {}
    try:
        msg = _get_json(f"https://api.crossref.org/works/{doi}", timeout=timeout)["message"]
        return dict(doi=msg.get("DOI"), title=(msg.get("title") or [""])[0], type=msg.get("type"),
                    container=(msg.get("container-title") or [None])[0], agency="crossref",
                    resolved=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    except Exception as e:
        tried["crossref"] = f"{type(e).__name__}: {e}"
    try:
        a = _get_json(f"https://api.datacite.org/dois/{doi}", timeout=timeout,
                      accept="application/json")["data"]["attributes"]
        return dict(doi=a.get("doi"), title=(a.get("titles") or [{}])[0].get("title", ""),
                    type=(a.get("types") or {}).get("resourceTypeGeneral"), container=a.get("publisher"),
                    agency="datacite", resolved=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    except Exception as e:
        tried["datacite"] = f"{type(e).__name__}: {e}"
    raise ReferenceRefusal(f"reference: the DOI {doi!r} resolved with neither registration agency "
                           f"({tried}); a reference record without a resolved identifier is refused")


# --------------------------------------------------------------- the code commit
def code_commit(*, require_clean=True):
    """The dmipy-sim commit a manifest row's ``published_commit`` names, refused unless it is fetchable.

    The git logic lives once, in :func:`dmipy_sim.replay.publish.code_commit` (the lower-level module every
    publish path, not only this protocol's, reads it from); this wraps it so a refusal here is a
    :class:`ReferenceRefusal` like every other refusal of this protocol, rather than the plain ``ValueError`` a
    standalone ``publish()`` call raises.
    """
    from .publish import code_commit as _code_commit
    try:
        return _code_commit(require_clean=require_clean)
    except ValueError as e:
        raise ReferenceRefusal(str(e)) from e


# --------------------------------------------------------------- records
def _canonical(payload):
    return json.dumps(payload, indent=1, sort_keys=True, default=str).encode()


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest(path):
    return dict(path=os.path.basename(path), sha256=_sha256_file(path), bytes=os.path.getsize(path))


class Records:
    """The family's ``records/`` directory: one file per stage, written once, each carrying the digest of the
    record before it and of its own inputs.

    A stage's record is returned unchanged when its ``inputs`` digest is the one on disk -- that is what
    "a stage asked to run twice on unchanged inputs returns the record it already wrote" means. A record whose
    inputs digest differs is REFUSED: the operator deletes it deliberately, so no later stage can find a
    number quietly restated.
    """

    def __init__(self, directory):
        self.dir = os.path.join(os.path.abspath(directory), "records")

    def path(self, stage):
        return os.path.join(self.dir, f"{stage}.json")

    def exists(self, stage):
        return os.path.exists(self.path(stage))

    def read(self, stage):
        if not self.exists(stage):
            raise ReferenceRefusal(f"{stage}: {self.path(stage)} is missing; run the {stage} stage first -- the "
                                   f"stages run in the order {', '.join(STAGES)} and none of them is skippable")
        with open(self.path(stage)) as fh:
            return json.load(fh)

    def digest(self, stage):
        return _sha256_file(self.path(stage))

    def check_chain(self, stage):
        """Verify that every earlier record is still the one its successor was written against."""
        for later in STAGES[1:STAGES.index(stage) + 1]:
            if not self.exists(later):
                break
            declared = self.read(later).get("previous_sha256")
            earlier = STAGES[STAGES.index(later) - 1]
            now = self.digest(earlier)
            if declared != now:
                raise ReferenceRefusal(f"{later}: records/{earlier}.json has changed since the {later} stage read it "
                                       f"({str(declared)[:12]} -> {now[:12]}); delete records/{later}.json onwards "
                                       f"and run the stages again")

    def reuse(self, stage, inputs_digest, previous_sha256):
        """The record this stage already wrote for these inputs, or ``None`` when it must measure.

        A stage calls this BEFORE it measures anything: that is what "a stage asked to run twice on unchanged
        inputs returns the record it already wrote" means, and a stage whose measurement is not a deterministic
        function of its inputs -- the design pilot's sampled memory is not -- would otherwise be unable to
        return it at all.

        Every record seals itself: ``body_sha256`` is the digest of its own body without that field, so an
        edited record fails its own seal and is refused here rather than handed to the next stage as a
        measurement. A record written for other inputs is refused too, naming them.
        """
        if not self.exists(stage):
            return None
        held = self.read(stage)
        sealed = held.get("body_sha256")
        if sealed != _sha256_bytes(_canonical({k: v for k, v in held.items() if k != "body_sha256"})):
            raise ReferenceRefusal(
                f"{stage}: {self.path(stage)} does not match its own body_sha256, so it has been edited since "
                f"the stage wrote it; an edited record is not a measurement -- delete it and the records after "
                f"it deliberately to re-measure")
        if (held.get("inputs_sha256"), held.get("previous_sha256")) != (inputs_digest, previous_sha256):
            raise ReferenceRefusal(
                f"{stage}: {self.path(stage)} was written for other inputs "
                f"({str(held.get('inputs_sha256'))[:12]} -> {inputs_digest[:12]}); a record is written once by "
                f"the stage that measured it -- delete it and the records after it deliberately to re-measure")
        return held

    def write(self, stage, payload, *, inputs_digest, previous_sha256):
        """Write one record, sealed with the digest of its own body. A record already there is reused through
        :meth:`reuse`, which every stage calls before it measures."""
        held = self.reuse(stage, inputs_digest, previous_sha256)
        if held is not None:
            return held
        sealed = dict(payload, stage=stage, inputs_sha256=inputs_digest, previous_sha256=previous_sha256)
        body = _canonical(dict(sealed, body_sha256=_sha256_bytes(_canonical(sealed))))
        os.makedirs(self.dir, exist_ok=True)
        with open(self.path(stage), "wb") as fh:
            fh.write(body)
        log.info("%s: records/%s.json written (%d bytes)", stage, stage, len(body))
        return json.loads(body)


# --------------------------------------------------------------- the tolerance, resolved over the records
def _resolve(path, context):
    node = context
    for key in str(path).split("."):
        if not isinstance(node, dict) or key not in node:
            where = sorted(node) if isinstance(node, dict) else type(node).__name__
            raise ReferenceRefusal(f"tolerance: the term {path!r} does not resolve -- {key!r} is not in {where}")
        node = node[key]
    return node


def _quadrature(*terms):
    """Independent standard uncertainties combined: the root of the sum of squares."""
    return float(sum(float(t) ** 2 for t in terms) ** 0.5)


def tolerance_of(tol, context):
    """``(terms, tolerance)``: every term of ``tol`` resolved over the records, combined in quadrature.

    A term is a record PATH, so the number it contributes was measured and written by the stage that measured
    it. A term that is a literal, that does not resolve, that is not a finite non-negative fraction, or whose
    name is a coverage factor or a resampled error bar, is refused.
    """
    if tol.get("combine") != "quadrature":
        raise ReferenceRefusal(f"tolerance: combine is {tol.get('combine')!r}; the terms of a standard uncertainty "
                               f"are combined in quadrature")
    paths = list(tol.get("terms") or ())
    if not paths:
        raise ReferenceRefusal("tolerance: no terms; a gate with no budget is not a gate")
    out = {}
    for p in paths:
        if not isinstance(p, str):
            raise ReferenceRefusal(f"tolerance: the term {p!r} is a literal; every term is a record path, so that "
                                   f"the number it contributes was measured by the stage that wrote it")
        last = p.split(".")[-1]
        if last in FORBIDDEN_TOLERANCE_KEYS:
            raise ReferenceRefusal(f"tolerance: the term {p!r} is a {last!r}; this tolerance is the standard "
                                   f"uncertainty, with no coverage factor and no resampled error bar")
        v = _resolve(p, context)
        if v is None or not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ReferenceRefusal(f"tolerance: the term {p!r} is {v!r}; a term is a number recorded upstream")
        v = float(v)
        if not (v >= 0.0) or v != v or v == float("inf"):
            raise ReferenceRefusal(f"tolerance: the term {p!r} is {v!r}; a relative standard uncertainty is finite "
                                   f"and non-negative")
        out[p] = v
    return out, _quadrature(*out.values())


def pass_band(n_comparisons, dof, *, false_failure_rate):
    """The multiplier ``k`` a gate of ``n_comparisons`` comparisons applies to its standard uncertainty so that
    the whole gate fails a CORRECT family with probability ``false_failure_rate``.

    A criterion of one standard error is not a criterion: a correct pack sits outside its own 1-sigma band a
    third of the time, and over several comparisons that is most runs. The band is the two-sided Student-t
    quantile at ``false_failure_rate / (2 n_comparisons)`` on the estimator's own degrees of freedom -- the
    Bonferroni form, so the rate is over the whole gate and not per comparison. It is derived from the
    comparison COUNT and the estimator, never from the data, and the design record carries it before the gate
    runs: that is what makes it different from a coverage factor typed in after a failure.
    """
    from scipy.stats import t as student
    m, nu, alpha = int(n_comparisons), int(dof), float(false_failure_rate)
    if m < 1 or nu < 1:
        raise ReferenceRefusal(f"pass band: {m} comparisons on {nu} degrees of freedom; both must be positive")
    if not (0.0 < alpha < 1.0):
        raise ReferenceRefusal(f"pass band: a false-failure rate of {alpha!r} is not a probability in (0, 1)")
    k = float(student.isf(0.5 * alpha / m, nu))
    one_sigma = 1.0 - (1.0 - 2.0 * float(student.sf(1.0, nu))) ** m      # what a one-sigma criterion would cost
    return dict(n_comparisons=m, dof=nu, false_failure_rate=alpha, k=k, one_sigma_false_failure_rate=one_sigma,
                distribution=f"Student-t on {nu} degrees of freedom",
                rule=(f"the gate makes {m} comparisons against a standard uncertainty on {nu} degrees of "
                      f"freedom; a band of {k:.4f} standard uncertainties per comparison fails a correct "
                      f"family with probability {alpha:g} over the whole gate (two-sided, Bonferroni). A "
                      f"one-sigma criterion would fail one about {one_sigma:.0%} of the time."))


# --------------------------------------------------------------- the gate
def _gate_checks(rec):
    """Every check the gate makes, as ``[{check, passed, detail}]``, from the records in ``rec`` alone.

    No number here is typed: a threshold is the pack's own certificate, the design record's tolerance terms,
    or the band the pack stores. The gate never walks, replays or inverts anything -- every quantity it
    compares was measured and written by the stage that measured it.
    """
    from .publish import _size                            # the byte format lives once, and carries no threshold
    src, ref, spc, des, wlk, pk = (rec[s] for s in ("source", "reference", "spec", "design", "walk", "pack"))
    checks = []

    def check(ok, what, detail, **numbers):
        checks.append(dict(check=what, passed=bool(ok), detail=detail, **numbers))

    # ---- the sources: a licence text and a digest per file
    for key, s in sorted(src["sources"].items()):
        text = s.get("licence_copy", {}) or {}
        stated = s.get("licence_stated", True)
        floor = MIN_LICENCE_TEXT_CHARS if stated else MIN_LICENCE_ABSENT_CHARS
        check(bool(text.get("sha256")) and int(text.get("chars") or 0) >= floor
              and (stated or not s.get("redistributes_bytes")),
              f"source/{key}/licence",
              (f"{s.get('licence_id')} copied verbatim from {s.get('licence_url')}: "
               if stated else
               f"the host at {s.get('licence_url')} states NO licence; its own words about rights are recorded "
               f"verbatim and no byte of it is redistributed: ")
              + f"{text.get('chars')} characters, sha256 {str(text.get('sha256'))[:12]}")
        missing = [f["cite_as"] for f in s["files"] if not f.get("sha256") or not f.get("bytes")]
        check(not missing, f"source/{key}/digests",
              f"{len(s['files'])} files digested" if not missing else f"no digest for {missing}")
        check(bool(s.get("host_record")), f"source/{key}/host-record",
              f"the host record is {s.get('host_record')!r}")

    # ---- the reference: a resolved DOI whose title matches
    if ref.get("absent"):
        check(True, "reference/absent", f"this family reproduces no published quantity: {ref.get('absent')}")
    else:
        check(bool(ref.get("doi")) and bool(ref.get("crossref", {}).get("resolved")) and ref.get("title_matches"),
              "reference/doi",
              f"{ref.get('doi')} resolved at {ref.get('crossref', {}).get('resolved')} as "
              f"{ref.get('crossref', {}).get('title')!r}, compared with the recorded title")

    # ---- per substrate
    for name in sorted(pk["substrates"]):
        sub = pk["substrates"][name]
        bound = sub["served_vs_channel_bound"]
        check(sub["served_vs_channel"] <= bound, f"{name}/served-equals-decoded",
              f"the signal the pack serves differs from its decoded channel by {sub['served_vs_channel']:.3g}, "
              f"against the codec's own error {bound:.3g} on that channel")

        predicted = set(des["derived"]["tiers_that_hold"])
        for tier, t in sorted(sub["tiers"].items()):
            # a tier the design PREDICTED would hold must hold; the trade note excuses only the tiers the
            # design already said would not, since otherwise a falsified prediction passes unseen
            says = tier in predicted
            ok = t["meets_target"] or (not says and bool(des.get("trade")))
            check(ok, f"{name}/tier-{tier}-{'holds-as-designed' if says else 'target-or-trade'}",
                  f"floor {t['floor']:.3g} and codec error {t['err']:.3g} against the target "
                  f"{t['target_floor']:.3g}; the design record "
                  + (f"predicted this tier would hold at {des['derived']['n_walkers']:,} walkers"
                     if says else "did not predict this tier would hold")
                  + (" and it does" if t["meets_target"] else
                     (" and it does NOT: the pilot's scaling was falsified by the pack's own certificate"
                      if says else f", and states the trade: {str(des.get('trade'))[:160]}")))

        wk = wlk["substrates"].get(name)
        if wk is None:
            raise ReferenceRefusal(
                f"gate: {name!r} is in the pack record and in no walk record, so the gate has no walk to hold it "
                f"to -- neither its counters nor its resident peak nor its budget. The two records describe one "
                f"set of substrates; delete records/pack.json onwards and run the walk stage for it")
        peak, budget = wk.get("peak_rss_bytes"), wk.get("budget_bytes")
        check(peak is None or (budget and peak <= budget), f"{name}/walk-within-budget",
              (f"the walk and its pack stage peaked at {_size(peak)} against the design's {_size(budget)} "
               f"budget" if peak is not None else
               f"the producing run recorded no resident peak for this walk, so the {_size(budget)} budget is "
               f"not checked against a measurement for this pack: "
               + str(wk.get("counters_not_recorded"))[:200]))

        declared = {w["label"] for w in des["waveforms"]}
        served = {w["label"] for w in sub["waveforms"]}
        check(declared == served, f"{name}/envelope-is-the-declared-one",
              f"the pack was checked against {sorted(served)}, the design declares {sorted(declared)}")
        for w in sub["waveforms"]:
            check(w["in_band"], f"{name}/{w['label']}-in-band",
                  f"waveform_band says {w['label']} needs {w['needed_hz']:.3g} Hz; the pack stores "
                  f"{w['pack_band_hz']:.3g} Hz")

        check(bool(sub["pack"]["license"]) and bool(sub["pack"]["citation"]) and bool(sub["pack"]["sha256"]),
              f"{name}/licence-and-digest",
              f"the pack carries {sub['pack']['license']}, its citation and sha256 {sub['pack']['sha256'][:12]}")
        cited = set(spc["substrates"][name]["cites_sha256"])
        known = {f["sha256"] for s in src["sources"].values() for f in s["files"]}
        needs_file = bool(spc["substrates"][name]["cites_files"])
        check(cited <= known and (bool(cited) or not needs_file), f"{name}/spec-cites-the-sources",
              (f"the spec cites {len(cited)} source digests, all of them in the source record" if cited
               else "this substrate is analytic: it has no released file to cite")
              if cited <= known else f"the spec cites digests no source record holds: {sorted(cited - known)}")
        rt = spc["substrates"][name]
        check(rt["round_trips"], f"{name}/spec-round-trips",
              f"spec_of(geometry_from_spec(spec)) == spec in {', '.join(rt.get('round_trip_compared') or ['every field'])}"
              + (f"; not compared: {', '.join(rt['round_trip_not_compared'])} -- the producer's own labelling, "
                 f"which spec_of reconstructs from the geometry and cannot reproduce"
                 if rt.get("round_trip_not_compared") else ""))

        for qname, got in sorted(sub["reproduced"].items()):
            q = next((x for x in ref["quantities"] if x["substrate"] == name and x["name"] == qname), None)
            if q is None:
                check(False, f"{name}/{qname}-has-a-reference", f"{qname} is reproduced but the reference record "
                                                                f"names no such quantity of {name}")
                continue
            check(got["grid"] == q["direct"]["grid"] and got["solver"] == q["direct"]["solver"],
                  f"{name}/{qname}-on-the-recorded-grid",
                  f"the reproduction ran on {got['grid']} with {got['solver']!r}; the reference record's direct "
                  f"measurement is on {q['direct']['grid']} with {q['direct']['solver']!r}")
            ctx = dict(reference=ref, design=des, spec=spc, walk=wlk, pack=sub,
                       quantity=dict(q, replayed=got["value"], replay_se=got["se"]))
            terms, u = tolerance_of(des["tolerance"], ctx)
            k = des["pass_band"]["k"]
            tol = k * u
            rel = abs(got["value"] - q["direct"]["value"]) / abs(q["direct"]["value"])
            check(rel <= tol, f"{name}/reproduces-{qname}",
                  f"replayed {got['value']:.6g} against the recorded direct {q['direct']['value']:.6g}: "
                  f"{rel:.3%} of it, {rel / u:.3g} standard uncertainties, against the design record's band of "
                  f"{k:.4g} ({tol:.3%}) at a declared false-failure rate of "
                  f"{des['pass_band']['false_failure_rate']:g} over {des['pass_band']['n_comparisons']} "
                  f"comparisons; the uncertainty is "
                  + ", ".join(f"{p}={v:.3g}" for p, v in sorted(terms.items())),
                  relative=rel, tolerance=tol, sigma=rel / u, standard_uncertainty=u, k=k, terms=terms)

            # and against THEM: the `reproduces-` check above compares two routes of one engine, and the
            # published number is what the family exists to reproduce. A publication that states an uncertainty
            # is compared within it; one that states none cannot be failed on, so the disagreement is RECORDED
            # as a number and the card's first paragraph states it.
            pub = q["published"]
            if pub["value"] is None:
                check(True, f"{name}/published-{qname}",
                      f"{pub['printed_in']} states no value for this quantity ({pub['uncertainty_is']})",
                      uncertainty_stated=False, relative=None)
            else:
                rel_pub = abs(got["value"] - pub["value"]) / abs(pub["value"])
                u_pub = _quadrature(u, pub["uncertainty"])
                stated = bool(pub["uncertainty"])
                tol_pub = k * u_pub
                check(rel_pub <= tol_pub if stated else True, f"{name}/published-{qname}",
                      f"replayed {got['value']:.6g} against {pub['value']:.6g} printed in {pub['printed_in']}, "
                      f"{pub['locator']}: {rel_pub:.3%} of it, {rel_pub / u_pub:.3g} standard uncertainties"
                      + (f", against the band of {k:.4g} ({tol_pub:.3%})" if stated else
                         f". The publication states no uncertainty ({pub['uncertainty_is']}), so this "
                         f"disagreement cannot be failed on and is recorded instead"),
                      relative=rel_pub, tolerance=tol_pub, sigma=rel_pub / u_pub, standard_uncertainty=u_pub,
                      k=k, uncertainty_stated=stated)
    return checks


def gate_verdict(checks):
    """``(passed, failures, per_substrate)``: a pack ships iff every check naming it and every family-level
    check passed. The check NAME carries which pack it is about, as ``"<substrate>/<check>"``."""
    family = [c for c in checks if c["check"].split("/")[0] in ("source", "reference")]
    names = sorted({c["check"].split("/")[0] for c in checks} - {"source", "reference"})
    per = {}
    for n in names:
        mine = [c for c in checks if c["check"].startswith(n + "/")]
        bad = [c["check"] for c in mine + family if not c["passed"]]
        per[n] = dict(passed=not bad, n_checks=len(mine), failures=bad)
    failures = [f"{c['check']}: {c['detail']}" for c in checks if not c["passed"]]
    return (not failures), failures, per


def _rss_bytes():
    """The process's resident bytes, read from ``VmRSS``. ``statm`` pages times 4096 under-reads by 16 on a
    64 kB-page host, so it is not read here."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise ReferenceRefusal("design: /proc/self/status states no VmRSS, so the memory budget cannot be measured")


class _PeakRSS:
    """The peak resident bytes of the PHASE it wraps, sampled.

    ``ru_maxrss`` is the whole process's high-water mark, so in a process that has already done something large
    -- a pytest session, a family script that walked before it packed -- it says nothing about this phase and
    the "resident bytes per walker" measured from it comes out zero or negative. A sampler measures the phase.
    """

    def __init__(self, interval_s=0.05):
        self.interval, self.peak, self._stop = float(interval_s), _rss_bytes(), None

    def __enter__(self):
        import threading
        self._stop = threading.Event()

        def sample():
            while not self._stop.wait(self.interval):
                self.peak = max(self.peak, _rss_bytes())
        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=self.interval * 10)
        self.peak = max(self.peak, _rss_bytes())
        return False


def _walk_from_pack(name, recorded, budget, design_n, family_dir):
    """The walk record of a pack that already exists: its own header and run record, plus the counters the
    engine reported in the log the declaration cites.

    Nothing is walked. The budget check is the RECORDED peak of that walk and its pack stage -- a measurement,
    where a fresh walk has only the pilot's projection. ``pack_path`` is recorded relative to ``family_dir``
    when the pack lives under it (the same spelling :meth:`ReferenceFamily._pack` reads back), else by
    basename alone -- never the machine's absolute path.
    """
    from .publish import header_of
    if not os.path.exists(recorded.pack_path):
        raise ReferenceRefusal(f"walk {name!r}: {recorded.pack_path} is not there; a recorded walk is read from the "
                               f"pack it produced")
    if not recorded.evidence:
        raise ReferenceRefusal(f"walk {name!r}: a walk record names the log the engine's counters were reported "
                               f"in, or the record that reports none of them; this one names nothing")
    declared = recorded.not_recorded or ()
    if isinstance(declared, str) or not set(declared) <= set(RECORDED_WALK_COUNTERS):
        raise ReferenceRefusal(
            f"walk {name!r}: not_recorded is {declared!r}; it is a tuple of counter NAMES drawn from "
            f"{RECORDED_WALK_COUNTERS}, per pack. A sentence is not a field list -- one that read 'for 118 of the "
            f"600, peak_rss_bytes' was matched as a substring and declared that counter absent on all 600")
    absent = [n for n in ("sub_steps", "illegal_crossings") if getattr(recorded, n) is None]
    unexplained = [n for n in absent if n not in declared]
    if unexplained:
        raise ReferenceRefusal(f"walk {name!r}: {', '.join(unexplained)} is None and not_recorded does not name "
                               f"it ({recorded.not_recorded!r}); a counter the producing run did not report is "
                               f"recorded as absent BY NAME, never left to be read as a measurement, and never "
                               f"filled in by hand")
    meta = header_of(recorded.pack_path)
    wp = meta.get("walk_params") or {}
    run = ((meta.get("provenance") or {}).get("run") or {}).get("walk") or {}
    if not wp.get("n_walkers"):
        raise ReferenceRefusal(f"walk {name!r}: {os.path.basename(recorded.pack_path)} carries no walk_params, so it "
                               f"does not describe its own walk and nothing can be recorded from it")
    if not (run.get("peak_rss_bytes") or (meta.get("provenance") or {}).get("certified", {}).get("peak_rss_gb")) \
            and "peak_rss_bytes" not in declared:
        raise ReferenceRefusal(
            f"walk {name!r}: {os.path.basename(recorded.pack_path)} carries no recorded resident peak for its walk, "
            f"so the memory budget cannot be checked against a measurement for it; name peak_rss_bytes in "
            f"not_recorded to record that, or supply a pack whose walk reported one")
    cert = ((meta.get("provenance") or {}).get("certified") or {})
    peak = int(max(int(run.get("peak_rss_bytes") or 0), int(float(cert.get("peak_rss_gb") or 0.0) * 1e9)))
    if not peak:
        peak = None
    elif peak > budget:
        raise ReferenceRefusal(f"walk {name!r}: the recorded walk and pack peaked at {peak / 1e9:.1f} GB against the "
                               f"design's {budget / 1e9:.1f} GB budget; the budget is a hard cap")
    n = int(wp["n_walkers"])
    from .publish import _size
    note = None if n == int(design_n) else (
        f"the pack was walked at {n:,} walkers where this design's pilot sets {int(design_n):,}; the walk is not "
        f"repeated, and what holds it to the budget is "
        + (f"its own RECORDED peak of {_size(peak)} rather than the pilot's projection" if peak is not None else
           "nothing measured: the producing run recorded no resident peak for it, which the walk record says"))
    abs_pack = os.path.abspath(recorded.pack_path)
    rel_pack = os.path.relpath(abs_pack, family_dir)
    pack_path = rel_pack if not rel_pack.startswith("..") else os.path.basename(abs_pack)
    return dict(from_pack=True, design_n_walkers=int(design_n), design_note=note,
                pack_path=pack_path,
                pack_sha256=_sha256_file(recorded.pack_path), n_walkers=int(wp["n_walkers"]),
                n_t=int(wp["n_t"]), dt_s=float(wp["dt_traj"]),
                sub_steps=None if recorded.sub_steps is None else int(recorded.sub_steps),
                illegal_crossings=(None if recorded.illegal_crossings is None
                                   else int(recorded.illegal_crossings)),
                counters_not_recorded=list(declared) or None, counters_evidence=recorded.evidence,
                seconds=round(float(run.get("wall_s") or 0.0), 1), peak_rss_bytes=peak,
                budget_checked=peak is not None, budget_bytes=int(budget), status=run.get("status"),
                code_commit=(run.get("code") or {}).get("commit"))


# --------------------------------------------------------------- the family
class ReferenceFamily:
    """One SubstrateCommons family, run as the nine stages of :data:`STAGES`.

    ``directory`` holds ``records/``, ``previews/`` and the family's ``README.md``; the packs live wherever
    :class:`Build` says. :meth:`run` runs every stage in order and returns the records; :meth:`stage` runs one,
    and refuses when the stage before it has not run or has changed since.
    """

    def __init__(self, name, directory, *, sources, reference, design, build, publication,
                 resolver=crossref, hub=None):
        self.name = str(name)
        self.dir = os.path.abspath(directory)
        self.sources = tuple(sources)
        self.reference = reference
        self.design = design
        self.build = build
        self.publication = publication
        self.resolver = resolver
        self._hub = hub
        self.records = Records(self.dir)
        if not self.build.specs:
            raise ReferenceRefusal(f"{self.name}: a family has at least one substrate")
        if self.build.walk is None and not self.build.recorded:
            raise ReferenceRefusal(f"{self.name}: build declares neither a walk nor a recorded walk per substrate")

    # ---------------- plumbing
    def _inputs(self, *parts):
        return _sha256_bytes(_canonical(list(parts)))

    def _prev(self, stage):
        i = STAGES.index(stage)
        if i == 0:
            return None, None
        self.records.check_chain(STAGES[i - 1])
        return self.records.read(STAGES[i - 1]), self.records.digest(STAGES[i - 1])

    def stage(self, name):
        """Run one stage. Refuses a name that is not a stage, and a stage whose predecessor has not run.

        A stage whose record is already there for these inputs returns it without measuring again; a stage
        cannot know its inputs digest before it has resolved them, so each stage asks :meth:`Records.reuse` at
        the point where it has and before it measures.
        """
        if name not in STAGES:
            raise ReferenceRefusal(f"{name!r} is not a stage of the reference-pack protocol; the stages are "
                                   f"{', '.join(STAGES)}")
        return getattr(self, f"_{name}")()

    def run(self):
        """Every stage, in order. Returns ``{stage: record}``."""
        return {s: self.stage(s) for s in STAGES}

    def read_all(self, upto="pack"):
        return {s: self.records.read(s) for s in STAGES[:STAGES.index(upto) + 1]}

    def hub(self):
        if self._hub is None:
            from ..fill.hub import Hub
            self._hub = Hub(self.publication.repo)
        return self._hub

    # ---------------- 1. source
    def _source(self):
        """Every input file with its URL, the host's licence VERBATIM, its sha256 and size, the host record.

        Refused without a licence text (a title is not a text) and without a digest per file. The licence is
        copied into ``records/licences/`` and that copy digested too, so an auditor can compare the copy with
        the host without trusting the record.
        """
        out = {}
        for s in self.sources:
            stated = s.licence_id != LICENCE_NONE_STATED
            floor = MIN_LICENCE_TEXT_CHARS if stated else MIN_LICENCE_ABSENT_CHARS
            if not s.licence_text or len(s.licence_text.strip()) < floor:
                raise ReferenceRefusal(
                    f"source {s.key!r}: licence_text is {len(s.licence_text.strip())} characters "
                    f"({s.licence_text.strip()[:60]!r}); "
                    + (f"a licence TITLE is not a licence text -- copy the text at {s.licence_url} verbatim (at "
                       f"least {MIN_LICENCE_TEXT_CHARS} characters)" if stated else
                       f"a host that states no licence is recorded by its own words about rights and by where "
                       f"they were looked for (at least {MIN_LICENCE_ABSENT_CHARS} characters), so the absence "
                       f"can be checked the way a licence would be"))
            if not stated and s.redistributes_bytes:
                raise ReferenceRefusal(
                    f"source {s.key!r}: its host states no licence, and this family declares that it "
                    f"redistributes the source's own bytes; a source whose licence nobody granted is read and "
                    f"cited, never republished (dmipy-sim#459: an unstated source needs the authors' written "
                    f"permission first)")
            if not s.host_record:
                raise ReferenceRefusal(f"source {s.key!r}: no host_record; a source record names the commit or the "
                                       f"version of the host record its files came from")
            if not s.files:
                raise ReferenceRefusal(f"source {s.key!r}: no files")
            lic_dir = os.path.join(self.records.dir, "licences")
            os.makedirs(lic_dir, exist_ok=True)
            lic = os.path.join(lic_dir, f"{s.key}-{s.licence_id}.txt")
            with open(lic, "w") as fh:
                fh.write(s.licence_text)
            files = []
            for f in s.files:
                if not os.path.exists(f.path):
                    raise ReferenceRefusal(f"source {s.key!r}: {f.path} is not there, so it cannot be digested; a "
                                           f"source record without a digest per file is refused")
                files.append(dict(_digest(f.path), cite_as=f.cite_as, role=f.role))
            out[s.key] = dict(url=s.url, host_record=s.host_record, licence_id=s.licence_id,
                              licence_url=s.licence_url, files=files,
                              licence_stated=bool(s.licence_id != LICENCE_NONE_STATED),
                              redistributes_bytes=bool(s.redistributes_bytes),
                              licence_copy=dict(_digest(lic), chars=len(s.licence_text),
                                                first_line=s.licence_text.strip().splitlines()[0][:120]))
        inputs = self._inputs([asdict(s) for s in self.sources])
        return self.records.write("source", dict(family=self.name, sources=out), inputs_digest=inputs,
                                  previous_sha256=None)

    # ---------------- 2. reference
    def _reference(self):
        """The published quantity as data, with the DOI resolved through Crossref and its title compared.

        Refused without a resolved DOI, without a verbatim quotation of the document that PRINTS the number,
        without an analytic standard error on our own direct measurement, and with any free parameter whose
        ownership is unstated. The record never states a grade: :func:`grade_of` assigns it by rule.
        """
        prev, prev_sha = self._prev("reference")
        r = self.reference
        if r is None:
            rec = dict(absent="this family reproduces no published quantity", quantities=[], parameters=[])
            inputs = self._inputs(prev_sha, None)
            return self.records.write("reference", rec, inputs_digest=inputs, previous_sha256=prev_sha)
        if r.published_kind not in PUBLISHED_KINDS:
            raise ReferenceRefusal(f"reference: published_kind is {r.published_kind!r}; it is one of {PUBLISHED_KINDS}")
        if r.sample_relation not in SAMPLE_RELATIONS:
            raise ReferenceRefusal(f"reference: sample_relation is {r.sample_relation!r}; it is one of "
                                   f"{SAMPLE_RELATIONS}")
        if not r.doi:
            raise ReferenceRefusal("reference: no DOI; a reference record without one is refused")
        cr = self.resolver(r.doi)
        matches = _normalise_title(cr.get("title")) == _normalise_title(r.title)
        if not matches:
            raise ReferenceRefusal(f"reference: {r.doi} resolves to {cr.get('title')!r}, the record says "
                                   f"{r.title!r}; the DOI and the title do not describe one work")
        if not r.parameters:
            raise ReferenceRefusal("reference: no free parameters recorded; a comparison rests on parameters and "
                                   "the record says whose each one is")
        params = []
        for p in r.parameters:
            if p.whose not in WHOSE:
                raise ReferenceRefusal(f"reference: the parameter {p.name!r} is marked {p.whose!r}; every free "
                                       f"parameter is one of {WHOSE}")
            if not p.where or not p.how:
                raise ReferenceRefusal(f"reference: the parameter {p.name!r} states no locator or no method")
            params.append(asdict(p))
        known = {name for name in self.build.specs}
        quantities, resolved_documents = [], {}
        for q in r.quantities:
            if q.substrate not in known:
                raise ReferenceRefusal(f"reference: the quantity {q.name!r} is about {q.substrate!r}, which is not "
                                       f"one of this family's substrates {sorted(known)}")
            pub, dr = q.published, q.direct
            if not pub.printed_in or not pub.locator:
                raise ReferenceRefusal(f"reference: {q.substrate}/{q.name} names no document or no locator")
            documents = {}
            if r.published_kind == "data":
                if not pub.data_url or not pub.data_sha256:
                    raise ReferenceRefusal(
                        f"reference: {q.substrate}/{q.name} is read from released data, so it records the data's "
                        f"URL and its sha256; it states {pub.data_url!r} and {pub.data_sha256!r}. A released "
                        f"array is the measurement and needs no printed line, but it needs its digest")
            elif not pub.verbatim.strip():
                raise ReferenceRefusal(f"reference: {q.substrate}/{q.name} quotes nothing from {pub.printed_in!r}; a "
                                       f"reference number is recorded with the verbatim line that prints it, so the "
                                       f"citation can be checked against the document rather than against us")
            if pub.document:
                doc = self.resolver(pub.document)
                if _normalise_title(doc.get("title")) not in _normalise_title(pub.printed_in):
                    raise ReferenceRefusal(
                        f"reference: {q.substrate}/{q.name} says the number is printed in {pub.printed_in!r} and "
                        f"cites {pub.document!r}, which resolves to {doc.get('title')!r}; the identifier and the "
                        f"document do not describe one thing")
                documents[pub.document] = doc
                resolved_documents[pub.document] = doc
            elif r.published_kind in ("number", "figure"):
                raise ReferenceRefusal(
                    f"reference: {q.substrate}/{q.name} names {pub.printed_in!r} as the document that prints the "
                    f"number but cites no identifier for it; the document the number comes FROM is resolved and "
                    f"title-compared, not left as prose (a thesis's DOI is a DataCite one -- Talabi's is "
                    f"10.25560/4261 -- so 'Crossref only' is not a reason to cite the paper instead)")
            if pub.uncertainty is None or not pub.uncertainty_is:
                raise ReferenceRefusal(f"reference: {q.substrate}/{q.name} states no uncertainty; a publication that "
                                       f"states none records 0.0 and says so in uncertainty_is")
            if dr.se_kind not in SE_KINDS:
                raise ReferenceRefusal(f"reference: {q.substrate}/{q.name}'s direct measurement has se_kind "
                                       f"{dr.se_kind!r}; a standard error is analytic ({', '.join(SE_KINDS)}), never "
                                       f"a fold spread or a split half, which are seed-dependent")
            if not (dr.se > 0.0) or dr.se >= 1.0:
                raise ReferenceRefusal(f"reference: {q.substrate}/{q.name}'s direct standard error is {dr.se!r}; it is "
                                       f"a relative standard error in (0, 1)")
            if not dr.se_derivation or not dr.grid or not dr.solver:
                raise ReferenceRefusal(f"reference: {q.substrate}/{q.name}'s direct measurement states no se "
                                       f"derivation, grid or solver; the grid is part of the measurement")
            quantities.append(dict(substrate=q.substrate, name=q.name, published=asdict(pub), direct=asdict(dr),
                                   documents=documents))
        if not quantities:
            raise ReferenceRefusal("reference: a reference record with a DOI records at least one quantity")
        ours_geometry = sorted(p["name"] for p in params if p["whose"] == "ours" and p["changes_geometry"])
        rec = dict(doi=r.doi, title=r.title, crossref=cr, title_matches=bool(matches),
                   published_kind=r.published_kind, same_released_geometry=not ours_geometry,
                   geometry_parameters_ours=ours_geometry,
                   geometry_derivation=(
                       "the geometry is the released one when no free parameter marked 'ours' changes it; "
                       + (f"these do: {', '.join(ours_geometry)}" if ours_geometry else
                          "none of this family's parameters does")),
                   sample=r.sample, sample_relation=r.sample_relation, quantities=quantities, parameters=params,
                   description=r.description, source_note=r.source_note, licence_note=r.licence_note,
                   caveats=dict(r.caveats))
        # the resolution TIME is an output of this stage, not an input to it: including it made the inputs
        # digest change on every run, so the real resolver could never return the record it had written and the
        # stage refused itself. The test resolver returns a fixed timestamp, which is why the tests did not see it
        inputs = self._inputs(prev_sha, asdict(r),
                              {k: v for k, v in sorted(cr.items()) if k != "resolved"},
                              {i: {k: v for k, v in sorted(d.items()) if k != "resolved"}
                               for i, d in sorted(resolved_documents.items())} or None)
        return self.records.write("reference", rec, inputs_digest=inputs, previous_sha256=prev_sha)

    # ---------------- 3. spec
    def _spec(self):
        """The substrate spec per pack, round-tripped, citing the source files by their digests."""
        prev, prev_sha = self._prev("spec")
        src = self.records.read("source")
        known = {f["sha256"]: f["cite_as"] for s in src["sources"].values() for f in s["files"]}
        from ..spec import geometry_from_spec, spec_of, validate
        out = {}
        for name, make in sorted(self.build.specs.items()):
            spec = make()
            validate(spec.to_dict())
            cited = sorted({w.surface.sha256 for w in spec.walls if w.surface.sha256})
            unknown = [c for c in cited if c not in known]
            if unknown:
                raise ReferenceRefusal(f"spec {name!r}: it cites the digests {unknown}, which the source record does "
                                       f"not hold; a spec cites its source files by the digest the source stage "
                                       f"measured")
            if not cited and any(w.surface.file for w in spec.walls):
                raise ReferenceRefusal(f"spec {name!r}: a wall cites a file but no sha256; the source record's digest "
                                       f"is what ties the spec to the released bytes")
            back = spec_of(geometry_from_spec(spec))
            a = spec.to_dict()                                 # the record carries the spec WHOLE
            mine, theirs = _pruned(a), _pruned(back.to_dict())  # the comparison drops what spec_of cannot make
            compared = sorted(set(mine) | set(theirs))
            differing = [k for k in compared if mine.get(k) != theirs.get(k)]
            if differing:
                raise ReferenceRefusal(f"spec {name!r}: spec_of(geometry_from_spec(spec)) differs in {differing}; a "
                                       f"spec the engine cannot rebuild is not the substrate that was walked")
            out[name] = dict(spec=a, spec_sha256=_sha256_bytes(_canonical(a)), cites_sha256=cited,
                             cites=[known[c] for c in cited], round_trips=True,
                             round_trip_compared=compared, round_trip_not_compared=list(ROUND_TRIP_NOT_COMPARED),
                             cites_files=bool(any(w.surface.file for w in spec.walls)),
                             realisation=spec.realisation, pack_id=self.build.pack_id[name])
        inputs = self._inputs(prev_sha, sorted(out), [v["spec_sha256"] for v in out.values()])
        return self.records.write("spec", dict(substrates=out), inputs_digest=inputs, previous_sha256=prev_sha)

    # ---------------- 4. design
    def _design(self):
        """The declared envelope, the target floor per tier, the memory budget, and a PILOT measured on the
        real window from which the walker count follows.

        The count is never an estimate: the pilot walks and PACKS ``pilot_n`` walkers over this family's own
        window and save grid, and the record carries the measured floor, the measured bytes per walker of the
        pack file and the measured resident bytes per walker of the pack stage. The floor scaling gives one
        count, the memory budget another, and the smaller wins; when they cannot both hold, the record states
        the trade and which tier holds.
        """
        prev, prev_sha = self._prev("design")
        spc, ref = self.records.read("spec"), self.records.read("reference")
        d = self.design
        waveform_labels = [label for label, _ in self.design.waveforms()]
        inputs = self._inputs(prev_sha, dict(asdict(d), waveforms=None, envelope=d.envelope), waveform_labels)
        held = self.records.reuse("design", inputs, prev_sha)
        if held is not None:                                  # unchanged inputs: the pilot is not walked again
            return held
        if self.build.walk is None:
            raise ReferenceRefusal("design: the pilot walks, so build.walk is required even for a family whose packs "
                                   "already exist; the pilot is what makes the walker count measured")
        n_t = int(round(d.window_s / d.dt_save_s)) + 1
        if d.pilot_substrate is not None and d.pilot_substrate not in spc["substrates"]:
            raise ReferenceRefusal(f"design: the pilot is declared on {d.pilot_substrate!r}, which is not one of "
                                   f"this family's {len(spc['substrates'])} substrates; the pilot walks a "
                                   f"substrate of the family")
        name = d.pilot_substrate or sorted(spc["substrates"])[0]
        spec = self.build.specs[name]()
        rss0 = _rss_bytes()
        t0 = time.time()
        with _PeakRSS() as sampler:
            walk = self.build.walk(spec, int(d.pilot_n), n_t=n_t)
        walk_s, walk_peak = time.time() - t0, sampler.peak
        if int(walk.positions.shape[1]) != n_t:
            raise ReferenceRefusal(f"design: the pilot recorded {walk.positions.shape[1]} saves, the design's window "
                                   f"is {n_t}; a pilot on a shorter window measures another pack's memory")
        from .bank import _master_arrays, _measure_floor, build_replay_pack
        floor0 = float(_measure_floor(_master_arrays(walk), d.envelope))
        t1 = time.time()
        import tempfile
        with _PeakRSS() as sampler:
            pk = build_replay_pack(walk, id=f"pilot/{self.name}", license="pilot", citation="pilot", K=int(d.K),
                                   envelope=d.envelope, position_container="bands", blt_container="bands",
                                   verbose=False)
            with tempfile.TemporaryDirectory(prefix="dmipy-pilot-") as tmp:
                f = os.path.join(tmp, "pilot.rpk")
                pk.save(f)
                pack_bytes = os.path.getsize(f)
        pack_s, pack_peak = time.time() - t1, sampler.peak
        per_walker_bytes = pack_bytes / float(d.pilot_n)
        per_walker_resident = max(pack_peak - rss0, 0) / float(d.pilot_n)
        # the resident cost of a pack cannot be under the bytes it holds, and the file's bytes per walker are
        # always measurable where a small pilot's resident increment can sit under the sampler's resolution
        per_walker_rss = max(per_walker_resident, per_walker_bytes)
        if not (per_walker_bytes > 0.0):
            raise ReferenceRefusal(f"design: the pilot's pack is {pack_bytes} bytes at {d.pilot_n} walkers, so it "
                                   f"measured {per_walker_bytes} bytes per walker; the walker count is set from a "
                                   f"measurement, never an estimate")
        fid = pk.meta["fidelity"]
        tiers, tier_floor = [], {}
        for t in d.tiers:
            if t.floor_key not in fid or t.err_key not in fid:
                raise ReferenceRefusal(f"design: the tier {t.name!r} reads {t.floor_key!r} / {t.err_key!r}, which the "
                                       f"pilot pack's certificate does not carry ({sorted(k for k in fid if isinstance(fid[k], (int, float)))}); "
                                       f"a declared tier is certified, never 'not applicable'")
            tiers.append(asdict(t))
            tier_floor[t.name] = float(fid[t.floor_key] or 0.0)
        target = min(float(t.target_floor) for t in d.tiers)
        n_floor = max(int(d.pilot_n), int(round(d.pilot_n * (floor0 / target) ** 2 * float(d.safety))))
        n_budget = int(max(0.0, (float(d.memory_budget_bytes) - rss0)) / per_walker_rss)
        n_walkers = min(n_floor, n_budget)                # the pilot already walked pilot_n, so that is the floor
        if n_walkers < int(d.pilot_n):
            raise ReferenceRefusal(
                f"design: the {d.memory_budget_bytes / 1e9:.1f} GB budget allows {n_budget:,} walkers, fewer than the "
                f"{d.pilot_n:,} the pilot already walked ({per_walker_rss / 1e6:.2f} MB resident per walker, "
                f"measured); this window cannot be packed within the budget at any useful count")
        trade = None
        holds = sorted(t.name for t in d.tiers
                       if tier_floor[t.name] * (float(d.pilot_n) / max(n_walkers, 1)) ** 0.5 <= float(t.target_floor))
        if n_floor > n_budget:
            trade = (f"the target floor {target:g} needs {n_floor:,} walkers and the {d.memory_budget_bytes / 1e9:.0f} GB "
                     f"memory budget allows {n_budget:,} (measured: {per_walker_rss / 1e6:.2f} MB resident per walker in "
                     f"the pack stage on this window). The design therefore sets {n_walkers:,} walkers; the tier that "
                     f"holds its target at that count is {', '.join(holds) or 'none'}, and every tier's achieved floor "
                     f"is certified in the pack and shown on the card.")
        waveforms, b_bound = [], max(float(x) for x in (d.envelope.get("bvals") or (0.0,)))
        for label, seq in self.design.waveforms():
            if label in {w["label"] for w in waveforms}:
                raise ReferenceRefusal(f"design: the waveform {label!r} is declared twice; the envelope is a SET of "
                                       f"waveforms and a duplicate hides one of them behind another's label")
            b = seq.b()
            if float(max(b)) > b_bound * (1.0 + 1e-6):
                raise ReferenceRefusal(
                    f"design: the waveform {label!r} reaches b = {float(max(b)):.4g} s/m^2 where the envelope the "
                    f"pack is BUILT to certifies {b_bound:.4g}; the envelope and the declared waveforms are one "
                    f"declaration, not two")
            waveforms.append(dict(label=label, b_max=float(max(b)), n_measurements=int(len(b)),
                                  T_s=float(seq.T), family=getattr(seq, "family", None)))
        if not waveforms:
            raise ReferenceRefusal("design: no waveform declared; an envelope a consumer is held to is a list of "
                                   "waveforms, and the pack stage checks every one against waveform_band")
        tol = dict(terms=list(self.design.tolerance.terms), combine=self.design.tolerance.combine)
        systematics = {}
        for s in d.systematics:
            if not s.measured_on or not s.evidence:
                raise ReferenceRefusal(f"design: the systematic {s.name!r} states no measurement or no evidence; a "
                                       f"term of the gate's budget is measured and carries the numbers it was "
                                       f"measured from")
            systematics[s.name] = asdict(s)
        # the number of comparisons the gate will make is a property of the reference record, so the band is
        # known before any of them is made: one `reproduces-` and one `published-` per recorded quantity
        n_cmp = 2 * max(1, len(ref.get("quantities") or ()))
        band = dict(pass_band(n_cmp, max(n_walkers - 1, 1), false_failure_rate=float(d.false_failure_rate)),
                    counted="one reproduces- and one published- comparison per quantity of the reference record",
                    dof_is="the walker count the design sets, minus one: the smallest count any term's standard "
                           "error is measured over")
        rec = dict(window_s=float(d.window_s), dt_save_s=float(d.dt_save_s), n_t=n_t, pass_band=band,
                   save_grid_why=d.save_grid_why, K=int(d.K), envelope=d.envelope, waveforms=waveforms,
                   tiers=tiers, memory_budget_bytes=int(d.memory_budget_bytes), tolerance=tol,
                   systematics=systematics,
                   pilot=dict(n_walkers=int(d.pilot_n), on_real_window=True, n_t=n_t, substrate=name,
                              substrate_is=("declared" if d.pilot_substrate else
                                            "the first of this family's substrates in sorted order, undeclared"),
                              walk_seconds=round(walk_s, 1), pack_seconds=round(pack_s, 1),
                              sub_steps=int(walk.sub_steps), illegal_crossings=int(walk.illegal_crossings or 0),
                              envelope_floor=floor0, tier_floor=tier_floor, pack_bytes=int(pack_bytes),
                              bytes_per_walker=per_walker_bytes, rss_bytes_per_walker=per_walker_rss,
                              resident_bytes_per_walker_sampled=per_walker_resident,
                              budget_is_set_by=("the sampled resident increment of the pack stage"
                                                if per_walker_resident >= per_walker_bytes else
                                                "the pack file's bytes per walker: the sampled resident "
                                                "increment at this pilot count was below it"),
                              rss_bytes_before=int(rss0), peak_rss_bytes_walk=int(walk_peak),
                              peak_rss_bytes_pack=int(pack_peak), codec_err=float(fid["err_max"]),
                              band_hz=float(pk.temporal_bandwidth_hz)),
                   derived=dict(target_floor=target, n_for_target_floor=n_floor, n_for_budget=n_budget,
                                n_walkers=n_walkers, tiers_that_hold=holds,
                                rule="the pilot's measured floor scaled as 1/sqrt(N) with the declared safety "
                                     "factor, capped by the pilot's measured resident bytes per walker against the "
                                     "memory budget; never an estimate"),
                   trade=trade)
        return self.records.write("design", rec, inputs_digest=inputs, previous_sha256=prev_sha)

    # ---------------- 5. walk
    def _walk(self):
        """The walk at the design, with the engine's own counters, and the memory budget checked before it.

        A family whose packs already exist declares a :class:`RecordedWalk` per substrate: the walk record is
        then written from the pack's own header and run record plus the counters the engine reported in the log
        that is cited, and nothing is walked again.

        The budget is enforced twice and the two are different claims. BEFORE the walk it is the design's cap:
        the count came from ``min(n_floor, n_budget)``, so the projection cannot exceed the budget and the check
        here is an assertion that the record it reads is the one the design wrote -- it fires when a design
        record has been produced another way. AFTER the walk it is a MEASUREMENT: the walk's own sampled peak,
        which the pilot's per-walker projection can under-read, and which refuses at the boundary.
        """
        prev, prev_sha = self._prev("walk")
        des = self.records.read("design")
        spc = self.records.read("spec")
        # the RecordedWalk contents, not just the names: they are WHAT the record says, so a change to one has
        # to invalidate the record rather than be returned unchanged by `reuse`
        inputs = self._inputs(prev_sha, sorted(spc["substrates"]),
                              {k: asdict(v) for k, v in sorted(self.build.recorded.items())})
        held = self.records.reuse("walk", inputs, prev_sha)
        if held is not None:                                  # unchanged inputs: nothing is walked again
            return held
        budget = int(des["memory_budget_bytes"])
        per_walker = float(des["pilot"]["rss_bytes_per_walker"])
        out = {}
        for name in sorted(spc["substrates"]):
            if name in self.build.recorded:
                r = self.build.recorded[name]
                out[name] = _walk_from_pack(name, r, budget, int(des["derived"]["n_walkers"]), self.dir)
            else:
                n = int(des["derived"]["n_walkers"])
                projected = int(des["pilot"]["rss_bytes_before"]) + int(per_walker * n)
                if projected > budget:
                    raise ReferenceRefusal(
                        f"walk {name!r}: {n:,} walkers project to {projected / 1e9:.1f} GB resident in the pack stage "
                        f"(measured {per_walker / 1e6:.2f} MB per walker) against the {budget / 1e9:.1f} GB budget; "
                        f"the design's own cap is min(n_floor, n_budget), so this record did not come from that "
                        f"derivation")
                out[name] = self._walk_now(name, n, des, projected)
        return self.records.write("walk", dict(substrates=out), inputs_digest=inputs, previous_sha256=prev_sha)

    def _walk_now(self, name, n, des, projected):
        spec = self.build.specs[name]()
        rss0 = _rss_bytes()
        t0 = time.time()
        with _PeakRSS() as sampler:
            walk = self.build.walk(spec, n, n_t=int(des["n_t"]))
        seconds, peak = time.time() - t0, sampler.peak
        out_dir = os.path.join(self.dir, "walks")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name.lower()}.safetensors")
        walk.save(path)
        if peak > int(des["memory_budget_bytes"]):        # the MEASURED peak, which the projection can under-read
            raise ReferenceRefusal(
                f"walk {name!r}: the walk peaked at {peak / 1e9:.2f} GB resident against the "
                f"{int(des['memory_budget_bytes']) / 1e9:.2f} GB budget, where the pilot projected "
                f"{projected / 1e9:.2f} GB; the budget is a hard cap and the measurement, not the projection, "
                f"is what it is held against")
        return dict(from_pack=False, walk_path=os.path.relpath(path, self.dir), walk_sha256=_sha256_file(path),
                    n_walkers=int(n), n_t=int(walk.positions.shape[1]), dt_s=float(walk.dt),
                    sub_steps=int(walk.sub_steps), illegal_crossings=int(walk.illegal_crossings or 0),
                    seconds=round(seconds, 1), peak_rss_bytes=int(peak), rss_bytes_before=int(rss0),
                    projected_rss_bytes=int(projected), budget_bytes=int(des["memory_budget_bytes"]),
                    code_commit=code_commit(require_clean=False))

    # ---------------- 6. pack
    def _pack(self):
        """The pack per substrate on the declared envelope, its certificate per tier with ``meets_target``, the
        served-vs-decoded check, every declared waveform against ``waveform_band``, and the reproduced quantity.

        These are the measurements the gate reads. A family whose packs exist reads them as they are: the pack
        is neither re-walked nor re-encoded, and its certificate is the one it was built with.
        """
        prev, prev_sha = self._prev("pack")
        des, ref, wlk = (self.records.read(s) for s in ("design", "reference", "walk"))
        inputs = self._inputs(prev_sha, sorted(wlk["substrates"]))
        held = self.records.reuse("pack", inputs, prev_sha)
        if held is not None:                                  # unchanged inputs: nothing is re-measured
            return held
        from .replay import ReplayPack
        out = {}
        for name in sorted(wlk["substrates"]):
            w = wlk["substrates"][name]
            if w["from_pack"]:
                local = w["pack_path"] if os.path.isabs(w["pack_path"]) else os.path.join(self.dir, w["pack_path"])
                pack = ReplayPack.load(local)
            else:
                local, pack = self._build_pack(name, w, des)
            out[name] = self._certify(name, pack, local, des, ref)
        return self.records.write("pack", dict(substrates=out), inputs_digest=inputs, previous_sha256=prev_sha)

    def _build_pack(self, name, w, des):
        from ..persistent_walk import PersistentWalk
        from .bank import build_replay_pack
        walk = PersistentWalk.load(os.path.join(self.dir, w["walk_path"]))
        pack = build_replay_pack(walk, id=self.build.pack_id[name], envelope=des["envelope"], K=int(des["K"]),
                                 position_container="bands", blt_container="bands",
                                 license=self.publication.licence, citation=self.publication.citation,
                                 verbose=False)
        out_dir = os.path.join(self.dir, "packs")
        os.makedirs(out_dir, exist_ok=True)
        local = os.path.join(out_dir, f"{name.lower()}.rpk")
        pack.save(local)
        return local, pack

    def _certify(self, name, pack, local, des, ref):
        fid = pack.meta["fidelity"]
        tiers = {}
        for t in des["tiers"]:
            floor, err = fid.get(t["floor_key"]), fid.get(t["err_key"])
            if floor is None or err is None:
                raise ReferenceRefusal(f"pack {name!r}: the tier {t['name']!r} has no {t['floor_key']} or "
                                       f"{t['err_key']} in the pack's certificate; a declared tier is certified, "
                                       f"never 'not applicable'")
            floor, err = float(floor), float(err)
            tiers[t["name"]] = dict(floor=floor, err=err, target_floor=float(t["target_floor"]),
                                    meets_target=bool(floor <= t["target_floor"] and err <= t["target_floor"]))
        served_tier = self.build.served_tier
        if served_tier not in tiers:
            raise ReferenceRefusal(f"pack {name!r}: the served-vs-decoded check is declared against the tier "
                                   f"{served_tier!r}, which the design does not declare ({sorted(tiers)})")
        waveforms = []
        for label, seq in self.design.waveforms():
            hz = float(pack.waveform_band(seq)[0])
            waveforms.append(dict(label=label, needed_hz=hz, pack_band_hz=float(pack.temporal_bandwidth_hz),
                                  in_band=bool(hz <= pack.temporal_bandwidth_hz)))
        reproduced = {}
        for q in ref.get("quantities", []):
            if q["substrate"] != name:
                continue
            got = self.build.reproduce(pack, q, q["direct"]["grid"])
            for key in ("value", "se", "se_kind", "se_derivation", "solver"):
                if got.get(key) in (None, ""):
                    raise ReferenceRefusal(f"pack {name!r}: the reproduction of {q['name']!r} states no {key}")
            if got["se_kind"] not in SE_KINDS:
                raise ReferenceRefusal(f"pack {name!r}: the reproduction of {q['name']!r} has se_kind "
                                       f"{got['se_kind']!r}; a standard error is analytic "
                                       f"({', '.join(SE_KINDS)}), never a fold spread or a split half")
            reproduced[q["name"]] = dict(got, grid=q["direct"]["grid"], unit=q["direct"]["unit"])
        from .publish import _built_commit_of
        cx = pack.meta.get("compression") or {}
        return dict(
            pack=dict(_digest(local), path_local=os.path.relpath(local, self.dir) if local.startswith(self.dir) else local,
                      id=pack.meta["id"], license=pack.meta.get("license"), citation=pack.meta.get("citation"),
                      K=int(pack.K), n_t=int(pack.n_t), dt_s=float(pack.dt), T_s=float((pack.n_t - 1) * pack.dt),
                      n_walkers=int(pack.n_walkers), band_hz=float(pack.temporal_bandwidth_hz),
                      channels=["positions", *sorted(cx.get("channels") or {})],
                      substrate_id=(pack.meta.get("substrate") or {}).get("id"),
                      built_commit=_built_commit_of(pack.meta)),
            certificate={k: v for k, v in fid.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
            tiers=tiers, served_vs_channel=float(self.build.served_vs_channel(pack)),
            served_vs_channel_bound=float(tiers[served_tier]["err"]), served_tier=served_tier,
            waveforms=waveforms, reproduced=reproduced)

    # ---------------- 7. gate
    def _gate(self):
        """The gate: deterministic, reading only the records, failing by name.

        Every number it compares was measured and written by the stage that measured it, so it never walks,
        replays or inverts anything, and no tolerance constant lives here: the budget is the design record's
        terms resolved over the records and combined in quadrature.
        """
        prev, prev_sha = self._prev("gate")
        rec = self.read_all("pack")
        checks = _gate_checks(rec)
        passed, failures, per = gate_verdict(checks)
        out = dict(passed=passed, n_checks=len(checks), checks=checks, failures=failures, per_substrate=per,
                   grade=grade_of(rec["reference"]),
                   inputs={s: self.records.digest(s) for s in STAGES[:STAGES.index("gate")]})
        inputs = self._inputs(prev_sha, out["inputs"])
        return self.records.write("gate", out, inputs_digest=inputs, previous_sha256=prev_sha)

    # ---------------- 8. card
    def _card(self):
        """The dataset card, rendered from the records (dmipy-sim#480), with its previews and its snippet RUN.

        Three orthogonal cross-sections per substrate come from :func:`dmipy_sim.spec.preview`, the manifest
        table from :func:`dmipy_sim.replay.publish.manifest_row` on the packs themselves, the reproduction table
        and its grade from the reference and pack records, and the gate's verdict per pack from the gate record.
        The "Use me" snippet is executed here with a ceiling of :data:`SNIPPET_CEILING_S`: a card whose snippet
        fails or times out is not written, so a card cannot teach a number it does not produce.
        """
        prev, prev_sha = self._prev("card")
        rec = self.read_all("pack")
        gate = self.records.read("gate")
        # the hold is rendered onto the card, so a change to it must re-render rather than be reused
        inputs = self._inputs(prev_sha, self.publication.repo, self.publication.snippet_substrate,
                              dict(sorted((self.publication.hold or {}).items())))
        held = self.records.reuse("card", inputs, prev_sha)
        if held is not None:                                  # unchanged inputs: the snippet is not run again
            return held
        from ..spec import SubstrateSpec, preview as spec_preview
        prev_dir = os.path.join(self.dir, "previews")
        previews = {}
        for name, s in sorted(rec["spec"]["substrates"].items()):
            path = os.path.join(prev_dir, f"{name.lower()}.png")
            previews[name] = spec_preview(SubstrateSpec.from_dict(s["spec"]), path,
                                          title=f"{name}: {s['spec'].get('description') or s['spec']['id']}")
        sub = self.publication.snippet_substrate
        if sub not in rec["pack"]["substrates"]:
            raise ReferenceRefusal(f"card: the snippet reads {sub!r}, which is not one of this family's packs "
                                   f"({sorted(rec['pack']['substrates'])})")
        row = rec["pack"]["substrates"][sub]["pack"]
        hub_path = self._hub_path(sub, row)
        shown = self.publication.snippet(f"hf://{self.publication.repo}/{hub_path}")
        local = row.get("path_local") or row["path"]
        local = local if os.path.isabs(local) else os.path.join(self.dir, local)
        ran = _run_snippet(self.publication.snippet(local), self.dir)
        # the commit of the code rendering this card right now (dmipy-sim#499): an OUTPUT of this stage, like the
        # reference stage's resolution time, never an input -- a later commit alone does not make a pack's card
        # stale, and the publish stage below is what checks this against the commit it is actually publishing at
        published_commit = code_commit(require_clean=False)
        card = _render_card(self.name, self.publication.repo, rec, gate, previews, shown, ran, hub_path,
                            hold=dict(self.publication.hold or {}), published_commit=published_commit)
        path = os.path.join(self.dir, "README.md")
        with open(path, "w") as fh:
            fh.write(card)
        out = dict(card=dict(_digest(path), chars=len(card)), previews=previews, grade=gate["grade"],
                   hold=dict(self.publication.hold or {}), published_commit=published_commit,
                   snippet=dict(shown_sha256=_sha256_bytes(shown.encode()), executed_on=os.path.basename(local),
                                executed_sha256=_sha256_bytes(self.publication.snippet(local).encode()),
                                seconds=ran["seconds"], stdout=ran["stdout"], ceiling_s=SNIPPET_CEILING_S,
                                substitution="the card shows the snippet reading the pack at its hub URI and it was "
                                             "executed against the local file of the same sha256, so the only "
                                             "difference between the two is the download"))
        return self.records.write("card", out, inputs_digest=inputs, previous_sha256=prev_sha)

    def _hub_path(self, name, row):
        if self.publication.pack_path is not None:
            return self.publication.pack_path(name)
        return f"packs/{str(row['id']).replace('/', '-')}.rpk"

    # ---------------- 9. publish
    def _publish(self):
        """The packs the gate passed, the records, the previews and the card, to the dataset the family names.

        The pack list comes from the gate's record and never from a glob; a pack the gate has not seen is
        fatal, since not being seen by the gate is what the gate is for. The code commit must be clean and
        pushed. Creating the dataset is an explicit, logged step. The organisation page is regenerated from the
        hub afterwards.
        """
        prev, prev_sha = self._prev("publish")
        rec = self.read_all("pack")
        gate, card = self.records.read("gate"), self.records.read("card")
        self.records.check_chain("card")
        for s, sha in gate["inputs"].items():
            if self.records.digest(s) != sha:
                raise ReferenceRefusal(f"publish: records/{s}.json changed after the gate ran "
                                       f"({sha[:12]} -> {self.records.digest(s)[:12]}); run the gate again")
        create_dataset, dry = bool(self.publication.create_dataset), bool(self.publication.dry)
        commit = code_commit()
        if card.get("published_commit") != commit:
            raise ReferenceRefusal(
                f"publish: the card states the publishing commit as {card.get('published_commit')!r}; this "
                f"publish is at {commit!r} -- run the card stage again so the card and the manifest name the "
                f"same one")
        per = gate["per_substrate"]
        passing = sorted(n for n in rec["pack"]["substrates"] if per.get(n, {}).get("passed"))
        withheld = sorted(n for n in rec["pack"]["substrates"] if not per.get(n, {}).get("passed"))
        if not passing:
            raise ReferenceRefusal(f"publish: the gate passed no pack of this family (withheld: {withheld}); "
                                   f"the failures are\n  " + "\n  ".join(gate["failures"]))
        created = None
        if create_dataset and not dry:
            from huggingface_hub import HfApi
            HfApi().create_repo(self.publication.repo, repo_type="dataset", exist_ok=True)
            created = dict(repo=self.publication.repo, at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            log.info("publish: dataset %s created (explicit)", self.publication.repo)
        hub = None if dry else self.hub()
        from .publish import MANIFEST, _load_manifest, publish as publish_pack
        already = {} if dry else {r.get("path"): r.get("sha256") for r in (_load_manifest(hub).get("packs") or [])}
        uploaded, kept = [], []
        for name in passing:
            row = rec["pack"]["substrates"][name]["pack"]
            path = self._hub_path(name, row)
            local = row.get("path_local") or row["path"]
            local = local if os.path.isabs(local) else os.path.join(self.dir, local)
            if already.get(path) == row["sha256"]:
                kept.append(dict(substrate=name, path=path, sha256=row["sha256"],
                                 why="the manifest already holds these bytes; nothing re-uploaded"))
                continue
            if dry:
                uploaded.append(dict(substrate=name, path=path, sha256=row["sha256"], uri=None, dry=True))
                continue
            uri = publish_pack(local, self.publication.repo, path=path, hub=hub, published_commit=commit,
                               message=f"{self.name}: {name} ({row['n_walkers']:,} walkers, K={row['K']}, "
                                       f"{row['T_s']:g} s at {row['dt_s'] * 1e6:.0f} us)")
            uploaded.append(dict(substrate=name, path=path, sha256=row["sha256"], uri=uri))
        # the gate's verdict belongs in the manifest, which is what a consumer reads: a withheld pack was left
        # advertised beside the others with nothing to say it had failed (its bytes are the maintainer's to
        # delete, not this stage's). Every row this family owns gets its verdict, including one uploaded by an
        # earlier run.
        held = dict(self.publication.hold or {})
        unknown = sorted(set(held) - set(rec["pack"]["substrates"]))
        if unknown:
            raise ReferenceRefusal(f"publish: a hold is declared on {unknown}, which this family has no pack of")
        failed_and_held = sorted(n for n in held if not per.get(n, {}).get("passed"))
        if failed_and_held:
            raise ReferenceRefusal(
                f"publish: {failed_and_held} are declared HELD and the gate FAILED them, so they are withheld "
                f"already; a hold is a reservation about a pack that ships, not a second way to withhold one")
        verdicts = {}
        manifest = None if dry else _load_manifest(hub)
        for name, sub in sorted(rec["pack"]["substrates"].items()):
            v = per.get(name, {})
            row = dict(gate="pass" if v.get("passed") else "fail", withheld=not v.get("passed"),
                       gate_failures=sorted(v.get("failures") or []), gate_record="records/gate.json")
            if name in held:                       # a DECLARED reservation, beside the verdict and never over it
                row["hold"] = held[name]
                row["held"] = True
            verdicts[self._hub_path(name, sub["pack"])] = row
        if manifest is not None:
            for row in manifest.get("packs") or []:
                if row.get("path") in verdicts:
                    row.update(verdicts[row["path"]])
        adds = {"README.md": os.path.join(self.dir, "README.md")}
        if manifest is not None:
            adds[MANIFEST] = json.dumps(manifest, indent=1).encode()
        for s in STAGES[:STAGES.index("card") + 1]:
            adds[f"records/{s}.json"] = self.records.path(s)
        for f in sorted(os.listdir(os.path.join(self.records.dir, "licences"))):
            adds[f"records/licences/{f}"] = os.path.join(self.records.dir, "licences", f)
        for name, p in sorted(card["previews"].items()):
            adds[f"previews/{p['path']}"] = os.path.join(self.dir, "previews", p["path"])
        if not dry:
            hub.commit(adds, [], f"{self.name}: the records, the previews and the card (protocol dmipy-sim#482)",
                       parent=hub.head())
        org = _organisation_after_publish(self.publication.repo, skip=(dry or not _is_the_real_hub(hub)))
        out = dict(repo=self.publication.repo, code_commit=commit, dataset_created=created, dry=bool(dry),
                   uploaded=uploaded, kept=kept, withheld=withheld, holds=held, manifest_verdicts=verdicts,
                   files=sorted(adds), organisation=org,
                   # when this ran, never who or on what machine
                   at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        inputs = self._inputs(prev_sha, [u["sha256"] for u in uploaded], sorted(adds), bool(dry))
        return self.records.write("publish", out, inputs_digest=inputs, previous_sha256=prev_sha)


# --------------------------------------------------------------- the card
def _run_snippet(source, cwd):
    """Run a card's "Use me" snippet with a ceiling of :data:`SNIPPET_CEILING_S` and return its output.

    A snippet that fails, or that a reader would not wait for, is not a snippet: the card is not written.
    """
    import sys
    import tempfile
    from .. import __file__ as pkg
    root = os.path.dirname(os.path.dirname(os.path.abspath(pkg)))
    # the snippet runs against the code that built the card, not against whatever is installed on the box: a
    # non-editable install in site-packages shadows a worktree the moment the cwd is not that worktree
    env = dict(os.environ, JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"),
               XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join([root, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
    with tempfile.TemporaryDirectory(prefix="dmipy-snippet-") as tmp:
        f = os.path.join(tmp, "use_me.py")
        with open(f, "w") as fh:
            fh.write(source)
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, f], capture_output=True, text=True, cwd=cwd, env=env,
                               timeout=SNIPPET_CEILING_S)
        except subprocess.TimeoutExpired as e:
            raise ReferenceRefusal(f"card: the \"Use me\" snippet did not finish in {SNIPPET_CEILING_S:g} s; a card "
                                   f"whose snippet a reader will not wait for is not written") from e
        seconds = time.time() - t0
    if r.returncode:
        raise ReferenceRefusal(f"card: the \"Use me\" snippet failed (exit {r.returncode}); a card that teaches a "
                               f"number it does not produce is not written:\n{r.stderr[-2000:]}")
    return dict(stdout=r.stdout.strip(), seconds=round(seconds, 1))


def _disagreements(gate):
    """Every comparison the card's first paragraph must state: a failed one, and a `published-` comparison
    whose disagreement is outside its band although the publication states no uncertainty to fail it on."""
    out = []
    for c in gate["checks"]:
        if "/reproduces-" not in c["check"] and "/published-" not in c["check"]:
            continue
        if not c["passed"]:
            out.append(c)
        elif c.get("relative") is not None and c.get("tolerance") is not None \
                and not c.get("uncertainty_stated", True) and c["relative"] > c["tolerance"]:
            out.append(c)
    return out


def _render_card(name, repo, rec, gate, previews, snippet_shown, snippet_ran, hub_path, hold=None,
                 published_commit=None):
    """The card: the manifest table, what is inside, the reproduction with its grade, the gate's verdict, and
    the snippet with the output it produced when this card was built.

    ``published_commit`` is the commit of the code rendering this card (dmipy-sim#499); the packs table's
    ``commit`` column names it beside each pack's own ``built_commit`` -- the commit that BUILT that pack's
    bytes, sealed in its header -- and labels each when the two differ, rather than printing one number under a
    heading that reads like the other."""
    from .publish import _fmt, _size, _commit_cell          # the card's number and byte formats live once
    ref, des, spc, wlk, pk = (rec[s] for s in ("reference", "design", "spec", "walk", "pack"))
    grade = gate["grade"]
    licences = sorted({v["pack"]["license"].lower() for v in pk["substrates"].values() if v["pack"].get("license")})
    L = ["---", f"license: {licences[0] if len(licences) == 1 else 'other'}",
         f"pretty_name: {repo.split('/')[-1]}", "---", "", f"# {repo}", ""]

    bad = _disagreements(gate)
    opening = [ref.get("description") or name]
    if ref.get("absent"):
        opening.append("**This family reproduces no published quantity** (" + str(ref["absent"]) + "), so its "
                       "reference grade is `none` and its card states no comparison.")
    else:
        opening.append(f"The reference it reproduces is [{ref['title']}](https://doi.org/{ref['doi']}) "
                       f"({ref['crossref'].get('container') or ref['crossref'].get('type')}), on "
                       f"{ref['sample_relation']}: **grade {grade}** by the rule of "
                       f"[dmipy-sim#459](https://github.com/dmrai-lab/dmipy-sim/issues/459).")
    if bad:
        failed = [c for c in bad if not c["passed"]]
        stated = [c for c in bad if c["passed"]]
        opening.append(("**The comparison does not hold.** " if failed else
                        "**These packs disagree with the published number.** ")
                       + " ".join(f"`{c['check'].split('/')[0]}`: {c['detail']}." for c in bad)
                       + (" The packs below are certified for what their certificate states; the gate says so."
                          if failed else "")
                       + (" No gate check fails on this: the publication states no uncertainty to fail it "
                          "against, so the disagreement is a recorded number rather than a verdict."
                          if stated else ""))
    if gate["passed"]:
        opening.append(f"Every one of the gate's {gate['n_checks']} checks passed.")
    else:
        opening.append(f"{len(gate['failures'])} of the gate's {gate['n_checks']} checks failed; "
                       f"**§ The gate** below names each one.")
    L += ["\n".join(opening), "",
          "Every number on this card is read from a record in `records/` of this dataset, written by the stage "
          "that measured it (the reference-pack protocol, "
          "[dmipy-sim#482](https://github.com/dmrai-lab/dmipy-sim/issues/482)); none of it is transcribed, so "
          "this card, the gate and any paper read the same files.", ""]

    head = (["pack", "substrate", "channels", "K", "band (Hz)", "T", "walkers"]
            + [f"{t['name']}: floor / err / target" for t in des["tiers"]] + ["size", "licence", "commit"])
    L += ["## The packs", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for n, v in sorted(pk["substrates"].items()):
        row, tiers = v["pack"], v["tiers"]
        cells = [f"`{row['id']}`", f"`{row['substrate_id']}`", ", ".join(row["channels"]), str(row["K"]),
                 f"{row['band_hz']:.3g}", f"{row['T_s']:g} s", f"{row['n_walkers']:,}"]
        for t in des["tiers"]:
            c = tiers[t["name"]]
            cells.append(f"{c['floor']:.3g} / {c['err']:.2g} / {c['target_floor']:.3g} "
                         f"{'**meets**' if c['meets_target'] else '_below target_'}")
        cells += [_size(row["bytes"]), row["license"], _commit_cell(row.get("built_commit"), published_commit)]
        L.append("| " + " | ".join(cells) + " |")
    hold = dict(hold or {})
    if hold:
        L += ["", "**Published with a hold.** "
              + " ".join(f"`{k}` ships and its manifest row says so, with a reservation recorded beside the "
                         f"verdict: {v}" for k, v in sorted(hold.items()))]
    per_pack_caveats = sorted(k for k in (ref.get("caveats") or {}) if k in pk["substrates"])
    if per_pack_caveats:
        L += ["", "**Read with a caveat of its own:** "
              + ", ".join(f"`{k}` (below, under § What is inside)" for k in per_pack_caveats)
              + ". A caveat keyed on a pack's own name is about THAT pack, and the card puts it beside it."]
    if des.get("trade"):
        L += ["", f"**The trade the design records:** {des['trade']}"]
    L += ["", f"The save grid is {des['dt_save_s'] * 1e6:.0f} us over {des['window_s']:g} s "
              f"({des['n_t']:,} saves): {des['save_grid_why']}", "",
          "| pack | walk | sub-steps | refused steps | wall time | peak resident | budget |",
          "|---|---|---|---|---|---|---|"]
    def counter(w, key):
        return "_not recorded_" if w.get(key) is None else f"{w[key]}"
    for n, w in sorted(wlk["substrates"].items()):
        L.append(f"| `{n}` | {w['n_walkers']:,} walkers"
                 + (" (recorded, not re-walked)" if w["from_pack"] else "")
                 + f" | {counter(w, 'sub_steps')} | {counter(w, 'illegal_crossings')} | {w['seconds']:.0f} s | "
                 + ("_not recorded_" if w.get("peak_rss_bytes") is None else _size(w["peak_rss_bytes"]))
                 + f" | {_size(w['budget_bytes'])} |")
    notes = [w.get("design_note") for w in wlk["substrates"].values() if w.get("design_note")]
    notes += [w["counters_not_recorded"] for w in wlk["substrates"].values() if w.get("counters_not_recorded")]
    if notes:
        L += ["", *[f"* {t}" for t in sorted(set(notes))]]
    L += [""]

    L += ["## What is inside", "",
          "Three orthogonal cross-sections through the centre of each substrate, rendered from the spec the pack "
          "embeds by `dmipy_sim.spec.preview` -- the same membership test the walk uses, so the picture cannot "
          "show a substrate the walk does not have.", ""]
    for n, p in sorted(previews.items()):
        frac = p["planes"][0]["pool_area_fraction"]
        real = spc["substrates"][n].get("realisation") or {}
        L += [f"### {n}", "", f"![{n}](previews/{p['path']})", "",
              f"* surface kinds: {', '.join(p['kinds'])}; pools {', '.join(p['pools'])}",
              f"* pixel {p['pixel_m'] * 1e6:.4g} um, scale bar {p['scale_bar_m'] * 1e6:g} um",
              "* area fraction in the x-centre section: "
              + ", ".join(f"{k} {v:.4f}" for k, v in sorted(frac.items()))
              + (f" (the spec's realisation: " + ", ".join(f"{k} {v:.4g}" for k, v in sorted(real.items())
                                                           if isinstance(v, float)) + ")" if real else ""), ""]
        mine = (ref.get("caveats") or {}).get(n)
        if mine:                        # a caveat keyed on this pack's name belongs BESIDE this pack
            L += [f"> **Caveat — `{n}`.** {mine}", ""]

    L += ["## The reproduction", ""]
    if ref.get("absent"):
        L += [f"There is none: {ref['absent']}. The grade is `none`.", ""]
    else:
        L += [f"**Grade {grade}.** " + grade_reason(ref),
              f"Their sample: {ref['sample']} ({ref['sample_relation']}).", "",
              "| substrate | quantity | replayed (this pack) | our direct walk | theirs | from | "
              "vs direct | band | holds | vs theirs |", "|" + "---|" * 10]
        for q in sorted(ref["quantities"], key=lambda x: (x["substrate"], x["name"])):
            got = pk["substrates"][q["substrate"]]["reproduced"].get(q["name"], {})
            c = next((c for c in gate["checks"]
                      if c["check"] == f"{q['substrate']}/reproduces-{q['name']}"), None)
            pubc = next((x for x in gate["checks"]
                         if x["check"] == f"{q['substrate']}/published-{q['name']}"), None)
            rel = "—" if not c else f"{c['relative']:.3%} ({c['sigma']:.2g} σ)"
            tol = "—" if not c else f"{c['k']:.3g} σ = {c['tolerance']:.3%}"
            vs_them = "—"
            if pubc and pubc.get("relative") is not None:
                vs_them = (f"{pubc['relative']:.3%} ({pubc['sigma']:.2g} σ)"
                           + ("" if pubc["passed"] else " **fails**")
                           + ("" if pubc.get("uncertainty_stated") else ", no uncertainty stated"))
            L.append(f"| {q['substrate']} | {q['name']} | **{_fmt(got.get('value'), '.6g')} "
                     f"{q['direct']['unit']}** | {q['direct']['value']:.6g} (SE {q['direct']['se']:.2%}, "
                     f"{q['direct']['se_kind']}, {q['direct']['n_walkers']:,} walkers) | "
                     f"{_fmt(q['published']['value'], '.6g')} {q['published']['unit']} | "
                     f"{q['published']['locator']} | {rel} | {tol} | "
                     f"{'yes' if (c and c['passed']) else 'NO'} | {vs_them} |")
        L += ["", "The quantity is compared on the grid the reference record states, with the solver it states; "
                  "the gate refuses a reproduction on any other grid, because the grid is part of the "
                  "measurement.", "",
              "**The free parameters, and whose they are:**", ""]
        for p in ref["parameters"]:
            L.append(f"* `{p['name']}` = {p['value']} {p['unit']} -- **{p['whose']}** ({p['where']}): {p['how']}")
        L += ["", "**Where each number comes from:**", ""]
        for q in sorted(ref["quantities"], key=lambda x: x["substrate"]):
            pub = q["published"]
            head = f"> {q['substrate']} -- {pub['printed_in']}, {pub['locator']}"
            for ident, doc in sorted((q.get("documents") or {}).items()):
                head += (f" ([{ident}](https://doi.org/{ident}), resolved via {doc.get('agency')} at "
                         f"{doc.get('resolved')} as {doc.get('title')!r})")
            L += [head + (f": `{pub['verbatim'].strip()}`" if pub.get("verbatim", "").strip() else
                          f": read from [{pub.get('data_url')}]({pub.get('data_url')}), sha256 "
                          f"`{str(pub.get('data_sha256'))[:12]}`"), ""]
        if ref.get("caveats"):
            L += ["**Caveats:**", ""] + [f"* **{k}** -- {v}" for k, v in sorted(ref["caveats"].items())] + [""]

    L += ["## The gate", "",
          f"Deterministic, reading only the records: {gate['n_checks']} checks, "
          f"{'all passed' if gate['passed'] else str(len(gate['failures'])) + ' failed'}. "
          "The tolerance is the design record's terms in quadrature -- an analytic standard error, the "
          "reference's own stated uncertainty, and any measured systematic -- with no coverage factor and no "
          "resampled error bar.", "", "| pack | verdict | checks | failures |", "|---|---|---|---|"]
    for n, v in sorted(gate["per_substrate"].items()):
        L.append(f"| `{n}` | {'**pass**' if v['passed'] else '**FAIL**'} | {v['n_checks']} | "
                 f"{', '.join('`' + f + '`' for f in v['failures']) or '—'} |")
    if gate["failures"]:
        L += ["", "The failures, as the gate states them:", ""] + [f"* {f}" for f in gate["failures"]]
    L += [""]

    L += ["## Use me", "",
          f"One call that reproduces one number of the table above. It was EXECUTED when this card was built "
          f"({snippet_ran['seconds']:.1f} s, ceiling {SNIPPET_CEILING_S:g} s), against the local file of the "
          f"same sha256 as `{hub_path}`, and printed:", "", "```", snippet_ran["stdout"], "```", "",
          "```python", snippet_shown.strip(), "```", ""]

    L += ["## Reference and licences", ""]
    if not ref.get("absent"):
        L += [f"* the reference: {ref['title']}, <https://doi.org/{ref['doi']}> -- **cited, never "
              f"redistributed**. {ref['source_note']}"]
    src = rec["source"]
    for k, s in sorted(src["sources"].items()):
        L.append(f"* `{k}`: <{s['url']}> at {s['host_record']} -- **{s['licence_id']}** "
                 f"(<{s['licence_url']}>), the host's text copied verbatim to "
                 f"`records/licences/{s['licence_copy']['path']}` ({s['licence_copy']['chars']} characters, "
                 f"sha256 `{s['licence_copy']['sha256'][:12]}`); "
                 + ", ".join(f"`{f['cite_as']}` (`{f['sha256'][:12]}`, {_size(f['bytes'])})" for f in s["files"]))
    if not ref.get("absent"):
        L += [f"* the packs: {ref['licence_note']}"]
    L += ["", "*Rendered by `dmipy_sim.replay.reference` from this dataset's `records/`. A bare republish of a "
              "pack regenerates the manifest-only card of `dmipy_sim.replay.publish` and drops this one; the "
              "protocol's publish stage writes this file last.*", ""]
    return "\n".join(L)


# --------------------------------------------------------------- the organisation page
#: The SubstrateCommons organisation and the Space its card lives in.
ORGANISATION = "SubstrateCommons"
ORGANISATION_CARD = f"{ORGANISATION}/README"


def organisation_index(org=ORGANISATION, *, api=None, get=None):
    """Every dataset of ``org`` with its pack count, total bytes and its own ``records/reference.json``.

    The one-line description, the source and the licence of a family come from THAT record -- the descriptions
    live with the family they describe, never in this renderer, which is why a family is on the page only once
    it has run the protocol.
    """
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    if get is None:
        from huggingface_hub import hf_hub_download
        def get(repo, path):
            return hf_hub_download(repo, path, repo_type="dataset")
    rows = []
    for d in api.list_datasets(author=org):
        name = d.id.split("/", 1)[1]
        n = total = None
        reference, packs = None, []
        try:
            m = json.load(open(get(d.id, "manifest.json")))
            packs = m.get("packs")
            n = len(packs) if isinstance(packs, (list, dict)) else (len(m.get("variants", {})) or None)
            total = m.get("total_bytes") or sum(int(r.get("bytes") or 0) for r in (packs or []) if isinstance(r, dict))
        except Exception:
            pass
        try:
            reference = json.load(open(get(d.id, "records/reference.json")))
        except Exception:
            pass
        rows.append(dict(name=name, repo=d.id, packs=n, bytes=total, reference=reference,
                         packs_rows=(packs if isinstance(packs, list) else [])))
    return sorted(rows, key=lambda r: r["name"])


def organisation_card(rows, *, example=None):
    """The organisation page, rendered from :func:`organisation_index`.

    ``example`` is the pack the "Use a pack" snippet reads -- ``owner/name/path`` of a pack a family's manifest
    holds and its gate PASSED. It hardcoded a path that did not exist and named the one pack the gate withheld;
    :func:`organisation_example` picks one from the index instead.

    Refused when any listed dataset has no ``records/reference.json``: the page states each family's grade and
    what it reproduces, and those are the reference record's words. A family that has not run the protocol has
    no such words, and inventing them in this renderer is how a page came to grade a figure as `A`.
    """
    for r in rows:
        if r["reference"] and not r["reference"].get("absent"):
            _protocol_reference(r["reference"], where=f"organisation: {r['name']}")
    missing = sorted(r["name"] for r in rows if not r["reference"])
    if missing:
        raise ReferenceRefusal(
            f"organisation: {', '.join(missing)} publish no records/reference.json, so this page has nothing to say "
            f"about what they reproduce. Run the reference-pack protocol (dmipy-sim#482) for them before "
            f"regenerating the page -- the descriptions belong to the families, not to this renderer")
    out = [f"# {ORGANISATION}\n",
           "**A library of replayable Monte-Carlo substrates for MRI microstructure.** Each dataset here is a "
           "*family* of replay packs: one substrate, walked once at a certified precision, stored so that any "
           "acquisition can be replayed on it later, on your machine, in seconds. A pack carries the walkers' "
           "paths in a band-limited form, their compartment occupancy, their wall contact, and the substrate's "
           "internal field, plus the specification of the substrate it came from and a certificate stating the "
           "Monte-Carlo floor and the codec error per tier. The tissue (T2, T1, surface relaxivity, "
           "susceptibility), the scanner's field and the pose are knobs applied at replay, never baked into the "
           "walk.\n",
           "Every family credits the people whose substrate it is, and every row below is read from that "
           "family's own records.\n",
           "## Use a pack\n",
           "```python\n"
           "# pip install dmipy-sim\n"
           "from dmipy_sim.replay import ReplayPack\n"
           "from dmipy_sim import sequences\n"
           f"pack = ReplayPack.load('hf://{example or ORGANISATION + '/<family>/packs/<pack>.rpk'}')\n"
           "seq = sequences.pgse([[1, 0, 0]], 0.04, 0.08, bvalues=[1e9], TE=0.15)\n"
           "S = pack.replay(seq)                                              # the bare diffusion signal\n"
           "S = pack.replay(seq, tissue=pack.nominal)                         # the paper's replay\n"
           "```\n",
           "A replay refuses what the pack cannot serve (a waveform beyond its temporal band, a tier it does not "
           "carry) rather than returning a plausible number. The format is specified in "
           "[replay-pack-spec](https://github.com/dmrai-lab/replay-pack-spec); the engine and the producers are "
           "[dmipy-sim](https://github.com/dmrai-lab/dmipy-sim).\n",
           "## Families\n",
           "| family | what it is | source | reference it reproduces | grade | packs | licence |",
           "|---|---|---|---|---|---|---|"]
    from .publish import _size
    for r in rows:
        ref = r["reference"]
        grade = grade_of(ref)
        what = "; ".join(filter(None, [ref.get("sample"), ref.get("sample_relation")])) if not ref.get("absent") \
            else str(ref.get("absent"))
        size = f" ({_size(r['bytes'])})" if r["bytes"] else ""
        out.append(f"| [{r['name']}](https://huggingface.co/datasets/{r['repo']}) | {ref.get('description', '')} | "
                   f"{ref.get('source_note', '')} | {what} | **{grade}** | "
                   f"{r['packs'] if r['packs'] is not None else '?'}{size} | {ref.get('licence_note', '')} |")
    out += ["",
            "Grades follow [dmipy-sim#459](https://github.com/dmrai-lab/dmipy-sim/issues/459) and are assigned BY "
            "RULE from the family's reference record: **A** a published number on the same released geometry, "
            "**B** a published figure or a number on something else, **analytic** a closed form, **none** no "
            "published reference. No family states its own grade.\n",
            "## How a family is made\n",
            "One producer reads the released substrate into a substrate specification; one walk at a floor set by "
            "a measured pilot; one pack per substrate with its certificate; a gate that reads the recorded "
            "numbers and refuses to publish a pack that does not reproduce them; a card generated from the "
            "records. The protocol is "
            "[dmipy-sim#482](https://github.com/dmrai-lab/dmipy-sim/issues/482), the card "
            "[dmipy-sim#480](https://github.com/dmrai-lab/dmipy-sim/issues/480), and every family's `records/` "
            "is published beside its packs.\n",
            "## Contribute, request, or object\n",
            "Open an issue on [dmipy-sim](https://github.com/dmrai-lab/dmipy-sim/issues): to propose a released "
            "substrate for a family (say where it is, its licence, and what was measured on it), to request an "
            "acquisition or a tier on an existing family, or, if you hold rights in a source and want a family "
            "taken down, to say so; the card is deleted on request.\n",
            "## Cite\n",
            "The Commons and the replay method are described in the replayable Monte-Carlo paper (Fick, 2026, in "
            "preparation); until it has a DOI, cite the family's own source paper and the dmipy-sim commit "
            "recorded in the family's manifest.\n"]
    return "\n".join(out)


def _is_the_real_hub(hub):
    """Whether ``hub`` is a dataset repository on the hub and not a stand-in (``FakeHub`` in the tests)."""
    from ..fill.hub import Hub
    return isinstance(hub, Hub)


def organisation_example(rows):
    """``owner/name/path`` of a pack the snippet on the organisation page may read: the first row of the first
    family whose manifest marks it gate-passed, so the page cannot teach a withheld pack."""
    for r in sorted(rows, key=lambda x: x["name"]):
        for row in r.get("packs_rows") or ():
            if row.get("gate") in (None, "pass") and not row.get("withheld") and row.get("path"):
                return f"{r['repo']}/{row['path']}"
    return None


def _organisation_after_publish(repo, *, skip):
    """Regenerate the organisation page from the hub after a publish, and record what happened.

    The page is rendered from every family's own ``records/reference.json``; a family that has not run the
    protocol has none, and the renderer refuses rather than invent a row. That refusal is recorded, not raised:
    this family's publish has already succeeded.
    """
    out = dict(page=ORGANISATION_CARD, after=repo, rendered=False, pushed=False, refused=None,
               families=None)
    if skip:
        out["refused"] = "not attempted: a dry run, or a hub that is not the organisation's"
        return out
    try:
        rows = organisation_index()
        out["families"] = [r["name"] for r in rows]
        text = organisation_card(rows, example=organisation_example(rows))
        out["rendered"] = True
    except ReferenceRefusal as e:
        out["refused"] = str(e)
        return out
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(ORGANISATION_CARD, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_file(path_or_fileobj=text.encode(), path_in_repo="README.md", repo_id=ORGANISATION_CARD,
                    repo_type="space",
                    commit_message="SubstrateCommons: the organisation page, rendered from the families' records")
    out["pushed"] = True
    out["sha256"] = _sha256_bytes(text.encode())
    return out
