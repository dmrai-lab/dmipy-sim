"""The **time-segment layout**: a walk contracted once against the time segments of its timing classes, so that
every acquisition piecewise constant on a stored partition -- at any amplitude, in any direction on each segment -- is
a few products per walker on the device (dmipy-sim#505, #770: a whole grid replayed while a user waits).

**Terms, not a shape.** A layout stores PARTITIONS (:class:`Partition`): TERMS, each an interval ``[a_s, b_s]`` of the
walk's clock and a temporal shape ``psi_s`` on it (:class:`Shape`, a stated record normalised to ``max |psi| = 1``), with
the sign of its class's effective gate on the interval (the coherence sign times the transverse gate of the RF schedule:
+1, -1, or 0 while the magnetisation is stored). A constant segment is the term ``psi = 1`` -- a PGSE timing class's two
lobes, a stimulated echo's lobe, mixing time and lobe, a double diffusion encoding's four lobes; a trapezoid with its
ramps, a cosine OGSE block or a designed waveform is a sampled shape; a tensor encoding is several shapes on ONE
interval. Per walker and term it holds the **term vector** ``m_s = gamma int psi_s(t) 1_[a_s, b_s](t) (r(t) - r(0))
dt`` (rad per T/m): the bands contracted against ``psi_s 1_[a_s, b_s]``
(:func:`~dmipy_sim.replay.compression.segment_projection`, exact for a shape held on equal cells: the closed form
:func:`~dmipy_sim.replay.compression.band_projection` sums piece by piece, so no quadrature stands between the stored
shape and the stored vector). An acquisition whose effective gradient is ``sum_s psi_s(t) a_s`` with the vector ``a_s``
(T/m, the term's **level**) has the phase ``sum_s a_s . m_s`` per walker: the band contraction of a replay, the start
dropping out because a refocused encoding has no net area ``sum_s a_s int psi_s``. PGSE, PGSTE, DDE, an unequal pair of
one timing class, a trapezoid at any amplitude and direction, a cosine OGSE, a tensor encoding at any rotation and
weighting (three vectors, nine numbers, per walker) are each a linear combination of the stored vectors formed in the
image kernel; no vector per acquisition is stored. The gate (the refocusing schedule and the coherence sign), not the
shape, fixes the relaxation, contact and field tiers, so partitions of one gate share their tier columns.

**The fit.** :meth:`Partition.levels` finds the levels of a sequence, and a sequence fits only when its effective
gradient IS ``sum_s psi_s(t) a_s`` on its grid to the float32 rounding of its samples: a shape's vector serves that shape
alone, so a different waveform is refused, never served with an error no certificate states. Disjoint constant segments
take the gradient's value on each; shaped or overlapping terms take the least squares fit of the gradient on the terms'
shapes, and its residual is held to the same tolerance, :data:`_RTOL` of the peak (the one statement of what "constant
on a segment" and "equal to its shapes" mean). A sequence that plays a gradient outside the terms or on a term its gate
stores, is off the shapes (the partition, the row, the term the residual lies in and the residual's relative norm are
named), or leaves a net moment, is refused by name (:meth:`Partition.levels`).

**Two sources, two certificates.** :func:`segment_rows` takes the walk's FLOAT coefficients (a
:class:`~dmipy_sim.coefficient_walk.CoefficientWalk`, read through the float containers, no quantiser) beside the pack
built from the same walk (``pack=``, its sibling: the floor target and the envelope are the pack's, and the rows'
provenance names the walk and the pack): two artefacts of one walk, each with its own certificate, the segments'
carrying the walk's terms (truncation, quadrature, field read, the accumulator's rounding) and its own rounding, and no
band quantisation. Given a quantised :class:`~dmipy_sim.replay.replay.ReplayPack` instead (``source: "pack"``) it
contracts the pack's bands: the certificate then carries the pack's band quantisation contracted through the partition
(:func:`quantisation_term`), and the vectors are held to what the floor's budget leaves of it, so their words are wider.

**Widths derived per term.** A rounding ``delta_s`` per component of ``m_s`` is a phase error of variance at most
``|a_s|^2 delta_s^2 / 12``. The envelope bounds the levels: ``|a_s| <= G_max`` (a shape has ``max |psi| = 1``) and the
encoding's b-value ``l^T B l <= b_max`` with ``B_st = gamma^2 int Q_s Q_t dt`` (``Q_s`` the running integral of
the term's shape) on the refocused levels ``sum_s l_s A_s = 0`` (``A_s = int psi_s dt``, the term's duration for a
segment; terms of zero area, a whole number of periods, leave no moment and constrain nothing; ``int |psi_s|`` bounds a
vector's magnitude and scales the net moment tolerance), so the largest level segment ``s`` carries is ``reach_s = min(G_max, sqrt(b_max
[N (N^T B N)^-1 N^T]_ss))``, ``N`` a basis of the refocused levels (:meth:`Partition.reach`; a PGSE pair's is
``sqrt(b_max / b_unit)``; a scanner class's levels are every axis within ``G_max`` on the refocused levels, the reach
``sqrt(3)`` times the largest per-axis level, :meth:`Partition.class_vertices`). The worst rounding over the envelope is then at most ``sum_s reach_s^2 delta_s^2 / 12``, held
at the floor's budget ``eps^2`` (:func:`~dmipy_sim.replay.compression.container_eps` at the floor target; from a pack,
``eps^2`` less the pack's quantisation term on the partition) with equal contributions, ``delta_s = sqrt(12 eps^2 / n)
/ reach_s`` over the ``n`` segments that carry a phase: the split that minimises the bits (stationary ``sum_s -3 log
delta_s + lambda sum_s reach_s^2 delta_s^2`` gives ``reach_s delta_s`` equal), where weighting the budget by ``(reach_s
d_s)^2`` would equalise the segments' relative precision instead. A segment the gate stores carries no phase and no
bits. Each component takes ``ceil(log2(a / delta_s + 1)) + 1`` bits against its largest ``|m|`` over the walkers
(:func:`field_code`), a two's-complement bit field; a walker's fields of one partition are placed first-fit into as few
32-bit words as hold them, no field across a word (:func:`word_layout`): a short partition shares one word, a long
one takes two. The manifest records per term ``{"t", "gate", "shape", "q_max", "floor_target", "step", "bits", "offsets",
"word", "scale"}`` (``shape`` the term's :meth:`Shape.to_meta` record, required: a term with none is refused by name;
``q_max`` the reach, ``word`` each component's word).

**The tiers by the same rule.** Per sequence group (the partitions that share a grid, an RF schedule and a readout: the
gate the tiers are read under) the layout holds each walker's pool (``uint8``), its gated contact and its gated field
channels (``t_<g>.npy``, one word array), and with ``background`` its background vector ``gamma int e(t) r(t) dt`` over
the acquisition (``bg_<g>.npy``; ``e`` the effective gate, what a magnet's constant gradient ``g0`` encodes as ``g0 .
n``). The contact enters as the weight ``exp((rho2 / D) c)``, a rounding ``delta`` a relative weight error ``(rho2 / D)
delta``, held at the envelope's largest ``rho2 / D``; the field as ``B0 (chi_iso A + chi_aniso B)`` with ``A, B`` the
channels contracted with coefficients of modulus at most 1, held at the envelope's largest ``B0 chi`` with the budget
split equally over the basis's twelve components (RPK.md 5.3: ``iso_local``, ``iso_P`` xx, yy, xy, xz, yz, ``aniso_G``
(6); ``iso_P_zz`` is never stored, the reader forms ``3 iso_local - iso_P_xx - iso_P_yy`` before the contraction, so
``iso_local``'s coefficient is bounded by 2 and its step halved); the background
at ``background_amplitude``. A field within half its step of zero on every row (a channel the gate refocuses, a
component the substrate's symmetry zeroes) rounds to zero on every row and is not stored. Each tier column is its own
term at the floor's budget. From a pack the layout's certificate also carries the pack's own quantisation of the contact
and the field through the gate (:func:`tier_quantisation_terms`) as terms of their own, beside its rounding (the widths
are the same as from the walk), and ``err_max`` covers both: the same bits, more error; a floor target below the pack's
own term, stated by the caller, is refused by name, since no width meets it. A rounding of rms ``sigma`` rad per walker moves a voxel's signal by about ``sigma`` times
its Monte-Carlo floor and biases it by ``sigma^2 / 2``; the manifest's ``certificate.rounding`` states each partition's
realised ``sigma^2`` beside its budget.

**Rows and consolidation.** :func:`segment_rows` writes one block's rows: per partition and tier column one float
file, the rows in (voxel, pool) order, the declarations in ``rows.json``. :func:`write_time_segment_layout` takes the
blocks' rows, derives every width from their largest values and TILES them: rows in tiles of ``TILE`` rows, every
tile the rows of one voxel half (a voxel's rows alternate between its two split halves in order), the last tile of a
half padded with zero weight, so a voxel's sum is a fused reduction over whole tiles and the sum over tiles a scatter
with as many collisions as a half has tiles (a scatter over rows serialises on a voxel's thousands of contiguous rows:
6.4 s per 2^19 rows against 1 ms tiled on an L40S). Files: ``w.npy`` ``(n_tiles, TILE)`` float32, ``tiles.npy``
``(n_tiles,)`` (``2 voxel + half``), ``p_<partition>.npy`` ``(n_tiles, TILE, n_words)`` uint32, with tiers ``pool.npy``
and ``t_<g>.npy``, with the background ``bg_<g>.npy``, and ``manifest.json`` (the partitions, the groups, the widths,
the floor, the source, the walk's identity, the certificate).

**The image.** :meth:`TimeSegmentLayout.image` takes a sequence (any rows, each piecewise constant on a stored
partition) and forms ``|sum_w w exp(i sum_s G_s . m_s)| / sum_w w`` per voxel with its split-half floor, on the device
from device-resident words: each word decoded in the kernel (a shift up and an arithmetic shift down per field, the
shifts constant per partition, a scale), float32 phases written out as products (a float32 matmul on CUDA is TF32),
complex64 exponentials, complex128 tile sums and accumulator. ``tissue=`` / ``scanner=`` / ``b0_direction=`` turn the
tiers into exactly :meth:`~dmipy_sim.replay.study.Primitives.reduction_terms`: the row weight ``amp exp(-tau_2 / T2_pool
- tau_1 / T1_pool + (rho2 / D) c)`` and the phase ``B0 (chi_iso A + chi_aniso B)``, ``A, B`` contracted from the decoded
channels for that direction. Two backends run that arithmetic, ``"jax"`` (the reference) and ``"torch"`` (Hugging
Face's shared GPU pool runs PyTorch only, and drops the device between calls).
"""
from __future__ import annotations

import collections
import functools
import hashlib
import itertools
import json
import os
import time
from dataclasses import dataclass, field, replace

import numpy as np

MANIFEST = "manifest.json"
FORMAT = "time_segments/2"
ROWS = "rows.json"
ROWS_FORMAT = "segment_rows/1"
TILE = 128                                                            # rows per tile
WORD_BITS = 32                                                        # the word a walker's fields are packed into
FIELD_BITS_MIN = 3
FIELD_BITS_MAX = 24                                                   # a field's integer is exact in a float32
#: T/m: the largest background gradient a layout's background vectors are certified for by default -- several times
#: the strongest catalogued magnet's own gradient over its anchored volume (the Swoop's 1.4 mT/m at 8 cm)
BACKGROUND_AMPLITUDE = 5e-3
_RTOL = 1e-5                                                          # a sequence's gradient is float32: the tolerance of every fit, of the peak
_AXES = np.eye(3)


def read_manifest(path):
    """The manifest of the layout in the directory ``path``; a manifest of another format is refused by name."""
    with open(os.path.join(path, MANIFEST)) as f:
        manifest = json.load(f)
    _check_format(manifest, path)
    return manifest


def _check_format(manifest, where):
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{where} is not a {FORMAT} layout ({manifest.get('format')!r}); a layout of another format is "
                         "written again from its walk's segment rows (segment_rows, write_time_segment_layout)")


# ------------------------------------------------------------------ the width rule and the words
def rounding_eps(floor_target, *, budget=None, kappa=None):
    """The floor's budget, rad of rms rounding phase: :func:`~dmipy_sim.replay.compression.container_eps` at
    ``floor_target``, the criterion the pack's containers are derived at."""
    from .compression import BUDGET, KAPPA, container_eps
    return float(container_eps(float(floor_target), budget=BUDGET if budget is None else budget,
                               kappa=KAPPA if kappa is None else kappa))


def field_code(a_max, step):
    """``(bits, scale)`` of fields whose values reach ``a_max`` at the quantisation ``step`` (arrays of one shape): the
    fewest bits whose largest code reaches ``a`` in steps of at most ``step``, ``ceil(log2(a / step + 1)) + 1``, at least
    :data:`FIELD_BITS_MIN`, and the scale ``a / (2^(bits-1) - 1) <= step``. A field
    within half a step of zero on every row rounds to zero on every row and holds 0 bits (it is not stored), as does a
    field whose step is infinite (it carries nothing). More than :data:`FIELD_BITS_MAX` bits is refused by name."""
    a = np.asarray(a_max, np.float64)
    st = np.broadcast_to(np.asarray(step, np.float64), a.shape)
    live = np.isfinite(st) & (st > 0) & (a > 0.5 * np.where(np.isfinite(st), st, 0.0))
    bits = np.zeros(a.shape, int)
    with np.errstate(divide="ignore", invalid="ignore"):
        need = np.ceil(np.log2(np.where(live, a / np.where(live, st, 1.0), 1.0) + 1.0)) + 1
    bits[live] = np.maximum(need[live], FIELD_BITS_MIN).astype(int)
    if np.any(bits > FIELD_BITS_MAX):
        raise ValueError(f"a field needs {int(bits.max())} bits at its derived step and a field holds at most "
                         f"{FIELD_BITS_MAX}: the floor target is below what the layout's words carry")
    scale = np.ones(a.shape)
    scale[live] = a[live] / (2.0 ** (bits[live] - 1) - 1)
    return bits, scale


def word_layout(bits):
    """``(((word, offset, bits), ...), n_words)``: each field's place, first fit in field order into 32-bit words, no
    field across a word; a 0-bit field holds no place (``(0, 0, 0)``)."""
    used, out = [], []
    for b in (int(x) for x in bits):
        if b == 0:
            out.append((0, 0, 0)); continue
        for i, u in enumerate(used):
            if u + b <= WORD_BITS:
                out.append((i, u, b)); used[i] += b
                break
        else:
            out.append((len(used), 0, b)); used.append(b)
    return tuple(out), len(used)


def pack_words(values, scale, layout, n_words):
    """``values`` ``(..., n_fields)`` as ``(..., n_words)`` uint32: field ``f`` rounded to ``scale[f]`` steps, two's
    complement in its place of :func:`word_layout`."""
    v = np.asarray(values, np.float64)
    out = np.zeros(v.shape[:-1] + (int(n_words),), np.uint64)
    for f, (w, off, b) in enumerate(layout):
        if b == 0:
            continue
        lim = 2 ** (b - 1) - 1
        k = np.clip(np.rint(v[..., f] / float(scale[f])), -lim, lim).astype(np.int64) & ((1 << b) - 1)
        out[..., w] |= k.astype(np.uint64) << np.uint64(off)
    return out.astype(np.uint32)


def _decode(words, scale, layout):
    """The fields of int32 ``words`` ``(..., n_words)`` on any array library (numpy, JAX, torch): each shifted to the top
    of its word and arithmetically back down, times its scale; ``None`` for a 0-bit field (it is zero)."""
    out = []
    for f, (w, off, b) in enumerate(layout):
        if b == 0:
            out.append(None); continue
        x = words[..., w]
        out.append(((x << (WORD_BITS - off - b)) >> (WORD_BITS - b)) * scale[f])
    return out


def unpack_words(words, scale, layout):
    """``(..., n_fields)`` float32: the values :func:`pack_words` stored (0 for a 0-bit field)."""
    w = np.asarray(words, np.uint32).view(np.int32)
    sc = np.asarray(scale, np.float32)
    vals = _decode(w, sc, layout)
    zero = np.zeros(w.shape[:-1], np.float32)
    return np.stack([zero if v is None else v.astype(np.float32) for v in vals], axis=-1)


# ------------------------------------------------------------------ partitions
def _sequence(seq):
    seq = seq.waveform if hasattr(seq, "waveform") else seq
    from ..acquisition.scanner_sequence import ScannerSequence
    if not isinstance(seq, ScannerSequence):
        raise TypeError(f"a time-segment layout reads a ScannerSequence; got {type(seq).__name__}")
    return seq


def _lobes(G, dt):
    """The maximal runs of samples on which every row's effective gradient is one non-zero constant, as ``[(a, b),
    ...]`` seconds: a sample-and-hold waveform's lobes."""
    n_t = G.shape[1]
    tol = _RTOL * (float(np.abs(G).max()) or 1.0)
    on = np.abs(G).max(axis=(0, 2)) > tol
    out, i = [], 0
    while i < n_t:
        if not on[i]:
            i += 1; continue
        j = i + 1
        while j < n_t and on[j] and np.abs(G[:, j] - G[:, i]).max() <= tol:
            j += 1
        out.append((i * float(dt), j * float(dt))); i = j
    return out


def _group_key(seq):
    """What the tiers of a sequence are read under: its grid, RF schedule and readout."""
    return json.dumps(dict(dt=float(seq.dt), n_t=int(seq.n_t), rf=seq.rf.to_dicts(), readout=[int(i) for i in seq.readout]),
                      sort_keys=True, default=float)


def _gate_sequence(seq):
    """``seq``'s gate alone: one row with no gradient on its grid, RF schedule and readout (what the tiers are read
    under)."""
    return replace(seq, G=np.zeros((1, int(seq.n_t), 3), np.float32), encoding=None)


@dataclass(frozen=True)
class Shape:
    """The temporal shape ``psi(t)`` of a term, a stated record and nothing else. ``kind`` is ``"constant"`` (``psi`` is
    1 on the term: a segment) or ``"sampled"`` (``psi`` is held on ``len(values)`` equal cells of the term, as a
    :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence` holds its samples: a trapezoid with its ramps, a
    cosine, a designed waveform). A shape is normalised, ``max |psi| = 1``, so that the amplitude (the term's level,
    T/m) bounds the gradient and the stored vector's scale means what a segment's does; ``values`` are the normalised
    cell values. The projection of ``psi`` on the bands is exact for either kind
    (:func:`~dmipy_sim.replay.compression.segment_projection`): no quadrature stands between the stored shape and the
    stored vector. Shapes read and write the manifest as :meth:`to_meta` / :meth:`from_meta`."""
    kind: str
    values: tuple = ()

    @classmethod
    def constant(cls):
        """``psi = 1``."""
        return cls("constant", ())

    @classmethod
    def sampled(cls, values):
        """``psi`` held on ``len(values)`` equal cells, ``values`` divided by their largest magnitude."""
        v = np.asarray(values, np.float64).ravel()
        if v.size == 0 or not np.all(np.isfinite(v)) or not np.any(v != 0.0):
            raise ValueError("a sampled shape holds finite values, not all zero")
        return cls("sampled", tuple((v / np.abs(v).max()).tolist()))

    @classmethod
    def of_samples(cls, values):
        """The shape of the waveform samples ``values`` (one per cell): :meth:`constant` when they are one value to
        :data:`_RTOL`, else :meth:`sampled`."""
        v = np.asarray(values, np.float64).ravel()
        if v.size and np.abs(v / (v[np.abs(v).argmax()] or 1.0) - 1.0).max() <= _RTOL:
            return cls.constant()
        return cls.sampled(v)

    @property
    def is_constant(self):
        return self.kind == "constant"

    @property
    def cells(self):
        return 1 if self.is_constant else len(self.values)

    def projection_values(self):
        """The cell values :func:`~dmipy_sim.replay.compression.segment_projection` weights by (``None``: the indicator)."""
        return None if self.is_constant else np.asarray(self.values, np.float64)

    def weight(self, tau):
        """``int |psi| dt`` over a term of duration ``tau``: what bounds a vector's magnitude (``tau`` for a constant)."""
        return float(tau) if self.is_constant else float(tau) * float(np.abs(self.values).mean())

    def area(self, tau):
        """``int psi dt`` over a term of duration ``tau``: the moment the term's level leaves (``tau`` for a constant).
        A sum within the rounding of its own ``int |psi|`` (``cells`` additions at the float64 unit roundoff) is zero."""
        if self.is_constant:
            return float(tau)
        v = np.asarray(self.values, np.float64)
        area, w = float(tau) * float(v.mean()), float(tau) * float(np.abs(v).mean())
        return 0.0 if abs(area) <= v.size * np.finfo(np.float64).eps * w else area

    def edges(self, a, b):
        """The cell edges of the term ``[a, b]``, ``(cells + 1,)`` seconds."""
        if self.is_constant:
            return np.array([a, b], np.float64)
        e = a + (b - a) * np.arange(self.cells + 1) / self.cells
        e[-1] = b
        return e

    def running(self, t, a, b):
        """``int_a^t psi dt`` at the times ``t`` for the term ``[a, b]`` (0 before it, its area after): piecewise linear
        between the cell edges."""
        t = np.asarray(t, np.float64)
        if self.is_constant:
            return np.minimum(np.clip(t - a, 0.0, None), b - a)
        e = self.edges(a, b)
        return np.interp(t, e, np.concatenate([[0.0], np.cumsum(np.asarray(self.values) * np.diff(e))]))

    def on_grid(self, a, b, dt, n_t, where):
        """``(i0, i1, psi)`` of the term ``[a, b]`` on a sequence's grid: its sample range and ``psi`` held on each
        sample; a term or cell edge off the grid, or a cell of a fraction of a sample, is refused naming ``where``."""
        i0, i1 = a / dt, b / dt
        if abs(i0 - round(i0)) > 1e-6 or abs(i1 - round(i1)) > 1e-6 or round(i1) > n_t:
            raise ValueError(f"{where} [{a:.6g}, {b:.6g}] s is not on the sequence's grid (dt = {dt:.6g} s, {n_t} samples)")
        i0, i1 = int(round(i0)), int(round(i1))
        if self.is_constant:
            return i0, i1, np.ones(i1 - i0)
        per, rem = divmod(i1 - i0, self.cells)
        if rem or per < 1:
            raise ValueError(f"the {self.cells} cells of {where} [{a:.6g}, {b:.6g}] s are not on the sequence's grid "
                             f"(dt = {dt:.6g} s, {i1 - i0} samples): a cell is a whole number of samples")
        return i0, i1, np.repeat(np.asarray(self.values, np.float64), per)

    def to_meta(self):
        return dict(kind="constant") if self.is_constant else dict(kind="sampled", normalisation="max|psi| = 1",
                                                                    cells=self.cells, values=list(self.values))

    @classmethod
    def from_meta(cls, rec, where):
        """The shape of the manifest record ``rec`` of ``where`` (a term); a record that is missing, names no known kind,
        lacks a field its kind needs or is not normalised is refused naming the field."""
        if not isinstance(rec, dict) or "kind" not in rec:
            raise ValueError(f"{where} has no shape record ('shape': {{'kind': 'constant'}} or {{'kind': 'sampled', "
                             "'cells': n, 'values': [...]}): every term records its shape explicitly")
        if rec["kind"] == "constant":
            return cls.constant()
        if rec["kind"] != "sampled":
            raise ValueError(f"{where} has the shape kind {rec['kind']!r}; a shape is 'constant' or 'sampled'")
        for k in ("cells", "values"):
            if k not in rec:
                raise ValueError(f"the sampled shape of {where} has no {k!r} field")
        v = np.asarray(rec["values"], np.float64)
        if int(rec["cells"]) != v.size or v.size == 0 or abs(float(np.abs(v).max()) - 1.0) > 1e-12:
            raise ValueError(f"the sampled shape of {where} is not {int(rec['cells'])} normalised cell values "
                             "(max |psi| = 1)")
        return cls("sampled", tuple(v.tolist()))


def _term_shapes(rec, name):
    """The shape records of a partition's ``rec`` (rows or manifest), one per term; a record that states none is
    refused naming the field."""
    if rec.get("shapes") is None:
        raise ValueError(f"partition {name!r} has no 'shapes' record: every term records its shape explicitly "
                         "(a constant segment records {'kind': 'constant'})")
    return rec["shapes"]


@dataclass(frozen=True)
class Partition:
    """Terms of the walk's clock, ``segments = ((a, b), ...)`` seconds in order of their start, each with a
    :class:`Shape` ``shapes[s]`` (``psi_s`` on ``[a_s, b_s]``; a constant segment is ``psi = 1``) and ``gate`` the sign
    of the effective gate on it (+1, -1, or 0 while the magnetisation is stored, where the term carries no phase);
    ``template`` the sequence the gate was read from (its grid, RF schedule and readout are the tiers' gate). Terms
    may overlap in time (a tensor encoding is three shapes on one interval); a partition of disjoint constant
    segments is the common case. :meth:`of` reads one from a sequence's lobes, :meth:`from_sequence` shapes from a
    sequence's waveforms."""
    name: str
    segments: tuple
    gate: tuple
    shapes: tuple
    template: object = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if not (len(self.segments) == len(self.gate) == len(self.shapes)) or not all(isinstance(s, Shape) for s in self.shapes):
            raise ValueError(f"partition {self.name!r} is one Shape and one gate per segment")

    @classmethod
    def of(cls, name, sequence, segments=None):
        """The partition of ``sequence``: its lobes (the maximal runs of one non-zero effective gradient), or the stated
        ``segments`` (``[(a, b), ...]`` s, in time order and disjoint), each a constant term with the effective gate's
        sign on it; a segment the gate changes sign inside (a refocusing pulse) is refused by name."""
        seq = _sequence(sequence)
        if segments is None:
            segments = _lobes(np.asarray(seq.G_eff, np.float64), seq.dt)
        return cls._make(name, seq, segments, None, disjoint=True)

    @classmethod
    def from_sequence(cls, name, sequence, *, terms=None, rows=0):
        """The partition of ``sequence`` whose terms take the shapes of its waveforms: term ``s`` is the interval
        ``terms[s]`` (``(a, b)`` s on the sequence's grid; by default the maximal runs of row ``rows[0]``'s non-zero
        effective gradient within one effective-gate sign) and its shape the normalised waveform of row ``rows[s]``
        (``rows`` an int for every term) along its own direction on the interval, one value per sample --
        :meth:`Shape.of_samples`, so a plateau is a constant term. Terms may repeat an interval with different rows
        (a tensor encoding's three shapes), and their shapes must be linearly independent on the grid. A row that is
        not along one direction on its term is refused by name. The level of the term in a fitted sequence
        (:meth:`levels`) is the amplitude and direction that waveform is played at."""
        seq = _sequence(sequence)
        G = np.asarray(seq.G_eff, np.float64); dt = float(seq.dt)
        rws = [int(rows)] * (len(terms) if terms is not None else 1) if np.ndim(rows) == 0 else [int(r) for r in rows]
        if any(not 0 <= r < G.shape[0] for r in rws):
            raise ValueError(f"rows {rws} are not measurements of the sequence ({G.shape[0]} rows)")
        if terms is None:
            e = np.asarray(seq.effective_gate, np.float64)
            tol = _RTOL * (float(np.abs(G[rws[0]]).max()) or 1.0)
            on = np.abs(G[rws[0]]).max(axis=1) > tol
            terms, i = [], 0
            while i < G.shape[1]:
                if not on[i]:
                    i += 1; continue
                j = i + 1
                while j < G.shape[1] and on[j] and e[j] == e[i]:
                    j += 1
                terms.append((i * dt, j * dt)); i = j
            rws = rws[:1] * len(terms)
        terms = [(float(a), float(b)) for a, b in terms]
        if len(rws) != len(terms):
            raise ValueError(f"rows is one measurement per term: {len(rws)} rows for {len(terms)} terms")
        shapes = []
        for s, ((a, b), r) in enumerate(zip(terms, rws)):
            i0, i1, _ = Shape.constant().on_grid(a, b, dt, G.shape[1], f"term {s} of partition {name!r}")
            v = G[r, i0:i1, :]
            peak = float(np.abs(v).max())
            if peak == 0.0:
                raise ValueError(f"row {r} plays no gradient on term {s} [{a:.6g}, {b:.6g}] s of partition {name!r}")
            g_hat = v[np.linalg.norm(v, axis=1).argmax()]; g_hat = g_hat / np.linalg.norm(g_hat)
            psi = v @ g_hat
            if np.abs(v - psi[:, None] * g_hat[None, :]).max() > _RTOL * peak:
                raise ValueError(f"row {r} is not along one direction on term {s} [{a:.6g}, {b:.6g}] s of partition "
                                 f"{name!r}: a term's shape is a scalar waveform (give each axis its own term)")
            shapes.append(Shape.of_samples(psi))
        return cls._make(name, seq, terms, shapes, disjoint=False)

    @classmethod
    def _make(cls, name, seq, segments, shapes, *, disjoint):
        segs = tuple((float(a), float(b)) for a, b in segments)
        if not segs:
            raise ValueError(f"partition {name!r} has no segment: give segments= or a sequence with a gradient")
        shapes = tuple(Shape.constant() for _ in segs) if shapes is None else tuple(shapes)
        for s, (a, b) in enumerate(segs):
            if not 0.0 <= a < b:
                raise ValueError(f"segment {s} of partition {name!r} is [{a:.6g}, {b:.6g}] s: a segment is 0 <= a < b")
            if s and a < segs[s - 1][(1 if disjoint else 0)] - 1e-12:
                raise ValueError(f"the segments of partition {name!r} are in time order" + (" and disjoint" if disjoint else ""))
        if max(b for _, b in segs) > seq.n_t * seq.dt * (1.0 + 1e-9):
            raise ValueError(f"partition {name!r} ends at {max(b for _, b in segs):.6g} s, beyond its sequence's grid")
        e = np.asarray(seq.effective_gate, np.float64)
        gate = []
        for s, (a, b) in enumerate(segs):
            i0 = int(np.floor(a / seq.dt + 1e-6)); i1 = max(int(np.ceil(b / seq.dt - 1e-6)), i0 + 1)
            vals = np.unique(e[i0:i1])
            if len(vals) != 1:
                raise ValueError(f"the effective gate changes sign inside segment {s} [{a:.6g}, {b:.6g}] s of partition "
                                 f"{name!r} (a pulse acts there): split the segment at the pulse")
            gate.append(int(np.rint(vals[0])))
        out = cls(str(name), segs, tuple(gate), shapes, seq)
        if not disjoint:
            out.design(seq.dt, seq.n_t)
        return out

    @property
    def is_constant(self):
        return all(sh.is_constant for sh in self.shapes)

    @property
    def is_indicator(self):
        """Whether every term is a constant segment disjoint from the others: the sequences of one level per segment."""
        return self.is_constant and all(self.segments[s][0] >= self.segments[s - 1][1] - 1e-12 for s in range(1, len(self.segments)))

    @property
    def durations(self):
        return np.array([b - a for a, b in self.segments], np.float64)

    @property
    def areas(self):
        """``(n_term,)``: ``int psi_s dt``, the moment a unit level on term ``s`` leaves (its duration for a segment)."""
        return np.array([sh.area(b - a) for sh, (a, b) in zip(self.shapes, self.segments)], np.float64)

    @property
    def weights(self):
        """``(n_term,)``: ``int |psi_s| dt``, what bounds the magnitude of a unit level's moment (its duration for a
        segment)."""
        return np.array([sh.weight(b - a) for sh, (a, b) in zip(self.shapes, self.segments)], np.float64)

    @property
    def carrying(self):
        """Per segment whether its gate lets a phase accrue there (``gate != 0``)."""
        return np.array([g != 0 for g in self.gate], bool)

    def projection_shapes(self, idx=None):
        """The ``shapes`` argument of :func:`~dmipy_sim.replay.compression.segment_projection` for the terms ``idx``
        (all when None); ``None`` when every one is a segment."""
        sel = range(len(self.shapes)) if idx is None else idx
        vals = [self.shapes[s].projection_values() for s in sel]
        return None if all(v is None for v in vals) else vals

    def design(self, dt, n_t):
        """``(n_t, n_term)``: ``psi_s`` held on each sample of the grid (0 outside term ``s``), and ``covered`` ``(n_t,)``
        the samples inside some term; a term off the grid is refused by name, and terms whose shapes are linearly
        dependent on the grid (their levels would not be determined) likewise."""
        D = np.zeros((int(n_t), len(self.segments))); covered = np.zeros(int(n_t), bool)
        for s, ((a, b), sh) in enumerate(zip(self.segments, self.shapes)):
            i0, i1, psi = sh.on_grid(a, b, float(dt), int(n_t), f"term {s} of partition {self.name!r}")
            D[i0:i1, s] = psi; covered[i0:i1] = True
        if not self.is_indicator:
            sv = np.linalg.svd(D, compute_uv=False)
            if sv.size and sv[-1] <= sv[0] * max(D.shape) * np.finfo(np.float64).eps:
                raise ValueError(f"the shapes of partition {self.name!r} are linearly dependent on the sequence's grid: "
                                 "the levels of its terms are not determined by a gradient")
        return D, covered

    def b_matrix(self):
        """``(n_seg, n_seg)``: ``B_st = gamma^2 int Q_s Q_t dt`` with ``Q_s(t) = int_a^t psi_s`` the running integral of
        term ``s``, so that an effective encoding with levels ``l`` on the terms has ``b = l^T B l`` (s/m^2) when it is
        refocused; exact (each ``Q_s`` is piecewise linear between the cell edges)."""
        from ..constants import GAMMA
        S = np.asarray(self.segments, np.float64)
        pts = np.unique(np.concatenate([[0.0]] + [sh.edges(a, b) for sh, (a, b) in zip(self.shapes, self.segments)]))
        Q = np.stack([sh.running(pts, a, b) for sh, (a, b) in zip(self.shapes, self.segments)])
        L = np.diff(pts); qa, qb = Q[:, :-1], Q[:, 1:]
        B = (L[None, None, :] * (2.0 * qa[:, None] * qa[None] + qa[:, None] * qb[None] + qb[:, None] * qa[None]
                                 + 2.0 * qb[:, None] * qb[None]) / 6.0).sum(-1)
        return GAMMA ** 2 * B

    def refocused_basis(self):
        """``(idx, N)``: the terms that carry a phase and an orthonormal basis ``N`` ``(len(idx), len(idx) - 1)`` of
        the levels on them that leave no net moment (``sum_s l_s A_s = 0``, ``A_s`` the term's :attr:`areas`); terms of
        zero area (a whole number of periods of a cosine) each leave none and ``N`` is the identity."""
        idx = np.flatnonzero(self.carrying)
        a = self.areas[idx]
        if idx.size and not np.any(a):
            return idx, np.eye(idx.size)
        if idx.size < 2:
            return idx, np.zeros((idx.size, 0))
        _, _, Vt = np.linalg.svd(a[None, :])
        return idx, Vt[1:].T

    def class_vertices(self, G_max):
        """``(idx, V)``: the terms that carry a phase and the vertices ``V`` ``(n_v, len(idx))`` of the refocused
        levels a scanner class plays on one axis, ``{|l_s| <= G_max, sum_s l_s A_s = 0}`` (``A_s`` the :attr:`areas`):
        a box cut by a plane, whose vertices have every level but one at ``+-G_max`` (the box's own, when every area
        is zero). A convex function of the levels (a phase variance, a b) is largest over them at one of these."""
        idx = np.flatnonzero(self.carrying)
        d = self.areas[idx]
        n = idx.size
        G = float(G_max)
        out = []
        if n and not np.any(d):
            out = [G * np.asarray(signs) for signs in itertools.product((-1.0, 1.0), repeat=n)]
        elif n >= 2:
            for i in np.flatnonzero(d):
                rest = [j for j in range(n) if j != i]
                for signs in itertools.product((-1.0, 1.0), repeat=n - 1):
                    v = np.zeros(n)
                    v[rest] = G * np.asarray(signs)
                    v[i] = -float(d[rest] @ v[rest]) / float(d[i])
                    if abs(v[i]) <= G * (1.0 + 1e-12):
                        out.append(v)
        if not out:
            return idx, np.zeros((0, n))
        return idx, np.unique(np.round(np.asarray(out) / G, 12), axis=0) * G

    def reach(self, G_max, b_max=None):
        """``(n_seg,)``: the largest level ``|G_s|`` (T/m, the vector's) an encoding within the envelope puts on each
        segment (0 on a segment the gate stores): over a scanner class's refocused levels (``b_max`` None,
        :meth:`class_vertices`) ``sqrt(3)`` times the largest per-axis ``|l_s|``, every axis at its own, else
        ``min(G_max, sqrt(b_max [N (N^T B N)^-1 N^T]_ss))`` over those within ``b_max``."""
        out = np.zeros(len(self.segments))
        if b_max is None:
            idx, V = self.class_vertices(G_max)
            if V.shape[0]:
                out[idx] = np.sqrt(3.0) * np.abs(V).max(axis=0)
            return out
        idx, N = self.refocused_basis()
        if N.shape[1] == 0:
            return out
        M = N.T @ self.b_matrix()[np.ix_(idx, idx)] @ N
        P = N @ np.linalg.solve(M, N.T)
        out[idx] = np.minimum(float(G_max), np.sqrt(float(b_max) * np.clip(np.diag(P), 0.0, None)))
        return out

    def levels(self, sequence):
        """``(n_meas, n_seg, 3)``: each row's effective gradient on each term (T/m), the vectors ``a_s`` with
        ``G(t) = sum_s psi_s(t) a_s`` to :data:`_RTOL` of the peak.

        Disjoint constant segments take the gradient's value on them, which must be constant on a segment. Otherwise (a
        shape, or terms that overlap) the levels are the least squares fit of ``G(t)`` on the terms' shapes on the
        sequence's grid, and the residual is held to the same tolerance. A sequence refused by name: off the terms'
        grid, a gradient outside every term, off the shapes (the row, the term the residual is largest in and its
        relative norm named), on a term its gate stores, or leaving a net moment."""
        seq = _sequence(sequence)
        G = np.asarray(seq.G_eff, np.float64); dt = float(seq.dt)
        n_meas, n_t, _ = G.shape
        peak = float(np.abs(G).max()) or 1.0; tol = _RTOL * peak
        design, covered = self.design(dt, n_t)
        if np.abs(G[:, ~covered, :]).max(initial=0.0) > tol:
            raise ValueError(f"the sequence plays a gradient outside the segments of partition {self.name!r}: it does not "
                             "fit the partition")
        if self.is_indicator:
            L = np.zeros((n_meas, len(self.segments), 3))
            for s, (a, b) in enumerate(self.segments):
                i0, i1, _ = self.shapes[s].on_grid(a, b, dt, n_t, f"segment {s} of partition {self.name!r}")
                lev = G[:, i0, :]
                if np.abs(G[:, i0:i1, :] - lev[:, None, :]).max() > tol:
                    raise ValueError(f"the sequence's effective gradient is not constant on segment {s} [{a:.6g}, {b:.6g}] s "
                                     f"of partition {self.name!r}: it does not fit the partition")
                L[:, s] = lev
        else:
            X = np.moveaxis(G, 1, 0).reshape(n_t, -1)
            sol = np.linalg.lstsq(design, X, rcond=None)[0]
            L = np.moveaxis(sol.reshape(len(self.segments), n_meas, 3), 0, 1)
            R = np.moveaxis((X - design @ sol).reshape(n_t, n_meas, 3), 0, 1)
            if np.abs(R).max(initial=0.0) > tol:
                self._refuse_residual(R, G, design)
        if np.abs(L[:, ~self.carrying]).max(initial=0.0) > tol:
            raise ValueError(f"the sequence plays an effective gradient on a segment of partition {self.name!r} its gate "
                             "stores: the partition holds no vector there")
        net = np.einsum("msc,s->mc", L, self.areas)
        if np.abs(net).max(initial=0.0) > _RTOL * peak * float(self.weights.sum()):
            raise ValueError(f"the sequence leaves a net moment at the readout on partition {self.name!r} (an unbalanced "
                             "encoding): a segment vector is the path relative to the walker's start, which only a "
                             "refocused encoding does not see")
        return L

    def _refuse_residual(self, R, G, design):
        """Refuse a least squares fit whose residual ``R`` ``(n_meas, n_t, 3)`` of ``G`` is beyond the tolerance, naming
        the row with the largest residual, the term it lies in and its relative norm."""
        m = int(np.abs(R).max(axis=(1, 2)).argmax())
        energy = ((R[m] ** 2).sum(1)[:, None] * (design != 0)).sum(0)
        term = int(energy.argmax())
        rel = float(np.sqrt((R[m] ** 2).sum() / max((G[m] ** 2).sum(), 1e-300)))
        a, b = self.segments[term]
        raise ValueError(f"the sequence's effective gradient does not fit the shapes of partition {self.name!r}: row {m} "
                         f"leaves a residual of relative norm {rel:.3g} (largest on term {term} [{a:.6g}, {b:.6g}] s), "
                         f"above the {_RTOL:.0e} of the peak a sequence is held to")

    def to_meta(self):
        return dict(segments=[list(s) for s in self.segments], gate=list(self.gate), shapes=[sh.to_meta() for sh in self.shapes])


def _partitions(partitions):
    """``{name: Partition}`` of ``{name: sequence}``, ``{name: Partition}`` or a list of partitions."""
    out = {}
    items = partitions.items() if isinstance(partitions, dict) else ((p.name, p) for p in partitions)
    for name, p in items:
        out[str(name)] = p if isinstance(p, Partition) else Partition.of(name, p)
    if not out:
        raise ValueError("no partitions: give at least one {name: sequence} or Partition")
    return out


# ------------------------------------------------------------------ the source
def segment_vectors(pack, partition):
    """``(n_w, n_term, 3)`` float64: every walker's term vectors ``gamma int psi_s(t) (r(t) - r(0)) dt`` (rad per T/m;
    ``psi_s = 1`` on a segment) of ``partition`` from the pack's coefficients, each window's bands against
    :func:`~dmipy_sim.replay.compression.segment_projection` (a window that opens after the last term adds nothing)."""
    from ..constants import GAMMA
    from .compression import read_position_coeffs, segment_projection
    S = np.asarray(partition.segments, np.float64).reshape(-1, 2)
    shapes = partition.projection_shapes()
    m = x0 = None
    for win, t0, T_w in pack._windows():
        if m is not None and t0 >= S[:, 1].max():
            break
        C = read_position_coeffs(win.arrays, dtype=np.float64, cached=False)   # the layout subtracts the start in place: a decode of its own
        if x0 is None:
            x0 = C[:, 0, :].copy()
        C[:, 0, :] -= x0
        W = segment_projection(S, T_w, C.shape[1] - 2, t0=t0, shapes=shapes)
        part = GAMMA * np.einsum("wjc,sj->wsc", C, W)
        m = part if m is None else m + part
    return m


def level_bound(partition, A, env):
    """The largest ``sum_c l_c^T A_c l_c`` over the encodings within the layout's envelope ``env``, ``A`` ``(n_axes,
    n, n)`` per axis over the segments that carry a phase (positive semi-definite): over a scanner class's levels
    (``env["b_max"]`` None) each axis at its own worst vertex (:meth:`Partition.class_vertices`, at most ``G_max`` per
    axis), else over the refocused levels with ``l^T B l <= b_max``, ``b_max`` times the largest generalised eigenvalue
    of ``(N^T A_c N, N^T B N)`` over the axes."""
    from scipy.linalg import eigh
    A = np.asarray(A, np.float64)
    if env.get("b_max") is None:
        _, V = partition.class_vertices(env["G_max"])
        if V.shape[0] == 0:
            return 0.0
        return float(sum(max(float(np.einsum("vs,st,vt->v", V, A[c], V).max()), 0.0) for c in range(A.shape[0])))
    idx, N = partition.refocused_basis()
    if N.shape[1] == 0:
        return 0.0
    M = N.T @ partition.b_matrix()[np.ix_(idx, idx)] @ N
    lam = max(float(eigh(N.T @ A[c] @ N, M, eigvals_only=True).max()) for c in range(A.shape[0]))
    return float(env["b_max"]) * max(lam, 0.0)


def quantisation_term(pack, partition, env):
    """The pack's band quantisation contracted through ``partition`` at its worst over the layout's envelope ``env``,
    rad^2: the bands' rounding (a step ``scale_k`` is a variance ``scale_k^2 / 12``, the largest over the scale tables)
    moves an encoding's phase by ``l^T A l`` with ``A_st = gamma^2 sum_k W_sk W_tk scale_k^2 / 12`` per axis, at its
    largest over the envelope's levels (:func:`level_bound`). Zero for the float container."""
    from ..constants import GAMMA
    from .compression import position_container, segment_projection
    if position_container(pack.arrays) != "bands":
        return 0.0
    idx, N = partition.refocused_basis()
    if N.shape[1] == 0:
        return 0.0
    S = np.asarray(partition.segments, np.float64)[idx]
    shapes = partition.projection_shapes(idx)
    A = None
    for win, t0, T_w in pack._windows():
        if A is not None and t0 >= S[:, 1].max():
            break
        sc = np.asarray(win.arrays["pos_band_scale"], np.float64)
        step2 = (sc.reshape(-1, sc.shape[-2], sc.shape[-1]) ** 2).max(axis=0)          # (n_axes, K)
        W = segment_projection(S, T_w, step2.shape[1], t0=t0, shapes=shapes)[:, 2:]
        Aw = np.stack([GAMMA ** 2 * (W * step2[c][None, :]) @ W.T / 12.0 for c in range(step2.shape[0])])
        A = Aw if A is None else A + Aw
    return level_bound(partition, A, env)


def _walker_max(arrays, f, slice_bytes=None):
    """The largest value of ``f`` over every walker of ``arrays``: ``f`` maps a walker range's tensors
    (:func:`~dmipy_sim.replay.compression.walker_rows`) to its elementwise largest value over those walkers, read a
    range of :func:`~dmipy_sim.replay.compression.slice_walkers` walkers at a time."""
    from .compression import position_shape, slice_walkers, walker_rows
    n, step, out = position_shape(arrays)[0], slice_walkers(arrays, slice_bytes), None
    for a in range(0, n, step):
        v = f(walker_rows(arrays, a, a + step))
        out = v if out is None else np.maximum(out, v)
    return out


def tier_quantisation_terms(pack, gate_seq, env, *, slice_bytes=None):
    """``(contact, field)``: the pack's own quantisation of the gated contact and of the gated field channels at their
    worst over the envelope, rad^2 of phase (the contact's relative weight error counted as a phase, as its width is).
    Each contraction is linear in the stored bands / modes, so a step ``s_k`` (the largest over the scale tables) is a
    variance ``w_k^2 s_k^2 / 12`` with ``w_k`` the contraction's weight on it: the contact's ``(rho2 / D)_max`` times the
    gated bridge's weights (:func:`~dmipy_sim.replay.replay.surface_logweight` of unit bands), each field component's
    ``B0 chi_max alpha_c gamma g_k`` with ``g_k`` the gate's cosine projection (``alpha`` 2 on ``iso_local``, which the
    layout forms ``iso_P_zz`` from; a pack's stored ``iso_P_zz`` is not read). A band or mode that is zero on every
    walker holds no step (its scale is a placeholder). Zero for a float container. The live bands and modes are read a
    walker range at a time (``slice_bytes``, :func:`~dmipy_sim.replay.compression.slice_walkers`)."""
    from ..constants import GAMMA
    from . import compression as cx
    from .bank import susc_path_coeffs
    from .replay import surface_logweight
    chi = np.asarray(gate_seq.chi_perp if gate_seq.chi_perp is not None else np.ones(gate_seq.n_t), np.float64)
    eg = np.asarray(gate_seq.effective_gate, np.float64)[:-1]               # the field's gate, the readout held over nothing
    ch = (pack.meta.get("compression") or {}).get("channels") or {}
    v_c = v_f = 0.0
    for win, t0, T_w in pack._windows():
        if t0 >= gate_seq.T:
            break
        cm = ch.get("boundary_local_time")
        if cm and "blt_band_scale" in win.arrays:
            sc = np.asarray(win.arrays["blt_band_scale"], np.float64)
            step2 = (sc.reshape(-1, sc.shape[-1]) ** 2).max(axis=0)
            live = _walker_max(win.arrays, lambda a: np.abs(cx.bridge_bands(a, cm)).max(axis=0), slice_bytes)[:step2.shape[0]] > 0
            step2 = np.where(live, step2, 0.0)
            K = step2.shape[0]
            a_, m_ = cx.encode_boundary_bands(np.zeros(K, np.float32), np.zeros(K, np.float32), np.eye(K), T=T_w,
                                              dtype=np.float32, container=None)
            w = np.asarray(surface_logweight(a_, 1.0, m_, chi, float(gate_seq.dt), t0=t0), np.float64)
            v_c += float(np.sum(w ** 2 * step2) / 12.0)
        pm = ch.get("susc_field_basis")
        if pm and "susc_path_scale" in win.arrays:
            S = np.asarray(win.arrays["susc_path_scale"], np.float64)
            step2 = (S.reshape((-1,) + S.shape[-2:]) ** 2).max(axis=0)          # (stored components, K)
            full = susc_path_coeffs(cx.walker_rows(win.arrays, 0, 1), pm)[1]
            amax = _walker_max(win.arrays, lambda a: np.abs(susc_path_coeffs(a, pm)[0]).max(axis=0), slice_bytes)   # (full components, K)
            stored = list(pm["channels"])
            live = np.stack([amax[full.index(nm), :step2.shape[1]] > 0 for nm in stored])
            step2 = np.where(live, step2, 0.0)
            g = cx.cosine_projection(eg.reshape(1, -1, 1), float(gate_seq.dt), T_w, step2.shape[1], t0=t0)[0, :, 0]
            alpha = np.array([2.0 if nm == "iso_local" else (0.0 if nm == "iso_P_zz" else 1.0) for nm in stored])
            v_f += float(np.sum((alpha[:, None] * GAMMA * g[None, :]) ** 2 * step2) / 12.0)
    return env["rho2_over_D_max"] ** 2 * v_c, env["B0_chi_max_T"] ** 2 * v_f


def _walk_view(walk, weights=None):
    """The walk's float coefficients as an uncertified in-memory :class:`~dmipy_sim.replay.replay.ReplayPack`: per
    window the positions, the contact and the field modes in the float containers and the occupancy's events, exactly
    the walk's own numbers, so that every reader of a pack contracts the walk before any quantiser."""
    from .bank import _coefficient_weights
    from . import compression as cx
    from .replay import ReplayPack
    w = _coefficient_weights(walk, weights)
    n, T_w = int(walk.n_windows), float(walk.T_window)
    arrays, channels = {}, {}
    names = walk.field_channel_names
    for i in range(n):
        cw = walk.window(i)
        p = "" if i == 0 else f"s{i}/"
        t0 = float((cw.block.get("window") or {}).get("t0", cw.block.get("t_start", 0.0)))
        a = dict(cx.pack_position_arrays(np.asarray(cw.position_coeffs)[:, 0], np.float32))
        if cw.boundary_bands is not None:
            a_, cm = cx.encode_boundary_bands(np.zeros(cw.n_walkers, np.float32), np.asarray(cw.boundary_endpoint)[:, 0],
                                              np.asarray(cw.boundary_bands)[:, 0], T=T_w, dtype=np.float32, container=None)
            a.update(a_); channels["boundary_local_time"] = cm
        if cw.compartment is not None:
            cols = {"comp": (np.asarray(cw.compartment), cw.comp_events)}
            if cw.bound_start is not None:
                cols["bound"] = (np.asarray(cw.bound_start), cw.bound_events)
            a_, cm = cx.encode_events_occupancy(cols, t0=t0)
            a.update(a_); channels["compartment"] = cm
        if cw.field_coeffs is not None:
            fc = np.asarray(cw.field_coeffs, np.float32)[:, 0, :int(walk.K_c3), :]
            a["susc_path_dct"] = np.ascontiguousarray(np.transpose(fc, (0, 2, 1)))
            channels["susc_field_basis"] = dict(channel="susc_field_basis", K=int(walk.K_c3), T=T_w, n_ch=len(names),
                                                channels=list(names), sampling="accumulated_in_walk",
                                                read_interval=walk.field_read_interval)
            channels["susceptibility_grid"] = dict(has_aniso="aniso_G_xx" in names, arrays_in_pack=False, replay_route="path")
        arrays.update({p + k: v for k, v in a.items()})
    if w is not None:
        arrays["spin_weights"] = np.asarray(w, np.float32)
    meta = dict(id=None, compression=dict(method=cx.POSITION_METHOD, K=int(walk.K), T=T_w, channels=channels),
                walk_params=dict(n_walkers=int(walk.n_walkers), T_max=float(walk.T), diffusivity=walk.diffusivity, seed=walk.seed,
                                 windows=dict(n=n, T=T_w, walks=[dict(first=0, last=n - 1, seed=walk.seed)]), block=dict(walk.block)),
                substrate=walk.spec.to_dict(), fidelity={}, view=dict(of="walk", containers="float"))
    return ReplayPack(arrays, meta)


def walk_identity(walk):
    """What names a walk: its seed, walkers, band, window, block and the sha256 of its spec and of its walkers' exact
    starts."""
    starts = np.ascontiguousarray(np.asarray(walk.position_coeffs)[:, 0, 0, :], np.float32)
    spec = None if walk.spec is None else json.dumps(walk.spec.to_dict(), sort_keys=True, default=str)
    return dict(seed=(None if walk.seed is None else str(walk.seed)), n_walkers=int(walk.n_walkers), K=int(walk.K), T=float(walk.T),
                n_windows=int(walk.n_windows), block_h=float(walk.h),
                spec_sha256=(None if spec is None else hashlib.sha256(spec.encode()).hexdigest()),
                starts_sha256=hashlib.sha256(starts.tobytes()).hexdigest())


def _envelope(meta, g_max):
    """What the widths are derived at, from a pack's header: the encodings -- a scanner class's (``acquisition.bound``
    ``"scanner_class"``: every level at most ``G_max`` per axis, ``b_max`` None) or those within ``b_max``
    (``replay_envelope.acquisition.b_max``) --, ``G_max`` (``g_max``, at most the class's, else the envelope's, else
    the pack's scanner class's), the largest ``rho2 / D`` (the contact tier's
    envelope, ``replay_envelope.tissue.rho2_over_D_max``; 0 for a pack with no contact channel, and a pack with one
    that states none is refused) and the largest ``B0 chi``: the one certified field scale, the largest of
    ``replay_envelope.acquisition.B0_list`` times ``replay_envelope.tissue.chi_max`` (0 for a pack with no field
    channel, and a pack with one that states either not is refused)."""
    env = meta.get("replay_envelope") or {}
    acq = env.get("acquisition") or {}
    by_class = acq.get("bound") == "scanner_class"
    if not by_class and acq.get("b_max") is None:
        raise ValueError("the pack declares no acquisition envelope (replay_envelope.acquisition.b_max or a scanner "
                         "class): a segment's width is derived from the encodings the pack is certified for")
    if by_class and g_max is not None and float(g_max) > float(acq["G_max"]) * (1.0 + 1e-12):
        raise ValueError(f"g_max = {float(g_max):.4g} T/m is beyond the {float(acq['G_max']):.4g} T/m per axis of the "
                         f"scanner class {acq['scanner']!r} the pack is certified for")
    if g_max is None:
        g_max = acq.get("G_max")
    if g_max is None and (meta.get("compression") or {}).get("scanner"):
        from ..acquisition.scanners import ScannerLimits
        g_max = ScannerLimits.of(meta["compression"]["scanner"]).G_max
    if g_max is None:
        raise ValueError("the pack names no scanner class and its envelope no G_max: give g_max= (T/m), the largest "
                         "gradient the layout is certified for")
    rho = (env.get("tissue") or {}).get("rho2_over_D_max")
    if rho is None:
        if ((meta.get("compression") or {}).get("channels") or {}).get("boundary_local_time") is not None:
            raise ValueError("the pack carries a contact channel and states no contact envelope "
                             "(replay_envelope.tissue.rho2_over_D_max): the contact's width is derived at the largest "
                             "rho2 / D its certificate serves")
        rho = 0.0
    b0chi = b0max = 0.0
    if ((meta.get("compression") or {}).get("channels") or {}).get("susc_field_basis") is not None:
        from .compression import envelope_B0_chi_max, envelope_B0_max
        tis = env.get("tissue") or {}
        if tis.get("chi_max") is None:
            raise ValueError("the pack carries a field channel and states no certified susceptibility scale "
                             "(replay_envelope.tissue.chi_max): the field's width is derived at the one certified B0 chi")
        stated = dict(B0_list=acq.get("B0_list"), chi_max=tis["chi_max"])
        b0chi, b0max = envelope_B0_chi_max(stated), envelope_B0_max(stated)
    return dict(b_max=(None if by_class else float(acq["b_max"])), G_max=float(g_max), rho2_over_D_max=float(rho),
                B0_chi_max_T=float(b0chi), B0_max_T=float(b0max),
                **(dict(bound="scanner_class", scanner=acq["scanner"]) if by_class else dict(bound="b_max")))


def _floor_target(meta, given):
    """The floor the widths are derived at: ``given``, else the pack's ``fidelity.target_floor``, else its median
    per-voxel floor."""
    if given is not None:
        return float(given)
    fid = meta.get("fidelity") or {}
    if fid.get("target_floor") is not None:
        return float(fid["target_floor"])
    pv = fid.get("per_voxel") or {}
    if pv.get("floor_median") is not None:
        return float(pv["floor_median"])
    raise ValueError("the pack declares no floor target (fidelity.target_floor) and no per-voxel certificate: give floor_target=")


def _group_tiers(view, gate_seq, *, tiers, background, background_amplitude, ref=None):
    """One sequence group's rows and record from ``view`` under the gate ``gate_seq``: the pool, the gated contact and
    field channels (with ``tiers``) and the background vector (with ``background``). ``ref`` is the record of the
    walkers before ``view``'s (a walker range's): their exposure times are the ones every walker's must equal."""
    from .replay import _band_phase, _path_field_channels
    from .study import Acquisition
    from ..acquisition.epg import pathway_weight
    rows, rec = {}, dict(pathway=float(pathway_weight(gate_seq)), tau_t2=None, tau_t1=None, contact=False, field=None,
                         background=False)
    n = view.n_walkers
    if tiers:
        prim = view.walker_primitives(Acquisition(gate_seq), tol=None)
        if prim.exposure_t2 is not None:
            e2 = np.asarray(prim.exposure_t2); e1 = np.asarray(prim.exposure_t1)
            if not np.allclose(e2.max(1), e2.sum(1)):
                raise ValueError("a walker was in more than one pool before the echo; a layout stores one pool per walker "
                                 "and needs a walk that does not cross")
            for tau, key in ((e2.max(1), "tau_t2"), (e1.max(1), "tau_t1")):
                t0 = tau[0] if ref is None or ref.get(key) is None else ref[key]
                if tau.size and not np.allclose(tau, t0, atol=1e-9):
                    raise ValueError(f"the {key} exposure differs between walkers of one gate; a non-crossing walk has one")
                rec[key] = float(t0) if tau.size else 0.0
            rows["pool"] = np.argmax(e2, axis=1).astype(np.uint8)
            rec["pools"] = int(e2.shape[1])
        else:
            rows["pool"] = np.zeros(n, np.uint8)
        if prim.contact is not None:
            rows["contact"] = np.asarray(prim.contact, np.float64); rec["contact"] = True
        if prim.field_iso is not None:
            P = view._prepare(gate_seq, tissue=None, scanner=None, orientation=None, compartment=None)
            Psi, names = _path_field_channels(P, gate_seq)
            keep = [i for i, nm in enumerate(names) if nm != "iso_P_zz"]       # the basis's components (RPK.md 5.3)
            Psi = np.asarray(Psi, np.float64)[:, keep]
            rows["field"] = Psi
            rec["field"] = dict(channels=[names[i] for i in keep], amax=None)              # the stored rows' (segment_rows)
        rec["D_walk"] = prim.D_walk
    if background:
        P = view._prepare(_background_sequence(gate_seq), tissue=None, scanner=None, orientation=None, compartment=None)
        if np.any(P["voxel"] != 1.0):
            raise ValueError("the background on this gate leaves a net moment at the readout: its effective gate does not "
                             "refocus a constant gradient")
        rows["background"] = np.asarray(_band_phase(P), np.float64)
        rec["background"] = True
    return rows, rec


def _positions_decoded(pack):
    """``pack`` (a walker range) with every window's position tensors decoded once into the float64 container
    (:func:`~dmipy_sim.replay.compression.pack_position_arrays`): the same coefficients to the bit, so that the
    range's readers -- each partition's vectors, each group's tiers and background -- decode none again."""
    from .compression import POSITION_AXES, pack_position_arrays, read_position_coeffs
    from .replay import ReplayPack
    arrays = dict(pack.arrays)
    shared = {k: v for k, v in arrays.items() if k == "band_block"}
    for p in sorted({k[:len(k) - len(k.split("/")[-1])] for k in arrays if k.split("/")[-1].startswith(POSITION_AXES)}):
        own = [k for k in arrays if k.startswith(p) and "/" not in k[len(p):] and k[len(p):].startswith(("pos_x", "pos_y", "pos_z", "pos_band_scale"))]
        C = read_position_coeffs(dict(shared, **{k[len(p):]: arrays[k] for k in own}), dtype=np.float64)
        for k in own:
            del arrays[k]
        arrays.update({p + k: v for k, v in pack_position_arrays(C, np.float64).items()})
    return ReplayPack(arrays, pack.meta, source=pack.source)


def _top(top, key, rows):
    """``top[key]`` the elementwise largest ``|value|`` of the stored (float32) ``rows`` and of the rows before them:
    a record's largest value is that of what the columns hold, the same bits whichever walker ranges they came in."""
    t = np.abs(rows).max(axis=0, initial=0.0)
    top[key] = np.maximum(top[key], t) if key in top else t


def _join_group_rec(rec, more):
    """``rec`` (a group's record over the walker ranges read so far) joined with ``more`` (the next range's): the pool
    count the larger of the two."""
    if more.get("pools") is not None:
        rec["pools"] = max(int(rec["pools"]), int(more["pools"]))


class _Spill:
    """Float32 columns written a walker range at a time to unsorted files in ``out_dir`` (``.<name>.unsorted``), then
    each gathered into ``<name>.npy`` in a row order a range at a time: no column is held whole in memory."""

    def __init__(self, out_dir):
        self.dir, self.cols = out_dir, {}

    def add(self, name, a):
        """Append rows ``a`` to column ``name`` as float32; returns them as stored."""
        a = np.ascontiguousarray(a, np.float32)
        path, shape, rows = self.cols.get(name) or (os.path.join(self.dir, f".{name}.unsorted"), a.shape[1:], 0)
        with open(path, "ab" if rows else "wb") as f:
            f.write(a.tobytes())
        self.cols[name] = (path, shape, rows + int(a.shape[0]))
        return a

    def save(self, order):
        from .compression import SLICE_BYTES
        for name, (path, shape, rows) in self.cols.items():
            src = np.memmap(path, np.float32, "r", shape=(rows,) + shape)
            dst = np.lib.format.open_memmap(os.path.join(self.dir, f"{name}.npy"), "w+", np.float32, (rows,) + shape)
            step = max(1, SLICE_BYTES // max(1, 4 * int(np.prod(shape, dtype=np.int64))))
            for i in range(0, rows, step):
                dst[i:i + step] = src[order[i:i + step]]
            dst.flush()
            del src, dst

    def discard(self):
        for path, _, _ in self.cols.values():
            if os.path.exists(path):
                os.remove(path)
        self.cols = {}


def _background_sequence(seq, amplitude=1.0):
    """The magnet's own gradient at ``amplitude`` along x, y and z as a three-row sequence on ``seq``'s grid, RF
    schedule and readout: a constant gradient through the pulses and the dead times
    (:meth:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence.with_background_gradient` on a silent waveform),
    so that its phases are a walker's **background vector** ``gamma int e(t) r(t) dt`` with ``e`` the sequence's
    :attr:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence.effective_gate` (the coherence sign, zero while
    stored) -- what a constant ``g0`` adds to the phase as ``g0 . n``."""
    from ..acquisition.waveforms import tile_waveform
    silent = tile_waveform(_gate_sequence(seq), 3)
    return silent.with_background_gradient(float(amplitude) * _AXES)


def segment_rows(source, partitions, out_dir, *, grid=None, pack=None, floor_target=None, g_max=None, tiers=False,
                 background=False, background_amplitude=BACKGROUND_AMPLITUDE, weights=None, slice_bytes=None):
    """One block's segment rows written to ``out_dir``: per partition (``{name: sequence}`` or :class:`Partition` objects)
    its segment vectors, per sequence group (the partitions sharing a grid, an RF schedule and a readout) with
    ``tiers`` the pool, gated contact and field channels, with ``background`` the background vector, the weights and
    each row's voxel on ``grid`` (the pack's per-voxel grid when None), every column float in (voxel, pool) order, and
    ``rows.json``: the partitions with their gates and reaches, the groups, the floor target and the envelope, the
    source and its certificate's terms.

    ``source`` is the walk (:class:`~dmipy_sim.coefficient_walk.CoefficientWalk`), read in its float coefficients, with
    ``pack`` the pack built from it (its sibling; refused when its walkers' starts are not the walk's), or a quantised
    :class:`~dmipy_sim.replay.replay.ReplayPack`. The floor target is ``floor_target``, else the pack's declared one, else
    its median per-voxel floor; the envelope the pack's (``g_max`` when it names no scanner class). Returns the record.

    The source is read a walker range at a time (:meth:`~dmipy_sim.replay.replay.ReplayPack.walkers`, ranges of
    :func:`~dmipy_sim.replay.compression.slice_walkers` at ``slice_bytes``), every column spilled to ``out_dir`` in
    walker order and gathered into (voxel, pool) order by one sort over all rows at the end; a record's largest
    values are those of the stored float32 rows, the largest over the ranges. The rows and the record do not depend on the ranges: memory is bounded
    by one range, not by the walker count."""
    from ..coefficient_walk import CoefficientWalk
    from .compression import slice_walkers
    from ..phantom.grid import Grid
    from .replay import ReplayPack
    parts = _partitions(partitions)
    if isinstance(source, CoefficientWalk):
        if not isinstance(pack, ReplayPack):
            raise ValueError("a walk's segment rows are the sibling of its pack: give pack= (the pack built from this walk), "
                             "whose floor target, envelope and certificate the rows are declared at")
        starts = np.asarray(source.position_coeffs)[:, 0, 0, :].astype(np.float32)
        if pack.n_walkers != source.n_walkers or not np.array_equal(pack.r0.astype(np.float32), starts):
            raise ValueError("pack= is not this walk's pack: its walkers' starts differ")
        view = _walk_view(source, weights)
        if not np.allclose(np.asarray(view.spin_weights, np.float32), np.asarray(pack.spin_weights, np.float32), rtol=1e-6):
            raise ValueError("the walk's weights are not its pack's: give the weights the pack was built with (weights=)")
        decl, kind = pack, "walk"
        src = dict(kind="walk", walk=walk_identity(source), pack=dict(id=pack.id, digest=pack.digest))
    elif isinstance(source, ReplayPack):
        if pack is not None:
            raise ValueError("pack= names a walk's sibling; a pack source is its own declaration")
        view = decl = source; kind = "pack"
        src = dict(kind="pack", walk=None, pack=dict(id=source.id, digest=source.digest))
    else:
        raise TypeError(f"segment rows are read from a CoefficientWalk (with its pack) or a ReplayPack; got {type(source).__name__}")
    meta = decl.meta
    if grid is None:
        g_meta = ((meta.get("fidelity") or {}).get("per_voxel") or {}).get("grid")
        if g_meta is None:
            raise ValueError("the pack carries no per-voxel grid: give grid=")
        grid = Grid.from_meta(g_meta)
    env = _envelope(meta, g_max)
    floor = _floor_target(meta, floor_target)
    eps = rounding_eps(floor)
    n = view.n_walkers
    os.makedirs(out_dir, exist_ok=True)
    ijk, inside = grid.bin(view.r0)
    if not inside.all():
        raise ValueError(f"{int((~inside).sum())} walkers start outside the grid: a row is placed in its start voxel")
    vox = np.ravel_multi_index(ijk.T, tuple(grid.shape)).astype(np.int64)
    part_rec, group_of = {}, {}
    groups = collections.OrderedDict()
    for name, p in parts.items():
        if p.template is None:
            raise ValueError(f"partition {name!r} carries no template sequence: build it with Partition.of")
        key = _group_key(p.template)
        if key not in groups:
            groups[key] = (f"g{len(groups)}", p.template, [])
        groups[key][2].append(name); group_of[name] = groups[key][0]
        reach = p.reach(env["G_max"], env["b_max"])
        if not np.any(reach > 0):
            raise ValueError(f"partition {name!r} has no refocused encoding within the envelope (fewer than two segments "
                             "carry a phase)")
        _check_band(view, p, env)
        v_src = 0.0 if kind == "walk" else quantisation_term(view, p, env)
        part_rec[name] = dict(p.to_meta(), group=group_of[name], reach=reach.tolist(), quantisation=float(v_src),
                              amax=None, column=f"p_{name}.npy")
    # the rows a walker range at a time, each column spilled in walker order; the records' largest values over them
    step = slice_walkers(view.arrays, slice_bytes)
    cols, group_rec, top = _Spill(out_dir), {}, {}
    pool = np.zeros(n, np.uint8)
    try:
        for a in range(0, n, step):
            sl = _positions_decoded(view.walkers(a, a + step))
            for name, p in parts.items():
                _top(top, ("p", name), cols.add(f"p_{name}", segment_vectors(sl, p)))
            for key, (g, seq, names) in groups.items():
                rows, rec = _group_tiers(sl, _gate_sequence(seq), tiers=tiers, background=background,
                                         background_amplitude=background_amplitude, ref=group_rec.get(g))
                if g not in group_rec:
                    group_rec[g] = rec
                else:
                    _join_group_rec(group_rec[g], rec)
                if "pool" in rows:
                    pool[a:a + len(rows["pool"])] = rows["pool"]
                for kind, col in (("contact", "contact"), ("field", "field"), ("background", "bg")):
                    if kind in rows:
                        _top(top, (g, kind), cols.add(f"{col}_{g}", rows[kind]))
        for name in parts:
            part_rec[name]["amax"] = top["p", name].tolist()
        for key, (g, seq, names) in groups.items():
            rec = group_rec[g]
            q_c, q_f = ((0.0, 0.0) if kind == "walk" or not tiers else
                        tier_quantisation_terms(view, _gate_sequence(seq), env, slice_bytes=slice_bytes))
            rec["quantisation"] = dict(contact=float(q_c), field=float(q_f))
            rec.update(partitions=names, rf=seq.rf.to_dicts(), dt=float(seq.dt), n_t=int(seq.n_t),
                       readout=[int(i) for i in seq.readout])
            if (g, "field") in top:
                rec["field"]["amax"] = top[g, "field"].tolist()
            if (g, "contact") in top:
                rec["contact_amax"] = float(top[g, "contact"].max(initial=0.0))
            if (g, "background") in top:
                rec["background_amax"] = top[g, "background"].tolist()
        order = np.lexsort((pool, vox))
        np.save(os.path.join(out_dir, "voxel.npy"), vox[order])
        np.save(os.path.join(out_dir, "w.npy"), np.asarray(view.spin_weights, np.float32)[order])
        if tiers:
            np.save(os.path.join(out_dir, "pool.npy"), pool[order])
        cols.save(order)
    finally:
        cols.discard()
    fid = meta.get("fidelity") or {}
    certificate = dict(terms=fid.get("terms"), source=kind,
                       pack_quantisation=(None if kind == "walk" else
                                          dict(err_quantisation=fid.get("err_quantisation"),
                                               err_surface_quantisation=fid.get("err_surface_quantisation"),
                                               err_susc_path=fid.get("err_susc_path"))))
    tier_rec = None
    if tiers:
        D_walk = next((r.get("D_walk") for r in group_rec.values() if r.get("D_walk") is not None), None)
        n_pools = next((r.get("pools") for r in group_rec.values() if r.get("pools") is not None), None)
        tier_rec = dict(relaxation=n_pools is not None, pools=n_pools, D_walk=D_walk, substrate=meta.get("substrate"))
    record = dict(format=ROWS_FORMAT, n_rows=int(n), grid=grid.to_meta(),
                  source=dict(src, floor_target=float(floor), floor_target_stated=floor_target is not None, eps=float(eps),
                              envelope=env,
                              background_amplitude=float(background_amplitude)),
                  certificate=certificate, partitions=part_rec, groups=group_rec, tiers=tier_rec, background=bool(background),
                  code=_code(), created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(out_dir, ROWS), "w") as f:
        json.dump(record, f, indent=1, default=float)
    return record


def _check_band(view, partition, env):
    """Refuse a partition whose template at the envelope's b_max (a class's: at ``G_max`` on its peak axis) needs more
    temporal band than the source stores (the
    replay's own check, :meth:`~dmipy_sim.replay.replay.ReplayPack._check_band`): a segment's sharp edges are the
    waveform's, and the bands beyond the pack's would carry phase the vectors do not."""
    from ..acquisition.waveforms import calc_b
    seq = partition.template
    b = np.asarray(calc_b(seq), np.float64)
    if not np.any(b > 0):
        return
    if env.get("b_max") is None:                                     # the class's strongest: its peak axis at G_max
        G = np.asarray(seq.G, np.float64)
        view._check_band(seq.with_gradient(G * float(env["G_max"]) / np.abs(G).max()))
        return
    view._check_band(seq.with_gradient(np.asarray(seq.G, np.float64) * np.sqrt(env["b_max"] / b.max())))


def read_rows(path):
    """The record of the segment rows in ``path`` (:func:`segment_rows`), refused when it is another format."""
    with open(os.path.join(path, ROWS)) as f:
        rec = json.load(f)
    if rec.get("format") != ROWS_FORMAT:
        raise ValueError(f"{path} holds no {ROWS_FORMAT} rows ({rec.get('format')!r})")
    return rec


# ------------------------------------------------------------------ consolidation
def _same(recs, key, what):
    first = json.dumps(recs[0][key], sort_keys=True, default=float)
    for r in recs[1:]:
        if json.dumps(r[key], sort_keys=True, default=float) != first:
            raise ValueError(f"the blocks' rows disagree on {what}: they are not rows of one layout")


def _partition_codes(recs, floor, eps):
    """Per partition the widths: the rounding budget ``eps^2`` less the largest source quantisation over the blocks,
    split equally over the segments that carry a phase, each segment's step ``sqrt(12 R / n) / reach_s``, the bits and
    scales against the blocks' largest values, and the fields' places."""
    out = {}
    for name, p0 in recs[0]["partitions"].items():
        reach = np.asarray(p0["reach"], np.float64)
        v_src = max(float(r["partitions"][name]["quantisation"]) for r in recs)
        R = eps ** 2 - v_src
        if R <= 0.0:
            raise ValueError(f"the pack's own band quantisation on partition {name!r} is {np.sqrt(v_src):.3g} rad, already "
                             f"the floor's budget of {eps:.3g} rad: write the segment rows from the walk")
        live = reach > 0
        step = np.full(reach.shape, np.inf); step[live] = np.sqrt(12.0 * R / live.sum()) / reach[live]
        amax = np.max([np.asarray(r["partitions"][name]["amax"], np.float64) for r in recs], axis=0)   # (n_seg, 3)
        bits, scale = field_code(amax, step[:, None])
        layout, n_words = word_layout(bits.ravel())
        realised = float(np.sum(reach ** 2 * np.where(live, scale.max(axis=1), 0.0) ** 2) / 12.0)
        segs = []
        for s, (t, e, sh) in enumerate(zip(p0["segments"], p0["gate"], _term_shapes(p0, name))):
            pl = layout[3 * s:3 * s + 3]
            segs.append(dict(t=list(t), gate=int(e), shape=sh, q_max=float(reach[s]), floor_target=float(floor), step=float(step[s]),
                             bits=[int(b) for b in bits[s]], offsets=[int(o) for _, o, _ in pl], word=[int(w) for w, _, _ in pl],
                             scale=[float(x) for x in scale[s]]))
        out[name] = dict(group=p0["group"], segments=segs, n_words=int(n_words), column=f"p_{name}.npy",
                         rounding=dict(budget=float(R), variance=realised, source_quantisation=float(v_src), eps=float(eps)))
    return out


def _tier_codes(recs, g, eps, env):
    """The group's tier word layout: the contact at ``sqrt(12) eps / (rho2/D)_max``, the basis's components (no
    ``iso_P_zz``) each at ``sqrt(12 / n) eps / (B0 chi_max alpha_c)``, a field that rounds to zero on every row of every
    block not stored."""
    g0 = recs[0]["groups"][g]
    fields = []
    if g0.get("contact"):
        a = max(float(r["groups"][g]["contact_amax"]) for r in recs)
        fields.append(dict(name="contact", amax=a, step=float(np.sqrt(12.0) * eps / env["rho2_over_D_max"]),
                           q_max=env["rho2_over_D_max"]))
    field_rec = None
    if g0.get("field"):
        names = list(g0["field"]["channels"])                         # the basis's components: no iso_P_zz (RPK.md 5.3)
        amax = np.max([np.asarray(r["groups"][g]["field"]["amax"], np.float64) for r in recs], axis=0)
        n = len(names)
        for nm in names:
            alpha = 2.0 if nm == "iso_local" else 1.0                  # iso_P_zz = 3 iso_local - xx - yy enters iso too
            fields.append(dict(name=nm, amax=float(amax[names.index(nm)]), q_max=env["B0_chi_max_T"] * alpha,
                               step=float(np.sqrt(12.0 / n) * eps / (env["B0_chi_max_T"] * alpha))))
        field_rec = dict(channels=names)
    bits, scale = field_code([f["amax"] for f in fields], [f["step"] for f in fields])
    if field_rec is not None:
        field_rec["stored"] = [f["name"] for f, b in zip(fields, bits) if b and f["name"] != "contact"]
        field_rec["zero"] = [f["name"] for f, b in zip(fields, bits) if not b and f["name"] != "contact"]
    keep = [i for i, b in enumerate(bits) if b]
    if not keep:
        return None, field_rec
    fields = [fields[i] for i in keep]; bits = bits[keep]; scale = scale[keep]
    layout, n_words = word_layout(bits)
    for f, b, s, (w, off, _) in zip(fields, bits, scale, layout):
        f.update(bits=int(b), scale=float(s), word=int(w), offset=int(off))
    return dict(fields=fields, n_words=int(n_words)), field_rec


def _source_tier_terms(recs, g, eps):
    """``{contact, field}``: the source's own quantisation of group ``g``'s tiers (rad^2, the largest over the blocks; 0
    from a walk). A floor target a caller STATED (``segment_rows(floor_target=)``) whose budget ``eps^2`` is below it is
    refused by name, since no width meets it; the pack's own declared target is taken as it is and the certificate
    states the error."""
    out = {k: max(float((r["groups"][g].get("quantisation") or {}).get(k, 0.0)) for r in recs) for k in ("contact", "field")}
    stated = any(r["source"].get("floor_target_stated") for r in recs)
    for k, q in out.items():
        if stated and q >= eps ** 2:
            raise ValueError(f"the pack's own quantisation of group {g}'s {k} is {np.sqrt(q):.3g} rad, above the floor "
                             f"target's budget of {eps:.3g} rad: no width meets that target from this pack -- ask for a "
                             "larger floor target, or write the segment rows from the walk")
    return out


def _tier_rounding(grp, env, background_amplitude):
    """``{contact, field, background}``: the realised rounding variance (rad^2) of a group's tier fields at their scales
    -- the contact's ``((rho2/D)_max scale)^2 / 12``, the components' ``sum (B0 chi_max alpha_c scale_c)^2 / 12``, the
    background's ``(g0_max max scale)^2 / 12``."""
    out = dict(contact=0.0, field=0.0, background=0.0)
    for f in ((grp.get("tier_words") or {}).get("fields") or []):
        v = (float(f["q_max"]) * float(f["scale"])) ** 2 / 12.0
        out["contact" if f["name"] == "contact" else "field"] += v
    bg = grp.get("background")
    if bg and bg.get("n_words"):
        out["background"] = (float(background_amplitude) * max(s for s, b in zip(bg["scale"], bg["bits"]) if b)) ** 2 / 12.0 \
            if any(bg["bits"]) else 0.0
    return out


def _certificate(r0, recs, partitions, groups, env, background_amplitude, eps):
    """The layout's certificate: the source's terms, per partition its rounding and its source's band quantisation, per
    group its tiers' rounding and its source's tier quantisation (a pack's: :func:`tier_quantisation_terms`), all
    rad^2, and ``err_max``, the worst image's signal error over the partitions with their group's tiers, half the summed
    phase variances (the criterion of every other term, as a field read's ``v / 2``); beside it ``err_rounding``, the
    same without the source's terms, which the floor target holds (``eps^2 / 2`` per term). From a pack ``err_max`` may
    exceed the floor target's: the same widths, the pack's own error added."""
    tiers = {g: dict(rounding=_tier_rounding(grp, env, background_amplitude), source=grp["source_quantisation"])
             for g, grp in groups.items()}
    err, err_r = 0.0, 0.0
    for name, p in partitions.items():
        t = tiers[p["group"]]
        rounding = p["rounding"]["variance"] + sum(t["rounding"].values())
        source = p["rounding"]["source_quantisation"] + sum(t["source"].values())
        err, err_r = max(err, 0.5 * (rounding + source)), max(err_r, 0.5 * rounding)
    return dict(source=r0["source"]["kind"], terms=r0["certificate"]["terms"],
                source_terms_per_block=[r["certificate"] for r in recs],
                rounding={name: p["rounding"] for name, p in partitions.items()}, tiers=tiers,
                err_max=float(err), err_rounding=float(err_r), floor_term=float(0.5 * eps ** 2))


def _background_code(recs, g, eps, amplitude):
    a = np.max([np.asarray(r["groups"][g]["background_amax"], np.float64) for r in recs], axis=0)
    step = float(np.sqrt(12.0) * eps / float(amplitude))
    bits, scale = field_code(a, step)
    layout, n_words = word_layout(bits)
    return dict(step=step, q_max=float(amplitude), bits=[int(b) for b in bits], scale=[float(x) for x in scale],
                word=[int(w) for w, _, _ in layout], offsets=[int(o) for _, o, _ in layout], n_words=int(n_words))


def _layout_of(rec):
    """``(layout, scale)`` of a partition's or a background's record: the fields' places and scales in field order."""
    if "segments" in rec:
        items = [(w, o, b, s) for seg in rec["segments"] for w, o, b, s in zip(seg["word"], seg["offsets"], seg["bits"], seg["scale"])]
    elif "fields" in rec:
        items = [(f["word"], f["offset"], f["bits"], f["scale"]) for f in rec["fields"]]
    else:
        items = list(zip(rec["word"], rec["offsets"], rec["bits"], rec["scale"]))
    return tuple((int(w), int(o) if b else 0, int(b)) if b else (0, 0, 0) for w, o, b, _ in items), \
        np.asarray([s for *_, s in items], np.float32)


def write_time_segment_layout(blocks, out_dir, *, progress=None):
    """The layout of the segment rows in the directories ``blocks`` (:func:`segment_rows`, one per block), written to
    ``out_dir``: the blocks' declarations checked equal (the partitions, the groups, the grid, the source's kind, the
    envelope's ``b_max`` and ``G_max``), every width derived from the blocks' largest values at the smallest floor
    target any block declares and the largest ``rho2 / D`` and ``B0 chi`` any states, the rows
    placed into tiles (each voxel's rows alternating between its split halves in block and row order) and packed, in two
    passes over the rows that hold one block at a time. ``progress(rows, seconds)`` is called after every block.
    Returns the manifest."""
    from ..phantom.grid import Grid
    blocks = [str(b) for b in ([blocks] if isinstance(blocks, (str, os.PathLike)) else blocks)]
    recs = [read_rows(b) for b in blocks]
    for key, what in (("grid", "the grid"), ("background", "the background")):
        _same(recs, key, what)
    for r in recs:
        for p_name, p in r["partitions"].items():
            _term_shapes(p, p_name)
    for r in recs[1:]:
        for k in ("kind", "background_amplitude"):
            if json.dumps(r["source"][k], sort_keys=True, default=float) != json.dumps(recs[0]["source"][k], sort_keys=True, default=float):
                raise ValueError(f"the blocks' rows disagree on the source's {k}: they are not rows of one layout")
        for k in ("b_max", "G_max", "bound"):
            if r["source"]["envelope"].get(k) != recs[0]["source"]["envelope"].get(k):
                raise ValueError(f"the blocks' rows disagree on the envelope's {k}: they are not rows of one layout")
        if {n: (p["segments"], p["gate"], _term_shapes(p, n), p["group"]) for n, p in r["partitions"].items()} != \
                {n: (p["segments"], p["gate"], _term_shapes(p, n), p["group"]) for n, p in recs[0]["partitions"].items()}:
            raise ValueError("the blocks' rows disagree on the partitions: they are not rows of one layout")
        if {g: (x["rf"], x["dt"], x["n_t"], x["readout"], x["contact"], bool(x["field"])) for g, x in r["groups"].items()} != \
                {g: (x["rf"], x["dt"], x["n_t"], x["readout"], x["contact"], bool(x["field"])) for g, x in recs[0]["groups"].items()}:
            raise ValueError("the blocks' rows disagree on the sequence groups: they are not rows of one layout")
    r0 = recs[0]; src = r0["source"]
    # the tiers' widths at the largest rho2 / D and B0 chi any block's source states (each pack's field read is its own)
    env = dict(src["envelope"], **{k: max(float(r["source"]["envelope"][k]) for r in recs) for k in ("rho2_over_D_max", "B0_chi_max_T", "B0_max_T")})
    grid = Grid.from_meta(r0["grid"]); n_vox = int(np.prod(grid.shape))
    floor = min(float(r["source"]["floor_target"]) for r in recs)       # the most demanding block's floor holds for all
    eps = rounding_eps(floor)
    partitions = _partition_codes(recs, floor, eps)
    groups = {}
    for g, g0 in r0["groups"].items():
        tw, field_rec = _tier_codes(recs, g, eps, env) if r0["tiers"] else (None, None)
        grp = {k: g0[k] for k in ("partitions", "rf", "dt", "n_t", "readout", "pathway", "tau_t2", "tau_t1")}
        grp.update(contact=bool(g0.get("contact")), field=field_rec, tier_words=tw,
                   column=(f"t_{g}.npy" if tw is not None else None),
                   source_quantisation=_source_tier_terms(recs, g, eps))
        if g0.get("background"):
            grp["background"] = dict(_background_code(recs, g, eps, src["background_amplitude"]), column=f"bg_{g}.npy")
        groups[g] = grp
    # pass 1: the rows per voxel, the tiling
    counts = np.zeros(n_vox, np.int64)
    for b in blocks:
        counts += np.bincount(np.load(os.path.join(b, "voxel.npy"), mmap_mode="r"), minlength=n_vox)
    if int(counts.sum()) != sum(int(r["n_rows"]) for r in recs):
        raise RuntimeError("the blocks' voxel columns do not hold their declared rows")
    k0 = -(-((counts + 1) // 2) // TILE); k1 = -(-(counts // 2) // TILE)
    start = np.concatenate([[0], np.cumsum(k0 + k1)])[:-1]
    t_half = np.stack([start, start + k0], axis=1)
    tiles = np.repeat(np.stack([2 * np.arange(n_vox), 2 * np.arange(n_vox) + 1], 1).ravel(), np.stack([k0, k1], 1).ravel()).astype(np.int32)
    n_tiles = int(len(tiles))
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "tiles.npy"), tiles)
    mm = lambda name, dtype, extra=(): np.lib.format.open_memmap(os.path.join(out_dir, name), mode="w+", dtype=dtype, shape=(n_tiles, TILE) + extra)
    out = {"w": (mm("w.npy", np.float32), None)}
    for name, p in partitions.items():                       # a word array of no words (every field zero) is not stored
        if p["n_words"]:
            out[f"p_{name}"] = (mm(p["column"], np.uint32, (p["n_words"],)), p)
        else:
            p["column"] = None
    if r0["tiers"]:
        out["pool"] = (mm("pool.npy", np.uint8), None)
    for g, grp in groups.items():
        if grp["tier_words"] is not None:
            out[f"t_{g}"] = (mm(grp["column"], np.uint32, (grp["tier_words"]["n_words"],)), grp)
        if grp.get("background"):
            if grp["background"]["n_words"]:
                out[f"bg_{g}"] = (mm(grp["background"]["column"], np.uint32, (grp["background"]["n_words"],)), grp["background"])
            else:
                grp["background"]["column"] = None
    # pass 2: each block's rows packed into their places
    seen = np.zeros(n_vox, np.int64); t0 = time.time(); done = 0
    for b, rec in zip(blocks, recs):
        vox = np.load(os.path.join(b, "voxel.npy"))
        n = len(vox)
        order = np.argsort(vox, kind="stable"); sv = vox[order]
        first = np.r_[True, sv[1:] != sv[:-1]] if n else np.zeros(0, bool)
        grp_start = np.maximum.accumulate(np.where(first, np.arange(n), 0)) if n else np.zeros(0, np.int64)
        rank = np.empty(n, np.int64); rank[order] = np.arange(n) - grp_start + seen[sv]
        seen += np.bincount(vox, minlength=n_vox)
        half = rank % 2; pos = rank // 2
        flat = (t_half[vox, half] + pos // TILE) * TILE + pos % TILE
        for name, (arr, code) in out.items():
            if name == "w" or name == "pool":
                arr.reshape(-1)[flat] = np.load(os.path.join(b, f"{name}.npy"))
            elif name.startswith("p_"):
                layout, scale = _layout_of(code)
                v = np.load(os.path.join(b, f"{name}.npy")).reshape(n, -1)
                arr.reshape(-1, code["n_words"])[flat] = pack_words(v, scale, layout, code["n_words"])
            elif name.startswith("t_"):
                g = name[2:]
                layout, scale = _layout_of(code["tier_words"])
                fields = code["tier_words"]["fields"]
                cols = []
                fl = np.load(os.path.join(b, f"field_{g}.npy")) if code["field"] else None
                for f in fields:
                    if f["name"] == "contact":
                        cols.append(np.load(os.path.join(b, f"contact_{g}.npy")))
                    else:
                        cols.append(fl[:, code["field"]["channels"].index(f["name"])])
                arr.reshape(-1, code["tier_words"]["n_words"])[flat] = pack_words(np.stack(cols, axis=1), scale, layout,
                                                                                  code["tier_words"]["n_words"])
            else:
                layout, scale = _layout_of(code)
                arr.reshape(-1, code["n_words"])[flat] = pack_words(np.load(os.path.join(b, f"{name}.npy")), scale, layout, code["n_words"])
        done += n
        if progress:
            progress(done, time.time() - t0)
    for arr, _ in out.values():
        arr.flush()
    tiers = None
    if r0["tiers"]:
        pools = [int(r["tiers"]["pools"]) for r in recs if r["tiers"].get("pools") is not None]   # the pool ids each block's rows address
        tiers = dict(r0["tiers"], relaxation=bool(pools), pools=max(pools) if pools else None, pool_column="pool.npy")
    manifest = dict(
        format=FORMAT, n_rows=int(counts.sum()), n_tiles=n_tiles, tile=TILE, n_voxels=n_vox, grid=r0["grid"],
        source=dict(kind=src["kind"], walk=src["walk"], pack=src["pack"], floor_target=floor, eps=eps, envelope=env,
                    background_amplitude=src["background_amplitude"],
                    blocks=[dict(path=os.path.basename(os.path.normpath(b)), n_rows=int(r["n_rows"]), walk=r["source"]["walk"],
                                 pack=r["source"]["pack"]) for b, r in zip(blocks, recs)]),
        certificate=_certificate(r0, recs, partitions, groups, env, src["background_amplitude"], eps),
        partitions=partitions, groups=groups, tiers=tiers,
        columns=dict(w="w.npy", tiles="tiles.npy"),
        bytes_per_walker=_bytes_per_walker(partitions, groups, tiers),
        code=_code(), created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(out_dir, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=1, default=float)
    return manifest


def _bytes_per_walker(partitions, groups, tiers):
    """The layout's stored bytes per walker by column kind (the weight, the partitions' words, the pool, the tiers'
    words, the background's words), padding not counted."""
    out = dict(w=4, partitions={n: 4 * int(p["n_words"]) for n, p in partitions.items()}, pool=(1 if tiers else 0),
               tiers={g: 4 * int(grp["tier_words"]["n_words"]) for g, grp in groups.items() if grp.get("tier_words")},
               background={g: 4 * int(grp["background"]["n_words"]) for g, grp in groups.items() if grp.get("background")})
    out["total"] = (out["w"] + sum(out["partitions"].values()) + out["pool"] + sum(out["tiers"].values())
                    + sum(out["background"].values()))
    return out


def _field_of_tiers(pack_id, tiers, groups):
    """The susceptibility field of a tier layout: its embedded spec's, with the groups' field columns as the stored
    field channel (:func:`~dmipy_sim.replay.replay.declared_susceptibility_field`)."""
    from .replay import declared_susceptibility_field
    return declared_susceptibility_field(dict(id=pack_id, substrate=tiers.get("substrate")),
                                         has_field=any(grp.get("field") for grp in groups.values()))


def _code():
    """The dmipy-sim that wrote a layout: the package version and its git commit when the tree is a checkout."""
    import subprocess
    from importlib.metadata import version, PackageNotFoundError
    from .. import __file__ as pkg
    root = os.path.dirname(os.path.dirname(os.path.abspath(pkg)))
    try:
        ver = version("dmipy-sim")
    except PackageNotFoundError:
        ver = None
    try:
        r = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        commit = r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return dict(package="dmipy-sim", version=ver, commit=commit)


# ------------------------------------------------------------------ the layout
class TimeSegmentLayout:
    """The time-segment layout at ``path``, tiles memory-mapped from disk: ``manifest``, ``grid``, ``partitions`` (the
    stored :class:`Partition` objects by name), and :meth:`image` -- a whole grid under any sequence whose effective gradient
    is a sum of the shapes of a stored partition's terms (piecewise constant on its segments, a trapezoid, a cosine, a
    tensor encoding), on the device. Open a Hub layout with :meth:`open`."""

    TILES = 4096                                                        # tiles per device chunk (2^19 rows)
    MEAS = 64                                                           # measurements padded to a multiple

    def __init__(self, path):
        from ..phantom.grid import Grid
        self.path = path
        self.manifest = read_manifest(path)
        self.grid = Grid.from_meta(self.manifest["grid"])
        self.n_rows = int(self.manifest["n_rows"]); self.n_tiles = int(self.manifest["n_tiles"]); self.tile = int(self.manifest["tile"])
        self.n_vox = int(np.prod(self.grid.shape))
        self.partitions = {n: Partition(n, tuple(tuple(s["t"]) for s in p["segments"]), tuple(int(s["gate"]) for s in p["segments"]),
                                        tuple(Shape.from_meta(s.get("shape"), f"term {i} of partition {n!r} in the manifest")
                                              for i, s in enumerate(p["segments"])))
                           for n, p in self.manifest["partitions"].items()}
        self.tiers = self.manifest.get("tiers")
        self.susceptibility_field = (None if not self.tiers else
                                     _field_of_tiers((self.manifest["source"].get("pack") or {}).get("id"), self.tiers,
                                                     self.manifest["groups"]))
        self._m = {}; self._device = {}; self._host = {}

    @staticmethod
    def open(uri, *, partitions=None, revision=None, workers=8):
        """The layout at ``uri``: a directory, or ``hf://owner/name/prefix`` fetched into the Hub cache (the manifest, the
        shared columns, the tiers and the words of ``partitions``, every partition when None) at ``revision``."""
        if not uri.startswith("hf://"):
            return TimeSegmentLayout(uri)
        from huggingface_hub import snapshot_download, hf_hub_download
        owner, name, prefix = uri[5:].split("/", 2); repo = f"{owner}/{name}"
        mf = hf_hub_download(repo, f"{prefix}/{MANIFEST}", repo_type="dataset", revision=revision)
        with open(mf) as f:
            manifest = json.load(f)
        _check_format(manifest, uri)
        local = snapshot_download(repo, repo_type="dataset", revision=revision, max_workers=workers,
                                  allow_patterns=[f"{prefix}/{f}" for f in TimeSegmentLayout.files(manifest, partitions)])
        return TimeSegmentLayout(os.path.join(local, prefix))

    @staticmethod
    def columns(manifest):
        """Every column of a layout by name -> file: ``w``, ``tiles``, ``p_<partition>``, with tiers ``pool`` and
        ``t_<g>`` per sequence group, with the background ``bg_<g>``."""
        cols = dict(manifest["columns"])
        cols.update({f"p_{n}": p["column"] for n, p in manifest["partitions"].items() if p.get("column")})
        if manifest.get("tiers"):
            cols["pool"] = manifest["tiers"]["pool_column"]
        for g, grp in manifest["groups"].items():
            if grp.get("column"):
                cols[f"t_{g}"] = grp["column"]
            if (grp.get("background") or {}).get("column"):
                cols[f"bg_{g}"] = grp["background"]["column"]
        return cols

    @staticmethod
    def files(manifest, partitions=None):
        """The files a layout needs for ``partitions`` (every partition when None): the manifest, the shared columns, the
        tier and background columns and the words of those partitions."""
        keep = set(manifest["partitions"]) if partitions is None else set(partitions)
        return [MANIFEST] + [f for name, f in TimeSegmentLayout.columns(manifest).items()
                             if not name.startswith("p_") or name[2:] in keep]

    def _column(self, name):
        if name not in self._m:
            self._m[name] = np.load(os.path.join(self.path, self.columns(self.manifest)[name]), mmap_mode="r")
        return self._m[name]

    def _group(self, partition):
        """``(g, group record)`` of ``partition``'s sequence group."""
        g = self.manifest["partitions"][partition]["group"]
        return g, self.manifest["groups"][g]

    def _code(self, partition):
        """``(scale, layout, n_words)`` of a partition's words."""
        rec = self.manifest["partitions"][partition]
        layout, scale = _layout_of(rec)
        return scale, layout, int(rec["n_words"])

    def _tier_code(self, g):
        """``(contact field's (layout, scale) or None, channel fields)`` of group ``g``'s tier words."""
        tw = self.manifest["groups"][g].get("tier_words")
        if tw is None:
            return None
        layout, scale = _layout_of(tw)
        return layout, scale, [f["name"] for f in tw["fields"]]

    def fit(self, sequence, partition=None):
        """``(partition, levels)``: the stored partition ``sequence`` fits (``partition`` names one, else the first that
        fits among those whose group plays the sequence's gate, then among the rest) and its rows' effective gradients
        on the segments ``(n_meas, n_seg, 3)``; refused by name when none fits."""
        if partition is None:
            key = _group_key(_sequence(sequence))
            own = [n for n in self.partitions if self._gate_key(n) == key]
            names = own + [n for n in self.partitions if n not in own]
        else:
            names = [partition]
        reasons = []
        for name in names:
            if name not in self.partitions:
                raise KeyError(f"no partition {name!r} in this layout; it holds {list(self.partitions)}")
            try:
                return name, self.partitions[name].levels(sequence)
            except ValueError as e:
                reasons.append(f"{name}: {e}")
        raise ValueError("the sequence fits no stored partition -- " + "; ".join(reasons))

    def _certified(self, partition, L):
        """Refuse levels beyond the envelope the partition's widths are derived at: a segment above ``G_max`` or an
        encoding above ``b_max``; for a scanner class's layout (``b_max`` None) a level above ``G_max`` on any axis."""
        env = self.manifest["source"]["envelope"]
        B = self.partitions[partition].b_matrix()
        b = np.einsum("msc,st,mtc->m", L, B, L)
        per_axis = env.get("b_max") is None
        g = (np.abs(L).max(initial=0.0) if per_axis else np.linalg.norm(L, axis=2).max(initial=0.0))
        if g > env["G_max"] * (1 + 1e-6):
            raise ValueError(f"the sequence plays {g:.4g} T/m{' on one axis' if per_axis else ''} on partition {partition!r} "
                             f"and its vectors are derived for at most G_max = {env['G_max']:.4g} T/m")
        if b.size and env.get("b_max") is not None and b.max() > env["b_max"] * (1 + 1e-6):
            raise ValueError(f"the sequence reaches b = {b.max():.4g} s/m^2 on partition {partition!r} and its vectors are "
                             f"derived for at most b_max = {env['b_max']:.4g} s/m^2 (the source's declared envelope)")
        return b

    def vectors(self, partition):
        """``(n_tiles, tile, n_seg, 3)`` float32: every row's segment vectors (0 on padding), decoded from the words."""
        scale, layout, _ = self._code(partition)
        n_seg = len(self.partitions[partition].segments)
        words = (self._column(f"p_{partition}") if f"p_{partition}" in self.columns(self.manifest)
                 else np.zeros((self.n_tiles, self.tile, 1), np.uint32))
        v = unpack_words(words, scale, layout)
        return v.reshape(v.shape[:2] + (n_seg, 3))

    def tier_columns(self, g):
        """``{name: (n_tiles, tile) float32}``: group ``g``'s decoded tier fields (the contact and the stored field
        channels)."""
        code = self._tier_code(g)
        if code is None:
            return {}
        layout, scale, names = code
        v = unpack_words(self._column(f"t_{g}"), scale, layout)
        return {nm: v[..., i] for i, nm in enumerate(names)}

    def terms(self, partition, tissue=None, scanner=None):
        """The reduction terms of a (tissue, scanner) pair on ``partition``'s gate -- exactly
        :meth:`~dmipy_sim.replay.study.Primitives.reduction_terms` on a descriptor of what the layout carries: ``None``
        for the bare image (no tissue, no scanner), else ``(logw_pool (n_pools,), rho2_over_D, a_iso, a_aniso,
        amplitude)``. A tier the pair needs that the layout lacks is refused by name, and a relaxivity beyond the
        ``rho2 / D`` the contact's widths are derived at (the source's declared envelope) likewise."""
        if tissue is None and scanner is None:
            return None
        if not self.tiers:
            raise ValueError("this layout holds the bare diffusion phase only (written without tiers=True); a tissue "
                             "or a scanner needs the tiers")
        from .study import Primitives
        from ..spec.substrate import SubstrateSpec
        from ..spec.tissue import _by_pool_id
        t = self.tiers; g, grp = self._group(partition)
        env = self.manifest["source"]["envelope"]
        spec = SubstrateSpec.from_dict(t["substrate"]) if t.get("substrate") else None

        def by_pool(values, what):
            if values is None:
                return None
            if spec is None:
                raise ValueError(f"{what} was given by pool but the layout embeds no substrate spec")
            if not isinstance(values, dict):
                raise ValueError(f"{what} is {{pool name: seconds}} over every pool of the spec {[p.name for p in spec.pools]}")
            return [float(v) for v in _by_pool_id(spec, values, what)]
        n_pools = int(t["pools"]) if t.get("relaxation") else None
        has_field = bool(grp.get("field"))
        desc = Primitives(w=np.ones(1), phi=np.zeros((1, 1)),
                          field_iso=np.zeros(1) if has_field else None, field_aniso=np.zeros(1) if has_field else None,
                          exposure_t2=np.zeros((1, n_pools)) if n_pools else None, exposure_t1=np.zeros((1, n_pools)) if n_pools else None,
                          contact=np.zeros(1) if grp.get("contact") else None, D_walk=t.get("D_walk"), voxel=np.ones(1),
                          pathway=float(grp["pathway"]), by_pool=by_pool, susceptibility_field=self.susceptibility_field,
                          rho2_over_D_max=env["rho2_over_D_max"],
                          field_envelope=(dict(B0_max=env["B0_max_T"], B0_chi_max=env["B0_chi_max_T"]) if has_field else None))
        rt = desc.reduction_terms(tissue, scanner)
        logw = -(float(grp["tau_t2"] or 0.0) * np.asarray(rt["invT2"], np.float64) + float(grp["tau_t1"] or 0.0) * np.asarray(rt["invT1"], np.float64))
        return (logw, float(rt["rho2_over_D"]), float(rt["a_iso"]), float(rt["a_aniso"]), float(rt["amplitude"]))

    def _gate_key(self, partition):
        """The gate ``partition``'s group was read under, spelled as :func:`_group_key` spells a sequence's."""
        _, grp = self._group(partition)
        return json.dumps(dict(dt=float(grp["dt"]), n_t=int(grp["n_t"]), rf=grp["rf"], readout=grp["readout"]), sort_keys=True, default=float)

    def _check_gate(self, partition, sequence):
        """Refuse a tiered image of a sequence whose gate (grid, RF schedule, readout) is not the one ``partition``'s
        tiers were read under."""
        g, grp = self._group(partition)
        if _group_key(_sequence(sequence)) != self._gate_key(partition):
            raise ValueError(f"the tiers of partition {partition!r} were read under group {g}'s gate (its grid, RF schedule "
                             "and readout), and this sequence plays another")

    @functools.cached_property
    def weights(self):
        """The per-voxel weight sums ``(n_vox, 2)`` (the two split halves), the denominator of every image (read once)."""
        w = np.asarray(self._column("w"), np.float64).sum(1); tiles = np.asarray(self._column("tiles"))
        return np.bincount(tiles, w, minlength=2 * self.n_vox).reshape(self.n_vox, 2)

    @functools.cached_property
    def live_voxels(self):
        """The flat indices of the voxels the layout has rows in, in order: the voxels a delivery is computed at."""
        return np.flatnonzero(self.weights.sum(1) > 0)

    # ---- the field direction's contraction
    def _field_spec(self, g, b0_direction, a_iso, a_aniso):
        """What the kernel needs to form group ``g``'s field phase from its tier words for ``b0_direction`` (a unit vector
        in the substrate frame), or ``None`` when the group stores no field or the pair's field scales are both zero:
        ``(layout, scale, coef, n_iso)``, the stored channels of the basis's twelve (RPK.md 5.3: ``iso_local``,
        ``iso_P`` xx, yy, xy, xz, yz, ``aniso_G`` (6)) in that order, their places in the words, their scales and their
        contraction weights in float64 -- :func:`~dmipy_sim.fields.hollow_cylinder.iso_weights` on the six isotropic
        channels (``iso_P_zz`` folded in by the trace identity, never formed), :func:`~dmipy_sim.fields.hollow_cylinder.q_of_H`
        on the six anisotropic ones -- and how many of them are isotropic. A channel that rounds to zero on every row
        (not stored) contributes nothing."""
        from ..fields.hollow_cylinder import iso_weights, q_of_H
        b0 = np.asarray(b0_direction, np.float64)
        if b0.shape != (3,) or abs(np.linalg.norm(b0) - 1.0) > 1e-6:
            raise ValueError("b0_direction is a unit vector")
        f = self.manifest["groups"][g].get("field")
        if not f or not f.get("stored") or (float(a_iso) == 0.0 and float(a_aniso) == 0.0):
            return None
        layout, scale, names = self._tier_code(g)
        weight = dict(zip(f["channels"][:6], iso_weights(b0)))
        weight.update(zip(f["channels"][6:12], q_of_H(b0)))
        iso = [nm for nm in f["channels"][:6] if nm in names]
        aniso = [nm for nm in f["channels"][6:12] if nm in names]
        order = iso + aniso
        return (tuple(layout[names.index(nm)] for nm in order),
                np.asarray([scale[names.index(nm)] for nm in order], np.float64),
                np.asarray([weight[nm] for nm in order], np.float64), len(iso))

    def tier_names(self, partition):
        """The tier columns of ``partition``'s group the kernel reads: the pool and the group's tier words (the contact
        and the field channels, decoded in the kernel)."""
        g, _ = self._group(partition)
        return ("pool", f"t_{g}")

    def release(self, keep=()):
        """Drop the device copies of every partition's words but ``keep``'s (the shared columns stay), on every backend."""
        for name in [n for n in self._device if n[1].startswith("p_") and n[1][2:] not in keep]:
            del self._device[name]

    def preload(self, partitions=None, background=True):
        """The padded host arrays of ``partitions``' words (every partition when None), the shared columns and the tiers
        kept in this process's memory, so that a device copy built later -- in a forked worker of a shared pool, which
        inherits them for free -- costs a host-to-device transfer and not a read of the layout; with ``background`` each
        partition's group's background words too, where the layout carries them."""
        parts = list(self.partitions if partitions is None else partitions)
        names = ["w", "tiles"] + [f"p_{p}" for p in parts]
        if self.tiers:
            for p in parts:
                names += list(self.tier_names(p))
        if background:
            names += [f"bg_{self._group(p)[0]}" for p in parts if self.manifest["groups"][self._group(p)[0]].get("background")]
        for name in dict.fromkeys(names):
            if name not in self._host:
                self._host[name] = self._padded(name)

    @functools.cached_property
    def _compact(self):
        """``(n_vox + 1,)`` int32: each voxel's index into :attr:`live_voxels` (the dump slot ``len(live)`` for a voxel
        without rows): a tile's slot in a delivery."""
        live = self.live_voxels
        compact = np.full(self.n_vox + 1, len(live), np.int32); compact[live] = np.arange(len(live), dtype=np.int32)
        return compact

    @property
    def n_pad(self):
        """The tiles padded to whole chunks of :attr:`TILES`."""
        return -(-self.n_tiles // self.TILES) * self.TILES

    def _host_chunk(self, name, s, e):
        """Tiles ``s:e`` of the column ``name`` on the host, padded past the layout's last tile (padding rows weigh
        nothing and scatter into the dump slot), words as their int32 view: a slice of the preloaded padded copy when
        there is one, else read from the memory-mapped column. ``slot`` is each tile's index into a delivery's live
        voxels."""
        if name in self._host:
            return self._host[name][s:e]
        n, real = e - s, max(0, min(e, self.n_tiles) - s)
        if name == "slot":
            a = np.full(n, len(self.live_voxels), np.int32); a[:real] = self._compact[np.asarray(self._column("tiles")[s:s + real]) // 2]
        elif name == "tiles":
            a = np.full(n, 2 * self.n_vox, np.int32); a[:real] = self._column("tiles")[s:s + real]
        elif name in ("w", "pool"):
            a = np.zeros((n, self.tile), np.uint8 if name == "pool" else np.float32); a[:real] = self._column(name)[s:s + real]
        elif name not in self.columns(self.manifest):           # a word array with no words (every field zero): one zero word
            a = np.zeros((n, self.tile, 1), np.int32)
        else:
            words = self._column(name)
            a = np.zeros((n,) + tuple(words.shape[1:]), np.uint32); a[:real] = words[s:s + real]
            a = a.view(np.int32)
        return a

    def _padded(self, name):
        """The column ``name`` on the host, padded to whole chunks of :attr:`TILES` (:meth:`_host_chunk`)."""
        return self._host[name] if name in self._host else self._host_chunk(name, 0, self.n_pad)

    @staticmethod
    def _put(backend, device):
        """The upload of a host column to the backend's device (torch indices as int64)."""
        if backend == "jax":
            import jax.numpy as jnp
            return lambda a, name: jnp.asarray(a)
        import torch
        return lambda a, name: torch.as_tensor(a.astype(np.int64) if name in ("tiles", "slot") else a, device=device)

    def _resident(self, backend, device, names):
        """The columns ``names`` whole on the backend's device, kept across calls."""
        put = self._put(backend, device)
        out = []
        for name in names:
            key = (backend, name, str(device))
            if key not in self._device:
                self._device[key] = put(self._padded(name), name)
            out.append(self._device[key])
        return tuple(out)

    def _chunks(self, backend, device, resident, names):
        """The columns ``names`` chunk by chunk of :attr:`TILES` tiles, a tuple of device arrays per chunk: slices of
        the device-resident columns (``resident``, :meth:`_resident`), or each chunk's tiles read from the layout (or
        the preloaded host copy), padded and uploaded on its own, so that one chunk of every column is on the device
        at a time and the words are decoded in the kernel step that reads them."""
        T = self.TILES
        if resident:
            cols = self._resident(backend, device, names)
            for s in range(0, self.n_pad, T):
                yield tuple(c[s:s + T] for c in cols)
            return
        put = self._put(backend, device)
        for s in range(0, self.n_pad, T):
            yield tuple(put(self._host_chunk(name, s, s + T), name) for name in names)

    def background_group(self, partition):
        """The sequence group whose background words ``partition`` reads, refused for a layout without them."""
        g, grp = self._group(partition)
        if not grp.get("background"):
            raise ValueError("this layout holds no background vectors: a magnet's own gradient needs them "
                             "(segment_rows(..., background=True))")
        return g

    def tier_maps(self, partition, tissue=None, scanner=None, *, backend="jax", device=None, resident=True,
                  b0_direction=(0.0, 0.0, 1.0)):
        """Per-voxel means over the walkers, at the walk's weights, of what each tier multiplies into :meth:`image` on
        ``partition``'s gate, each on the grid (NaN where the layout has no rows): ``pool`` (the weight fraction per
        pool, ``{name: grid}``), ``contact`` (the gated contact term; None without the contact tier),
        ``contact_weight`` (``exp(rho2_over_D contact)`` at the pair's rho2; None without a tissue), ``phase`` and
        ``phase_std`` (the mean and the spread of the field's phase offset at the pair's field and ``b0_direction``;
        None without a scanner). The tier words are decoded chunk by chunk on the device, as in :meth:`image`
        (``resident`` as there)."""
        if backend not in ("jax", "torch"):
            raise ValueError(f"backend is 'jax' or 'torch', got {backend!r}")
        if not self.tiers:
            raise ValueError("this layout holds the bare diffusion phase only (written without tiers=True)")
        tm = self.terms(partition, tissue, scanner)
        g, grp = self._group(partition)
        pools = [(q["name"], int(q["id"])) for q in self.tiers["substrate"]["pools"]]
        if backend == "torch" and device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        rho_D, a_iso, a_aniso = (0.0, 0.0, 0.0) if tm is None else (float(tm[1]), float(tm[2]), float(tm[3]))
        fspec = self._field_spec(g, b0_direction, a_iso, a_aniso) if tm is not None else None
        code = self._tier_code(g)
        names = [("pool", name, pid) for name, pid in pools]
        if grp.get("contact"):
            names.append(("contact",))
            if tm is not None:
                names.append(("contact_weight",))
        if grp.get("field") and tm is not None:
            names += [("phase",), ("phase_sq",)]
        on_torch = backend == "torch"

        def rows(w, pool, tw, xp):
            out = []
            c = 0.0 * w
            if code is not None and "contact" in code[2]:
                layout, scale, fnames = code
                sc = xp.asarray(scale) if not on_torch else xp.as_tensor(scale, device=tw.device)
                c = _decode(tw, sc, layout)[fnames.index("contact")]
            ph = 0.0 * w
            if fspec is not None:
                ph = _field_offset(tw, fspec, a_iso, a_aniso, w, on_torch)
            for q in names:
                if q[0] == "pool":
                    out.append((pool == q[2]).to(w.dtype) if on_torch else (pool == q[2]).astype(w.dtype))
                elif q[0] == "contact":
                    out.append(c)
                elif q[0] == "contact_weight":
                    out.append(xp.exp(rho_D * c))
                else:
                    out.append(ph if q[0] == "phase" else ph * ph)
            return out

        n_halves = 2 * self.n_vox + 1
        chunks = self._chunks(backend, device, resident, ("w", "tiles") + self.tier_names(partition))
        with _x64(backend):
            if fspec is not None:
                fspec = _device_field_spec(fspec, backend, device)
            acc = (_voxel_sums_torch if on_torch else _voxel_sums_jax)(rows, len(names), chunks, n_halves)
        num = acc[:, :2 * self.n_vox].reshape(len(names), self.n_vox, 2).sum(2)
        den = self.weights.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(den > 0, num / den, np.nan).reshape((len(names),) + tuple(self.grid.shape))
        by = {q[0] if q[0] != "pool" else ("pool", q[1]): m for q, m in zip(names, means)}
        maps = dict(pool={name: by[("pool", name)] for name, _ in pools}, contact=by.get("contact"), contact_weight=by.get("contact_weight"),
                    phase=by.get("phase"), phase_std=None)
        if "phase" in by:
            maps["phase_std"] = np.sqrt(np.maximum(by["phase_sq"] - by["phase"] ** 2, 0.0))
        return maps

    def delivery(self, sequence, scanner, grid=None, *, partition=None, to_scanner=None, nonlinearity=True, background=True,
                 concomitant=True, transmit=True):
        """What ``scanner`` (a :class:`~dmipy_sim.acquisition.scanners.ScannerLimits`) plays at every voxel with rows when
        the layout's grid sits in its bore as ``grid`` says (the layout's own grid, whose isocenter is the grid centre,
        when None; :meth:`~dmipy_sim.phantom.Grid.centred_at` moves it), for ``sequence`` on its stored partition:
        :func:`~dmipy_sim.phantom.bore.delivered_vectors` of each segment's commanded plateau at :attr:`live_voxels`,
        for :meth:`image`'s ``delivered``. The term switches are those of
        :func:`~dmipy_sim.phantom.bore.delivered_gradient` and ``transmit``."""
        from ..phantom.bore import delivered_vectors
        grid = self.grid if grid is None else grid
        if tuple(grid.shape) != tuple(self.grid.shape):
            raise ValueError(f"the grid placed in the bore is the layout's own ({tuple(self.grid.shape)}), got {tuple(grid.shape)}")
        name, L = self.fit(sequence, partition)
        if not self.partitions[name].is_constant:
            raise ValueError(f"partition {name!r} has shaped terms: a machine's delivered gradient (its Maxwell term is "
                             "quadratic in the waveform) is one vector per constant segment, not per shape")
        e = np.asarray(self.partitions[name].gate, np.float64)
        plateau = L * np.where(e == 0.0, 0.0, e)[None, :, None]          # the commanded (physical) gradient: e = +-1
        live = self.live_voxels
        ijk = np.stack(np.unravel_index(live, tuple(self.grid.shape)), axis=1)
        d = delivered_vectors(scanner, grid, plateau, e, voxels=ijk, to_scanner=to_scanner, nonlinearity=nonlinearity,
                              background=background, concomitant=concomitant, transmit=transmit)
        durations = self.partitions[name].durations
        net = np.abs(np.einsum("vmsc,s->vmc", d.q, durations)).max(initial=0.0)
        if net > _RTOL * (float(np.abs(d.q).max(initial=0.0)) or 1.0) * float(durations.sum()):
            raise ValueError("the machine's delivered gradient leaves a net moment at the readout (a Maxwell term "
                             "the encoding does not refocus, as a bipolar pair's does not): a segment vector is the path "
                             "relative to the walker's start, which only a refocused encoding does not see")
        return Delivered(partition=name, voxels=live, levels=L, q=d.q, g0=d.g0, kappa=d.kappa)

    def image(self, sequence, *, partition=None, backend="jax", device=None, resident=True, tissue=None, scanner=None,
              b0_direction=(0.0, 0.0, 1.0), delivered=None):
        """``(S, floor)`` of the grid under ``sequence`` (a :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence`
        whose rows are sums of the shapes of a stored partition's terms; ``partition`` names one, else the first it fits): ``S``
        ``grid.shape + (n_meas,)``, the weighted ensemble magnitude per voxel (NaN where the layout has no rows),
        ``floor`` ``grid.shape`` its split-half floor ``max_m |S_a - S_b| / 2``. A row beyond the envelope the widths are
        derived at (``G_max``, ``b_max``) is refused by name. ``backend`` runs the arithmetic on JAX (its default
        device) or torch (``device``, CUDA when torch sees one, else its CPU). The packed words -- the partition's, the
        tiers', the background's -- reach the device as they are stored and are decoded in the kernel step that reads
        them, a chunk of :attr:`TILES` tiles at a time; ``resident`` keeps every column the image reads whole on the
        device across calls, ``resident=False`` reads and uploads each chunk on its own (one chunk of each column on the
        device at a time, nothing kept), to the same bits. ``tissue`` and ``scanner`` turn the stored tiers into the weights and the field phase of
        :meth:`terms` (the sequence must play the partition's group's gate); both None is the bare image, scaled by the
        sequence's pathway amplitude. ``b0_direction`` is the field's direction in the substrate frame.

        ``delivered`` (:meth:`delivery`, for this sequence) is what a machine plays at each voxel in place of the
        commanded gradient: each row's phase is ``sum_s q_vs . m_s + g0_v . n`` with ``q_vs`` the voxel's delivered
        vector on segment ``s``, ``n`` the row's background vector, and each voxel's signal is multiplied by its pathway
        amplitude at its transmit scale over the nominal one
        (:func:`~dmipy_sim.acquisition.epg.transmit_amplitude`)."""
        from ..acquisition.epg import pathway_weight
        if backend not in ("jax", "torch"):
            raise ValueError(f"backend is 'jax' or 'torch', got {backend!r}")
        name, L = self.fit(sequence, partition)
        self._certified(name, L)
        tm = self.terms(name, tissue, scanner)
        if tm is not None:
            self._check_gate(name, sequence)
        scale, layout, _ = self._code(name)
        n_seg = len(self.partitions[name].segments)
        M = L.shape[0]; M_pad = -(-M // self.MEAS) * self.MEAS
        n_halves = 2 * self.n_vox + 1
        if backend == "torch" and device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        g, grp = self._group(name)
        tier_names = self.tier_names(name) if tm is not None else ()
        tier = None
        if tm is not None:
            contact = None
            t_code = self._tier_code(g)
            if t_code is not None and "contact" in t_code[2]:
                t_layout, t_scale, t_names = t_code
                i = t_names.index("contact")
                contact = ((t_layout[i],), t_scale[i:i + 1])
            fspec = self._field_spec(g, b0_direction, tm[2], tm[3])
            tier = (contact, fspec, np.asarray(tm[0], np.float32), float(tm[1]), float(tm[2]), float(tm[3]), float(tm[4]))
        else:
            self._field_spec(g, b0_direction, 0.0, 0.0)             # the direction checked whatever the tiers
        factor = None; bg = None
        names = (f"p_{name}", "w", "tiles")
        if delivered is not None:
            live = self._check_delivered(delivered, name, L)
            names += ("slot",)
            Q = np.zeros((len(live) + 1, M_pad, n_seg, 3), np.float32); Q[:len(live), :M] = delivered.q
            if delivered.g0 is not None:
                gb = self.background_group(name)
                b_layout, b_scale = _layout_of(self.manifest["groups"][gb]["background"])
                G0 = np.zeros((len(live) + 1, 3), np.float32); G0[:len(live)] = delivered.g0
                bg = (b_layout, b_scale, G0)
            if delivered.kappa is not None:
                from ..acquisition.epg import transmit_amplitude
                factor = np.ones(self.n_vox)
                factor[live] = transmit_amplitude(_sequence(sequence), delivered.kappa) / transmit_amplitude(_sequence(sequence), [1.0])[0]
            levels = Q
        else:
            levels = np.zeros((M_pad, n_seg, 3), np.float32); levels[:M] = L
        names += tuple(tier_names) + ((f"bg_{gb}",) if bg is not None else ())
        chunks = self._chunks(backend, device, resident, names)
        with _x64(backend):
            acc = (_sums_jax if backend == "jax" else _sums_torch)(chunks, scale, layout, levels, delivered is not None, tier, bg,
                                                                   n_halves, backend, device)
        num = acc[:2 * self.n_vox, :M].reshape(self.n_vox, 2, M)
        den = self.weights
        S = np.full((self.n_vox, M), np.nan); floor = np.full(self.n_vox, np.nan)
        any_ = den.sum(1) > 0; both = (den > 0).all(1)
        pw = float(pathway_weight(_sequence(sequence))) if tm is None else 1.0   # the tiered path applied it as amp
        pv = pw if factor is None else pw * factor
        S[any_] = (pv if np.ndim(pv) == 0 else pv[any_, None]) * np.abs(num[any_].sum(1) / den[any_].sum(1)[:, None])
        pb = pv if np.ndim(pv) == 0 else pv[both, None]
        Sa = pb * np.abs(num[both, 0] / den[both, 0][:, None]); Sb = pb * np.abs(num[both, 1] / den[both, 1][:, None])
        floor[both] = 0.5 * np.abs(Sa - Sb).max(1)
        return S.reshape(tuple(self.grid.shape) + (M,)), floor.reshape(self.grid.shape)

    def _check_delivered(self, delivered, partition, L):
        """The live voxels, once ``delivered`` is checked to be this layout's, this partition's and these levels'."""
        live = self.live_voxels
        if delivered.partition != partition:
            raise ValueError(f"the delivery was computed for the partition {delivered.partition!r}, not {partition!r}")
        if not np.array_equal(np.asarray(delivered.voxels), live):
            raise ValueError("the delivery was computed for other voxels than the layout's")
        if delivered.q.shape != (len(live),) + L.shape or not np.allclose(delivered.levels, L, rtol=1e-9, atol=0.0):
            raise ValueError("the delivery was computed for another sequence than the image's")
        if delivered.g0 is not None:
            amax = float(self.manifest["source"]["background_amplitude"])
            if float(np.linalg.norm(delivered.g0, axis=1).max(initial=0.0)) > amax * (1 + 1e-9):
                raise ValueError(f"the magnet's gradient reaches {np.linalg.norm(delivered.g0, axis=1).max():.3g} T/m and the "
                                 f"layout's background vectors are certified to {amax:g} T/m")
        return live


@dataclass(frozen=True)
class Delivered:
    """:meth:`TimeSegmentLayout.delivery`: what a machine plays at each of a layout's voxels with rows (``voxels``, flat
    indices) for a sequence on ``partition`` whose commanded effective levels are ``levels`` ``(n_meas, n_seg, 3)``:
    the delivered effective vector on every segment ``q`` ``(n_voxels, n_meas, n_seg, 3)``, the magnet's own gradient
    ``g0`` ``(n_voxels, 3)`` (None without one) and the transmit scale ``kappa`` ``(n_voxels,)`` (None without one)
    (:class:`~dmipy_sim.phantom.bore.VectorDelivery`)."""
    partition: str
    voxels: np.ndarray
    levels: np.ndarray
    q: np.ndarray
    g0: np.ndarray = None
    kappa: np.ndarray = None


# ------------------------------------------------------------------ the kernels
class _x64:
    """JAX's 64-bit types for the duration of a JAX image (the field channels in float64, the tile sums and the
    accumulator in complex128), the flag restored after; nothing for torch."""

    def __init__(self, backend):
        self.on = backend == "jax"

    def __enter__(self):
        if self.on:
            import jax
            self.was = bool(jax.config.read("jax_enable_x64"))
            jax.config.update("jax_enable_x64", True)
        return self

    def __exit__(self, *exc):
        if self.on:
            import jax
            jax.config.update("jax_enable_x64", self.was)


def _device_field_spec(fspec, backend, device):
    """:meth:`TimeSegmentLayout._field_spec` with its scales and weights on the backend's device (float64)."""
    layout, scale, coef, n_iso = fspec
    if backend == "torch":
        import torch
        return layout, torch.as_tensor(scale, device=device), torch.as_tensor(coef, device=device), n_iso
    import jax.numpy as jnp
    return layout, jnp.asarray(scale, jnp.float64), jnp.asarray(coef, jnp.float64), n_iso


def _field_terms_of_words(tw, fspec, on_torch):
    """``(iso, aniso)`` float64 of every row from its int32 tier words ``tw`` ``(chunk, tile, n_words)``: each stored
    channel's integer shifted out of its word, made float64 and times its scale (exact: a code of at most 24 bits
    times a float32 scale), then weighted by the direction's contraction coefficients and summed in channel order, the
    products written out (a float32 matmul on CUDA is TF32); ``None`` for a term with no stored channel."""
    layout, scale, coef, n_iso = fspec
    iso = aniso = None
    for j, (w, off, b) in enumerate(layout):
        x = tw[..., w]
        k = (x << (WORD_BITS - off - b)) >> (WORD_BITS - b)
        if on_torch:
            import torch
            c = k.to(torch.float64) * scale[j]
        else:
            import jax.numpy as jnp
            c = k.astype(jnp.float64) * scale[j]
        t = c * coef[j]
        if j < n_iso:
            iso = t if iso is None else iso + t
        else:
            aniso = t if aniso is None else aniso + t
    return iso, aniso


def _field_offset(tw, fspec, a_iso, a_aniso, w, on_torch):
    """The field's phase offset of every row, float32: ``a_iso iso + a_aniso aniso`` with the terms of
    :func:`_field_terms_of_words` rounded to float32 first."""
    iso, aniso = _field_terms_of_words(tw, fspec, on_torch)
    f32 = w.dtype
    fi = iso.to(f32) if on_torch and iso is not None else (iso.astype(f32) if iso is not None else 0.0 * w)
    fa = aniso.to(f32) if on_torch and aniso is not None else (aniso.astype(f32) if aniso is not None else 0.0 * w)
    return a_iso * fi + a_aniso * fa


def _phase(comps, levels, per_tile):
    """``sum_{s,c} m_sc L_sc`` ``(chunk, tile, M_pad)`` from the decoded components ``comps`` (segment-major, ``None``
    for a 0-bit field) and the levels: ``(M_pad, n_seg, 3)``, or with ``per_tile`` ``(chunk, M_pad, n_seg, 3)`` (a
    delivery's per voxel); the products written out (a float32 matmul on CUDA is TF32)."""
    ph = None
    for f, c in enumerate(comps):
        if c is None:
            continue
        s, ax = divmod(f, 3)
        term = c[:, :, None] * (levels[:, None, :, s, ax] if per_tile else levels[None, None, :, s, ax])
        ph = term if ph is None else ph + term
    return ph


def _row_terms(ph, w, pool, tw, nb, slot, tier, bg, on_torch):
    """The phase and the weight of each row with the background (``g0_v . n``, the voxel's ``g0`` by ``slot``, ``n``
    decoded from the background words ``nb``) and the tiers (the weight ``w amp exp(logw_pool[pool] + rho_D c)`` with
    the contact ``c`` decoded from the tier words ``tw``, the field offset :func:`_field_offset` from the same words)
    folded in, on JAX or (``on_torch``) torch."""
    if bg is not None:
        b_layout, b_scale, G0 = bg
        g0 = G0[slot]
        off = None
        for c, v in enumerate(_decode(nb, b_scale, b_layout)):
            if v is not None:
                t = v * g0[:, None, c]
                off = t if off is None else off + t
        if off is not None:
            ph = ph + off[:, :, None]
    if tier is None:
        return ph, w
    contact, fspec, logw_pool, rho_D, a_iso, a_aniso, amp = tier
    if fspec is not None:
        ph = ph + _field_offset(tw, fspec, a_iso, a_aniso, w, on_torch)[:, :, None]
    if on_torch:
        import torch
        lw, exp = logw_pool[pool.long()], torch.exp
    else:
        import jax.numpy as jnp
        lw, exp = logw_pool[pool.astype(jnp.int32)], jnp.exp
    if contact is not None:
        lw = lw + rho_D * _decode(tw, contact[1], contact[0])[0]
    return ph, w * amp * exp(lw)


_KERNELS = {}


def _compiled(n_halves, layout, per_tile, tiers, contact, f_static, bg_layout):
    """The JAX chunk kernel: the words decoded, the phase, the tiers and the background folded in, ``w exp(i ph)`` in
    complex64 summed over each tile's rows (a fused reduction), the tile sums scattered into their voxel halves in
    complex128. One per (voxel halves, field layout, delivery, tiers, contact, field channels, background layout)."""
    import jax
    import jax.numpy as jnp
    key = (n_halves, layout, per_tile, tiers, contact, f_static, bg_layout)
    if key not in _KERNELS:
        @jax.jit
        def kernel(words, scale, w, tiles, levels, slot, pool, tw, c_scale, f_scale, f_coef, logw_pool, rho_D, a_iso, a_aniso,
                   amp, nb, b_scale, G0, acc):
            comps = _decode(words, scale, layout)
            lev = levels[slot] if per_tile else levels
            ph = _phase(comps, lev, per_tile)
            if ph is None:
                ph = jnp.zeros(words.shape[:2] + (levels.shape[-3],), jnp.float32)
            fspec = None if f_static is None else (f_static[0], f_scale, f_coef, f_static[1])
            tier = (((contact, c_scale) if contact is not None else None, fspec, logw_pool, rho_D, a_iso, a_aniso, amp)
                    if tiers else None)
            bg = (bg_layout, b_scale, G0) if bg_layout is not None else None
            ph, wf = _row_terms(ph, w, pool, tw, nb, slot, tier, bg, False)
            E = jnp.exp(1j * ph) * wf[:, :, None]
            return acc + jax.ops.segment_sum(E.sum(1).astype(jnp.complex128), tiles, num_segments=n_halves)
        _KERNELS[key] = kernel
    return _KERNELS[key]


def _sums_jax(chunks, scale, layout, levels, per_tile, tier, bg, n_halves, backend=None, device=None):
    """The per-voxel-half sums ``(n_halves, M_pad)`` complex128 on JAX: the compiled kernel over the ``chunks``
    (:meth:`TimeSegmentLayout._chunks`: the partition's words, the weights, the tiles, then the slot with a delivery, the
    pool and the tier words with the tiers, the background words with a background)."""
    import jax.numpy as jnp
    contact = None if tier is None or tier[0] is None else tier[0][0]
    fspec = None if tier is None else tier[1]
    f_static = None if fspec is None else (fspec[0], fspec[3])
    bg_layout = None if bg is None else bg[0]
    kernel = _compiled(n_halves, layout, per_tile, tier is not None, contact, f_static, bg_layout)
    M = levels.shape[-3]
    acc = jnp.zeros((n_halves, M), jnp.complex128)
    z = jnp.zeros(1, jnp.float32); zi = jnp.zeros((1, 1, 1), jnp.int32); z64 = jnp.zeros(1, jnp.float64)
    lev, sc = jnp.asarray(levels), jnp.asarray(scale)
    if tier is not None:
        ct, _, lw, rho_D, a_iso, a_aniso, amp = tier
        c_scale = jnp.asarray(ct[1]) if ct is not None else z
        f_scale, f_coef = (z64, z64) if fspec is None else (jnp.asarray(fspec[1], jnp.float64), jnp.asarray(fspec[2], jnp.float64))
        tconst = (c_scale, f_scale, f_coef, jnp.asarray(lw), jnp.float32(rho_D), jnp.float32(a_iso), jnp.float32(a_aniso), jnp.float32(amp))
    if bg is not None:
        _, b_scale, G0 = bg
        bconst = (jnp.asarray(b_scale), jnp.asarray(G0))
    for cols in chunks:
        words, w, tiles = cols[:3]; k = 3
        slot = cols[k] if per_tile else z
        k += 1 if per_tile else 0
        if tier is not None:
            pool, tw = cols[k:k + 2]; k += 2
            targs = (pool, tw) + tconst
        else:
            targs = (z, zi, z, z64, z64, z, z, z, z, z)
        bargs = (cols[k],) + bconst if bg is not None else (zi, z, z)
        acc = kernel(words, sc, w, tiles, lev, slot, *targs, *bargs, acc)
    return np.asarray(acc)


def _tile_sums_torch(words, scale, layout, w, levels, slot, pool, tw, nb, tier, bg, per_tile):
    """One chunk's tile sums on torch, ``(chunk, M_pad)`` complex128: the decoded words, the phase with the tiers and
    the background, ``w exp(i ph)`` in complex64, the reduce over each tile's rows."""
    import torch
    comps = _decode(words, scale, layout)
    lev = levels[slot.long()] if per_tile else levels
    ph = _phase(comps, lev, per_tile)
    if ph is None:
        ph = torch.zeros(words.shape[:2] + (levels.shape[-3],), dtype=torch.float32, device=words.device)
    ph, wf = _row_terms(ph, w, pool, tw, nb, slot.long() if per_tile else None, tier, bg, True)
    return torch.polar(wf[:, :, None].expand_as(ph), ph).sum(1).to(torch.complex128)


_TORCH_KERNEL = {}


def _torch_kernel():
    """The chunk function fused by ``torch.compile`` when the compiler is available (eager otherwise; measured on the
    L40S, DiSCo's layout, 184 measurements: eager 3.3 s per image, compiled 1.44 s, JAX 0.97 s)."""
    if "fn" not in _TORCH_KERNEL:
        import torch
        try:
            _TORCH_KERNEL["fn"] = torch.compile(_tile_sums_torch, dynamic=False)
        except Exception:                                     # no compiler on this platform: eager
            _TORCH_KERNEL["fn"] = _tile_sums_torch
    return _TORCH_KERNEL["fn"]


def _sums_torch(chunks, scale, layout, levels, per_tile, tier, bg, n_halves, backend=None, device=None):
    """The same sums on torch: the tile sums of every chunk (:func:`_tile_sums_torch`, compiled when possible)
    ``index_add_``-ed into a complex128 accumulator (deterministic under ``torch.use_deterministic_algorithms(True)``)."""
    import torch
    dev = torch.device(device)
    lev = torch.as_tensor(levels, device=dev); sc = torch.as_tensor(scale, device=dev)
    t = None
    if tier is not None:
        ct, fspec, lw, rho_D, a_iso, a_aniso, amp = tier
        ct = None if ct is None else (ct[0], torch.as_tensor(ct[1], device=dev))
        fspec = None if fspec is None else _device_field_spec(fspec, "torch", dev)
        t = (ct, fspec, torch.as_tensor(lw, device=dev), rho_D, a_iso, a_aniso, amp)
    b = None
    if bg is not None:
        b_layout, b_scale, G0 = bg
        b = (b_layout, torch.as_tensor(b_scale, device=dev), torch.as_tensor(G0, device=dev))
    acc = torch.zeros((n_halves, levels.shape[-3]), dtype=torch.complex128, device=dev)
    fn = _torch_kernel()
    with torch.no_grad():
        for cols in chunks:
            words, w, tiles = cols[:3]; k = 3
            slot = cols[k] if per_tile else None
            k += 1 if per_tile else 0
            pool = tw = nb = None
            if tier is not None:
                pool, tw = cols[k:k + 2]; k += 2
            if bg is not None:
                nb = cols[k]
            args = (words, sc, layout, w, lev, slot, pool, tw, nb, t, b, per_tile)
            try:
                ts = fn(*args)
            except Exception:                                  # the compiled path failed at run time: eager, once for all
                _TORCH_KERNEL["fn"] = fn = _tile_sums_torch
                ts = fn(*args)
            acc.index_add_(0, tiles, ts)
    return acc.cpu().numpy()


def _voxel_sums_torch(rows, n_q, chunks, n_halves):
    """The weighted per-voxel-half sums ``(n_q, n_halves)`` float64 on torch of the ``n_q`` per-row quantities
    ``rows(w, pool, tw, torch)`` gives for each chunk ``(w, tiles, pool, tw)`` of ``chunks``."""
    import torch
    acc = None
    with torch.no_grad():
        for w, tiles, pool, tw in chunks:
            if acc is None:
                acc = torch.zeros((n_q, n_halves), dtype=torch.float64, device=w.device)
            ts = torch.stack([(q * w).sum(1) for q in rows(w, pool, tw, torch)]).to(torch.float64)
            acc.index_add_(1, tiles, ts)
    return acc.cpu().numpy()


def _voxel_sums_jax(rows, n_q, chunks, n_halves):
    """:func:`_voxel_sums_torch` on JAX (``segment_sum`` into the voxel halves)."""
    import jax
    import jax.numpy as jnp
    acc = jnp.zeros((n_halves, n_q), jnp.float64)
    for w, tiles, pool, tw in chunks:
        ts = jnp.stack([(q * w).sum(1) for q in rows(w, pool, tw, jnp)], axis=1).astype(jnp.float64)
        acc = acc + jax.ops.segment_sum(ts, tiles, num_segments=n_halves)
    return np.asarray(acc).T
