"""The **time-segment layout**: a walk contracted once against the time segments of its timing classes, so that
every acquisition piecewise constant on a stored partition -- at any amplitude, in any direction on each segment -- is
a few products per walker on the device (dmipy-sim#505, #770: a whole grid replayed while a user waits).

**A partition, not a shape.** A layout stores PARTITIONS (:class:`Partition`): consecutive time segments ``[a_s,
b_s]`` of the walk's clock -- a PGSE timing class's two lobes, a stimulated echo's lobe, mixing time and lobe, a double
diffusion encoding's four lobes -- each with the sign of its class's effective gate on it (the coherence sign times
the transverse gate of the RF schedule: +1, -1, or 0 while the magnetisation is stored). Per walker and segment it holds
the **segment vector** ``m_s = gamma int_{a_s}^{b_s} (r(t) - r(0)) dt`` (rad per T/m): the bands contracted against
the segment's indicator in closed form (:func:`~dmipy_sim.replay.compression.segment_projection`, the projection
:func:`~dmipy_sim.replay.compression.band_projection` sums piece by piece). An acquisition whose effective gradient is
the vector ``G_s`` (T/m) on segment ``s`` and zero outside the segments has the phase ``sum_s G_s . m_s`` per walker:
the band contraction of a replay, the start dropping out because a refocused encoding has no net area. PGSE, PGSTE,
DDE and an unequal pair of one timing class are each a linear combination of the stored vectors formed in the image
kernel; no vector per acquisition is stored. A sequence that is not piecewise constant on a stored partition, plays a
gradient outside its segments or on a segment its gate stores, or leaves a net moment, is refused by name
(:meth:`Partition.levels`).

**Two sources, two certificates.** :func:`segment_rows` takes the walk's FLOAT coefficients (a
:class:`~dmipy_sim.coefficient_walk.CoefficientWalk`, read through the float containers, no quantiser) beside the pack
built from the same walk (``pack=``, its sibling: the floor target and the envelope are the pack's, and the rows'
provenance names the walk and the pack): two artefacts of one walk, each with its own certificate, the segments'
carrying the walk's terms (truncation, quadrature, field read, the accumulator's rounding) and its own rounding, and no
band quantisation. Given a quantised :class:`~dmipy_sim.replay.replay.ReplayPack` instead (``source: "pack"``) it
contracts the pack's bands: the certificate then carries the pack's band quantisation contracted through the partition
(:func:`quantisation_term`), and the vectors are held to what the floor's budget leaves of it, so their words are wider.

**Widths derived per segment.** A rounding ``delta_s`` per component of ``m_s`` is a phase error of variance at most
``|G_s|^2 delta_s^2 / 12``. The envelope bounds the levels: ``|G_s| <= G_max`` and the encoding's b-value ``l^T B l <=
b_max`` with ``B_st = gamma^2 int Q_s Q_t dt`` (``Q_s`` the running integral of the segment's indicator) on the
refocused levels ``sum_s l_s d_s = 0``, so the largest level segment ``s`` carries is ``reach_s = min(G_max, sqrt(b_max
[N (N^T B N)^-1 N^T]_ss))``, ``N`` a basis of the refocused levels (:meth:`Partition.reach`; a PGSE pair's is
``sqrt(b_max / b_unit)``). The worst rounding over the envelope is then at most ``sum_s reach_s^2 delta_s^2 / 12``, held
at the floor's budget ``eps^2`` (:func:`~dmipy_sim.replay.compression.container_eps` at the floor target; from a pack,
``eps^2`` less the pack's quantisation term on the partition) with equal contributions, ``delta_s = sqrt(12 eps^2 / n)
/ reach_s`` over the ``n`` segments that carry a phase: the split that minimises the bits (stationary ``sum_s -3 log
delta_s + lambda sum_s reach_s^2 delta_s^2`` gives ``reach_s delta_s`` equal), where weighting the budget by ``(reach_s
d_s)^2`` would equalise the segments' relative precision instead. A segment the gate stores carries no phase and no
bits. Each component takes ``ceil(log2(a / delta_s)) + 1`` bits against its largest ``|m|`` over the walkers
(:func:`field_code`), a two's-complement bit field; a walker's fields of one partition are placed first-fit into as few
32-bit words as hold them, no field across a word (:func:`word_layout`): a short partition shares one word, a long
one takes two. The manifest records per segment ``{"q_max", "floor_target", "step", "bits", "offsets", "word",
"scale"}`` (``q_max`` the reach, ``word`` each component's word).

**The tiers by the same rule.** Per sequence group (the partitions that share a grid, an RF schedule and a readout: the
gate the tiers are read under) the layout holds each walker's pool (``uint8``), its gated contact and its gated field
channels (``t_<g>.npy``, one word array), and with ``background`` its background vector ``gamma int e(t) r(t) dt`` over
the acquisition (``bg_<g>.npy``; ``e`` the effective gate, what a magnet's constant gradient ``g0`` encodes as ``g0 .
n``). The contact enters as the weight ``exp((rho2 / D) c)``, a rounding ``delta`` a relative weight error ``(rho2 / D)
delta``, held at the envelope's largest ``rho2 / D``; the field as ``B0 (chi_iso A + chi_aniso B)`` with ``A, B`` the
channels contracted with coefficients of modulus at most 1 (2 for ``iso_local`` when ``iso_P_zz`` is implied), held at
the envelope's largest ``B0 chi`` with the budget split equally over the channels; ``iso_P_zz`` is not stored when the
rows' trace identity ``iso_P_xx + iso_P_yy + iso_P_zz = 3 iso_local`` holds within half the step the channels take
without it (the pack checks the same residual; reconstructing the channel then moves no rounded value); the background
at ``background_amplitude``. A field within half its step of zero on every row (a channel the gate refocuses, a
component the substrate's symmetry zeroes) rounds to zero on every row and is not stored. Each tier column is its own term at the floor's budget.

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
_RTOL = 1e-5                                                          # a sequence's gradient is float32
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
    """``(bits, scale)`` of fields whose values reach ``a_max`` at the quantisation ``step`` (arrays of one shape):
    ``ceil(log2(a / step)) + 1`` bits, at least :data:`FIELD_BITS_MIN`, the scale ``a / (2^(bits-1) - 1)``. A field
    within half a step of zero on every row rounds to zero on every row and holds 0 bits (it is not stored), as does a
    field whose step is infinite (it carries nothing). More than :data:`FIELD_BITS_MAX` bits is refused by name."""
    a = np.asarray(a_max, np.float64)
    st = np.broadcast_to(np.asarray(step, np.float64), a.shape)
    live = np.isfinite(st) & (st > 0) & (a > 0.5 * np.where(np.isfinite(st), st, 0.0))
    bits = np.zeros(a.shape, int)
    with np.errstate(divide="ignore", invalid="ignore"):
        need = np.ceil(np.log2(np.where(live, a / np.where(live, st, 1.0), 1.0))) + 1
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
class Partition:
    """Consecutive time segments of the walk's clock, ``segments = ((a, b), ...)`` seconds in time order and disjoint,
    with ``gate`` the sign of the effective gate on each (+1, -1, or 0 while the magnetisation is stored, where the
    segment carries no phase); ``template`` the sequence the gate was read from (its grid, RF schedule and readout are
    the tiers' gate). :meth:`of` reads one from a sequence."""
    name: str
    segments: tuple
    gate: tuple
    template: object = field(default=None, compare=False, repr=False)

    @classmethod
    def of(cls, name, sequence, segments=None):
        """The partition of ``sequence``: its lobes (the maximal runs of one non-zero effective gradient), or the stated
        ``segments`` (``[(a, b), ...]`` s), each with the effective gate's sign on it; a segment the gate changes sign
        inside (a refocusing pulse) is refused by name."""
        seq = _sequence(sequence)
        if segments is None:
            segments = _lobes(np.asarray(seq.G_eff, np.float64), seq.dt)
        segs = tuple((float(a), float(b)) for a, b in segments)
        if not segs:
            raise ValueError(f"partition {name!r} has no segment: give segments= or a sequence with a gradient")
        for s, (a, b) in enumerate(segs):
            if not 0.0 <= a < b:
                raise ValueError(f"segment {s} of partition {name!r} is [{a:.6g}, {b:.6g}] s: a segment is 0 <= a < b")
            if s and a < segs[s - 1][1] - 1e-12:
                raise ValueError(f"the segments of partition {name!r} are in time order and disjoint")
        if segs[-1][1] > seq.n_t * seq.dt * (1.0 + 1e-9):
            raise ValueError(f"partition {name!r} ends at {segs[-1][1]:.6g} s, beyond its sequence's grid")
        e = np.asarray(seq.effective_gate, np.float64)
        gate = []
        for s, (a, b) in enumerate(segs):
            i0 = int(np.floor(a / seq.dt + 1e-6)); i1 = max(int(np.ceil(b / seq.dt - 1e-6)), i0 + 1)
            vals = np.unique(e[i0:i1])
            if len(vals) != 1:
                raise ValueError(f"the effective gate changes sign inside segment {s} [{a:.6g}, {b:.6g}] s of partition "
                                 f"{name!r} (a pulse acts there): split the segment at the pulse")
            gate.append(int(np.rint(vals[0])))
        return cls(str(name), segs, tuple(gate), seq)

    @property
    def durations(self):
        return np.array([b - a for a, b in self.segments], np.float64)

    @property
    def carrying(self):
        """Per segment whether its gate lets a phase accrue there (``gate != 0``)."""
        return np.array([g != 0 for g in self.gate], bool)

    def b_matrix(self):
        """``(n_seg, n_seg)``: ``B_st = gamma^2 int Q_s Q_t dt`` with ``Q_s(t)`` the running integral of segment ``s``'s
        indicator, so that an effective encoding with levels ``l`` on the segments has ``b = l^T B l`` (s/m^2) when it is
        refocused; exact (each ``Q_s`` is piecewise linear)."""
        from ..constants import GAMMA
        S = np.asarray(self.segments, np.float64)
        pts = np.unique(np.concatenate([[0.0], S.ravel()]))
        Q = np.minimum(np.clip(pts[None, :] - S[:, :1], 0.0, None), (S[:, 1] - S[:, 0])[:, None])
        L = np.diff(pts); qa, qb = Q[:, :-1], Q[:, 1:]
        B = (L[None, None, :] * (2.0 * qa[:, None] * qa[None] + qa[:, None] * qb[None] + qb[:, None] * qa[None]
                                 + 2.0 * qb[:, None] * qb[None]) / 6.0).sum(-1)
        return GAMMA ** 2 * B

    def refocused_basis(self):
        """``(idx, N)``: the segments that carry a phase and an orthonormal basis ``N`` ``(len(idx), len(idx) - 1)`` of
        the levels on them that leave no net moment (``sum_s l_s d_s = 0``)."""
        idx = np.flatnonzero(self.carrying)
        if idx.size < 2:
            return idx, np.zeros((idx.size, 0))
        _, _, Vt = np.linalg.svd(self.durations[idx][None, :])
        return idx, Vt[1:].T

    def reach(self, G_max, b_max):
        """``(n_seg,)``: the largest level (T/m) an encoding within the envelope puts on each segment, ``min(G_max,
        sqrt(b_max [N (N^T B N)^-1 N^T]_ss))`` over the refocused levels (0 on a segment the gate stores)."""
        out = np.zeros(len(self.segments))
        idx, N = self.refocused_basis()
        if N.shape[1] == 0:
            return out
        M = N.T @ self.b_matrix()[np.ix_(idx, idx)] @ N
        P = N @ np.linalg.solve(M, N.T)
        out[idx] = np.minimum(float(G_max), np.sqrt(float(b_max) * np.clip(np.diag(P), 0.0, None)))
        return out

    def levels(self, sequence):
        """``(n_meas, n_seg, 3)``: each row's effective gradient on each segment (T/m), refused by name when the
        sequence is not constant on a segment, has a segment edge off its grid, plays a gradient outside the segments
        or on a segment its gate stores, or leaves a net moment (a segment vector is relative to the walker's start,
        which only a refocused encoding does not see)."""
        seq = _sequence(sequence)
        G = np.asarray(seq.G_eff, np.float64); dt = float(seq.dt)
        n_meas, n_t, _ = G.shape
        peak = float(np.abs(G).max()) or 1.0; tol = _RTOL * peak
        covered = np.zeros(n_t, bool); L = np.zeros((n_meas, len(self.segments), 3))
        for s, (a, b) in enumerate(self.segments):
            i0, i1 = a / dt, b / dt
            if abs(i0 - round(i0)) > 1e-6 or abs(i1 - round(i1)) > 1e-6 or round(i1) > n_t:
                raise ValueError(f"segment {s} [{a:.6g}, {b:.6g}] s of partition {self.name!r} is not on the sequence's "
                                 f"grid (dt = {dt:.6g} s, {n_t} samples)")
            i0, i1 = int(round(i0)), int(round(i1))
            lev = G[:, i0, :]
            if np.abs(G[:, i0:i1, :] - lev[:, None, :]).max() > tol:
                raise ValueError(f"the sequence's effective gradient is not constant on segment {s} [{a:.6g}, {b:.6g}] s "
                                 f"of partition {self.name!r}: it does not fit the partition")
            L[:, s] = lev; covered[i0:i1] = True
        if np.abs(G[:, ~covered, :]).max(initial=0.0) > tol:
            raise ValueError(f"the sequence plays a gradient outside the segments of partition {self.name!r}: it does not "
                             "fit the partition")
        if np.abs(L[:, ~self.carrying]).max(initial=0.0) > tol:
            raise ValueError(f"the sequence plays an effective gradient on a segment of partition {self.name!r} its gate "
                             "stores: the partition holds no vector there")
        net = np.einsum("msc,s->mc", L, self.durations)
        if np.abs(net).max(initial=0.0) > _RTOL * peak * float(self.durations.sum()):
            raise ValueError(f"the sequence leaves a net moment at the readout on partition {self.name!r} (an unbalanced "
                             "encoding): a segment vector is the path relative to the walker's start, which only a "
                             "refocused encoding does not see")
        return L

    def to_meta(self):
        return dict(segments=[list(s) for s in self.segments], gate=list(self.gate))


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
def segment_vectors(pack, segments):
    """``(n_w, n_seg, 3)`` float64: every walker's segment vectors ``gamma int_seg (r(t) - r(0)) dt`` (rad per T/m) from
    the pack's coefficients, each window's bands against :func:`~dmipy_sim.replay.compression.segment_projection`
    (a window that opens after the last segment adds nothing)."""
    from ..constants import GAMMA
    from .compression import read_position_coeffs, segment_projection
    S = np.asarray(segments, np.float64).reshape(-1, 2)
    m = x0 = None
    for win, t0, T_w in pack._windows():
        if m is not None and t0 >= S[:, 1].max():
            break
        C = read_position_coeffs(win.arrays, dtype=np.float64)
        if x0 is None:
            x0 = C[:, 0, :].copy()
        C[:, 0, :] -= x0
        W = segment_projection(S, T_w, C.shape[1] - 2, t0=t0)
        part = GAMMA * np.einsum("wjc,sj->wsc", C, W)
        m = part if m is None else m + part
    return m


def quantisation_term(pack, partition, b_max):
    """The pack's band quantisation contracted through ``partition`` at its worst over the envelope, rad^2: the bands'
    rounding (a step ``scale_k`` is a variance ``scale_k^2 / 12``, the largest over the scale tables) moves an
    encoding's phase by ``l^T A l`` with ``A_st = gamma^2 sum_k W_sk W_tk scale_k^2 / 12`` per axis, whose largest value
    over the refocused levels with ``l^T B l <= b_max`` is ``b_max`` times the largest generalised eigenvalue of ``(N^T
    A N, N^T B N)``. Zero for the float container."""
    from scipy.linalg import eigh
    from ..constants import GAMMA
    from .compression import position_container, segment_projection
    if position_container(pack.arrays) != "bands":
        return 0.0
    idx, N = partition.refocused_basis()
    if N.shape[1] == 0:
        return 0.0
    S = np.asarray(partition.segments, np.float64)[idx]
    A = None
    for win, t0, T_w in pack._windows():
        if A is not None and t0 >= S[:, 1].max():
            break
        sc = np.asarray(win.arrays["pos_band_scale"], np.float64)
        step2 = (sc.reshape(-1, sc.shape[-2], sc.shape[-1]) ** 2).max(axis=0)          # (n_axes, K)
        W = segment_projection(S, T_w, step2.shape[1], t0=t0)[:, 2:]
        Aw = np.stack([GAMMA ** 2 * (W * step2[c][None, :]) @ W.T / 12.0 for c in range(step2.shape[0])])
        A = Aw if A is None else A + Aw
    M = N.T @ partition.b_matrix()[np.ix_(idx, idx)] @ N
    lam = max(float(eigh(N.T @ A[c] @ N, M, eigvals_only=True).max()) for c in range(A.shape[0]))
    return float(b_max) * max(lam, 0.0)


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
                                                channels=list(names), iso_P_zz="stored", sampling="accumulated_in_walk",
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


def _envelope(meta, spec, D, g_max):
    """What the widths are derived at, from a pack's header: ``b_max`` (``replay_envelope.acquisition.b_max``), ``G_max``
    (``g_max``, else the envelope's, else the pack's scanner class's), the largest ``rho2 / D`` (the contact tier's
    envelope, else the certificate's battery top over ``D``) and the largest ``B0 chi`` (the field read's, else the
    field rule's scale)."""
    from .compression import default_envelope
    from ..engine.accumulate import field_rule_scale
    env = meta.get("replay_envelope") or {}
    acq = env.get("acquisition") or {}
    if acq.get("b_max") is None:
        raise ValueError("the pack declares no acquisition envelope (replay_envelope.acquisition.b_max): a segment's width "
                         "is derived from the largest b the pack is certified for")
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
        if not D:
            raise ValueError("the pack records no diffusivity: the contact's width is derived at the largest rho2 / D")
        rho = max(acq.get("rho2_list") or default_envelope().get("rho2_list") or [1e-4]) / float(D)
    b0chi = (env.get("field_read") or {}).get("B0_chi_max_T")
    if b0chi is None:
        B0, chi = field_rule_scale(spec, acq)
        b0chi = B0 * chi
    return dict(b_max=float(acq["b_max"]), G_max=float(g_max), rho2_over_D_max=float(rho), B0_chi_max_T=float(b0chi))


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


def _group_tiers(view, gate_seq, *, tiers, background, background_amplitude):
    """One sequence group's rows and record from ``view`` under the gate ``gate_seq``: the pool, the gated contact and
    field channels (with ``tiers``) and the background vector (with ``background``)."""
    from .replay import _band_phase, _path_field_channels
    from .study import Acquisition
    from ..acquisition.epg import pathway_weight
    rows, rec = {}, dict(pathway=float(pathway_weight(gate_seq)), tau_t2=None, tau_t1=None, contact=False, field=None,
                         background=False)
    n = view.n_walkers
    if tiers:
        prim = view.walker_primitives(Acquisition(gate_seq))
        if prim.exposure_t2 is not None:
            e2 = np.asarray(prim.exposure_t2); e1 = np.asarray(prim.exposure_t1)
            if not np.allclose(e2.max(1), e2.sum(1)):
                raise ValueError("a walker was in more than one pool before the echo; a layout stores one pool per walker "
                                 "and needs a walk that does not cross")
            for tau, key in ((e2.max(1), "tau_t2"), (e1.max(1), "tau_t1")):
                if tau.size and not np.allclose(tau, tau[0], atol=1e-9):
                    raise ValueError(f"the {key} exposure differs between walkers of one gate; a non-crossing walk has one")
                rec[key] = float(tau[0]) if tau.size else 0.0
            rows["pool"] = np.argmax(e2, axis=1).astype(np.uint8)
            rec["pools"] = int(e2.shape[1])
        else:
            rows["pool"] = np.zeros(n, np.uint8)
        if prim.contact is not None:
            rows["contact"] = np.asarray(prim.contact, np.float64); rec["contact"] = True
        if prim.field_iso is not None:
            P = view._prepare(gate_seq, tissue=None, scanner=None, orientation=None, compartment=None)
            Psi, names = _path_field_channels(P, gate_seq)
            Psi = np.asarray(Psi, np.float64)
            rows["field"] = Psi
            rec["field"] = dict(channels=list(names), amax=np.abs(Psi).max(axis=0).tolist())
            if {"iso_local", "iso_P_xx", "iso_P_yy", "iso_P_zz"} <= set(names):
                i0, ix, iy, iz = (names.index(k) for k in ("iso_local", "iso_P_xx", "iso_P_yy", "iso_P_zz"))
                rec["field"]["trace_abs"] = float(np.abs(Psi[:, ix] + Psi[:, iy] + Psi[:, iz] - 3.0 * Psi[:, i0]).max(initial=0.0))
        rec["D_walk"] = prim.D_walk
    if background:
        P = view._prepare(_background_sequence(gate_seq), tissue=None, scanner=None, orientation=None, compartment=None)
        if np.any(P["voxel"] != 1.0):
            raise ValueError("the background on this gate leaves a net moment at the readout: its effective gate does not "
                             "refocus a constant gradient")
        rows["background"] = np.asarray(_band_phase(P), np.float64)
        rec["background"] = True
    return rows, rec


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
                 background=False, background_amplitude=BACKGROUND_AMPLITUDE, weights=None):
    """One block's segment rows written to ``out_dir``: per partition (``{name: sequence}`` or :class:`Partition`\\ s)
    its segment vectors, per sequence group (the partitions sharing a grid, an RF schedule and a readout) with
    ``tiers`` the pool, gated contact and field channels, with ``background`` the background vector, the weights and
    each row's voxel on ``grid`` (the pack's per-voxel grid when None), every column float in (voxel, pool) order, and
    ``rows.json``: the partitions with their gates and reaches, the groups, the floor target and the envelope, the
    source and its certificate's terms.

    ``source`` is the walk (:class:`~dmipy_sim.coefficient_walk.CoefficientWalk`), read in its float coefficients, with
    ``pack`` the pack built from it (its sibling; refused when its walkers' starts are not the walk's), or a quantised
    :class:`~dmipy_sim.replay.replay.ReplayPack`. The floor target is ``floor_target``, else the pack's declared one, else
    its median per-voxel floor; the envelope the pack's (``g_max`` when it names no scanner class). Returns the record."""
    from ..coefficient_walk import CoefficientWalk
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
    env = _envelope(meta, decl.substrate, decl.diffusivity, g_max)
    floor = _floor_target(meta, floor_target)
    eps = rounding_eps(floor)
    n = view.n_walkers
    os.makedirs(out_dir, exist_ok=True)
    ijk, inside = grid.bin(view.r0)
    if not inside.all():
        raise ValueError(f"{int((~inside).sum())} walkers start outside the grid: a row is placed in its start voxel")
    vox = np.ravel_multi_index(ijk.T, tuple(grid.shape)).astype(np.int64)
    cols, part_rec, group_of = {}, {}, {}
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
        m = segment_vectors(view, p.segments)
        v_src = 0.0 if kind == "walk" else quantisation_term(view, p, env["b_max"])
        cols[f"p_{name}"] = m
        part_rec[name] = dict(p.to_meta(), group=group_of[name], reach=reach.tolist(), quantisation=float(v_src),
                              amax=np.abs(m).max(axis=0).tolist(), column=f"p_{name}.npy")
    group_rec = {}
    pool = np.zeros(n, np.uint8)
    for key, (g, seq, names) in groups.items():
        rows, rec = _group_tiers(view, _gate_sequence(seq), tiers=tiers, background=background,
                                 background_amplitude=background_amplitude)
        rec.update(partitions=names, rf=seq.rf.to_dicts(), dt=float(seq.dt), n_t=int(seq.n_t),
                   readout=[int(i) for i in seq.readout])
        if "pool" in rows:
            pool = rows["pool"]
        if "contact" in rows:
            cols[f"contact_{g}"] = rows["contact"]; rec["contact_amax"] = float(np.abs(rows["contact"]).max(initial=0.0))
        if "field" in rows:
            cols[f"field_{g}"] = rows["field"]
        if "background" in rows:
            cols[f"bg_{g}"] = rows["background"]; rec["background_amax"] = np.abs(rows["background"]).max(axis=0).tolist()
        group_rec[g] = rec
    order = np.lexsort((pool, vox))
    np.save(os.path.join(out_dir, "voxel.npy"), vox[order])
    np.save(os.path.join(out_dir, "w.npy"), np.asarray(view.spin_weights, np.float32)[order])
    if tiers:
        np.save(os.path.join(out_dir, "pool.npy"), pool[order])
    for name, a in cols.items():
        np.save(os.path.join(out_dir, f"{name}.npy"), np.asarray(a, np.float32)[order])
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
                  source=dict(src, floor_target=float(floor), eps=float(eps), envelope=env,
                              background_amplitude=float(background_amplitude)),
                  certificate=certificate, partitions=part_rec, groups=group_rec, tiers=tier_rec, background=bool(background),
                  code=_code(), created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(out_dir, ROWS), "w") as f:
        json.dump(record, f, indent=1, default=float)
    return record


def _check_band(view, partition, env):
    """Refuse a partition whose template at the envelope's b_max needs more temporal band than the source stores (the
    replay's own check, :meth:`~dmipy_sim.replay.replay.ReplayPack._check_band`): a segment's sharp edges are the
    waveform's, and the bands beyond the pack's would carry phase the vectors do not."""
    from ..acquisition.waveforms import calc_b
    seq = partition.template
    b = np.asarray(calc_b(seq), np.float64)
    if not np.any(b > 0):
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
        for s, (t, e) in enumerate(zip(p0["segments"], p0["gate"])):
            pl = layout[3 * s:3 * s + 3]
            segs.append(dict(t=list(t), gate=int(e), q_max=float(reach[s]), floor_target=float(floor), step=float(step[s]),
                             bits=[int(b) for b in bits[s]], offsets=[int(o) for _, o, _ in pl], word=[int(w) for w, _, _ in pl],
                             scale=[float(x) for x in scale[s]]))
        out[name] = dict(group=p0["group"], segments=segs, n_words=int(n_words), column=f"p_{name}.npy",
                         rounding=dict(budget=float(R), variance=realised, source_quantisation=float(v_src), eps=float(eps)))
    return out


def _tier_codes(recs, g, eps, env):
    """The group's tier word layout: the contact at ``sqrt(12) eps / (rho2/D)_max``, the field channels each at
    ``sqrt(12 / n) eps / (B0 chi_max alpha_c)`` (``iso_P_zz`` implied when the blocks' largest trace residual is within
    half that step), a field that rounds to zero on every row of every block not stored."""
    g0 = recs[0]["groups"][g]
    fields = []
    if g0.get("contact"):
        a = max(float(r["groups"][g]["contact_amax"]) for r in recs)
        fields.append(dict(name="contact", amax=a, step=float(np.sqrt(12.0) * eps / env["rho2_over_D_max"]),
                           q_max=env["rho2_over_D_max"]))
    field_rec = None
    if g0.get("field"):
        names = list(g0["field"]["channels"])
        amax = np.max([np.asarray(r["groups"][g]["field"]["amax"], np.float64) for r in recs], axis=0)
        residual = None
        if "trace_abs" in g0["field"]:
            residual = max(float(r["groups"][g]["field"]["trace_abs"]) for r in recs)
        # iso_P_zz is implied when the trace identity's residual is within half the step the channels take with it
        # implied: reconstructing it then moves no rounded value
        n_implied = len(names) - 1
        step_implied = np.sqrt(12.0 / n_implied) * eps / env["B0_chi_max_T"]
        implied = residual is not None and residual <= 0.5 * step_implied
        cand = [nm for nm in names if not (implied and nm == "iso_P_zz")]
        n = len(cand)
        for nm in cand:
            alpha = 2.0 if (implied and nm == "iso_local") else 1.0
            fields.append(dict(name=nm, amax=float(amax[names.index(nm)]), q_max=env["B0_chi_max_T"] * alpha,
                               step=float(np.sqrt(12.0 / n) * eps / (env["B0_chi_max_T"] * alpha))))
        field_rec = dict(channels=names, iso_P_zz=("implied" if implied else "stored"), trace_residual=residual)
    if not fields:
        return None, field_rec
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
    ``out_dir``: the blocks' declarations checked equal, every width derived from the blocks' largest values, the rows
    placed into tiles (each voxel's rows alternating between its split halves in block and row order) and packed, in two
    passes over the rows that hold one block at a time. ``progress(rows, seconds)`` is called after every block.
    Returns the manifest."""
    from ..phantom.grid import Grid
    blocks = [str(b) for b in ([blocks] if isinstance(blocks, (str, os.PathLike)) else blocks)]
    recs = [read_rows(b) for b in blocks]
    for key, what in (("grid", "the grid"), ("background", "the background")):
        _same(recs, key, what)
    for r in recs[1:]:
        for k in ("kind", "floor_target", "eps", "envelope", "background_amplitude"):
            if json.dumps(r["source"][k], sort_keys=True, default=float) != json.dumps(recs[0]["source"][k], sort_keys=True, default=float):
                raise ValueError(f"the blocks' rows disagree on the source's {k}: they are not rows of one layout")
        if {n: (p["segments"], p["gate"], p["group"]) for n, p in r["partitions"].items()} != \
                {n: (p["segments"], p["gate"], p["group"]) for n, p in recs[0]["partitions"].items()}:
            raise ValueError("the blocks' rows disagree on the partitions: they are not rows of one layout")
        if {g: (x["rf"], x["dt"], x["n_t"], x["readout"], x["contact"], bool(x["field"])) for g, x in r["groups"].items()} != \
                {g: (x["rf"], x["dt"], x["n_t"], x["readout"], x["contact"], bool(x["field"])) for g, x in recs[0]["groups"].items()}:
            raise ValueError("the blocks' rows disagree on the sequence groups: they are not rows of one layout")
    r0 = recs[0]; src = r0["source"]; env = src["envelope"]
    grid = Grid.from_meta(r0["grid"]); n_vox = int(np.prod(grid.shape))
    floor, eps = float(src["floor_target"]), float(src["eps"])
    partitions = _partition_codes(recs, floor, eps)
    groups = {}
    for g, g0 in r0["groups"].items():
        tw, field_rec = _tier_codes(recs, g, eps, env) if r0["tiers"] else (None, None)
        grp = {k: g0[k] for k in ("partitions", "rf", "dt", "n_t", "readout", "pathway", "tau_t2", "tau_t1")}
        grp.update(contact=bool(g0.get("contact")), field=field_rec, tier_words=tw,
                   column=(f"t_{g}.npy" if tw is not None else None))
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
    for name, p in partitions.items():
        out[f"p_{name}"] = (mm(p["column"], np.uint32, (p["n_words"],)), p)
    if r0["tiers"]:
        out["pool"] = (mm("pool.npy", np.uint8), None)
    for g, grp in groups.items():
        if grp["tier_words"] is not None:
            out[f"t_{g}"] = (mm(grp["column"], np.uint32, (grp["tier_words"]["n_words"],)), grp)
        if grp.get("background"):
            out[f"bg_{g}"] = (mm(grp["background"]["column"], np.uint32, (grp["background"]["n_words"],)), grp["background"])
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
        tiers = dict(r0["tiers"], pool_column="pool.npy")
    manifest = dict(
        format=FORMAT, n_rows=int(counts.sum()), n_tiles=n_tiles, tile=TILE, n_voxels=n_vox, grid=r0["grid"],
        source=dict(kind=src["kind"], walk=src["walk"], pack=src["pack"], floor_target=floor, eps=eps, envelope=env,
                    background_amplitude=src["background_amplitude"],
                    blocks=[dict(path=os.path.basename(os.path.normpath(b)), n_rows=int(r["n_rows"]), walk=r["source"]["walk"],
                                 pack=r["source"]["pack"]) for b, r in zip(blocks, recs)]),
        certificate=dict(source=src["kind"], terms=r0["certificate"]["terms"],
                         source_terms_per_block=[r["certificate"] for r in recs],
                         rounding={name: p["rounding"] for name, p in partitions.items()}),
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
    stored :class:`Partition`\\ s by name), and :meth:`image` -- a whole grid under any sequence piecewise constant on a
    stored partition, on the device. Open a Hub layout with :meth:`open`."""

    TILES = 4096                                                        # tiles per device chunk (2^19 rows)
    MEAS = 64                                                           # measurements padded to a multiple
    #: distinct (sequence group, B0 direction) field-term pairs held at once, host and every backend's device copies
    #: together (dmipy-sim#517): a page with free B0 angles asks for more directions than fit in memory, so this is a
    #: small LRU -- a pair beyond the bound evicts the least recently used one (:meth:`_touch_field_direction`)
    FIELD_DIRECTION_CACHE = 4
    FIELD_CHUNK_TILES = 65_536                   # tiles per contraction chunk

    def __init__(self, path):
        from ..phantom.grid import Grid
        self.path = path
        self.manifest = read_manifest(path)
        self.grid = Grid.from_meta(self.manifest["grid"])
        self.n_rows = int(self.manifest["n_rows"]); self.n_tiles = int(self.manifest["n_tiles"]); self.tile = int(self.manifest["tile"])
        self.n_vox = int(np.prod(self.grid.shape))
        self.partitions = {n: Partition(n, tuple(tuple(s["t"]) for s in p["segments"]), tuple(int(s["gate"]) for s in p["segments"]))
                           for n, p in self.manifest["partitions"].items()}
        self.tiers = self.manifest.get("tiers")
        self.susceptibility_field = (None if not self.tiers else
                                     _field_of_tiers((self.manifest["source"].get("pack") or {}).get("id"), self.tiers,
                                                     self.manifest["groups"]))
        self._m = {}; self._device = {}; self._host = {}
        self._field_direction_lru = collections.OrderedDict()     # (group, direction key) -> None, MRU last

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
        cols.update({f"p_{n}": p["column"] for n, p in manifest["partitions"].items()})
        if manifest.get("tiers"):
            cols["pool"] = manifest["tiers"]["pool_column"]
        for g, grp in manifest["groups"].items():
            if grp.get("column"):
                cols[f"t_{g}"] = grp["column"]
            if grp.get("background"):
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
        fits) and its rows' effective gradients on the segments ``(n_meas, n_seg, 3)``; refused by name when none fits."""
        names = list(self.partitions) if partition is None else [partition]
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
        encoding above ``b_max``."""
        env = self.manifest["source"]["envelope"]
        B = self.partitions[partition].b_matrix()
        b = np.einsum("msc,st,mtc->m", L, B, L)
        g = np.linalg.norm(L, axis=2).max(initial=0.0)
        if g > env["G_max"] * (1 + 1e-6):
            raise ValueError(f"the sequence plays {g:.4g} T/m on partition {partition!r} and its vectors are derived for at "
                             f"most G_max = {env['G_max']:.4g} T/m")
        if b.size and b.max() > env["b_max"] * (1 + 1e-6):
            raise ValueError(f"the sequence reaches b = {b.max():.4g} s/m^2 on partition {partition!r} and its vectors are "
                             f"derived for at most b_max = {env['b_max']:.4g} s/m^2 (the source's declared envelope)")
        return b

    def vectors(self, partition):
        """``(n_tiles, tile, n_seg, 3)`` float32: every row's segment vectors (0 on padding), decoded from the words."""
        scale, layout, _ = self._code(partition)
        n_seg = len(self.partitions[partition].segments)
        v = unpack_words(self._column(f"p_{partition}"), scale, layout)
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
        amplitude)``. A tier the pair needs that the layout lacks is refused by name."""
        if tissue is None and scanner is None:
            return None
        if not self.tiers:
            raise ValueError("this layout holds the bare diffusion phase only (written without tiers=True); a tissue "
                             "or a scanner needs the tiers")
        from .study import Primitives
        from ..spec.substrate import SubstrateSpec
        from ..spec.tissue import _by_pool_id
        t = self.tiers; g, grp = self._group(partition)
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
                          pathway=float(grp["pathway"]), by_pool=by_pool, susceptibility_field=self.susceptibility_field)
        rt = desc.reduction_terms(tissue, scanner)
        logw = -(float(grp["tau_t2"] or 0.0) * np.asarray(rt["invT2"], np.float64) + float(grp["tau_t1"] or 0.0) * np.asarray(rt["invT1"], np.float64))
        return (logw, float(rt["rho2_over_D"]), float(rt["a_iso"]), float(rt["a_aniso"]), float(rt["amplitude"]))

    def _check_gate(self, partition, sequence):
        """Refuse a tiered image of a sequence whose gate (grid, RF schedule, readout) is not the one ``partition``'s
        tiers were read under."""
        g, grp = self._group(partition)
        seq = _sequence(sequence)
        mine = json.dumps(dict(dt=float(grp["dt"]), n_t=int(grp["n_t"]), rf=grp["rf"], readout=grp["readout"]), sort_keys=True, default=float)
        if _group_key(seq) != mine:
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

    # ---- the field direction's terms
    @staticmethod
    def _direction_key(b0_direction):
        """The one spelling of a field direction in the field-term columns' names (``field_iso_<g>@<key>``)."""
        b0 = np.asarray(b0_direction, np.float64)
        if b0.shape != (3,) or abs(np.linalg.norm(b0) - 1.0) > 1e-6:
            raise ValueError("b0_direction is a unit vector")
        return ",".join(f"{x:.6f}" for x in b0)

    _b0_key = _direction_key((0.0, 0.0, 1.0))      # the frame's z, the pose the tiers were gated under

    def _set_b0_direction(self, b0_direction):
        self._b0_key = self._direction_key(b0_direction)

    @staticmethod
    def _field_direction_of(name):
        """``(g, d)`` of a field-term column name (``field_iso_<g>@<d>`` / ``field_aniso_<g>@<d>``), else ``None``."""
        if not name.startswith(("field_iso_", "field_aniso_")):
            return None
        g, d = name.split("_", 2)[2].split("@")
        return g, d

    def _touch_field_direction(self, g, d):
        """Mark ``(g, d)``'s field terms as just used in the bounded field-direction cache (dmipy-sim#517): past
        :attr:`FIELD_DIRECTION_CACHE` distinct pairs the host and every backend's device copies of the least recently
        used pair are dropped."""
        key = (g, d)
        if key in self._field_direction_lru:
            self._field_direction_lru.move_to_end(key)
        else:
            self._field_direction_lru[key] = None
        while len(self._field_direction_lru) > self.FIELD_DIRECTION_CACHE:
            og, od = next(iter(self._field_direction_lru))
            del self._field_direction_lru[(og, od)]
            for name in (f"field_iso_{og}@{od}", f"field_aniso_{og}@{od}"):
                self._host.pop(name, None)
                for dk in [k for k in self._device if k[1] == name]:
                    del self._device[dk]

    def _channels_of(self, g, words, xp):
        """The full channel set (``iso_local, iso_P (6)[, aniso_G (6)]``) of group ``g`` from int32 tier ``words`` on
        ``xp`` (numpy or torch), float64: the stored channels decoded, ``iso_P_zz`` from the trace identity when it is
        implied, a channel not stored zero."""
        grp = self.manifest["groups"][g]; f = grp["field"]
        layout, scale, names = self._tier_code(g)
        vals = dict(zip(names, _decode(words, scale, layout)))
        full = []
        zero = None
        for nm in f["channels"]:
            v = vals.get(nm)
            if v is None and not (nm == "iso_P_zz" and f["iso_P_zz"] == "implied"):
                if zero is None:
                    zero = words[..., 0] * 0
                v = zero
            full.append(v)
        if f["iso_P_zz"] == "implied":
            i = f["channels"].index("iso_P_zz")
            full[i] = 3.0 * full[f["channels"].index("iso_local")] - full[f["channels"].index("iso_P_xx")] - full[f["channels"].index("iso_P_yy")]
        if xp == "torch":
            import torch
            return torch.stack([x.to(torch.float64) for x in full], dim=-1)
        return np.stack([np.asarray(x, np.float64) for x in full], axis=-1)

    def _field_terms(self, g, b0_direction, device=None):
        """``(iso, aniso)`` per row of group ``g`` for ``b0_direction`` (a unit vector in the substrate frame), from the
        decoded channels through the one contraction, zeros when the group carries no field: on the host into two float32
        arrays chunk by chunk, or, given a torch ``device``, the words uploaded chunk by chunk and decoded and contracted
        there."""
        from ..fields.hollow_cylinder import field_terms
        grp = self.manifest["groups"][g]
        T = self.TILES; n_pad = -(-self.n_tiles // T) * T
        if device is None:
            iso = np.zeros((n_pad, self.tile), np.float32); aniso = np.zeros((n_pad, self.tile), np.float32)
        else:
            import torch
            iso = torch.zeros((n_pad, self.tile), dtype=torch.float32, device=device)
            aniso = torch.zeros((n_pad, self.tile), dtype=torch.float32, device=device)
        if grp.get("field") and grp["field"]["stored"]:
            words = self._column(f"t_{g}")
            b0 = np.asarray(b0_direction, np.float64)
            has_aniso = "aniso_G_xx" in grp["field"]["channels"]
            for s in range(0, self.n_tiles, self.FIELD_CHUNK_TILES):
                e = min(s + self.FIELD_CHUNK_TILES, self.n_tiles)
                w32 = np.asarray(words[s:e]).view(np.int32)
                if device is None:
                    a, b = field_terms(self._channels_of(g, w32, "numpy"), b0)
                    iso[s:e] = a
                    if has_aniso and b is not None:
                        aniso[s:e] = b
                else:
                    import torch
                    a, b = field_terms(self._channels_of(g, torch.as_tensor(w32, device=device), "torch"), b0)
                    iso[s:e] = a.to(torch.float32)
                    if has_aniso and b is not None:
                        aniso[s:e] = b.to(torch.float32)
        return iso, aniso

    def tier_names(self, partition):
        """The device-side tier columns of ``partition``'s group: the pool, the group's tier words, its field terms for
        the current field direction."""
        g, _ = self._group(partition)
        d = self._b0_key
        return ("pool", f"t_{g}", f"field_iso_{g}@{d}", f"field_aniso_{g}@{d}")

    def _is_tier_column(self, name):
        return name == "pool" or name.startswith(("t_", "field_iso_", "field_aniso_", "bg_"))

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

    def _padded(self, name):
        """The column ``name`` on the host, padded to whole chunks of ``TILES`` (padding rows weigh nothing and scatter
        into the dump slot), words as their int32 view; the preloaded copy when there is one. A field-direction column
        touches the bounded field-direction cache."""
        fd = self._field_direction_of(name)
        if name in self._host:
            if fd is not None:
                self._touch_field_direction(*fd)
            return self._host[name]
        T = self.TILES; n_pad = -(-self.n_tiles // T) * T
        if fd is not None:
            g, d = fd
            iso, aniso = self._field_terms(g, [float(x) for x in d.split(",")])
            self._host[f"field_iso_{g}@{d}"], self._host[f"field_aniso_{g}@{d}"] = iso, aniso
            self._touch_field_direction(g, d)
            return self._host[name]
        if name == "slot":                                       # each tile's index into a delivery's live voxels
            live = self.live_voxels
            compact = np.full(self.n_vox + 1, len(live), np.int32); compact[live] = np.arange(len(live), dtype=np.int32)
            a = np.full(n_pad, len(live), np.int32); a[:self.n_tiles] = compact[np.asarray(self._column("tiles")) // 2]
        elif name == "tiles":
            a = np.full(n_pad, 2 * self.n_vox, np.int32); a[:self.n_tiles] = self._column("tiles")
        elif name in ("w", "pool"):
            a = np.zeros((n_pad, self.tile), np.uint8 if name == "pool" else np.float32); a[:self.n_tiles] = self._column(name)
        elif name.startswith("t_") and not self.manifest["groups"][name[2:]].get("column"):
            a = np.zeros((n_pad, self.tile, 1), np.int32)                # a group with no tier words: one zero word
        else:
            words = self._column(name)
            a = np.zeros((n_pad,) + tuple(words.shape[1:]), np.uint32); a[:self.n_tiles] = words
            a = a.view(np.int32)
        return a

    def _resident(self, backend, device, resident, names):
        """The columns ``names`` on the backend's device, kept across calls when ``resident``, else built afresh. Field
        terms for a direction the host has not contracted are contracted on the device (torch)."""
        if backend == "jax":
            import jax.numpy as jnp
            put = lambda a, name: jnp.asarray(a)
        else:
            import torch
            put = lambda a, name: torch.as_tensor(a.astype(np.int64) if name in ("tiles", "slot") else a, device=device)
        out = []; made = {}
        for name in names:
            key = (backend, name, str(device))
            fd = self._field_direction_of(name)
            if key not in self._device:
                if backend == "torch" and fd is not None and name not in self._host:
                    if name not in made:
                        g, d = fd
                        iso, aniso = self._field_terms(g, [float(x) for x in d.split(",")], device=device)
                        made.update({f"field_iso_{g}@{d}": iso, f"field_aniso_{g}@{d}": aniso})
                    arr = made[name]
                else:
                    arr = put(self._padded(name), name)
                if not resident:
                    if fd is not None:
                        self._touch_field_direction(*fd)
                    out.append(arr); continue
                self._device[key] = arr
            if fd is not None:
                self._touch_field_direction(*fd)
            out.append(self._device[key])
        return tuple(out)

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
        None without a scanner)."""
        if backend not in ("jax", "torch"):
            raise ValueError(f"backend is 'jax' or 'torch', got {backend!r}")
        if not self.tiers:
            raise ValueError("this layout holds the bare diffusion phase only (written without tiers=True)")
        tm = self.terms(partition, tissue, scanner)
        self._set_b0_direction(b0_direction)
        g, grp = self._group(partition)
        pools = [(q["name"], int(q["id"])) for q in self.tiers["substrate"]["pools"]]
        if backend == "torch" and device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        w, tiles, pool, tw, fiso, faniso = self._resident(backend, device, resident, ("w", "tiles") + self.tier_names(partition))
        rho_D, a_iso, a_aniso = (0.0, 0.0, 0.0) if tm is None else (float(tm[1]), float(tm[2]), float(tm[3]))
        code = self._tier_code(g)
        names = [("pool", name, pid) for name, pid in pools]
        if grp.get("contact"):
            names.append(("contact",))
            if tm is not None:
                names.append(("contact_weight",))
        if grp.get("field") and tm is not None:
            names += [("phase",), ("phase_sq",)]

        def rows(sl, xp):
            out = []
            c = 0.0 * w[sl]
            if code is not None and "contact" in code[2]:
                layout, scale, fnames = code
                sc = xp.asarray(scale) if backend == "jax" else xp.as_tensor(scale, device=tw.device)
                c = _decode(tw[sl], sc, layout)[fnames.index("contact")]
            for q in names:
                if q[0] == "pool":
                    out.append((pool[sl] == q[2]).to(w.dtype) if backend == "torch" else (pool[sl] == q[2]).astype(w.dtype))
                elif q[0] == "contact":
                    out.append(c)
                elif q[0] == "contact_weight":
                    out.append(xp.exp(rho_D * c))
                else:
                    ph = a_iso * fiso[sl] + a_aniso * faniso[sl]
                    out.append(ph if q[0] == "phase" else ph * ph)
            return out

        n_halves = 2 * self.n_vox + 1
        acc = (_voxel_sums_torch if backend == "torch" else _voxel_sums_jax)(rows, len(names), w, tiles, n_halves, self.TILES)
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
        e = np.asarray(self.partitions[name].gate, np.float64)
        plateau = L * np.where(e == 0.0, 0.0, e)[None, :, None]          # the commanded (physical) gradient: e = +-1
        live = self.live_voxels
        ijk = np.stack(np.unravel_index(live, tuple(self.grid.shape)), axis=1)
        d = delivered_vectors(scanner, grid, plateau, e, voxels=ijk, to_scanner=to_scanner, nonlinearity=nonlinearity,
                              background=background, concomitant=concomitant, transmit=transmit)
        return Delivered(partition=name, voxels=live, levels=L, q=d.q, g0=d.g0, kappa=d.kappa)

    def image(self, sequence, *, partition=None, backend="jax", device=None, resident=True, tissue=None, scanner=None,
              b0_direction=(0.0, 0.0, 1.0), delivered=None):
        """``(S, floor)`` of the grid under ``sequence`` (a :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence`
        whose rows are piecewise constant on a stored partition; ``partition`` names one, else the first it fits): ``S``
        ``grid.shape + (n_meas,)``, the weighted ensemble magnitude per voxel (NaN where the layout has no rows),
        ``floor`` ``grid.shape`` its split-half floor ``max_m |S_a - S_b| / 2``. A row beyond the envelope the widths are
        derived at (``G_max``, ``b_max``) is refused by name. ``backend`` runs the arithmetic on JAX (its default
        device) or torch (``device``, CUDA when torch sees one, else its CPU); ``resident=False`` uploads the columns for
        this call only. ``tissue`` and ``scanner`` turn the stored tiers into the weights and the field phase of
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
        self._set_b0_direction(b0_direction)
        scale, layout, _ = self._code(name)
        n_seg = len(self.partitions[name].segments)
        M = L.shape[0]; M_pad = -(-M // self.MEAS) * self.MEAS
        n_halves = 2 * self.n_vox + 1
        if backend == "torch" and device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        g, grp = self._group(name)
        tier_names = self.tier_names(name) if tm is not None else ()
        contact = None
        t_code = self._tier_code(g)
        if tm is not None and t_code is not None and "contact" in t_code[2]:
            t_layout, t_scale, t_names = t_code
            i = t_names.index("contact")
            contact = ((t_layout[i],), t_scale[i:i + 1])
        factor = None; bg = None
        if delivered is not None:
            live = self._check_delivered(delivered, name, L)
            names = ("w", "tiles", "slot") + tuple(tier_names)
            if delivered.g0 is not None:
                gb = self.background_group(name)
                names += (f"bg_{gb}",)
                b_layout, b_scale = _layout_of(self.manifest["groups"][gb]["background"])
            cols = self._resident(backend, device, resident, (f"p_{name}",) + names)
            words, w, tiles, slot = cols[:4]
            rest = cols[4:]
            tier_cols = rest[:len(tier_names)]
            Q = np.zeros((len(live) + 1, M_pad, n_seg, 3), np.float32); Q[:len(live), :M] = delivered.q
            G0 = np.zeros((len(live) + 1, 3), np.float32)
            if delivered.g0 is not None:
                G0[:len(live)] = delivered.g0
                bg = (rest[len(tier_names)], b_layout, b_scale, G0)
            if delivered.kappa is not None:
                from ..acquisition.epg import transmit_amplitude
                factor = np.ones(self.n_vox)
                factor[live] = transmit_amplitude(_sequence(sequence), delivered.kappa) / transmit_amplitude(_sequence(sequence), [1.0])[0]
            levels, sl = Q, slot
        else:
            cols = self._resident(backend, device, resident, (f"p_{name}", "w", "tiles") + tuple(tier_names))
            words, w, tiles = cols[:3]
            tier_cols = cols[3:]
            levels = np.zeros((M_pad, n_seg, 3), np.float32); levels[:M] = L
            sl = None
        tier = None
        if tm is not None:
            pool, tw, fiso, faniso = tier_cols
            tier = (pool, tw, fiso, faniso, contact, np.asarray(tm[0], np.float32), float(tm[1]), float(tm[2]), float(tm[3]), float(tm[4]))
        run = _sums_jax if backend == "jax" else _sums_torch
        acc = run(words, scale, layout, w, tiles, levels, sl, tier, bg, n_halves, self.TILES)
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


def _row_terms(ph, w, tier, bg, slot, on_torch):
    """The phase and the weight of each row with the background (``g0_v . n``, the voxel's ``g0`` by ``slot``) and the
    tiers (the weight ``w amp exp(logw_pool[pool] + rho_D c)``, the field offset ``a_iso iso + a_aniso aniso``) folded
    in, on JAX or (``on_torch``) torch."""
    if bg is not None:
        nb_words, b_layout, b_scale, G0 = bg
        g0 = G0[slot]
        off = None
        for c, v in enumerate(_decode(nb_words, b_scale, b_layout)):
            if v is not None:
                t = v * g0[:, None, c]
                off = t if off is None else off + t
        if off is not None:
            ph = ph + off[:, :, None]
    if tier is None:
        return ph, w
    pool, tw, fiso, faniso, contact, logw_pool, rho_D, a_iso, a_aniso, amp = tier
    ph = ph + (a_iso * fiso + a_aniso * faniso)[:, :, None]
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


def _compiled(n_halves, layout, per_tile, tiers, contact, bg_layout):
    """The JAX chunk kernel: the words decoded, the phase, the tiers and the background folded in, ``w exp(i ph)`` in
    complex64 summed over each tile's rows (a fused reduction), the tile sums scattered into their voxel halves in
    complex128. One per (voxel halves, field layout, delivery, tiers, contact, background layout)."""
    import jax
    import jax.numpy as jnp
    key = (n_halves, layout, per_tile, tiers, contact, bg_layout)
    if key not in _KERNELS:
        @jax.jit
        def kernel(words, scale, w, tiles, levels, slot, pool, tw, fiso, faniso, c_scale, logw_pool, rho_D, a_iso, a_aniso, amp,
                   nb, b_scale, G0, acc):
            comps = _decode(words, scale, layout)
            lev = levels[slot] if per_tile else levels
            ph = _phase(comps, lev, per_tile)
            if ph is None:
                ph = jnp.zeros(words.shape[:2] + (levels.shape[-3],), jnp.float32)
            tier = ((pool, tw, fiso, faniso, (contact, c_scale) if contact is not None else None, logw_pool, rho_D, a_iso, a_aniso, amp)
                    if tiers else None)
            bg = (nb, bg_layout, b_scale, G0) if bg_layout is not None else None
            ph, wf = _row_terms(ph, w, tier, bg, slot, False)
            E = (jnp.exp(1j * ph) * wf[:, :, None]).astype(jnp.complex128)
            return acc + jax.ops.segment_sum(E.sum(1), tiles, num_segments=n_halves)
        _KERNELS[key] = kernel
    return _KERNELS[key]


def _sums_jax(words, scale, layout, w, tiles, levels, slot, tier, bg, n_halves, chunk):
    """The per-voxel-half sums ``(n_halves, M_pad)`` complex128 on JAX: the compiled chunk kernel over the tiles."""
    import jax.numpy as jnp
    per_tile = slot is not None
    contact = None if tier is None or tier[4] is None else tier[4][0]
    bg_layout = None if bg is None else bg[1]
    kernel = _compiled(n_halves, layout, per_tile, tier is not None, contact, bg_layout)
    M = levels.shape[-3]
    acc = jnp.zeros((n_halves, M), jnp.complex128)
    z = jnp.zeros(1, jnp.float32); zi = jnp.zeros((1, 1, 1), jnp.int32)
    lev, sc = jnp.asarray(levels), jnp.asarray(scale)
    if tier is not None:
        pool, tw, fiso, faniso, ct, lw, rho_D, a_iso, a_aniso, amp = tier
        c_scale = jnp.asarray(ct[1]) if ct is not None else z
        tconst = (jnp.asarray(lw), jnp.float32(rho_D), jnp.float32(a_iso), jnp.float32(a_aniso), jnp.float32(amp))
    if bg is not None:
        nb, _, b_scale, G0 = bg
        bconst = (jnp.asarray(b_scale), jnp.asarray(G0))
    for i in range(0, words.shape[0], chunk):
        s = slice(i, i + chunk)
        targs = (pool[s], tw[s], fiso[s], faniso[s], c_scale) + tconst if tier is not None else (z, zi, z, z, z, z, z, z, z, z)
        bargs = (nb[s],) + bconst if bg is not None else (zi, z, z)
        acc = kernel(words[s], sc, w[s], tiles[s], lev, slot[s] if per_tile else z, *targs, *bargs, acc)
    return np.asarray(acc)


def _tile_sums_torch(words, scale, layout, w, levels, slot, tier, bg, per_tile):
    """One chunk's tile sums on torch, ``(chunk, M_pad)`` complex128: the decoded words, the phase with the tiers and
    the background, ``w exp(i ph)`` in complex64, the reduce over each tile's rows."""
    import torch
    comps = _decode(words, scale, layout)
    lev = levels[slot.long()] if per_tile else levels
    ph = _phase(comps, lev, per_tile)
    if ph is None:
        ph = torch.zeros(words.shape[:2] + (levels.shape[-3],), dtype=torch.float32, device=words.device)
    ph, wf = _row_terms(ph, w, tier, bg, slot.long() if per_tile else None, True)
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


def _sums_torch(words, scale, layout, w, tiles, levels, slot, tier, bg, n_halves, chunk):
    """The same sums on torch: the tile sums of every chunk (:func:`_tile_sums_torch`, compiled when possible)
    ``index_add_``-ed into a complex128 accumulator (deterministic under ``torch.use_deterministic_algorithms(True)``)."""
    import torch
    dev = words.device
    per_tile = slot is not None
    lev = torch.as_tensor(levels, device=dev); sc = torch.as_tensor(scale, device=dev)
    if tier is not None:
        pool, tw, fiso, faniso, ct, lw, rho_D, a_iso, a_aniso, amp = tier
        ct = None if ct is None else (ct[0], torch.as_tensor(ct[1], device=dev))
        lw = torch.as_tensor(lw, device=dev)
    if bg is not None:
        nb, b_layout, b_scale, G0 = bg
        b_scale = torch.as_tensor(b_scale, device=dev); G0 = torch.as_tensor(G0, device=dev)
    acc = torch.zeros((n_halves, levels.shape[-3]), dtype=torch.complex128, device=dev)
    fn = _torch_kernel()
    with torch.no_grad():
        for i in range(0, words.shape[0], chunk):
            s = slice(i, i + chunk)
            t = None if tier is None else (pool[s], tw[s], fiso[s], faniso[s], ct, lw, rho_D, a_iso, a_aniso, amp)
            b = None if bg is None else (nb[s], b_layout, b_scale, G0)
            args = (words[s], sc, layout, w[s], lev, slot[s] if per_tile else None, t, b, per_tile)
            try:
                ts = fn(*args)
            except Exception:                                  # the compiled path failed at run time: eager, once for all
                _TORCH_KERNEL["fn"] = fn = _tile_sums_torch
                ts = fn(*args)
            acc.index_add_(0, tiles[s], ts)
    return acc.cpu().numpy()


def _voxel_sums_torch(rows, n_q, w, tiles, n_halves, chunk):
    """The weighted per-voxel-half sums ``(n_q, n_halves)`` float64 on torch of the ``n_q`` per-row quantities
    ``rows(sl, torch)`` gives for a chunk of tiles."""
    import torch
    acc = torch.zeros((n_q, n_halves), dtype=torch.float64, device=w.device)
    with torch.no_grad():
        for i in range(0, w.shape[0], chunk):
            sl = slice(i, i + chunk)
            ts = torch.stack([(q * w[sl]).sum(1) for q in rows(sl, torch)]).to(torch.float64)
            acc.index_add_(1, tiles[sl], ts)
    return acc.cpu().numpy()


def _voxel_sums_jax(rows, n_q, w, tiles, n_halves, chunk):
    """:func:`_voxel_sums_torch` on JAX (``segment_sum`` into the voxel halves)."""
    import jax
    import jax.numpy as jnp
    acc = jnp.zeros((n_halves, n_q), jnp.float64)
    for i in range(0, w.shape[0], chunk):
        sl = slice(i, i + chunk)
        ts = jnp.stack([(q * w[sl]).sum(1) for q in rows(sl, jnp)], axis=1).astype(jnp.float64)
        acc = acc + jax.ops.segment_sum(ts, tiles[sl], num_segments=n_halves)
    return np.asarray(acc).T
