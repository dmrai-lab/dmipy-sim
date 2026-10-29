"""The **time-segment layout**: a walk contracted once against the time segments of its timing classes, so that
every acquisition piecewise constant on a stored partition -- at any amplitude, in any direction on each segment -- is
a few products per walker on the device (dmipy-sim#505, #770: a whole grid replayed while a user waits).

The layout's reader -- the format, :class:`Shape`, :class:`Partition` and its fit, the words, the tiers and the image
on numpy -- is :mod:`dmipy_replay.replay.time_segments`; this module writes the layout (:func:`segment_rows`,
:func:`write_time_segment_layout`) and its :class:`TimeSegmentLayout` adds the JAX (the default) and torch backends of
the image and a machine's delivered gradient (:meth:`TimeSegmentLayout.delivery`).

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

from dmipy_replay.replay.time_segments import TimeSegmentLayout as _TimeSegmentLayout
import dmipy_replay.replay.compression as _r_compression
import collections
import functools
import hashlib
import itertools
import json
import os
import time
from dataclasses import dataclass, field, replace

import numpy as np
from dmipy_replay.replay.time_segments import Delivered, FORMAT, MANIFEST, Partition, TILE, WORD_BITS, _AXES, _RTOL, _decode, _group_key, _layout_of, _phase, _row_terms

ROWS = "rows.json"
ROWS_FORMAT = "segment_rows/1"
FIELD_BITS_MIN = 3
FIELD_BITS_MAX = 24                                                   # a field's integer is exact in a float32
#: T/m: the largest background gradient a layout's background vectors are certified for by default -- several times
#: the strongest catalogued magnet's own gradient over its anchored volume (the Swoop's 1.4 mT/m at 8 cm)
BACKGROUND_AMPLITUDE = 5e-3






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












def _gate_sequence(seq):
    """``seq``'s gate alone: one row with no gradient on its grid, RF schedule and readout (what the tiers are read
    under)."""
    return replace(seq, G=np.zeros((1, int(seq.n_t), 3), np.float32), encoding=None)




def _term_shapes(rec, name):
    """The shape records of a partition's ``rec`` (rows or manifest), one per term; a record that states none is
    refused naming the field."""
    if rec.get("shapes") is None:
        raise ValueError(f"partition {name!r} has no 'shapes' record: every term records its shape explicitly "
                         "(a constant segment records {'kind': 'constant'})")
    return rec["shapes"]




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
    from dmipy_replay.constants import GAMMA
    from dmipy_replay.replay.compression import read_position_coeffs, segment_projection
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
    from dmipy_replay.constants import GAMMA
    from dmipy_replay.replay.compression import position_container, segment_projection
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
    from dmipy_replay.replay.compression import position_shape, walker_rows
    from .compression import slice_walkers
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
    from dmipy_replay.constants import GAMMA
    from . import compression as cx
    from dmipy_replay.replay.field import susc_path_coeffs
    from dmipy_replay.replay.replay import surface_logweight
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
            live = _walker_max(win.arrays, lambda a: np.abs(_r_compression.bridge_bands(a, cm)).max(axis=0), slice_bytes)[:step2.shape[0]] > 0
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
            full = susc_path_coeffs(_r_compression.walker_rows(win.arrays, 0, 1), pm)[1]
            amax = _walker_max(win.arrays, lambda a: np.abs(susc_path_coeffs(a, pm)[0]).max(axis=0), slice_bytes)   # (full components, K)
            stored = list(pm["channels"])
            live = np.stack([amax[full.index(nm), :step2.shape[1]] > 0 for nm in stored])
            step2 = np.where(live, step2, 0.0)
            g = _r_compression.cosine_projection(eg.reshape(1, -1, 1), float(gate_seq.dt), T_w, step2.shape[1], t0=t0)[0, :, 0]
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
    meta = dict(id=None, compression=dict(method=_r_compression.POSITION_METHOD, K=int(walk.K), T=T_w, channels=channels),
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
        from dmipy_replay.acquisition.scanners import ScannerLimits
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
    from dmipy_replay.replay.replay import _band_phase, _path_field_channels
    from .study import Acquisition
    from dmipy_replay.acquisition.epg import pathway_weight
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
    from dmipy_replay.replay.compression import POSITION_AXES, read_position_coeffs
    from .compression import pack_position_arrays
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
    from dmipy_replay.acquisition.waveforms import tile_waveform
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
    from dmipy_replay.phantom.grid import Grid
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
    from dmipy_replay.acquisition.waveforms import calc_b
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




def write_time_segment_layout(blocks, out_dir, *, progress=None):
    """The layout of the segment rows in the directories ``blocks`` (:func:`segment_rows`, one per block), written to
    ``out_dir``: the blocks' declarations checked equal (the partitions, the groups, the grid, the source's kind, the
    envelope's ``b_max`` and ``G_max``), every width derived from the blocks' largest values at the smallest floor
    target any block declares and the largest ``rho2 / D`` and ``B0 chi`` any states, the rows
    placed into tiles (each voxel's rows alternating between its split halves in block and row order) and packed, in two
    passes over the rows that hold one block at a time. ``progress(rows, seconds)`` is called after every block.
    Returns the manifest."""
    from dmipy_replay.phantom.grid import Grid
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
class TimeSegmentLayout(_TimeSegmentLayout):
    """:class:`dmipy_replay.replay.time_segments.TimeSegmentLayout` (the manifest, the fit, the words, the tiers and the
    image) with the JAX and torch backends of the image beside numpy's (:attr:`BACKENDS`; JAX the default) and a
    machine's delivered gradient per voxel (:meth:`delivery`)."""

    DEFAULT_BACKEND = "jax"

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



class _JaxBackend:
    """The image's arithmetic on JAX (its default device): :func:`_sums_jax`, :func:`_voxel_sums_jax`, 64-bit types
    for the duration of a call (:class:`_x64`)."""
    name = "jax"
    torch = False

    def device(self, device):
        return device

    def put(self, device):
        import jax.numpy as jnp
        return lambda a, name: jnp.asarray(a)

    def context(self):
        return _x64("jax")

    def field_spec(self, fspec, device):
        return _device_field_spec(fspec, "jax", device)

    def sums(self, chunks, scale, layout, levels, per_tile, tier, bg, n_halves, device):
        return _sums_jax(chunks, scale, layout, levels, per_tile, tier, bg, n_halves, "jax", device)

    voxel_sums = staticmethod(_voxel_sums_jax)


class _TorchBackend:
    """The image's arithmetic on torch (``device``; CUDA when torch sees one, else its CPU): :func:`_sums_torch`,
    :func:`_voxel_sums_torch`; tile and slot indices as int64."""
    name = "torch"
    torch = True

    def device(self, device):
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def put(self, device):
        import torch
        return lambda a, name: torch.as_tensor(a.astype(np.int64) if name in ("tiles", "slot") else a, device=device)

    def context(self):
        import contextlib
        return contextlib.nullcontext()

    def field_spec(self, fspec, device):
        return _device_field_spec(fspec, "torch", device)

    def sums(self, chunks, scale, layout, levels, per_tile, tier, bg, n_halves, device):
        return _sums_torch(chunks, scale, layout, levels, per_tile, tier, bg, n_halves, "torch", device)

    voxel_sums = staticmethod(_voxel_sums_torch)


TimeSegmentLayout.BACKENDS = dict(_TimeSegmentLayout.BACKENDS, jax=_JaxBackend(), torch=_TorchBackend())
