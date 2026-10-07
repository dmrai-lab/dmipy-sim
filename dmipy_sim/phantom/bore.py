"""What the machine does to a phantom, as a function of where each voxel sits in the bore.

A magnet's field is not uniform, and the way it departs from uniformity is a property of the MACHINE while
where the sample sits in it is a property of the GRID. This module is the one place the two meet: it renders
a catalogued field law (:meth:`~dmipy_sim.acquisition.scanners.ScannerLimits.b0_offset`) onto a grid and
hands back what a replay already accepts as ``off_resonance``.

Nothing here is a new kind of input. A phantom has taken a callable of scanner coordinates for its
macroscopic layers all along; this supplies one that a machine, rather than a person, is the author of.

Every function takes the machine as a :class:`~dmipy_sim.acquisition.scanners.ScannerLimits` and asks it
the one question each map needs -- ``has_field_law``, ``has_gradient_nonlinearity``, ``has_transmit_profile``
-- so a machine that publishes nothing yields ``None``, which is what a replay already means by "no such
term".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["b0_offset_map", "b1_scale_map", "background_gradient_map", "gradient_tensor_map",
           "delivered_gradient", "delivered_weights", "delivered_moments", "encoding_classes", "EncodingClasses"]


def _model(scanner):
    """The machine as the model these maps read parameters from; a bare field strength is refused, because
    a strength has no shape, no coils and no coil profile to render."""
    from ..acquisition.scanners import ScannerLimits
    if not isinstance(scanner, ScannerLimits):
        raise TypeError(f"a bore map reads a ScannerLimits (ScannerLimits.of('swoop')); got {type(scanner).__name__}")
    return scanner


def _bore(grid, to_scanner):
    """``(R, into_bore)``: the rotation taking the grid's axes to the scanner's -- the grid's own unless the
    caller states one -- and the map from positions in metres to displacements from isocentre in the BORE's
    frame, which is the frame every field law is written in.

    The rotation is taken from the grid by default because forgetting it does not fail: an oblique grid then
    silently evaluates every law at the wrong place, and for a law with a preferred direction, as a
    single-yoke magnet's is, it gets the sign of the asymmetry wrong over part of the volume.
    """
    R = grid.to_scanner if to_scanner is None else to_scanner
    R = None if R is None else np.asarray(R, dtype=np.float64)
    if R is not None:
        if R.shape != (3, 3):
            raise ValueError(f"to_scanner is the 3x3 rotation taking grid axes to scanner axes; got {R.shape}")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
            raise ValueError("to_scanner is not a rotation: a grid's axes are orthonormal in the scanner")
    iso = np.asarray(grid.isocenter_m, dtype=np.float64)

    def into_bore(positions_m):
        d = np.asarray(positions_m, dtype=np.float64).reshape(-1, 3) - iso
        return d if R is None else d @ R.T

    return R, into_bore


def _b0_axis(scanner, R=None):
    """The machine's field direction IN THE GRID'S FRAME, defaulting to the bore axis the formula assumes."""
    v = scanner.b0_axis
    if v is None:
        return (0.0, 0.0, 1.0)
    v = np.asarray(v, dtype=np.float64)
    return v if R is None else v @ R


def b0_offset_map(scanner, grid, *, to_scanner=None, delta_T_K=0.0):
    """A callable giving the static field's departure from uniformity, in tesla, at each voxel of ``grid``.

    Pass the result straight to ``off_resonance=`` of any :meth:`~dmipy_sim.phantom.Phantom.replay`; it has
    the signature a phantom already evaluates for a layer, ``f(positions_m) -> (n,)``.

    ``to_scanner`` is the rotation taking the grid's axes to the scanner's. It defaults to the grid's own
    (:attr:`~dmipy_sim.phantom.Grid.to_scanner`, which :meth:`~dmipy_sim.phantom.Grid.from_oblique_affine`
    sets), so an oblique grid is handled without the caller remembering. Pass it explicitly only to override
    what the grid says.

    ``delta_T_K`` adds the UNIFORM offset a magnet this much warmer holds
    (:meth:`~dmipy_sim.acquisition.scanners.ScannerLimits.b0_drift`). It is a separate argument from the
    shape rather than part of it because it is separate physics: the shape is fixed and spatial, the drift
    is uniform and moves. A scanner cancels the uniform part by re-tuning and cannot cancel the shape, so
    what belongs here is the drift ACCRUED SINCE THE LAST RE-CENTRING, not the drift since the magnet was
    built. On the Swoop that interval is catalogued (``f0_recentering_interval``, 339 s).

    ``None`` when the machine publishes no profile and no coefficient, which is every machine but a
    permanent-magnet one; ``off_resonance=None`` is then exactly right, being what a replay already means
    by "no field offset".
    """
    scanner = _model(scanner)
    drift = 0.0
    if delta_T_K:
        drift = scanner.b0_drift(delta_T_K)
        if drift is None:
            raise ValueError(
                f"{scanner.name!r} has no catalogued temperature coefficient, so a drift of {delta_T_K} K "
                f"cannot be rendered. A superconducting magnet has none because it has no room temperature "
                f"to drift with; this is refused rather than silently ignored")
        drift = float(drift)
    if not scanner.has_field_law:
        if not drift:
            return None
        return lambda positions_m: np.full(np.asarray(positions_m, np.float64).reshape(-1, 3).shape[0], drift)
    _R, into_bore = _bore(grid, to_scanner)
    return lambda positions_m: scanner.b0_offset(into_bore(positions_m)) + drift


def b1_scale_map(scanner, grid, *, to_scanner=None):
    """A callable giving the transmit scale at each voxel of ``grid``: 1 nominal, what multiplies every flip
    angle. Pass it to ``transmit=`` of :meth:`~dmipy_sim.phantom.Phantom.replay`.

    Unlike :func:`b0_offset_map`, a phantom does NOT pick this up on its own from the scanner, and the
    asymmetry is deliberate. A field offset is arithmetic on a contraction the replay was doing anyway; a
    transmit scale acts on the pulses, so it moves the whole replay onto the vector-Bloch route -- one
    magnetisation propagation per distinct scale and pose, instead of one contraction. Something that
    changes the cost of a replay by orders of magnitude is asked for, not assumed.

    It also only works on a frames-mode phantom, because that route does. See ``transmit_tolerance`` on
    :meth:`~dmipy_sim.phantom.Phantom.replay` for what a smooth map costs and how to afford it.
    """
    scanner = _model(scanner)
    if not scanner.has_transmit_profile:
        return None
    _R, into_bore = _bore(grid, to_scanner)
    return lambda positions_m: scanner.b1_scale(into_bore(positions_m))


def background_gradient_map(scanner, grid, *, to_scanner=None):
    """A callable giving the magnet's OWN encoding gradient, in T/m, at each voxel of ``grid``: ``(n, 3)``
    in the GRID's frame, ready for
    :meth:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence.with_background_gradient`.

    This is :func:`b0_offset_map`'s companion and the same law differentiated, but the rotation enters
    TWICE and in opposite directions, which is the one thing to get right. A position goes forward into the
    bore to evaluate the law there; the gradient that comes back is a VECTOR in the bore's frame and has to
    be brought back into the grid's, because that is the frame the sequence's ``G`` is written in. An offset
    is a scalar and needs only the first half, so the asymmetry between the two functions is real rather
    than an oversight.

    ``None`` when the machine publishes no profile.
    """
    scanner = _model(scanner)
    if not scanner.has_field_law:
        return None
    R, into_bore = _bore(grid, to_scanner)

    def gradient(positions_m):
        g = np.atleast_2d(scanner.b0_gradient(into_bore(positions_m)))
        return g if R is None else g @ R                # gradient: the bore's frame back into the grid's

    return gradient


def gradient_tensor_map(scanner, grid, *, to_scanner=None):
    """A callable giving the gradient-nonlinearity tensor ``L`` at each voxel: ``(n, 3, 3)`` in the GRID's
    frame, so it can multiply a sequence's ``G`` directly.

    The rotation enters twice, as it does for
    :func:`background_gradient_map`, but a TENSOR comes back differently from a vector: a position goes
    forward into the bore as ``d R^T``, and the tensor evaluated there returns as ``R^T L R``, which is the
    similarity transform rather than a single product. Getting that wrong leaves a tensor that is still
    symmetric and still plausible, so it does not announce itself.

    ``None`` when the machine has no catalogued coefficients.
    """
    scanner = _model(scanner)
    if not scanner.has_gradient_nonlinearity:
        return None
    R, into_bore = _bore(grid, to_scanner)

    def tensor(positions_m):
        L = np.atleast_3d(scanner.gradient_tensor(into_bore(positions_m))).reshape(-1, 3, 3)
        return L if R is None else np.einsum("ji,njk,kl->nil", R, L, R)      # R^T L R, the similarity transform

    return tensor


def delivered_gradient(scanner, grid, sequence, *, voxels=None, to_scanner=None,
                       nonlinearity=True, background=True, concomitant=True):
    """The gradient each voxel ACTUALLY receives: ``(n_voxels, n_meas, n_t, 3)`` in the grid's frame.

    Three things stand between the gradient a sequence prescribes and the one a spin at ``r`` sees:

    * the coils' **nonlinearity**, ``g -> L(r) g``, which mis-scales and TILTS the encoding direction;
    * the magnet's own **background** gradient, constant in time -- on through the pulses and the dead
      times alike, because a magnet does not switch off -- which the RF
      sign carries like any other gradient and whose contribution to ``b`` does not refocus;
    * the coils' **concomitant** (Maxwell) term, whose extra encoding gradient is quadratic in ``G(t)`` --
      so it varies through the sequence and does not reverse when the coils do.

    This is the reference route: it builds the acquisition as played at each voxel through
    :meth:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence.with_gradient_nonlinearity`,
    ``with_background_gradient`` and ``with_concomitant``, which is exact and costs a sequence per voxel.
    :func:`delivered_weights` is the same thing in the replay's own coefficient space, at a few thousand
    flops per voxel, and the two are held against each other by test.
    """
    scanner = _model(scanner)
    idx = grid.every_voxel if voxels is None else voxels
    pos = grid.positions_m(idx)
    R, into_bore = _bore(grid, to_scanner)
    d_grid = np.asarray(pos, dtype=np.float64).reshape(-1, 3) - np.asarray(grid.isocenter_m, dtype=np.float64)
    B0 = scanner.field_T
    Lf = gradient_tensor_map(scanner, grid, to_scanner=R) if nonlinearity else None
    Ls = None if Lf is None else Lf(pos)
    gmap = background_gradient_map(scanner, grid, to_scanner=R) if background else None
    g0 = None if gmap is None else gmap(pos)
    out = np.empty((len(d_grid), sequence.n_meas, sequence.G.shape[1], 3), dtype=np.float64)
    for k, seq in enumerate(_delivered(scanner, R, d_grid, sequence, Ls, g0, concomitant and B0)):
        out[k] = seq.G
    return out


def _delivered(scanner, R, d_grid, sequence, Ls, g0, concomitant_B0):
    """The acquisition as played at each voxel, one sequence at a time, in the grid's frame."""
    for k in range(len(d_grid)):
        seq = sequence
        if Ls is not None:
            seq = seq.with_gradient_nonlinearity(Ls[k])
        if g0 is not None:
            seq = seq.with_background_gradient(g0[k])
        if concomitant_B0:
            # every argument in the SAME frame as G, which is the grid's: the position and the field axis
            # both come back through R. Reading them in the bore while G is in the grid is a mixed frame,
            # and it is 105 per cent of the concomitant term -- a different quantity, not a perturbation.
            seq = seq.with_concomitant(d_grid[k], float(concomitant_B0), b0_axis=_b0_axis(scanner, R))
        yield seq


def concomitant_phase_map(scanner, grid, sequence, *, voxels=None, to_scanner=None):
    """The order-0 concomitant phase of every voxel, ``(n_voxels, n_meas, n_readouts)`` in radians: the Maxwell
    term's VALUE at the voxel's centre integrated through the coherence sign to each readout, which the
    voxel's whole signal turns by (:attr:`ScannerSequence.concomitant_phase_rad`, dmipy-sim#394). Read from the
    same per-voxel sequences as :func:`delivered_gradient`, the nonlinearity applied first since the term is
    the coil's own gradient's. ``None`` for a scanner without a field strength."""
    scanner = _model(scanner)
    B0 = scanner.field_T
    if not B0:
        return None
    idx = grid.every_voxel if voxels is None else voxels
    pos = grid.positions_m(idx)
    R, _into = _bore(grid, to_scanner)
    d_grid = np.asarray(pos, dtype=np.float64).reshape(-1, 3) - np.asarray(grid.isocenter_m, dtype=np.float64)
    Lf = gradient_tensor_map(scanner, grid, to_scanner=R)
    Ls = None if Lf is None else Lf(pos)
    n_read = len(sequence.readout) if sequence.readout else 1
    out = np.zeros((len(d_grid), sequence.n_meas, n_read))
    for k, seq in enumerate(_delivered(scanner, R, d_grid, sequence, Ls, None, B0)):
        out[k] = seq.concomitant_phase_rad
    return out


class EncodingClasses:
    """The acquisitions of one :func:`encoding_classes` call, composed ONE AT A TIME.

    A class is identified by its member's exact tensor / background / position values alone -- a 3x3, a
    3-vector and a 3-vector, a few dozen bytes -- which is what this object stores, keyed by the bin a voxel
    fell in. :meth:`__getitem__` (and iteration, which is just every index in order) builds the
    :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence` :func:`delivered_gradient` would have
    built, from those numbers, when a caller asks for it, and keeps nothing after returning it.

    Composing a class is :func:`with_gradient_nonlinearity` / :func:`with_background_gradient` /
    :func:`with_concomitant`: elementwise array ops over the base sequence's grid, measured at tens of
    microseconds a class on a 485-measurement protocol (``tests/test_bore_encoding_classes_memory.py``) --
    cheap next to the ~12 MB (``G`` and ``imposed_gradient``, both float32) that composing one and KEEPING it
    costs. So nothing here caches a composed class: the class that stays in memory after it is read is
    exactly what grew without bound over a magnet's own gradient across a head (dmipy-sim#563) -- a handful
    of classes for a uniform scanner, tens of thousands for a permanent magnet's ``g0`` over a whole head.
    A reader that holds the whole thing, as ``list(classes)`` or ``ReplayPack.pose_responses`` do, pays the
    same cost it always did; one that takes a class at a time and lets it go, as
    :meth:`~dmipy_sim.replay.replay.ReplayPack.pose_series` does, never holds a second one.

    ``len()``, ``classes[c]`` (any index or slice) and iteration are the whole interface, and behave as the
    list :func:`encoding_classes` used to return -- the SAME sequence, to the bit, since it is composed from
    the same member's exact values either way.
    """

    __slots__ = ("_sequence", "_has_L", "_has_g0", "_has_c", "_Ls", "_g0", "_d_grid", "_members", "_B0", "_b0_axis")

    def __init__(self, sequence, has_L, has_g0, has_c, Ls, g0, d_grid, members, B0, b0_axis):
        self._sequence = sequence
        self._has_L, self._has_g0, self._has_c = has_L, has_g0, has_c
        self._Ls, self._g0, self._d_grid = Ls, g0, d_grid
        self._members = members
        self._B0, self._b0_axis = B0, b0_axis

    def __len__(self):
        return int(self._members.shape[0])

    def _compose(self, c):
        k = int(self._members[c])
        seq = self._sequence
        if self._has_L:
            seq = seq.with_gradient_nonlinearity(self._Ls[k])
        if self._has_g0:
            seq = seq.with_background_gradient(self._g0[k])
        if self._has_c:
            seq = seq.with_concomitant(self._d_grid[k], self._B0, b0_axis=self._b0_axis)
        return seq

    def __getitem__(self, c):
        n = len(self)
        if isinstance(c, slice):
            return [self._compose(i) for i in range(*c.indices(n))]
        i = c + n if c < 0 else c
        if not (0 <= i < n):
            raise IndexError(c)
        return self._compose(i)

    def __iter__(self):
        for c in range(len(self)):
            yield self._compose(c)

    def __repr__(self):
        return f"EncodingClasses(n_classes={len(self)})"


def encoding_classes(scanner, grid, sequence, voxel_index, *, tolerance=1e-3, to_scanner=None,
                     nonlinearity=True, background=True, concomitant=True):
    """The acquisition as the machine plays it at each voxel of ``voxel_index``, BINNED into the distinct
    ways the gradient is delivered: ``(class_of_voxel, played)`` with ``class_of_voxel`` ``(n,)`` an index
    into ``played``, an :class:`EncodingClasses` -- each member composed as :func:`delivered_gradient`
    composes them, lazily, from the stored key alone (dmipy-sim#563). ``None`` when the machine brings none
    of the three terms (dmipy-sim#377).

    A phantom replays each pack once per pose expansion, so a machine whose gradient depends on where the
    voxel sits would cost one expansion per voxel. The three terms are binned on their inputs instead, each
    to what a fraction ``tolerance`` of ``b`` allows -- the tensor to ``tolerance / 2`` (``b`` goes as the
    square of the gradient), the background to that fraction of the largest commanded gradient, and the
    position the Maxwell term is read at to ``tolerance B0 / G_max`` -- and one acquisition is composed per
    distinct bin from a member's exact values. The count is the cost, and it follows the size of the effect:
    a 3 T magnet with no catalogued shape has only the concomitant term, a few centimetres wide in position,
    so a head is a handful of classes; a permanent magnet's tensor varies by percent across a head and costs
    hundreds at a tolerance of a thousandth, and its own ``g0`` over a whole head at a tight tolerance is tens
    of thousands. ``tolerance=None`` bins nothing: every distinct voxel its own class, exact.

    Nothing here holds a composed class: ``played`` stores each class's bin (a 3x3, a 3-vector, a 3-vector --
    the member's exact values) and builds the full ``ScannerSequence`` only when :class:`EncodingClasses`
    is indexed or iterated, so this function's own memory is ``O(n_voxels)`` regardless of the class count.
    """
    from ..replay.phantom import quantise
    scanner = _model(scanner)
    B0 = scanner.field_T
    has_L = bool(nonlinearity and scanner.has_gradient_nonlinearity)
    has_g0 = bool(background and scanner.has_field_law)
    has_c = bool(concomitant and B0 is not None)
    if not (has_L or has_g0 or has_c):
        return None
    idx = np.asarray(voxel_index)
    pos = grid.positions_m(idx)
    R, _into_bore = _bore(grid, to_scanner)
    d_grid = np.asarray(pos, dtype=np.float64).reshape(-1, 3) - np.asarray(grid.isocenter_m, dtype=np.float64)
    n = d_grid.shape[0]
    G_max = max(float(np.abs(np.asarray(sequence.G)).max()), 1e-30)
    tol = None if tolerance is None else float(tolerance)
    Ls = gradient_tensor_map(scanner, grid, to_scanner=R)(pos) if has_L else None
    g0 = background_gradient_map(scanner, grid, to_scanner=R)(pos) if has_g0 else None
    parts = []
    if has_L:
        parts.append(quantise(Ls.reshape(n, 9), None if tol is None else 0.5 * tol))
    if has_g0:
        parts.append(quantise(g0, None if tol is None else 0.5 * tol * G_max))
    if has_c:
        parts.append(quantise(d_grid, None if tol is None else tol * float(B0) / G_max))
    key = np.concatenate(parts, axis=1)
    # return_index: the first occurrence, in voxel_index's own order, of each unique row -- the same member a
    # class's bin stood for before this was lazy (`np.flatnonzero(inverse == c)[0]`), read off np.unique's own
    # bookkeeping instead of a second O(n_classes * n) scan over `inverse`.
    _uniq, members, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    b0_axis = _b0_axis(scanner, R)
    played = EncodingClasses(sequence, has_L, has_g0, has_c, Ls, g0, d_grid, members,
                             None if B0 is None else float(B0), b0_axis)
    return inverse, played


def delivered_weights(scanner, grid, sequence, *, K, n_t, dt_pack, voxels=None, to_scanner=None,
                      nonlinearity=True, background=True, concomitant=True):
    """Per-voxel replay weights ``W`` for the delivered gradient: ``(n_voxels, (K+2)*3, n_meas)``.

    The same physics as :func:`delivered_gradient`, in the space the replay contracts in, at a few thousand
    flops per voxel instead of a sequence build and a fresh projection. What makes that possible is that
    ``compression.bridge_projection`` is EXACTLY linear in the gradient (checked to 1e-15), so each term
    can be projected once and combined per voxel:

    * ``L(r)`` is a 3x3 mix of the base weights' axis index -- projecting ``L G`` is ``L`` applied to the
      projection of ``G``;
    * the background is a constant vector times the RF sign, so its projection is an outer product of
      ``g0(r)`` with one precomputed time course;
    * the concomitant term is the exact field magnitude's departure from ``B0 + B_n`` at the voxel
      (:meth:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence.with_concomitant`), which is not
      polynomial in position and is quadratic in the DELIVERED gradient ``L(r) G``, so it is projected per
      voxel from the composed acquisition -- one projection of a single increment per voxel. The magnet's
      background is left out of it, for a reason of SIZE rather than principle: ``div B = 0`` forces
      transverse components on the magnet's inhomogeneity too, and those beat against the coils' in ``|B|``
      to give a real cross term, measured on the Swoop at 2-10 per cent of the concomitant gradient and
      1.7e-3 of ``b`` over the 8 cm validity sphere. Carrying it needs the magnet's TRANSVERSE field, which
      the catalogue does not record.

    None of that is an approximation: it is the same sum in a different order, which is why the reference
    route exists to check it rather than to be replaced by it.
    """
    from ..replay.replay import _compile_effective
    from ..replay._replay_kernel import effective_gradient

    scanner = _model(scanner)
    idx = grid.every_voxel if voxels is None else voxels
    pos = grid.positions_m(idx)
    R, into_bore = _bore(grid, to_scanner)
    d = np.asarray(pos, dtype=np.float64).reshape(-1, 3) - np.asarray(grid.isocenter_m, dtype=np.float64)
    B0 = scanner.field_T
    b0_axis = _b0_axis(scanner, R)

    def project(G):
        """W for a physical G, carried through the RF sign AND onto the pack's save grid, exactly as the
        replay does: ``ReplayPack._prepare`` builds its weights as
        ``effective_gradient(G_eff, dt_waveform, n_t_pack, dt_pack)`` -- the sign folded in, then RESAMPLED
        onto the grid the walk was saved on. Weights that skip the resample are self-consistent and
        incompatible with the pack, and a parity test between two routes that both skip it cannot see it.
        """
        G_eff = np.asarray(sequence.with_gradient(G).G_eff, dtype=np.float64)
        on_pack = effective_gradient(G_eff, float(sequence.dt), int(n_t), float(dt_pack))
        return _compile_effective(on_pack, dt_pack, K, n_t)

    W = project(np.asarray(sequence.G, dtype=np.float64))                       # ((K+2)*3, n_meas)
    n_c = (K + 2)
    out = np.repeat(W[None], len(d), axis=0)

    Lf = gradient_tensor_map(scanner, grid, to_scanner=R) if nonlinearity else None
    Ls = None if Lf is None else Lf(pos)
    if Ls is not None:
        base = W.reshape(n_c, 3, -1)
        out = np.einsum("nij,kjm->nkim", Ls, base).reshape(len(d), n_c * 3, -1)

    gmap = background_gradient_map(scanner, grid, to_scanner=R) if background else None
    if gmap is not None:
        unit = [project(np.broadcast_to(e, sequence.G.shape)).reshape(n_c, 3, -1) for e in np.eye(3)]
        out = out + np.einsum("ni,ikjm->nkjm", gmap(pos), np.stack(unit)).reshape(len(d), n_c * 3, -1)

    if concomitant and B0:
        for k in range(len(d)):
            played = sequence if Ls is None else sequence.with_gradient_nonlinearity(Ls[k])
            extra = (np.asarray(played.with_concomitant(d[k], float(B0), b0_axis=b0_axis).G, dtype=np.float64)
                     - np.asarray(played.G, dtype=np.float64))
            out[k] = out[k] + project(extra)

    return out


@dataclass(frozen=True)
class MomentDelivery:
    """What a machine plays at each voxel for an acquisition of one two-valued shape (:func:`delivered_moments`):
    ``q`` ``(n_voxels, n_meas, 3)`` T/m, the delivered encoding vector a walker's shape moment is dotted with;
    ``g0`` ``(n_voxels, 3)`` T/m, the magnet's own gradient its background moment is dotted with (None: no field
    shape catalogued or the term off); ``kappa`` ``(n_voxels,)``, the transmit scale every flip is multiplied by (None:
    no transmit profile catalogued or the term off). In the grid's frame."""
    q: np.ndarray
    g0: np.ndarray = None
    kappa: np.ndarray = None


def delivered_moments(scanner, grid, profile, amplitudes, directions, *, voxels=None, to_scanner=None,
                      nonlinearity=True, background=True, concomitant=True, transmit=True):
    """The gradient each voxel of ``grid`` receives for an acquisition that plays one shape ``profile`` (``(n_t,)``,
    peak 1) at ``amplitudes`` (T/m) along unit ``directions`` (``(n_meas, 3)``, the grid's frame), in the terms a
    walker's two moments take (:class:`MomentDelivery`): the same physics as :func:`delivered_gradient`, reduced.

    Measurement ``i`` commands ``G_i(t) = a_i s(t) u_i``. The coils deliver ``L G_i`` (linear in ``G``: the shape is
    kept, the vector becomes ``a_i L u_i``); the magnet adds its constant ``g0`` (its own shape, the effective gate
    alone: a second moment); and the Maxwell term adds ``gc(L G_i(t))``, a POINTWISE function of the coils' gradient
    (:func:`~dmipy_sim.acquisition.scanner_sequence.concomitant_terms`) that vanishes with it. When the shape is
    two-valued, ``s(t)`` in ``{0, 1}`` -- square pulses of one polarity, as every square-pulse spin echo and
    stimulated echo plays them -- ``gc(L G_i(t)) = s(t) gc(a_i L u_i)`` exactly, so the Maxwell gradient is one more
    vector on the shape: ``q_i = a_i L u_i + gc(a_i L u_i)``. A shape with ramps or a second level is refused for
    the Maxwell term (its time course is then ``s^2``-like and not the shape's), with the bridge route
    (:func:`delivered_weights`) named. None of this is an approximation: the phase is linear in the waveform and
    each piece is the waveform the reference route builds, so the two are held equal by test.

    ``voxels`` ``(n, 3)`` indices (every voxel when None); the term switches as in :func:`delivered_gradient`, with
    ``transmit`` reading the machine's transmit profile (:func:`b1_scale_map`). Each catalogued law refuses a voxel
    beyond its anchor."""
    scanner = _model(scanner)
    idx = grid.every_voxel if voxels is None else np.asarray(voxels)
    pos = grid.positions_m(idx)
    R, _into = _bore(grid, to_scanner)
    d = np.asarray(pos, np.float64).reshape(-1, 3) - np.asarray(grid.isocenter_m, np.float64)
    s = np.asarray(profile, np.float64).reshape(-1)
    a = np.asarray(amplitudes, np.float64).reshape(-1)
    u = np.asarray(directions, np.float64).reshape(-1, 3)
    if u.shape[0] != a.shape[0]:
        raise ValueError(f"one direction per amplitude: {u.shape[0]} directions, {a.shape[0]} amplitudes")
    G = a[:, None] * u                                                       # (n_meas, 3): the commanded plateau
    Lf = gradient_tensor_map(scanner, grid, to_scanner=R) if nonlinearity else None
    if Lf is None:
        q = np.broadcast_to(G[None], (len(d),) + G.shape).copy()
    else:
        q = np.einsum("vij,mj->vmi", Lf(pos), G)
    B0 = scanner.field_T
    if concomitant and B0:
        if not np.all(np.isclose(s, 0.0, atol=1e-6) | np.isclose(s, 1.0, atol=1e-6)):
            raise ValueError("the Maxwell gradient follows the coils' gradient sample by sample, and this shape is not "
                             "two-valued (ramps, or lobes of two levels), so its time course is not the shape's: it is "
                             "not one vector on the shape's moment. Replay this acquisition on the bridge route "
                             "(delivered_weights), or play the shape with square pulses of one polarity")
        from ..acquisition.scanner_sequence import concomitant_terms
        gc, _Bc = concomitant_terms(q, d[:, None, :], float(B0), _b0_axis(scanner, R))
        q = q + gc
    gmap = background_gradient_map(scanner, grid, to_scanner=R) if background else None
    g0 = None if gmap is None else np.atleast_2d(gmap(pos)).astype(np.float64)
    bmap = b1_scale_map(scanner, grid, to_scanner=R) if transmit else None
    kappa = None if bmap is None else np.atleast_1d(np.asarray(bmap(pos), np.float64))
    return MomentDelivery(q=q, g0=g0, kappa=kappa)
