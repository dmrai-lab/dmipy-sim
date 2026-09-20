"""What the machine does to a phantom, as a function of where each voxel sits in the bore.

A magnet's field is not uniform, and the way it departs from uniformity is a property of the MACHINE while
where the sample sits in it is a property of the GRID. This module is the one place the two meet: it renders
a catalogued field law (:meth:`~dmipy_sim.acquisition.scanners.ScannerLimits.b0_offset`) onto a grid and
hands back what a replay already accepts as ``off_resonance``.

Nothing here is a new kind of input. A phantom has taken a callable of scanner coordinates for its
macroscopic layers all along; this supplies one that a machine, rather than a person, is the author of.
"""
from __future__ import annotations

from itertools import product as _product

import numpy as np

__all__ = ["b0_offset_map", "b1_scale_map", "background_gradient_map", "delivered_b", "delivered_b_map", "b_polynomial", "gradient_tensor_map"]


def b0_offset_map(scanner, grid, *, to_scanner=None, delta_T_K=0.0):
    """A callable giving the static field's departure from uniformity, in tesla, at each voxel of ``grid``.

    Pass the result straight to ``off_resonance=`` of any :meth:`~dmipy_sim.phantom.Phantom.replay`; it has
    the signature a phantom already evaluates for a layer, ``f(positions_m) -> (n,)``.

    ``to_scanner`` is the rotation taking the grid's axes to the scanner's. It defaults to the grid's own
    (:attr:`~dmipy_sim.phantom.Grid.to_scanner`, which :meth:`~dmipy_sim.phantom.Grid.from_oblique_affine`
    sets), so an oblique grid is handled without the caller remembering -- which matters because forgetting
    does not fail. It silently evaluates the law at the wrong place, and for a law with a preferred
    direction, as a single-yoke magnet's is, it gets the sign of the asymmetry wrong over part of the
    volume. Pass it explicitly only to override what the grid says.

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
    drift = 0.0
    if delta_T_K:
        drift = getattr(scanner, "b0_drift", lambda _dT: None)(delta_T_K)
        if drift is None:
            raise ValueError(
                f"{getattr(scanner, 'name', scanner)!r} has no catalogued temperature coefficient, so a "
                f"drift of {delta_T_K} K cannot be rendered. A superconducting magnet has none because it "
                f"has no room temperature to drift with; this is refused rather than silently ignored")
        drift = float(drift)
    if getattr(scanner, "b0_harmonic_Z2", None) is None:
        if not drift:
            return None
        return lambda positions_m: np.full(np.asarray(positions_m, np.float64).reshape(-1, 3).shape[0], drift)
    iso = np.asarray(grid.isocenter_m, dtype=np.float64)
    R = grid.to_scanner if to_scanner is None else to_scanner
    R = None if R is None else np.asarray(R, dtype=np.float64)
    if R is not None:
        if R.shape != (3, 3):
            raise ValueError(f"to_scanner is the 3x3 rotation taking grid axes to scanner axes; got {R.shape}")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
            raise ValueError("to_scanner is not a rotation: a grid's axes are orthonormal in the scanner")

    def field(positions_m):
        d = np.asarray(positions_m, dtype=np.float64).reshape(-1, 3) - iso
        if R is not None:
            d = d @ R.T                      # the grid's frame into the bore's, where the law is stated
        return scanner.b0_offset(d) + drift

    return field


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
    if getattr(scanner, "b1_axial_falloff", None) is None and getattr(scanner, "b1_calibration_offset", None) is None:
        return None
    iso = np.asarray(grid.isocenter_m, dtype=np.float64)
    R = grid.to_scanner if to_scanner is None else to_scanner
    R = None if R is None else np.asarray(R, dtype=np.float64)

    def transmit(positions_m):
        d = np.asarray(positions_m, dtype=np.float64).reshape(-1, 3) - iso
        if R is not None:
            d = d @ R.T
        return scanner.b1_scale(d)

    return transmit


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
    if getattr(scanner, "b0_harmonic_Z2", None) is None:
        return None
    iso = np.asarray(grid.isocenter_m, dtype=np.float64)
    R = None if to_scanner is None else np.asarray(to_scanner, dtype=np.float64)
    if R is not None:
        if R.shape != (3, 3):
            raise ValueError(f"to_scanner is the 3x3 rotation taking grid axes to scanner axes; got {R.shape}")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
            raise ValueError("to_scanner is not a rotation: a grid's axes are orthonormal in the scanner")

    def gradient(positions_m):
        d = np.asarray(positions_m, dtype=np.float64).reshape(-1, 3) - iso
        if R is not None:
            d = d @ R.T                      # position: the grid's frame into the bore's
        g = np.atleast_2d(scanner.b0_gradient(d))
        if R is not None:
            g = g @ R                        # gradient: the bore's frame back into the grid's
        return g

    return gradient


def delivered_b(scanner, grid, sequence, *, to_scanner=None, voxels=None):
    """``(n_voxels, n_meas)`` -- the b value each voxel actually receives, against the one the sequence
    prescribes at isocentre.

    A magnet's own gradient encodes diffusion alongside the pulsed one, so what a voxel is measured at is
    not what was asked for. The ratio ``delivered_b / sequence.b()`` is the Swoop paper's ``a(r)``, and
    because an ADC fitted against the prescribed b absorbs the whole discrepancy, ``a - 1`` IS the
    fractional ADC error at that voxel -- which is how the paper's up-to-16.1 % figure is reproduced rather
    than asserted.

    The cross term is what dominates and what makes this a per-DIRECTION effect rather than a scale: it is
    linear in the background gradient, so it changes sign with the diffusion direction, and a symmetric
    direction set therefore has its mean error largely cancel while each individual measurement keeps its
    own. Reporting only the mean would hide the effect entirely.
    """
    gmap = background_gradient_map(scanner, grid, to_scanner=to_scanner)
    if gmap is None:
        return None
    idx = grid.every_voxel if voxels is None else voxels
    g = gmap(grid.offset_m(idx).reshape(-1, 3) + np.asarray(grid.isocenter_m, dtype=np.float64))
    out = np.empty((g.shape[0], sequence.n_meas), dtype=np.float64)
    for i, gv in enumerate(g):
        out[i] = sequence.with_background_gradient(gv).b()
    return out


# ── the delivered b over a whole grid, in one pass ──────────────────────────────────────────────────
#
# Both of the terms a magnet adds to an acquisition are AFFINE in position. The background gradient is
# `B0 (a xhat + 2 c r)`, affine by inspection. The concomitant field's gradient is `M(t) r` with `M`
# built from the coils' own G and nothing else -- also affine, and with no constant part, which is why
# it vanishes at isocentre where the background one does not.
#
# The b value is a quadratic functional of the effective gradient, so an affine dependence on position
# makes `b(r)` an exact QUADRATIC FORM: `b(r) = b0 + v . r + r^T A r`. Ten coefficients per measurement,
# whatever the grid. That is what turns an image from one sequence rebuild per voxel into ten.

#: the delivered b is a polynomial in position of this degree, exactly. The background gradient is the
#: gradient of a solid-harmonic field truncated at l=3, so it is QUADRATIC in position; the concomitant
#: gradient is linear. q is the time integral of both, so it is quadratic, and b integrates |q|^2 -- which
#: makes it quartic. A quadratic fit, which is what an l<=2 field law would have needed, leaves a residual
#: of five parts in ten thousand; a quartic one is exact to the precision G is stored in.
_POLY_DEGREE = 4


def _poly_exponents(degree=_POLY_DEGREE):
    return [e for e in _product(range(degree + 1), repeat=3) if sum(e) <= degree]


def _poly_features(r, degree=_POLY_DEGREE):
    """``(n, n_terms)``: every monomial in three variables up to ``degree``, in a fixed order."""
    r = np.asarray(r, dtype=np.float64).reshape(-1, 3)
    return np.stack([r[:, 0] ** i * r[:, 1] ** j * r[:, 2] ** k
                     for i, j, k in _poly_exponents(degree)], axis=1)


def _probe_points(radius, degree=_POLY_DEGREE):
    """Positions that determine a polynomial of this degree: a deterministic quasi-lattice, sized to the
    number of coefficients and conditioned well enough to solve exactly rather than in least squares."""
    n = len(_poly_exponents(degree))
    rng = np.random.default_rng(20260920)
    best, best_cond = None, np.inf
    for _ in range(40):
        P = rng.uniform(-1.0, 1.0, size=(n, 3))
        P[0] = 0.0
        F = _poly_features(P * radius, degree)
        c = np.linalg.cond(F)
        if c < best_cond:
            best, best_cond = P.copy(), c
    return best * radius


def b_polynomial(played_at, *, probe_radius=0.05, degree=_POLY_DEGREE):
    """``(n_meas, n_terms)`` -- the coefficients of the exact polynomial ``b(r)`` that ``played_at`` produces.

    ``played_at(r)`` returns the acquisition as it is actually played at one position. The dependence IS
    polynomial of this degree, so evaluating at exactly as many well-conditioned points as there are
    coefficients recovers it exactly rather than approximately, and :func:`delivered_b_map` then costs a
    matrix product per grid instead of a sequence rebuild per voxel.

    Solving for the coefficients rather than deriving them in closed form is deliberate. The b integral has a
    quadrature convention, and a second implementation of it here would be a second thing to keep true.
    Probing uses the acquisition's own :meth:`b`, so the batched answer cannot drift from the exact one;
    :func:`delivered_b` remains the oracle that says so.
    """
    probes = _probe_points(float(probe_radius), degree)
    F = _poly_features(probes, degree)
    if np.linalg.matrix_rank(F) < F.shape[1]:
        raise ValueError(f"the probe set does not determine a degree-{degree} polynomial")
    B = np.stack([np.asarray(played_at(p).b(), dtype=np.float64) for p in probes])
    return np.linalg.solve(F, B).T


def delivered_b_map(scanner, grid, sequence, *, to_scanner=None, voxels=None,
                    background=True, concomitant=True, probe_radius=0.05, report=None):
    """``(n_voxels, n_meas)`` -- the b every voxel actually receives, in ONE pass over the grid.

    Two separate things a magnet does to a diffusion measurement, and they are different physics even
    though they arrive the same way:

    ``background`` is the magnet's OWN field gradient, which is constant in time and non-zero at isocentre
    (a single-yoke magnet has an odd term that survives differentiation). Its cross term with the pulsed
    gradient flips sign with the diffusion direction.

    ``concomitant`` is the gradient coils' Maxwell term, which is quadratic in ``G(t)`` and therefore varies
    through the sequence and does NOT flip when the coils reverse. It is exactly zero at isocentre and
    scales as ``1 / B0``, which is what makes it a low-field problem rather than a clinical one.

    Both are affine in position, so the delivered b is an exact quadratic form and the whole grid costs ten
    probe evaluations. ``None`` when the machine publishes neither.
    """
    gmap = background_gradient_map(scanner, grid, to_scanner=to_scanner) if background else None
    B0 = getattr(scanner, "field_T", None)
    if gmap is None and not (concomitant and B0):
        return None

    R = None if to_scanner is None else np.asarray(to_scanner, dtype=np.float64)

    def played_at(r_bore):
        """The acquisition as played at one point, stated in the BORE's frame -- which is the frame both
        the field law and the Maxwell formula are written in."""
        seq = sequence
        if gmap is not None:
            g = np.atleast_2d(scanner.b0_gradient(np.asarray(r_bore, np.float64)[None]))[0]
            seq = seq.with_background_gradient(g)
        if concomitant and B0:
            seq = seq.with_concomitant(np.asarray(r_bore, np.float64), B0)
        return seq

    coeff = b_polynomial(played_at, probe_radius=probe_radius)

    idx = grid.every_voxel if voxels is None else voxels
    d = grid.offset_m(idx).reshape(-1, 3)
    if R is not None:
        d = d @ R.T                                    # the grid's frame into the bore's
    out = _poly_features(d) @ coeff.T             # (n_vox, n_meas)
    if report is not None:
        report.update(n_probes=len(_poly_exponents()), n_voxels=d.shape[0],
                      background=gmap is not None, concomitant=bool(concomitant and B0))
    return out


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
    if getattr(scanner, "d_scale_y_dx", None) is None:
        return None
    iso = np.asarray(grid.isocenter_m, dtype=np.float64)
    R = None if to_scanner is None else np.asarray(to_scanner, dtype=np.float64)
    if R is not None:
        if R.shape != (3, 3):
            raise ValueError(f"to_scanner is the 3x3 rotation taking grid axes to scanner axes; got {R.shape}")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
            raise ValueError("to_scanner is not a rotation: a grid's axes are orthonormal in the scanner")

    def tensor(positions_m):
        d = np.asarray(positions_m, dtype=np.float64).reshape(-1, 3) - iso
        if R is not None:
            d = d @ R.T
        L = np.atleast_3d(scanner.gradient_tensor(d)).reshape(-1, 3, 3)
        if R is not None:
            L = np.einsum("ji,njk,kl->nil", R, L, R)          # R^T L R, the similarity transform
        return L

    return tensor


def delivered_gradient(scanner, grid, sequence, *, voxels=None, to_scanner=None,
                       nonlinearity=True, background=True, concomitant=True):
    """The gradient each voxel ACTUALLY receives: ``(n_voxels, n_meas, n_t, 3)`` in the grid's frame.

    Three things stand between the gradient a sequence prescribes and the one a spin at ``r`` sees, and all
    three are already modelled elsewhere in this package without ever reaching a signal (dmipy-sim#369):

    * the coils' **nonlinearity**, ``g -> L(r) g``, which mis-scales and TILTS the encoding direction;
    * the magnet's own **background** gradient, constant in time and non-zero at isocentre, which the RF
      sign carries like any other gradient and whose contribution to ``b`` does not refocus;
    * the coils' **concomitant** (Maxwell) term, whose extra encoding gradient is quadratic in ``G(t)`` --
      so it varies through the sequence and does not reverse when the coils do.

    This is the reference route: it builds the acquisition as played at each voxel through
    :meth:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence.with_background_gradient` and
    ``with_concomitant``, which is exact and costs a sequence per voxel. :func:`delivered_weights` is the
    same thing in the replay's own coefficient space, at a few thousand flops per voxel, and the two are
    held against each other by test.
    """
    idx = grid.every_voxel if voxels is None else voxels
    d = grid.offset_m(idx).reshape(-1, 3)
    R = None if to_scanner is None else np.asarray(to_scanner, dtype=np.float64)
    if R is not None:
        d = d @ R.T                                          # the grid's frame into the bore's
    B0 = getattr(scanner, "field_T", None)
    L = gradient_tensor_map(scanner, grid, to_scanner=to_scanner) if nonlinearity else None
    out = np.empty((len(d), sequence.n_meas, sequence.G.shape[1], 3), dtype=np.float64)
    Ls = None if L is None else L(grid.positions_m(idx))
    for k, r in enumerate(d):
        seq = sequence
        if Ls is not None:
            seq = seq.with_gradient_nonlinearity(Ls[k])
        if background and getattr(scanner, "b0_harmonic_Z2", None) is not None:
            seq = seq.with_background_gradient(np.atleast_2d(scanner.b0_gradient(r[None]))[0])
        if concomitant and B0:
            seq = seq.with_concomitant(r, float(B0), b0_axis=_b0_axis(scanner))
        out[k] = seq.G
    return out


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
    * the concomitant term's extra gradient is quadratic in ``G(t)`` but LINEAR IN POSITION (``B_c`` is
      quadratic in ``r``), so three column projections done once combine by the voxel's own coordinates.

    None of that is an approximation: it is the same sum in a different order, which is why the reference
    route exists to check it rather than to be replaced by it.
    """
    from ..replay.replay import _compile_effective
    from ..acquisition.rf import RFSchedule                      # noqa: F401  (documents where sign lives)

    idx = grid.every_voxel if voxels is None else voxels
    pos = grid.positions_m(idx)
    d = grid.offset_m(idx).reshape(-1, 3)
    R = None if to_scanner is None else np.asarray(to_scanner, dtype=np.float64)
    d_bore = d if R is None else d @ R.T
    B0 = getattr(scanner, "field_T", None)

    def project(seq_like_G):
        """W for a physical G, carried through the RF sign exactly as the replay does."""
        return _compile_effective(_effective(sequence, seq_like_G), dt_pack, K, n_t)

    W = project(np.asarray(sequence.G, dtype=np.float64))                       # ((K+2)*3, n_meas)
    n_c = (K + 2)
    out = np.repeat(W[None], len(d), axis=0)

    if nonlinearity:
        Lf = gradient_tensor_map(scanner, grid, to_scanner=to_scanner)
        if Lf is not None:
            base = W.reshape(n_c, 3, -1)
            out = np.einsum("nij,kjm->nkim", Lf(pos), base).reshape(len(d), n_c * 3, -1)

    if background and getattr(scanner, "b0_harmonic_Z2", None) is not None:
        unit = [project(np.broadcast_to(e, sequence.G.shape)).reshape(n_c, 3, -1) for e in np.eye(3)]
        g0 = scanner.b0_gradient(d_bore)
        if R is not None:
            g0 = g0 @ R                                                        # back into the grid's frame
        out = out + np.einsum("ni,ikjm->nkjm", g0, np.stack(unit)).reshape(len(d), n_c * 3, -1)

    if concomitant and B0:
        linear_only = (not nonlinearity) and not (background and getattr(scanner, "b0_harmonic_Z2", None))
        if concomitant == "linearised" or linear_only:
            # The concomitant's extra gradient is quadratic in G(t) but LINEAR IN POSITION, so three column
            # projections done once combine by the voxel's own coordinates. Exact when nothing else has
            # changed the gradient -- and an APPROXIMATION when L or a background is also on, because the
            # Maxwell term is then quadratic in the DELIVERED gradient and its cross terms are per voxel.
            # Measured on this magnet over a 9 cm grid: 2.8 per cent of the concomitant term, 0.09 per cent
            # of G. Cheap and bounded, but not the default.
            cols = [project(sequence.with_concomitant(e, float(B0), b0_axis=_b0_axis(scanner)).G
                            - np.asarray(sequence.G, dtype=np.float64)).reshape(n_c, 3, -1)
                    for e in np.eye(3)]
            out = out + np.einsum("ni,ikjm->nkjm", d_bore, np.stack(cols)).reshape(len(d), n_c * 3, -1)
        else:
            # The Maxwell term is quadratic in the gradient the COILS deliver, so it reads L(r) G rather
            # than the nominal G. The magnet's background is left out, and the honest reason is a SIZE and
            # not a principle: div B = 0 forces transverse components on the magnet's inhomogeneity too,
            # and those beat against the coils' in |B| to give a real cross term, linear in G(t). Measured
            # on this machine over the 8 cm validity sphere it is 2-10 per cent of the modelled concomitant
            # gradient and 1.7e-3 of b -- smaller than the concomitant term itself and far smaller than the
            # background's own 22 per cent, but not zero. Carrying it properly needs the magnet's TRANSVERSE
            # field, which the catalogue does not record; feeding g0 to a formula written for a coil is a
            # different wrong answer, not a better one.
            coil_G = delivered_gradient(scanner, grid, sequence, voxels=voxels, to_scanner=to_scanner,
                                        nonlinearity=nonlinearity, background=False, concomitant=False)
            for k in range(len(d)):
                out[k] = out[k] + project(_concomitant_of(sequence, coil_G[k], d_bore[k], float(B0),
                                          b0_axis=_b0_axis(scanner)))

    return out


def _b0_axis(scanner):
    """The machine's field direction, defaulting to the bore axis the concomitant formula assumes."""
    v = getattr(scanner, "b0_axis", None)
    return (0.0, 0.0, 1.0) if v is None else v


def _concomitant_of(sequence, G_delivered, position_m, B0_T, b0_axis=(0.0, 0.0, 1.0)):
    """The concomitant extra gradient of the gradient the COILS deliver, as a physical ``G`` increment.

    ``G_delivered`` is ``L(r) G``. Substituting the nominal ``G`` drops the cross terms between the coils'
    nonlinearity and their own concomitant field; substituting ``L G`` is the best LOCAL approximation and
    not the exact field of a nonlinear coil, whose transverse completion depends on its whole harmonic
    expansion rather than on the gradient at one point (measured residual ~2 per cent of the term, against
    ~3 per cent for the nominal gradient).

    The magnet's background is deliberately absent, for a reason of SIZE rather than principle: it does
    produce a genuine cross term (1.7e-3 of b here), but carrying it needs the magnet's transverse field,
    which the catalogue does not record.
    """
    from dataclasses import replace as _replace
    played = _replace(sequence, G=np.asarray(G_delivered, dtype=np.float64))
    return (played.with_concomitant(position_m, B0_T, b0_axis=b0_axis).G
            - np.asarray(G_delivered, dtype=np.float64))


def _effective(sequence, G):
    """``G_eff`` for a physical ``G`` under this sequence's RF schedule -- the sign the replay applies."""
    sign = sequence.rf.sign(np.arange(np.asarray(G).shape[1]) * sequence.dt)
    return np.asarray(G, dtype=np.float64) * np.asarray(sign, dtype=np.float64)[None, :, None]
