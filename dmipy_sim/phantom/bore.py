"""What the machine does to a phantom, as a function of where each voxel sits in the bore.

A magnet's field is not uniform, and the way it departs from uniformity is a property of the MACHINE while
where the sample sits in it is a property of the GRID. This module is the one place the two meet: it renders
a catalogued field law (:meth:`~dmipy_sim.acquisition.scanners.ScannerLimits.b0_offset`) onto a grid and
hands back what a replay already accepts as ``off_resonance``.

Nothing here is a new kind of input. A phantom has taken a callable of scanner coordinates for its
macroscopic layers all along; this supplies one that a machine, rather than a person, is the author of.
"""
from __future__ import annotations

import numpy as np

__all__ = ["b0_offset_map", "b1_scale_map", "background_gradient_map", "delivered_b", "b_quadratic", "delivered_b_map"]


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
    if getattr(scanner, "b0_quadratic", None) is None:
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
    if getattr(scanner, "b0_quadratic", None) is None:
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

_QUADRATIC_TERMS = 10          # 1 + 3 linear + 6 symmetric-quadratic


def _quadratic_features(r):
    """``(n, 10)``: the monomials of a general quadratic in three variables, in the order the coefficients
    are solved for."""
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    one = np.ones_like(x)
    return np.stack([one, x, y, z, x * x, y * y, z * z, x * y, x * z, y * z], axis=1)


def b_quadratic(played_at, *, probe_radius=0.05):
    """``(n_meas, 10)`` -- the coefficients of the exact quadratic ``b(r)`` that ``played_at`` produces.

    ``played_at(r)`` returns the acquisition as it is actually played at one position; this evaluates it at
    ten probe positions and solves for the quadratic they determine. Ten is not a sampling: the dependence
    IS quadratic, so ten well-placed points recover it exactly rather than approximately, and
    :func:`delivered_b_map` then costs a matrix product per grid instead of a rebuild per voxel.

    Solving for the coefficients rather than deriving them in closed form is deliberate. The b integral has
    a quadrature convention -- rectangular q, trapezoidal in time -- and a second implementation of it here
    would be a second thing to keep true. Probing uses the acquisition's own :meth:`b`, so the batched
    answer cannot drift from the exact one; :func:`delivered_b` remains the oracle that says so.
    """
    # a well-conditioned probe set: the origin, +-one radius on each axis, and three diagonal points that
    # pin the cross terms. Deliberately not random -- the solve should be reproducible.
    u = float(probe_radius)
    probes = np.array([[0.0, 0.0, 0.0],
                       [u, 0, 0], [-u, 0, 0], [0, u, 0], [0, -u, 0], [0, 0, u], [0, 0, -u],
                       [u, u, 0], [u, 0, u], [0, u, u]], dtype=np.float64)
    F = _quadratic_features(probes)
    if np.linalg.matrix_rank(F) < _QUADRATIC_TERMS:
        raise ValueError("the probe set does not determine a quadratic; probe_radius must be non-zero")
    B = np.stack([np.asarray(played_at(p).b(), dtype=np.float64) for p in probes])   # (10, n_meas)
    return np.linalg.solve(F, B).T                                                   # (n_meas, 10)


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

    coeff = b_quadratic(played_at, probe_radius=probe_radius)

    idx = grid.every_voxel if voxels is None else voxels
    d = grid.offset_m(idx).reshape(-1, 3)
    if R is not None:
        d = d @ R.T                                    # the grid's frame into the bore's
    out = _quadratic_features(d) @ coeff.T             # (n_vox, n_meas)
    if report is not None:
        report.update(n_probes=_QUADRATIC_TERMS, n_voxels=d.shape[0],
                      background=gmap is not None, concomitant=bool(concomitant and B0))
    return out
