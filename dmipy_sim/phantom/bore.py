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

__all__ = ["b0_offset_map", "b1_scale_map", "background_gradient_map", "delivered_b"]


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
