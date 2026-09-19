"""A diffusion-prepared RF train replayed over a whole phantom, orientation distributions and all.

This is what :mod:`dmipy_sim.replay.pathways` is for. The vector-Bloch route needs one rotation per slot and
so refuses an ODF phantom (dmipy-sim#338); a train decomposed into microscopic gates does not, because each
gate is an ordinary phase sum and the pose expansion already carries those.

The cost model is the point. Building the gate expansions is the expensive part and it happens ONCE per
substrate; the transmit scale then enters as a re-weighting of coefficients already built, so a phantom
whose every voxel has its own flip angle costs one state propagation per distinct scale rather than one
magnetisation propagation per voxel. Binning the scales (``transmit_tolerance``) is what keeps that count
finite for a map a machine produced, which is smooth.
"""
from __future__ import annotations

import numpy as np

from ..constants import GAMMA

__all__ = ["replay_train"]


def replay_train(phantom, waveform, *, echo=-1, transmit=None, transmit_tolerance=1e-2,
                 off_resonance=None, scanner=None, pose=None, packs=None, proton_density=None,
                 keep=None, complex_signal=False, jax=None, report=None):
    """``(voxel_index, S)`` for an RF train over every voxel of ``phantom``, at one echo of the train.

    ``transmit`` is the per-voxel flip-angle scale -- a scalar, a volume, or a callable of scanner
    coordinates, exactly as :meth:`~dmipy_sim.phantom.Phantom.replay` takes one. It is BINNED to
    ``transmit_tolerance`` before use: the flip angle reaches the signal through a sine, so an error of
    ``tol`` in the scale is an error of the same order in the signal and never larger, and binning is what
    makes a smooth map cost a handful of re-weightings instead of one per voxel.

    ``off_resonance`` enters as it does on the ordinary route: a static offset dephasing through the
    acquisition's own coherence gate.
    """
    from .pathways import train_response
    from ..replay.phantom import transmit_classes
    from ..spec.tissue import Tissue

    f = phantom.file if hasattr(phantom, "file") else phantom
    if f.mode not in ("odf_sh", "peaks", "frames", "bingham"):
        raise ValueError(f"unknown orientation mode {f.mode!r}")

    R_s = None
    if pose is not None:
        from .replay import _pose_matrix
        R_s = _pose_matrix(pose)

    # a scalar, a volume or a callable of scanner coordinates, resolved to one value per voxel
    mapper = phantom._map if hasattr(phantom, "_map") else (lambda v, n: v)
    off_resonance = mapper(off_resonance, "off_resonance")
    kappa = mapper(transmit, "transmit")
    if kappa is None:
        kappa = np.ones(f.n_voxels)
    kappa = np.broadcast_to(np.asarray(kappa, np.float64), (f.n_voxels,))
    binned = transmit_classes(kappa, transmit_tolerance)
    scales = np.unique(binned)

    loaded = f._loaded_packs(packs)
    sid, frac = f.substrate_id, f.geometric_fraction
    m0 = f._m0(proton_density)

    # The band to expand at is the one the DISTRIBUTION retains, not the one the response reaches.
    # Composing is an inner product, so an ODF of order 8 cannot see a response's order 38 -- expanding
    # that far is work thrown away, and for a brain it is the difference between seconds and minutes.
    if keep is None:
        keep = (int(f.meta["orientation"].get("lmax", 8)), 0)

    # one set of gate expansions per substrate -- the expensive part, and it happens once
    trains = {}
    for i, sub in enumerate(f.substrates):
        if sub["kind"] != "pack" or i not in loaded:
            continue
        trains[i] = train_response(loaded[i], waveform, keep=keep,
                                   tissue=Tissue.from_meta(sub.get("tissue")), scanner=scanner, pose=R_s)
    if not trains:
        raise ValueError("a train replay needs at least one pack substrate")

    first = next(iter(trains.values()))
    probe = first.at(1.0, echo=echo)
    keep_l, keep_n = probe.lmax, probe.nmax
    vp, F = f.slot_coefficients(keep_l, keep_n)
    ids = sid[vp[:, 0], vp[:, 1]].astype(int)
    weight = frac[vp[:, 0], vp[:, 1]].astype(np.float64) * m0[vp[:, 0], ids]

    n_meas = probe.coeffs.shape[0]

    # Every (substrate, transmit scale) pair has its own coefficients, and every slot belongs to exactly
    # one pair -- so the whole phantom is ONE gather and one contraction, rather than a loop over pairs.
    pairs, coeff = {}, []
    for i, tr in trains.items():
        for scale in scales:
            pairs[(i, float(scale))] = len(coeff)
            coeff.append(np.asarray(tr.at(float(scale), echo=echo).retained(keep_l, keep_n), np.complex128))
    coeff = np.stack(coeff)                                   # (n_pairs, n_meas, n_feat)
    slot_scale = binned[vp[:, 0]]
    which = np.array([pairs.get((int(i), float(sc)), -1) for i, sc in zip(ids, slot_scale)])
    live = which >= 0

    if jax:
        S = _contract_jax(F[live], coeff, which[live], weight[live], vp[live, 0], f.n_voxels, n_meas)
    else:
        S = np.zeros((f.n_voxels, n_meas), np.complex128)
        part = np.einsum("sf,smf->sm", F[live].astype(np.complex128), coeff[which[live]])
        np.add.at(S, vp[live, 0], weight[live][:, None] * part)

    dB0 = f.layer_values("delta_B0_T", off_resonance)
    if dB0 is not None:
        S = S * np.exp(1j * GAMMA * dB0[:, None] * f.gate_integral(waveform))
    if report is not None:
        report.update(n_scales=len(scales), n_gates=first.n_gates, lmax=keep_l,
                      n_echoes=len(first.readouts))
    return f.voxel_index, (S if complex_signal else np.abs(S))


def _contract_jax(F, coeff, which, weight, voxel, n_voxels, n_meas):
    """The contraction and the scatter-add on an accelerator.

    Both halves are the shape a GPU is for: a gather of each slot's coefficients followed by one contraction
    over the orientation features, then a segment sum into voxels. Complex is carried as a real pair because
    a segment sum over complex is not uniformly supported.
    """
    import jax, jax.numpy as jnp
    Fj = jnp.asarray(np.ascontiguousarray(F), jnp.float32)
    Cr = jnp.asarray(np.ascontiguousarray(coeff.real), jnp.float32)
    Ci = jnp.asarray(np.ascontiguousarray(coeff.imag), jnp.float32)
    wj = jnp.asarray(np.ascontiguousarray(weight), jnp.float32)
    idx = jnp.asarray(np.ascontiguousarray(which), jnp.int32)
    vox = jnp.asarray(np.ascontiguousarray(voxel), jnp.int32)

    @jax.jit
    def go(Fj, Cr, Ci, wj, idx, vox):
        gr, gi = Cr[idx], Ci[idx]                                  # (n_slots, n_meas, n_feat)
        re = jnp.einsum("sf,smf->sm", Fj, gr) * wj[:, None]
        im = jnp.einsum("sf,smf->sm", Fj, gi) * wj[:, None]
        return (jax.ops.segment_sum(re, vox, num_segments=n_voxels),
                jax.ops.segment_sum(im, vox, num_segments=n_voxels))

    re, im = go(Fj, Cr, Ci, wj, idx, vox)
    return np.asarray(re, np.float64) + 1j * np.asarray(im, np.float64)
