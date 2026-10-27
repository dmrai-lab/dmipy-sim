"""Replay-pack assembler — walk once, compress, self-certify, freeze into a ``.rpk``.

:func:`build_replay_pack` turns a **master walk** (from
:func:`dmipy_sim.engine.core.simulate_trajectories`) into a compressed,
self-describing replay pack: the position ensemble is compressed by a :mod:`dmipy_sim.replay.compression`
codec (``bridge_dst``), the tier channels the certified envelope needs are carried
(bulk relaxation via the compartment map, surface relaxivity via the boundary-local-time channel,
magnetization transfer via the bound-fraction channel), and the pack MEASURES its own replay
fidelity against the split-half Monte-Carlo floor before it is written. The result is a single
``safetensors`` file readable by :func:`dmipy_sim.replay.replay.read_rpk` and replayable by the
:mod:`dmipy_sim.replay.replay` / :mod:`dmipy_sim.replay.trajectories` operators.

:func:`build_to_floor` is the bank's default generation policy: size the walker count so the
Monte-Carlo floor meets a target ``sigma_star``, then build a pack whose codec error meets it too.

This is the *producer* side of the substrate bank. Publishing/pulling packs to a HuggingFace
dataset (the federation layer) lives separately. The susceptibility (field) replay tier needs the
per-walker susceptibility-basis channel and its ``Q(H)`` contraction (a separate module) and is not
assembled here yet.
"""
from __future__ import annotations
import functools
import json
import logging

log = logging.getLogger(__name__)

import numpy as np

from ..run import Run, _code
from ..engine.accumulate import FIELD_RULE_CHI, FIELD_RULE_B0, field_read_term, field_read_B0_chi_max, field_rule_scale

from . import compression as _cx
from ..acquisition.rf import RFEvent
from .replay import ReplayPack, read_rpk, write_rpk
from ..fields.hollow_cylinder import CHANNEL_NAMES as _PATH_CHANNELS, ISO_CHANNEL_NAMES as _ISO_PATH_CHANNELS
from ..spec.substrate import susceptibility_field_of

__all__ = ["build_replay_pack", "build_to_floor", "frame_from_axis", "frame_from_bundles", "frame_of_spec", "check_frame_against_walk",
           "check_frame_against_geometry", "read_rpk", "write_rpk", "RPK_SCHEMA_VERSION"]

RPK_SCHEMA_VERSION = "0.7"


# --------------------------------------------------------------- master-walk normalisation
def seed_value(seed):
    """A walk's seed as provenance: an int for a walk, the list of shard seeds for a pack merged from a fill.

    A merged pack has no single seed, so the list is what its provenance carries and what every derived
    pack (a prefix, a re-encoding) carries on."""
    if seed is None:
        return 0
    if isinstance(seed, (list, tuple, np.ndarray)):
        return [int(v) for v in np.asarray(seed).reshape(-1)]
    return int(seed)


# --------------------------------------------------------------- substrate frames
def frame_from_axis(axis, *, in_plane=None):
    """Deterministic orthonormal substrate frame R (3x3, columns [x, y, z]) with z = `axis`
    (the primary fibre direction) and a FIXED perpendicular x/y basis (Gram-Schmidt seeded
    from the global axis least aligned with z). A single fibre vector leaves a free rotation
    about itself; this pins x/y so directions are reproducible run-to-run and gradient schemes
    are oriented unambiguously. For an isotropic substrate any axis works — the frame is still
    fixed, giving uniform behaviour across packs. ``in_plane`` (a spec's ``frame.in_plane``) pins
    ``y`` instead: its component perpendicular to ``z``, the direction a secondary bundle opens
    into (RPK.md 4.2), and ``x = y × z``."""
    z = np.asarray(axis, float); z = z / np.linalg.norm(z)
    if in_plane is not None:
        y = np.asarray(in_plane, float); y = y - z * float(y @ z)
        if np.linalg.norm(y) < 1e-9:
            raise ValueError("in_plane is parallel to the axis")
        y /= np.linalg.norm(y)
        return np.column_stack([np.cross(y, z), y, z])
    seed = np.eye(3)[int(np.argmin(np.abs(z)))]        # global axis least aligned with z
    x = seed - z * float(seed @ z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def frame_of_spec(spec):
    """The substrate frame a spec declares, as the 3x3 basis: ``frame.axis`` (and ``frame.in_plane`` when
    given) through :func:`frame_from_axis`."""
    fr = getattr(spec, "frame", None)
    if fr is None:
        return None
    return frame_from_axis(fr.axis, in_plane=getattr(fr, "in_plane", None))


def check_frame_against_walk(traj, F, *, w=None, bundle_axes=None, tol_deg=5.0, anisotropy=1.1):
    """Refuse a HAND-DECLARED substrate frame the walk contradicts (RPK.md 4.2, dmipy-sim#194): the principal axis
    of the walkers' end-to-end displacements must lie within ``tol_deg`` of the span of the declared bundle
    axes (``bundle_axes``, default the frame's ``z``). A walk whose displacement covariance has no dominant
    axis declares nothing and passes: dominant means an eigenvalue ratio above ``anisotropy`` AND above the
    spread finite sampling gives an isotropic walk -- each eigenvalue of an isotropic sample covariance
    fluctuates by ``sqrt(2 / n_eff)``, the ratio of the largest to the next by about ``2.4 / sqrt(n_eff)`` at
    one sigma, so ``1 + 10 / sqrt(n_eff)`` is the four-sigma allowance the angle tolerance also uses -- so a
    small walk of free water is not read as oriented by its noise (113 free walkers gave a ratio of 1.56 on
    one platform's realisation and passed on another's). A walk with two comparable axes is checked against the plane only when
    two or more bundles are declared. Returns the angle (degrees)."""
    X = np.asarray(traj)                                          # only the two endpoints of each walker are read
    d = np.asarray(X[:, -1, :], np.float64) - np.asarray(X[:, 0, :], np.float64)
    w = np.ones(d.shape[0]) if w is None else np.asarray(w, np.float64)
    ok = np.isfinite(d).all(1) & np.isfinite(w) & (w > 0)        # a walker with no position at the end says nothing
    d, w = d[ok], w[ok]
    if d.shape[0] < 4:
        return 0.0
    d = d - (w[:, None] * d).sum(0) / w.sum()
    C = (d * w[:, None]).T @ d / w.sum()
    tr = float(np.trace(C))
    if tr <= 0:
        return 0.0                                                # nobody moved
    lam, V = np.linalg.eigh(C / tr)                               # ascending; unit trace keeps LAPACK away from
    lam, V = lam[::-1], V[:, ::-1]                                # its tolerance floor at 1e-12 m^2
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    dominant = max(float(anisotropy), 1.0 + 10.0 / np.sqrt(max(n_eff, 1.0)))
    if lam[1] <= 0 or lam[0] / lam[1] < dominant:
        if lam[2] <= 0 or lam[1] / lam[2] < dominant or bundle_axes is None or len(bundle_axes) < 2:
            return 0.0                                            # no dominant axis: nothing to contradict
        probe = V[:, :2]                                          # a plane of two comparable axes
    else:
        probe = V[:, :1]
    F = np.asarray(F, np.float64).reshape(3, 3)
    B = np.asarray(bundle_axes if bundle_axes is not None else [F[:, 2]], np.float64)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    Q, _ = np.linalg.qr(B.T)                                      # an orthonormal basis of the declared span
    Q = Q[:, :np.linalg.matrix_rank(B)]
    worst = 0.0
    for k in range(probe.shape[1]):
        v = probe[:, k]
        r = v - Q @ (Q.T @ v)
        worst = max(worst, float(np.degrees(np.arcsin(min(1.0, np.linalg.norm(r))))))
    # the principal axis of n_eff samples is itself uncertain by ~ sqrt(l1 l2) / (l1 - l2) / sqrt(n_eff) radians
    # (Anderson): a small walk is refused only beyond four of those, never for its own sampling noise
    k = probe.shape[1]
    sigma = np.degrees(np.sqrt(lam[k - 1] * lam[k]) / max(lam[k - 1] - lam[k], 1e-300) / np.sqrt(n_eff))
    if worst > max(float(tol_deg), 4.0 * float(sigma)):
        raise ValueError(f"the walk's principal displacement axis {np.round(V[:, 0], 3).tolist()} is {worst:.1f} deg from "
                         f"the declared substrate frame (axis {np.round(F[:, 2], 3).tolist()}"
                         f"{'' if bundle_axes is None else f', bundles {np.round(B, 3).tolist()}'}): the frame does not "
                         f"describe this substrate (RPK.md 4.2). Declare it from the substrate's own structure -- "
                         f"substrate_frame=frame_from_axis(axis) or the spec's frame -- not from a guess")
    return worst


def check_frame_against_geometry(sub, F, *, tol_deg=5.0):
    """Refuse a STRUCTURAL substrate frame (``frame.source == "structural"``, dmipy-sim#538) that disagrees with
    the shape it was derived from: the principal axis of the length-weighted tangent dyadic of every
    ``swept_polyline`` wall's centerline(s) in the spec ``sub`` must lie within ``tol_deg`` of the frame's ``z``.
    This reads only the spec's wall geometry, never a walk's trajectory, so it is cheap and needs no walk at all
    -- the right test for a frame :func:`dmipy_sim.spec.build.spec_of` derived from the substrate's own structure,
    where :func:`check_frame_against_walk` is the wrong one: a curved tube's walkers' principal displacement axis
    depends on the walk duration and the finite tube length, and need not agree with the structural axis at any
    tolerance. Returns the angle (degrees), or 0.0 when ``sub`` declares no ``swept_polyline`` wall to check
    against."""
    centerlines = []
    for w in sub.get("walls") or []:
        surf = (w.get("surface") or {}) if isinstance(w, dict) else {}
        if surf.get("kind") != "swept_polyline":
            continue
        if surf.get("centerline") is not None:
            centerlines.append(np.asarray(surf["centerline"], np.float64))
        for cl in (surf.get("instances") or {}).get("centerlines") or []:
            centerlines.append(np.asarray(cl, np.float64))
    D = np.zeros((3, 3))
    for cl in centerlines:
        if cl.ndim != 2 or cl.shape[0] < 2:
            continue
        t = np.diff(cl, axis=0)
        length = np.linalg.norm(t, axis=1)
        ok = length > 0
        if not ok.any():
            continue
        u = t[ok] / length[ok, None]
        D += (u * length[ok, None]).T @ u                             # sum_i length_i * outer(u_i, u_i)
    tr = float(np.trace(D))
    if tr <= 0:
        return 0.0                                                    # no swept_polyline wall: nothing to check
    lam, V = np.linalg.eigh(D / tr)
    axis = V[:, -1]
    z = np.asarray(F, np.float64).reshape(3, 3)[:, 2]
    z = z / np.linalg.norm(z)
    angle = float(np.degrees(np.arccos(np.clip(abs(float(axis @ z)), 0.0, 1.0))))
    if angle > float(tol_deg):
        raise ValueError(f"the substrate's own length-weighted tangent axis {np.round(axis, 3).tolist()} is "
                         f"{angle:.1f} deg from the declared structural frame (axis {np.round(z, 3).tolist()}): "
                         f"this frame was derived from the substrate's own shape by spec_of, and now disagrees "
                         f"with it -- it was edited by hand after. Rebuild it from the geometry (spec_of), do "
                         f"not hand-edit frame.axis on a structural frame")
    return angle


def frame_from_bundles(axes, *, primary=0, weights=None, tol=1e-3):
    """Substrate frame for a MULTI-bundle (e.g. crossing) substrate, anchored to a PRIMARY bundle.

    Do NOT use a PCA over all positions here: for two crossing bundles PCA returns the *bisector*,
    which is no real bundle's axis and rotates as the crossing angle opens. Instead pin the primary
    bundle (z) and let the most-transverse secondary define the plane (-> +y); x = y x z. With one
    bundle / all parallel it degrades to :func:`frame_from_axis`. `axes` is (n_bundles, 3) of
    per-bundle mean tangents (NOT a vertex PCA)."""
    A = np.asarray(axes, float)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    if primary is None:
        primary = int(np.argmax(weights)) if weights is not None else 0
    z = A[primary]
    perp = A - (A @ z)[:, None] * z[None, :]
    mag = np.linalg.norm(perp, axis=1); mag[primary] = 0.0
    j = int(np.argmax(mag))
    if mag[j] < tol:
        return frame_from_axis(z)
    y = perp[j] / mag[j]
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


# --------------------------------------------------------------- fidelity helpers
def _envelope_summary(env):
    return dict(b_max=float(max(env["bvals"])), ogse_periods=list(env.get("ogse_periods", [])),
                B0_list=list(env.get("B0_list", [])),
                note="temporal band set by max OGSE period / min delta")


#: rho2 / D values a window's contact envelope is read on: log-spaced from 1 m^-1 to the battery's largest
SURFACE_ENVELOPE_POINTS = 256


def surface_envelope(ell, w, *, rho2_over_D_hi, reference=None, tol=2.0, n_grid=SURFACE_ENVELOPE_POINTS):
    """The contact tier's envelope on one window: the largest ``rho2 / D`` up to which the window's decoded contact
    ``ell`` ``(n_w, n_t)`` serves a physical signal within its certificate. ``(edge, err, floor)``.

    On a log grid from 1 m^-1 to ``rho2_over_D_hi``, at every point: the floor is the standard deviation of the
    split-half difference of the ungated attenuation ``<w exp(rho2/D s)>``, ``2 sd_w / sqrt(n_eff)`` (``s`` the
    window's summed contact; the reference's when one is given) -- the spread of the certificate's split-half
    floor rather than one draw of it, so that the edge does not move with the luck of one split; the error is the
    larger of the attenuation's distance from the reference's (``reference``, the same window's per-walker summed
    contact from the walk the channel was encoded from) and the unphysical gain ``<w (exp(rho2/D r) - 1)>``, with
    ``r`` each walker's largest rise of the cumulative contact over any stretch of the window -- a contact only
    ever lowers the weight, so a rise is band ripple, and it is what a gate over that stretch would turn into a
    weight above one. The edge is the largest grid point below which every point is finite, has its error within
    ``tol`` floors and its gain within one floor; ``err`` and ``floor`` are the largest of each up to it. 0 when
    even the first point fails."""
    ell = np.asarray(ell, np.float64)
    w = np.asarray(w, np.float64)
    n_w = ell.shape[0]
    B = np.cumsum(ell[:, 1:], axis=1)                              # the window's first save ends no step of it
    B = np.concatenate([np.zeros((n_w, 1)), B], axis=1)
    s = B[:, -1]
    rise = np.max(B - np.minimum.accumulate(B, axis=1), axis=1)
    s_ref = s if reference is None else np.asarray(reference, np.float64)
    p = w / w.sum()
    n_eff = 1.0 / float(np.sum(p ** 2))
    edge = err = floor = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        for rd in np.geomspace(1.0, float(rho2_over_D_hi), int(n_grid)):
            a_ref = np.exp(rd * s_ref)
            m_ref = float(p @ a_ref)
            fl = 2.0 * float(np.sqrt(p @ (a_ref - m_ref) ** 2)) / np.sqrt(n_eff)
            gain = float(p @ np.expm1(rd * rise))
            e = max(abs(m_ref - float(p @ np.exp(rd * s))), gain)
            if not (np.isfinite(fl) and np.isfinite(e)) or e > tol * fl or gain > fl:
                break
            edge, err, floor = float(rd), max(err, e), max(floor, fl)
    return edge, float(err), float(floor)


def restate_maxima(fid):
    """A certificate's ``err_max`` / ``floor_max`` re-read from its terms -- every ``err_*`` / ``floor_*`` row and the
    positions' per-family rows -- and ``within_2x_floor`` from them."""
    errs = [v for k, v in fid.items() if k.startswith("err_") and k != "err_max" and isinstance(v, (int, float))]
    floors = [v for k, v in fid.items() if k.startswith("floor_") and k != "floor_max" and isinstance(v, (int, float))]
    for f in (fid.get("per_family") or {}).values():
        errs.append(f["err_max"]); floors.append(f["floor_max"])
    if errs:
        fid["err_max"] = float(max(errs))
    if floors:
        fid["floor_max"] = float(max(floors))
    if "err_max" in fid and "floor_max" in fid:
        fid["within_2x_floor"] = bool(fid["err_max"] <= 2.0 * fid["floor_max"])
    return fid


def _census_rows(ijk_all, inside):
    """The occupied voxels (unique rows of ``ijk_all[inside]``) and every walker's row into them, ``-1`` outside:
    the per-(voxel, pool) census as one vectorised ``np.unique`` inverse lookup -- no per-walker Python dict probe
    (``ijk_all`` of 157k walkers cost 1.2 s that way)."""
    n_w = ijk_all.shape[0]
    row = np.full(n_w, -1, np.int64)
    if not inside.any():
        return np.zeros((0, ijk_all.shape[1]), ijk_all.dtype), row
    ijk, inv = np.unique(ijk_all[inside], axis=0, return_inverse=True)
    row[inside] = inv.reshape(-1)
    return ijk, row


def _walker_coeffs(pieces, bounds, start, stop):
    """``(stop - start, K + 2, n_axes)`` float64: the coefficients of walkers ``start:stop`` of a walk held as
    consecutive ``pieces`` (position tensors; ``bounds`` their cumulative walker counts), decoded from each piece it
    spans."""
    out = []
    for p, lo, hi in zip(pieces, bounds[:-1], bounds[1:]):
        if lo < stop and hi > start:
            out.append(_cx.read_position_coeffs(_cx.walker_rows(p, max(start, lo) - lo, min(stop, hi) - lo), dtype=np.float64))
    return out[0] if len(out) == 1 else np.concatenate(out)


def voxel_floor_coded(positions, dt, n_t, grid, comp, env, *, w=None, device="auto", chunk=None):
    """The per-(voxel, pool) split-half floor of a walk meant to be partitioned, from its bridge coefficients over the
    window of ``n_t`` block ends ``dt`` apart (:func:`voxel_fidelity` without the dense oracle: the same binning by start position -- the exact first
    endpoint -- and pool (``comp`` a track, its first sample, or one label per walker), the same fixed split, the
    phases by :func:`compression.coded_phases`). ``positions`` are the walk's position tensors (a pack's arrays), or a
    list of them for a walk held in consecutive pieces (a merge's shards, in walker order); ``chunk`` walkers are
    decoded at a time (default :func:`~dmipy_sim.replay.compression.slice_walkers`). Returns ``(ijk, pools, n, floor,
    err)`` with ``err`` all ``nan``: the codec error is the certifying pack's, or the walk-free certificate's."""
    pieces = list(positions) if isinstance(positions, (list, tuple)) else [positions]
    bounds = np.cumsum([0] + [_cx.position_shape(p)[0] for p in pieces]); n_w = int(bounds[-1])
    chunk = _cx.slice_walkers(pieces[0]) if chunk is None else int(chunk)
    G, _ = _cx.acquisition_battery(int(n_t), float(dt), env); n_m = G.shape[0]
    W2 = _cx.coded_projection(dt, G, n_t, _cx.position_shape(pieces[0])[1] - 2)
    ijk_all, inside = grid.bin(np.concatenate([_cx.position_starts(p) for p in pieces]))
    pid = None if comp is None else (np.asarray(comp) if np.ndim(comp) == 1 else np.asarray(comp)[:, 0]).astype(np.int64)
    pools = sorted(set(np.unique(pid).tolist())) if pid is not None else [0]
    pid = pid if pid is not None else np.zeros(n_w, np.int64)
    ijk, row = _census_rows(ijk_all, inside)
    col = np.searchsorted(pools, pid)
    ww = np.ones(n_w) if w is None else np.asarray(w, np.float64)
    half = (np.random.default_rng(0).permutation(n_w) % 2).astype(bool)
    n_v, n_p = ijk.shape[0], len(pools)
    S_a = np.zeros((n_v, n_p, n_m), complex); S_b = np.zeros_like(S_a)
    W_a = np.zeros((n_v, n_p)); W_b = np.zeros((n_v, n_p)); N = np.zeros((n_v, n_p), np.int64)
    for i in range(0, n_w, chunk):
        idx = np.arange(i, min(i + chunk, n_w)); idx = idx[row[idx] >= 0]
        if not len(idx):
            continue
        C = _walker_coeffs(pieces, bounds, i, min(i + chunk, n_w))
        e = np.exp(1j * _cx.coded_phases(C if len(idx) == len(C) else C[idx - i], dt, G, n_t, device=device, projection=W2)) * ww[idx, None]
        r, c, h = row[idx], col[idx], half[idx]
        np.add.at(S_a, (r[h], c[h]), e[h]); np.add.at(S_b, (r[~h], c[~h]), e[~h])
        np.add.at(W_a, (r[h], c[h]), ww[idx][h]); np.add.at(W_b, (r[~h], c[~h]), ww[idx][~h])
        np.add.at(N, (r, c), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        floor = 0.5 * np.abs(S_a / W_a[..., None] - S_b / W_b[..., None]).max(-1)
    floor[(W_a == 0) | (W_b == 0)] = np.nan
    return ijk, pools, N, floor, np.full_like(floor, np.nan)


def _merged_start_pool(pks):
    """The pool each walker of shards ``pks`` (in walker order) starts in, ``(n, 1)``; ``None`` without an occupancy
    channel."""
    cols = [_start_pool(pk.arrays, pk.meta) for pk in pks]
    return None if any(c is None for c in cols) else np.concatenate(cols)


def _start_pool(arrays, meta):
    """The pool each walker starts in (a column ``(n, 1)``), from the occupancy channel; ``None`` without one."""
    ch = (meta["compression"].get("channels") or {}).get("compartment")
    if ch is None:
        return None
    col = next(c for c in ch["columns"] if c["name"] == "comp")
    return _cx.occupancy_events(arrays, col)[0][:, None]


def per_voxel_summary(grid_meta, pools, cert, **extra):
    """``fidelity.per_voxel`` for the certificate rows ``cert`` ``(n_voxels, n_pools, 3)`` (walkers, floor, error per
    voxel and pool) on the grid ``grid_meta``: the counts and the floors' and errors' extremes, with ``extra`` keys
    (``shards``, ``recertified``) beside them."""
    c = np.asarray(cert, np.float64)
    n, floor, err = c[:, :, 0], c[:, :, 1], c[:, :, 2]; ok = np.isfinite(floor)
    return dict(grid=grid_meta, pools=[int(p) for p in pools], n_voxels=int(c.shape[0]),
                walkers_min=int(n[n > 0].min()) if (n > 0).any() else 0,
                floor_max=float(np.nanmax(floor)) if ok.any() else None,
                floor_median=float(np.nanmedian(floor)) if ok.any() else None,
                err_max=float(np.nanmax(err)) if np.isfinite(err).any() else None,
                within_2x_floor_fraction=(float(np.mean(err[ok & np.isfinite(err)] <= 2.0 * floor[ok & np.isfinite(err)]))
                                          if (ok & np.isfinite(err)).any() else None),
                thin_voxels=int(((n > 0) & (n < 2)).sum()), **extra)


def voxel_certificate(pack, grid, *, envelope=None, device="auto"):
    """The per-voxel certificate of a pack that carries none, as its arrays and its summary: a shard whose writer
    measures no floor (the grid is the recipe's, not the shard's). The walkers are binned on ``grid`` by where they
    start and by pool and the split-half floor of each (voxel, pool) is read from the coefficients
    (:func:`voxel_floor_coded`, over ``envelope``'s battery, default the envelope every pack is built against), the
    codec-error column ``nan``: the pack's error is ``fidelity.err_max`` and its terms. Returns
    ``({"voxel_ijk": ..., "voxel_certificate": ...}, per_voxel)``, the pair a pack built with ``voxel_grid=``
    carries."""
    arrays, meta = pack.arrays, pack.meta
    w = None if "spin_weights" not in arrays else np.asarray(arrays["spin_weights"], np.float64)
    ijk, pools, n, floor, err = voxel_floor_coded(arrays, float(pack.dt), int(pack.n_t), grid, _start_pool(arrays, meta),
                                                  envelope or _cx.default_envelope(), w=w, device=device)
    cert = np.stack([n.astype(np.float64), floor, err], axis=-1)
    return (dict(voxel_ijk=ijk.astype(np.int32), voxel_certificate=cert.astype(np.float32)),
            per_voxel_summary(grid.to_meta(), pools, cert, shards=1, recertified=False))


def voxel_floor(pack, grid, waveform, *, shells=None, **replay_knobs):
    """The per-voxel, per-pool split-half Monte-Carlo floor of a pack's replay of ``waveform`` -- the acquisition
    the pack is meant for, rather than the envelope battery the build certifies against -- as dense volumes on
    ``grid`` (substrate-attached): ``(floors, counts)`` with ``floors[pool name][shell]`` the rms over the shell's
    measurements of ``|S_a - S_b| / 2`` from a fixed split of each voxel's walkers of that pool (the rms, not the
    max: a max over 90 directions of a few-walker difference sits ~2 sigma above the error of any one
    measurement, and the plan would over-count by 5x), and
    ``counts[pool name]`` the walkers per voxel. ``shells`` groups the measurements (``{name: index array}``;
    default one group, ``"all"``). Walkers are binned by where they started (``pack.r0``, the partition's rule).
    The input of :func:`dmipy_sim.spec.seeding.plan_seeding` for a target acquisition."""
    from ..replay.compression import occupancy_events
    if pack.substrate is None:
        raise ValueError("this pack embeds no spec, so its pools have no names")
    names = {p.id: p.name for p in pack.substrate.pools}
    ch = pack.meta.get("compression", {}).get("channels", {}) or {}
    if "compartment" not in ch:
        raise ValueError("the per-pool floor needs the pack's compartment channel")
    col = next(c for c in ch["compartment"]["columns"] if c["name"] == "comp")
    ids = occupancy_events(pack.arrays, col)[0]                                  # the pool at the walk's start
    w, ew, E = pack.walker_signals(waveform, **replay_knobs)
    n_w, n_m = E.shape
    ijk, inside = grid.bin(pack.r0)
    flat = np.where(inside, np.ravel_multi_index(tuple(np.clip(ijk, 0, np.asarray(grid.shape) - 1).T), grid.shape), -1)
    half = (np.random.default_rng(0).permutation(n_w) % 2).astype(bool)
    shells = shells or {"all": np.arange(n_m)}
    floors, counts = {}, {}
    n_v = grid.n_voxels
    for pid, name in names.items():
        m = (ids.astype(int) == int(pid)) & (flat >= 0)
        cnt = np.bincount(flat[m], minlength=n_v)
        counts[name] = cnt.reshape(grid.shape).astype(float)
        floors[name] = {}
        S_a = np.zeros((n_v, n_m), complex); S_b = np.zeros_like(S_a); W_a = np.zeros(n_v); W_b = np.zeros(n_v)
        ma, mb = m & half, m & ~half
        np.add.at(S_a, flat[ma], ew[ma, None] * E[ma]); np.add.at(W_a, flat[ma], w[ma])
        np.add.at(S_b, flat[mb], ew[mb, None] * E[mb]); np.add.at(W_b, flat[mb], w[mb])
        with np.errstate(invalid="ignore", divide="ignore"):
            diff = 0.5 * np.abs(S_a / W_a[:, None] - S_b / W_b[:, None])
        diff[(W_a == 0) | (W_b == 0)] = np.nan
        for sh, idx in shells.items():
            floors[name][sh] = (np.nan_to_num(np.sqrt(np.nanmean(diff[:, np.asarray(idx)] ** 2, axis=1)), nan=0.0).reshape(grid.shape)
                                if len(idx) else np.zeros(grid.shape))
    return floors, counts


def voxel_fidelity_volumes(pack):
    """The per-voxel certificate of a pack built with ``voxel_grid=`` as dense volumes on that grid:
    ``(grid, {pool name: floor volume}, {pool name: walker-count volume})``, the pools named by the pack's
    embedded spec -- the input of :func:`dmipy_sim.spec.seeding.plan_seeding`."""
    from ..phantom.grid import Grid
    pv = (pack.meta.get("fidelity") or {}).get("per_voxel")
    if pv is None or "voxel_certificate" not in pack.arrays:
        raise ValueError("this pack carries no per-voxel certificate; build it with voxel_grid=")
    if pack.substrate is None:
        raise ValueError("this pack embeds no spec, so its pools have no names")
    names = {p.id: p.name for p in pack.substrate.pools}
    grid = Grid.from_meta(pv["grid"]); ijk = np.asarray(pack.arrays["voxel_ijk"], np.int64)
    cert = np.asarray(pack.arrays["voxel_certificate"], np.float64)              # (n_v, n_pools, 3): n, floor, err
    floors, counts = {}, {}
    for j, pid in enumerate(pv["pools"]):
        fl = np.zeros(grid.shape); n = np.zeros(grid.shape)
        fl[tuple(ijk.T)] = np.nan_to_num(cert[:, j, 1], nan=0.0); n[tuple(ijk.T)] = cert[:, j, 0]
        floors[names[int(pid)]] = fl; counts[names[int(pid)]] = n
    return grid, floors, counts


_ISO_P_NAMES = ("xx", "yy", "zz", "xy", "xz", "yz")


def _cpmg_gate(n_t, n_pulses, t=None):
    """Transverse-phase gate for a CPMG train of ``n_pulses`` 180s at the usual (k+1/2)*TE/n times, on ``n_t`` samples
    at the fractions ``t`` of the window (uniform by default)."""
    t = np.arange(n_t) / max(n_t - 1, 1) if t is None else np.asarray(t, np.float64)
    s = np.ones(n_t)
    for k in range(n_pulses):                       # each 180 flips the accumulated-phase sign
        s[t >= (k + 0.5) / n_pulses] *= -1.0
    return s


def _ste_gate(n_t, t=None):
    """Transverse-phase gate of a stimulated echo filling the window: transverse for its first quarter, stored along
    z (accruing nothing) for the half between the store and the recall, then transverse with the recall's sign."""
    t = np.arange(n_t) / max(n_t - 1, 1) if t is None else np.asarray(t, np.float64)
    return np.where(t < 0.25, 1.0, np.where(t < 0.75, 0.0, -1.0))


def _surface_gates(n_t, n_pulses=4):
    """The gates the contact tier is certified under on an ``n_t``-sample window: the battery's
    (:func:`_battery_gates`) TRANSVERSE ON-TIME, ``|chi|``. The surface weight ``exp(-(rho / D) l)`` is a positive,
    monotone functional of the local time ``l`` and a 180 degree pulse does not change its sign, so a refocusing
    pulse leaves the gate on: a gradient echo, a spin echo and a CPMG train read the whole window, and a stimulated
    echo is off while its magnetisation is stored along B0. Each distinct gate once."""
    out = []
    for g in _battery_gates(n_t, n_pulses):
        g = np.abs(g)
        if not any(np.array_equal(g, o) for o in out):
            out.append(g)
    return out


def _battery_gates(n_t, n_pulses, t=None):
    """The gates the field tier is certified under on an ``n_t``-sample window (at the fractions ``t`` of it, uniform
    by default): a gradient echo, a spin echo, the CPMG train at the tier's depth ``n_pulses``, and a stimulated echo
    (its mixing time stored)."""
    return [np.ones(n_t), _cpmg_gate(n_t, 1, t), _cpmg_gate(n_t, n_pulses, t), _ste_gate(n_t, t)]


#: the bands the field tier is tried at, in order, when its band is derived on the walk
SUSC_PATH_LADDER = (16, 32, 64, 128, 256, 512, 1024, 2048)


def derive_field_band(series, names, h, env=None, *, K_max, B0=None, chi=None, w=None, ladder=SUSC_PATH_LADDER,
                      floor_scale=1.0, target_floor=None):
    """The field channel's band derived on a PILOT's block means: ``(K_c3, record)``.

    ``series`` ``(n_w, n_blocks, n_ch)`` are the pilot walkers' block means of the field's channels (the walk's own
    reads, trapezoid over each block of ``h`` seconds; :attr:`~dmipy_sim.coefficient_walk.CoefficientWalk.raw_field`
    without its first sample). The series' cosine modes are ``c_k = sqrt(h) X_k`` with ``X`` its orthonormal DCT-II
    (the block-mean quadrature of ``int e_k b dt``), exact at ``K = n_blocks``; the rungs of ``ladder`` below it are
    read in ascending band, each as the pack's certificate reads it -- the modes truncated to the rung and held in
    the container :class:`DerivedFieldContainer` derives for the rung's own mode scales at the rung's floor,
    contracted for each field strength and direction of the envelope's battery, decoded on the block grid and gated
    by GRE, spin echo, the CPMG train the rung serves and a stimulated echo (:func:`_battery_gates`) -- against the
    split-half floor of the pilot's own series. The first rung within the floor is the band; ``K_max`` (the position
    band) is the point past which the tier costs more than the walk it rides on, and the record says when the ladder
    went beyond it. A refocusing depth the envelope declares (``max_refocus_pulses``) puts a floor of twice that
    under the ladder. When no rung reaches the floor at all, the rung of least error is taken, the narrowest among
    those within three per cent of it. The floor the rungs are held to is the PRODUCTION walk's: the pilot's
    split-half floor times ``floor_scale`` (``sqrt(n_pilot / n_walkers)``: a pilot of a thousand walkers has a floor
    four times the ten-thousand-walker walk it sizes, and a band within the pilot's floor alone falls short of the
    walk's). The band is fixed BEFORE the production walk (#742): the walk accumulates ``K_c3`` modes and nothing is
    re-derived on its samples afterwards; the pack's container is derived afterwards from the modes the walk
    accumulated (:func:`build_replay_pack`), the rung's container on the pilot's scales being the record's
    ``container``. Each rung's ``err_truncation`` is its error with the modes held in float: what the band alone
    drops.

    With target_floor, selection and the container use that explicit numerical error
    budget independently of walker count. No fallback above budget is admitted.
    The finite gate battery remains diagnostic; callers must validate their query domain.
    floor_scale is retained for legacy split-half diagnostics and cannot be combined
    with target_floor.
    """
    from scipy.fft import idct
    if target_floor is not None:
        target_floor = float(target_floor)
        if not np.isfinite(target_floor) or target_floor <= 0 or float(floor_scale) != 1.0:
            raise ValueError("an explicit field error budget is positive and is not scaled by a walker count")
    series = np.asarray(series, np.float64)
    n_w, n_t, n_ch = series.shape
    names = list(names)
    env = env or _cx.default_envelope()
    depth = int(env.get("max_refocus_pulses") or 0)
    min_K = 2 * depth
    rungs = sorted(set([min(int(k), n_t) for k in ladder if k < n_t] + [n_t] + ([min(min_K, n_t)] if min_K else [])))
    rungs = [k for k in rungs if k >= min_K] or [n_t]
    top = rungs[-1]
    arrays_top, meta_top = susc_path_encode_series(series, names, K=top, container=None, layout="wtc", dtype=np.float64,
                                                   device="numpy", T=float(h) * n_t)
    coeffs = np.asarray(arrays_top["susc_path_dct"], np.float64) / np.sqrt(float(h))    # the DCT-II of the means: the series' own grid
    has_aniso = any(n.startswith("aniso_G") for n in meta_top["channels"])
    w = np.ones(n_w) if w is None else np.asarray(w, np.float64)
    chi_i = float(chi) if chi is not None else float(FIELD_RULE_CHI)
    ca = 0.1 * chi_i if has_aniso else 0.0
    B0s = list(env.get("B0_list") or ([float(B0)] if B0 is not None else [3.0, 7.0]))
    perm = np.random.RandomState(0).permutation(n_w); A, Bh = perm[:n_w // 2], perm[n_w // 2:]
    settings = [(b0, th) for b0 in B0s for th in (env.get("theta_deg") or [0, 90])]
    raw = {}
    from ..fields.hollow_cylinder import contract
    for b0, th in settings:
        t = np.deg2rad(float(th)); d = [np.sin(t), 0.0, np.cos(t)]
        raw[(b0, th)] = np.asarray(contract(series, d, B0=b0, chi_iso=chi_i, chi_aniso=ca), np.float64)         # (n_w, n_t)
    stored = meta_top["channels"]

    def read(a_k, m_k, k, gates):
        """the worst gated-phase error of the modes ``a_k`` against the series, and the series' own split-half floor"""
        C, _names = susc_path_coeffs(a_k, m_k)
        e_max = f_max = 0.0
        for b0, th in settings:
            t = np.deg2rad(float(th)); d = [np.sin(t), 0.0, np.cos(t)]
            cd = susc_path_field(C, d, B0=b0, chi_iso=chi_i, chi_aniso=ca, has_aniso=has_aniso)       # (n_w, k)
            f_dec = idct(np.pad(cd, ((0, 0), (0, n_t - k))), type=2, norm="ortho", axis=1)
            e, f = _gate_battery(raw[(b0, th)], f_dec, gates, w, A, Bh, float(h))
            e_max, f_max = max(e_max, e), max(f_max, f)
        return e_max, f_max

    err, floor, trunc, cont, order = {}, {}, {}, {}, []
    chosen = None
    for k in rungs:
        n_p = _depth(k, depth)
        gates = _battery_gates(n_t, max(1, n_p))
        meta_k = dict(meta_top, K=k, max_refocus_pulses=n_p)
        c_k = np.asarray(coeffs[:, :, :k], np.float64)
        e_tr, f_max = read({"susc_path_dct": c_k}, dict(meta_k, bits=None, dtype="float64"), k, gates)   # the band alone, modes in float
        fl = f_max * float(floor_scale) if target_floor is None else target_floor
        ranges = DerivedFieldContainer(floor=fl, B0=max(B0s), chi_iso=chi_i, chi_aniso=ca).ranges(np.abs(c_k).max(axis=0), stored, float(h) * n_t)
        a_k, m_k = _quantise_susc_path(c_k, meta_k, ranges)
        e_max, _ = read(a_k, m_k, k, gates)
        err[k], floor[k], trunc[k], cont[k] = e_max, fl, e_tr, ranges; order.append(k)
        if e_max <= fl:
            chosen = k; break
    if chosen is None:
        if target_floor is not None:
            raise ValueError(f"no field band/container meets the requested error budget {target_floor:g}; "
                             f"best measured error {min(err.values()):g}")
        best = min(err.values())
        chosen = min([k for k in order if err[k] <= 1.03 * best])
    K = chosen
    record = dict(rule="pilot",
                  criterion="the first rung in ascending band whose codec error on the certificate's battery, in the container derived "
                            "for its mode scales at its floor, is within the pilot's floor; failing every rung, the narrowest among the least errors",
                  ladder=[dict(K=int(k), err=float(err[k]), err_truncation=float(trunc[k]), floor=float(floor[k]),
                               container=[dict(r) for r in cont[k]]) for k in order],
                  K=int(K), K_max=int(K_max), min_K=int(min_K), depth=(int(depth) or None), above_position_band=bool(K > int(K_max)),
                  within_floor=bool(err[K] <= floor[K]), err=float(err[K]), err_truncation=float(trunc[K]), floor=float(floor[K]),
                  container=[dict(r) for r in cont[K]],
                  floor_is=("the explicit numerical field error budget" if target_floor is not None else
                            "the pilot's split-half floor scaled to the production walk's count (x floor_scale)"),
                  floor_scale=float(floor_scale), B0_list=[float(b) for b in B0s], chi_iso=chi_i, chi_aniso=float(ca), n_blocks=int(n_t))
    return int(K), record


def derive_field_band_per_pool(series, pools, names, h, env=None, *, K_max, B0=None, chi=None, floor_scale=1.0, target_floor=None):
    """The field channel's band on a pilot spread over the walk's pools: :func:`derive_field_band` on each pool's
    walkers (``pools`` ``{name: index array into series}``), the walk taking the widest band any pool needs -- a pool
    whose field is smooth (a uniform lumen, a far extra space) must not set the band of one whose field is not.
    ``(K_c3, record)``: the record is the binding pool's (its ladder, its error and floor at the band; ``binding_pool``),
    with every pool's own under ``pools``."""
    per, recs = {}, {}
    for name, idx in pools.items():
        idx = np.asarray(idx)
        if idx.size < 2:
            continue
        k_p, rec = derive_field_band(np.asarray(series)[idx], names, h, env, K_max=K_max, B0=B0, chi=chi, floor_scale=floor_scale, target_floor=target_floor)
        recs[str(name)] = (int(k_p), rec)
        per[str(name)] = dict({k: v for k, v in rec.items() if k != "ladder"}, walkers=int(idx.size))
    if not recs:
        return derive_field_band(series, names, h, env, K_max=K_max, B0=B0, chi=chi, floor_scale=floor_scale, target_floor=target_floor)
    binding = max(recs, key=lambda nm: (recs[nm][0], recs[nm][1]["err"] / max(recs[nm][1]["floor"], 1e-300)))
    return int(recs[binding][0]), dict(recs[binding][1], binding_pool=binding, pools=per,
                                       per_pool="each pool's band by its own pilot walkers, the widest taken")


def _gate_battery(f_raw, f_dec, gates, w, A, B, dt):
    """The worst gated-phase replay error between two per-walker field series over ``gates``, and the split-half
    floor of the reference: ``(err, floor)``."""
    from ..constants import GAMMA
    wmean = lambda c, idx: float(np.sum(w[idx] * c[idx]) / np.sum(w[idx]))
    err = floor = 0.0
    dt = np.broadcast_to(np.asarray(dt, np.float64), (f_raw.shape[1],))                # a sample's interval (s)
    for g in gates:
        cr = np.cos(GAMMA * (f_raw * (g * dt)[None, :]).sum(1))
        cd = np.cos(GAMMA * (f_dec * (g * dt)[None, :]).sum(1))
        err = max(err, abs(wmean(cr, slice(None)) - wmean(cd, slice(None))))
        floor = max(floor, abs(wmean(cr, A) - wmean(cr, B)))
    return err, floor


def path_codec(bits):
    """The path channel's storage knob in its one normal form: ``None`` (the float form), an ``int`` 8 or 16 (one
    native depth over every mode, the registry's single-depth container), or the tuple ``((upto, bits), ...)`` of a
    container of ranges of modes (:func:`compression.band_curve`'s list is accepted, its ranges naming ``modes`` or
    ``bands``) at depths 3..16."""
    if bits is None:
        return None
    if isinstance(bits, (int, np.integer)):
        if int(bits) not in (8, 16):
            raise ValueError(f"susc_path bits must be 8, 16 or a container of ranges (got {bits!r})")
        return int(bits)
    c = _container(bits)
    if isinstance(c, _cx.BandCurve):
        return c
    if len(c) == 1 and c[0][1] in _cx.NATIVE_BITS:
        return int(c[0][1])
    return c


def path_codec_of(pm):
    """:func:`path_codec` of a path channel's metadata: its ``container`` when it has one, else its ``bits``."""
    if pm.get("container") is not None:
        return path_codec(pm["container"])
    return path_codec(pm.get("bits") or None)


#: the field directions the container is derived and certified for: the cube's three axes, six face diagonals and four
#: body diagonals (the contraction is quadratic in the direction, so its worst case over the sphere sits among them)
FIELD_DIRECTIONS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1],
                             [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], np.float64)


def field_channel_weights(stored, *, B0, chi_iso, chi_aniso=0.0, directions=FIELD_DIRECTIONS):
    """The weight of each STORED channel in the field ``dB`` (T) a replay contracts, per direction: ``(n_dir,
    n_stored)`` (:func:`~dmipy_sim.fields.hollow_cylinder.contract` on unit channels of the canonical order, twelve
    with an ``aniso_G`` channel stored, else six; ``iso_P_zz`` is formed from the trace inside the contraction, so its
    weight is already carried by ``iso_local``, ``iso_P_xx`` and ``iso_P_yy``). A component the pack leaves out has
    no weight to carry."""
    from ..fields.hollow_cylinder import contract
    stored = list(stored)
    full = list(_PATH_CHANNELS if any(n.startswith("aniso_G") for n in stored) else _ISO_PATH_CHANNELS)
    n = len(full)
    ca = float(chi_aniso) if n >= 12 else 0.0
    W = np.stack([contract(np.eye(n), d, B0=B0, chi_iso=chi_iso, chi_aniso=ca, axis=-1) for d in np.asarray(directions, np.float64)])
    return W[:, [full.index(c) for c in stored]]


def _contracted_modes(X, W):
    """``(n_dir, K)``: the amplitude per mode ``X`` ``(n_ch, K)`` of the stored channels (their maximum, or their
    quantisation step) as the contracted field sees it, the channels' weights ``W`` ``(n_dir, n_ch)`` added in
    quadrature."""
    return np.sqrt(np.einsum("dc,ck->dk", np.asarray(W, np.float64) ** 2, np.asarray(X, np.float64) ** 2))


class DerivedFieldContainer:
    """The default container of the field channel, derived per pack by :meth:`ranges` from the mode scales the encoder
    measures and the floor the pack is ordered at (``floor`` None is filled by the builder). The step is common to
    the modes in the contracted field ``dB`` at the envelope's largest ``B0`` and the worst field direction: a mode
    of maximum ``a_k`` (the stored channels contracted, :func:`field_channel_weights`) at ``b`` bits has step ``a_k /
    (2^(b-1) - 1)``, and a gate of unit modulus weighs the modes by ``g_k`` with ``sum_k g_k^2 <= T``, so one step
    ``s`` costs a phase variance of at most ``gamma^2 T s^2 / 12`` whatever the gate:
    :func:`compression.derive_container` with ``phase = gamma^2 T`` holds it to the pack's quantisation budget,
    ``bits_k = ceil(log2(a_k / s)) + 1`` clipped to 3..16, equal-bit runs grown to whole bytes. The depth is the
    mode's, not the band's: the field is nearly white above its mean, so the modes' depths sit within a few bits
    of one another and the byte count is carried by the floor."""

    def __init__(self, floor=None, B0=None, chi_iso=None, chi_aniso=0.0, budget=_cx.BUDGET):
        self.floor, self.B0, self.chi_iso, self.chi_aniso, self.budget = floor, B0, chi_iso, chi_aniso, float(budget)

    def bound(self, **kw):
        d = dict(floor=self.floor, B0=self.B0, chi_iso=self.chi_iso, chi_aniso=self.chi_aniso, budget=self.budget); d.update(kw)
        return DerivedFieldContainer(**d)

    def ranges(self, amax, stored, T):
        """``[{"modes": [k0, k1], "bits": b}, ...]`` for the stored channels' per-mode maxima ``amax`` ``(n_ch, K)``."""
        if self.floor is None or self.B0 is None or self.chi_iso is None:
            raise ValueError("a derived field container needs the pack's floor and the field's B0 and chi; the builder supplies them")
        from ..constants import GAMMA
        W = field_channel_weights(stored, B0=self.B0, chi_iso=self.chi_iso, chi_aniso=self.chi_aniso)
        a = _contracted_modes(amax, W).max(axis=0)
        r = _cx.derive_container(a, a.shape[0], max(float(self.floor), 1e-300), phase=GAMMA ** 2 * float(T), budget=self.budget)
        return [{"modes": [int(x["bands"][0]), int(x["bands"][1])], "bits": int(x["bits"])} for x in r]

    def __repr__(self):
        return f"DerivedFieldContainer(floor={self.floor}, B0={self.B0}, budget={self.budget:g})"


class FieldCodec:
    """A stated layout of the field channel: its container (``None``: float) and the channels stored. What a shard of a
    fill and a window of a pack take from the pack they belong to, so that every piece of one pack has the same
    tensors; built from a pack's descriptor ``compression.channels.susc_field_basis``
    (``build_replay_pack(field_container=descriptor)``)."""

    def __init__(self, container, channels):
        self.container, self.channels = container, list(channels)

    @classmethod
    def of(cls, pm):
        return cls(path_codec_of(pm), pm["channels"])

    def __repr__(self):
        return f"FieldCodec({self.container!r}, channels={self.channels})"


def _field_container(spec):
    """The field channel's container from the builder's knob: ``None`` (the float container), ``"derived"`` (the
    default, :class:`DerivedFieldContainer`), a :class:`DerivedFieldContainer`, a pack's field descriptor (a
    :class:`FieldCodec`: its container and channels), or an explicit container of ranges of modes
    (:func:`path_codec`)."""
    if spec is None or isinstance(spec, (DerivedFieldContainer, FieldCodec)):
        return spec
    if isinstance(spec, dict) and "channels" in spec:
        return FieldCodec.of(spec)
    if isinstance(spec, str):
        if spec != "derived":
            raise ValueError(f"the field container is None, 'derived', or ranges of modes; got {spec!r}")
        return DerivedFieldContainer()
    return path_codec(spec)


def field_quantisation(arrays, pm, env, *, T, h, B0, chi_iso, chi_aniso=0.0):
    """The rms phase (rad) the stored field container adds: the realised steps of ``susc_path_scale`` (a step is
    its own width, noise ``step^2 / 12``), contracted at the field strength ``B0`` and the worst direction
    (:func:`field_channel_weights`), weighted by the battery's gates (:func:`_battery_gates` at the depth the pack
    advertises, their cosine projections on the window) -- the worst gate of the worst direction. 0 for the float
    container."""
    from ..constants import GAMMA
    if "susc_path_scale" not in arrays:
        return 0.0
    K = int(pm["K"])
    step = np.asarray(arrays["susc_path_scale"], np.float64)
    step = step.max(axis=0) if step.ndim == 3 else step                   # a merged pack: the widest block's
    W = field_channel_weights(pm["channels"], B0=B0, chi_iso=chi_iso, chi_aniso=chi_aniso)
    n_t = int(round(float(T) / float(h))) + 1
    gates = np.stack(_battery_gates(n_t, max(1, int(pm.get("max_refocus_pulses") or K // 2))))
    g = _cx.cosine_projection(gates[:, :, None], float(h), float(T), K)[:, :, 0]               # (n_g, K)
    var = _contracted_modes(step, W) ** 2 / 12.0                                              # (n_dir, K)
    return float(GAMMA * np.sqrt(np.einsum("gk,dk->gd", g ** 2, var).max()))


_ZERO_STEP = float(np.finfo(np.float32).tiny)


def _quantise_susc_path(coeffs, meta, container, *, device="numpy"):
    """The integer container of the path-field coefficients: a per-(channel, mode) scale (the quantisation step; a
    mode that is zero in every walker has the smallest float32 for a step, which decodes it to zero) -- for a
    container of ranges (:func:`path_codec`), each range of modes at its own depth, bit-packed along the modes
    below or between the native depths (:func:`compression.quantise_bands`' layout), under ``susc_path_dct_b<i>[u<bits>]``
    with the ranges in ``meta["container"]``; or, for the single-depth form, one native depth over every mode. In the single-depth form the
    reduction (the scale's max-abs over every walker) and the round/clip run on the JAX device when ``device``
    resolves to one (float32: the container is already lossy, so this loses nothing the quantisation does not),
    host numpy (float64, the reference) otherwise."""
    bits = path_codec(container)
    if not isinstance(bits, (int, np.integer)):
        K = coeffs.shape[2]
        ranges = _cx._band_ranges(bits, K)
        amax = np.abs(coeffs).max(axis=0)                            # (n_ch, K)
        scale = np.ones(amax.shape, np.float64); arrays, cont = {}, []
        for i, (k0, k1, b) in enumerate(ranges):
            lim = 2 ** (b - 1) - 1
            sc = amax[:, k0:k1] / lim; sc[sc == 0] = _ZERO_STEP
            scale[:, k0:k1] = sc
            q = np.clip(np.rint(coeffs[:, :, k0:k1] / sc[None]), -lim, lim)           # (n_w, n_ch, m)
            t = _cx._store_range(np.swapaxes(q, 1, 2), b)                              # (n_w, m | bytes, n_ch)
            arrays[_cx.range_name("susc_path_dct", i, b)] = np.ascontiguousarray(np.swapaxes(t, 1, 2))
            cont.append({"modes": [int(k0), int(k1)], "bits": int(b)})
        meta["bits"] = None; meta["dtype"] = "bands"; meta["container"] = cont
        arrays["susc_path_scale"] = np.asarray(scale, np.float32)
        return arrays, meta
    itype = np.int8 if bits == 8 else np.int16
    lim = 2 ** (bits - 1) - 1
    if _cx.resolve_device(device) == "numpy":
        scale = np.abs(coeffs).max(axis=0) / lim                     # (n_ch, K), per channel AND band
        scale[scale == 0] = _ZERO_STEP
        q = np.clip(np.rint(coeffs / scale), -lim, lim).astype(itype)
    else:
        import jax.numpy as jnp
        c = jnp.asarray(coeffs, jnp.float32)
        scale = np.asarray(jnp.max(jnp.abs(c), axis=0)) / lim
        scale[scale == 0] = _ZERO_STEP
        q = np.asarray(jnp.clip(jnp.round(c / jnp.asarray(scale, jnp.float32)[None]), -lim, lim)).astype(itype)
    meta["bits"] = int(bits); meta["dtype"] = np.dtype(itype).name
    return {"susc_path_dct": q, "susc_path_scale": np.asarray(scale, np.float32)}, meta


def _depth(K, declared):
    """The refocusing depth a path channel of ``K`` bands serves: the declared train when there is one, never more than
    the ``K / 2`` pulses the band can gate; ``K / 2`` itself when none is declared."""
    return int(min(int(K) // 2, int(declared))) if declared else int(K) // 2


def susc_path_encode_series(series, names, *, K=32, container=None, dtype=np.float16, layout="wct", device="auto",
                            chunk=20_000, T=None, max_refocus_pulses=None):
    """The field channel's tensors and descriptor from a field series of block means (``container`` None: the float
    form at ``dtype``; else :func:`path_codec`'s), in the canonical channel order with
    ``names`` (the twelve of :data:`~dmipy_sim.fields.hollow_cylinder.CHANNEL_NAMES`, or the six isotropic ones):
    ``(n_w, n_ch, n_t)`` (``layout="wct"``, what a decoded path channel gives) or ``(n_w, n_t, n_ch)``
    (``layout="wtc"``, the interval means a walk sampled), encoded in walker chunks of ``chunk`` on ``device``
    without a copy of the whole series. ``T`` is the
    window the series spans (s): the modes are the DCT-II of the ``n_t`` means times ``sqrt(T / n_t)``, the
    block-mean quadrature of the window's orthonormal cosine modes (:func:`derive_field_band` reads the DCT itself).
    The pilot's route: a production pack stores the modes the walk accumulated (:func:`susc_path_encode_modes`)."""
    if layout not in ("wct", "wtc"):
        raise ValueError("layout is 'wct' (n_w, n_ch, n_t) or 'wtc' (n_w, n_t, n_ch)")
    series = np.asarray(series)
    n_w = series.shape[0]; n_ch = series.shape[1] if layout == "wct" else series.shape[2]
    n_t = series.shape[2] if layout == "wct" else series.shape[1]
    names = list(names)
    if len(names) != n_ch:
        raise ValueError(f"{len(names)} channel names for {n_ch} channels")
    _refuse_trace_component(names)
    K = int(min(K, n_t))
    take = lambda sl: (np.transpose(series[sl], (0, 2, 1)) if layout == "wct" else series[sl])   # (rows, n_t, n_ch)
    coeffs = np.empty((n_w, n_ch, K), np.float64)
    for i in range(0, n_w, chunk):
        ch = take(slice(i, i + chunk))                           # the series' own dtype, read in place -- no copy
        coeffs[i:i + chunk] = _cx.dct_bands_series(ch, K, device=device)   # one cached device jit for every chunk
    if T is not None:
        coeffs *= np.sqrt(float(T) / n_t)
    meta = dict(channel="susc_field_basis", K=K, n_ch=n_ch, channels=names,
                max_refocus_pulses=_depth(K, max_refocus_pulses), **({} if T is None else dict(T=float(T))))
    if container is None:
        meta["bits"] = None; meta["dtype"] = np.dtype(dtype).name
        return {"susc_path_dct": np.asarray(coeffs, dtype)}, meta
    return _quantise_susc_path(coeffs, meta, container, device=device)


def _refuse_trace_component(names):
    """``iso_P_zz`` is never a stored or accumulated component (RPK.md 5.3): a reader forms it from ``3 iso_local -
    iso_P_xx - iso_P_yy``. A producer that hands one over is refused by name."""
    if "iso_P_zz" in names:
        raise ValueError("the field channel carries twelve components: iso_P_zz is never stored (RPK.md 5.3), a reader forms it "
                         "as 3 iso_local - iso_P_xx - iso_P_yy; this field offers it")


def has_susc_path(arrays):
    """Whether the arrays carry a path channel, in the single-depth tensor or in a container's ranges."""
    return any(k.startswith("susc_path_dct") for k in arrays)


def susc_path_coeffs(arrays, meta):
    """Dequantised C3 coefficients in the canonical channel order -- ``(n_w, n_ch_full, K)``, twelve channels
    (``iso_local``, five ``iso_P``, six ``aniso_G``) or the six isotropic ones: a component the pack leaves out
    (``zero_components``, an absent ``aniso_G`` block) is zero. ``iso_P_zz`` is not a channel here: the contraction
    forms it from the trace (:func:`~dmipy_sim.fields.hollow_cylinder.field_terms`), and a descriptor that lists it is
    refused.

    The ONE place that undoes the storage container. ``susc_path_dct`` may be an integer array whose
    scale lives in ``susc_path_scale`` (see :func:`susc_path_encode`), so reading it raw yields
    integers ~1e3 off with no error -- every consumer goes through here."""
    if meta.get("container") is not None:
        parts = []
        for i, r in enumerate(meta["container"]):
            t = np.swapaxes(np.asarray(arrays[_cx.range_name("susc_path_dct", i, r["bits"])]), 1, 2)
            parts.append(np.swapaxes(np.asarray(_cx._load_range(t, int(r["bits"]), int(r["modes"][1] - r["modes"][0])), np.float64), 1, 2))
        C = np.concatenate(parts, axis=2)
    else:
        C = np.asarray(arrays["susc_path_dct"], np.float64)
    if "susc_path_scale" in arrays:
        S = np.asarray(arrays["susc_path_scale"], np.float64)
        if S.ndim == 3:                                            # a merged pack: one scale table per shard, walkers by block
            blk = np.asarray(arrays["band_block"], np.int64)
            for b in np.unique(blk):
                C[blk == b] *= S[b][None]
        else:
            C = C * S[None]
    names = list(meta["channels"])
    _refuse_trace_component(names)
    if names and all(n in _PATH_CHANNELS for n in names):
        full = list(_PATH_CHANNELS if any(n.startswith("aniso_G") for n in names) else _ISO_PATH_CHANNELS)
        if names != full:
            out = np.zeros((C.shape[0], len(full), C.shape[2]), C.dtype)
            out[:, [full.index(n) for n in names]] = C
            C, names = out, full
    return C, names


def susc_path_field(b, b0_dir, *, B0, chi_iso, chi_aniso=0.0, has_aniso=False):
    """``dB(t)`` per walker ``(n_w, n_t)`` from the decoded path channels ``(n_w, n_ch, n_t)`` for one
    ``(B0, direction, chi)``: :func:`~dmipy_sim.fields.hollow_cylinder.contract` along the channel axis."""
    from ..fields.hollow_cylinder import contract
    aniso = chi_aniso if (has_aniso and chi_aniso and b.shape[1] >= 12) else 0.0
    return contract(b, b0_dir, B0=B0, chi_iso=chi_iso, chi_aniso=aniso, axis=1)


def held_voxels(cert):
    """The rows of a voxel certificate ``(n_v, n_pools, 3)`` (walkers, floor, err per pool) a shard certifies: walkers
    in the voxel and a floor measured for at least one pool. A row with walkers but no floor is a stray, a walker
    seeded on a block's face and binned into the neighbour's voxel."""
    c = np.asarray(cert, np.float64)
    return (c[:, :, 0].sum(1) > 0) & np.isfinite(c[:, :, 1]).any(1)


def grid_periodic_of(grid_meta):
    """Per axis whether a field grid is one period of its field, from a pack's ``susceptibility_grid`` meta
    (``periodic``) or a master (``susc_grid_periodic``): the ``periodic=`` that
    :func:`~dmipy_sim.fields.susceptibility_field.sample_grid` reads it with. A grid that declares none is clamped."""
    per = grid_meta.get("periodic", grid_meta.get("susc_grid_periodic"))
    return (False, False, False) if per is None else tuple(bool(p) for p in per)


def grid_basis_of(arrays, grid_meta):
    """The pack's stored field basis as :func:`~dmipy_sim.fields.susceptibility_field.assemble_field` reads it."""
    return {"iso_local": np.asarray(arrays["susc_grid_iso_local"], np.float64),
            "iso_P": np.asarray(arrays["susc_grid_iso_P"], np.float64),
            "aniso_G": (np.asarray(arrays["susc_grid_aniso_G"], np.float64) if "susc_grid_aniso_G" in arrays else None),
            "shape": tuple(grid_meta["shape"]), "voxel_size": np.asarray(grid_meta["voxel_size"], float)}


# --------------------------------------------------------------- susceptibility replay (consume)
# --------------------------------------------------------------- pack generation
#: per-channel numbers a codec MEASURES on the walk it encoded (not parameters): two shards of one fill differ in them
_MEASURED_CHANNEL_KEYS = ("trace_residual", "band")
#: the field channel's read, which each pool of a bundle makes at its own interval: a record of the walk, not a codec
#: parameter (:func:`charge_field_read` states the bundle's)
_FIELD_READ_CHANNEL_KEYS = ("read", "read_interval")


def _codec_signature(comp):
    """The codec parameters of a pack's ``compression`` meta, without what is measured per pack (the precision tiers,
    a channel's trace residual): what two shards of one fill must agree on."""
    sig = {k: v for k, v in comp.items() if k != "precision_tiers"}
    if "channels" in sig:
        sig["channels"] = {c: ({k: v for k, v in m.items() if k not in _MEASURED_CHANNEL_KEYS + _FIELD_READ_CHANNEL_KEYS} if isinstance(m, dict) else m)
                           for c, m in sig["channels"].items()}
    return sig


def union_weights(w, shard, voxel, pool):
    """The walker weights of merged shards, renormalised to the union. A walk weights every walker of a (voxel,
    pool) alike, by its census of the voxel's fraction of that pool over its own count there, so one shard's
    walkers of a (voxel, pool) sum to its fraction. The union's walkers of that (voxel, pool) are weighted alike
    too, whichever shard they came from, by the mean of the shards' fractions (each census an independent
    estimate) over the union's count -- a 16 % first pass merged with its 84 % top-up as-is would weigh both
    passes equally, and the union's floor would be 36 % worse than its count deserves. ``shard``, ``voxel`` and
    ``pool`` are per walker; a walker outside the grid (``voxel`` -1) keeps its weight."""
    w = np.asarray(w, np.float64).copy(); shard = np.asarray(shard); voxel = np.asarray(voxel); pool = np.asarray(pool)
    held = voxel >= 0
    _, cell = np.unique(np.stack([voxel[held], pool[held]], 1), axis=0, return_inverse=True); cell = cell.reshape(-1)
    n_union = np.bincount(cell).astype(np.float64)
    _, own, n_own = np.unique(np.stack([shard[held], cell], 1), axis=0, return_inverse=True, return_counts=True); own = own.reshape(-1)
    fraction_own = np.bincount(own, weights=w[held])                        # each shard's fraction of its (voxel, pool)
    cell_of_own = np.zeros(len(n_own), np.int64); cell_of_own[own] = cell
    shards_in_cell = np.bincount(cell_of_own, minlength=len(n_union)).astype(np.float64)
    fraction = np.bincount(cell_of_own, weights=fraction_own, minlength=len(n_union)) / shards_in_cell
    w[held] = fraction[cell] / n_union[cell]
    return w


def _walk_identity(wp):
    """The walk parameters that shards of one walk share: everything but the walker count, the seeds and the
    field read interval (each pool reads the field at its own, :func:`charge_field_read`), the window table's
    walks compared without theirs."""
    out = {k: v for k, v in wp.items() if k not in ("n_walkers", "seed", "qv", "field_read_interval")}      # qv: each shard's own measure
    if isinstance(out.get("block"), dict):                  # the block's definition, not what each shard measured on it
        out["block"] = {k: out["block"].get(k) for k in ("h", "m", "n_blocks", "rule")}
    seg = out.get("windows")
    if isinstance(seg, dict):
        out["windows"] = dict(seg, walks=[{k: v for k, v in w.items() if k != "seed"} for w in seg.get("walks", [])])
    return out


def _envelope_identity(env):
    """A pack's ``replay_envelope`` without its field read (a pool's own, :func:`charge_field_read`): what shards of
    one walk share."""
    return {k: v for k, v in env.items() if k != "field_read"} if isinstance(env, dict) else env


def _finite_or_none(x):
    """``x`` as a float, or ``None`` where it is not finite (a read series that does not vary bounds no ``B0 chi``)."""
    return float(x) if np.isfinite(float(x)) else None


def _merged_qv(qvs, n):
    """``{"qv": ...}``: the walkers' mean quadratic variation of shards of one walk (RPK.md 9.5.5), each entry the
    walker-weighted mean of the shards' (``n`` their walker counts); ``{}`` when a shard carries none."""
    if not qvs or any(not isinstance(q, dict) for q in qvs):
        return {}
    w = np.asarray(n, np.float64) / float(np.sum(n))
    out = {}
    for k in qvs[0]:
        vals = [q.get(k) for q in qvs]
        if any(v is None for v in vals):
            continue
        a = (w.reshape((-1,) + (1,) * np.asarray(vals[0]).ndim) * np.asarray(vals, np.float64)).sum(0)
        out[k] = [float(x) for x in a] if np.ndim(a) else float(a)
    return dict(qv=out)


def field_read_record(meta):
    """What a pack states about its field read, or ``None`` without a field channel: the interval
    (``walk_params.field_read_interval``), the term (``fidelity.terms.field_read``), the read series' quadratic
    variation (``walk_params.qv.field_increments``, the walkers' mean) and the walker count it is the mean over, the
    envelope's entry and the path channel's own record of the read."""
    wp = meta.get("walk_params") or {}
    interval = wp.get("field_read_interval")
    if interval is None:
        return None
    ch = ((meta.get("compression") or {}).get("channels") or {}).get("susc_field_basis") or {}
    return dict(interval=float(interval), term=((meta.get("fidelity") or {}).get("terms") or {}).get("field_read"),
                qv_increments=(wp.get("qv") or {}).get("field_increments"), n_walkers=int(wp.get("n_walkers") or 0),
                envelope=(meta.get("replay_envelope") or {}).get("field_read"), read=ch.get("read"),
                channel_interval=ch.get("read_interval"))


def charge_field_read(meta, records):
    """``meta`` (the metadata of a pack made of several shards, or of a layout of several blocks) states the field read
    of ``records`` (:func:`field_read_record` per shard) as the producer states a bundle's
    (:func:`~dmipy_sim.spec.walk._join_pool_walks`): every shard reads the field at its own interval (a pool's read
    snaps to a whole number of its own rounds per block), and the pack records the COARSEST -- in
    ``walk_params.field_read_interval``, the envelope's ``field_read`` and the path channel's ``read`` -- with the
    term charged at it on the merged read series: ``walk_params.qv.field_increments`` the walker-weighted mean of the
    shards', ``fidelity.terms.field_read`` half of ``v_read`` at the coarsest interval
    (:func:`~dmipy_sim.engine.accumulate.field_read_term` at the envelope's ``B0`` and ``chi``, the largest trapezoid
    constant per channel any shard charged), the envelope's ``B0_chi_max_T`` at it, and ``err_max`` covers the terms.
    The shards' own intervals are the ``provenance`` record's. ``meta`` is changed in place; its nested dicts are
    replaced, never edited."""
    if all(r is None for r in records):
        return meta
    if any(r is None for r in records):
        raise ValueError("some of the shards read the field and others do not; the shards of one walk carry one set of channels")
    if any(r["qv_increments"] is None for r in records):
        raise ValueError("a shard carries no walk_params.qv.field_increments (RPK.md 9.5.5), which the merged field read's "
                         "term is charged on; re-walk it with a conformant producer")
    coarse = records[int(np.argmax([r["interval"] for r in records]))]
    n = np.asarray([max(r["n_walkers"], 1) for r in records], np.float64)
    qv = (n[:, None] * np.asarray([r["qv_increments"] for r in records], np.float64)).sum(0) / n.sum()
    consts = [((r.get("read") or {}).get("constants")) for r in records]
    consts = None if all(c is None for c in consts) else np.max([c for c in consts if c is not None], axis=0)
    env = coarse["envelope"] or {}
    at = env.get("term_at") or {}
    term = field_read_term(coarse["interval"], qv, at.get("B0"), at.get("chi"), consts)
    wp = dict(meta["walk_params"], field_read_interval=coarse["interval"])
    wp["qv"] = dict(wp.get("qv") or {}, field_increments=[float(x) for x in qv])
    meta["walk_params"] = wp
    if coarse["envelope"] is not None:
        meta["replay_envelope"] = dict(meta["replay_envelope"], field_read=dict(
            coarse["envelope"], B0_chi_max_T=_finite_or_none(field_read_B0_chi_max(coarse["interval"], qv, at.get("floor") or 1e-3, consts))))
    comp = dict(meta["compression"]); chans = dict(comp["channels"]); pm = dict(chans["susc_field_basis"])
    for key, val in (("read", coarse["read"]), ("read_interval", coarse["channel_interval"])):
        if val is not None:
            pm[key] = val
    chans["susc_field_basis"] = pm; comp["channels"] = chans; meta["compression"] = comp
    fid = dict(meta["fidelity"]); fid["terms"] = dict(fid["terms"], field_read=float(term))
    meta["fidelity"] = _cover_terms(fid)
    return meta


def _spec_identity(spec):
    """What makes two embedded specs the same substrate: the spec without its ``provenance`` (who wrote it, when,
    with which software, from which local path), plus the sha256 of every file it cites (the tracks ARE the
    substrate). Two shards of one fill embed one spec written by two workers."""
    if not isinstance(spec, dict):
        return spec
    out = {k: v for k, v in spec.items() if k != "provenance"}
    files = (spec.get("provenance") or {}).get("files") or []
    out["cited_sha256"] = sorted(str(f.get("sha256")) for f in files if isinstance(f, dict))
    return out


def _agree(a, b, rtol=1e-9, atol=1e-12):
    """Whether two JSON-like values agree: floats to rounding, the rest exactly, recursively."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return bool(np.isclose(a, b, rtol=rtol, atol=atol))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_agree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_agree(x, y) for x, y in zip(a, b))
    return a == b


def merge_packs(packs, *, id, out_path=None, overlap="refuse", envelope=None, device="auto"):
    """One pack from the shards of one walk: the packs of voxel blocks of the same substrate, walked with the
    same parameters and codec (a distributed fill: each device seeds and walks its block and packs it with
    ``voxel_grid=``). Every walker-indexed array is concatenated shard after shard. The per-voxel certificate:
    with ``overlap="refuse"`` the shards hold disjoint voxels and the certificate is the union of their rows
    (two shards holding the same voxel are refused); with ``overlap="recertify"`` shards may share voxels -- the
    rounds a small machine walks one block in, or a top-up pass of the same block -- the walker weights are
    renormalised to the union (:func:`union_weights`) and the per-voxel floor
    of the union is read afresh from the merged coefficients (:func:`voxel_floor_coded`, over ``envelope``'s
    battery, default the envelope every pack is built against), the codec-error column ``nan`` as in an
    inherited certificate. The fidelity summary is the conservative one over the shards (the largest error and
    floor, every family within its floor only when every shard was); the precision tiers are recomputed and
    declared unshuffled (the walkers are ordered by shard). Provenance lists the shards and the field read interval
    of each. The shards of a pool bundle read the field at the intervals of their own rounds, so the interval is not
    an identity key: the merged pack states the coarsest and the term charged at it (:func:`charge_field_read`).
    Shards that carry no per-voxel certificate (a walker's shards) keep their walkers' ``voxel_ijk`` ``(n, 3)``, by
    which a recertifying merge renormalises their weights to the union. ``packs`` are :class:`ReplayPack` objects
    or paths. The merged walk is never decoded whole: the floors are read a walker range at a time from the shards'
    own tensors. With ``out_path`` the merged tensors are streamed from the shards' to the file
    (:func:`write_rpk`) and the pack returned is mapped from it (:func:`read_rpk`); without, they are concatenated in
    memory."""
    with Run("merge_packs", params=dict(id=id, n_packs=len(packs), overlap=overlap, out_path=out_path)) as run:
        from ..phantom.grid import Grid
        if overlap not in ("refuse", "recertify"):
            raise ValueError("overlap is 'refuse' (disjoint voxel blocks) or 'recertify' (shards may share voxels; the floors are re-read)")
        pks = [pk if isinstance(pk, ReplayPack) else read_rpk(pk) for pk in packs]
        for pk in pks:
            pk.susceptibility_field                                      # every shard holds the equivalence
        if len(pks) < 2:
            raise ValueError("merge_packs takes at least two shards")
        def same(key, get):
            """The shards' value of ``key``, which must agree; floats agree to rounding (two shards of one spec
            computed the substrate frame on different machines and differ by an ulp), everything else exactly."""
            vals = [get(pk) for pk in pks]
            for v in vals[1:]:
                if not _agree(v, vals[0]):
                    raise ValueError(f"the shards differ in {key}: {vals[0]!r} vs {v!r}")
            return vals[0]
        comp = same("compression", lambda pk: _codec_signature(pk.meta["compression"]))
        wp = same("walk_params", lambda pk: _walk_identity(pk.meta["walk_params"]))
        same("substrate", lambda pk: _spec_identity(pk.meta.get("substrate")))
        same("replay_envelope", lambda pk: _envelope_identity(pk.meta.get("replay_envelope")))
        pv0 = same("per-voxel grid", lambda pk: ((pk.meta.get("fidelity") or {}).get("per_voxel") or {}).get("grid"))
        same("array names", lambda pk: sorted(pk.arrays))
        n = [int(pk.meta["walk_params"]["n_walkers"]) for pk in pks]
        reads = [field_read_record(pk.meta) for pk in pks]
        arrays = {}
        scale_keys = [k for k in pks[0].arrays if k.endswith("_band_scale") or k.split("/")[-1] == "susc_path_scale"]   # every window's
        # a scale table with a block axis: the band scales carry one from the start ((n_blocks, ...)); the path channel's
        # (n_ch, K) gains one here
        table = lambda pk, k: (np.asarray(pk.arrays[k])[None] if (k.split("/")[-1] == "susc_path_scale" and np.asarray(pk.arrays[k]).ndim == 2) else np.asarray(pk.arrays[k]))
        if scale_keys:                                                   # per-pack scale tables: stack the shards' and give
            blocks, off = [], 0                                          # every walker its block
            for pk, m in zip(pks, n):
                nb = int(table(pk, scale_keys[0]).shape[0])
                blk = np.asarray(pk.arrays["band_block"], np.int64) if "band_block" in pk.arrays else np.zeros(m, np.int64)
                blocks.append(blk + off); off += nb
            arrays["band_block"] = np.concatenate(blocks).astype(np.uint16 if off < 65536 else np.int32)
            for k in scale_keys:
                arrays[k] = np.concatenate([table(pk, k) for pk in pks])
        for k in pks[0].arrays:
            if k in ("voxel_certificate", "band_block") or (k == "voxel_ijk" and pv0 is not None) or k in scale_keys:
                continue                                                      # (without a certificate ``voxel_ijk`` is the walkers' own)
            if k.startswith("susc_grid_"):                                    # the substrate's field grid: one table, every shard's
                for pk in pks[1:]:
                    if not np.array_equal(np.asarray(pk.arrays[k]), np.asarray(pks[0].arrays[k])):
                        raise ValueError(f"the shards differ in the field grid {k!r}")
                arrays[k] = np.asarray(pks[0].arrays[k])
                continue
            parts = [pk.arrays[k] for pk in pks]                              # the shards' pieces, concatenated as written
            base = k.split("/")[-1]
            if base.endswith(("_events_t", "_events_label")):                 # an events stream (RPK.md 9.5.7): the shards'
                arrays[k] = parts                                             # records follow each other in walker order
                continue
            if base.endswith("_events_offsets"):                              # the offsets re-based onto the merged stream
                offs, base_off = [parts[0]], int(parts[0][-1])
                for a in parts[1:]:
                    offs.append(np.asarray(a, np.int64)[1:] + base_off); base_off += int(a[-1])
                arrays[k] = np.concatenate(offs).astype(np.int64)
                continue
            if not all(np.ndim(a) >= 1 and np.shape(a)[0] == m and np.shape(a)[1:] == np.shape(parts[0])[1:] for a, m in zip(parts, n)):
                raise ValueError(f"array {k!r} is not walker-leading in every shard; it cannot be concatenated")
            arrays[k] = parts
        fid = dict(pks[0].meta.get("fidelity") or {})
        for key, agg in (("err_max", max), ("floor_max", max), ("noise_floor", max), ("err_surface", max), ("floor_surface", max)):
            vals = [(pk.meta.get("fidelity") or {}).get(key) for pk in pks]
            if all(v is not None for v in vals):
                fid[key] = float(agg(vals))
        if all("within_2x_floor" in (pk.meta.get("fidelity") or {}) for pk in pks):
            fid["within_2x_floor"] = bool(all((pk.meta["fidelity"]["within_2x_floor"]) for pk in pks))
        fid.pop("per_family", None)
        if pv0 is not None:
            ijk = np.concatenate([np.asarray(pk.arrays["voxel_ijk"], np.int64) for pk in pks])
            # the certificate's pool axis is the union of the shards' pools (a round of a pass may hold none of a
            # sparse pool): a pool a shard lacks has no walkers there, its floor and error unread
            pools_of = [[int(p_) for p_ in pk.meta["fidelity"]["per_voxel"]["pools"]] for pk in pks]
            pools = sorted(set().union(*pools_of))

            def aligned(pk, own):
                c_ = np.asarray(pk.arrays["voxel_certificate"], np.float64)
                out = np.full((c_.shape[0], len(pools), 3), np.nan); out[:, :, 0] = 0.0
                for j_, p_ in enumerate(own):
                    out[:, pools.index(p_)] = c_[:, j_]
                return out
            cert = np.concatenate([aligned(pk, own) for pk, own in zip(pks, pools_of)])
            held = held_voxels(cert)
            key = np.ravel_multi_index(tuple(ijk[held].T), tuple(Grid.from_meta(pv0).shape))
            uk, cnt = np.unique(key, return_counts=True)
            if (cnt > 1).any() and overlap == "refuse":
                dup = np.unravel_index(uk[cnt > 1][0], tuple(Grid.from_meta(pv0).shape))
                raise ValueError(f"two shards hold walkers in the same voxel (e.g. {tuple(int(x) for x in dup)}); a voxel belongs to one "
                                 "shard, unless the merge recertifies (overlap='recertify')")
            if overlap == "recertify":                                        # the union's floors from the merged coefficients
                grid = Grid.from_meta(pv0)
                comp = _merged_start_pool(pks)                                    # the pool at the walk's start
                _w = None
                if "spin_weights" in arrays:                                  # the union's weights, not the shards'
                    _vox, _in = grid.bin(np.concatenate([_cx.position_starts(pk.arrays) for pk in pks]))
                    _key = np.where(_in, np.ravel_multi_index(tuple(np.clip(_vox, 0, np.asarray(grid.shape) - 1).T), tuple(grid.shape)), -1)
                    _pid = np.asarray(comp)[:, 0].astype(np.int64) if comp is not None else np.zeros(len(_key), np.int64)
                    _w = union_weights(np.concatenate(arrays["spin_weights"]).astype(np.float64), np.repeat(np.arange(len(pks)), n), _key, _pid)
                    arrays["spin_weights"] = _w.astype(np.float32)
                _ijk, _pools_r, _n, _floor, _err = voxel_floor_coded([pk.arrays for pk in pks], float(pks[0].dt), int(pks[0].n_t),
                                                                     grid, comp, envelope or _cx.default_envelope(), w=_w, device=device)
                if [int(p_) for p_ in _pools_r] != [int(p_) for p_ in pools]:
                    raise ValueError(f"the merged walkers' pools {list(_pools_r)} differ from the shards' certificate pools {pools}")
                ijk, cert = _ijk, np.stack([_n.astype(np.float64), _floor, _err], axis=-1)
                rows = np.arange(len(_ijk))
            else:
                # one row per voxel: the shard that holds it, else the first shard's empty row
                order = {}
                for r_, (i_, c_) in enumerate(zip(map(tuple, ijk), cert)):
                    if i_ not in order or c_[:, 0].sum() > 0:
                        order[i_] = r_
                rows = np.array(sorted(order.values()))
            arrays["voxel_ijk"] = ijk[rows].astype(np.int32); c = cert[rows]
            arrays["voxel_certificate"] = c.astype(np.float32)
            fid["per_voxel"] = per_voxel_summary(pv0, pools, c, shards=len(pks), recertified=bool(overlap == "recertify"))
        elif overlap == "recertify" and "spin_weights" in arrays and "voxel_ijk" in arrays:
            # shards without a certificate (a writer that measures no floor) carry their walkers' own voxels: the union's
            # weights need no grid, a voxel is a row of ``voxel_ijk``
            comp = _merged_start_pool(pks)
            _, _key = np.unique(np.concatenate(arrays["voxel_ijk"]).astype(np.int64), axis=0, return_inverse=True)
            _pid = np.asarray(comp)[:, 0].astype(np.int64) if comp is not None else np.zeros(len(_key), np.int64)
            arrays["spin_weights"] = union_weights(np.concatenate(arrays["spin_weights"]).astype(np.float64), np.repeat(np.arange(len(pks)), n),
                                                   _key.reshape(-1), _pid).astype(np.float32)
        n_all = int(sum(n))
        comp_meta = dict(pks[0].meta["compression"])
        if "channels" in comp_meta:                                      # the measured numbers: the worst over the shards
            chans = {c: (dict(m) if isinstance(m, dict) else m) for c, m in comp_meta["channels"].items()}
            for c, m in chans.items():
                if isinstance(m, dict):
                    for k in _MEASURED_CHANNEL_KEYS:
                        vals = [((pk.meta["compression"].get("channels") or {}).get(c) or {}).get(k) for pk in pks]
                        if all(v is not None for v in vals):
                            m[k] = float(max(vals)) if not isinstance(vals[0], dict) else vals[0]
            comp_meta["channels"] = chans
        if comp_meta.get("walker_preserving"):
            comp_meta["precision_tiers"] = _precision_tiers(arrays, n_all, float(fid.get("floor_max") or 0.0), False)
        meta = dict(pks[0].meta)
        seeds = [pk.meta["walk_params"]["seed"] for pk in pks]
        seg0 = pks[0].meta["walk_params"].get("windows") or {}
        meta.update(id=id, compression=comp_meta, fidelity=fid,
                    walk_params=dict(pks[0].meta["walk_params"], n_walkers=n_all, seed=seeds,
                                     windows=dict(seg0, walks=[dict(w, seed=seeds) for w in seg0.get("walks", [])]),
                                     **_merged_qv([pk.meta["walk_params"].get("qv") for pk in pks], n)),
                    provenance=dict(pks[0].meta.get("provenance") or {}, shards=[
                        dict(id=pk.meta.get("id"), n_walkers=int(m), **({} if r is None else dict(field_read_interval=r["interval"])))
                        for pk, m, r in zip(pks, n, reads)]))
        charge_field_read(meta, reads)                                   # each shard read the field at its own interval; the pack states the coarsest
        ReplayPack(arrays, meta).susceptibility_field                    # the merged pack holds the equivalence
        if out_path is None:
            return ReplayPack({k: np.concatenate(v) if isinstance(v, list) else v for k, v in arrays.items()}, meta)
        write_rpk(out_path, arrays, meta)
        run.artifact(out_path)
        return read_rpk(out_path)


WINDOW_T = 0.1          #: the storage rule's window (s): a pack stores its walk in windows of this duration (RPK.md 4.3)
STORAGE_BAND_MARGIN = 1.4  #: the storage rule's margin over a class's certified band (the paper's K = 256 per 100 ms for the human classes)


def _walk_scanner(walk):
    """The scanner class a walk's band was derived for (``CoefficientWalk.scanner``, set by ``walk_spec``), or None."""
    return getattr(walk, "scanner", None) if not isinstance(walk, dict) else walk.get("scanner")


def class_band(scanner):
    """The band (Hz) a walk for scanner class ``scanner`` resolves its step for: the class's stored band
    (:func:`storage_band_hz`, the certified band times its margin); ``None`` -- the block's Nyquist frequency -- when
    no class is named."""
    return None if scanner is None else float(storage_band_hz(scanner))


def storage_band_hz(scanner):
    """The band a pack built for ``scanner`` is stored at (Hz): the class's certified band -- the highest frequency any
    waveform the class can deliver needs of a pack, the catalogue's ``band.certified_band_hz`` -- times
    :data:`STORAGE_BAND_MARGIN`. A class the catalogue holds no certified band for is refused by name: the band is a
    measured property of the class (the closed-form certificate of the replayable-MC paper, ``figures/fig_band_data.py``),
    never a guess from its amplitude or slew."""
    from ..acquisition.scanners import ScannerLimits
    lim = ScannerLimits.of(scanner)
    if lim.certified_band_hz is None:
        raise ValueError(f"the catalogue holds no certified band for scanner class {lim.name!r}: a pack cannot be stored for it "
                         f"until band.certified_band_hz is measured and cited (the replayable-MC paper's fig_band_data on the class's "
                         f"envelope); pass K= or temporal_bandwidth_hz= to store at a stated band instead")
    return float(STORAGE_BAND_MARGIN) * float(lim.certified_band_hz)


def mode_count(T, scanner=None, *, temporal_bandwidth_hz=None):
    """The band count ``K = ceil(2 f T)`` of a window of ``T`` seconds stored at ``f`` = :func:`storage_band_hz` of
    ``scanner`` (or a stated ``temporal_bandwidth_hz``): the one derivation every producer reads (#643)."""
    f = float(temporal_bandwidth_hz) if temporal_bandwidth_hz is not None else storage_band_hz(scanner)
    return max(2, int(np.ceil(2.0 * f * float(T))))
_SHARED_KEYS = ("spin_weights", "comp_static", "band_block", "voxel_ijk", "voxel_certificate")


def _is_shared_key(k):
    """Whether a tensor is the walk's rather than a window's: per-walker constants and substrate tables."""
    return k in _SHARED_KEYS or k.startswith("susc_grid_")


def combine_window_fidelity(fids):
    """The whole walk's certificate as the BOUND over its windows' (RPK.md 4.3): every error term the sum over
    the windows (a replay across them sums their contributions, which may add coherently), every floor the largest,
    ``combined = "bound_over_windows"``, and ``certified = "measured"`` (every window certifies itself: walk-free terms,
    or the battery on its raw subset; RPK.md 9.4 rule 4). What a pack of several windows, a prefix of whole windows or
    an appended continuation carries."""
    fids = [dict(f) for f in fids]
    out = dict(metric=fids[0].get("metric"), combined="bound_over_windows", windows=fids, certified="measured")
    keys = sorted({k for f in fids for k in f if k.startswith(("err_", "floor_"))})
    for k in keys:
        vals = [f.get(k) for f in fids]
        if any(v is None for v in vals):
            continue
        out[k] = float(sum(vals)) if k.startswith("err_") else float(max(vals))
    fams = sorted({fam for f in fids for fam in (f.get("per_family") or {})})
    if fams:
        out["per_family"] = {fam: dict(err_max=float(sum(f["per_family"][fam]["err_max"] for f in fids if fam in (f.get("per_family") or {}))),
                                       floor_max=float(max(f["per_family"][fam]["floor_max"] for f in fids if fam in (f.get("per_family") or {}))))
                             for fam in fams}
    if "noise_floor" in fids[0]:
        out["noise_floor"] = float(max(f.get("noise_floor", 0.0) for f in fids))
    if all(isinstance(f.get("terms"), dict) for f in fids):                # the bounded terms add like the errors
        t0 = fids[0]["terms"]
        out["terms"] = dict(t0, truncation=float(sum(f["terms"]["truncation"] for f in fids)),
                            quadrature={c: float(sum(f["terms"]["quadrature"].get(c, 0.0) for f in fids))
                                        for c in t0["quadrature"]})
        if any("rounding" in f["terms"] for f in fids):
            out["terms"]["rounding"] = float(sum(f["terms"].get("rounding", 0.0) for f in fids))
        if any("field_read" in f["terms"] for f in fids):
            out["terms"]["field_read"] = float(sum(f["terms"].get("field_read", 0.0) for f in fids))
    if "err_max" in out and "floor_max" in out:
        out["within_2x_floor"] = bool(out["err_max"] <= 2.0 * out["floor_max"])
    return _cover_terms(out)


def _round_sig(x, sig):
    """``x`` rounded to ``sig`` significant decimal digits; 0, nan and inf pass through unchanged."""
    if x == 0 or not np.isfinite(x):
        return x
    return float(f"{x:.{sig - 1}e}")


def _round_floats(x, sig):
    """``x`` with every float leaf rounded to ``sig`` significant digits, recursing through dicts, lists and
    tuples; bools and ints (and everything else) pass through unchanged."""
    if isinstance(x, bool):
        return x
    if isinstance(x, float):
        return _round_sig(x, sig)
    if isinstance(x, dict):
        return {k: _round_floats(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_round_floats(v, sig) for v in x)
    return x


def _round_certificate(fid, sig=6):
    """The certificate's statistics (``fidelity.*``: every ``err_*``/``floor_*``, ``per_family``, ``per_voxel``,
    the per-window list) rounded to ``sig`` significant digits before they are written.

    6 digits is far below any floor's MEANING -- a split-half Monte-Carlo floor is itself an estimate, not an
    exact quantity -- and far above the 17th digit a float64 reduction (a sum over many walkers or windows)
    differs in when the same build runs on a different core count, BLAS build or thread schedule (dmipy-sim#541:
    the same ``fidelity.per_family.OGSE3.err_max`` read ``0.0018340476685469705`` on one host and
    ``0.00183404766854697`` on another). Rounded, the two write the same bytes.
    """
    return _round_floats(fid, sig)


def _write_run_sidecar(out_path, run, walk):
    """The run that built the pack, beside it (``<out_path>.run.json``) rather than inside it: the run's id,
    timestamps, wall time, pid, host and its absolute record path (:class:`~dmipy_sim.run.Run`) are not a
    function of what was built, so a pack's bytes stay a function of its content only when they live here.

    Beside the pack, not inside the run's own record directory: a :class:`~dmipy_sim.run.Run` persists a
    directory of its own only once it outlives the sampler interval or is given ``run_dir=`` explicitly, so most
    of a bank's packs -- the small or fast ones, this module's own tests among them -- would have no such
    directory to hold a sidecar. A file written next to the pack it describes exists exactly when the pack does.

    Names the pack's own sha256 (:func:`dmipy_sim.fill.hub.sha256_of`, what a consumer pins --
    :func:`dmipy_sim.replay.publish.manifest_row`), so the sidecar for a pack found by its hash is the one its
    bytes point to. Returns the sidecar's path."""
    from ..fill.hub import sha256_of
    w = getattr(walk, "run", None)
    sha = sha256_of(out_path)
    sidecar = dict(pack_sha256=sha, pack=run.summary,
                   walk=(None if w is None else (w.summary if hasattr(w, "summary") else w)),
                   work=getattr(walk, "work", None))
    path = out_path + ".run.json"
    with open(path, "w") as f:
        json.dump(sidecar, f, indent=1, default=str)
    run.artifact(path, sha256=sha)
    return path


def _precision_tiers(arrays, n_walkers, floor_max, walkers_shuffled):
    """How many leading walkers a consumer must read to reach a given statistical floor.

    Every walker-indexed array is stored walkers-LEADING, so the first ``n`` walkers are one
    contiguous byte range per tensor -- a range GET / ``safetensors.get_slice(name)[0:n]``, no need to
    pull the whole pack. The MC floor scales as 1/sqrt(n), so n(eps) = n_walkers*(floor_max/eps)^2.

    This is only meaningful if walker ORDER is exchangeable. Producers that seed pool-by-pool (e.g.
    intra-axonal then myelin) must shuffle, or a prefix silently returns a one-pool substrate; the
    producer asserts that by setting ``walkers_shuffled`` on the master. Without it the tiers are
    still reported but flagged unusable, rather than quietly offering a biased read.
    """
    per_walker, fixed, whole = 0, 0, []
    for k, v in arrays.items():
        ps = [np.asarray(x) for x in v] if isinstance(v, list) else [np.asarray(v)]   # a merge's tensor: its shards' pieces
        nbytes = sum(int(x.nbytes) for x in ps)
        if all(x.ndim >= 1 for x in ps) and sum(int(x.shape[0]) for x in ps) == n_walkers:
            per_walker += int(nbytes // max(n_walkers, 1))
        else:
            # Not walker-leading -- e.g. a run-length channel whose runs vary per walker. Row n of such
            # an array is not walker n, so it cannot be prefix-sliced and must be read whole. Counting
            # it as fixed overhead keeps a small tier's quoted size honest instead of understating it.
            fixed += nbytes; whole.append(k)
    tiers = []
    if floor_max > 0:
        for eps in (1e-2, 3e-3, 1e-3):
            n = int(min(n_walkers, max(1, np.ceil(n_walkers * (floor_max / eps) ** 2))))
            tiers.append(dict(eps=float(eps), n_walkers=n, bytes=int(n * per_walker + fixed),
                              reachable=bool(n < n_walkers or floor_max <= eps)))
    return dict(bytes_per_walker=per_walker, fixed_bytes=fixed,
                not_prefix_sliceable=sorted(whole), floor_at_full_n=float(floor_max),
                walkers_shuffled=bool(walkers_shuffled),
                usable=bool(walkers_shuffled), tiers=tiers,
                note=("prefix reads are unbiased" if walkers_shuffled else
                      "WALKER ORDER NOT DECLARED SHUFFLED -- a prefix may be a biased sub-ensemble"))


def _container(spec):
    """A band container from the builder's knob: ``None`` (the float container), ``"derived"`` (the default,
    :data:`compression.BAND_CONTAINER`: derived per pack against its floor) or an explicit ``((upto, bits), ...)`` or the
    ``[{"bands", "bits"}, ...]`` of :func:`compression.band_curve`."""
    if spec is None:
        return None
    if isinstance(spec, (_cx.BandCurve, _cx.DerivedContainer)):
        return spec
    if isinstance(spec, str):
        if spec != "derived":
            raise ValueError(f"container must be None, 'derived', or ((upto, bits), ...); got {spec!r}")
        return _cx.BAND_CONTAINER
    return tuple((None if u is None else int(u), int(b)) for u, b in
                 ((r["bands"][1] if "bands" in r else r["modes"][1], r["bits"]) if isinstance(r, dict) else r for r in spec))


def _measured_truncation(fid):
    """A certificate measured against the walk's own block ends has measured what the bands beyond ``K`` drop: its
    ``err_max`` bounds the truncation over the battery, so ``terms.truncation`` is the smaller of it and the free
    bridge's walk-free bound (``truncation_from = "pilot"`` when the measurement is the smaller, RPK.md 9.5.5)."""
    t = fid.get("terms")
    if isinstance(t, dict) and fid.get("err_max") is not None and float(fid["err_max"]) < float(t.get("truncation", 0.0)):
        t["truncation"] = float(fid["err_max"]); t["truncation_from"] = "pilot"
    return fid


def _cover_terms(fid):
    """``err_max`` is at least the sum of ``fidelity.terms`` (RPK.md 9.5.5): each term is a maximum over the battery,
    so their sum can exceed the largest per-measurement total; a pack reports the larger. ``within_2x_floor`` follows."""
    t = fid.get("terms")
    if not isinstance(t, dict) or fid.get("err_max") is None:
        return fid
    total = (float(t.get("truncation", 0.0)) + float(sum((t.get("quadrature") or {}).values()))
             + float(t.get("field_read", 0.0)) + float(t.get("rounding", 0.0)))
    if total > float(fid["err_max"]):
        fid["err_max"] = total
    if fid.get("floor_max") is not None:
        fid["within_2x_floor"] = bool(float(fid["err_max"]) <= 2.0 * float(fid["floor_max"]))
    return fid


def _terms_of(fid):
    """``fidelity.terms`` (RPK.md 9.5.5) of a certificate that carries its bounded terms as ``err_truncation`` /
    ``err_quadrature`` / ``err_quadrature_surface``."""
    q = dict(positions=float(fid.get("err_quadrature", 0.0)))
    if fid.get("err_quadrature_surface") is not None and fid.get("has_contact"):
        q["local_time"] = float(fid["err_quadrature_surface"])
    f = fid.get("field")
    if f is not None:                                                 # iff the field channel is present
        q["field"] = float(f["quadrature"])
    out = dict(truncation=float(fid.get("err_truncation", 0.0)), truncation_from="band_variance", quadrature=q)
    if f is not None:
        out["field_read"] = float(f["field_read"])
    if fid.get("err_rounding") is not None:                           # REQUIRED iff block.precision is float32
        out["rounding"] = float(fid["err_rounding"])
    return out


def _battery_b_max(env):
    """The largest b (s/m^2) of the envelope's battery (:func:`compression.acquisition_battery`)."""
    return float(max(list(env.get("bvals") or [0.0]) + [float(env.get("shortd_b") or 0.0)]))


def _derived_certificate(scales, *, T, h, K, m, D, env, floor, qv_pos=None, qv_contact=None, rho_over_D=None,
                         rounding_rel=None, field=None, h_round=None):
    """The walk-free certificate of a pack built from a coefficient walk, per measurement of the envelope's battery
    (on the window's block ends) and per family: the TRUNCATION, the phase the bands beyond ``K`` would carry for a
    free walk at ``D`` (:func:`compression.band_phase_variances`), half of it as the signal it costs; the QUADRATURE
    of the block table at ``(m, h)``, per channel ``rel(m, h) x QV`` with ``QV`` the channel's measured quadratic
    variation (the walkers' mean): on the positions half of ``gamma^2 sum_{k,c} W_kc^2 rel(m, h, omega_k) T QV_c /
    (pi k)^2`` (``QV_c = 2 D T`` on a free walk, where it is the free band variance's statement), on the contact half
    of ``(rho / D)^2 sum_k rel(m, h, omega_k) 2 QV_B / (pi k)^2`` per jump of a gate at the envelope's largest ``rho /
    D`` (an ungated contact is the exact endpoint) (:func:`compression.quadrature_rel`); the QUANTISATION of the
    stored containers, the random phase ``eps^2 = gamma^2 sum_k W_k^2 step_k^2 / 12`` of the positions' realised steps
    and the field's (``field["quantisation_eps"]``, :func:`field_quantisation`: its own phase, added in variance), split
    by what walkers do to it: its bias ``eps^2 / 2`` (the attenuation ``<e^{i delta}>``, which no walker count removes)
    is the numerical term ``err_quantisation`` (``err_quantisation_field`` the field's alone), and its random part,
    independent per walker, rides with the floor: the standard deviations add, so the quantised pack's standard error
    is at most the walkers' plus ``eps / sqrt(n_eff)`` (``floor_quantisation``, ``floor_total_max`` the sum; measured on a
    DiSCo-like K = 260 strands walk: the attenuation within 30 % of ``eps^2 / 2`` including the 3-bit bands' gain); and the
    split-half floor ``floor`` of the float walkers (:func:`compression.measure_floor_coded`). The error per
    measurement is the sum of the three. ``rounding_rel`` (a float32 accumulation's measured relative rounding of
    its bands, ``block.k0_residual_relative``) adds the ROUNDING term of RPK.md 9.5.5: a relative error ``r`` on
    every band is a phase error correlated with the phase, ``|<e^{i phi (1 + r)}> - <e^{i phi}>| <= r sqrt(<phi^2>)
    <= r sqrt(2 b_max D)``, with ``b_max`` the battery's largest b and ``D`` the bounds' diffusivity."""
    from ..constants import GAMMA
    T, h, K = float(T), float(h), int(K)
    n_t = int(round(T / h)) + 1
    G, meta = _cx.acquisition_battery(n_t, h, env)
    pb, total = _cx.band_phase_variances(G, h, T, K, float(D))
    trunc = 0.5 * np.clip(total - pb.sum(axis=1), 0.0, None)
    W = _cx.band_projection(G, h, T, K)[:, 2:, :]                                            # (n_meas, K, 3)
    k = np.arange(1, K + 1)
    rel = _cx.quadrature_rel(m, h, np.pi * k / T)
    if h_round is not None:                           # an adaptive walk: a round's interior path is unresolved (order 0 at the round)
        rel = rel + _cx.quadrature_rel(0, float(h_round), np.pi * k / T)
    qv = np.full(3, 2.0 * float(D) * T) if qv_pos is None else np.asarray(qv_pos, np.float64)
    quad = 0.5 * GAMMA ** 2 * np.einsum("mkc,k,c->m", W ** 2, rel * T / (np.pi * k) ** 2, qv)
    quad_surface = 0.0
    if qv_contact is not None and rho_over_D is not None:
        quad_surface = 0.5 * float(rho_over_D) ** 2 * float((rel * 2.0 * float(qv_contact) / (np.pi * k) ** 2).sum())
    if scales is None:
        eps = np.zeros_like(trunc)
    else:
        step2 = np.mean(np.asarray(scales, np.float64) ** 2, axis=0)                           # (3, K) over the walkers
        eps = np.sqrt(GAMMA ** 2 * np.einsum("mkc,ck->m", W ** 2, step2) / 12.0)
    eps_f = 0.0 if field is None else float(field.get("quantisation_eps", 0.0))               # the field's, a phase of its own
    quant_of = lambda e: e ** 2 / 2.0                                                         # the bias; the random part is the floor's
    eps_all = np.sqrt(eps ** 2 + eps_f ** 2)                                                  # independent phases add in variance
    quant = quant_of(eps_all)
    floor_q = eps_all / np.sqrt(float(floor.get("n_eff") or 1.0 / float(floor["noise_floor"]) ** 2))
    rounding = None if rounding_rel is None else float(rounding_rel) * float(np.sqrt(2.0 * _battery_b_max(env) * float(D)))
    field_err = 0.0 if field is None else float(field["quadrature"]) + float(field["field_read"])     # the field's terms (:func:`_field_terms`)
    err = trunc + quad + quant + quad_surface + (0.0 if rounding is None else rounding) + field_err
    fams = sorted({x["fam"] for x in meta})
    per = {}
    for f in fams:
        j = np.array([i for i, x in enumerate(meta) if x["fam"] == f])
        per[f] = dict(err_max=float(err[j].max()), floor_max=float(floor["per_family"][f]))
    return dict(metric="max_abs_complex_signal_error", certified="measured", measured_by="walk_free_terms",
                err_truncation=float(trunc.max()), err_quadrature=float(quad.max()), err_quantisation=float(quant.max()),
                err_quadrature_surface=float(quad_surface),
                **({} if rounding is None else dict(err_rounding=rounding)),
                **({} if field is None else dict(err_quadrature_field=float(field["quadrature"]), err_field_read=float(field["field_read"]),
                                                 err_quantisation_field=float(quant_of(eps_f)))),
                err_max=float(err.max()), floor_max=float(floor["floor_max"]), noise_floor=float(floor["noise_floor"]),
                floor_quantisation=float(floor_q.max()), floor_total_max=float(floor["floor_max"]) + float(floor_q.max()),
                within_2x_floor=bool(err.max() <= 2.0 * float(floor["floor_max"])), per_family=per,
                quadrature=dict(m=int(m), h=h, f_band_hz=float(K) / (2.0 * T), h_round=(None if h_round is None else float(h_round)),
                                qv_positions_m2=(None if qv_pos is None else [float(x) for x in qv_pos]),
                                qv_contact_m2=(None if qv_contact is None else float(qv_contact)),
                                surface_rho_over_D=(None if rho_over_D is None else float(rho_over_D)),
                                rel_at_band_edge=float(_cx.quadrature_rel(m, h, np.pi * K / T))),
                bounds_diffusivity=float(D))


def _coefficient_weights(cw, weights):
    """The walkers' proton-density weights of a coefficient walk: ``weights``, else the walk's own, else the spec's
    water fractions by pool; a walker in a pool the spec says holds no water is refused by name."""
    spec = cw.spec
    comp = cw.compartment
    comp0 = None if comp is None else (np.asarray(comp) if np.ndim(comp) == 1 else np.asarray(comp)[:, 0])
    if weights is None and cw.weights is None and comp0 is not None:
        wf = [p.water_fraction for p in sorted(spec.pools, key=lambda p: p.id)]
        pool0 = np.rint(comp0).astype(int)
        dry = [(p, int(np.sum(pool0 == p.id))) for p in spec.pools if p.water_fraction == 0.0 and np.any(pool0 == p.id)]
        if dry:
            raise ValueError("the walk has walkers in a pool the spec says holds no water: "
                             + ", ".join(f"{n:,} in pool {p.id} ({p.name})" for p, n in dry)
                             + "; such walkers would weigh 0 and the pack would silently omit them. Seed the pools the "
                               "spec gives water to, give the pool its water in the spec, or pass weights= explicitly")
        if any(f != 1.0 for f in wf):
            weights = np.asarray(wf, float)[pool0]
    if weights is None and cw.weights is not None:
        weights = cw.weights
    if weights is not None:
        weights = np.asarray(weights, float).reshape(-1)
        if weights.shape[0] != cw.n_walkers:
            raise ValueError(f"weights has {weights.shape[0]} entries for {cw.n_walkers} walkers")
    return weights


def join_windows(packs, *, fidelity, walks, walkers_shuffled):
    """The packs of the consecutive windows of one walk -- each one window of the same block, codec and band, as
    :func:`build_replay_pack` writes one -- as ONE pack's ``(arrays, meta)`` (RPK.md 4.3): window 0's tensors under the
    channel names and window ``i``'s under ``s{i}/``, the tensors the walk shares (``_SHARED_KEYS``, the field grids)
    once, refused when two windows disagree in one; the channel numbers a codec measures the worst over the windows;
    the window table ``{n, T, walks}`` and the block record over the whole walk; ``fidelity`` the whole's certificate
    and the precision tiers read from it. The metadata is window 0's otherwise."""
    arrays = dict(packs[0].arrays)
    for i, pk in enumerate(packs[1:], start=1):
        for k, v in pk.arrays.items():
            if _is_shared_key(k):
                if not np.array_equal(np.asarray(v), np.asarray(arrays[k])):
                    raise ValueError(f"window {i} disagrees with window 0 in the shared tensor {k!r}")
                continue
            arrays[f"s{i}/{k}"] = v
    meta = json.loads(json.dumps(packs[0].meta))
    n_windows = len(packs)
    T_win = float(packs[0].meta["walk_params"]["windows"]["T"])
    cm = meta["compression"]
    chans = cm.get("channels") or {}
    for c, mm in chans.items():                                   # the measured numbers: the worst over the windows
        if isinstance(mm, dict):
            for k in _MEASURED_CHANNEL_KEYS:
                vals = [((pk.meta["compression"].get("channels") or {}).get(c) or {}).get(k) for pk in packs]
                if all(v is not None for v in vals):
                    mm[k] = float(max(vals)) if not isinstance(vals[0], dict) else vals[0]
    n_walkers = int(packs[0].n_walkers)
    if cm.get("walker_preserving"):
        cm["precision_tiers"] = _precision_tiers(arrays, n_walkers, float(fidelity.get("floor_max") or 0.0), bool(walkers_shuffled))
    blk = meta["walk_params"]["block"]
    blk["n_blocks"] = int(n_windows) * int(blk["n_blocks"])
    for key in ("k0_residual_m", "k0_residual_relative"):                  # what the windows measured: the worst
        vals = [(pk.meta["walk_params"].get("block") or {}).get(key) for pk in packs]
        if all(v is not None for v in vals):
            blk[key] = float(max(vals))
    meta["walk_params"].update(T_max=float(n_windows * T_win),
                               windows=dict(n=int(n_windows), T=T_win, walks=list(walks)))
    qvs = [pk.meta["walk_params"].get("qv") for pk in packs]
    if all(isinstance(q, dict) for q in qvs):                                 # summed over the walk's increments
        meta["walk_params"]["qv"] = {k: (float(sum(q[k] for q in qvs)) if not isinstance(qvs[0][k], list)
                                         else [float(sum(x)) for x in zip(*[q[k] for q in qvs])]) for k in qvs[0]}
    meta["fidelity"] = fidelity
    return arrays, meta


# ------------------------------------------------------------------------------------------ the field channel (C3)
def susc_path_encode_modes(coeffs, names, *, T, container, atol_zero=1e-4, trace_residual=None, max_refocus_pulses=None,
                           read_interval=None, device="numpy", dtype=np.float32):
    """The field channel's tensors and descriptor from the modes a walk ACCUMULATED (``coeffs`` ``(n_w, K, n_ch)``, the
    window's orthonormal cosine modes ``c_k = int e_k b dt``, ``k = 0..K-1``, in the canonical channel order with
    ``names``: the twelve of :data:`~dmipy_sim.fields.hollow_cylinder.CHANNEL_NAMES`, or the six isotropic ones).
    ``iso_P_zz`` is never a component (RPK.md 5.3: a reader forms it from the trace); a walk that offers it is refused.
    A component whose modes are zero in every walker to ``atol_zero`` of the largest channel's (the z-components of a
    straight cylinder along z, the ``aniso_G`` block of a source without an anisotropic part) is not stored: it is
    named in ``zero_components`` and a reader returns zeros. The modes are stored in ``container``: ranges of modes at
    their own depths (:func:`path_codec`; a :class:`DerivedFieldContainer` is derived here, against the stored
    channels' own scales and its floor), or one native depth 8 / 16, under ``susc_path_dct`` with ``susc_path_scale``
    (CODEC_REGISTRY ``susc_field_basis``); ``None`` stores the float tensor at ``dtype``; a :class:`FieldCodec` states
    the channels as well (the certifying pack's, which a shard of a fill takes), and a walk that carries a component
    the layout leaves out is refused by name. The descriptor carries ``K``, ``T`` (the window), the channels, the
    read interval, the refocusing depth the band serves and ``trace_residual`` (a diagnostic: the identity's residual
    the field source measured, :attr:`~dmipy_sim.fields.susceptibility_field.FieldGrid.trace_residual`)."""
    coeffs = np.transpose(np.asarray(coeffs, np.float64), (0, 2, 1))                   # (n_w, n_ch, K)
    names = list(names)
    n_w, n_ch, K = coeffs.shape
    if len(names) != n_ch:
        raise ValueError(f"{len(names)} channel names for {n_ch} channels")
    _refuse_trace_component(names)
    codec = container if isinstance(container, FieldCodec) else None
    if codec is not None:
        container = codec.container
    amax = np.abs(coeffs).max(axis=(0, 2))
    zero = [n for i, n in enumerate(names) if amax[i] <= atol_zero * float(amax.max())]
    if codec is not None:
        stored = list(codec.channels)
        missing = [n for n in stored if n not in names]
        if missing:
            raise ValueError(f"the stated field layout stores {missing}, which the walk's channels do not carry")
        lost = [n for n in names if n not in stored and n not in zero]
        if lost:
            raise ValueError(f"the stated field layout leaves out {lost}, which these modes carry above {atol_zero} of the "
                             "largest channel; the walk is refused rather than a component dropped")
        keep = [names.index(n) for n in stored]
        zero = [n for n in names if n not in stored]
    else:
        if zero and len(zero) == len(names):
            zero = zero[1:]
        keep = [i for i, n in enumerate(names) if n not in zero]
    meta = dict(channel="susc_field_basis", K=int(K), T=float(T), n_ch=len(keep), channels=[names[i] for i in keep],
                trace_residual=(None if trace_residual is None else float(trace_residual)),
                max_refocus_pulses=_depth(K, max_refocus_pulses), sampling="accumulated_in_walk")
    if zero:
        meta["zero_components"] = zero
    if read_interval is not None:
        meta["read_interval"] = float(read_interval)
    coeffs = np.ascontiguousarray(coeffs[:, keep, :])
    if container is None:
        meta["bits"] = None; meta["dtype"] = np.dtype(dtype).name
        return {"susc_path_dct": np.asarray(coeffs, dtype)}, meta
    if isinstance(container, DerivedFieldContainer):
        container = container.ranges(np.abs(coeffs).max(axis=0), meta["channels"], T)
    return _quantise_susc_path(coeffs, meta, container, device=device)


def _mode_sample_matrix(K, T, n_t, start=0, stop=None, edges=None):
    """``(K, n)``: the orthonormal cosine modes of a window of ``T`` seconds read as the samples ``start .. stop - 1``
    of ``n_t`` -- sample 0 the series at ``t = 0``, sample ``j >= 1`` its mean over the block ``[(j - 1) h, j h]``,
    ``h = T / (n_t - 1)`` (the block-mean reading a raw subset's block means are compared with), or over
    ``[edges[j - 1], edges[j]]`` with ``edges`` ``(n_t,)`` the instants of a series on intervals of their own lengths
    (a walk's read instants)."""
    n_t = int(n_t); stop = n_t if stop is None else int(stop); start = int(start)
    h = float(T) / (n_t - 1)
    E = (np.arange(n_t) * h) if edges is None else np.asarray(edges, np.float64)
    j = np.arange(start, stop)
    k = np.arange(int(K))[:, None]
    om = np.pi * k / float(T)
    D = np.zeros((int(K), j.size))
    e_k0 = np.sqrt(2.0 / float(T)) * np.ones((int(K), 1)); e_k0[0] = 1.0 / np.sqrt(float(T))
    at0 = j == 0
    if at0.any():
        D[:, at0] = e_k0
    jb = j[~at0]
    if jb.size:
        a, b = E[jb - 1], E[jb]
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.sqrt(2.0 / float(T)) * (np.sin(om * b[None, :]) - np.sin(om * a[None, :])) / om / (b - a)[None, :]
        m[0] = 1.0 / np.sqrt(float(T))
        D[:, ~at0] = m
    return D


def susc_path_decode(arrays, meta, *, n_t, n_w=None, n_cut=None, start=0, edges=None):
    """The field's channels per walker on ``n_t`` samples of the window ``meta["T"]`` -- sample 0 the series at ``t =
    0``, sample ``j`` its mean over block ``j`` (:func:`_mode_sample_matrix`) -- from the stored modes, the omitted components
    zero: ``(field (n_w, n_ch_full, n_samples), names)`` in the canonical channel order. ``start`` / ``n_cut``
    read the samples ``start .. n_cut - 1`` only. A reader's sampling of the stored function (RPK.md 9.5.4), what
    the Bloch route and the dense oracles read; the scalar route contracts the modes (:func:`path_field_integral`)."""
    C, names = susc_path_coeffs(arrays, meta)
    if n_w is not None:
        C = C[:int(n_w)]
    D = _mode_sample_matrix(C.shape[2], float(meta["T"]), int(n_t), start, n_cut, edges=edges)
    return C @ D, names


def path_field_integral(arrays, meta, waveform, T, *, t0=None, n_w=None):
    """The gated path integral of every field channel per walker, ``(Psi (n_w, n_ch), names)``: ``Psi = gamma int
    gate(t) b(t) dt = gamma sum_k g_k c_k`` with ``g_k`` the gate's cosine projection on the window ``[t0, t0 + T]``
    (:func:`~dmipy_sim.replay.compression.cosine_projection`; Parseval, RPK.md 9.5.4) and ``c_k`` the stored modes.
    The names are in the canonical order, what :func:`~dmipy_sim.fields.hollow_cylinder.field_terms` reads."""
    from ..constants import GAMMA
    Cs, names = susc_path_coeffs(arrays, meta)
    if tuple(names) not in (tuple(_PATH_CHANNELS), tuple(_ISO_PATH_CHANNELS)):
        raise ValueError(f"the path channels are stored as {names}, not in the canonical order {tuple(_PATH_CHANNELS)}")
    if n_w is not None:
        Cs = Cs[:n_w]
    gate = waveform.effective_gate if waveform.gate is None else waveform.gate
    gate = np.asarray(gate, np.float64).reshape(-1)[:-1]                       # the readout sample is held over nothing (#225)
    g = _cx.cosine_projection(gate.reshape(1, -1, 1), float(waveform.dt), float(T), Cs.shape[2], t0=t0)[0, :, 0]
    return GAMMA * np.einsum("k,wck->wc", g, Cs), names


def _field_scale(env, spec=None):
    """``(chi_iso, chi_aniso)`` the field certificates are measured at: the spec's nominal of its magnetic pools, else the
    certificate's defaults (:data:`~dmipy_sim.engine.accumulate.FIELD_RULE_CHI`, a tenth of it anisotropic)."""
    chi_i, chi_a = None, None
    if spec is not None:
        for p in getattr(spec, "field_source_pools", []):
            s = p.susceptibility
            if s.chi_iso:
                chi_i = max(chi_i or 0.0, abs(float(s.chi_iso)))
            if s.chi_aniso:
                chi_a = max(chi_a or 0.0, abs(float(s.chi_aniso)))
    chi_i = float(chi_i) if chi_i else float(FIELD_RULE_CHI)
    chi_a = float(chi_a) if chi_a else 0.1 * chi_i
    return chi_i, chi_a


def read_interval_means(reads):
    """A raw subset's field reads at the walk's read instants ``(n_w, n_reads + 1, n_ch)`` as the series a field
    oracle integrates: sample 0 the start's read, sample ``i`` the mean of the chord over read interval ``i``, ``(b_{i-1}
    + b_i) / 2`` -- the trapezoid the walk accumulated, interval by interval."""
    b = np.asarray(reads, np.float64)
    return np.concatenate([b[:, :1], 0.5 * (b[:, :-1] + b[:, 1:])], axis=1)


def read_block_means(reads, edges, reads_per_block, h):
    """A raw subset's reads at the walk's read instants as each block's mean of the chord through them, ``(n_w,
    n_blocks + 1, n_ch)`` (sample 0 the start's read): exact, since every block ends on a read."""
    b = np.asarray(reads, np.float64); E = np.asarray(edges, np.float64)
    trap = 0.5 * (b[:, :-1] + b[:, 1:]) * np.diff(E)[None, :, None]
    n_w, n_r, n_ch = trap.shape
    blocks = trap.reshape(n_w, n_r // int(reads_per_block), int(reads_per_block), n_ch).sum(2) / float(h)
    return np.concatenate([b[:, :1], blocks], axis=1)


def _susc_path_fidelity(series_raw, arrays, pm, gm, env, *, h=None, edges=None, w=None, chi=None):
    """Certify the field channel AT ITS DECLARED CAPABILITY on a raw subset's reads: the stored (truncated, quantised)
    modes decoded on the walk's read intervals (``edges`` the read instants, ``series_raw`` the chord's mean over each
    interval, :func:`read_interval_means`; or ``h`` a uniform interval) against the walk's own reads, each interval
    weighted by its own length, under the gates the pack claims
    (:func:`_battery_gates`: GRE, spin echo, the CPMG train at ``max_refocus_pulses`` and a stimulated echo), over the
    envelope's field strengths and directions, against the split-half floor of the reference."""
    series_raw = np.asarray(series_raw, np.float64)                                    # (n_w, n_t, n_ch), sample 0 the start's
    n_w, n_t = series_raw.shape[0], series_raw.shape[1]
    w = np.ones(n_w) if w is None else np.asarray(w, np.float64)
    b_dec, _ = susc_path_decode(arrays, pm, n_t=n_t, n_w=n_w, edges=edges)
    chi_i, ca = chi if chi is not None else _field_scale(env)
    if not gm.get("has_aniso"):
        ca = 0.0
    n_p = int(pm.get("max_refocus_pulses") or 1)
    if edges is None:
        gates, dt = _battery_gates(n_t, n_p), float(h)
    else:
        E = np.asarray(edges, np.float64)
        gates = _battery_gates(n_t, n_p, t=E / E[-1])
        dt = np.concatenate([[0.0], np.diff(E)])                         # each interval its own length; the start holds none
    perm = np.random.RandomState(0).permutation(n_w); A, B = perm[:n_w // 2], perm[n_w // 2:]
    err = floor = 0.0
    from ..fields.hollow_cylinder import contract
    for B0 in (env.get("B0_list") or [3.0, 7.0]):
        for th in (env.get("theta_deg") or [0, 90]):
            t = np.deg2rad(float(th)); d = [np.sin(t), 0.0, np.cos(t)]
            f_raw = contract(series_raw, d, B0=B0, chi_iso=chi_i, chi_aniso=ca)
            f_dec = susc_path_field(b_dec, d, B0=B0, chi_iso=chi_i, chi_aniso=ca, has_aniso=bool(gm.get("has_aniso")))
            e, f = _gate_battery(f_raw, f_dec, gates, w, A, B, dt)
            err, floor = max(err, e), max(floor, f)
    return dict(err=float(err), floor=float(floor), n_pulses_certified=n_p, n_walkers=int(n_w))


def _susc_path_bloch_fidelity(series_raw, arrays, pm, gm, env, *, h, w=None, chi=None, n_sub=8000):
    """Certify the field channel ON THE VECTOR-BLOCH PATH -- the primary consumer -- on a raw subset: the scalar gate in
    :func:`_susc_path_fidelity` integrates the right quantity but does not exercise the Bloch operator, where the RF
    rotations do NOT commute with the field's z-precession and relaxation interleaves between pulses. This drives
    ``replay_bloch`` (no gradient: the positions do not enter) on the walk's read intervals (``h``, ``series_raw`` as
    :func:`_susc_path_fidelity` takes it) with the STORED channel decoded there against the walk's own reads, at spin
    echo and at the CPMG train the pack advertises, against the split-half floor OF THAT SUBSAMPLE."""
    from ..constants import GAMMA
    from .trajectories import replay_bloch
    series_raw = np.asarray(series_raw, np.float64)
    k = int(min(n_sub, series_raw.shape[0]))
    series_raw = series_raw[:k]
    n_t = series_raw.shape[1]
    pos = np.broadcast_to(np.zeros((1, 1, 3)), (k, n_t, 3))  # no gradient: the positions are never read
    dt = float(h); TE = (n_t - 1) * dt
    w = np.ones(k) if w is None else np.asarray(w, np.float64)[:k]
    b_dec, _ = susc_path_decode(arrays, pm, n_t=n_t, n_w=k)
    chi_i, ca = chi if chi is not None else _field_scale(env)
    if not gm.get("has_aniso"):
        ca = 0.0
    n_p = int(pm.get("max_refocus_pulses") or 1)

    def rf_train(npul):
        ev = [RFEvent(0.0, 90.0, 'Mz→Mxy')]
        for j in range(npul):
            ev.append(RFEvent((j + 0.5) * TE / npul, 180.0, 'refocus', axis_deg=90.0))
        return ev

    G0 = np.zeros((1, n_t, 3))                       # b=0 isolates the susceptibility physics

    def sig(field, rf, idx=None):
        f = field if idx is None else field[idx]
        p = pos if idx is None else pos[idx]
        ww = w if idx is None else w[idx]
        out = replay_bloch(p, dt, G0, dt, rf, T2=None, T1=None, extra_phase_per_step=GAMMA * dt * f, weights=ww)
        a = np.asarray(out[0] if isinstance(out, tuple) else out)
        return float(np.abs(np.atleast_1d(a.ravel())[-1]))

    perm = np.random.RandomState(0).permutation(k); A, B = perm[:k // 2], perm[k // 2:]
    err = floor = 0.0
    from ..fields.hollow_cylinder import contract
    B0s = (env.get("B0_list") or [7.0])[:1]          # one field strength: the tier scales linearly in B0
    for B0 in B0s:
        for th in (env.get("theta_deg") or [90]):
            t = np.deg2rad(float(th)); d = [np.sin(t), 0.0, np.cos(t)]
            f_raw = contract(series_raw, d, B0=B0, chi_iso=chi_i, chi_aniso=ca)
            f_dec = susc_path_field(b_dec, d, B0=B0, chi_iso=chi_i, chi_aniso=ca, has_aniso=bool(gm.get("has_aniso")))
            for npul in (1, n_p):
                rf = rf_train(npul)
                sr = sig(f_raw, rf)
                err = max(err, abs(sig(f_dec, rf) - sr))
                floor = max(floor, abs(sig(f_raw, rf, A) - sig(f_raw, rf, B)))
    return dict(err=float(err), floor=float(floor), n_pulses=n_p, n_walkers=k)


def _surface_terms_walk_free(endpoint, bands, arrays, meta, env, *, T, h, D, w=None, quadrature=0.0, gates=None):
    """The contact tier's WALK-FREE certificate (RPK.md 9.4 rule 4, 9.5.5): the surface-weighted signal ``<exp((rho2 /
    D) int chi dB)>`` at every ``rho2`` of the envelope's ``rho2_list`` under the transverse on-time gates ``gates``
    (``(n_s,)`` arrays on the block grid; default :func:`_surface_gates`, the battery's), read from the STORED bridge
    (:func:`~dmipy_sim.replay.compression.surface_logweight_bridge` on the pack's own arrays) against the same reading
    of the walk's float ``bands``, every band the walk accumulated: the container's quantisation of the surface rows
    (the stored bands against the walk's float bands of the stored count) and the truncation of the bands the pack
    does not store (those against all of ``bands``; zero when it stores them all), plus the local time's quadrature
    term ``quadrature`` (``err_quadrature_surface`` of :func:`_derived_certificate`); the floor the coded split-half
    spread of the stored reading. ``(err, floor, quantisation, truncation, quadrature, rho2_list)``, the largest over
    the rows; no raw path is read."""
    n = int(np.asarray(endpoint).shape[0])
    w = np.ones(n) if w is None else np.asarray(w, np.float64)
    rhos = [float(r) for r in (env.get("rho2_list") or [1e-4])]
    n_s = int(round(float(T) / float(h))) + 1
    bands = np.asarray(bands, np.float64)
    K_s = _cx.c2_bands_K(arrays, meta)
    if K_s > bands.shape[1]:
        raise ValueError(f"the pack stores {K_s} contact bands; the walk's float bands given are {bands.shape[1]}")
    float_of = lambda b: _cx.encode_boundary_bands(np.zeros(n, np.float32), np.asarray(endpoint, np.float32), b,
                                                   T=float(T), dtype=np.float64, container=None)
    exact, meta_x = float_of(bands[:, :K_s])
    full = float_of(bands) if K_s < bands.shape[1] else None
    perm = np.random.RandomState(0).permutation(n); A, B = perm[:n // 2], perm[n // 2:]
    wmean = lambda x, idx: float(np.sum(w[idx] * x[idx]) / np.sum(w[idx]))
    quant = trunc = floor = 0.0
    for rho in rhos:
        r = rho / float(D)
        for gate in [None] + (_surface_gates(n_s) if gates is None else list(gates)):
            s_dec = np.exp(np.asarray(_cx.surface_logweight_bridge(arrays, meta, r, gate, h), np.float64))
            s_x = np.exp(np.asarray(_cx.surface_logweight_bridge(exact, meta_x, r, gate, h), np.float64))
            quant = max(quant, abs(wmean(s_dec, slice(None)) - wmean(s_x, slice(None))))
            if full is not None:
                s_f = np.exp(np.asarray(_cx.surface_logweight_bridge(full[0], full[1], r, gate, h), np.float64))
                trunc = max(trunc, abs(wmean(s_x, slice(None)) - wmean(s_f, slice(None))))
            floor = max(floor, abs(wmean(s_dec, A) - wmean(s_dec, B)))
    return dict(err=float(quant + trunc + float(quadrature)), floor=float(floor), quantisation=float(quant),
                truncation=float(trunc), quadrature=float(quadrature), n_walkers=n, rho2_list=rhos)


def _contact_floor(endpoint, bands, env, *, T, h, D, w=None):
    """The floor the contact channel's container is derived against: the coded split-half floor of the surface reading
    of the walk's float bands (:func:`_surface_terms_walk_free`), which the container barely moves."""
    n = int(np.asarray(endpoint).shape[0])
    a, m = _cx.encode_boundary_bands(np.zeros(n, np.float32), np.asarray(endpoint, np.float32), np.asarray(bands, np.float64),
                                     T=float(T), dtype=np.float64, container=None)
    return float(_surface_terms_walk_free(endpoint, bands, a, m, env, T=T, h=h, D=D, w=w)["floor"])


def _surface_fidelity(raw_boundary, arrays, meta, env, *, h, D, w=None):
    """The contact tier's measured certificate (RPK.md 9.4 rule 4) on the raw subset: the surface-weighted signal
    ``<exp((rho2 / D) int chi dB)>`` at every ``rho2`` of the envelope's ``rho2_list`` under the battery's transverse
    on-time gates (:func:`_surface_gates`: the whole window, and a stimulated echo's two transverse quarters; the
    ungated total is the bridge's exact endpoint), from the stored bridge
    (:func:`~dmipy_sim.replay.compression.surface_logweight_bridge`) against the raw block contacts, and the
    split-half floor of the raw reading: ``(err, floor)`` the largest over the rows."""
    raw = np.asarray(raw_boundary, np.float64)                                   # (n_raw, n_blocks + 1): block j's contact at sample j
    n_raw, n_s = raw.shape
    w = np.ones(n_raw) if w is None else np.asarray(w, np.float64)
    rhos = [float(r) for r in (env.get("rho2_list") or [1e-4])]
    perm = np.random.RandomState(0).permutation(n_raw); A, B = perm[:n_raw // 2], perm[n_raw // 2:]
    wmean = lambda x, idx: float(np.sum(w[idx] * x[idx]) / np.sum(w[idx]))
    err = floor = 0.0
    for rho in rhos:
        r = rho / float(D)
        for gate in [None] + _surface_gates(n_s):
            lw_dec = np.asarray(_cx.surface_logweight_bridge(arrays, meta, r, gate, h), np.float64)[:n_raw]
            lw_raw = r * (raw[:, 1:].sum(1) if gate is None else raw[:, 1:] @ np.asarray(gate, np.float64)[:-1])
            s_dec, s_raw = np.exp(lw_dec), np.exp(lw_raw)
            err = max(err, abs(wmean(s_dec, slice(None)) - wmean(s_raw, slice(None))))
            floor = max(floor, abs(wmean(s_raw, A) - wmean(s_raw, B)))
    return dict(err=float(err), floor=float(floor), n_walkers=int(n_raw), rho2_list=rhos)


def _field_terms(T, h, m, K_c3, qv_field, env, chi, read):
    """The field channel's bounded terms (RPK.md 9.5.5): ``quadrature.field``, half of ``(gamma B0 chi)^2 sum_k g_k^2
    rel(m, omega_k h) QV_f / T`` over the battery's gates (:func:`_battery_gates`, their cosine projections on the
    window) at the envelope's largest ``B0`` and the certificate's chi, and ``field_read``, half of ``v_read = (gamma
    B0 chi)^2 dt_f^2 (sum_c sqrt(kappa_c QV_c))^2`` (:func:`~dmipy_sim.engine.accumulate.field_read_term`) from ``read`` =
    ``{dt_f, qv_increments, constants, B0, chi}``: the walk's read interval, its read series' quadratic variation per
    channel (the walkers' mean), the trapezoid's constants and the envelope's ``B0`` and the rule's ``chi``
    (:func:`~dmipy_sim.engine.accumulate.field_rule_scale`)."""
    from ..constants import GAMMA
    n_t = int(round(float(T) / float(h))) + 1
    gates = np.stack(_battery_gates(n_t, max(1, int(K_c3) // 2)))                       # (n_g, n_t) on the block grid
    g = _cx.cosine_projection(gates[:, :, None], float(h), float(T), int(K_c3))[:, :, 0]     # (n_g, K_c3)
    k = np.arange(int(K_c3))
    rel = _cx.quadrature_rel(m, h, np.pi * k / float(T))
    B0 = float(max(env.get("B0_list") or [FIELD_RULE_B0]))
    qv = float(np.sum(qv_field))                                                           # over the components: a bound
    v = (GAMMA * B0 * float(chi)) ** 2 * (g ** 2 * rel[None, :]).sum(axis=1) * qv / float(T)
    term = field_read_term(read["dt_f"], read["qv_increments"], read["B0"], read["chi"], read.get("constants"))
    return dict(quadrature=float(0.5 * v.max()), field_read=float(term), B0=B0, chi=float(chi),
                read=dict(B0=float(read["B0"]), chi=float(read["chi"]), interval_s=float(read["dt_f"]),
                          B0_chi_max_T=_finite_or_none(field_read_B0_chi_max(read["dt_f"], read["qv_increments"], read.get("floor", 1e-3),
                                                                             read.get("constants")))))


# ------------------------------------------------------------------------------------------ the pack from the walk
def build_replay_pack(walk, *, id, license, citation, weights=None, envelope=None, K=None, temporal_bandwidth_hz=None,
                      scanner=None, sigma_star=None, provenance=None, blt_temporal_K=None, blt_dtype=np.float32,
                      susc_path_K=None, field_container="derived", voxel_grid=None, position_container="derived", blt_container="derived",
                      diffusivity=None, substrate_frame=None, out_path=None, verbose=False, fidelity=None, device="auto",
                      window_T=None, pack_license=None, pack_license_url=None):
    """Assemble a self-certifying replay pack from a :class:`~dmipy_sim.coefficient_walk.CoefficientWalk`: the bands
    stored as the walk accumulated them, at its ``K`` or a stated smaller band, window by window (RPK.md 4.3), in
    the builder's containers; the contact (C2) in the same form; the occupancy (C1) as the static label or the
    events of a walk whose walkers change pool (RPK.md 9.5.7), the MT bound column likewise; the field (C3) as the
    modes the walk read along the path (``susc_path_K`` a smaller band of the walk's ``K_c3``). There is no other input: a pack is never built from samples.

    ``license`` is the SOURCE substrate's licence, which a pack never relicenses (RPK.md 10); ``pack_license`` is
    the terms the pack file itself is distributed under, as an SPDX expression, with ``pack_license_url`` where
    its full text lives; absent, the pack is distributed under ``license``.

    **The band is the scanner class's** (#643): ``K = ceil(2 f T)`` per window with ``f`` = :func:`storage_band_hz`
    of ``scanner`` (the class the walk was walked for, ``CoefficientWalk.scanner``, which :func:`~dmipy_sim.spec.walk.walk_spec`
    sets) times :data:`STORAGE_BAND_MARGIN`, never more than the walk's own ``K``; ``K`` or ``temporal_bandwidth_hz``
    state the band directly, and a walk with no class and no stated band is packed at its own ``K`` (what it was
    walked at). Nothing is searched on the walk: the certificate measures whether the band holds, it does not
    choose it.

    ``position_container``, ``blt_container`` and ``field_container`` are the band containers of the position,
    contact and field channels: ``"derived"`` (the default) is :data:`compression.BAND_CONTAINER` for the first two and
    :class:`DerivedFieldContainer` for the field, each derived per pack against the pack's floor; ``None`` the float
    container; an explicit ``((upto, bits), ...)`` or :func:`compression.band_curve` list is taken as given.

    ``fidelity`` is what the pack certifies (RPK.md 9.4 rule 4): ``"derived"`` (the default), the walk-free
    certificate (:func:`_derived_certificate`: the truncation from the free band variances, the block quadrature
    ``rel(m, h) x QV`` per channel, the field read's term, the container's quantisation, the float32 rounding, the
    coded split-half floors; the field channel's pilot error beside them), or ``"measured"``, which also replays the
    battery on the walk's raw subset against its own coefficients -- the positions, and the field channel on the
    scalar gates and on the vector-Bloch path (:func:`_susc_path_bloch_fidelity`) -- and is refused by name on a walk
    that kept none. ``voxel_grid`` (a :class:`~dmipy_sim.phantom.Grid`, substrate-attached) adds the per-voxel
    certificate a partition needs (:func:`voxel_floor_coded`). ``device`` runs the coded floors on the JAX device
    (``"auto"``: when it is a GPU). Returns a :class:`~dmipy_sim.replay.replay.ReplayPack`; writes it to
    ``out_path`` if given.
    """
    from ..coefficient_walk import CoefficientWalk
    _work = getattr(walk, "work", None)
    if _work and _work.get("exhausted_steps"):
        raise ValueError(
            f"{_work['exhausted_steps']} walker-steps of this walk exhausted its bounce budget of {_work['bounce_budget']}: "
            f"part of their paths went untested, so the walk is not the record of the physics it claims. Walk it again "
            f"with a larger bounce_budget= (certify_bounce_budget sizes one from a pilot) and build the pack from that.")
    if not isinstance(walk, CoefficientWalk):
        raise TypeError(f"build_replay_pack takes the CoefficientWalk a producer returns (simulate_trajectories, walk_spec, "
                        f"simulate_trajectories_adaptive, simulate_mt_trajectories); got {type(walk).__name__}. A pack is "
                        "the bands a walk accumulated; it is never built from samples on a grid")
    cw = walk
    if fidelity is None:
        fidelity = "derived"
    if fidelity not in ("derived", "measured"):
        raise ValueError(f"a coefficient walk certifies itself walk-free (fidelity='derived') or on the raw subset it kept "
                         f"(fidelity='measured'); got fidelity={fidelity!r}")
    if fidelity == "measured" and cw.raw_positions is None:
        raise ValueError("fidelity='measured' replays the battery on raw positions, and this coefficient walk kept none: "
                         "walk it with simulate_trajectories(raw_subset=n), or build with fidelity='derived'")
    if cw.spec is None:
        raise ValueError("a pack embeds the substrate spec its walk was driven by, and this walk carries none")
    T_win = float(cw.T_window)
    if window_T is not None and abs(T_win - float(window_T)) > 1e-9 * T_win and T_win > float(window_T):
        raise ValueError(f"this walk was accumulated in windows of {T_win * 1e3:.4g} ms and a pack stores windows of "
                         f"window_T = {float(window_T) * 1e3:.4g} ms (RPK.md 4.3): walk it with that window_T, or pass "
                         f"window_T >= {T_win:.6g}")
    weights = _coefficient_weights(cw, weights)
    kw = dict(license=license, citation=citation, weights=weights, envelope=envelope, scanner=scanner,
              sigma_star=sigma_star, provenance=provenance, blt_dtype=blt_dtype, position_container=position_container,
              blt_container=blt_container, diffusivity=diffusivity, substrate_frame=substrate_frame, verbose=verbose,
              fidelity=fidelity, device=device, pack_license=pack_license, pack_license_url=pack_license_url)
    n = int(cw.n_windows)
    with Run("build_replay_pack", params=dict(id=id, K=K, fidelity=fidelity, device=device, out_path=out_path,
                                              n_windows=n)) as run:
        if n == 1:
            pack = _build_coefficient_window(cw, id=id, K=K, temporal_bandwidth_hz=temporal_bandwidth_hz,
                                             blt_temporal_K=blt_temporal_K, susc_path_K=susc_path_K,
                                             field_container=field_container, voxel_grid=voxel_grid, run=run, **kw)
        else:
            packs = []
            for i in range(n):
                run.phase(f"window {i + 1} of {n}")
                pk = _build_coefficient_window(cw.window(i), id=id, K=K, temporal_bandwidth_hz=temporal_bandwidth_hz,
                                               blt_temporal_K=blt_temporal_K, susc_path_K=susc_path_K,
                                               field_container=field_container, voxel_grid=(voxel_grid if i == 0 else None),
                                               run=run, **kw)
                if i == 0:                                    # window 0's band, containers and channel bands are every window's
                    cm0 = pk.meta["compression"]
                    K, temporal_bandwidth_hz = int(pk.K), None
                    if cm0.get("container"):
                        kw["position_container"] = cm0["container"]
                    ch0 = cm0.get("channels") or {}
                    if ch0.get("boundary_local_time") is not None:
                        blt_temporal_K = int(ch0["boundary_local_time"]["K"])
                        if ch0["boundary_local_time"].get("container"):
                            kw["blt_container"] = ch0["boundary_local_time"]["container"]
                    if ch0.get("susc_field_basis") is not None:
                        susc_path_K = int(ch0["susc_field_basis"]["K"])
                        field_container = FieldCodec.of(ch0["susc_field_basis"])
                packs.append(pk)
            fid = _round_certificate(combine_window_fidelity([pk.meta["fidelity"] for pk in packs]))
            if sigma_star is not None:
                fid.update(target_floor=float(sigma_star),
                           meets_target=bool(fid["err_max"] <= sigma_star and fid["floor_max"] <= sigma_star))
            seed = seed_value(cw.seed)
            arrays, meta = join_windows(packs, fidelity=fid, walks=[dict(first=0, last=n - 1, seed=seed)],
                                        walkers_shuffled=True)
            pack = ReplayPack(arrays, meta, source=out_path)
        if out_path is not None:
            run.phase("write")
            write_rpk(out_path, {k: v for k, v in pack.arrays.items() if v is not None}, pack.meta)
            run.artifact(out_path)
            _write_run_sidecar(out_path, run, cw)
        if verbose:
            log.info(f"[pack] {id} K={pack.K} windows={n} err={pack.meta['fidelity']['err_max']:.4g} "
                     f"floor={pack.meta['fidelity']['floor_max']:.4g}")
        return pack


def _build_coefficient_window(cw, *, id, license, citation, weights, envelope, K, temporal_bandwidth_hz, scanner,
                              sigma_star, provenance, blt_temporal_K, blt_dtype, susc_path_K, field_container, voxel_grid,
                              position_container, blt_container, diffusivity, substrate_frame, verbose, fidelity, device,
                              pack_license, pack_license_url, run):
    """One window of :func:`build_replay_pack` as a pack of its own (``cw`` a one-window walk; its block record names
    the window's start on the walk's clock when it is a later window)."""
    T, h = float(cw.T), float(cw.h)
    t0 = float((cw.block.get("window") or {}).get("t0", cw.block.get("t_start", 0.0)))
    n_t = int(cw.n_blocks) + 1                                 # the battery's grid: the window's block ends
    spec = cw.spec
    sub = spec.to_dict()
    has_field = cw.field_coeffs is not None
    susceptibility_field_of(sub, name=id, stores_field=has_field)
    if substrate_frame is None:
        substrate_frame = frame_of_spec(spec)
    if substrate_frame is not None:
        if (sub.get("frame") or {}).get("source") == "structural":
            check_frame_against_geometry(sub, substrate_frame)
        else:
            bundles = (sub.get("realisation") or {}).get("bundles")
            Cw = np.asarray(cw.position_coeffs)[:, 0]
            ends = np.stack([Cw[:, 0, :], Cw[:, 0, :] + Cw[:, 1, :]], axis=1)       # the walk's two exact endpoints
            check_frame_against_walk(ends, substrate_frame, w=weights,
                                     bundle_axes=(None if not bundles else [b["axis"] for b in bundles]))
    D = float(diffusivity if diffusivity is not None else (cw.diffusivity or 0.0))
    if not D > 0.0:
        raise ValueError("the walk-free certificate bounds the truncation at the walk's diffusivity, which this walk does not record")
    band_class = None
    if K is None and temporal_bandwidth_hz is None and (scanner or cw.scanner) is not None:
        band_class = scanner if scanner is not None else cw.scanner       # the class the walk was walked for (#643)
        temporal_bandwidth_hz = storage_band_hz(band_class)
    if K is None:                                             # else the walk's own band: what it was walked at
        K = cw.K if temporal_bandwidth_hz is None else min(mode_count(T, temporal_bandwidth_hz=temporal_bandwidth_hz), cw.K)
    K = int(K)
    if K > cw.K:
        raise ValueError(f"the walk accumulated {cw.K} bands per window; a pack of {K} needs a walk with K >= {K}")
    env = envelope or _cx.default_envelope()
    C = np.asarray(cw.position_coeffs, np.float64)[:, 0, :K + 2, :]
    run.phase("floor")
    floor = _cx.measure_floor_coded(C, h, n_t, env, w=weights, device=device)
    floor_t = float(sigma_star) if sigma_star is not None else float(floor["floor_max"])
    pc = _container(position_container)
    if isinstance(pc, _cx.DerivedContainer):
        pc = pc.bound(floor=floor_t, dt=h, env=env)
    pos_meta = dict(method=_cx.POSITION_METHOD, K=K, T=T)
    if pc is None:
        pos_arrays = _cx.pack_position_arrays(C, np.float32)
    else:
        pc = _cx.resolve_container(pc, C[:, 2:, :], n_t)
        pos_arrays = _cx.pack_position_arrays(C, np.float32, container=pc)
        pos_meta["container"] = [{"bands": [k0, k1], "bits": b} for k0, k1, b in _cx._band_ranges(pc, K)]
    Cq = _cx.read_position_coeffs(pos_arrays, dtype=np.float64)
    run.phase("certificate")
    scales = None if pc is None else _cx.band_scales(pos_arrays, "pos_band_scale", Cq.shape[0])
    qv_b = None if cw.qv_boundary is None else float(np.mean(cw.qv_boundary))
    chi_scale = _field_scale(env, spec)
    field_terms = None
    if has_field:
        fr = cw.block.get("field_read") or {}
        K_f = int(cw.K_c3 if susc_path_K is None else susc_path_K)
        if K_f < 1 or K_f > int(cw.K_c3):
            raise ValueError(f"the walk accumulated {cw.K_c3} field modes per window; susc_path_K={K_f} asks for more (or none)")
        if cw.qv_field_increments is None:
            raise ValueError("this walk carries no read series' quadratic variation (CoefficientWalk.qv_field_increments), which "
                             "the field read's term is measured on (RPK.md 9.5.5); walk it again")
        B0_r, chi_r = field_rule_scale(spec, env)
        band = dict(cw.block.get("field_band") or {})
        fc = _field_container(field_container)
        if isinstance(fc, DerivedFieldContainer):                  # derived against the pack's floor, like the positions'
            fc = fc.bound(floor=floor_t, B0=B0_r, chi_iso=chi_scale[0], chi_aniso=chi_scale[1])
        run.phase("field channel")
        names = list(cw.field_basis.channel_names)
        _a, _pm = susc_path_encode_modes(np.asarray(cw.field_coeffs, np.float64)[:, 0, :K_f, :], names, T=T, container=fc,
                                         trace_residual=getattr(cw.field_basis, "trace_residual", None),
                                         max_refocus_pulses=env.get("max_refocus_pulses"), read_interval=cw.field_read_interval,
                                         device=device)
        _pm["band"] = dict(band) if band else dict(rule="given", K=K_f)
        _pm["read"] = {k: v for k, v in (cw.block.get("field_read") or {}).items()}
        field_terms = _field_terms(T, h, cw.m, K_f, np.asarray(cw.qv_field, np.float64)[:, 0].mean(axis=0), env, chi_scale[0],
                                   dict(dt_f=float(cw.field_read_interval), B0=B0_r, chi=chi_r, floor=float(fr.get("floor", 1e-3)),
                                        qv_increments=np.asarray(cw.qv_field_increments, np.float64)[:, 0].mean(axis=0),
                                        constants=fr.get("constants")))
        field_terms["quantisation_eps"] = field_quantisation(_a, _pm, env, T=T, h=h, B0=B0_r, chi_iso=chi_scale[0], chi_aniso=chi_scale[1])
    fid = _derived_certificate(scales, T=T, h=h, K=K, m=cw.m, D=D, env=env, floor=floor,
                               qv_pos=(None if cw.qv_positions is None
                                       else np.asarray(cw.qv_positions, np.float64)[:, 0].mean(axis=0)),
                               qv_contact=qv_b, rho_over_D=(None if qv_b is None else max(env.get("rho2_list") or [1e-4]) / D),
                               rounding_rel=(float(cw.block.get("k0_residual_relative") or 0.0)
                                             if cw.block.get("precision", "float32") == "float32" else None),
                               field=field_terms, h_round=cw.block.get("round_s"))
    if fidelity == "measured":
        raw = np.asarray(cw.raw_positions, np.float64)
        n_raw = raw.shape[0]
        G, gmeta = _cx.acquisition_battery(n_t, h, env)
        phi_raw = _cx._phases_by_chunk(raw, h, G)
        phi_dec = _cx.coded_phases(Cq[:n_raw], h, G, n_t, device=device)
        wr = None if weights is None else np.asarray(weights, np.float64)[:n_raw]
        err = np.abs(_cx._mean_signal(phi_dec, w=wr) - _cx._mean_signal(phi_raw, w=wr))
        per = {f: dict(err_max=float(err[[i for i, x in enumerate(gmeta) if x["fam"] == f]].max()),
                       floor_max=fid["per_family"][f]["floor_max"]) for f in fid["per_family"]}
        fid.update(certified="measured", measured_by="raw_subset", measured_on=int(n_raw), err_derived=fid["err_max"], err_max=float(err.max()),
                   per_family=per, within_2x_floor=bool(err.max() <= 2.0 * fid["floor_max"]))
    arrays = dict(pos_arrays)
    chan_meta = {}
    channels = {"gradient": True, "susceptibility": False, "T1T2": False, "rho2": False, "mt": cw.bound_start is not None}
    if has_field:
        from ..fields.susceptibility_field import FieldGrid, FieldGridRecord
        from ..fields.strand_field import StrandFieldBasis, StrandFieldRecord
        from ..fields.hollow_cylinder import HollowCylinderField
        field = cw.field_basis
        arrays.update(_a); chan_meta["susc_field_basis"] = _pm
        if isinstance(field, (FieldGrid, FieldGridRecord)):
            fm = field.meta
            chan_meta["susceptibility_grid"] = dict(
                origin=fm["origin"], voxel_size=fm["voxel_size"], shape=fm["shape"], has_aniso=bool(fm["has_aniso"]),
                periodic=fm["periodic"], arrays_in_pack=False, replay_route="path", raster=fm.get("raster"))
        elif isinstance(field, HollowCylinderField):
            chan_meta["susceptibility_grid"] = dict(has_aniso=True, arrays_in_pack=False, replay_route="path", source=field.meta)
        elif isinstance(field, (StrandFieldBasis, StrandFieldRecord)):
            chan_meta["susceptibility_grid"] = dict(has_aniso=("aniso_G_xx" in names), arrays_in_pack=False,
                                                    replay_route="path", source=field.meta)
        else:
            raise TypeError(f"a walk's field_basis is a FieldGrid, a HollowCylinderField or a StrandFieldBasis (or their records), got {type(field).__name__}")
        channels["susceptibility"] = True
        # the pilot's measurement of the band at its rung: what the band alone drops from the pilot's reads, with the
        # field container's quantisation read back from the stored steps
        if band.get("err_truncation") is not None and int(band.get("K", K_f)) == K_f:
            fid.update(err_susc_path=float(band["err_truncation"]) + float(fid["err_quantisation_field"]), floor_susc_path=float(band["floor"]),
                       susc_path_pulses_certified=int(_pm["max_refocus_pulses"]), susc_path_measured_on="pilot",
                       susc_path_walkers=int(band.get("pilot_walkers", 0)))
        if fidelity == "measured" and cw.raw_field is not None:
            run.phase("certificate path")
            series, edges = read_interval_means(cw.raw_field), cw.field_read_instants   # the walk's own reads, interval by interval
            wr = None if weights is None else np.asarray(weights, np.float64)[:series.shape[0]]
            _pf = _susc_path_fidelity(series, arrays, _pm, chan_meta["susceptibility_grid"], env, edges=edges, w=wr, chi=chi_scale)
            # the Bloch route on the block grid: the chord's exact block means (a block ends on a read)
            _bf = _susc_path_bloch_fidelity(read_block_means(cw.raw_field, edges, cw.reads_per_block, h), arrays, _pm,
                                            chan_meta["susceptibility_grid"], env, h=h, w=wr, chi=chi_scale)
            fid.update(err_susc_path=_pf["err"], floor_susc_path=_pf["floor"], susc_path_pulses_certified=_pf["n_pulses_certified"],
                       susc_path_measured_on="raw_subset", susc_path_walkers=int(_pf["n_walkers"]),
                       err_susc_bloch=_bf["err"], floor_susc_bloch=_bf["floor"], susc_bloch_walkers=_bf["n_walkers"])
        for ek, fk in (("err_susc_path", "floor_susc_path"), ("err_susc_bloch", "floor_susc_bloch")):
            if fid.get(ek) is not None:
                fid["err_max"] = max(float(fid["err_max"]), float(fid[ek]))
                fid["floor_max"] = max(float(fid["floor_max"]), float(fid[fk]))
        fid["within_2x_floor"] = bool(fid["err_max"] <= 2.0 * fid["floor_max"])
    comp = cw.compartment
    if comp is not None:
        cols = {"comp": (np.asarray(comp), cw.comp_events)}
        if cw.bound_start is not None:
            cols["bound"] = (np.asarray(cw.bound_start), cw.bound_events)
        a_, cm_ = _cx.encode_events_occupancy(cols, t0=t0)
        arrays.update(a_); chan_meta["compartment"] = cm_
        if weights is not None:
            arrays["spin_weights"] = np.asarray(weights, np.float32)
        channels["T1T2"] = True
    elif cw.bound_start is not None:
        raise ValueError("an MT (C4) pack carries its bound pool as a C1 occupancy column, so it needs the compartment channel too.")
    if cw.boundary_bands is not None:
        Kb = K if blt_temporal_K is None else int(blt_temporal_K)
        if Kb > cw.K:
            raise ValueError(f"the walk accumulated {cw.K} contact bands; blt_temporal_K={Kb} asks for more")
        bc = _container(blt_container)
        if isinstance(bc, _cx.DerivedContainer):
            # held to the contact's own floor (the surface reading's split-half spread) where it is the tighter
            bc = bc.bound(floor=min(floor_t, _contact_floor(np.asarray(cw.boundary_endpoint)[:, 0], np.asarray(cw.boundary_bands)[:, 0, :Kb],
                                                            env, T=T, h=h, D=D, w=weights)),
                          dt=h, env=env, weight=max(env.get("rho2_list") or [1e-4]) / D)
        a_, cm_ = _cx.encode_boundary_bands(np.zeros(cw.n_walkers, np.float32), np.asarray(cw.boundary_endpoint)[:, 0],
                                            np.asarray(cw.boundary_bands)[:, 0, :Kb], T=T, dtype=blt_dtype, container=bc)
        arrays.update(a_); chan_meta["boundary_local_time"] = cm_; channels["rho2"] = True
        # the contact tier's own certificate (``err_surface`` / ``floor_surface``, what a family's contact tier reads):
        # walk-free from the stored bridge against every band the walk accumulated -- the container's quantisation of
        # the surface rows and the truncation of the bands beyond the stored Kb beside the local time's quadrature
        # term, and the coded split-half floor of the surface rows; a measured certificate replaces it with the raw
        # subset's reading
        run.phase("certificate contact")
        _sf = _surface_terms_walk_free(np.asarray(cw.boundary_endpoint)[:, 0], np.asarray(cw.boundary_bands)[:, 0], arrays, cm_,
                                       env, T=T, h=h, D=D, w=weights, quadrature=float(fid.get("err_quadrature_surface", 0.0)))
        fid.update(err_surface=_sf["err"], floor_surface=_sf["floor"], surface_certified="walk_free", surface_rho2_list=_sf["rho2_list"],
                   err_surface_quantisation=_sf["quantisation"], err_surface_truncation=_sf["truncation"],
                   err_surface_quadrature=_sf["quadrature"])
        if fidelity == "measured" and cw.raw_boundary is not None:
            wr = None if weights is None else np.asarray(weights, np.float64)[:cw.raw_boundary.shape[0]]
            _sf = _surface_fidelity(cw.raw_boundary, arrays, cm_, env, h=h, D=D, w=wr)
            fid.update(err_surface=_sf["err"], floor_surface=_sf["floor"], surface_certified="measured",
                       surface_measured_on=int(_sf["n_walkers"]), surface_rho2_list=_sf["rho2_list"])
        fid["err_max"] = max(float(fid["err_max"]), float(_sf["err"])); fid["floor_max"] = max(float(fid["floor_max"]), float(_sf["floor"]))
        fid["within_2x_floor"] = bool(fid["err_max"] <= 2.0 * float(fid["floor_max"]))
    if voxel_grid is not None:
        from ..phantom.grid import Grid
        if not isinstance(voxel_grid, Grid) or voxel_grid.attach != "substrate":
            raise TypeError("voxel_grid must be a dmipy_sim.phantom.Grid attached to the substrate")
        _w = None if weights is None else np.asarray(weights, np.float64)
        _ijk, _pools, _n, _floor, _err = voxel_floor_coded(pos_arrays, h, n_t, voxel_grid, comp, env, w=_w, device=device)
        arrays["voxel_ijk"] = _ijk.astype(np.int32)
        arrays["voxel_certificate"] = np.stack([_n.astype(np.float32), _floor.astype(np.float32), _err.astype(np.float32)], axis=-1)
        fid = dict(fid, per_voxel=per_voxel_summary(voxel_grid.to_meta(), _pools, np.stack([_n, _floor, _err], axis=-1)))
        if sigma_star is not None and np.isfinite(_floor).any():
            fid["per_voxel"]["meets_target"] = bool(np.nanmax(_floor) <= sigma_star)
    if sigma_star is not None:
        fid.update(target_floor=float(sigma_star),
                   meets_target=bool(fid["err_max"] <= sigma_star and fid["floor_max"] <= sigma_star))
    comp_meta = dict(method=_cx.POSITION_METHOD, K=K, walker_preserving=True, container=pos_meta.get("container"),
                     temporal_bandwidth_hz=float(K) / (2.0 * T))
    if band_class is not None:
        comp_meta["scanner"] = str(band_class)
    comp_meta["precision_tiers"] = _precision_tiers(arrays, cw.n_walkers, float(fid.get("floor_max") or 0.0), True)
    if chan_meta:
        comp_meta["channels"] = chan_meta
    fid = _round_certificate(fid)
    seed = seed_value(cw.seed)
    block = {k: v for k, v in dict(cw.block).items() if k not in ("window", "quadratic_variation", "field_read", "field_band", "calibration_scope", "calibration_pools", "events", "t_start")}
    qv = dict(positions=[float(x) for x in np.asarray(cw.qv_positions, np.float64)[:, 0].mean(axis=0)])
    if cw.boundary_bands is not None:
        qv["local_time"] = float(np.mean(cw.qv_boundary))
    wp_field = {}
    if channels["susceptibility"]:
        qv["field"] = [float(x) for x in np.asarray(cw.qv_field, np.float64)[:, 0].mean(axis=0)]
        qv["field_increments"] = [float(x) for x in np.asarray(cw.qv_field_increments, np.float64)[:, 0].mean(axis=0)]
        wp_field["field_read_interval"] = float(cw.field_read_interval)
    fid["terms"] = _terms_of(dict(fid, has_contact=cw.boundary_bands is not None, field=field_terms))
    _cover_terms(_measured_truncation(fid) if fid.get("measured_by") == "raw_subset" else fid)
    env_acq = _envelope_summary(env)
    if channels["susceptibility"]:
        fr = cw.block.get("field_read") or {}
        env_acq["field_modes"] = int(chan_meta["susc_field_basis"]["K"])
        env_acq["max_refocusing_pulses"] = int(chan_meta["susc_field_basis"]["max_refocus_pulses"])
    replay_env = dict(gradient=True, bulk_relaxation=channels["T1T2"], surface_relaxivity=channels["rho2"],
                      field=channels["susceptibility"], magnetization_transfer=channels["mt"], diffusivity_fixed=True,
                      acquisition=env_acq)
    if channels["susceptibility"]:
        fr = cw.block.get("field_read") or {}
        # RPK.md 9.4 rule 5: a read at the step narrows the pack's reach, stated in the envelope, never hidden in fidelity
        rd = field_terms["read"]
        replay_env["field_read"] = dict(interval_s=float(cw.field_read_interval), at_step=bool(fr.get("at_step", False)),
                                        B0_chi_max_T=rd["B0_chi_max_T"],
                                        term_at=dict(B0=rd["B0"], chi=rd["chi"], floor=fr.get("floor")))
    meta = dict(
        rpk_schema_version=RPK_SCHEMA_VERSION, id=id, compression=comp_meta,
        walk_params=dict(n_walkers=cw.n_walkers, T_max=T, diffusivity=D, seed=seed,
                         windows=dict(n=1, T=T, walks=[dict(first=0, last=0, seed=seed)]),
                         block=block, qv=qv, **wp_field,
                         substrate_frame=(None if substrate_frame is None else np.asarray(substrate_frame, float).tolist())),
        replay_envelope=replay_env,
        fidelity=fid, provenance=dict(provenance or {}, code=_code(), walk=dict(kind="coefficient", stepping=cw.stepping)),
        license=license, citation=citation, substrate=sub)
    if pack_license:
        meta["pack_license"] = str(pack_license)
        if pack_license_url:
            meta["pack_license_url"] = str(pack_license_url)
    susceptibility_field_of(sub, name=id, stores_field=bool(channels["susceptibility"]))
    return ReplayPack(arrays, meta, source=None)


def build_to_floor(make_model, *, id, envelope=None, sigma_star=1e-3, pilot_n=8000, safety=1.4, max_n=400000, verbose=True, **bp):
    """Adaptive floor-targeting generation policy (the bank default).

    Size the walker count so the split-half Monte-Carlo floor <= ``sigma_star``, then build the pack at the band
    ``bp`` states (``scanner=``, ``K=`` or ``temporal_bandwidth_hz=``; #643: the band is never searched) and record
    whether its codec error meets ``sigma_star`` too (``fidelity.meets_target``) -- converging to a defined
    precision instead of a wasteful ultra-high N. ``make_model(n_walkers)`` MUST return a fresh
    :class:`~dmipy_sim.coefficient_walk.CoefficientWalk` on the SAME fixed geometry (only the walker count changes);
    the floor is read from its coefficients (:func:`compression.measure_floor_coded`, no path). Records
    ``sigma_star`` + the achieved floor in the pack.
    """
    env = envelope or _cx.default_envelope()

    def floor_of(cw):
        C = np.asarray(cw.position_coeffs, np.float64)[:, 0]
        return float(_cx.measure_floor_coded(C, float(cw.h), int(cw.n_blocks) // int(cw.n_windows) + 1, env, w=cw.weights)["floor_max"])
    model = make_model(pilot_n)
    f0 = floor_of(model)
    n_star = int(min(max_n, max(pilot_n, round(pilot_n * (f0 / sigma_star) ** 2 * safety))))
    if verbose:
        log.info(f"[floor-target] pilot N={pilot_n}: floor={f0:.4g}; sigma*={sigma_star:.4g} -> N*~{n_star}")
    if n_star == pilot_n:                            # the pilot meets the floor: it is the walk
        f = f0
    else:
        model = None                                 # the pilot is released before the sized walk is made, not after
        model = make_model(n_star); f = floor_of(model)
    while f > sigma_star and n_star < max_n:         # undershoot -> re-estimate from the last measurement and walk again,
        n_star = int(min(max_n, max(n_star + 1, round(n_star * (f / sigma_star) ** 2 * safety))))   # until met or capped
        if verbose:
            log.info(f"[floor-target] floor={f:.4g} > sigma*; topping up to N*={n_star}")
        model = None                                 # one walk in memory at a time: a rebinding would hold both
        model = make_model(n_star); f = floor_of(model)
    if verbose:
        log.info(f"[floor-target] N={n_star}: achieved floor={f:.4g} "
              f"({'<=' if f <= sigma_star else '>'} sigma*)")
    return build_replay_pack(model, id=id, envelope=env, sigma_star=sigma_star, verbose=verbose, **bp)


# ------------------------------------------------------------------------------------ a window of a pack, re-projected
def _cos_int(a, w, L):
    """``int_0^L cos(a + w tau) dtau``, elementwise, ``L cos(a)`` where ``w = 0``."""
    a = np.asarray(a, np.float64); w = np.asarray(w, np.float64)
    small = np.abs(w) < 1e-12
    ws = np.where(small, 1.0, w)
    return np.where(small, L * np.cos(a), (np.sin(a + ws * L) - np.sin(a)) / ws)


def bridge_window_matrix(T, K, t0, T_new, K_new):
    """``(K_new + 2, K + 2)``: the linear map taking a bridge's stored vector ``[x0, d, beta_1..K]`` on ``[0, T]`` to the
    stored vector of the SAME function restricted to the window ``[t0, t0 + T_new]`` and re-projected on that
    window's basis -- the window's exact start ``I(t0)`` and displacement ``I(t0 + T_new) - I(t0)``, and the bands
    ``beta'_j = int_0^T' u'(tau) s'_j(tau) dtau`` of its bridge residual, each ``int`` in closed form. The chord
    contributes nothing to ``u'`` (its restriction is again a chord), so the map on the bands is from the parent's
    bands alone; what the series holds beyond ``K_new`` on the window is the re-projection's truncation, which the
    certificate measures (RPK.md 9.5.4: the band-limited prefix is the function cut and re-projected)."""
    T, T_new, t0 = float(T), float(T_new), float(t0)
    K, K_new = int(K), int(K_new)
    k = np.arange(1, K + 1); om = np.pi * k / T
    j = np.arange(1, K_new + 1); omp = np.pi * j / T_new
    s_t0 = np.sqrt(2.0 / T) * np.sin(om * t0)
    s_t1 = np.sqrt(2.0 / T) * np.sin(om * (t0 + T_new))
    M = np.zeros((K_new + 2, K + 2))
    M[0, 0] = 1.0; M[0, 1] = t0 / T; M[0, 2:] = s_t0
    M[1, 1] = T_new / T; M[1, 2:] = s_t1 - s_t0
    if K and K_new:
        # I1[j, k] = int_0^T' sin(om_k (t0 + tau)) sin(om'_j tau) dtau
        #          = 1/2 [ int cos(om_k t0 + (om_k - om'_j) tau) - int cos(om_k t0 + (om_k + om'_j) tau) ]
        a = (om * t0)[None, :]
        I1 = 0.5 * (_cos_int(a, om[None, :] - omp[:, None], T_new) - _cos_int(a, om[None, :] + omp[:, None], T_new))
        sign = (-1.0) ** j
        I2 = (1.0 - sign) / omp                                           # int_0^T' sin(om'_j tau) dtau
        I3 = -T_new * sign / omp                                          # int_0^T' tau sin(om'_j tau) dtau
        delta = s_t1 - s_t0
        A = np.sqrt(2.0 / T) * np.sqrt(2.0 / T_new) * I1 - np.sqrt(2.0 / T_new) * (np.sin(om * t0) * np.sqrt(2.0 / T))[None, :] * I2[:, None] \
            - np.sqrt(2.0 / T_new) * (delta[None, :] / T_new) * I3[:, None]
        M[2:, 2:] = A
    return M


def mode_window_matrix(T, K, t0, T_new, K_new):
    """``(K_new, K)``: the linear map taking a field's cosine modes ``c_k`` on ``[0, T]`` to the modes of the same
    function on the window ``[t0, t0 + T_new]``, ``c'_j = int_0^T' e_k(t0 + tau) e'_j(tau) dtau`` in closed form."""
    T, T_new, t0 = float(T), float(T_new), float(t0)
    k = np.arange(int(K)); om = np.pi * k / T
    j = np.arange(int(K_new)); omp = np.pi * j / T_new
    nk = np.full(int(K), np.sqrt(2.0 / T)); nk[0] = 1.0 / np.sqrt(T)
    nj = np.full(int(K_new), np.sqrt(2.0 / T_new)); nj[0] = 1.0 / np.sqrt(T_new)
    a = (om * t0)[None, :]
    B = 0.5 * (_cos_int(a, om[None, :] - omp[:, None], T_new) + _cos_int(a, om[None, :] + omp[:, None], T_new))
    return nj[:, None] * nk[None, :] * B


def window_of_pack(pack, t0, T_new, *, K=None, blt_K=None, path_K=None, id=None, envelope=None, position_container=None,
                   blt_container=None, blt_dtype=np.float32, field_container=None, device="auto", provenance=None):
    """The window ``[t0, t0 + T_new]`` of a one-window ``pack`` as a pack of its own, every band-stored channel the
    stored function cut and re-projected on the window's basis (:func:`bridge_window_matrix`,
    :func:`mode_window_matrix`; RPK.md 9.5.4), the events sliced to the window, exactly -- what ``prefix`` (``t0 =
    0``) and ``rewindow`` build. ``K`` / ``blt_K`` / ``path_K`` are the window's bands (the parent's bands per second
    by default, at least 2); ``position_container``, ``blt_container`` and ``field_container`` the window's
    containers (None: the parent's own ranges; ``"derived"``: derived for the window's floor). Its certificate: the walk-free terms at the parent's block, plus the RE-PROJECTION's
    truncation measured on the battery -- the parent's exact replay of the window (its bands with the waveform at
    ``t0``) against the child's, per channel -- recorded as ``err_reprojection`` and in ``err_max``."""
    from ..constants import GAMMA
    from .compression import read_position_coeffs, cosine_projection, band_projection, surface_logweight_bridge
    if pack.n_windows != 1:
        raise ValueError("window_of_pack reads a one-window pack (pack.window(i) for a window of a table)")
    T, h = float(pack.T_window), float(pack.dt)
    t0, T_new = float(t0), float(T_new)
    if t0 < -1e-12 or t0 + T_new > T * (1.0 + 1e-9) or T_new <= 0:
        raise ValueError(f"the window [{t0:.6g}, {t0 + T_new:.6g}] s is not within the pack's walk of {T:.6g} s")
    n_new = max(int(round(T_new / h)), 2)
    T_new = float(T_new)
    env = envelope or _cx.default_envelope()
    ch = dict(pack.meta.get("compression", {}).get("channels", {}) or {})
    wp = pack.meta["walk_params"]
    K_new = int(K) if K is not None else max(2, int(np.ceil(pack.K * T_new / T)))
    K_new = min(K_new, int(pack.K))
    # positions
    C = read_position_coeffs(pack.arrays, dtype=np.float64)                               # (n_w, K+2, 3)
    M = bridge_window_matrix(T, pack.K, t0, T_new, K_new)
    Cn = np.einsum("jk,wkc->wjc", M, C)
    n_t_new = n_new + 1; h_new = T_new / n_new
    G, gmeta = _cx.acquisition_battery(n_t_new, h_new, env)
    # the battery's last sample is the readout, held over nothing: dropped on the parent's side too, where the longer
    # window would otherwise integrate it
    # the battery is on the WINDOW's clock; on the parent's clock it starts at t0, which band_projection's t0 (the
    # window's start on the waveform's clock) expresses as -t0
    W_par = band_projection(G[:, :-1], h_new, T, pack.K, t0=-t0)                          # the parent's exact reading of the window
    W_chi = band_projection(G[:, :-1], h_new, T_new, K_new)
    phi_par = GAMMA * np.einsum("wkc,mkc->wm", C, W_par)
    phi_chi = GAMMA * np.einsum("wkc,mkc->wm", Cn, W_chi)
    w = np.asarray(pack.spin_weights, np.float64)
    err_pos = float(np.abs(_cx._mean_signal(phi_chi, w=w) - _cx._mean_signal(phi_par, w=w)).max())
    floor = _cx.measure_floor_coded(Cn, h_new, n_t_new, env, w=w, device=device)
    floor_t = float(floor["floor_max"])
    pc = _container(position_container if position_container is not None else _container_of_meta(pack.meta["compression"].get("container")))
    if isinstance(pc, _cx.DerivedContainer):
        pc = pc.bound(floor=floor_t, dt=h_new, env=env)
    pos_meta = dict(method=_cx.POSITION_METHOD, K=K_new, T=T_new)
    if pc is None:
        pos_arrays = _cx.pack_position_arrays(Cn, np.float32)
    else:
        pc = _cx.resolve_container(pc, Cn[:, 2:, :], n_t_new)
        pos_arrays = _cx.pack_position_arrays(Cn, np.float32, container=pc)
        pos_meta["container"] = [{"bands": [k0, k1], "bits": b} for k0, k1, b in _cx._band_ranges(pc, K_new)]
    scales = None if pc is None else _cx.band_scales(pos_arrays, "pos_band_scale", Cn.shape[0])
    D = float(wp["diffusivity"])
    blk = wp["block"]
    qv = wp.get("qv") or {}
    qv_pos = None if qv.get("positions") is None else np.asarray(qv["positions"], np.float64) * (T_new / T)
    qv_b = None if qv.get("local_time") is None else float(qv["local_time"]) * (T_new / T)
    rho = max(env.get("rho2_list") or [1e-4]) / D
    arrays = dict(pos_arrays)
    chan_meta = {}
    channels = dict(gradient=True, susceptibility=False, T1T2=False, rho2=False, mt=False)
    err_rep = dict(positions=err_pos)
    # the contact bridge
    if "boundary_local_time" in ch and pack.has_surface:
        bm = dict(ch["boundary_local_time"]); bm.setdefault("K", _cx.c2_bands_K(pack.arrays, bm))
        a = np.asarray(pack.arrays["blt_start"], np.float64); e = np.asarray(pack.arrays["blt_endpoint"], np.float64)
        Bb = _cx.bridge_bands(pack.arrays, bm)
        Kb_new = int(blt_K) if blt_K is not None else max(2, int(np.ceil(int(bm["K"]) * T_new / T)))
        Kb_new = min(Kb_new, int(bm["K"]))
        Mb = bridge_window_matrix(T, int(bm["K"]), t0, T_new, Kb_new)
        vec = np.concatenate([a[:, None], (e - a)[:, None], Bb], axis=1)                   # (n_w, K+2)
        vn = vec @ Mb.T
        a_new, e_new = vn[:, 0], vn[:, 0] + vn[:, 1]
        bc = _container(blt_container if blt_container is not None else _container_of_meta(bm.get("container")))
        if isinstance(bc, _cx.DerivedContainer):
            bc = bc.bound(floor=min(floor_t, _contact_floor((e_new - a_new).astype(np.float32), vn[:, 2:], env, T=T_new, h=h_new, D=D, w=w)),
                          dt=h_new, env=env, weight=rho)
        a_, cm_ = _cx.encode_boundary_bands(np.zeros_like(a_new, dtype=np.float32), (e_new - a_new).astype(np.float32),
                                            vn[:, 2:], T=T_new, dtype=blt_dtype, container=bc)
        arrays.update(a_); chan_meta["boundary_local_time"] = cm_; channels["rho2"] = True
        # the re-projection's error on the contact: the gated contact of the battery's gates, parent at t0 vs child
        gates = np.stack(_battery_gates(n_t_new, 1))
        lw_par = np.stack([surface_logweight_bridge(pack.arrays, bm, rho, gt, h_new, t0=-t0) for gt in gates])
        lw_chi = np.stack([surface_logweight_bridge(a_, cm_, rho, gt, h_new) for gt in gates])
        e_rep = max(float(np.abs(np.exp(lw_chi) - np.exp(lw_par)).mean(axis=1).max()), 0.0)
        err_rep["local_time"] = e_rep
    # the field modes
    field_terms = None
    pm = ch.get("susc_field_basis")
    if pm is not None:
        Cf, names = susc_path_coeffs(pack.arrays, pm)                                     # (n_w, n_ch_full, K_f)
        Kf_new = int(path_K) if path_K is not None else max(2, int(np.ceil(int(pm["K"]) * T_new / T)))
        Kf_new = min(Kf_new, int(pm["K"]))
        Mf = mode_window_matrix(T, int(pm["K"]), t0, T_new, Kf_new)
        Cfn = np.einsum("jk,wck->wcj", Mf, Cf)
        chi_i, ca = _field_scale(env, pack.substrate)
        B0_r, chi_r = field_rule_scale(pack.substrate, env)
        fc = _field_container(field_container if field_container is not None else FieldCodec.of(pm))
        if isinstance(fc, DerivedFieldContainer):
            fc = fc.bound(floor=floor_t, B0=B0_r, chi_iso=chi_i, chi_aniso=ca)
        a_, pm_new = susc_path_encode_modes(np.transpose(Cfn, (0, 2, 1)), names, T=T_new, container=fc, trace_residual=pm.get("trace_residual"),
                                            max_refocus_pulses=env.get("max_refocus_pulses") or pm.get("max_refocus_pulses"),
                                            read_interval=pm.get("read_interval"), device=device)
        pm_new["band"] = dict(rule="reprojected", parent_K=int(pm["K"]), K=Kf_new)
        pm_new["read"] = pm.get("read")
        arrays.update(a_); chan_meta["susc_field_basis"] = pm_new
        chan_meta["susceptibility_grid"] = dict(ch.get("susceptibility_grid") or {})
        channels["susceptibility"] = True
        gm = chan_meta["susceptibility_grid"]
        gates = np.stack(_battery_gates(n_t_new, max(1, Kf_new // 2)))
        g_par = cosine_projection(gates[:, :-1, None], h_new, T, int(pm["K"]), t0=-t0)[:, :, 0]
        g_chi = cosine_projection(gates[:, :-1, None], h_new, T_new, Kf_new)[:, :, 0]
        Cq_new, _ = susc_path_coeffs(a_, pm_new)
        e_rep = 0.0
        for B0 in (env.get("B0_list") or [3.0, 7.0]):
            for th in (env.get("theta_deg") or [0, 90]):
                tt = np.deg2rad(float(th)); d = [np.sin(tt), 0.0, np.cos(tt)]
                f_par = susc_path_field(np.einsum("gk,wck->wcg", g_par, Cf), d, B0=B0, chi_iso=chi_i, chi_aniso=(ca if gm.get("has_aniso") else 0.0), has_aniso=bool(gm.get("has_aniso")))
                f_chi = susc_path_field(np.einsum("gk,wck->wcg", g_chi, Cq_new), d, B0=B0, chi_iso=chi_i, chi_aniso=(ca if gm.get("has_aniso") else 0.0), has_aniso=bool(gm.get("has_aniso")))
                S_par = (w[:, None] * np.exp(1j * GAMMA * f_par)).sum(0) / w.sum()
                S_chi = (w[:, None] * np.exp(1j * GAMMA * f_chi)).sum(0) / w.sum()
                e_rep = max(e_rep, float(np.abs(S_chi - S_par).max()))
        err_rep["field"] = e_rep
        qv_f = None if qv.get("field") is None else np.asarray(qv["field"], np.float64) * (T_new / T)
        if qv.get("field_increments") is None:
            raise ValueError("this pack carries no walk_params.qv.field_increments (RPK.md 9.5.5), which the field read's term "
                             "of a window is charged on; re-walk it with a conformant producer")
        read = pm.get("read") or {}
        at = ((pack.meta.get("replay_envelope") or {}).get("field_read") or {}).get("term_at") or {}
        field_terms = _field_terms(T_new, h_new, int(blk["m"]), Kf_new, qv_f if qv_f is not None else np.zeros(1), env, chi_i,
                                   dict(dt_f=float(wp["field_read_interval"]), B0=B0_r, chi=chi_r, floor=float(at.get("floor") or 1e-3),
                                        qv_increments=np.asarray(qv["field_increments"], np.float64) * (T_new / T),
                                        constants=read.get("constants")))
        field_terms["quantisation_eps"] = field_quantisation(a_, pm_new, env, T=T_new, h=h_new, B0=B0_r, chi_iso=chi_i, chi_aniso=ca)
    # the occupancy: events sliced to the window
    if "compartment" in ch:
        from ..coefficient_walk import events_within
        cols = {}
        for col in ch["compartment"]["columns"]:
            start, ev = _cx.occupancy_events(pack.arrays, col)
            if ev is None:
                cols[col["name"]] = (start, None)
            else:
                ev_w, lab0 = events_within(ev, start, t0, t0 + T_new)
                cols[col["name"]] = (lab0, ev_w)
        a_, cm_ = _cx.encode_events_occupancy(cols, t0=t0)
        arrays.update(a_); chan_meta["compartment"] = cm_; channels["T1T2"] = True
        channels["mt"] = any(c["name"] == "bound" for c in cm_["columns"])
        if "spin_weights" in pack.arrays:
            arrays["spin_weights"] = np.asarray(pack.arrays["spin_weights"], np.float32)
    fid = _derived_certificate(scales, T=T_new, h=h_new, K=K_new, m=int(blk["m"]), D=D, env=env, floor=floor,
                               qv_pos=qv_pos, qv_contact=qv_b, rho_over_D=(None if qv_b is None else rho),
                               rounding_rel=(float(blk.get("k0_residual_relative") or 0.0) if blk.get("precision", "float32") == "float32" else None),
                               field=field_terms, h_round=blk.get("round_s"))
    fid["err_reprojection"] = float(max(err_rep.values()))
    fid["reprojection"] = {k: float(v) for k, v in err_rep.items()}
    fid["err_max"] = float(fid["err_max"] + fid["err_reprojection"])
    fid["within_2x_floor"] = bool(fid["err_max"] <= 2.0 * fid["floor_max"])
    fid["terms"] = _terms_of(dict(fid, has_contact=channels["rho2"], field=field_terms))
    fid["terms"]["reprojection"] = fid["err_reprojection"]
    _cover_terms(fid)
    comp_meta = dict(method=_cx.POSITION_METHOD, K=K_new, walker_preserving=True, container=pos_meta.get("container"),
                     temporal_bandwidth_hz=float(K_new) / (2.0 * T_new))
    if pack.meta["compression"].get("scanner"):
        comp_meta["scanner"] = pack.meta["compression"]["scanner"]
    comp_meta["precision_tiers"] = _precision_tiers(arrays, pack.n_walkers, float(fid.get("floor_max") or 0.0),
                                                    bool(pack.meta["compression"].get("precision_tiers", {}).get("walkers_shuffled", True)))
    comp_meta["channels"] = chan_meta
    fid = _round_certificate(fid)
    env_out = dict(pack.meta.get("replay_envelope") or {})
    env_out.update(bulk_relaxation=channels["T1T2"], surface_relaxivity=channels["rho2"], field=channels["susceptibility"],
                   magnetization_transfer=channels["mt"], acquisition=dict(env_out.get("acquisition") or _envelope_summary(env)))
    if pm is not None:
        env_out["acquisition"]["field_modes"] = int(chan_meta["susc_field_basis"]["K"])
        env_out["acquisition"]["max_refocusing_pulses"] = int(chan_meta["susc_field_basis"]["max_refocus_pulses"])
    env_out.pop("tissue", None)                                       # the contact envelope is restated on the new window
    wp_new = dict(wp, T_max=T_new, windows=dict(n=1, T=T_new, walks=[dict(first=0, last=0, seed=wp.get("seed"))]),
                  block=dict(blk, n_blocks=n_new), qv={k: v for k, v in dict(positions=(None if qv_pos is None else [float(x) for x in qv_pos]),
                                                                              local_time=qv_b, field=(None if pm is None or qv.get("field") is None else [float(x) for x in np.asarray(qv["field"]) * (T_new / T)]),
                                                                              field_increments=(None if pm is None or qv.get("field_increments") is None else [float(x) for x in np.asarray(qv["field_increments"]) * (T_new / T)])).items() if v is not None})
    meta = dict(rpk_schema_version=RPK_SCHEMA_VERSION, id=(id or pack.meta.get("id")), compression=comp_meta, walk_params=wp_new,
                replay_envelope=env_out, fidelity=fid,
                provenance=dict(provenance or copy_provenance(pack.meta.get("provenance")), code=_code(),
                                window=dict(parent_id=pack.meta.get("id"), parent_digest=pack.digest, t0=t0, T=T_new, parent_T=T,
                                            parent_K=int(pack.K), K=K_new,
                                            note="the parent's stored functions cut to the window and re-projected on its basis "
                                                 "(bridge_window_matrix, mode_window_matrix); the re-projection's truncation is "
                                                 "measured on the battery against the parent's exact reading of the window "
                                                 "(fidelity.err_reprojection)")),
                license=pack.meta.get("license"), citation=pack.meta.get("citation"), substrate=pack.meta.get("substrate"))
    for key in ("pack_license", "pack_license_url"):
        if pack.meta.get(key):
            meta[key] = pack.meta[key]
    return ReplayPack(arrays, meta, source=None)


def copy_provenance(prov):
    import copy
    return copy.deepcopy(prov or {})


def _container_of_meta(ranges):
    """The builder's container knob from a stored ranges list (``[{"bands": [k0, k1], "bits": b}, ...]``): the float
    container for None."""
    if not ranges:
        return None
    return tuple((None if i == len(ranges) - 1 else int(r["bands"][1]), int(r["bits"])) for i, r in enumerate(ranges))
