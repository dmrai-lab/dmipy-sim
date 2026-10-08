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
import json
import logging

log = logging.getLogger(__name__)

import numpy as np

from ..persistent_walk import PersistentWalk
from ..run import Run, _code

from . import compression as _cx
from ..acquisition.rf import RFEvent
from .replay import ReplayPack, read_rpk, write_rpk
from ..spec.substrate import susceptibility_field_of

__all__ = ["build_replay_pack", "build_to_floor", "frame_from_axis", "frame_from_bundles", "frame_of_spec", "check_frame_against_walk",
           "check_frame_against_geometry", "read_rpk", "write_rpk", "RPK_SCHEMA_VERSION"]

RPK_SCHEMA_VERSION = "0.6"


# --------------------------------------------------------------- master-walk normalisation
def _master_arrays(src) -> dict:
    """Normalise a master walk to the dict consumed by :func:`build_replay_pack`.

    ``src`` is a :class:`~dmipy_sim.persistent_walk.PersistentWalk` (what ``simulate_trajectories`` and
    ``simulate_mt_trajectories`` return), or its bank dict / ``.npz`` (``PersistentWalk._bank_dict``) exposing
    at least ``traj`` (n_walkers, n_t, 3), ``dt_traj`` and ``T_max``;
    the tier channels (``comp`` for bulk relaxation, ``dlog_b`` for
    surface relaxivity, ``bfrac`` for MT) are optional."""
    if isinstance(src, PersistentWalk):
        src = src._bank_dict()
    if not (isinstance(src, dict) or hasattr(src, "files")):
        raise TypeError(
            "build_replay_pack expects a PersistentWalk (the output of simulate_trajectories(..., "
            "tiers=\"all\")) or a master-walk dict / .npz with keys "
            "traj/dt_traj/T_max[/comp/dlog_b/bfrac]; "
            f"got {type(src).__name__}.")
    keys = src.files if hasattr(src, "files") else src.keys()
    m = {k: src[k] for k in keys}
    g = lambda k, d=None: (np.asarray(m[k]) if k in m and m[k] is not None else d)
    traj = m["traj"] if _cx.is_lazy(m["traj"]) else np.asarray(m["traj"])
    scal = lambda k: (float(np.asarray(m[k])) if m.get(k) is not None else None)   # idempotent re-normalise
    return dict(traj=traj, dt_traj=float(np.asarray(m["dt_traj"])),
                T_max=float(np.asarray(m["T_max"])), comp=g("comp"), comp0=g("comp0"),
                w=g("w"), dlog_b=g("dlog_b"), bfrac=g("bfrac"),
                # static field-grid susceptibility channel (dict of grids + world origin + chi)
                susc_field_basis=(m.get("susc_field_basis") if isinstance(m, dict) else None),
                susc_field_sampler=(m.get("susc_field_sampler") if isinstance(m, dict) else None),
                susc_field_samples=(m.get("susc_field_samples") if isinstance(m, dict) else None),
                susc_field_every=int(m.get("susc_field_every", 1) or 1) if isinstance(m, dict) else 1,
                susc_field_fill=(m.get("susc_field_fill") if isinstance(m, dict) else None),
                susc_grid_origin=(np.asarray(m["susc_grid_origin"]) if "susc_grid_origin" in m else None),
                susc_grid_raster=m.get("susc_grid_raster"),
                susc_grid_periodic=(tuple(bool(p) for p in m["susc_grid_periodic"]) if m.get("susc_grid_periodic") is not None
                                    else None),
                susc_chi_iso=scal("susc_chi_iso"), delta_chi_a=scal("delta_chi_a"),
                cell_size=scal("cell_size"), R=g("R"), D_intra=scal("D_intra"),
                substrate_frame=g("substrate_frame"),
                walkers_shuffled=bool(m.get("walkers_shuffled", False)),
                substrate=(m.get("substrate") if isinstance(m, dict) else None),
                n_walkers=int(traj.shape[0]), seed=seed_value(m.get("seed", 0)))


def _stores_field(m):
    """Whether the master ``m`` carries the field basis its pack's field channel (C3) is built from."""
    return m.get("susc_field_basis") is not None or m.get("susc_field_sampler") is not None


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
    X = traj if _cx.is_lazy(traj) else np.asarray(traj)           # only the two endpoints of each walker are read
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


def _measure_floor(m, env):
    """Split-half Monte-Carlo floor of the RAW walk over the envelope (decoded == raw ->
    fidelity err is 0, so ``floor_max`` is the substrate's own finite-N statistical noise).
    The walk is read as stored, in float64 chunks of at most :data:`compression.CHUNK_BYTES`."""
    traj = m["traj"] if _cx.is_lazy(m["traj"]) else np.asarray(m["traj"])
    return float(_cx.measure_fidelity(traj, float(m["dt_traj"]), traj, env)["floor_max"])


def _surface_fidelity(m, arrays, chan_meta, env, *, segment_T=None):
    """Certify the surface tier (C2): the surface-relaxivity attenuation reconstructed from the
    STORED boundary channel vs the RAW per-step boundary local time, over a rho2 battery, against
    the split-half MC floor of the raw surface signal. Returns ``dict(err, floor)`` or None.

    The whole-duration ``err``/``floor`` above is blind to the BAND: the bridge
    (:func:`compression.encode_boundary_bridge`) stores the summed contact exactly in its two
    endpoints at every K, so a band too coarse to resolve a later cut still reproduces the whole
    walk's total. When ``segment_T`` is given and this walk is longer than one window of that
    duration, ``err_window``/``floor_window`` add the same battery and split-half floor on the summed
    contact of EACH window of ``segment_T`` of the STORED channel
    (:func:`compression.decode_boundary_bridge`'s ``n_cut``/``start``, dropping the window's own
    first save as :func:`_window_master` does) against the window's own raw saves -- the worst window
    kept (dmipy-sim#528). Only the bridge codec carries a band to probe this way; the exact
    sparse/dense fallback never adds the window term."""
    raw = m.get("dlog_b")
    has_stored = _cx.has_c2(arrays) or any(k in arrays for k in ("blt_dense_q", "blt_counts"))
    if raw is None or not has_stored:
        return None
    raw = np.asarray(raw); n_w, n_t = raw.shape[0], raw.shape[1]         # as stored; float64 per walker chunk below
    w = np.asarray(m["w"], np.float64) if m.get("w") is not None else np.ones(n_w)
    D = float(m.get("D_intra") or 0.0) or 1.0
    decode = _cx.decode_boundary_bridge if _cx.has_c2(arrays) else _cx.decode_boundary_local_time
    # the log-weight at rho2/D = 1 is each walker's summed local time: raw and decoded, taken per chunk so
    # that neither the raw channel in float64 nor the decoded channel is ever held whole
    s_raw = np.empty(n_w); s_dec = np.empty(n_w)
    step = max(1, int(_cx.CHUNK_BYTES // (8 * n_t)))
    for lo in range(0, n_w, step):
        hi = min(lo + step, n_w)
        s_raw[lo:hi] = _cx.surface_logweight_series(raw[lo:hi], 1.0)
        s_dec[lo:hi] = _cx.surface_logweight_series(decode(arrays, chan_meta, slice(lo, hi)), 1.0)
    perm = np.random.RandomState(0).permutation(n_w); A, B = perm[:n_w // 2], perm[n_w // 2:]
    fac = lambda sl, idx: float(np.sum(w[idx] * np.exp(sl[idx])) / np.sum(w[idx]))
    rho2_list = env.get("rho2_list") or [1e-5, 3e-5, 1e-4]
    err = floor = 0.0
    for rho2 in rho2_list:
        rd = float(rho2) / D
        sl_raw = rd * s_raw
        sl_dec = rd * s_dec
        err = max(err, abs(fac(sl_raw, slice(None)) - fac(sl_dec, slice(None))))
        floor = max(floor, abs(fac(sl_raw, A) - fac(sl_raw, B)))
    out = dict(err=float(err), floor=float(floor))
    dt = m.get("dt_traj")
    if segment_T and dt and _cx.has_c2(arrays):
        dt = float(dt); T_walk = (n_t - 1) * dt
        if T_walk > float(segment_T) * (1.0 + 1e-9):
            steps = max(1, int(round(float(segment_T) / dt)))
            n_windows = (n_t - 1) // steps
            if n_windows >= 1:
                err_w = floor_w = 0.0
                for i in range(n_windows):
                    lo, hi = i * steps, i * steps + steps + 1
                    dec_win = decode(arrays, chan_meta, n_cut=hi, start=lo)
                    raw_win = raw[:, lo:hi]
                    dec_in = dec_win[:, 1:]; raw_in = raw_win[:, 1:]        # the window's own first save ends no step of it
                    s_raw_w = raw_in.sum(axis=1).astype(np.float64)
                    s_dec_w = dec_in.sum(axis=1).astype(np.float64)
                    for rho2 in rho2_list:
                        rd = float(rho2) / D
                        e = abs(fac(rd * s_raw_w, slice(None)) - fac(rd * s_dec_w, slice(None)))
                        f = abs(fac(rd * s_raw_w, A) - fac(rd * s_raw_w, B))
                        if f > 0 and e / f > (err_w / floor_w if floor_w else -1.0):
                            err_w, floor_w = e, f
                out["err_window"] = float(err_w)
                out["floor_window"] = float(floor_w)
    return out


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


def voxel_fidelity(traj, dt, decoded_pos, grid, comp, env, *, w=None, logw=None, chunk=20_000):
    """The pack's fidelity PER VOXEL AND POOL, for a walk meant to be partitioned: every walker is binned by where
    it started (``grid.bin``, the partition's own rule) and by its pool; per (voxel, pool) the ensemble signal
    over the envelope's acquisition battery is formed from the raw and the decoded paths, and a split-half
    Monte-Carlo floor from the voxel's own walkers. Returns ``(ijk, pools, n, floor, err)``: the occupied voxels
    ``(n_v, 3)``, the pool ids, and ``(n_v, n_pools)`` arrays of the walker count, the floor (max over the
    battery of ``|S_a - S_b| / 2``) and the codec error (max over the battery of ``|S_dec - S_raw|``); a voxel
    with fewer than two walkers of a pool has ``nan`` for its floor. Walker chunks keep the memory at
    ``chunk x n_meas`` complex."""
    traj = np.asarray(traj, np.float64); dec = np.asarray(decoded_pos, np.float64)
    n_w, n_t = traj.shape[0], traj.shape[1]
    G, _ = _cx.acquisition_battery(n_t, dt, env); n_m = G.shape[0]
    ijk_all, inside = grid.bin(traj[:, 0])
    pools = sorted(set(np.unique(np.asarray(comp)[:, 0]).tolist())) if comp is not None else [0]
    pid = np.asarray(comp)[:, 0].astype(np.int64) if comp is not None else np.zeros(n_w, np.int64)
    ijk, row = _census_rows(ijk_all, inside)
    col = np.searchsorted(pools, pid)
    ww = np.ones(n_w) if w is None else np.asarray(w, np.float64)
    lw = np.zeros(n_w) if logw is None else np.asarray(logw, np.float64)
    half = (np.random.default_rng(0).permutation(n_w) % 2).astype(bool)      # a fixed split of every voxel's walkers
    n_v, n_p = ijk.shape[0], len(pools)
    S_raw = np.zeros((n_v, n_p, n_m), complex); S_dec = np.zeros_like(S_raw)
    S_a = np.zeros_like(S_raw); S_b = np.zeros_like(S_raw)
    W = np.zeros((n_v, n_p)); W_a = np.zeros((n_v, n_p)); W_b = np.zeros((n_v, n_p)); N = np.zeros((n_v, n_p), np.int64)
    for i in range(0, n_w, chunk):
        sl = slice(i, i + chunk); m = row[sl] >= 0
        if not m.any():
            continue
        idx = np.arange(i, min(i + chunk, n_w))[m]
        e_raw = np.exp(lw[idx, None] + 1j * _cx._walker_phases(traj[idx], dt, G)) * ww[idx, None]
        e_dec = np.exp(lw[idx, None] + 1j * _cx._walker_phases(dec[idx], dt, G)) * ww[idx, None]
        r, c, h = row[idx], col[idx], half[idx]
        np.add.at(S_raw, (r, c), e_raw); np.add.at(S_dec, (r, c), e_dec)
        np.add.at(S_a, (r[h], c[h]), e_raw[h]); np.add.at(S_b, (r[~h], c[~h]), e_raw[~h])
        np.add.at(W, (r, c), ww[idx]); np.add.at(W_a, (r[h], c[h]), ww[idx][h]); np.add.at(W_b, (r[~h], c[~h]), ww[idx][~h])
        np.add.at(N, (r, c), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        err = np.abs(S_dec / W[..., None] - S_raw / W[..., None]).max(-1)
        floor = 0.5 * np.abs(S_a / W_a[..., None] - S_b / W_b[..., None]).max(-1)
    err[W == 0] = np.nan; floor[(W_a == 0) | (W_b == 0)] = np.nan
    return ijk, pools, N, floor, err


def voxel_floor_coded(C, dt, n_t, grid, comp, env, *, w=None, device="auto", chunk=200_000):
    """The per-(voxel, pool) split-half floor of a walk meant to be partitioned, from its bridge coefficients
    (:func:`voxel_fidelity` without the dense oracle: the same binning by start position -- the exact first
    endpoint -- and pool, the same fixed split, the phases by :func:`compression.coded_phases`). Returns
    ``(ijk, pools, n, floor, err)`` with ``err`` all ``nan``: the codec error is the certifying pack's."""
    C = np.asarray(C); n_w = C.shape[0]
    G, _ = _cx.acquisition_battery(int(n_t), float(dt), env); n_m = G.shape[0]
    ijk_all, inside = grid.bin(np.asarray(C[:, 0, :], np.float64))
    pools = sorted(set(np.unique(np.asarray(comp)[:, 0]).tolist())) if comp is not None else [0]
    pid = np.asarray(comp)[:, 0].astype(np.int64) if comp is not None else np.zeros(n_w, np.int64)
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
        e = np.exp(1j * _cx.coded_phases(C[idx], dt, G, n_t, device=device)) * ww[idx, None]
        r, c, h = row[idx], col[idx], half[idx]
        np.add.at(S_a, (r[h], c[h]), e[h]); np.add.at(S_b, (r[~h], c[~h]), e[~h])
        np.add.at(W_a, (r[h], c[h]), ww[idx][h]); np.add.at(W_b, (r[~h], c[~h]), ww[idx][~h])
        np.add.at(N, (r, c), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        floor = 0.5 * np.abs(S_a / W_a[..., None] - S_b / W_b[..., None]).max(-1)
    floor[(W_a == 0) | (W_b == 0)] = np.nan
    return ijk, pools, N, floor, np.full_like(floor, np.nan)


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
    from ..replay.compression import decode_occupancy
    if pack.substrate is None:
        raise ValueError("this pack embeds no spec, so its pools have no names")
    names = {p.id: p.name for p in pack.substrate.pools}
    ch = pack.meta.get("compression", {}).get("channels", {}) or {}
    if "compartment" not in ch:
        raise ValueError("the per-pool floor needs the pack's compartment channel")
    ids = np.asarray(decode_occupancy(pack.arrays, ch["compartment"])["comp"]); ids = ids[:, 0] if ids.ndim == 2 else ids
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


def _field_of(m):
    """The master's field source as a sampler (``channels(points)`` / ``field(points, ...)``): the grid as a
    :class:`FieldGrid`, or the strand substrate's :class:`StrandFieldBasis`; None without a field tier."""
    fb = m.get("susc_field_basis")
    if fb is not None:
        from ..fields.susceptibility_field import FieldGrid
        return FieldGrid(fb, np.asarray(m["susc_grid_origin"], float), periodic=grid_periodic_of(m))
    return m.get("susc_field_sampler")


def _field_along(field, traj, b0_dir, *, B0, chi_iso, chi_aniso, walkers_per_chunk=None):
    """``(n_w, n_t)`` field (Tesla) along every walker's path, in walker chunks."""
    traj = np.asarray(traj, np.float64); n_w, n_t = traj.shape[0], traj.shape[1]
    step = walkers_per_chunk or max(1, int(2e8 // max(n_t * 13, 1)))
    out = np.empty((n_w, n_t), np.float64)
    for i in range(0, n_w, step):
        out[i:i + step] = field.field(traj[i:i + step].reshape(-1, 3), b0_dir, B0=B0, chi_iso=chi_iso,
                                      chi_aniso=chi_aniso).reshape(-1, n_t)
    return out


def _raw_field(m, field, traj, b0_dir, *, B0, chi_iso, chi_aniso, idx=None):
    """The reference field along the walk for the path certificates: the walk's own samples (the interval means
    it stored) when it took them, else the field evaluated at the stored points."""
    from ..fields.hollow_cylinder import contract
    samples = m.get("susc_field_samples")
    if samples is not None:
        sm = np.asarray(samples if idx is None else samples[idx], np.float64)
        return contract(sm, b0_dir, B0=B0, chi_iso=chi_iso, chi_aniso=chi_aniso)
    return _field_along(field, traj, b0_dir, B0=B0, chi_iso=chi_iso, chi_aniso=chi_aniso)


def _susc_grid_fidelity(m, arrays, gm, decoded_pos, dt, env):
    """Certify the static field-grid tier (SE + GRE): the STORED f16 grid sampled at the DECODED
    trajectory vs the RAW f64 grid at the FULL-resolution trajectory (folds in f16 quantisation AND
    the position-codec error), against the split-half MC floor of the raw signal. Returns dict or None."""
    from ..constants import GAMMA
    from ..fields.susceptibility_field import assemble_field, sample_grid
    fb = m.get("susc_field_basis")
    if fb is None or "susc_grid_iso_local" not in arrays:
        return None
    origin = np.asarray(gm["origin"], float); vs = np.asarray(gm["voxel_size"], float); per = grid_periodic_of(gm)
    raw_traj = np.asarray(m["traj"], np.float64); n_w, n_t = raw_traj.shape[0], raw_traj.shape[1]
    w = np.asarray(m["w"], np.float64) if m.get("w") is not None else np.ones(n_w)
    braw = {"iso_local": np.asarray(fb["iso_local"], np.float64), "iso_P": np.asarray(fb["iso_P"], np.float64),
            "aniso_G": (np.asarray(fb["aniso_G"], np.float64) if fb.get("aniso_G") is not None else None),
            "shape": tuple(fb["shape"]), "voxel_size": vs}
    bsto = grid_basis_of(arrays, gm)
    chi_i = float(m.get("susc_chi_iso") or 1.06e-6)
    # Certify the ANISOTROPIC channels even when the substrate's reference delta_chi_a is 0 (Winther
    # used isotropic myelin). The channels are geometry only and chi_aniso is a replay knob, so if we
    # store aniso_G we must verify it reproduces -- with ca=0 the gate would silently skip 6 of 13
    # channels and pass. The probe magnitude below is a CERTIFICATION value, not a physical claim.
    ca = float(m.get("delta_chi_a") or 0.0)
    if gm.get("has_aniso") and ca == 0.0:
        ca = 0.1 * float(m.get("susc_chi_iso") or 1.06e-6)
    if not gm.get("has_aniso"):
        ca = 0.0
    se = np.sign(0.5 * (n_t - 1) - np.arange(n_t)).astype(float); gre = np.ones(n_t)
    perm = np.random.RandomState(0).permutation(n_w); A, B = perm[:n_w // 2], perm[n_w // 2:]
    wmean = lambda c, idx: float(np.sum(w[idx] * c[idx]) / np.sum(w[idx]))
    err = floor = 0.0
    for B0 in (env.get("B0_list") or [3.0, 7.0]):
        for th in (env.get("theta_deg") or [0, 90]):
            t = np.deg2rad(float(th)); d = [np.sin(t), 0.0, np.cos(t)]
            s_raw = sample_grid(assemble_field(braw, d, B0=B0, chi_iso=chi_i, chi_aniso=ca), raw_traj, origin, vs, periodic=per)
            s_dec = sample_grid(assemble_field(bsto, d, B0=B0, chi_iso=chi_i, chi_aniso=ca), decoded_pos, origin, vs, periodic=per)
            for gate in (se, gre):
                cr = np.cos(GAMMA * dt * (s_raw * gate[None, :]).sum(1))
                cd = np.cos(GAMMA * dt * (s_dec * gate[None, :]).sum(1))
                err = max(err, abs(wmean(cr, slice(None)) - wmean(cd, slice(None))))
                floor = max(floor, abs(wmean(cr, A) - wmean(cr, B)))
    return dict(err=float(err), floor=float(floor))


_ISO_P_NAMES = ("xx", "yy", "zz", "xy", "xz", "yz")


def _cpmg_gate(n_t, n_pulses):
    """Transverse-phase gate for a CPMG train of ``n_pulses`` 180s at the usual (k+1/2)*TE/n times."""
    t = np.arange(n_t) / max(n_t - 1, 1)
    s = np.ones(n_t)
    for k in range(n_pulses):                       # each 180 flips the accumulated-phase sign
        s[t >= (k + 0.5) / n_pulses] *= -1.0
    return s


def _ste_gate(n_t):
    """Transverse-phase gate of a stimulated echo filling the window: transverse for its first quarter, stored along
    z (accruing nothing) for the half between the store and the recall, then transverse with the recall's sign."""
    t = np.arange(n_t) / max(n_t - 1, 1)
    return np.where(t < 0.25, 1.0, np.where(t < 0.75, 0.0, -1.0))


def _battery_gates(n_t, n_pulses):
    """The gates the field tier is certified under on an ``n_t``-sample window: a gradient echo, a spin echo, the CPMG
    train at the tier's depth ``n_pulses``, and a stimulated echo (its mixing time stored)."""
    return [np.ones(n_t), _cpmg_gate(n_t, 1), _cpmg_gate(n_t, n_pulses), _ste_gate(n_t)]


def _susc_path_bloch_fidelity(m, arrays, pm, gm, env, n_sub=8000):
    """Certify the susc_path tier ON THE VECTOR-BLOCH PATH -- the primary consumer.

    The scalar gate in :func:`_susc_path_fidelity` integrates the right quantity, but it does not
    exercise the Bloch operator, where the RF rotations do NOT commute with the field's z-precession
    and relaxation interleaves between pulses. A pack that claims CPMG capability should be certified
    by the operator that actually replays CPMG, so this drives ``replay_bloch`` with the STORED
    (K-truncated, f16) channels against the RAW full-resolution field, at spin echo and at the CPMG
    train the pack advertises.

    Walker-subsampled (``n_sub``) because the quantity measured is codec error, which is per-walker;
    the comparison floor is the split-half floor OF THAT SUBSAMPLE, so the verdict stays self-consistent.
    """
    from ..constants import GAMMA
    from .trajectories import replay_bloch
    field = _field_of(m)
    if field is None or "susc_path_dct" not in arrays:
        return None
    n_w, n_t = m["traj"].shape[0], m["traj"].shape[1]            # the walk is read for the subsample alone
    dt = float(m["dt_traj"]); TE = (n_t - 1) * dt
    k = int(min(n_sub, n_w))
    pos = np.asarray(m["traj"][:k], np.float64)
    w = (np.asarray(m["w"], np.float64)[:k] if m.get("w") is not None else np.ones(k))
    b_dec, _ = susc_path_decode(arrays, pm, n_w=k)
    chi_i = float(m.get("susc_chi_iso") or 1.06e-6)
    # Certify the ANISOTROPIC channels even when the substrate's reference delta_chi_a is 0 (Winther
    # used isotropic myelin). The channels are geometry only and chi_aniso is a replay knob, so if we
    # store aniso_G we must verify it reproduces -- with ca=0 the gate would silently skip 6 of 13
    # channels and pass. The probe magnitude below is a CERTIFICATION value, not a physical claim.
    ca = float(m.get("delta_chi_a") or 0.0)
    if gm.get("has_aniso") and ca == 0.0:
        ca = 0.1 * float(m.get("susc_chi_iso") or 1.06e-6)
    if not gm.get("has_aniso"):
        ca = 0.0
    n_p = int(pm.get("max_refocus_pulses") or 1)

    def rf_train(npul):
        ev = [RFEvent(0.0, 90.0, 'Mz→Mxy')]
        for j in range(npul):
            ev.append(RFEvent((j + 0.5) * TE / npul, 180.0, 'refocus', axis_deg=90.0))
        return ev

    G0 = np.zeros((1, n_t, 3))                       # b=0 isolates the susceptibility physics
    every = int(m.get("susc_field_every", 1) or 1)

    def sig(field, rf, idx=None):
        f = field if idx is None else field[idx]
        if every > 1:                                                     # a field on its own grid, held over its saves
            f = np.repeat(f, every, axis=1)[:, :n_t]
        p = pos if idx is None else pos[idx]
        ww = w if idx is None else w[idx]
        out = replay_bloch(p, dt, G0, dt, rf, T2=None, T1=None,
                           extra_phase_per_step=GAMMA * dt * f, weights=ww)
        a = np.asarray(out[0] if isinstance(out, tuple) else out)
        return float(np.abs(np.atleast_1d(a.ravel())[-1]))

    perm = np.random.RandomState(0).permutation(k); A, B = perm[:k // 2], perm[k // 2:]
    err = floor = 0.0
    B0s = (env.get("B0_list") or [7.0])[:1]          # one field strength: the tier scales linearly in B0
    for B0 in B0s:
        for th in (env.get("theta_deg") or [90]):
            t = np.deg2rad(float(th)); d = [np.sin(t), 0.0, np.cos(t)]
            f_raw = _raw_field(m, field, pos, d, B0=B0, chi_iso=chi_i, chi_aniso=ca, idx=slice(0, k))
            f_dec = susc_path_field(b_dec, d, B0=B0, chi_iso=chi_i, chi_aniso=ca,
                                    has_aniso=bool(gm.get("has_aniso")))
            for npul in (1, n_p):
                rf = rf_train(npul)
                sr = sig(f_raw, rf)
                err = max(err, abs(sig(f_dec, rf) - sr))
                floor = max(floor, abs(sig(f_raw, rf, A) - sig(f_raw, rf, B)))
    return dict(err=float(err), floor=float(floor), n_pulses=n_p, n_walkers=k)


#: the bands the field tier is tried at, in order, when its band is derived on the walk
SUSC_PATH_LADDER = (16, 32, 64, 128, 256, 512, 1024, 2048)


def derive_susc_path_K(m, field, env, *, K_max, ladder=SUSC_PATH_LADDER, containers=(8, 16)):
    """The field tier's band and container, derived on the walk it will store: ``(K, bits, record)``.

    The path channel's band is the pack's to choose per substrate, since what it must resolve is the field a
    walker sees along its path, and that depends on the geometry (a sheath two nodes wide changes the field at
    every step, a smooth far field does not) and on the gates the tier serves; and its container is the pack's
    too, since an integer container's error is set by the largest coefficient of a band over the walkers and does
    not fall with the band (on the CATERPillar pilot the 8-bit tier sits at 1.6 times its floor at every band, the
    16-bit one at a third of it by 256). So both are measured rather than fixed. The field is encoded once at the
    top of ``ladder`` (the exact series, ``n_t`` bands, closes it), and the rungs are read in ascending band, the
    narrower container first, each as the pack's own certificate reads it -- the coefficients truncated to the
    rung and quantised, contracted for each field strength and direction of the envelope's battery, decoded and
    gated by GRE, spin echo, the CPMG train the rung serves and a stimulated echo (:func:`_battery_gates`) -- against the
    split-half floor of the reference. The first pair within the floor is the band and the container: the
    cheapest in bytes per walker that keeps the tier's accuracy. ``K_max``, the position channel's band, is the
    point past which the tier costs more than the walk it rides on; the ladder continues beyond it when the
    floor is not yet reached, and the record says so. A refocusing depth the envelope declares
    (``max_refocus_pulses``) puts a floor of twice that under the ladder, since the tier serves ``K / 2``
    pulses. When no pair reaches the floor at all (a container's own error above it at the exact series), the
    pair of least error is taken, the cheapest among those within three per cent of it. ``record`` lists every
    pair read.
    """
    from scipy.fft import idct
    traj = m["traj"]; n_w = traj.shape[0]          # read whole only to sample a field the walk did not sample itself
    every = int(m.get("susc_field_every", 1) or 1)
    n_t = len(range(0, traj.shape[1], every)); dt = float(m["dt_traj"]) * every
    # the band bounds the refocusing depth the tier serves, K / 2 pulses; a train the envelope declares is the depth
    # every rung is read at and certified for, and a floor under the ladder, so a pack built for a twelve-pulse train
    # is never derived at eight and a wide band is not tested as a claim to a thousand pulses
    depth = int(env.get("max_refocus_pulses") or 0)
    min_K = 2 * depth
    rungs = sorted(set([min(int(k), n_t) for k in ladder if k < n_t] + [n_t] + ([min(min_K, n_t)] if min_K else [])))
    rungs = [k for k in rungs if k >= min_K] or [n_t]
    top = rungs[-1]
    if m.get("susc_field_samples") is not None:
        arrays_top, meta_top = susc_path_encode_series(np.asarray(m["susc_field_samples"]), list(field.channel_names), K=top, bits=None,
                                                       layout="wtc", dt=dt, dtype=np.float32)
    else:
        arrays_top, meta_top = susc_path_encode(field, np.asarray(traj, np.float64), K=top, bits=None, dtype=np.float32)
    coeffs = np.asarray(arrays_top["susc_path_dct"], np.float32)               # (n_w, n_ch, top)
    has_aniso = coeffs.shape[1] >= 12
    w = np.asarray(m["w"], np.float64) if m.get("w") is not None else np.ones(n_w)
    chi_i = float(m.get("susc_chi_iso") or 1.06e-6)
    ca = float(m.get("delta_chi_a") or 0.0)
    if has_aniso and ca == 0.0:
        ca = 0.1 * chi_i
    if not has_aniso:
        ca = 0.0
    perm = np.random.RandomState(0).permutation(n_w); A, B = perm[:n_w // 2], perm[n_w // 2:]
    settings = [(B0, th) for B0 in (env.get("B0_list") or [3.0, 7.0]) for th in (env.get("theta_deg") or [0, 90])]
    raw = {}                                                                                    # the reference per setting, once
    for B0, th in settings:
        t = np.deg2rad(float(th)); d = [np.sin(t), 0.0, np.cos(t)]
        raw[(B0, th)] = np.asarray(_raw_field(m, field, traj, d, B0=B0, chi_iso=chi_i, chi_aniso=ca), np.float32)
    err, floor, order = {}, {}, []
    chosen = None
    for k in rungs:
        for bits in containers:
            e_max = f_max = 0.0
            n_p = _depth(k, depth)                                                                 # the depth this rung would serve
            a_k, m_k = _quantise_susc_path(np.asarray(coeffs[:, :, :k], np.float64), dict(meta_top, K=k, max_refocus_pulses=n_p), bits)
            C, _names = susc_path_coeffs(a_k, m_k)                                                 # dequantised, zz re-inserted
            gates = _battery_gates(n_t, max(1, n_p))
            for B0, th in settings:
                t = np.deg2rad(float(th)); d = [np.sin(t), 0.0, np.cos(t)]
                cd = susc_path_field(C, d, B0=B0, chi_iso=chi_i, chi_aniso=ca, has_aniso=has_aniso)   # (n_w, k): linear in the channels
                f_dec = idct(np.pad(cd, ((0, 0), (0, n_t - k))), type=2, norm="ortho", axis=1)
                e, f = _gate_battery(raw[(B0, th)], f_dec, gates, w, A, B, dt)
                e_max, f_max = max(e_max, e), max(f_max, f)
            err[(k, bits)], floor[(k, bits)] = e_max, f_max; order.append((k, bits))
            if e_max <= f_max:
                chosen = (k, bits); break
        if chosen is not None:
            break
    if chosen is None:                                                    # nothing within the floor: the least error, cheaply
        best = min(err.values())
        chosen = min([p for p in order if err[p] <= 1.03 * best], key=lambda p: (p[0] * p[1], p[0]))
    K, bits = chosen
    record = dict(rule="derived",
                  criterion="the first (band, container) pair in ascending band, the narrower container first, whose codec error on "
                            "the certificate's battery is within its floor; failing every pair, the cheapest among the least errors",
                  ladder=[dict(K=int(k), bits=int(b), err=float(err[(k, b)]), floor=float(floor[(k, b)])) for k, b in order],
                  K_max=int(K_max), min_K=int(min_K), depth=(int(depth) or None), above_position_band=bool(K > int(K_max)),
                  within_floor=bool(err[chosen] <= floor[chosen]))
    return int(K), int(bits), record


def _susc_path_fidelity(m, arrays, pm, gm, env):
    """Certify the susc_path_dct tier AT ITS DECLARED CAPABILITY.

    Compares the stored (K-truncated, f16) channels against the raw full-resolution field sampled on
    the true trajectory, under the gates the pack claims to support (:func:`_battery_gates`) -- GRE, spin
    echo, the CPMG train at ``max_refocus_pulses`` and a stimulated echo. The CPMG gate is the binding one:
    truncation error grows with gate bandwidth, so certifying on SE alone would pass a pack that fails the trains
    it advertises.
    """
    field = _field_of(m)
    if field is None or "susc_path_dct" not in arrays:
        return None
    traj = m["traj"]; n_w = traj.shape[0]          # read whole only to sample a field the walk did not sample itself
    every = int(m.get("susc_field_every", 1) or 1)
    n_t = len(range(0, traj.shape[1], every))                    # the channel's own grid
    dt = float(m["dt_traj"]) * every
    w = np.asarray(m["w"], np.float64) if m.get("w") is not None else np.ones(n_w)
    b_dec, _ = susc_path_decode(arrays, pm)
    chi_i = float(m.get("susc_chi_iso") or 1.06e-6)
    # Certify the ANISOTROPIC channels even when the substrate's reference delta_chi_a is 0 (Winther
    # used isotropic myelin). The channels are geometry only and chi_aniso is a replay knob, so if we
    # store aniso_G we must verify it reproduces -- with ca=0 the gate would silently skip 6 of 13
    # channels and pass. The probe magnitude below is a CERTIFICATION value, not a physical claim.
    ca = float(m.get("delta_chi_a") or 0.0)
    if gm.get("has_aniso") and ca == 0.0:
        ca = 0.1 * float(m.get("susc_chi_iso") or 1.06e-6)
    if not gm.get("has_aniso"):
        ca = 0.0
    n_p = int(pm.get("max_refocus_pulses") or 1)
    gates = _battery_gates(n_t, n_p)
    perm = np.random.RandomState(0).permutation(n_w); A, B = perm[:n_w // 2], perm[n_w // 2:]
    err = floor = 0.0
    for B0 in (env.get("B0_list") or [3.0, 7.0]):
        for th in (env.get("theta_deg") or [0, 90]):
            t = np.deg2rad(float(th)); d = [np.sin(t), 0.0, np.cos(t)]
            f_raw = _raw_field(m, field, traj, d, B0=B0, chi_iso=chi_i, chi_aniso=ca)
            f_dec = susc_path_field(b_dec, d, B0=B0, chi_iso=chi_i, chi_aniso=ca,
                                    has_aniso=bool(gm.get("has_aniso")))
            e, f = _gate_battery(f_raw, f_dec, gates, w, A, B, dt)
            err, floor = max(err, e), max(floor, f)
    return dict(err=float(err), floor=float(floor), n_pulses_certified=n_p)


def _gate_battery(f_raw, f_dec, gates, w, A, B, dt):
    """The worst gated-phase replay error between two per-walker field series over ``gates``, and the split-half
    floor of the reference: ``(err, floor)``."""
    from ..constants import GAMMA
    wmean = lambda c, idx: float(np.sum(w[idx] * c[idx]) / np.sum(w[idx]))
    err = floor = 0.0
    for g in gates:
        cr = np.cos(GAMMA * dt * (f_raw * g[None, :]).sum(1))
        cd = np.cos(GAMMA * dt * (f_dec * g[None, :]).sum(1))
        err = max(err, abs(wmean(cr, slice(None)) - wmean(cd, slice(None))))
        floor = max(floor, abs(wmean(cr, A) - wmean(cr, B)))
    return err, floor


def susc_path_series_fidelity(series_raw, arrays, pm, gm, *, w, dt, env=None, chi_iso=1.06e-6, chi_aniso=None):
    """Certify a path channel against a REFERENCE SERIES ``(n_w, n_ch, n_t)`` in the canonical channel order (what a
    re-encoding of a decoded channel is measured against, since the grid is not in the pack): the same battery
    as the producer's (:func:`_battery_gates`: GRE, spin echo, the CPMG train at the channel's
    ``max_refocus_pulses`` and a stimulated echo), over the envelope's ``B0_list`` and ``theta_deg`` -- against the
    split-half floor of the reference."""
    env = env or {}
    series_raw = np.asarray(series_raw, np.float64); n_w, n_t = series_raw.shape[0], series_raw.shape[2]
    b_dec, _ = susc_path_decode(arrays, pm, n_w=n_w)
    has_aniso = bool(gm.get("has_aniso")) and series_raw.shape[1] >= 13
    ca = (0.1 * chi_iso if has_aniso else 0.0) if chi_aniso is None else float(chi_aniso)
    n_p = int(pm.get("max_refocus_pulses") or 1)
    gates = _battery_gates(n_t, n_p)
    perm = np.random.RandomState(0).permutation(n_w); A, B = perm[:n_w // 2], perm[n_w // 2:]
    w = np.ones(n_w) if w is None else np.asarray(w, np.float64)
    err = floor = 0.0
    for B0 in (env.get("B0_list") or [3.0, 7.0]):
        for th in (env.get("theta_deg") or [0, 90]):
            t = np.deg2rad(float(th)); d = [np.sin(t), 0.0, np.cos(t)]
            f_raw = susc_path_field(series_raw, d, B0=B0, chi_iso=chi_iso, chi_aniso=ca, has_aniso=has_aniso)
            f_dec = susc_path_field(b_dec, d, B0=B0, chi_iso=chi_iso, chi_aniso=ca, has_aniso=has_aniso)
            e, f = _gate_battery(f_raw, f_dec, gates, w, A, B, float(dt))
            err, floor = max(err, e), max(floor, f)
    return dict(err=float(err), floor=float(floor), n_pulses_certified=n_p)


def susc_path_encode(field, traj, *, K=32, bits=8, dtype=np.float16, atol_trace=1e-6, max_refocus_pulses=None):
    """Encode the off-resonance field ALONG each walker's path as K temporal DCT-II coefficients.

    ``field`` is the substrate's field source -- a :class:`~dmipy_sim.fields.susceptibility_field.FieldGrid` or a
    :class:`~dmipy_sim.fields.strand_field.StrandFieldBasis` -- read through ``channels(points)``; ``traj`` the
    full-resolution walk ``(n_w, n_t, 3)``. The ``iso_P_zz`` channel is implied by the trace identity
    ``iso_P_xx + iso_P_yy + iso_P_zz = 3 iso_local`` and left out when the source satisfies it to ``atol_trace``.
    """
    from scipy.fft import dct
    traj = np.asarray(traj, np.float64)
    n_w, n_t = traj.shape[0], traj.shape[1]
    names_all = list(field.channel_names)
    step = max(1, int(2e8 // max(n_t * 13, 1)))                 # walkers per chunk: ~1.6 GB of channels
    first = field.channels(traj[:min(step, n_w)].reshape(-1, 3))
    tr = first[:, 1] + first[:, 2] + first[:, 3]
    scale = float(np.max(np.abs(first[:, 0]))) or 1.0
    trace_res = float(np.max(np.abs(tr - 3.0 * first[:, 0]))) / (3.0 * scale)
    drop_zz = bool(trace_res <= atol_trace)
    keep = [i for i, n in enumerate(names_all) if not (drop_zz and n == "iso_P_zz")]
    names = [names_all[i] for i in keep]
    grids = keep                                                  # the channel columns stored
    K = int(min(K, n_t))
    coeffs = np.empty((n_w, len(grids), K), np.float64)
    for i in range(0, n_w, step):                      # walker chunks: the channels of a chunk, then their DCT
        ch = first if i == 0 else field.channels(traj[i:i + step].reshape(-1, 3))
        ch = ch.reshape(-1, n_t, ch.shape[-1])
        for c, col in enumerate(grids):
            coeffs[i:i + step, c, :] = dct(ch[:, :, col], type=2, norm="ortho", axis=1)[:, :K]
    meta = dict(channel="susc_path_dct", K=K, n_t=int(n_t), n_ch=len(grids), channels=names,
                iso_P_zz=("implied" if drop_zz else "stored"), trace_residual=trace_res,
                max_refocus_pulses=_depth(K, max_refocus_pulses))
    if bits is None:
        meta["bits"] = None; meta["dtype"] = np.dtype(dtype).name
        return {"susc_path_dct": np.asarray(coeffs, dtype)}, meta
    if bits not in (8, 16):
        raise ValueError("susc_path bits must be 8, 16, or None (got %r); sub-byte depths need "
                         "bit-packing to save bytes and 6-bit measured above the MC floor" % (bits,))
    return _quantise_susc_path(coeffs, meta, bits)


def _quantise_susc_path(coeffs, meta, bits, *, device="numpy"):
    """The integer container of the path-field coefficients: a per-(channel, band) scale, ``bits`` wide -- the
    reduction (the scale's max-abs over every walker) and the round/clip run on the JAX device when ``device``
    resolves to one (float32: the container is already lossy, so this loses nothing the quantisation does not),
    host numpy (float64, the reference) otherwise."""
    itype = np.int8 if bits == 8 else np.int16
    lim = 2 ** (bits - 1) - 1
    if _cx.resolve_device(device) == "numpy":
        scale = np.abs(coeffs).max(axis=0) / lim                     # (n_ch, K), per channel AND band
        scale[scale == 0] = 1.0
        q = np.clip(np.rint(coeffs / scale), -lim, lim).astype(itype)
    else:
        import jax.numpy as jnp
        c = jnp.asarray(coeffs, jnp.float32)
        scale = np.asarray(jnp.max(jnp.abs(c), axis=0)) / lim
        scale[scale == 0] = 1.0
        q = np.asarray(jnp.clip(jnp.round(c / jnp.asarray(scale, jnp.float32)[None]), -lim, lim)).astype(itype)
    meta["bits"] = int(bits); meta["dtype"] = np.dtype(itype).name
    return {"susc_path_dct": q, "susc_path_scale": np.asarray(scale, np.float32)}, meta


def _depth(K, declared):
    """The refocusing depth a path channel of ``K`` bands serves: the declared train when there is one, never more than
    the ``K / 2`` pulses the band can gate; ``K / 2`` itself when none is declared."""
    return int(min(int(K) // 2, int(declared))) if declared else int(K) // 2


def susc_path_encode_series(series, names, *, K=32, bits=8, dtype=np.float16, layout="wct", atol_trace=1e-4, device="auto",
                            chunk=20_000, dt=None, max_refocus_pulses=None):
    """:func:`susc_path_encode` from the per-save field series itself, in the canonical channel order with
    ``names``: ``(n_w, n_ch, n_t)`` (``layout="wct"``, what a decoded path channel gives) or ``(n_w, n_t, n_ch)``
    (``layout="wtc"``, the interval means a walk sampled), encoded in walker chunks of ``chunk`` on ``device``
    without a copy of the whole series. The ``iso_P_zz`` channel is implied by the trace identity
    ``iso_P_xx + iso_P_yy + iso_P_zz = 3 iso_local`` and left out when the series satisfies it to ``atol_trace``
    (the closed-form strand field does exactly, and a series sampled in float32 over a few hundred strands keeps it
    to ~1e-5; a windowed k-space grid breaks it at the percent level); the decoder re-inserts it. ``dt``
    is the series' own step (the walk's save step times ``field_sample_every``), recorded so the replay gates the
    channel on its grid; absent, the channel is on the pack's save grid."""
    if layout not in ("wct", "wtc"):
        raise ValueError("layout is 'wct' (n_w, n_ch, n_t) or 'wtc' (n_w, n_t, n_ch)")
    series = np.asarray(series)
    n_w = series.shape[0]; n_ch = series.shape[1] if layout == "wct" else series.shape[2]
    n_t = series.shape[2] if layout == "wct" else series.shape[1]
    names = list(names)
    if len(names) != n_ch:
        raise ValueError(f"{len(names)} channel names for {n_ch} channels")
    K = int(min(K, n_t))
    take = lambda sl: (np.transpose(series[sl], (0, 2, 1)) if layout == "wct" else series[sl])   # (rows, n_t, n_ch)
    # the trace identity is judged on a sample of walkers in float64 (a whole chunk in float64 was 1.7 GB of host copy)
    probe = np.asarray(take(slice(0, min(2000, n_w))), np.float64)
    drop_zz = False; trace_res = None
    if {"iso_local", "iso_P_xx", "iso_P_yy", "iso_P_zz"} <= set(names):
        i0, ix, iy, iz = (names.index(n) for n in ("iso_local", "iso_P_xx", "iso_P_yy", "iso_P_zz"))
        tr = probe[..., ix] + probe[..., iy] + probe[..., iz]
        scale = float(np.max(np.abs(probe[..., i0]))) or 1.0
        trace_res = float(np.max(np.abs(tr - 3.0 * probe[..., i0]))) / (3.0 * scale)
        drop_zz = bool(trace_res <= atol_trace)
    keep = [i for i, n in enumerate(names) if not (drop_zz and n == "iso_P_zz")]
    drop = None if len(keep) == n_ch else keep                   # a dropped channel is selected past the host copy
    coeffs = np.empty((n_w, len(keep), K), np.float64)
    for i in range(0, n_w, chunk):
        ch = take(slice(i, i + chunk))                           # the series' own dtype, read in place -- no copy
        coeffs[i:i + chunk] = _cx.dct_bands_series(ch, K, device=device, keep=drop)   # one cached device jit for every chunk
    meta = dict(channel="susc_path_dct", K=K, n_t=int(n_t), n_ch=len(keep), channels=[names[i] for i in keep],
                iso_P_zz=("implied" if drop_zz else "stored"), trace_residual=trace_res,
                max_refocus_pulses=_depth(K, max_refocus_pulses))
    if dt is not None:
        meta["dt"] = float(dt)
    if bits is None:
        meta["bits"] = None; meta["dtype"] = np.dtype(dtype).name
        return {"susc_path_dct": np.asarray(coeffs, dtype)}, meta
    if bits not in (8, 16):
        raise ValueError("susc_path bits must be 8, 16, or None")
    return _quantise_susc_path(coeffs, meta, bits, device=device)


def susc_path_coeffs(arrays, meta):
    """Dequantised C3 coefficients with ``iso_P_zz`` re-inserted -- ``(n_w, n_ch_full, K)``.

    The ONE place that undoes the storage container. ``susc_path_dct`` may be an integer array whose
    scale lives in ``susc_path_scale`` (see :func:`susc_path_encode`), so reading it raw yields
    integers ~1e3 off with no error -- every consumer goes through here. The zz identity is applied
    in coefficient space; the DCT is linear, so this is identical to applying it after the transform.
    """
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
    if meta.get("iso_P_zz") == "implied":
        i_loc, i_xx, i_yy = (names.index(n) for n in ("iso_local", "iso_P_xx", "iso_P_yy"))
        at = names.index("iso_P_xy")
        C = np.insert(C, at, 3.0 * C[:, i_loc] - C[:, i_xx] - C[:, i_yy], axis=1)
        names = names[:at] + ["iso_P_zz"] + names[at:]
    return C, names


def susc_path_decode(arrays, meta, *, n_w=None, n_cut=None, start=0):
    """Reconstruct b_c(t) per walker from the stored coefficients; re-inserts iso_P_zz if implied.

    Returns ``(field, names)`` with field ``(n_w, n_ch_full, n_t)`` in the canonical channel order
    (iso_local, iso_P_xx..yz, [aniso_G_xx..yz]) so the Q(H) contraction indexes it directly. ``start`` / ``n_cut``
    ask for the samples ``start .. n_cut - 1`` only (a prefix, or a window of the walk): one product against the
    inverse DCT-II basis evaluated there, instead of the whole series (dmrai-lab/dmipy-sim#449 item 3).
    """
    from scipy.fft import idct
    C, names = susc_path_coeffs(arrays, meta)
    if n_w is not None:
        C = C[:int(n_w)]
    n_t = int(meta["n_t"])
    stop = n_t if n_cut is None else min(int(n_cut), n_t)
    if int(start) > 0 or stop < n_t:
        K = C.shape[2]
        k = np.arange(K)[:, None]; n = np.arange(int(start), stop)[None, :]
        D = np.sqrt(2.0 / n_t) * np.cos(np.pi * k * (2 * n + 1) / (2.0 * n_t)); D[0] = np.sqrt(1.0 / n_t)   # (K, saves)
        return C @ D, names
    b = idct(C, type=2, norm="ortho", axis=2, n=n_t) if C.shape[2] == n_t else \
        idct(np.pad(C, ((0, 0), (0, 0), (0, n_t - C.shape[2]))), type=2, norm="ortho", axis=2)
    return b, names


def susc_path_field(b, b0_dir, *, B0, chi_iso, chi_aniso=0.0, has_aniso=False):
    """``dB(t)`` per walker ``(n_w, n_t)`` from the decoded path channels ``(n_w, n_ch, n_t)`` for one
    ``(B0, direction, chi)``: :func:`~dmipy_sim.fields.hollow_cylinder.contract` along the channel axis."""
    from ..fields.hollow_cylinder import contract
    aniso = chi_aniso if (has_aniso and chi_aniso and b.shape[1] >= 13) else 0.0
    return contract(b, b0_dir, B0=B0, chi_iso=chi_iso, chi_aniso=aniso, axis=1)


_PATH_CHANNELS = ("iso_local", "iso_P_xx", "iso_P_yy", "iso_P_zz", "iso_P_xy", "iso_P_xz", "iso_P_yz",
                  "aniso_G_xx", "aniso_G_yy", "aniso_G_zz", "aniso_G_xy", "aniso_G_xz", "aniso_G_yz")


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


def path_field_integral(arrays, meta, waveform, n_t, dt, *, t0=None, n_w=None):
    """The gated path integral of every field channel per walker, ``(Psi (n_w, n_ch), names)``: ``Psi = gamma dt_f
    sum_t gate(t) c_w(t)``, read from the channel's DCT coefficients against the DCT of the field gate on the
    channel's own grid, for the window starting at ``t0`` on the waveform clock. The names are in the canonical
    order (``iso_local``, the six ``iso_P``, then the six ``aniso_G`` when stored), which is what the contraction
    :func:`~dmipy_sim.fields.hollow_cylinder.field_terms` reads."""
    from scipy.fft import dct
    from ..constants import GAMMA
    from ._replay_kernel import field_gate
    from .replay import _path_grid
    Cs, names = susc_path_coeffs(arrays, meta)
    if tuple(names) != _PATH_CHANNELS[:len(names)]:
        raise ValueError(f"the path channels are stored as {names}, not in the canonical order {_PATH_CHANNELS[:len(names)]}")
    if n_w is not None:
        Cs = Cs[:n_w]
    n_tf, dt_f = _path_grid(meta, n_t, dt)
    gate_hat = dct(field_gate(waveform, n_tf, dt_f, t0=t0), type=2, norm="ortho")[:Cs.shape[2]]
    return (GAMMA * dt_f) * np.einsum("k,wck->wc", gate_hat, Cs), names


# --------------------------------------------------------------- susceptibility replay (consume)
# --------------------------------------------------------------- pack generation
#: per-channel numbers a codec MEASURES on the walk it encoded (not parameters): two shards of one fill differ in them
_MEASURED_CHANNEL_KEYS = ("trace_residual", "band")


def _codec_signature(comp):
    """The codec parameters of a pack's ``compression`` meta, without what is measured per pack (the precision tiers,
    a channel's trace residual): what two shards of one fill must agree on."""
    sig = {k: v for k, v in comp.items() if k != "precision_tiers"}
    if "channels" in sig:
        sig["channels"] = {c: ({k: v for k, v in m.items() if k not in _MEASURED_CHANNEL_KEYS} if isinstance(m, dict) else m)
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
    """The walk parameters that shards of one walk share: everything but the walker count and the seeds, the
    segment table's walks compared without theirs."""
    out = {k: v for k, v in wp.items() if k not in ("n_walkers", "seed")}
    seg = out.get("segments")
    if isinstance(seg, dict):
        out["segments"] = dict(seg, walks=[{k: v for k, v in w.items() if k != "seed"} for w in seg.get("walks", [])])
    return out


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
    declared unshuffled (the walkers are ordered by shard). Provenance lists the shards. ``packs`` are
    :class:`ReplayPack` objects or paths."""
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
        same("replay_envelope", lambda pk: pk.meta.get("replay_envelope"))
        pv0 = same("per-voxel grid", lambda pk: ((pk.meta.get("fidelity") or {}).get("per_voxel") or {}).get("grid"))
        same("array names", lambda pk: sorted(pk.arrays))
        n = [int(pk.meta["walk_params"]["n_walkers"]) for pk in pks]
        arrays = {}
        scale_keys = [k for k in pks[0].arrays if k.endswith("_band_scale") or k.split("/")[-1] == "susc_path_scale"]   # every segment's
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
            if k in ("voxel_ijk", "voxel_certificate", "band_block") or k in scale_keys:
                continue
            if k.startswith("susc_grid_"):                                    # the substrate's field grid: one table, every shard's
                for pk in pks[1:]:
                    if not np.array_equal(np.asarray(pk.arrays[k]), np.asarray(pks[0].arrays[k])):
                        raise ValueError(f"the shards differ in the field grid {k!r}")
                arrays[k] = np.asarray(pks[0].arrays[k])
                continue
            parts = [np.asarray(pk.arrays[k]) for pk in pks]
            base = k.split("/")[-1]
            if base.endswith(("_rle_vals", "_rle_lens")):                      # a stream (RPK.md 9.4 rule 7): the shards'
                arrays[k] = np.concatenate(parts)                             # records follow each other in walker order
                continue
            if not all(a.shape[0] == m and a.shape[1:] == parts[0].shape[1:] for a, m in zip(parts, n)):
                raise ValueError(f"array {k!r} is not walker-leading in every shard; it cannot be concatenated")
            arrays[k] = np.concatenate(parts)
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
                ch = (comp_meta_ch := (pks[0].meta["compression"].get("channels") or {})).get("compartment")
                comp = None
                if ch is not None:
                    if ch.get("columns") and all(c.get("kind") == "static" for c in ch["columns"]) and "comp_static" in arrays:
                        comp = np.asarray(arrays["comp_static"])[:, None]
                    else:
                        comp = np.asarray(_cx.decode_occupancy(arrays, ch)["comp"])
                C = _cx.read_position_coeffs(arrays, dtype=np.float64)
                _w = None
                if "spin_weights" in arrays:                                  # the union's weights, not the shards'
                    _vox, _in = grid.bin(C[:, 0, :])
                    _key = np.where(_in, np.ravel_multi_index(tuple(np.clip(_vox, 0, np.asarray(grid.shape) - 1).T), tuple(grid.shape)), -1)
                    _pid = np.asarray(comp)[:, 0].astype(np.int64) if comp is not None else np.zeros(len(_key), np.int64)
                    _w = union_weights(np.asarray(arrays["spin_weights"], np.float64), np.repeat(np.arange(len(pks)), n), _key, _pid)
                    arrays["spin_weights"] = _w.astype(np.float32)
                _ijk, _pools_r, _n, _floor, _err = voxel_floor_coded(C, float(wp["dt_traj"]), int(comp_meta_ch and pks[0].meta["walk_params"]["n_t"]),
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
            _n, _floor, _err = c[:, :, 0], c[:, :, 1], c[:, :, 2]; _ok = np.isfinite(_floor)
            fid["per_voxel"] = dict(grid=pv0, pools=pools, n_voxels=int(len(rows)),
                                    walkers_min=int(_n[_n > 0].min()) if (_n > 0).any() else 0,
                                    floor_max=float(np.nanmax(_floor)) if _ok.any() else None,
                                    floor_median=float(np.nanmedian(_floor)) if _ok.any() else None,
                                    err_max=float(np.nanmax(_err)) if np.isfinite(_err).any() else None,
                                    within_2x_floor_fraction=(float(np.mean(_err[_ok] <= 2.0 * _floor[_ok])) if _ok.any() else None),
                                    thin_voxels=int(((_n > 0) & (_n < 2)).sum()), shards=len(pks),
                                    recertified=bool(overlap == "recertify"))
        n_all = int(sum(n))
        comp_meta = dict(pks[0].meta["compression"])
        if "channels" in comp_meta:                                      # the measured numbers: the worst over the shards
            chans = {c: (dict(m) if isinstance(m, dict) else m) for c, m in comp_meta["channels"].items()}
            for c, m in chans.items():
                if isinstance(m, dict):
                    for k in _MEASURED_CHANNEL_KEYS:
                        vals = [((pk.meta["compression"].get("channels") or {}).get(c) or {}).get(k) for pk in pks]
                        if all(v is not None for v in vals):
                            m[k] = float(max(vals))
            comp_meta["channels"] = chans
        if comp_meta.get("walker_preserving"):
            comp_meta["precision_tiers"] = _precision_tiers(arrays, n_all, float(fid.get("floor_max") or 0.0), False)
        meta = dict(pks[0].meta)
        seeds = [pk.meta["walk_params"]["seed"] for pk in pks]
        seg0 = pks[0].meta["walk_params"].get("segments") or {}
        meta.update(id=id, compression=comp_meta, fidelity=fid,
                    walk_params=dict(pks[0].meta["walk_params"], n_walkers=n_all, seed=seeds,
                                     segments=dict(seg0, walks=[dict(w, seed=seeds) for w in seg0.get("walks", [])])),
                    provenance=dict(pks[0].meta.get("provenance") or {}, shards=[dict(id=pk.meta.get("id"), n_walkers=int(m)) for pk, m in zip(pks, n)]))
        out = ReplayPack(arrays, meta)
        out.susceptibility_field                                         # the merged pack holds the equivalence
        if out_path is not None:
            write_rpk(out_path, arrays, meta)
            run.artifact(out_path)
        return out


SEGMENT_T = 0.1          #: the storage rule's window (s): a pack stores its walk in windows of this duration (RPK.md 4.3)
STORAGE_BAND_MARGIN = 1.4  #: the storage rule's margin over a class's certified band (the paper's K = 256 per 100 ms for the human classes)


def _walk_scanner(walk):
    """The scanner class a walk's save grid was derived for (``PersistentWalk.scanner``, set by ``walk_spec``), or None."""
    return getattr(walk, "scanner", None) if not isinstance(walk, dict) else walk.get("scanner")


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


def segment_plan(n_t, dt, segment_T):
    """``(n_segments, n_t per segment)`` for a walk of ``n_t`` saves at ``dt`` stored in windows of ``segment_T``
    seconds (RPK.md 4.3): one window when the walk is within it, else the walk must be a whole number of windows
    on its save grid, and is refused otherwise. A pack declares its segment length; there is no other layout."""
    n_t = int(n_t); dt = float(dt)
    if segment_T is None or not float(segment_T) > 0.0:
        raise ValueError("segment_T is the duration of the windows a pack stores its walk in (RPK.md 4.3), the storage "
                         f"rule's {SEGMENT_T:g} s by default; got {segment_T!r}")
    T = (n_t - 1) * dt
    if T <= float(segment_T) * (1.0 + 1e-9):
        return 1, n_t
    steps = int(round(float(segment_T) / dt))
    if steps < 2 or abs(steps * dt - float(segment_T)) > 1e-9 * float(segment_T):
        raise ValueError(f"a segment of {float(segment_T):.6g} s is not a whole number of saves at dt = {dt:.6g} s; "
                         "walk on a save grid that divides the segment")
    if (n_t - 1) % steps:
        raise ValueError(f"the walk's {T:.6g} s is not a whole number of {float(segment_T):.6g} s segments on its save "
                         f"grid ({n_t - 1} steps, {steps} per segment); walk a whole number of segments, or pass the "
                         "segment_T that divides it")
    return (n_t - 1) // steps, steps + 1


def _window_master(m, k0, k1):
    """The master arrays of the window of saves ``[k0, k1]`` as a fresh walk of that window would have recorded
    them: positions and tracks sliced, the contact of the window's first save 0 (it ends no step of the window),
    the field samples the window's on the field's own grid, everything else the walk's."""
    w = dict(m)
    w["traj"] = np.asarray(m["traj"])[:, k0:k1 + 1]
    w["T_max"] = float(k1 - k0) * float(m["dt_traj"])
    for key in ("comp", "bfrac"):
        a = m.get(key)
        if a is not None and np.asarray(a).ndim == 2:
            w[key] = np.asarray(a)[:, k0:k1 + 1]
    if m.get("dlog_b") is not None:
        d = np.array(np.asarray(m["dlog_b"])[:, k0:k1 + 1], copy=True)
        if k0 > 0:
            d[:, 0] = 0.0
        w["dlog_b"] = d
    if m.get("susc_field_samples") is not None:
        every = int(m.get("susc_field_every", 1) or 1)
        if k0 % every or k1 % every:
            raise ValueError(f"the field was sampled every {every} saves and a segment boundary at save {k0 if k0 % every else k1} "
                             "is not on that grid; walk with a segment that is a whole number of field samples")
        w["susc_field_samples"] = np.asarray(m["susc_field_samples"])[:, k0 // every:k1 // every + 1]
    return w


def combine_segment_fidelity(fids):
    """The whole walk's certificate as the BOUND over its segments' (RPK.md 4.3): every error term the sum over
    the windows (a replay across them sums their contributions, which may add coherently), every floor the largest,
    and ``certified = "bounded"``. What a prefix of whole windows or an appended continuation carries; a walk built
    in one go measures the positions battery over the whole instead."""
    fids = [dict(f) for f in fids]
    out = dict(metric=fids[0].get("metric"), certified="bounded", segments=fids)
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
    if "err_max" in out and "floor_max" in out:
        out["within_2x_floor"] = bool(out["err_max"] <= 2.0 * out["floor_max"])
    return out


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
    the per-segment list) rounded to ``sig`` significant digits before they are written.

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
        a = np.asarray(v)
        if a.ndim >= 1 and a.shape[0] == n_walkers:
            per_walker += int(a.nbytes // max(n_walkers, 1))
        else:
            # Not walker-leading -- e.g. a run-length channel whose runs vary per walker. Row n of such
            # an array is not walker n, so it cannot be prefix-sliced and must be read whole. Counting
            # it as fixed overhead keeps a small tier's quoted size honest instead of understating it.
            fixed += int(a.nbytes); whole.append(k)
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


def _select_boundary_codec(m, dlog, env, tol, dtype, verbose=False, container=None, *, min_K=None, segment_T=None):
    """Choose the C2 (boundary-local-time) codec by COST subject to the surface-fidelity gate.

    The bridge form (:func:`compression.encode_boundary_bridge`) is flat in ``n_t`` and, at K=32 in
    f16, lands thousands of times below the surface split-half floor; exact sparse CSR costs ~one
    entry per wall contact instead, so it scales with walk length (574 B/walker at n_t=1601 for an
    axon, and worse the longer the walk). Cost each bridge candidate, keep the cheapest that passes
    the gate, and fall back to exact sparse if none do.

    ``min_K`` raises the ladder's smallest rung (16 bands per storage-rule window when this build is a
    walk built as one window but longer than the storage rule's window, RPK.md 4.3 -- the measured
    table is in the PR, not here) instead of the plain ladder's 8; it still doubles from there, same as
    always, until the certificate passes. ``segment_T`` adds :func:`_surface_fidelity`'s window term
    (``err_window``/``floor_window``): a candidate passes only when both the whole-duration term and
    the window term are within ``tol`` of their floors (dmipy-sim#528)."""
    base = 8 if not min_K else int(min_K)
    cands = []
    for K in (base, base * 2, base * 4, base * 8):
        a, mm = _cx.encode_boundary_bridge(dlog, K=int(K), dtype=dtype, container=container)
        cf = _surface_fidelity(m, a, mm, env, segment_T=segment_T)
        nb = sum(int(np.asarray(v).nbytes) for v in a.values()) / max(len(dlog), 1)
        cands.append((nb, K, a, mm, cf))
        ok = cf is not None and cf["err"] <= tol * cf["floor"]
        if ok and "err_window" in cf:
            ok = cf["err_window"] <= tol * cf["floor_window"]
        if ok:
            if verbose:
                log.info(f"[bank] C2 codec: boundary_bridge K={K} {np.dtype(dtype).name} "
                      f"({nb:.0f} B/walker, err={cf['err']:.2e} vs floor {cf['floor']:.2e}"
                      + ("" if "err_window" not in cf else
                         f", window err={cf['err_window']:.2e} vs floor {cf['floor_window']:.2e}") + ")")
            return a, mm
    a, mm = _cx.encode_boundary_local_time(dlog)
    if verbose:
        nb = sum(int(np.asarray(v).nbytes) for v in a.values()) / max(len(dlog), 1)
        best = min(cands, key=lambda c: (c[4] or {}).get("err", np.inf))
        log.info(f"[bank] C2 codec: no bridge K passed the surface gate (best K={best[1]} "
              f"err={(best[4] or {}).get('err', float('nan')):.2e}); using exact sparse "
              f"({nb:.0f} B/walker)")
    return a, mm


def preflight_master(m, *, susc_path_K=None, sigma_star=None, K=None):
    """Check a master walk against the tiers its content will assemble. Returns a list of
    problems, empty if it will work.

    Every one of these is knowable from metadata, yet each cost a full walk to discover:

    * the CACTUS master emits the per-walker ``susc_basis`` unless the producer asks for
      ``field_store='grid'``; ``build_replay_pack`` then refuses the master outright -- AFTER the
      walk, because the walk happens inside ``_master_arrays``.
    * a master carrying ``susc_field_basis=None`` silently yields a pack with no field tier, so
      the pack looks fine and is missing a capability. Presence of the KEY is not enough; the
      value is what matters.
    * an auto-selected ``K`` is sized against this walk's own floor. For a pack that will later be
      merged with others, the floor falls as 1/sqrt(N) while codec error does not, so a K that
      passes here can make compression the dominant error in the union.

    Call it before walking where possible, or at least before encoding.
    """
    bad = []
    if m.get("PhiC") is not None or m.get("susc_basis") is not None:
        bad.append("master carries the private susceptibility forms (susc_basis/PhiC), which are "
                   "refused here; rebuild the master with field_store='grid', or drop them and "
                   "rely on susc_field_basis")
    if susc_path_K and m.get("susc_field_basis") is None and m.get("susc_field_sampler") is None:
        bad.append("susc_path_K was requested but the master carries no field basis (grid or sampler), so the field tier (C3) "
                   "cannot be assembled and the pack would be silently missing it "
                   "(field_store='grid' is what produces it)")
    if sigma_star is not None and K is None:
        bad.append(f"sigma_star={sigma_star:g} with K unpinned: a pack built to an absolute target states its band "
                   f"(K=, temporal_bandwidth_hz= or scanner=) so that every pack of the target shares it. Pin K.")
    return bad



def _walk_master(walk, *, weights=None, diffusivity=None, substrate_frame=None):
    """The bank's master dict from a PersistentWalk plus the substrate metadata; a bank dict / .npz passes
    through unchanged (it carries any field tier as its own keys already).

    The field tier is the walk's, not a build-time choice: ``walk.field_basis`` (built by
    :func:`~dmipy_sim.spec.walk.walk_spec` whenever the spec declares a susceptibility source) travels
    whenever it is set, and the channel it becomes needs ``walk.field_samples`` -- sampled in the walk
    (a gridded source by every walk, a strand source with ``adaptive_steps=True``) or read afterwards (:func:`~dmipy_sim.spec.walk.fill_field`). A spec that
    declares a source but whose walk carries neither is refused here BY NAME: the ambiguity ("was the field
    meant to be in this pack or not") is exactly the defect this refuses, rather than silently guessing
    (the grid :func:`~dmipy_sim.fields.susceptibility_field.field_grid_of` would derive from the geometry's
    type, dmipy-sim#539)."""
    from ..persistent_walk import PersistentWalk
    from ..fields.susceptibility_field import FieldGrid
    if not isinstance(walk, PersistentWalk):
        if any(v is not None for v in (weights, diffusivity, substrate_frame)):
            raise TypeError("weights=, diffusivity= and substrate_frame= go with a PersistentWalk; a master "
                            "dict carries them as its own keys")
        return walk
    geometry = walk.geometry
    spec = walk.spec if walk.spec is not None else (getattr(geometry, "spec", None) if geometry is not None else None)
    if weights is None and walk.weights is None and spec is not None and walk.compartment is not None:
        wf = [p.water_fraction for p in sorted(spec.pools, key=lambda p: p.id)]
        pool0 = np.asarray(walk.compartment)[:, 0].astype(int)
        dry = [(p, int(np.sum(pool0 == p.id))) for p in spec.pools if p.water_fraction == 0.0 and np.any(pool0 == p.id)]
        if dry:
            raise ValueError("the walk has walkers in a pool the spec says holds no water: "
                             + ", ".join(f"{n:,} in pool {p.id} ({p.name})" for p, n in dry)
                             + "; such walkers would weigh 0 and the pack would silently omit them. Seed the pools the "
                               "spec gives water to, give the pool its water in the spec, or pass weights= explicitly")
        if any(f != 1.0 for f in wf):                     # the seeding rule's weights, from the spec
            weights = np.asarray(wf, float)[pool0]
    field = walk.field_basis
    if (spec is not None and susceptibility_field_of(spec) == "present" and walk.field_samples is None
            and (walk.field_deferred or field is None)):
        if walk.field_deferred:
            raise ValueError(f"this walk's spec {spec.id!r} declares a susceptibility source and recorded a "
                             "deferred field tier, but the walk carries no field samples yet; fill it first "
                             "with fill_field(walk, basis)")
        raise ValueError(f"this walk's spec {spec.id!r} declares a susceptibility source but the walk carries "
                         "no field samples and recorded no deferral; sample the field in the walk, or defer it "
                         "and fill it with fill_field")
    if weights is None and walk.weights is not None:
        weights = walk.weights
    extra = {}
    if spec is not None:
        extra["substrate"] = spec.to_dict()
        if substrate_frame is None:                       # the pack declares what its spec declares (RPK.md 4.2)
            substrate_frame = frame_of_spec(spec)
    if weights is not None:
        w = np.asarray(weights, float).reshape(-1)
        if w.shape[0] != walk.n_walkers:
            raise ValueError(f"weights has {w.shape[0]} entries for {walk.n_walkers} walkers")
        extra["w"] = w
    if field is not None:
        from ..fields.strand_field import StrandFieldBasis, StrandFieldRecord
        if isinstance(field, FieldGrid):
            extra.update(susc_field_basis=field.basis, susc_grid_origin=np.asarray(field.origin, float),
                         susc_grid_raster=getattr(field, "certificate", None), susc_grid_periodic=tuple(field.periodic))
        elif isinstance(field, (StrandFieldBasis, StrandFieldRecord)):
            if isinstance(field, StrandFieldRecord) and walk.field_samples is None:
                raise ValueError("the walk carries the record of its field basis but no field samples: rebuild the basis from "
                                 "the spec (walk_spec) to sample the field, or fill_field(walk, basis)")
            extra["susc_field_sampler"] = field
        else:
            raise TypeError("a walk's field_basis must be a fields.susceptibility_field.FieldGrid (basis, origin) or a "
                            f"fields.strand_field.StrandFieldBasis, got {type(field).__name__}")
        if walk.field_samples is not None:                       # sampled in the walk, or read back by fill_field
            extra["susc_field_samples"] = np.asarray(walk.field_samples, np.float32)
            extra["susc_field_every"] = int(getattr(walk, "field_sample_every", 1) or 1)
            if walk.field_fill is not None:
                extra["susc_field_fill"] = walk.field_fill
    if diffusivity is not None:
        extra["D_intra"] = float(diffusivity)
    if substrate_frame is not None:
        extra["substrate_frame"] = np.asarray(substrate_frame, float)
    extra["walkers_shuffled"] = True        # the producer draws walkers i.i.d.: any prefix is a fair subsample
    return walk._bank_dict(**extra)

def _container(spec):
    """A band container from the builder's knob: ``None`` (the float container), ``"bands"`` (the registry's
    default, :data:`compression.BAND_CONTAINER`) or an explicit ``((upto, bits), ...)``."""
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec != "bands":
            raise ValueError(f"container must be None, 'bands' or ((upto, bits), ...); got {spec!r}")
        return _cx.BAND_CONTAINER
    return tuple((None if u is None else int(u), int(b)) for u, b in spec)


def build_replay_pack(walk, *, id, license, citation, weights=None,
                      method=_cx.POSITION_METHOD, envelope=None, tol=2.0, K=None, temporal_bandwidth_hz=None, scanner=None,
                      sigma_star=None, provenance=None,
                      blt_temporal_K=None, blt_dtype=np.float16, susc_path_K="auto", susc_path_bits=8, voxel_grid=None,
                      position_container=None, blt_container=None,
                      diffusivity=None, substrate_frame=None, out_path=None, verbose=False,
                      fidelity="measured", fidelity_from=None, device="auto", segment_T=SEGMENT_T, _occupancy_runs=False,
                      _window_of_plan=False):
    """Compress a persistent walk and assemble a self-certifying replay pack.

    The walk is stored in SEGMENTS of ``segment_T`` seconds (RPK.md 4.3; the storage rule's 100 ms): a walk within
    one is one segment, a longer one a whole number of them on its save grid (refused otherwise), each window
    encoded with the same codec at the same ``K`` and certified on its own, the whole certified over the full
    walk; segment 0's tensors under the channel names, segment ``i`` under ``s{i}/``. ``K`` and ``susc_path_K``
    are per segment. A prefix of whole segments is then a range of the file, and a walk is continued by
    appending segments (:meth:`~dmipy_sim.replay.replay.ReplayPack.truncate`, :func:`continue_walk`).

    **The band is the scanner class's** (#643): ``K = ceil(2 f T)`` per window with ``f`` =
    :func:`storage_band_hz` of ``scanner`` -- the class's certified band (the catalogue's ``band.certified_band_hz``,
    the replayable-MC paper's closed-form certificate: the highest frequency any waveform the class can deliver
    needs) times :data:`STORAGE_BAND_MARGIN`. ``scanner`` defaults to the class the walk's save grid was derived for
    (``PersistentWalk.scanner``, which :func:`~dmipy_sim.spec.walk.walk_spec` sets); a walk with no class is refused
    by name unless ``K`` or ``temporal_bandwidth_hz`` states the band. Nothing is searched on the walk: the
    certificate measures whether the band holds, it does not choose it. The pack records the class under
    ``compression.scanner`` beside ``temporal_bandwidth_hz``.

    ``fidelity`` is what this pack certifies (RPK.md 9.4 rule 4): ``"measured"`` replays the envelope's battery
    on the raw and the decoded walk and reports the codec error against the split-half floor, per tier; a pack
    that is one block of a fill -- the same substrate, walk parameters, save grid, codec and containers as a
    pack already measured, other walkers -- passes ``fidelity="inherited"`` with ``fidelity_from=`` that
    certifying pack (or its meta): the codec error and the per-tier terms are the certifying pack's, this pack
    reads its OWN split-half floor (whole and per voxel) from its stored coefficients over the same battery,
    no path is decoded, and a codec parameter that differs from the cited pack is refused. ``device`` runs the
    band transforms and the coded floors on the JAX device (``"auto"``: when it is a GPU).

    ``walk`` is the :class:`~dmipy_sim.persistent_walk.PersistentWalk` a producer returned (the
    walk's bank dict / ``.npz`` is also accepted). The tiers assembled are the ones the
    walk CARRIES, with what they need read from the geometry the walk was run on
    (``walk.geometry`` / ``walk.spec``): **gradient** (C0, always); **bulk relaxation** (C1) when the walk
    has a compartment channel (the pools' T2 / T1 are replay knobs; the pack carries none); **surface relaxivity**
    (C2) when the walk has the boundary local time; **magnetization transfer** (C4) when it has the
    bound fraction; **field** (C3) when the walk carries a field basis (``walk.field_basis``, built by
    :func:`~dmipy_sim.spec.walk.walk_spec` whenever its spec declares a susceptibility source, or attached
    directly) WITH field samples (``walk.field_samples``, sampled in the walk at the sub-step -- a grid by every
    walk, a strand basis with ``adaptive_steps=True`` -- or read afterwards by :func:`~dmipy_sim.spec.walk.fill_field`): a
    :class:`~dmipy_sim.fields.susceptibility_field.FieldGrid` basis stores its grid (a mesh or myelinated
    substrate), a :class:`~dmipy_sim.fields.strand_field.StrandFieldBasis` the per-segment closed form of a
    strand substrate (path channel only: it has no grid); B0, its direction and the susceptibilities are
    replay knobs, never stored. A walk whose spec declares a source but carries neither field samples nor a
    recorded deferral is refused BY NAME (:func:`_walk_master`): sample the field in the walk, or
    ``walk_spec(..., defer_field=True)`` and ``fill_field`` it before packing. A spec without a source builds
    a pack without the tier. Every pack embeds the substrate spec its walk was driven by, and a walk without one is
    refused; the spec's pools declare the field, and the pack holds the equivalence
    (:func:`~dmipy_sim.spec.substrate.susceptibility_field_of`: some pool magnetic <=> ``"field"`` in its tiers
    <=> the pack stores C3), checked on the walk before anything is encoded and on the channels built.
    ``susc_path_K`` is the field tier's own band: ``"auto"`` (the default) --
    the band and the container (``susc_path_bits``) derived on this walk as the cheapest pair whose codec
    error on the certificate's battery is within its floor (:func:`derive_susc_path_K`), certified against the
    walk's own samples; the grid route instead (the grid read at the decoded positions) when the positions are
    lossless and the samples are the grid read at the saved positions (:func:`~dmipy_sim.spec.walk.fill_field`),
    since that route then reproduces them exactly at no channel's cost -- a number, or ``None`` for the grid
    alone. ``weights`` are per-walker proton-density weights (default: the pools' water fractions
    by compartment, else uniform).

    The position ensemble is compressed by ``method`` (default ``bridge_dst``: endpoints plus a
    Brownian bridge on the sine basis, which holds both endpoints exactly and pairs its first two
    coefficients with the gradient moments -- see :func:`compression.encode_bridge_dst`); ``K``
    (mode count) is chosen automatically to keep the *measured* replay error within ``tol`` x the
    Monte-Carlo floor over ``envelope`` (default :func:`compression.default_envelope`) unless given.
    ``voxel_grid`` (a :class:`~dmipy_sim.phantom.Grid`, substrate-attached) adds the PER-VOXEL certificate a
    partition needs: walkers binned by where they started and by pool, and per (voxel, pool) the split-half
    floor and the codec error over the acquisition battery, stored as ``voxel_ijk`` / ``voxel_certificate``
    with a summary in ``fidelity["per_voxel"]`` (:func:`voxel_fidelity`; :func:`voxel_fidelity_volumes` reads
    it back, :func:`dmipy_sim.spec.seeding.plan_seeding` turns a pilot's into the next walk's counts).

    A contact (C2) channel chosen automatically (``blt_temporal_K`` not given) is banded by
    :func:`_select_boundary_codec`: for a walk built as ONE window but longer than the storage rule's
    window (``segment_T``, RPK.md 4.3), the ladder's floor is 16 bands per storage-rule window, so that a
    later cut of this pack into windows (:meth:`~dmipy_sim.replay.replay.ReplayPack.resegment`) keeps the
    stored contact within the window's own floor rather than only the whole walk's (dmipy-sim#528) -- a
    window built directly as one segment of an already-decided plan (``_build_segmented``, ``resegment``,
    ``prefix``, a continuation) keeps its own endpoints and the plain ladder, since it IS the window, not
    something that will be cut again.

    Returns a :class:`dmipy_sim.replay.replay.ReplayPack`; writes it to ``out_path`` if given.
    """
    _work = getattr(walk, "work", None)
    if _work and _work.get("exhausted_steps"):
        raise ValueError(
            f"{_work['exhausted_steps']} walker-steps of this walk exhausted its bounce budget of {_work['bounce_budget']}: "
            f"part of their paths went untested, so the walk is not the record of the physics it claims. Walk it again "
            f"with a larger bounce_budget= (certify_bounce_budget sizes one from a pilot) and build the pack from that.")
    with Run("build_replay_pack", params=dict(id=id, K=K, fidelity=fidelity, device=device, out_path=out_path)) as run:
        src = _walk_master(walk, weights=weights, diffusivity=diffusivity, substrate_frame=substrate_frame)
        _cx.require_position_method(method)
        m = _master_arrays(src)
        # a window of an already-planned pack may carry its field channel beside the walk (the parent's grid)
        susceptibility_field_of(m.get("substrate"), name=id,
                                stores_field=(None if _window_of_plan and not _stores_field(m) else _stores_field(m)))
        n_segments, n_seg = segment_plan(m["traj"].shape[1], m["dt_traj"], segment_T)
        if n_segments > 1:
            kw = dict(id=id, license=license, citation=citation, method=method, envelope=envelope, tol=tol, K=K,
                      temporal_bandwidth_hz=temporal_bandwidth_hz, scanner=scanner, sigma_star=sigma_star,
                      provenance=provenance, blt_temporal_K=blt_temporal_K, blt_dtype=blt_dtype, susc_path_K=susc_path_K,
                      susc_path_bits=susc_path_bits, voxel_grid=voxel_grid, position_container=position_container,
                      blt_container=blt_container, verbose=verbose, fidelity=fidelity, fidelity_from=fidelity_from, device=device)
            return _build_segmented(m, n_segments, n_seg, run, walk, out_path, **kw)
        if m.get("substrate_frame") is not None:
            sub = m.get("substrate") or {}
            if (sub.get("frame") or {}).get("source") == "structural":
                # a frame spec_of derived from the substrate's own shape: checked against that shape, cheap and
                # walk-free, before anything about the walk itself is touched (#538)
                check_frame_against_geometry(sub, m["substrate_frame"])
            else:
                # a hand-declared frame the walk contradicts is refused (RPK.md 4.2, #194)
                bundles = (sub.get("realisation") or {}).get("bundles") if isinstance(sub, dict) else None
                check_frame_against_walk(m["traj"], m["substrate_frame"], w=m.get("w"),
                                         bundle_axes=(None if not bundles else [b["axis"] for b in bundles]))
        env = envelope or _cx.default_envelope()
        if fidelity not in ("measured", "inherited"):
            raise ValueError("fidelity is 'measured' (the battery on this walk) or 'inherited' (a block of a fill citing its certifying pack)")
        cert = None
        if fidelity == "inherited":
            if fidelity_from is None:
                raise ValueError("fidelity='inherited' needs fidelity_from=: the certifying pack of the fill, or its meta")
            cert = fidelity_from.meta if hasattr(fidelity_from, "meta") else dict(fidelity_from)
            cc, cf = cert["compression"], cert["fidelity"]
            if cc.get("method") != method:
                raise ValueError(f"the certifying pack stores positions by {cc.get('method')!r}, this build by {method!r}")
            if K is None and temporal_bandwidth_hz is None:
                K = int(cc["K"])
            if cf.get("certified", "measured") != "measured":
                raise ValueError("a certifying pack carries a measured fidelity; a pack that inherited one cannot certify another")
        elif fidelity_from is not None:
            raise ValueError("fidelity_from= goes with fidelity='inherited'")
        X = m["traj"] if _cx.is_lazy(m["traj"]) else np.asarray(m["traj"])   # as stored, or lazily: read per walker chunk
        dt = float(m["dt_traj"])
        # a window built directly (one segment of a plan _build_segmented/resegment/prefix/a continuation already
        # decided) carries its own endpoints and needs none of this; only a walk built as ONE window that is itself
        # longer than the storage rule's window risks a later cut it cannot see (#528)
        _c2_window_T = None if _window_of_plan else SEGMENT_T
        _c2_min_K = None
        if _c2_window_T is not None and (X.shape[1] - 1) * dt > float(_c2_window_T) * (1.0 + 1e-9):
            _c2_min_K = 16 * int(np.ceil((X.shape[1] - 1) * dt / float(_c2_window_T)))
        # The band (#643): a stated K, else a stated frequency (#199: K bands over T resolve up to K / (2T)), else the
        # scanner class the pack is stored for -- the one given, else the one the walk's save grid was derived for.
        # Nothing is searched: the band is the class's, and the certificate measures whether it holds.
        band_class = None
        if K is None and temporal_bandwidth_hz is None and cert is None:
            band_class = scanner if scanner is not None else _walk_scanner(walk)
            if band_class is None:
                raise ValueError("build_replay_pack needs the band to store the positions at: pass scanner= (the class the pack is "
                                 "for -- a walk from walk_spec carries the one its save grid was derived for), K= or "
                                 "temporal_bandwidth_hz=. The band is never searched on the walk (#643)")
            temporal_bandwidth_hz = storage_band_hz(band_class)
        if K is None and temporal_bandwidth_hz is not None:
            K = mode_count((X.shape[1] - 1) * dt, temporal_bandwidth_hz=temporal_bandwidth_hz)
        if cert is not None:                              # the codec error is the certifying pack's; the floor is this walk's
            pos_arrays, pos_meta, _ = _cx.encode(X, method, K, container=_container(position_container), device=device)
            cc, cf = cert["compression"], cert["fidelity"]
            same = dict(K=(int(pos_meta.get("K", K)), int(cc["K"])), n_t=(int(X.shape[1]), int(cc["n_t"])),
                        container=(pos_meta.get("container"), cc.get("container")),
                        dt_traj=(dt, float(cert["walk_params"]["dt_traj"])))
            for name, (mine, theirs) in same.items():
                if (abs(mine - theirs) > 1e-12 * abs(theirs) if name == "dt_traj" else mine != theirs):
                    raise ValueError(f"this build's {name} is {mine!r}, the certifying pack's {theirs!r}: a block inherits a "
                                     "certificate only with the codec it was measured for")
            _Cc = _cx.read_position_coeffs(pos_arrays, dtype=np.float64)
            fl = _cx.measure_floor_coded(_Cc, dt, X.shape[1], env, device=device)     # unweighted, as measure_fidelity reads it
            fid = dict(metric=cf["metric"], err_max=float(cf["err_max"]), floor_max=fl["floor_max"], noise_floor=fl["noise_floor"],
                       within_2x_floor=bool(float(cf["err_max"]) <= 2.0 * fl["floor_max"]),
                       per_family={f: dict(err_max=float(cf["per_family"][f]["err_max"]), floor_max=fl["per_family"][f])
                                   for f in fl["per_family"] if f in cf.get("per_family", {})},
                       certified="inherited",
                       inherited_from=dict(id=cert["id"], err_max=float(cf["err_max"]), floor_max=float(cf["floor_max"])))
        else:
            pos_arrays, pos_meta, _ = _cx.encode(X, method, K, container=_container(position_container), device=device)
            run.phase("certificate positions")
            fid = _cx.measure_fidelity(X, dt, _cx.decoder(pos_arrays, pos_meta), env)   # decoded per chunk, never whole
        if cert is None:
            fid["certified"] = "measured"
        if sigma_star is not None:                       # adaptive floor-target policy (build_to_floor)
            fid = dict(fid, target_floor=float(sigma_star),
                       meets_target=bool(fid["err_max"] <= sigma_star and fid["floor_max"] <= sigma_star))

        arrays = dict(pos_arrays)
        chan_meta = {}                                   # per-channel codec params
        channels = {"gradient": True, "susceptibility": False, "T1T2": False, "rho2": False,
                    "mt": (m.get("bfrac") is not None)}
        # STATIC field-grid susceptibility channel: store the geometry-only field-basis grids ONCE
        # (a substrate property); replay assembles the field for any (B0,dir,chi) and samples it along
        # the pos-codec-decoded trajectory (ReplayPack.replay with a field). O(N_vox) not O(N_w*N_t) and SE-exact (a static
        # field at a frozen point cancels under the SE gate to machine precision). f16 grids: O(1) geometry.
        _field = _field_of(m)
        _band_record = None
        if isinstance(susc_path_K, str):
            if susc_path_K != "auto":
                raise ValueError(f"susc_path_K is a band, 'auto' (derived on the walk) or None; got {susc_path_K!r}")
            if _field is None:
                susc_path_K = None
            elif (m.get("susc_field_basis") is not None and _cx.is_lossless_at(method, int(K), int(X.shape[1]))
                  and (m.get("susc_field_samples") is None or m.get("susc_field_fill"))):
                susc_path_K = None                             # the grid at lossless positions IS the samples: exact, no channel
            else:
                run.phase("field band")
                susc_path_K, susc_path_bits, _band_record = derive_susc_path_K(m, _field, env, K_max=int(K))
        if _field is not None and m.get("susc_field_basis") is None:
            # a strand substrate's per-segment field: no grid to store, the path channel is the tier
            if not susc_path_K:
                raise ValueError("a StrandFieldBasis has no grid to store: the field tier (C3) needs susc_path_K")
            if m.get("susc_field_samples") is None:
                raise ValueError("this walk carries a field basis but no field samples: sample the field in the "
                                 "walk (walk_spec), or fill_field(walk, basis) before building the pack")
            chan_meta["susceptibility_grid"] = dict(has_aniso=("aniso_G_xx" in _field.channel_names), arrays_in_pack=False,
                                                    replay_route="path", source=_field.meta)
            channels["susceptibility"] = True
            # the samples are the walk's interval means (sampled in the walk) or fill_field's save-resolution
            # read -- either way the pack's path channel is their DCT, not a fresh sample of the trajectory
            _a, _pm = susc_path_encode_series(np.asarray(m["susc_field_samples"]), list(_field.channel_names), K=int(susc_path_K),
                                              bits=susc_path_bits, layout="wtc", device=device,    # no copy of the samples
                                              dt=float(m["dt_traj"]) * int(m.get("susc_field_every", 1)),
                                              max_refocus_pulses=env.get("max_refocus_pulses"))
            _pm["sampling"] = "read_at_save_resolution" if m.get("susc_field_fill") else "interval_mean_in_walk"
            if m.get("susc_field_fill"):
                _pm["field_fill"] = m["susc_field_fill"]
            if _band_record is not None:
                _pm["band"] = _band_record
            arrays.update(_a); chan_meta["susceptibility_path"] = _pm
        if m.get("susc_field_basis") is not None:
            fb = m["susc_field_basis"]
            # The GRID route samples the field at codec-DECODED positions, so it is only sound when the
            # position codec is lossless. The PATH route stores the walk's own samples (the sub-step interval
            # means taken in the walk, or fill_field's read at the saved positions), which is precisely what
            # frees the positions to be lossy -- so the two cannot both be advertised: shipping grid arrays
            # next to lossy positions would offer a replay route whose accuracy silently depends on a property
            # the pack no longer has. Path wins when present; the grid rides beside it only where it
            # reproduces the samples exactly, at lossless positions and samples read at those positions.
            # A full-rank walker-preserving codec is an exact rewrite; the rank is n_t for
            # n_t-2 for bridge_dst, which stores two endpoints outside the bands.
            _pos_lossless = _cx.is_lossless_at(method, int(K), int(X.shape[1]))
            _grid_in_pack = (not susc_path_K) or (_pos_lossless and m.get("susc_field_fill") is not None)
            if _grid_in_pack:
                arrays["susc_grid_iso_local"] = np.asarray(fb["iso_local"], np.float16)
                arrays["susc_grid_iso_P"] = np.asarray(fb["iso_P"], np.float16)
                if fb.get("aniso_G") is not None:
                    arrays["susc_grid_aniso_G"] = np.asarray(fb["aniso_G"], np.float16)
            chan_meta["susceptibility_grid"] = dict(
                origin=np.asarray(m["susc_grid_origin"], float).tolist(),
                voxel_size=np.asarray(fb["voxel_size"], float).tolist(),
                shape=[int(s) for s in fb["shape"]], has_aniso=(fb.get("aniso_G") is not None),
                periodic=list(grid_periodic_of(m)), arrays_in_pack=bool(_grid_in_pack),
                replay_route=("grid+path" if (_grid_in_pack and susc_path_K)
                              else ("path" if susc_path_K else "grid")),
                raster=m.get("susc_grid_raster"))
            channels["susceptibility"] = True
            # PATH form (C3, preferred): the field sampled along each walker's FULL-RESOLUTION path and
            # compressed in time. Decouples the susceptibility tier from the position codec -- which is
            # what lets the positions go back to K << n_t, since the only reason they had to be stored
            # losslessly was that grid-sampling needed exact r(t). See susc_path_encode for why K is a
            # gate-bandwidth capability rather than a fidelity knob.
            if susc_path_K:
                if m.get("susc_field_samples") is None:
                    raise ValueError("this walk carries a field basis but no field samples: sample the field in "
                                     "the walk (walk_spec), or fill_field(walk, basis) before building the pack")
                _a, _pm = susc_path_encode_series(np.asarray(m["susc_field_samples"]), list(_field.channel_names), K=int(susc_path_K),
                                                  bits=susc_path_bits, layout="wtc", device=device,
                                                  dt=float(m["dt_traj"]) * int(m.get("susc_field_every", 1)),
                                                  max_refocus_pulses=env.get("max_refocus_pulses"))
                _pm["sampling"] = "read_at_save_resolution" if m.get("susc_field_fill") else "interval_mean_in_walk"
                if m.get("susc_field_fill"):
                    _pm["field_fill"] = m["susc_field_fill"]
                if _band_record is not None:
                    _pm["band"] = _band_record
                arrays.update(_a); chan_meta["susceptibility_path"] = _pm
        # C1 (occupancy): the geometric compartment plus, when the walk bound spins, the MT bound
        # pool as a SECOND COLUMN on an independent axis -- not a channel of its own. Replay weights
        # the per-pool rates by occupancy either way; what makes MT a distinct tier is the replay
        # side (vector-Bloch RF, bound-pool knobs, equilibrium start), not the storage.
        if m.get("comp") is not None:
            _cols = {"comp": np.asarray(m["comp"])}
            if m.get("bfrac") is not None:
                _cols["bound"] = np.asarray(m["bfrac"]); channels["mt"] = True
            _a, _cm = _cx.encode_occupancy(_cols, force_runs=_occupancy_runs)
            arrays.update(_a); chan_meta["compartment"] = _cm
            if m.get("w") is not None:
                arrays["spin_weights"] = np.asarray(m["w"], np.float32)
            channels["T1T2"] = True
        elif m.get("bfrac") is not None:
            raise ValueError("an MT (C4) pack carries its bound pool as a C1 occupancy column, so "
                             "it needs the compartment channel too.")
        # dense per-walker physics channels get their own codecs (compression.py):
        # boundary local time -> sparse/dense or the cumulative bridge.
        if m.get("dlog_b") is not None:
            if cert is not None:                                   # the cited pack's channel, parameter for parameter
                _cb = (cert["compression"].get("channels") or {}).get("boundary_local_time")
                if _cb is None:
                    raise ValueError("this walk records wall contact but the certifying pack carries no C2 channel")
                if blt_temporal_K is not None and int(blt_temporal_K) != int(_cb["K"]):
                    raise ValueError(f"blt_temporal_K={blt_temporal_K} but the certifying pack's C2 has K={_cb['K']}")
                _a, _mm = _cx.encode_boundary_bridge(np.asarray(m["dlog_b"]), K=int(_cb["K"]), dtype=blt_dtype,
                                                     container=_container(blt_container), device=device)
                if _mm.get("container") != _cb.get("container") or _mm.get("dtype") != _cb.get("dtype"):
                    raise ValueError("the C2 container or dtype differs from the certifying pack's")
            elif blt_temporal_K:
                _a, _mm = _cx.encode_boundary_bridge(np.asarray(m["dlog_b"]), K=int(blt_temporal_K),
                                                 dtype=blt_dtype, container=_container(blt_container), device=device)
            else:
                _a, _mm = _select_boundary_codec(m, np.asarray(m["dlog_b"]), env, tol,
                                                 blt_dtype, verbose, container=_container(blt_container),
                                                 min_K=_c2_min_K, segment_T=_c2_window_T)
            arrays.update(_a); chan_meta["boundary_local_time"] = _mm; channels["rho2"] = True

        # Surface tier (C2) fidelity: certify the boundary channel reproduces the surface-relaxivity
        # signal from its stored coeffs, vs the raw boundary local time.
        if channels["rho2"] and chan_meta.get("boundary_local_time") is not None and cert is None:
            run.phase("certificate surface")
            _cf = _surface_fidelity(m, arrays, chan_meta["boundary_local_time"], env, segment_T=_c2_window_T)
            if _cf is not None:
                fid = dict(fid, err_surface=_cf["err"], floor_surface=_cf["floor"],
                           err_max=max(float(fid.get("err_max", 0.0)), _cf["err"]),
                           floor_max=max(float(fid.get("floor_max", 0.0)), _cf["floor"]))
                if "err_window" in _cf:
                    fid.update(err_surface_window=_cf["err_window"], floor_surface_window=_cf["floor_window"],
                               err_max=max(float(fid["err_max"]), _cf["err_window"]),
                               floor_max=max(float(fid["floor_max"]), _cf["floor_window"]))
                fid["within_2x_floor"] = bool(fid["err_max"] <= 2.0 * fid["floor_max"])
                if sigma_star is not None:
                    fid["meets_target"] = bool(fid["err_max"] <= sigma_star and fid["floor_max"] <= sigma_star)

        # Field tier (C3) fidelity: certify the stored f16 grid sampled at the decoded trajectory
        # reproduces the raw-grid/true-trajectory susceptibility signal (SE + GRE, split-half floor).
        if channels["susceptibility"] and chan_meta.get("susceptibility_path") is not None and cert is not None:
            _cp = (cert["compression"].get("channels") or {}).get("susceptibility_path") or {}
            if int(_cp.get("K", -1)) != int(susc_path_K) or int(_cp.get("bits", -1)) != int(susc_path_bits):
                raise ValueError("the path channel's K or bits differ from the certifying pack's")
        if channels["susceptibility"] and chan_meta.get("susceptibility_path") is not None and cert is None:
            run.phase("certificate path")
            _pf = _susc_path_fidelity(m, arrays, chan_meta["susceptibility_path"],
                                      chan_meta["susceptibility_grid"], env)
            if _pf is not None:
                _bf = _susc_path_bloch_fidelity(m, arrays, chan_meta["susceptibility_path"],
                                                chan_meta["susceptibility_grid"], env)
                fid = dict(fid, err_susc_path=_pf["err"], floor_susc_path=_pf["floor"],
                           susc_path_pulses_certified=_pf["n_pulses_certified"],
                           err_susc_bloch=(None if _bf is None else _bf["err"]),
                           floor_susc_bloch=(None if _bf is None else _bf["floor"]),
                           susc_bloch_walkers=(None if _bf is None else _bf["n_walkers"]),
                           err_max=max(float(fid.get("err_max", 0.0)), _pf["err"],
                                       (0.0 if _bf is None else _bf["err"])),
                           floor_max=max(float(fid.get("floor_max", 0.0)), _pf["floor"],
                                         (0.0 if _bf is None else _bf["floor"])))
                fid["within_2x_floor"] = bool(fid["err_max"] <= 2.0 * fid["floor_max"])
                if sigma_star is not None:
                    fid["meets_target"] = bool(fid["err_max"] <= sigma_star and fid["floor_max"] <= sigma_star)
        if cert is not None:                                   # the certifying pack's per-tier terms, its codec on this walk
            fid.update({k: v for k, v in cert["fidelity"].items()
                        if k.startswith(("err_", "floor_", "susc_")) and k not in ("err_max", "floor_max")})
        if channels["susceptibility"] and "susc_grid_iso_local" in arrays and cert is None:
            _dpos = _cx.decode(pos_arrays, pos_meta)
            run.phase("certificate grid")
            _gf = _susc_grid_fidelity(m, arrays, chan_meta["susceptibility_grid"], _dpos, dt, env)
            if _gf is not None:
                fid = dict(fid, err_susc_se=_gf["err"], floor_susc_se=_gf["floor"],
                           err_max=max(float(fid.get("err_max", 0.0)), _gf["err"]),
                           floor_max=max(float(fid.get("floor_max", 0.0)), _gf["floor"]))
                fid["within_2x_floor"] = bool(fid["err_max"] <= 2.0 * fid["floor_max"])
                if sigma_star is not None:
                    fid["meets_target"] = bool(fid["err_max"] <= sigma_star and fid["floor_max"] <= sigma_star)

        n_t = X.shape[1]
        if voxel_grid is not None:
            from ..phantom.grid import Grid
            if not isinstance(voxel_grid, Grid) or voxel_grid.attach != "substrate":
                raise TypeError("voxel_grid must be a dmipy_sim.phantom.Grid attached to the substrate")
            _w = np.asarray(m["w"], np.float64) if m.get("w") is not None else None
            if cert is not None:
                _ijk, _pools, _n, _floor, _err = voxel_floor_coded(_Cc, dt, X.shape[1], voxel_grid, m.get("comp"), env, w=_w, device=device)
            else:
                _dpos = _cx.decode(pos_arrays, pos_meta)
                run.phase("certificate per voxel")
                _ijk, _pools, _n, _floor, _err = voxel_fidelity(X, dt, _dpos, voxel_grid, m.get("comp"), env, w=_w)
            arrays["voxel_ijk"] = _ijk.astype(np.int32)
            arrays["voxel_certificate"] = np.stack([_n.astype(np.float32), _floor.astype(np.float32), _err.astype(np.float32)], axis=-1)
            _ok = np.isfinite(_floor)
            fid = dict(fid, per_voxel=dict(grid=voxel_grid.to_meta(), pools=[int(p) for p in _pools], n_voxels=int(_ijk.shape[0]),
                                           walkers_min=int(_n[_n > 0].min()) if (_n > 0).any() else 0,
                                           floor_max=float(np.nanmax(_floor)) if _ok.any() else None,
                                           floor_median=float(np.nanmedian(_floor)) if _ok.any() else None,
                                           err_max=float(np.nanmax(_err)) if np.isfinite(_err).any() else None,
                                           within_2x_floor_fraction=(float(np.mean(_err[_ok] <= 2.0 * _floor[_ok])) if _ok.any() else None),
                                           thin_voxels=int(((_n > 0) & (_n < 2)).sum())))
            if sigma_star is not None and _ok.any():
                fid["per_voxel"]["meets_target"] = bool(np.nanmax(_floor) <= sigma_star and np.nanmax(_err) <= sigma_star)
        comp_meta = dict(method=method, K=int(pos_meta.get("K", K)),        # the K stored: the codec clamps a short walk
                         walker_preserving=True, n_t=int(n_t),
                         container=pos_meta.get("container"),               # None: the float container; else the band ranges
                         temporal_bandwidth_hz=float(int(pos_meta.get("K", K)) / (2.0 * (int(n_t) - 1) * dt)))   # K bands over T (#199)
        if band_class is not None:
            comp_meta["scanner"] = str(band_class)                    # the class the band was derived from (#643)
        comp_meta["precision_tiers"] = _precision_tiers(arrays, int(m["n_walkers"]),
                                                        float(fid.get("floor_max") or 0.0),
                                                        bool(m.get("walkers_shuffled")))
        if chan_meta:
            comp_meta["channels"] = chan_meta      # per-channel codec params (Q, scale, ...)
        fid = _round_certificate(fid)
        meta = dict(
            rpk_schema_version=RPK_SCHEMA_VERSION, id=id,
            compression=comp_meta,
            walk_params=dict(n_walkers=int(m["n_walkers"]), n_t=int(n_t), dt_traj=dt,
                             T_max=float(m["T_max"]), diffusivity=m.get("D_intra"), seed=seed_value(m["seed"]),
                             segments=dict(n=1, n_t=int(n_t), T=float((int(n_t) - 1) * dt),
                                           walks=[dict(first=0, last=0, seed=seed_value(m["seed"]))]),
                             cell_size=m.get("cell_size"),
                             substrate_frame=(None if m.get("substrate_frame") is None
                                              else np.asarray(m["substrate_frame"], float).tolist())),
            replay_envelope=dict(gradient=True,
                                 bulk_relaxation=channels["T1T2"],
                                 surface_relaxivity=channels["rho2"],
                                 field=channels["susceptibility"],
                                 magnetization_transfer=channels["mt"],
                                 diffusivity_fixed=True, acquisition=_envelope_summary(env)),
            fidelity=fid, provenance=dict(provenance or {}, code=_code()), license=license, citation=citation)
        meta["substrate"] = m["substrate"]               # the spec the walk was driven by (#130)
        stored = bool(channels["susceptibility"])
        susceptibility_field_of(meta["substrate"], name=id, stores_field=(None if _window_of_plan and not stored else stored))
        run.phase("write")
        pack = ReplayPack(arrays, meta, source=out_path)
        if out_path is not None:
            write_rpk(out_path, {k: v for k, v in arrays.items() if v is not None}, meta)
            run.artifact(out_path)
            _write_run_sidecar(out_path, run, walk)
        if verbose:
            log.info(f"[pack] {id} method={method} K={K} err={fid['err_max']:.4f} "
                  f"floor={fid['floor_max']:.4f} within2x={fid['within_2x_floor']}")
        return pack


def join_segments(packs, *, fidelity, walks, walkers_shuffled):
    """The packs of the consecutive windows of one walk -- each one window of the same saves, codec and band, as
    :func:`build_replay_pack` writes one -- as ONE pack's ``(arrays, meta)`` (RPK.md 4.3): window 0's tensors under the
    channel names and window ``i``'s under ``s{i}/``, the tensors the walk shares (``_SHARED_KEYS``, the field grids)
    once, refused when two windows disagree in one; the channel numbers a codec measures the worst over the windows;
    the segment table ``{n, n_t, T, walks}``; ``fidelity`` the whole's certificate and the precision tiers read from
    it. The metadata is window 0's otherwise."""
    arrays = dict(packs[0].arrays)
    for i, pk in enumerate(packs[1:], start=1):
        for k, v in pk.arrays.items():
            if _is_shared_key(k):
                if not np.array_equal(np.asarray(v), np.asarray(arrays[k])):
                    raise ValueError(f"segment {i} disagrees with segment 0 in the shared tensor {k!r}")
                continue
            arrays[f"s{i}/{k}"] = v
    meta = json.loads(json.dumps(packs[0].meta))
    n_seg = int(packs[0].n_t); steps = n_seg - 1
    n_segments = len(packs)
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
    dt = float(packs[0].dt)
    meta["walk_params"].update(n_t=int(n_segments * steps + 1), T_max=float(n_segments * steps * dt),
                               segments=dict(n=int(n_segments), n_t=int(n_seg), T=float(steps * dt), walks=list(walks)))
    meta["fidelity"] = fidelity
    return arrays, meta


def _window_certificate(fidelity_from, i, n_segments):
    """The certifying pack's certificate for window ``i`` of a segmented build: a certifying pack stored in windows
    certifies window by window (its ``fidelity.segments[i]``, each a measured one-window certificate), so a window
    inherits the error of the window it mirrors and the whole's bound is the sum over the windows once, as the
    certifying pack's own is; a one-window certifying pack certifies every window with its whole. ``None`` stays
    ``None`` (a measured build). A certifying pack with another number of windows is refused by name."""
    if fidelity_from is None:
        return None
    cert = fidelity_from.meta if hasattr(fidelity_from, "meta") else dict(fidelity_from)
    segs = (cert.get("fidelity") or {}).get("segments")
    if not segs:
        return cert
    if len(segs) != int(n_segments):
        raise ValueError(f"the certifying pack holds {len(segs)} window(s), this build {n_segments}: a window inherits the certificate "
                         "of the window it mirrors")
    return dict(cert, fidelity=dict(segs[int(i)]))


def _build_segmented(m, n_segments, n_seg, run, walk, out_path, *, id, K, temporal_bandwidth_hz, scanner, blt_temporal_K, susc_path_K,
                     fidelity, fidelity_from, envelope, sigma_star=None, **kw):
    """:func:`build_replay_pack` for a walk of ``n_segments`` windows of ``n_seg`` saves: every window built as a
    pack of its own from the walk's arrays of that window (:func:`_window_master`), with segment 0's band, contact
    codec and occupancy form, then assembled -- the windows' tensors under ``s{i}/`` beside the tensors the walk
    shares, the whole's positions battery measured over the full walk and its tier terms the bound over the
    windows (:func:`combine_segment_fidelity`), every window's certificate kept under ``fidelity.segments``."""
    dt = float(m["dt_traj"]); steps = n_seg - 1
    T_seg = steps * dt
    comp = m.get("comp")
    crosses = comp is not None and np.asarray(comp).ndim == 2 and bool(np.any(np.asarray(comp)[:, 1:] != np.asarray(comp)[:, :-1]))
    if K is None and temporal_bandwidth_hz is None and fidelity_from is None:
        band_class = scanner if scanner is not None else _walk_scanner(walk)
        if band_class is None:
            raise ValueError("build_replay_pack needs the band to store the positions at: pass scanner=, K= or temporal_bandwidth_hz= (#643)")
        temporal_bandwidth_hz = storage_band_hz(band_class)
    if temporal_bandwidth_hz is not None and K is None:
        K = mode_count(T_seg, temporal_bandwidth_hz=temporal_bandwidth_hz)
    packs = []
    for i in range(n_segments):
        run.phase(f"segment {i + 1} of {n_segments}")
        w = _window_master(m, i * steps, (i + 1) * steps)
        cert_i = _window_certificate(fidelity_from, i, n_segments)
        if i == 0:
            pk = build_replay_pack(w, id=f"{id}", K=K, scanner=scanner, blt_temporal_K=blt_temporal_K, susc_path_K=susc_path_K, fidelity=fidelity,
                                   fidelity_from=cert_i, envelope=envelope, segment_T=T_seg, _occupancy_runs=crosses, sigma_star=sigma_star,
                                   _window_of_plan=True, **kw)
            K = int(pk.K)
            pm0 = (pk.meta["compression"].get("channels") or {}).get("susceptibility_path")
            if pm0 is not None:
                susc_path_K = int(pm0["K"])                   # a band derived on window 0 is every window's
            c2 = (pk.meta["compression"].get("channels") or {}).get("boundary_local_time")
            if c2 is not None:
                if c2.get("mode") != "bridge_dst":
                    raise ValueError("a walk stored in segments keeps its contact channel in the bridge form, and segment 0 chose "
                                     f"{c2.get('mode')!r}; pass blt_temporal_K=")
                blt_temporal_K = int(c2["K"])
        else:
            pk = build_replay_pack(w, id=f"{id}", K=K, scanner=scanner, blt_temporal_K=blt_temporal_K, susc_path_K=susc_path_K, fidelity=fidelity,
                                   fidelity_from=cert_i, envelope=envelope, segment_T=T_seg, _occupancy_runs=crosses, sigma_star=sigma_star,
                                   _window_of_plan=True, voxel_grid=None, **{k_: v_ for k_, v_ in kw.items() if k_ != "voxel_grid"})
        packs.append(pk)
    # the whole: the positions battery measured over the full walk (the walk is in hand), the tier terms bounded
    fid = combine_segment_fidelity([pk.meta["fidelity"] for pk in packs])
    if fidelity == "measured":
        run.phase("certificate whole")
        X = np.asarray(m["traj"], np.float64)
        pos = np.concatenate([pk.positions()[:, (0 if j == 0 else 1):] for j, pk in enumerate(packs)], axis=1)
        whole = _cx.measure_fidelity(X, dt, pos, envelope or _cx.default_envelope())
        del X, pos
        tier_err = [fid[k] for k in fid if k.startswith("err_") and k != "err_max"]
        tier_floor = [fid[k] for k in fid if k.startswith("floor_") and k != "floor_max"]
        fid.update(per_family=whole["per_family"], noise_floor=whole["noise_floor"],
                   err_max=float(max([whole["err_max"]] + tier_err)), floor_max=float(max([whole["floor_max"]] + tier_floor)),
                   certified="measured", positions="measured over the whole walk", tiers="bounded over the segments")
        fid["within_2x_floor"] = bool(fid["err_max"] <= 2.0 * fid["floor_max"])
    if sigma_star is not None:                                 # the floor-target policy's verdict on the whole
        fid.update(target_floor=float(sigma_star), meets_target=bool(fid["err_max"] <= sigma_star and fid["floor_max"] <= sigma_star))
    fid = _round_certificate(fid)
    arrays, meta = join_segments(packs, fidelity=fid, walks=[dict(first=0, last=int(n_segments) - 1, seed=seed_value(m["seed"]))],
                                 walkers_shuffled=bool(m.get("walkers_shuffled")))
    # meta["provenance"]["code"] is already window 0's (every window is itself a build_replay_pack call, #541)
    run.phase("write")
    pack = ReplayPack(arrays, meta, source=out_path)
    if out_path is not None:
        write_rpk(out_path, {k: v for k, v in arrays.items() if v is not None}, meta)
        run.artifact(out_path)
        _write_run_sidecar(out_path, run, walk)
    return pack


def build_to_floor(make_model, *, id, envelope=None, sigma_star=1e-3, pilot_n=8000,
                   safety=1.4, max_n=400000, walk=None, method="bridge_dst", verbose=True, **bp):
    """Adaptive floor-targeting generation policy (the bank default).

    Size the walker count so the split-half Monte-Carlo floor <= ``sigma_star``, then build the pack at the band
    ``bp`` states (``scanner=``, ``K=`` or ``temporal_bandwidth_hz=``; #643: the band is never searched) and record
    whether its codec error meets ``sigma_star`` too (``fidelity.meets_target``) -- converging to a defined
    precision instead of a wasteful ultra-high N. ``make_model(n_walkers)`` MUST return a fresh master walk (dict/.npz) on
    the SAME fixed geometry (only the walker count changes). ``walk(model)`` returns its master dict
    (default: the model already IS one). Records ``sigma_star`` + the achieved floor in the pack.
    """
    env = envelope or _cx.default_envelope()
    _walk = walk or _master_arrays
    model = make_model(pilot_n)
    f0 = _measure_floor(_walk(model), env)
    n_star = int(min(max_n, max(pilot_n, round(pilot_n * (f0 / sigma_star) ** 2 * safety))))
    if verbose:
        log.info(f"[floor-target] pilot N={pilot_n}: floor={f0:.4g}; sigma*={sigma_star:.4g} -> N*~{n_star}")
    if n_star == pilot_n:                            # the pilot meets the floor: it is the walk
        f = f0
    else:
        model = None                                 # the pilot is released before the sized walk is made, not after
        model = make_model(n_star); f = _measure_floor(_walk(model), env)
    while f > sigma_star and n_star < max_n:         # undershoot -> re-estimate from the last measurement and walk again,
        n_star = int(min(max_n, max(n_star + 1, round(n_star * (f / sigma_star) ** 2 * safety))))   # until met or capped
        if verbose:
            log.info(f"[floor-target] floor={f:.4g} > sigma*; topping up to N*={n_star}")
        model = None                                 # one walk in memory at a time: a rebinding would hold both
        model = make_model(n_star); f = _measure_floor(_walk(model), env)
    if verbose:
        log.info(f"[floor-target] N={n_star}: achieved floor={f:.4g} "
              f"({'<=' if f <= sigma_star else '>'} sigma*)")
    # build_replay_pack normalises the raw model itself (idempotent if already a master dict)
    return build_replay_pack(model, id=id, envelope=env, method=method,
                             sigma_star=sigma_star, verbose=verbose, **bp)
