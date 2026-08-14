"""Replay-pack assembler — walk once, compress, self-certify, freeze into a ``.rpk``.

:func:`build_replay_pack` turns a **master walk** (from
:func:`dmipy_sim.core.simulate_trajectories`, ``save_relaxation_data=True``) into a compressed,
self-describing replay pack: the position ensemble is compressed by a :mod:`dmipy_sim.compression`
codec (default ``temporal_dct``), the tier channels the certified envelope needs are carried
(bulk relaxation via the compartment map, surface relaxivity via the boundary-local-time channel,
magnetization transfer via the bound-fraction channel), and the pack MEASURES its own replay
fidelity against the split-half Monte-Carlo floor before it is written. The result is a single
``safetensors`` file readable by :func:`dmipy_sim.replay.read_rpk` and replayable by the
:mod:`dmipy_sim.replay` / :mod:`dmipy_sim.trajectories` operators.

:func:`build_to_floor` is the bank's default generation policy: size the walker count so the
Monte-Carlo floor meets a target ``sigma_star``, then build a pack whose codec error meets it too.

This is the *producer* side of the substrate bank. Publishing/pulling packs to a HuggingFace
dataset (the federation layer) lives separately. The susceptibility (field) replay tier needs the
per-walker susceptibility-basis channel and its ``Q(H)`` contraction (a separate module) and is not
assembled here yet.
"""
from __future__ import annotations

import numpy as np

from . import compression as _cx
from .replay import ReplayPack, read_rpk, write_rpk

__all__ = ["build_replay_pack", "build_to_floor", "replay_susc", "frame_from_axis", "frame_from_bundles",
           "read_rpk", "write_rpk", "RPK_SCHEMA_VERSION"]

RPK_SCHEMA_VERSION = "1.1"


# --------------------------------------------------------------- master-walk normalisation
def _master_arrays(src) -> dict:
    """Normalise a raw master walk (dict or ``.npz``) to the common master dict consumed by
    :func:`build_replay_pack`. ``src`` must expose at least ``traj`` (n_walkers, n_t, 3),
    ``dt_traj`` and ``T_max``; the tier channels (``comp``/``T2_per_comp``/``T1_per_comp`` for
    bulk relaxation, ``dlog_b`` for surface relaxivity, ``bfrac`` for MT) are optional."""
    if not (isinstance(src, dict) or hasattr(src, "files")):
        raise TypeError(
            "build_replay_pack expects a master-walk dict / .npz (the output of "
            "simulate_trajectories(..., save_relaxation_data=True), assembled into a dict with "
            "keys traj/dt_traj/T_max[/comp/T2_per_comp/T1_per_comp/dlog_b/bfrac]); "
            f"got {type(src).__name__}.")
    keys = src.files if hasattr(src, "files") else src.keys()
    m = {k: src[k] for k in keys}
    g = lambda k, d=None: (np.asarray(m[k]) if k in m and m[k] is not None else d)
    traj = np.asarray(m["traj"])
    scal = lambda k: (float(np.asarray(m[k])) if m.get(k) is not None else None)   # idempotent re-normalise
    return dict(traj=traj, dt_traj=float(np.asarray(m["dt_traj"])),
                T_max=float(np.asarray(m["T_max"])), comp=g("comp"), comp0=g("comp0"),
                w=g("w"), dlog_b=g("dlog_b"), bfrac=g("bfrac"),
                # static field-grid susceptibility channel (dict of grids + world origin + chi)
                susc_field_basis=(m.get("susc_field_basis") if isinstance(m, dict) else None),
                susc_grid_origin=(np.asarray(m["susc_grid_origin"]) if "susc_grid_origin" in m else None),
                susc_chi_iso=scal("susc_chi_iso"), delta_chi_a=scal("delta_chi_a"),
                # analytic phasor maps / per-walker basis are NOT assembled publicly (grid form only)
                PhiC=g("PhiC"), PhiS=g("PhiS"), Phi0=g("Phi0"), susc_basis=g("susc_basis"),
                mt_params=m.get("mt_params") if isinstance(m, dict) else None,
                cell_size=scal("cell_size"), R=g("R"), D_intra=scal("D_intra"),
                T2_per_comp=g("T2_per_comp"), T1_per_comp=g("T1_per_comp"),
                substrate_frame=g("substrate_frame"),
                walkers_shuffled=bool(m.get("walkers_shuffled", False)),
                n_walkers=int(traj.shape[0]), seed=int(np.asarray(m.get("seed", 0))))


# --------------------------------------------------------------- substrate frames
def frame_from_axis(axis):
    """Deterministic orthonormal substrate frame R (3x3, columns [x, y, z]) with z = `axis`
    (the primary fibre direction) and a FIXED perpendicular x/y basis (Gram-Schmidt seeded
    from the global axis least aligned with z). A single fibre vector leaves a free rotation
    about itself; this pins x/y so directions are reproducible run-to-run and gradient schemes
    are oriented unambiguously. For an isotropic substrate any axis works — the frame is still
    fixed, giving uniform behaviour across packs."""
    z = np.asarray(axis, float); z = z / np.linalg.norm(z)
    seed = np.eye(3)[int(np.argmin(np.abs(z)))]        # global axis least aligned with z
    x = seed - z * float(seed @ z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


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
    fidelity err is 0, so ``floor_max`` is the substrate's own finite-N statistical noise)."""
    traj = np.asarray(m["traj"], np.float64)
    return float(_cx.measure_fidelity(traj, float(m["dt_traj"]), traj, env)["floor_max"])


def _surface_fidelity(m, arrays, chan_meta, env):
    """Certify the surface tier (C2): the surface-relaxivity attenuation reconstructed from the
    STORED boundary channel vs the RAW per-step boundary local time, over a rho battery, against
    the split-half MC floor of the raw surface signal. Returns ``dict(err, floor)`` or None."""
    raw = m.get("dlog_b")
    has_stored = any(k in arrays for k in ("blt_dct_coeffs", "blt_dense_q", "blt_counts"))
    if raw is None or not has_stored:
        return None
    raw = np.asarray(raw, np.float64); n_w = raw.shape[0]
    w = np.asarray(m["w"], np.float64) if m.get("w") is not None else np.ones(n_w)
    D = float(m.get("D_intra") or 0.0) or 1.0
    if "blt_dct_coeffs" in arrays:
        decoded = _cx.decode_boundary_dct(arrays, chan_meta)
    else:
        decoded = _cx.decode_boundary_local_time(arrays, chan_meta)
    perm = np.random.RandomState(0).permutation(n_w); A, B = perm[:n_w // 2], perm[n_w // 2:]
    fac = lambda sl, idx: float(np.sum(w[idx] * np.exp(sl[idx])) / np.sum(w[idx]))
    err = floor = 0.0
    for rho in (env.get("rho_list") or [1e-5, 3e-5, 1e-4]):
        rd = float(rho) / D
        sl_raw = _cx.surface_logweight(raw, rd)
        sl_dec = _cx.surface_logweight(decoded, rd)
        err = max(err, abs(fac(sl_raw, slice(None)) - fac(sl_dec, slice(None))))
        floor = max(floor, abs(fac(sl_raw, A) - fac(sl_raw, B)))
    return dict(err=float(err), floor=float(floor))


def _susc_grid_fidelity(m, arrays, gm, decoded_pos, dt, env):
    """Certify the static field-grid tier (SE + GRE): the STORED f16 grid sampled at the DECODED
    trajectory vs the RAW f64 grid at the FULL-resolution trajectory (folds in f16 quantisation AND
    the position-codec error), against the split-half MC floor of the raw signal. Returns dict or None."""
    from .constants import GAMMA
    from .susceptibility_field import assemble_field, sample_grid
    fb = m.get("susc_field_basis")
    if fb is None or "susc_grid_iso_local" not in arrays:
        return None
    origin = np.asarray(gm["origin"], float); vs = np.asarray(gm["voxel_size"], float)
    raw_traj = np.asarray(m["traj"], np.float64); n_w, n_t = raw_traj.shape[0], raw_traj.shape[1]
    w = np.asarray(m["w"], np.float64) if m.get("w") is not None else np.ones(n_w)
    braw = {"iso_local": np.asarray(fb["iso_local"], np.float64), "iso_P": np.asarray(fb["iso_P"], np.float64),
            "aniso_G": (np.asarray(fb["aniso_G"], np.float64) if fb.get("aniso_G") is not None else None),
            "shape": tuple(fb["shape"]), "voxel_size": vs}
    bsto = {"iso_local": np.asarray(arrays["susc_grid_iso_local"], np.float64),
            "iso_P": np.asarray(arrays["susc_grid_iso_P"], np.float64),
            "aniso_G": (np.asarray(arrays["susc_grid_aniso_G"], np.float64) if "susc_grid_aniso_G" in arrays else None),
            "shape": tuple(gm["shape"]), "voxel_size": vs}
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
            s_raw = sample_grid(assemble_field(braw, d, B0=B0, chi_iso=chi_i, chi_aniso=ca), raw_traj, origin, vs)
            s_dec = sample_grid(assemble_field(bsto, d, B0=B0, chi_iso=chi_i, chi_aniso=ca), decoded_pos, origin, vs)
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
    from .constants import GAMMA
    from .susceptibility_field import assemble_field, sample_grid
    from .trajectories import replay_bloch
    fb = m.get("susc_field_basis")
    if fb is None or "susc_path_dct" not in arrays:
        return None
    traj = np.asarray(m["traj"], np.float64)
    n_w, n_t = traj.shape[0], traj.shape[1]
    dt = float(m["dt_traj"]); TE = (n_t - 1) * dt
    k = int(min(n_sub, n_w))
    pos = traj[:k]
    w = (np.asarray(m["w"], np.float64)[:k] if m.get("w") is not None else np.ones(k))
    origin = np.asarray(gm["origin"], float); vs = np.asarray(fb["voxel_size"], float)
    braw = {"iso_local": np.asarray(fb["iso_local"], np.float64),
            "iso_P": np.asarray(fb["iso_P"], np.float64),
            "aniso_G": (np.asarray(fb["aniso_G"], np.float64) if fb.get("aniso_G") is not None else None),
            "shape": tuple(fb["shape"]), "voxel_size": vs}
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
        ev = [dict(t_s=0.0, flip_deg=90.0, axis_deg=0.0)]
        for j in range(npul):
            ev.append(dict(t_s=(j + 0.5) * TE / npul, flip_deg=180.0, axis_deg=90.0))
        return ev

    G0 = np.zeros((1, n_t, 3))                       # b=0 isolates the susceptibility physics
    def sig(field, rf, idx=None):
        f = field if idx is None else field[idx]
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
            f_raw = sample_grid(assemble_field(braw, d, B0=B0, chi_iso=chi_i, chi_aniso=ca),
                                pos, origin, vs, periodic=False)
            f_dec = susc_path_field(b_dec, d, B0=B0, chi_iso=chi_i, chi_aniso=ca,
                                    has_aniso=bool(gm.get("has_aniso")))
            for npul in (1, n_p):
                rf = rf_train(npul)
                sr = sig(f_raw, rf)
                err = max(err, abs(sig(f_dec, rf) - sr))
                floor = max(floor, abs(sig(f_raw, rf, A) - sig(f_raw, rf, B)))
    return dict(err=float(err), floor=float(floor), n_pulses=n_p, n_walkers=k)


def _susc_path_fidelity(m, arrays, pm, gm, env):
    """Certify the susc_path_dct tier AT ITS DECLARED CAPABILITY.

    Compares the stored (K-truncated, f16) channels against the raw full-resolution field sampled on
    the true trajectory, under the gates the pack claims to support -- GRE, spin echo, AND the CPMG
    train at ``max_refocus_pulses``. The CPMG gate is the binding one: truncation error grows with
    gate bandwidth, so certifying on SE alone would pass a pack that fails the trains it advertises.
    """
    from .constants import GAMMA
    from .susceptibility_field import assemble_field, sample_grid
    fb = m.get("susc_field_basis")
    if fb is None or "susc_path_dct" not in arrays:
        return None
    traj = np.asarray(m["traj"], np.float64); n_w, n_t = traj.shape[0], traj.shape[1]
    dt = float(m["dt_traj"])
    w = np.asarray(m["w"], np.float64) if m.get("w") is not None else np.ones(n_w)
    origin = np.asarray(gm["origin"], float); vs = np.asarray(fb["voxel_size"], float)
    braw = {"iso_local": np.asarray(fb["iso_local"], np.float64),
            "iso_P": np.asarray(fb["iso_P"], np.float64),
            "aniso_G": (np.asarray(fb["aniso_G"], np.float64) if fb.get("aniso_G") is not None else None),
            "shape": tuple(fb["shape"]), "voxel_size": vs}
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
    gates = [np.ones(n_t), _cpmg_gate(n_t, 1), _cpmg_gate(n_t, n_p)]
    perm = np.random.RandomState(0).permutation(n_w); A, B = perm[:n_w // 2], perm[n_w // 2:]
    wmean = lambda c, idx: float(np.sum(w[idx] * c[idx]) / np.sum(w[idx]))
    err = floor = 0.0
    for B0 in (env.get("B0_list") or [3.0, 7.0]):
        for th in (env.get("theta_deg") or [0, 90]):
            t = np.deg2rad(float(th)); d = [np.sin(t), 0.0, np.cos(t)]
            f_raw = sample_grid(assemble_field(braw, d, B0=B0, chi_iso=chi_i, chi_aniso=ca),
                                traj, origin, vs, periodic=False)
            f_dec = susc_path_field(b_dec, d, B0=B0, chi_iso=chi_i, chi_aniso=ca,
                                    has_aniso=bool(gm.get("has_aniso")))
            for g in gates:
                cr = np.cos(GAMMA * dt * (f_raw * g[None, :]).sum(1))
                cd = np.cos(GAMMA * dt * (f_dec * g[None, :]).sum(1))
                err = max(err, abs(wmean(cr, slice(None)) - wmean(cd, slice(None))))
                floor = max(floor, abs(wmean(cr, A) - wmean(cr, B)))
    return dict(err=float(err), floor=float(floor), n_pulses_certified=n_p)


def susc_path_encode(fb, traj, origin, *, K=32, dtype=np.float16, atol_trace=1e-6):
    """Encode the off-resonance field ALONG each walker's path as K temporal DCT-II coefficients.

    The obstacle to compressing susceptibility is that phi_chi = gamma*int s(t)*dB(r(t)) dt is
    NONLINEAR in r, so a low-rank/DCT model of r(t) has no Parseval shortcut the way the gradient
    phase does. The fix is to move the compression to the other side of the nonlinearity: evaluate
    the field basis along the FULL-resolution path first, then compress the resulting scalars
    b_c(t). Those are very red (the field varies on ~1um scales while r(t) is broadband), so K<<n_t.

    Stored per walker: n_ch channels x K coefficients, where the channels are geometry only -- B0
    direction/magnitude and chi stay replay knobs via the same Q(H) contraction the grid form uses.

    ``iso_P_zz`` is dropped when the exact identity ``P_xx + P_yy + P_zz == 3*iso_local`` holds
    (it does to machine precision for an un-apodised source grid), and reconstructed on read. If a
    k-space window was applied to iso_P but not iso_local the identity is broken, so it is CHECKED
    rather than assumed and all six components are kept when it fails.

    **K is a gate-bandwidth capability, not a field-fidelity knob.** b_c(t) is broadband -- the walker
    crosses ~1um field structure every save step -- so truncation does NOT reproduce the field (K=512
    still misses ~60% of its peak). What it reproduces is the *phase*, because phi only sees b(t)
    integrated against the sequence gate s(t), and s(t) is narrowband. Measured requirement on the
    vector-Bloch path (n_t=1601, TE=36ms, MC floor 9.4e-3, tolerance floor/3):

        CPMG pulses  1   2   4   8   16  32  64
        min K        8   8   16  16  32  64  128        =>  K >= 2 * n_refocus

    So ``max_refocus_pulses = K//2`` is recorded in the metadata; a consumer replaying a denser train
    than that is outside the pack's certified range. Do not read K as "the field is accurate to X".
    """
    from scipy.fft import dct
    from .susceptibility_field import sample_grid
    traj = np.asarray(traj, np.float64)
    n_w, n_t = traj.shape[0], traj.shape[1]
    vs = np.asarray(fb["voxel_size"], float)
    il = np.asarray(fb["iso_local"], np.float64)
    iP = np.asarray(fb["iso_P"], np.float64)
    aG = np.asarray(fb["aniso_G"], np.float64) if fb.get("aniso_G") is not None else None

    # the identity is a property of the GRIDS, so test it there (cheap) rather than along paths
    tr = iP[0] + iP[1] + iP[2]
    scale = float(np.max(np.abs(il))) or 1.0
    trace_res = float(np.max(np.abs(tr - 3.0 * il))) / (3.0 * scale)
    drop_zz = bool(trace_res <= atol_trace)

    names = ["iso_local"] + [f"iso_P_{n}" for n in _ISO_P_NAMES if not (drop_zz and n == "zz")]
    grids = [il] + [iP[i] for i, n in enumerate(_ISO_P_NAMES) if not (drop_zz and n == "zz")]
    if aG is not None:
        names += [f"aniso_G_{n}" for n in _ISO_P_NAMES]
        grids += [aG[i] for i in range(6)]

    K = int(min(K, n_t))
    coeffs = np.empty((n_w, len(grids), K), np.float64)
    for c, g in enumerate(grids):                      # one channel at a time: n_w*n_t floats, not n_ch*
        coeffs[:, c, :] = dct(sample_grid(g, traj, origin, vs, periodic=False),
                              type=2, norm="ortho", axis=1)[:, :K]
    meta = dict(channel="susc_path_dct", K=K, n_t=int(n_t), n_ch=len(grids), channels=names,
                iso_P_zz=("implied" if drop_zz else "stored"), trace_residual=trace_res,
                dtype=np.dtype(dtype).name, max_refocus_pulses=K // 2)
    return {"susc_path_dct": np.asarray(coeffs, dtype)}, meta


def susc_path_decode(arrays, meta, *, n_w=None):
    """Reconstruct b_c(t) per walker from the stored coefficients; re-inserts iso_P_zz if implied.

    Returns ``(field, names)`` with field ``(n_w, n_ch_full, n_t)`` in the canonical channel order
    (iso_local, iso_P_xx..yz, [aniso_G_xx..yz]) so the Q(H) contraction indexes it directly.
    """
    from scipy.fft import idct
    C = np.asarray(arrays["susc_path_dct"], np.float64)
    if n_w is not None:
        C = C[:int(n_w)]
    n_t = int(meta["n_t"])
    b = idct(C, type=2, norm="ortho", axis=2, n=n_t) if C.shape[2] == n_t else \
        idct(np.pad(C, ((0, 0), (0, 0), (0, n_t - C.shape[2]))), type=2, norm="ortho", axis=2)
    names = list(meta["channels"])
    if meta.get("iso_P_zz") == "implied":
        i_loc = names.index("iso_local")
        i_xx, i_yy = names.index("iso_P_xx"), names.index("iso_P_yy")
        zz = 3.0 * b[:, i_loc] - b[:, i_xx] - b[:, i_yy]
        b = np.insert(b, names.index("iso_P_xy"), zz, axis=1)
        names = names[:names.index("iso_P_xy")] + ["iso_P_zz"] + names[names.index("iso_P_xy"):]
    return b, names


def susc_path_field(b, b0_dir, *, B0, chi_iso, chi_aniso=0.0, has_aniso=False):
    """Contract decoded path channels into dB(t) for one (B0, direction, chi) -- mirrors assemble_field."""
    from .susceptibility_field import _q_of_H
    q = _q_of_H(b0_dir)
    dB = chi_iso * B0 * (b[:, 0] - np.einsum("c,cwt->wt", q, np.swapaxes(b[:, 1:7], 0, 1)))
    if has_aniso and chi_aniso and b.shape[1] >= 13:
        dB = dB + chi_aniso * B0 * np.einsum("c,cwt->wt", q, np.swapaxes(b[:, 7:13], 0, 1))
    return dB


# --------------------------------------------------------------- susceptibility replay (consume)
def _pack_positions(pack):
    """Reconstruct the (n_w, n_t, 3) trajectory from a pack's position codec."""
    cx = pack.meta.get("compression", {})
    meta = {"method": cx.get("method", "temporal_dct"), "K": int(cx.get("K", 0)),
            "n_t": int(cx.get("n_t") or pack.n_t)}
    wp = _cx.is_walker_preserving(meta["method"])
    return _cx.decode(pack.arrays, meta, n_walkers=(pack.n_walkers if wp else None))


def _se_gate(n_t, dt, refocus_time):
    """Transverse-phase gate s(t): +1 before a 180 at ``refocus_time`` (s), -1 after; balanced so a
    static field refocuses exactly (sum s = 0). ``None`` -> gradient echo (s == +1)."""
    if refocus_time is None:
        return np.ones(n_t)
    t = np.arange(n_t) * dt
    s = np.sign(refocus_time - t).astype(float)
    d = int(round(s.sum()))
    if d != 0:
        side = np.where(s == np.sign(d))[0]
        s[side[np.argsort(-np.abs(t[side] - refocus_time))[:abs(d)]]] = 0.0
    return s


def replay_susc(pack, waveform, *, b0_dir=(0.0, 0.0, 1.0), B0=0.0, chi_iso=0.0, chi_aniso=0.0,
                refocus_time=None, relaxation=True, complex_signal=False):
    """Replay a gradient waveform on a static field-grid pack WITH susceptibility (the C3 consume
    path). Reconstructs the trajectory, accrues the gradient phase and the susceptibility phase
    (``assemble_field`` for ``(b0_dir,B0,chi_iso,chi_aniso)`` sampled along the walk, gated by the SE
    ``refocus_time``) in the SAME complex mean so the diffusion x susceptibility cross-term is kept,
    plus optional per-compartment relaxation. ``waveform`` has ``.G`` (n_meas,n_t,3) and ``.dt``;
    returns the (complex or magnitude) signal (n_meas,)."""
    from .constants import GAMMA
    from .susceptibility_field import assemble_field, sample_grid
    chans = (pack.meta.get("compression", {}).get("channels", {}) or {})
    pm = chans.get("susceptibility_path")
    has_grid = "susc_grid_iso_local" in pack.arrays
    if pm is None and not has_grid:
        raise ValueError("pack carries no susceptibility channel (neither susc_path_dct nor susc_grid_*).")
    G = np.asarray(getattr(waveform, "G", waveform), np.float64)
    dt = float(getattr(waveform, "dt", pack.dt))
    pos = np.asarray(_pack_positions(pack), np.float64)                 # (n_w, n_t, 3)
    n_w, n_t = pos.shape[0], pos.shape[1]
    w = np.asarray(pack.arrays.get("spin_weights", np.ones(n_w)), np.float64)
    gm = chans["susceptibility_grid"]
    if pm is not None:
        # PATH route (preferred): the field was sampled along the full-resolution walk at build time,
        # so it does not depend on the position codec being lossless.
        b, _ = susc_path_decode(pack.arrays, pm, n_w=n_w)
        dB = susc_path_field(b, b0_dir, B0=B0, chi_iso=chi_iso, chi_aniso=chi_aniso,
                             has_aniso=bool(gm.get("has_aniso")))
    else:
        basis = {"iso_local": np.asarray(pack.arrays["susc_grid_iso_local"], np.float64),
                 "iso_P": np.asarray(pack.arrays["susc_grid_iso_P"], np.float64),
                 "aniso_G": (np.asarray(pack.arrays["susc_grid_aniso_G"], np.float64)
                             if "susc_grid_aniso_G" in pack.arrays else None),
                 "shape": tuple(gm["shape"]), "voxel_size": np.asarray(gm["voxel_size"], float)}
        dB = sample_grid(assemble_field(basis, b0_dir, B0=B0, chi_iso=chi_iso, chi_aniso=chi_aniso),
                         pos, np.asarray(gm["origin"], float), gm["voxel_size"], periodic=False)
    phi_x = GAMMA * dt * (dB * _se_gate(n_t, dt, refocus_time)[None, :]).sum(1)                # (n_w,)
    phi_G = GAMMA * dt * np.einsum("mtj,wtj->mw", G, pos)                                       # (n_meas,n_w)
    logw = np.zeros(n_w)
    ch = (pack.meta.get("compression", {}).get("channels", {}) or {})
    if relaxation and "comp_rle_vals" in pack.arrays and pack.meta.get("per_comp", {}).get("T2"):
        comp = _cx.decode_compartment(pack.arrays, ch.get("compartment", {}))
        pc = pack.meta["per_comp"]
        logw = _cx.relaxation_logweight(comp, pc["T2"], pc.get("T1"), dt)
    ew = w * np.exp(logw)
    S = (ew[None, :] * np.exp(1j * (phi_G + phi_x[None, :]))).sum(1) / w.sum()
    return S if complex_signal else np.abs(S)


# --------------------------------------------------------------- pack generation
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


def _select_boundary_codec(m, dlog, env, tol, dtype, verbose=False):
    """Choose the C2 (boundary-local-time) codec by COST subject to the surface-fidelity gate.

    The historical default was sparse CSR, which is exact but costs ~one entry per wall contact, so it
    scales with walk length (574 B/walker at n_t=1601 for an axon, and worse the longer you record).
    The detrended-cumulative DCT is flat in n_t and, at K=32 in f16, lands thousands of times below the
    surface split-half floor -- but it was opt-in, so the *default* pack got the expensive channel.
    Cost each candidate, keep the cheapest that passes, and fall back to exact sparse if none do.
    """
    cands = []
    for K in (8, 16, 32, 64):
        a, mm = _cx.encode_boundary_dct(dlog, K=int(K), dtype=dtype)
        cf = _surface_fidelity(m, a, mm, env)
        nb = sum(int(np.asarray(v).nbytes) for v in a.values()) / max(len(dlog), 1)
        cands.append((nb, K, a, mm, cf))
        if cf is not None and cf["err"] <= tol * cf["floor"]:
            if verbose:
                print(f"[bank] C2 codec: boundary_dct K={K} {np.dtype(dtype).name} "
                      f"({nb:.0f} B/walker, err={cf['err']:.2e} vs floor {cf['floor']:.2e})", flush=True)
            return a, mm
    a, mm = _cx.encode_boundary_local_time(dlog)
    if verbose:
        nb = sum(int(np.asarray(v).nbytes) for v in a.values()) / max(len(dlog), 1)
        best = min(cands, key=lambda c: (c[4] or {}).get("err", np.inf))
        print(f"[bank] C2 codec: no DCT K passed the surface gate (best K={best[1]} "
              f"err={(best[4] or {}).get('err', float('nan')):.2e}); using exact sparse "
              f"({nb:.0f} B/walker)", flush=True)
    return a, mm


def build_replay_pack(src, *, id, method="temporal_dct", envelope=None, tol=2.0, K=None,
                      err_target=None, sigma_star=None,
                      license, citation, provenance=None, surface_relaxivity=False,
                      blt_temporal_K=None, blt_dtype=np.float16, susc_path_K=None, mt=False,
                      out_path=None, verbose=False):
    """Compress a master walk and assemble a self-certifying replay pack.

    ``src`` is a raw master dict / ``.npz`` (see :func:`_master_arrays`). The position ensemble is
    compressed by ``method`` (default ``temporal_dct``); ``K`` (mode count) is chosen automatically
    to keep the *measured* replay error within ``tol``x the Monte-Carlo floor over ``envelope``
    (default :func:`compression.default_envelope`) unless given. Walker-preserving methods
    (``temporal_dct``, ``lowrank``) carry the full per-walker channel space; distributional methods
    (``gaussian``/``marginal``) are gradient-only.

    Tiers assembled: **gradient** (always), **bulk relaxation** (from the compartment map +
    per-compartment T1/T2, when ``comp``/``T2_per_comp`` are present), **surface relaxivity**
    (``surface_relaxivity=True``, needs ``dlog_b``), **magnetization transfer** (``mt=True``, needs
    ``bfrac``). The **susceptibility (field)** tier is not assembled here yet (needs the
    susceptibility-basis channel + Q(H) contraction) — a master carrying ``susc_basis``/``PhiC`` is
    rejected. Returns a :class:`dmipy_sim.replay.ReplayPack`; writes it to ``out_path`` if given.
    """
    m = _master_arrays(src)
    if m.get("PhiC") is not None or m.get("susc_basis") is not None:
        raise NotImplementedError(
            "the analytic phasor-map (PhiC) and per-walker (susc_basis) susceptibility forms are "
            "not assembled publicly; use the STATIC FIELD-GRID form (master key 'susc_field_basis', "
            "e.g. from dmipy_sim.mesh_axon.mesh_axon_master).")
    env = envelope or _cx.default_envelope()
    X = np.asarray(m["traj"], np.float64)
    dt = float(m["dt_traj"])
    wp_method = _cx.is_walker_preserving(method)
    if K is None:
        K, fid = _cx.auto_select_modes(X, X, dt, method=method, env=env, tol=tol,
                                       err_target=err_target, verbose=verbose)
    else:
        arrays0, meta0, _ = _cx.encode(X, method, K)
        pos = _cx.decode(arrays0, meta0, n_walkers=(X.shape[0] if wp_method else None))
        fid = _cx.measure_fidelity(X, dt, pos, env)
    if sigma_star is not None:                       # adaptive floor-target policy (build_to_floor)
        fid = dict(fid, target_floor=float(sigma_star),
                   meets_target=bool(fid["err_max"] <= sigma_star and fid["floor_max"] <= sigma_star))

    pos_arrays, pos_meta, _ = _cx.encode(X, method, K)
    arrays = dict(pos_arrays)
    chan_meta = {}                                   # per-channel codec params
    channels = {"gradient": True, "susceptibility": False, "T1T2": False, "rho": False,
                "mt": (m.get("bfrac") is not None) or (m.get("mt_params") is not None)}
    # STATIC field-grid susceptibility channel: store the geometry-only field-basis grids ONCE
    # (a substrate property); replay assembles the field for any (B0,dir,chi) and samples it along
    # the pos-codec-decoded trajectory (replay_susc). O(N_vox) not O(N_w*N_t) and SE-exact (a static
    # field at a frozen point cancels under the SE gate to machine precision). f16 grids: O(1) geometry.
    if m.get("susc_field_basis") is not None:
        fb = m["susc_field_basis"]
        # The GRID route samples the field at codec-DECODED positions, so it is only sound when the
        # position codec is lossless. The PATH route samples the FULL-RESOLUTION trajectory at build
        # time, which is precisely what frees the positions to be lossy -- so the two cannot both be
        # advertised: shipping grid arrays next to lossy positions would offer a replay route whose
        # accuracy silently depends on a property the pack no longer has. Path wins when present;
        # the grid is then published as a separate per-substrate companion artefact, not per walker.
        # temporal_dct at K=n_t is an exact rewrite; other codecs are not lossless at any K.
        _pos_lossless = (method == "temporal_dct") and int(K) >= int(X.shape[1])
        _grid_in_pack = (not susc_path_K) or _pos_lossless
        if _grid_in_pack:
            arrays["susc_grid_iso_local"] = np.asarray(fb["iso_local"], np.float16)
            arrays["susc_grid_iso_P"] = np.asarray(fb["iso_P"], np.float16)
            if fb.get("aniso_G") is not None:
                arrays["susc_grid_aniso_G"] = np.asarray(fb["aniso_G"], np.float16)
        chan_meta["susceptibility_grid"] = dict(
            origin=np.asarray(m["susc_grid_origin"], float).tolist(),
            voxel_size=np.asarray(fb["voxel_size"], float).tolist(),
            shape=[int(s) for s in fb["shape"]], has_aniso=(fb.get("aniso_G") is not None),
            chi_iso=(m.get("susc_chi_iso")), delta_chi_a=(m.get("delta_chi_a")),
            arrays_in_pack=bool(_grid_in_pack),
            replay_route=("grid+path" if (_grid_in_pack and susc_path_K)
                          else ("path" if susc_path_K else "grid")))
        channels["susceptibility"] = True
        # PATH form (C3, preferred): the field sampled along each walker's FULL-RESOLUTION path and
        # compressed in time. Decouples the susceptibility tier from the position codec -- which is
        # what lets the positions go back to K << n_t, since the only reason they had to be stored
        # losslessly was that grid-sampling needed exact r(t). See susc_path_encode for why K is a
        # gate-bandwidth capability rather than a fidelity knob.
        if susc_path_K:
            _a, _pm = susc_path_encode(fb, np.asarray(m["traj"], np.float64),
                                       np.asarray(m["susc_grid_origin"], float),
                                       K=int(susc_path_K), dtype=np.float16)
            arrays.update(_a); chan_meta["susceptibility_path"] = _pm
    if wp_method:
        # compartment map (RLE / quantized-RLE) -> per-compartment T1/T2; surface/MT opt-in.
        if m.get("comp") is not None and m.get("T2_per_comp") is not None:
            _a, _cm = _cx.encode_compartment(np.asarray(m["comp"]))
            arrays.update(_a); chan_meta["compartment"] = _cm
            if m.get("w") is not None:
                arrays["spin_weights"] = np.asarray(m["w"], np.float32)
            channels["T1T2"] = True
        # dense per-walker physics channels get their own codecs (compression.py):
        # boundary local time -> sparse/dense or (temporal_K) cumulative-DCT; bound_frac -> RLE.
        if surface_relaxivity and m.get("dlog_b") is not None:
            if blt_temporal_K:
                _a, _mm = _cx.encode_boundary_dct(np.asarray(m["dlog_b"]), K=int(blt_temporal_K),
                                                 dtype=blt_dtype)
            else:
                _a, _mm = _select_boundary_codec(m, np.asarray(m["dlog_b"]), env, tol,
                                                 blt_dtype, verbose)
            arrays.update(_a); chan_meta["boundary_local_time"] = _mm; channels["rho"] = True
        if mt and m.get("bfrac") is not None:
            _a, _mm = _cx.encode_bound_fraction(np.asarray(m["bfrac"]), Q=256)
            arrays.update(_a); chan_meta["bound_fraction"] = _mm; channels["mt"] = True

    # Surface tier (C2) fidelity: certify the boundary channel reproduces the surface-relaxivity
    # signal from its stored coeffs, vs the raw boundary local time.
    if channels["rho"] and chan_meta.get("boundary_local_time") is not None:
        _cf = _surface_fidelity(m, arrays, chan_meta["boundary_local_time"], env)
        if _cf is not None:
            fid = dict(fid, err_surface=_cf["err"], floor_surface=_cf["floor"],
                       err_max=max(float(fid.get("err_max", 0.0)), _cf["err"]),
                       floor_max=max(float(fid.get("floor_max", 0.0)), _cf["floor"]))
            fid["within_2x_floor"] = bool(fid["err_max"] <= 2.0 * fid["floor_max"])
            if sigma_star is not None:
                fid["meets_target"] = bool(fid["err_max"] <= sigma_star and fid["floor_max"] <= sigma_star)

    # Field tier (C3) fidelity: certify the stored f16 grid sampled at the decoded trajectory
    # reproduces the raw-grid/true-trajectory susceptibility signal (SE + GRE, split-half floor).
    if channels["susceptibility"] and chan_meta.get("susceptibility_path") is not None:
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
    if channels["susceptibility"] and m.get("susc_field_basis") is not None:
        _dpos = _cx.decode(pos_arrays, pos_meta, n_walkers=(X.shape[0] if wp_method else None))
        _gf = _susc_grid_fidelity(m, arrays, chan_meta["susceptibility_grid"], _dpos, dt, env)
        if _gf is not None:
            fid = dict(fid, err_susc_se=_gf["err"], floor_susc_se=_gf["floor"],
                       err_max=max(float(fid.get("err_max", 0.0)), _gf["err"]),
                       floor_max=max(float(fid.get("floor_max", 0.0)), _gf["floor"]))
            fid["within_2x_floor"] = bool(fid["err_max"] <= 2.0 * fid["floor_max"])
            if sigma_star is not None:
                fid["meets_target"] = bool(fid["err_max"] <= sigma_star and fid["floor_max"] <= sigma_star)

    n_t = X.shape[1]
    comp_meta = dict(method=method, K=int(K), walker_preserving=bool(wp_method), n_t=int(n_t))
    if wp_method:
        comp_meta["precision_tiers"] = _precision_tiers(arrays, int(m["n_walkers"]),
                                                        float(fid.get("floor_max") or 0.0),
                                                        bool(m.get("walkers_shuffled")))
    if chan_meta:
        comp_meta["channels"] = chan_meta      # per-channel codec params (Q, scale, ...)
    meta = dict(
        rpk_schema_version=RPK_SCHEMA_VERSION, id=id,
        compression=comp_meta,
        walk_params=dict(n_walkers=int(m["n_walkers"]), n_t=int(n_t), dt_traj=dt,
                         T_max=float(m["T_max"]), diffusivity=m.get("D_intra"), seed=int(m["seed"]),
                         cell_size=m.get("cell_size"),
                         substrate_frame=(None if m.get("substrate_frame") is None
                                          else np.asarray(m["substrate_frame"], float).tolist())),
        per_comp=dict(T2=(None if m.get("T2_per_comp") is None else np.asarray(m["T2_per_comp"]).tolist()),
                      T1=(None if m.get("T1_per_comp") is None else np.asarray(m["T1_per_comp"]).tolist()),
                      R=(None if m.get("R") is None else np.asarray(m["R"]).tolist())),
        replay_envelope=dict(gradient=True,
                             bulk_relaxation=channels["T1T2"],
                             surface_relaxivity=channels["rho"],
                             field=channels["susceptibility"],
                             magnetization_transfer=channels["mt"],
                             diffusivity_fixed=True, acquisition=_envelope_summary(env)),
        fidelity=fid, provenance=provenance or {}, license=license, citation=citation)
    if m.get("mt_params") is not None:              # parametric two-pool qMT (pool-level knob)
        meta["mt"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                      for k, v in m["mt_params"].items()}
    pack = ReplayPack(arrays, meta, source=out_path)
    if out_path is not None:
        write_rpk(out_path, {k: v for k, v in arrays.items() if v is not None}, meta)
    if verbose:
        print(f"[pack] {id} method={method} K={K} err={fid['err_max']:.4f} "
              f"floor={fid['floor_max']:.4f} within2x={fid['within_2x_floor']}", flush=True)
    return pack


def build_to_floor(make_model, *, id, envelope=None, sigma_star=1e-3, pilot_n=8000,
                   safety=1.4, max_n=400000, walk=None, method="temporal_dct", verbose=True, **bp):
    """Adaptive floor-targeting generation policy (the bank default).

    Size the walker count so the split-half Monte-Carlo floor <= ``sigma_star``, then build a pack
    whose codec error is <= ``sigma_star`` too — converging to a defined precision instead of a
    wasteful ultra-high N. ``make_model(n_walkers)`` MUST return a fresh master walk (dict/.npz) on
    the SAME fixed geometry (only the walker count changes). ``walk(model)`` returns its master dict
    (default: the model already IS one). Records ``sigma_star`` + the achieved floor in the pack.
    """
    env = envelope or _cx.default_envelope()
    _walk = walk or _master_arrays
    f0 = _measure_floor(_walk(make_model(pilot_n)), env)
    n_star = int(min(max_n, max(pilot_n, round(pilot_n * (f0 / sigma_star) ** 2 * safety))))
    if verbose:
        print(f"[floor-target] pilot N={pilot_n}: floor={f0:.4g}; sigma*={sigma_star:.4g} -> N*~{n_star}", flush=True)
    model = make_model(n_star); f = _measure_floor(_walk(model), env)
    if f > sigma_star and n_star < max_n:            # undershoot -> one re-estimate/top-up
        n_star = int(min(max_n, round(n_star * (f / sigma_star) ** 2 * safety)))
        if verbose:
            print(f"[floor-target] floor={f:.4g} > sigma*; topping up to N*={n_star}", flush=True)
        model = make_model(n_star); f = _measure_floor(_walk(model), env)
    if verbose:
        print(f"[floor-target] N={n_star}: achieved floor={f:.4g} "
              f"({'<=' if f <= sigma_star else '>'} sigma*)", flush=True)
    # build_replay_pack normalises the raw model itself (idempotent if already a master dict)
    return build_replay_pack(model, id=id, envelope=env, method=method,
                             err_target=sigma_star, sigma_star=sigma_star, verbose=verbose, **bp)
