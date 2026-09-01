"""Replay packs (``.rpk``) and the compiled-scheme forward — the shared replay primitive.

A **replay pack** stores the state of one converged Monte-Carlo walk (walker trajectories as truncated
DCT-II coefficients, plus a spin weight and an optional surface boundary-local-time channel) so the
diffusion-weighted signal for *any* gradient waveform can be reconstructed without re-simulating. It is a
single ``safetensors`` file: the arrays are the tensors, the JSON metadata sits in the ``"rpk"`` header
key. Producers (e.g. the substrate generator) write them; consumers (dmipy-fit compartments, dmipy-design
waveform optimization) replay them.

The forward is exact and cheap. For a stored trajectory ``r_i(t) = idct(C_i)`` the signal is

    E = < w_i exp(i phi_i) > / < w_i > ,   phi_i(m) = gamma * dt * sum_t G_m(t) . r_i(t)

and because the phase is linear in position and the DCT-II is orthonormal (Parseval),

    phi_i(m) = sum_{k,c} C_{i,k,c} * Ghat_{m,k,c},   Ghat = gamma * dt * DCT(G_m)[:K].

The waveform projection ``Ghat`` (= :func:`compile_scheme`) is independent of the walkers; each forward is
then one dense matmul ``C @ W`` + a weighted complex mean (:func:`replay_signal`). This is the SAME math
whether the acquisition is fixed and the substrate varies (fitting) or the substrate is fixed and the
waveform varies (design) — in the latter it is differentiable in ``G``, so it drives gradient-based
waveform/B1 optimization. A JAX twin (:func:`replay_signal_jax`) supplies the autodiff/GPU path.

Surface relaxivity is exact, via the stored boundary local time (``blt_dct``): a per-walker reweight by
``exp((rho/D) * sum_t chi(t) ell_i(t))``, optionally coherence-gated by an occupancy schedule ``chi``.
"""
import json

import numpy as np

from .constants import GAMMA

__all__ = ["ReplayPack", "read_rpk", "write_rpk",
           "compile_scheme", "replay_signal", "replay_signal_jax", "surface_logweight"]


# ------------------------------- .rpk container I/O -------------------------------
def write_rpk(path, arrays, metadata):
    """Write a replay pack: ``arrays`` (name -> ndarray) as safetensors tensors, ``metadata`` (a dict)
    serialised to JSON under the ``"rpk"`` header key."""
    from safetensors.numpy import save_file
    meta = dict(metadata)
    tens = {k: np.ascontiguousarray(v) for k, v in arrays.items()}
    save_file(tens, str(path), metadata={"rpk": json.dumps(meta)})


def read_rpk(path):
    """Read a replay pack into a :class:`ReplayPack` (arrays + metadata)."""
    from safetensors import safe_open
    arrays = {}
    with safe_open(str(path), framework="numpy") as f:
        hdr = f.metadata() or {}
        meta = json.loads(hdr.get("rpk") or hdr.get("json") or "{}")
        for k in f.keys():
            arrays[k] = f.get_tensor(k)
    return ReplayPack(arrays, meta, source=str(path))


class ReplayPack:
    """A loaded replay pack: the channel ``arrays`` plus ``meta``. Convenience accessors mirror the
    walk parameters needed to replay (``n_t``, ``dt``, ``K``, ``n_walkers``)."""

    def __init__(self, arrays, meta, source=None):
        self.arrays = dict(arrays)
        self.meta = dict(meta)
        self.source = source

    @property
    def position_coeffs(self):
        """``(n_walkers, K+2, n_axes)``: two exact endpoints, then ``K`` sine bands per axis.

        The leading two entries are NOT bands -- they are ``r(0)`` and ``r(T)-r(0)`` -- so this
        array must never be handed to a band-basis contraction.  Raises on a pre-layout pack
        rather than guessing: see compression.read_position_coeffs.
        """
        from .compression import read_position_coeffs, require_position_method
        require_position_method(self.method)
        return read_position_coeffs(self.arrays, dtype=np.float32)

    @property
    def dct_coeffs(self):
        raise AttributeError(
            "ReplayPack.dct_coeffs is gone. Positions are stored as bridge_dst -- two exact "
            "endpoints followed by sine bands -- so the name described the wrong basis and the "
            "array it returned would be contracted against the wrong one. Use "
            "ReplayPack.position_coeffs, and compile the scheme with replay.compile_scheme, "
            "which emits the matching layout.")

    @property
    def spin_weights(self):
        w = self.arrays.get("spin_weights")
        return np.ones(self.position_coeffs.shape[0]) if w is None else w

    @property
    def blt_dct(self):
        raise AttributeError(
            "blt_dct is retired: C2 is stored in the bridge form as 'blt_bridge_dst' plus the "
            "exact endpoints 'blt_start'/'blt_endpoint'. Use pack.arrays for the tensors, or "
            "compression.decode_boundary_bridge for the per-save series.")

    @property
    def boundary_local_time(self):
        """The C2 channel's exact cumulative total per walker, or None if the pack has no C2."""
        return self.arrays.get("blt_endpoint")

    @property
    def K(self):
        """Retained sine bands -- NOT the stored coefficient count, which is ``K + 2``.

        Reading the width off the array instead would hand ``K + 2`` to compile_scheme, whose
        output would then be the right shape to multiply and the wrong thing to multiply by.
        Declared and stored are cross-checked so they cannot drift apart.
        """
        k = int(self.meta.get("compression", {}).get("K", -1))
        stored = int(self.position_coeffs.shape[1])
        if k < 0:
            raise ValueError("pack declares no compression.K; refusing to infer it from the "
                             "array width, which counts the two endpoints as well")
        if stored != k + 2:
            raise ValueError(f"pack declares K={k} but stores {stored} coefficients per axis; "
                             f"expected {k + 2} (two endpoints + K bands)")
        return k

    @property
    def n_coeffs(self):
        """Stored coefficients per axis, ``K + 2``."""
        return int(self.position_coeffs.shape[1])

    @property
    def n_walkers(self):
        return int(self.position_coeffs.shape[0])

    @property
    def n_t(self):
        wp = self.meta.get("walk_params", {})
        return int(self.meta.get("n_t") or wp.get("n_t"))

    @property
    def dt(self):
        wp = self.meta.get("walk_params", {})
        return float(self.meta.get("dt") or wp.get("dt_traj") or wp.get("dt"))

    # ---- producer metadata (present on packs written by dmipy_sim.bank.build_replay_pack) ----
    @property
    def id(self):
        return self.meta.get("id")

    @property
    def method(self):
        return self.meta.get("compression", {}).get("method")

    license = property(lambda self: self.meta.get("license"))
    citation = property(lambda self: self.meta.get("citation"))
    fidelity = property(lambda self: self.meta.get("fidelity"))
    replay_envelope = property(lambda self: self.meta.get("replay_envelope"))
    provenance = property(lambda self: self.meta.get("provenance"))


# ------------------------------- compiled-scheme forward -------------------------------
def compile_scheme(G, dt, K, gyromagnetic_ratio=GAMMA, *, n_t=None, method=None):
    """Compile an acquisition into its temporal-basis projection ``W``.

    ``G`` is the gradient waveform on the pack save grid, shape ``(n_meas, n_t, 3)`` [T/m]; ``dt`` the save
    interval [s]; ``K`` the pack's retained-mode count; ``method`` the pack's position codec.
    Shape is ``(3(K+2), n_meas)``: the first two rows per axis are the gradient moments
    ``M0`` and ``M1``, which a motion-compensated waveform makes vanish, followed by the sine
    bands.  ``method`` is accepted only to let a caller assert the pack's codec; a retired one
    raises rather than selecting a different basis. Reusable across every pack on this grid (fitting) and
    every fit iteration; in design it is recomputed per candidate waveform (cheap: an FFT + scale)."""
    from scipy.fft import dst
    from .compression import require_position_method
    require_position_method(method or "bridge_dst")
    G = np.asarray(G, np.float64)
    if True:
        # first two rows per axis are the gradient moments -- the columns a motion-compensated
        # waveform annihilates -- followed by the sine bands of the pinned residual
        from .compression import bridge_moment_rows
        n_t = int(n_t or G.shape[1])
        M0, M1 = bridge_moment_rows(G, n_t)
        Ghat = dst(G[:, 1:-1, :], type=1, norm="ortho", axis=1)[:, :K, :]
        W = np.concatenate([M0[:, None, :], M1[:, None, :], Ghat], axis=1)  # (n_meas, K+2, 3)
        n_meas = W.shape[0]
        return (gyromagnetic_ratio * dt * W).reshape(n_meas, (K + 2) * 3).T


def surface_logweight(arrays, rho_over_D, chan_meta=None, chi_hat=None):
    """Per-walker surface log-weight ``(rho/D) * sum_t chi(t) ell_i(t)`` from the C2 channel.

    C2 is stored in the bridge form (``blt_bridge_dst`` + the two exact endpoints), so the
    UNGATED total contact is ``blt_endpoint`` read directly -- it is the exact cumulative
    ``L(T)``, not something reconstructed from bands, which is the whole reason the endpoint is
    held exactly. A coherence gate needs the per-save series, so that branch decodes.

    Takes the pack's ``arrays`` rather than one tensor: the channel is three tensors now, and a
    signature that accepted just the coefficient block invited passing the wrong one.
    """
    from .compression import decode_boundary_bridge
    if "blt_bridge_dst" not in arrays:
        raise ValueError(
            "surface relaxivity was requested but this pack carries no C2 channel "
            "(no 'blt_bridge_dst'). A pack written before the C2 bridge form stored "
            "'blt_dct_coeffs', which is retired -- re-encode it. Returning the signal without "
            "the requested attenuation would be a plausible wrong number.")
    if chi_hat is None:
        return float(rho_over_D) * np.asarray(arrays["blt_endpoint"], np.float64)
    meta = dict(chan_meta or {})
    meta.setdefault("n_t", int(np.asarray(chi_hat).shape[0]))
    meta.setdefault("K", int(np.asarray(arrays["blt_bridge_dst"]).shape[1]))
    ell = np.asarray(decode_boundary_bridge(arrays, meta), np.float64)
    chi = np.asarray(chi_hat, np.float64)[: ell.shape[1]]
    return float(rho_over_D) * (ell[:, : chi.shape[0]] @ chi)


def replay_signal(pack, W, *, rho_over_D=0.0, chi_hat=None, complex_signal=False):
    """Replay a compiled scheme ``W`` (from :func:`compile_scheme`) against ``pack`` (a :class:`ReplayPack`
    or a plain arrays dict). Returns ``E`` per measurement (magnitude unless ``complex_signal``).

    ``rho_over_D`` > 0 activates the exact surface-relaxivity replay via the pack's boundary local time
    (a per-walker signal loss decaying the whole signal, including ``b=0``); ``chi_hat`` coherence-gates it.
    """
    a = pack.arrays if isinstance(pack, ReplayPack) else pack
    from .compression import read_position_coeffs
    from .compression import require_position_method
    if isinstance(pack, ReplayPack):
        require_position_method(pack.method)
    C = read_position_coeffs(a, dtype=np.float64)
    N_w, K, _ = C.shape
    if W.shape[0] != K * 3:
        raise ValueError(
            f"compiled scheme has {W.shape[0]} rows for {K * 3} stored coefficients "
            f"({K} per axis = 2 endpoints + {K - 2} bands). Compile with "
            f"compile_scheme(G, dt, pack.K, n_t=pack.n_t) -- passing the stored width instead "
            f"of pack.K produces a scheme that multiplies cleanly and means nothing.")
    w0 = np.asarray(a.get("spin_weights", np.ones(N_w)), np.float64)
    phi = C.reshape(N_w, K * 3) @ W                            # (N_w, n_meas)
    w_eff = w0
    if rho_over_D:
        # asked for, so it must happen: a missing C2 channel raises inside surface_logweight
        # rather than being skipped. The previous form looked up a key the bridge rename
        # retired, so `rho_over_D` was silently ignored and callers got an unattenuated signal.
        cm = ((pack.meta.get("compression", {}).get("channels", {}) or {}).get("boundary_local_time")
              if isinstance(pack, ReplayPack) else None)
        w_eff = w0 * np.exp(surface_logweight(a, rho_over_D, cm, chi_hat))
    S = (w_eff[:, None] * np.exp(1j * phi)).sum(0) / w0.sum()
    return S if complex_signal else np.abs(S)


def replay_signal_jax(position_coeffs, spin_weights, W, *, blt_dct=None, rho_over_D=0.0,
                      n_t=None, chi_hat=None):
    """JAX/autodiff twin of :func:`replay_signal` — differentiable in the compiled scheme ``W`` (hence in
    the waveform ``G`` that produced it) and jittable. ``position_coeffs`` is
    ``(n_walkers, K+2, n_axes)`` -- two endpoints then sine bands -- and ``W`` must come from
    :func:`compile_scheme`, which emits the matching row order. Returns the complex signal (take ``abs`` for
    magnitude). This is the forward a gradient-based waveform/B1 optimizer differentiates through."""
    import jax.numpy as jnp
    C = jnp.asarray(position_coeffs)
    N_w, K, _ = C.shape
    phi = C.reshape(N_w, K * 3) @ jnp.asarray(W)
    w0 = jnp.asarray(spin_weights)
    w_eff = w0
    if blt_dct is not None and rho_over_D:
        blt = jnp.asarray(blt_dct)
        s = (jnp.sqrt(n_t) * blt[:, 0] if chi_hat is None
             else blt[:, : jnp.asarray(chi_hat).shape[0]] @ jnp.asarray(chi_hat))
        w_eff = w0 * jnp.exp(rho_over_D * s)
    return (w_eff[:, None] * jnp.exp(1j * phi)).sum(0) / w0.sum()
