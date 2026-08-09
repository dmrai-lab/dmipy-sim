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
    def dct_coeffs(self):
        return self.arrays["dct_coeffs"]                       # (n_walkers, K, 3)

    @property
    def spin_weights(self):
        w = self.arrays.get("spin_weights")
        return np.ones(self.dct_coeffs.shape[0]) if w is None else w

    @property
    def blt_dct(self):
        return self.arrays.get("blt_dct")                      # (n_walkers, Kb) or None

    @property
    def K(self):
        return int(self.dct_coeffs.shape[1])

    @property
    def n_walkers(self):
        return int(self.dct_coeffs.shape[0])

    @property
    def n_t(self):
        wp = self.meta.get("walk_params", {})
        return int(self.meta.get("n_t") or wp.get("n_t"))

    @property
    def dt(self):
        wp = self.meta.get("walk_params", {})
        return float(self.meta.get("dt") or wp.get("dt_traj") or wp.get("dt"))


# ------------------------------- compiled-scheme forward -------------------------------
def compile_scheme(G, dt, K, gyromagnetic_ratio=GAMMA):
    """Compile an acquisition into its temporal-basis projection ``W`` (3K, n_meas).

    ``G`` is the gradient waveform on the pack save grid, shape ``(n_meas, n_t, 3)`` [T/m]; ``dt`` the save
    interval [s]; ``K`` the pack's DCT-mode count. Reusable across every pack on this grid (fitting) and
    every fit iteration; in design it is recomputed per candidate waveform (cheap: an FFT + scale)."""
    from scipy.fft import dct
    G = np.asarray(G, np.float64)
    Ghat = dct(G, type=2, norm="ortho", axis=1)[:, :K, :]      # (n_meas, K, 3)
    n_meas = Ghat.shape[0]
    return (gyromagnetic_ratio * dt * Ghat).reshape(n_meas, K * 3).T   # (3K, n_meas)


def surface_logweight(blt_dct, rho_over_D, n_t, chi_hat=None):
    """Per-walker surface log-weight ``(rho/D) * sum_t chi(t) ell_i(t)`` from the boundary-local-time DCT
    coefficients ``blt_dct`` (n_walkers, Kb). Un-gated (``chi_hat=None``): exact total contact
    ``sqrt(n_t) * beta0`` (the DC coefficient). Coherence-gated: contract with ``chi_hat`` = DCT of the
    occupancy schedule (Parseval)."""
    blt = np.asarray(blt_dct, np.float64)
    if chi_hat is None:
        s = np.sqrt(n_t) * blt[:, 0]
    else:
        chi_hat = np.asarray(chi_hat, np.float64)[: blt.shape[1]]
        s = blt[:, : chi_hat.shape[0]] @ chi_hat
    return rho_over_D * s


def replay_signal(pack, W, *, rho_over_D=0.0, chi_hat=None, complex_signal=False):
    """Replay a compiled scheme ``W`` (from :func:`compile_scheme`) against ``pack`` (a :class:`ReplayPack`
    or a plain arrays dict). Returns ``E`` per measurement (magnitude unless ``complex_signal``).

    ``rho_over_D`` > 0 activates the exact surface-relaxivity replay via the pack's boundary local time
    (a per-walker signal loss decaying the whole signal, including ``b=0``); ``chi_hat`` coherence-gates it.
    """
    a = pack.arrays if isinstance(pack, ReplayPack) else pack
    C = np.asarray(a["dct_coeffs"], np.float64)
    N_w, K, _ = C.shape
    w0 = np.asarray(a.get("spin_weights", np.ones(N_w)), np.float64)
    phi = C.reshape(N_w, K * 3) @ W                            # (N_w, n_meas)
    w_eff = w0
    blt = a.get("blt_dct")
    if blt is not None and rho_over_D:
        n_t = pack.n_t if isinstance(pack, ReplayPack) else None
        w_eff = w0 * np.exp(surface_logweight(blt, rho_over_D, n_t, chi_hat))
    S = (w_eff[:, None] * np.exp(1j * phi)).sum(0) / w0.sum()
    return S if complex_signal else np.abs(S)


def replay_signal_jax(dct_coeffs, spin_weights, W, *, blt_dct=None, rho_over_D=0.0, n_t=None, chi_hat=None):
    """JAX/autodiff twin of :func:`replay_signal` — differentiable in the compiled scheme ``W`` (hence in
    the waveform ``G`` that produced it) and jittable. Returns the complex signal (take ``abs`` for
    magnitude). This is the forward a gradient-based waveform/B1 optimizer differentiates through."""
    import jax.numpy as jnp
    C = jnp.asarray(dct_coeffs)
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
