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

Surface relaxivity is exact, via the stored boundary local time (the C2 channel, bridge form): a
per-walker reweight by ``exp((rho/D) * sum_t chi(t) ell_i(t))``, optionally coherence-gated by an
occupancy schedule ``chi`` (:func:`surface_logweight`).
"""
import json
from functools import cached_property

import numpy as np

from ..constants import GAMMA

__all__ = ["ReplayPack", "PoseSpectra", "read_rpk", "write_rpk",
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


class PoseSpectra:
    """A pack's response over every pose of its substrate for one acquisition: per measurement the two-axis spectrum
    ``Lambda`` (:func:`sh_convolution.coupled_spectrum_at`), the gradient direction and the shared B0 direction, all
    in the scanner frame. ``compose(fod)`` gives the signal of a distribution of poses; ``at(direction)`` the signal
    of one pose read from the same spectra (a peak, RPH.md 4)."""

    def __init__(self, lams, gdir, b0_dir, l_g, l_b, misfit):
        self.lams, self.gdir, self.b0_dir, self.l_g, self.l_b, self.misfit = lams, np.asarray(gdir, float), \
            np.asarray(b0_dir, float), int(l_g), int(l_b), float(misfit)

    @property
    def n_meas(self):
        return len(self.lams)

    def compose(self, fod):
        """``(n_meas,)`` complex: the pose-averaged signal for a :class:`~dmipy_sim.replay.fod.FOD`."""
        from .sh_convolution import apply_odf_coupled
        out = np.empty(self.n_meas, np.complex128)
        for i, lam in enumerate(self.lams):
            out[i] = apply_odf_coupled(lam, fod.coeffs, self.gdir[i], self.b0_dir, l_fod=fod.lmax, l_g=self.l_g, l_b=self.l_b)
        return out

    def at(self, direction):
        """``(n_meas,)`` complex: the response at one pose (the substrate axis along ``direction``), evaluated from
        the spectra -- the zero-dispersion limit a peak stands for."""
        from scipy.special import eval_legendre
        n = np.asarray(direction, float); n = n / np.linalg.norm(n)
        out = np.empty(self.n_meas, np.complex128)
        for i, lam in enumerate(self.lams):
            u, v = float(n @ self.gdir[i]), float(n @ self.b0_dir)
            kv = np.cross(self.gdir[i], self.b0_dir); s_k = np.linalg.norm(kv)
            wt = float(n @ (kv / s_k)) if s_k > 1e-12 else 0.0
            out[i] = sum(c * eval_legendre(l1, u) * eval_legendre(l2, v) * (wt ** p)
                         for (l1, l2, p), c in zip(lam["terms"], lam["coeffs"]))
        return out


class ReplayPack:
    """A replay pack: the channel ``arrays`` plus ``meta``, with ``load`` / ``save`` and the one
    consume path, :meth:`replay`. Accessors mirror the walk parameters (``n_t``, ``dt``, ``K``,
    ``n_walkers``) and the tiers carried (``has_relaxation``, ``has_surface``, ``has_field``)."""

    def __init__(self, arrays, meta, source=None):
        self.arrays = dict(arrays)
        self.meta = dict(meta)
        self.source = source

    @classmethod
    def load(cls, path):
        """Read a ``.rpk`` file."""
        return read_rpk(path)

    def save(self, path):
        """Write this pack to a ``.rpk`` file."""
        write_rpk(path, {k: v for k, v in self.arrays.items() if v is not None}, self.meta)
        self.source = str(path)
        return path

    # ---- tiers carried ----
    @property
    def has_relaxation(self):
        """C1: a compartment channel, so per-pool T2 / T1 given at replay can be applied."""
        return "comp_rle_vals" in self.arrays

    @property
    def substrate(self):
        """The :class:`~dmipy_sim.spec.SubstrateSpec` the walk was driven by, when the pack embeds one."""
        d = self.meta.get("substrate")
        if d is None:
            return None
        from ..spec.substrate import SubstrateSpec
        return SubstrateSpec.from_dict(d)

    @property
    def has_surface(self):
        """C2: the boundary local time in the bridge form."""
        return "blt_bridge_dst" in self.arrays

    @property
    def has_field(self):
        """C3: a susceptibility path channel or the static field grid."""
        ch = (self.meta.get("compression", {}).get("channels", {}) or {})
        return ch.get("susceptibility_path") is not None or "susc_grid_iso_local" in self.arrays

    @property
    def diffusivity(self):
        """The walk's diffusivity (m^2/s) when the producer recorded it."""
        return self.meta.get("walk_params", {}).get("diffusivity")

    def positions(self):
        """The ``(n_walkers, n_t, 3)`` trajectory decoded from the position codec (float64)."""
        from .compression import decode, is_walker_preserving, require_position_method
        cx = self.meta.get("compression", {})
        meta = {"method": require_position_method(cx.get("method")), "K": int(cx.get("K", 0)),
                "n_t": int(cx.get("n_t") or self.n_t)}
        wp = is_walker_preserving(meta["method"])
        return np.asarray(decode(self.arrays, meta, n_walkers=(self.n_walkers if wp else None)), np.float64)

    def _by_pool(self, values, what):
        """Per-pool values as a list by id; a ``{name: value}`` dict resolves through the embedded spec."""
        if values is None:
            return None
        if isinstance(values, dict):
            spec = self.substrate
            if spec is None:
                raise ValueError(f"{what} given by pool name but the pack embeds no substrate spec; give a list by id")
            out = [0.0] * len(spec.pools)
            for name, v in values.items():
                out[spec.pool(name).id] = float(v)
            return out
        return [float(v) for v in np.asarray(values, float).reshape(-1)]

    def replay(self, waveform, *, tissue="nominal", T2=None, T1=None, rho=None, D=None, B0=None,
               b0_dir=(0.0, 0.0, 1.0), chi_iso=None, chi_aniso=0.0, refocus_time="auto", compartment=None,
               orientation=None, fod=None, complex_signal=False):
        """The signal of ``waveform`` on this pack, with every tier the pack carries and the request asks for.

        ``waveform`` is a :class:`~dmipy_sim.acquisition.waveforms.Waveform` / :class:`~dmipy_sim.sequences.Sequence`
        (``G`` (n_meas, n_t_wf, 3) in T/m, ``dt``), or a bare ``G`` already on the pack's save grid; it
        is resampled onto the pack grid (``n_t`` samples of ``dt``, zero outside the waveform).

        * **gradient** (C0): always, in mode space from the position coefficients -- unless a field is
          requested, when the trajectory is decoded and the two phases accrue in one complex mean so
          their cross-term is kept.
        * **bulk relaxation** (C1): ``T2`` (and ``T1``, under the waveform's coherence gate) per pool
          id, or a ``{pool name: value}`` dict resolved through the embedded substrate spec; the pack
          carries the occupancy channel and no value, so nothing is applied unless given.
        * ``tissue``: where the physical values come from. ``"nominal"`` (default) is the **nominal replay**:
          the values the embedded substrate spec declares (pool T2 / T1, wall rho, the field source's chi,
          the calibration field ``nominal_field_T`` as B0) -- what a paper's pack reproduces by firing a
          pulse at it; a pack without a spec, or a spec that declares no values, replays the gradient alone.
          ``False`` is the bare diffusion signal. A :class:`~dmipy_sim.spec.Tissue` supplies the values
          yourself. In every case an explicit keyword wins.
        * **orientation**: the substrate's pose in the scanner -- a 3x3 rotation (substrate frame -> lab) or the
          lab direction its axis (``spec.frame.axis``, default z) points along. Exact by pose covariance: the
          gradient and B0 are rotated into the substrate frame together.
        * **fod**: a :class:`~dmipy_sim.replay.fod.FOD` -- the signal of a distribution of poses of this
          substrate, composed through the two-axis Gaunt route (:mod:`sh_convolution`): the gradient and
          field axes move together, so the composition is a spherical convolution only when there is no
          field. Needs one gradient direction per measurement; with a field, the pack's path channel. A bare
          coefficient array is refused: the basis must be declared (``FOD.from_sh``, ``FOD.native``).
        * **surface relaxivity** (C2): ``rho`` (m/s) with the walk's diffusivity ``D`` (the pack's
          recorded value unless given); requires the boundary local time.
        * **field** (C3): ``B0`` (T) with ``b0_dir`` and the susceptibility ``chi_iso`` (required) and
          ``chi_aniso``; the pack stores the substrate's geometry-only basis and no susceptibility value,
          so these are the replay's to give; ``refocus_time="auto"`` reads the 180 from the waveform's
          RF schedule (``None`` = gradient echo); requires the field tier.

        ``compartment`` restricts the ensemble mean to one pool id (or a boolean walker mask).
        A tier that is requested but not carried raises rather than returning a plausible number.
        """
        from .compression import read_position_coeffs
        from ._replay_kernel import gradient_phase, se_gate
        if orientation is not None and fod is not None:
            raise ValueError("orientation= and fod= exclude each other: one pose, or a distribution of poses")
        P = self._prepare(waveform, tissue=tissue, T2=T2, T1=T1, rho=rho, D=D, B0=B0, b0_dir=b0_dir, chi_iso=chi_iso,
                          chi_aniso=chi_aniso, orientation=orientation, compartment=compartment)
        if fod is not None:
            from .fod import FOD
            if not isinstance(fod, FOD):
                raise TypeError("fod must be a dmipy_sim.replay.fod.FOD -- a bare coefficient array has no basis, and the "
                                "Gaunt composition is silently wrong in a non-orthonormal one; use FOD.from_sh(coeffs, "
                                "basis=...) / FOD.native(coeffs) / FOD.watson(...)")
            S = self._pose_spectra(P, refocus_time, waveform).compose(fod)
            return S if complex_signal else np.abs(S)
        n_w, dt, n_t, Geff = P["n_w"], P["dt"], P["n_t"], P["Geff"]
        if P["B0"] is None:
            C = read_position_coeffs(self.arrays, dtype=np.float64)
            W = _compile_effective(Geff, dt, self.K, n_t)
            phi = C.reshape(n_w, self.n_coeffs * 3) @ W                              # (n_w, n_meas)
        else:
            if not self.has_field:
                raise ValueError("B0 was given but the pack carries no field tier (C3); build it with field=FieldGrid(...)")
            from .bank import susc_path_decode, susc_path_field
            from ..fields.susceptibility_field import assemble_field, sample_grid
            ch, b0_dir, B0, chi_aniso = P["ch"], P["b0_dir"], P["B0"], P["chi_aniso"]
            gm = ch["susceptibility_grid"]
            if P["chi_iso"] is None:
                raise ValueError("B0 was given without chi_iso: the pack carries the substrate's field basis, "
                                 "not a susceptibility; give chi_iso (and chi_aniso) at replay")
            chi_i = float(P["chi_iso"])
            pos = self.positions()
            pm = ch.get("susceptibility_path")
            if pm is not None:
                b, _ = susc_path_decode(self.arrays, pm, n_w=n_w)
                dB = susc_path_field(b, b0_dir, B0=float(B0), chi_iso=chi_i, chi_aniso=chi_aniso,
                                     has_aniso=bool(gm.get("has_aniso")))
            else:
                basis = {"iso_local": np.asarray(self.arrays["susc_grid_iso_local"], np.float64),
                         "iso_P": np.asarray(self.arrays["susc_grid_iso_P"], np.float64),
                         "aniso_G": (np.asarray(self.arrays["susc_grid_aniso_G"], np.float64)
                                     if "susc_grid_aniso_G" in self.arrays else None),
                         "shape": tuple(gm["shape"]), "voxel_size": np.asarray(gm["voxel_size"], float)}
                dB = sample_grid(assemble_field(basis, b0_dir, B0=float(B0), chi_iso=chi_i, chi_aniso=chi_aniso),
                                 pos, np.asarray(gm["origin"], float), gm["voxel_size"], periodic=False)
            if refocus_time == "auto":
                refocus_time = _refocus_time_of(waveform)
            phi_x = GAMMA * dt * (dB * se_gate(n_t, dt, refocus_time)[None, :]).sum(1)    # (n_w,)
            phi = gradient_phase(Geff, pos, dt).T + phi_x[:, None]                             # (n_w, n_meas)
        S = (P["ew"][:, None] * np.exp(1j * phi)).sum(0) / P["norm"]
        return S if complex_signal else np.abs(S)

    def _prepare(self, waveform, *, tissue, T2, T1, rho, D, B0, b0_dir, chi_iso, chi_aniso, orientation, compartment):
        """Everything a replay resolves before it reads positions: the waveform's exact per-save weights (rotated
        into the substrate frame when a pose is given), the knobs (nominal, a Tissue, or explicit), the per-walker
        weights with the relaxation and surface terms applied, and the compartment selection."""
        from .compression import require_position_method, decode_occupancy, relaxation_logweight
        from ._replay_kernel import effective_gradient, bin_gate
        require_position_method(self.method)
        G = np.asarray(getattr(waveform, "G", waveform), np.float64)
        if G.ndim == 2:
            G = G[None]
        dt_wf = float(getattr(waveform, "dt", self.dt))
        n_t, dt = self.n_t, self.dt
        chi = getattr(waveform, "chi_perp", None)
        if chi is not None:                              # the gate on the occupancy / contact channels: averaged over
            chi = np.asarray(chi, np.float64).reshape(-1)  # each save's accumulation interval, never resampled
            chi = bin_gate(chi, dt_wf if chi.shape[0] == G.shape[1] else dt, n_t, dt)[0]
        ch = (self.meta.get("compression", {}).get("channels", {}) or {})
        n_w = self.n_walkers
        w = np.asarray(self.spin_weights, np.float64)
        if isinstance(tissue, str):
            if tissue != "nominal":
                raise ValueError("tissue must be 'nominal', False, or a Tissue")
            spec = self.substrate
            if spec is not None:
                from ..spec.tissue import Tissue
                tissue = Tissue.from_spec(spec)
            else:
                tissue = None
        elif tissue is False:
            tissue = None
        if tissue is not None:
            k = tissue.knobs()
            T2 = k["T2"] if T2 is None else T2; T1 = k["T1"] if T1 is None else T1
            rho = k["rho"] if rho is None else rho; B0 = k["B0"] if B0 is None else B0
            chi_iso = k["chi_iso"] if chi_iso is None else chi_iso
            if chi_aniso == 0.0: chi_aniso = k["chi_aniso"]
            if tuple(b0_dir) == (0.0, 0.0, 1.0): b0_dir = k["b0_dir"]
        if orientation is not None:
            R = self._rotation_of(orientation)
            G = G @ R                                                     # R^T g per sample
            b0_dir = tuple(np.asarray(R, float).T @ np.asarray(b0_dir, float))
        Geff = effective_gradient(G, dt_wf, n_t, dt)                     # exact per-save weights (n_meas, n_t, 3)
        logw = np.zeros(n_w)
        if T2 is not None or T1 is not None:
            if not self.has_relaxation:
                raise ValueError("T2 / T1 were given but the pack carries no compartment channel (C1); build it "
                                 "from a walk with tiers='all'")
            comp = decode_occupancy(self.arrays, ch["compartment"])["comp"]
            T2v = self._by_pool(T2, "T2"); T1v = self._by_pool(T1, "T1")
            n_ids = int(np.max(comp)) + 1
            if T2v is None:
                T2v = [0.0] * n_ids                                   # no T2 decay, T1 only
            if T1v is None:
                T1v = [0.0] * n_ids                                   # no T1 term
            if len(T2v) < n_ids or (T1v is not None and len(T1v) < n_ids):
                raise ValueError(f"the compartment channel uses pool ids up to {n_ids - 1}; T2 / T1 must be given "
                                 f"for every id (got {len(T2v)}{'' if T1v is None else f' / {len(T1v)}'})")
            logw = logw + relaxation_logweight(comp, T2v, T1v, dt, chi)
        if rho is not None and float(rho) != 0.0:
            D_walk = self.diffusivity if D is None else D
            if D_walk is None:
                raise ValueError("rho needs the walk's diffusivity: the pack did not record it, pass D=")
            logw = logw + surface_logweight(self.arrays, float(rho) / float(D_walk),
                                            ch.get("boundary_local_time"), chi)      # raises without C2
        ew = w * np.exp(logw)
        norm = w.sum()
        ew, norm = self._select(compartment, ew, norm, w, ch, n_w)
        return dict(G=G, Geff=Geff, dt=dt, n_t=n_t, dt_wf=dt_wf, ch=ch, n_w=n_w, w=w, ew=ew, norm=norm, B0=B0,
                    b0_dir=b0_dir, chi_iso=chi_iso, chi_aniso=chi_aniso)

    def pose_spectra(self, waveform, *, tissue="nominal", T2=None, T1=None, rho=None, D=None, B0=None,
                     b0_dir=(0.0, 0.0, 1.0), chi_iso=None, chi_aniso=0.0, refocus_time="auto", compartment=None,
                     l_g=8, l_b=6):
        """The pack's response over every pose of its substrate, one two-axis spectrum per measurement
        (:class:`PoseSpectra`): what a replay phantom composes against each voxel's orientation. Same knobs as
        :meth:`replay`; the acquisition is in the scanner frame, the substrate is rotated under it."""
        P = self._prepare(waveform, tissue=tissue, T2=T2, T1=T1, rho=rho, D=D, B0=B0, b0_dir=b0_dir, chi_iso=chi_iso,
                          chi_aniso=chi_aniso, orientation=None, compartment=compartment)
        return self._pose_spectra(P, refocus_time, waveform, l_g=l_g, l_b=l_b)

    def _select(self, compartment, ew, norm, w, ch, n_w):
        """Restrict the ensemble mean to ``compartment`` (a pool id or a walker mask)."""
        from .compression import decode_occupancy
        if compartment is None:
            return ew, norm
        sel = np.asarray(compartment)
        if sel.dtype != bool:
            if "compartment" not in ch:
                raise ValueError("compartment= by id needs the pack's compartment channel")
            ids = np.asarray(decode_occupancy(self.arrays, ch["compartment"])["comp"])
            ids = ids[:, 0] if ids.ndim == 2 else ids
            sel = ids.astype(int) == int(sel)
        if sel.shape[0] != n_w:
            raise ValueError(f"compartment mask has {sel.shape[0]} entries for {n_w} walkers")
        if not sel.any():
            raise ValueError("compartment selection matched no walkers")
        return np.where(sel, ew, 0.0), w[sel].sum()

    @property
    def frame_axis(self):
        """The substrate's own axis (``spec.frame.axis``; z when the pack embeds no spec)."""
        spec = self.substrate
        a = np.asarray(spec.frame.axis if spec is not None else (0.0, 0.0, 1.0), float)
        return a / np.linalg.norm(a)

    def _rotation_of(self, orientation):
        """A 3x3 rotation (substrate frame -> lab), or the lab direction the substrate axis points along."""
        from .sh_convolution import _rotations_from_axis
        o = np.asarray(orientation, float)
        if o.shape == (3, 3):
            if not np.allclose(o @ o.T, np.eye(3), atol=1e-6) or np.linalg.det(o) < 0:
                raise ValueError("orientation must be a proper rotation matrix (R R^T = I, det +1)")
            return o
        if o.shape == (3,):
            return _rotations_from_axis(self.frame_axis, o[None, :])[0]
        raise ValueError("orientation is a (3, 3) rotation or a (3,) axis direction")

    def _pose_spectra(self, P, refocus_time, waveform, l_g=8, l_b=6, n_theta=32, n_phi=64):
        """Per measurement: the pack's response over a sphere of poses (each pose = the gradient and B0
        counter-rotated, from per-walker contractions hoisted once) and its two-axis spectrum ``Lambda``."""
        from scipy.fft import dct
        from .gaunt import sphere_quadrature
        from .sh_convolution import coupled_spectrum_at, _rotations_from_axis
        from .compression import read_position_coeffs
        from ._replay_kernel import se_gate
        G, dt, n_t, ew, norm, B0 = P["Geff"], P["dt"], P["n_t"], P["ew"], P["norm"], P["B0"]
        b0_dir, chi_iso, chi_aniso = P["b0_dir"], P["chi_iso"], P["chi_aniso"]
        n_meas = G.shape[0]
        n_w = ew.shape[0]
        gdir = np.zeros((n_meas, 3)); prof = np.zeros((n_meas, n_t))
        for i in range(n_meas):
            A = G[i]
            k = int(np.argmax(np.linalg.norm(A, axis=1)))
            if np.linalg.norm(A[k]) == 0.0:
                gdir[i] = self.frame_axis                                          # b = 0: any direction
                continue
            g = A[k] / np.linalg.norm(A[k])
            p = A @ g
            if np.linalg.norm(A - p[:, None] * g[None, :]) > 1e-9 * np.linalg.norm(A):
                raise ValueError(f"measurement {i} has no single gradient direction (a multi-axis waveform); the pose "
                                 f"composition expands the response in the fibre-gradient angle and needs one")
            gdir[i], prof[i] = g, p
        C = read_position_coeffs(self.arrays, dtype=np.float64).reshape(n_w, -1)
        q = np.stack([C @ _compile_effective(np.einsum("it,a->ita", prof, np.eye(3)[a]), dt, self.K, n_t)
                      for a in range(3)], axis=-1)                                   # (n_w, n_meas, 3)
        dirs, wq = sphere_quadrature(n_theta, n_phi)
        R = _rotations_from_axis(self.frame_axis, dirs)                               # (n_dirs, 3, 3)
        b = np.asarray(b0_dir, float); b = b / np.linalg.norm(b)
        if B0 is not None:
            if not self.has_field:
                raise ValueError("B0 was given but the pack carries no field tier (C3)")
            pm = self.meta.get("compression", {}).get("channels", {}).get("susceptibility_path")
            if pm is None:
                raise ValueError("the pose composition with a field needs the pack's susc_path channel (C3 path route)")
            if chi_iso is None:
                raise ValueError("B0 was given without chi_iso; give chi_iso (and chi_aniso)")
            from .bank import susc_path_coeffs
            Cs, names = susc_path_coeffs(self.arrays, pm)
            if refocus_time == "auto":
                refocus_time = _refocus_time_of(waveform)
            gate_hat = dct(se_gate(n_t, dt, refocus_time), type=2, norm="ortho")[:Cs.shape[2]]
            Psi = (GAMMA * dt) * np.einsum("k,wck->wc", gate_hat, Cs)                # (n_w, n_ch)
            bp = np.einsum("nji,j->ni", R, b)
            Q = np.stack([bp[:, 0] ** 2, bp[:, 1] ** 2, bp[:, 2] ** 2, 2 * bp[:, 0] * bp[:, 1], 2 * bp[:, 0] * bp[:, 2],
                          2 * bp[:, 1] * bp[:, 2]], axis=1)
            i_p = names.index("iso_P_xx")
            phi_chi = float(chi_iso) * float(B0) * (Psi[:, names.index("iso_local")][None, :] - Q @ Psi[:, i_p:i_p + 6].T)
            if chi_aniso and "aniso_G_xx" in names:
                ia = names.index("aniso_G_xx")
                phi_chi = phi_chi + float(chi_aniso) * float(B0) * (Q @ Psi[:, ia:ia + 6].T)
            Ew = np.exp(1j * phi_chi) * ew[None, :]                                    # (n_dirs, n_w)
        else:
            Ew = np.broadcast_to(ew[None, :].astype(np.complex128), (dirs.shape[0], n_w))
        lams, worst = [], 0.0
        for i in range(n_meas):
            gp = np.einsum("nji,j->ni", R, gdir[i])                                   # R^T g
            E = (Ew * np.exp(1j * (gp @ q[:, i, :].T))).sum(1) / norm                  # response over poses
            lam, resid, _rank = coupled_spectrum_at(lambda d, E=E: E, gdir[i], b, l_g=l_g, l_b=l_b,
                                                   chiral=(B0 is not None), _grid=(dirs, wq))
            # the fit's misfit in signal units; a finite ensemble leaves a roughness of order the pack's own
            # floor (the same walkers seen from every pose), which is not a defect of the expansion
            worst = max(worst, float(resid) * float(np.sqrt(np.mean(np.abs(E) ** 2))))
            lams.append(lam)
        floor = 1.0 / np.sqrt(n_w)
        if worst > 2.0 * floor:
            import warnings
            warnings.warn(f"the pack's response over poses is not that of an axially symmetric substrate within its own "
                          f"Monte-Carlo floor (fit misfit {worst:.3f} in signal units vs floor {floor:.3f}): the two-axis "
                          f"expansion (l_g={l_g}, l_b={l_b}) does not represent it and the pose composition is unreliable. "
                          f"Compose a non-axisymmetric substrate by averaging over poses instead.", UserWarning, stacklevel=4)
        return PoseSpectra(lams, gdir, b, int(l_g), int(l_b), worst)

    @cached_property
    def position_coeffs(self):
        """``(n_walkers, K+2, n_axes)``: two exact endpoints, then ``K`` sine bands per axis.

        The leading two entries are NOT bands -- they are ``r(0)`` and ``r(T)-r(0)`` -- so this
        array must never be handed to a band-basis contraction.  Raises on a pre-layout pack
        rather than guessing: see compression.read_position_coeffs.

        Stacked from the per-axis tensors once; a pack's arrays do not change after loading, and
        ``K``, ``n_walkers`` and ``n_coeffs`` all read this block.
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

    # ---- producer metadata (present on packs written by dmipy_sim.replay.bank.build_replay_pack) ----
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



def _refocus_time_of(waveform):
    """The time of the waveform's 180 (the first refocusing pulse of its RF schedule), or ``None``
    for a schedule without one (a gradient echo)."""
    for e in (getattr(waveform, "rf_events", None) or []):
        if int(round(float(e.get("flip_deg", 0)))) == 180:
            return float(e["t_s"])
    return None


# ------------------------------- compiled-scheme forward -------------------------------
def compile_scheme(G, dt, K, gyromagnetic_ratio=GAMMA, *, n_t=None, method=None, dt_pack=None):
    """Compile an acquisition into its temporal-basis projection ``W``: the exact integral of the waveform
    against the stored path, in mode space.

    ``G`` is the gradient waveform ``(n_meas, n_wf, 3)`` [T/m] on ITS OWN grid ``dt`` [s]; ``dt_pack`` the pack's
    save interval (default: the waveform is on the pack grid, ``dt_pack = dt``); ``n_t`` the pack's save count
    (default ``G.shape[1]`` on the pack grid); ``K`` the pack's retained-mode count; ``method`` the pack's
    position codec, accepted only to let a caller assert it. Shape is ``(3(K+2), n_meas)``: per axis the two
    gradient moments ``M0`` and ``M1``, which a motion-compensated waveform makes vanish, then the sine bands.
    Nothing is resampled: the waveform enters through its exact per-save weights
    (:func:`_replay_kernel.effective_gradient`), so an edge between saves carries exactly its b. Reusable across
    every pack on one grid (fitting) and every fit iteration; in design it is recomputed per candidate."""
    from .compression import require_position_method
    from ._replay_kernel import effective_gradient
    require_position_method(method or "bridge_dst")
    G = np.asarray(G, np.float64)
    if dt_pack is None:
        dt_pack = dt
        n_t = int(n_t or G.shape[1])
    elif n_t is None:
        raise ValueError("a waveform on its own grid needs the pack's n_t")
    return _compile_effective(effective_gradient(G, dt, int(n_t), dt_pack), dt_pack, K, int(n_t), gyromagnetic_ratio)


def _compile_effective(Geff, dt_pack, K, n_t, gyromagnetic_ratio=GAMMA):
    """``W`` from per-save weights already on the pack grid (:func:`_replay_kernel.effective_gradient`)."""
    from .compression import bridge_projection
    W = bridge_projection(np.asarray(Geff, np.float64), int(n_t), K)              # (n_meas, K+2, 3)
    return (gyromagnetic_ratio * float(dt_pack) * W).reshape(W.shape[0], (K + 2) * 3).T


def surface_logweight(arrays, rho_over_D, chan_meta=None, chi_hat=None):
    """Per-walker surface log-weight ``(rho/D) * sum_t chi(t) ell_i(t)`` from the C2 channel.

    C2 is stored in the bridge form (``blt_bridge_dst`` + the two exact endpoints), so the
    UNGATED total contact is ``blt_endpoint`` read directly -- it is the exact cumulative
    ``L(T)``, not something reconstructed from bands, which is the whole reason the endpoint is
    held exactly. A coherence gate needs the per-save series, so that branch decodes.

    Takes the pack's ``arrays`` rather than one tensor: the channel is three tensors now, and a
    signature that accepted just the coefficient block invited passing the wrong one.
    """
    from .compression import decode_boundary_bridge, surface_logweight_series
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
    return surface_logweight_series(ell[:, : chi.shape[0]], rho_over_D, chi)


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


def replay_signal_jax(position_coeffs, spin_weights, W, *, surface_logw=None):
    """JAX/autodiff twin of :func:`replay_signal` -- differentiable in the compiled scheme ``W``
    (hence in the waveform ``G`` that produced it) and jittable. ``position_coeffs`` is
    ``(n_walkers, K+2, n_axes)`` -- two endpoints then sine bands -- and ``W`` must come from
    :func:`compile_scheme`, which emits the matching row order.

    ``surface_logw`` is the per-walker surface log-weight from :func:`surface_logweight` (it does
    not depend on ``W``, so it is computed once on the host and passed in). Returns the complex
    signal (take ``abs`` for magnitude). This is the forward a gradient-based waveform/B1
    optimizer differentiates through."""
    import jax.numpy as jnp
    C = jnp.asarray(position_coeffs)
    N_w, K, _ = C.shape
    phi = C.reshape(N_w, K * 3) @ jnp.asarray(W)
    w0 = jnp.asarray(spin_weights)
    w_eff = w0 if surface_logw is None else w0 * jnp.exp(jnp.asarray(surface_logw))
    return (w_eff[:, None] * jnp.exp(1j * phi)).sum(0) / w0.sum()
