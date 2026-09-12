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
from ..acquisition.rf import RFSchedule
from ..acquisition.scanner_sequence import Protocol, ScannerSequence

__all__ = ["ReplayPack", "PoseResponse", "read_rpk", "write_rpk",
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


class PoseResponse:
    """A pack's response over every pose of its substrate, for one acquisition, as SO(3) coefficients.

    The pose of a substrate is a rotation, so the response of one measurement is a function on SO(3), and this
    holds its expansion in the real Wigner basis (:mod:`dmipy_sim.replay.so3`). Two bands are involved and they
    are not the same:

    * the band this was **projected** at, ``(lmax, nmax)``, which follows the response's own sharpness -- the
      accumulated phase amplitude :attr:`phase_amplitude`, in radians. It sets what had to be sampled, since a
      rule that resolves only the retained band folds everything above it into the coefficients kept.
    * the band a composition **retains**, which follows the orientation distribution: composing is an inner
      product, so a distribution has no reach above its own order and none at all outside ``n = 0`` when it
      leaves the substrate's azimuth unstated. :meth:`compose` restricts to that band, exactly.

    ``misfit`` is the worst case, over rotations off the projection's own grid, of what the expansion could not
    reproduce -- in signal units, to be read against the pack's Monte-Carlo floor. A worst case rather than a
    spread, because a composition is exposed to individual poses and a spread bounds nothing.
    """

    def __init__(self, coeffs, lmax, nmax, misfit, floor, phase_amplitude, n_samples):
        self.coeffs = np.asarray(coeffs)
        self.lmax, self.nmax = int(lmax), int(nmax)
        self.misfit = np.asarray(misfit, float)
        self.floor, self.n_samples = float(floor), int(n_samples)
        self.phase_amplitude = float(phase_amplitude)
        self.route = "quadrature" if n_samples > 0 else "closed"      # which route built it (#197)
        self.n_bodies = None                                          # distinct waveforms contracted (closed form)
        self.field_lmax = 0

    @property
    def n_meas(self):
        return self.coeffs.shape[0]

    @property
    def asymmetry(self):
        """The share of the response's energy that depends on the substrate's own azimuth.

        Zero for an axially symmetric substrate, and the reason a pose has to be a rotation rather than an axis
        when it is not: an axis-only representation cannot carry that share, and a rule that samples over
        directions alone folds it into the ``n = 0`` coefficients a composition does use.
        """
        from . import so3
        num = den = 0.0
        for c in self.coeffs:
            _per_l, per_n = so3.energy(c, self.lmax, self.nmax)
            num += float(per_n[1:].sum())
            den += float(per_n.sum())
        return num / max(den, 1e-300)

    def retained(self, lmax, nmax):
        """The coefficients a distribution of this band reaches: ``(n_meas, n_features)``.

        Dropping the rest is lossless, not an approximation -- the distribution has no coefficients there to
        multiply. For an ODF of order 8 this is 45 numbers per measurement out of the 969 an unrestricted
        expansion carries, which is what makes a per-voxel composition a short dot product.
        """
        from .so3 import truncate_coeffs
        return truncate_coeffs(self.coeffs, self.lmax, self.nmax, lmax, nmax)

    def compose(self, distribution):
        """``(n_meas,)`` complex: the signal of a voxel holding this distribution of poses."""
        from .so3 import Distribution
        if not isinstance(distribution, Distribution):
            raise TypeError("compose takes a dmipy_sim.replay.so3.Distribution (Distribution.pose / axis / "
                            "axis_density / watson / bingham / uniform), so that what the coefficients mean is "
                            "carried with them")
        from .so3 import truncate_coeffs
        # both sides come down to the band they share. Dropping the distribution's higher terms is safe for the
        # same reason dropping the response's is: the response was certified to reproduce itself at its own
        # band, so it has nothing above it for those terms to multiply.
        keep_l = min(distribution.lmax, self.lmax)
        keep_n = min(distribution.nmax, self.nmax)
        f = distribution.coeffs
        if (distribution.lmax, distribution.nmax) != (keep_l, keep_n):
            f = truncate_coeffs(f, distribution.lmax, distribution.nmax, keep_l, keep_n)
        return self.retained(keep_l, keep_n) @ f

    def at(self, R):
        """``(n_meas,)`` complex: the response at one stated pose, evaluated from the expansion.

        A single pose is a distribution of unbounded band, so this reads the whole expansion, and it is the one
        case whose accuracy is the pointwise :attr:`misfit`. When one pose is all that is wanted,
        :meth:`ReplayPack.replay` with ``orientation=R`` answers it exactly and costs nothing to certify.
        """
        from .so3 import so3_design
        A = so3_design(self.lmax, np.asarray(R, np.float64).reshape(1, 3, 3), self.nmax)
        return self.coeffs @ A[0]


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
               b0_dir=(0.0, 0.0, 1.0), chi_iso=None, chi_aniso=0.0, compartment=None,
               orientation=None, complex_signal=False):
        """The signal of ``waveform`` on this pack, with every tier the pack carries and the request asks for.

        ``waveform`` is a :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence`: its ``G_eff``, the
        effective gradient, is what this route integrates, and its RF schedule says where the 180 is (a schedule
        without one is a gradient echo). It is read on the pack grid (``n_t`` samples of ``dt``, zero outside
        the waveform).

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
        * **orientation**: the substrate's pose. **One pose** -- a 3x3 rotation (substrate frame -> lab), or the
          lab direction its axis (``spec.frame.axis``, default z) points along -- is exact by pose covariance:
          the gradient and B0 are rotated into the substrate frame together, and no expansion is involved.
          **A distribution of poses** -- a :class:`~dmipy_sim.replay.so3.Distribution` (one pose, an axis
          density, a Watson cone, a Bingham fan) or an :class:`~dmipy_sim.replay.fod.FOD`, which is read as an
          axis density with no statement about the substrate's own azimuth -- goes through
          :meth:`pose_response` and is composed on SO(3). An FOD's basis must be declared (``FOD.from_sh``,
          ``FOD.native``); a bare coefficient array is refused, since the convention cannot be inferred.
        * **surface relaxivity** (C2): ``rho`` (m/s) with the walk's diffusivity ``D`` (the pack's
          recorded value unless given); requires the boundary local time.
        * **field** (C3): ``B0`` (T) with ``b0_dir`` and the susceptibility ``chi_iso`` (required) and
          ``chi_aniso``; the pack stores the substrate's geometry-only basis and no susceptibility value,
          so these are the replay's to give; the 180 the static field refocuses at is the waveform's own
          (``waveform.rf.refocus_time``); requires the field tier.

        ``compartment`` restricts the ensemble mean to one pool id (or a boolean walker mask).
        A tier that is requested but not carried raises rather than returning a plausible number.
        """
        from .compression import read_position_coeffs
        from ._replay_kernel import gradient_phase, se_gate
        waveform = waveform.waveform if hasattr(waveform, "waveform") else waveform
        if isinstance(waveform, Protocol):                # a multi-TE scheme: each sequence replayed, placed at its rows
            kw = dict(tissue=tissue, T2=T2, T1=T1, rho=rho, D=D, B0=B0, b0_dir=b0_dir, chi_iso=chi_iso,
                      chi_aniso=chi_aniso, compartment=compartment, orientation=orientation, complex_signal=complex_signal)
            return waveform.scatter([self.replay(seq, **kw) for seq in waveform])
        dist = _as_distribution(orientation)
        if dist is not None:
            S = self.pose_response(waveform, tissue=tissue, T2=T2, T1=T1, rho=rho, D=D, B0=B0, b0_dir=b0_dir,
                                    chi_iso=chi_iso, chi_aniso=chi_aniso,
                                    compartment=compartment).compose(dist)
            return S if complex_signal else np.abs(S)
        P = self._prepare(waveform, tissue=tissue, T2=T2, T1=T1, rho=rho, D=D, B0=B0, b0_dir=b0_dir, chi_iso=chi_iso,
                          chi_aniso=chi_aniso, orientation=orientation, compartment=compartment)
        phi = self._walker_phases(P, waveform)
        S = (P["ew"][:, None] * np.exp(1j * phi)).sum(0) / P["norm"]
        return S if complex_signal else np.abs(S)

    def walker_signals(self, waveform, *, tissue="nominal", T2=None, T1=None, rho=None, D=None, B0=None,
                       b0_dir=(0.0, 0.0, 1.0), chi_iso=None, chi_aniso=0.0, orientation=None, compartment=None,
                       b1_scale=None, off_resonance_T=None):
        """The replay **before the ensemble mean**: ``(w, ew, E)`` with ``w`` the walkers' statistical weights
        ``(n_w,)``, ``ew`` those weights with the relaxation and surface terms applied, and ``E`` the complex
        signal factor of every walker at every measurement, ``(n_w, n_meas)``.

        :meth:`replay` is ``(ew[:, None] * E).sum(0) / w.sum()``; any other grouping of the walkers -- by the
        voxel they started in, which is what partitions one walk into a phantom (RPH.md) -- is the same sum
        over its members with its own normaliser. Knobs resolve as in :meth:`replay`. With ``b1_scale`` (a
        scalar or per walker) or ``off_resonance_T`` (a scalar or per walker) given, ``E`` comes from the
        RF-aware route (:meth:`replay_bloch`): the transverse magnetisation of each walker at the readout,
        with the relaxation the route applied itself, so ``ew`` is then ``w``.
        """
        waveform = waveform.waveform if hasattr(waveform, "waveform") else waveform
        if b1_scale is None and off_resonance_T is None:
            P = self._prepare(waveform, tissue=tissue, T2=T2, T1=T1, rho=rho, D=D, B0=B0, b0_dir=b0_dir, chi_iso=chi_iso,
                              chi_aniso=chi_aniso, orientation=orientation, compartment=compartment)
            w = np.asarray(self.spin_weights, np.float64)
            return w, P["ew"], np.exp(1j * self._walker_phases(P, waveform))
        E = self.replay_bloch(waveform, b1_scale=b1_scale, off_resonance_T=off_resonance_T, tissue=tissue, T2=T2, T1=T1,
                              rho=rho, D=D, B0=B0, b0_dir=b0_dir, chi_iso=chi_iso, chi_aniso=chi_aniso,
                              orientation=orientation, compartment=compartment, complex_signal=True, per_walker=True)
        w = np.asarray(self.spin_weights, np.float64)
        return w, w, E

    def _walker_phases(self, P, waveform):
        """``(n_w, n_meas)`` accumulated phase of every walker under the prepared acquisition ``P``."""
        from .compression import read_position_coeffs
        from ._replay_kernel import gradient_phase, se_gate
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
            phi_x = GAMMA * dt * (dB * se_gate(n_t, dt, waveform.rf.refocus_time)[None, :]).sum(1)    # (n_w,)
            phi = gradient_phase(Geff, pos, dt).T + phi_x[:, None]                             # (n_w, n_meas)
        return phi

    def replay_bloch(self, waveform, *, b1_scale=None, off_resonance_T=None, tissue="nominal", T2=None, T1=None,
                     rho=None, D=None, B0=None, b0_dir=(0.0, 0.0, 1.0), chi_iso=None, chi_aniso=0.0,
                     orientation=None, compartment=None, jax=False, complex_signal=False, per_walker=False):
        """The RF-aware replay: each walker's magnetisation vector propagated through the actual sequence
        operators on this pack's walk (:func:`~dmipy_sim.replay.trajectories.replay_bloch`).

        This route reads the waveform's PHYSICAL ``G`` and applies its pulses; :meth:`replay` reads ``G_eff``
        with the pulses folded in. The magnitude route of :meth:`replay` assumes ideal pulses and reads the
        signal as a phase sum, so it cannot carry anything that acts on the magnetisation vector: a flip angle that is not nominal
        (``b1_scale``), a finite pulse, a pulse train's coherence pathways. Those are what this route is for,
        and it costs a propagation per piece instead of a projection.

        Knobs are the same as :meth:`replay` and resolve the same way, nominal by default; the pose rotates the
        acquisition and the field direction as it does there. ``b1_scale`` scales every flip angle, as a scalar
        or per walker. ``off_resonance_T`` is a **uniform** static field offset (a field-map value, in T) every
        walker precesses in through the actual pulses -- what a macroscopic layer of a phantom is (RPH.md 5.1).
        ``per_walker`` returns every walker's transverse magnetisation at the readout, ``(n_w, n_meas)`` complex,
        unweighted, instead of the ensemble mean (single-readout sequences).
        """
        from .trajectories import replay_bloch as _rb, replay_bloch_jax as _rbj
        from .compression import decode_occupancy, decode_boundary_bridge
        P = self._prepare(waveform, tissue=tissue, T2=T2, T1=T1, rho=rho, D=D, B0=B0, b0_dir=b0_dir,
                          chi_iso=chi_iso, chi_aniso=chi_aniso, orientation=orientation, compartment=compartment,
                          relaxation=False, surface=False)
        rf = waveform.rf
        if not rf:
            raise ValueError("the Bloch route replays an RF schedule and this sequence carries none. Without a "
                             "pulse there is nothing this route adds over replay().")
        ch, dt, n_t = P["ch"], P["dt"], P["n_t"]
        pos = self.positions()
        kw = dict(weights=P["ew"] / P["norm"], echo_steps=_echo_saves(waveform, dt))
        if b1_scale is not None:
            kw["b1_scale"] = b1_scale
        T2v, T1v = P["T2"], P["T1"]
        if T2v is not None or T1v is not None:
            comp = decode_occupancy(self.arrays, ch["compartment"])["comp"]
            n_ids = int(np.max(comp)) + 1
            # the Bloch route reads a rate as 1/T, so "no decay in this pool" is an infinite time, not a zero
            # one; a zero would make the rate infinite and return an identically dark signal
            def per_pool(v, what):
                out = self._by_pool(v, what)
                if out is None:
                    return [np.inf] * n_ids
                return [np.inf if t is None or float(t) <= 0.0 else float(t) for t in out]
            kw.update(comp_traj=comp, T2_per_comp=per_pool(T2v, "T2"), T1_per_comp=per_pool(T1v, "T1"))
        if P["rho"] is not None and float(P["rho"]) != 0.0:
            D_walk = self.diffusivity if P["D"] is None else P["D"]
            if D_walk is None:
                raise ValueError("rho needs the walk's diffusivity: the pack did not record it, pass D=")
            if "blt_bridge_dst" not in self.arrays:
                raise ValueError("surface relaxivity was requested but this pack carries no C2 channel")
            meta = dict(ch.get("boundary_local_time") or {})
            meta.setdefault("n_t", n_t)
            meta.setdefault("K", int(np.asarray(self.arrays["blt_bridge_dst"]).shape[1]))
            kw.update(dlog_boundary_unit=decode_boundary_bridge(self.arrays, meta),
                      surface_relaxivity=float(P["rho"]), D=float(D_walk))
        extra = None
        if P["B0"] is not None:
            extra = GAMMA * dt * self._field_along_walk(P, pos)
        if off_resonance_T is not None and np.any(np.asarray(off_resonance_T, np.float64) != 0.0):
            off = np.asarray(off_resonance_T, np.float64).reshape(-1)
            if off.size not in (1, pos.shape[0]):
                raise ValueError(f"off_resonance_T is a scalar or one value per walker ({pos.shape[0]}); got {off.shape}")
            uniform = GAMMA * dt * np.broadcast_to(off[:, None], (pos.shape[0], n_t))
            extra = uniform if extra is None else extra + uniform
        if extra is not None:
            kw["extra_phase_per_step"] = extra
        if per_walker:
            if kw.get("echo_steps") is not None:
                raise ValueError("per_walker reads each walker at the readout of a single-echo sequence")
            kw.update(echo_steps=[n_t - 1], echo_per_walker=True)
            out = (_rbj if jax else _rb)(pos, dt, P["G"], P["dt_wf"], rf, **kw)
            E = np.asarray(out[0] if isinstance(out, tuple) else out)                # (n_meas, 1, n_w)
            E = np.asarray(E).reshape(E.shape[0], -1, pos.shape[0])[:, -1, :].T     # (n_w, n_meas)
            return E if complex_signal else np.abs(E)
        out = (_rbj if jax else _rb)(pos, dt, P["G"], P["dt_wf"], rf, **kw)
        S = np.asarray(out[0] if isinstance(out, tuple) else out)
        return S if complex_signal else np.abs(S)

    def _field_along_walk(self, P, pos):
        """The susceptibility off-resonance each walker sees at each save, from whichever C3 route the pack
        carries: the compressed path coefficients, or the stored field basis sampled along the walk."""
        from .bank import susc_path_decode, susc_path_field
        from ..fields.susceptibility_field import assemble_field, sample_grid
        if not self.has_field:
            raise ValueError("B0 was given but the pack carries no field tier (C3)")
        if P["chi_iso"] is None:
            raise ValueError("B0 was given without chi_iso: the pack carries the substrate's field basis, "
                             "not a susceptibility; give chi_iso (and chi_aniso) at replay")
        gm = P["ch"]["susceptibility_grid"]
        pm = P["ch"].get("susceptibility_path")
        if pm is not None:
            b, _ = susc_path_decode(self.arrays, pm, n_w=P["n_w"])
            return susc_path_field(b, P["b0_dir"], B0=float(P["B0"]), chi_iso=float(P["chi_iso"]),
                                   chi_aniso=P["chi_aniso"], has_aniso=bool(gm.get("has_aniso")))
        basis = {"iso_local": np.asarray(self.arrays["susc_grid_iso_local"], np.float64),
                 "iso_P": np.asarray(self.arrays["susc_grid_iso_P"], np.float64),
                 "aniso_G": (np.asarray(self.arrays["susc_grid_aniso_G"], np.float64)
                             if "susc_grid_aniso_G" in self.arrays else None),
                 "shape": tuple(gm["shape"]), "voxel_size": np.asarray(gm["voxel_size"], float)}
        return sample_grid(assemble_field(basis, P["b0_dir"], B0=float(P["B0"]), chi_iso=float(P["chi_iso"]),
                                          chi_aniso=P["chi_aniso"]),
                           pos, np.asarray(gm["origin"], float), gm["voxel_size"], periodic=False)

    def _prepare(self, waveform, *, tissue, T2, T1, rho, D, B0, b0_dir, chi_iso, chi_aniso, orientation, compartment,
                 relaxation=True, surface=True):
        """Everything a replay resolves before it reads positions: the waveform's exact per-save weights (rotated
        into the substrate frame when a pose is given), the knobs (nominal, a Tissue, or explicit), the per-walker
        weights with the relaxation and surface terms applied, and the compartment selection."""
        from .compression import require_position_method, decode_occupancy, relaxation_logweight
        from ._replay_kernel import effective_gradient, bin_gate
        require_position_method(self.method)
        # two gradients: the PHYSICAL one (``G``, for the vector-Bloch route, which applies the pulses itself)
        # and the EFFECTIVE one (``G_eff``, for the scalar routes, the pulses folded in)
        waveform = waveform.waveform if hasattr(waveform, "waveform") else waveform
        if not isinstance(waveform, ScannerSequence):
            raise TypeError(f"a replay takes a ScannerSequence (a bare gradient array says nothing about its pulses); "
                            f"got {type(waveform).__name__}")
        G = np.asarray(waveform.G, np.float64)
        G_eff = np.asarray(waveform.G_eff, np.float64)
        dt_wf = float(waveform.dt)
        n_t, dt = self.n_t, self.dt
        chi = waveform.chi_perp
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
            G, G_eff = G @ R, G_eff @ R                                   # R^T g per sample: stored coordinates
            b0_dir = tuple(np.asarray(R, float).T @ np.asarray(b0_dir, float))
        Geff = effective_gradient(G_eff, dt_wf, n_t, dt)                 # exact per-save weights of the effective gradient
        logw = np.zeros(n_w)
        if (T2 is not None or T1 is not None) and relaxation:
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
        if rho is not None and float(rho) != 0.0 and surface:
            D_walk = self.diffusivity if D is None else D
            if D_walk is None:
                raise ValueError("rho needs the walk's diffusivity: the pack did not record it, pass D=")
            logw = logw + surface_logweight(self.arrays, float(rho) / float(D_walk),
                                            ch.get("boundary_local_time"), chi)      # raises without C2
        ew = w * np.exp(logw)
        norm = w.sum()
        ew, norm = self._select(compartment, ew, norm, w, ch, n_w)
        return dict(G=G, Geff=Geff, dt=dt, n_t=n_t, dt_wf=dt_wf, ch=ch, n_w=n_w, w=w, ew=ew, norm=norm, B0=B0,
                    b0_dir=b0_dir, chi_iso=chi_iso, chi_aniso=chi_aniso, T2=T2, T1=T1, rho=rho, D=D)

    def pose_response(self, waveform, *, tissue="nominal", T2=None, T1=None, rho=None, D=None, B0=None,
                      b0_dir=(0.0, 0.0, 1.0), chi_iso=None, chi_aniso=0.0, compartment=None,
                      method="auto", keep=None):
        """The pack's response over every pose of its substrate, for one acquisition: a :class:`PoseResponse` whose
        coefficients a voxel's orientation distribution contracts against (RPH.md 6).

        **Closed form, by default (#197).** For a single-direction encoding the response is a sum of plane waves in
        each walker's rotated moment, and its harmonics are the Rayleigh expansion, computed per walker with no
        rotation ever evaluated; with a field the response is that expansion times the field factor's, coupled
        with the real Clebsch-Gordan tables. The band follows the walkers' phase amplitudes by construction and
        nothing is chosen. ``keep = (lmax, nmax)`` restricts what is computed to what the composition retains: an
        ODF or peaks composition keeps ``n = 0``, a frame keeps everything; ``None`` in either slot means the
        response's own band.

        **The quadrature, by name.** An encoding whose moment matrix is not rank one -- a b-tensor or multi-axis
        waveform -- has no plane-wave expansion, and takes the sampled route: the response evaluated on an SO(3)
        quadrature sized to its phase amplitude, projected, and certified off the grid (``misfit``, a worst case).
        ``method="quadrature"`` asks for it explicitly; ``method="closed"`` refuses what the closed form cannot
        take rather than falling back. :attr:`PoseResponse.route` says which was used.

        Knobs are the same as :meth:`replay` and resolve the same way; the pose is not one of them, since every
        pose is what is being expanded.
        """
        P = self._prepare(waveform, tissue=tissue, T2=T2, T1=T1, rho=rho, D=D, B0=B0, b0_dir=b0_dir, chi_iso=chi_iso,
                          chi_aniso=chi_aniso, orientation=None, compartment=compartment)
        if method not in ("auto", "closed", "quadrature"):
            raise ValueError("method is 'auto', 'closed' (the per-walker Rayleigh expansion) or 'quadrature'")
        if method != "quadrature":
            closed = self._pose_coeffs_closed(P, waveform, keep=keep)
            if closed is not None:
                return closed
            if method == "closed":
                raise ValueError("the closed-form pose expansion needs a single-direction encoding on every measurement "
                                 "and no field: this acquisition or replay has neither, so use method='quadrature'")
        return self._pose_coeffs(P, waveform, keep=keep)

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
    def substrate_frame(self):
        """The substrate's intrinsic frame (RPK.md 4.2): a ``(3, 3)`` right-handed basis in stored coordinates, column
        3 the primary structural axis, ``d_stored = F d_canonical``. From ``walk_params.substrate_frame`` when the
        producer declared it, else from the embedded spec's ``frame`` (axis, optional in-plane), else the identity --
        the pack's positions are then taken to be in the canonical frame already."""
        from .bank import frame_from_axis
        wp = self.meta.get("walk_params", {}) or {}
        F = wp.get("substrate_frame")
        if F is None:
            F = self.meta.get("substrate_frame")
        if F is not None:
            F = np.asarray(F, np.float64).reshape(3, 3)
        else:
            spec = self.substrate
            fr = getattr(spec, "frame", None) if spec is not None else None
            if fr is None:
                return np.eye(3)
            a = np.asarray(fr.axis, np.float64); a = a / np.linalg.norm(a)
            F = np.asarray(frame_from_axis(a), np.float64).reshape(3, 3)
        if not np.allclose(F @ F.T, np.eye(3), atol=1e-6) or np.linalg.det(F) < 0:
            raise ValueError("the pack's substrate_frame is not a proper rotation")
        return F

    @property
    def frame_axis(self):
        """The substrate's own axis in stored coordinates: column 3 of :attr:`substrate_frame`."""
        return self.substrate_frame[:, 2].copy()

    def _rotation_of(self, orientation):
        """The rotation taking STORED coordinates to the lab: from a 3x3 pose of the canonical substrate frame
        (``R``, so stored -> lab is ``R F^T`` with ``F`` the pack's :attr:`substrate_frame`), or from the lab
        direction the substrate's axis points along (the azimuth left as the frame's)."""
        from .so3 import rotation_of
        o = np.asarray(orientation, float)
        if o.shape == (3, 3):
            if not np.allclose(o @ o.T, np.eye(3), atol=1e-6) or np.linalg.det(o) < 0:
                raise ValueError("orientation must be a proper rotation matrix (R R^T = I, det +1)")
            return o @ self.substrate_frame.T
        if o.shape == (3,):
            return rotation_of(o, self.frame_axis)
        raise ValueError("orientation is a (3, 3) rotation, a (3,) axis direction, or a distribution of poses "
                         "(dmipy_sim.replay.so3.Distribution, or an FOD read as an axis density)")

    def _pose_coeffs_closed(self, P, waveform, keep=None, tol=1e-8, l_cap=64):
        """The pose expansion in closed form (#197): the response of a single-direction encoding is a sum of plane
        waves in the rotated moment of each walker, and a plane wave's harmonics are the Rayleigh expansion.

        Measurement ``i`` plays ``G_i(t) = g_i s_i(t)``; walker ``w``'s phase at pose ``R`` is ``kappa g^ . R m^``
        with ``m_w = gamma sum_t s_i(t) r_w(t) dt`` (the moment, in the canonical frame) and ``kappa = |g||m|``. Then

            exp(i kappa g^.R m^) = 4 pi sum_l i^l j_l(kappa) sum_m Y_lm(g^) sum_n M^l(R)[m, n] Y_ln(m^)

        so in the basis ``sqrt(2l+1) M^l(R)[m, n]`` the ``(l, m, n)`` coefficient of the ensemble is

            c^l_mn = 4 pi i^l / sqrt(2l+1) * Y_lm(g^_i) * sum_w ew_w j_l(kappa_w) Y_ln(m^_w) / norm

        -- one contraction over walkers per order, no rotation ever evaluated. ``j_l(kappa)`` dies above
        ``l ~ kappa``, so the band is the largest phase amplitude and nothing is chosen: orders are added until
        their weighted Bessel tail is below ``tol``. ``keep`` restrains the band as before: an ODF or peaks
        composition keeps ``n = 0`` only, which here is the Legendre polynomial of the moment's angle to the
        substrate axis and never a roll quadrature.

        **The field.** The susceptibility phase of a walker is ``a_w + u^T A_w u`` in the field direction
        ``u = R^T b`` (:meth:`_field_quadratic`), a function on the sphere whose harmonics ``a_l'm'`` are read
        off a product quadrature exact to its own band (the band follows the phase amplitude ``|A_w|``, orders
        added until the energy above them is below ``tol``). The response is then the product of two expansions,
        contracted with the real coupling tables (:func:`so3.coupling`) on the lab index (``Y_lm(g^)`` with
        ``Y_l'n'(b^)``) and on the body index (``j_l Y_ln(m^)`` with ``a_l'm'``); no ``g x B0`` frame exists.
        Returns None when a measurement is not single-direction (a b-tensor encoding): that takes the quadrature.
        """
        from scipy.special import spherical_jn
        from . import so3
        from .compression import read_position_coeffs
        Geff, dt, n_t, ew, norm = P["Geff"], P["dt"], P["n_t"], P["ew"], P["norm"]
        n_meas, n_w = Geff.shape[0], ew.shape[0]
        field = self._field_quadratic(P, waveform) if P["B0"] is not None else None   # (a_w, A_w) or None
        G = np.asarray(Geff, np.float64)
        g_hat = np.zeros((n_meas, 3)); s_wave = np.zeros((n_meas, n_t))
        for i in range(n_meas):
            Gi = G[i]
            if not np.any(Gi):
                g_hat[i] = (0.0, 0.0, 1.0)                                     # a b = 0 row: no phase at any pose
                continue
            _u, sv, vt = np.linalg.svd(Gi, full_matrices=False)
            if sv[1] > 1e-6 * sv[0]:                                    # G is stored float32; a direction is one to that
                return None                                                    # rank > 1: not a single direction
            g, sw = vt[0], Gi @ vt[0]
            lead = int(np.flatnonzero(np.abs(sw) > 1e-6 * np.abs(sw).max())[0])
            if sw[lead] < 0:                       # one spelling of (direction, waveform): the first lobe positive
                g, sw = -g, -sw
            g_hat[i] = g; s_wave[i] = sw
        # the body of a coefficient depends on the waveform's shape and amplitude only, never on its direction:
        # measurements that play the same s_i(t) -- a shell -- share one body, and their directions enter as
        # harmonics afterwards. Group by the played waveform, exactly, and contract once per group.
        group, first = _group_waveforms(s_wave, rtol=1e-5)                   # float32 G: 1e-5 is the same waveform
        n_grp = len(first)
        s_grp = s_wave[first]                                                  # (n_grp, n_t)
        C = read_position_coeffs(self.arrays, dtype=np.float64).reshape(n_w, -1)
        e = np.eye(3)
        m = np.empty((n_w, n_grp, 3))
        for b_ in range(3):                                                    # m_w[b] = gamma sum_t s(t) r_w(t)_b dt
            W = _compile_effective(s_grp[:, :, None] * e[b_][None, None, :], dt, self.K, n_t)
            m[:, :, b_] = C @ W
        m = m @ self.substrate_frame                                           # stored -> canonical: F^T m, per walker
        kappa = np.linalg.norm(m, axis=2)                                      # (n_w, n_grp), radians
        safe = np.where(kappa > 0, kappa, 1.0)
        m_hat = m / safe[:, :, None]
        m_hat[kappa == 0] = (0.0, 0.0, 1.0)
        w = np.asarray(ew, np.float64) / float(norm)
        # the band: orders until the weighted Bessel tail is below tol for the worst group
        k_max = float(kappa.max()) if kappa.size else 0.0
        L = int(np.ceil(k_max)) + 2
        while L < l_cap:
            tail = (2 * (L + 1) + 1) * (np.abs(w)[:, None] * np.abs(spherical_jn(L + 1, kappa))).sum(0).max()
            if tail < tol:
                break
            L += 1
        # the field factor's harmonics per walker, and the band the product reaches
        if field is None:
            L_f, F_sh = 0, None
        else:
            F_sh, L_f = self._field_harmonics(field, tol=tol, l_cap=l_cap)      # (n_w, (L_f+1)^2) complex
        L_tot = L + L_f
        want_l, want_n = (None, None) if keep is None else (keep[0], keep[1])
        keep_l = L_tot if want_l is None else min(int(want_l), L_tot)
        keep_n = L_tot if want_n is None else min(int(want_n), L_tot)
        n_feat = so3.n_so3_coeffs(keep_l, keep_n)
        coeffs = np.zeros((n_meas, n_feat), np.complex128)
        cos_z = m_hat[:, :, 2]
        J = [spherical_jn(l, kappa) for l in range(L + 1)]                      # (n_w, n_grp) per order
        Yg = so3.real_sh(L, g_hat, full=True)                                   # (n_meas, (L+1)^2): the lab side
        if field is None:
            # ---- gradient only: one body per order and group, outer product with the direction harmonics
            bodies = [None] * (keep_l + 1)                                      # per order: (n_grp, 2k+1)
            if keep_n == 0:                                                     # n = 0 only: the Legendre of the angle to the axis
                for l in range(keep_l + 1):
                    bodies[l] = np.sqrt((2 * l + 1) / (4 * np.pi)) * ((w[:, None] * J[l]) * _legendre(l, cos_z)).sum(0)[:, None]
            else:
                # the body side: every walker's moment direction's harmonics, all orders in one recurrence pass,
                # in chunks of groups sized to ~256 MB
                for l in range(keep_l + 1):
                    bodies[l] = np.empty((n_grp, 2 * (so3._n_cols(l, keep_n) // 2) + 1))
                n_cols = (keep_l + 1) ** 2
                step = max(1, int(2.5e8 / (8 * n_w * n_cols)))
                for lo in range(0, n_grp, step):
                    sl = slice(lo, min(lo + step, n_grp))
                    nc = sl.stop - sl.start
                    Y = so3.real_sh(keep_l, m_hat[:, sl, :].reshape(-1, 3), full=True).reshape(n_w, nc, n_cols)
                    for l in range(keep_l + 1):
                        k = so3._n_cols(l, keep_n) // 2
                        blk = so3.sh_block(l, True)
                        Yl = Y[:, :, blk.start + l - k:blk.start + l + k + 1]                 # (n_w, nc, 2k+1)
                        bodies[l][sl] = np.einsum("wi,wim->im", w[:, None] * J[l][:, sl], Yl)
            off = 0
            for l in range(keep_l + 1):
                k = so3._n_cols(l, keep_n) // 2
                blk = so3.sh_block(l, True)
                body_i = bodies[l][group]                                                      # (n_meas, 2k+1)
                block = (4 * np.pi * (1j ** l) / np.sqrt(2 * l + 1)) * Yg[:, blk][:, :, None] * body_i[:, None, :]
                coeffs[:, off:off + (2 * l + 1) * (2 * k + 1)] = block.reshape(n_meas, -1)
                off += (2 * l + 1) * (2 * k + 1)
        else:
            # ---- gradient x field: the outer product of the two body expansions per walker, summed over the
            # walkers per group, then coupled on both indices into the total order L_tot
            b_lab = np.asarray(P["b0_dir"], np.float64); b_lab = b_lab / np.linalg.norm(b_lab)
            Yb = so3.real_sh(L_f, b_lab[None, :], full=True)[0]                 # ((L_f+1)^2,): the field direction, lab side
            n_cols = (L + 1) ** 2
            Ym = np.empty((n_w, n_grp, n_cols))                                # the moment harmonics, all groups
            step = max(1, int(2.5e8 / (8 * n_w * n_cols)))
            for lo in range(0, n_grp, step):
                sl = slice(lo, min(lo + step, n_grp)); nc = sl.stop - sl.start
                Ym[:, sl, :] = so3.real_sh(L, m_hat[:, sl, :].reshape(-1, 3), full=True).reshape(n_w, nc, n_cols)
            offs = {}
            off = 0
            for Lc in range(keep_l + 1):
                offs[Lc] = off; off += (2 * Lc + 1) * (2 * (so3._n_cols(Lc, keep_n) // 2) + 1)
            for l in range(L + 1):
                bl = so3.sh_block(l, True)
                if l > keep_l + L_f:
                    break
                # body for every field order at once: B_all[(g, n), (l', m')] = sum_w w j_l(kappa) Y_ln(m^) a_l'm'(w),
                # one matrix product over the walkers per gradient order
                X = ((w[:, None] * J[l])[:, :, None] * Ym[:, :, bl]).reshape(n_w, -1)       # (n_w, n_grp (2l+1))
                B_all = (X.T @ F_sh.real + 1j * (X.T @ F_sh.imag)).reshape(n_grp, 2 * l + 1, -1)   # (n_grp, 2l+1, (L_f+1)^2); real products
                for lp in range(L_f + 1):
                    blp = so3.sh_block(lp, True)
                    Ls = [Lc for Lc in range(abs(l - lp), min(l + lp, keep_l) + 1)]
                    if not Ls:
                        continue
                    B = B_all[:, :, blp].reshape(n_grp, -1)                                 # (n_grp, (2l+1)(2l'+1))
                    # lab: Lam[i, (m, n')] = Y_lm(g^_i) Y_l'n'(b^)
                    Lam = (Yg[:, bl][:, :, None] * Yb[blp][None, None, :]).reshape(n_meas, -1)
                    K = so3.coupling(l, lp)
                    for Lc in Ls:
                        KL = K[Lc]                                              # ((2l+1)(2l'+1), 2Lc+1)
                        kk = so3._n_cols(Lc, keep_n) // 2
                        lab = Lam @ KL                                          # (n_meas, 2Lc+1)
                        body = B @ KL.conj()[:, Lc - kk:Lc + kk + 1]            # (n_grp, 2kk+1)
                        block = (4 * np.pi * (1j ** l) / np.sqrt(2 * Lc + 1)) * lab[:, :, None] * body[group][:, None, :]
                        o = offs[Lc]
                        coeffs[:, o:o + (2 * Lc + 1) * (2 * kk + 1)] += block.reshape(n_meas, -1)
        # what the expansion cannot hold pointwise: the orders above the band it was built to, as a bound from
        # |P_l| <= 1 -- below tol by construction
        tail = np.zeros(n_grp)
        for l in range(L + 1, L + 4):
            tail += (2 * l + 1) * (np.abs(w)[:, None] * np.abs(spherical_jn(l, kappa))).sum(0)
        out = PoseResponse(coeffs, keep_l, keep_n, misfit=tail[group], floor=1.0 / np.sqrt(n_w), phase_amplitude=k_max,
                           n_samples=0)
        out.n_bodies = n_grp                                                   # the distinct waveforms contracted
        out.field_lmax = L_f
        out.route = "closed"
        return out

    def _field_quadratic(self, P, waveform):
        """The susceptibility phase of every walker as ``a_w + u^T A_w u`` in the field direction ``u`` expressed in
        the CANONICAL frame: ``(a (n_w,), A (n_w, 3, 3))``, the gate-integrated path field basis of the pack (C3 path
        route) scaled by ``B0``, ``chi_iso``, ``chi_aniso``. Raises, as the quadrature route does, when the pack
        cannot supply it."""
        from scipy.fft import dct
        from ._replay_kernel import se_gate
        from .bank import susc_path_coeffs
        B0, chi_iso, chi_aniso = P["B0"], P["chi_iso"], P["chi_aniso"]
        if not self.has_field:
            raise ValueError("B0 was given but the pack carries no field tier (C3)")
        pm = self.meta.get("compression", {}).get("channels", {}).get("susceptibility_path")
        if pm is None:
            raise ValueError("the pose expansion with a field needs the pack's susc_path channel (C3 path route)")
        if chi_iso is None:
            raise ValueError("B0 was given without chi_iso; give chi_iso (and chi_aniso)")
        dt, n_t = P["dt"], P["n_t"]
        Cs, names = susc_path_coeffs(self.arrays, pm)
        gate_hat = dct(se_gate(n_t, dt, waveform.rf.refocus_time), type=2, norm="ortho")[:Cs.shape[2]]
        Psi = (GAMMA * dt) * np.einsum("k,wck->wc", gate_hat, Cs)               # (n_w, n_ch)
        i_p = names.index("iso_P_xx")
        i_a = names.index("aniso_G_xx") if "aniso_G_xx" in names else None
        a = float(chi_iso) * float(B0) * Psi[:, names.index("iso_local")]
        six = -float(chi_iso) * float(B0) * Psi[:, i_p:i_p + 6]
        if chi_aniso and i_a is not None:
            six = six + float(chi_aniso) * float(B0) * Psi[:, i_a:i_a + 6]
        xx, yy, zz, xy, xz, yz = six.T                                         # u^T A u = xx x^2 + ... + 2 xy x y + ...
        A = np.empty((six.shape[0], 3, 3))
        A[:, 0, 0], A[:, 1, 1], A[:, 2, 2] = xx, yy, zz
        A[:, 0, 1] = A[:, 1, 0] = xy; A[:, 0, 2] = A[:, 2, 0] = xz; A[:, 1, 2] = A[:, 2, 1] = yz
        F = self.substrate_frame                                               # stored -> canonical: A_c = F^T A F
        A = np.einsum("ab,wbc,cd->wad", F.T, A, F)
        return a, A

    def _field_harmonics(self, field, tol=1e-8, l_cap=48):
        """The harmonics of ``exp(i (a_w + u^T A_w u))`` over the sphere per walker, ``(n_w, (L'+1)^2)``, by a product
        quadrature exact to the band ``L'`` chosen from the phase amplitude: orders are added until the energy in
        the last one is below ``tol`` of the total."""
        from . import so3
        a, A = field
        amp = float(np.abs(np.linalg.eigvalsh(A)).max()) if A.size else 0.0
        Lp = int(np.ceil(2.0 * amp)) + 4
        while True:
            dirs, wq = so3.sphere_quadrature(Lp + 2, 2 * Lp + 2)
            q = np.einsum("qa,wab,qb->wq", dirs, A, dirs)                     # (n_w, n_q)
            f = np.exp(1j * (a[:, None] + q))
            Y = so3.real_sh(Lp, dirs, full=True)                               # (n_q, (Lp+1)^2)
            F = (f * wq[None, :]) @ Y                                           # (n_w, (Lp+1)^2)
            top = so3.sh_block(Lp, True)
            if (np.abs(F[:, top]) ** 2).sum(1).max() <= tol * (np.abs(F) ** 2).sum(1).max() or Lp >= l_cap:
                return F, Lp
            Lp += 4

    def _pose_coeffs(self, P, waveform, band=None, keep=None, margin=2, n_check=256, seed=0,
                     chunk=256, over=2, band_cap=12, strict=True):
        """Sample the response over rotations, project it, and check it off the grid.

        The gradient term is exact for any waveform, single- or multi-axis: the phase of walker ``w`` at pose
        ``R`` is ``<R, M_w>`` with ``M_w[a, b] = sum_t Geff[t, a] r_w[t, b]``, so the walk is contracted once
        per measurement and every pose is then a 3x3 inner product. The field term is the same second-rank form
        in the field direction carried into the substrate frame, reached through the six quadratic products of
        the rotated field direction.

        The band follows from those contractions rather than from a setting: ``||M_w||`` is the phase the pose
        modulates, in radians, and the harmonic content of ``exp(i <U, M_w>)`` reaches about that order -- the
        cutoff that makes ``j_l(x)`` negligible for ``l`` beyond ``x``. The quadrature is then oversampled past
        the projection band, because a rule exact only for the band being kept folds everything above it into
        those coefficients, and does so differently at different frames. The misfit is measured at rotations off
        that grid, where such folding cannot hide, and as a worst case rather than a spread.
        """
        from scipy.fft import dct
        from . import so3
        from .compression import read_position_coeffs
        from ._replay_kernel import se_gate
        Geff, dt, n_t, ew, norm, B0 = P["Geff"], P["dt"], P["n_t"], P["ew"], P["norm"], P["B0"]
        b0_dir, chi_iso, chi_aniso = P["b0_dir"], P["chi_iso"], P["chi_aniso"]
        n_meas, n_w = Geff.shape[0], ew.shape[0]

        C = read_position_coeffs(self.arrays, dtype=np.float64).reshape(n_w, -1)
        # The gradient term at pose R is <R, M_w> with M_w[a, b] = sum_t Geff[i, t, a] r_w[t, b]: the walk is
        # contracted against the waveform once, and every pose after that is a 3x3 inner product. Exact for a
        # multi-axis waveform too, which an expansion in one gradient direction could not take at all.
        Q = np.empty((n_w, n_meas, 3, 3))
        e = np.eye(3)
        for a in range(3):
            for b_ in range(3):
                W = _compile_effective(Geff[:, :, a][:, :, None] * e[b_][None, None, :], dt, self.K, n_t)
                Q[:, :, a, b_] = C @ W

        Psi = names = i_p = i_a = None
        if B0 is not None:
            if not self.has_field:
                raise ValueError("B0 was given but the pack carries no field tier (C3)")
            pm = self.meta.get("compression", {}).get("channels", {}).get("susceptibility_path")
            if pm is None:
                raise ValueError("the pose expansion with a field needs the pack's susc_path channel (C3 path route)")
            if chi_iso is None:
                raise ValueError("B0 was given without chi_iso; give chi_iso (and chi_aniso)")
            from .bank import susc_path_coeffs
            Cs, names = susc_path_coeffs(self.arrays, pm)
            gate_hat = dct(se_gate(n_t, dt, waveform.rf.refocus_time), type=2, norm="ortho")[:Cs.shape[2]]
            Psi = (GAMMA * dt) * np.einsum("k,wck->wc", gate_hat, Cs)               # (n_w, n_ch)
            i_p = names.index("iso_P_xx")
            i_a = names.index("aniso_G_xx") if "aniso_G_xx" in names else None
        b = np.asarray(b0_dir, float); b = b / np.linalg.norm(b)

        # a pose R is a rotation of the CANONICAL substrate frame: the stored walk is first turned into it
        # (F^T, RPK.md 4.2) and then by R -- the identity for a pack whose frame is the identity
        Qf_axis = self.substrate_frame.T

        def response(R):
            """The ensemble signal of every measurement at every one of these poses."""
            E = np.empty((R.shape[0], n_meas), np.complex128)
            for lo in range(0, R.shape[0], int(chunk)):
                sl = slice(lo, min(lo + int(chunk), R.shape[0]))
                Rc = R[sl] @ Qf_axis
                if Psi is None:
                    Ew = np.broadcast_to(ew[None, :].astype(np.complex128), (Rc.shape[0], n_w))
                else:
                    bs = np.einsum("nji,j->ni", Rc, b)                              # the field in the substrate frame
                    Qf = np.stack([bs[:, 0] ** 2, bs[:, 1] ** 2, bs[:, 2] ** 2, 2 * bs[:, 0] * bs[:, 1],
                                   2 * bs[:, 0] * bs[:, 2], 2 * bs[:, 1] * bs[:, 2]], axis=1)
                    phi_chi = float(chi_iso) * float(B0) * (Psi[:, names.index("iso_local")][None, :]
                                                            - Qf @ Psi[:, i_p:i_p + 6].T)
                    if chi_aniso and i_a is not None:
                        phi_chi = phi_chi + float(chi_aniso) * float(B0) * (Qf @ Psi[:, i_a:i_a + 6].T)
                    Ew = np.exp(1j * phi_chi) * ew[None, :]
                for i in range(n_meas):
                    E[sl, i] = (Ew * np.exp(1j * np.einsum("nab,wab->nw", Rc, Q[:, i]))).sum(1) / norm
            return E

        # the phase the pose modulates, per measurement: the sampling band follows this, not a setting
        amp = np.linalg.norm(Q.reshape(n_w, n_meas, 9), axis=2)
        phi_amp = float(np.percentile(amp, 95)) if amp.size else 0.0
        if band is None:
            S_L = int(max(int(np.ceil(phi_amp)) + int(margin), 2))
        else:
            S_L = int(band[0]) if np.ndim(band) else int(band)
        if S_L > int(band_cap):
            raise ValueError(
                f"this acquisition sweeps {phi_amp:.1f} radians of phase on this pack, so its pose response "
                f"reaches order ~{S_L}, beyond the cap of {band_cap}. That is a real cost, not a setting: the "
                f"expansion is worth building to share one walk over many poses, and at this sharpness a direct "
                f"replay per pose (orientation=R) is the exact route for it.")
        S_N = S_L
        # `keep` may leave either index open with None, meaning "whatever the response carries"
        want_l, want_n = (None, None) if keep is None else (keep[0], keep[1])
        keep_l = S_L if want_l is None else min(int(want_l), S_L)
        keep_n = S_N if want_n is None else min(int(want_n), S_N)

        # sample beyond the retained band and project onto it, in blocks: a rule exact only for what is kept
        # folds everything above it into those coefficients (and differently at different frames)
        Rq, wq, _dirs, _rolls = so3.so3_quadrature(S_L + int(over), S_N + int(over))
        n_feat = so3.n_so3_coeffs(keep_l, keep_n)
        coeffs = np.zeros((n_feat, n_meas), np.complex128)
        for lo in range(0, Rq.shape[0], int(chunk)):
            sl = slice(lo, min(lo + int(chunk), Rq.shape[0]))
            A = so3.so3_design(keep_l, Rq[sl], keep_n)
            coeffs += (A * wq[sl, None]).T @ response(Rq[sl])
        floor = 1.0 / np.sqrt(n_w)

        Rc, Ac = so3.haar_design(keep_l, keep_n, int(n_check), int(seed))
        if (keep_l, keep_n) == (S_L, S_N):
            # the retained band is the whole response: certify pointwise, worst case, off the grid
            misfit = np.abs(Ac @ coeffs - response(Rc)).max(axis=0)
        else:
            # only part of the response is retained, so a pointwise comparison is not the question. What has to
            # hold is that the retained coefficients are alias-free: refine the grid and require them to stand.
            Rf, wf, _d, _r = so3.so3_quadrature(S_L + 2 * int(over) + 1, S_N + 2 * int(over) + 1)
            fine = np.zeros_like(coeffs)
            for lo in range(0, Rf.shape[0], int(chunk)):
                sl = slice(lo, min(lo + int(chunk), Rf.shape[0]))
                A = so3.so3_design(keep_l, Rf[sl], keep_n)
                fine += (A * wf[sl, None]).T @ response(Rf[sl])
            misfit = np.abs(Ac @ (coeffs - fine)).max(axis=0)
        if misfit.max() > floor:
            msg = (
                f"this pack's pose response is not represented at (lmax, nmax) = ({keep_l}, {keep_n}): the worst "
                f"case away from the projection's own grid is {misfit.max():.4f} in signal units, against the "
                f"pack's Monte-Carlo floor of {floor:.4f}. The response's phase amplitude is {phi_amp:.1f} rad, "
                f"so it reaches about that order; retain more of it. Composing "
                f"here would return a plausible wrong number rather than a wrong-looking one.")
            if strict:
                raise ValueError(msg)
            import warnings
            warnings.warn(msg, UserWarning, stacklevel=3)
        return PoseResponse(coeffs.T, keep_l, keep_n, misfit, floor, phi_amp, Rq.shape[0] + Rc.shape[0])


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
    def r0(self):
        """``(n_walkers, 3)``: where every walker **started**, in metres, read from the stored coefficients.

        The position codec keeps ``r(0)`` as an exact entry rather than a reconstructed band, so this is the
        stored value to float32 (of order 0.1 nm on a millimetre coordinate), identical at every ``K``, and it
        costs one slice of the coefficient block -- no path is decoded. It is what partitions a walk into voxels
        (RPH.md: ``voxel = floor((r0 - origin) / voxel_size)``), which is why it must not depend on the band.
        """
        from .compression import read_position_coeffs, require_position_method
        require_position_method(self.method)
        return np.ascontiguousarray(read_position_coeffs(self.arrays, dtype=np.float64)[:, 0, :])

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



def _as_distribution(orientation):
    """``orientation=`` as a distribution of poses, or None when it names a single pose.

    An :class:`~dmipy_sim.replay.fod.FOD` is read as an axis density with no statement about the substrate's own
    azimuth, which is what an ODF says.
    """
    from .fod import FOD
    from .so3 import Distribution
    if isinstance(orientation, Distribution):
        return orientation
    if isinstance(orientation, FOD):
        return Distribution.axis_density(orientation)
    return None


def _echo_saves(waveform, dt_pack):
    """The pack-grid save indices of a multi-echo readout, or ``None`` when the sequence is read at its last
    sample (the Bloch replay then returns the final magnetisation)."""
    readout = tuple(int(i) for i in waveform.readout)
    if readout == (waveform.n_t - 1,):
        return None
    return [int(round(i * float(waveform.dt) / float(dt_pack))) for i in readout]


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


def _group_waveforms(s, rtol=1e-5):
    """Group rows of ``s (n, n_t)`` that are the same waveform to ``rtol`` of the largest amplitude: ``(group (n,),
    first (n_grp,))``. A tolerance, not a rounding, so two rows a rounding boundary apart stay together."""
    s = np.asarray(s, np.float64)
    scale = float(np.abs(s).max()) or 1.0
    tol = rtol * scale
    group = np.full(s.shape[0], -1, np.int64)
    first = []
    for i in range(s.shape[0]):
        if group[i] >= 0:
            continue
        same = np.flatnonzero((group < 0) & (np.abs(s - s[i]).max(axis=1) <= tol))
        group[same] = len(first)
        first.append(i)
    return group, np.asarray(first, np.int64)


def _legendre(l, x):
    """``P_l(x)`` by the three-term recurrence, vectorised over ``x``."""
    x = np.asarray(x, np.float64)
    if l == 0:
        return np.ones_like(x)
    p0, p1 = np.ones_like(x), x.copy()
    for k in range(1, l):
        p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
    return p1


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
