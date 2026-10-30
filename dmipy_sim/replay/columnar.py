"""The columnar layout of a replay pack (dmipy-sim#290), the reader being :class:`dmipy_replay.replay.columnar.ColumnarPack`:
:class:`ColumnarPack` is that reader with this package's :class:`~dmipy_sim.replay.replay.ReplayPack` as its views, the JAX
voxel sums as an image backend (the default here), and the image of a :class:`~dmipy_sim.replay.study.Study`
(:meth:`ColumnarPack.image_study`). The layout is written by :mod:`dmipy_sim.fill.consolidate`.

    pack = open_columnar("hf://SubstrateCommons/disco-replay/disco")      # or ReplayPack.open(...)
    plan = pack.plan(seq, tissue=..., scanner=...)                          # bands, modes, tiers, bytes: before any transfer
    view = pack.view(K=32, voxels=[(20, 20, 20)])                           # one voxel's rows at 32 bands: an ordinary ReplayPack
    S, floor, plan = pack.image([seq], settings=[(None, None), (tissue, 3.0)])   # every setting's volume from one pass
"""
from __future__ import annotations
import time

import numpy as np

from dmipy_replay.replay.columnar import ColumnarPack as _ColumnarPack, Source  # noqa: F401

__all__ = ["ColumnarPack", "open_columnar", "Source"]

_ROWS, _SEGS = 1 << 16, 256                                     # rows and voxel runs padded to multiples: few compiled shapes


def _jax_voxel_sums():
    """The JAX voxel sums (``(phi, ew, starts) -> (n_seg, n_meas)`` complex128, :attr:`ColumnarPack.IMAGE_BACKENDS`):
    ``sum_rows ew exp(i phi)`` over each run of rows that begins at ``starts``, a ``segment_sum`` in double precision
    on the JAX device, the rows and runs padded to multiples so that a few shapes compile."""
    import functools
    import jax
    import jax.numpy as jnp

    @functools.partial(jax.jit, static_argnums=3)
    def voxel_sums(phi, ew, seg, n_seg):
        return jax.ops.segment_sum(jnp.exp(1j * phi) * ew[:, None], seg, num_segments=n_seg)

    def sums(phi, ew, starts):
        was_x64 = bool(jax.config.read("jax_enable_x64"))
        jax.config.update("jax_enable_x64", True)
        try:
            n, n_seg = len(phi), len(starts)
            seg = np.repeat(np.arange(n_seg), np.diff(np.r_[starts, n]))
            n_pad = -(-n // _ROWS) * _ROWS; s_pad = -(-(n_seg + 1) // _SEGS) * _SEGS   # a spare segment takes the padding
            seg_p = np.full(n_pad, n_seg, np.int32); seg_p[:n] = seg
            p = np.zeros((n_pad, phi.shape[1])); p[:n] = phi
            e = np.zeros(n_pad); e[:n] = ew
            return np.asarray(voxel_sums(jnp.asarray(p), jnp.asarray(e), jnp.asarray(seg_p), s_pad))[:n_seg]
        finally:
            jax.config.update("jax_enable_x64", was_x64)
    return sums


class ColumnarPack(_ColumnarPack):
    """:class:`dmipy_replay.replay.columnar.ColumnarPack` (the plan, the views, the image) with this package's
    :class:`~dmipy_sim.replay.replay.ReplayPack` as its views, ``"jax"`` among :attr:`IMAGE_BACKENDS` and the default of
    :meth:`image`, and the image of a :class:`~dmipy_sim.replay.study.Study` (:meth:`image_study`)."""

    IMAGE_BACKENDS = dict(_ColumnarPack.IMAGE_BACKENDS, jax=_jax_voxel_sums)

    def __init__(self, uri, *, revision="main", workers=8):
        from .replay import ReplayPack
        super().__init__(uri, revision=revision, workers=workers)
        self.pack_class = ReplayPack

    def image(self, seq, *, tissue=None, scanner=None, settings=None, tol=0.25, voxels=None, chunk_rows=2_000_000,
              backend="jax", progress=None):
        """:meth:`dmipy_replay.replay.columnar.ColumnarPack.image` on the ``backend`` (JAX by default), or for a
        :class:`~dmipy_sim.replay.study.Study` :meth:`image_study`."""
        from .study import Study
        if isinstance(seq, Study):
            return self.image_study(seq, tol=tol, chunk_rows=chunk_rows, progress=progress)
        return super().image(seq, tissue=tissue, scanner=scanner, settings=settings, tol=tol, voxels=voxels,
                             chunk_rows=chunk_rows, backend=backend, progress=progress)

    def image_study(self, study, *, tol=0.25, chunk_rows=2_000_000, progress=None):
        """``(S, floor, plan)`` for a :class:`~dmipy_sim.replay.study.Study` (dmipy-sim#297): ``S`` is
        ``(pairs, *grid, n_meas)``, ``floor`` ``(pairs, *grid)`` the split-half floor of each volume (the rows of a
        voxel in two halves, ``max_m |S_a - S_b| / 2`` on the S0-normalised signal), ``plan`` what was read. One pass
        over the rows: per row group the primitives of every acquisition once (the bands and the path channel
        contracted, the exposures and the contact read), on the device; every (tissue, scanner) pair then an
        elementwise term and two segment sums."""
        import functools
        import jax
        import jax.numpy as jnp
        was_x64 = bool(jax.config.read("jax_enable_x64"))
        jax.config.update("jax_enable_x64", True)
        try:
            pairs = [study.resolved(k) for k in range(len(study))]
            plans = [self.plan(a.waveform, tissue=t_, scanner=s_, tol=tol) for a in study.protocol for t_, s_ in pairs]
            plan = dict(plans[0], K=max(p["K"] for p in plans), modes=max(p["modes"] for p in plans), contact=any(p["contact"] for p in plans),
                        relaxation=any(p["relaxation"] for p in plans), settings=len(pairs), study=study.to_meta())
            self.src.bytes_read = self.src.requests = 0; t0 = time.time()
            n_meas = study.protocol.n_meas; n_vox = int(np.prod(self.grid.shape))
            num = np.zeros((len(pairs), 2, n_vox, n_meas), complex); den = np.zeros((2, n_vox)); rows = 0
            ROWS, SEGS = 1 << 16, 256

            @functools.partial(jax.jit, static_argnums=(10,))
            def pair_sums(phi, fi, fa, et2, et1, ct, ct1, w, seg2, terms, n_seg2):
                """``Primitives.signals`` on the device: `(ew[:, None] * E)` summed over each voxel's rows, with every
                term taken from :meth:`~dmipy_sim.replay.study.Primitives.reduction_terms` and none re-derived here."""
                a_i, a_a, rho_D, rho1_D, invT2, invT1, amp, vox = terms
                logw = -(et2 @ invT2) - (et1 @ invT1) + rho_D * ct + rho1_D * ct1
                ph = phi + (a_i * fi + a_a * fa)[:, None]
                E = jnp.exp(1j * ph) * vox[None, :]                       # replay._signal_factor
                return jax.ops.segment_sum(E * (amp * w * jnp.exp(logw))[:, None], seg2, num_segments=n_seg2)
            for pk in self.iter_views(chunk_rows=chunk_rows, K=plan["K"], modes=plan["modes"], contact=plan["contact"]):
                n = pk.n_walkers
                ijk, _ = self.grid.bin(pk.r0); v = np.ravel_multi_index(ijk.T, self.grid.shape)
                starts = np.flatnonzero(np.r_[True, v[1:] != v[:-1]]); n_seg = len(starts)
                seg = np.repeat(np.arange(n_seg), np.diff(np.r_[starts, n])); half = (np.arange(n) - starts[seg]) % 2
                n_pad = -(-n // ROWS) * ROWS; s_pad = -(-(n_seg + 1) // SEGS) * SEGS
                seg2 = np.full(n_pad, 2 * n_seg, np.int32); seg2[:n] = 2 * seg + half
                w_all = None
                for a, sl in zip(study.protocol, study.protocol.slices):
                    prim = pk.walker_primitives(a, tol=None); w_all = prim.w
                    pad = lambda x, shape: (np.zeros((n_pad,) + shape) if x is None else np.concatenate([np.asarray(x, np.float64), np.zeros((n_pad - n,) + shape)]))
                    n_pools = prim.n_pools or 1
                    phi = pad(prim.phi, (prim.phi.shape[1],)); fi = pad(prim.field_iso, ()); fa = pad(prim.field_aniso, ())
                    et2 = pad(prim.exposure_t2, (n_pools,)); et1 = pad(prim.exposure_t1, (n_pools,))
                    ct = pad(prim.contact, ()); ct1 = pad(prim.contact_t1, ()); w = pad(prim.w, ())
                    dev = [jnp.asarray(x) for x in (phi, fi, fa, et2, et1, ct, ct1, w)] + [jnp.asarray(seg2)]
                    for k, (t_, s_) in enumerate(pairs):
                        rt = prim.reduction_terms(t_, s_)                        # every per-pair term, resolved once
                        terms = (jnp.float64(rt["a_iso"]), jnp.float64(rt["a_aniso"]), jnp.float64(rt["rho2_over_D"]),
                                 jnp.float64(rt["rho1_over_D"]), jnp.asarray(rt["invT2"]), jnp.asarray(rt["invT1"]),
                                 jnp.float64(rt["amplitude"]), jnp.asarray(rt["voxel"], jnp.float64))
                        sums = np.asarray(pair_sums(*dev, terms, 2 * s_pad))[:2 * n_seg].reshape(n_seg, 2, -1)
                        np.add.at(num[k, 0, :, sl], v[starts], sums[:, 0]); np.add.at(num[k, 1, :, sl], v[starts], sums[:, 1])
                wh = np.zeros((2, n)); wh[half, np.arange(n)] = w_all
                np.add.at(den[0], v[starts], np.add.reduceat(wh[0], starts)); np.add.at(den[1], v[starts], np.add.reduceat(wh[1], starts))
                rows += n
                if progress:
                    progress(rows, self.src.bytes_read, time.time() - t0)
            S = np.full((len(pairs), n_vox, n_meas), np.nan); floor = np.full((len(pairs), n_vox), np.nan)
            both = (den > 0).all(0); any_ = den.sum(0) > 0
            tot = num.sum(1); S[:, any_] = np.abs(tot[:, any_] / den.sum(0)[any_][None, :, None])
            Sa = np.abs(num[:, 0][:, both] / den[0][both][None, :, None]); Sb = np.abs(num[:, 1][:, both] / den[1][both][None, :, None])
            floor[:, both] = 0.5 * np.abs(Sa - Sb).max(-1)
            plan.update(bytes_read=self.src.bytes_read, requests=self.src.requests, seconds=time.time() - t0, rows=rows)
            return S.reshape((len(pairs),) + tuple(self.grid.shape) + (n_meas,)), floor.reshape((len(pairs),) + tuple(self.grid.shape)), plan
        finally:
            jax.config.update("jax_enable_x64", was_x64)


def open_columnar(uri, *, revision="main", workers=8):
    """The columnar layout at ``uri`` (a directory, or ``hf://owner/name/prefix``) as a :class:`ColumnarPack`."""
    return ColumnarPack(uri, revision=revision, workers=workers)
