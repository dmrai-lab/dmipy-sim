"""Another simulator's walk as a coefficient walk: positions sampled on a save grid become the position channel (C0)
of a replay pack, and nothing else.

:func:`walk_from_positions` takes the positions ``(n_walkers, n_saves, 3)`` (metres, unwrapped) another engine
recorded at ``t = j sample_interval`` and returns the :class:`~dmipy_sim.coefficient_walk.CoefficientWalk` of ONE window of
``T = (n_saves - 1) sample_interval``, which :func:`~dmipy_sim.replay.bank.build_replay_pack` packs like any producer's walk.
No file is read here: a file format is the caller's (``examples/foreign/`` has one script per engine).

**The bands.** The window's start and displacement are the first and last save. The bridge ``y_i = x_i - x_0 - (i / N)
(x_N - x_0)`` (``N = n_saves - 1`` intervals of ``h = sample_interval``) is zero at both ends; ``beta_k`` is the projection of
its piecewise-linear interpolant ``L`` on ``s_k = sqrt(2 / T) sin(pi k t / T)``. The quadrature is EXACT for ``L``: the
integral of a hat function ``h_i`` against ``sin(omega t)`` is ``h sinc^2(omega h / 2 pi) sin(omega t_i)``, so

    beta_k = sqrt(h) sinc^2(k / 2N) * DST1_ortho(y_1..y_{N-1})_k ,        sinc(x) = sin(pi x) / (pi x),

the orthonormal DST-I of the interior samples (``sqrt(2 / N) sum_i y_i sin(pi k i / N)``). (The block table's order-0
accumulation, the midpoint rule, has ``sinc`` where this has ``sinc^2``; it is not exact for a path through saves.)

**The band limit.** The interpolant's bridge has ``N - 1`` degrees of freedom, the interior samples, and the sine
bands ``k = 1..N-1`` are an invertible transform of them (the DST-I); ``s_N`` vanishes at every save, so
``beta_N = 0`` and no higher band is a function of the samples. ``K <= K_max = N - 1`` is what the samples
resolve; a larger ``K`` is refused. Near ``K_max`` the bands are interpolation, not measurement; the certificate
charges them (below).

**The numerical term.** The true path is ``L`` plus, on each interval, a Brownian bridge independent of the saves
(Brownian between saves, per axis ``2 D`` variance rate). So the true band is ``beta_k = beta_k^L + e_k`` with
``e_k`` independent of ``beta_k^L``, ``Var(e_k) = Var(beta_k) - Var(beta_k^L)``: the free bridge's band variance
``2 D T^2 / (pi k)^2`` less the interpolant's, ``h sinc^4(k / 2N)`` times the DST-I variance ``D h / (2 sin^2(pi k /
2N))``. With ``x = pi k / 2N``:

    Var(e_k) = (D h^2 / 2) (x^2 - sin^2 x) / x^4   <=   D h^2 / 6      (per axis, 1 <= k <= N - 1),

the quotient falling monotonically from ``1/3`` at ``x -> 0`` to ``0.24`` at ``x = pi / 2``. The bound ``D h^2 / 6 =
2 D h^2 / 12`` is independent of ``k`` and of ``T``: the interpolation error is white across the bands. The ``e_k`` are uncorrelated across ``k`` (the free bridge's sine bands are its Karhunen-Loeve coordinates, and the
DST-I diagonalises the discrete bridge), so over a gradient waveform with sine weights ``W_kc`` the replayed phase
carries the variance ``gamma^2 sum_{k,c} W_kc^2 Var(e_k)`` and the signal at most half of it:
``err_quadrature <= gamma^2 sum_{k,c} W_kc^2 D h^2 / 12``. That is the pack's
quadrature term, from ``D`` and ``sample_interval`` alone (:func:`~dmipy_sim.replay.bank._derived_certificate`, ``band_var``),
and the truncation beyond ``K`` is charged as for every pack. The bound is derived for a free Brownian path; a
restricted walk's bridge is no rougher (the same statement the block quadrature makes), and it is stated at the
walker's ``D`` as given. A foreign engine's own step error (an engine stepping coarser than its saves) is not in it.

**Channels.** Only the position channel is produced. The pool the walkers were seeded in is the static occupancy
label (C1, no events); the contact (C2), the field (C3) and any change of pool are absent, and a replay that asks
for them is refused by the reader by name (their definitions differ across engines, and a pack never mixes them under
one name).
"""
import numpy as np

from .coefficient_walk import CoefficientWalk

#: the producer record's required keys
PRODUCER_REQUIRED = ("engine", "version")

#: positions are chunked over walkers to this many float64 values (a chunk of the interior bridge)
_CHUNK_VALUES = 32_000_000


def band_limit(n_saves):
    """The largest ``K`` the samples resolve: ``n_saves - 2`` (the interior samples of the bridge; ``s_N`` vanishes at every save)."""
    return int(n_saves) - 2


def interpolation_variance(D, sample_interval, K, N):
    """Var of the interpolation error ``e_k`` of the bands ``k = 1..K`` of a free Brownian path at diffusivity ``D``
    through saves ``sample_interval`` apart over ``N`` intervals, per axis (m^2): ``(D h^2 / 2) (x^2 - sin^2 x) / x^4`` with ``x = pi k / 2N``."""
    x = np.pi * np.arange(1, int(K) + 1) / (2.0 * int(N))
    return 0.5 * float(D) * float(sample_interval) ** 2 * (x ** 2 - np.sin(x) ** 2) / x ** 4


def interpolation_variance_bound(D, sample_interval):
    """The bound on every band's interpolation variance, per axis (m^2): ``D sample_interval^2 / 6``."""
    return float(D) * float(sample_interval) ** 2 / 6.0


def _bands(y, N, K, h):
    """Sine bands ``1..K`` of the piecewise-linear bridges ``y`` ``(n, N + 1, 3)`` (zero at both ends)."""
    from scipy.fft import dst
    k = np.arange(1, K + 1)
    scale = np.sqrt(h) * np.sinc(k / (2.0 * N)) ** 2
    d = dst(y[:, 1:-1, :], type=1, norm="ortho", axis=1)[:, :K, :]
    return d * scale[None, :, None]


def walk_from_positions(positions, *, sample_interval, K, pool, spec, D, producer, times=None, seed=None):
    """The coefficient walk of positions another engine sampled on a uniform save grid.

    Parameters
    ----------
    positions : (n_walkers, n_saves, 3) array
        Metres, unwrapped (continuous through any periodic boundary), save ``j`` at ``t = j sample_interval``: the grid
        starts at ``t = 0`` and is uniform. A float32 export is accepted; its resolution is recorded.
    sample_interval : float
        The save interval (s).
    K : int
        Bands per axis. Required: at most :func:`band_limit` (``n_saves - 2``), else refused by name.
    pool : str
        The name (in ``spec.pools``) of the pool every walker was seeded in; the occupancy of the pack is that
        static label. A walk whose walkers leave their pool is not a C0-only walk of this pool.
    spec : :class:`~dmipy_replay.spec.substrate.SubstrateSpec`
        The substrate the engine walked.
    D : float
        The diffusivity the engine walked at (m^2/s); the interpolation bound is stated at it.
    producer : dict
        The producing engine: at least ``engine`` and ``version`` (strings); any further JSON-serialisable keys (the
        engine's seed, step, configuration) are kept. Required.
    times : (n_saves,) array, optional
        The engine's recorded save times; checked against ``j sample_interval``, and refused when they differ.
    seed : int, optional
        The engine's seed, recorded as the walk's.

    Returns
    -------
    :class:`~dmipy_sim.coefficient_walk.CoefficientWalk`
        One window, position channel and the static pool label only; ``m = 0``, ``h = sample_interval``.
    """
    from dmipy_replay.spec.substrate import SubstrateSpec
    pos = np.asarray(positions)
    if pos.ndim != 3 or pos.shape[2] != 3 or not np.issubdtype(pos.dtype, np.floating):
        raise ValueError(f"positions are a float array (n_walkers, n_saves, 3); got shape {pos.shape} dtype {pos.dtype}")
    n, n_saves = int(pos.shape[0]), int(pos.shape[1])
    if n < 1 or n_saves < 4:
        raise ValueError(f"a walk of saves needs at least one walker and 4 saves (K >= 1 resolves n_saves - 2); got {n} x {n_saves}")
    sample_interval = float(sample_interval)
    if not (np.isfinite(sample_interval) and sample_interval > 0.0):
        raise ValueError(f"sample_interval is the save interval in seconds, > 0; got {sample_interval!r}")
    if times is not None:
        t = np.asarray(times, np.float64)
        want = sample_interval * np.arange(n_saves)
        if t.shape != want.shape or np.abs(t - want).max() > 1e-6 * sample_interval:
            raise ValueError(f"the saves are not on the uniform grid t = j * sample_interval from t = 0 (sample_interval = {sample_interval:.6g} s): "
                             f"the recorded times differ from it by up to {np.abs(t - want).max() if t.shape == want.shape else float('nan'):.3g} s "
                             "(or are not n_saves long); resample the walk onto a uniform grid at the engine, not here")
    if not np.isfinite(pos).all():
        raise ValueError("positions hold a non-finite value")
    D = float(D)
    if not (np.isfinite(D) and D > 0.0):
        raise ValueError(f"D is the diffusivity the engine walked at (m^2/s), > 0; got {D!r}")
    if not isinstance(spec, SubstrateSpec):
        raise TypeError(f"spec is the SubstrateSpec the engine walked, got {type(spec).__name__}")
    if not isinstance(producer, dict) or any(not isinstance(producer.get(k), str) or not producer[k].strip() for k in PRODUCER_REQUIRED):
        raise ValueError(f"producer names the engine that walked: a dict with non-empty string {PRODUCER_REQUIRED}; got {producer!r}")
    import json
    json.dumps(producer)
    names = [p.name for p in spec.pools]
    if pool not in names:
        raise ValueError(f"pool {pool!r} is not a pool of the substrate; its pools are {names}")
    pool_id = int(next(p.id for p in spec.pools if p.name == pool))
    N = n_saves - 1
    K_max = band_limit(n_saves)
    if not isinstance(K, (int, np.integer)) or int(K) < 1:
        raise ValueError(f"K is the number of bands per axis, an integer >= 1; got {K!r}")
    K = int(K)
    if K > K_max:
        raise ValueError(f"K = {K} is above the band limit of this walk: {n_saves} saves are {N} intervals, whose bridge has "
                         f"{N - 1} interior samples, so the saves resolve at most K = {K_max} bands (s_{N} vanishes at every save); "
                         f"ask for K <= {K_max} or save more often")
    h = sample_interval
    T = N * h
    step_max = float(np.abs(np.diff(pos.astype(np.float64), axis=1)).max())
    if step_max > 20.0 * np.sqrt(2.0 * D * h):
        raise ValueError(f"a step of {step_max:.3g} m between two saves is {step_max / np.sqrt(2.0 * D * h):.1f} sigma of free diffusion at "
                         f"D = {D:.3g} m^2/s and sample_interval = {h:.3g} s: the positions are wrapped at a periodic boundary or D/sample_interval are wrong; "
                         "pass them unwrapped, with the D and sample_interval of the walk")
    coeffs = np.empty((n, 1, K + 2, 3), np.float32)
    qv = np.empty((n, 1, 3), np.float32)
    chunk = max(1, int(_CHUNK_VALUES // (3 * n_saves)))
    cast_err = 0.0
    for a in range(0, n, chunk):
        x = pos[a:a + chunk].astype(np.float64)
        disp = x[:, -1] - x[:, 0]
        frac = (np.arange(N + 1) / N)[None, :, None]
        y = x - x[:, :1] - disp[:, None, :] * frac
        c = np.concatenate([x[:, :1], disp[:, None, :], _bands(y, N, K, h)], axis=1)
        coeffs[a:a + chunk, 0] = c
        cast_err = max(cast_err, float(np.abs(coeffs[a:a + chunk, 0, :2].astype(np.float64) - c[:, :2]).max()))
        qv[a:a + chunk, 0] = (np.diff(x, axis=1) ** 2).sum(axis=1)
    qv_mean = qv[:, 0].astype(np.float64).mean(axis=0)
    res = float(np.spacing(np.float32(np.abs(pos).max()))) if pos.dtype == np.float32 else float(np.spacing(np.abs(pos).max()))
    source = dict(kind="foreign_save_grid", **producer, sample_interval=h, n_saves=n_saves, K_max=K_max, K=K,
                  interpolant="piecewise_linear", band_quadrature="exact for the piecewise-linear interpolant: sqrt(dt) sinc^2(k/2N) DST-I",
                  interpolation_variance_bound_m2="D sample_interval^2 / 6 per band and axis",
                  interpolation_variance_bound_value_m2=interpolation_variance_bound(D, h),
                  position_dtype=str(pos.dtype), position_resolution_m=res, channels=["positions", "compartment"],
                  channels_absent=["boundary_local_time", "susceptibility_field", "pool_changes"])
    block = dict(h=h, m=0, n_blocks=N, rule="given", precision="float32", k0_residual_m=cast_err,
                 k0_residual_relative=float(np.finfo(np.float32).eps) / 2.0, source=source,
                 quadratic_variation=dict(positions_mean_m2=[float(v) for v in qv_mean],
                                          positions_over_2DT=[float(v / (2.0 * D * T)) for v in qv_mean]))
    return CoefficientWalk(
        position_coeffs=coeffs, T=T, K=K, m=0, block=block, n_windows=1, dt_sim=h, sub_steps=1,
        compartment=np.full(n, pool_id, np.int8), comp_events=None, qv_positions=qv, seed=seed, diffusivity=D, spec=spec,
        stepping=dict(kind="foreign_save_grid", engine=producer["engine"], version=producer["version"]))
