"""Read calibration against closed-form linear and quadratic fields, including shortened intervals."""
import numpy as np
import pytest
from dmipy_sim.engine.accumulate import field_read_statistics


def test_linear_fields_have_exact_trapezoids_and_length_weighted_short_intervals():
    dt = .1
    time = np.arange(11) * dt
    reads = np.stack([time, 2 * time], axis=-1)[None]
    result = field_read_statistics(reads, step_s=dt, rounds_per_block=5, blocks_per_window=2, checks=(3, 5))
    assert np.max(np.abs(result["checks"][:, :, 0])) < 1e-15
    # Two blocks, each a three-round interval and a shortened two-round interval.
    expected = 2 * dt ** 2 * (3 ** 2 + (2 * 2 / 3) ** 2)
    np.testing.assert_allclose(result["checks"][0, 0, 1, 0], expected * np.array([1, 4]))
    np.testing.assert_allclose(result["qv_increments"][0, 0], 10 * dt ** 2 * np.array([1, 4]))


def test_quadratic_integral_errors_accumulate_over_a_window_before_squaring():
    dt = .02
    t = np.arange(21) * dt
    reads = np.broadcast_to(t[None, :, None] ** 2, (3, 21, 1))
    result = field_read_statistics(reads, step_s=dt, rounds_per_block=5, blocks_per_window=2, checks=(3,))
    # Composite trapezoid error for t^2 is sum(interval_length^3)/6.
    per_block = dt ** 3 * (3 ** 3 + 2 ** 3 - 5) / 6
    np.testing.assert_allclose(result["checks"][:, :, 0, 0, 0], 2 * per_block, rtol=1e-10)
    assert result["qv_field"].shape == (3, 2, 1)


def test_nonfinite_or_incomplete_reference_series_is_refused():
    with pytest.raises(ValueError, match="whole windows"):
        field_read_statistics(np.zeros((2, 10, 1)), step_s=.1, rounds_per_block=5, blocks_per_window=1, checks=(3,))
    with pytest.raises(ValueError, match="finite"):
        field_read_statistics(np.full((2, 6, 1), np.nan), step_s=.1, rounds_per_block=5, blocks_per_window=1, checks=(3,))


def _jax_pilot(path, read, sub_steps, n_bw, n_win, checks, dt):
    """The JAX pilot's accumulation (moment_scan and ring_scan as _walk_geometry' pilot runs them: a read at every
    step, the trapezoid's checks, each block's mean emitted) of the walker that steps along ``path`` ``(n_steps + 1, 3)``:
    ``(block means (n_blocks, n_ch), qv_d (n_win, n_ch), chk (n_win, 2, n_checks, n_ch))``."""
    import jax
    import jax.numpy as jnp
    from dmipy_sim.engine.accumulate import Kind, moment_scan, ring_scan
    P = jnp.asarray(path, jnp.float32)
    n_ch = int(read(P[0]).shape[0])
    kind = Kind(inner=lambda c, _: ((c[0] + 1,), None), pos_of=lambda c: P[c[0]], read_pos_of=lambda c: P[c[0]])
    h = jnp.float32(sub_steps * dt)

    def interval(c, b):
        c, mom = moment_scan(kind, c, sub_steps, 1, read=read, n_ch=n_ch, n_read=1, dt_f=dt, step0=b * sub_steps,
                             checks=checks, keep_reads=False)
        return c, mom[4][0] / h, mom

    @jax.jit
    def walk():
        c = ((jnp.int32(0),), read(P[0]), None, None)
        means, qds, chks = [], [], []
        for w in range(n_win):
            c, _, _, _, e, _, (_, _, _, qd, ec) = ring_scan(interval, c, n_bw, 4, 1, has_blt=False, n_ch=n_ch,
                                                            emit=lambda ys: ys, offset=w * n_bw, n_checks=len(checks))
            means.append(e); qds.append(qd); chks.append(ec)
        return jnp.concatenate(means), jnp.stack(qds), jnp.stack(chks)
    return tuple(np.asarray(x, np.float64) for x in walk())


def test_the_host_statistics_are_the_jax_pilots_accumulation():
    """A rough walker path (a Brownian walk across a field that varies on the step's scale) read at every step: the
    block means, the read series' quadratic variation per window and the trapezoid's checks the JAX pilot accumulates
    in float32 equal field_read_statistics of the same reads (float64 sums of the float32 reads) to the float32
    summation bound, |error| <= (L + 4) u sum |terms| (u = 2^-24; L the float32 additions of the pilot's chain: the
    block's steps, the ring's blocks, the window's rings; 4 the rounding of each term), per entry. A backend's pilot
    hands the host the reads and the host computes these (_walk_geometry(field_pilot=)): the definitions are
    the JAX pilot's."""
    import jax.numpy as jnp
    from dmipy_sim.engine.accumulate import read_checks, RING
    sub_steps, n_bw, n_win, dt = 23, 70, 2, 1e-6
    n_steps = sub_steps * n_bw * n_win
    rng = np.random.default_rng(5)
    path = np.cumsum(np.vstack([np.zeros((1, 3)), rng.standard_normal((n_steps, 3)) * 0.05e-6]), axis=0).astype(np.float32)
    k = np.array([[3.1, -1.7, 2.3], [-0.9, 4.2, 1.1], [2.0, 0.5, -3.3]]) / 1e-6

    def read(r):
        ph = jnp.asarray(k, jnp.float32) @ r.astype(jnp.float32)   # three channels varying on a 0.3 um scale
        return jnp.stack([jnp.sin(ph[0]), jnp.cos(ph[1]) + 0.5, ph[2] * jnp.float32(0.1)])
    checks = read_checks(sub_steps)
    means_j, qd_j, chk_j = _jax_pilot(path, read, sub_steps, n_bw, n_win, checks, dt)
    import jax
    reads = np.asarray(jax.vmap(read)(jnp.asarray(path)))[None]
    st = field_read_statistics(reads, step_s=dt, rounds_per_block=sub_steps, blocks_per_window=n_bw, checks=checks)
    u = 2.0 ** -24
    L = sub_steps + RING + -(-n_bw // RING)
    tol = (L + 4) * u
    v = reads[0].astype(np.float64)
    inc = 0.5 * (v[:-1] + v[1:]) * dt                                             # (n_steps, n_ch) the trapezoid increments
    # block means: the block's sum of |inc| over h
    abs_mean = np.abs(inc).reshape(n_win * n_bw, sub_steps, -1).sum(1) / (sub_steps * dt)
    assert np.all(np.abs(means_j - st["means"][0]) <= tol * abs_mean + 1e-30)
    # the read series' variation: nonnegative terms, the sum itself
    assert np.all(np.abs(qd_j - st["qv_increments"][0]) <= tol * st["qv_increments"][0])
    # the checks: E_d sums |fine| and |coarse| terms over the window; the coarse QV is nonnegative
    abs_inc_w = np.abs(inc).reshape(n_win, -1, inc.shape[1]).sum(1)                # (n_win, n_ch)
    E_j, E_h = chk_j[:, 0], st["checks"][0][:, 0]
    assert np.all(np.abs(E_j - E_h) <= 2 * tol * abs_inc_w[:, None, :]), np.abs(E_j - E_h).max()
    assert np.abs(E_h).max() > 1e3 * tol * abs_inc_w.max()                      # the checks are resolved, not rounding
    Q_j, Q_h = chk_j[:, 1], st["checks"][0][:, 1]
    assert np.all(np.abs(Q_j - Q_h) <= tol * Q_h)
