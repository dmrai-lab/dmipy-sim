"""The record of a walk's membrane trials at its plain walls.

A plain wall (:class:`~dmipy_sim.geometry.PackedCylinders`, :class:`~dmipy_sim.geometry.PackedSpheres`, a plain
cylinder of :class:`~dmipy_sim.geometry.PackedMyelinatedCylinders`) decides a transmission trial at every encounter of
a step, with probability ``p = x / (1 + x)``, ``x = 2 (kappa / D) d_perp``
(:func:`~dmipy_sim.geometry._boundary.transmit_probability`). A walk asked for it (``record_trials=``
:class:`TrialRecordSpec` on :func:`~dmipy_sim.engine.core._walk_geometry` and :func:`~dmipy_sim.walk`) returns, per
walker and per (object, time band) cell, the count of its trials ``n``, the count of granted trials ``N`` and the ``M``
sums ``S_j = sum u T_j(2 u / u_max - 1)``, ``u = 2 d_perp / R`` the trial's overshoot (``R`` the radius of the object
met), ``T_j`` the Chebyshev polynomial of the first kind and ``u_max`` the object's a priori bound on ``u``
(:func:`u_max_of`): the :class:`TrialRecord` of :attr:`~dmipy_sim.coefficient_walk.CoefficientWalk.trial_record`.

The rules, each a condition a consumer's arithmetic relies on:

* every encounter of a plain wall is a trial and is recorded, the second and later encounters of a step included and
  an object of permeability 0 included (its trials are always refused: ``N`` is 0, ``n`` counts them); a step the walk
  discards (a walker held bound through its dwell) records none;
* a trial belongs to the step it is decided in, and a step to the band whose interval contains the step's END time
  ``(step + 1) dt``: band ``k`` is the left-open, right-closed ``(edge_k, edge_k+1]``, so a step ending exactly on an
  edge belongs to the earlier band, the final band contains ``T_max``, and a storage window's bands hold exactly its
  own steps; a scheduled permeability is read at the step's midpoint (:func:`~dmipy_sim.engine.physics.membrane_kappa`);
* the counts are integers and the sums are held in the walk's accumulation dtype; nothing is rounded or dithered, and
  the record draws nothing (the walk's random stream is the walk's without the record);
* the record reads the walk and enters none of its arithmetic: one step of a wall kernel is the same bits with and
  without it. The whole walk with it and without it are two compiled programs: the same bits on the recording build
  and host, while another JAX build may arrange them so that a position's last bits differ, which the reflections can
  grow into another encounter -- two valid walks of one law. The record is exactly consistent with its own walk (the
  record turns ``Geometry.count_walls`` on, of which it is the cell-resolved form: ``sum n`` over the cells of an
  all-plain cell is ``work["n_hits"]`` and ``sum N`` is ``work["n_crossings"]``).

Its memory is ``n_objects * n_bands * (2 * 4 + M * itemsize)`` bytes per walker (:func:`bytes_per_walker`), refused
above :data:`MAX_BYTES_PER_WALKER`.
"""
from dataclasses import dataclass, field

import numpy as np

#: The most bands a record holds.
MAX_BANDS = 64
#: The most bytes a record holds per walker.
MAX_BYTES_PER_WALKER = 65536
#: An edge within this fraction of a physics step of a step's end is that step's end (the rounding of seconds).
EDGE_SNAP = 1e-9


@dataclass(frozen=True)
class TrialRecordSpec:
    """What a walk records of its membrane trials: ``modes`` Chebyshev sums per cell (``M``, at least 1) over the time
    bands ``band_edges`` (seconds, strictly increasing, from 0 to the walk's ``T_max``, at most :data:`MAX_BANDS` bands,
    containing every boundary of the walk's storage windows so that a band never straddles two windows)."""
    modes: int
    band_edges: tuple

    def __post_init__(self):
        m = self.modes
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or int(m) < 1:
            raise ValueError(f"trial_record: modes must be a whole number of at least 1, got {m!r}")
        edges = np.asarray(self.band_edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
            raise ValueError(f"trial_record: band_edges must be at least two finite times in seconds, got {self.band_edges!r}")
        if not np.all(np.diff(edges) > 0):
            raise ValueError(f"trial_record: band_edges must be strictly increasing, got {self.band_edges!r}")
        if edges.size - 1 > MAX_BANDS:
            raise ValueError(f"trial_record: {edges.size - 1} bands exceed {MAX_BANDS}")
        object.__setattr__(self, "modes", int(m))
        object.__setattr__(self, "band_edges", tuple(float(e) for e in edges))

    @property
    def n_bands(self):
        return len(self.band_edges) - 1


def edge_steps(spec, T_max, dt, n_steps, window_steps, window_s=None):
    """The step-end index of every band edge of ``spec`` for a walk of ``n_steps`` physics steps of ``dt`` seconds over
    ``T_max`` seconds in storage windows of ``window_steps`` steps (``window_s`` seconds each, the walk's
    ``window_T``): edge ``e`` is at the last step end ``j dt <= e``, ``j = floor(e / dt)`` (an edge within
    :data:`EDGE_SNAP` of a step of a step end is that step end, the rounding of seconds), so the steps ending in
    ``(edge_k, edge_k+1]`` are those of index ``j`` with ``steps[k] < j + 1 <= steps[k + 1]``. Refused by name: edges that
    do not run from 0 to ``T_max``, and edges that miss a window boundary, checked on the times as given (an edge
    farther than :data:`EDGE_SNAP` of a step from the boundary misses it, before any edge is placed on a step end)."""
    edges = np.asarray(spec.band_edges, np.float64)
    if edges[0] != 0.0 or edges[-1] != float(T_max):
        raise ValueError(f"trial_record: band_edges must run from 0 to T_max = {float(T_max)!r}, got "
                         f"[{float(edges[0])!r}, {float(edges[-1])!r}]")
    window_steps = int(window_steps)
    tol = EDGE_SNAP * float(dt)
    for w in range(1, int(n_steps) // window_steps):
        t = w * (float(window_s) if window_s is not None else window_steps * float(dt))
        if not np.any(np.abs(edges - t) <= tol):
            raise ValueError(f"trial_record: band_edges must contain the window boundary {t!r}")
    q = edges / float(dt)
    near = np.rint(q)
    steps = np.where(np.abs(q - near) <= EDGE_SNAP, near, np.floor(q)).astype(np.int64)
    steps[0], steps[-1] = 0, int(n_steps)                     # the walk's first and last instants
    return steps


def u_max_of(radii, D, dt):
    """The a priori bound on a trial's overshoot ``u = 2 d_perp / R`` at each object: ``2 LONGEST_STEP h / R``,
    ``h = sqrt(6 D dt)`` with ``D`` the larger diffusivity of the two pools the object's wall separates, since
    ``d_perp`` is at most the step and the step law's longest step is ``LONGEST_STEP h``
    (:data:`~dmipy_sim.math.step_law.LONGEST_STEP`). ``(n_objects,)`` float64."""
    from .math.step_law import LONGEST_STEP
    radii, D = np.asarray(radii, np.float64), np.asarray(D, np.float64)
    return 2.0 * float(LONGEST_STEP) * np.sqrt(6.0 * D * float(dt)) / radii


def bytes_per_walker(n_objects, n_bands, modes, itemsize):
    """The record's bytes per walker: ``n_objects * n_bands * (2 * 4 + modes * itemsize)`` (two int32 counts and
    ``modes`` sums of ``itemsize`` bytes per cell)."""
    return int(n_objects) * int(n_bands) * (8 + int(modes) * int(itemsize))


def check_bytes(n_objects, n_bands, modes, itemsize):
    """:func:`bytes_per_walker`, refused by name above :data:`MAX_BYTES_PER_WALKER`."""
    b = bytes_per_walker(n_objects, n_bands, modes, itemsize)
    if b > MAX_BYTES_PER_WALKER:
        raise ValueError(f"trial_record: {b} bytes per walker exceed {MAX_BYTES_PER_WALKER}")
    return b


def chebyshev_terms(x, modes, xp):
    """``(T_0(x), ..., T_{modes-1}(x))`` stacked on a new last axis, by the three-term recurrence in ``x``'s dtype."""
    out = [xp.ones_like(x)]
    if modes > 1:
        out.append(x)
    for _ in range(2, modes):
        out.append(2 * x * out[-1] - out[-2])
    return xp.stack(out, axis=-1)


def device_record(n_objects, steps, u_max, modes, dtype):
    """The record as a walk carries it per walker: ``(init, add)``. ``init()`` is the empty record ``(n (n_objects,
    n_bands) int32, N (n_objects, n_bands) int32, S (n_objects, n_bands, modes) dtype, exceed int32)``; ``add(rec,
    log, step, keep=None)`` adds the trials of a :class:`~dmipy_sim.geometry._boundary.TrialLog` of the walk-wide step
    ``step`` (int32) in the band whose ``(edge_k, edge_k+1]`` holds the step's end, ``keep`` False discarding the step
    (a walker held bound); ``exceed`` counts the trials whose ``u`` exceeds the object's ``u_max``."""
    import jax.numpy as jnp
    n_bands = int(len(steps) - 1)
    interior = jnp.asarray(np.asarray(steps[1:-1], np.int64), jnp.int32)
    umax = jnp.asarray(np.asarray(u_max, np.float64), dtype)
    modes, n_objects = int(modes), int(n_objects)

    def init():
        z = jnp.zeros((n_objects, n_bands), jnp.int32)
        return (z, z, jnp.zeros((n_objects, n_bands, modes), dtype), jnp.int32(0))

    def add(rec, log, step, keep=None):
        n, N, S, exceed = rec
        valid = log.valid if keep is None else (log.valid & keep)
        band = jnp.sum(interior < (jnp.asarray(step, jnp.int32) + 1)).astype(jnp.int32)    # (edge_k, edge_k+1]
        obj = log.obj.astype(jnp.int32)
        n = n.at[obj, band].add(valid.astype(jnp.int32))
        N = N.at[obj, band].add((valid & log.granted).astype(jnp.int32))
        u = log.u.astype(dtype)
        um = umax[obj]
        terms = u[:, None] * chebyshev_terms(2 * u / um - 1, modes, jnp)
        S = S.at[obj, band].add(jnp.where(valid[:, None], terms, jnp.zeros((), dtype)))
        exceed = exceed + jnp.sum(valid & (u > um)).astype(jnp.int32)
        return (n, N, S, exceed)

    return init, add


@dataclass(frozen=True)
class TrialRecord:
    """A walk's membrane trials at its plain walls, per walker (rows in walker order: row ``w`` is walker
    ``first_walker + w`` of the seed) and per (object, time band) cell.

    Attributes
    ----------
    n : (n_walkers, n_objects, n_bands) int32
        The trials: every encounter of the object's wall by a step ending in the band.
    N : (n_walkers, n_objects, n_bands) int32
        The granted trials (crossings).
    S : (n_walkers, n_objects, n_bands, modes) in the walk's accumulation dtype
        ``S_j = sum u T_j(2 u / u_max[o] - 1)`` over the cell's trials, ``u = 2 d_perp / R_o`` the overshoot.
    u_max : (n_objects,) float64
        Each object's bound on ``u`` (:func:`u_max_of`).
    band_edges : tuple of float
        The bands' edges in seconds on the walk's clock; band ``k`` holds the steps ending in ``(edge_k, edge_k+1]``.
    modes : int
        ``M``, the sums per cell.
    object_radii : (n_objects,) float64
        The radius ``R_o`` of each object (m), in the geometry's object order.
    """
    n: np.ndarray
    N: np.ndarray
    S: np.ndarray
    u_max: np.ndarray
    band_edges: tuple
    modes: int
    object_radii: np.ndarray = field(repr=False)

    @property
    def nbytes(self):
        return int(self.n.nbytes + self.N.nbytes + self.S.nbytes + self.u_max.nbytes + self.object_radii.nbytes)

    def take(self, idx):
        """The record of the walkers ``idx`` (an index array or a slice), in that order."""
        import dataclasses
        return dataclasses.replace(self, n=self.n[idx], N=self.N[idx], S=self.S[idx])

    def bands(self, t0, t1):
        """The record of the bands within ``[t0, t1]`` (seconds), which must be band edges."""
        import dataclasses
        edges = np.asarray(self.band_edges)
        i0 = int(np.flatnonzero(np.isclose(edges, t0, rtol=0.0, atol=EDGE_SNAP * max(abs(t1 - t0), 1e-300)))[0])
        i1 = int(np.flatnonzero(np.isclose(edges, t1, rtol=0.0, atol=EDGE_SNAP * max(abs(t1 - t0), 1e-300)))[0])
        return dataclasses.replace(self, n=self.n[:, :, i0:i1], N=self.N[:, :, i0:i1], S=self.S[:, :, i0:i1],
                                   band_edges=tuple(self.band_edges[i0:i1 + 1]))

    def same_layout(self, other):
        """Whether ``other`` records the same objects, bands and modes."""
        return (self.band_edges == other.band_edges and self.modes == other.modes
                and np.array_equal(self.u_max, other.u_max) and np.array_equal(self.object_radii, other.object_radii))

    @staticmethod
    def concatenate(records):
        """The records of contiguous walker ranges joined in order (the same layout, refused by name otherwise)."""
        import dataclasses
        r0 = records[0]
        for j, r in enumerate(records[1:], 1):
            if not r0.same_layout(r):
                raise ValueError(f"trial_record: walk {j} records other objects, bands or modes than walk 0")
        return dataclasses.replace(r0, n=np.concatenate([r.n for r in records]), N=np.concatenate([r.N for r in records]),
                                   S=np.concatenate([r.S for r in records]))

    def to_arrays(self):
        """The record as the tensors of a walk file."""
        return {"trial_n": self.n, "trial_N": self.N, "trial_S": self.S, "trial_u_max": self.u_max,
                "trial_band_edges": np.asarray(self.band_edges, np.float64), "trial_object_radii": self.object_radii}

    @classmethod
    def from_arrays(cls, arrays):
        """The record from :meth:`to_arrays`, or None when the file holds none."""
        if "trial_n" not in arrays:
            return None
        S = arrays["trial_S"]
        return cls(n=arrays["trial_n"], N=arrays["trial_N"], S=S, u_max=arrays["trial_u_max"],
                   band_edges=tuple(float(e) for e in arrays["trial_band_edges"]), modes=int(S.shape[-1]),
                   object_radii=arrays["trial_object_radii"])
