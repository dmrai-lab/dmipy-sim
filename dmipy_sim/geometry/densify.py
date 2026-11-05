"""Dense periodic packings of disks and spheres: one compression path, snapshotted at packing fractions.

Random sequential addition (:mod:`.packing`) saturates near 0.55 for disks and 0.38 for spheres (Feder 1980,
Cooper 1988). Beyond that, a packing is made by compression: the soft-particle protocol of Xu, Blawzdziewicz &
O'Hern (Phys. Rev. E 71, 061306, 2005), as Desmond & Weeks (Phys. Rev. E 80, 051305, 2009) state it -- particles
placed at random at a packing fraction far below jamming, "all disk diameters are slowly expanded subject to the
fixed size ratio" by a step in packing fraction, and after each step any overlap is removed by minimising the
harmonic overlap energy ``V = 1/2 sum (sigma_ij - r_ij)^2`` over the overlapping pairs until it is zero. Below the
jamming point a zero-energy state exists and the minimiser reaches it; at the jamming point it does not (O'Hern,
Silbert, Liu & Nagel, Phys. Rev. E 68, 011306, 2003). The minimiser is FIRE (Bitzek, Koskinen, Gaehler, Moseler &
Gumbsch, Phys. Rev. Lett. 97, 170201, 2006), with that paper's parameters and the time steps of :data:`FIRE`.

A path here is one such run:

- **the start** is a random-sequential-addition packing (largest first, :func:`.packing.pack_cylinders` /
  :func:`.packing.pack_spheres`) at :data:`DENSIFY_START`;
- **the trunk** compresses it in steps of :data:`DENSIFY_STEP` in packing fraction, the stops
  ``DENSIFY_START + k DENSIFY_STEP``: each step shrinks the periodic cell affinely with the radii fixed (the
  protocol's expansion at a fixed size ratio, up to the global scale) and relaxes the overlaps;
- **a snapshot** at a packing fraction on a stop is the trunk's state there; one between two stops is the state
  of the stop below compressed to it and relaxed, a branch the trunk does not continue from.

So a snapshot depends on the radii, the seed and the gap alone, never on which other snapshots were asked for, and
every snapshot of a path holds the same objects (index ``i`` is one object, its radius never changes), each a
small compression of the one before.

**No overlap, by construction.** The minimised contact distance of a pair is ``r_i + r_j + (1 + _GAP_PAD) g`` for
the stated gap ``g``, and a relaxation is accepted only once every overlap of it is below ``_GAP_PAD - _GAP_HOLD``
of ``g``: every clear gap of a snapshot is then at least ``(1 + _GAP_HOLD) g`` in float64, which
:func:`.packing.periodic_min_gap` confirms on the snapshot before it is returned. The walk holds the centres in
float32, whose rounding of a gap is a few ``L 2^-24`` in a cell of side ``L``; a gap below :data:`FLOAT32_GAP` ``L``
is refused.

The neighbour list is a Verlet list (pairs within their contact distance plus ``_SKIN`` mean radii, from
:class:`scipy.spatial.cKDTree`), rebuilt when the affine compression and the largest displacement since the build
could bring an unlisted pair into contact. Every global sum of the minimiser is sequential
(``np.add.accumulate``), the pair list is sorted, and the forces are summed per object in pair order
(``np.bincount``), so a path is a deterministic function of its inputs.
"""
import numpy as np

from .packing import pack_cylinders, pack_spheres, periodic_min_gap

#: The packing fraction a path starts at: a random-sequential-addition packing, far below RSA's own saturation.
DENSIFY_START = 0.05
#: The step of a path in packing fraction between two stops of its trunk: Desmond & Weeks (2009)'s starting
#: expansion step, ``delta phi = 1e-3``.
DENSIFY_STEP = 1e-3
#: The highest packing fraction a path is asked to reach, per dimension (2: disks, the cross-sections of
#: parallel cylinders; 3: spheres), in the contact radii the minimiser holds apart (each radius plus
#: ``(1 + _GAP_PAD) g / 2``). It lies below the protocol's jamming point: 0.842 for 50:50 bidisperse disks of
#: diameter ratio 1.4 and 0.648 for such spheres (O'Hern, Langer, Liu & Nagel, Phys. Rev. Lett. 88, 075507, 2002),
#: 0.639 for monodisperse spheres (O'Hern et al. 2003); polydispersity raises it (Desmond & Weeks, Phys. Rev. E 90,
#: 022204, 2014). This path jams, on 1,000 objects, at 0.838 for those disks and 0.639 for those spheres, 0.636 for
#: monodisperse spheres, and 0.843 / 0.848 / 0.858 for disks and 0.650 / 0.665 / 0.680 for spheres of the Gamma law
#: by quantile at CV 0.25 / 0.45 / 0.65 (truncated at the 1 % water floor); monodisperse disks order partly into
#: hexagonal patches and jam higher, at 0.866 (at 0.8, 24 % of 2,000 equal disks have a local bond order
#: ``|psi_6| > 0.9``, against 5 % at CV 0.25 and 1 % at CV 0.45). The relaxation slows without bound as jamming
#: nears, so the limit keeps a margin below the lowest.
DENSIFY_LIMIT = {2: 0.82, 3: 0.62}

#: FIRE's parameters (Bitzek et al. 2006: ``N_min = 5``, ``f_inc = 1.1``, ``f_dec = 0.5``, ``alpha_start = 0.1``,
#: ``f_alpha = 0.99``), its one adjustable parameter, the largest time step ``dt_max``, and the first time step
#: ``dt_start``, both in units of ``sqrt(m / k)`` for unit mass and unit overlap stiffness. Measured on 3,000
#: objects (Gamma CV 0.45, to 0.8 for disks and 0.6 for spheres): ``dt_max`` 0.5 was the fastest of
#: 0.25 / 0.5 / 1.0, and starting at it 42-48 % fewer than starting at a tenth of it.
FIRE = {"n_min": 5, "f_inc": 1.1, "f_dec": 0.5, "alpha_start": 0.1, "f_alpha": 0.99, "dt_max": 0.5,
        "dt_start": 0.5}
#: How far past the stated gap the minimiser pushes (in units of the gap), and how much of it every accepted
#: relaxation keeps: every clear gap of a snapshot is at least ``(1 + _GAP_HOLD) g``.
_GAP_PAD = 0.5
_GAP_HOLD = 0.05
#: The Verlet list's skin, in mean radii.
_SKIN = 0.3
#: The smallest gap a path holds, in units of the largest cell side it snapshots: eight times the float32 rounding
#: ``L 2^-24`` of a coordinate, against the few such roundings a gap's float32 evaluation in the walk's wall kernel
#: accumulates (two centres, the minimum-image wrap, two radii).
FLOAT32_GAP = 2.0 ** -21
#: FIRE iterations one relaxation may take before the path is declared jammed.
MAX_RELAX_ITERATIONS = 100_000
#: A relaxation whose largest net force has fallen below this fraction of its largest overlap is force-balanced
#: with the overlap still there: jammed.
_JAM_FORCE = 1e-9


class Jammed(RuntimeError):
    """A relaxation of the path found no zero-overlap state: the packing is at or past its jamming point."""

    def __init__(self, fraction, iterations, overlap, reason):
        self.fraction, self.iterations, self.overlap = float(fraction), int(iterations), float(overlap)
        super().__init__(f"the compression jammed at packing fraction {fraction:.4f}: {reason} after {iterations} "
                         f"FIRE iterations, the largest overlap {overlap:.3e} mean radii")


def _sequential_sum(x):
    """The sum of ``x`` in index order: independent of how NumPy blocks a reduction on this CPU."""
    return float(np.add.accumulate(x.ravel())[-1]) if x.size else 0.0


def _norm2(a):
    """The squared Euclidean norm of every row of ``a`` (``(n, 2)`` or ``(n, 3)``), component by component."""
    out = a[:, 0] * a[:, 0] + a[:, 1] * a[:, 1]
    if a.shape[1] == 3:
        out += a[:, 2] * a[:, 2]
    return out


def _cell_volume_factor(dim):
    return np.pi if dim == 2 else 4.0 * np.pi / 3.0


class _Path:
    """The state of one compression path, in units of the mean radius: positions ``x`` in ``[-L/2, L/2)`` (unwrapped
    between list builds), the cell side ``L``, the Verlet list."""

    def __init__(self, rho, x, L, gap):
        self.rho, self.x, self.L = rho, x, float(L)
        self.n, self.dim = x.shape
        self.pad = (1.0 + _GAP_PAD) * gap               # added to r_i + r_j in the minimised contact distance
        self.tol = (_GAP_PAD - _GAP_HOLD) * gap          # the largest overlap of it an accepted relaxation keeps
        self.sigma_max = 2.0 * float(rho.max()) + self.pad
        self.iterations = 0
        self._build()

    def copy(self):
        other = object.__new__(_Path)
        other.__dict__.update(self.__dict__)
        other.x = self.x.copy()
        return other

    def _build(self):
        from scipy.spatial import cKDTree
        L = self.L
        self.x = self.x - L * np.floor(self.x / L + 0.5)
        tree = cKDTree(self.x + L / 2.0, boxsize=L)
        pairs = tree.query_pairs(min(self.sigma_max + _SKIN, 0.4999 * L), output_type="ndarray")
        if len(pairs):
            pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
            i, j = pairs[:, 0], pairs[:, 1]
            dq = self.x[i] - self.x[j]
            dq -= L * np.round(dq / L)
            d = np.sqrt(_norm2(dq))
            keep = d < self.rho[i] + self.rho[j] + self.pad + _SKIN
            i, j = i[keep], j[keep]
        else:
            i = j = np.zeros(0, np.int64)
        self.i, self.j = i, j
        self.sigma = self.rho[i] + self.rho[j] + self.pad
        self.s_build, self.L_build = self.x / L, L

    def _list_holds(self):
        """No pair outside the list can be in contact: ``(sigma_max + skin) L / L_build - 2 max displacement``
        still exceeds ``sigma_max`` (the affine compression and the moves since the build both close gaps)."""
        moved = (self.x / self.L - self.s_build) * self.L
        disp = float(np.sqrt(np.max(_norm2(moved))))
        return (self.sigma_max + _SKIN) * self.L / self.L_build - 2.0 * disp >= self.sigma_max

    def compress(self, L_new):
        """Shrink the cell affinely to side ``L_new``."""
        self.x = self.x * (float(L_new) / self.L)
        self.L = float(L_new)

    def _overlaps(self):
        L = self.L
        dq = self.x[self.i] - self.x[self.j]
        dq -= L * np.round(dq / L)
        d = np.sqrt(_norm2(dq))
        return dq, d, self.sigma - d

    def relax(self, fraction):
        """FIRE on the harmonic overlap energy until every overlap is at most ``tol``; :class:`Jammed` if the
        energy cannot reach that."""
        p = FIRE
        v = np.zeros_like(self.x)
        dt, alpha, n_pos = p["dt_start"], p["alpha_start"], 0
        for it in range(MAX_RELAX_ITERATIONS + 1):
            if not self._list_holds():
                self._build()
            dq, d, ov = self._overlaps()
            hit = ov > 0.0
            worst = float(ov[hit].max()) if hit.any() else 0.0
            if worst <= self.tol:
                self.iterations += it
                return it
            if it == MAX_RELAX_ITERATIONS:
                raise Jammed(fraction, it, worst, "no zero-overlap state was reached")
            w = (ov[hit] / d[hit])[:, None] * dq[hit]          # the force on i from j, k = 1 on the overlap
            ih, jh = self.i[hit], self.j[hit]
            F = np.empty_like(self.x)
            for a in range(self.dim):
                F[:, a] = np.bincount(ih, w[:, a], self.n) - np.bincount(jh, w[:, a], self.n)
            f2 = F * F
            if it and float(np.sqrt(np.max(_norm2(F)))) < _JAM_FORCE * worst:
                raise Jammed(fraction, it, worst, "the forces balanced with the overlap still there")
            power = _sequential_sum(F * v)
            if power > 0.0:
                f_norm = np.sqrt(_sequential_sum(f2))
                v_norm = np.sqrt(_sequential_sum(v * v))
                v = (1.0 - alpha) * v + (alpha * v_norm / f_norm) * F
                if n_pos > p["n_min"]:
                    dt = min(dt * p["f_inc"], p["dt_max"])
                    alpha *= p["f_alpha"]
                n_pos += 1
            else:
                v[:] = 0.0
                dt *= p["f_dec"]
                alpha, n_pos = p["alpha_start"], 0
            v = v + dt * F
            self.x = self.x + dt * v
        raise AssertionError("unreachable")

    def centers(self):
        L = self.L
        return self.x - L * np.floor(self.x / L + 0.5)


def _stop_index(fraction):
    """The trunk's last stop at or below ``fraction``, and whether ``fraction`` is that stop."""
    k = int(np.floor((fraction - DENSIFY_START) / DENSIFY_STEP + 1e-9))
    on = abs(DENSIFY_START + k * DENSIFY_STEP - fraction) <= 1e-12
    return k, on


def _start(radii, dim, rbar, seed, max_attempts):
    rho = radii / rbar
    pack = pack_cylinders if dim == 2 else pack_spheres
    x, L, _ = pack(rho, target_vf=DENSIFY_START, seed=seed, max_attempts=max_attempts)
    return rho, x, L


def contact_fraction(radii, fraction, dim, min_gap):
    """The packing fraction of the contact radii a path holds apart (each radius plus ``(1 + _GAP_PAD) g / 2``) in
    the cell of ``radii`` at ``fraction``: what :data:`DENSIFY_LIMIT` bounds."""
    r = np.asarray(radii, np.float64)
    c = _cell_volume_factor(dim)
    L = (c * np.sum(r ** dim) / float(fraction)) ** (1.0 / dim)
    return float(c * np.sum((r + 0.5 * (1.0 + _GAP_PAD) * float(min_gap)) ** dim) / L ** dim)


def densify_packing(radii, fractions, dim, *, seed, min_gap, max_attempts=100_000):
    """The snapshots of ONE compression path of ``radii`` (metres) in a periodic cell of dimension ``dim``
    (2: disks, the cross-sections of parallel cylinders; 3: spheres) at each packing fraction of ``fractions``.

    Returns a list, in the order of ``fractions``, of ``(centers, L)``: centres ``(N, dim)`` in ``[-L/2, L/2)``
    (metres; object ``i`` is ``radii[i]`` in every snapshot) and the cell side ``L`` (metres) at which
    ``sum(c_d r^dim) / L^dim`` is that fraction. Every clear gap, periodic images included, is at least
    ``min_gap`` (metres). ``seed`` draws the random-sequential-addition start (``max_attempts`` trials per object
    there). The method is the module's: see its docstring.

    Refused (``ValueError``): a fraction below :data:`DENSIFY_START`, or whose contact radii
    (:func:`contact_fraction`) pack beyond :data:`DENSIFY_LIMIT`, a gap below :data:`FLOAT32_GAP` times the cell side
    of the sparsest snapshot. A relaxation that jams raises :class:`Jammed`, naming the fraction.
    """
    radii = np.asarray(radii, np.float64).ravel()
    dim = int(dim)
    if dim not in DENSIFY_LIMIT:
        raise ValueError(f"dim must be 2 (disks) or 3 (spheres), got {dim}")
    if len(radii) == 0 or np.any(radii <= 0):
        raise ValueError("radii must be positive, at least one")
    g = float(min_gap)
    if not g > 0.0:
        raise ValueError(f"min_gap must be positive (metres): a compression path holds every gap at it, got {min_gap!r}")
    fractions = [float(f) for f in fractions]
    for f in fractions:
        if f < DENSIFY_START - 1e-12:
            raise ValueError(f"packing fraction {f:.4f} is below the densification path's start DENSIFY_START = "
                             f"{DENSIFY_START}")
        reach = contact_fraction(radii, f, dim, g)
        if reach > DENSIFY_LIMIT[dim]:
            raise ValueError(f"packing fraction {f:.4f} ({reach:.4f} in the contact radii, each radius plus "
                             f"{0.5 * (1.0 + _GAP_PAD):.2f} min_gap) is beyond the densification path's reachable "
                             f"limit DENSIFY_LIMIT = {DENSIFY_LIMIT[dim]} in {dim}-D")
    widest = (_cell_volume_factor(dim) * float(np.sum(radii ** dim)) / min(fractions)) ** (1.0 / dim)
    if g < FLOAT32_GAP * widest:
        raise ValueError(f"min_gap {g:.3e} m is below FLOAT32_GAP x the cell side {widest:.3e} m of the sparsest "
                         f"snapshot ({FLOAT32_GAP * widest:.3e} m): the walk's float32 centres would not hold it")
    rbar = float(np.mean(radii))
    rho, x, L0 = _start(radii, dim, rbar, seed, max_attempts)
    vol = _cell_volume_factor(dim) * float(np.sum(rho ** dim))
    side = lambda f: (vol / f) ** (1.0 / dim)
    trunk = _Path(rho, x, L0, g / rbar)
    order = sorted(range(len(fractions)), key=lambda q: fractions[q])
    out = [None] * len(fractions)
    k_done = -1
    for q in order:
        f = fractions[q]
        k, on = _stop_index(f)
        while k_done < k:
            k_done += 1
            phi = DENSIFY_START + k_done * DENSIFY_STEP
            trunk.compress(side(phi))
            trunk.relax(phi)
        state = trunk
        if not on:
            state = trunk.copy()
            state.compress(side(f))
            state.relax(f)
        centers = state.centers() * rbar
        L = state.L * rbar
        gap = periodic_min_gap(centers, radii, L)
        if gap < g:
            raise AssertionError(f"a densified snapshot at {f:.4f} has a gap {gap:.3e} m below min_gap {g:.3e} m")
        out[q] = (centers, L)
    return out


def jamming_fraction(radii, dim, *, seed, min_gap, max_attempts=100_000):
    """The packing fraction (in the radii, not the contact radii) at which this module's compression path of
    ``radii`` jams: the trunk is compressed past :data:`DENSIFY_LIMIT` until a relaxation jams, and the fraction
    returned is the last stop that relaxed (the jamming point lies within one :data:`DENSIFY_STEP` above it)."""
    radii = np.asarray(radii, np.float64).ravel()
    rbar = float(np.mean(radii))
    rho, x, L0 = _start(radii, int(dim), rbar, seed, max_attempts)
    vol = _cell_volume_factor(dim) * float(np.sum(rho ** dim))
    path = _Path(rho, x, L0, float(min_gap) / rbar)
    k = 0
    while True:
        phi = DENSIFY_START + k * DENSIFY_STEP
        path.compress((vol / phi) ** (1.0 / dim))
        try:
            path.relax(phi)
        except Jammed:
            return DENSIFY_START + (k - 1) * DENSIFY_STEP
        k += 1
