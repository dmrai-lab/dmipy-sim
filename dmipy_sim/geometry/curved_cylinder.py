"""CurvedCylinder — a sphere-swept polyline geometry for curving fibres (e.g. DiSCo strands).

The intra-axonal space of a constant-radius fibre that follows an arbitrary curved
centerline is exactly ``{ r : dist(r, centerline_polyline) < R }`` — the Minkowski
sum of the polyline with a ball of radius R. This is smooth everywhere (cylindrical
along each segment, a spherical patch at every joint), so it has none of the
kink/gap/overlap artifacts of a chain of straight finite cylinders, and it captures
the local fibre orientation (which varies along the strand).

Reflection is specular at the wall crossing: the step's exit from the tube (the last of its
segments' capsules along the step, which overlap at every joint) is found on the shared quadric,
the displacement left after it is mirrored about the outward normal there -- the radial direction
from the nearest centerline point, so the cylinder of a segment and the sphere of a joint are one
rule -- and a second bounce takes a remainder that leaves again. A walker keeps to its own tube.

The step rule of the family is ``reflection_step_fraction`` and, the contact channel having been measured
at the same step, ``surface_substep_frac``: ``step_l <= R_min / STEP_FRACTION`` with ``STEP_FRACTION = 3``,
the coarsest step at which every observable a replay pack stores stays within the Monte Carlo floor of
a walk at ``R_min / 12`` on the adversarial fixtures of the family, one million walkers each, 20 ms,
D = 0.6e-9: the thinnest DiSCo strand (R 0.72 um) at its sharpest joint (173 deg, a hairpin), and the
exterior pool in the 0.1 R slot between two such strands. At R/3 the PGSE signals at b = 1000 / 3090 /
13190 s/mm2 are within 1.5 x the split-half floor (0.0005 intra, 0.0011 extra) -- what two independent
walks differ by -- the contact channel is within 0.15 % (intra) and 0.1 % of the R/4 walk (extra), the
displacement variance within 0.35 %, and no walker ends on the wrong side. At R/2 the slot fixture
moves: its contact channel by 1.2 % against R/3 and R/4, its b = 3090 signal to 2.6 x the floor. On
DiSCo's densest region (200k walkers) every level from R/6 to R/2 is within the floor.

Impermeable and analytic — no triangle mesh, no spatial grid — so it is ~orders of
magnitude cheaper than walking the equivalent triangulated tube.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..engine.tables import jit_with_tables
from ._boundary import keep_side_radial, ray_quadric_t, specular, off_wall, representable_nudge
from ._grid import bucket_by_bbox, NEIGHBOUR_OFFSETS, gather
import numpy as np

from .base import Geometry, LengthScales


_BIG = jnp.float32(1e30)


def _capsule_spans(r, step, A, AB, AB2, rr, valid):
    """``(tau_in, tau_out, hit)`` per candidate segment: the interval of the ray ``r + tau * step`` inside the
    capsule of radius ``rr`` swept along the segment ``(A, AB)``. A capsule is convex, so the ray is inside it on
    one interval: the earliest entry and the latest exit over its three pieces -- the infinite cylinder cut to
    the segment's span, and the ball at either end -- each from the shared quadric."""
    L = jnp.sqrt(AB2 + jnp.float32(1e-30)); u = AB / L[:, None]
    rA = r[None, :] - A
    rp = rA - (rA * u).sum(1)[:, None] * u; sp = step[None, :] - (step @ u.T)[:, None] * u
    aa = (sp * sp).sum(1) + jnp.float32(1e-30); bb = (rp * sp).sum(1); cc = (rp * rp).sum(1) - rr * rr
    c_in, c_out, c_disc = ray_quadric_t(aa, bb, cc)
    t0 = (rA * AB).sum(1) / AB2; t1 = (step @ AB.T) / AB2                 # the span coordinate along the ray
    t1s = jnp.where(jnp.abs(t1) > 1e-30, t1, 1e-30)
    s_lo = jnp.minimum(-t0 / t1s, (1.0 - t0) / t1s); s_hi = jnp.maximum(-t0 / t1s, (1.0 - t0) / t1s)
    flat = jnp.abs(t1) <= 1e-30                                          # the ray runs across the axis
    s_lo = jnp.where(flat, jnp.where((t0 >= 0) & (t0 <= 1), -_BIG, _BIG), s_lo)
    s_hi = jnp.where(flat, jnp.where((t0 >= 0) & (t0 <= 1), _BIG, -_BIG), s_hi)
    cyl_in = jnp.maximum(c_in, s_lo); cyl_out = jnp.minimum(c_out, s_hi)
    cyl_hit = (c_disc >= 0) & (cyl_in <= cyl_out)
    ss = step @ step + jnp.float32(1e-30)
    a_in, a_out, a_disc = ray_quadric_t(ss, rA @ step, (rA * rA).sum(1) - rr * rr)
    rB = rA - AB
    b_in, b_out, b_disc = ray_quadric_t(ss, rB @ step, (rB * rB).sum(1) - rr * rr)
    a_hit = a_disc >= 0; b_hit = b_disc >= 0
    tau_in = jnp.minimum(jnp.where(cyl_hit, cyl_in, _BIG), jnp.minimum(jnp.where(a_hit, a_in, _BIG), jnp.where(b_hit, b_in, _BIG)))
    tau_out = jnp.maximum(jnp.where(cyl_hit, cyl_out, -_BIG), jnp.maximum(jnp.where(a_hit, a_out, -_BIG), jnp.where(b_hit, b_out, -_BIG)))
    hit = valid & (cyl_hit | a_hit | b_hit)
    return tau_in, tau_out, hit


def _union_exit(r, step, A, AB, AB2, rr, valid):
    """``(tau, j, leaves)``: where the ray ``r + tau * step`` leaves the union of the candidate capsules and which
    capsule it leaves through. The intervals inside each capsule are chained from ``tau = 0`` (a tube's capsules
    overlap at its joints, so a step runs from one into the next); ``leaves`` when the chain ends on the step."""
    tau_in, tau_out, hit = _capsule_spans(r, step, A, AB, AB2, rr, valid)
    eps = jnp.float32(1e-6)
    def chain(_, tau):
        ext = hit & (tau_in <= tau + eps) & (tau_out > tau)
        return jnp.where(ext.any(), jnp.where(ext, tau_out, -_BIG).max(), tau)
    tau = jax.lax.fori_loop(0, 3, chain, jnp.float32(0.0))
    j = jnp.argmax(jnp.where(hit & (tau_out >= tau - eps) & (tau_in <= tau + eps), tau_out, -_BIG))
    inside0 = (hit & (tau_in <= eps) & (tau_out > -eps)).any()
    return tau, j, inside0 & (tau < 1.0)


def _interior_bounce(r, step, A, AB, AB2, rr, valid, NUDGE):
    """One specular bounce of an interior walker off the wall of its tube, the union of the capsules given:
    ``(r_start, rem, r_end, d_perp, bounced)``. The exit crossing of the union is found along the step, the
    displacement left after it is mirrored about the wall's outward normal there (the radial direction from the
    nearest centerline point, so the tube's cylinder and the joint's sphere are one rule), and the walker is set
    ``NUDGE`` inside the wall; ``d_perp`` is that remainder on the normal, the contact the base slab rule reads.
    ``r_start`` / ``rem`` are the point off the wall and the mirrored remainder, the next bounce's ray."""
    tau, j, leaves = _union_exit(r, step, A, AB, AB2, rr, valid)
    exit_ = r + tau * step
    Aj = A[j]; ABj = AB[j]
    t = jnp.clip(((exit_ - Aj) @ ABj) / AB2[j], 0.0, 1.0); Q = Aj + t * ABj
    ne = exit_ - Q
    n = ne / (jnp.linalg.norm(ne) + jnp.float32(1e-30))
    rem = specular((1.0 - tau) * step, n)
    r_start = off_wall(exit_, n, True, NUDGE)
    r_end = r_start + rem
    d_perp = jnp.abs(((1.0 - tau) * step) @ n)
    return r_start, rem, r_end, jnp.where(leaves, d_perp, jnp.float32(0.0)), leaves


def _reflect_interior(r, step, A, AB, AB2, rr, valid, tube, NUDGE):
    """The interior wall interaction: ``(r_out, d_perp)``. A walker is confined to ITS tube -- the one it is deepest
    in where the step starts (``tube`` labels each candidate segment's tube) -- and that tube is the union of its
    own segments' capsules, which overlap at every joint. Two specular bounces off its wall (a grazing exit off a
    thin tube can send the remainder out again), then the guarantee: a walker still outside its tube is put
    ``NUDGE`` inside the segment it is least outside of, the tie rule every geometry shares."""
    r_new = r + step
    t0 = jnp.clip(((r[None, :] - A) * AB).sum(1) / AB2, 0.0, 1.0)
    d0 = jnp.linalg.norm(r[None, :] - (A + t0[:, None] * AB), axis=1)
    own = tube[jnp.argmax(jnp.where(valid, rr - d0, -_BIG))]
    mine = valid & (tube == own)
    s1, rem1, e1, dp1, b1 = _interior_bounce(r, step, A, AB, AB2, rr, mine, NUDGE)
    s2, rem2, e2, dp2, b2 = _interior_bounce(s1, rem1, A, AB, AB2, rr, mine, NUDGE)
    r_out = jnp.where(b1, jnp.where(b2, e2, e1), r_new)
    d_perp = jnp.where(b1, dp1 + jnp.where(b2, dp2, 0.0), jnp.float32(0.0))
    t = jnp.clip(((r_out[None, :] - A) * AB).sum(1) / AB2, 0.0, 1.0)
    Q = A + t[:, None] * AB
    d = jnp.linalg.norm(r_out[None, :] - Q, axis=1)
    k = jnp.argmax(jnp.where(mine, rr - d, -_BIG))                            # its tube's segment it is deepest in
    r_out, _ = keep_side_radial(r_out, r_out - Q[k], rr[k], True, NUDGE)
    return r_out, d_perp


def _exterior_bounce(r, step, A, AB, AB2, rr, valid, NUDGE):
    """One specular bounce of an exterior walker off the first tube its step enters: ``(r_start, rem, r_end,
    d_perp, hit)``. The tube is the one the endpoint is deepest in; the entry crossing is the first root of the
    shared quadric on that segment's cylinder, the radial part of the displacement left after it is mirrored
    about the outward normal there (the axial part continues), the walker is set ``NUDGE`` outside the wall, and
    ``d_perp`` is that remainder on the normal, the contact the exact packed cylinders read. ``r_start`` / ``rem``
    are the point off the wall and the mirrored remainder, the next bounce's ray."""
    r_new = r + step
    t = jnp.clip(((r_new[None, :] - A) * AB).sum(1) / AB2, 0.0, 1.0)
    d = jnp.linalg.norm(r_new[None, :] - (A + t[:, None] * AB), axis=1)
    inside = valid & (d < rr)
    i = jnp.argmax(jnp.where(inside, rr - d, -jnp.inf))              # the entered (deepest) tube
    Ai = A[i]; ABi = AB[i]; rh = rr[i]
    u = ABi / jnp.sqrt(AB2[i] + jnp.float32(1e-30))                  # segment axis (unit)
    rp = (r - Ai) - ((r - Ai) @ u) * u                               # start, radial to axis
    sp = step - (step @ u) * u                                       # step, radial to axis
    aa = sp @ sp + jnp.float32(1e-30); bb = 2.0 * (rp @ sp); cc = rp @ rp - rh * rh
    tau, _, _ = ray_quadric_t(aa, jnp.float32(0.5) * bb, cc)
    tau = jnp.clip(tau, 0.0, 1.0)                                    # first surface crossing
    entry = r + tau * step
    rem = (1.0 - tau) * step
    rem_ax = (rem @ u) * u
    rem_p = rem - rem_ax
    ne = (entry - Ai) - ((entry - Ai) @ u) * u
    nhat = ne / (jnp.linalg.norm(ne) + jnp.float32(1e-30))          # outward radial normal
    r_start = off_wall(entry, nhat, False, NUDGE)
    rem_ref = rem_ax + specular(rem_p, nhat)
    hit = inside.any()
    return r_start, rem_ref, r_start + rem_ref, jnp.where(hit, jnp.abs(rem_p @ nhat), jnp.float32(0.0)), hit


def _reflect_exterior(r, step, A, AB, AB2, rr, valid, NUDGE):
    """The exterior wall interaction against the candidate capsules: ``(r_out, d_perp)``. Two specular bounces (in
    a gap narrower than the step the remainder enters the facing tube), then the guarantee: a walker that still
    ends inside a tube is put ``NUDGE`` outside the one it is deepest in, the tie rule every geometry shares."""
    r_new = r + step
    s1, rem1, e1, dp1, h1 = _exterior_bounce(r, step, A, AB, AB2, rr, valid, NUDGE)
    s2, rem2, e2, dp2, h2 = _exterior_bounce(s1, rem1, A, AB, AB2, rr, valid, NUDGE)
    r_out = jnp.where(h1, jnp.where(h2, e2, e1), r_new)
    d_perp = jnp.where(h1, dp1 + jnp.where(h2, dp2, 0.0), jnp.float32(0.0))
    t = jnp.clip(((r_out[None, :] - A) * AB).sum(1) / AB2, 0.0, 1.0)
    Q = A + t[:, None] * AB
    d = jnp.linalg.norm(r_out[None, :] - Q, axis=1)
    k = jnp.argmax(jnp.where(valid, rr - d, -_BIG))                   # the tube it is deepest in, if any
    r_out, _ = keep_side_radial(r_out, r_out - Q[k], rr[k], False, NUDGE)
    return r_out, d_perp


#: The step rule of the curved-tube family, ``step_l <= R_min / STEP_FRACTION`` (the module docstring): measured on the
#: family's adversarial fixtures, one million walkers each, and the value every driver takes through
#: ``reflection_step_fraction`` and ``surface_substep_frac``.
STEP_FRACTION = 3.0


class CurvedCylinder(Geometry):
    reflection_step_fraction = STEP_FRACTION
    surface_substep_frac = STEP_FRACTION

    def __init__(self, centerline, radius: float, surface_relaxivity_t2=None):
        self.surface_relaxivity_t2 = None if surface_relaxivity_t2 is None else float(surface_relaxivity_t2)
        cl = np.asarray(centerline, np.float64)          # (P, 3) metres
        if cl.ndim != 2 or cl.shape[0] < 2:
            raise ValueError("centerline must be (P>=2, 3)")
        #: the wall nudge: a fraction of the radius, and never below what a float32 coordinate of this strand can carry
        self.nudge_m = representable_nudge(1e-4 * float(radius), np.abs(cl).max() + float(radius))
        self.centerline = cl
        self.radius = float(radius)
        A = cl[:-1]
        B = cl[1:]
        self._A = jnp.asarray(A, jnp.float32)            # (M, 3)
        self._AB = jnp.asarray(B - A, jnp.float32)       # (M, 3)
        seglen2 = np.maximum(((B - A) ** 2).sum(1), 1e-30)
        self._AB2 = jnp.asarray(seglen2, jnp.float32)    # (M,)
        self._seglen = np.sqrt(seglen2)

    @property
    def length_scales(self):
        return LengthScales(min_feature=self.radius)

    # ---- containment / geometry ----
    def _nearest(self, r):
        """Nearest point on the centerline polyline to r, and the distance."""
        rA = r[None, :] - self._A                                   # (M,3)
        t = jnp.clip((rA * self._AB).sum(1) / self._AB2, 0.0, 1.0)  # (M,)
        Q = self._A + t[:, None] * self._AB                         # (M,3)
        dvec = r[None, :] - Q
        d2 = (dvec * dvec).sum(1)                                   # (M,)
        i = jnp.argmin(d2)
        return Q[i], jnp.sqrt(d2[i])

    def classify_position(self, r):
        """Compartment id: 1 inside the tube, 0 outside."""
        _, d = self._nearest(r)
        return jnp.where(d < jnp.float32(self.radius), jnp.int32(1), jnp.int32(0))

    def volume(self) -> float:
        return float(self._seglen.sum() * np.pi * self.radius ** 2)

    # ---- seeding: uniform inside the tube (arc-uniform x disk-uniform) ----
    def init_positions(self, n_walkers, key):
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2 ** 30)))
        cl = self.centerline
        seg = cl[1:] - cl[:-1]
        L = self._seglen
        cumL = np.cumsum(L)
        total = float(cumL[-1])
        s = rng.uniform(0.0, total, n_walkers)
        idx = np.searchsorted(cumL, s, side="right")
        idx = np.clip(idx, 0, len(L) - 1)
        s0 = np.concatenate([[0.0], cumL])[idx]
        frac = np.clip((s - s0) / L[idx], 0.0, 1.0)
        C = cl[:-1][idx] + frac[:, None] * seg[idx]
        T = seg[idx] / L[idx][:, None]
        # arbitrary perpendicular frame per point
        ref = np.tile(np.array([0.0, 0.0, 1.0]), (n_walkers, 1))
        par = np.abs((T * ref).sum(1)) > 0.9
        ref[par] = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(T, ref); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(T, e1)
        rr = (self.radius - self.nudge_m) * np.sqrt(rng.uniform(0.0, 1.0, n_walkers))   # inside by the nudge: a seed
        th = rng.uniform(0.0, 2 * np.pi, n_walkers)                                       # on the wall reads as outside
        off = rr[:, None] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
        return jnp.asarray(C + off, jnp.float32)

    # ---- specular reflection off the swept-tube wall ----
    def reflect(self, r, step):
        return self._reflect_contact(r, step)[0]

    def reflect_with_log_weight(self, r, step, rho_over_D):
        """The reflection with the surface-relaxation log-weight ``-2 (rho / D) d_perp`` of the wall contact:
        ``d_perp`` is the displacement left after the wall crossing read on the wall's normal, the perpendicular
        distance the base slab rule uses."""
        r_out, d_perp = self._reflect_contact(r, step)
        return r_out, -2.0 * rho_over_D * d_perp

    def _reflect_contact(self, r, step):
        R = jnp.float32(self.radius)
        NUDGE = jnp.float32(self.nudge_m)
        M = self._A.shape[0]
        return _reflect_interior(r, step, self._A, self._AB, self._AB2, jnp.full((M,), R, jnp.float32),
                                 jnp.ones((M,), bool), jnp.zeros((M,), jnp.int32), NUDGE)


class CurvedMyelinatedCylinder(CurvedCylinder):
    """A myelinated curved axon: concentric intra / myelin / extra shells swept along a
    curved centerline. Compartment by distance-to-centerline d:
      1 intra  (d < r_in),  2 myelin (r_in <= d < r_out),  0 extra (d >= r_out).
    Impermeable band-confined reflection keeps each walker in the shell it started in, so
    the three compartments are independent: ``init_positions`` seeds the pool named at construction
    (``pool=``) and the walk takes that compartment's diffusivity (intra ~free, myelin
    ~stuck D->0, extra free). Because the local tangent varies along the strand, the
    myelin annulus carries the orientation-varying susceptibility source. (Single-pass
    per-walker-D, mirroring ``MyelinatedCylinder._is_myelinated``, is the later
    optimisation; impermeable shells make the separate-walk form exact.)
    """

    def __init__(self, centerline, r_in: float, r_out: float, pool="intra", surface_relaxivity_t2=None):
        super().__init__(centerline, r_out, surface_relaxivity_t2)   # base extent = outer radius
        if not (r_out > r_in > 0):
            raise ValueError("need r_out > r_in > 0")
        self.r_in = float(r_in)
        self.r_out = float(r_out)
        self.nudge_m = representable_nudge(1e-4 * self.r_in, np.abs(self.centerline).max() + self.r_out)
        if pool not in ("intra", "myelin", "extra"):
            raise ValueError(f"pool must be 'intra', 'myelin' or 'extra', got {pool!r}")
        self.pool = pool                                   # the shell init_positions seeds
        self.radius = float(r_in)                      # auto-tune to the finest wall

    def classify_position(self, r):
        """Compartment id: 0 extra (d >= r_out), 1 intra (d < r_in), 2 myelin."""
        _, d = self._nearest(r)
        return jnp.where(d < jnp.float32(self.r_in), jnp.int32(1),
                         jnp.where(d < jnp.float32(self.r_out), jnp.int32(2), jnp.int32(0)))

    def reflect(self, r, step):
        return self._reflect_contact(r, step)[0]

    def reflect_with_log_weight(self, r, step, rho_over_D):
        """The band-confined reflection with the contact log-weight of the wall hit (either edge of the band)."""
        r_out, d_perp = self._reflect_contact(r, step)
        return r_out, -2.0 * rho_over_D * d_perp

    def _reflect_contact(self, r, step):
        r_in = jnp.float32(self.r_in); r_out = jnp.float32(self.r_out)
        NUDGE = jnp.float32(self.nudge_m)
        _, do = self._nearest(r)                       # band of the OLD position
        lo = jnp.where(do < r_in, jnp.float32(0.0), jnp.where(do < r_out, r_in, r_out))
        hi = jnp.where(do < r_in, r_in, jnp.where(do < r_out, r_out, jnp.float32(np.inf)))
        r_new = r + step
        Q, d = self._nearest(r_new)
        n = (r_new - Q) / (d + jnp.float32(1e-30))
        dt = d
        dt = jnp.where(d >= hi, 2.0 * hi - d - NUDGE, dt)  # mirror at the band's outer wall
        dt = jnp.where(d <= lo, 2.0 * lo - d + NUDGE, dt)  # mirror at the band's inner wall
        d_perp = jnp.maximum(jnp.where(jnp.isfinite(hi), d - hi, jnp.float32(-1.0)), jnp.float32(0.0)) \
            + jnp.where(lo > 0, jnp.maximum(lo - d, jnp.float32(0.0)), jnp.float32(0.0))       # the overshoot past either edge
        # Equality counts as the wrong side for BOTH neighbours (see _boundary): a walker
        # landing exactly on r_in or r_out belongs to neither band, and the strict `>` / `<`
        # used here previously left that tie unresolved -- the same defect that let walkers
        # change compartment without moving in the analytic geometries (#86). A mirror alone
        # also has no guarantee, so clamp the result into [lo, hi] explicitly.
        dt = jnp.clip(dt, lo + NUDGE, jnp.where(jnp.isfinite(hi), hi - NUDGE, dt))
        return Q + dt * n, d_perp

    def init_positions(self, n_walkers, key, pool=None):
        """Walkers seeded uniformly in the shell ``pool`` (default: the geometry's ``pool``) along the strand."""
        shell = self.pool if pool is None else pool
        lo, hi = {"intra": (0.0, self.r_in),
                  "myelin": (self.r_in, self.r_out),
                  "extra": (self.r_out, 1.5 * self.r_out)}[shell]
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2 ** 30)))
        cl = self.centerline; seg = cl[1:] - cl[:-1]; L = self._seglen
        cumL = np.cumsum(L); total = float(cumL[-1])
        s = rng.uniform(0.0, total, n_walkers)
        idx = np.clip(np.searchsorted(cumL, s, side="right"), 0, len(L) - 1)
        s0 = np.concatenate([[0.0], cumL])[idx]
        frac = np.clip((s - s0) / L[idx], 0.0, 1.0)
        C = cl[:-1][idx] + frac[:, None] * seg[idx]
        T = seg[idx] / L[idx][:, None]
        ref = np.tile(np.array([0.0, 0.0, 1.0]), (n_walkers, 1))
        ref[np.abs((T * ref).sum(1)) > 0.9] = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(T, ref); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(T, e1)
        lo, hi = (lo + self.nudge_m if lo > 0 else 0.0), hi - self.nudge_m           # inside the band by the nudge
        rr = np.sqrt(rng.uniform(lo ** 2, hi ** 2, n_walkers))   # uniform-in-area radius
        th = rng.uniform(0.0, 2 * np.pi, n_walkers)
        off = rr[:, None] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
        return jnp.asarray(C + off, jnp.float32)


class PackedCurvedCylinders(Geometry):
    """Extra-axonal diffusion around a pack of curved tubes, accelerated by a sparse grid
    over the tube *segments* (~O(#segments), not #triangles). An extra walker must not
    enter any tube: each step gathers the tube segments in the walker's 27-cell
    neighbourhood, and if the step would put it inside any tube (dist-to-segment < r_out)
    it reflects specularly off that tube's outer wall at the entry crossing (a second bounce takes
    a remainder that enters the facing tube of a narrow gap). An interior walker (``interior=True``)
    keeps to its own tube -- the one it is deepest in where its step starts -- and reflects at its
    exit from that tube; the strands are separate axons even where the data lets them touch (0.33 %
    of DiSCo's inner tube volume lies in two strands). This is the ~100x-lighter counterpart of a
    triangle-mesh grid for the same geometry. The step rule is the family's ``STEP_FRACTION``.
    """
    reflection_step_fraction = STEP_FRACTION
    surface_substep_frac = STEP_FRACTION

    def __init__(self, centerlines, radii, cell_size=None, interior=False, box=None, box_reflect=True,
                 surface_relaxivity_t2=None):
        self.surface_relaxivity_t2 = None if surface_relaxivity_t2 is None else float(surface_relaxivity_t2)
        # interior=False: extra-axonal (bounce off tube exteriors, stay outside all tubes)
        # interior=True : intra-axonal, all tubes at once (each walker confined inside its
        #                 own tube), one grid/one JIT for all tubes.
        # box=(lo, hi) : a finite, non-periodic voxel; with box_reflect its faces are mirrors that never
        #                carry a walker across a tube wall (the fold is refused, like an escaping step).
        self.interior = bool(interior)
        self.box = None if box is None else (np.asarray(box[0], float), np.asarray(box[1], float))
        self.box_reflect = bool(box_reflect) and self.box is not None
        if self.box_reflect:
            self._lo, self._hi = jnp.asarray(self.box[0], jnp.float32), jnp.asarray(self.box[1], jnp.float32)
        self.centerlines = [np.asarray(cl, np.float64) for cl in centerlines]     # what the pack was built from
        self.radii = np.asarray(radii, np.float64).reshape(-1)
        A, AB, rr = [], [], []
        for cl, R in zip(centerlines, radii):
            cl = np.asarray(cl, np.float64)
            A.append(cl[:-1]); AB.append(cl[1:] - cl[:-1]); rr.append(np.full(len(cl) - 1, float(R)))
        A = np.vstack(A); AB = np.vstack(AB); rout = np.concatenate(rr)
        self._seg_tube = jnp.asarray(np.concatenate([np.full(len(cl) - 1, k) for k, cl in enumerate(centerlines)]), jnp.int32)
        self._A = jnp.asarray(A, jnp.float32)
        self._AB = jnp.asarray(AB, jnp.float32)
        self._AB2 = jnp.asarray(np.maximum((AB ** 2).sum(1), 1e-30), jnp.float32)
        self._rout = jnp.asarray(rout, jnp.float32)
        self._Rmin = float(rout.min()); self._Rmax = float(rout.max())
        self.radius = self._Rmin                       # auto-tune to the finest wall
        cs = float(cell_size) if cell_size else (4.0 * self._Rmin / 6.0 + 2.0 * self._Rmax)
        self.cell_size = cs
        lo = np.minimum(A, A + AB) - self._Rmax
        hi = np.maximum(A, A + AB) + self._Rmax
        self.gmin = lo.min(0) - cs
        self.dims = np.maximum(1, np.ceil((hi.max(0) + cs - self.gmin) / cs).astype(int))
        corners = [self.gmin, self.gmin + self.dims * cs] + ([self.box[0], self.box[1]] if self.box is not None else [])
        #: the wall nudge: a fraction of the smallest radius, and never below what a float32 coordinate of this pack can carry
        self.nudge_m = representable_nudge(1e-4 * self._Rmin, np.abs(np.concatenate(corners)).max())
        loc = np.clip(np.floor((lo - self.gmin) / cs).astype(int), 0, self.dims - 1)
        hic = np.clip(np.floor((hi - self.gmin) / cs).astype(int), 0, self.dims - 1)
        cell, self.C, _max_occ, _overflow = bucket_by_bbox(loc, hic, self.dims, None)
        self._CELL = jnp.asarray(cell, jnp.int32)
        self._DIMS = tuple(int(x) for x in self.dims)
        self._dims_arr = jnp.asarray(self._DIMS, jnp.int32)
        self._GMIN = jnp.asarray(self.gmin, jnp.float32)
        self._CS = jnp.float32(cs)
        self._OFF = jnp.asarray(NEIGHBOUR_OFFSETS)

    #: the device tables every jitted program of this geometry reads -- passed as arguments at every call
    #: (:func:`~dmipy_sim.engine.tables.jit_with_tables`), never captured: a program per batch shape and candidate
    #: width that embedded them held hundreds of copies of the segment tables on the host
    TABLES = ("_A", "_AB", "_AB2", "_rout", "_seg_tube", "_CELL")

    def classify_positions_exact(self, pts, chunk=100_000):
        """The exact labels of a batch of host-side points, in chunks; the program is built once per instance
        and reads the tables as arguments."""
        f = getattr(self, "_classify_batch", None)
        if f is None:
            f = self._classify_batch = jit_with_tables(self, self.TABLES, jax.vmap(self.classify_position))
        pts = np.asarray(pts, np.float32)
        return jnp.concatenate([f(jnp.asarray(pts[i:i + chunk])) for i in range(0, pts.shape[0], chunk)])

    def _gather(self, r):
        return gather(self._CELL, self._OFF, self._GMIN, self._CS, self._dims_arr, r)

    @property
    def length_scales(self):
        # a real tube radius (the family's step rule applies) and a segment grid (the lookup rule applies)
        return LengthScales(min_feature=self._Rmin, lookup_cell=self.cell_size)

    classify_returns_object_id = True

    def classify_position(self, r):
        """Compartment id: ``k + 1`` inside tube ``k`` (1-indexed as every packed geometry; where tubes overlap or
        a thin tube runs close to a fat one, the tube the point is DEEPEST inside -- inside ANY tube, not only the
        nearest axis, which misread a walker in a fat tube close to a thin one's axis as outside), 0 outside every
        tube. The record of an interior walk carried pool 0 without this, and the pack then weighted every walker
        with the extra-cellular water fraction."""
        cand, valid = self._gather(r)
        A = self._A[cand]; AB = self._AB[cand]; AB2 = self._AB2[cand]; rr = self._rout[cand]
        t = jnp.clip(((r[None, :] - A) * AB).sum(1) / AB2, 0.0, 1.0)
        d = jnp.sqrt(((r[None, :] - (A + t[:, None] * AB)) ** 2).sum(1))
        depth = jnp.where(valid, rr - d, -jnp.inf)                   # positive inside a tube
        i = jnp.argmax(depth)
        return jnp.where(depth[i] > 0, self._seg_tube[cand[i]] + 1, 0).astype(jnp.int32)

    def inside_any(self, P, chunk=50000):
        """(n,3) → (n,) bool: is each point inside ANY tube (dist-to-segment < r_out)?
        Grid-accelerated (each point tests only its 27-cell segment neighbourhood) and
        GPU-vmapped in chunks — the fast primitive for seeding the extra-axonal space
        (rejection over ~O(#segments-per-cell), not the whole pack)."""
        P = np.asarray(P, np.float32)
        out = np.empty(P.shape[0], bool)

        # Built ONCE per instance, not per call. jax.jit caches compiled programs on the identity of
        # the function object it wraps, so a jit defined in this method body was a fresh object every
        # call and recompiled every time -- and `sample_outside` calls this in a rejection LOOP, so
        # that was a recompile per iteration. The closure captures this instance's segment arrays,
        # which are fixed at construction, so caching per instance (not module-wide) is the correct
        # scope. Measured on the same pattern in mesh.py: ~1.4 s per call -> 0.0002 s once hoisted.
        _batch = getattr(self, "_inside_any_batch", None)
        if _batch is None:
            def _batch_body(Pb):
                def one(p):
                    cand, valid = self._gather(p)
                    A = self._A[cand]; AB = self._AB[cand]; AB2 = self._AB2[cand]; rr = self._rout[cand]
                    t = jnp.clip(((p[None, :] - A) * AB).sum(1) / AB2, 0.0, 1.0)
                    d = jnp.linalg.norm(p[None, :] - (A + t[:, None] * AB), axis=1)
                    return (valid & (d < rr)).any()
                return jax.vmap(one)(Pb)
            _batch = self._inside_any_batch = jit_with_tables(self, self.TABLES, _batch_body)

        for i in range(0, P.shape[0], chunk):
            out[i:i + chunk] = np.asarray(_batch(jnp.asarray(P[i:i + chunk])))
        return out

    def radial_directors(self, P, chunk=50000):
        """``(n, 3)`` -> ``(n, 3)`` unit vectors from the nearest centerline point to each point: the sheath's
        radial (lipid) director at that point, exact from the geometry rather than from the gradient of a
        voxelised mask (dmipy-sim#213). Grid-accelerated like :meth:`inside_any`; a point with no segment in its
        neighbourhood gets the zero vector."""
        P = np.asarray(P, np.float32)
        out = np.zeros((P.shape[0], 3), np.float32)
        _batch = getattr(self, "_radial_batch", None)
        if _batch is None:
            def _batch_body(Pb):
                def one(p):
                    cand, valid = self._gather(p)
                    A = self._A[cand]; AB = self._AB[cand]; AB2 = self._AB2[cand]
                    t = jnp.clip(((p[None, :] - A) * AB).sum(1) / AB2, 0.0, 1.0)
                    Q = A + t[:, None] * AB
                    d2 = jnp.where(valid, ((p[None, :] - Q) ** 2).sum(1), jnp.inf)
                    i = jnp.argmin(d2)
                    v = p - Q[i]
                    n = jnp.sqrt((v * v).sum())
                    return jnp.where(valid.any() & (n > 0), v / jnp.maximum(n, 1e-30), jnp.zeros(3, v.dtype))
                return jax.vmap(one)(Pb)
            _batch = self._radial_batch = jit_with_tables(self, self.TABLES, _batch_body)
        for i in range(0, P.shape[0], chunk):
            out[i:i + chunk] = np.asarray(_batch(jnp.asarray(P[i:i + chunk])))
        return out

    def sample_outside(self, n_walkers, rng, bounds=None):
        """Uniformly sample `n_walkers` points in the extra-axonal space (outside all
        tubes). `bounds=(lo,hi)` overrides the tube bounding box (e.g. the voxel domain)."""
        lo = np.asarray(bounds[0]) if bounds else self.gmin
        hi = np.asarray(bounds[1]) if bounds else (self.gmin + self.dims * self.cell_size)
        acc = []; got = 0
        while got < n_walkers:
            P = rng.uniform(lo, hi, (max(n_walkers, 100000) * 2, 3))
            P = P[~self.inside_any(P)]
            acc.append(P); got += len(P)
        return np.concatenate(acc)[:n_walkers].astype(np.float32)

    def _inside_one(self, p, cand=None, valid=None):
        """Pure-JAX membership of one point in any tube, against the given candidate segments or, without them,
        the 27-cell gather at ``p``."""
        if cand is None:
            cand, valid = self._gather(p)
        A = self._A[cand]; AB = self._AB[cand]; AB2 = self._AB2[cand]; rr = self._rout[cand]
        t = jnp.clip(((p[None, :] - A) * AB).sum(1) / AB2, 0.0, 1.0)
        d = jnp.linalg.norm(p[None, :] - (A + t[:, None] * AB), axis=1)
        return (valid & (d < rr)).any()

    def wall_scales(self, P, chunk=100_000):
        """``(n, 3) -> (d_wall (n,), R_near (n,))``: each point's distance to the nearest wall it can hit -- the
        nearest tube's surface (outside all tubes for the exterior pool, the confining tube's wall for the interior
        one) and, when the pack mirrors at a box, the nearest face -- and the radius of that nearest tube, the
        curvature scale of the wall. What an adaptive walk steps by (:mod:`dmipy_sim.engine.adaptive`): a walker
        farther from every wall than its round's excursion takes one free step, the rest step at the family's
        step rule of THEIR tube rather than the pack's smallest. A point with no tube in reach gets ``inf`` and the
        pack's largest radius."""
        P = np.asarray(P, np.float32)
        out_d = np.empty(P.shape[0], np.float32); out_r = np.empty(P.shape[0], np.float32)
        _batch = self._wall_scales_device()
        for i in range(0, P.shape[0], chunk):
            d, r = _batch(jnp.asarray(P[i:i + chunk]))
            out_d[i:i + chunk] = np.asarray(d); out_r[i:i + chunk] = np.asarray(r)
        return out_d, out_r

    def _wall_scales_device(self):
        """The jitted ``(n, 3) -> (d_wall, R_near)`` of :meth:`wall_scales` on device arrays, built once."""
        _batch = getattr(self, "_wall_scales_batch", None)
        if _batch is None:
            interior = self.interior; box = self.box_reflect
            lo = self._lo if box else None; hi = self._hi if box else None
            R_max = jnp.float32(self._Rmax)

            def _batch_body(Pb):
                def one(p):
                    cand, valid = self._gather(p)
                    A = self._A[cand]; AB = self._AB[cand]; AB2 = self._AB2[cand]; rr = self._rout[cand]
                    t = jnp.clip(((p[None, :] - A) * AB).sum(1) / AB2, 0.0, 1.0)
                    d = jnp.linalg.norm(p[None, :] - (A + t[:, None] * AB), axis=1)
                    dd = jnp.where(valid, d, jnp.inf)
                    i = jnp.argmin(dd)
                    if interior:
                        d_wall = jnp.where(valid.any(), rr[i] - dd[i], jnp.inf)
                    else:
                        d_wall = jnp.where(valid, d - rr, jnp.inf).min()
                    R_near = jnp.where(valid.any(), rr[i], R_max)
                    if box:
                        d_wall = jnp.minimum(d_wall, jnp.minimum((p - lo).min(), (hi - p).min()))
                    return jnp.maximum(d_wall, 0.0), R_near
                return jax.vmap(one)(Pb)
            _batch = self._wall_scales_batch = jit_with_tables(self, self.TABLES, _batch_body)
        return _batch

    def _fold(self, r, r_new, cand=None, valid=None):
        """Mirror into the voxel, never across a tube wall. The membership of the mirrored point is tested against
        ``cand`` / ``valid`` when given -- the segments the step was reflected against, which cover the mirror
        image (it lies within two steps of ``r_new``) -- else against a gather of its own; that second gather per
        step was 20x the cost of the reflection itself."""
        if not self.box_reflect:
            return r_new
        span = self._hi - self._lo
        x = (r_new - self._lo) % (2.0 * span)
        folded = self._lo + jnp.where(x > span, 2.0 * span - x, x)
        ok = self._inside_one(folded, cand, valid) == self.interior
        return jnp.where(ok, folded, r)

    def _step_with(self, r, step, cand, valid):
        """One wall interaction and the box fold against one candidate list: ``(r_out, d_perp)``."""
        r_ref, d_perp = self._reflect_with(r, step, cand, valid)
        return self._fold(r, r_ref, cand, valid), d_perp

    def reflect(self, r, step):
        cand, valid = self._gather(r + step)
        return self._step_with(r, step, cand, valid)[0]

    def reflect_with_log_weight(self, r, step, rho_over_D):
        """The reflection with the contact log-weight ``-2 (rho / D) d_perp``: the radial part of the displacement
        left after the wall crossing, on that wall's normal, as the exact packed cylinders read it -- inside, the
        exit from the walker's own tube; outside, the entry into a tube."""
        cand, valid = self._gather(r + step)
        r_out, d_perp = self._step_with(r, step, cand, valid)
        return r_out, -2.0 * rho_over_D * d_perp

    def _reflect(self, r, step):
        """The wall interaction against every segment near the step's end (the 27-cell gather)."""
        cand, valid = self._gather(r + step)
        return self._reflect_with(r, step, cand, valid)

    def reach_candidates(self, r, reach, k):
        """``(cand (k,), valid (k,), n_within)``: the segments whose SURFACE lies within ``reach`` of ``r`` -- every
        segment a walker can meet while it stays within ``reach`` of ``r`` -- the nearest ``k`` of them padded
        with invalid entries, and how many there were (more than ``k``: the list is short, and the caller must
        widen it). What a round of an adaptive walk gathers once and steps against (:mod:`engine.adaptive`)."""
        cand, valid = self._gather(r)
        A = self._A[cand]; AB = self._AB[cand]; AB2 = self._AB2[cand]; rr = self._rout[cand]
        t = jnp.clip(((r[None, :] - A) * AB).sum(1) / AB2, 0.0, 1.0)
        d = jnp.linalg.norm(r[None, :] - (A + t[:, None] * AB), axis=1)
        within = valid & (d - rr < reach)                             # the surface within reach
        # compact the hits into the first k slots by a prefix sum (order is immaterial to the interaction; a
        # top-k sort of thousands of candidates per walker per round was the round's cost)
        slot = jnp.cumsum(within) - 1
        put = within & (slot < k)
        out_c = jnp.zeros(k, cand.dtype).at[jnp.where(put, slot, k)].set(cand, mode="drop")
        out_v = jnp.zeros(k, bool).at[jnp.where(put, slot, k)].set(True, mode="drop")
        return out_c, out_v, within.sum()

    def _reflect_with(self, r, step, cand, valid):
        """The wall interaction against the given candidate segments (``cand`` indices, ``valid`` mask): the one
        implementation behind :meth:`_reflect` and the cached-candidate round of an adaptive walk."""
        NUDGE = jnp.float32(self.nudge_m)
        A = self._A[cand]; AB = self._AB[cand]; AB2 = self._AB2[cand]; rr = self._rout[cand]
        if self.interior:
            # each walker keeps to its own tube; the strands are separate axons even where the data lets them touch
            return _reflect_interior(r, step, A, AB, AB2, rr, valid, self._seg_tube[cand], NUDGE)
        # the exterior pool: specular off the first tube the step enters
        return _reflect_exterior(r, step, A, AB, AB2, rr, valid, NUDGE)

    def sample_inside(self, n, rng):
        """``n`` points uniform by volume inside the tubes (segment volume, then the disc, then the length), the
        whole strand set: no box."""
        A = np.asarray(self._A); AB = np.asarray(self._AB); AB2 = np.asarray(self._AB2); rout = np.asarray(self._rout)
        L = np.sqrt(AB2)
        w = (rout ** 2) * L; w = w / w.sum()
        idx = rng.choice(len(A), size=int(n), p=w)
        C = A[idx] + rng.uniform(0.0, 1.0, int(n))[:, None] * AB[idx]
        T = AB[idx] / L[idx][:, None]
        ref = np.tile(np.array([0., 0., 1.]), (int(n), 1))
        ref[np.abs((T * ref).sum(1)) > 0.9] = np.array([1., 0., 0.])
        e1 = np.cross(T, ref); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(T, e1)
        rad = (rout[idx] - self.nudge_m) * np.sqrt(rng.uniform(0., 1., int(n)))     # inside by the nudge: a seed on the wall reads as outside
        th = rng.uniform(0., 2 * np.pi, int(n))
        return C + rad[:, None] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)

    def init_positions(self, n_walkers, key):
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2 ** 30)))
        if self.interior:
            pts = self.sample_inside(n_walkers, rng)
            if self.box is not None:                       # strands overrun the voxel: seed only inside it
                keep = ((pts >= self.box[0]) & (pts <= self.box[1])).all(1)
                pts = pts[keep]
                while len(pts) < n_walkers:
                    more = self.sample_inside(n_walkers, rng)
                    more = more[((more >= self.box[0]) & (more <= self.box[1])).all(1)]
                    pts = np.concatenate([pts, more])[:n_walkers]
            return jnp.asarray(pts, jnp.float32)
        # extra: rejection outside all tubes, grid-accelerated (see sample_outside/inside_any)
        return jnp.asarray(self.sample_outside(n_walkers, rng, bounds=self.box), jnp.float32)
