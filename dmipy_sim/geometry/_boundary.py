"""Shared boundary primitives: one implementation of each rule, for every geometry.

Why this module exists
----------------------
The compartment sentinel (dmrai-lab/dmipy-sim#86) was written seven times, once per
geometry, and the copies drifted into three different spellings of the same rule:

    Sphere/Cylinder/Ellipsoid/PackedSpheres   jnp.where(inside, d >= R, d <= R)
    PackedCylinders                           geo_in != want_in
    PermeableSlab1D/PermeableShell            (x < wall) != side0

Those are not equivalent. The `!=` forms leave the tie unresolved: a walker landing
EXACTLY on the surface is on neither side, and in float32 that tie is real -- |q| - R
evaluates to exactly 0.0 for such endpoints, and two equally-correct-looking spellings
of "is it inside" then disagree depending on how XLA fuses them. That cost a debugging
cycle on PackedCylinders and left the slab and the shell with the defective form still
in place after the others were fixed.

A rule that must hold identically for every geometry belongs in one function. The
geometry supplies its own surface parametrisation; the DECISION lives here.

The invariant, stated once
--------------------------
A walker may change compartment only where a crossing was granted. Everywhere else its
side is fixed, and equality counts as the wrong side for BOTH compartments -- an interior
walker must be strictly inside, an exterior walker strictly outside. That is what makes
the rule total: there is no coordinate for which it declines to decide.
"""
import jax.numpy as jnp

# Plain Python floats, NOT jnp scalars. A module-level jnp value is a DEVICE buffer created
# at import, and `gpu.py` calls `jax.clear_caches()` on session teardown -- after which every
# use raises "Array has been deleted with shape=float32[]". As weak-typed Python floats these
# promote against float32 operands without changing the arithmetic.
_TINY = 1e-30

# Half-width of the "too close to call" band around a surface, RELATIVE to the radius.
# float32 has ~1.2e-7 relative precision, so a point within a few ulps of |q| = R is
# classified differently by mathematically identical spellings: `norm(q) < R` takes a
# sqrt, `dot(q,q) < R*R` does not, and near the wall they disagree. The geometries do not
# even agree among themselves -- Sphere and the packed classes compare squares, while
# PermeableShell compares the norm -- so there is no single spelling to match. Treat
# anything inside this band as ON the wall and move it off. 1e-6 is ~8x float32 eps and
# ~100x tighter than a typical NUDGE (1e-4 R), so a corrected point is unambiguous under
# every spelling while the band itself is far too thin to perturb the physics.
_SURF_EPS = 1e-6


def keep_side_radial(pos, q, R, want_inside, nudge, active=True):
    """Force ``pos`` strictly onto ``want_inside``'s side of the surface |q| = R.

    Covers every geometry whose boundary is a level set of a distance: Sphere and
    Cylinder (object at the frame origin, so ``q`` is the position itself, 3-D or the
    2-D in-plane part), PackedCylinders and PackedSpheres (``q`` the minimum-image
    offset to the relevant object), and PermeableShell (``q`` the radial component, so
    the axial coordinate is untouched).

    Parameters
    ----------
    pos : the proposed end-of-step position, in whatever frame ``q`` is measured.
    q : offset from the surface's centre to ``pos``; ``|q| = R`` is the wall.
    R : the wall radius.
    want_inside : bool, the side the walker must be on.
    nudge : how far off the wall to place a corrected walker.

    Returns
    -------
    (pos_corrected, wrong) -- ``wrong`` is the sentinel-fired flag, worth counting.
    """
    d2 = jnp.sum(q * q, axis=-1)
    R2 = R * R
    # decide on SQUARED distances against a tie-band (see _SURF_EPS): a point closer to the
    # wall than float32 can resolve belongs to neither side and is moved off it
    wrong = jnp.where(want_inside, d2 >= R2 * (1.0 - _SURF_EPS),
                                   d2 <= R2 * (1.0 + _SURF_EPS))
    wrong = jnp.logical_and(active, wrong)
    d = jnp.sqrt(jnp.maximum(d2, jnp.float32(0.0)))
    target = jnp.where(want_inside, R - nudge, R + nudge)
    # move ALONG q, so components orthogonal to it (e.g. a cylinder's axial coordinate)
    # are preserved exactly
    return jnp.where(wrong, pos + q * (target / jnp.maximum(d, _TINY) - jnp.float32(1.0)),
                     pos), wrong


def keep_side_planar(x, x_wall, want_below, nudge, active=True):
    """Force scalar coordinate ``x`` strictly onto ``want_below``'s side of ``x_wall``.

    For PermeableSlab1D, whose membrane is a plane rather than a curved surface. The
    crossing test there is a STRICT sign change, so a step landing exactly on the
    membrane fires no reflection while the classifier reads the far compartment.
    """
    wrong = jnp.logical_and(active, jnp.where(want_below, x >= x_wall, x <= x_wall))
    return jnp.where(wrong, jnp.where(want_below, x_wall - nudge, x_wall + nudge), x), wrong


def keep_side_quadric(pos, Q, want_inside, rel_nudge, active=True):
    """Force ``pos`` strictly onto ``want_inside``'s side of the quadric Q = 1.

    For Ellipsoid, where the surface is ``r.D.r = 1`` rather than a constant distance.
    Scaling by sqrt(target/Q) moves along the ray from the centre, which is the
    ellipsoid's own radial direction, so the corrected point stays on the same ray.
    ``rel_nudge`` is the offset as a FRACTION of the semiaxis, since "distance off the
    wall" is direction-dependent here.
    """
    wrong = jnp.where(want_inside, Q >= jnp.float32(1.0) - _SURF_EPS,
                                   Q <= jnp.float32(1.0) + _SURF_EPS)
    target = jnp.where(want_inside, (jnp.float32(1.0) - rel_nudge) ** 2,
                                    (jnp.float32(1.0) + rel_nudge) ** 2)
    wrong = jnp.logical_and(active, wrong)
    return jnp.where(wrong, pos * jnp.sqrt(target / jnp.maximum(Q, _TINY)), pos), wrong


# ─────────────────────────────────────────────────────────────────────────────
# Ray/surface intersection and reflection.
#
# Before this, `geometries.py` carried 12 hand-written hit computations, 16 copies of the
# specular formula and 5 of the transmit probability. Four classes (Sphere, Cylinder,
# PackedCylinders, PackedSpheres) computed the SAME discriminant, differing only in 2-D vs
# 3-D and scalar vs batched -- distinctions jnp handles for free. Ellipsoid and
# PermeableShell used the general quadric, of which the other four are the A = 1 case.
#
# One implementation each means a fix lands once instead of nine times, which is the whole
# point: the compartment sentinel had to be fixed in seven places and two were missed.
# ─────────────────────────────────────────────────────────────────────────────


def ray_sphere_t(q, d, R):
    """Entry/exit parameters of the ray ``q + t d`` against ``|x| = R``, for UNIT ``d``.

    Works scalar or batched: ``q`` and ``d`` are reduced over their last axis, so a
    single 3-vector, a 2-D in-plane vector, or an (N, k) stack of offsets to N objects
    all go through unchanged.

    Returns ``(t_entry, t_exit, disc)``. ``disc < 0`` means the ray misses; the roots are
    computed from the clipped discriminant so they stay finite either way, and callers
    decide what a miss means using ``disc``.
    """
    dp = jnp.sum(d * q, axis=-1)
    disc = dp * dp - (jnp.sum(q * q, axis=-1) - R * R)
    sq = jnp.sqrt(jnp.maximum(disc, jnp.float32(0.0)))
    return -dp - sq, -dp + sq, disc


def ray_quadric_t(A, B, C):
    """Entry/exit parameters for ``A t^2 + 2 B t + C = 0`` (the general quadric).

    ``ray_sphere_t`` is the ``A = 1`` case. Used where the surface is an ellipsoid, or a
    cylinder whose ray direction is not normalised in the radial subspace.
    """
    disc = B * B - A * C
    sq = jnp.sqrt(jnp.maximum(disc, jnp.float32(0.0)))
    A_safe = jnp.maximum(A, _TINY)
    return (-B - sq) / A_safe, (-B + sq) / A_safe, disc


def specular(d, n):
    """Mirror direction ``d`` about the surface whose unit normal is ``n``."""
    return d - jnp.float32(2.0) * jnp.dot(d, n) * n


def transmit_probability(kappa_over_D, d_perp):
    """Powles membrane transmission probability, ``min(1, 2 kappa/D * d_perp)``."""
    return jnp.minimum(jnp.float32(1.0),
                       jnp.float32(2.0) * kappa_over_D * d_perp)


def off_wall(r_hit, n_out, stay_inside, nudge):
    """Move a point sitting exactly ON a surface to ``nudge`` clear of it, correct side.

    ``r_hit`` is snapped to the surface, so it is precisely the coordinate the strict
    containment tests cannot classify. Stepping off along the outward normal is what
    makes the reflected walker unambiguously its own compartment's.

    Frame-agnostic: pass a lab position with its outward normal, or a local offset
    ``q_hit`` relative to an object's centre (the packed geometries do the latter, which
    is what guarantees ``|q| = R +/- nudge`` exactly).
    """
    return r_hit + jnp.where(stay_inside, -nudge, nudge) * n_out


def step_off_wall(r_hit, n_out, stay_inside, d_refl, remaining, nudge):
    """``off_wall`` then travel the remaining path along ``d_refl``.

    The remaining distance is reduced by the nudge, so total path length is preserved.
    """
    return (off_wall(r_hit, n_out, stay_inside, nudge)
            + d_refl * jnp.maximum(remaining - nudge, jnp.float32(0.0)))


def bind_probability(kappa_over_D, local_time):
    """Probability a spin binds at a wall, from boundary LOCAL TIME: ``min(1, kappa/D * l)``.

    Distinct from :func:`transmit_probability`, which is driven by the penetration depth of
    a single hit and carries the Powles factor of 2. This is the local-time formulation used
    by the vector-Bloch and MT walkers, where the wall contact is accumulated rather than
    resolved hit-by-hit. Three identical copies of it existed (physics, bloch, mt_walk); the
    two rules are easy to conflate, which is reason to name both rather than neither.
    """
    return jnp.minimum(jnp.float32(1.0), kappa_over_D * local_time)
