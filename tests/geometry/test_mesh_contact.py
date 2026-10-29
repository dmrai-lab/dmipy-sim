"""A mesh wall's contact is the push its reflection gives the walker: one impact on a facet of a coarse closed prism,
fired from inside at a known point of the facet, and its boundary local time (``-dlog_w / rho_over_D``) against the
displacement the wall gave the walker along the facet's inward normal (its end against the free path's end). The
octagonal prism's vertex-interpolated normal is tilted from its facets by up to 22.5 degrees, so a contact read off it
would fall short of the push by that cosine; read off the facet normal it is the push to float32 rounding, and the
walk's mean local time is then the prism's own ``D t S/V``."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_sim.geometry.mesh import Mesh

UM = 1e-6


def _prism(R=3.0 * UM, L=12.0 * UM, n_ang=8, n_ax=6):
    """A closed n-gon prism along z (vertices on the circle of radius R), the two caps as fans."""
    th = np.linspace(0.0, 2 * np.pi, n_ang, endpoint=False)
    zs = np.linspace(-L / 2, L / 2, n_ax + 1)
    V = [(R * np.cos(t), R * np.sin(t), z) for z in zs for t in th]
    F = []
    for iz in range(n_ax):
        for it in range(n_ang):
            a, b = iz * n_ang + it, iz * n_ang + (it + 1) % n_ang
            F += [(a, b, b + n_ang), (a, b + n_ang, a + n_ang)]
    V += [(0.0, 0.0, -L / 2), (0.0, 0.0, L / 2)]
    off = n_ax * n_ang
    for it in range(n_ang):
        F += [(len(V) - 2, (it + 1) % n_ang, it), (len(V) - 1, off + it, off + (it + 1) % n_ang)]
    V, F = np.asarray(V), np.asarray(F, np.int64)
    return Mesh(V, F, periodic=False, voxel_min=[-R - UM, -R - UM, -L / 2 - UM], voxel_max=[R + UM, R + UM, L / 2 + UM],
                feature_radius=R), V, F


def test_a_facet_reflection_counts_the_push_it_gives():
    mesh, V, F = _prism()
    lateral = np.where(np.abs(np.ptp(V[F][:, :, 2], axis=1)) > 0)[0]           # the tube's facets, not the caps'
    rng = np.random.default_rng(3)
    rows = []
    for t in rng.choice(lateral, 24, replace=False):
        A, B, C = V[F[t]]
        n_out = np.cross(B - A, C - A); n_out /= np.linalg.norm(n_out)
        if np.dot(n_out, A) < 0:
            n_out = -n_out
        for (u, v) in ((0.05, 0.05), (0.45, 0.45), (0.1, 0.8), (0.3, 0.3)):  # near a vertex, an edge, inside
            target = A + u * (B - A) + v * (C - A)
            for incidence in (0.0, 30.0, 60.0):
                tang = np.cross(n_out, [0.0, 0.0, 1.0]); tang /= np.linalg.norm(tang)
                a = np.deg2rad(incidence)
                dh = np.cos(a) * n_out + np.sin(a) * (np.cos(1.0) * tang + np.sin(1.0) * np.array([0.0, 0.0, 1.0]))
                dh /= np.linalg.norm(dh)
                start = target - 0.05 * UM * dh                                   # inside, 50 nm from the facet
                rows.append((start, 0.1 * UM * dh, n_out))
    start = np.stack([r[0] for r in rows]); step = np.stack([r[1] for r in rows]); n_out = np.stack([r[2] for r in rows])
    fire = jax.jit(jax.vmap(lambda p, s: mesh.interact(p, s, rho_over_D=1.0)))
    hit = fire(jnp.asarray(start, jnp.float32), jnp.asarray(step, jnp.float32))
    r = np.asarray(hit.r, np.float64)
    contact = -np.asarray(hit.dlog_w, np.float64)
    pushed = ((start + step - r) * n_out).sum(1)                               # the end's displacement inward
    assert (pushed > 0).all(), "every row reflects off its facet"
    rel = np.abs(contact - pushed) / pushed
    assert rel.max() < 1e-4, f"contact against the push: worst {rel.max():.2e} (median {np.median(rel):.2e})"
