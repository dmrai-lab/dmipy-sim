"""Monte-Carlo accuracy of the mesh geometry vs the analytic geometries.

A triangulated shape must reproduce the analytic signal of the same shape to the MC noise floor. The mesh of a sphere is
a polyhedron inscribed in it, so the analytic reference is the sphere of the mesh's own volume (its surface is within
0.1 % of the mesh's at subdivision 3, where the sphere of the mesh's circumscribing radius would be 0.9 % off in
volume): the residual is then the surface discretisation, which is far below the noise.

The mesh walk is the expensive one in the suite: 100 sub-steps per walker at subdivision 3 (the engine takes one
sub-step per waveform step there) and about 2e3 walker-sub-steps per second on four CPU cores, so the walk is tier 2
and its size is the smallest that resolves a 0.13 signal difference.
"""
import numpy as np
import numpy.testing as npt
import jax
import jax.numpy as jnp
import pytest

from dmipy_sim import simulate, Sphere, Cylinder, Mesh, set_b, pgse, pgste
from dmipy_replay.acquisition.scanner_sequence import ScannerSequence
from dmipy_sim.engine._fused import simulate_fused
from dmipy_sim import Compartments, Pool
from tests.conftest import SD_COS, z_family

trimesh = pytest.importorskip("trimesh")

D = 2e-9
R = 5e-6
SEED = 123


def _pgse(nb, n_t, TE=40e-3, axis=0):
    dt = TE / (n_t - 1)
    G = np.zeros((1, n_t, 3), np.float32)
    G[0, 1:int(0.4 * n_t), axis] = 1.0
    G[0, -int(0.4 * n_t):-1, axis] = -1.0
    Gt = jnp.tile(jnp.array(G), (nb, 1, 1))
    return set_b(ScannerSequence(G=Gt, dt=dt), np.linspace(1, 2e9, nb))


def _ico(sub, r=R):
    m = trimesh.creation.icosphere(subdivisions=sub, radius=r)
    return np.asarray(m.vertices, float), np.asarray(m.faces, int)


def _open_tube(r, L, nt=48, nz=40):
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    zs = np.linspace(0, L, nz)
    V = np.array([[r * np.cos(t), r * np.sin(t), z] for z in zs for t in th])
    F = []
    for iz in range(nz - 1):
        for j in range(nt):
            a = iz * nt + j; b = iz * nt + (j + 1) % nt
            c = (iz + 1) * nt + j; d = (iz + 1) * nt + (j + 1) % nt
            F.append([a, b, d]); F.append([a, d, c])
    return V, np.array(F)


N_REFERENCE = 16_000      # the analytic sphere is cheap: its noise is a seventh of the mesh walk's
# The mesh walk: the difference of its signal and the sphere's at b = 0, 1 and 2 ms/um^2 is held to ``tol`` at the
# Bonferroni quantile of three comparisons; a difference of 0.13 fails with the margin the noise passes 0.064.
# The walker's cos(phase) spreads by at most SD_COS, so
#   N = (2 z SD_COS / 0.127)^2 (1 + N / N_REFERENCE) = 2,000.
N_MESH_MC = 2_000
TE_MC = 10e-3             # s: restricted by the 5 um sphere (6 um of diffusion), and a tenth of the sub-steps of 40 ms
RHO_MC = 8e-5             # m/s: Brownstein-Tarr 3 rho / R = 48 /s, the wall relaxes S from 1 to 0.62 in TE_MC


def _equal_volume_radius(V, F):
    return (3.0 * abs(trimesh.Trimesh(V, F, process=False).volume) / (4.0 * np.pi)) ** (1.0 / 3.0)


@pytest.mark.gpu
def test_diffusion_and_surface_relaxivity_match_analytic_sphere():
    """The mesh sphere's signal over a b-ladder, with relaxing walls, is the analytic sphere's of the same volume.

    Claim: the signals agree to ``tol`` = 0.064 at b = 0 (where only the relaxation acts: the wall relaxes S from 1 to
    0.62, so a mesh without relaxivity is 0.38 off), 1 and 2 ms/um^2 (restricted diffusion). Statistic: the difference
    of the mesh walk's walker mean (N = 2,000) and the sphere's (16,000), ``z SD_COS sqrt(1 / N + 1 / N_ref)``
    at z = 3.59 (three comparisons)."""
    V, F = _ico(3)
    R_eq = _equal_volume_radius(V, F)
    wf = _pgse(3, 100, TE=TE_MC)
    pools = Compartments(intra=Pool(surface_relaxivity_t2=RHO_MC), extra=Pool(surface_relaxivity_t2=RHO_MC))
    s_mesh = np.asarray(simulate_fused(N_MESH_MC, D, wf, Mesh(V, F, compartments=pools), seed=SEED, require_gpu=False))
    s_ana = np.asarray(simulate_fused(N_REFERENCE, D, wf, Sphere(radius=R_eq, surface_relaxivity_t2=RHO_MC), seed=SEED, require_gpu=False))
    tol = z_family(3) * SD_COS * np.sqrt(1.0 / N_MESH_MC + 1.0 / N_REFERENCE)
    assert np.abs(s_mesh - s_ana).max() < tol, f"mesh {np.round(s_mesh, 4)} vs analytic Sphere {np.round(s_ana, 4)} (tolerance {tol:.4f})"


def test_directional_permeability():
    """Symmetric κ (dict) reproduces the scalar; a rectifying (direction-dependent) κ drives net flux -- outflow-only
    empties cells, inflow-only retains them.

    The membrane's verdict is deterministic where the transmit probability is one, so the claim is held on the wall
    interaction itself: from points inside and outside a mesh sphere, radial steps across the wall with
    ``2 kappa/D d_perp >= 1`` end on the far side when that direction's kappa is the nominal and on the near side
    (reflected) when it is zero, for the 320 directions through the facets' centres; the symmetric dict is the scalar to
    the bit at a probability below one."""
    V, F = _ico(2)
    kod = 1e9                                        # 2 kappa/D d_perp >= 1 for every step of the table (d_perp >= 1 um)
    centroids = np.asarray(V, float)[np.asarray(F)].mean(axis=1)                  # through the middle of every facet, never an edge
    n = centroids / np.linalg.norm(centroids, axis=1)[:, None]
    inside, outside = 0.8 * R * n, 1.2 * R * n                                       # 0.4 R radial steps cross the wall
    starts = np.concatenate([inside, outside]).astype(np.float32)
    steps = np.concatenate([outside - inside, inside - outside]).astype(np.float32)

    def ends(perm, kod=kod):
        mesh = Mesh(V, F, permeability=perm)
        r = jax.vmap(lambda p, q, k: mesh.interact(p, q, kappa_over_D=kod, key=k).r)(
            jnp.asarray(starts), jnp.asarray(steps), jax.random.split(jax.random.PRNGKey(1), len(starts)))
        return np.asarray(r), np.asarray(jax.vmap(mesh.classify_position)(r))

    half = len(n)
    cases = {"outflow only": ({"intra_to_extra": 1.0, "extra_to_intra": 0.0}, 0, 0),
             "inflow only": ({"intra_to_extra": 0.0, "extra_to_intra": 1.0}, 1, 1),
             "both ways": ({"intra_to_extra": 1.0, "extra_to_intra": 1.0}, 0, 1)}
    for name, (perm, from_inside, from_outside) in cases.items():
        _, lab = ends(perm)
        assert (lab[:half] == from_inside).all() and (lab[half:] == from_outside).all(), (
            f"{name}: {(lab[:half] != from_inside).sum()} steps from inside and {(lab[half:] != from_outside).sum()} from "
            f"outside ended on the wrong side of the wall")
    _, lab = ends(None, kod=0.0)                     # an impermeable mesh is walked with kappa/D = 0
    assert (lab[:half] == 1).all() and (lab[half:] == 0).all(), "a closed membrane let a step through"

    mild = 2e-5 / D                                  # a probability below one: 2 kappa/D d_perp for d_perp ~ 0.02 um
    near = np.concatenate([0.98 * R * n, 1.02 * R * n]).astype(np.float32)
    short = (np.concatenate([n, -n]) * 0.05 * R).astype(np.float32)
    keys = jax.random.split(jax.random.PRNGKey(2), len(near))

    def mild_ends(perm):
        mesh = Mesh(V, F, permeability=perm)
        return np.asarray(jax.vmap(lambda p, q, k: mesh.interact(p, q, kappa_over_D=mild, key=k).r)(jnp.asarray(near), jnp.asarray(short), keys))
    npt.assert_array_equal(mild_ends(2e-5), mild_ends({"intra_to_extra": 2e-5, "extra_to_intra": 2e-5}))
