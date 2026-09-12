"""The strand field basis (the per-segment superposition) against the rasterised k-space route on a curved
sheathed strand, its domain mean, and the cutoff certificate on a packed set of strands."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.fields.strand_field import StrandFieldBasis
from dmipy_sim.fields.hollow_cylinder import CHANNEL_NAMES
from dmipy_sim.fields.susceptibility_field import predicate_field_basis


def _kinked_strand(n=7, step=3e-6, turn_deg=20.0, seed=0):
    """A polyline with `turn_deg` joints (DiSCo's median joint is 21 degrees), through the origin."""
    rng = np.random.default_rng(seed)
    pts = [np.array([0.0, 0.0, -0.5 * (n - 1) * step])]; u = np.array([0.0, 0.0, 1.0])
    for _ in range(n - 1):
        ax = np.cross(u, rng.normal(size=3)); ax /= np.linalg.norm(ax); t = np.deg2rad(turn_deg)
        u = np.cos(t) * u + np.sin(t) * ax; u /= np.linalg.norm(u)
        pts.append(pts[-1] + step * u)
    return np.array(pts)


def test_a_curved_strand_matches_the_rasterised_route():
    """One sheathed strand with 20-degree joints: the superposition's 13 channels against the k-space route on a
    box, inside the lumen, the sheath and outside (tolerances from the DiSCo strand with a 54-degree joint: 6 % of
    the component's maximum at the joint, 2 % near, 0.3 % far)."""
    cl = _kinked_strand(); a, b = 0.8e-6, 1.4e-6
    lo, hi = np.full(3, -7e-6), np.full(3, 7e-6); res = 0.15e-6
    inner = d.PackedCurvedCylinders([cl], [a], interior=True); outer = d.PackedCurvedCylinders([cl], [b], interior=True)
    basis, origin, _ = predicate_field_basis(inner.inside_any, outer.inside_any, lo, hi, res=res, mask_supersample=2,
                                             director=inner.radial_directors)
    shape = tuple(basis["shape"]); vs = np.asarray(basis["voxel_size"]); org = np.asarray(origin)
    ax = [org[k] + (np.arange(shape[k]) + 0.5) * vs[k] for k in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij"); P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    K = np.concatenate([basis["iso_local"].reshape(-1, 1), basis["iso_P"].reshape(6, -1).T, basis["aniso_G"].reshape(6, -1).T], 1)
    sel = np.all(np.abs(P) < 3.5e-6, axis=1)                                  # away from the box's images
    sf = StrandFieldBasis([cl], [a], [b], cutoff_m=20e-6, domain=(lo, hi))
    C = sf.channels(P[sel]); Kc = K[sel]
    # the true distance to the swept polyline, for the regions
    A_ = cl[:-1]; AB = cl[1:] - cl[:-1]; AB2 = (AB ** 2).sum(1)
    t = np.clip(((P[sel][:, None, :] - A_[None]) * AB[None]).sum(-1) / AB2[None], 0, 1)
    dist = np.linalg.norm(P[sel][:, None, :] - (A_[None] + t[..., None] * AB[None]), axis=-1).min(1)
    regions = {"lumen": dist < a - 0.15e-6, "sheath": (dist > a + 0.15e-6) & (dist < b - 0.15e-6),
               "near": (dist > b + 0.15e-6) & (dist < 3 * b), "far": dist > 3 * b}
    tol = {"lumen": 0.08, "sheath": 0.08, "near": 0.04, "far": 0.03}      # far: the 14 um box's images (b / 10 um)^2
    group_scale = {0: np.abs(Kc[:, 0]).max(), 1: np.abs(Kc[:, 1:7]).max(), 2: np.abs(Kc[:, 7:13]).max()}
    for j, name in enumerate(CHANNEL_NAMES):
        scale = group_scale[0 if j == 0 else (1 if j < 7 else 2)]             # relative to the group's field scale
        Cj = C[:, j] - C[:, j].mean() + Kc[:, j].mean()                       # the grid's mean is over its whole box
        for r, s in regions.items():
            assert s.sum() > 50, r
            err = np.sqrt(np.mean((Cj[s] - Kc[s, j]) ** 2)) / scale
            assert err < tol[r], (name, r, err)


def test_the_domain_mean_is_the_closed_form():
    """The channels average to zero over the domain (the closed-form mean equals the sampled one)."""
    rng = np.random.default_rng(1)
    cls, ra, rb = [], [], []
    for k in range(40):                                                       # straight strands, random directions
        u = rng.normal(size=3); u /= np.linalg.norm(u); c0 = rng.uniform(-20e-6, 20e-6, 3)
        cls.append(np.stack([c0 - 60e-6 * u, c0 + 60e-6 * u])); ra.append(1e-6); rb.append(1.4e-6)
    lo, hi = np.full(3, -20e-6), np.full(3, 20e-6)
    sf = StrandFieldBasis(cls, ra, rb, cutoff_m=60e-6, domain=(lo, hi))
    P = rng.uniform(lo, hi, (60000, 3))
    sampled = sf.channels(P).mean(0)                                           # mean subtracted: should be ~0
    scale = np.abs(sf.mean).max()
    assert scale > 1e-4 and np.abs(sampled).max() < 0.05 * scale, (sampled / scale)


def test_the_cutoff_error_falls_with_the_cutoff_and_the_field_contracts():
    rng = np.random.default_rng(2)
    cls, ra, rb = [], [], []
    for k in range(120):
        c0 = np.array([rng.uniform(-40e-6, 40e-6), rng.uniform(-40e-6, 40e-6), 0.0])
        cls.append(np.stack([c0 + [0, 0, -50e-6], c0 + [0, 0, 50e-6]])); ra.append(1e-6); rb.append(1.4e-6)
    sf = StrandFieldBasis(cls, ra, rb, cutoff_m=5e-6)
    P = rng.uniform(-10e-6, 10e-6, (2000, 3))
    e5 = sf.cutoff_error(P); e20 = sf.with_cutoff(20e-6).cutoff_error(P)
    assert e5["iso"] > e20["iso"] and e5["aniso"] > e20["aniso"] and e20["iso"] < 0.1, (e5, e20)
    f = sf.field(P, [1.0, 0.0, 0.0], B0=3.0, chi_iso=-0.05e-6, chi_aniso=-0.1e-6)
    assert f.shape == (2000,) and np.isfinite(f).all() and np.abs(f).max() > 1e-9
    # B0 along every fibre: no field outside the sheaths (a uniform value: the subtracted domain mean)
    centres = np.array([c[0, :2] for c in cls]); dist = np.linalg.norm(P[:, None, :2] - centres[None], axis=-1).min(1)
    f_par = sf.field(P, [0.0, 0.0, 1.0], B0=3.0, chi_aniso=-0.1e-6)
    assert (dist > 1.4e-6).sum() > 500 and np.abs(f_par[dist > 1.4e-6] - f_par[dist > 1.4e-6].mean()).max() < 1e-15
    assert sf.meta["kind"] == "strand_superposition" and sf.meta["n_strands"] == 120
