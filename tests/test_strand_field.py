"""The strand field basis (the finite-line superposition over every segment) against the rasterised k-space route
on a curved sheathed strand, its domain mean, and the cutoff certificate on a packed set of strands."""
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
    """One sheathed strand with 20-degree joints (DiSCo's median): the superposition's 13 channels against the
    k-space route on a box, inside the lumen, the sheath and outside (measured 4.8 / 3.5 / 1.6 / 0.9 % of the
    component's maximum; a 54-degree joint, DiSCo's sharpest, measures 8.9 / 8.4 / 3.6 / 1.0 %)."""
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
    """The channels average to zero over the domain (the closed-form mean equals the sampled one). The closed form
    counts every strand's length, so where two sheaths overlap it counts the local terms (iso_local, P / 2 of M_P)
    twice while the field carries the nearest strand's alone: that residual is the overlap's local terms, measured
    by the same membership rule."""
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
    d_all = np.full(len(P), np.inf); twice = np.zeros((len(P), 7)); nearest = np.zeros((len(P), 7))
    for c, a, b in zip(cls, ra, rb):
        A = c[0]; AB = c[1] - c[0]; u = AB / np.linalg.norm(AB); t = np.clip(((P - A) @ AB) / (AB @ AB), 0, 1)
        d = np.linalg.norm(P - (A + t[:, None] * AB), axis=1); m = ((d > a) & (d < b)).astype(float)
        P2 = 0.5 * np.array([1 - u[0] ** 2, 1 - u[1] ** 2, 1 - u[2] ** 2, -u[0] * u[1], -u[0] * u[2], -u[1] * u[2]])
        loc = m[:, None] * np.concatenate([[1 / 3], P2])[None]
        twice += loc; new = d < d_all; nearest = np.where(new[:, None], loc, nearest); d_all = np.minimum(d_all, d)
    overlap = (twice - nearest).mean(0)                                        # the local terms the closed form counts twice
    assert scale > 1e-4 and np.abs(sampled[:7] + overlap).max() < 0.08 * scale, ((sampled[:7] + overlap) / scale)
    assert np.abs(sampled[7:]).max() < 0.08 * scale, (sampled / scale)        # 60k samples of a field whose outside terms average out by cos 2a


def test_the_cutoff_error_falls_with_the_cutoff_and_the_field_contracts():
    rng = np.random.default_rng(2)
    cls, ra, rb = [], [], []
    for k in range(120):
        c0 = np.array([rng.uniform(-40e-6, 40e-6), rng.uniform(-40e-6, 40e-6), 0.0])
        cls.append(np.stack([c0 + [0, 0, -1e-3], c0 + [0, 0, 1e-3]])); ra.append(1e-6); rb.append(1.4e-6)   # a millimetre each way: the ends' dipoles at 1e-9 of the field
    sf = StrandFieldBasis(cls, ra, rb, cutoff_m=5e-6)
    P = rng.uniform(-10e-6, 10e-6, (2000, 3))
    e5 = sf.cutoff_error(P); e20 = sf.with_cutoff(20e-6).cutoff_error(P)
    assert e5["iso"] > e20["iso"] and e5["aniso"] > e20["aniso"] and e20["iso"] < 0.1, (e5, e20)
    f = sf.field(P, [1.0, 0.0, 0.0], B0=3.0, chi_iso=-0.05e-6, chi_aniso=-0.1e-6)
    assert f.shape == (2000,) and np.isfinite(f).all() and np.abs(f).max() > 1e-9
    # B0 along every fibre: no field outside the sheaths (a uniform value: the subtracted domain mean)
    centres = np.array([c[0, :2] for c in cls]); dist = np.linalg.norm(P[:, None, :2] - centres[None], axis=-1).min(1)
    f_par = sf.field(P, [0.0, 0.0, 1.0], B0=3.0, chi_aniso=-0.1e-6)
    assert (dist > 1.4e-6).sum() > 500 and np.abs(f_par[dist > 1.4e-6] - f_par[dist > 1.4e-6].mean()).max() < 1e-12
    assert sf.meta["kind"] == "strand_superposition" and sf.meta["n_strands"] == 120
    with pytest.raises(ValueError, match="segments_max"):                    # more segments within reach than allowed: refused, not dropped
        StrandFieldBasis(cls, ra, rb, cutoff_m=20e-6, segments_max=4).channels(P[:50])


def _segment_field(P, A_, B_, a, b):
    """The exact field of one segment at points ``P`` (float64): the dipole kernel integrated along it, times the
    annulus area; ``M_P = A L``, ``M_A = -A sym(L T)``; rho continued at the sheath's surface inside it."""
    AB = B_ - A_; L = np.linalg.norm(AB); u = AB / L
    zA = (P - A_) @ u; rv = (P - A_) - zA[:, None] * u; rho = np.maximum(np.linalg.norm(rv, axis=1), 1e-30)
    rc = np.maximum(rho, b * (1 + 1e-6)); rv = rv * (rc / rho)[:, None]; r2 = rc ** 2
    def prim(z):
        rr = np.sqrt(r2 + z * z); r3 = rr ** 3
        return np.stack([z / (r2 * rr), z * (2 * z * z + 3 * r2) / (3 * r2 * r2 * r3), -1 / (3 * r3), z ** 3 / (3 * r2 * r3)])
    i0, j0, j1, j2 = prim(L - zA) - prim(-zA)
    E = np.eye(3); rr_ = rv[:, :, None] * rv[:, None, :]; ru = rv[:, :, None] * u[None, None, :] + u[None, :, None] * rv[:, None, :]; UU = np.outer(u, u)
    Lt = (E * i0[:, None, None] - 3 * (rr_ * j0[:, None, None] - ru * j1[:, None, None] + UU * j2[:, None, None])) / (4 * np.pi)
    A = np.pi * (b ** 2 - a ** 2); T = 0.5 * (E - UU) - E / 3; LT = Lt @ T
    sym6 = lambda M: np.stack([M[:, 0, 0], M[:, 1, 1], M[:, 2, 2], M[:, 0, 1], M[:, 0, 2], M[:, 1, 2]], 1)
    return np.concatenate([np.zeros((len(P), 1)), sym6(A * Lt), sym6(-0.5 * A * (LT + np.swapaxes(LT, 1, 2)))], 1)


def _finite_line_sum(cls, ra, rb, P, cutoff):
    """The reference: every segment within ``cutoff`` (capsule distance) of a point, its exact segment field, summed
    (float64); within the nearest strand's gate, that strand's terms are its nearest segment's whole infinite
    cylinder (the lowest index at a tie)."""
    from dmipy_sim.fields.hollow_cylinder import hollow_cylinder_basis
    out = np.zeros((len(P), 13)); d_best = np.full(len(P), np.inf); corr = np.zeros((len(P), 13))
    for c, a, b in zip(cls, ra, rb):
        A_ = c[:-1]; AB = c[1:] - c[:-1]; L = np.linalg.norm(AB, axis=1); U = AB / L[:, None]
        zA = ((P[:, None, :] - A_[None]) * U[None]).sum(-1)
        t = np.clip(zA / L[None], 0, 1); d = np.linalg.norm(P[:, None, :] - (A_[None] + t[..., None] * AB[None]), axis=-1)
        K = np.stack([_segment_field(P, A_[k], A_[k] + AB[k], a, b) for k in range(len(A_))], 1)          # (n, s, 13)
        own = (K * (d < cutoff)[..., None]).sum(1); out += own
        d1 = d.min(1); j = (d <= d1[:, None] + 1e-6 * cutoff).argmax(1); new = d1 < d_best - 1e-6 * cutoff
        x = np.clip((d1 - b) / (StrandFieldBasis.NEAREST_GATE_RADII * b), 0, 1); g = x * x * (3 - 2 * x)
        rv = (P - A_[j]) - zA[np.arange(len(P)), j][:, None] * U[j]
        Cj = np.asarray(hollow_cylinder_basis(rv, U[j], np.full(len(P), a), np.full(len(P), b)), np.float64)
        this = (1 - g)[:, None] * (Cj - own) * (d1 < cutoff)[:, None]
        corr = np.where(new[:, None], this, corr); d_best = np.where(new, d1, d_best)
    return out + corr


def test_a_segment_is_the_line_up_close_and_a_dipole_far_away():
    """`segment_basis`: a very long segment seen from beside its middle is the infinite cylinder's outside formula;
    a short one seen from far away is the point dipole of its moment (the isotropic channels: `A L (I - 3 r r^T) /
    4 pi r^3`); a line cut into pieces sums to the line."""
    from dmipy_sim.fields.hollow_cylinder import segment_basis, hollow_cylinder_basis
    a, b = 1.0e-6, 1.5e-6; u = np.array([0.3, -0.2, 1.0]); u /= np.linalg.norm(u)
    rng = np.random.default_rng(3); v = rng.normal(size=3); rv = v - (v @ u) * u; rv *= 6e-6 / np.linalg.norm(rv)
    long_ = np.asarray(segment_basis(rv, u, -1.0, 1.0, a, b)); inf_ = np.asarray(hollow_cylinder_basis(rv, u, a, b))
    np.testing.assert_allclose(long_[1:], inf_[1:], rtol=0, atol=1e-6 * np.abs(inf_[1:]).max())
    L = 2e-6; zc = 60e-6; r = rv + zc * u; rhat = r / np.linalg.norm(r); A = np.pi * (b ** 2 - a ** 2)   # (L / r)^2 = 1e-3; float32 holds a 2 um piece at 60 um
    far = np.asarray(segment_basis(rv, u, -zc - L / 2, -zc + L / 2, a, b))
    dip = A * L * (np.eye(3) - 3 * np.outer(rhat, rhat)) / (4 * np.pi * np.linalg.norm(r) ** 3)
    want = np.array([dip[0, 0], dip[1, 1], dip[2, 2], dip[0, 1], dip[0, 2], dip[1, 2]])
    np.testing.assert_allclose(far[1:7], want, rtol=5e-3, atol=2e-3 * np.abs(want).max())
    cuts = np.linspace(-40e-6, 40e-6, 9)
    whole = np.asarray(segment_basis(rv, u, cuts[0], cuts[-1], a, b))
    pieces = sum(np.asarray(segment_basis(rv, u, z1, z2, a, b)) for z1, z2 in zip(cuts[:-1], cuts[1:]))
    np.testing.assert_allclose(pieces, whole, rtol=0, atol=1e-6 * np.abs(whole).max())


def test_the_bounded_evaluation_is_the_full_superposition():
    """The grid gather, the duplicate removal and the top-k of nearest segments reproduce the brute-force sum over
    every segment within the cutoff, at cutoffs below and above the segment length (a segment gathered from several
    cells counts once, and the padding never shadows segment zero -- both were defects once); on kinked strands
    with many segments each, every segment within the cutoff counts with its own exact field."""
    rng = np.random.default_rng(2); cls, ra, rb = [], [], []
    for k in range(120):
        c0 = np.array([rng.uniform(-40e-6, 40e-6), rng.uniform(-40e-6, 40e-6), 0.0])
        cls.append(np.stack([c0 + [0, 0, -50e-6], c0 + [0, 0, 50e-6]])); ra.append(1e-6); rb.append(1.4e-6)
    P = rng.uniform(-10e-6, 10e-6, (7, 3))
    for cutoff in (20e-6, 40e-6, 90e-6):
        sf = StrandFieldBasis(cls, ra, rb, cutoff_m=cutoff)
        np.testing.assert_allclose(sf.channels(P) + sf.mean, _finite_line_sum(cls, ra, rb, P, cutoff), atol=2e-7, rtol=0)
    cls2 = [_kinked_strand(n=9, step=6e-6, turn_deg=25.0, seed=k) + rng.uniform(-15e-6, 15e-6, 3) for k in range(30)]
    ra2, rb2 = [1e-6] * 30, [1.5e-6] * 30
    P2 = np.concatenate([P, np.vstack([c[3] for c in cls2[:5]])])     # the joints themselves
    sf2 = StrandFieldBasis(cls2, ra2, rb2, cutoff_m=12e-6)
    np.testing.assert_allclose(sf2.channels(P2) + sf2.mean, _finite_line_sum(cls2, ra2, rb2, P2, 12e-6), atol=5e-7, rtol=0)   # float32 in the evaluator


def test_the_field_is_continuous_across_joints_and_free_ends():
    """Along a line through a tangle of kinked strands, some ending inside the box, the field beyond the
    nearest-segment gate changes by a bounded amount per 10 nm step (the nearest-segment rule jumped at every
    joint's bisector by up to the field's rms); and a straight strand's factors telescope: a strand cut into many
    segments is the one-segment strand."""
    rng = np.random.default_rng(4); cls, ra, rb = [], [], []
    for k in range(30):
        c = _kinked_strand(n=9, step=5e-6, turn_deg=40.0, seed=k) + rng.uniform(-12e-6, 12e-6, 3)
        cls.append(c); ra.append(1e-6); rb.append(1.5e-6)
    sf = StrandFieldBasis(cls, ra, rb, cutoff_m=30e-6)
    n = 3001; line = np.stack([np.linspace(-15e-6, 15e-6, n), np.full(n, 0.3e-6), np.full(n, -0.2e-6)], 1)
    C = sf.channels(line); scale = C.std(0)
    beyond = np.ones(n, bool)             # beyond every strand's gate, and off every segment's extended sheath surfaces
    for c, a, b in zip(cls, ra, rb):
        A_ = c[:-1]; AB = c[1:] - c[:-1]; AB2 = (AB ** 2).sum(1); U = AB / np.sqrt(AB2)[:, None]
        t = np.clip(((line[:, None, :] - A_[None]) * AB[None]).sum(-1) / AB2[None], 0, 1)
        d = np.linalg.norm(line[:, None, :] - (A_[None] + t[..., None] * AB[None]), axis=-1).min(1)
        beyond &= d > (1.0 + StrandFieldBasis.NEAREST_GATE_RADII) * b + 0.05e-6
        z = ((line[:, None, :] - A_[None]) * U[None]).sum(-1); rho = np.linalg.norm(line[:, None, :] - A_[None] - z[..., None] * U[None], axis=-1)
        beyond &= ((np.abs(rho - a) > 0.05e-6) & (np.abs(rho - b) > 0.05e-6)).all(1)
    keep = beyond[1:] & beyond[:-1]
    assert keep.sum() > 500, keep.sum()
    step = np.abs(np.diff(C, axis=0))[keep].max(0)
    assert (step < 0.02 * scale).all(), step / scale
    one = [np.array([[0, 0, -40e-6], [0, 0, 40e-6]]) + 1e-6]
    many = [np.stack([np.zeros(17), np.zeros(17), np.linspace(-40e-6, 40e-6, 17)], 1) + 1e-6]
    Q = rng.uniform(-8e-6, 8e-6, (300, 3))
    np.testing.assert_allclose(StrandFieldBasis(many, [1e-6], [1.5e-6], cutoff_m=100e-6).channels(Q),
                               StrandFieldBasis(one, [1e-6], [1.5e-6], cutoff_m=100e-6).channels(Q), atol=1e-6, rtol=0)
