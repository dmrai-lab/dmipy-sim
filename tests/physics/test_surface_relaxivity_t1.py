"""Physics tests: the longitudinal surface relaxivity ``rho1`` (dmipy-sim#574).

The replay equation's contact channel is general in the two surface relaxivities, like C1 is in
T2 and T1: the C2 gate is ``rho2 * chi_perp + rho1 * chi_parallel`` against the SAME stored
boundary local time, with ``chi_parallel = active - chi_perp`` the complement of the coherence gate
(on while the magnetisation is stored along B0, e.g. a stimulated echo's mixing time). No new
walk, no new channel -- a second contraction of the one C2 series already read for ``rho2``.

Reference for the stimulated-echo check: the exact Brownstein-Tarr series for a sphere of radius
``R``, diffusivity ``D``, Robin surface relaxivity ``rho1`` on a UNIFORM initial magnetisation (the
forward engine has no ``rho1`` term to walk directly -- see ``dmipy_sim/replay/replay.py``'s
refusal in ``replay_bloch`` -- so this is the analytical slow-diffusion-limit reference the issue
allows in that case). Writing ``u = r * C`` turns the radial Robin problem into a 1-D rod on
``[0, R]`` with ``u(0) = 0`` and ``u'(R) = (1/R - rho1/D) u(R)``, a standard Sturm-Liouville
problem; its eigenvalues ``x_n = k_n R`` solve ``tan(x) = x / (1 - rho1 R / D)`` and its
coefficients for a uniform initial condition are elementary integrals of ``sin``. The two limits
check themselves: ``rho1 -> 0`` gives ``tan(x) = x``, the reflecting (Neumann) sphere, decay 0;
``rho1 R / D -> 0`` gives the well-known fast-diffusion rate ``rho1 (S/V) = 3 rho1 / R``.
"""
import numpy as np
import pytest
from scipy.optimize import brentq

import dmipy_sim as d
from dmipy_sim import sequences as seq
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec.tissue import Tissue

D0 = 2e-9          # m^2/s
R = 8e-6           # m
N_WALKERS = 45_000
T_MAX = 0.1        # s -- one segment (RPK.md 4.3)
DT_SAVE = 1e-4     # s
SEED = 11

ENV = dict(bvals=[0.0], dirs=[[1, 0, 0]], ogse_periods=[2], shortd_b=1e9,
           shortd_deltas_frac=[0.05], delta_frac=0.2, Delta_frac=0.5, rho2_list=[1e-5])


@pytest.fixture(scope="module")
def sphere_pack():
    walk = d.simulate_trajectories(N_WALKERS, D0, d.Sphere(R), T_MAX, DT_SAVE, seed=SEED, require_gpu=False)
    return build_replay_pack(walk, id="test/sphere-rho1", K=128, envelope=ENV, license="x", citation="x")


# --------------------------------------------------------------------------- the analytical reference
def _lowest_roots(b, n_modes):
    """The positive roots ``x_n`` of ``tan(x) = x / (1 - b)`` for ``n = 1 .. n_modes`` (excluding the
    spurious ``x = 0``, a root of the equation for every ``b`` but the sphere's only when ``b = 0``)."""
    def f(x):
        return np.sin(x) * (1.0 - b) - x * np.cos(x)
    roots, lo = [], 1e-6
    for n in range(1, n_modes + 1):
        hi = n * np.pi - 1e-6
        xs = np.linspace(lo, hi, 4000)
        fs = f(xs)
        idx = np.flatnonzero(np.sign(fs[:-1]) != np.sign(fs[1:]))
        a, c = (xs[idx[-1]], xs[idx[-1] + 1]) if idx.size else (lo, hi)
        roots.append(brentq(f, a, c, xtol=1e-13, rtol=1e-14))
        lo = n * np.pi + 1e-6
    return np.array(roots)


def sphere_robin_decay(t, D, R, rho2, n_modes=60):
    """``M(t) / M(0)`` of a uniformly-magnetised sphere under a Robin (partially-relaxing) boundary:
    the exact Brownstein-Tarr radial series (see module docstring for the derivation)."""
    t = np.atleast_1d(np.asarray(t, float))
    if rho2 == 0.0:
        return np.ones_like(t)
    b = rho2 * R / D
    x = _lowest_roots(b, n_modes)
    k = x / R
    I = (np.sin(x) - x * np.cos(x)) / k ** 2              # integral_0^R r sin(k r) dr
    J = (2 * x - np.sin(2 * x)) / (4 * k)                  # integral_0^R sin(k r)^2 dr
    coef = 3.0 * I ** 2 / (J * R ** 3)
    return np.array([np.sum(coef * np.exp(-D * k ** 2 * tt)) for tt in t])


def test_the_reference_series_is_self_consistent():
    """The two limits the series must reproduce, independent of any Monte Carlo: no relaxivity is no
    decay, and a weak one is the textbook fast-diffusion rate rho1 * (S/V) = 3 rho1 / R."""
    t = np.array([0.0, 0.01, 0.05])
    np.testing.assert_array_equal(sphere_robin_decay(t, D0, R, 0.0), np.ones_like(t))
    rho1 = 1e-7                                            # rho1 R / D = 4e-4: deep in the well-mixed regime
    got = sphere_robin_decay(t, D0, R, rho1)
    ref = np.exp(-(3.0 * rho1 / R) * t)
    np.testing.assert_allclose(got, ref, rtol=2e-3)


# --------------------------------------------------------------------------- acceptance: rho1 = 0
def test_rho1_zero_is_bit_identical_to_absent(sphere_pack):
    """A rho1 of 0 or None changes nothing: no new floating-point operation runs on that path (the
    longitudinal term is skipped, not computed-and-multiplied-by-zero), for a spin echo, a stimulated
    echo and a gradient echo alike."""
    pk = sphere_pack
    waveforms = dict(
        spin_echo=seq.pgse([[1, 0, 0]], 2e-3, 6e-3, bvalues=[5e8], n_t=pk.n_t, slew_rate=np.inf),
        stimulated_echo=seq.pgste([[1, 0, 0]], 1e-3, 0.02, bvalues=[3e8], n_t=pk.n_t, slew_rate=np.inf),
        gradient_echo=seq.gre(4e-3, n_t=pk.n_t, slew_rate=np.inf),
    )
    for name, wf in waveforms.items():
        baseline = pk.replay(wf, tissue=Tissue(rho2=4e-6))
        zero = pk.replay(wf, tissue=Tissue(rho2=4e-6, rho1=0.0))
        absent = pk.replay(wf, tissue=Tissue(rho2=4e-6, rho1=None))
        np.testing.assert_array_equal(baseline, zero, err_msg=f"{name}: rho1=0.0 is not bit-identical")
        np.testing.assert_array_equal(baseline, absent, err_msg=f"{name}: rho1=None is not bit-identical")
    # and with no rho2 at all
    wf = waveforms["stimulated_echo"]
    np.testing.assert_array_equal(pk.replay(wf), pk.replay(wf, tissue=Tissue(rho1=0.0)))


def test_rho1_zero_is_bit_identical_in_the_study_and_pose_routes(sphere_pack):
    """The same bit-identity through the other two depths that read the C2 channel: the study
    primitives (`Primitives.contact_t1`) and the pose closed form."""
    pk = sphere_pack
    wf = seq.pgste([[1, 0, 0]], 1e-3, 0.02, bvalues=[3e8], n_t=pk.n_t, slew_rate=np.inf)
    w1, ew1, E1 = pk.walker_signals(wf, tissue=Tissue(rho2=4e-6))
    w2, ew2, E2 = pk.walker_signals(wf, tissue=Tissue(rho2=4e-6, rho1=0.0))
    np.testing.assert_array_equal(ew1, ew2)
    np.testing.assert_array_equal(E1, E2)
    pr1 = pk.pose_response(wf, tissue=Tissue(rho2=4e-6))
    pr2 = pk.pose_response(wf, tissue=Tissue(rho2=4e-6, rho1=0.0))
    np.testing.assert_array_equal(np.asarray(pr1.coeffs), np.asarray(pr2.coeffs))


# --------------------------------------------------------------------------- acceptance: spin echo independence
def test_spin_echo_is_independent_of_rho1(sphere_pack):
    """A spin echo never stores magnetisation (chi_perp == active throughout, so chi_parallel == 0
    everywhere): rho1 has nothing to gate and the signal is unchanged, at any rho1."""
    pk = sphere_pack
    wf = seq.pgse([[1, 0, 0]], 2e-3, 6e-3, bvalues=[5e8], n_t=pk.n_t, slew_rate=np.inf)
    baseline = pk.replay(wf, tissue=Tissue(rho2=3e-6))
    for rho1 in (1e-6, 1e-4, 1e-2):
        np.testing.assert_array_equal(baseline, pk.replay(wf, tissue=Tissue(rho2=3e-6, rho1=rho1)),
                                      err_msg=f"a spin echo changed with rho1={rho1:g}")


# --------------------------------------------------------------------------- acceptance: stimulated echo vs reference
def test_stimulated_echo_matches_the_sphere_robin_series(sphere_pack):
    """A store-wait-recall schedule with b ~ 0 (so only the storage-period wall relaxation acts): the
    replayed signal, relative to its rho1 = 0 value, matches the exact Robin-sphere series to the
    Monte Carlo floor (``1/sqrt(N)``) plus the MC's own discretisation bias -- the same scale
    (a few percent) ``tests/physics/test_surface_relaxivity.py`` already accepts for the T2 analogue."""
    pk = sphere_pack
    rho1 = 3e-4                                            # rho1 R / D = 1.2: a genuine Robin (not well-mixed) test
    floor = 1.0 / np.sqrt(N_WALKERS)
    for TM in (0.005, 0.01, 0.02, 0.04):
        wf = seq.pgste([[1, 0, 0]], DT_SAVE, TM, bvalues=[0.0], n_t=pk.n_t, slew_rate=np.inf)
        S0 = float(pk.replay(wf, tissue=Tissue(rho2=0.0))[0])
        S1 = float(pk.replay(wf, tissue=Tissue(rho2=0.0, rho1=rho1))[0])
        ratio = S1 / S0
        ref = float(sphere_robin_decay([TM], D0, R, rho1)[0])
        tol = max(5 * floor, 0.06 * ref)
        assert abs(ratio - ref) < tol, f"TM={TM}: ratio={ratio:.5f} vs reference={ref:.5f} (tol {tol:.4f})"
    # monotone in TM: longer storage, more wall contact, less signal
    TMs = np.array([0.005, 0.02, 0.05])
    ratios = [float(pk.replay(seq.pgste([[1, 0, 0]], DT_SAVE, TM, bvalues=[0.0], n_t=pk.n_t, slew_rate=np.inf),
                              tissue=Tissue(rho2=0.0, rho1=rho1))[0])
             for TM in TMs]
    assert ratios[0] > ratios[1] > ratios[2]


# --------------------------------------------------------------------------- acceptance: replay vs the pose closed form
def test_replay_and_the_pose_closed_form_agree(sphere_pack):
    """The pose expansion at the identity pose composes back to the plain replay (#197): since
    :meth:`~dmipy_sim.replay.replay.ReplayPack.pose_response` reads the SAME ``_prepare`` weights
    (``ew``), this is a structural check that rho1 reaches the closed form too, not a new route."""
    from dmipy_sim.replay.so3 import Distribution
    pk = sphere_pack
    wf = seq.pgste([[1, 0, 0]], 1e-3, 0.02, bvalues=[3e8], n_t=pk.n_t, slew_rate=np.inf)
    tissue = Tissue(rho2=2e-6, rho1=8e-6)
    S_replay = np.asarray(pk.replay(wf, tissue=tissue))
    pr = pk.pose_response(wf, tissue=tissue)
    S_pose = np.abs(np.asarray(pr.compose(Distribution.pose(np.eye(3)))))
    np.testing.assert_allclose(S_replay, S_pose, rtol=1e-4)


# --------------------------------------------------------------------------- acceptance: the device (columnar) route
def test_the_columnar_device_route_applies_rho1(tmp_path):
    """The device reduction (`ColumnarPack.image_study`) must read `rho1` exactly as the host study
    route does -- the kernel that had grown its own copy once before (#484) and silently dropped a
    term; this is the same class of bug, so it gets the same style of test."""
    import os
    from dmipy_sim.io.strands import write_tck
    from dmipy_sim.phantom import Grid
    from dmipy_sim.replay.bank import merge_packs
    from dmipy_sim.replay.replay import ReplayPack
    from dmipy_sim.fill.consolidate import consolidate
    from dmipy_sim.spec import disco_spec, walk_spec, StratifiedByVoxel
    from dmipy_sim.replay.study import Protocol, Study

    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp_path / "t.tck"), str(tmp_path / "d.txt")
    write_tck(tck, cls_, coordinate_unit_m=25e-6)
    np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    spec = disco_spec(tck, dia, side_m=20e-6, field=False)
    grid = Grid(shape=(2, 2, 2), voxel_size_m=(10e-6,) * 3, origin_m=(5e-6,) * 3)
    os.makedirs(str(tmp_path / "shards"))
    packs = []
    for b in (0, 1):
        want = np.zeros(grid.shape, np.int64); want[b] = 12
        w = walk_spec(spec, T_max=8e-4, dt_save=2e-4, seed=23 + b, require_gpu=False,
                      seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}))
        packs.append(build_replay_pack(w, id=f"t/{b}", license="x", citation="x", K=3, voxel_grid=grid,
                                       out_path=str(tmp_path / "shards" / f"block-000{b}.p1.rpk")))
    consolidate(str(tmp_path / "shards"), str(tmp_path / "layout"), blocks=[0, 1], id="t/columns")
    merged = merge_packs(packs, id="t/merged")
    col = ReplayPack.open(str(tmp_path / "layout"))
    seq1 = d.pgste([[1, 0, 0]], 0.2e-3, 0.4e-3, bvalues=[0.0], n_t=merged.n_t, slew_rate=np.inf)
    tissue = Tissue(T2={"intra": 0.03, "extra": 0.08, "myelin": 0.01}, rho2=1e-5, rho1=8e-6)
    study = Study(Protocol([seq1]), tissues=[None, tissue], scanners=[None])
    assert study.needs_contact
    S, floor, plan = col.image(study, tol=1e-9, chunk_rows=5)
    ijk, _ = grid.bin(merged.r0)
    v = np.ravel_multi_index(ijk.T, grid.shape)
    for k in range(len(study)):
        tk, sk = study.resolved(k)
        w, ew, E = merged.walker_signals(seq1, tissue=tk, scanner=sk)
        keys, inv = np.unique(v, return_inverse=True)
        num = np.zeros((len(keys), E.shape[1]), complex)
        den = np.zeros(len(keys))
        np.add.at(num, inv, ew[:, None] * E)
        np.add.at(den, inv, w)
        np.testing.assert_allclose(S[k].reshape(-1, E.shape[1])[keys], np.abs(num / den[:, None]),
                                   rtol=1e-9, atol=1e-12)
