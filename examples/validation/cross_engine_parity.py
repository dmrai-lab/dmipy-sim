"""Cross-engine parity fixtures: MC/DC's undulating axons and Disimpy's cylinder mesh, walked here.

Two published Monte-Carlo references, each walked here on the SAME geometry at the SAME acquisition and
compared at signal level, then published as one certified pack per fixture to ``SubstrateCommons/parity-fixtures``.

* **MC/DC** (Rafael-Patino et al. 2020, Front. Neuroinform. 14:8, LGPL-2.1,
  https://github.com/jonhrafe/Robust-Monte-Carlo-Simulations): single undulating axon PLY meshes, the
  ``ActiveAxG140_PM.scheme`` ActiveAx protocol (372 measurements, TE 53.52 ms, four shells), and the raw
  ``*_DWI.bfloat`` their engine produced. Their signal is a Monte-Carlo estimate too, so the tolerance is the
  two floors together: theirs from the walker count the b = 0 entry states (50,000) and the phase variance
  this walk measures, ours from the split-half of this walk.
* **Disimpy** (Kerkelae et al. 2020, JOSS 5(52):2527, MIT, https://github.com/kerkelae/disimpy):
  ``tests/cylinder_mesh_closed.pkl`` against ``misst_cylinder_signal_smalldelta_30ms_bigdelta_40ms_radius_5um.txt``.
  MISST is an exact eigenfunction solution, so the tolerance is our floor alone.

This module is the reproduction: the fixture constants, the substrate specs, the walks, the references and
the two Monte-Carlo floors. ``tests/validation/test_cross_engine_parity.py`` asserts on it, and the pack
builder that publishes the family to ``SubstrateCommons/parity-fixtures`` imports it.

    DMIPY_SIM_MCDC_ROBUST_DIR=... DMIPY_SIM_DISIMPY_DIR=... python examples/validation/cross_engine_parity.py
"""
import os
import pickle
import time

import numpy as np

SEED = 0

# ---------------------------------------------------------------------------------------- MC/DC, their run
MCDC_TE = 0.05352            #: s, the configuration's `duration` and every row's TE
MCDC_D = 0.6e-9              #: m^2/s, the configuration's `diffusivity`
MCDC_N_T = 2677              #: saves over TE -> dt 20 us; 10704 / 2676 = 4, so every lobe edge is still on a sample
MCDC_SCHEME = "ActiveAxG140_PM.scheme"
MCDC_CONF = "uAxon_d_1.0_amp_0.0_wL_4.0.conf"
MCDC_LICENSE = "LGPL-2.1"
MCDC_CITATION = ("Rafael-Patino J, Romascano D, Ramirez-Manzanares A, Canales-Rodriguez EJ, Girard G, "
                 "Thiran J-P (2020) Robust Monte-Carlo Simulations in Diffusion-MRI: Effect of the Substrate "
                 "Complexity and Parameter Choice on the Reproducibility of Results. Front. Neuroinform. 14:8, "
                 "the meshes, the scheme and the reference signals (LGPL-2.1); "
                 "Fick RHJ (2026), the replay pack, SubstrateCommons")

#: The undulating axons this family walks: (amplitude um, wavelength um), three spanning the released
#: amp 0.0-2.6 x wL 4-32 um grid at d = 1 um.
#:
#: **10 of the 31 released meshes are UNCAPPED tubes** -- 40 boundary edges, two open rims of 20 edges --
#: and enclose no volume, so they are not substrates and are not walked here: every `wL 4.0` up to amp 1.6,
#: every `wL 8.0` up to amp 1.0, and `amp 0.0 / wL 4.0`. MC/DC walks them because its walkers never reach
#: the rim (the released initial-walker lists span the central ~40 um of a 250 um tube); a spec cannot,
#: because an open surface does not say which side is intra.
MCDC_FIXTURES = [(0.2, 32.0), (1.0, 12.0), (2.6, 4.0)]

# -------------------------------------------------------------------------------------- Disimpy, their test
DISIMPY_D = 2e-9             #: m^2/s, `test_mesh_diffusion`
DISIMPY_TE = 70e-3           #: s, the window the fixture's gradient array spans
DISIMPY_DELTA = 30e-3        #: s, the timing MISST was evaluated at (the reference file's name)
DISIMPY_BIGDELTA = 40e-3     #: s
DISIMPY_N_T = 1401           #: saves over TE -> dt 50 us; 1400 = 7 x 200, so 30 ms, 40 ms and TE are on samples
DISIMPY_BVALUES = np.linspace(1.0, 3e9, 100)
DISIMPY_RADIUS = 5e-6
DISIMPY_LICENSE = "MIT"
DISIMPY_CITATION = ("Kerkelae L, Nery F, Hall MG, Clark CA (2020) Disimpy: A massively parallel Monte Carlo "
                    "simulator for generating diffusion-weighted MRI data in Python. JOSS 5(52):2527, the "
                    "cylinder mesh and the MISST reference signal (MIT); "
                    "Drobnjak I, Zhang H, Hall MG, Alexander DC (2011) MISST, the exact reference; "
                    "Fick RHJ (2026), the replay pack, SubstrateCommons")

SIGMA = 3e-3                 #: the split-half floor each pack is sized to
PILOT_N = 4000


def peak_rss_gb():
    """Peak resident memory in GB, from ``VmRSS``: ``statm`` pages are 64 kB on this family of boxes and
    multiplying them by 4096 under-reads by 16x."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / (1024 ** 2)
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


# --------------------------------------------------------------------------------------------------- MC/DC
def mcdc_paths(data, amp, wL):
    stem = f"uAxon_d_1.0_amp_{amp}_wL_{wL}"
    return dict(
        ply=os.path.join(data, "Experiments-mesh -files", "Undulated-fibres", "d_1", f"{stem}.ply"),
        ini=os.path.join(data, "Experiments-mesh -files", "Undulated-fibres", "d_1", f"{stem}_ini_points.txt"),
        dwi=os.path.join(data, "Experiments-raw-signals.", "Undulated_fibers", "d_1",
                         f"uAxon_amp_{amp}_wL_{wL}", f"{stem}_DWI.bfloat"),
        scheme=os.path.join(data, "Simulator-Conf-files", MCDC_SCHEME),
        conf=os.path.join(data, "Simulator-Conf-files", MCDC_CONF), stem=stem)


def mcdc_spec(data, amp, wL):
    from dmipy_sim.io import mcdc
    from dmipy_sim.spec import mcdc_axon_spec
    p = mcdc_paths(data, amp, wL)
    conf = mcdc.read_conf(p["conf"])
    return mcdc_axon_spec(
        p["ply"], scale=conf["mesh_scale"], D=conf["diffusivity"],
        voxel=(conf["voxel_min"], conf["voxel_max"]), ini_walkers=p["ini"],
        cite_ply_as=os.path.basename(p["ply"]), id=f"mcdc-robust/{p['stem']}",
        description=(f"MC/DC undulating axon, d = 1 um, undulation amplitude {amp} um, wavelength {wL} um "
                     f"(Rafael-Patino et al. 2020): one closed lumen in free space, walked at their own "
                     f"diffusivity and seeded from their own initial-walker list."))


def mcdc_walk(data, amp, wL, n, *, n_t=MCDC_N_T, batch=25_000, require_gpu=None, sub_steps=None):
    """One trajectory walk of an MC/DC axon, seeded MC/DC's way (their list, read cyclically)."""
    import dmipy_sim as d
    from dmipy_sim.io import mcdc
    from dmipy_sim.spec import geometry_from_spec
    p = mcdc_paths(data, amp, wL)
    geom = geometry_from_spec(mcdc_spec(data, amp, wL))
    r0 = mcdc.seed_positions(mcdc.read_ini_walkers(p["ini"]), n)
    dt = MCDC_TE / (n_t - 1)
    return d.simulate_trajectories(int(n), MCDC_D, geom, T_max=MCDC_TE, dt_save=dt, seed=SEED, tiers="all",
                                   r0=r0, walker_batch_size=batch, require_gpu=require_gpu, sub_steps=sub_steps)


def mcdc_reference(data, amp, wL, seq):
    """Their released signal, normalised by the walker count its b = 0 entries state."""
    from dmipy_sim.io import mcdc
    p = mcdc_paths(data, amp, wL)
    S = mcdc.read_bfloat(p["dwi"], n_measurements=seq.G.shape[0])
    N = mcdc.walker_count(S, seq.b())
    return S / N, N


# ------------------------------------------------------------------------------------------------- Disimpy
def disimpy_mesh(data):
    """``(V, F, repairs)`` of ``tests/cylinder_mesh_closed.pkl``: a capped 49-gon tube of radius 5 um, length 25 um.

    As stored the surface reads as OPEN -- 352 vertices, 112 boundary edges -- because it writes a coincident
    duplicate of every seam vertex, so the faces either side of a seam index different copies of the same point
    and the edge between them is counted twice as a boundary. Merging the duplicates
    (:func:`~dmipy_sim.geometry.mesh.merge_duplicate_vertices`, the repair the repo's ``load_ply`` applies to a
    file) leaves 296 vertices, 588 faces and **0 boundary edges**: the mesh is closed and its name is right.
    Its area is 0.999 of the ideal cylinder-plus-caps, so the geometry is what MISST was evaluated on to the
    facet.

    The repair is not cosmetic: a walker seeded by ray parity against the unmerged surface lands outside the
    lumen (measured: radii to 7.06 um in a 5 um tube).
    """
    from dmipy_sim.geometry.mesh import merge_duplicate_vertices, surface_topology
    with open(os.path.join(data, "disimpy", "tests", "cylinder_mesh_closed.pkl"), "rb") as fh:
        d = pickle.load(fh)
    V0, F0 = np.asarray(d["vertices"], float), np.asarray(d["faces"], np.int64)
    V, F, n_merged = merge_duplicate_vertices(V0, F0)
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    topo = surface_topology(V, F)
    if topo["boundary_edges"]:
        raise ValueError(f"cylinder_mesh_closed.pkl is open after the merge: {topo}")
    return V, F, dict(vertices_before=int(len(V0)), vertices_after=int(len(V)),
                      faces_before=int(len(F0)), faces_after=int(len(F)),
                      boundary_edges_before=int(surface_topology(V0, F0)["boundary_edges"]),
                      boundary_edges_after=0)


def disimpy_geometry(data):
    """The walked geometry, built from :func:`disimpy_spec` so the walk (and any pack of it) carries a spec
    that names the fixture rather than the anonymous one a ``Mesh`` from arrays writes for itself."""
    from dmipy_sim.spec import geometry_from_spec
    return geometry_from_spec(disimpy_spec(data))


def _disimpy_mesh_object(data):
    from dmipy_sim.geometry.mesh import Mesh
    V, F, _ = disimpy_mesh(data)
    lo = np.zeros(3)
    hi = np.array([1e-5, 1e-5, 2.5e-5])          # the voxel Disimpy's `substrates.mesh(padding=0)` builds
    return Mesh(V, F, periodic=False, voxel_min=lo, voxel_max=hi, box_reflect=True, pool="intra",
                feature_radius=DISIMPY_RADIUS)


def disimpy_spec(data):
    """The fixture's substrate spec. The surface is DERIVED -- the repaired ``cylinder_mesh_closed.pkl`` --
    so it is written into the surface cache under its content hash and cited by that file's basename, and
    the pickle it came from is cited by sha256 in the provenance."""
    import dataclasses
    from dmipy_sim.spec import spec_of
    from dmipy_sim.fill.hub import sha256_of
    pkl = os.path.join(data, "disimpy", "tests", "cylinder_mesh_closed.pkl")
    g = _disimpy_mesh_object(data)
    spec = spec_of(g, id="disimpy/cylinder_mesh_closed", provenance={
        "source": "Disimpy (Kerkelae et al. 2020), tests/cylinder_mesh_closed.pkl, MIT",
        "files": [{"path": "cylinder_mesh_closed.pkl", "sha256": sha256_of(pkl)}],
        "transformations": [
            "duplicate vertices merged (352 -> 296): as stored the surface reads as open with 112 boundary "
            "edges, and is closed with 0 after the merge; its area is 0.999 of the ideal cylinder plus caps",
            "voxel [0,0,0]..[10,10,25] um: Disimpy's substrates.mesh with padding = 0",
            f"radius {DISIMPY_RADIUS * 1e6:g} um along z, walls purely reflecting (no relaxivity in the walk)",
            "the surface is DERIVED from the pickle and written into $DMIPY_SIM_SURFACE_DIR under its content "
            "hash; rebuild it with examples/validation/cross_engine_parity.disimpy_mesh"],
        "created": time.strftime("%Y-%m-%d")})
    wall = spec.walls[0]
    surface = dataclasses.replace(wall.surface, file=os.path.basename(wall.surface.file))
    return dataclasses.replace(spec, walls=[dataclasses.replace(wall, surface=surface)]).validate()


def disimpy_sequence(*, delta=DISIMPY_DELTA, Delta=DISIMPY_BIGDELTA, n_t=DISIMPY_N_T):
    """The fixture's acquisition: PGSE along x, ``delta`` / ``Delta``, 100 b-values to 3e9 s/m^2, TE 70 ms.

    Disimpy's test builds a 700-sample array whose lobes are 299 samples from index 1 and 400, i.e.
    ``delta = 29.943 ms`` and ``Delta = 39.957 ms``, and compares it to a MISST signal computed at exactly
    30 / 40 ms. This sequence is MISST's timing, since MISST is the exact reference; the fixture's own array
    timing is worth 2.6e-4 of signal here (measured: max abs dS 0.02103 against 0.02077 at 8,000 walkers), so
    the choice does not decide anything.
    """
    from dmipy_sim import pgse
    return pgse([[1.0, 0.0, 0.0]] * len(DISIMPY_BVALUES), delta, Delta, bvalues=DISIMPY_BVALUES,
                TE=DISIMPY_TE, n_t=int(n_t), slew_rate=np.inf)


def disimpy_walk(data, n, *, n_t=DISIMPY_N_T, batch=50_000, require_gpu=None, sub_steps=None):
    """One trajectory walk of the Disimpy cylinder. ``sub_steps`` pins the sub-step count for the ladder."""
    import dmipy_sim as d
    dt = DISIMPY_TE / (int(n_t) - 1)
    return d.simulate_trajectories(int(n), DISIMPY_D, disimpy_geometry(data), T_max=DISIMPY_TE, dt_save=dt,
                                   seed=SEED, tiers="all", walker_batch_size=batch, require_gpu=require_gpu,
                                   sub_steps=sub_steps)


def disimpy_analytic_walk(n, *, n_t=DISIMPY_N_T, require_gpu=None, sub_steps=None):
    """The same walk on the ANALYTIC cylinder of the same radius: identical waveform, seed and walker count.

    This is how faceting and discretisation are separated from the engine (the repo's rule): whatever the mesh
    and the analytic geometry of the same shape share is the engine's, and whatever differs is the surface's
    representation.
    """
    import dmipy_sim as d
    from dmipy_sim import Cylinder
    return d.simulate_trajectories(int(n), DISIMPY_D, Cylinder(radius=DISIMPY_RADIUS, orientation=(0, 0, 1)),
                                   T_max=DISIMPY_TE, dt_save=DISIMPY_TE / (int(n_t) - 1), seed=SEED,
                                   tiers="all", require_gpu=require_gpu, sub_steps=sub_steps)


def disimpy_reference(data):
    return np.loadtxt(os.path.join(data, "disimpy", "tests",
                                   "misst_cylinder_signal_smalldelta_30ms_bigdelta_40ms_radius_5um.txt"))


# ------------------------------------------------------------------------------------- floors and the table
def walk_cos_phi(walk, seq, *, chunk=4000):
    """Per-walker ``cos(phi)`` for every measurement of ``seq``, from the walk's own positions: ``(n_meas, n_w)``.

    The engine's own gradient phase (``replay._replay_kernel.effective_gradient`` resamples the sequence onto the
    walk's save grid, ``gradient_phase`` accumulates ``gamma dt sum G.r``), which is what ``simulate`` computes
    fused and what a pack's bands reconstruct. The floors are measured through this, and the pack's own
    ``walker_primitives(seq).phi`` is compared against it.
    """
    from dmipy_sim.replay import _replay_kernel as rk
    n_t = walk.positions.shape[1]
    G = rk.effective_gradient(np.asarray(seq.G_eff, np.float64), float(seq.dt), n_t, float(walk.dt))
    out = []
    for lo in range(0, walk.positions.shape[0], chunk):
        phi = np.asarray(rk.gradient_phase(G, np.asarray(walk.positions[lo:lo + chunk], np.float32), float(walk.dt)))
        out.append(np.cos(np.asarray(phi, np.float64)))
    return np.concatenate(out, axis=1)


def pack_cos_phi(pack, seq):
    """Per-walker ``cos(phi)`` the PUBLISHED pack gives: ``walker_primitives(seq).phi``, the bands contracted
    once. ``(n_meas, n_w)``, so it lines up with :func:`walk_cos_phi`."""
    return np.cos(np.asarray(pack.walker_primitives(seq).phi, np.float64)).T


def floors(per_walker, n_theirs=None):
    """``(ours, theirs, dof, exact)`` per measurement: the **analytic standard error of the ensemble mean**.

    The signal of a measurement is the mean of ``cos(phi)`` over walkers, so the one-sigma noise of that mean
    is ``sd(cos(phi)) / sqrt(N)`` -- computed from the same walkers, per measurement, with no randomness of its
    own. ``theirs`` is the same per-walker spread at THEIR walker count, because their released file is one
    realisation and carries no spread; ``dof`` is ``N - 1``, conservatively the dof of our sample variance
    alone (their mean adds ~50,000 more, so a band derived at this dof is the wider one).

    A **split half** was used here before and is wrong for this job: half the absolute difference of two
    half-means is a one-degree-of-freedom draw, not an estimate of the noise, and it is shared by every
    measurement through one permutation. Measured on the published ``mcdc-1.0-12.0`` pack over seeds 0..199,
    the worst measurement moved between **2.09 and 3.51 sigma** (median 3.20) and the number of 3-sigma
    exceedances between **0 and 3** (median 2), so the family-wise check at 3.206 sigma failed for **49.5 %**
    of seeds. The analytic estimator gives **2.869 sigma and 0 of 360** for the same pack, once and for all.
    The "2 of 372 outside their own band" that motivated the gate's thresholds was the estimator, not the walk.

    ``exact`` marks the measurements with **zero** per-walker spread: a ``b = 0`` row has ``phi = 0`` for every
    walker, so its mean is 1 exactly and its standard error is exactly 0. There is no band to be inside, so
    such a row is compared exactly instead of in sigma and carries no degrees of freedom -- 12 of the ActiveAx
    scheme's 372 measurements, and their ``|dS|`` is 0.0 on all three packs. Dividing by their zero (or by a
    1e-12 floor, as this did) is what made the statistic ``nan``.
    """
    per_walker = np.asarray(per_walker, float)
    n_m, n_w = per_walker.shape
    sd = per_walker.std(axis=1, ddof=1)
    ours = sd / np.sqrt(float(n_w))
    theirs = sd / np.sqrt(float(n_theirs)) if n_theirs else np.zeros_like(sd)
    return ours, theirs, int(n_w - 1), (sd == 0.0)


def parity(cos_phi, reference, n_theirs=None, *, k=3.0):
    """The parity record of one fixture: our signal, theirs, the floors, and the worst measurement in sigma.

    ``n_theirs`` is their walker count when their reference is itself a Monte-Carlo estimate (MC/DC) and
    ``None`` when it is exact (MISST), in which case only our floor enters the band.

    Every number here is a property of the walkers and the reference, not of a seed. The zero-variance
    measurements are reported separately (``n_exact``, ``max_abs_diff_exact``) because they have no band; the
    sigma statistics are over the rest, and ``n_live`` is how many that is -- which is what the family-wise
    threshold must be derived for.
    """
    cos_phi = np.asarray(cos_phi, float)
    ours = cos_phi.mean(axis=1)
    ref = np.asarray(reference, float)
    if ours.shape != ref.shape:
        raise ValueError(f"{ours.shape} measurements against a reference of {ref.shape}")
    f_ours, f_theirs, dof, exact = floors(cos_phi, n_theirs)
    d = np.abs(ours - ref)
    live = ~exact
    se = np.sqrt(f_ours ** 2 + f_theirs ** 2)
    sigma = np.where(live, d / np.where(live, se, 1.0), 0.0)
    i = int(np.argmax(d))
    j = int(np.flatnonzero(live)[np.argmax(sigma[live])]) if live.any() else -1
    return dict(n_meas=int(len(ours)), n_live=int(live.sum()), n_exact=int(exact.sum()), dof=int(dof),
                k=float(k), max_abs_diff=float(d.max()), at_measurement=i,
                ours_at=float(ours[i]), theirs_at=float(ref[i]), rms_diff=float(np.sqrt((d ** 2).mean())),
                max_abs_diff_exact=float(d[exact].max()) if exact.any() else 0.0,
                floor_ours_max=float(f_ours.max()), floor_ours_med=float(np.median(f_ours)),
                floor_theirs_max=float(f_theirs.max()),
                worst_sigma=float(sigma[live].max()) if live.any() else 0.0, worst_sigma_at=j,
                n_over_k_sigma=int((sigma[live] > k).sum()) if live.any() else 0,
                tol_max=float(k * se.max()))


def multiplicity_thresholds(n_live, dof, *, k_per=3.0, expected_family_wise=0.5, poisson_sigma=3.0):
    """The two thresholds a parity comparison over ``n_live`` measurements needs, derived under H0 **with the
    estimator the gate actually uses** -- the analytic standard error of the ensemble mean, whose studentised
    statistic is Student-t on ``dof = N - 1`` degrees of freedom.

    Deriving them under the wrong estimator is how this went wrong the first time. With the split half that
    stood here, the per-measurement statistic has ONE degree of freedom, its 3-sigma exceedance rate over 372
    measurements is 2.5 rather than 1.0, and the family-wise expectation at k = 3.206 is 1.34 rather than 0.5 --
    so the thresholds were a normal-quantile calculation about a statistic that was not normal, and the pass
    margin was 0.6 %. The estimator is now the analytic one, ``dof`` is ~1e5, and these ARE the t-quantiles.

    * ``max_exceedances`` -- the Poisson upper bound ``mean + poisson_sigma sqrt(mean)`` on the expected count
      ``n_live * p(k_per)``, rounded up. A SYSTEMATIC offset pushes many measurements out at once and breaks
      this; noise does not.
    * ``k_family_wise`` -- the band at which the expected number of exceedances over the whole family drops
      below ``expected_family_wise``, so nothing at all should cross it. One wild measurement breaks this.

    ``n_live`` excludes the zero-variance measurements: a ``b = 0`` row has no spread, no band and no degrees of
    freedom, and is compared exactly instead (12 of the ActiveAx scheme's 372).
    """
    from math import ceil, sqrt
    from scipy.stats import t as student
    m, nu = int(n_live), int(dof)
    if m < 1 or nu < 1:
        raise ValueError(f"n_live {m} and dof {nu} must both be positive")
    p_per = 2.0 * float(student.sf(k_per, nu))                        # two-sided, on the estimator's own dof
    mean = m * p_per
    max_exc = int(ceil(mean + poisson_sigma * sqrt(mean)))
    k_fw = float(student.isf(0.5 * float(expected_family_wise) / m, nu))
    return dict(n_live=m, dof=nu, estimator="analytic standard error of the ensemble mean, sd/sqrt(N)",
                distribution=f"Student-t on {nu} dof", k_per_measurement=float(k_per),
                p_per_measurement=p_per, expected_exceedances=mean, poisson_sigma=float(poisson_sigma),
                max_exceedances=max_exc, expected_family_wise=float(expected_family_wise),
                k_family_wise=k_fw,
                rule=(f"a {k_per:g}-sigma per-measurement band over {m} measurements with a standard error on "
                      f"{nu} degrees of freedom is exceeded {mean:.3f} times by chance, so the gate allows up "
                      f"to {max_exc} exceedances (the Poisson {poisson_sigma:g}-sigma upper bound on that "
                      f"count) and requires the WORST measurement inside {k_fw:.3f} sigma, the band whose "
                      f"expected family-wise exceedance count is {expected_family_wise:g}. Both are derived "
                      f"from the measurement count and the estimator's degrees of freedom, never from the "
                      f"data being gated."))


# --------------------------------------------------------------------------------------------- the envelope
def mcdc_envelope():
    """What an MC/DC-axon pack is certified FOR: the fixture's own scheme, and nothing wider.

    The four ActiveAx shells as (b, delta, Delta) over the scheme's three orthogonal-ish axes, so
    ``waveform_band`` is checked against the acquisition the pack exists to reproduce.
    """
    return dict(bvals=[0.0, 1.925e9, 1.932e9, 3.094e9, 1.319e10],
                dirs=[[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1]],
                delta_frac=0.01774 / MCDC_TE, Delta_frac=0.03578 / MCDC_TE,
                shortd_b=3.094e9, shortd_deltas_frac=[0.01015 / MCDC_TE, 0.00762 / MCDC_TE],
                ogse_periods=[1, 2])        # not in the scheme; a stricter band tier than PGSE alone


def disimpy_envelope():
    """What the Disimpy pack is certified FOR: the fixture's own PGSE, b to 3e9 s/m^2 at 30 / 40 ms along x."""
    return dict(bvals=[0.0, 1.5e9, 3.0e9], dirs=[[1, 0, 0], [0, 0, 1], [1, 0, 1]],
                delta_frac=DISIMPY_DELTA / DISIMPY_TE, Delta_frac=DISIMPY_BIGDELTA / DISIMPY_TE,
                shortd_b=3.0e9, shortd_deltas_frac=[DISIMPY_DELTA / DISIMPY_TE],
                ogse_periods=[1, 2])        # not in the fixture; a stricter band tier than PGSE alone




# ------------------------------------------------------------------------------- the step, not the engine
def step_ladder(walks, reference, *, n_theirs=None):
    """Is what is left between two engines the ENGINE, or the step?

    ``walks`` is ``[(step_m, PersistentWalk, sequence), ...]`` at decreasing sub-step length on the same
    substrate, seed and acquisition. Returns the signed difference from ``reference`` at each step, the
    per-measurement Richardson extrapolation to zero step (first order, from the two finest steps, since a
    reflecting-wall Monte-Carlo step is first-order in the step length) and the floors.

    A walk whose gap to an EXACT reference shrinks with the step and extrapolates into the Monte-Carlo floor
    is a discretisation bias, not a disagreement. One that does not is a disagreement.
    """
    steps = np.array([float(h) for h, _, _ in walks], float)
    order = np.argsort(-steps)                                     # coarsest first
    steps = steps[order]
    cos = [walk_cos_phi(walks[i][1], walks[i][2]) for i in order]
    ours = np.stack([c.mean(axis=1) for c in cos])                 # (n_steps, n_meas)
    ref = np.asarray(reference, float)
    d = ours - ref[None, :]
    h1, h2 = steps[-2], steps[-1]
    extrap = (d[-1] * h1 - d[-2] * h2) / (h1 - h2)                 # d(h) = d0 + c h, at h = 0
    f_ours, f_theirs, dof, exact = floors(cos[-1], n_theirs)
    live = ~exact
    se = np.sqrt(f_ours ** 2 + f_theirs ** 2)
    return dict(steps_m=steps.tolist(), dof=int(dof), n_live=int(live.sum()),
                max_abs_diff=[float(np.abs(x).max()) for x in d],
                extrapolated_max_abs_diff=float(np.abs(extrap).max()),
                floor_ours_max=float(f_ours.max()), floor_theirs_max=float(f_theirs.max()),
                tol_max=float(3.0 * se[live].max()) if live.any() else 0.0,
                extrapolated_worst_sigma=float(np.abs(extrap[live] / se[live]).max()) if live.any() else 0.0,
                monotone=bool(np.all(np.diff([float(np.abs(x).max()) for x in d]) < 0)))


def disimpy_step_ladder(data, n, sub_steps=(1, 2, 4), *, require_gpu=None):
    """``step_ladder`` on the Disimpy cylinder against MISST: one walk per sub-step count, same seed."""
    import dmipy_sim as d
    dt = DISIMPY_TE / (DISIMPY_N_T - 1)
    seq = disimpy_sequence()
    walks = []
    for ss in sub_steps:
        w = disimpy_walk(data, n, require_gpu=require_gpu, sub_steps=int(ss))
        walks.append((float(np.sqrt(6 * DISIMPY_D * w.dt_sim)), w, seq))
    return step_ladder(walks, disimpy_reference(data))


# ------------------------------------------------------------------------------------------- the reproduction
def main():
    """Walk both fixtures at a modest walker count and print the parity table."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcdc-data", default=os.environ.get("DMIPY_SIM_MCDC_ROBUST_DIR"))
    ap.add_argument("--disimpy-data", default=os.environ.get("DMIPY_SIM_DISIMPY_DIR"))
    ap.add_argument("--n-walkers", type=int, default=20_000)
    a = ap.parse_args()
    from dmipy_sim.io import mcdc
    print(f"| fixture | N | max|dS| | rms | our floor | their floor | tolerance | worst sigma |")
    print(f"|---|---|---|---|---|---|---|---|")
    if a.mcdc_data:
        os.environ["DMIPY_SIM_SURFACE_DIR"] = os.path.abspath(
            os.path.join(a.mcdc_data, "Experiments-mesh -files", "Undulated-fibres", "d_1"))
        seq = mcdc.read_scheme(os.path.join(a.mcdc_data, "Simulator-Conf-files", MCDC_SCHEME), n_t=MCDC_N_T)
        for amp, wL in MCDC_FIXTURES:
            ref, n_theirs = mcdc_reference(a.mcdc_data, amp, wL, seq)
            walk = mcdc_walk(a.mcdc_data, amp, wL, a.n_walkers)
            p = parity(walk_cos_phi(walk, seq), ref, n_theirs)
            print(f"| MC/DC amp {amp} wL {wL} | {a.n_walkers:,} | {p['max_abs_diff']:.5f} | {p['rms_diff']:.5f} "
                  f"| {p['floor_ours_max']:.5f} | {p['floor_theirs_max']:.5f} | {p['tol_max']:.5f} "
                  f"| {p['worst_sigma']:.2f} |")
    if a.disimpy_data:
        walk = disimpy_walk(a.disimpy_data, a.n_walkers)
        p = parity(walk_cos_phi(walk, disimpy_sequence()), disimpy_reference(a.disimpy_data))
        print(f"| Disimpy cylinder vs MISST | {a.n_walkers:,} | {p['max_abs_diff']:.5f} | {p['rms_diff']:.5f} "
              f"| {p['floor_ours_max']:.5f} | exact | {p['tol_max']:.5f} "
              f"| {p['worst_sigma']:.2f} |")


if __name__ == "__main__":
    main()
