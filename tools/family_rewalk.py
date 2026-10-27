"""The replayable-MC paper's substrate families walked once each, on a stated certificate envelope.

Three families: the canonical pores (planes, cylinders and spheres, 0.1 to 20 um, the ids and seeds of
``examples/substrate_bank/build_canonical_pores.py``), the closed permeable slab (4 um, a membrane at its middle,
kappa 10 and 100 um/s, a field layer in one compartment) and the CACTUS bundle (inner and outer meshes of each
strand, three pools: intra, frozen myelin water, extra). Each pack is walked once at every tier its substrate
carries and built with :func:`~dmipy_sim.replay.bank.build_replay_pack` on the family's ENVELOPE:

* the gradient side for a scanner class (``scanner``: every waveform within its ``G_max`` per axis),
* the contact tier at the surface relaxivities ``rho2_list`` (m/s),
* the field tier at the strengths ``B0_list`` (T) and the susceptibility scale ``chi_max`` (the spec's magnetic
  pools', or the family's where its field is not a pool's).

The defaults are the DiSCo recipe's (connectome_2, rho2 <= 3 um/s, B0 in 1.5 / 3 / 7 T); ``--envelope`` replaces
any of them from a JSON file.

The walker count is the toggle-proof rule. Every physics toggle of a replay (relaxation, relaxivity, a field, a
pool's weight) multiplies a walker's contribution by a factor of modulus at most one, so the standard error of a
signal in units of the unrelaxed water signal obeys ``SE^2 <= sum_p a_p^2 / n_p`` for pools of water share
``a_p`` walked by ``n_p`` walkers. At equal cost per walker the cheapest counts that hold ``SE <= f`` for every
toggle and every waveform are ``n_p = a_p / f^2``: ``1 / f^2`` walkers in all, seeded in proportion to the water
(:func:`toggle_proof_counts` states the general, cost-weighted form ``n_p = a_p sum_q a_q sqrt(c_q / c_p) / f^2``).

A pore needs fewer: no waveform of the class dephases a walker confined to a small pore, and a pilot bounds what
any refocused waveform within the class's ``G_max`` per axis on the axes the pore CONFINES (the plane's normal,
the cylinder's cross-section, all three for the sphere) and any contact toggle up to the envelope's ``rho2`` can
make that variance (:func:`phase_variance_bound`, :func:`toggle_variance_bound`). That is what a pore's pack
certifies: its standard error is at most ``f`` for every such waveform and toggle; along a free axis the
attenuation is free diffusion's, analytic, and not what the count is set for. ``plan --pilot N`` walks the
pilot per pore at the pack's own seed (``pack_seed(id)``) and sets ``n = max(n_min, V / f^2)``; production walks
``N = n_min`` (8,000), so a pore whose count is ``n_min`` IS its pilot, packed there and walked once, and a pore
that needs more is walked afresh by ``walk --plan`` at ``pack_seed(id + "/after-pilot")``: its walkers are then
independent of the pilot's, whose variance chose their number, and the pilot is not merged into it.

Stages::

    python tools/family_rewalk.py plan pores                          # counts, step resolution, cost, backend
    python tools/family_rewalk.py plan pores --pilot 4000 --out DIR   # the counts from a pilot bound per pore
    python tools/family_rewalk.py walk pores --out DIR --backend cuda # walk + pack every substrate not yet in DIR
    python tools/family_rewalk.py check DIR                           # every pack: stated envelope, certificate, a replay

``walk --keep-walk DIR`` also saves each substrate's float ``CoefficientWalk`` to ``DIR/<pack id>.safetensors``
(``--keep-walk-ids`` restricts it to the substrates named there), its path and sha256 in the pack's receipt;
``walk --raw-subset N`` keeps the block ends, contact and field reads of ``N`` walkers and builds the pack
``fidelity="measured"`` (its certificate replayed on that subset against the pack's own coefficients). The CACTUS
bundle's walk drops the subset in its join, so that family refuses the option.

``walk`` is resumable: a substrate whose pack and receipt exist in ``DIR`` is not walked again, and
``--shard i/n`` takes every n-th substrate from the i-th, so n jobs share one family. ``--smoke`` is the same
pipeline on a few substrates at a few hundred walkers over a short window, for a CPU (``JAX_PLATFORMS=cpu``).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: The certificate envelope every family states: DiSCo's (tessera's recipe ``pack``: rho2_max_m_s, B0_list_T; its
#: ``walk.scanner``). The battery's shapes are the library default's; ``chi_max`` is the spec's unless a family
#: states its own.
ENVELOPE = dict(scanner="connectome_2", rho2_list=[3e-6], B0_list=[1.5, 3.0, 7.0])

#: Bands per 100 ms storage window: the storage band (1280 Hz), DiSCo's.
K_BANDS = 256
WINDOW_T = 0.1


# ------------------------------------------------------------------ the walker rule
def toggle_proof_counts(shares, floor, costs=None):
    """Walkers per pool so that every replay's standard error is at most ``floor`` (units of the unrelaxed water
    signal) for ANY physics toggle: minimise ``sum_p n_p c_p`` subject to ``sum_p a_p^2 / n_p <= floor^2``, which is
    ``n_p = a_p sum_q a_q sqrt(c_q / c_p) / floor^2``. ``shares`` maps a pool to its water share ``a_p`` (normalised
    here); ``costs`` maps a pool to its cost per walker (equal when None, which gives ``n_p = a_p / floor^2``).
    Returns ``{pool: n_p}`` (ceilings)."""
    a = {p: float(v) for p, v in shares.items() if float(v) > 0.0}
    tot = sum(a.values())
    if not tot > 0.0 or not float(floor) > 0.0:
        raise ValueError("toggle_proof_counts needs positive water shares and a positive floor")
    a = {p: v / tot for p, v in a.items()}
    c = {p: 1.0 for p in a} if costs is None else {p: float(costs[p]) for p in a}
    if not all(v > 0.0 for v in c.values()):
        raise ValueError("every pool's cost per walker must be positive")
    s = sum(a[q] * np.sqrt(c[q]) for q in a)
    return {p: int(np.ceil(a[p] * s / np.sqrt(c[p]) / float(floor) ** 2)) for p in a}


def toggle_proof_bound(shares, counts):
    """``sqrt(sum_p a_p^2 / n_p)``: the standard error any toggle of a replay is held under at these counts."""
    tot = sum(float(v) for v in shares.values())
    return float(np.sqrt(sum((float(shares[p]) / tot) ** 2 / int(counts[p]) for p in counts if int(counts[p]) > 0)))


# ------------------------------------------------------------------ the pilot's variance bound
#: The pilot's statistical margin on a sampled second moment: the estimate's relative standard error is
#: sqrt(2 / (n - 1)) for a Gaussian quantity, and the bound takes three of them above the largest of the whole
#: pilot's and its two halves' estimates.
PILOT_MARGIN_SE = 3.0
#: The surface relaxivities (fractions of the envelope's largest) and the readout times (every block end) the
#: toggles' weights are read at.
RHO_FRACTIONS = tuple(np.linspace(0.0, 1.0, 17))


def _top_eigenvalue(Y):
    """The largest eigenvalue of ``Y^T Y / (m - 1)`` for ``Y`` (m, p), through the smaller Gram side."""
    m = Y.shape[0]
    A = (Y @ Y.T) if m < Y.shape[1] else (Y.T @ Y)
    return float(np.linalg.eigvalsh(A / max(m - 1, 1))[-1])


def projected_gram(T, K):
    """``A_jl = <P g_j, P g_l>`` for the basis a pack's path is stored on beyond its start, ``g = [t / T, s_1 .. s_K]``
    (``s_k = sqrt(2 / T) sin(pi k t / T)``), ``P`` the removal of the time mean over ``[0, T]``: what a refocused
    waveform (``int G dt = 0``) reads of a path. Closed form."""
    T, k = float(T), np.arange(1, int(K) + 1)
    m = np.sqrt(2.0 / T) * (1.0 - (-1.0) ** k) / (np.pi * k)            # (1 / T) int s_k
    A = np.empty((int(K) + 1, int(K) + 1))
    A[0, 0] = T / 12.0                                                    # t / T: T / 3 less T (1 / 2)^2
    lin = np.sqrt(2.0 / T) * T * (-1.0) ** (k + 1) / (np.pi * k) - 0.5 * T * m
    A[0, 1:] = A[1:, 0] = lin
    A[1:, 1:] = np.eye(int(K)) - T * np.outer(m, m)
    return A


def phase_variance_bound(position_coeffs, T, *, G_max, axes=(0, 1, 2), margin_se=PILOT_MARGIN_SE):
    """The largest variance over walkers of the phase a pack SERVES, over every refocused waveform of a scanner
    class on the ``axes`` it is certified for: ``sup Var(phi)``, ``phi = gamma sum_j C_j W_j`` the replay's own phase
    (``C = [x(0), x(T) - x(0), beta_1..beta_K]`` per axis, ``W`` the waveform's :func:`compression.band_projection`),
    ``|G_axis| <= G_max`` on [0, T], ``int G dt = 0`` and ``G`` zero on every other axis (a pore's free axes are not
    confined, and a waveform along one dephases any count of walkers).

    A refocused waveform has ``W_0 = 0`` and reads the stored path through ``P g_j`` (:func:`projected_gram`), so
    ``Var(phi) = gamma^2 alpha^T A S A alpha`` with ``G = sum alpha_j P g_j`` its component in their span, ``S`` the
    covariance over walkers of ``C_1..`` and ``||G||^2 >= alpha^T A alpha``; every axis has ``int G_axis^2 <= G_max^2 T``
    (Bessel's inequality), so

        Var(phi) <= gamma^2 G_max^2 T  min( n_axes lambda_max(A^1/2 S_joint A^1/2), (sum_axis sqrt(lambda_axis))^2 ).

    Exact for the served estimator: the bands beyond ``K`` are not in it (their error is the certificate's
    truncation). ``S`` is the pilot's, centred over the walkers, each eigenvalue the largest of the whole pilot's and
    its two halves' times ``1 + margin_se sqrt(2 / (n / 2 - 1))``. Returns a dict with the bound ``var_phi`` and its
    parts."""
    from dmipy_sim.constants import GAMMA
    C = np.asarray(position_coeffs, np.float64)
    if C.ndim == 4:
        if C.shape[1] != 1:
            raise ValueError("the pilot bound reads one storage window; this walk has "
                             f"{C.shape[1]}")
        C = C[:, 0]
    C = C[:, 1:, :][:, :, list(axes)]                                     # beyond the start, on the certified axes
    n, nb, n_ax = C.shape
    ev, U = np.linalg.eigh(projected_gram(T, nb - 1))
    R = (U * np.sqrt(np.clip(ev, 0.0, None))) @ U.T                      # A^1/2
    Y = np.einsum("jl,wla->wja", R, C - C.mean(axis=0, keepdims=True))  # A^1/2 (C - mean): Var(phi) = gamma^2 |.|^2
    perm = np.random.RandomState(0).permutation(n)
    parts = (slice(None), perm[:n // 2], perm[n // 2:])
    grow = 1.0 + float(margin_se) * np.sqrt(2.0 / max(n // 2 - 1, 1))
    lam_axis = [max(_top_eigenvalue(Y[idx, :, a]) for idx in parts) * grow for a in range(n_ax)]
    lam_joint = max(_top_eigenvalue(Y[idx].reshape(len(Y[idx]), nb * n_ax)) for idx in parts) * grow
    scale = GAMMA ** 2 * float(G_max) ** 2 * float(T)
    joint = scale * n_ax * lam_joint
    per_axis = scale * float(sum(np.sqrt(l) for l in lam_axis)) ** 2
    return dict(var_phi=float(min(joint, per_axis)), var_phi_joint=float(joint), var_phi_per_axis=float(per_axis),
                axes=[int(a) for a in axes], lambda_axis_m2s=[float(l) for l in lam_axis],
                lambda_joint_m2s=float(lam_joint), margin=float(grow), n_pilot=int(n), K=int(nb - 1),
                G_max=float(G_max), T=float(T))


def toggle_variance_bound(var_phi, raw_boundary=None, *, D, rho2_max, rho_fractions=RHO_FRACTIONS):
    """The largest ``Var(w e^{i phi})`` over the toggles the envelope allows, in units of the unrelaxed signal.

    ``w`` is the walker's weight, ``exp((rho2 / D) L(t))`` with ``L`` its cumulative contact (``raw_boundary``, the
    pilot's per-block contact; the relaxivities ``rho_fractions x rho2_max``, every block end ``t``); a pool's bulk
    relaxation multiplies every walker of a one-pool substrate alike, which scales the variance down, so the bound
    is read at none. For any constant ``c``, ``Var(Z) <= E|Z - c|^2``; with ``c = E[w] e^{i mu}``, ``mu = E phi``,

        Var(w e^{i phi}) <= ( sd(w) + E[w] sqrt(E|e^{i phi} - e^{i mu}|^2) )^2 <= ( sd(w) + E[w] sqrt(min(Var phi, 4)) )^2

    (Minkowski; ``|e^{ia} - e^{ib}| <= |a - b|``, so the CENTRED second moment of the phase is the one that enters),
    and ``Var(Z) <= E[w^2] <= 1`` besides. Without a contact channel ``w = 1`` and the bound is
    ``min(1, Var(phi))``. Returns ``(V, where)``: the bound and the ``(rho2, t index)`` it was largest at."""
    sd_e = float(np.sqrt(min(float(var_phi), 4.0)))
    if raw_boundary is None or not float(rho2_max) > 0.0:
        return float(min(1.0, sd_e ** 2)), dict(rho2=0.0, t_index=None)
    B = np.asarray(raw_boundary, np.float64)
    L = np.concatenate([np.zeros((B.shape[0], 1)), np.cumsum(B[:, 1:], axis=1)], axis=1)   # the contact at each block end
    best, where = 0.0, dict(rho2=0.0, t_index=None)
    for fr in rho_fractions:
        rho = float(fr) * float(rho2_max)
        W = np.exp((rho / float(D)) * L)
        m, sd, Ew2 = W.mean(axis=0), W.std(axis=0, ddof=1), (W ** 2).mean(axis=0)
        V = np.minimum(Ew2, (sd + m * sd_e) ** 2)
        j = int(np.argmax(V))
        if V[j] > best:
            best, where = float(V[j]), dict(rho2=rho, t_index=j)
    return float(min(best, 1.0)), where


def pilot_count(V, floor, n_min):
    """The pore's walkers: ``max(n_min, ceil(V / f^2))``, ``f`` the floor as a standard error."""
    return int(max(int(n_min), int(np.ceil(float(V) / float(floor) ** 2))))


# ------------------------------------------------------------------ the envelope
def family_envelope(base, overrides=None, **family):
    """The envelope a family's packs are certified under: the library's default battery, :data:`ENVELOPE` over it,
    the family's own declaration ``base`` over that (a family builder's ``ENVELOPE``), then ``family`` keys, then
    ``overrides`` (the ``--envelope`` file)."""
    from dmipy_sim.replay import compression as cx
    env = dict(cx.default_envelope())
    env.update(ENVELOPE)
    env.update(base or {})
    env.update(family)
    env.update(overrides or {})
    return env


def stated(pack):
    """What a pack STATES of its envelope (``replay_envelope``): the scanner class and its ``G_max``, the contact
    tier's ``rho2_over_D_max``, the field's ``chi_max`` and ``B0_list``; None for a key it does not state."""
    env = pack.meta.get("replay_envelope") or {}
    acq, tis = env.get("acquisition") or {}, env.get("tissue") or {}
    return dict(scanner=acq.get("scanner"), G_max=acq.get("G_max"), bound=acq.get("bound"), b_max=acq.get("b_max"),
                B0_list=acq.get("B0_list"), rho2_over_D_max=tis.get("rho2_over_D_max"), chi_max=tis.get("chi_max"),
                surface_relaxivity=env.get("surface_relaxivity"), field=env.get("field"),
                bulk_relaxation=env.get("bulk_relaxation"))


def missing_statements(pack):
    """The envelope statements a pack of this campaign must carry and does not: the scanner class always, the
    relaxivity bound where it stores the contact, the field strengths and chi where it stores the field."""
    s = stated(pack)
    out = []
    if not s["scanner"] or s["G_max"] is None:
        out.append("acquisition.scanner/G_max")
    if s["surface_relaxivity"] and s["rho2_over_D_max"] is None:
        out.append("tissue.rho2_over_D_max")
    if s["field"] and (not s["B0_list"] or s["chi_max"] is None):
        out.append("acquisition.B0_list/tissue.chi_max")
    return out


# ------------------------------------------------------------------ the families
def _pores_module():
    path = os.path.join(ROOT, "examples", "substrate_bank", "build_canonical_pores.py")
    import dmipy_sim  # noqa: F401  (bound first: the builder puts its own checkout on sys.path)
    spec = importlib.util.spec_from_file_location("build_canonical_pores", path)
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


class Family:
    """One family's recipe: its substrates by name, each with a pack id, a dataset path and the call that walks
    it; its envelope, walker floor, bands and licences."""
    name = None
    floor = 5e-3
    blt_K = 8
    license = "CC-BY-4.0"
    pack_license = None
    citation = "Fick RHJ (2026), replayable Monte-Carlo packs, SubstrateCommons"
    #: Whether the family walks pool by pool through ``_walk_bundle``, whose join drops the walk's raw subset.
    bundle = False

    def __init__(self, args):
        self.args = args
        self.raw_subset = int(getattr(args, "raw_subset", 0) or 0)
        if self.raw_subset > 0 and self.bundle:
            raise SystemExit(f"--raw-subset {self.raw_subset}: the {self.name} family is a multi-pool bundle walk, and "
                             "the bundle's join drops the raw subset, so a pack has nothing to be measured on; walk "
                             "it without --raw-subset (a derived certificate)")
        self.T = float(args.T) if args.T else WINDOW_T
        self.K = int(args.K) if args.K else int(np.ceil(K_BANDS * self.T / WINDOW_T))
        self.floor = float(args.floor) if args.floor else self.floor
        self.overrides = json.load(open(args.envelope)) if args.envelope else {}

    def substrates(self):
        raise NotImplementedError

    def shares(self, name):
        return {"water": 1.0}

    def counts(self, name):
        planned = getattr(self, "planned", {}).get(name)
        if planned is not None:                          # a pilot's bound (plan --pilot), read back by walk --plan
            return dict(planned)
        if self.args.smoke:
            return {p: max(1, int(round(int(self.args.smoke_n) * a))) for p, a in self.shares(name).items()}
        return toggle_proof_counts(self.shares(name), self.floor)

    def n_walkers(self, name):
        return int(sum(self.counts(name).values()))


class Pores(Family):
    """The canonical pores: three shapes at every 0.1 um from 0.1 to 20 um, one pool, impermeable, no field;
    walked by ``simulate_trajectories`` at the pack's own seed (``pack_seed`` of its id)."""
    name = "pores"
    blt_K = 8
    #: The axes each shape confines, in its own frame (the cylinder along z, the plates' normal along x): the
    #: family's packs are certified for waveforms on these (``build_canonical_pores``'s envelope caveat).
    CONFINED = {"sphere": (0, 1, 2), "cylinder": (0, 1), "plane": (0,)}

    def __init__(self, args):
        super().__init__(args)
        self.m = _pores_module()
        dia = tuple(args.diameters) if args.diameters else ((5.0, 20.0) if args.smoke else self.m.DIAMETERS_UM)
        self.rows = self.m.grid_rows(diameters_um=dia)
        self.envelope = family_envelope(self.m.ENVELOPE, self.overrides)
        self.license = self.m.LICENCE
        self.citation = self.m.CITATION

    def substrates(self):
        return {n: dict(id=r["id"], path=r["path"], shape=r["shape"], d_um=r["d_um"]) for n, r in sorted(self.rows.items())}

    def geometry(self, name):
        r = self.rows[name]
        return self.m.geometry(r["shape"], r["d_um"])

    def cost(self, name, backend):
        from dmipy_sim.engine.core import walk_cost
        return walk_cost(self.n_walkers(name), self.m.D0, self.geometry(name), self.T, K=self.K, tiers="all",
                         backend=backend, window_T=self.T)

    def walk(self, name, seed, backend):
        import dmipy_sim as d
        from dmipy_sim.spec import geometry_from_spec
        r = self.rows[name]
        spec = self.m.spec_of(r["shape"], r["d_um"])
        return d.simulate_trajectories(self.n_walkers(name), self.m.D0, geometry_from_spec(spec), T_max=self.T, K=self.K,
                                       seed=int(seed), tiers="all", require_gpu=(backend != "jax"), backend=backend,
                                       window_T=self.T, raw_subset=self.raw_subset, envelope=self.envelope)

    def pack_kw(self, name, walk):
        return dict(K=self.K, blt_temporal_K=min(self.blt_K, self.K), diffusivity=self.m.D0)

    def pilot(self, name, n, seed, backend):
        """``(walk, record)``: a pilot of ``n`` walkers that keeps its block ends, and the pore's walker count from it --
        the variance any refocused waveform of the envelope's class on the pore's confined axes and any contact
        toggle up to its ``rho2`` can give a walker's contribution (:func:`phase_variance_bound`,
        :func:`toggle_variance_bound`), over the floor squared. The pilot is walked as a pack is (the window, the
        band, every tier, the envelope), so a pore whose count is the pilot's packs it."""
        import dmipy_sim as d
        from dmipy_sim.acquisition.scanners import ScannerLimits
        from dmipy_sim.spec import geometry_from_spec
        r = self.rows[name]
        w = d.simulate_trajectories(int(n), self.m.D0, geometry_from_spec(self.m.spec_of(r["shape"], r["d_um"])),
                                    T_max=self.T, K=self.K, seed=int(seed), tiers="all", require_gpu=(backend != "jax"),
                                    backend=backend, window_T=self.T, raw_subset=int(n), envelope=self.envelope)
        G_max = float(self.envelope.get("G_max") or ScannerLimits.of(self.envelope["scanner"]).G_max)
        axes = self.CONFINED[r["shape"]] if self.args.pilot_axes == "confined" else (0, 1, 2)
        ph = phase_variance_bound(w.position_coeffs, self.T, G_max=G_max, axes=axes)
        V, where = toggle_variance_bound(ph["var_phi"], w.raw_boundary, D=self.m.D0,
                                         rho2_max=max(self.envelope.get("rho2_list") or [0.0]))
        n_pore = pilot_count(V, self.floor, self.args.n_min)
        free = [a for a in (0, 1, 2) if a not in axes]
        return w, dict(phase=ph, V=V, toggle_at=where, n=n_pore, floor=self.floor, n_min=int(self.args.n_min),
                       sub_steps=int(w.sub_steps), certified_axes=list(axes), free_axes=free,
                       certifies=(f"the standard error of any refocused waveform of {self.envelope['scanner']} on the "
                                  f"pore's confined axes {list(axes)} under any toggle up to rho2 "
                                  f"{max(self.envelope.get('rho2_list') or [0.0]):g} m/s is at most {self.floor:g}"
                                  + (f"; along its free axes {free} the attenuation is free diffusion's, analytic, "
                                     "and not what the count is set for" if free else "")))


class Slab(Family):
    """The closed permeable slab: [0, L] split at L/2 by a membrane of permeability kappa, reflecting x faces, every
    walker starting in compartment A = [0, L/2]. Compartment B's pool is magnetic (``chi_iso`` = -0.1 ppm, which
    the envelope's ``chi_max`` is read from) and its field basis is the indicator of B, a layer at ``chi B0`` the
    walkers exchange in and out of, read along the path."""
    name = "slab"
    blt_K = 256
    L = 4.0e-6
    D0 = 2.0e-9
    KAPPAS = (1.0e-5, 1.0e-4)
    CHI = -1.0e-7                      #: compartment B's chi_iso: the off-resonant layer
    FIELD_NODES = 400
    FIELD_ROUTE = "grid"               #: the layer's basis is a raster read at the walker's position

    def __init__(self, args):
        super().__init__(args)
        self.envelope = family_envelope(None, self.overrides, max_refocus_pulses=8)

    def substrates(self):
        kap = self.KAPPAS if not self.args.kappas else tuple(self.args.kappas)
        return {f"slab-L{self.L * 1e6:.1f}um-k{k * 1e6:g}um_s": dict(
            id=f"permeable-slab/L{self.L * 1e6:.1f}um-kappa{k * 1e6:g}um_s",
            path=f"packs/L{self.L * 1e6:.1f}um_kappa{k * 1e6:g}um_s.rpk", kappa=float(k)) for k in kap}

    def geometry(self, name):
        """The slab as a two-voxel label volume (``--slab-geometry label``, the default: A the first voxel, B the
        second, the x faces reflecting, y and z periodic, every face between them the membrane; an engine class the
        CUDA backend walks) or as ``PermeableSlab1D`` (``analytic``). Both are the same closed slab under the same
        Powles rule, ``p = min(1, 2 kappa d_perp / D)`` per encounter, and resolve the same step."""
        kappa = self.substrates()[name]["kappa"]
        if self.args.slab_geometry == "analytic":
            import dmipy_sim as d
            return d.PermeableSlab1D(self.L, kappa)
        from dmipy_sim.geometry.label_volume import LabelVolume
        return LabelVolume(np.array([1, 0], np.uint8).reshape(2, 1, 1), (self.L / 2, self.L, self.L),
                           periodic=(False, True, True), pools={0: "extra", 1: "intra"}, pool="intra",
                           permeability=kappa)

    def spec(self, name):
        """The slab's spec with compartment B (pool ``extra``) magnetic and the field tier declared: the field rule
        (some pool magnetic <=> ``"field"`` in the tiers <=> C3 stored) the pack is held to."""
        import dataclasses
        from dmipy_sim.spec.substrate import Susceptibility
        s = self.geometry(name).spec
        return s.replace(susceptibility={"extra": Susceptibility(chi_iso=self.CHI)},
                         validity=dataclasses.replace(s.validity, tiers=list(s.validity.tiers) + ["field"]))

    def field_grid(self):
        """A geometry-only basis on a 1-D grid: ``iso_local`` 1 in compartment B and 0 in A, no ``iso_P``."""
        from dmipy_sim.fields.susceptibility_field import FieldGrid
        nx = self.FIELD_NODES
        vs = np.array([self.L / nx, 1.0, 1.0])
        x = (np.arange(nx) + 0.5) * vs[0]
        iso_local = (x > self.L / 2).astype(np.float32).reshape(nx, 1, 1)
        basis = dict(iso_local=iso_local, iso_P=np.zeros((6, nx, 1, 1), np.float32), shape=(nx, 1, 1), voxel_size=vs)
        return FieldGrid(basis, np.array([0.0, -0.5, -0.5]))

    def cost(self, name, backend):
        from dmipy_sim.engine.core import walk_cost
        return walk_cost(self.n_walkers(name), self.D0, self.geometry(name), self.T, K=self.K, tiers="all",
                         backend=backend, window_T=self.T, events_capacity=64)

    def walk(self, name, seed, backend):
        import dmipy_sim as d
        return d.simulate_trajectories(self.n_walkers(name), self.D0, self.spec(name), T_max=self.T, K=self.K,
                                       seed=int(seed), tiers="all", require_gpu=(backend != "jax"), backend=backend,
                                       window_T=self.T, raw_subset=self.raw_subset, field_basis=self.field_grid(),
                                       envelope=self.envelope)

    def pack_kw(self, name, walk):
        return dict(K=self.K, blt_temporal_K=min(self.blt_K, self.K), diffusivity=self.D0,
                    provenance=dict(substrate=type(self.geometry(name)).__name__, L_m=self.L, kappa_m_s=self.substrates()[name]["kappa"],
                                    D0=self.D0, field="iso_local = 1 in compartment B (x > L/2), 0 in A"))


class Cactus(Family):
    """The CACTUS bundle: every strand an inner and an outer mesh in a periodic cell, three pools (intra, myelin
    water frozen in its sheath, extra), the myelin sheath's susceptibility field rasterised and read along the
    path. Walked by ``walk_spec`` pool by pool, seeded in proportion to the water (``thin``), which is the
    toggle-proof count at equal cost per walker."""
    name = "cactus"
    floor = 3e-3
    blt_K = 256
    license = "MIT"                                    #: the CACTUS source's
    pack_license = "CC-BY-NC-SA-4.0 OR LicenseRef-Commercial"
    bundle = True
    citation = "Villarreal-Haro et al. 2023, CACTUS (https://github.com/Juanitovh/CACTUS)"

    def __init__(self, args):
        super().__init__(args)
        self.envelope = family_envelope(None, self.overrides, max_refocus_pulses=8)
        self.run_dir = args.cactus_run or (_smoke_cactus_run(args) if args.smoke else None)
        if self.run_dir is None:
            raise SystemExit("cactus: --cactus-run DIR (a CACTUS run directory: optimized_final.txt and "
                             "meshes/simulations/strand_*_{inner,outer}_erode_*.ply)")
        self._spec = None

    def spec(self):
        if self._spec is None:
            from dmipy_sim.spec import cactus_spec
            self._spec = cactus_spec(self.run_dir, side_um=(self.args.smoke_side_um if self.args.smoke else None))
        return self._spec

    def substrates(self):
        base = os.path.basename(os.path.normpath(self.run_dir)).replace("cactus_", "").replace("_", "")
        tag = f"{base}-{self.T * 1e3:g}ms"
        return {f"cactus-{tag}": dict(id=f"cactus/{tag}", path=f"packs/{tag.replace('-', '_')}.rpk")}

    def shares(self, name):
        """Water share per seeded pool: its volume fraction (a probe of the cell) times its water fraction."""
        from dmipy_sim.spec.walk import WalkContext
        spec = self.spec()
        tests = WalkContext(spec).tests
        probe = np.random.default_rng(99).uniform(tests.lo, tests.hi, (int(self.args.n_probe), 3))
        return {tests.pools[pid].name: float(tests.member(pid)(probe).mean()) * float(tests.wf[pid]) for pid in tests.seeded}

    #: How a mesh sheath's field reaches the walk: a raster (``FieldGrid``) read at the walker's position. The CUDA
    #: backend samples a strand basis only (``CudaBackend.refuses``), so a CUDA walk of this family refuses its field.
    FIELD_ROUTE = "grid"

    def pool_costs(self, name, backend):
        """``{pool: WalkCost}`` of each diffusing pool's walk, the geometry ``walk_spec`` walks it in (inside its
        inner walls or outside its outer ones, the cell's faces as the domain says), without the field read."""
        from dmipy_sim.engine.core import walk_cost
        from dmipy_sim.spec.walk import WalkContext
        spec = self.spec()
        t = WalkContext(spec).tests
        counts, feature, out = self.counts(name), float(spec.validity.smallest_feature), {}
        for pid in t.seeded:
            pool = t.pools[pid]
            if not pool.D:
                continue                                        # a frozen pool takes no step
            geo = (t.boundary(t.inside_w[pid]).geometry("intra", t.lo, t.hi, t.periodic, t.reflect, feature) if t.inside_w[pid]
                   else t.boundary(t.outside_w[pid]).geometry("extra", t.lo, t.hi, t.periodic, t.reflect, feature))
            out[pool.name] = walk_cost(max(1, counts.get(pool.name, 1)), float(pool.D), geo, self.T, K=self.K,
                                       tiers="all", backend=backend, window_T=self.T)
        return out

    def cost(self, name, backend):
        """The pools' walks as one cost (walker-steps summed, the finest step's resolution); the field's read is
        priced by the walk itself."""
        costs = self.pool_costs(name, backend)
        first = min(costs.values(), key=lambda c: c.dt_sim)
        import dataclasses
        return dataclasses.replace(first, walker_steps=int(sum(c.walker_steps for c in costs.values())),
                                   n_walkers=int(sum(c.n_walkers for c in costs.values())))

    def walk(self, name, seed, backend):
        from dmipy_sim.spec import walk_spec
        kw = dict(field_res=float(self.args.field_res) * 1e-6) if self.args.field_res else {}
        return walk_spec(self.spec(), self.n_walkers(name), self.T, scanner=self.envelope["scanner"], seed=int(seed),
                         K=self.K, window_T=self.T, envelope=self.envelope, backend=backend,
                         require_gpu=(backend != "jax"), **kw)

    def pack_kw(self, name, walk):
        return dict(K=self.K, blt_temporal_K=min(self.blt_K, self.K), pack_license=self.pack_license,
                    provenance=dict(source=os.path.abspath(self.run_dir)))


FAMILIES = {"pores": Pores, "slab": Slab, "cactus": Cactus}


def _smoke_cactus_run(args):
    """A CACTUS-shaped run directory of two capped myelinated tubes along z overrunning a small periodic cell:
    the same producer (``cactus_spec``), walk and pack as the real bundle, at a CPU's size."""
    import trimesh
    out = os.path.join(os.path.abspath(args.out or "."), "_smoke_cactus_run")
    sim = os.path.join(out, "meshes", "simulations")
    if os.path.isdir(sim) and len(os.listdir(sim)) == 4:
        return out
    os.makedirs(sim, exist_ok=True)
    side = float(args.smoke_side_um)
    for i, (cx, cy) in enumerate(((-0.25 * side, -0.2 * side), (0.25 * side, 0.2 * side))):
        for kind, r in (("inner", 1.0), ("outer", 1.4)):
            m = trimesh.creation.cylinder(radius=r, height=1.6 * side, sections=48)
            m.apply_translation([cx, cy, 0.0])
            m.export(os.path.join(sim, f"strand_{i:05d}_{kind}_erode_0.ply"))
    return out


# ------------------------------------------------------------------ the stages
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _commit(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def _sim_identity():
    """Where the ``dmipy_sim`` that walked is imported from and its checkout's commit (an installed copy shadowing
    the intended checkout is the trap this records), and the CUDA backend's when it is installed."""
    import dmipy_sim
    sim = os.path.dirname(os.path.abspath(dmipy_sim.__file__))
    out = dict(path=sim, commit=_commit(sim))
    try:
        import dmipy_sim_cuda
        cu = os.path.dirname(os.path.abspath(dmipy_sim_cuda.__file__))
        out["cuda"] = dict(path=cu, commit=_commit(cu))
    except ImportError:
        out["cuda"] = None
    return out


def _device():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        return out or "cpu"
    except Exception:
        return "cpu"


def _shard(names, shard):
    if not shard:
        return names
    i, n = (int(x) for x in shard.split("/"))
    return names[i::n]


def _plan_backend(backend):
    """The backend a plan asks: the installed one of that name, else for ``"cuda"`` the dmipy-sim-cuda checkout's
    :class:`CudaBackend` on the path (its refusals need no built library), so a CPU host states what the card
    would refuse."""
    from dmipy_sim.engine.backends import resolve
    try:
        resolve(backend)
        return backend
    except ValueError:
        if backend != "cuda":
            raise
        from dmipy_sim_cuda.backend import CudaBackend
        return CudaBackend()


def _backend_check(fam, name, backend):
    """``(accepted, cost or reason)`` for walking ``name`` on ``backend``: the walk's own resolution
    (:func:`~dmipy_sim.engine.core.walk_cost`), refused where the walk refuses."""
    backend = _plan_backend(backend)
    try:
        c = fam.cost(name, backend)
        return True, c
    except ValueError as e:
        return False, str(e)


def stage_plan(fam, args):
    from dmipy_sim.replay.reference import pack_seed
    rows, tot_steps, tot_n = [], 0, 0
    fam.planned = {}
    for name, sub in _shard(list(fam.substrates().items()), args.shard):
        pilot = None
        if int(args.pilot) > 0:
            if not hasattr(fam, "pilot"):
                raise SystemExit(f"{fam.name}: no pilot bound for this family (one pool, no field: the pores)")
            held = os.path.join(os.path.abspath(args.out), "pilots", f"{name}.json") if args.out else None
            if held and os.path.exists(held):               # a plan resumed: the pilot is not walked again
                pilot = json.load(open(held))
            else:
                t0 = time.time()
                walk, pilot = fam.pilot(name, int(args.pilot), pack_seed(sub["id"]), args.pilot_backend or args.backend)
                pilot["seconds"] = round(time.time() - t0, 1)
                pilot["pilot_is_pack"] = bool(pilot["n"] <= int(args.pilot))
                if pilot["pilot_is_pack"]:                 # the count is the pilot's: the pilot is the pack, walked once
                    pilot["n"] = int(args.pilot)
                fam.pilots = dict(getattr(fam, "pilots", {}), **{name: pilot})
                fam.planned[name] = {p: pilot["n"] for p in fam.shares(name)}
                if pilot["pilot_is_pack"] and args.out:
                    pack_and_receipt(fam, name, sub, walk, seed=pack_seed(sub["id"]), seed_is="pack_seed(id)",
                                     t_walk=pilot["seconds"], args=args, out=os.path.abspath(args.out),
                                     walked_as="pilot")
                del walk
                if held:
                    os.makedirs(os.path.dirname(held), exist_ok=True)
                    with open(held + ".part", "w") as fh:
                        json.dump(pilot, fh, indent=1, default=str)
                    os.replace(held + ".part", held)
            fam.pilots = dict(getattr(fam, "pilots", {}), **{name: pilot})
            fam.planned[name] = {p: pilot["n"] for p in fam.shares(name)}
        counts = fam.counts(name)
        n = int(sum(counts.values()))
        ok, c = _backend_check(fam, name, args.backend)
        refusal = None if ok else c
        if not ok and args.backend != "jax":         # the resolution is the walk's whatever executes it
            ok_j, c = _backend_check(fam, name, "jax")
            c = c if ok_j else None
        field_refusal = None
        if getattr(fam, "FIELD_ROUTE", None) == "grid" and args.backend == "cuda":
            field_refusal = "the field is read from a FieldGrid, and the CUDA backend samples a StrandFieldBasis only"
        row = dict(name=name, id=sub["id"], n_walkers=n, counts=counts,
                   se_bound=toggle_proof_bound(fam.shares(name), counts), backend=args.backend,
                   accepted=bool(ok) and field_refusal is None, refusal=refusal, field_refusal=field_refusal,
                   pilot=pilot)
        if c is not None and not isinstance(c, str):
            row.update(sub_steps=int(c.sub_steps), dt_sim_s=float(c.dt_sim), n_blocks=int(c.n_blocks),
                       walker_steps=int(c.walker_steps), bytes_per_walker=int(c.bytes_per_walker),
                       binding_rule=str(c.binding_rule))
            tot_steps += int(c.walker_steps)
        tot_n += n
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "counts"}, default=str), flush=True)
    rate = float(args.rate)
    summary = dict(family=fam.name, n_substrates=len(rows), n_walkers=tot_n, walker_steps=tot_steps,
                   gpu_hours_at_rate=tot_steps / rate / 3600.0, rate_walker_steps_per_s=rate,
                   floor=fam.floor, T=fam.T, K=fam.K, blt_K=fam.blt_K, envelope=fam.envelope,
                   walker_rule=("pilot bound: n = max(n_min, V / f^2)" if int(args.pilot) > 0
                                else "toggle-proof: n_p = a_p / f^2"), pilot_n=int(args.pilot), n_min=int(args.n_min),
                   refused=[r["name"] for r in rows if not r["accepted"]])
    print(json.dumps(summary, default=str), flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"plan_{fam.name}.json"), "w") as fh:
            json.dump(dict(summary=summary, rows=rows), fh, indent=1, default=str)
    return summary


def load_plan(fam, path):
    """A plan's per-substrate counts and pilot bounds onto ``fam``: what ``walk --plan`` walks."""
    rows = json.load(open(path))["rows"]
    fam.planned = {r["name"]: r["counts"] for r in rows}
    fam.pilots = {r["name"]: r.get("pilot") for r in rows}


#: The seed of a pore's pack when the pilot set a count above its own: a stream of its own, so that the walkers the
#: pack averages are independent of the pilot's whose variance chose their number.
AFTER_PILOT = "/after-pilot"


def seed_rule(fam, name, sub):
    """``(seed, rule)`` a substrate's pack is walked at: ``pack_seed(id)`` (the pilot's own seed, so a pore whose count
    is the pilot's IS its pilot), or ``pack_seed(id + AFTER_PILOT)`` for a pore whose pilot set a larger count."""
    from dmipy_sim.replay.reference import pack_seed
    pilot = getattr(fam, "pilots", {}).get(name)
    if pilot and not pilot.get("pilot_is_pack"):
        return pack_seed(sub["id"] + AFTER_PILOT), f"pack_seed(id + {AFTER_PILOT!r}): fresh walkers, independent of the pilot"
    return pack_seed(sub["id"]), "pack_seed(id)"


def keep_ids(spec):
    """The substrate ids ``--keep-walk-ids`` names: the lines or commas of a file, or the comma-separated list itself
    (``None`` when unset: every substrate keeps its walk)."""
    if not spec:
        return None
    text = open(spec).read() if os.path.isfile(spec) else spec
    return {x.strip() for x in text.replace(",", "\n").splitlines() if x.strip()}


def keep_walk(name, sub, walk, args):
    """``{path, sha256, bytes}`` of ``walk`` saved to ``<--keep-walk>/<pack id>.safetensors``, or ``None`` where the
    option is off or ``--keep-walk-ids`` leaves this substrate out (named by its pack id or its name)."""
    if not args.keep_walk:
        return None
    ids = keep_ids(args.keep_walk_ids)
    if ids is not None and sub["id"] not in ids and name not in ids:
        return None
    path = os.path.join(os.path.abspath(args.keep_walk), sub["id"] + ".safetensors")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    walk.save(path + ".part")
    os.replace(path + ".part", path)
    return dict(path=path, sha256=_sha256(path), bytes=os.path.getsize(path))


def pack_and_receipt(fam, name, sub, walk, *, seed, seed_is, t_walk, args, out, walked_as):
    """``walk`` packed on the family's envelope into ``out`` with its receipt: what the walk stage and a pilot that
    IS its pack (``walked_as="pilot"``) both write."""
    from dmipy_sim.replay.bank import build_replay_pack
    path = os.path.join(out, sub["path"])
    receipt_path = os.path.join(out, "receipts", f"{name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.makedirs(os.path.dirname(receipt_path), exist_ok=True)
    kept = keep_walk(name, sub, walk, args)
    fidelity = "measured" if fam.raw_subset > 0 else "derived"
    t1 = time.time()
    kw = fam.pack_kw(name, walk)
    prov = dict(kw.pop("provenance", {}) or {})
    pilot = getattr(fam, "pilots", {}).get(name)
    rule = ("pilot bound: SE^2 <= V / n for any refocused waveform of the class on the confined axes and any toggle, "
            "n = max(n_min, V / f^2)" if pilot else
            "toggle-proof: SE^2 <= sum_p a_p^2 / n_p for any toggle, n_p = a_p / f^2 at equal cost")
    prov.update(campaign=args.campaign, walked_as=walked_as, seed_rule=seed_is, walker_rule=dict(
        rule=rule, floor=fam.floor, counts=fam.counts(name), smoke=bool(args.smoke),
        **({} if not pilot else dict(V=pilot["V"], var_phi=pilot["phase"]["var_phi"],
                                     certified_axes=pilot["certified_axes"], free_axes=pilot["free_axes"],
                                     n_pilot=pilot["phase"]["n_pilot"], n_min=pilot["n_min"]))))
    pack = build_replay_pack(walk, id=sub["id"], license=fam.license, citation=fam.citation,
                             envelope=fam.envelope, fidelity=fidelity, provenance=prov,
                             out_path=path + ".part", **kw)
    os.replace(path + ".part", path)
    t_pack = time.time() - t1
    fid = pack.meta.get("fidelity") or {}
    work = dict(getattr(walk, "work", None) or {})
    receipt = dict(
        name=name, id=sub["id"], path=sub["path"], sha256=_sha256(path), bytes=os.path.getsize(path),
        n_walkers=int(pack.n_walkers), counts=fam.counts(name), seed=int(seed), seed_rule=seed_is,
        walked_as=walked_as, backend=args.pilot_backend if walked_as == "pilot" and args.pilot_backend else args.backend,
        device=_device(), walk_seconds=round(t_walk, 2), pack_seconds=round(t_pack, 2),
        sub_steps=int(walk.sub_steps), illegal_crossings=int(walk.illegal_crossings or 0),
        exhausted_steps=int(work.get("exhausted_steps") or 0), block=dict(walk.block or {}),
        K=int(fam.K), blt_K=int(fam.blt_K), T=float(fam.T), floor_target=float(fam.floor),
        raw_subset=int(fam.raw_subset), fidelity=fidelity, measured_by=fid.get("measured_by"),
        measured_on=fid.get("measured_on"),
        kept_walk=kept,
        err_max=fid.get("err_max"), floor_max=fid.get("floor_max"), terms=fid.get("terms"),
        envelope=fam.envelope, stated=stated(pack), missing_statements=missing_statements(pack),
        pilot=pilot, dmipy_sim=_sim_identity(), smoke=bool(args.smoke))
    with open(receipt_path + ".part", "w") as fh:
        json.dump(receipt, fh, indent=1, default=str)
    os.replace(receipt_path + ".part", receipt_path)
    print(f"{name}: {receipt['n_walkers']} walkers ({walked_as}), walk {t_walk:.1f} s, pack {t_pack:.1f} s, "
          f"{receipt['bytes'] / 1e6:.2f} MB, err_max {receipt['err_max']}, floor {receipt['floor_max']}, "
          f"missing {receipt['missing_statements']}", flush=True)
    return receipt


def stage_walk(fam, args):
    if args.plan:
        load_plan(fam, args.plan)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "receipts"), exist_ok=True)
    done, failed = [], []
    for name, sub in _shard(list(fam.substrates().items()), args.shard):
        path = os.path.join(out, sub["path"])
        receipt_path = os.path.join(out, "receipts", f"{name}.json")
        if os.path.exists(path) and os.path.exists(receipt_path):
            print(f"{name}: pack and receipt present, not walked again", flush=True)
            continue
        seed, seed_is = seed_rule(fam, name, sub)
        t0 = time.time()
        try:
            walk = fam.walk(name, seed, args.backend)
            pack_and_receipt(fam, name, sub, walk, seed=seed, seed_is=seed_is, t_walk=time.time() - t0, args=args,
                             out=out, walked_as="walk")
        except Exception as e:
            failed.append(name)
            print(f"{name}: FAILED {type(e).__name__}: {e}", flush=True)
            if args.stop_on_error:
                raise
            continue
        done.append(name)
    print(json.dumps(dict(family=fam.name, walked=len(done), failed=failed)), flush=True)
    if failed:
        raise SystemExit(1)


def stage_check(args):
    """Every pack under ``DIR``: it loads, states the campaign's envelope (refused when ``--require-stated``), its
    certificate against its floor, and the replay of a PGSE at b = 1000 s/mm^2 across x is a signal in (0, 1]."""
    from dmipy_sim import pgse
    from dmipy_sim.replay.replay import ReplayPack
    rows, bad = [], []
    for dirpath, _, files in os.walk(os.path.abspath(args.out)):
        for f in sorted(files):
            if not f.endswith(".rpk"):
                continue
            path = os.path.join(dirpath, f)
            pk = ReplayPack.load(path)
            T = (int(pk.n_t) - 1) * float(pk.dt)
            delta, Delta = 0.1 * T, 0.5 * T
            seq = pgse([[1.0, 0.0, 0.0]], delta, Delta, bvalues=[1e9], n_t=400, slew_rate=np.inf)
            S = float(np.abs(np.asarray(pk.replay(seq))).ravel()[0])
            fid = pk.meta.get("fidelity") or {}
            miss = missing_statements(pk)
            row = dict(pack=os.path.relpath(path, args.out), n_walkers=int(pk.n_walkers), S_pgse_b1000=S,
                       err_max=fid.get("err_max"), floor_max=fid.get("floor_max"), missing_statements=miss,
                       stated=stated(pk))
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
            if not (0.0 < S <= 1.0 + 1e-6) or (args.require_stated and miss):
                bad.append(row["pack"])
    print(json.dumps(dict(checked=len(rows), bad=bad)), flush=True)
    if bad or not rows:
        raise SystemExit(1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stage", choices=("plan", "walk", "check"))
    p.add_argument("family", nargs="?", choices=tuple(FAMILIES))
    p.add_argument("--out", default=None, help="the family's output directory (packs/, receipts/)")
    p.add_argument("--backend", default="jax", help="'cuda' for production, 'jax' for a CPU smoke")
    p.add_argument("--envelope", default=None, help="a JSON file of envelope keys over the family's (scanner, "
                                                    "rho2_list, B0_list, chi_max, ...)")
    p.add_argument("--floor", type=float, default=None, help="the toggle-proof standard-error floor f")
    p.add_argument("--T", type=float, default=None, help="the walk's window (s); 0.1 by default")
    p.add_argument("--K", type=int, default=None, help="bands per window; 256 per 100 ms by default")
    p.add_argument("--shard", default=None, help="i/n: every n-th substrate from the i-th")
    p.add_argument("--campaign", default="replayable-mc-rewalk-2026-10")
    p.add_argument("--diameters", type=float, nargs="+", default=None, help="pores: a sub-grid (um)")
    p.add_argument("--kappas", type=float, nargs="+", default=None, help="slab: permeabilities (m/s)")
    p.add_argument("--slab-geometry", choices=("label", "analytic"), default="label",
                   help="slab: a two-voxel LabelVolume (CUDA-walked) or PermeableSlab1D (JAX only)")
    p.add_argument("--cactus-run", default=None, help="cactus: the CACTUS run directory")
    p.add_argument("--field-res", type=float, default=None, help="cactus: the field raster's node spacing (um)")
    p.add_argument("--n-probe", type=int, default=200_000, help="cactus: probe points for the pools' water shares")
    p.add_argument("--rate", type=float, default=1.0e9, help="plan: walker-steps per second the GPU-hours assume")
    p.add_argument("--pilot", type=int, default=0, help="plan: walkers of a per-substrate pilot whose variance bound "
                                                        "sets the count (pores; production: n_min); 0 the toggle-proof a / f^2")
    p.add_argument("--n-min", type=int, default=8000, help="plan: the fewest walkers a pilot bound sets (the "
                                                            "production pilot walks this many, and a pore at it packs its pilot)")
    p.add_argument("--pilot-backend", default=None, help="plan: the backend the pilot walks on (--backend's by default)")
    p.add_argument("--pilot-axes", choices=("confined", "all"), default="confined",
                   help="plan: the gradient axes the pilot bound covers: the pore's confined axes, or all three")
    p.add_argument("--plan", default=None, help="walk: a plan file whose per-substrate counts are walked")
    p.add_argument("--smoke", action="store_true", help="a few substrates, a few hundred walkers: the CPU check")
    p.add_argument("--smoke-n", type=int, default=300)
    p.add_argument("--smoke-side-um", type=float, default=6.0)
    p.add_argument("--require-stated", action="store_true", help="check: refuse a pack that does not state the envelope")
    p.add_argument("--allow-jax", action="store_true", help="walk: a production walk on the JAX kernels")
    p.add_argument("--keep-walk", default=None, help="walk: a directory the float coefficient walk of each substrate is "
                                                      "saved to, as <pack id>.safetensors (path and sha256 in the receipt)")
    p.add_argument("--keep-walk-ids", default=None, help="walk: a file (lines or commas) or a comma-separated list of "
                                                          "the pack ids (or names) that keep their walk; all by default")
    p.add_argument("--raw-subset", type=int, default=0, help="walk: walkers whose block ends, contact and field reads "
                                                              "are kept; above 0 the pack is built fidelity='measured' "
                                                              "(not for the cactus bundle)")
    p.add_argument("--stop-on-error", action="store_true")
    a = p.parse_args(argv)
    if a.keep_walk_ids and not a.keep_walk:
        p.error("--keep-walk-ids needs --keep-walk")
    if a.raw_subset < 0:
        p.error("--raw-subset is a count of walkers, at least 0")
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    if a.stage == "check":
        if not a.out:
            p.error("check needs --out")
        return stage_check(a)
    if a.family is None:
        p.error(f"{a.stage} needs a family")
    fam = FAMILIES[a.family](a)
    if a.stage == "plan":
        return stage_plan(fam, a)
    if not a.out:
        p.error("walk needs --out")
    if not a.smoke and a.backend == "jax" and not a.allow_jax:
        p.error("a production walk runs on --backend cuda; --smoke for the CPU check, --allow-jax for a family the "
                "CUDA backend refuses and the campaign walks on the JAX kernels (its receipts say so)")
    return stage_walk(fam, a)


if __name__ == "__main__":
    main()
