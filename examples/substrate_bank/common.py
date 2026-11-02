"""What the reference-family builders of this directory share: the CC BY 4.0 legal code a source record carries,
the producer walk every pack is built from, and the T2 log-mean a pack's stored contact channel serves.

Each builder binds its own constants (its diffusivity, its bulk T2, its storage windows) and passes them here; none
of these functions holds a family's number.
"""
import os

import numpy as np

#: The licence the Figshare records and this repository's own packs are released under.
CC_BY_4_URL = "https://creativecommons.org/licenses/by/4.0/legalcode.txt"


def cc_by_4_legal_code(cache_dir):
    """The CC BY 4.0 legal code, fetched once into ``cache_dir`` and read from there after: the licence TEXT a
    source record copies verbatim, because a title is not a licence."""
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "cc-by-4.0-legalcode.txt")
    if not os.path.exists(cache):
        req = urllib.request.Request(CC_BY_4_URL, headers={"User-Agent": "dmipy-sim (https://github.com/dmrai-lab/dmipy-sim)"})
        with urllib.request.urlopen(req, timeout=60) as fh, open(cache, "wb") as out:
            out.write(fh.read())
    return open(cache, encoding="utf-8").read()


def walk(spec, n, *, D, T, K, seed, sub_steps=None, window_T=None):
    """The walk a family's pilot runs and its packs are built from: ``n`` walkers at diffusivity ``D`` over ``T``,
    no relaxivity, every tier stored, ``K`` bands per storage window (``window_T``; None is the storage rule's 100 ms
    windows) in the band's own blocks, at the pack's own ``seed`` (``pack_seed`` of its id); ``sub_steps`` is the step
    pilot's coarse count, None the rule's. A :class:`~dmipy_sim.coefficient_walk.CoefficientWalk`."""
    from dmipy_sim.engine.core import _walk_geometry
    from dmipy_sim.spec import geometry_from_spec
    return _walk_geometry(int(n), float(D), geometry_from_spec(spec), T_max=float(T), K=int(K), seed=int(seed),
                          tiers="all", require_gpu=False, walker_batch_size=2000, sub_steps=sub_steps,
                          window_T=None if window_T is None else float(window_T))


def t2_grid(grid):
    """The T2 values (s) a reference record's log-spaced grid names: ``n`` from ``from_s`` to ``to_s``."""
    return np.logspace(np.log10(grid["from_s"]), np.log10(grid["to_s"]), int(grid["n"]))


def decay_and_log_mean(pack, rho2, grid, *, D, T2B, chunk=4000):
    """``(t, S, T2 log-mean, its relative delta-method standard error)`` from the pack's stored C2 channel.

    ``S(t) = <exp((rho2/D) L(t) - t/T2B)>`` on the recorded sampling (``grid["sample_ms"]``), with ``D`` the walk's
    diffusivity and ``T2B`` the bulk T2; the standard error is the log-mean's gradient with respect to the decay it is
    inverted from, projected on each walker's own decay -- an analytic standard error, not a fold spread. The walkers
    are read in chunks of ``chunk``, so no float64 copy of the whole walk is held.
    """
    from examples.validation.talabi_micro_ct_rocks import log_mean_gradient
    ell = pack.contact()
    if ell is None:
        raise ValueError(f"{pack.meta['id']} carries no C2 channel, so it cannot serve a relaxation decay")
    n_w, n_t = ell.shape
    every = max(1, int(round(grid["sample_ms"] * 1e-3 / pack.dt)))
    t = np.arange(n_t)[every::every] * pack.dt

    def per_walker(lo):
        L = np.cumsum(np.asarray(ell[lo:lo + chunk], np.float64), axis=1)[:, every::every]
        return np.exp(float(rho2) / float(D) * L - t / float(T2B))

    acc = np.zeros(len(t))
    for lo in range(0, n_w, chunk):
        acc += per_walker(lo).sum(axis=0)
    S = acc / n_w
    lm, g = log_mean_gradient(t, S, t2_grid(grid))
    proj = np.empty(n_w)
    for lo in range(0, n_w, chunk):
        s = per_walker(lo)
        proj[lo:lo + s.shape[0]] = s @ g
    return t, S, lm, float(np.std(proj, ddof=1) / np.sqrt(n_w) / lm)
