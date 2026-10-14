"""dmipy-sim's one CPU thread budget: BLAS, torch and XLA's CPU intra-op pool, all at one
process-wide cap (``DMIPY_SIM_CPU_THREADS``, default ``min(8, os.cpu_count())``), so a
many-core box does not hand every pool one thread per core (dmrai-lab/dmipy-sim#700: an
uncapped 72-core run turned a 23 s pose/codec call into 300-2200 s, OpenBLAS alone taking
one thread per core and spin-waiting on a path of many small LAPACK calls).

:func:`cap_cpu_threads` is called exactly once, as the first statement of the package import
(:mod:`dmipy_sim`), before any submodule of this package is imported -- the same window
``engine._gpu_config.apply_gpu_mem_cap`` uses for the GPU memory cap, and for the same reason:
JAX reads ``XLA_FLAGS`` once, when its CPU backend first initialises, and never again, so the
flag must land before the package's first ``import jax`` (inside ``engine.core``). This module
imports :mod:`numpy` and :mod:`scipy.linalg` at its own top level for the opposite reason --
:mod:`threadpoolctl` only sees BLAS libraries that are already loaded (a limit applied before
a library loads caps nothing, since it is not retroactive) -- so by the time
:func:`cap_cpu_threads` runs, both OpenBLAS copies this process can load (numpy's and scipy's
own, distinct shared objects) are already in hand and get capped for real, while jax is still
untouched. torch is different again: it is an optional, lazily-imported backend
(dmrai-lab/dmipy-sim#603) that no CPU pose/codec path needs by default, so it is never
force-imported here -- :func:`cap_cpu_threads` caps it only when some caller already imported
it (checked through ``sys.modules``), which costs nothing and is called again, cheaply
(idempotent past the first successful application), by any code path that imports torch
itself. None of this adds an import or a check to the GPU route beyond this one cheap pass.

The #564 one-build guard is a floor under this cap: :func:`cpu_threads` can be
set above :data:`dmipy_sim._blas.SAFE_THREADS`, but the build #564 found hanging past it never
runs above it regardless of what was asked, so :func:`_cap_blas` clamps to the lower of the
two only on that build and passes the request through unchanged on every other.
"""
import os
import sys
import warnings

import numpy as _np          # noqa: F401  (forces numpy's BLAS to load before the first cap)
import scipy.linalg as _sla  # noqa: F401  (scipy ships its OWN separate OpenBLAS; force that too)

_LIMITER = None       # the threadpoolctl controller; kept alive at module level (never entered
                       # as a context manager, so nothing ever restores the uncapped pools)
_BLAS_XLA_DONE = False
_TORCH_DONE = False


def cpu_threads():
    """The CPU thread budget: ``DMIPY_SIM_CPU_THREADS``, default ``min(8, os.cpu_count())``.

    Refuses by name a value that is not a positive integer.
    """
    raw = os.environ.get("DMIPY_SIM_CPU_THREADS")
    if raw is None:
        return min(8, os.cpu_count() or 8)
    stripped = raw.strip()
    try:
        n = int(stripped)
    except ValueError:
        raise ValueError(
            f"DMIPY_SIM_CPU_THREADS={raw!r} must be a positive integer.") from None
    if n <= 0 or str(n) != stripped:
        raise ValueError(f"DMIPY_SIM_CPU_THREADS={raw!r} must be a positive integer.")
    return n


def cap_cpu_threads():
    """Apply :func:`cpu_threads` to BLAS, torch and XLA's CPU intra-op pool.

    Idempotent and cheap past the first call (see the module docstring for why it is called
    once from the package import and again wherever a CPU path has just imported torch)."""
    n = cpu_threads()
    global _BLAS_XLA_DONE, _TORCH_DONE
    if not _BLAS_XLA_DONE:
        _cap_blas(n)
        _cap_xla(n)
        _BLAS_XLA_DONE = True
    if not _TORCH_DONE:
        torch = sys.modules.get("torch")
        if torch is not None:
            torch.set_num_threads(n)
            _TORCH_DONE = True


def _cap_blas(n):
    global _LIMITER
    from threadpoolctl import threadpool_limits
    from ._blas import _affected, SAFE_THREADS
    limit = min(n, SAFE_THREADS) if _affected() else n
    _LIMITER = threadpool_limits(limits=limit, user_api="blas")


def _cap_xla(n):
    flag = f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={n}"
    if "jax" in sys.modules:
        if "DMIPY_SIM_CPU_THREADS" in os.environ:          # an explicit budget that cannot reach XLA is said; the default is silent
            warnings.warn(
                "DMIPY_SIM_CPU_THREADS is set but jax was imported before dmipy_sim; XLA's CPU intra-op "
                "thread count is read once when jax's backend initialises, so this process keeps jax's "
                "own default for it (BLAS and torch are capped). Import dmipy_sim before jax to cap XLA too.",
                RuntimeWarning)
        return
    existing = os.environ.get("XLA_FLAGS", "")
    if flag not in existing:
        os.environ["XLA_FLAGS"] = (existing + " " + flag).strip()
