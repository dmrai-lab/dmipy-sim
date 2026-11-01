"""The CPU thread count dmipy-sim owns: BLAS and torch at one cap (dmrai-lab/dmipy-sim#700), and the floor that cap keeps on the one BLAS build that fails multithreaded (#564).

The cap is ONE number, taken from the first of: the ``n`` passed to :func:`cap_cpu_threads`, the environment
variable ``DMIPY_SIM_CPU_THREADS`` (read when the package is imported), :data:`DEFAULT_CPU_THREADS` bounded by
the host's core count. A value that is not a positive integer is refused by name. Uncapped, OpenBLAS takes one
thread per core and spin-waits on a path of many small LAPACK calls: on a 72-core host a 23 s pose response ran
300-2200 s.

``import dmipy_sim`` applies the cap; :func:`cap_cpu_threads` called later re-applies it with a stated count.
torch is never imported here: it is capped when it is already loaded, and again by the code that loads it.
XLA's CPU client has no thread-count setting; its pool follows the process's CPU affinity (``taskset``), which
this module leaves alone.

OpenBLAS 0.3.28, as the scipy-openblas wheels ship it, fails in its multithreaded drivers on two architectures:
on aarch64 (``neoversen1`` build, Neoverse-V1 runtime core, a 72-core GH200) its LAPACK drivers never return (an
SVD of a 1455 x 1002 matrix, a symmetric eigensolve of its Gram and a column-pivoted QR: under a second on one
thread, unfinished after a minute on 64, 17 s a call at 16); on x86-64 (Haswell runtime core; scipy 1.15.3, the
last scipy for Python 3.10) scipy's L-BFGS-B segfaults from 4 threads on (dmipy-design's codesign test, 2 threads
pass). On that version, whatever its architecture, BLAS is held to :data:`SAFE_THREADS` whatever count was asked;
on every other build BLAS gets exactly the count asked.
"""
import os
import sys

import numpy  # noqa: F401  (numpy's BLAS must be loaded before threadpoolctl can cap it)
import scipy.linalg  # noqa: F401  (scipy loads its own OpenBLAS copy)

ENV_VAR = "DMIPY_SIM_CPU_THREADS"
DEFAULT_CPU_THREADS = 8
"""The cap when neither an argument nor :data:`ENV_VAR` states one, bounded by ``os.cpu_count()``."""

FAULTY_BUILD = dict(internal_api="openblas", version="0.3.28")
"""The BLAS build of #564 and of the L-BFGS-B segfault: OpenBLAS 0.3.28, on any architecture."""

SAFE_THREADS = 2
"""The threads BLAS keeps on that build: the x86-64 segfault begins at 4 threads, the aarch64 stall at 16."""

_threads = None          # the cap in force; None until the first application
_torch_applied = None    # the count torch was last set to


def cpu_threads(n=None):
    """The CPU thread cap: ``n`` if given, else ``DMIPY_SIM_CPU_THREADS``, else
    ``min(DEFAULT_CPU_THREADS, os.cpu_count())``. Refuses a value that is not a positive integer."""
    if n is None:
        raw = os.environ.get(ENV_VAR)
        if raw is None:
            return min(DEFAULT_CPU_THREADS, os.cpu_count() or DEFAULT_CPU_THREADS)
        src, val = f"{ENV_VAR}={raw!r}", raw.strip()
        try:
            n = int(val)
        except ValueError:
            n = 0
        if str(n) != val:
            n = 0
    else:
        src, val = f"n={n!r}", n
        if isinstance(n, bool) or not isinstance(n, int):
            n = 0
    if n <= 0:
        raise ValueError(f"{src} must be a positive integer.")
    return n


def _affected():
    """The loaded BLAS pools of :data:`FAULTY_BUILD`, whatever their current thread count: a pool already held to
    :data:`SAFE_THREADS` stays one, so that applying the cap again never raises it back."""
    from threadpoolctl import threadpool_info
    sig = FAULTY_BUILD
    return [p for p in threadpool_info()
            if p.get("user_api") == "blas" and p.get("internal_api") == sig["internal_api"]
            and p.get("version") == sig["version"]]


def _cap_blas(n):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=min(n, SAFE_THREADS) if _affected() else n, user_api="blas")


def _cap_torch(n):
    global _torch_applied
    torch = sys.modules.get("torch")
    if torch is not None and _torch_applied != n:
        torch.set_num_threads(n)
        _torch_applied = n


def cap_cpu_threads(n=None):
    """Cap BLAS and torch (if loaded) at ``cpu_threads(n)`` and return the count.

    ``n`` states the cap; without it the cap is ``DMIPY_SIM_CPU_THREADS`` or the default (:func:`cpu_threads`).
    Called with no argument it is idempotent: it does nothing once a cap is in force, except cap torch
    when torch has been loaded since."""
    global _threads
    if n is None and _threads is not None:
        _cap_torch(_threads)
        return _threads
    count = cpu_threads(n)
    _cap_blas(count)
    _cap_torch(count)
    _threads = count
    return count
