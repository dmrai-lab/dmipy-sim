"""The thread cap for the LAPACK calls a known BLAS build hangs on (dmrai-lab/dmipy-sim#564).

scipy-openblas 0.3.28 as numpy's wheels ship it on aarch64 (``USE64BITINT DYNAMIC_ARCH NO_AFFINITY neoversen1
MAX_THREADS=64``; the runtime core is reported as Neoverse-V1 on a 72-core GH200) never returns from its
multithreaded LAPACK drivers: an SVD of a 1455 x 1002 matrix, a symmetric eigensolve of its 1002 x 1002 Gram
and a column-pivoted QR of it run in well under a second on one thread and do not finish in a minute on 64 (at
16 the eigensolve took 17 s a call). A guarded SVD of that matrix at 64 threads returns in 0.3 s. :func:`lapack_threads` caps such a call at :data:`SAFE_THREADS` when the BLAS loaded in this process is
that build, read at runtime from threadpoolctl, and is inert on every other build: it changes nothing there, so
the numbers are the ones the call gives unguarded.
"""
import contextlib

HANG_SIGNATURE = dict(internal_api="openblas", version="0.3.28", architectures=("neoversev1", "neoversen1"))
"""The build of #564: OpenBLAS 0.3.28 on a Neoverse core (as threadpoolctl reports the runtime core)."""

SAFE_THREADS = 8
"""The threads a guarded call keeps: the stall begins at 16 threads (#564), and 8 returns at once."""


def _affected():
    """The loaded BLAS pools that are #564's build and run more threads than :data:`SAFE_THREADS`."""
    from threadpoolctl import threadpool_info
    sig = HANG_SIGNATURE
    return [p for p in threadpool_info()
            if p.get("user_api") == "blas" and p.get("internal_api") == sig["internal_api"]
            and p.get("version") == sig["version"] and str(p.get("architecture", "")).lower() in sig["architectures"]
            and int(p.get("num_threads") or 0) > SAFE_THREADS]


def lapack_threads():
    """A context for ONE LAPACK call: the BLAS capped at :data:`SAFE_THREADS` threads when this process loaded
    #564's build with more than that, a null context otherwise."""
    if not _affected():
        return contextlib.nullcontext()
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=SAFE_THREADS, user_api="blas")
