"""GPU device guard + memory management for Monte Carlo runs.

Project rule (see CLAUDE.md): **large MC runs MUST be on GPU — never a silent
CPU fallback.**  A heavy walk that quietly drops to CPU looks like it is working
but is ~100× slower; this module makes that loud.

It also gives you the tools to keep a long interactive/batch session from leaving
the GPU occupied:

- :func:`gpu_available` — is a CUDA/ROCm device visible to JAX?
- :func:`check_gpu` — the guard called at the top of ``simulate``
  (``require_gpu=None`` warns on large CPU runs,
  ``True`` raises, ``False`` opts out — e.g. float64 CPU reference checks).
- :func:`free_gpu_memory` — release device memory held by *this* process.
- :func:`gpu_session` — context manager that frees on exit.
- :func:`list_gpu_processes` — best-effort report of who is holding the GPU
  (for diagnosing *other* processes' zombie buffers; this module never kills).

Walker batching for small GPUs lives in ``core.simulate(walker_batch_size=...)``,
not here — splitting a
run into walker chunks bounds peak device memory to one chunk.
"""
import gc
import warnings

# A CPU run at or above this many walkers is treated as "large" and warned about
# when require_gpu is left at its default (None).
LARGE_RUN_WALKERS = 10_000

_GPU_PLATFORMS = ("gpu", "cuda", "rocm")


def gpu_available() -> bool:
    """True if JAX can see a CUDA/ROCm device on the default backend."""
    try:
        import jax
        return any(getattr(d, "platform", "") in _GPU_PLATFORMS for d in jax.devices())
    except Exception:
        return False


def _devices_str() -> str:
    try:
        import jax
        return repr(jax.devices())
    except Exception as e:  # pragma: no cover
        return f"<jax.devices() failed: {e}>"


def check_gpu(n_walkers: int = None, require_gpu=None, what: str = "MC simulation") -> bool:
    """Guard against a silent CPU fallback for a Monte Carlo run.

    Parameters
    ----------
    n_walkers : int, optional
        Walker count of the run being guarded.  Used only to decide whether a
        CPU run is "large" enough to warn about when ``require_gpu is None``.
    require_gpu : {None, True, False}
        - ``True``  — raise :class:`RuntimeError` if no GPU is visible.
        - ``False`` — never warn or raise (explicit CPU opt-in, e.g. a float64
          reference check that is meant to run on CPU).
        - ``None``  — (default) warn (don't raise) when a *large* run
          (``n_walkers >= LARGE_RUN_WALKERS``) is about to run on CPU.
    what : str
        Label used in the message.

    Returns
    -------
    bool
        Whether the run will execute on a GPU.
    """
    if require_gpu is False:
        return gpu_available()

    on_gpu = gpu_available()
    if on_gpu:
        return True

    msg = (f"{what}: no GPU device visible to JAX (jax.devices()={_devices_str()}); "
           f"this will run on CPU and be far slower. Confirm a CUDA device is "
           f"visible (do not set JAX_PLATFORMS=cpu for heavy runs). Pass "
           f"require_gpu=False to silence this, or require_gpu=True to enforce GPU.")
    if require_gpu is True:
        raise RuntimeError(msg)
    if n_walkers is None or n_walkers >= LARGE_RUN_WALKERS:
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    return False


def free_gpu_memory(aggressive: bool = False, clear_compilation_cache: bool = True) -> int:
    """Release GPU/device memory held by this process.

    By default this is **safe to call while live JAX arrays exist**: it clears
    the XLA compilation cache (often the largest persistent consumer after many
    differently-shaped runs) and runs a GC pass, which frees the device buffers
    of any arrays that are no longer referenced.  Use this between heavy phases
    of a long session, or in a ``finally`` block.

    Parameters
    ----------
    aggressive : bool
        If True, additionally ``.delete()`` *every* live JAX array on every
        backend.  This reclaims the most memory but **invalidates all live JAX
        arrays in the process** — including arrays cached on geometry objects
        (e.g. ``geometry._inner_radii_jax``).  Only use it when you are done
        with the current objects (rebuild geometries afterwards).
    clear_compilation_cache : bool
        Clear the JIT compilation cache (``jax.clear_caches``).  Frees compiled
        executables; the next call recompiles.

    Returns
    -------
    int
        Number of live arrays deleted (0 unless ``aggressive=True``).
    """
    import jax
    if clear_compilation_cache:
        try:
            jax.clear_caches()
        except Exception:
            pass
    deleted = 0
    if aggressive:
        try:
            for arr in jax.live_arrays():
                try:
                    arr.delete()
                    deleted += 1
                except Exception:
                    pass
        except Exception:
            pass
    gc.collect()
    return deleted


class gpu_session:
    """Context manager that frees device memory on exit.

    >>> with gpu_session():
    ...     sig = simulate(...)
    # device buffers released here

    ``aggressive`` is forwarded to :func:`free_gpu_memory`; leave it False unless
    you are tearing down all JAX state (it invalidates live arrays).
    """

    def __init__(self, aggressive: bool = False, clear_compilation_cache: bool = True):
        self.aggressive = aggressive
        self.clear_compilation_cache = clear_compilation_cache

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        free_gpu_memory(aggressive=self.aggressive,
                        clear_compilation_cache=self.clear_compilation_cache)
        return False


def list_gpu_processes():
    """Best-effort list of processes holding GPU memory, via ``nvidia-smi``.

    Returns a list of dicts ``{pid, used_mib, name}`` (empty if nvidia-smi is
    unavailable).  Reporting only — this module never kills processes; to clear
    a *zombie from another run* you must kill that PID yourself (and leave other
    projects' processes alone).
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_memory,process_name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
    except Exception:
        return []
    procs = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            procs.append({"pid": int(parts[0]),
                          "used_mib": int(parts[1]) if parts[1].isdigit() else None,
                          "name": parts[2]})
    return procs
