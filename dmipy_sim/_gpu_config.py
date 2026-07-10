"""GPU memory configuration — must be applied before JAX initialises.

Reads the ``DMIPY_GPU_MEM_GB`` environment variable (a hard ceiling, in
gigabytes, on the JAX/XLA GPU memory arena) and translates it into the XLA
pre-allocation environment variables that bound device-memory use. Also exposes
:func:`configure` for setting the cap from Python before the first JAX import.

Motivation: JAX/XLA otherwise pre-allocates ~75% of the GPU on first use, which
is fine on a large card but hostile on shared or small hardware. This lets a
user cap usage (e.g. ``DMIPY_GPU_MEM_GB=8``) so dmipy is a good GPU citizen.
"""
import os
import sys
import warnings


def _detect_gpu_total_gb():
    """Total memory of GPU 0 in GB via nvidia-smi (works before JAX init)."""
    try:
        import subprocess
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total',
             '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL, timeout=5)
        return int(out.decode().splitlines()[0].strip()) / 1024.0
    except Exception:
        return None


def apply_gpu_mem_cap():
    """Translate ``DMIPY_GPU_MEM_GB`` into an XLA pre-allocation cap.

    No-op if the variable is unset. Warns (and does nothing) if JAX has already
    been imported, since the cap can only take effect before XLA initialises.
    """
    gb = os.environ.get('DMIPY_GPU_MEM_GB')
    if not gb:
        return
    if 'jax' in sys.modules:
        warnings.warn(
            "DMIPY_GPU_MEM_GB is set but JAX is already imported; the GPU memory "
            "cap cannot take effect. Set it (or call configure()) before importing "
            "jax or dmipy.", RuntimeWarning)
        return
    try:
        gb_f = float(gb)
    except ValueError:
        warnings.warn(
            "DMIPY_GPU_MEM_GB={!r} is not a number; ignoring.".format(gb),
            RuntimeWarning)
        return
    total = _detect_gpu_total_gb() or 16.0
    frac = max(0.01, min(0.98, gb_f / total))
    # Hard ceiling: pre-allocate exactly this fraction and never grow past it.
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'true')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '{:.4f}'.format(frac))


def configure(max_gpu_gb=None):
    """Set runtime GPU limits before JAX initialises.

    Call at the very top of your script, before importing ``jax`` or running any
    dmipy computation::

        import dmipy_sim
        dmipy_sim.configure(max_gpu_gb=8)   # cap the GPU arena at 8 GB

    Parameters
    ----------
    max_gpu_gb : float, optional
        Hard ceiling (in GB) on the JAX/XLA GPU memory arena. ``None`` leaves the
        current behaviour unchanged.
    """
    if max_gpu_gb is not None:
        if 'jax' in sys.modules:
            raise RuntimeError(
                "configure(max_gpu_gb=...) must be called before JAX is imported; "
                "'jax' is already in sys.modules. Set the DMIPY_GPU_MEM_GB env var "
                "before launching Python instead.")
        os.environ['DMIPY_GPU_MEM_GB'] = str(max_gpu_gb)
        apply_gpu_mem_cap()
