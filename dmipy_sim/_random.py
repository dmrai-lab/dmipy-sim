"""Seed construction shared by the walk, geometry and acquisition helpers."""

import jax.numpy as jnp
import numpy as np


def seed_key(seed):
    """Return the raw Threefry key of an integer seed in `[0, 2**64)`.

    The two uint32 words are ordered high first, independently of JAX's x64 mode.
    """
    if (not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_, np.timedelta64))
            or not 0 <= int(seed) < 2**64):
        raise ValueError(f"seed must be an integer in [0, 2^64); got {seed!r}")
    high, low = divmod(int(seed), 2**32)
    return jnp.asarray((high, low), dtype=jnp.uint32)
