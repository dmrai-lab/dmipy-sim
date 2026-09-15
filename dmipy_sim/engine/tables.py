"""A jitted function whose traced body reads an object's device tables as ARGUMENTS.

A jitted closure that reads a device array embeds it in every executable it lowers, as a constant: JAX copies
the array back to the host at each lowering (``mlir.ir_constant``), XLA keeps another copy inside the program,
and a walk that lowers one program per batch shape, per candidate-list width and per radius class holds hundreds
of copies of the same segment tables (a 20k-walker DiSCo rehearsal: 223 host copies, 3.4 GB in Python and far
more in the runtime, of tables whose one copy is 30 MB). Passing the tables as arguments makes them traced
values: one executable per shape, no copy anywhere, the same arrays handed in at every call.
"""
from __future__ import annotations

import jax


def jit_with_tables(obj, names, fn, **jit_kw):
    """``jax.jit(fn)`` with ``obj``'s attributes ``names`` (device arrays) passed as arguments instead of captured:
    during the trace the attributes are swapped for the tracers of the tables handed in, so ``fn``'s body -- and
    every method of ``obj`` it calls -- reads them as traced values; after the trace they are restored. The
    returned callable takes ``fn``'s own arguments and passes ``obj``'s current tables at every call, so a
    caller does not see the tables at all."""
    names = tuple(names)

    def traced(tables, *args):
        saved = {n: getattr(obj, n) for n in names}
        try:
            for n in names:
                object.__setattr__(obj, n, tables[n])
            return fn(*args)
        finally:
            for n, v in saved.items():
                object.__setattr__(obj, n, v)
    jitted = jax.jit(traced, **jit_kw)

    def call(*args):
        return jitted({n: getattr(obj, n) for n in names}, *args)
    call.jitted = jitted
    return call
