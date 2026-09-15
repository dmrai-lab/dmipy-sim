"""A jitted program that reads an object's device tables as arguments embeds none of them as constants
(`engine.tables.jit_with_tables`): the same numbers as the closure, no table in the lowered program, the tables
restored on the object after the trace, and a program whose closure captured them held a copy per lowering."""
import numpy as np
import jax
import jax.numpy as jnp

from dmipy_sim.engine.tables import jit_with_tables


class _Thing:
    def __init__(self):
        self._T = jnp.asarray(np.full((4096, 3), 12345.5, np.float32))
        self._U = jnp.asarray(np.arange(4096, dtype=np.int32))

    def value(self, i):
        return self._T[self._U[i]].sum()


def test_the_tables_are_arguments_not_constants():
    t = _Thing(); T0, U0 = t._T, t._U
    f = jit_with_tables(t, ("_T", "_U"), jax.vmap(t.value))
    idx = jnp.asarray([0, 7, 4095])
    np.testing.assert_allclose(np.asarray(f(idx)), 3 * 12345.5, rtol=1e-6)
    assert t._T is T0 and t._U is U0                                     # restored after the trace
    text = f.jitted.lower({"_T": t._T, "_U": t._U}, idx).as_text()
    assert "12345.5" not in text and "dense<[" not in text.replace("dense<[0, 7, 4095]", "")   # no table constant
    g = jax.jit(jax.vmap(t.value))
    assert "12345.5" in g.lower(idx).as_text() or "dense" in g.lower(idx).as_text()          # the closure embeds it
    t._T = jnp.asarray(np.full((4096, 3), 1.0, np.float32))                                   # the same program, new tables
    np.testing.assert_allclose(np.asarray(f(idx)), 3.0, rtol=1e-6)
