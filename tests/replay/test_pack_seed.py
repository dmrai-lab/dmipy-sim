"""One walk seed per pack, derived from the pack's id (``reference.pack_seed``), and the canonical-pores family
that walks every pack at its own."""
import os
import subprocess
import sys

from dmipy_sim.replay.reference import pack_seed
from examples.substrate_bank import build_canonical_pores as B


def test_two_packs_of_equal_walker_count_walk_different_seeds():
    ids = [B.pack_id(shape, d) for shape in ("plane", "cylinder", "sphere") for d in (0.1, 8.5, 20.0)]
    assert len({pack_seed(i) for i in ids}) == len(ids)


def test_the_seed_is_a_pure_function_of_the_id_across_processes():
    pid = B.pack_id("plane", 8.5)
    assert pid == "canonical-pores/plane/d08.50um"
    code = ("from dmipy_sim.replay.reference import pack_seed;"
            "print(pack_seed('canonical-pores/plane/d08.50um'))")
    out = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                          env={**os.environ, "PYTHONHASHSEED": h, "JAX_PLATFORMS": "cpu"}).stdout.strip()
           for h in ("1", "2")}
    assert out == {str(pack_seed(pid))}
    assert pack_seed(pid) == pack_seed(str(pid))


def test_the_seed_is_a_key_jax_takes():
    import jax
    for i in (B.pack_id(s, d) for s in ("plane", "cylinder", "sphere") for d in (0.1, 5.0, 20.0)):
        s = pack_seed(i)
        assert isinstance(s, int) and 0 <= s < 2 ** 32
        jax.random.PRNGKey(s)


def test_the_walk_records_the_seed_it_was_given():
    w = B.walk(B.spec_of("plane", 8.5), 64, n_t=5, seed=pack_seed(B.pack_id("plane", 8.5)))
    assert w.seed == pack_seed(B.pack_id("plane", 8.5))
