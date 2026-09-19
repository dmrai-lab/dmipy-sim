"""Every assembler is findable from the module that defines it.

An assembler that is not in ``__all__`` and not named in the module docstring's list is one the next author
does not find, and the cost of not finding it is a fifth assembler that does the same job. This is the audit's
pattern (#315) applied to the one extension point the cookbook sends people to.
"""
from __future__ import annotations

import inspect

from dmipy_sim.sequences import assemble

# The protocol an assembler must implement; ``n_t_of`` and ``windows`` are the multi-echo extras.
ASSEMBLER_PROTOCOL = ("te_min", "layout")


def assemblers():
    """Every public class in the module implementing the assembler protocol."""
    out = []
    for name, obj in vars(assemble).items():
        if name.startswith("_") or not inspect.isclass(obj):
            continue
        if obj.__module__ != assemble.__name__:
            continue
        if all(hasattr(obj, m) for m in ASSEMBLER_PROTOCOL):
            out.append(name)
    return sorted(out)


def test_there_are_assemblers_to_check():
    found = assemblers()
    assert {"SpinEcho", "StimulatedEcho", "GradientEcho", "EchoTrain", "PreparedEchoTrain"} <= set(found), found


def test_every_assembler_is_exported():
    missing = [n for n in assemblers() if n not in assemble.__all__]
    assert not missing, f"assemblers missing from __all__: {missing}"


def test_every_assembler_is_named_in_the_module_docstring():
    doc = assemble.__doc__ or ""
    missing = [n for n in assemblers() if n not in doc]
    assert not missing, (f"assemblers the module docstring does not list: {missing}. The docstring is where "
                         f"an author looks before writing a new one.")
