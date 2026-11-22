"""Recorded bits: the hashes of fixed walks, one set per family of compiled CPU arithmetic, selected by the canary.

A compiled walk's float32 bits depend on the JAX build (:data:`RECORDED_WITH`) and on the arithmetic XLA's CPU backend
emits for the host, which differs between families of CPUs (an instruction-set cap does not remove the difference).
The recorded values are therefore kept per family in :data:`FAMILIES`, keyed by the family's canary: the hash of a
small fixed walk (:func:`canary`) through the same concentric wall kernel. The ``test_recorded_bits_*`` tests read the
values of this host's family through the ``recorded_bits`` fixture (:func:`recorded_values`).

The canary only SELECTS the family. A matching canary is not evidence that the recorded bits match: a compile setting
can leave the canary unchanged and still change the bits of a recorded walk. The recorded tests themselves are that
evidence.

Where the recorded tests run: wherever jax / jaxlib are :data:`RECORDED_WITH` and the canary names a recorded family.
In the pinned ``recorded-bits`` CI job (which sets :data:`RECORDED_BITS_ENV` to ``1``) another build is an error and a
canary of no recorded family is a failure that prints the canary; anywhere else either is a skip with that reason.
``tools/record_recorded_bits.py`` prints a host's canary and the values in the form of :data:`FAMILIES`.
"""
import hashlib
import os

import numpy as np
import pytest

#: The JAX build the recorded bits were recorded with (jax, jaxlib): another build compiles the same walk to other
#: float32 bits (measured: jaxlib 0.12 differs from 0.10 on every float array).
RECORDED_WITH = ("0.10.2", "0.10.2")
#: The variable the pinned ``recorded-bits`` CI job sets to ``1``: there a build other than :data:`RECORDED_WITH` is an
#: error and a canary of no recorded family a failure; elsewhere both are skips.
RECORDED_BITS_ENV = "DMIPY_SIM_RECORDED_BITS"

#: The recorded values per family, keyed by the family's canary (:func:`canary`): for each, the hashes of the
#: all-myelinated reference walks of ``tests/geometry/test_mixed_myelinated_plain_cell.py`` (``recorded_hashes``):
#: ``field`` and ``no_field`` per array of the walk, ``forward`` per output of the forward signal. The arrays that
#: carry the field (``field_coeffs``, ``qv_field``, ``qv_field_increments``) are the walk's with the cell's exactly
#: periodic field (:class:`~dmipy_sim.fields.hollow_cylinder.HollowCylinderField`).
FAMILIES = {
    "e74c68c7486e5194fea424efc88122e001a185a20b6925b53b1b9f79a1cef439": {
        "field": {
            "position_coeffs": "50414eae62de40d03fb55fa05f74c7afce290f7ffaec55234701f2376300f330",
            "boundary_endpoint": "a22f9c158be02a6e4ec7063f7637c052cfccee718de77cd3e950a22de0a83d49",
            "boundary_bands": "098c6c8e905a3a0b619897f98f923364273f9f823a7784116e10b99a5d71d523",
            "compartment": "94a6c8a51255a63f755262119e0751907ee258b1a45d5f6fd60bce2bf896cfef",
            "comp_events": "c80c27f359a2e1e78d9cc18697d56de853dbe5b236a4e097b891fd7319a5bb61",
            "bound_start": "d13d4a8b3b8add19b5970157f09d00c12cbda4fed4d74d8493156523f7069b66",
            "bound_events": "d85e665e87915a0547074574123091b10559d70c45a417a9d658aab4736c0b1f",
            "field_coeffs": "f7c85c60296503b29d3fc80373a23b31b172b6e7ccf01b31f57d364b5fdf79ec",
            "qv_positions": "de399f6e9c861c167f17dab32dbf632fecd8bc964cfc4c2286ff121cd3fa55df",
            "qv_boundary": "0b5973718144555252a88ae97325eb9cbf3323b8455295366ef56147b023c434",
            "qv_field": "99b50df513aa4b2cf6558fc3164fe3a94592b821d0962fcc75c8e0b694fc4251",
            "qv_field_increments": "ec76d65574b0e5a9ae586234c7289b088d7034511271e8cb18bbc9595f38c359",
        },
        "no_field": {
            "position_coeffs": "95df9377574fccd72ab08f03a1aafdb982058727d31d8e84bd3bfb6f38afff18",
            "boundary_endpoint": "e30075c774f5cdfc00c7026a1e1dde5e56d0621caf6e428b65c7b4de29266571",
            "boundary_bands": "355bb44d16026b0a5a8f447b343af9ddf358926fdabf00f3e550fcf8319c6257",
            "compartment": "94a6c8a51255a63f755262119e0751907ee258b1a45d5f6fd60bce2bf896cfef",
            "comp_events": "7ea15196b67da26fe05a79d221cec25db800801ea85e4a9d577af14cec143e6f",
            "qv_positions": "82f56ba77355742f2bb68b0a092f1ee1762fcc87f3b100327903053b20f08587",
            "qv_boundary": "4caf516cb3c64fb27cc5de493cb757f2eb4d89c43ac7eaf6aa7de44d3bf74be6",
        },
        "forward": ["4d4424174b850381ba58cfd4c2cc13430c3eb66d97b48d0dfea1aca5f1e6611f",
                    "c411c50a8a9d328af16bb9757147c4000467fd76f7793a9ba63c08a40fe46ed0",
                    "dbc2ededdf5901ec41ad255a99e309c84b034c7128ba778065ca43a302cdbb30",
                    "2fc0fd100afb026caaaadfcf399535365c56914cc96bf882993c749005908a97",
                    "7d943647b442068342534466ce4c5526b6ef1081908dc93df2908e3386503a1d"],
    },
}


def canary():
    """The canary: the sha256 of ``position_coeffs`` of a small fixed all-myelinated walk (16 walkers, 0.5 ms, four
    sheathed cylinders, permeable walls) through the concentric wall kernel, on the CPU. Its value names the family of
    compiled arithmetic this host belongs to; it says nothing about whether the recorded walks match."""
    from dmipy_sim.compartments import Compartments, Pool
    from dmipy_sim.engine.core import _walk_geometry
    from dmipy_sim.geometry import PackedMyelinatedCylinders
    D, R = 2e-9, 2e-6
    g = np.array([0.7, 0.8, 0.7, 0.75])
    outer = np.array([R, 0.8 * R, R, 0.9 * R])
    centres = np.array([[0.0, 0.0], [2.2, 0.0], [0.0, 2.2], [2.2, 2.2]]) * R
    cell = PackedMyelinatedCylinders(outer * g, g, centres, 4.4 * R, N_max=4, kappa_inner=4e-5, kappa_outer=4e-5,
                                     compartments=Compartments(intra=Pool(D=D), myelin=Pool(D=0.0), extra=Pool(D=D)))
    w = _walk_geometry(16, D, cell, 5e-4, 5e-5, tiers="all", seed=3, require_gpu=False, K=4)
    return hashlib.sha256(np.ascontiguousarray(w.position_coeffs).tobytes()).hexdigest()


def in_recorded_bits_job():
    """True in the pinned ``recorded-bits`` CI job (:data:`RECORDED_BITS_ENV` set to ``1``)."""
    return os.environ.get(RECORDED_BITS_ENV) == "1"


def check_build(build, *, in_job):
    """Refuse a JAX build other than :data:`RECORDED_WITH`: an assertion error in the job, a skip elsewhere."""
    if build == RECORDED_WITH:
        return
    reason = f"recorded bits are bits of jax / jaxlib {RECORDED_WITH}; this is {build}"
    if in_job:
        raise AssertionError(f"the recorded-bits job runs jax / jaxlib {RECORDED_WITH}, this is {build}")
    pytest.skip(reason)


def family_of(canary_hash, *, in_job):
    """The recorded values of the family whose canary is ``canary_hash``, from :data:`FAMILIES`.

    A canary of no recorded family fails the test in the recorded-bits job (``in_job``) and skips it elsewhere, both
    naming the canary. The canary selects the family and nothing more: it is NOT evidence that this host's bits are the
    family's, since a compile setting can keep the canary and change a recorded walk; the recorded tests decide that."""
    if canary_hash in FAMILIES:
        return FAMILIES[canary_hash]
    reason = (f"this host's canary {canary_hash} names no recorded family (recorded: {sorted(FAMILIES)}); "
              f"tools/record_recorded_bits.py prints its values")
    if in_job:
        pytest.fail(f"unrecorded CPU family: {reason}", pytrace=False)
    pytest.skip(f"unrecorded CPU family: {reason}")


_CANARY = []


def recorded_values():
    """This host's family's recorded values, after :func:`check_build` and :func:`family_of` (the canary is walked
    once per process)."""
    import jax
    import jaxlib
    in_job = in_recorded_bits_job()
    check_build((jax.__version__, jaxlib.__version__), in_job=in_job)
    if not _CANARY:
        _CANARY.append(canary())
    return family_of(_CANARY[0], in_job=in_job)
