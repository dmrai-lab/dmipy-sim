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
    "22ccc499baa1797363d54e57d2b5dce6e6ae3c09fcd5a101f88cb9d4815b9eeb": {
        "field": {
            "position_coeffs": "cf784afe6f1372399844bcab7cb93f574ba8a88c778f5d7fd0f2bcaaa1c43351",
            "boundary_endpoint": "d3cc0706df8367d39925c199520600e49f0fc0fcb20553b454fd1f2ca2e34d76",
            "boundary_bands": "5e8da891248e713afa01dfd0cb836a41b84b5337bf47eaa2f64714e8ebed68d9",
            "compartment": "9d97d188917dec920eabca932dad8e76680a4634394776226c1896b470ca47ab",
            "comp_events": "8e63dde602a3ea9378abbb8dd463aaebc6238cc6e953d967d2966ef2ae3ec8d6",
            "bound_start": "d13d4a8b3b8add19b5970157f09d00c12cbda4fed4d74d8493156523f7069b66",
            "bound_events": "5fceeabde86a91a42fdcd3e4387e8ad12dc05e256ba127dcdead4e397e1b86ef",
            "field_coeffs": "7e198220810f6fd9b2bf128a32656745e769c5099bc22f1f3fa6b4fdbd9a0457",
            "qv_positions": "24b73a3becd1be85e234e41d7e3499964c0ac3281be51779a953841ff6065480",
            "qv_boundary": "35896dec343b87e45003e85c66e66c4dfe8f236a58904f8b1b39a2059f978dd8",
            "qv_field": "2d9b600a293a5c74a0e592a4daa93f1c3e4fc3d72288f15ff2bba037c0c692b9",
            "qv_field_increments": "42fecf9fcbb90733b3deddb8b79bd8915920e662ebf8d9639c4f0a4372dbbcfa",
        },
        "no_field": {
            "position_coeffs": "a0b73bfba5cf61b8a45af3b108f1dbdf14218e8f381f33259ad75ef8bd3f6735",
            "boundary_endpoint": "4fb5140ff86c9562da177c02646822b49bdf944854ba7604d8634a9c934d8123",
            "boundary_bands": "e5e86fd4ab9d6972470ba4e0c93adbc6d1b1567b257c76136c1f86b11c8e6882",
            "compartment": "9d97d188917dec920eabca932dad8e76680a4634394776226c1896b470ca47ab",
            "comp_events": "2f56bc4c3598c6f0629ca864e727aa46b27c10d550146be5795ae23bc25caea0",
            "qv_positions": "85c42a002f7dffe5e7961621d7893c417ffb0636759eaf93e39594a867b59db3",
            "qv_boundary": "63b3a85e00fba12436b9bda927206485a1adb4c454f996cf76813d3b6a897fc7",
        },
        "forward": ["97145e59d848907a4f0c22bafa96a722fb6b91d609ef3dcfabaf79c75d613caf",
                    "af197e46ba0cfc590830fd5cae9f874da496969454db10331188111b7a2c07a7",
                    "b43b8c74066db5a8cfb4c87116414e3dab2f8e7ea1fae1a62645aae9d4a1b0ea",
                    "7ac4047d43a83524622f202a99027af4df10ebcbf607c8079b8401c0a0405ac4",
                    "e5f307e5bd533a68a1eabbe7c01d8f203f188f40e5d1b7fe51245d446cd12b2d"],
    },
    "71a3eff2d8554b5c1d9b4ba3f78575b636f50281a3de6c62a08a0bd5e459ae58": {
        "field": {
            "position_coeffs": "1e24584f8b69eb81d7a20577212d049b4a39d9fad81d06254f5bf844c4a23c89",
            "boundary_endpoint": "07dcdc414151478a130cfdb7d5e6b4ad9e6bf7974842fa37136ce0a728f7b1f8",
            "boundary_bands": "2011a5bb8054280b2ecda6eceb4bba907fe3f3007e186ebcdb2038131ccaef02",
            "compartment": "9d97d188917dec920eabca932dad8e76680a4634394776226c1896b470ca47ab",
            "comp_events": "8e63dde602a3ea9378abbb8dd463aaebc6238cc6e953d967d2966ef2ae3ec8d6",
            "bound_start": "d13d4a8b3b8add19b5970157f09d00c12cbda4fed4d74d8493156523f7069b66",
            "bound_events": "5fceeabde86a91a42fdcd3e4387e8ad12dc05e256ba127dcdead4e397e1b86ef",
            "field_coeffs": "6d883f5790310f7796daf9f5598337cae94a04674564d29ab5c72424ffa0e2d9",
            "qv_positions": "db0980f9294b73fa9b48a80e6595ca0ecbf14fd0d9d0eae07bb433a2204243c0",
            "qv_boundary": "9702e0d46ac4d36a8561fe5f564aca4c8d3f755f11516f652172e0d91261d390",
            "qv_field": "078a77dff2fb7669676cec35646485d4bc9cb2b2a478f500c566f081951e0525",
            "qv_field_increments": "7eb99296c49eb8099fab89bc7763dcc20fa7d8b9ca00dcf9ffa20072abfe63ee",
        },
        "no_field": {
            "position_coeffs": "2c3ff1b7479fc28fda231daa411cd5d29cab703a10f3ad45e8d942d4f0b860f5",
            "boundary_endpoint": "91209249d3cfee857d741f2e7cea0ff16e79dc037bd1d5fca8a2707eaa1452a9",
            "boundary_bands": "bf537107fc57b7fadb571ef7c8704147bb60cef18ba19bba63f2b89ea21cc783",
            "compartment": "9d97d188917dec920eabca932dad8e76680a4634394776226c1896b470ca47ab",
            "comp_events": "2f56bc4c3598c6f0629ca864e727aa46b27c10d550146be5795ae23bc25caea0",
            "qv_positions": "9817631e68112b312e074b230bd0bc01988cac30a9a555e1a10ff1ee84dd8e65",
            "qv_boundary": "cfae902d2c581d091108097b10c89dc6a0ce796b24945457a8c0c3d5baef2a5b",
        },
        "forward": ["c2d5c7d4fc9bf1d5bfef8a0e4c89c2fc6db589578aa245ecf9b69cce853abd4f",
                    "af197e46ba0cfc590830fd5cae9f874da496969454db10331188111b7a2c07a7",
                    "64c397c8733b4e81c7d7eac42f90cb73d8ce36f6cc6e7b24705c6c4767debf56",
                    "d2868ae53f19de9c8fede8432c22ea0c3b5c9d87eb160c9164abfb14546ee3ab",
                    "7b7a491a00920b19d6da47a73250e6a1bee97fc02d9c442dee9d00260f5bcbcb"],
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
