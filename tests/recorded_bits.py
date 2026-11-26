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
from unittest.mock import patch

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
#: ``fixed_step_default`` holds the full block and saved-array hashes of the default fixed-step field walk.
FAMILIES = {
    "d0d7a6f831ecca7f9d94e401041709594f0f2ecf0cd90194cc97f1efb76590ea": {
        "field": {
            "position_coeffs": "3886a6f2918f49d9db8150cab5b0db35708e2004a6c523a077c02f522c99d22b",
            "boundary_endpoint": "a6468cc76b61593a891a20cdbb8a4122f4b8a7fb72cb56dcab8d481c255515d9",
            "boundary_bands": "483a79705963fd16b2e5bd44a523670193fc2c57c1de8b06e21d3c2beebf8635",
            "compartment": "94a6c8a51255a63f755262119e0751907ee258b1a45d5f6fd60bce2bf896cfef",
            "comp_events": "c80c27f359a2e1e78d9cc18697d56de853dbe5b236a4e097b891fd7319a5bb61",
            "bound_start": "d13d4a8b3b8add19b5970157f09d00c12cbda4fed4d74d8493156523f7069b66",
            "bound_events": "d85e665e87915a0547074574123091b10559d70c45a417a9d658aab4736c0b1f",
            "field_coeffs": "0885d62d5a6173f22669523cdd92fe3c501039ae9864283b589993db9bcd2316",
            "qv_positions": "01dc9bd13aa70c2848fa2f52d99302059f79419abfbc41c943fe86c26f3b53f0",
            "qv_boundary": "2a10f33414d66a537f794a962c6cd4eaa352d75ff9a7938c8bb0a7137e8333b9",
            "qv_field": "4c685966cc3d7f4046fe27b4ed59cc46047fa35850ed95bb95a2f9439ea5c15f",
            "qv_field_increments": "af21ef398b73a5b1748f3bdd6bc419cba9d538515b59e2475eaf44582d665ec0",
        },
        "no_field": {
            "position_coeffs": "2e5fca48f9c1f003775cf741a47b3a37b89cb2382fdb834ea4c9d466bba2c3a9",
            "boundary_endpoint": "44bef2ce5439f01e63b5716e10c4f9b2c89cc5c4b928c9e26f21975166d0dc8b",
            "boundary_bands": "15d7ff33823cbea0ad3e5382e51c343b9573546695856daabad520f8e00054e6",
            "compartment": "94a6c8a51255a63f755262119e0751907ee258b1a45d5f6fd60bce2bf896cfef",
            "comp_events": "7ea15196b67da26fe05a79d221cec25db800801ea85e4a9d577af14cec143e6f",
            "qv_positions": "081ed72cdf2fae25c262654a2c9b172e9f3d5d9bb8be4341ed6d2c1d2849836c",
            "qv_boundary": "25802701c078b6cc96f6366a1101e6313d60c927bbff9af86ca01d2e3b002281",
        },
        "forward": ["bd76a93f00d43a0d0427a92dd1462aeafa0cf503abace23519d05dd7842777a9",
                    "c411c50a8a9d328af16bb9757147c4000467fd76f7793a9ba63c08a40fe46ed0",
                    "45d80fd89aaa416b48aa75c31b5a677af425783f6a89df7ee9c24892ea3195f3",
                    "8eb40ffebb5d9d2c85cef70c3df5b61a2e4e1f22d9661bb99419e958518bacc8",
                    "d74f0785f9b4e8d7cafade4125099db0551920906277888b20293a58be3ccef9"],
    },
    "e74c68c7486e5194fea424efc88122e001a185a20b6925b53b1b9f79a1cef439": {
        # From the default walk at d2867a9 with time.time fixed at 1000 s.
        "fixed_step_default": {
            "boundary_bands": "c2b4e7f01cdad7545689f0327b9db2f20d0371a42e42b71f014ea13023f0bb5b",
            "boundary_endpoint": "4d964515c4cafd4572b3707244d4766651c80a071c4838260a195bec4cdeaef7",
            "field_coeffs": "c9595d30efbc1885c15f5e82fbbe33777a5f1428e7ff2f3cc993d9c84611df13",
            "position_coeffs": "d21a361471fd25ff07a33fca8c9be65361f122835d642f7989c569398d1fdce1",
            "qv_boundary": "aaabb75945ca9f3206caefd7225c05e1e901bb27b609b7b8f1e6744ecc24852b",
            "qv_field": "f72fdd3a50cb92f6d46f608db6216303049808821eb25da37bf0b517023b45ed",
            "qv_field_increments": "67ed3b30aa8bb3087e8381e73104f284ad2890bf45fcb248c84ec28f967035b7",
            "qv_positions": "266cdebff7a95b7fe0adcc84ba9ff13c2d564614ba2c560015112d8aa7942b57",
            "compartment": "4a59c1136395d4124da5604dfd60e7d6739bf17eb1d80a6d40f84fc839f5f00e",
            "block": "ce1443b33ae6e40e0fce297b8b6187891ca6b19a8ebfc0cf3a8fa87a20cf69fa",
        },
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


def fixed_step_default_hashes(path):
    """Hash the default fixed-step field walk's full block and saved tensors with a pinned clock."""
    from safetensors.numpy import load_file
    from dmipy_sim import walk
    from dmipy_sim.engine.calibration import canonical
    from tests._envelope import walk_envelope
    from tests.test_spec_field_closed_form import _cell

    with patch("time.time", return_value=1000.0):
        spec = _cell().spec
        w = walk(spec, 24, T_max=5e-4, block_h=1.25e-4, envelope=walk_envelope(of=spec), seed=3,
                 require_gpu=False, K=4, K_c3=4, field_read_interval=1.25e-4)
    w.save(path)
    hashes = {name: hashlib.sha256(array.tobytes()).hexdigest() for name, array in load_file(str(path)).items()}
    hashes["block"] = hashlib.sha256(canonical(w.block).encode()).hexdigest()
    return hashes


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
