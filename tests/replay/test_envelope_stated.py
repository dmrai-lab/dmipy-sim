"""A pack's certificate is read over the envelope it states, and a replay beyond it is refused.

The build envelope states the surface relaxivities ``rho2_list`` (m/s), the field strengths ``B0_list`` (T) and the
susceptibility scale ``chi_max`` (``compression.certified_envelope``: the library default where it states none, the
substrate's magnetic pools' scale for ``chi_max``). The pack records them (``replay_envelope.tissue.rho2_over_D_max``
over the walk's ``D``, ``replay_envelope.acquisition.B0_list``, ``replay_envelope.tissue.chi_max``); every field term is
read at the one certified ``B0 chi``, the largest ``B0`` times ``chi_max``; the field read's own limit is
``field_read.read_limit_B0_chi_T``, a different quantity.
"""
import warnings

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.accumulate import field_read_term, spec_chi_scale
from dmipy_sim.io.strands import write_tck, concat_centerlines
from dmipy_sim.replay import time_segments
from dmipy_sim.replay.bank import build_replay_pack, _round_sig
from dmipy_sim.replay.compression import certified_envelope, default_envelope
from dmipy_sim.spec import disco_spec, walk_spec
from dmipy_sim.spec.tissue import Tissue

BLOCK = 1e-4
BATTERY = dict(bvals=[0.0, 1e9], dirs=[[1, 0, 0], [0, 0, 1]], ogse_periods=[2], shortd_b=1e9, shortd_deltas_frac=[0.1],
               delta_frac=0.2, Delta_frac=0.5)


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("envelope")
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp / "t.tck"), str(tmp / "d.txt")
    write_tck(tck, *concat_centerlines(cls_), coordinate_unit_m=25e-6)
    np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    return disco_spec(tck, dia, side_m=20e-6)


@pytest.fixture(scope="module")
def walk(spec):
    """A DiSCo-like strand walk with the contact and the field on."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return walk_spec(spec, 60, 8e-4, BLOCK, seed=3, n_probe=20_000, require_gpu=False, adaptive_steps=True, K=4,
                         field_cutoff_max_m=25e-6)


def _pack(walk, **env):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_replay_pack(walk, id="t/envelope", license="x", citation="x", K=4, susc_path_K=4, device="numpy",
                                 envelope=dict(BATTERY, **env), blt_temporal_K=2)


@pytest.fixture(scope="module")
def disco_like(walk):
    """DiSCo's envelope: rho2 <= 3 um/s, B0 1.5 / 3 / 7 T, chi the substrate's."""
    return _pack(walk, rho2_list=[3e-6], B0_list=[1.5, 3.0, 7.0])


def _seq(pk):
    return d.gre(4e-4, gradient_directions=[[1, 0, 0]], bvalues=[0.0], delta=1e-4, Delta=2e-4, n_t=pk.n_t, slew_rate=np.inf)


def _surface_quadrature(pk, rho2):
    """The local time's quadrature term at ``rho2``, computed from the pack's recorded block and quadratic variation
    as ``bank._derived_certificate`` states it: half of ``(rho2 / D)^2 sum_k rel_k 2 QV_B / (pi k)^2``, rounded as the
    certificate is written."""
    from dmipy_sim.replay import compression as cx
    q, T, K = pk.fidelity["quadrature"], float(pk.T_window), int(pk.K)
    k = np.arange(1, K + 1)
    rel = cx.quadrature_rel(int(q["m"]), float(q["h"]), np.pi * k / T)
    if q["h_round"] is not None:
        rel = rel + cx.quadrature_rel(0, float(q["h_round"]), np.pi * k / T)
    r = float(rho2) / float(pk.diffusivity)
    return _round_sig(0.5 * r ** 2 * float((rel * 2.0 * float(q["qv_contact_m2"]) / (np.pi * k) ** 2).sum()), 6)


def test_the_contact_certificate_is_read_at_the_stated_relaxivity(walk, disco_like):
    """The surface quadrature term is the one at the STATED rho2 (computed again from the pack's own record), and
    the pack states the range it was certified over."""
    D = disco_like.diffusivity
    wider = _pack(walk, rho2_list=[1e-5], B0_list=[1.5, 3.0, 7.0])
    for pk, rho2 in ((disco_like, 3e-6), (wider, 1e-5)):
        assert pk.fidelity["quadrature"]["surface_rho_over_D"] == _round_sig(rho2 / D, 6)         # written as the certificate is
        assert pk.fidelity["err_quadrature_surface"] > 0
        assert pk.fidelity["err_quadrature_surface"] == pytest.approx(_surface_quadrature(pk, rho2), rel=2e-6)   # to its sixth digit
    assert disco_like.rho2_over_D_max == pytest.approx(3e-6 / D) and wider.rho2_over_D_max == pytest.approx(1e-5 / D)
    assert disco_like.fidelity["surface_rho2_list"] == [3e-6]


def test_a_replay_beyond_the_stated_relaxivity_is_refused(disco_like):
    seq = _seq(disco_like)
    disco_like.replay(seq, tissue=Tissue(rho2=3e-6))
    with pytest.raises(ValueError, match="contact envelope"):
        disco_like.replay(seq, tissue=Tissue(rho2=1e-5))


def test_an_envelope_that_states_none_is_certified_at_the_library_default(walk, spec):
    """An envelope without ``rho2_list``, ``B0_list`` or ``chi_max`` is built at the default's (and the substrate's
    chi), written into the pack, and certifies exactly as the same envelope stated in full."""
    bare, full = _pack(walk), _pack(walk, **{k: certified_envelope(BATTERY, spec)[k] for k in ("rho2_list", "B0_list", "chi_max")})
    D = bare.diffusivity
    assert default_envelope()["rho2_list"] == [1e-5, 3e-5, 1e-4] and default_envelope()["B0_list"] == [3.0, 7.0]
    assert bare.rho2_over_D_max == pytest.approx(1e-4 / D)
    assert bare.replay_envelope["acquisition"]["B0_list"] == [3.0, 7.0]
    assert bare.replay_envelope["tissue"]["chi_max"] == pytest.approx(spec_chi_scale(spec))
    assert bare.fidelity == full.fidelity and bare.replay_envelope == full.replay_envelope


def test_every_field_term_is_read_at_the_one_certified_scale(walk, spec, disco_like):
    """The field read's term is charged at the largest ``B0`` and ``chi_max``; the quadrature and the read terms scale
    with ``(B0 chi)^2``; the read's own limit is named apart, and a segment layout's widths are derived at the
    certified scale."""
    env = disco_like.replay_envelope
    chi = spec_chi_scale(spec)
    assert env["acquisition"]["B0_list"] == [1.5, 3.0, 7.0] and env["tissue"]["chi_max"] == pytest.approx(chi)
    fe = disco_like.field_envelope
    assert fe["B0_max"] == 7.0 and fe["B0_chi_max"] == pytest.approx(7.0 * chi)
    fr = env["field_read"]
    assert fr["term_at"]["B0"] == 7.0 and fr["term_at"]["chi"] == pytest.approx(chi)
    assert "read_limit_B0_chi_T" in fr and "B0_chi_max_T" not in fr
    qv = disco_like.meta["walk_params"]["qv"]["field_increments"]
    consts = disco_like.meta["compression"]["channels"]["susc_field_basis"]["read"]["constants"]
    assert disco_like.fidelity["terms"]["field_read"] == pytest.approx(
        field_read_term(disco_like.meta["walk_params"]["field_read_interval"], qv, 7.0, chi, consts), rel=1e-12)
    low = _pack(walk, rho2_list=[3e-6], B0_list=[3.0])
    t7, t3 = disco_like.fidelity["terms"], low.fidelity["terms"]
    # both terms are (gamma B0 chi)^2 times the walk's own quantities, exactly (fidelity.terms carries them unrounded):
    # the ratio is (7 / 3)^2 to float64's rounding
    assert t7["field_read"] / t3["field_read"] == pytest.approx((7.0 / 3.0) ** 2, rel=1e-12)
    assert t7["quadrature"]["field"] / t3["quadrature"]["field"] == pytest.approx((7.0 / 3.0) ** 2, rel=1e-12)
    seg = time_segments._envelope(disco_like.meta, 0.08)
    assert seg["B0_chi_max_T"] == pytest.approx(7.0 * chi) and seg["B0_max_T"] == 7.0
    assert seg["rho2_over_D_max"] == pytest.approx(disco_like.rho2_over_D_max)


def test_a_replay_beyond_the_certified_field_is_refused(disco_like, spec):
    seq = _seq(disco_like)
    chi = spec_chi_scale(spec)
    at_scale = Tissue(chi_iso=-chi / 2, chi_aniso=-chi / 2)
    disco_like.replay(seq, scanner=7.0, tissue=at_scale)
    with pytest.raises(ValueError, match="B0 = 9.4 T is beyond"):
        disco_like.replay(seq, scanner=9.4, tissue=at_scale)
    with pytest.raises(ValueError, match="certified field scale"):
        disco_like.replay(seq, scanner=7.0, tissue=Tissue(chi_iso=-2 * chi))
    disco_like.replay(seq, scanner=3.0, tissue=Tissue(chi_iso=-2 * chi))             # within B0 chi at 3 T


def test_a_layout_of_a_pack_that_states_no_envelope_is_refused(disco_like):
    """A contact or field pack that does not state the range its certificate was read over gives no width."""
    import copy
    for key, match in (("rho2_over_D_max", "contact envelope"), ("chi_max", "susceptibility scale")):
        meta = copy.deepcopy(disco_like.meta)
        del meta["replay_envelope"]["tissue"][key]
        with pytest.raises(ValueError, match=match):
            time_segments._envelope(meta, 0.08)
