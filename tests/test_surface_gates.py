"""The contact tier's gates are the transverse ON-TIME (``bank._surface_gates``).

The surface weight ``exp(-(rho / D) l)`` is a positive, monotone functional of the local time ``l``: a 180 degree pulse
does not change its sign, the gate for C2 is on / off (``ScannerSequence.chi_perp``, ones for a spin echo), never the
+-1 of the phase's coherence sign. The walk-free certificate (``err_surface`` / ``floor_surface``) and the measured one
read the same gates, so a PGSE and a spin echo of one TE certify alike and the two routes agree on a pack with a
refocused battery.
"""
import warnings

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import build_replay_pack, sequences
from dmipy_sim.acquisition.rf import RFEvent, RFSchedule
from dmipy_sim.replay import bank

D0, R, T, H = 2e-9, 3e-6, 0.02, 5e-4
N_S = int(round(T / H)) + 1                                   # 41 samples: 40 blocks of 0.5 ms


def _env(rho2_list):
    return dict(bvals=[0.0, 1e9, 3e9], dirs=[[1, 0, 0], [0, 0, 1]], ogse_periods=[2], shortd_b=1e9,
                shortd_deltas_frac=[0.05], B0_list=[], theta_deg=[0], delta_frac=0.2, Delta_frac=0.5,
                rho2_list=list(rho2_list))


@pytest.fixture(scope="module")
def cylinder_walk():
    """A reflecting cylinder of radius 3 um walked on the CPU with the contact channel and every walker kept raw."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return d.simulate_trajectories(1500, D0, d.Cylinder(radius=R, orientation=[0, 0, 1]), T, H, seed=1,
                                       require_gpu=False, K=32, raw_subset=1500, window_T=T)


@pytest.fixture(scope="module")
def packs(cylinder_walk):
    """The walk-free and the measured pack of the one walk, at the default contact envelope."""
    kw = dict(id="test/cylinder", license="CC-BY-4.0", citation="test", envelope=_env([1e-5, 1e-4]), K=32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (build_replay_pack(cylinder_walk, fidelity="derived", **kw),
                build_replay_pack(cylinder_walk, fidelity="measured", **kw))


def _replay_gate(seq):
    """The gate the replay reads the contact through: the sequence's ``chi_perp``, ones when it is transverse
    throughout (a spin echo), on the block grid of the walk."""
    chi = np.ones(seq.n_t) if seq.chi_perp is None else np.asarray(seq.chi_perp, np.float64)
    t_seq = np.arange(seq.n_t) * seq.dt
    return np.interp(np.arange(N_S) * H, t_seq, chi)


def test_the_surface_gates_are_the_transverse_on_time_of_every_schedule():
    """Every gate is 0 or 1, and each schedule's transverse gate (gradient echo, spin echo, a CPMG train, a stimulated
    echo with its mixing time stored) is one of them: a refocusing pulse leaves the gate on."""
    gates = bank._surface_gates(N_S)
    assert all(set(np.unique(g)) <= {0.0, 1.0} for g in gates)
    assert len(gates) == 2                                    # the whole window, and the stimulated echo's two quarters
    ev = lambda t, flip, label: RFEvent(t * T, flip, label)
    schedules = {
        "gre": RFSchedule([ev(0.0, 90, "excitation")]),
        "se": RFSchedule([ev(0.0, 90, "excitation"), ev(0.25, 180, "refocus")]),
        "cpmg4": RFSchedule([ev(0.0, 90, "excitation")] + [ev((k + 0.5) / 4, 180, "refocus") for k in range(4)]),
        "ste": RFSchedule([ev(0.0, 90, "excitation"), ev(0.25, 90, "store"), ev(0.75, 90, "recall")]),
    }
    for name, rf in schedules.items():
        chi = rf.transverse_gate(N_S, H).astype(np.float64)
        assert any(np.array_equal(chi, g) for g in gates), name
    # the sign a refocusing pulse puts on the PHASE is not a gate of the contact: the battery's gates carry it, these do not
    assert min(g.min() for g in bank._battery_gates(N_S, 4)) == -1.0 and min(g.min() for g in gates) == 0.0


def test_a_pgse_and_a_spin_echo_of_the_same_te_certify_alike(packs):
    """The contact rows read through a PGSE's gate and through a spin echo's, at one TE, are the same rows: the
    certificate depends on the transverse on-time and not on the gradient between the pulses. Read through the
    coherence SIGN (the phase's gate) the same pack's rows are an order of magnitude larger."""
    from dmipy_sim.replay import compression as cx
    pk, _ = packs
    ch = pk.meta["compression"]["channels"]["boundary_local_time"]
    endpoint, bands = np.asarray(pk.arrays["blt_endpoint"], np.float64), cx.bridge_bands(pk.arrays, ch)
    pgse = sequences.pgse([[1, 0, 0]], 4e-3, 10e-3, bvalues=[1e9], TE=T, n_t=201, slew_rate=np.inf)
    se = sequences.cpmg(1, T / 2, n_t_per_echo=100)

    def rows(gate):
        return bank._surface_terms_walk_free(endpoint, bands, pk.arrays, ch, _env([1e-4, 1e-3]), T=T, h=H, D=D0, gates=[gate])

    a, b = rows(_replay_gate(pgse)), rows(_replay_gate(se))
    assert a["err"] == b["err"] and a["floor"] == b["floor"]
    assert a["floor"] < 1e-2
    signed = rows(bank._cpmg_gate(N_S, 1))                    # the spin echo's coherence sign: +1, then -1 after the pulse
    assert signed["floor"] > 10.0 * a["floor"], (signed["floor"], a["floor"])


def test_the_walk_free_contact_term_agrees_with_the_measured_one(packs):
    """On a pack whose battery is refocused (spin echo, a four-pulse train, a stimulated echo) the walk-free
    ``err_surface`` is the measured one (the raw subset's dense reading) to the floor, both a small fraction of it."""
    wf, ms = packs
    f_wf, f_ms = wf.fidelity, ms.fidelity
    assert f_wf["surface_certified"] == "walk_free" and f_ms["surface_certified"] == "measured"
    for f in (f_wf, f_ms):
        assert f["err_surface"] <= 2.0 * f["floor_surface"] and f["err_surface"] < 1e-2 and f["floor_surface"] < 1e-2
    assert abs(f_wf["err_surface"] - f_ms["err_surface"]) <= 0.25 * f_ms["floor_surface"], (f_wf["err_surface"], f_ms["err_surface"])


def test_the_walk_free_contact_term_charges_the_bands_the_pack_does_not_store(cylinder_walk):
    """A pack storing fewer contact bands than the walk accumulated (``blt_temporal_K``) is charged their truncation
    walk-free: in the float container the stored bridge has no quantisation, its whole error against the dense reading
    is the bands left out, and the walk-free ``err_surface`` is the measured one to a quarter of the floor."""
    kw = dict(id="test/cylinder", license="CC-BY-4.0", citation="test", envelope=_env([1e-5, 1e-4]), K=32,
              blt_temporal_K=4, blt_container=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_wf = build_replay_pack(cylinder_walk, fidelity="derived", **kw).fidelity
        f_ms = build_replay_pack(cylinder_walk, fidelity="measured", **kw).fidelity
    assert f_wf["err_surface_quantisation"] == 0.0 and f_wf["err_surface_truncation"] > 0.0
    assert f_wf["err_surface"] >= f_wf["err_surface_truncation"]
    assert abs(f_wf["err_surface"] - f_ms["err_surface"]) <= 0.25 * f_ms["floor_surface"], (f_wf["err_surface"], f_ms["err_surface"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = build_replay_pack(cylinder_walk, fidelity="derived", **dict(kw, blt_temporal_K=32)).fidelity
    assert full["err_surface_truncation"] == 0.0
