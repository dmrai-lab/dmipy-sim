"""A diffusion-prepared RF train over a whole ODF phantom -- what the vector-Bloch route cannot do.

`replay_bloch` needs one rotation per slot, so it refuses an `odf_sh` phantom (dmipy-sim#338) and a brain
is `odf_sh`. Decomposing the train into microscopic gates removes the obstacle: each gate is an ordinary
phase sum, and the pose expansion has always carried those.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.phantom import Grid, Inert, ODF, PackSubstrate, Phantom
from dmipy_sim.replay import read_rpk
from dmipy_sim.replay.bank import build_replay_pack

SH = (6, 6, 2)


@pytest.fixture(scope="module")
def pack_path(tmp_path_factory):
    out = tmp_path_factory.mktemp("pk") / "wm.rpk"
    walk = d.simulate_trajectories(600, 2e-9, d.FreeDiffusion(), 0.14, 2.5e-4, seed=0, require_gpu=False)
    build_replay_pack(walk, id="test/wm", license="x", citation="x", K=10, out_path=str(out), segment_T=0.14)   # one 140 ms window
    return str(out)


@pytest.fixture(scope="module")
def brain(pack_path):
    grid = Grid(shape=SH, voxel_size_m=(2.5e-2,) * 3,
                origin_m=tuple(-0.5 * (n - 1) * 2.5e-2 for n in SH), isocenter_m=(0.0, 0.0, 0.0))
    c = np.zeros(SH + (45,), np.float32)
    c[..., 0] = 1.0 / np.sqrt(4 * np.pi)
    c[..., 3] = 0.3
    wm = PackSubstrate(pack_path, m0=0.7, name="wm")
    ph = Phantom.compose(grid, fractions={wm: np.ones(SH, np.float32)},
                         orientation={wm: ODF(c, basis="mrtrix3")}, remainder=Inert(name="bg"))
    return ph, read_rpk(pack_path), grid


def _train(n_echo=3, beta=150.0):
    return sequences.splice([[1.0, 0, 0]], 15e-3, 25e-3, n_echo, 10e-3, bvalues=[1e9],
                            TE_prep=80e-3, beta_deg=beta, n_t_per_echo=40)


def test_a_train_replays_over_an_odf_phantom_which_the_bloch_route_refuses(brain):
    """The point of the whole decomposition, stated as a test."""
    ph, pack, _grid = brain
    assert ph.mode == "odf_sh"
    with pytest.raises(ValueError, match="frames-mode"):
        ph.file.replay_bloch(_train(), packs={0: pack})
    S = ph.replay_train(_train(), packs={0: pack})
    assert S.shape[:3] == SH and np.all(np.isfinite(S))
    assert np.nanmean(np.abs(S)) > 0


def test_the_flip_angle_changes_the_signal_as_the_pathways_say(brain):
    """A reduced flip is not a scale factor: the pathways part and the echoes stop being equal. A route that
    merely attenuated would pass a weaker test than this one."""
    ph, pack, _g = brain
    perfect = np.abs(ph.replay_train(_train(beta=180.0), packs={0: pack}))
    reduced = np.abs(ph.replay_train(_train(beta=120.0), packs={0: pack}))
    assert np.nanmean(reduced) < np.nanmean(perfect)

    # and echo to echo, a perfect train is flat where a reduced one is not (no relaxation, no train gradient)
    flat = [np.nanmean(np.abs(ph.replay_train(_train(beta=180.0), echo=e, packs={0: pack}))) for e in range(3)]
    bent = [np.nanmean(np.abs(ph.replay_train(_train(beta=120.0), echo=e, packs={0: pack}))) for e in range(3)]
    assert np.ptp(flat) < 1e-6 * np.mean(flat)
    assert np.ptp(bent) > 1e-2 * np.mean(bent)


def test_a_transmit_map_costs_a_reweighting_not_a_reexpansion(brain):
    """The cost claim. The gate expansions do not depend on the transmit scale, so a map with many scales
    costs one state propagation each rather than one magnetisation propagation per voxel -- and binning
    bounds that count for a map a machine produced, which is smooth."""
    ph, pack, grid = brain
    smooth = np.linspace(0.8, 1.0, ph.n_voxels).reshape(SH)
    fine, coarse = {}, {}
    S_fine = ph.replay_train(_train(), transmit=smooth, transmit_tolerance=1e-2, packs={0: pack}, report=fine)
    S_coarse = ph.replay_train(_train(), transmit=smooth, transmit_tolerance=1e-1, packs={0: pack}, report=coarse)
    assert fine["n_scales"] > coarse["n_scales"]                 # binning is what bounds the count
    assert coarse["n_gates"] == fine["n_gates"]                  # and it does not touch the expansions
    rel = np.nanmax(np.abs(np.abs(S_fine) - np.abs(S_coarse))) / np.nanmax(np.abs(S_fine))
    assert rel < 0.2, f"a tenth of a flip angle moved the signal by {rel:.3f}"


def test_the_gate_count_is_set_by_the_preparation_not_the_train(brain):
    """A longer train does not cost more expansions -- which is what makes seventy echoes possible at all."""
    ph, pack, _g = brain
    counts = []
    for n_echo in (2, 4, 8):
        rep = {}
        ph.replay_train(_train(n_echo=n_echo), packs={0: pack}, report=rep)
        counts.append(rep["n_gates"])
    assert len(set(counts)) == 1, f"gate count moved with the train: {counts}"


def test_the_accelerated_route_agrees_with_the_plain_one(brain):
    """The GPU path is a different arrangement of the same contraction, so it must agree to the precision it
    works in -- float32 -- and not merely correlate."""
    jax = pytest.importorskip("jax")
    ph, pack, _g = brain
    smooth = np.linspace(0.85, 1.0, ph.n_voxels).reshape(SH)
    a = ph.replay_train(_train(), transmit=smooth, packs={0: pack}, jax=False)
    b = ph.replay_train(_train(), transmit=smooth, packs={0: pack}, jax=True)
    rel = np.nanmax(np.abs(np.abs(a) - np.abs(b))) / max(np.nanmax(np.abs(a)), 1e-30)
    assert rel < 1e-5, f"the two routes differ by {rel:.2e}"


# ── a drifting magnet (dmipy-sim#285 item 5) ────────────────────────────────────────────────────────
def test_off_resonance_is_gated_like_the_gradient_not_applied_at_the_end(brain):
    """The correctness point. A field offset accrues only while magnetisation is TRANSVERSE, so it is gated
    by the same F+/F-/Z pattern the gradient is. The refocused pathway comes back to zero signed transverse
    time and so refocuses an offset exactly; a pathway that slept through an interval does not. A train
    therefore has no single coherence gate, and applying one factor at the end would be wrong for every
    pathway but one."""
    from dmipy_sim.replay.pathways import train_response
    _ph, pack, _g = brain
    tr = train_response(pack, _train(), keep=(8, 0))
    tau = tr.tau
    assert any(abs(v) < 1e-9 for v in tau.values()), "no refocused pathway"
    assert any(abs(v) > 1e-3 for v in tau.values()), "no pathway that sleeps"
    # the one that never leaves the transverse plane accrues the whole preparation
    assert max(tau.values()) == pytest.approx(2 * max(v for v in tau.values() if 0 < v < max(tau.values())),
                                              rel=1e-6)


def test_a_spin_echo_refocuses_a_drifting_magnet_exactly(brain):
    """One pathway survives a perfect 180, its signed transverse time is zero, so a UNIFORM offset leaves
    both magnitude and phase untouched however far the magnet has drifted."""
    from dmipy_sim.replay.pathways import train_response
    from dmipy_sim.replay import so3
    _ph, pack, _g = brain
    se = _train(n_echo=1, beta=180.0)
    tr = train_response(pack, se, keep=(8, 0))
    A = so3.so3_design(8, np.eye(3)[None], 0)
    base = complex((A @ tr.at(1.0, echo=-1, dw=0.0).coeffs.T).reshape(-1)[0])
    for hz in (100.0, 1000.0, 2725.0):                       # far beyond any drift this magnet reaches
        s = complex((A @ tr.at(1.0, echo=-1, dw=2 * np.pi * hz).coeffs.T).reshape(-1)[0])
        assert abs(s) / abs(base) == pytest.approx(1.0, abs=1e-9)
        assert abs(np.angle(s / base)) < 1e-9


def test_a_refocusing_train_is_exactly_insensitive_to_a_uniform_offset(brain):
    """#285 filed drift as low priority on the assumption that "the SPLICE magnitude combination is
    insensitive to a slow phase ramp". It is more than insensitive: with hard pulses it is EXACT, and the
    reason is structural rather than a cancellation that happens to be good.

    A uniform offset multiplies every transverse coefficient by the same factor, so it can rotate the
    signal but never redistribute it between coherence orders. The train's echo is carried by the refocused
    pathway, whose signed transverse time is zero, so there is no second pathway to interfere with and the
    magnitude is untouched at any offset.

    An earlier version of this test recorded a bounded +-1.5 % envelope. That was an artefact: the
    off-resonance operator conjugated by the sign of the dephasing index, which made it non-uniform and let
    it move amplitude between orders -- something no static field offset can do. See
    tests/test_epg_off_resonance.py, which checks the operator against an isochromat sum.

    What this does NOT say is that a real magnet's drift is harmless. It says a UNIFORM offset is refocused
    under HARD pulses. Finite pulses excite about a tilted effective field when off-resonant, so the flip
    angle itself becomes offset-dependent -- a channel this route does not model."""
    from dmipy_sim.replay.pathways import train_response
    from dmipy_sim.replay import so3
    _ph, pack, _g = brain
    tr = train_response(pack, _train(n_echo=4, beta=150.0), keep=(8, 0))
    A = so3.so3_design(8, np.eye(3)[None], 0)

    def at(hz):
        return complex((A @ tr.at(1.0, echo=-1, dw=2 * np.pi * hz).coeffs.T).reshape(-1)[0])

    base = at(0.0)
    for hz in (13.0, 100.0, 545.0, 1400.0, 2725.0):
        assert abs(at(hz)) / abs(base) == pytest.approx(1.0, abs=1e-9)
    # the echo really is carried by the pathway that refocuses, which is why
    live = [g for g, w in tr.weights(1.0).items() if abs(w[-1]) > 1e-6]
    assert len(live) == 1 and tr.tau[live[0]] == pytest.approx(0.0, abs=1e-12)
