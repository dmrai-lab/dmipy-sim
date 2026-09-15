"""The particle-mesh split of the strand field (#217): the closed form within `near_m` of a point, a coarse grid
of the switch-weighted far part beyond. On DiSCo's centre the superposition summed up to 289 strands per point at
the 50 um cutoff; the split sums the few within the switch and reads one trilinear value for the rest."""
import numpy as np
import pytest

from dmipy_sim.fields.strand_field import StrandFieldBasis, FarGrid
from dmipy_sim.io.strands import write_tck
from dmipy_sim.spec import disco_spec, walk_spec


def _strands(rng, n=40, side=40e-6, n_pts=7):
    """Random gently bent strands crossing a box end to end (as DiSCo's do: a strand ends at the domain's surface,
    never inside it), polylines with small kinks, radii of DiSCo's kind."""
    cls, ri, ro = [], [], []
    for _ in range(n):
        p0 = rng.uniform(0, side, 3); d = rng.normal(size=3); d /= np.linalg.norm(d)
        t = np.linspace(-1.2 * side, 1.2 * side, n_pts)
        pts = p0[None, :] + t[:, None] * d[None, :] + rng.normal(0, 0.3e-6, (n_pts, 3)) * (np.abs(t) < side)[:, None]
        cls.append(pts); r = rng.uniform(0.7e-6, 2.1e-6); ri.append(r); ro.append(r / 0.7)
    return cls, np.array(ri), np.array(ro)


def test_the_split_equals_the_superposition():
    """Near + far at random points equals the plain superposition to the grid's interpolation error; the grid round
    trips through a file; the meta records it; a mismatched cutoff is refused."""
    rng = np.random.default_rng(0)
    cls, ri, ro = _strands(rng)
    lo, hi = np.zeros(3), np.full(3, 40e-6)
    plain = StrandFieldBasis(cls, ri, ro, cutoff_m=30e-6, domain=(lo, hi))
    with pytest.raises(ValueError, match="nearest-segment gate"):
        plain.build_far_grid(1.0e-6, 4e-6)                        # a switch starting inside the largest sheath's gate
    far = plain.build_far_grid(1.0e-6, 12e-6, blend_m=4e-6)
    assert far.shape == (41, 41, 41) and far.values.dtype == np.float16 and far.blend_m == 4e-6
    split = plain.with_far(far)
    assert split.gather_radius_m == 12e-6 and split.meta["far_grid"]["sha256"] == far.sha256
    P = rng.uniform(2e-6, 38e-6, (2000, 3))
    c_plain = plain.channels(P); c_split = split.channels(P)
    rel = np.sqrt(((c_split - c_plain) ** 2).sum()) / np.sqrt((c_plain ** 2).sum())
    assert rel < 5e-3, rel                                        # measured 2e-3 (float16 grid, tricubic read)
    assert np.abs(c_split - c_plain).max() < 1e-2 * np.abs(c_plain).max()
    with pytest.raises(ValueError, match="cutoff"):
        plain.with_cutoff(60e-6).with_far(far)
    with pytest.raises(ValueError, match="without a far grid"):
        split.build_far_grid(1e-6, 12e-6)


def test_the_far_grid_round_trips_and_the_walk_reads_it(tmp_path):
    """A DiSCo-form spec walked with the far grid: the same positions as without (the field does not steer the
    walk), the field samples within the split's error of the plain ones, the pack's field meta carries the grid."""
    rng = np.random.default_rng(1)
    cls, ri, ro = _strands(rng, n=12, side=20e-6)
    tck, dia = str(tmp_path / "t.tck"), str(tmp_path / "d.txt")
    write_tck(tck, [c + 0.0 for c in cls], coordinate_unit_m=25e-6); np.savetxt(dia, 2 * ri / 1e-3)
    spec = disco_spec(tck, dia, side_m=20e-6)
    plain = StrandFieldBasis(cls, ri, ro, cutoff_m=25e-6, domain=(np.zeros(3), np.full(3, 20e-6)))
    far = plain.build_far_grid(0.5e-6, 10e-6, blend_m=3e-6)
    far.save(str(tmp_path / "far.npy")); back = FarGrid.load(str(tmp_path / "far.npy"))
    np.testing.assert_array_equal(back.values, far.values); assert back.meta == far.meta
    kw = dict(T_max=6e-4, dt_save=5e-5, seed=5, n_probe=20_000, require_gpu=False, field=True, adaptive_steps=True,
              field_cutoff_m=25e-6, field_cutoff_max_m=25e-6)
    w0 = walk_spec(spec, 40, **kw)
    w1 = walk_spec(spec, 40, field_far=str(tmp_path / "far.npy"), **kw)
    np.testing.assert_array_equal(w1.positions, w0.positions)
    d = w1.field_samples - w0.field_samples
    assert np.sqrt((d ** 2).sum()) / np.sqrt((w0.field_samples ** 2).sum()) < 3e-2
    assert w1.field_basis.far is not None and w1.field_basis.certificate["far_grid"]["sha256"] == far.sha256
    from dmipy_sim.replay.bank import build_replay_pack
    pk = build_replay_pack(w1, id="t/far", license="x", citation="x", K=6, susc_path_K=4, device="numpy")
    assert pk.meta["compression"]["channels"]["susceptibility_grid"]["source"]["far_grid"]["near_m"] == 10e-6


def test_free_ends_and_the_exact_grid_equals_the_cutoff_grid():
    """A strand's field ends at its free end with its finite-line factor (1/2 on the end plane, the dipole tail
    beyond), so strands ending inside the domain -- DiSCo's 24,392 ends do -- leave the far part smooth: a fixture
    with free ends reads to a tenth of a percent, and the read error falls with the spacing. The all-strands far
    grid equals the cutoff grid when the cutoff spans the box."""
    rng = np.random.default_rng(3)
    cls, ri, ro = [], [], []
    side = 40e-6
    for _ in range(40):                                              # strands ending INSIDE the box
        p0 = rng.uniform(0, side, 3); d = rng.normal(size=3); d /= np.linalg.norm(d)
        pts = np.stack([p0 - 0.35 * side * d, p0 + rng.normal(0, 0.3e-6, 3), p0 + 0.35 * side * d])
        cls.append(pts); r = rng.uniform(0.7e-6, 2.1e-6); ri.append(r); ro.append(r / 0.7)
    ri, ro = np.array(ri), np.array(ro)
    lo, hi = np.zeros(3), np.full(3, side)
    plain = StrandFieldBasis(cls, ri, ro, cutoff_m=80e-6, domain=(lo, hi))
    P = rng.uniform(2e-6, 38e-6, (1500, 3)); c0 = plain.channels(P)
    rel = {}
    for h in (2.0e-6, 1.0e-6):
        far = plain.build_far_grid(h, 14e-6, blend_m=4e-6, dtype=np.float32)
        c1 = plain.with_far(far).channels(P)
        rel[h] = np.sqrt(((c1 - c0) ** 2).sum()) / np.sqrt((c0 ** 2).sum())
    assert rel[1.0e-6] < 5e-3 and rel[1.0e-6] < rel[2.0e-6], rel              # measured 0.19 % -> 0.11 % (the taper read 1 %)
    far = plain.build_far_grid(1.0e-6, 14e-6, blend_m=4e-6, dtype=np.float32)
    exact = plain.build_far_grid(1.0e-6, 14e-6, blend_m=4e-6, dtype=np.float32, all_strands=True, chunk=512)
    assert exact.cutoff_m == pytest.approx(float(np.linalg.norm(hi - lo)))
    np.testing.assert_allclose(exact.values, far.values, atol=1e-5, rtol=0)     # 80 um spans the box: the same sum
