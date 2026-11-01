"""The envelopes test packs state. A test walks for the ``connectom`` class's band (:func:`walk_envelope`, the
class's preset with the test's field and contact bounds), and a pack whose waveforms are beyond every catalogue
scanner (ideal square lobes, strong gradients) states the test's own wide limits (:func:`stated`); a replay beyond
either is still refused."""
from dmipy_replay import Envelope, envelope_of
from dmipy_sim.replay.compression import BATTERY_KEYS, battery_reach  # noqa: F401  (battery_reach: the tests' G_max for a battery-read certificate)

TEST_G_MAX = 100.0          #: T/m per axis: the test's own limit, above every waveform a test plays
TEST_BAND_HZ = 1e9          #: Hz: wider than any test pack stores, so a pack states the band it stores
TEST_RHO2_MAX = 1e-4        #: m/s: the surface relaxivity a test's contact tier is certified up to
TEST_B0_LIST = (3.0, 7.0)   #: T: the field strengths a test's field tier is read at


def walk_envelope(scanner="connectom", *, of=None, G_max=None, band_hz=None, B0_list=TEST_B0_LIST, chi_max=None,
                  rho2_max=TEST_RHO2_MAX, max_refocusing_pulses=None):
    """The envelope of scanner class ``scanner`` (:func:`dmipy_replay.envelope_of`) with the test's bounds, its
    ``G_max`` or ``band_hz`` replaced where given. ``of`` is the spec (or geometry) walked: its own susceptibility scale
    (``spec_chi_scale``) is the ``chi_max`` where none is given."""
    if chi_max is None and of is not None:
        from dmipy_sim.engine.accumulate import spec_chi_scale
        chi_max = spec_chi_scale(getattr(of, "spec", of))
    e = envelope_of(scanner, B0_list=B0_list, chi_max=chi_max, rho2_max=rho2_max,
                    max_refocusing_pulses=max_refocusing_pulses)
    return e.replace(**{k: v for k, v in dict(G_max=G_max, band_hz=band_hz).items() if v is not None})


def stated(settings=None, G_max=None, *, of=None, band_hz=TEST_BAND_HZ, chi_max=None):
    """``dict(envelope=, battery=)`` for ``build_replay_pack`` from a test's certificate ``settings``: the battery keys
    (``compression.BATTERY_KEYS``) are the battery, and the envelope is the test's own, ``G_max`` and ``band_hz`` with
    the settings' field strengths (``B0_list``; the test's where they state none), surface relaxivities (``rho2_list``: their largest is ``rho2_max``),
    ``chi_max`` (``chi_max`` where they state none, else the susceptibility scale of the walk ``of``) and refocusing depth (``max_refocus_pulses``) where they state them."""
    s = dict(settings or {})
    if chi_max is None and of is not None:                   # the walk's substrate's own susceptibility scale
        from dmipy_sim.engine.accumulate import spec_chi_scale
        chi_max = spec_chi_scale(getattr(of, "spec", of))
    battery = {k: v for k, v in s.items() if k in BATTERY_KEYS} or None
    rho = s.get("rho2_list", [TEST_RHO2_MAX])
    env = Envelope(name="test", band_hz=band_hz, G_max=(G_max if G_max is not None else s.get("G_max", TEST_G_MAX)), B0_list=tuple(s.get("B0_list") or TEST_B0_LIST),
                   chi_max=s.get("chi_max", chi_max),
                   rho2_max=(max(rho) if rho else None), max_refocusing_pulses=s.get("max_refocus_pulses"))
    return dict(envelope=env, battery=battery)

