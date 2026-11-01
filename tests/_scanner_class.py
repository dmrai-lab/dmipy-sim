"""The scanner class a test pack states when the waveforms it replays are beyond every catalogue scanner (ideal square
lobes, strong gradients): the stated limits are the test's own, generous, and a replay beyond them is still refused."""
from dmipy_sim.replay.compression import default_envelope

TEST_G_MAX = 100.0          # T/m
TEST_SLEW_MAX = 1e12        # T/m/s


def stated_class(envelope=None, G_max=TEST_G_MAX):
    """``envelope`` (the default's when None) with a scanner class of the test's own limits stated."""
    return dict(default_envelope() if envelope is None else envelope, scanner="connectome_2", G_max=G_max,
                slew_max=TEST_SLEW_MAX)
