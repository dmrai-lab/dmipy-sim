"""Physics test suite configuration.

Sets XLA_PYTHON_CLIENT_PREALLOCATE=false before JAX is initialized so that
JAX allocates GPU memory on demand rather than reserving 75% up-front.
This is required for the permeability tests at N=1M walkers and long run
times (n_t up to ~6000 steps), which need ~10 GB of GPU memory but the
preallocation would leave only ~12 GB free on a 46 GB card.
"""

import os

# Must be set before any JAX import.  The physics conftest.py is loaded
# by pytest before the test modules, so this is the correct place.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
