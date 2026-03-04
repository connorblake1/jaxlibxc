"""jaxlibxc: Exchange-correlation functionals in JAX.

A reimplementation of libxc using JAX for automatic differentiation,
JIT compilation, and GPU acceleration.
"""

# Enable float64 -- non-negotiable for DFT accuracy
import jax
jax.config.update("jax_enable_x64", True)

from .functional import Functional
from ._registry import available, get as get_functional
from ._types import Family, Kind

# Import functional subpackages to trigger registration
from . import lda
from . import gga
from . import mgga

__all__ = [
    'Functional',
    'available',
    'get_functional',
    'Family',
    'Kind',
]
