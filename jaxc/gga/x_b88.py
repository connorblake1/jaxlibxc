"""Becke 88 exchange functional.

Translated from maple/gga_exc/gga_x_b88.mpl.

F(x) = 1 + beta/X_FACTOR_C * x^2 / (1 + gamma*beta*x*arcsinh(x))

References:
    Becke, Phys. Rev. A 38, 3098 (1988)
"""

import jax.numpy as jnp

from .._types import FunctionalInfo, FunctionalDef, Family, Kind
from .._registry import register
from .._exchange import gga_exchange
from .._constants import X_FACTOR_C


def _b88_enhance(params, xs):
    """B88 enhancement factor."""
    beta = params['beta']
    gamma = params['gamma']
    x = xs
    return 1.0 + (beta / X_FACTOR_C) * x**2 / (1.0 + gamma * beta * x * jnp.arcsinh(x))


def _b88_energy(params, rs, z, xt, xs0, xs1):
    """B88 exchange energy density per electron."""
    enhance = lambda xs: _b88_enhance(params, xs)
    return gga_exchange(enhance, rs, z, xs0, xs1)


# GGA_X_B88 (ID 106)
register(FunctionalDef(
    info=FunctionalInfo(
        number=106,
        name='gga_x_b88',
        family=Family.GGA,
        kind=Kind.EXCHANGE,
    ),
    energy_fn=_b88_energy,
    default_params={
        'beta': 0.0042,
        'gamma': 6.0,
    },
    n_internal=3,
))
