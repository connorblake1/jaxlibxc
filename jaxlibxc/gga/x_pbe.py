"""PBE exchange functional and variants.

Translated from maple/gga_exc/gga_x_pbe.mpl.

PBE enhancement factor: F(s) = 1 + kappa - kappa / (1 + mu*s^2/kappa)
where s = X2S * x_sigma (reduced gradient per spin).

References:
    Perdew, Burke, Ernzerhof, PRL 77, 3865 (1996)
"""

import jax.numpy as jnp

from .._types import FunctionalInfo, FunctionalDef, Family, Kind
from .._registry import register
from .._exchange import gga_exchange
from .._constants import X2S, MU_GE, MU_PBE, KAPPA_PBE


def _pbe_enhance(params, xs):
    """PBE enhancement factor F(xs).

    F(s) = 1 + kappa * (1 - kappa / (kappa + mu * s^2))
         = 1 + mu * s^2 / (1 + mu * s^2 / kappa)
    where s = X2S * xs.
    """
    kappa = params['kappa']
    mu = params['mu']
    s = X2S * xs
    s2 = s**2
    return 1.0 + kappa * (1.0 - kappa / (kappa + mu * s2))


def _pbe_x_energy(params, rs, z, xt, xs0, xs1):
    """PBE exchange energy density per electron."""
    enhance = lambda xs: _pbe_enhance(params, xs)
    return gga_exchange(enhance, rs, z, xs0, xs1)


# Standard PBE exchange (ID 101)
register(FunctionalDef(
    info=FunctionalInfo(
        number=101,
        name='gga_x_pbe',
        family=Family.GGA,
        kind=Kind.EXCHANGE,
    ),
    energy_fn=_pbe_x_energy,
    default_params={
        'kappa': 0.8040,
        'mu': 0.2195149727645171,
    },
    n_internal=3,  # xt, xs0, xs1
))

# PBEsol exchange (ID 116)
register(FunctionalDef(
    info=FunctionalInfo(
        number=116,
        name='gga_x_pbe_sol',
        family=Family.GGA,
        kind=Kind.EXCHANGE,
    ),
    energy_fn=_pbe_x_energy,
    default_params={
        'kappa': 0.8040,
        'mu': float(MU_GE),  # 10/81
    },
    n_internal=3,
))

# revPBE exchange (ID 102) -- Zhang-Yang, 1998
register(FunctionalDef(
    info=FunctionalInfo(
        number=102,
        name='gga_x_pbe_r',
        family=Family.GGA,
        kind=Kind.EXCHANGE,
    ),
    energy_fn=_pbe_x_energy,
    default_params={
        'kappa': 1.245,
        'mu': 0.2195149727645171,
    },
    n_internal=3,
))

# PBE-TCA exchange
register(FunctionalDef(
    info=FunctionalInfo(
        number=400,
        name='gga_x_pbe_tca',
        family=Family.GGA,
        kind=Kind.EXCHANGE,
    ),
    energy_fn=_pbe_x_energy,
    default_params={
        'kappa': 1.227,
        'mu': 0.2195149727645171,
    },
    n_internal=3,
))

# RPBE exchange (ID 144) -- Hammer et al 1999
register(FunctionalDef(
    info=FunctionalInfo(
        number=144,
        name='gga_x_rpbe',
        family=Family.GGA,
        kind=Kind.EXCHANGE,
    ),
    energy_fn=_pbe_x_energy,
    default_params={
        'kappa': 0.8040,
        'mu': 0.2195149727645171,
    },
    n_internal=3,
))

# APBE exchange (ID 184)
register(FunctionalDef(
    info=FunctionalInfo(
        number=184,
        name='gga_x_apbe',
        family=Family.GGA,
        kind=Kind.EXCHANGE,
    ),
    energy_fn=_pbe_x_energy,
    default_params={
        'kappa': 0.8040,
        'mu': 0.260,
    },
    n_internal=3,
))
