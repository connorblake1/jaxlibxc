"""Lee-Yang-Parr (LYP) GGA correlation functional.

Translated from maple/gga_exc/gga_c_lyp.mpl.

References:
    Lee, Yang, Parr, Phys. Rev. B 37, 785 (1988)
    Miehlich, Savin, Stoll, Preuss, CPL 157, 200 (1989)
"""

import jax.numpy as jnp
import numpy as np

from .._types import FunctionalInfo, FunctionalDef, Family, Kind
from .._registry import register
from .._utils import opz_pow_n
from .._constants import RS_FACTOR


_CF = 3.0 / 10.0 * (3.0 * np.pi**2) ** (2.0 / 3.0)
_AUX6 = 1.0 / 2.0 ** (8.0 / 3.0)
_AUX4 = _AUX6 / 4.0
_AUX5 = _AUX4 / 18.0  # _AUX4 / (9*2)


def _lyp_energy(params, rs, z, xt, xs0, xs1):
    """LYP correlation energy density per electron.

    Uses rr = rs/RS_FACTOR = n_total^(-1/3) as the density parameter.
    """
    a = params['a']
    b = params['b']
    c = params['c']
    d = params['d']

    rr = rs / RS_FACTOR  # = n^(-1/3)

    # Derived quantities
    omega = b * jnp.exp(-c * rr) / (1.0 + d * rr)
    delta = (c + d / (1.0 + d * rr)) * rr
    z2 = z**2
    one_mz2 = 1.0 - z2

    opz83 = opz_pow_n(z, 8.0/3.0)
    omz83 = opz_pow_n(-z, 8.0/3.0)
    opz113 = opz_pow_n(z, 11.0/3.0)
    omz113 = opz_pow_n(-z, 11.0/3.0)
    opz2 = opz_pow_n(z, 2.0)
    omz2 = opz_pow_n(-z, 2.0)

    # Term 1: -(1-z^2)/(1+d*rr)
    t1 = -one_mz2 / (1.0 + d * rr)

    # Term 2: gradient-dependent
    t2 = -xt**2 * (one_mz2 * (47.0 - 7.0 * delta) / 72.0 - 2.0 / 3.0)

    # Term 3: kinetic energy correction
    t3 = -_CF / 2.0 * one_mz2 * (opz83 + omz83)

    # Term 4
    t4 = _AUX4 * one_mz2 * (5.0/2.0 - delta/18.0) * (xs0**2 * opz83 + xs1**2 * omz83)

    # Term 5
    t5 = _AUX5 * one_mz2 * (delta - 11.0) * (xs0**2 * opz113 + xs1**2 * omz113)

    # Term 6: cross-spin terms
    t6 = -_AUX6 * (
        2.0/3.0 * (xs0**2 * opz83 + xs1**2 * omz83)
        - opz2 * xs1**2 * omz83 / 4.0
        - omz2 * xs0**2 * opz83 / 4.0
    )

    return a * (t1 + omega * (t2 + t3 + t4 + t5 + t6))


# GGA_C_LYP (ID 131)
register(FunctionalDef(
    info=FunctionalInfo(
        number=131,
        name='gga_c_lyp',
        family=Family.GGA,
        kind=Kind.CORRELATION,
    ),
    energy_fn=_lyp_energy,
    default_params={
        'a': 0.04918,
        'b': 0.132,
        'c': 0.2533,
        'd': 0.349,
    },
    n_internal=3,
))
