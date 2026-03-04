"""SCAN (Strongly Constrained and Appropriately Normed) meta-GGA exchange.

Translated from maple/mgga_exc/mgga_x_scan.mpl.

References:
    Sun, Ruzsinszky, Perdew, PRL 115, 036402 (2015)
"""

import jax.numpy as jnp
import numpy as np

from .._types import FunctionalInfo, FunctionalDef, Family, Kind
from .._registry import register
from .._exchange import mgga_exchange
from .._constants import X2S, K_FACTOR_C, MU_GE
from .._numerical import my_piecewise3

# Machine epsilon for cutoff computations
_DBL_EPSILON = np.finfo(np.float64).eps

# Fixed constants from the maple file
_SCAN_A1 = 4.9479
_SCAN_H0X = 1.174
_SCAN_B2 = np.sqrt(5913.0 / 405000.0)
_SCAN_B1 = (511.0 / 13500.0) / (2.0 * _SCAN_B2)
_SCAN_B3 = 0.5


def _scan_enhance(params, xs, u, t):
    """SCAN enhancement factor F(xs, u, t).

    t here is tau_s / n_s^(5/3) (per-spin kinetic energy density).
    """
    c1 = params['c1']
    c2 = params['c2']
    d = params['d']
    k1 = params['k1']

    # Derived constants
    b4 = MU_GE**2 / k1 - 1606.0 / 18225.0 - _SCAN_B1**2

    # p = (X2S * xs)^2
    p = X2S**2 * xs**2

    # alpha = (t - xs^2/8) / K_FACTOR_C
    alpha = (t - xs**2 / 8.0) / K_FACTOR_C

    # f_alpha interpolation (piecewise with exponential cutoff)
    log_eps = -jnp.log(_DBL_EPSILON)

    # Left branch: exp(-c1 * a / (1 - a)) for a <= 1
    left_cutoff = log_eps / (log_eps + c1)
    a_left = jnp.minimum(alpha, left_cutoff)
    # Clamp denominator to avoid division by zero at a=1
    denom_left = jnp.maximum(1.0 - a_left, 1e-30)
    f_alpha_left0 = jnp.exp(-c1 * a_left / denom_left)
    f_alpha_left = jnp.where(alpha > left_cutoff, 0.0, f_alpha_left0)

    # Right branch: -d * exp(c2 / (1 - a)) for a > 1
    right_cutoff = (-jnp.log(_DBL_EPSILON / jnp.abs(d)) + c2) / (-jnp.log(_DBL_EPSILON / jnp.abs(d)))
    a_right = jnp.maximum(alpha, right_cutoff)
    denom_right = jnp.minimum(1.0 - a_right, -1e-30)
    f_alpha_right0 = -d * jnp.exp(c2 / denom_right)
    f_alpha_right = jnp.where(alpha < right_cutoff, 0.0, f_alpha_right0)

    f_alpha = jnp.where(alpha <= 1.0, f_alpha_left, f_alpha_right)

    # h1x(y) = 1 + k1 * (1 - k1 / (k1 + y))
    y = (MU_GE * p
         + b4 * p**2 * jnp.exp(-b4 * p / MU_GE)
         + (_SCAN_B1 * p + _SCAN_B2 * (1.0 - alpha) * jnp.exp(-_SCAN_B3 * (1.0 - alpha)**2))**2)
    h1x = 1.0 + k1 * (1.0 - k1 / (k1 + y))

    # gx(x) = 1 - exp(-a1 / sqrt(X2S * x))
    gx = 1.0 - jnp.exp(-_SCAN_A1 / jnp.sqrt(jnp.maximum(X2S * xs, 1e-30)))

    return (h1x * (1.0 - f_alpha) + _SCAN_H0X * f_alpha) * gx


def _scan_x_energy(params, rs, z, xt, xs0, xs1, u0, u1, t0, t1):
    """SCAN exchange energy density per electron."""
    enhance = lambda xs, u, t: _scan_enhance(params, xs, u, t)
    return mgga_exchange(enhance, rs, z, xs0, xs1, u0, u1, t0, t1)


# MGGA_X_SCAN (ID 263)
register(FunctionalDef(
    info=FunctionalInfo(
        number=263,
        name='mgga_x_scan',
        family=Family.MGGA,
        kind=Kind.EXCHANGE,
    ),
    energy_fn=_scan_x_energy,
    default_params={
        'c1': 0.667,
        'c2': 0.8,
        'd': 1.24,
        'k1': 0.065,
    },
    n_internal=7,
))

# revSCAN exchange (ID 581)
register(FunctionalDef(
    info=FunctionalInfo(
        number=581,
        name='mgga_x_revscan',
        family=Family.MGGA,
        kind=Kind.EXCHANGE,
    ),
    energy_fn=_scan_x_energy,
    default_params={
        'c1': 0.607,
        'c2': 0.7,
        'd': 1.37,
        'k1': 0.065,
    },
    n_internal=7,
))
