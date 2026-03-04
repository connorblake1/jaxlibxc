"""SCAN meta-GGA correlation functional.

Translated from maple/mgga_exc/mgga_c_scan.mpl + includes.

The SCAN correlation interpolates between PBE correlation (with Hu-Langreth
beta) and a special e0 term via the f_alpha switching function.

f = f_pbe + f_alpha(alpha_c) * (e0 - f_pbe)

References:
    Sun, Ruzsinszky, Perdew, PRL 115, 036402 (2015)
"""

import jax.numpy as jnp
import numpy as np

from .._types import FunctionalInfo, FunctionalDef, Family, Kind
from .._registry import register
from .._utils import mphi, tt, f_zeta, opz_pow_n
from .._numerical import safe_log, my_piecewise3
from .._constants import (
    X2S, K_FACTOR_C, MU_GE, DEFAULT_ZETA_THRESHOLD,
)
from ..lda.c_pw import _pw92_energy, _PW92_MOD_PARAMS

_DBL_EPSILON = np.finfo(np.float64).eps

# --- PBE correlation with Hu-Langreth beta (from gga_c_regtpss.mpl) ---

_BETA_A = 0.066724550603149220
_BETA_B = 0.1
_BETA_C = 0.1778
_GAMMA_PBE = (1.0 - np.log(2.0)) / np.pi**2


def _pbe_corr_HL(pw_params, rs, z, xt):
    """PBE correlation with Hu-Langreth beta(rs) and SCAN-modified f2.

    In the SCAN include chain, gga_c_scan_e0.mpl redefines f2 to:
      f2 = beta * (1 - g(rs, z, t)) / (gamma * A)
    where g = (1 + 4*A*t^2)^(-1/4), replacing the standard PBE f2.
    """
    ec_pw = _pw92_energy(pw_params, rs, z)
    phi = mphi(z)
    t = tt(rs, z, xt)

    # Hu-Langreth beta
    beta = _BETA_A * (1.0 + _BETA_B * rs) / (1.0 + _BETA_C * rs)
    gamma = _GAMMA_PBE

    A = beta / (gamma * (jnp.exp(-ec_pw / (gamma * phi**3)) - 1.0))

    # SCAN-modified f2: uses scan_e0_g instead of standard PBE
    scan_e0_g = (1.0 + 4.0 * A * t**2) ** (-0.25)
    f2 = beta * (1.0 - scan_e0_g) / (gamma * A)

    H = gamma * phi**3 * safe_log(1.0 + f2)

    return ec_pw + H


# --- SCAN e0 term (from gga_c_scan_e0.mpl) ---

_SCAN_B1C = 0.0285764
_SCAN_B2C = 0.0889
_SCAN_B3C = 0.125541
_SCAN_CHI_INFTY = 0.12802585262625815
_SCAN_G_CNST = 2.363


def _scan_eclda0(rs):
    """SCAN LDA-like correlation: -b1c / (1 + b2c*sqrt(rs) + b3c*rs)."""
    return -_SCAN_B1C / (1.0 + _SCAN_B2C * jnp.sqrt(rs) + _SCAN_B3C * rs)


def _scan_g_infty(s):
    """g_infty(s) = 1 / (1 + 4*chi_infty*s^2)^(1/4)."""
    return 1.0 / (1.0 + 4.0 * _SCAN_CHI_INFTY * s**2)**0.25


def _scan_Gc(z):
    """Gc(z) = (1 - G_cnst * (2^(1/3) - 1) * f_zeta(z)) * (1 - z^12)."""
    return (1.0 - _SCAN_G_CNST * (2.0**(1.0/3.0) - 1.0) * f_zeta(z)) * (1.0 - z**12)


def _scan_e0(pw_params, rs, z, xt):
    """SCAN e0 term: (eclda0 + H0) * Gc.

    Also uses PBE correlation machinery for the scan_e0_g function.
    """
    eclda0 = _scan_eclda0(rs)
    s = X2S * 2.0**(1.0/3.0) * xt  # reduced gradient scaled

    # H0 = b1c * log(1 + (exp(-eclda0/b1c) - 1) * (1 - g_infty(s)))
    H0 = _SCAN_B1C * safe_log(
        1.0 + (jnp.exp(-eclda0 / _SCAN_B1C) - 1.0) * (1.0 - _scan_g_infty(s))
    )

    return (eclda0 + H0) * _scan_Gc(z)


# --- f_alpha switching function (reused from SCAN exchange) ---

def _scan_f_alpha(alpha, c1, c2, d):
    """SCAN f_alpha interpolation function."""
    log_eps = -jnp.log(_DBL_EPSILON)

    # Left branch
    left_cutoff = log_eps / (log_eps + c1)
    a_left = jnp.minimum(alpha, left_cutoff)
    denom_left = jnp.maximum(1.0 - a_left, 1e-30)
    f_left0 = jnp.exp(-c1 * a_left / denom_left)
    f_left = jnp.where(alpha > left_cutoff, 0.0, f_left0)

    # Right branch
    right_cutoff = (-jnp.log(_DBL_EPSILON / jnp.abs(d)) + c2) / (-jnp.log(_DBL_EPSILON / jnp.abs(d)))
    a_right = jnp.maximum(alpha, right_cutoff)
    denom_right = jnp.minimum(1.0 - a_right, -1e-30)
    f_right0 = -d * jnp.exp(c2 / denom_right)
    f_right = jnp.where(alpha < right_cutoff, 0.0, f_right0)

    return jnp.where(alpha <= 1.0, f_left, f_right)


# --- SCAN correlation energy ---

def _scan_c_energy(params, rs, z, xt, xs0, xs1, u0, u1, t0, t1):
    """SCAN correlation energy per electron.

    f = f_pbe(rs, z, xt) + f_alpha(alpha_c) * (e0 - f_pbe)

    alpha_c = (t_total - xt^2/8) / (K_FACTOR_C * t_total(z, 1, 1))
    where t_total = ts0*(1+z)^(5/3)/2^(5/3) + ts1*(1-z)^(5/3)/2^(5/3)
    """
    pw_params = _PW92_MOD_PARAMS

    # PBE correlation with Hu-Langreth beta
    f_pbe = _pbe_corr_HL(pw_params, rs, z, xt)

    # SCAN e0 term
    e0 = _scan_e0(pw_params, rs, z, xt)

    # alpha for correlation
    # t_total(z, ts0, ts1) = ts0*((1+z)/2)^(5/3) + ts1*((1-z)/2)^(5/3)
    opz53 = opz_pow_n(z, 5.0/3.0) / 2.0**(5.0/3.0)
    omz53 = opz_pow_n(-z, 5.0/3.0) / 2.0**(5.0/3.0)
    t_tot = t0 * opz53 + t1 * omz53

    # t_total(z, 1, 1) for uniform case
    t_unif = opz53 + omz53

    alpha_c = (t_tot - xt**2 / 8.0) / (K_FACTOR_C * t_unif)

    # f_alpha with SCAN correlation parameters
    c1 = 0.64
    c2 = 1.5
    d = 0.7
    fa = _scan_f_alpha(alpha_c, c1, c2, d)

    return f_pbe + fa * (e0 - f_pbe)


# MGGA_C_SCAN (ID 267)
register(FunctionalDef(
    info=FunctionalInfo(
        number=267,
        name='mgga_c_scan',
        family=Family.MGGA,
        kind=Kind.CORRELATION,
    ),
    energy_fn=_scan_c_energy,
    default_params={},
    n_internal=7,
))
