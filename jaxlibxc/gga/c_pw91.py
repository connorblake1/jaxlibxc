"""PW91 correlation functional.

Translated from maple/gga_exc/gga_c_pw91.mpl.
Uses PW92 LDA correlation as base, adds gradient correction H = H0 + H1.

References:
    Perdew, Chevary, Vosko, Jackson, Pederson, Singh, Fiolhais,
    Phys. Rev. B 46, 6671 (1992)
"""

import jax.numpy as jnp
import numpy as np

from .._types import FunctionalInfo, FunctionalDef, Family, Kind
from .._registry import register
from .._utils import mphi, tt
from .._numerical import safe_log
from ..lda.c_pw import _pw92_energy, _PW92_PARAMS


# PW91 constants (from maple source)
_PW91_CC0 = 4.235e-3
_PW91_ALPHA = 0.09
_PW91_NU = 16.0 / np.pi * (3.0 * np.pi**2) ** (1.0 / 3.0)
_PW91_BETA = _PW91_NU * _PW91_CC0

_PW91_C1 = _PW91_BETA**2 / (2.0 * _PW91_ALPHA)
_PW91_C2 = 2.0 * _PW91_ALPHA / _PW91_BETA

# Rasolt-Geldart Pade parametrization of C_xc(rs)
# C_xc(rs) = (a1 + a2*rs + a3*rs^2) / (1000 * (b1 + b2*rs + b3*rs^2))
_RS_A = np.array([2.568, 23.266, 0.007389])
_RS_B = np.array([1.0, 8.723, 0.472])

# H1 constants
_CXC0 = 2.568e-3
_CX = -0.001667
_H_A1 = -100.0 * 4.0 / np.pi * (4.0 / (9.0 * np.pi)) ** (1.0 / 3.0)


def _c_xc(rs):
    """Rasolt-Geldart C_xc(rs) parametrization."""
    return (_RS_A[0] + _RS_A[1] * rs + _RS_A[2] * rs**2) / (
        1000.0 * (_RS_B[0] + _RS_B[1] * rs + _RS_B[2] * rs**2))


def _pw91_c_energy(params, rs, z, xt, xs0, xs1):
    """PW91 correlation energy density per electron.

    f(rs, z, xt) = f_pw(rs, z) + H0(rs, z, t) + H1(rs, z, t)
    """
    # PW92 correlation as base (uses standard params, not modified)
    pw_params = {
        'pp':     params['pw_pp'],
        'a':      params['pw_a'],
        'alpha1': params['pw_alpha1'],
        'beta1':  params['pw_beta1'],
        'beta2':  params['pw_beta2'],
        'beta3':  params['pw_beta3'],
        'beta4':  params['pw_beta4'],
        'fz20':   params['pw_fz20'],
    }
    ec_pw = _pw92_energy(pw_params, rs, z)

    # Reduced gradient and phi
    phi = mphi(z)
    t = tt(rs, z, xt)

    # Equation (14): A(rs, z)
    A = _PW91_C2 / (jnp.exp(-2.0 * _PW91_ALPHA * ec_pw
                              / (phi**3 * _PW91_BETA**2)) - 1.0)

    # Equation (13): H0(rs, z, t)
    t2 = t**2
    t4 = t2**2
    H0 = _PW91_C1 * phi**3 * safe_log(
        1.0 + _PW91_C2 * (t2 + A * t4) / (1.0 + A * t2 + A**2 * t4))

    # Equation (15): H1(rs, z, t)
    H1 = _PW91_NU * (_c_xc(rs) - _CXC0 - 3.0 * _CX / 7.0) * \
        phi**3 * t2 * jnp.exp(_H_A1 * rs * phi**4 * t2)

    return ec_pw + H0 + H1


def _make_pw91_c_params(pw_params):
    """Merge PW92 params with 'pw_' prefix."""
    params = {}
    for k, v in pw_params.items():
        params[f'pw_{k}'] = v
    return params


# GGA_C_PW91 (ID 134)
register(FunctionalDef(
    info=FunctionalInfo(
        number=134,
        name='gga_c_pw91',
        family=Family.GGA,
        kind=Kind.CORRELATION,
    ),
    energy_fn=_pw91_c_energy,
    default_params=_make_pw91_c_params(_PW92_PARAMS),
    n_internal=3,
))
