"""PBE correlation functional.

Translated from maple/gga_exc/gga_c_pbe.mpl.
Includes PW92 LDA correlation as base, adds gradient correction.

E_c^PBE = E_c^PW(rs, z) + H(rs, z, t)

References:
    Perdew, Burke, Ernzerhof, PRL 77, 3865 (1996)
"""

import jax.numpy as jnp
import numpy as np

from .._types import FunctionalInfo, FunctionalDef, Family, Kind
from .._registry import register
from .._utils import mphi, tt, f_zeta
from .._numerical import safe_log
from ..lda.c_pw import _pw92_energy, _PW92_MOD_PARAMS


def _pbe_c_energy(params, rs, z, xt, xs0, xs1):
    """PBE correlation energy density per electron.

    f(rs, z, xt) = f_pw(rs, z) + H(rs, z, t)

    where t = xt / (4 * 2^(1/3) * phi(z) * sqrt(rs)) is the reduced gradient
    and H is the gradient correction.
    """
    beta = params['beta']
    gamma = params['gamma']
    BB = params['BB']

    # PW92 correlation as base (uses modified params)
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

    # Reduced gradient t
    phi = mphi(z)
    t = tt(rs, z, xt)

    # Equation (8): A(rs, z, t)
    A = beta / (gamma * (jnp.exp(-ec_pw / (gamma * phi**3)) - 1.0))

    # Equation (7) components
    t2 = t**2
    f1 = t2 + BB * A * t2**2
    f2 = beta * f1 / (gamma * (1.0 + A * f1))

    # Equation (7): H = gamma * phi^3 * log(1 + f2)
    H = gamma * phi**3 * safe_log(1.0 + f2)

    return ec_pw + H


# Merge PW92_MOD params into PBE params with 'pw_' prefix
def _make_pbe_c_params(beta, gamma, BB, pw_params):
    params = {'beta': beta, 'gamma': gamma, 'BB': BB}
    for k, v in pw_params.items():
        params[f'pw_{k}'] = v
    return params


_GAMMA_PBE = (1.0 - np.log(2.0)) / np.pi**2

# GGA_C_PBE (ID 130)
register(FunctionalDef(
    info=FunctionalInfo(
        number=130,
        name='gga_c_pbe',
        family=Family.GGA,
        kind=Kind.CORRELATION,
    ),
    energy_fn=_pbe_c_energy,
    default_params=_make_pbe_c_params(
        beta=0.06672455060314922,
        gamma=_GAMMA_PBE,
        BB=1.0,
        pw_params=_PW92_MOD_PARAMS,
    ),
    n_internal=3,
))

# GGA_C_PBE_SOL (ID 133) -- same beta/gamma but different is same actually
register(FunctionalDef(
    info=FunctionalInfo(
        number=133,
        name='gga_c_pbe_sol',
        family=Family.GGA,
        kind=Kind.CORRELATION,
    ),
    energy_fn=_pbe_c_energy,
    default_params=_make_pbe_c_params(
        beta=0.046,
        gamma=_GAMMA_PBE,
        BB=1.0,
        pw_params=_PW92_MOD_PARAMS,
    ),
    n_internal=3,
))
