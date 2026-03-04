"""Perdew-Wang 1992 (PW92) LDA correlation functional.

Translated from maple/lda_exc/lda_c_pw.mpl.
This is the foundational LDA correlation used by PBE and many other functionals.

References:
    Perdew & Wang, Phys. Rev. B 45, 13244 (1992)
"""

import jax.numpy as jnp

from .._types import FunctionalInfo, FunctionalDef, Family, Kind
from .._registry import register
from .._utils import f_zeta
from .._numerical import safe_log, safe_sqrt
from .._constants import DEFAULT_ZETA_THRESHOLD


def _g_aux(pp, a, alpha1, beta1, beta2, beta3, beta4, rs):
    """Equation (10) auxiliary: denominator inside log."""
    return (beta1 * jnp.sqrt(rs)
            + beta2 * rs
            + beta3 * rs**1.5
            + beta4 * rs**(pp + 1.0))


def _g(pp, a, alpha1, beta1, beta2, beta3, beta4, rs):
    """Equation (10): parametrization of correlation energy components.

    g(rs) = -2a(1 + alpha1*rs) * log(1 + 1/(2a * g_aux(rs)))
    """
    aux = _g_aux(pp, a, alpha1, beta1, beta2, beta3, beta4, rs)
    return -2.0 * a * (1.0 + alpha1 * rs) * safe_log(1.0 + 1.0 / (2.0 * a * aux))


def _pw92_energy(params, rs, z):
    """PW92 correlation energy density per electron.

    f(rs, zeta) = g1(rs) + zeta^4 * f_zeta(z) * [g2(rs) - g1(rs) + g3(rs)/fz20]
                  - f_zeta(z) * g3(rs) / fz20

    Args:
        params: dict with keys 'pp', 'a', 'alpha1', 'beta1'...'beta4', 'fz20'
                each is array of shape (3,) for [paramagnetic, ferromagnetic, spin-corr]
    """
    pp = params['pp']
    a = params['a']
    alpha1 = params['alpha1']
    beta1 = params['beta1']
    beta2 = params['beta2']
    beta3 = params['beta3']
    beta4 = params['beta4']
    fz20 = params['fz20']

    g1 = _g(pp[0], a[0], alpha1[0], beta1[0], beta2[0], beta3[0], beta4[0], rs)
    g2 = _g(pp[1], a[1], alpha1[1], beta1[1], beta2[1], beta3[1], beta4[1], rs)
    g3 = _g(pp[2], a[2], alpha1[2], beta1[2], beta2[2], beta3[2], beta4[2], rs)

    fz = f_zeta(z)
    z4 = z**4

    return g1 + z4 * fz * (g2 - g1 + g3 / fz20) - fz * g3 / fz20


# Default PW92 parameters
_PW92_PARAMS = {
    'pp':     jnp.array([1.0, 1.0, 1.0]),
    'a':      jnp.array([0.031091, 0.015545, 0.016887]),
    'alpha1': jnp.array([0.21370, 0.20548, 0.11125]),
    'beta1':  jnp.array([7.5957, 14.1189, 10.357]),
    'beta2':  jnp.array([3.5876, 6.1977, 3.6231]),
    'beta3':  jnp.array([1.6382, 3.3662, 0.88026]),
    'beta4':  jnp.array([0.49294, 0.62517, 0.49671]),
    'fz20':   jnp.array(1.709921),
}

# Modified PW parameters (PW_MOD, used in PBE)
_PW92_MOD_PARAMS = {
    'pp':     jnp.array([1.0, 1.0, 1.0]),
    'a':      jnp.array([0.0310907, 0.01554535, 0.0168869]),
    'alpha1': jnp.array([0.21370, 0.20548, 0.11125]),
    'beta1':  jnp.array([7.5957, 14.1189, 10.357]),
    'beta2':  jnp.array([3.5876, 6.1977, 3.6231]),
    'beta3':  jnp.array([1.6382, 3.3662, 0.88026]),
    'beta4':  jnp.array([0.49294, 0.62517, 0.49671]),
    'fz20':   jnp.array(1.709920934161365617563962776245),
}

# Register LDA_C_PW (ID 12)
register(FunctionalDef(
    info=FunctionalInfo(
        number=12,
        name='lda_c_pw',
        family=Family.LDA,
        kind=Kind.CORRELATION,
    ),
    energy_fn=_pw92_energy,
    default_params=_PW92_PARAMS,
    n_internal=0,
))

# Register LDA_C_PW_MOD (ID 13)
register(FunctionalDef(
    info=FunctionalInfo(
        number=13,
        name='lda_c_pw_mod',
        family=Family.LDA,
        kind=Kind.CORRELATION,
    ),
    energy_fn=_pw92_energy,
    default_params=_PW92_MOD_PARAMS,
    n_internal=0,
))
