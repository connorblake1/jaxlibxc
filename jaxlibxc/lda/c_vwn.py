"""Vosko-Wilk-Nusair (VWN) LDA correlation functionals.

Translated from maple/vwn.mpl and maple/lda_exc/lda_c_vwn.mpl.
Includes VWN5 (standard), VWN_RPA, VWN3, and VWN1 variants.

References:
    Vosko, Wilk, Nusair, Can. J. Phys. 58, 1200 (1980)
"""

import jax.numpy as jnp
import numpy as np

from .._types import FunctionalInfo, FunctionalDef, Family, Kind
from .._registry import register
from .._utils import f_zeta
from .._numerical import safe_log


# --- VWN parameters (from vwn.mpl, divided by 2 for Hartree) ---

# VWN5 parameters (QMC fit)
_A_VWN = np.array([0.0310907, 0.01554535, -1.0 / (6.0 * np.pi**2)])
_B_VWN = np.array([3.72744, 7.06042, 1.13107])
_C_VWN = np.array([12.9352, 18.0578, 13.0045])
_X0_VWN = np.array([-0.10498, -0.32500, -0.0047584])

# RPA parameters
_A_RPA = np.array([0.0310907, 0.01554535, -1.0 / (6.0 * np.pi**2)])
_B_RPA = np.array([13.0720, 20.1231, 1.06835])
_C_RPA = np.array([42.7198, 101.578, 11.4813])
_X0_RPA = np.array([-0.409286, -0.743294, -0.228344])

# f''(0) normalization
_FPP_VWN = 4.0 / (9.0 * (2.0**(1.0/3.0) - 1.0))


def _Q(b, c):
    """Q = sqrt(4c - b^2)."""
    return jnp.sqrt(4.0 * c - b**2)


def _fx(b, c, rs):
    """fx(rs) = rs + b*sqrt(rs) + c."""
    return rs + b * jnp.sqrt(rs) + c


def _f_aux(A, b, c, x0, rs):
    """VWN auxiliary function (Padé approximant form).

    f_aux = A * [log(rs/fx) + (f1 - f2*f3)*arctan(Q/(2*sqrt(rs)+b)) - f2*log((sqrt(rs)-x0)^2/fx)]
    """
    Q = _Q(b, c)
    fx = _fx(b, c, rs)
    sqrs = jnp.sqrt(rs)

    f1 = 2.0 * b / Q
    f2 = b * x0 / (x0**2 + b * x0 + c)
    f3 = 2.0 * (2.0 * x0 + b) / Q

    return A * (
        safe_log(rs / fx)
        + (f1 - f2 * f3) * jnp.arctan(Q / (2.0 * sqrs + b))
        - f2 * safe_log((sqrs - x0)**2 / fx)
    )


def _DMC(rs):
    """DMC correction: f_aux(VWN[2]) - f_aux(VWN[1]) (Maple 1-indexed).

    = f_aux(ferromagnetic) - f_aux(paramagnetic) = index 1 - index 0 in Python.
    """
    return (_f_aux(_A_VWN[1], _B_VWN[1], _C_VWN[1], _X0_VWN[1], rs)
            - _f_aux(_A_VWN[0], _B_VWN[0], _C_VWN[0], _X0_VWN[0], rs))


# --- VWN5 (standard VWN) ---

def _vwn5_energy(params, rs, z):
    """VWN5 correlation energy per electron.

    f(rs, z) = f_aux(VWN[1]) + f_aux(VWN[3])*f_zeta(z)*(1-z^4)/fpp
               + DMC(rs)*f_zeta(z)*z^4
    """
    fz = f_zeta(z)
    z4 = z**4

    ec0 = _f_aux(_A_VWN[0], _B_VWN[0], _C_VWN[0], _X0_VWN[0], rs)
    alpha_c = _f_aux(_A_VWN[2], _B_VWN[2], _C_VWN[2], _X0_VWN[2], rs)
    dmc = _DMC(rs)

    return ec0 + alpha_c * fz * (1.0 - z4) / _FPP_VWN + dmc * fz * z4


# --- VWN_RPA ---

def _vwn_rpa_energy(params, rs, z):
    """VWN RPA correlation energy per electron.

    f(rs, z) = f_aux(RPA[1])*(1 - f_zeta(z)) + f_aux(RPA[2])*f_zeta(z)
    """
    fz = f_zeta(z)
    ec_para = _f_aux(_A_RPA[0], _B_RPA[0], _C_RPA[0], _X0_RPA[0], rs)
    ec_ferro = _f_aux(_A_RPA[1], _B_RPA[1], _C_RPA[1], _X0_RPA[1], rs)
    return ec_para * (1.0 - fz) + ec_ferro * fz


# Register VWN5 (ID 7)
register(FunctionalDef(
    info=FunctionalInfo(
        number=7,
        name='lda_c_vwn',
        family=Family.LDA,
        kind=Kind.CORRELATION,
    ),
    energy_fn=_vwn5_energy,
    default_params={},
    n_internal=0,
))

# Register VWN_RPA (ID 8)
register(FunctionalDef(
    info=FunctionalInfo(
        number=8,
        name='lda_c_vwn_rpa',
        family=Family.LDA,
        kind=Kind.CORRELATION,
    ),
    energy_fn=_vwn_rpa_energy,
    default_params={},
    n_internal=0,
))
