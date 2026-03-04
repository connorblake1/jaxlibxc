"""LDA exchange functional.

Translated from maple/lda_exc/lda_x.mpl and lda_x_2d.mpl.

f(rs, z) = alpha * [piecewise(screen_up, 0, lda_x_spin(rs, z))
                   + piecewise(screen_dn, 0, lda_x_spin(rs, -z))]
"""

import jax.numpy as jnp
import numpy as np

from .._types import FunctionalInfo, FunctionalDef, Family, Kind
from .._registry import register
from .._utils import lda_x_spin, screen_dens, n_total, opz_pow_n
from .._numerical import my_piecewise3
from .._constants import (
    DEFAULT_DENS_THRESHOLD, DEFAULT_ZETA_THRESHOLD, RS_FACTOR, DIMENSIONS,
)


def _lda_x_energy(params, rs, z):
    """LDA exchange energy density per electron.

    Args:
        params: dict with 'alpha' key
        rs: Wigner-Seitz radius
        z: spin polarization zeta

    Returns:
        zk: exchange energy per electron (scalar)
    """
    alpha = params['alpha']
    dens_thr = DEFAULT_DENS_THRESHOLD
    zeta_thr = DEFAULT_ZETA_THRESHOLD

    up = my_piecewise3(
        screen_dens(rs, z, dens_thr),
        0.0,
        lda_x_spin(rs, z, zeta_thr)
    )
    dn = my_piecewise3(
        screen_dens(rs, -z, dens_thr),
        0.0,
        lda_x_spin(rs, -z, zeta_thr)
    )
    return alpha * (up + dn)


# Register LDA_X (ID 1)
register(FunctionalDef(
    info=FunctionalInfo(
        number=1,
        name='lda_x',
        family=Family.LDA,
        kind=Kind.EXCHANGE,
    ),
    energy_fn=_lda_x_energy,
    default_params={'alpha': 1.0},
    n_internal=0,
))

# 2D LDA exchange -- completely different formula from 3D.
# From maple/lda_exc/lda_x_2d.mpl:
#   f = ax * f_zeta_2d(z) / rs_2d
# where ax = -4*sqrt(2)/(3*pi), f_zeta_2d = [(1+z)^(3/2) + (1-z)^(3/2)]/2,
# and rs_2d = 1/(sqrt(pi)*sqrt(n)).
_AX_2D = -4.0 * np.sqrt(2.0) / (3.0 * np.pi)
_RS_FACTOR_2D = 1.0 / np.sqrt(np.pi)


def _lda_x_2d_energy(params, rs, z):
    """2D LDA exchange energy density per electron.

    Uses the 2D formula from libxc: f = ax * f_zeta_2d(z) / rs_2d.
    Since the autodiff engine passes the 3D rs, we convert via density.
    """
    alpha = params['alpha']
    zeta_thr = DEFAULT_ZETA_THRESHOLD

    # Recover density from 3D rs: n = (RS_FACTOR_3D / rs)^3
    n = (RS_FACTOR / rs) ** DIMENSIONS
    # 2D Wigner-Seitz radius: rs_2d = 1 / (sqrt(pi) * sqrt(n))
    rs_2d = _RS_FACTOR_2D / jnp.sqrt(jnp.maximum(n, DEFAULT_DENS_THRESHOLD))
    # f_zeta_2d(z) = [(1+z)^(3/2) + (1-z)^(3/2)] / 2
    fz_2d = (opz_pow_n(z, 1.5, zeta_thr)
             + opz_pow_n(-z, 1.5, zeta_thr)) / 2.0

    return alpha * _AX_2D * fz_2d / rs_2d


# Register LDA_X_2D (ID 19)
register(FunctionalDef(
    info=FunctionalInfo(
        number=19,
        name='lda_x_2d',
        family=Family.LDA,
        kind=Kind.EXCHANGE,
    ),
    energy_fn=_lda_x_2d_energy,
    default_params={'alpha': 1.0},
    n_internal=0,
))

