"""LDA exchange functional.

Translated from maple/lda_exc/lda_x.mpl.

f(rs, z) = alpha * [piecewise(screen_up, 0, lda_x_spin(rs, z))
                   + piecewise(screen_dn, 0, lda_x_spin(rs, -z))]
"""

import jax.numpy as jnp

from .._types import FunctionalInfo, FunctionalDef, Family, Kind
from .._registry import register
from .._utils import lda_x_spin, screen_dens
from .._numerical import my_piecewise3
from .._constants import DEFAULT_DENS_THRESHOLD, DEFAULT_ZETA_THRESHOLD


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

# Register LDA_X_2D (ID 19)
register(FunctionalDef(
    info=FunctionalInfo(
        number=19,
        name='lda_x_2d',
        family=Family.LDA,
        kind=Kind.EXCHANGE,
    ),
    energy_fn=_lda_x_energy,
    default_params={'alpha': 1.0},
    n_internal=0,
))

# Register Xalpha (ID 550) -- same energy_fn, different alpha
register(FunctionalDef(
    info=FunctionalInfo(
        number=550,
        name='lda_x_slater',
        family=Family.LDA,
        kind=Kind.EXCHANGE,
    ),
    energy_fn=_lda_x_energy,
    default_params={'alpha': 2.0 / 3.0},  # Slater's Xalpha with alpha=2/3
    n_internal=0,
))
