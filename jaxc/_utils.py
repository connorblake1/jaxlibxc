"""Core utility functions translated from libxc maple/util.mpl.

These are the building blocks used by all functionals: spin interpolation,
density conversions, screening, and exchange/kinetic base formulas.
"""

import jax.numpy as jnp

from ._constants import (
    RS_FACTOR, DIMENSIONS, LDA_X_FACTOR, K_FACTOR_C, X_FACTOR_C,
    F_ZETA_DENOM, DEFAULT_DENS_THRESHOLD, DEFAULT_ZETA_THRESHOLD,
)
from ._numerical import my_piecewise3, my_piecewise5


# --- Density conversions ---

def r_ws(n):
    """Wigner-Seitz radius from electron density."""
    return RS_FACTOR / jnp.maximum(n, 1e-30) ** (1.0 / DIMENSIONS)


def n_total(rs):
    """Total electron density from Wigner-Seitz radius."""
    return (RS_FACTOR / rs) ** DIMENSIONS


def n_spin(rs, z):
    """Spin density: n_s = (1+z) * n_total / 2."""
    return (1.0 + z) * n_total(rs) / 2.0


# --- Spin polarization functions ---

def opz_pow_n(z, n, zeta_threshold=DEFAULT_ZETA_THRESHOLD):
    """Protected (1+z)^n: avoids 0^n when z ~ -1.

    opz_pow_n(z, n) = (zeta_threshold)^n  if 1+z <= zeta_threshold
                      (1+z)^n             otherwise
    """
    safe_base = jnp.where(1.0 + z <= zeta_threshold,
                          zeta_threshold,
                          1.0 + z)
    return jnp.power(safe_base, n)


def z_thr(z, zeta_threshold=DEFAULT_ZETA_THRESHOLD):
    """Screen extreme values of zeta to avoid singularities.

    z_thr(z) = zeta_threshold - 1   if 1+z <= zeta_threshold
             = 1 - zeta_threshold   if 1-z <= zeta_threshold
             = z                    otherwise
    """
    return my_piecewise5(
        1.0 + z <= zeta_threshold, zeta_threshold - 1.0,
        1.0 - z <= zeta_threshold, 1.0 - zeta_threshold,
        z
    )


def f_zeta(z, zeta_threshold=DEFAULT_ZETA_THRESHOLD):
    """Spin-polarization interpolation function (Perdew 1992, Eq. 9).

    f(z) = [(1+z)^(4/3) + (1-z)^(4/3) - 2] / [2^(4/3) - 2]
    """
    return (opz_pow_n(z, 4.0/3.0, zeta_threshold)
            + opz_pow_n(-z, 4.0/3.0, zeta_threshold)
            - 2.0) / F_ZETA_DENOM


def mphi(z, zeta_threshold=DEFAULT_ZETA_THRESHOLD):
    """Spin scaling factor: phi(z) = [(1+z)^(2/3) + (1-z)^(2/3)] / 2."""
    return (opz_pow_n(z, 2.0/3.0, zeta_threshold)
            + opz_pow_n(-z, 2.0/3.0, zeta_threshold)) / 2.0


def tt(rs, z, xt, zeta_threshold=DEFAULT_ZETA_THRESHOLD):
    """Reduced gradient for correlation: t = xt / (4 * 2^(1/3) * phi(z) * sqrt(rs))."""
    phi = mphi(z, zeta_threshold)
    return xt / (4.0 * 2.0 ** (1.0/3.0) * phi * jnp.sqrt(rs))


# --- LDA base formulas ---

def lda_x_spin(rs, z, zeta_threshold=DEFAULT_ZETA_THRESHOLD):
    """LDA exchange energy density per electron for one spin channel.

    E_x^LDA = LDA_X_FACTOR * (1+z)^(4/3) * 2^(-4/3) * (RS_FACTOR/rs)
    """
    return (LDA_X_FACTOR
            * opz_pow_n(z, 1.0 + 1.0/DIMENSIONS, zeta_threshold)
            * 2.0 ** (-1.0 - 1.0/DIMENSIONS)
            * (RS_FACTOR / rs))


def lda_k_spin(rs, z, zeta_threshold=DEFAULT_ZETA_THRESHOLD):
    """LDA kinetic energy density per electron for one spin channel."""
    return (K_FACTOR_C
            * opz_pow_n(z, 5.0/3.0, zeta_threshold)
            * 2.0 ** (-5.0/3.0)
            * (RS_FACTOR / rs) ** 2)


# --- Screening functions ---

def screen_dens(rs, z, dens_threshold=DEFAULT_DENS_THRESHOLD):
    """Returns True where spin density is below threshold."""
    return n_spin(rs, z) <= dens_threshold


def screen_dens_zeta(rs, z, dens_threshold=DEFAULT_DENS_THRESHOLD,
                     zeta_threshold=DEFAULT_ZETA_THRESHOLD):
    """Screen by both density and extreme zeta values."""
    return (n_spin(rs, z) <= dens_threshold) | (1.0 + z <= zeta_threshold)
