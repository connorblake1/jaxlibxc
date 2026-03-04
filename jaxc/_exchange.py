"""Universal exchange and kinetic wrappers.

These wrap an enhancement factor function F(xs) into a full exchange
energy density, handling spin decomposition and density screening.

Translation of gga_exchange, mgga_exchange, etc. from util.mpl.
"""

import jax.numpy as jnp

from ._constants import DEFAULT_DENS_THRESHOLD, DEFAULT_ZETA_THRESHOLD
from ._utils import (
    lda_x_spin, lda_k_spin, screen_dens, screen_dens_zeta, z_thr,
)
from ._numerical import my_piecewise3


def gga_exchange(enhance_fn, rs, z, xs0, xs1,
                 dens_threshold=DEFAULT_DENS_THRESHOLD,
                 zeta_threshold=DEFAULT_ZETA_THRESHOLD):
    """Standard GGA exchange: E_x = sum_s [E_x^LDA(n_s) * F(xs_s)].

    enhance_fn: callable (xs) -> F(xs), the enhancement factor.
    rs: Wigner-Seitz radius
    z: spin polarization zeta
    xs0, xs1: per-spin reduced gradients

    Returns: exchange energy density per electron (scalar)
    """
    z_safe = z_thr(z, zeta_threshold)
    z_safe_neg = z_thr(-z, zeta_threshold)

    # Spin-up contribution
    up = my_piecewise3(
        screen_dens(rs, z, dens_threshold),
        0.0,
        lda_x_spin(rs, z_safe, zeta_threshold) * enhance_fn(xs0)
    )
    # Spin-down contribution
    dn = my_piecewise3(
        screen_dens(rs, -z, dens_threshold),
        0.0,
        lda_x_spin(rs, z_safe_neg, zeta_threshold) * enhance_fn(xs1)
    )
    return up + dn


def gga_exchange_nsp(enhance_fn, rs, z, xs0, xs1,
                     dens_threshold=DEFAULT_DENS_THRESHOLD,
                     zeta_threshold=DEFAULT_ZETA_THRESHOLD):
    """Non-separable GGA exchange: enhance_fn(rs, z, xs)."""
    z_safe = z_thr(z, zeta_threshold)
    z_safe_neg = z_thr(-z, zeta_threshold)

    up = my_piecewise3(
        screen_dens(rs, z, dens_threshold),
        0.0,
        lda_x_spin(rs, z_safe, zeta_threshold) * enhance_fn(rs, z_safe, xs0)
    )
    dn = my_piecewise3(
        screen_dens(rs, -z, dens_threshold),
        0.0,
        lda_x_spin(rs, z_safe_neg, zeta_threshold) * enhance_fn(rs, z_safe_neg, xs1)
    )
    return up + dn


def gga_kinetic(enhance_fn, rs, z, xs0, xs1,
                dens_threshold=DEFAULT_DENS_THRESHOLD,
                zeta_threshold=DEFAULT_ZETA_THRESHOLD):
    """GGA kinetic energy: same pattern as exchange but with kinetic base."""
    z_safe = z_thr(z, zeta_threshold)
    z_safe_neg = z_thr(-z, zeta_threshold)

    up = my_piecewise3(
        screen_dens(rs, z, dens_threshold),
        0.0,
        lda_k_spin(rs, z_safe, zeta_threshold) * enhance_fn(xs0)
    )
    dn = my_piecewise3(
        screen_dens(rs, -z, dens_threshold),
        0.0,
        lda_k_spin(rs, z_safe_neg, zeta_threshold) * enhance_fn(xs1)
    )
    return up + dn


def mgga_exchange(enhance_fn, rs, z, xs0, xs1, u0, u1, t0, t1,
                  dens_threshold=DEFAULT_DENS_THRESHOLD,
                  zeta_threshold=DEFAULT_ZETA_THRESHOLD):
    """Standard meta-GGA exchange: enhance_fn(xs, u, t)."""
    z_safe = z_thr(z, zeta_threshold)
    z_safe_neg = z_thr(-z, zeta_threshold)

    up = my_piecewise3(
        screen_dens(rs, z, dens_threshold),
        0.0,
        lda_x_spin(rs, z_safe, zeta_threshold) * enhance_fn(xs0, u0, t0)
    )
    dn = my_piecewise3(
        screen_dens(rs, -z, dens_threshold),
        0.0,
        lda_x_spin(rs, z_safe_neg, zeta_threshold) * enhance_fn(xs1, u1, t1)
    )
    return up + dn


def mgga_exchange_nsp(enhance_fn, rs, z, xs0, xs1, u0, u1, t0, t1,
                      dens_threshold=DEFAULT_DENS_THRESHOLD,
                      zeta_threshold=DEFAULT_ZETA_THRESHOLD):
    """Non-separable meta-GGA exchange: enhance_fn(rs, z, xs, u, t)."""
    z_safe = z_thr(z, zeta_threshold)
    z_safe_neg = z_thr(-z, zeta_threshold)

    up = my_piecewise3(
        screen_dens(rs, z, dens_threshold),
        0.0,
        lda_x_spin(rs, z_safe, zeta_threshold) * enhance_fn(rs, z_safe, xs0, u0, t0)
    )
    dn = my_piecewise3(
        screen_dens(rs, -z, dens_threshold),
        0.0,
        lda_x_spin(rs, z_safe_neg, zeta_threshold) * enhance_fn(rs, z_safe_neg, xs1, u1, t1)
    )
    return up + dn
