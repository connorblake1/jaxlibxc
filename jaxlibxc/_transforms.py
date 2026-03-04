"""Variable transformations from user inputs to internal variables.

Converts (rho, sigma, lapl, tau) -> (rs, zeta, xt, xs0, xs1, u0, u1, t0, t1)
with density thresholding matching libxc's work_{lda,gga,mgga}_inc.c.
"""

import jax.numpy as jnp

from ._constants import (
    RS_FACTOR, DIMENSIONS, DEFAULT_DENS_THRESHOLD,
    DEFAULT_SIGMA_THRESHOLD, DEFAULT_TAU_THRESHOLD,
)


def transform_lda_unpol(rho, dens_threshold=DEFAULT_DENS_THRESHOLD):
    """Transform unpolarized LDA inputs to (rs, zeta=0).

    Args:
        rho: array of shape (N,) -- total electron density

    Returns:
        rs: Wigner-Seitz radius, shape (N,)
        zeta: spin polarization (all zeros), shape (N,)
        mask: boolean mask where density is above threshold
    """
    n = jnp.maximum(rho, dens_threshold)
    rs = RS_FACTOR / n ** (1.0 / DIMENSIONS)
    zeta = jnp.zeros_like(rs)
    mask = rho >= dens_threshold
    return rs, zeta, mask


def transform_lda_pol(rho, dens_threshold=DEFAULT_DENS_THRESHOLD):
    """Transform polarized LDA inputs to (rs, zeta).

    Args:
        rho: array of shape (N, 2) -- [rho_up, rho_down]

    Returns:
        rs, zeta, mask
    """
    rho_up = jnp.maximum(rho[:, 0], dens_threshold)
    rho_dn = jnp.maximum(rho[:, 1], dens_threshold)
    n = rho_up + rho_dn
    rs = RS_FACTOR / n ** (1.0 / DIMENSIONS)
    zeta = (rho_up - rho_dn) / n
    mask = (rho[:, 0] + rho[:, 1]) >= dens_threshold
    return rs, zeta, mask


def transform_gga_unpol(rho, sigma,
                        dens_threshold=DEFAULT_DENS_THRESHOLD,
                        sigma_threshold=DEFAULT_SIGMA_THRESHOLD):
    """Transform unpolarized GGA inputs to (rs, zeta, xt, xs0, xs1).

    Args:
        rho: (N,) total density
        sigma: (N,) squared gradient magnitude

    Returns:
        rs, zeta, xt, xs0, xs1, mask
    """
    n = jnp.maximum(rho, dens_threshold)
    sigma_safe = jnp.maximum(sigma, sigma_threshold**2)

    rs = RS_FACTOR / n ** (1.0 / DIMENSIONS)
    zeta = jnp.zeros_like(rs)

    # Reduced gradient: x = |grad n| / n^(4/3)
    xt = jnp.sqrt(sigma_safe) / n ** (4.0 / 3.0)
    # For unpolarized: xs0 = xs1 = xt / 2^(1/3) (per-spin reduced gradient)
    xs0 = xt * 2.0 ** (1.0 / 3.0)
    xs1 = xs0

    mask = rho >= dens_threshold
    return rs, zeta, xt, xs0, xs1, mask


def transform_gga_pol(rho, sigma,
                      dens_threshold=DEFAULT_DENS_THRESHOLD,
                      sigma_threshold=DEFAULT_SIGMA_THRESHOLD):
    """Transform polarized GGA inputs to (rs, zeta, xt, xs0, xs1).

    Args:
        rho: (N, 2) [rho_up, rho_down]
        sigma: (N, 3) [sigma_uu, sigma_ud, sigma_dd]

    Returns:
        rs, zeta, xt, xs0, xs1, mask
    """
    rho_up = jnp.maximum(rho[:, 0], dens_threshold)
    rho_dn = jnp.maximum(rho[:, 1], dens_threshold)
    n = rho_up + rho_dn

    sigma_uu = jnp.maximum(sigma[:, 0], sigma_threshold**2)
    sigma_dd = jnp.maximum(sigma[:, 2], sigma_threshold**2)
    # Clamp cross-term by Cauchy-Schwarz
    s_ave = 0.5 * (sigma_uu + sigma_dd)
    sigma_ud = jnp.clip(sigma[:, 1], -s_ave, s_ave)

    rs = RS_FACTOR / n ** (1.0 / DIMENSIONS)
    zeta = (rho_up - rho_dn) / n

    # Total reduced gradient: xt = |grad n| / n^(4/3)
    sigma_tot = sigma_uu + 2.0 * sigma_ud + sigma_dd
    sigma_tot = jnp.maximum(sigma_tot, sigma_threshold**2)
    xt = jnp.sqrt(sigma_tot) / n ** (4.0 / 3.0)

    # Per-spin reduced gradients: xs_s = |grad n_s| / n_s^(4/3)
    xs0 = jnp.sqrt(sigma_uu) / rho_up ** (4.0 / 3.0)
    xs1 = jnp.sqrt(sigma_dd) / rho_dn ** (4.0 / 3.0)

    mask = (rho[:, 0] + rho[:, 1]) >= dens_threshold
    return rs, zeta, xt, xs0, xs1, mask


def transform_mgga_unpol(rho, sigma, lapl, tau,
                         dens_threshold=DEFAULT_DENS_THRESHOLD,
                         sigma_threshold=DEFAULT_SIGMA_THRESHOLD,
                         tau_threshold=DEFAULT_TAU_THRESHOLD):
    """Transform unpolarized MGGA inputs.

    Returns:
        rs, zeta, xt, xs0, xs1, u0, u1, t0, t1, mask
    """
    n = jnp.maximum(rho, dens_threshold)
    tau_safe = jnp.maximum(tau, tau_threshold)

    # Enforce Fermi hole curvature: sigma <= 8*rho*tau
    sigma_clamped = jnp.minimum(jnp.maximum(sigma, sigma_threshold**2),
                                8.0 * n * tau_safe)

    rs, zeta, xt, xs0, xs1, mask = transform_gga_unpol(
        rho, sigma_clamped, dens_threshold, sigma_threshold)

    # Per-spin kinetic energy density: ts = tau_s / n_s^(5/3)
    n_half = n / 2.0
    t0 = tau_safe / (2.0 * n_half ** (5.0 / 3.0))
    t1 = t0

    # Laplacian: us = lapl_s / n_s^(5/3)
    u0 = lapl / (2.0 * n_half ** (5.0 / 3.0))
    u1 = u0

    return rs, zeta, xt, xs0, xs1, u0, u1, t0, t1, mask


def transform_mgga_pol(rho, sigma, lapl, tau,
                       dens_threshold=DEFAULT_DENS_THRESHOLD,
                       sigma_threshold=DEFAULT_SIGMA_THRESHOLD,
                       tau_threshold=DEFAULT_TAU_THRESHOLD):
    """Transform polarized MGGA inputs.

    Returns:
        rs, zeta, xt, xs0, xs1, u0, u1, t0, t1, mask
    """
    rho_up = jnp.maximum(rho[:, 0], dens_threshold)
    rho_dn = jnp.maximum(rho[:, 1], dens_threshold)

    tau0 = jnp.maximum(tau[:, 0], tau_threshold)
    tau1 = jnp.maximum(tau[:, 1], tau_threshold)

    # Enforce Fermi hole curvature: sigma_ss <= 8*rho_s*tau_s
    sigma_clamped = sigma.copy()
    sigma_uu = jnp.minimum(jnp.maximum(sigma[:, 0], sigma_threshold**2),
                           8.0 * rho_up * tau0)
    sigma_dd = jnp.minimum(jnp.maximum(sigma[:, 2], sigma_threshold**2),
                           8.0 * rho_dn * tau1)
    sigma_clamped = sigma_clamped.at[:, 0].set(sigma_uu)
    sigma_clamped = sigma_clamped.at[:, 2].set(sigma_dd)

    rs, zeta, xt, xs0, xs1, mask = transform_gga_pol(
        rho, sigma_clamped, dens_threshold, sigma_threshold)

    t0 = tau0 / rho_up ** (5.0 / 3.0)
    t1 = tau1 / rho_dn ** (5.0 / 3.0)

    u0 = lapl[:, 0] / rho_up ** (5.0 / 3.0)
    u1 = lapl[:, 1] / rho_dn ** (5.0 / 3.0)

    return rs, zeta, xt, xs0, xs1, u0, u1, t0, t1, mask
