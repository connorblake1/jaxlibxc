"""Automatic differentiation engine for computing XC derivatives.

Given an energy_fn that returns zk (energy per electron), this module
computes all derivatives (vrho, vsigma, vtau, v2rho2, ...) using
jax.grad and jax.jacfwd.
"""

import jax
import jax.numpy as jnp
from functools import partial

from ._types import Family


def _exc_per_volume_lda(rho_pt, energy_fn, params,
                        dens_threshold, zeta_threshold, polarized):
    """Energy per unit volume at a single grid point (LDA).

    rho_pt: scalar (unpol) or (2,) array (pol)
    Returns: n * eps_xc (energy per volume)
    """
    if polarized:
        rho_up = rho_pt[0]
        rho_dn = rho_pt[1]
        n = rho_up + rho_dn
        n_safe = jnp.maximum(n, dens_threshold)
        from ._constants import RS_FACTOR, DIMENSIONS
        rs = RS_FACTOR / n_safe ** (1.0 / DIMENSIONS)
        zeta = (jnp.maximum(rho_up, dens_threshold)
                - jnp.maximum(rho_dn, dens_threshold)) / n_safe
    else:
        n = rho_pt
        n_safe = jnp.maximum(n, dens_threshold)
        from ._constants import RS_FACTOR, DIMENSIONS
        rs = RS_FACTOR / n_safe ** (1.0 / DIMENSIONS)
        zeta = 0.0

    eps_xc = energy_fn(params, rs, zeta)
    return n_safe * eps_xc


def _exc_per_volume_gga(rho_pt, sigma_pt, energy_fn, params,
                        dens_threshold, sigma_threshold,
                        zeta_threshold, polarized):
    """Energy per unit volume at a single grid point (GGA).

    rho_pt: scalar or (2,) array
    sigma_pt: scalar or (3,) array [sigma_uu, sigma_ud, sigma_dd]
    """
    from ._constants import RS_FACTOR, DIMENSIONS

    if polarized:
        rho_up = jnp.maximum(rho_pt[0], dens_threshold)
        rho_dn = jnp.maximum(rho_pt[1], dens_threshold)
        n = rho_up + rho_dn

        sigma_uu = jnp.maximum(sigma_pt[0], sigma_threshold**2)
        sigma_dd = jnp.maximum(sigma_pt[2], sigma_threshold**2)
        # Clip sigma_ud by Cauchy-Schwarz; use stop_gradient on bounds
        # since libxc treats the clip as preprocessing.
        # Use jnp.where instead of jnp.clip to get correct gradient at boundary
        s_ave = jax.lax.stop_gradient(0.5 * (sigma_uu + sigma_dd))
        sigma_ud = jnp.where(sigma_pt[1] > s_ave, s_ave,
                             jnp.where(sigma_pt[1] < -s_ave, -s_ave, sigma_pt[1]))

        rs = RS_FACTOR / n ** (1.0 / DIMENSIONS)
        zeta = (rho_up - rho_dn) / n

        sigma_tot = sigma_uu + 2.0 * sigma_ud + sigma_dd
        sigma_tot = jnp.maximum(sigma_tot, sigma_threshold**2)
        xt = jnp.sqrt(sigma_tot) / n ** (4.0 / 3.0)
        xs0 = jnp.sqrt(sigma_uu) / rho_up ** (4.0 / 3.0)
        xs1 = jnp.sqrt(sigma_dd) / rho_dn ** (4.0 / 3.0)
    else:
        n = jnp.maximum(rho_pt, dens_threshold)
        sigma_safe = jnp.maximum(sigma_pt, sigma_threshold**2)

        rs = RS_FACTOR / n ** (1.0 / DIMENSIONS)
        zeta = 0.0
        xt = jnp.sqrt(sigma_safe) / n ** (4.0 / 3.0)
        xs0 = xt * 2.0 ** (1.0 / 3.0)
        xs1 = xs0

    eps_xc = energy_fn(params, rs, zeta, xt, xs0, xs1)
    return n * eps_xc


def _exc_per_volume_mgga(rho_pt, sigma_pt, lapl_pt, tau_pt,
                         energy_fn, params,
                         dens_threshold, sigma_threshold,
                         tau_threshold, zeta_threshold, polarized):
    """Energy per unit volume at a single grid point (MGGA)."""
    from ._constants import RS_FACTOR, DIMENSIONS

    if polarized:
        rho_up = jnp.maximum(rho_pt[0], dens_threshold)
        rho_dn = jnp.maximum(rho_pt[1], dens_threshold)
        n = rho_up + rho_dn

        sigma_uu = jnp.maximum(sigma_pt[0], sigma_threshold**2)
        sigma_dd = jnp.maximum(sigma_pt[2], sigma_threshold**2)

        tau0 = jnp.maximum(tau_pt[0], tau_threshold)
        tau1 = jnp.maximum(tau_pt[1], tau_threshold)

        # Enforce Fermi hole curvature: sigma_ss <= 8*rho_s*tau_s
        # Use stop_gradient on the bound so derivatives don't flow through the clamp
        bound_uu = jax.lax.stop_gradient(8.0 * rho_up * tau0)
        bound_dd = jax.lax.stop_gradient(8.0 * rho_dn * tau1)
        sigma_uu = jnp.minimum(sigma_uu, bound_uu)
        sigma_dd = jnp.minimum(sigma_dd, bound_dd)

        s_ave = jax.lax.stop_gradient(0.5 * (sigma_uu + sigma_dd))
        sigma_ud = jnp.clip(sigma_pt[1], -s_ave, s_ave)

        rs = RS_FACTOR / n ** (1.0 / DIMENSIONS)
        zeta = (rho_up - rho_dn) / n

        sigma_tot = sigma_uu + 2.0 * sigma_ud + sigma_dd
        sigma_tot = jnp.maximum(sigma_tot, sigma_threshold**2)
        xt = jnp.sqrt(sigma_tot) / n ** (4.0 / 3.0)
        xs0 = jnp.sqrt(sigma_uu) / rho_up ** (4.0 / 3.0)
        xs1 = jnp.sqrt(sigma_dd) / rho_dn ** (4.0 / 3.0)
        t0 = tau0 / rho_up ** (5.0 / 3.0)
        t1 = tau1 / rho_dn ** (5.0 / 3.0)
        u0 = lapl_pt[0] / rho_up ** (5.0 / 3.0)
        u1 = lapl_pt[1] / rho_dn ** (5.0 / 3.0)
    else:
        n = jnp.maximum(rho_pt, dens_threshold)
        sigma_safe = jnp.maximum(sigma_pt, sigma_threshold**2)
        tau_safe = jnp.maximum(tau_pt, tau_threshold)

        # Enforce Fermi hole curvature: sigma <= 8*rho*tau
        # Use stop_gradient on the bound so derivatives don't flow through the clamp
        sigma_safe = jnp.minimum(sigma_safe, jax.lax.stop_gradient(8.0 * n * tau_safe))

        rs = RS_FACTOR / n ** (1.0 / DIMENSIONS)
        zeta = 0.0
        xt = jnp.sqrt(sigma_safe) / n ** (4.0 / 3.0)
        xs0 = xt * 2.0 ** (1.0 / 3.0)
        xs1 = xs0

        n_half = n / 2.0
        t0 = tau_safe / (2.0 * n_half ** (5.0 / 3.0))
        t1 = t0
        u0 = lapl_pt / (2.0 * n_half ** (5.0 / 3.0))
        u1 = u0

    eps_xc = energy_fn(params, rs, zeta, xt, xs0, xs1, u0, u1, t0, t1)
    return n * eps_xc


def compute_exc(energy_fn, params, family, polarized, inputs, thresholds):
    """Compute energy density zk = eps_xc (per electron).

    Args:
        energy_fn: the functional's energy function
        params: parameter dict
        family: Family enum
        polarized: bool
        inputs: dict with 'rho', optionally 'sigma', 'lapl', 'tau'
        thresholds: dict with 'dens', 'sigma', 'tau', 'zeta' thresholds

    Returns:
        zk: array of shape (N, 1)
    """
    dens_thr = thresholds['dens']
    zeta_thr = thresholds['zeta']

    rho = inputs['rho']

    if family == Family.LDA:
        if polarized:
            N = rho.shape[0]
            def _point(rho_pt):
                return _exc_per_volume_lda(
                    rho_pt, energy_fn, params, dens_thr, zeta_thr, True)
            exc_vol = jax.vmap(_point)(rho)
            n = rho[:, 0] + rho[:, 1]
        else:
            N = rho.shape[0]
            def _point(rho_pt):
                return _exc_per_volume_lda(
                    rho_pt, energy_fn, params, dens_thr, zeta_thr, False)
            exc_vol = jax.vmap(_point)(rho)
            n = rho

        n_safe = jnp.maximum(n, dens_thr)
        zk = exc_vol / n_safe
        mask = n >= dens_thr
        zk = jnp.where(mask, zk, 0.0)
        return zk.reshape(-1, 1)

    elif family == Family.GGA:
        sigma = inputs['sigma']
        sigma_thr = thresholds['sigma']

        if polarized:
            def _point(rho_pt, sigma_pt):
                return _exc_per_volume_gga(
                    rho_pt, sigma_pt, energy_fn, params,
                    dens_thr, sigma_thr, zeta_thr, True)
            exc_vol = jax.vmap(_point)(rho, sigma)
            n = rho[:, 0] + rho[:, 1]
        else:
            def _point(rho_pt, sigma_pt):
                return _exc_per_volume_gga(
                    rho_pt, sigma_pt, energy_fn, params,
                    dens_thr, sigma_thr, zeta_thr, False)
            exc_vol = jax.vmap(_point)(rho, sigma)
            n = rho

        n_safe = jnp.maximum(n, dens_thr)
        zk = exc_vol / n_safe
        mask = n >= dens_thr
        zk = jnp.where(mask, zk, 0.0)
        return zk.reshape(-1, 1)

    elif family == Family.MGGA:
        sigma = inputs['sigma']
        lapl = inputs.get('lapl', jnp.zeros_like(inputs['tau']))
        tau = inputs['tau']
        sigma_thr = thresholds['sigma']
        tau_thr = thresholds['tau']

        if polarized:
            def _point(rho_pt, sigma_pt, lapl_pt, tau_pt):
                return _exc_per_volume_mgga(
                    rho_pt, sigma_pt, lapl_pt, tau_pt,
                    energy_fn, params,
                    dens_thr, sigma_thr, tau_thr, zeta_thr, True)
            exc_vol = jax.vmap(_point)(rho, sigma, lapl, tau)
            n = rho[:, 0] + rho[:, 1]
        else:
            def _point(rho_pt, sigma_pt, lapl_pt, tau_pt):
                return _exc_per_volume_mgga(
                    rho_pt, sigma_pt, lapl_pt, tau_pt,
                    energy_fn, params,
                    dens_thr, sigma_thr, tau_thr, zeta_thr, False)
            exc_vol = jax.vmap(_point)(rho, sigma, lapl, tau)
            n = rho

        n_safe = jnp.maximum(n, dens_thr)
        zk = exc_vol / n_safe
        mask = n >= dens_thr
        zk = jnp.where(mask, zk, 0.0)
        return zk.reshape(-1, 1)


def compute_vxc_lda(energy_fn, params, polarized, inputs, thresholds):
    """Compute 1st derivatives for LDA: vrho.

    Returns dict with 'vrho'.
    """
    dens_thr = thresholds['dens']
    zeta_thr = thresholds['zeta']
    rho = inputs['rho']

    if polarized:
        def _point(rho_pt):
            return _exc_per_volume_lda(
                rho_pt, energy_fn, params, dens_thr, zeta_thr, True)
        vrho = jax.vmap(jax.grad(_point))(rho)
        mask = (rho[:, 0] + rho[:, 1]) >= dens_thr
        vrho = jnp.where(mask[:, None], vrho, 0.0)
    else:
        def _point(rho_pt):
            return _exc_per_volume_lda(
                rho_pt, energy_fn, params, dens_thr, zeta_thr, False)
        vrho = jax.vmap(jax.grad(_point))(rho)
        mask = rho >= dens_thr
        vrho = jnp.where(mask, vrho, 0.0)
        vrho = vrho.reshape(-1, 1)

    return {'vrho': vrho}


def compute_vxc_gga(energy_fn, params, polarized, inputs, thresholds):
    """Compute 1st derivatives for GGA: vrho, vsigma."""
    dens_thr = thresholds['dens']
    sigma_thr = thresholds['sigma']
    zeta_thr = thresholds['zeta']
    rho = inputs['rho']
    sigma = inputs['sigma']

    if polarized:
        def _point(rho_pt, sigma_pt):
            return _exc_per_volume_gga(
                rho_pt, sigma_pt, energy_fn, params,
                dens_thr, sigma_thr, zeta_thr, True)
        grad_fn = jax.grad(_point, argnums=(0, 1))
        vrho, vsigma = jax.vmap(grad_fn)(rho, sigma)
        mask = (rho[:, 0] + rho[:, 1]) >= dens_thr
        vrho = jnp.where(mask[:, None], vrho, 0.0)
        vsigma = jnp.where(mask[:, None], vsigma, 0.0)
    else:
        def _point(rho_pt, sigma_pt):
            return _exc_per_volume_gga(
                rho_pt, sigma_pt, energy_fn, params,
                dens_thr, sigma_thr, zeta_thr, False)
        grad_fn = jax.grad(_point, argnums=(0, 1))
        vrho, vsigma = jax.vmap(grad_fn)(rho, sigma)
        mask = rho >= dens_thr
        vrho = jnp.where(mask, vrho, 0.0).reshape(-1, 1)
        vsigma = jnp.where(mask, vsigma, 0.0).reshape(-1, 1)

    return {'vrho': vrho, 'vsigma': vsigma}


def compute_vxc_mgga(energy_fn, params, polarized, inputs, thresholds):
    """Compute 1st derivatives for MGGA: vrho, vsigma, vlapl, vtau."""
    dens_thr = thresholds['dens']
    sigma_thr = thresholds['sigma']
    tau_thr = thresholds['tau']
    zeta_thr = thresholds['zeta']
    rho = inputs['rho']
    sigma = inputs['sigma']
    lapl = inputs.get('lapl', jnp.zeros_like(inputs['tau']))
    tau = inputs['tau']

    if polarized:
        def _point(rho_pt, sigma_pt, lapl_pt, tau_pt):
            return _exc_per_volume_mgga(
                rho_pt, sigma_pt, lapl_pt, tau_pt,
                energy_fn, params,
                dens_thr, sigma_thr, tau_thr, zeta_thr, True)
        grad_fn = jax.grad(_point, argnums=(0, 1, 2, 3))
        vrho, vsigma, vlapl, vtau = jax.vmap(grad_fn)(rho, sigma, lapl, tau)
        mask = (rho[:, 0] + rho[:, 1]) >= dens_thr
        vrho = jnp.where(mask[:, None], vrho, 0.0)
        vsigma = jnp.where(mask[:, None], vsigma, 0.0)
        vlapl = jnp.where(mask[:, None], vlapl, 0.0)
        vtau = jnp.where(mask[:, None], vtau, 0.0)
    else:
        def _point(rho_pt, sigma_pt, lapl_pt, tau_pt):
            return _exc_per_volume_mgga(
                rho_pt, sigma_pt, lapl_pt, tau_pt,
                energy_fn, params,
                dens_thr, sigma_thr, tau_thr, zeta_thr, False)
        grad_fn = jax.grad(_point, argnums=(0, 1, 2, 3))
        vrho, vsigma, vlapl, vtau = jax.vmap(grad_fn)(rho, sigma, lapl, tau)
        mask = rho >= dens_thr
        vrho = jnp.where(mask, vrho, 0.0).reshape(-1, 1)
        vsigma = jnp.where(mask, vsigma, 0.0).reshape(-1, 1)
        vlapl = jnp.where(mask, vlapl, 0.0).reshape(-1, 1)
        vtau = jnp.where(mask, vtau, 0.0).reshape(-1, 1)

    return {'vrho': vrho, 'vsigma': vsigma, 'vlapl': vlapl, 'vtau': vtau}


def compute_fxc_lda(energy_fn, params, polarized, inputs, thresholds):
    """Compute 2nd derivatives for LDA: v2rho2."""
    dens_thr = thresholds['dens']
    zeta_thr = thresholds['zeta']
    rho = inputs['rho']

    if polarized:
        def _point(rho_pt):
            return _exc_per_volume_lda(
                rho_pt, energy_fn, params, dens_thr, zeta_thr, True)
        hess_fn = jax.jacfwd(jax.grad(_point))
        v2rho2 = jax.vmap(hess_fn)(rho)
        mask = (rho[:, 0] + rho[:, 1]) >= dens_thr
        # v2rho2 shape: (N, 2, 2) -> flatten to (N, 3) [uu, ud, dd]
        v2rho2_flat = jnp.stack([
            v2rho2[:, 0, 0], v2rho2[:, 0, 1], v2rho2[:, 1, 1]
        ], axis=-1)
        v2rho2_flat = jnp.where(mask[:, None], v2rho2_flat, 0.0)
    else:
        def _point(rho_pt):
            return _exc_per_volume_lda(
                rho_pt, energy_fn, params, dens_thr, zeta_thr, False)
        hess_fn = jax.jacfwd(jax.grad(_point))
        v2rho2 = jax.vmap(hess_fn)(rho)
        mask = rho >= dens_thr
        v2rho2_flat = jnp.where(mask, v2rho2, 0.0).reshape(-1, 1)

    return {'v2rho2': v2rho2_flat}


def compute_fxc_gga(energy_fn, params, polarized, inputs, thresholds):
    """Compute 2nd derivatives for GGA: v2rho2, v2rhosigma, v2sigma2."""
    dens_thr = thresholds['dens']
    sigma_thr = thresholds['sigma']
    zeta_thr = thresholds['zeta']
    rho = inputs['rho']
    sigma = inputs['sigma']

    if polarized:
        def _point(rho_pt, sigma_pt):
            return _exc_per_volume_gga(
                rho_pt, sigma_pt, energy_fn, params,
                dens_thr, sigma_thr, zeta_thr, True)

        # Use jacfwd of grad to get full Hessian
        def grad_fn(rho_pt, sigma_pt):
            return jax.grad(_point, argnums=(0, 1))(rho_pt, sigma_pt)

        def hess_fn(rho_pt, sigma_pt):
            return jax.jacfwd(grad_fn, argnums=(0, 1))(rho_pt, sigma_pt)

        # hess returns ((d2/drho_drho, d2/drho_dsigma), (d2/dsigma_drho, d2/dsigma_dsigma))
        hess = jax.vmap(hess_fn)(rho, sigma)
        mask = (rho[:, 0] + rho[:, 1]) >= dens_thr

        # v2rho2: (N,2,2) -> (N,3) [uu, ud, dd]
        v2rho2 = hess[0][0]
        v2rho2_flat = jnp.stack([
            v2rho2[:, 0, 0], v2rho2[:, 0, 1], v2rho2[:, 1, 1]
        ], axis=-1)

        # v2rhosigma: (N,2,3) -> (N,6) [u_uu, u_ud, u_dd, d_uu, d_ud, d_dd]
        v2rhosigma = hess[0][1]
        v2rhosigma_flat = v2rhosigma.reshape(-1, 6)

        # v2sigma2: (N,3,3) -> (N,6) [uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd]
        v2sigma2 = hess[1][1]
        v2sigma2_flat = jnp.stack([
            v2sigma2[:, 0, 0], v2sigma2[:, 0, 1], v2sigma2[:, 0, 2],
            v2sigma2[:, 1, 1], v2sigma2[:, 1, 2], v2sigma2[:, 2, 2]
        ], axis=-1)

        v2rho2_flat = jnp.where(mask[:, None], v2rho2_flat, 0.0)
        v2rhosigma_flat = jnp.where(mask[:, None], v2rhosigma_flat, 0.0)
        v2sigma2_flat = jnp.where(mask[:, None], v2sigma2_flat, 0.0)
    else:
        def _point(rho_pt, sigma_pt):
            return _exc_per_volume_gga(
                rho_pt, sigma_pt, energy_fn, params,
                dens_thr, sigma_thr, zeta_thr, False)

        def grad_fn(rho_pt, sigma_pt):
            return jax.grad(_point, argnums=(0, 1))(rho_pt, sigma_pt)

        def hess_fn(rho_pt, sigma_pt):
            return jax.jacfwd(grad_fn, argnums=(0, 1))(rho_pt, sigma_pt)

        hess = jax.vmap(hess_fn)(rho, sigma)
        mask = rho >= dens_thr

        v2rho2_flat = jnp.where(mask, hess[0][0], 0.0).reshape(-1, 1)
        v2rhosigma_flat = jnp.where(mask, hess[0][1], 0.0).reshape(-1, 1)
        v2sigma2_flat = jnp.where(mask, hess[1][1], 0.0).reshape(-1, 1)

    return {
        'v2rho2': v2rho2_flat,
        'v2rhosigma': v2rhosigma_flat,
        'v2sigma2': v2sigma2_flat,
    }
