"""Test fixtures and helpers for jaxc tests."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# Standard test densities for unpolarized calculations
UNPOL_RHO = np.array([0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0])

# Standard test densities for polarized calculations (N, 2)
POL_RHO = np.array([
    [0.05, 0.05],
    [0.1, 0.05],
    [0.3, 0.2],
    [0.5, 0.5],
    [0.8, 0.2],
    [1.0, 0.0001],
    [2.0, 1.5],
    [5.0, 3.0],
])

# Standard sigma values for GGA tests (same length as UNPOL_RHO)
UNPOL_SIGMA = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 10.0])

# Polarized sigma: (N, 3) = [sigma_uu, sigma_ud, sigma_dd]
POL_SIGMA = np.array([
    [0.01, 0.005, 0.01],
    [0.05, 0.02, 0.03],
    [0.1, 0.05, 0.08],
    [0.2, 0.1, 0.2],
    [0.5, 0.1, 0.1],
    [1.0, 0.001, 0.001],
    [2.0, 1.0, 1.5],
    [5.0, 2.0, 3.0],
])

# Tolerances
TOL_EXC = 5e-8
TOL_VXC = 5e-5
TOL_FXC = 5e-4


def relative_error(x, y):
    """Relative error metric matching libxc's xc-error."""
    return np.abs(x - y) / (1.0 + np.maximum(np.abs(x), np.abs(y)))


def assert_close(actual, reference, tol, label=""):
    """Assert values match within relative tolerance."""
    a = np.array(actual).ravel()
    r = np.array(reference).ravel()
    assert a.shape == r.shape, f"{label} shape mismatch: {a.shape} vs {r.shape}"
    err = relative_error(a, r)
    max_err = np.max(err)
    if max_err > tol:
        idx = np.argmax(err)
        raise AssertionError(
            f"{label} max relative error {max_err:.2e} > {tol:.2e} "
            f"at index {idx}: actual={a[idx]:.10e} ref={r[idx]:.10e}"
        )


def pyscf_eval_xc(func_name, rho, spin=0, deriv=1):
    """Evaluate functional using pyscf's libxc interface.

    Args:
        func_name: libxc functional name (e.g. 'lda_x', 'gga_x_pbe')
        rho: for spin=0: (N,) array; for spin=1: (N, 2) in jaxc format
        spin: 0 for unpolarized, 1 for polarized
        deriv: derivative order

    Returns:
        exc, vxc, fxc, kxc (matching pyscf convention)
    """
    from pyscf.dft import libxc

    if spin == 0:
        return libxc.eval_xc(func_name, rho, spin=0, deriv=deriv)
    else:
        # Convert from jaxc format (N,2) to pyscf format (2,N)
        rho_pyscf = rho.T  # (2, N)
        return libxc.eval_xc(func_name, rho_pyscf, spin=1, deriv=deriv)


def pyscf_eval_gga(func_name, rho, sigma, spin=0, deriv=1):
    """Evaluate GGA functional using pyscf.

    For spin=0: rho is (N,), sigma is (N,)
    For spin=1: rho is (N,2), sigma is (N,3)

    pyscf wants: spin=0 -> (nvar, N) where nvar=4 [rho, grad_x, grad_y, grad_z]
                 spin=1 -> (2, nvar, N)
    But for GGA, pyscf uses a condensed format where sigma is reconstructed.
    """
    from pyscf.dft import libxc

    N = rho.shape[0] if rho.ndim > 0 else 1

    if spin == 0:
        # pyscf GGA unpol: (5, N) = [rho, sigma, lapl, tau, ?] -- actually (4,N) or just rho with deriv
        # Actually pyscf GGA wants rho as (4, N) = [rho, |grad rho|_x, |grad rho|_y, |grad rho|_z]
        # But sigma = |grad rho|^2, so we fake it with grad = (sqrt(sigma), 0, 0)
        grad_x = np.sqrt(np.maximum(sigma, 0.0))
        rho_4 = np.array([rho, grad_x, np.zeros(N), np.zeros(N)])
        return libxc.eval_xc(func_name, rho_4, spin=0, deriv=deriv)
    else:
        # pyscf polarized GGA: (2, 4, N)
        grad_x_up = np.sqrt(np.maximum(sigma[:, 0], 0.0))
        grad_x_dn = np.sqrt(np.maximum(sigma[:, 2], 0.0))
        rho_up = np.array([rho[:, 0], grad_x_up, np.zeros(N), np.zeros(N)])
        rho_dn = np.array([rho[:, 1], grad_x_dn, np.zeros(N), np.zeros(N)])
        rho_8 = np.array([rho_up, rho_dn])
        return libxc.eval_xc(func_name, rho_8, spin=1, deriv=deriv)
