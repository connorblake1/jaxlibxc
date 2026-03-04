"""Tests for meta-GGA functionals against pyscf/libxc reference."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import jaxlibxc
from .conftest import TOL_EXC, TOL_VXC, assert_close


def _make_mgga_inputs(spin=0):
    """Generate consistent test inputs for MGGA."""
    if spin == 0:
        rho = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
        sigma = np.array([0.01, 0.05, 0.1, 0.3, 1.0, 3.0])
        tau = np.array([0.02, 0.1, 0.2, 0.5, 1.5, 5.0])
        lapl = np.zeros_like(rho)
        N = len(rho)
        grad_x = np.sqrt(sigma)
        pyscf_inp = np.array([rho, grad_x, np.zeros(N), np.zeros(N), lapl, tau])
        jaxlibxc_inp = {
            'rho': jnp.array(rho), 'sigma': jnp.array(sigma),
            'lapl': jnp.array(lapl), 'tau': jnp.array(tau),
        }
        return pyscf_inp, jaxlibxc_inp
    else:
        N = 5
        rho_p = np.array([
            [0.1, 0.05], [0.3, 0.2], [0.5, 0.3],
            [0.8, 0.2], [1.0, 0.5],
        ])
        grad_up = np.array([
            [0.1, 0.0, 0.0], [0.2, 0.1, 0.0], [0.3, 0.1, 0.0],
            [0.4, 0.2, 0.1], [0.5, 0.3, 0.2],
        ])
        grad_dn = np.array([
            [0.08, 0.0, 0.0], [0.15, 0.05, 0.0], [0.2, 0.0, 0.0],
            [0.15, 0.0, 0.0], [0.3, 0.2, 0.1],
        ])
        sigma_uu = np.sum(grad_up**2, axis=1)
        sigma_dd = np.sum(grad_dn**2, axis=1)
        sigma_ud = np.sum(grad_up * grad_dn, axis=1)
        sigma_p = np.column_stack([sigma_uu, sigma_ud, sigma_dd])
        # Use tau values well above von Weizsacker limit to avoid
        # alpha < 0 edge cases in SCAN-type functionals
        tau_p = np.array([
            [0.05, 0.02], [0.2, 0.12], [0.4, 0.2],
            [0.8, 0.1], [1.0, 0.4],
        ])
        lapl_p = np.zeros_like(tau_p)

        rho_up = np.array([rho_p[:,0], grad_up[:,0], grad_up[:,1], grad_up[:,2], lapl_p[:,0], tau_p[:,0]])
        rho_dn = np.array([rho_p[:,1], grad_dn[:,0], grad_dn[:,1], grad_dn[:,2], lapl_p[:,1], tau_p[:,1]])
        pyscf_inp = np.array([rho_up, rho_dn])
        jaxlibxc_inp = {
            'rho': jnp.array(rho_p), 'sigma': jnp.array(sigma_p),
            'lapl': jnp.array(lapl_p), 'tau': jnp.array(tau_p),
        }
        return pyscf_inp, jaxlibxc_inp


def _test_mgga_functional(name, spin=0):
    from pyscf.dft import libxc
    pyscf_inp, jaxlibxc_inp = _make_mgga_inputs(spin)
    ref_exc, ref_vxc, _, _ = libxc.eval_xc(name, pyscf_inp, spin=spin, deriv=1)

    func = jaxlibxc.Functional(name, spin='polarized' if spin else 'unpolarized')
    out = func.compute(jaxlibxc_inp, do_exc=True, do_vxc=True)

    label = f"{name} {'pol' if spin else 'unpol'}"
    assert_close(out['zk'], ref_exc, TOL_EXC, f"{label} exc")
    assert_close(out['vrho'], ref_vxc[0], TOL_VXC, f"{label} vrho")
    assert_close(out['vsigma'], ref_vxc[1], TOL_VXC, f"{label} vsigma")
    assert_close(out['vtau'], ref_vxc[3], TOL_VXC, f"{label} vtau")


class TestSCANX:
    def test_unpol(self):
        _test_mgga_functional('mgga_x_scan', spin=0)

    def test_pol(self):
        _test_mgga_functional('mgga_x_scan', spin=1)


class TestSCANC:
    def test_unpol(self):
        _test_mgga_functional('mgga_c_scan', spin=0)

    def test_pol(self):
        _test_mgga_functional('mgga_c_scan', spin=1)


class TestMGGAJit:
    def test_scan_jit(self):
        func = jaxlibxc.Functional('mgga_x_scan', spin='unpolarized')
        rho = jnp.array([0.5, 1.0])
        sigma = jnp.array([0.1, 0.3])
        tau = jnp.array([0.2, 0.5])
        lapl = jnp.zeros(2)

        @jax.jit
        def compute(rho, sigma, lapl, tau):
            return func.compute(
                {'rho': rho, 'sigma': sigma, 'lapl': lapl, 'tau': tau},
                do_exc=True)['zk']

        zk1 = compute(rho, sigma, lapl, tau)
        zk2 = compute(rho, sigma, lapl, tau)
        np.testing.assert_array_equal(np.array(zk1), np.array(zk2))
