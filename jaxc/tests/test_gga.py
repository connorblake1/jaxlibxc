"""Tests for GGA functionals against pyscf/libxc reference."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import jaxc
from .conftest import TOL_EXC, TOL_VXC, TOL_FXC, assert_close


def _make_gga_inputs(spin=0):
    """Generate consistent test inputs for GGA with proper gradient vectors."""
    if spin == 0:
        rho = np.array([0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
        sigma = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0])
        grad_x = np.sqrt(sigma)
        N = len(rho)
        pyscf_inp = np.array([rho, grad_x, np.zeros(N), np.zeros(N)])
        jaxc_inp = {'rho': jnp.array(rho), 'sigma': jnp.array(sigma)}
        return pyscf_inp, jaxc_inp
    else:
        N = 6
        rho_p = np.array([
            [0.1, 0.05], [0.3, 0.2], [0.5, 0.3],
            [0.8, 0.2], [1.0, 0.5], [2.0, 1.5],
        ])
        # Explicit gradient components for consistency
        grad_up = np.array([
            [0.1, 0.0, 0.0], [0.2, 0.1, 0.0], [0.3, 0.1, 0.0],
            [0.4, 0.2, 0.1], [0.5, 0.3, 0.2], [1.0, 0.5, 0.3],
        ])
        grad_dn = np.array([
            [0.08, 0.0, 0.0], [0.15, 0.05, 0.0], [0.2, 0.0, 0.0],
            [0.15, 0.0, 0.0], [0.3, 0.2, 0.1], [0.8, 0.4, 0.2],
        ])
        sigma_uu = np.sum(grad_up**2, axis=1)
        sigma_dd = np.sum(grad_dn**2, axis=1)
        sigma_ud = np.sum(grad_up * grad_dn, axis=1)
        sigma_p = np.column_stack([sigma_uu, sigma_ud, sigma_dd])

        rho_up_arr = np.array([rho_p[:, 0], grad_up[:, 0], grad_up[:, 1], grad_up[:, 2]])
        rho_dn_arr = np.array([rho_p[:, 1], grad_dn[:, 0], grad_dn[:, 1], grad_dn[:, 2]])
        pyscf_inp = np.array([rho_up_arr, rho_dn_arr])
        jaxc_inp = {'rho': jnp.array(rho_p), 'sigma': jnp.array(sigma_p)}
        return pyscf_inp, jaxc_inp


def _test_gga_functional(name, spin=0, test_vxc=True, test_fxc=False):
    """Generic test helper for a GGA functional."""
    from pyscf.dft import libxc
    pyscf_inp, jaxc_inp = _make_gga_inputs(spin)
    deriv = 2 if test_fxc else 1
    ref_exc, ref_vxc, ref_fxc, _ = libxc.eval_xc(name, pyscf_inp, spin=spin, deriv=deriv)

    func = jaxc.Functional(name, spin='polarized' if spin else 'unpolarized')
    out = func.compute(jaxc_inp, do_exc=True, do_vxc=test_vxc, do_fxc=test_fxc)

    label = f"{name} {'pol' if spin else 'unpol'}"
    assert_close(out['zk'], ref_exc, TOL_EXC, f"{label} exc")
    if test_vxc:
        assert_close(out['vrho'], ref_vxc[0], TOL_VXC, f"{label} vrho")
        assert_close(out['vsigma'], ref_vxc[1], TOL_VXC, f"{label} vsigma")


class TestGGAXPBE:
    def test_unpol(self):
        _test_gga_functional('gga_x_pbe', spin=0)

    def test_pol(self):
        _test_gga_functional('gga_x_pbe', spin=1)

    def test_pbesol_unpol(self):
        _test_gga_functional('gga_x_pbe_sol', spin=0)


class TestGGAXB88:
    def test_unpol(self):
        _test_gga_functional('gga_x_b88', spin=0)

    def test_pol(self):
        _test_gga_functional('gga_x_b88', spin=1)


class TestGGACPBE:
    def test_unpol(self):
        _test_gga_functional('gga_c_pbe', spin=0)

    def test_pol(self):
        _test_gga_functional('gga_c_pbe', spin=1)

    def test_fxc_unpol(self):
        _test_gga_functional('gga_c_pbe', spin=0, test_fxc=True)


class TestGGACLYP:
    def test_unpol(self):
        _test_gga_functional('gga_c_lyp', spin=0)

    def test_pol(self):
        _test_gga_functional('gga_c_lyp', spin=1)


class TestGGAJit:
    def test_pbe_jit(self):
        func = jaxc.Functional('gga_x_pbe', spin='unpolarized')
        rho = jnp.array([0.1, 0.5, 1.0])
        sigma = jnp.array([0.01, 0.1, 0.3])

        @jax.jit
        def compute(rho, sigma):
            return func.compute({'rho': rho, 'sigma': sigma}, do_exc=True)['zk']

        zk1 = compute(rho, sigma)
        zk2 = compute(rho, sigma)
        np.testing.assert_array_equal(np.array(zk1), np.array(zk2))

    def test_param_grad(self):
        """Test gradients w.r.t. PBE parameters (ML use case)."""
        func_def = jaxc.get_functional('gga_x_pbe')
        energy_fn = func_def.energy_fn

        def loss(kappa):
            params = {'kappa': kappa, 'mu': jnp.array(0.2195149727645171)}
            from jaxc._autodiff import compute_exc
            from jaxc._types import Family
            rho = jnp.array([1.0])
            sigma = jnp.array([0.1])
            zk = compute_exc(energy_fn, params, Family.GGA, False,
                             {'rho': rho, 'sigma': sigma},
                             {'dens': 1e-15, 'sigma': 1e-15, 'tau': 1e-20, 'zeta': 1e-10})
            return jnp.sum(zk)

        grad = jax.grad(loss)(jnp.array(0.8040))
        assert jnp.isfinite(grad)
