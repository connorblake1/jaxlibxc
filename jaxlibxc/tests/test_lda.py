"""Tests for LDA functionals against pyscf/libxc reference."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import jaxlibxc
from .conftest import (
    UNPOL_RHO, POL_RHO, TOL_EXC, TOL_VXC, TOL_FXC,
    assert_close, pyscf_eval_xc,
)


class TestLDAX:
    """Test LDA exchange functional."""

    def test_unpol_exc(self):
        func = jaxlibxc.Functional('lda_x', spin='unpolarized')
        ref_exc, _, _, _ = pyscf_eval_xc('lda_x', UNPOL_RHO, spin=0, deriv=0)
        out = func.compute({'rho': jnp.array(UNPOL_RHO)}, do_exc=True)
        assert_close(out['zk'].flatten(), ref_exc, TOL_EXC, "LDA_X unpol exc")

    def test_unpol_vrho(self):
        func = jaxlibxc.Functional('lda_x', spin='unpolarized')
        ref_exc, ref_vxc, _, _ = pyscf_eval_xc('lda_x', UNPOL_RHO, spin=0, deriv=1)
        out = func.compute({'rho': jnp.array(UNPOL_RHO)}, do_vxc=True)
        assert_close(out['vrho'].flatten(), ref_vxc[0], TOL_VXC, "LDA_X unpol vrho")

    def test_unpol_v2rho2(self):
        func = jaxlibxc.Functional('lda_x', spin='unpolarized')
        _, _, ref_fxc, _ = pyscf_eval_xc('lda_x', UNPOL_RHO, spin=0, deriv=2)
        out = func.compute({'rho': jnp.array(UNPOL_RHO)}, do_fxc=True)
        assert_close(out['v2rho2'].flatten(), ref_fxc[0], TOL_FXC, "LDA_X unpol v2rho2")

    def test_pol_exc(self):
        func = jaxlibxc.Functional('lda_x', spin='polarized')
        ref_exc, _, _, _ = pyscf_eval_xc('lda_x', POL_RHO, spin=1, deriv=0)
        out = func.compute({'rho': jnp.array(POL_RHO)}, do_exc=True)
        assert_close(out['zk'].flatten(), ref_exc, TOL_EXC, "LDA_X pol exc")

    def test_pol_vrho(self):
        func = jaxlibxc.Functional('lda_x', spin='polarized')
        _, ref_vxc, _, _ = pyscf_eval_xc('lda_x', POL_RHO, spin=1, deriv=1)
        out = func.compute({'rho': jnp.array(POL_RHO)}, do_vxc=True)
        assert_close(out['vrho'], ref_vxc[0], TOL_VXC, "LDA_X pol vrho")

    def test_jit(self):
        """Verify JIT compilation works."""
        func = jaxlibxc.Functional('lda_x', spin='unpolarized')
        rho = jnp.array(UNPOL_RHO)

        @jax.jit
        def compute_exc(rho):
            return func.compute({'rho': rho}, do_exc=True)['zk']

        zk1 = compute_exc(rho)
        zk2 = compute_exc(rho)  # second call uses cached JIT
        np.testing.assert_array_equal(np.array(zk1), np.array(zk2))

    def test_grad_wrt_params(self):
        """Verify gradients w.r.t. parameters are finite (ML use case)."""
        func_def = jaxlibxc.get_functional('lda_x')
        energy_fn = func_def.energy_fn

        def loss(alpha_val):
            params = {'alpha': alpha_val}
            rho = jnp.array([1.0])
            from jaxlibxc._autodiff import compute_exc
            from jaxlibxc._types import Family
            zk = compute_exc(
                energy_fn, params, Family.LDA, False,
                {'rho': rho},
                {'dens': 1e-15, 'sigma': 1e-15, 'tau': 1e-20, 'zeta': 1e-10})
            return jnp.sum(zk)

        grad = jax.grad(loss)(jnp.array(1.0))
        assert jnp.isfinite(grad), f"Gradient w.r.t. alpha is not finite: {grad}"

    def test_autodiff_vs_finite_diff(self):
        """Check JAX autodiff matches finite differences."""
        func = jaxlibxc.Functional('lda_x', spin='unpolarized')
        rho = jnp.array([0.5])
        out = func.compute({'rho': rho}, do_vxc=True)
        vrho_ad = float(out['vrho'].flatten()[0])

        # Finite difference
        h = 1e-6
        rho_p = jnp.array([0.5 + h])
        rho_m = jnp.array([0.5 - h])
        exc_p = float(func.compute({'rho': rho_p}, do_exc=True)['zk'].flatten()[0])
        exc_m = float(func.compute({'rho': rho_m}, do_exc=True)['zk'].flatten()[0])
        # vrho = d(n*exc)/dn
        n_p, n_m = 0.5 + h, 0.5 - h
        vrho_fd = (n_p * exc_p - n_m * exc_m) / (2.0 * h)
        assert abs(vrho_ad - vrho_fd) < 1e-6, f"AD: {vrho_ad}, FD: {vrho_fd}"


class TestLDAX2D:
    """Test 2D LDA exchange against pyscf/libxc."""

    def test_unpol_exc(self):
        func = jaxlibxc.Functional('lda_x_2d', spin='unpolarized')
        ref_exc, _, _, _ = pyscf_eval_xc('lda_x_2d', UNPOL_RHO, spin=0, deriv=0)
        out = func.compute({'rho': jnp.array(UNPOL_RHO)}, do_exc=True)
        assert_close(out['zk'].flatten(), ref_exc, TOL_EXC, "LDA_X_2D unpol exc")

    def test_unpol_vrho(self):
        func = jaxlibxc.Functional('lda_x_2d', spin='unpolarized')
        _, ref_vxc, _, _ = pyscf_eval_xc('lda_x_2d', UNPOL_RHO, spin=0, deriv=1)
        out = func.compute({'rho': jnp.array(UNPOL_RHO)}, do_vxc=True)
        assert_close(out['vrho'].flatten(), ref_vxc[0], TOL_VXC, "LDA_X_2D unpol vrho")

    def test_pol_exc(self):
        func = jaxlibxc.Functional('lda_x_2d', spin='polarized')
        ref_exc, _, _, _ = pyscf_eval_xc('lda_x_2d', POL_RHO, spin=1, deriv=0)
        out = func.compute({'rho': jnp.array(POL_RHO)}, do_exc=True)
        assert_close(out['zk'].flatten(), ref_exc, TOL_EXC, "LDA_X_2D pol exc")

    def test_differs_from_3d(self):
        """2D LDA exchange must differ from 3D."""
        rho = jnp.array([0.1, 0.5, 1.0, 5.0])
        inp = {'rho': rho}
        f3d = jaxlibxc.Functional('lda_x', spin='unpolarized')
        f2d = jaxlibxc.Functional('lda_x_2d', spin='unpolarized')
        zk_3d = f3d.compute(inp, do_exc=True)['zk']
        zk_2d = f2d.compute(inp, do_exc=True)['zk']
        assert not np.allclose(np.array(zk_3d), np.array(zk_2d), atol=1e-10), \
            "2D and 3D LDA exchange should differ"


class TestPytree:
    """Test Functional pytree registration."""

    def test_jit_sees_param_change(self):
        """Verify JIT retraces when params change (no stale cache)."""
        func = jaxlibxc.Functional('lda_x', spin='unpolarized')
        rho = jnp.array([1.0])

        @jax.jit
        def get_exc(f, rho):
            return f.compute({'rho': rho}, do_exc=True)['zk']

        zk1 = get_exc(func, rho)

        # Mutate params: alpha=2.0 should give 2x the energy
        func._params = {'alpha': jnp.array(2.0)}
        zk2 = get_exc(func, rho)

        assert not np.allclose(np.array(zk1), np.array(zk2)), \
            "JIT should see param change via pytree, not return stale result"
        np.testing.assert_allclose(np.array(zk2), 2.0 * np.array(zk1), rtol=1e-12)

    def test_pytree_roundtrip(self):
        """Verify flatten/unflatten preserves Functional state."""
        func = jaxlibxc.Functional('gga_x_pbe', spin='polarized')
        leaves, treedef = jax.tree_util.tree_flatten(func)
        func2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert func2.get_name() == 'gga_x_pbe'
        assert func2._polarized is True
        for k in func._params:
            np.testing.assert_array_equal(
                np.array(func._params[k]), np.array(func2._params[k]))


class TestLDACPW:
    """Test PW92 correlation (after Phase 2 implementation)."""
    pass


class TestLDACVWN:
    """Test VWN correlation (after Phase 2 implementation)."""
    pass
