"""User-facing Functional class with pylibxc-compatible API."""

import jax
import jax.numpy as jnp

from ._types import Family, FunctionalDef, MixedDef
from ._registry import get as registry_get
from ._constants import (
    DEFAULT_DENS_THRESHOLD, DEFAULT_SIGMA_THRESHOLD,
    DEFAULT_TAU_THRESHOLD, DEFAULT_ZETA_THRESHOLD,
)
from . import _autodiff


class Functional:
    """Exchange-correlation functional with pylibxc-compatible interface.

    Usage:
        func = Functional("lda_x", spin="unpolarized")
        out = func.compute({"rho": rho}, do_exc=True, do_vxc=True)
        # out["zk"], out["vrho"]
    """

    def __init__(self, name_or_id, spin="unpolarized"):
        """Initialize functional.

        Args:
            name_or_id: functional name (str) or ID (int)
            spin: "unpolarized" (or 1) or "polarized" (or 2)
        """
        self._def = registry_get(name_or_id)
        self._info = self._def.info

        if isinstance(spin, str):
            self._polarized = spin.lower() == "polarized"
            self._nspin = 2 if self._polarized else 1
        else:
            self._nspin = int(spin)
            self._polarized = self._nspin == 2

        # Copy default params (user can modify)
        self._params = {k: jnp.array(v) for k, v in self._def.default_params.items()}

        # Thresholds
        self._dens_threshold = DEFAULT_DENS_THRESHOLD
        self._sigma_threshold = DEFAULT_SIGMA_THRESHOLD
        self._tau_threshold = DEFAULT_TAU_THRESHOLD
        self._zeta_threshold = DEFAULT_ZETA_THRESHOLD

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        self._params = {k: jnp.array(v) for k, v in new_params.items()}

    @property
    def family(self):
        return self._info.family

    @property
    def kind(self):
        return self._info.kind

    def get_number(self):
        return self._info.number

    def get_name(self):
        return self._info.name

    def get_family(self):
        return self._info.family

    def get_kind(self):
        return self._info.kind

    def set_dens_threshold(self, val):
        self._dens_threshold = float(val)

    def set_sigma_threshold(self, val):
        self._sigma_threshold = float(val)

    def set_tau_threshold(self, val):
        self._tau_threshold = float(val)

    def set_zeta_threshold(self, val):
        self._zeta_threshold = float(val)

    def _thresholds(self):
        return {
            'dens': self._dens_threshold,
            'sigma': self._sigma_threshold,
            'tau': self._tau_threshold,
            'zeta': self._zeta_threshold,
        }

    def _prepare_inputs(self, inp):
        """Validate and reshape inputs into standard form."""
        if isinstance(inp, dict):
            out = {}
            rho = jnp.asarray(inp['rho'], dtype=jnp.float64)
            if self._polarized:
                if rho.ndim == 1:
                    rho = rho.reshape(-1, 2)
                out['rho'] = rho
                N = rho.shape[0]
            else:
                rho = rho.ravel()
                out['rho'] = rho
                N = rho.shape[0]

            if 'sigma' in inp:
                sigma = jnp.asarray(inp['sigma'], dtype=jnp.float64)
                if self._polarized:
                    if sigma.ndim == 1:
                        sigma = sigma.reshape(-1, 3)
                    out['sigma'] = sigma
                else:
                    out['sigma'] = sigma.ravel()

            if 'lapl' in inp:
                lapl = jnp.asarray(inp['lapl'], dtype=jnp.float64)
                if self._polarized:
                    if lapl.ndim == 1:
                        lapl = lapl.reshape(-1, 2)
                out['lapl'] = lapl

            if 'tau' in inp:
                tau = jnp.asarray(inp['tau'], dtype=jnp.float64)
                if self._polarized:
                    if tau.ndim == 1:
                        tau = tau.reshape(-1, 2)
                out['tau'] = tau

            return out, N
        else:
            raise TypeError("Input must be a dict with 'rho' key")

    def compute(self, inp, do_exc=True, do_vxc=False, do_fxc=False,
                do_kxc=False, do_lxc=False):
        """Compute functional outputs.

        Args:
            inp: dict with 'rho' and optionally 'sigma', 'lapl', 'tau'
            do_exc: compute energy density zk
            do_vxc: compute 1st derivatives
            do_fxc: compute 2nd derivatives
            do_kxc: compute 3rd derivatives (not yet implemented)
            do_lxc: compute 4th derivatives (not yet implemented)

        Returns:
            dict with requested outputs
        """
        inputs, N = self._prepare_inputs(inp)
        thresholds = self._thresholds()
        energy_fn = self._def.energy_fn
        params = self._params
        family = self._info.family
        result = {}

        if do_exc:
            result['zk'] = _autodiff.compute_exc(
                energy_fn, params, family, self._polarized,
                inputs, thresholds)

        if do_vxc:
            if family == Family.LDA:
                vxc = _autodiff.compute_vxc_lda(
                    energy_fn, params, self._polarized, inputs, thresholds)
            elif family == Family.GGA:
                vxc = _autodiff.compute_vxc_gga(
                    energy_fn, params, self._polarized, inputs, thresholds)
            elif family == Family.MGGA:
                vxc = _autodiff.compute_vxc_mgga(
                    energy_fn, params, self._polarized, inputs, thresholds)
            result.update(vxc)

        if do_fxc:
            if family == Family.LDA:
                fxc = _autodiff.compute_fxc_lda(
                    energy_fn, params, self._polarized, inputs, thresholds)
            elif family == Family.GGA:
                fxc = _autodiff.compute_fxc_gga(
                    energy_fn, params, self._polarized, inputs, thresholds)
            result.update(fxc)

        return result
