"""Mixed/hybrid functional infrastructure.

A mixed functional is a weighted linear combination of primitive functionals,
optionally including exact (HF) exchange.

B3LYP = (1-a0-ax)*LDA_X + ax*B88_X + (1-ac)*VWN_RPA + ac*LYP + a0*HF

Mixing coefficients are stored in params as 'coeff_0', 'coeff_1', etc.
so they participate in JAX autodiff (enabling ML optimization of mixing weights).
"""

import jax.numpy as jnp

from ._types import Family, Kind, FunctionalInfo, FunctionalDef, MixedDef
from ._registry import get as registry_get, register


def make_mixed_energy(component_specs):
    """Create a mixed energy function from component specifications.

    Args:
        component_specs: list of (coefficient, functional_name)
            Each component's default params will be used.

    Returns:
        (energy_fn, max_family, default_params) where default_params includes
        mixing coefficients as 'coeff_0', 'coeff_1', ... and component params
        as 'comp_0_<key>', 'comp_1_<key>', ...
    """
    # Resolve components at definition time
    resolved = []
    default_params = {}
    for i, (coeff, name) in enumerate(component_specs):
        func_def = registry_get(name)
        # Store mixing coefficient in params
        default_params[f'coeff_{i}'] = coeff
        # Store component params with prefix
        for k, v in func_def.default_params.items():
            default_params[f'comp_{i}_{k}'] = v
        resolved.append((i, func_def))

    # Determine highest family level
    max_family = max(fd.info.family for _, fd in resolved)

    # Build the energy function that reads coefficients and component params
    # from the params dict (making them differentiable)
    if max_family == Family.LDA:
        def energy_fn(params, rs, z):
            total = 0.0
            for i, func_def in resolved:
                coeff = params[f'coeff_{i}']
                comp_params = {k: params[f'comp_{i}_{k}']
                               for k in func_def.default_params}
                total = total + coeff * func_def.energy_fn(comp_params, rs, z)
            return total
    elif max_family == Family.GGA:
        def energy_fn(params, rs, z, xt, xs0, xs1):
            total = 0.0
            for i, func_def in resolved:
                coeff = params[f'coeff_{i}']
                comp_params = {k: params[f'comp_{i}_{k}']
                               for k in func_def.default_params}
                if func_def.info.family == Family.LDA:
                    total = total + coeff * func_def.energy_fn(comp_params, rs, z)
                else:
                    total = total + coeff * func_def.energy_fn(
                        comp_params, rs, z, xt, xs0, xs1)
            return total
    else:  # MGGA
        def energy_fn(params, rs, z, xt, xs0, xs1, u0, u1, t0, t1):
            total = 0.0
            for i, func_def in resolved:
                coeff = params[f'coeff_{i}']
                comp_params = {k: params[f'comp_{i}_{k}']
                               for k in func_def.default_params}
                if func_def.info.family == Family.LDA:
                    total = total + coeff * func_def.energy_fn(comp_params, rs, z)
                elif func_def.info.family == Family.GGA:
                    total = total + coeff * func_def.energy_fn(
                        comp_params, rs, z, xt, xs0, xs1)
                else:
                    total = total + coeff * func_def.energy_fn(
                        comp_params, rs, z, xt, xs0, xs1, u0, u1, t0, t1)
            return total

    return energy_fn, max_family, default_params


def register_mixed(number, name, component_specs, hyb_exx=0.0):
    """Register a mixed/hybrid functional.

    Args:
        number: functional ID
        name: functional name (e.g. 'hyb_gga_xc_b3lyp')
        component_specs: list of (coefficient, functional_name)
        hyb_exx: fraction of exact exchange
    """
    energy_fn, max_family, default_params = make_mixed_energy(component_specs)

    # n_internal depends on family
    if max_family == Family.LDA:
        n_internal = 0
    elif max_family == Family.GGA:
        n_internal = 3
    else:
        n_internal = 7

    return register(FunctionalDef(
        info=FunctionalInfo(
            number=number,
            name=name,
            family=max_family,
            kind=Kind.EXCHANGE_CORRELATION,
        ),
        energy_fn=energy_fn,
        default_params=default_params,
        n_internal=n_internal,
    ))
