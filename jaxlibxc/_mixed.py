"""Mixed/hybrid functional infrastructure.

A mixed functional is a weighted linear combination of primitive functionals,
optionally including exact (HF) exchange.

B3LYP = (1-a0-ax)*LDA_X + ax*B88_X + (1-ac)*VWN_RPA + ac*LYP + a0*HF
"""

import jax.numpy as jnp

from ._types import Family, Kind, FunctionalInfo, FunctionalDef, MixedDef
from ._registry import get as registry_get, register


def _mixed_energy_fn(components, rs, z, xt=None, xs0=None, xs1=None,
                     u0=None, u1=None, t0=None, t1=None):
    """Evaluate a mixed functional as weighted sum of components.

    components: list of (coeff, func_def, comp_params)
    """
    total = 0.0
    for coeff, func_def, comp_params in components:
        family = func_def.info.family
        if family == Family.LDA:
            total = total + coeff * func_def.energy_fn(comp_params, rs, z)
        elif family == Family.GGA:
            total = total + coeff * func_def.energy_fn(comp_params, rs, z, xt, xs0, xs1)
        elif family == Family.MGGA:
            total = total + coeff * func_def.energy_fn(
                comp_params, rs, z, xt, xs0, xs1, u0, u1, t0, t1)
    return total


def make_mixed_energy(component_specs):
    """Create a mixed energy function from component specifications.

    Args:
        component_specs: list of (coefficient, functional_name)
            Each component's default params will be used.

    Returns:
        energy_fn compatible with the autodiff engine.
    """
    # Resolve components at definition time
    resolved = []
    for coeff, name in component_specs:
        func_def = registry_get(name)
        comp_params = {k: jnp.array(v) for k, v in func_def.default_params.items()}
        resolved.append((coeff, func_def, comp_params))

    # Determine highest family level
    max_family = max(fd.info.family for _, fd, _ in resolved)

    if max_family == Family.LDA:
        def energy_fn(params, rs, z):
            total = 0.0
            for coeff, func_def, comp_params in resolved:
                total = total + coeff * func_def.energy_fn(comp_params, rs, z)
            return total
    elif max_family == Family.GGA:
        def energy_fn(params, rs, z, xt, xs0, xs1):
            total = 0.0
            for coeff, func_def, comp_params in resolved:
                if func_def.info.family == Family.LDA:
                    total = total + coeff * func_def.energy_fn(comp_params, rs, z)
                else:
                    total = total + coeff * func_def.energy_fn(
                        comp_params, rs, z, xt, xs0, xs1)
            return total
    else:  # MGGA
        def energy_fn(params, rs, z, xt, xs0, xs1, u0, u1, t0, t1):
            total = 0.0
            for coeff, func_def, comp_params in resolved:
                if func_def.info.family == Family.LDA:
                    total = total + coeff * func_def.energy_fn(comp_params, rs, z)
                elif func_def.info.family == Family.GGA:
                    total = total + coeff * func_def.energy_fn(
                        comp_params, rs, z, xt, xs0, xs1)
                else:
                    total = total + coeff * func_def.energy_fn(
                        comp_params, rs, z, xt, xs0, xs1, u0, u1, t0, t1)
            return total

    return energy_fn, max_family


def register_mixed(number, name, component_specs, hyb_exx=0.0):
    """Register a mixed/hybrid functional.

    Args:
        number: functional ID
        name: functional name (e.g. 'hyb_gga_xc_b3lyp')
        component_specs: list of (coefficient, functional_name)
        hyb_exx: fraction of exact exchange
    """
    energy_fn, max_family = make_mixed_energy(component_specs)

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
        default_params={},  # Mixed functionals don't have top-level params
        n_internal=n_internal,
    ))
