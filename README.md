HIGHLY EXPERIMENTAL - CLAUDE OPUS 4.6 WROTE THIS ENTIRE THING FROM SCRATCH - USE AT YOUR OWN RISK
# jaxlibxc

[![Documentation Status](https://readthedocs.org/projects/jaxlibxc/badge/?version=latest)](https://jaxlibxc.readthedocs.io/en/latest/?badge=latest)

A pure Python/JAX reimplementation of [libxc](https://libxc.gitlab.io/), the library of exchange-correlation (XC) functionals for density functional theory (DFT).

**[Documentation](https://jaxlibxc.readthedocs.io)** | **[Installation](https://jaxlibxc.readthedocs.io/en/latest/installation.html)** | **[Quick Start](https://jaxlibxc.readthedocs.io/en/latest/quickstart.html)**

## Why?

libxc provides 649 XC functionals implemented in C with symbolically-generated derivatives from Maple. This works, but:

- Adding new functionals requires Maple codegen expertise
- Derivatives are fixed at compile time (up to 4th order)
- No GPU acceleration
- Parameters are not differentiable (can't optimize them with gradient descent)

**jaxlibxc** solves all of these by expressing each functional as a pure Python function and using JAX for:

- **Automatic differentiation**: `jax.grad` computes all derivative orders (vrho, vsigma, v2rho2, ...) from the energy density alone -- eliminating ~80% of libxc's code
- **JIT compilation**: `jax.jit` compiles functionals to optimized XLA code
- **GPU/TPU acceleration**: Runs on any JAX-supported backend
- **Differentiable parameters**: `jax.grad` w.r.t. functional parameters enables ML optimization of XC functionals

## Installation

```bash
git clone https://github.com/connorblake1/jaxlibxc.git
cd jaxlibxc
pip install -e .
```

Requires Python 3.9+ and JAX with float64 support (enabled automatically). See the [installation guide](https://jaxlibxc.readthedocs.io/en/latest/installation.html) for GPU setup and optional dependencies.

## Quick Start

```python
import jaxlibxc

# Evaluate PBE exchange
func = jaxlibxc.Functional('gga_x_pbe', spin='unpolarized')
out = func.compute(
    {'rho': rho, 'sigma': sigma},
    do_exc=True,   # energy per particle
    do_vxc=True,   # first derivatives
    do_fxc=True,   # second derivatives
)
# out['zk'], out['vrho'], out['vsigma'], out['v2rho2'], ...
```

### Polarized calculations

```python
func = jaxlibxc.Functional('gga_c_lyp', spin='polarized')
out = func.compute({
    'rho': rho_pol,       # (N, 2) array [rho_up, rho_down]
    'sigma': sigma_pol,   # (N, 3) array [sigma_uu, sigma_ud, sigma_dd]
}, do_exc=True, do_vxc=True)
```

### JIT compilation

```python
import jax

@jax.jit
def compute_energy(rho, sigma):
    return func.compute({'rho': rho, 'sigma': sigma}, do_exc=True)['zk']
```

### Differentiable parameters (ML use case)

```python
func = jaxlibxc.Functional('gga_x_pbe', spin='unpolarized')

def loss(kappa):
    func.params = {**func.params, 'kappa': kappa}
    return jnp.sum(func.compute(inp, do_exc=True)['zk'] * rho)

grad = jax.grad(loss)(jnp.array(0.8040))  # gradient w.r.t. kappa
```

## Implemented Functionals

### Primitives (with full autodiff)

| Family | Functional | Variants |
|--------|-----------|----------|
| LDA | `lda_x` | Slater, 2D |
| LDA | `lda_c_pw` | PW92, PW92-mod |
| LDA | `lda_c_vwn` | VWN5, VWN-RPA |
| GGA | `gga_x_pbe` | PBE, PBEsol, revPBE, RPBE, APBE, TCA |
| GGA | `gga_x_b88` | Becke 88 |
| GGA | `gga_c_pbe` | PBE, PBEsol |
| GGA | `gga_c_lyp` | Lee-Yang-Parr |
| MGGA | `mgga_x_scan` | SCAN, revSCAN |
| MGGA | `mgga_c_scan` | SCAN correlation |

### Hybrids (mixed functionals)

| Functional | Composition |
|-----------|-------------|
| `hyb_gga_xc_b3lyp` | 0.08 LDA_X + 0.72 B88 + 0.19 VWN_RPA + 0.81 LYP + 0.20 HF |
| `hyb_gga_xc_b3lyp5` | B3LYP with VWN5 |
| `hyb_gga_xc_b3pw91` | B3PW91 |
| `hyb_gga_xc_pbeh` | PBE0 (0.75 PBE_X + PBE_C + 0.25 HF) |

Run `jaxlibxc.available()` for the full list.

## Accuracy

All functionals are validated against libxc (via pyscf) and independent reference sources:

- **ExchCXX** (12-digit precision reference values)
- **Density Functional Repository** (standardized test densities)
- **Libxc.jl** (Julia wrapper reference)
- **BrOH molecular densities** (realistic DFT grid inputs spanning core to tail regions)

Tolerances: `5e-8` (energy), `5e-5` (1st derivatives), `5e-4` (2nd derivatives).

55 tests pass across all implemented functionals.

## Architecture

Each functional is a pure function `(params, rs, zeta, xs, ...) -> scalar` that returns the XC energy density per electron at a single grid point. JAX handles everything else:

```
User inputs (rho, sigma, tau)
    |
    v
Variable transforms (_transforms.py)  -->  (rs, zeta, xt, xs0, xs1, t0, t1)
    |
    v
Energy density function (e.g., gga/x_pbe.py)  -->  scalar zk
    |
    v
jax.grad / jax.jacfwd (_autodiff.py)  -->  vrho, vsigma, vtau, v2rho2, ...
    |
    v
jax.vmap (functional.py)  -->  vectorized over grid points
```

## License

This project is licensed under the [Mozilla Public License 2.0](LICENSE), the same license used by libxc.

This is an independent reimplementation. The mathematical formulas for XC functionals are from published scientific literature. The implementation structure and variable transformations are derived from the libxc source code (Copyright (C) 2006-2021 M.A.L. Marques et al., MPL 2.0).

### Citation

If you use this library in published work, please cite both this project and the original libxc:

> S. Lehtola, C. Steigemann, M. J. T. Oliveira, and M. A. L. Marques,
> "Recent developments in libxc -- A comprehensive library of functionals
> for density functional theory", SoftwareX 7, 1-5 (2018).

## Contributing

This project is in active development. ~24 primitive functionals and 4 hybrids are implemented. The framework supports mechanical addition of the remaining ~245 primitives and ~370 mixed functionals following established patterns. See the [contributing guide](https://jaxlibxc.readthedocs.io/en/latest/contributing.html) for how to add new functionals.
