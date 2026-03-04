# jaxlibxc

A pure Python/JAX reimplementation of [libxc](https://libxc.gitlab.io/), the library of exchange-correlation functionals for density functional theory (DFT).

**jaxlibxc** expresses each XC functional as a pure Python function and uses JAX to provide:

- **Automatic differentiation** -- `jax.grad` computes all derivative orders from the energy density alone
- **JIT compilation** -- `jax.jit` compiles functionals to optimized XLA code
- **GPU/TPU acceleration** -- runs on any JAX-supported backend
- **Differentiable parameters** -- enables ML optimization of XC functional parameters

## Contents

```{toctree}
:maxdepth: 2

installation
quickstart
user_guide/functionals
user_guide/derivatives
user_guide/ml_optimization
api/index
contributing
```

## Quick Example

```python
import jaxlibxc

func = jaxlibxc.Functional("gga_x_pbe", spin="unpolarized")
out = func.compute(
    {"rho": rho, "sigma": sigma},
    do_exc=True,   # energy per particle
    do_vxc=True,   # first derivatives
    do_fxc=True,   # second derivatives
)
# out["zk"], out["vrho"], out["vsigma"], out["v2rho2"], ...
```

## License

[Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/), the same license used by libxc.
