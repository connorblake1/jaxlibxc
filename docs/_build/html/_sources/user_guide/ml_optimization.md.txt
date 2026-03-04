# Differentiable Parameters for ML

## Overview

Because jaxlibxc is built on JAX, functional parameters are fully differentiable. This enables gradient-based optimization of XC functionals -- a capability not available in libxc.

## Accessing parameters

Every functional has a `params` dict of tunable parameters:

```python
import jaxlibxc

func = jaxlibxc.Functional("gga_x_pbe", spin="unpolarized")
print(func.params)
# {'kappa': Array(0.804, dtype=float64),
#  'mu': Array(0.2195..., dtype=float64)}
```

## Modifying parameters

```python
import jax.numpy as jnp

func.params = {**func.params, "kappa": jnp.array(0.9)}
out = func.compute({"rho": rho, "sigma": sigma}, do_exc=True)
```

## Gradient w.r.t. parameters

Use `jax.grad` to differentiate a loss function through the functional evaluation:

```python
import jax

func = jaxlibxc.Functional("gga_x_pbe", spin="unpolarized")

def loss(kappa):
    func.params = {**func.params, "kappa": kappa}
    zk = func.compute({"rho": rho, "sigma": sigma}, do_exc=True)["zk"]
    return jnp.sum(zk * rho)  # total XC energy

# Gradient of total energy w.r.t. kappa
grad_kappa = jax.grad(loss)(jnp.array(0.8040))
```

## Use cases

### Fitting to reference data

```python
# Reference energies from higher-level theory
ref_energies = ...

def loss(kappa, mu):
    func.params = {"kappa": kappa, "mu": mu}
    zk = func.compute(inp, do_exc=True)["zk"]
    predicted = jnp.sum(zk * rho)
    return (predicted - ref_energies) ** 2

# Optimize with any JAX-compatible optimizer
from jax.example_libraries.optimizers import adam
opt_init, opt_update, get_params = adam(1e-3)
```

### Neural network-enhanced functionals

Since functional parameters flow through standard JAX operations, they can be outputs of a neural network:

```python
import jax.numpy as jnp

def nn_functional(nn_params, rho, sigma):
    # Neural network predicts enhancement factor parameters
    kappa = nn_forward(nn_params, features)
    func.params = {**func.params, "kappa": kappa}
    return func.compute({"rho": rho, "sigma": sigma}, do_exc=True)["zk"]
```

## Parameter reference

| Functional | Parameters |
|------------|-----------|
| `gga_x_pbe` | `kappa`, `mu` |
| `gga_x_pbe_sol` | `kappa`, `mu` (different defaults) |
| `gga_x_b88` | `beta` |
| `gga_c_pbe` | `beta`, `gamma` |
| `gga_c_lyp` | `A`, `B`, `C`, `D` |
