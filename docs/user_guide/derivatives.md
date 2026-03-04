# Automatic Derivatives

## How it works

The core design of jaxlibxc: define only the energy density function, then use JAX's automatic differentiation to compute all derivatives.

Each functional implements a single function:

```
energy_fn(params, rs, zeta, ...) -> eps_xc  (scalar)
```

The autodiff engine in `_autodiff.py` then constructs:

```python
# Energy per unit volume at a single grid point
exc_per_volume = n * energy_fn(params, rs, zeta, ...)

# First derivatives via jax.grad
vrho = jax.grad(exc_per_volume, argnums=0)  # w.r.t. rho

# Second derivatives via jax.jacfwd(jax.grad(...))
v2rho2 = jax.jacfwd(jax.grad(exc_per_volume))

# Vectorize over grid points
vrho_grid = jax.vmap(vrho)(rho_grid)
```

## Derivative levels

| Flag | Derivatives | LDA outputs | GGA outputs |
|------|-------------|-------------|-------------|
| `do_exc` | 0th (energy) | `zk` | `zk` |
| `do_vxc` | 1st | `vrho` | `vrho`, `vsigma` |
| `do_fxc` | 2nd | `v2rho2` | `v2rho2`, `v2rhosigma`, `v2sigma2` |

Meta-GGA first derivatives additionally include `vlapl` and `vtau`.

## Preprocessing clamps and stop_gradient

libxc applies input sanitization *before* computing derivatives:

- Density thresholding: `rho = max(rho, 1e-15)`
- Sigma clamping: `sigma = max(sigma, 1e-30)`
- Fermi hole curvature (MGGA): `sigma <= 8 * rho * tau`
- Cauchy-Schwarz: `|sigma_ud| <= 0.5 * (sigma_uu + sigma_dd)`

These clamps are physical constraints, not part of the functional definition. If JAX differentiated through them, derivatives would be wrong at clamped points. The solution:

```python
# Use jax.lax.stop_gradient on clamp bounds
bound = jax.lax.stop_gradient(8.0 * rho * tau)
sigma_safe = jnp.minimum(sigma, bound)
```

This ensures the clamp affects the *value* but not the *gradient*.

## Numerical safety

JAX evaluates *both branches* of `jnp.where`, even the "dead" one. Both must produce finite values for `jax.grad` to work:

```python
# BAD: NaN gradient when x <= 0
jnp.where(x > 0, jnp.log(x), 0.0)

# GOOD: safe_log clamps input
from jaxlibxc._numerical import safe_log
jnp.where(x > 0, safe_log(x), 0.0)
```

The `_numerical` module provides `safe_pow`, `safe_log`, `safe_sqrt`, and `safe_div` for this purpose.

## Comparison with libxc

| Aspect | libxc | jaxlibxc |
|--------|-------|----------|
| Derivative generation | Maple symbolic CAS | JAX autodiff |
| Code per functional | ~500-2000 lines (energy + all derivatives) | ~50-100 lines (energy only) |
| Max derivative order | 4th (fixed at compile time) | Arbitrary (limited by memory) |
| Adding a functional | Write Maple + regenerate C | Write Python energy function |
