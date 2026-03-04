# Quick Start

## Basic usage

Create a functional, provide electron density, and compute outputs:

```python
import jax.numpy as jnp
import jaxlibxc

# Create an unpolarized LDA exchange functional
func = jaxlibxc.Functional("lda_x", spin="unpolarized")

# Electron density at 5 grid points
rho = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])

# Compute energy density and first derivatives
out = func.compute({"rho": rho}, do_exc=True, do_vxc=True)
print(out["zk"])    # energy per electron, shape (5, 1)
print(out["vrho"])  # d(n*eps)/d(rho), shape (5, 1)
```

## GGA functionals

GGA functionals require `sigma = |grad(rho)|^2` in addition to `rho`:

```python
func = jaxlibxc.Functional("gga_x_pbe", spin="unpolarized")
out = func.compute(
    {"rho": rho, "sigma": sigma},
    do_exc=True,
    do_vxc=True,
)
# out["zk"], out["vrho"], out["vsigma"]
```

## Meta-GGA functionals

Meta-GGAs additionally require the kinetic energy density `tau`:

```python
func = jaxlibxc.Functional("mgga_x_scan", spin="unpolarized")
out = func.compute(
    {"rho": rho, "sigma": sigma, "lapl": lapl, "tau": tau},
    do_exc=True,
    do_vxc=True,
)
# out["zk"], out["vrho"], out["vsigma"], out["vlapl"], out["vtau"]
```

## Spin-polarized calculations

Pass `spin="polarized"` and provide spin-resolved inputs:

```python
func = jaxlibxc.Functional("gga_c_lyp", spin="polarized")

# rho: (N, 2) array [rho_up, rho_down]
# sigma: (N, 3) array [sigma_uu, sigma_ud, sigma_dd]
out = func.compute(
    {"rho": rho_pol, "sigma": sigma_pol},
    do_exc=True,
    do_vxc=True,
)
```

## Second derivatives

Request second derivatives with `do_fxc=True`:

```python
func = jaxlibxc.Functional("lda_c_vwn", spin="unpolarized")
out = func.compute({"rho": rho}, do_exc=True, do_fxc=True)
print(out["v2rho2"])  # d2(n*eps)/d(rho)^2
```

For GGA functionals, second derivatives include `v2rho2`, `v2rhosigma`, and `v2sigma2`.

## Hybrid functionals

Hybrid functionals (e.g., B3LYP, PBE0) work identically. The exact exchange fraction is stored in the definition but must be handled externally when integrating with a DFT code:

```python
func = jaxlibxc.Functional("hyb_gga_xc_b3lyp", spin="unpolarized")
out = func.compute(
    {"rho": rho, "sigma": sigma},
    do_exc=True,
    do_vxc=True,
)
```

## Listing available functionals

```python
print(jaxlibxc.available())
```

## JIT compilation

Wrap computation in `jax.jit` for best performance:

```python
import jax

@jax.jit
def compute_energy(rho, sigma):
    return func.compute({"rho": rho, "sigma": sigma}, do_exc=True)["zk"]

# First call compiles; subsequent calls are fast
zk = compute_energy(rho, sigma)
```
