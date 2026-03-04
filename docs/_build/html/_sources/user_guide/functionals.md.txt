# Functionals

## Overview

jaxlibxc implements exchange-correlation (XC) functionals organized by family:

- **LDA** (Local Density Approximation) -- depends only on electron density
- **GGA** (Generalized Gradient Approximation) -- adds density gradient dependence
- **Meta-GGA** (MGGA) -- adds kinetic energy density dependence

Each functional also has a *kind*: exchange, correlation, exchange-correlation, or kinetic.

## Implemented functionals

### LDA primitives

| Name | Description |
|------|-------------|
| `lda_x` | Slater exchange (Dirac 1930) |
| `lda_x_2d` | 2D Slater exchange |
| `lda_c_pw` | Perdew-Wang 92 correlation |
| `lda_c_pw_mod` | Perdew-Wang 92 (modified) |
| `lda_c_vwn` | Vosko-Wilk-Nusair 5 correlation |
| `lda_c_vwn_rpa` | Vosko-Wilk-Nusair RPA correlation |

### GGA primitives

| Name | Description |
|------|-------------|
| `gga_x_pbe` | Perdew-Burke-Ernzerhof exchange |
| `gga_x_pbe_sol` | PBEsol exchange |
| `gga_x_pbe_r` | revPBE exchange |
| `gga_x_rpbe` | RPBE exchange (Hammer et al.) |
| `gga_x_apbe` | APBE exchange |
| `gga_x_pbe_tca` | TCA exchange |
| `gga_x_b88` | Becke 88 exchange |
| `gga_c_pbe` | PBE correlation |
| `gga_c_pbe_sol` | PBEsol correlation |
| `gga_c_lyp` | Lee-Yang-Parr correlation |

### Meta-GGA primitives

| Name | Description |
|------|-------------|
| `mgga_x_scan` | SCAN exchange (Sun et al. 2015) |
| `mgga_x_revscan` | Revised SCAN exchange |
| `mgga_c_scan` | SCAN correlation |

### Hybrid functionals

| Name | Composition |
|------|-------------|
| `hyb_gga_xc_b3lyp` | 0.08 LDA_X + 0.72 B88 + 0.19 VWN_RPA + 0.81 LYP + 0.20 HF |
| `hyb_gga_xc_b3lyp5` | B3LYP with VWN5 instead of VWN_RPA |
| `hyb_gga_xc_b3pw91` | B3PW91 |
| `hyb_gga_xc_pbeh` | PBE0 (0.75 PBE_X + PBE_C + 0.25 HF) |

## Input variables by family

### LDA

| Input | Shape (unpolarized) | Shape (polarized) |
|-------|--------------------|--------------------|
| `rho` | `(N,)` | `(N, 2)` -- `[rho_up, rho_down]` |

### GGA

| Input | Shape (unpolarized) | Shape (polarized) |
|-------|--------------------|--------------------|
| `rho` | `(N,)` | `(N, 2)` |
| `sigma` | `(N,)` -- `|grad rho|^2` | `(N, 3)` -- `[sigma_uu, sigma_ud, sigma_dd]` |

### Meta-GGA

| Input | Shape (unpolarized) | Shape (polarized) |
|-------|--------------------|--------------------|
| `rho` | `(N,)` | `(N, 2)` |
| `sigma` | `(N,)` | `(N, 3)` |
| `lapl` | `(N,)` -- Laplacian of density | `(N, 2)` |
| `tau` | `(N,)` -- kinetic energy density | `(N, 2)` |

## Output variables

| Output | Meaning | Shape (unpol) | Shape (pol) |
|--------|---------|---------------|-------------|
| `zk` | Energy per electron | `(N, 1)` | `(N, 1)` |
| `vrho` | d(n*eps)/d(rho) | `(N, 1)` | `(N, 2)` |
| `vsigma` | d(n*eps)/d(sigma) | `(N, 1)` | `(N, 3)` |
| `vlapl` | d(n*eps)/d(lapl) | `(N, 1)` | `(N, 2)` |
| `vtau` | d(n*eps)/d(tau) | `(N, 1)` | `(N, 2)` |
| `v2rho2` | d2(n*eps)/d(rho)^2 | `(N, 1)` | `(N, 3)` |
| `v2rhosigma` | d2(n*eps)/d(rho)d(sigma) | `(N, 1)` | `(N, 6)` |
| `v2sigma2` | d2(n*eps)/d(sigma)^2 | `(N, 1)` | `(N, 6)` |

## Internal variables

jaxlibxc transforms user inputs into internal variables matching libxc conventions:

| Variable | Meaning |
|----------|---------|
| `rs` | Wigner-Seitz radius: `(3/(4*pi*n))^(1/3)` |
| `zeta` | Spin polarization: `(n_up - n_down) / n` |
| `xt` | Total reduced gradient: `|grad n| / n^(4/3)` |
| `xs0`, `xs1` | Per-spin reduced gradients |
| `t0`, `t1` | Reduced kinetic energy densities per spin |
| `u0`, `u1` | Reduced Laplacians per spin |
