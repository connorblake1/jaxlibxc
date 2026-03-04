# CLAUDE.md -- Agent Guide for jaxlibxc

## Project Overview

jaxlibxc is a pure Python/JAX reimplementation of libxc (C library of 649 XC functionals for DFT). Each functional defines only the energy density; JAX autodiff computes all derivatives.

## Quick Commands

```bash
# Run all tests (requires pyscf for reference values)
python -m pytest jaxlibxc/tests/ -v

# Run specific test families
python -m pytest jaxlibxc/tests/test_lda.py -v
python -m pytest jaxlibxc/tests/test_gga.py -v
python -m pytest jaxlibxc/tests/test_mgga.py -v
python -m pytest jaxlibxc/tests/test_external_validation.py -v

# Quick smoke test (no pyscf needed)
python -c "import jaxlibxc; print(jaxlibxc.available())"
```

## Package Structure

```
jaxlibxc/
  __init__.py              # Public API, enables float64
  _constants.py            # RS_FACTOR, X_FACTOR_C, K_FACTOR_C, X2S, thresholds
  _numerical.py            # safe_pow, safe_log, safe_sqrt, piecewise3/5
  _types.py                # Family, Kind, FunctionalInfo, FunctionalDef, MixedDef
  _transforms.py           # (rho, sigma, tau) -> (rs, zeta, xt, xs, ts) transforms
  _utils.py                # f_zeta, opz_pow_n, mphi, lda_x_spin, screen_dens
  _exchange.py             # gga_exchange, mgga_exchange wrappers
  _registry.py             # Global name/ID -> FunctionalDef dict
  _autodiff.py             # compute_derivatives(): zk -> vrho/vsigma/vtau via jax.grad
  _mixed.py                # MixedFunctional: weighted sum of primitives
  functional.py            # Functional class (user-facing, pylibxc-compatible API)
  lda/                     # LDA primitives
    x.py                   # LDA exchange (Slater, 2D)
    c_pw.py                # Perdew-Wang 92 correlation
    c_vwn.py               # Vosko-Wilk-Nusair correlation (VWN5, VWN-RPA)
  gga/                     # GGA primitives
    x_pbe.py               # PBE exchange + 6 variants (PBEsol, revPBE, RPBE, APBE, TCA)
    x_b88.py               # Becke 88 exchange
    c_pbe.py               # PBE correlation + PBEsol
    c_lyp.py               # Lee-Yang-Parr correlation
    _mixed.py              # B3LYP, PBE0, B3PW91
  mgga/                    # Meta-GGA primitives
    x_scan.py              # SCAN exchange + revSCAN
    c_scan.py              # SCAN correlation
  tests/
    conftest.py            # Fixtures, pyscf oracle helpers, tolerances
    test_lda.py            # LDA tests vs pyscf
    test_gga.py            # GGA tests vs pyscf
    test_mgga.py           # MGGA tests vs pyscf
    test_external_validation.py  # 30 tests vs ExchCXX, df_repo, Libxc.jl, BrOH
```

## How to Add a New Functional

### Adding a primitive GGA exchange functional

1. Create `jaxlibxc/gga/x_<name>.py`
2. Define the enhancement factor function `F(xs)` and an `energy_fn(params, rs, zeta, xt, xs0, xs1)`
3. Use `gga_exchange(enhance_fn, ...)` from `_exchange.py` -- it handles spin decomposition
4. Register with `register(FunctionalDef(info, energy_fn, default_params, n_internal=3))`
5. Import in `gga/__init__.py`
6. Add tests in `tests/test_gga.py`

Pattern to follow: `gga/x_pbe.py`

### Adding a primitive GGA correlation functional

1. Create `jaxlibxc/gga/c_<name>.py`
2. Define `energy_fn(params, rs, zeta, xt, xs0, xs1)` that returns scalar eps_c
3. Register with `n_internal=3`
4. Pattern: `gga/c_pbe.py` (includes LDA correlation via import chain)

### Adding a primitive MGGA functional

1. Create `jaxlibxc/mgga/x_<name>.py` or `mgga/c_<name>.py`
2. `energy_fn(params, rs, zeta, xt, xs0, xs1, u0, u1, t0, t1)` -- note 7 extra args
3. Register with `n_internal=7`
4. For exchange: use `mgga_exchange(enhance_fn, ...)` from `_exchange.py`
5. Pattern: `mgga/x_scan.py`

### Adding a mixed/hybrid functional

1. Add entry in the relevant `_mixed.py` file (e.g., `gga/_mixed.py`)
2. Call `register_mixed(...)` with component names and coefficients
3. Pattern: see `gga/_mixed.py` for B3LYP, PBE0

## Key Design Decisions

### Autodiff via jax.grad (in `_autodiff.py`)

The core trick: define `exc_per_volume(rho_pt, sigma_pt, ...) = n * energy_fn(params, ...)`, then:
- `vrho = jax.grad(exc_per_volume, argnums=0)` -- first derivative
- `v2rho2 = jax.jacfwd(jax.grad(...))` -- second derivative
- `jax.vmap` vectorizes over grid points

### Critical: stop_gradient on preprocessing clamps

libxc applies input sanitization (density thresholds, sigma clamps) BEFORE computing derivatives. JAX's autodiff would differentiate through these clamps, producing wrong derivatives. Solution: `jax.lax.stop_gradient` on clamp bounds.

Key examples in `_autodiff.py`:
- **Fermi hole curvature**: `sigma_safe = jnp.minimum(sigma, stop_gradient(8*n*tau))`
- **Cauchy-Schwarz**: `sigma_ud = jnp.where(sigma_ud > stop_gradient(s_ave), ...)`

### Numerical safety (`_numerical.py`)

- `jnp.where` evaluates BOTH branches -- both must be finite for `jax.grad`
- Use `safe_pow`, `safe_log`, `safe_sqrt` which clamp inputs before operations
- `my_piecewise3/5` implements piecewise functions via nested `jnp.where`
- `jnp.minimum(x, b)` gives gradient 0.5 at x==b -- use `jnp.where(x > b, b, x)` instead for clamps

### Variable conventions

Internal variables match libxc's `work_{lda,gga,mgga}_inc.c`:
- `rs` = Wigner-Seitz radius = `(3/(4*pi*n))^(1/3)`
- `zeta` = spin polarization = `(n_up - n_dn) / n`
- `xt` = total reduced gradient = `|grad n| / n^(4/3)`
- `xs0, xs1` = per-spin reduced gradients
- `t0, t1` = reduced kinetic energy density per spin
- `u0, u1` = reduced Laplacian per spin

## Testing

### Tolerances
- Energy (zk): `5e-8`
- 1st derivatives (vrho, vsigma, vtau): `5e-5`
- 2nd derivatives (v2rho2, ...): `5e-4`
- Error metric: `|x - y| / (1 + max(|x|, |y|))` (matches libxc's xc-error)

### Reference sources
- **Primary**: pyscf/libxc (in test_lda/gga/mgga.py)
- **ExchCXX**: 12-digit precision hardcoded values
- **Density Functional Repository**: Standardized polarized test points
- **Libxc.jl**: Julia wrapper cross-validation
- **BrOH molecular grid**: Realistic DFT densities (core/valence/tail)

### Dependencies for testing
- `pyscf` (provides libxc bindings for oracle comparison)
- `pytest`
- `jax`, `jaxlib`, `numpy`

## Common Gotchas

1. **float64 is required**. `jax.config.update("jax_enable_x64", True)` is set in `__init__.py`. Without it, accuracy drops to ~1e-7 and many tests fail.

2. **Both branches of `jnp.where` must be finite**. Even the "dead" branch is evaluated by JAX. Example: `jnp.where(x > 0, jnp.log(x), 0.0)` will NaN when x <= 0. Use `safe_log` instead.

3. **`jnp.minimum(x, b)` gradient at boundary**: When x == b, gradient is 0.5 to each argument. For preprocessing clamps, use `jnp.where(x > b, b, x)` to get full gradient to x.

4. **New functionals must be imported in `__init__.py` of their subpackage** (e.g., `gga/__init__.py`) to trigger registration.

## License

MPL 2.0, same as libxc. See LICENSE file.
