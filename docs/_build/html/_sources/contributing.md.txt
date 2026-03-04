# Contributing

## Adding a new functional

### GGA exchange

1. Create `jaxlibxc/gga/x_<name>.py`
2. Define the enhancement factor `F(xs)` and an energy function:

```python
from .._exchange import gga_exchange
from .._types import Family, Kind, FunctionalInfo, FunctionalDef
from .._registry import register

def _enhance(xs):
    """Enhancement factor F(xs)."""
    ...
    return F

def energy_fn(params, rs, zeta, xt, xs0, xs1):
    return gga_exchange(_enhance, rs, zeta, xs0, xs1)

register(FunctionalDef(
    info=FunctionalInfo(number=ID, name="gga_x_name", family=Family.GGA, kind=Kind.EXCHANGE),
    energy_fn=energy_fn,
    default_params={"param1": value1},
    n_internal=3,
))
```

3. Import in `gga/__init__.py`
4. Add tests in `tests/test_gga.py`

**Pattern to follow**: `gga/x_pbe.py`

### GGA correlation

1. Create `jaxlibxc/gga/c_<name>.py`
2. Define `energy_fn(params, rs, zeta, xt, xs0, xs1)` returning scalar `eps_c`
3. Register with `n_internal=3`

**Pattern to follow**: `gga/c_pbe.py`

### Meta-GGA

1. Create `jaxlibxc/mgga/x_<name>.py` or `mgga/c_<name>.py`
2. Energy function signature includes extra arguments:

```python
def energy_fn(params, rs, zeta, xt, xs0, xs1, u0, u1, t0, t1):
    ...
```

3. Register with `n_internal=7`
4. For exchange, use `mgga_exchange` from `_exchange.py`

**Pattern to follow**: `mgga/x_scan.py`

### Mixed/hybrid functional

1. Add entry in the relevant `_mixed.py` (e.g., `gga/_mixed.py`)
2. Call `register_mixed(...)` with component names and coefficients

**Pattern to follow**: `gga/_mixed.py` for B3LYP, PBE0

## Testing

### Running tests

```bash
python -m pytest jaxlibxc/tests/ -v
```

### Test structure

Tests compare jaxlibxc against pyscf/libxc reference values using a relative error metric:

```
error = |x - y| / (1 + max(|x|, |y|))
```

### Tolerances

| Quantity | Tolerance |
|----------|-----------|
| Energy (`zk`) | `5e-8` |
| 1st derivatives (`vrho`, `vsigma`, `vtau`) | `5e-5` |
| 2nd derivatives (`v2rho2`, ...) | `5e-4` |

### Dependencies

- `pyscf` -- provides libxc bindings for oracle comparison
- `pytest`

## Common gotchas

1. **float64 is required.** `jax.config.update("jax_enable_x64", True)` is set in `__init__.py`. Without it, accuracy drops to ~1e-7.

2. **Both branches of `jnp.where` must be finite.** Even the "dead" branch is evaluated by JAX. Use `safe_log`, `safe_pow`, `safe_sqrt` from `_numerical.py`.

3. **`jnp.minimum(x, b)` gradient at boundary:** When `x == b`, gradient is 0.5 to each argument. For preprocessing clamps, use `jnp.where(x > b, b, x)` to get full gradient to `x`.

4. **New functionals must be imported** in the `__init__.py` of their subpackage (e.g., `gga/__init__.py`) to trigger registration.
