# Installation

## Requirements

- Python 3.9+
- JAX with float64 support (enabled automatically by jaxlibxc)

## From source

```bash
git clone https://github.com/connorblake1/jaxlibxc.git
cd jaxlibxc
pip install -e .
```

## Dependencies

Core dependencies (installed automatically):

- `jax>=0.4.0`
- `jaxlib>=0.4.0`
- `numpy>=1.22`

### Optional: testing

```bash
pip install -e ".[test]"
```

This installs `pytest` and `pyscf` (which provides libxc bindings for reference comparisons).

### Optional: documentation

```bash
pip install -e ".[docs]"
```

## Verifying the installation

```python
import jaxlibxc
print(jaxlibxc.available())
# ['gga_c_lyp', 'gga_c_pbe', 'gga_x_b88', 'gga_x_pbe', ...]
```

## GPU support

jaxlibxc runs on any JAX backend. To use a GPU, install the appropriate `jaxlib` variant:

```bash
# CUDA 12
pip install jax[cuda12]

# For other backends, see https://jax.readthedocs.io/en/latest/installation.html
```

No code changes are needed -- JAX automatically uses the available accelerator.
