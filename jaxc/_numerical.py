"""Numerically safe operations for JAX autodiff.

All functions are designed so that both branches of jnp.where produce
finite values, which is critical for JAX gradient computation (both branches
are always evaluated, even if only one result is selected).
"""

import jax.numpy as jnp

# Small epsilon for clamping
_EPS = 1e-30


def my_piecewise3(cond, val_true, val_false):
    """Differentiable piecewise: if cond then val_true else val_false.

    Both val_true and val_false MUST be finite for all inputs.
    This is the JAX translation of libxc's my_piecewise3.
    """
    return jnp.where(cond, val_true, val_false)


def my_piecewise5(cond1, val1, cond2, val2, val3):
    """Five-argument piecewise: if cond1 then val1 elif cond2 then val2 else val3.

    Translation of libxc's my_piecewise5.
    """
    return jnp.where(cond1, val1, jnp.where(cond2, val2, val3))


def safe_pow(x, n):
    """x^n with clamped base to avoid NaN gradients at x=0."""
    return jnp.power(jnp.maximum(x, _EPS), n)


def safe_log(x):
    """log(x) with clamped argument."""
    return jnp.log(jnp.maximum(x, _EPS))


def safe_sqrt(x):
    """sqrt(x) with clamped argument."""
    return jnp.sqrt(jnp.maximum(x, _EPS))


def safe_div(x, y):
    """x/y with denominator clamped away from zero."""
    return x / jnp.where(jnp.abs(y) < _EPS, jnp.sign(y) * _EPS + _EPS, y)
