# Numerical Safety

Numerically safe operations for JAX autodiff. All functions ensure both branches of `jnp.where` produce finite values, which is critical for gradient computation.

```{eval-rst}
.. automodule:: jaxlibxc._numerical
   :members:
   :undoc-members:
```
