# jaxlibxc Codebase Audit Results

## Summary

All 55 existing tests pass, but the test suite has significant coverage gaps.
Targeted probing reveals **3 confirmed functional bugs** and **3 registry/ID errors**
that would cause failures if properly tested against libxc.

---

## CONFIRMED BUGS (would fail tests vs libxc)

### BUG 1: RPBE uses wrong enhancement factor (CRITICAL)

**File:** `gga/x_pbe.py:104-118`
**Functional:** `gga_x_rpbe` (ID 144)

RPBE (Hammer, Hansen, Norskov 1999) has an **exponential** enhancement factor:
```
F_RPBE(s) = 1 + kappa * (1 - exp(-mu*s^2/kappa))
```

But the code registers RPBE using the **PBE** enhancement factor:
```
F_PBE(s) = 1 + kappa - kappa/(1 + mu*s^2/kappa)
```

These are mathematically different functions. The code produces values **identical**
to PBE at all inputs, while pyscf/libxc correctly returns different values.

**Measured error vs libxc:** up to 8.8e-4 relative error at moderate gradients.

**Root cause:** RPBE is registered with the same `_pbe_x_energy` function and same
parameters as PBE, but RPBE requires a different functional form.

---

### BUG 2: LDA_X_2D uses 3D formula (CRITICAL)

**File:** `lda/x.py:59-70`
**Functional:** `lda_x_2d` (ID 19)

The 2D LDA exchange has a completely different formula from the 3D version:
- 3D: `eps_x = -(3/4)(3/pi)^(1/3) * n^(1/3)`
- 2D: `eps_x = -(8/(3*sqrt(pi))) * n^(1/2)` (uses `X_FACTOR_2D_C`, `DIMENSIONS=2`)

The code registers `lda_x_2d` with the **same energy function** as `lda_x` (3D),
just with `alpha=1.0`. The 2D constant `X_FACTOR_2D_C` defined in `_constants.py:23`
is never used anywhere.

**Measured error vs libxc:** completely wrong scaling (n^(1/3) vs n^(1/2)).
At rho=5.0: jaxlibxc gives -1.263, pyscf gives -2.379 (89% relative error).

---

### BUG 3: B3PW91 uses PBE correlation instead of PW91 correlation

**File:** `gga/_mixed.py:37-48`
**Functional:** `hyb_gga_xc_b3pw91` (ID 401)

B3PW91 should use `gga_c_pw91` (Perdew-Wang 1991) as its GGA correlation component.
The code uses `gga_c_pbe` instead, with the comment "PW91 ~ PBE for this purpose".

While PW91 and PBE correlation are similar, they differ enough to cause test failures:

**Measured errors vs libxc:**
- Energy (exc): max rel error 4.96e-5 (borderline at 5e-5 tolerance)
- vrho: max rel error 1.68e-4 (FAILS at 5e-5 tolerance)
- vsigma: max rel error 7.09e-4 (FAILS at 5e-4 tolerance)
- Polarized vsigma: max rel error 7.09e-4 (FAILS)

---

## REGISTRY/ID ERRORS (wrong metadata)

### ID ERROR 1: lda_c_pw_mod registered as ID 13

**File:** `lda/c_pw.py:102-113`

In libxc, ID 13 is `lda_c_pz` (Perdew-Zunger 1981), NOT `lda_c_pw_mod`.
`lda_c_pw_mod` is not a separate functional in libxc -- it's the same as `lda_c_pw`
but with higher-precision parameters, used internally by PBE.

Impact: Lookup by ID 13 would return PW_MOD instead of Perdew-Zunger.

### ID ERROR 2: mgga_x_revscan registered as ID 581

**File:** `mgga/x_scan.py:105-120`

The CLAUDE.md correctly states revSCAN is ID 456, but the code registers it as 581.
In libxc, ID 581 is `mgga_x_revscan_vv10`, a different functional.
ID 456 is the correct `mgga_x_revscan`.

### ID ERROR 3: lda_x_slater registered as ID 550

**File:** `lda/x.py:72-83`

In libxc, ID 550 does not correspond to "Slater's Xalpha". The `lda_x` functional
itself supports an `_alpha` parameter. While the energy values with alpha=2/3 are
correct, the ID mapping is wrong.

---

## DEAD CODE / INCONSISTENCIES

### _transforms.py is dead code

**File:** `_transforms.py` (181 lines)

This module implements variable transforms (rho, sigma, tau) -> (rs, zeta, xt, ...),
but it is **never imported by the autodiff engine** (`_autodiff.py`). The autodiff
engine replicates all transforms inline.

Worse, `_transforms.py` has an INFERIOR implementation:
- `transform_mgga_unpol` (line 130-131) uses `jnp.minimum` for the Fermi hole
  curvature clamp WITHOUT `jax.lax.stop_gradient`, while `_autodiff.py` (line 135)
  correctly uses `stop_gradient`.
- `transform_mgga_pol` (line 165-170) similarly lacks `stop_gradient`.
- `transform_gga_pol` (line 99) uses `jnp.clip` for sigma_ud instead of
  `jnp.where` with `stop_gradient`.

If anyone called these transforms for derivative computation, they'd get wrong results.

### Inconsistent sigma_ud clipping in _autodiff.py

In `_autodiff.py`:
- GGA polarized (line 63): Uses `jnp.where` for sigma_ud clipping (correct per CLAUDE.md)
- MGGA polarized (line 114): Uses `jnp.clip` for sigma_ud clipping (different style)

Both have `stop_gradient` on the bounds so both are functionally correct, but the
inconsistency suggests one was written by different logic than the other.

### jnp.minimum for Fermi hole curvature in MGGA

In `_autodiff.py` lines 110-111:
```python
sigma_uu = jnp.minimum(sigma_uu, bound_uu)
```

CLAUDE.md explicitly warns: "`jnp.minimum(x, b)` gradient at boundary: When x == b,
gradient is 0.5 to each argument." While the bound is behind `stop_gradient` so the
0.5 vs 1.0 distinction only matters at the exact boundary (measure-zero in practice),
this contradicts the project's own documented best practice.

---

## TEST COVERAGE GAPS

The following registered functionals have NO tests against pyscf/libxc:

| Functional | Issue |
|---|---|
| `gga_x_rpbe` (ID 144) | **WRONG** - uses PBE formula |
| `lda_x_2d` (ID 19) | **WRONG** - uses 3D formula |
| `hyb_gga_xc_b3pw91` (ID 401) | **WRONG** - uses PBE_C instead of PW91_C |
| `lda_c_pw_mod` (ID 13) | Untested (works correctly, wrong ID) |
| `gga_x_pbe_tca` (ID 400) | Untested (works correctly) |
| `gga_x_apbe` (ID 184) | Untested (works correctly) |
| `gga_c_pbe_sol` (ID 133) | Untested (works correctly) |
| `hyb_gga_xc_b3lyp5` (ID 475) | Untested vs pyscf (works correctly) |
| `mgga_x_revscan` (ID 581/456) | Untested vs pyscf (works correctly, wrong ID) |
| `lda_x_slater` (ID 550) | Not in libxc, untested |

Additionally, these derivative orders are not tested:
- GGA 2nd derivatives polarized (fxc) - actually passes when tested
- MGGA 2nd derivatives (any)
- LDA 2nd derivatives polarized - actually passes when tested
- MGGA polarized vlap
- All 3rd/4th derivatives

---

## FUNCTIONALS THAT PASSED INDEPENDENT VERIFICATION

These functionals that ARE tested match pyscf/libxc to machine precision:

| Functional | exc error | vrho error | Notes |
|---|---|---|---|
| lda_x | <1e-16 | <1e-16 | Verified against analytical formula too |
| lda_c_pw | <1e-16 | <1e-16 | Verified against df_repo too |
| lda_c_vwn | <1e-16 | <1e-16 | |
| lda_c_vwn_rpa | <1e-16 | <1e-16 | |
| gga_x_pbe | <1e-16 | <1e-16 | Verified against ExchCXX, Libxc.jl too |
| gga_x_pbe_sol | <1e-16 | <1e-16 | |
| gga_x_pbe_r (revPBE) | <1e-16 | <1e-16 | |
| gga_x_b88 | <1e-15 | <1e-15 | |
| gga_c_pbe | <1e-16 | <1e-16 | |
| gga_c_lyp | <1e-16 | <1e-16 | Verified against ExchCXX, df_repo too |
| hyb_gga_xc_b3lyp | <1e-16 | <1e-16 | Polarized too |
| hyb_gga_xc_pbeh (PBE0) | <1e-16 | <1e-16 | Polarized too |
| mgga_x_scan | <1e-16 | <1e-16 | Verified against ExchCXX too |
| mgga_x_revscan | <1e-16 | <1e-16 | (but wrong ID) |
| mgga_c_scan | <1e-16 | <1e-16 | |
