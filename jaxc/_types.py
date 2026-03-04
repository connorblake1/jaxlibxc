"""Type definitions for jaxc."""

import enum
from typing import NamedTuple, Callable, Optional


class Family(enum.IntEnum):
    LDA = 1
    GGA = 2
    MGGA = 4


class Kind(enum.IntEnum):
    EXCHANGE = 0
    CORRELATION = 1
    EXCHANGE_CORRELATION = 2
    KINETIC = 3


class FunctionalInfo(NamedTuple):
    number: int
    name: str
    family: Family
    kind: Kind
    references: tuple = ()


class FunctionalDef(NamedTuple):
    """Definition of a primitive functional in the registry."""
    info: FunctionalInfo
    energy_fn: Callable       # (params, rs, z, ...) -> scalar zk
    default_params: dict      # default parameter values
    # number of internal variable arguments: LDA=0, GGA=3, MGGA=7
    # (xt, xs0, xs1) for GGA; (xt, xs0, xs1, u0, u1, t0, t1) for MGGA
    n_internal: int = 0


class MixedDef(NamedTuple):
    """Definition of a mixed/hybrid functional."""
    info: FunctionalInfo
    components: tuple         # tuple of (functional_name, coefficient) or coeff_fn
    default_params: dict
    hyb_exx: float = 0.0     # exact exchange fraction
