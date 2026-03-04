"""Global registry mapping functional names/IDs to their definitions."""

from ._types import FunctionalDef, MixedDef

# Global registries
_BY_NAME: dict[str, FunctionalDef | MixedDef] = {}
_BY_ID: dict[int, FunctionalDef | MixedDef] = {}


def register(func_def):
    """Register a functional definition (primitive or mixed)."""
    name = func_def.info.name.lower()
    fid = func_def.info.number
    _BY_NAME[name] = func_def
    _BY_ID[fid] = func_def
    return func_def


def get(name_or_id):
    """Look up a functional by name (str) or ID (int)."""
    if isinstance(name_or_id, int):
        if name_or_id not in _BY_ID:
            raise KeyError(f"Unknown functional ID: {name_or_id}")
        return _BY_ID[name_or_id]
    name = name_or_id.lower().replace("-", "_")
    if name not in _BY_NAME:
        raise KeyError(f"Unknown functional: {name_or_id}")
    return _BY_NAME[name]


def available():
    """Return sorted list of all registered functional names."""
    return sorted(_BY_NAME.keys())
