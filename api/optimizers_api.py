"""
api.optimizers_api -- turn the algos.Optimizer registry into JSON the frontend can render a
form from, with zero per-optimizer frontend code.

algos/optimizers.py and algos/problems.py both use `from __future__ import annotations`, so every
annotation on `Optimizer.solve` is a *string* at runtime. typing.get_type_hints() re-resolves them
against the function's module globals -- Parameter.annotation alone would give literal text.
"""
from __future__ import annotations
import inspect
import typing

from algos import Optimizer

_SKIP = {"self", "problem", "kw"}
_TYPE_MAP = {int: "integer", float: "number", str: "string", bool: "boolean"}


def _resolve(ann):
    """(json-type, nullable). Unwraps Optional/`T | None`; unknown -> ('string', False)."""
    if ann is None or ann is inspect._empty:
        return "string", False
    origin = typing.get_origin(ann)
    if origin is typing.Union or (origin is not None and str(origin) == "types.UnionType"):
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        nullable = len(args) < len(typing.get_args(ann))
        base = _TYPE_MAP.get(args[0], "string") if args else "string"
        return base, nullable
    return _TYPE_MAP.get(ann, "string"), False


def _param_schema(name, ann, default):
    jtype, nullable = _resolve(ann)
    return {
        "name": name,
        "type": jtype,
        "default": None if default is inspect._empty else default,
        "nullable": nullable or default is None,
    }


def describe(cls) -> dict:
    inst = cls()
    try:
        hints = typing.get_type_hints(cls.solve)
    except Exception:
        hints = {}
    params = []
    for pname, p in inspect.signature(cls.solve).parameters.items():
        if pname in _SKIP or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        params.append(_param_schema(pname, hints.get(pname), p.default))
    return {
        "name": cls.name,
        "kinds": list(cls.kinds),
        "available": bool(inst.available),
        "docstring": inspect.getdoc(cls) or "",
        "params": params,
    }


def list_optimizers(kind: str | None = None) -> list[dict]:
    out = []
    for name, cls in sorted(Optimizer.registry().items()):
        if kind is not None and kind not in cls.kinds:
            continue
        out.append(describe(cls))
    return out
