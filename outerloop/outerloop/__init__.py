

from __future__ import annotations

from importlib import import_module
from typing import Any

from outerloop._version import __version__


_LAZY_PUBLIC_API = {
    "Config": ("outerloop.config", "Config"),
    "OuterLoop": ("outerloop.controller", "OuterLoop"),
    "run_evolution": ("outerloop.api", "run_evolution"),
    "evolve_function": ("outerloop.api", "evolve_function"),
    "evolve_algorithm": ("outerloop.api", "evolve_algorithm"),
    "evolve_code": ("outerloop.api", "evolve_code"),
    "EvolutionResult": ("outerloop.api", "EvolutionResult"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_PUBLIC_API.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_PUBLIC_API))

__all__ = [
    "Config",
    "OuterLoop",
    "__version__",
    "run_evolution",
    "evolve_function",
    "evolve_algorithm",
    "evolve_code",
    "EvolutionResult",
]
