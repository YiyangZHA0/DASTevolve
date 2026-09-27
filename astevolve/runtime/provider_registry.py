

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Dict, Generic, TypeVar


T = TypeVar("T")


class ProviderRegistry(Generic[T]):


    def __init__(self, normalizer: Callable[[str], str] | None = None):
        self._normalizer = normalizer or (lambda value: value.strip().lower())
        self._factories: Dict[str, Callable[[], T]] = {}
        self._canonical_names: list[str] = []

    def register(
        self,
        name: str,
        factory: Callable[[], T],
        *,
        aliases: Iterable[str] = (),
        replace: bool = False,
    ) -> None:


        canonical = self._normalizer(name)
        keys = [canonical, *(self._normalizer(alias) for alias in aliases)]
        for key in keys:
            if key in self._factories and not replace:
                raise ValueError(f"Provider {key!r} is already registered")
        for key in keys:
            self._factories[key] = factory
        if canonical not in self._canonical_names:
            self._canonical_names.append(canonical)

    def create(self, name: str) -> T:


        key = self._normalizer(name)
        try:
            factory = self._factories[key]
        except KeyError as exc:
            available = ", ".join(self.available()) or "<none>"
            raise ValueError(f"Unknown provider {name!r}; available: {available}") from exc
        return factory()

    def available(self) -> tuple[str, ...]:


        return tuple(self._canonical_names)
