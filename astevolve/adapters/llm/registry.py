

from __future__ import annotations

from threading import RLock
from typing import Any, Callable, Mapping, Tuple

from astevolve.application.ports.llm import LanguageModel


LanguageModelFactory = Callable[[Mapping[str, Any]], LanguageModel]


def _provider_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("language-model provider name must be non-empty")
    normalized = value.strip().lower().replace("-", "_")
    if not normalized.replace("_", "").isalnum():
        raise ValueError("language-model provider name must be alphanumeric")
    return normalized


class LanguageModelProviderRegistry:


    def __init__(self) -> None:
        self._factories: dict[str, LanguageModelFactory] = {}
        self._lock = RLock()

    def register(
        self,
        name: str,
        factory: LanguageModelFactory,
        *,
        replace: bool = False,
    ) -> None:
        provider = _provider_name(name)
        if not callable(factory):
            raise TypeError("language-model provider factory must be callable")
        with self._lock:
            if provider in self._factories and not replace:
                raise ValueError(
                    f"language-model provider {provider!r} is already registered"
                )
            self._factories[provider] = factory

    def available(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._factories))

    def create(
        self, name: str, config: Mapping[str, Any] | None = None
    ) -> LanguageModel:
        provider = _provider_name(name)
        if config is not None and not isinstance(config, Mapping):
            raise TypeError("language-model provider config must be a mapping or None")
        with self._lock:
            factory = self._factories.get(provider)
            available = tuple(sorted(self._factories))
        if factory is None:
            choices = ", ".join(available) if available else "<none>"
            raise ValueError(
                f"unknown language-model provider {provider!r}; available: {choices}"
            )
        model = factory(dict(config or {}))
        if not callable(getattr(model, "complete", None)):
            raise TypeError(
                f"language-model provider {provider!r} did not return a LanguageModel"
            )
        return model


language_model_providers = LanguageModelProviderRegistry()


def register_language_model_provider(
    name: str,
    factory: LanguageModelFactory,
    *,
    replace: bool = False,
) -> None:
    language_model_providers.register(name, factory, replace=replace)


def available_language_model_providers() -> Tuple[str, ...]:
    return language_model_providers.available()


def create_language_model(
    name: str, config: Mapping[str, Any] | None = None
) -> LanguageModel:
    return language_model_providers.create(name, config)


__all__ = [
    "LanguageModelFactory",
    "LanguageModelProviderRegistry",
    "available_language_model_providers",
    "create_language_model",
    "language_model_providers",
    "register_language_model_provider",
]
