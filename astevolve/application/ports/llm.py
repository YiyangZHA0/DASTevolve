

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Protocol, Tuple


@dataclass(frozen=True)
class ChatMessage:


    role: str
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("chat message role must be non-empty")
        if not isinstance(self.content, str):
            raise TypeError("chat message content must be a string")


@dataclass(frozen=True)
class LanguageModelRequest:


    messages: Tuple[ChatMessage, ...]
    system_message: str = ""
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.messages, tuple) or any(
            not isinstance(message, ChatMessage) for message in self.messages
        ):
            raise TypeError("messages must be a tuple of ChatMessage values")
        if not isinstance(self.system_message, str):
            raise TypeError("system_message must be a string")
        for name in ("temperature", "top_p"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number or None")
        if self.top_p is not None and not 0.0 < float(self.top_p) <= 1.0:
            raise ValueError("top_p must be within (0, 1]")
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer or None")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError("seed must be an integer or None")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class LanguageModelResponse:


    text: str
    provider: str
    model: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)
    raw: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("language-model response text must be a string")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("language-model response provider must be non-empty")
        if not isinstance(self.model, str):
            raise TypeError("language-model response model must be a string")
        if not isinstance(self.usage, Mapping):
            raise TypeError("language-model response usage must be a mapping")
        object.__setattr__(self, "usage", dict(self.usage))


class LanguageModel(Protocol):


    async def complete(self, request: LanguageModelRequest) -> LanguageModelResponse:
        ...
