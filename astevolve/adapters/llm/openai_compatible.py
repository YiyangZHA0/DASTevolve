

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
import math
import os
import re
import threading
from typing import Any, Mapping
from urllib.parse import urlsplit

from astevolve.application.ports.llm import LanguageModelRequest, LanguageModelResponse


class OpenAICompatibleError(RuntimeError):
    pass


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty normalized string")
    return value


_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_PROVIDER_SEED_MODULUS = 2**31


def _provider_seed(seed: int) -> int:


    return seed % _PROVIDER_SEED_MODULUS


def _validated_base_url(value: Any) -> str:
    text = _text(value, "base_url")
    try:
        parsed = urlsplit(text)
    except ValueError as exc:
        raise ValueError("base_url must be a credential-free HTTP(S) URL") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base_url must be a credential-free HTTP(S) URL")
    return text


def _consume_task(task: asyncio.Future[Any]) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        pass
    except BaseException:

        pass


@dataclass(frozen=True)
class OpenAICompatibleLanguageModel:


    model: str
    base_url: str | None = None
    api_key_env: str = "ASTEVOLVE_LLM_API_KEY"
    timeout_seconds: float = 300.0
    provider_name: str = "openai_compatible"
    client: Any = None
    _owned_client: Any = field(default=None, init=False, repr=False, compare=False)
    _owned_loop: asyncio.AbstractEventLoop | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _client_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
        compare=False,
    )
    _closed: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _text(self.model, "model")
        api_key_env = _text(self.api_key_env, "api_key_env")
        if not _ENVIRONMENT_NAME.fullmatch(api_key_env):
            raise ValueError("api_key_env must name an environment variable")
        _text(self.provider_name, "provider_name")
        if self.base_url is not None:
            object.__setattr__(self, "base_url", _validated_base_url(self.base_url))
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")

    async def _get_client(self) -> Any:
        loop = asyncio.get_running_loop()
        with self._client_lock:
            if self._closed:
                raise OpenAICompatibleError("language-model client is closed")
            if self.client is not None:
                return self.client
            if self._owned_client is not None:
                if self._owned_loop is not loop:
                    raise OpenAICompatibleError(
                        "language-model client must be used on its owning event loop"
                    )
                return self._owned_client
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise OpenAICompatibleError(
                    "language-model credential environment variable is not set"
                )
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise OpenAICompatibleError(
                    "the optional openai package is required for this provider"
                ) from exc
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": float(self.timeout_seconds),
            }
            if self.base_url is not None:
                kwargs["base_url"] = self.base_url
            try:
                owned = AsyncOpenAI(**kwargs)
            except Exception:
                raise OpenAICompatibleError(
                    "language-model client initialization failed"
                ) from None
            object.__setattr__(self, "_owned_client", owned)
            object.__setattr__(self, "_owned_loop", loop)
            return owned

    async def aclose(self) -> None:


        loop = asyncio.get_running_loop()
        with self._client_lock:
            if self._closed:
                return
            owned = self._owned_client
            owner = self._owned_loop
            if owned is not None and owner is not loop:
                raise OpenAICompatibleError(
                    "language-model client must be closed on its owning event loop"
                )
            object.__setattr__(self, "_closed", True)
            object.__setattr__(self, "_owned_client", None)
            object.__setattr__(self, "_owned_loop", None)
        if owned is None:
            return
        close = getattr(owned, "aclose", None)
        if callable(close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    task = asyncio.ensure_future(result)
                    done, _pending = await asyncio.wait(
                        (task,), timeout=float(self.timeout_seconds)
                    )
                    if not done:
                        task.cancel()
                        task.add_done_callback(_consume_task)
                        raise OpenAICompatibleError(
                            "language-model client close exceeded timeout"
                        )
                    task.result()
            except asyncio.CancelledError:
                raise
            except OpenAICompatibleError:
                raise
            except Exception:
                raise OpenAICompatibleError(
                    "language-model client close failed"
                ) from None

    async def __aenter__(self) -> "OpenAICompatibleLanguageModel":
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def complete(self, request: LanguageModelRequest) -> LanguageModelResponse:
        if not isinstance(request, LanguageModelRequest):
            raise TypeError("request must be LanguageModelRequest")
        messages = []
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})
        for message in request.messages:
            if message.role not in {"system", "user", "assistant", "tool"}:
                raise OpenAICompatibleError(
                    f"unsupported chat message role: {message.role!r}"
                )
            messages.append({"role": message.role, "content": message.content})
        params: dict[str, Any] = {"model": self.model, "messages": messages}
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        provider_seed: int | None = None
        if request.seed is not None:
            provider_seed = _provider_seed(request.seed)
            params["seed"] = provider_seed


        params["timeout"] = float(self.timeout_seconds)
        client = await self._get_client()
        try:
            operation = client.chat.completions.create(**params)
            if not inspect.isawaitable(operation):
                raise TypeError("provider create() must return an awaitable")
            task = asyncio.ensure_future(operation)
            done, _pending = await asyncio.wait(
                (task,), timeout=float(self.timeout_seconds)
            )
            if not done:
                task.cancel()
                task.add_done_callback(_consume_task)
                raise OpenAICompatibleError("language-model transport exceeded timeout")
            response = task.result()
        except asyncio.CancelledError:
            raise
        except OpenAICompatibleError:
            raise
        except Exception:
            raise OpenAICompatibleError(
                "language-model transport request failed"
            ) from None
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise OpenAICompatibleError(
                "provider response has no chat completion content"
            ) from exc
        if content is None:
            raise OpenAICompatibleError("provider returned empty completion content")
        usage_value: Mapping[str, Any] = {}
        usage = getattr(response, "usage", None)
        if usage is not None:
            if callable(getattr(usage, "model_dump", None)):
                dumped = usage.model_dump()
                usage_value = dumped if isinstance(dumped, Mapping) else {}
            elif isinstance(usage, Mapping):
                usage_value = usage
        normalized_usage = dict(usage_value)
        if request.seed is not None:
            normalized_usage["_astevolve_native_seed"] = request.seed
            normalized_usage["_astevolve_provider_seed"] = provider_seed
        return LanguageModelResponse(
            text=str(content),
            provider=self.provider_name,
            model=str(getattr(response, "model", None) or self.model),
            usage=normalized_usage,
            raw=response,
        )


_CONFIG_FIELDS = {
    "model",
    "base_url",
    "api_key_env",
    "timeout_seconds",
    "provider_name",
}


def create_openai_compatible_model(
    config: Mapping[str, Any],
) -> OpenAICompatibleLanguageModel:
    if not isinstance(config, Mapping):
        raise TypeError("OpenAI-compatible provider config must be a mapping")
    unknown = sorted(set(config) - _CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"unknown OpenAI-compatible config fields: {unknown}")
    model = _text(config.get("model"), "model")
    return OpenAICompatibleLanguageModel(
        model=model,
        base_url=config.get("base_url"),
        api_key_env=config.get("api_key_env", "ASTEVOLVE_LLM_API_KEY"),
        timeout_seconds=config.get("timeout_seconds", 300.0),
        provider_name=config.get("provider_name", "openai_compatible"),
    )


__all__ = [
    "OpenAICompatibleError",
    "OpenAICompatibleLanguageModel",
    "create_openai_compatible_model",
]
