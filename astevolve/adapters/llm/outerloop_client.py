

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astevolve.application.ports.llm import LanguageModelRequest, LanguageModelResponse


@dataclass
class OuterLoopLanguageModelAdapter:


    client: Any
    provider_name: str = "outerloop"

    async def complete(self, request: LanguageModelRequest) -> LanguageModelResponse:
        kwargs = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.seed is not None:
            kwargs["seed"] = request.seed
        text = await self.client.generate_with_context(
            request.system_message,
            [{"role": message.role, "content": message.content} for message in request.messages],
            **kwargs,
        )
        return LanguageModelResponse(
            text=str(text),
            provider=self.provider_name,
            model=str(getattr(self.client, "model", "")),
        )
