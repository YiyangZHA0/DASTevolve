

import asyncio
import logging
import random
from typing import Callable, Dict, List, Optional, Tuple

from outerloop.llm.base import LLMInterface
from outerloop.llm.openai import OpenAILLM
from outerloop.config import LLMModelConfig

logger = logging.getLogger(__name__)

_PROVIDER_REGISTRY: Dict[str, Callable[[LLMModelConfig], LLMInterface]] = {
    "openai": lambda cfg: OpenAILLM(cfg),
}

try:
    from outerloop.llm.claude_code import ClaudeCodeLLM

    _PROVIDER_REGISTRY["claude_code"] = lambda cfg: ClaudeCodeLLM(cfg)
except ImportError:
    pass


def _normalize_provider(value: Optional[str]) -> str:
    provider = str(value or "openai").strip().lower().replace("-", "_")
    return {
        "openai_compatible": "openai",
        "claude": "claude_code",
        "claude_cli": "claude_code",
    }.get(provider, provider)


def register_llm_provider(
    name: str,
    factory: Callable[[LLMModelConfig], LLMInterface],
    *,
    replace: bool = False,
) -> None:


    if not isinstance(name, str) or not name.strip():
        raise ValueError("LLM provider name must be non-empty")
    provider = _normalize_provider(name)
    if not callable(factory):
        raise TypeError("LLM provider factory must be callable")
    if provider in _PROVIDER_REGISTRY and not replace:
        raise ValueError(f"LLM provider {provider!r} is already registered")
    _PROVIDER_REGISTRY[provider] = factory


def available_llm_providers() -> Tuple[str, ...]:
    return tuple(sorted(_PROVIDER_REGISTRY))


def _create_model(model_cfg: LLMModelConfig) -> LLMInterface:


    if model_cfg.init_client:
        return model_cfg.init_client(model_cfg)

    provider = _normalize_provider(getattr(model_cfg, "provider", None))
    try:
        factory = _PROVIDER_REGISTRY[provider]
    except KeyError as exc:
        raise ValueError(
            f"Unknown LLM provider {provider!r}; available: "
            + ", ".join(available_llm_providers())
        ) from exc
    return factory(model_cfg)


class LLMEnsemble:


    def __init__(self, models_cfg: List[LLMModelConfig]):
        self.models_cfg = models_cfg


        self.models = [_create_model(model_cfg) for model_cfg in models_cfg]


        self.weights = [model.weight for model in models_cfg]
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]


        self.random_state = random.Random()

        if (
            models_cfg
            and hasattr(models_cfg[0], "random_seed")
            and models_cfg[0].random_seed is not None
        ):
            self.random_state.seed(models_cfg[0].random_seed)
            logger.debug(
                f"LLMEnsemble: Set random seed to {models_cfg[0].random_seed} for deterministic model selection"
            )


        if len(models_cfg) > 1 or not hasattr(logger, "_ensemble_logged"):
            logger.info(
                f"Initialized LLM ensemble with models: "
                + ", ".join(
                    f"{model.name} (weight: {weight:.2f})"
                    for model, weight in zip(models_cfg, self.weights)
                )
            )
            logger._ensemble_logged = True

    async def generate(self, prompt: str, **kwargs) -> str:

        model = self._sample_model()
        return await model.generate(prompt, **kwargs)

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:

        model = self._sample_model()
        return await model.generate_with_context(system_message, messages, **kwargs)

    def _sample_model(self) -> LLMInterface:

        index = self.random_state.choices(range(len(self.models)), weights=self.weights, k=1)[0]
        sampled_model = self.models[index]
        logger.info(f"Sampled model: {vars(sampled_model)['model']}")
        return sampled_model

    async def generate_multiple(self, prompt: str, n: int, **kwargs) -> List[str]:

        tasks = [self.generate(prompt, **kwargs) for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def parallel_generate(self, prompts: List[str], **kwargs) -> List[str]:

        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def generate_all_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:

        responses = []
        for model in self.models:
            responses.append(await model.generate_with_context(system_message, messages, **kwargs))
        return responses
