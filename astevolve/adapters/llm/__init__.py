

from .outerloop_client import OuterLoopLanguageModelAdapter
from .registry import (
    LanguageModelFactory,
    LanguageModelProviderRegistry,
    available_language_model_providers,
    create_language_model,
    language_model_providers,
    register_language_model_provider,
)
from .openai_compatible import (
    OpenAICompatibleError,
    OpenAICompatibleLanguageModel,
    create_openai_compatible_model,
)

if "openai_compatible" not in available_language_model_providers():
    register_language_model_provider(
        "openai_compatible", create_openai_compatible_model
    )

__all__ = [
    "LanguageModelFactory",
    "LanguageModelProviderRegistry",
    "OuterLoopLanguageModelAdapter",
    "OpenAICompatibleError",
    "OpenAICompatibleLanguageModel",
    "available_language_model_providers",
    "create_language_model",
    "create_openai_compatible_model",
    "language_model_providers",
    "register_language_model_provider",
]
