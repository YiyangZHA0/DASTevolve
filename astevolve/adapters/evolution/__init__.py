"""Evolution adapters; optional runtimes load only when requested."""
from importlib import import_module

_EXPORTS = {
    "OuterLoopRun": "outerloop_runtime",
    "CallableProposalEvaluator": "native_runtime",
    "CallableProposalSource": "native_runtime",
    "DesignSearchProposalEvaluator": "native_runtime",
    "StaticRevisionProposalSource": "native_runtime",
    "LLMProposalError": "llm_proposal",
    "LLMProposalPolicy": "llm_proposal",
    "StructuredLLMProposalSource": "llm_proposal",
    "NativeCaseProposalEvaluator": "case_runtime",
    "NativeCaseRuntimeError": "case_runtime",
    "create_case_design_evaluator": "case_runtime",
    "NativeLLMRuntimeError": "configured_llm",
    "create_configured_llm_proposal_source": "configured_llm",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
