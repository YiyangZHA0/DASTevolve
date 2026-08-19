

from __future__ import annotations

from dataclasses import fields, is_dataclass
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TypeVar

from astevolve.search.config import SAConfig
from astevolve.search.run_memory import InnerRunMemory
from engine.experiment_identity import EvaluatorDescriptor
from engine.history_lifecycle import current_history_execution_context
from engine.history_runtime import evaluate_exact_persistently, register_sequence_occurrence


FastScorer = Callable[
    [Dict[str, str], List[tuple[float, Any]], SAConfig, Dict[str, Any]],
    Tuple[Dict[str, float], Dict[str, float], float],
]


StructureEvidence = TypeVar("StructureEvidence")
STRUCTURE_EVIDENCE_CACHE_VERSION = "astevolve.structure_evidence.v3"


_OPERATIONAL_STRUCTURE_CONFIG_FIELDS = {
    "structure_batch_size",
    "structure_parallel_workers",
    "structure_service_url",
    "structure_service_token",
    "structure_service_timeout",
}


def _structure_evidence_cache_scope() -> Optional[str]:


    context = current_history_execution_context()
    if context is None:
        return None
    scope = str(context.scope)
    prefix, separator, lineage = scope.rpartition(":")
    if (
        separator
        and scope.startswith("native:")
        and len(lineage) == 16
        and all(character in "0123456789abcdefABCDEF" for character in lineage)
    ):
        return f"{prefix}:structure-evidence"
    return scope


def _descriptor_config(cfg: Any) -> Any:
    if cfg is None:
        return None
    if is_dataclass(cfg) and not isinstance(cfg, type):
        values = {field.name: getattr(cfg, field.name) for field in fields(cfg)}
    elif isinstance(getattr(cfg, "__dict__", None), dict):
        values = dict(cfg.__dict__)
    else:
        return cfg
    for field in _OPERATIONAL_STRUCTURE_CONFIG_FIELDS:
        values.pop(field, None)
    return values


def _scientific_structure_arguments(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    arguments = dict(kwargs)
    arguments["cfg"] = _descriptor_config(arguments.get("cfg"))
    compiled = arguments.get("compiled")
    if isinstance(compiled, dict):
        normalized_compiled = dict(compiled)
        normalized_compiled.pop("_plddt", None)
        normalized_compiled.pop("_struct_cache", None)
        arguments["compiled"] = normalized_compiled
    return arguments


def _estimated_structure_launches(
    kwargs: Dict[str, Any], *, provider: str = ""
) -> float:
    compiled = kwargs.get("compiled")
    if not isinstance(compiled, dict):
        return 1.0
    design_state = compiled.get("_design_state")
    states = design_state.get("complex_states") if isinstance(design_state, dict) else None
    if not isinstance(states, list):
        return 1.0
    valid_states = [state for state in states if isinstance(state, dict)]
    launches = len(valid_states)
    normalized_provider = str(provider or "").strip().lower().replace("-", "_")
    if normalized_provider in {"protenix", "protenix_mini"} and valid_states:
        cfg = kwargs.get("cfg")
        requested_batch_size = int(getattr(cfg, "structure_batch_size", 0) or 0)
        if requested_batch_size < 0:
            return float(max(1, launches))
        metric_counts: Dict[str, int] = {}
        for state in valid_states:
            metric = str(state.get("metric", "plddt"))
            metric_counts[metric] = metric_counts.get(metric, 0) + 1
        return float(
            sum(
                math.ceil(count / (requested_batch_size or count))
                for count in metric_counts.values()
            )
        )
    return float(max(1, launches))


def evaluate_structure_evidence_persistently(
    sequence: Mapping[str, str],
    *,
    run_memory: Optional[InnerRunMemory],
    provider: str,
    model: str,
    operation: str,
    request_state: Any,
    scientific_settings: Any,
    seed: Optional[int],
    compute: Callable[[], StructureEvidence],
    estimated_cost: float = 1.0,
) -> Tuple[StructureEvidence, Dict[str, Any]]:


    normalized_provider = str(provider or "").strip().lower()
    normalized_model = str(model or "").strip()
    normalized_operation = str(operation or "").strip()
    if not normalized_provider or not normalized_model or not normalized_operation:
        raise ValueError("structure evidence identity fields must be non-empty")
    config = {
        "provider": normalized_provider,
        "operation": normalized_operation,
        "scientific_settings": scientific_settings,
    }
    state = {"provider_request": request_state}
    cache_scope = _structure_evidence_cache_scope()


    try:
        descriptor = EvaluatorDescriptor.create(
            tool=f"structure_evidence:{normalized_provider}",
            tool_version=STRUCTURE_EVIDENCE_CACHE_VERSION,
            model=normalized_model,
            config=config,
            state=state,
            seed=seed,
        )
    except (TypeError, ValueError):
        descriptor = None

    def evaluate_durably() -> Dict[str, Any]:
        evidence, artifact = evaluate_exact_persistently(
            sequence,
            tool=f"structure_evidence:{normalized_provider}",
            tool_version=STRUCTURE_EVIDENCE_CACHE_VERSION,
            model=normalized_model,
            config=config,
            state=state,
            seed=seed,
            compute=compute,
            estimated_cost=float(estimated_cost),


            retry_failed=True,
            cache_scope=cache_scope,
        )
        artifact = dict(artifact)
        artifact["cache_scope"] = cache_scope
        return {
            "evidence": evidence,
            "cache_artifact": artifact,
        }

    if run_memory is None or descriptor is None:
        envelope = evaluate_durably()
        return envelope["evidence"], dict(envelope["cache_artifact"])

    lookup = run_memory.get_or_compute_structure(
        sequence,
        f"{STRUCTURE_EVIDENCE_CACHE_VERSION}|{descriptor.descriptor_hash}",
        evaluate_durably,
    )
    envelope = dict(lookup.value)
    artifact = dict(envelope["cache_artifact"])
    if lookup.cache_hit:


        artifact["served_by_inner_run_cache"] = True
        artifact["evaluation_invoked"] = False
        artifact["occurrence_cache_hit"] = True
    return envelope["evidence"], artifact


def score_fast_with_run_memory(
    seqs: Dict[str, str],
    terms_fast: List[tuple[float, Any]],
    cfg: SAConfig,
    compiled: Dict[str, Any],
    run_memory: Optional[InnerRunMemory],
    *,
    scorer: FastScorer,
) -> Tuple[Dict[str, float], Dict[str, float], float, bool]:


    if run_memory is None:
        breakdown, progen, fast = scorer(seqs, terms_fast, cfg, compiled)
        return breakdown, progen, float(fast), False

    lookup = run_memory.get_or_compute_fast(
        seqs,
        "fast_objective.v1",
        lambda: (lambda result: {
            "breakdown": result[0],
            "progen": result[1],
            "fast": float(result[2]),
        })(scorer(seqs, terms_fast, cfg, compiled)),
    )
    payload = lookup.value
    return (
        dict(payload["breakdown"]),
        dict(payload["progen"]),
        float(payload["fast"]),
        bool(lookup.cache_hit),
    )


def evaluate_structure_with_run_memory(
    candidate: Dict[str, Any],
    *,
    run_memory: Optional[InnerRunMemory],
    provider: str,
    stage: str,
    evaluator: Callable[..., Dict[str, Any]],
    **kwargs: Any,
) -> Dict[str, Any]:


    register_sequence_occurrence(
        candidate["seqs"],
        role="structure_evaluator_input",
        context_id=(
            f"structure:{stage}:{provider}:"
            f"{candidate.get('variant_id') or candidate.get('seq_hash') or 'candidate'}"
        ),
        metadata={"stage": stage, "provider": provider},
    )
    result = dict(
        evaluator(
            candidate,
            provider=provider,
            stage=stage,
            structure_evidence_run_memory=run_memory,
            **kwargs,
        )
    )


    for field in (
        "variant_id",
        "parent_id",
        "seq_hash",
        "seqs",
        "move",
        "is_parent_baseline",
        "causal_context",
        "design_action_provenance",
        "design_action_validation",
        "compiled_portfolio_request_hash",
        "portfolio_realization_receipts",
        "portfolio_pair_receipts",
        "portfolio_realization_summary",
        "portfolio_dispatch_validation",
    ):
        if field in candidate:
            result[field] = candidate[field]

    cache_artifact = result.get("persistent_evaluation_cache")
    inner_hit = bool(
        isinstance(cache_artifact, Mapping)
        and cache_artifact.get("served_by_inner_run_cache")
    )
    result["inner_run_structure_cache_hit"] = inner_hit
    if isinstance(cache_artifact, Mapping):
        occurrence_hit = bool(
            cache_artifact.get("cache_hit")
            or cache_artifact.get("occurrence_cache_hit")
            or inner_hit
        )
        invoked = bool(cache_artifact.get("evaluation_invoked")) and not inner_hit
        dispatch = dict(result.get("structure_evaluation_dispatch") or {})
        dispatch.update(
            {
                "scope": "structure_evidence",
                "cache_hit": occurrence_hit,
                "evaluation_invoked": invoked,
                "reevaluated": invoked,
            }
        )
        result["structure_evaluation_dispatch"] = dispatch
    return result


__all__ = [
    "STRUCTURE_EVIDENCE_CACHE_VERSION",
    "evaluate_structure_evidence_persistently",
    "evaluate_structure_with_run_memory",
    "score_fast_with_run_memory",
]
