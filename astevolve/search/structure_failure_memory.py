

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
import hashlib
import json
import math
from typing import Any, Dict, Optional

from astevolve.search.artifact_io import _seqs_hash


STRUCTURE_FAILURE_MEMORY_VERSION = "astevolve.structure_failure_memory.v1"
STRUCTURE_FAILURE_CONTEXT_VERSION = "astevolve.structure_failure_context.v2"
STRUCTURE_FAILURE_SUPPRESSION_VERSION = (
    "astevolve.structure_failure_suppression.v1"
)
DEFAULT_MAX_EXACT_FAILURES = 256
DEFAULT_MAX_REHABILITATED = 64

_SCIENTIFIC_SCORE_FIELDS = frozenset(
    {
        "evaluator_backends",
        "evaluator_plugins",
        "evaluator_weights",
        "fast_loss_nonneg",
        "hard_gate_disqualified_score",
        "hard_gate_failure_score_scale",
        "inner_evaluator_loss_weight",
        "inner_hard_gate_fail_penalty",
        "plddt_scale",
        "clash_scale",
        "plugin_config",
        "plugin_resolution",
    }
)

_STRUCTURE_PROTOCOL_FIELDS = (
    "structure_model",
    "structure_model_name",
    "structure_prescreen_enabled",
    "structure_prescreen_model",
    "structure_prescreen_model_name",
    "structure_prescreen_top_frac",
    "structure_prescreen_min_candidates",
    "structure_prescreen_max_candidates",
    "structure_prescreen_forward_all_to_screen",
    "structure_screen_enabled",
    "structure_screen_model",
    "structure_screen_model_name",
    "structure_rerank_enabled",
    "structure_rerank_model",
    "structure_rerank_model_name",
    "structure_service_backend",
    "structure_service_url",
    "protenix_model_name",
    "protenix_seed",
    "protenix_complex_use_msa",
    "protenix_complex_cycle",
    "protenix_complex_step",
    "protenix_complex_sample",
    "protenix_complex_use_default_params",
    "af3_model_dir",
    "af3_run_data_pipeline",
    "af3_db_dir",
    "af3_num_recycles",
    "af3_num_diffusion_samples",
    "af3_flash_attention_implementation",
    "esmfold2_mode",
    "esmfold2_num_loops",
    "esmfold2_num_sampling_steps",
    "esmfold2_num_diffusion_samples",
)


def _provider_name(value: Any) -> str:
    name = str(value or "unknown").strip().lower()
    aliases = {
        "af3": "alphafold3",
        "alpha_fold_3": "alphafold3",
    }
    return aliases.get(name, name)


def _canonical(value: Any) -> Any:


    if isinstance(value, Mapping):
        return {
            str(key): _canonical(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else str(value)
    return str(value)


def _digest(value: Any) -> str:
    payload = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scientific_score_projection(
    score_config: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:


    config = score_config if isinstance(score_config, Mapping) else {}
    selected: Dict[str, Any] = {}
    for key, value in config.items():
        name = str(key)
        if (
            name in _SCIENTIFIC_SCORE_FIELDS
            or name.startswith("weight_")
            or name.endswith("_weight")
            or name.endswith("_penalty")
            or name.endswith("_scale")
        ):
            selected[name] = _canonical(value)
    return {key: selected[key] for key in sorted(selected)}


def _structure_protocol_projection(cfg: Any) -> Dict[str, Any]:
    return {
        field: _canonical(getattr(cfg, field, None))
        for field in _STRUCTURE_PROTOCOL_FIELDS
    }


def _terminal_provider(cfg: Any) -> tuple[str, str, Any]:


    if bool(getattr(cfg, "structure_screen_enabled", False)) and bool(
        getattr(cfg, "structure_rerank_enabled", False)
    ):
        dispatch = _provider_name(getattr(cfg, "structure_rerank_model", ""))
        model_name = (
            getattr(cfg, "structure_rerank_model_name", None)
            or getattr(cfg, "structure_model_name", None)
        )
    elif bool(getattr(cfg, "structure_screen_enabled", False)):
        dispatch = _provider_name(getattr(cfg, "structure_screen_model", ""))
        model_name = (
            getattr(cfg, "structure_screen_model_name", None)
            or getattr(cfg, "structure_model_name", None)
        )
    else:
        dispatch = _provider_name(getattr(cfg, "structure_model", ""))
        model_name = getattr(cfg, "structure_model_name", None)

    effective = dispatch
    if dispatch == "service":
        effective = _provider_name(
            getattr(cfg, "structure_service_backend", "unknown")
        )
    if not model_name:
        if effective == "protenix":
            model_name = getattr(cfg, "protenix_model_name", None)
        elif effective == "alphafold3":
            model_name = getattr(cfg, "af3_model_dir", None)
    return effective, dispatch, model_name


def build_structure_failure_context(
    cfg: Any,
    score_config: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:


    provider, dispatch_provider, model_name = _terminal_provider(cfg)
    protocol = _structure_protocol_projection(cfg)
    objective = {
        "direction": "minimize",
        "selection_objective": str(
            getattr(cfg, "structure_selection_objective", "legacy_additive")
        ).strip().lower(),
        "scientific_score_config": _scientific_score_projection(score_config),
        "multistate_objectives_enabled": bool(
            getattr(cfg, "multistate_objectives_enabled", True)
        ),
        "multistate_objective_weight": float(
            getattr(cfg, "multistate_objective_weight", 1.0)
        ),
    }
    protocol["terminal_provider"] = provider
    protocol["dispatch_provider"] = dispatch_provider
    protocol["terminal_model_name"] = _canonical(model_name or provider)
    model_identity_digest = _digest(protocol)
    objective_digest = _digest(objective)
    identity = {
        "schema_version": STRUCTURE_FAILURE_CONTEXT_VERSION,
        "provider": provider,
        "dispatch_provider": dispatch_provider,
        "model_identity_digest": model_identity_digest,
        "objective_digest": objective_digest,
        "direction": "minimize",
    }
    return {
        **identity,
        "context_key": _digest(identity)[:32],
        "selection_objective": objective["selection_objective"],
    }


def _validated_context(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    required = (
        "context_key",
        "provider",
        "dispatch_provider",
        "model_identity_digest",
        "objective_digest",
        "direction",
    )
    if any(not str(value.get(field) or "").strip() for field in required):
        return None
    if str(value.get("schema_version")) != STRUCTURE_FAILURE_CONTEXT_VERSION:
        return None
    identity = {
        "schema_version": STRUCTURE_FAILURE_CONTEXT_VERSION,
        "provider": _provider_name(value.get("provider")),
        "dispatch_provider": _provider_name(value.get("dispatch_provider")),
        "model_identity_digest": str(value.get("model_identity_digest")),
        "objective_digest": str(value.get("objective_digest")),
        "direction": str(value.get("direction")).strip().lower(),
    }
    expected = _digest(identity)[:32]
    if str(value.get("context_key")) != expected:
        return None
    return {
        **identity,
        "context_key": expected,
        "selection_objective": str(
            value.get("selection_objective") or "unknown"
        ),
    }


def _memory_block(memory: Any) -> Optional[Mapping[str, Any]]:
    if not isinstance(memory, Mapping):
        return None
    adaptive = memory.get("adaptive_memory")
    if isinstance(adaptive, Mapping):
        memory = adaptive
    block = memory.get("structure_failure_memory")
    if not isinstance(block, Mapping):
        return None
    if str(block.get("schema_version")) != STRUCTURE_FAILURE_MEMORY_VERSION:
        return None
    return block


def _candidate_hash(candidate: Mapping[str, Any]) -> str:
    seqs = candidate.get("seqs")
    if isinstance(seqs, Mapping) and seqs:
        return _seqs_hash(
            {str(chain): str(sequence) for chain, sequence in seqs.items()}
        )
    return str(candidate.get("seq_hash") or "").strip()


def _failure_reference_roots(entry: Any) -> tuple[str, ...]:


    if not isinstance(entry, Mapping):
        return ()
    values = entry.get("reference_root_seq_hashes")
    roots: set[str] = set()
    if isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray)
    ):
        roots.update(str(value).strip() for value in values)


    singular = str(entry.get("reference_root_seq_hash") or "").strip()
    if singular:
        roots.add(singular)
    roots.discard("")
    return tuple(sorted(roots))


def _failure_matches_reference_root(entry: Any, root_seq_hash: str) -> bool:
    root = str(root_seq_hash or "").strip()
    return bool(root and root in _failure_reference_roots(entry))


def _is_repair(candidate: Mapping[str, Any]) -> bool:
    if any(
        bool(candidate.get(field))
        for field in ("hard_gate_repair", "gate_repair_candidate", "repair_candidate")
    ):
        return True
    if str(candidate.get("proposal_tier") or "").strip().lower() == "repair":
        return True
    move = candidate.get("move")
    if not isinstance(move, Mapping):
        return False
    plan = move.get("mutation_plan")
    if not isinstance(plan, Mapping):
        plan = {}
    return (
        str(plan.get("tier") or "").strip().lower() == "repair"
        or str(plan.get("action") or "").strip().lower() == "repair_node"
        or str(move.get("tier") or "").strip().lower() == "repair"
        or str(move.get("operator_phase") or "").strip().lower() == "repair"
    )


def exact_failure_suppression_reason(
    candidate: Mapping[str, Any],
    memory: Optional[Mapping[str, Any]],
    context: Optional[Mapping[str, Any]],
    *,
    root_seq_hash: str = "",
) -> Optional[str]:


    validated = _validated_context(context)
    block = _memory_block(memory)
    if validated is None or block is None:
        return None
    seq_hash = _candidate_hash(candidate)
    if not seq_hash or seq_hash == str(root_seq_hash or ""):
        return None
    if _is_repair(candidate):
        return None
    contexts = block.get("contexts")
    if not isinstance(contexts, Mapping):
        return None
    context_block = contexts.get(validated["context_key"])
    if not isinstance(context_block, Mapping):
        return None
    stored_context = _validated_context(context_block.get("context"))
    if stored_context is None or stored_context != validated:
        return None
    successes = context_block.get("rehabilitated_sequences")
    if isinstance(successes, Mapping) and seq_hash in successes:
        return None
    failures = context_block.get("exact_failures")
    if not isinstance(failures, Mapping) or seq_hash not in failures:
        return None
    if not _failure_matches_reference_root(
        failures.get(seq_hash), root_seq_hash
    ):
        return None
    return "same_context_exact_expensive_failure"


def suppress_exact_structure_failures(
    candidates: Sequence[Dict[str, Any]],
    *,
    memory: Optional[Mapping[str, Any]],
    context: Optional[Mapping[str, Any]],
    root_seq_hash: str = "",
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:


    validated = _validated_context(context)
    block = _memory_block(memory)
    audit: Dict[str, Any] = {
        "schema_version": STRUCTURE_FAILURE_SUPPRESSION_VERSION,
        "enabled": bool(validated is not None),
        "context_key": (
            validated.get("context_key") if validated is not None else None
        ),
        "candidate_count_before": len(candidates),
        "candidate_count_after": len(candidates),
        "suppressed_candidate_count": 0,
        "suppressed_unique_sequence_count": 0,
        "suppression_reasons": {},
        "exemptions": {
            "root": 0,
            "hard_gate_repair": 0,
            "rehabilitated_success": 0,
            "reference_root_mismatch": 0,
            "reference_root_unavailable": 0,
            "unscoped_legacy_failure": 0,
        },
        "memory_available": bool(block is not None),
    }
    if validated is None or block is None:
        return list(candidates), audit

    contexts = block.get("contexts")
    active_block = (
        contexts.get(validated["context_key"])
        if isinstance(contexts, Mapping)
        else None
    )
    failures = (
        active_block.get("exact_failures")
        if isinstance(active_block, Mapping)
        else {}
    )
    successes = (
        active_block.get("rehabilitated_sequences")
        if isinstance(active_block, Mapping)
        else {}
    )
    failures = failures if isinstance(failures, Mapping) else {}
    successes = successes if isinstance(successes, Mapping) else {}
    audit["active_failure_entry_count"] = len(failures)
    audit["active_failure_reference_count"] = sum(
        len(_failure_reference_roots(entry))
        for entry in failures.values()
    )
    audit["current_root_scoped_failure_entry_count"] = sum(
        1
        for entry in failures.values()
        if _failure_matches_reference_root(entry, root_seq_hash)
    )

    kept: list[Dict[str, Any]] = []
    suppressed_hashes: set[str] = set()
    for candidate in candidates:
        seq_hash = _candidate_hash(candidate)
        if seq_hash and seq_hash == str(root_seq_hash or ""):
            audit["exemptions"]["root"] += 1
            kept.append(candidate)
            continue
        if _is_repair(candidate) and seq_hash in failures:
            audit["exemptions"]["hard_gate_repair"] += 1
            kept.append(candidate)
            continue
        if seq_hash in successes:
            audit["exemptions"]["rehabilitated_success"] += 1
            kept.append(candidate)
            continue
        if seq_hash in failures:
            reference_roots = _failure_reference_roots(
                failures.get(seq_hash)
            )
            if not str(root_seq_hash or "").strip():
                audit["exemptions"]["reference_root_unavailable"] += 1
            elif not reference_roots:
                audit["exemptions"]["unscoped_legacy_failure"] += 1
            elif str(root_seq_hash) not in reference_roots:
                audit["exemptions"]["reference_root_mismatch"] += 1
        reason = exact_failure_suppression_reason(
            candidate,
            memory,
            validated,
            root_seq_hash=root_seq_hash,
        )
        if reason is None:
            kept.append(candidate)
            continue
        audit["suppressed_candidate_count"] += 1
        audit["suppression_reasons"][reason] = (
            int(audit["suppression_reasons"].get(reason, 0)) + 1
        )
        if seq_hash:
            suppressed_hashes.add(seq_hash)
    audit["candidate_count_after"] = len(kept)
    audit["suppressed_unique_sequence_count"] = len(suppressed_hashes)


    audit["suppressed_sequence_hashes"] = sorted(suppressed_hashes)
    return kept, audit


def attach_structure_failure_suppression_summary(
    search_artifacts: MutableMapping[str, Any],
    stage_summaries: Mapping[str, Mapping[str, Any]],
    *,
    context: Mapping[str, Any],
    enabled: bool,
) -> None:


    stages = {
        str(stage): dict(summary["cross_round_failure_suppression"])
        for stage, summary in stage_summaries.items()
        if isinstance(summary.get("cross_round_failure_suppression"), Mapping)
    }
    reasons = sorted(
        {
            str(reason)
            for item in stages.values()
            for reason in (item.get("suppression_reasons") or {})
        }
    )
    search_artifacts["structure_failure_suppression"] = {
        "schema_version": "astevolve.structure_failure_suppression_summary.v1",
        "enabled": bool(enabled),
        "context_key": context.get("context_key"),
        "suppressed_candidate_count": sum(
            int(item.get("suppressed_candidate_count", 0) or 0)
            for item in stages.values()
        ),
        "suppression_reasons": {
            reason: sum(
                int((item.get("suppression_reasons") or {}).get(reason, 0) or 0)
                for item in stages.values()
            )
            for reason in reasons
        },
        "stages": stages,
    }


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bounded_contexts(
    contexts: MutableMapping[str, Any],
    *,
    max_exact_failures: int,
    max_rehabilitated: int,
) -> tuple[int, int]:
    failure_rows = []
    success_rows = []
    for context_key in sorted(contexts):
        block = contexts.get(context_key)
        if not isinstance(block, MutableMapping):
            continue
        failures = block.get("exact_failures")
        if isinstance(failures, MutableMapping):
            for seq_hash, entry in failures.items():
                payload = entry if isinstance(entry, Mapping) else {}
                failure_rows.append(
                    (
                        str(payload.get("last_seen") or ""),
                        int(payload.get("count") or 0),
                        str(context_key),
                        str(seq_hash),
                    )
                )
        successes = block.get("rehabilitated_sequences")
        if isinstance(successes, MutableMapping):
            for seq_hash, entry in successes.items():
                payload = entry if isinstance(entry, Mapping) else {}
                success_rows.append(
                    (
                        str(payload.get("last_seen") or ""),
                        str(context_key),
                        str(seq_hash),
                    )
                )

    keep_failures = {
        (context_key, seq_hash)
        for _seen, _count, context_key, seq_hash in sorted(
            failure_rows, reverse=True
        )[: max(0, int(max_exact_failures))]
    }
    keep_successes = {
        (context_key, seq_hash)
        for _seen, context_key, seq_hash in sorted(
            success_rows, reverse=True
        )[: max(0, int(max_rehabilitated))]
    }
    evicted_failures = 0
    evicted_successes = 0
    for context_key in list(sorted(contexts)):
        block = contexts.get(context_key)
        if not isinstance(block, MutableMapping):
            contexts.pop(context_key, None)
            continue
        failures = block.get("exact_failures")
        if not isinstance(failures, MutableMapping):
            failures = {}
            block["exact_failures"] = failures
        for seq_hash in list(failures):
            if (str(context_key), str(seq_hash)) not in keep_failures:
                failures.pop(seq_hash, None)
                evicted_failures += 1
        successes = block.get("rehabilitated_sequences")
        if not isinstance(successes, MutableMapping):
            successes = {}
            block["rehabilitated_sequences"] = successes
        for seq_hash in list(successes):
            if (str(context_key), str(seq_hash)) not in keep_successes:
                successes.pop(seq_hash, None)
                evicted_successes += 1
        if not failures and not successes:
            contexts.pop(context_key, None)
    return evicted_failures, evicted_successes


def update_structure_failure_memory(
    adaptive_memory: MutableMapping[str, Any],
    feedback: Any,
    *,
    timestamp: str,
    max_exact_failures: int = DEFAULT_MAX_EXACT_FAILURES,
    max_rehabilitated: int = DEFAULT_MAX_REHABILITATED,
) -> Dict[str, Any]:


    summary = {
        "schema_version": STRUCTURE_FAILURE_MEMORY_VERSION,
        "feedback_available": False,
        "failures_added_or_updated": 0,
        "rehabilitated": 0,
        "root_ignored": 0,
        "ineligible_ignored": 0,
        "provider_mismatch_ignored": 0,
        "success_protected_ignored": 0,
        "evicted_failures": 0,
        "evicted_rehabilitated": 0,
    }
    if not isinstance(feedback, Mapping):
        return summary
    context = _validated_context(feedback.get("scientific_context"))
    candidates = feedback.get("candidates")
    if context is None or not isinstance(candidates, Sequence):
        return summary
    summary["feedback_available"] = True

    block = adaptive_memory.get("structure_failure_memory")
    if not isinstance(block, MutableMapping) or str(
        block.get("schema_version")
    ) != STRUCTURE_FAILURE_MEMORY_VERSION:
        block = {
            "schema_version": STRUCTURE_FAILURE_MEMORY_VERSION,
            "max_exact_failures": int(max_exact_failures),
            "max_rehabilitated_sequences": int(max_rehabilitated),
            "contexts": {},
            "total_failure_observations": 0,
            "total_rehabilitations": 0,
        }
        adaptive_memory["structure_failure_memory"] = block
    contexts = block.get("contexts")
    if not isinstance(contexts, MutableMapping):
        contexts = {}
        block["contexts"] = contexts
    context_block = contexts.get(context["context_key"])
    if not isinstance(context_block, MutableMapping) or _validated_context(
        context_block.get("context")
    ) != context:
        context_block = {
            "context": dict(context),
            "exact_failures": {},
            "rehabilitated_sequences": {},
        }
        contexts[context["context_key"]] = context_block
    failures = context_block["exact_failures"]
    successes = context_block["rehabilitated_sequences"]
    root_hash = str(feedback.get("root_seq_hash") or "")
    permitted_providers = {
        _provider_name(context["provider"]),
        _provider_name(context["dispatch_provider"]),
    }

    for row in candidates:
        if not isinstance(row, Mapping):
            summary["ineligible_ignored"] += 1
            continue
        seq_hash = str(row.get("seq_hash") or "").strip()
        if not seq_hash or seq_hash == root_hash or str(
            row.get("candidate_role") or ""
        ) == "parent_baseline":
            summary["root_ignored"] += 1
            failures.pop(seq_hash, None)
            successes.pop(seq_hash, None)
            continue
        if _provider_name(row.get("provider")) not in permitted_providers:
            summary["provider_mismatch_ignored"] += 1
            continue
        if not bool(row.get("structure_signal_available")):
            summary["ineligible_ignored"] += 1
            continue

        delta = _finite(row.get("root_relative_aligned_energy"))
        improved = delta is not None and delta < -1e-12
        selected = bool(row.get("selected"))
        if selected or improved:
            failures.pop(seq_hash, None)
            existing_success = successes.get(seq_hash)
            first_seen = (
                str(existing_success.get("first_seen"))
                if isinstance(existing_success, Mapping)
                and existing_success.get("first_seen")
                else str(timestamp)
            )
            successes[seq_hash] = {
                "first_seen": first_seen,
                "last_seen": str(timestamp),
                "reason": "selected" if selected else "root_relative_improvement",
                "root_relative_aligned_energy": delta,
            }
            summary["rehabilitated"] += 1
            block["total_rehabilitations"] = int(
                block.get("total_rehabilitations") or 0
            ) + 1
            continue
        if seq_hash in successes:
            summary["success_protected_ignored"] += 1
            continue
        existing = failures.get(seq_hash)
        existing = dict(existing) if isinstance(existing, Mapping) else {}
        reason = str(row.get("rejection_reason") or "unselected_expensive_finalist")
        reason_counts = existing.get("failure_reason_counts")
        reason_counts = dict(reason_counts) if isinstance(reason_counts, Mapping) else {}
        reason_counts[reason] = int(reason_counts.get(reason) or 0) + 1
        observed_delta = delta
        prior_best = _finite(existing.get("best_root_relative_aligned_energy"))
        prior_worst = _finite(existing.get("worst_root_relative_aligned_energy"))
        reference_roots = set(_failure_reference_roots(existing))
        if root_hash:
            reference_roots.add(root_hash)
        failures[seq_hash] = {
            "count": int(existing.get("count") or 0) + 1,
            "first_seen": str(existing.get("first_seen") or timestamp),
            "last_seen": str(timestamp),
            "last_root_relative_aligned_energy": observed_delta,
            "best_root_relative_aligned_energy": (
                min(prior_best, observed_delta)
                if prior_best is not None and observed_delta is not None
                else observed_delta if observed_delta is not None else prior_best
            ),
            "worst_root_relative_aligned_energy": (
                max(prior_worst, observed_delta)
                if prior_worst is not None and observed_delta is not None
                else observed_delta if observed_delta is not None else prior_worst
            ),
            "last_aligned_energy": _finite(row.get("aligned_energy")),


            "reference_root_seq_hashes": sorted(reference_roots),
            "last_failure_reason": reason,
            "failure_reason_counts": {
                key: int(reason_counts[key]) for key in sorted(reason_counts)
            },
            "last_gate_reasons": sorted(
                str(value) for value in (row.get("gate_reasons") or [])
            ),
        }
        summary["failures_added_or_updated"] += 1
        block["total_failure_observations"] = int(
            block.get("total_failure_observations") or 0
        ) + 1

    block["max_exact_failures"] = max(0, int(max_exact_failures))
    block["max_rehabilitated_sequences"] = max(
        0, int(max_rehabilitated)
    )
    evicted_failures, evicted_successes = _bounded_contexts(
        contexts,
        max_exact_failures=max_exact_failures,
        max_rehabilitated=max_rehabilitated,
    )
    summary["evicted_failures"] = evicted_failures
    summary["evicted_rehabilitated"] = evicted_successes
    summary["context_key"] = context["context_key"]
    summary["exact_failure_entry_count"] = sum(
        len(item.get("exact_failures", {}))
        for item in contexts.values()
        if isinstance(item, Mapping)
    )
    summary["rehabilitated_entry_count"] = sum(
        len(item.get("rehabilitated_sequences", {}))
        for item in contexts.values()
        if isinstance(item, Mapping)
    )
    return summary


__all__ = [
    "DEFAULT_MAX_EXACT_FAILURES",
    "DEFAULT_MAX_REHABILITATED",
    "STRUCTURE_FAILURE_CONTEXT_VERSION",
    "STRUCTURE_FAILURE_MEMORY_VERSION",
    "STRUCTURE_FAILURE_SUPPRESSION_VERSION",
    "attach_structure_failure_suppression_summary",
    "build_structure_failure_context",
    "exact_failure_suppression_reason",
    "suppress_exact_structure_failures",
    "update_structure_failure_memory",
]
