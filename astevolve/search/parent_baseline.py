

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

from astevolve.search.artifact_io import _seqs_hash
from astevolve.search.energy_reporting import fast_energy_record
from astevolve.search.candidate_validation import (
    _candidate_fast_filter,
    _required_final_mutation_coverage,
)
from astevolve.search.config import SAConfig
from astevolve.search.semantic_coverage import _semantic_coverage_hard_enabled


PARENT_BASELINE_COMPARISON_VERSION = "astevolve.parent_baseline_comparison.v1"
STRUCTURE_EVALUATION_DISPATCH_VERSION = "astevolve.structure_evaluation_dispatch.v1"


def is_parent_baseline(candidate: Mapping[str, Any]) -> bool:


    return bool(candidate.get("is_parent_baseline")) or (
        str(candidate.get("variant_id") or "") == "root"
        and str(candidate.get("candidate_role") or "") == "parent_baseline"
    )


def build_parent_baseline_candidate(
    seqs: Mapping[str, str],
    breakdown: Mapping[str, Any],
    progen: Mapping[str, Any],
    fast_loss: float,
    *,
    template_seqs: Optional[Dict[str, str]],
    fixed_residues: Optional[Dict[str, Dict[int, str]]],
    compiled: Dict[str, Any],
    cfg: SAConfig,
) -> Dict[str, Any]:


    root_seqs = {str(chain_id): str(sequence) for chain_id, sequence in seqs.items()}
    fast_filter = _candidate_fast_filter(
        root_seqs, template_seqs, fixed_residues, compiled, cfg
    )
    candidate = {
        "variant_id": "root",
        "parent_id": None,
        "candidate_role": "parent_baseline",
        "is_parent_baseline": True,
        "seq_hash": _seqs_hash(root_seqs),
        "seqs": root_seqs,
        "fast_loss": float(fast_loss),
        "constraint_penalty": float(breakdown.get("total", 0.0) or 0.0),
        "progen_loglik_avg": float(progen.get("loglik_avg", 0.0) or 0.0),
        "progen_loglik_sum": float(progen.get("loglik_sum", 0.0) or 0.0),
        "energy": fast_energy_record(
            fast_loss=fast_loss,
            constraint_penalty=breakdown.get("total", 0.0),
            progen_loglik_avg=progen.get("loglik_avg", 0.0),
            progen_weight=cfg.progen_weight,
            hard_gate_pass=bool(fast_filter.get("pass", True)),
        ),
        "fast_filter": fast_filter,
    }


    if _semantic_coverage_hard_enabled(cfg):
        candidate["semantic_final_mutation_coverage"] = (
            _required_final_mutation_coverage(
                root_seqs, template_seqs, compiled, cfg
            )
        )
    return candidate


def include_parent_baseline(
    selected_mutants: List[Dict[str, Any]],
    parent: Mapping[str, Any],
) -> List[Dict[str, Any]]:


    parent_hash = str(parent.get("seq_hash") or _seqs_hash(parent["seqs"]))
    return [dict(parent)] + [
        candidate
        for candidate in selected_mutants
        if not is_parent_baseline(candidate)
        and str(candidate.get("seq_hash") or _seqs_hash(candidate["seqs"]))
        != parent_hash
    ]


def evaluate_structure_candidate_dispatched(
    candidate: Dict[str, Any],
    *,
    evaluator: Callable[..., Dict[str, Any]],
    **kwargs: Any,
) -> Dict[str, Any]:


    result = evaluator(candidate, **kwargs)
    result["structure_evaluation_dispatch"] = {
        "schema_version": STRUCTURE_EVALUATION_DISPATCH_VERSION,
        "scope": "inner_run",
        "cache_hit": False,
        "evaluation_invoked": True,
        "reevaluated": True,
        "backend_cache_observable": False,
        "backend_cache_hit": None,
    }
    return result


def public_chai_results(
    results: List[Dict[str, Any]],
    *,
    limit: int = 10,
) -> List[Dict[str, Any]]:


    if limit <= 0 or not results:
        return []
    parent_results = [item for item in results if is_parent_baseline(item)]
    if not parent_results:
        return results[:limit]
    highest_fidelity_parent = parent_results[-1]
    non_parent = [item for item in results if not is_parent_baseline(item)]
    return [highest_fidelity_parent] + non_parent[: max(0, limit - 1)]


def build_parent_baseline_comparison(
    parent: Mapping[str, Any],
    evaluated_results: List[Dict[str, Any]],
    selection_results: List[Dict[str, Any]],
    selection_decision: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:


    parent_evaluations = [
        item for item in evaluated_results if is_parent_baseline(item)
    ]
    parent_selection = next(
        (item for item in selection_results if is_parent_baseline(item)), None
    )
    decision = dict(selection_decision or {})
    parent_selection_id = (
        str(parent_selection.get("selection_candidate_id") or "root")
        if parent_selection is not None
        else None
    )
    ordered_ids = [str(value) for value in decision.get("ordered_ids", []) or []]
    selection_row = (
        dict(parent_selection.get("feasibility_selection") or {})
        if parent_selection is not None
        else {}
    )
    selected_candidate_id = decision.get("selected_candidate_id")
    selected_is_parent = bool(
        parent_selection_id is not None
        and str(selected_candidate_id) == parent_selection_id
    )

    evaluations: List[Dict[str, Any]] = []
    for item in parent_evaluations:
        dispatch = dict(item.get("structure_evaluation_dispatch") or {})
        evaluations.append(
            {
                "stage": str(item.get("structure_stage") or "unknown"),
                "provider": str(item.get("structure_provider") or "unknown"),
                "model_name": item.get("structure_model_name"),
                "dispatch_cache_hit": bool(dispatch.get("cache_hit", False)),
                "evaluation_invoked": bool(dispatch.get("evaluation_invoked", False)),
                "reevaluated": bool(dispatch.get("reevaluated", False)),
                "backend_cache_observable": bool(
                    dispatch.get("backend_cache_observable", False)
                ),
                "backend_cache_hit": dispatch.get("backend_cache_hit"),
            }
        )

    return {
        "schema_version": PARENT_BASELINE_COMPARISON_VERSION,
        "parent_variant_id": "root",
        "parent_seq_hash": str(parent.get("seq_hash") or ""),
        "candidate_role": "parent_baseline",
        "mutant_quota_consumed": False,
        "evaluated": bool(parent_evaluations),
        "evaluations": evaluations,
        "selection": {
            "included": parent_selection is not None,
            "candidate_id": parent_selection_id,
            "feasible": selection_row.get("feasible"),
            "gate_reasons": list(selection_row.get("gate_reasons", []) or []),
            "raw_objective": selection_row.get("raw_objective"),
            "rank": (
                ordered_ids.index(parent_selection_id) + 1
                if parent_selection_id in ordered_ids
                else None
            ),
            "selected_candidate_id": selected_candidate_id,
            "selected_is_parent": selected_is_parent,
            "selected_stage": (
                parent_selection.get("structure_stage")
                if selected_is_parent and parent_selection is not None
                else None
            ),
        },
    }


__all__ = [
    "PARENT_BASELINE_COMPARISON_VERSION",
    "STRUCTURE_EVALUATION_DISPATCH_VERSION",
    "build_parent_baseline_candidate",
    "build_parent_baseline_comparison",
    "evaluate_structure_candidate_dispatched",
    "include_parent_baseline",
    "is_parent_baseline",
    "public_chai_results",
]
