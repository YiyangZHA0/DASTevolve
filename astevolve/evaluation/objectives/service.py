

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from astevolve.evaluation.objectives.scorers import (
    _score_objective,
    supported_objective_types,
)
from astevolve.evaluation.objectives.support import _clamp01


EVALUATOR_TERM_OBJECTIVE_TYPE = "evaluator_term"
EVALUATOR_TERM_ROLES = frozenset(
    {
        "hard_gate",
        "soft_archive_descriptor",
        "soft_hidden_label_audit",
        "soft_objective",
        "soft_report_only",
    }
)


def validate_multistate_objective_specs(
    objective_specs: Optional[Sequence[Dict[str, Any]]],
    *,
    evaluator_term_binding_available: bool = False,
) -> Dict[str, Any]:


    errors: List[str] = []
    seen_names: set[str] = set()
    runtime_bound_terms = 0
    specs = list(objective_specs or [])
    supported_types = set(supported_objective_types())
    for index, raw_spec in enumerate(specs, start=1):
        if not isinstance(raw_spec, dict):
            errors.append(f"objective_spec_not_mapping:{index}")
            continue
        name = str(raw_spec.get("name") or f"objective_{index}").strip()
        if name in seen_names:
            errors.append(f"duplicate_objective_name:{name}")
        seen_names.add(name)

        explicit_type = str(raw_spec.get("type") or "").strip().lower()
        legacy_kind = str(raw_spec.get("kind") or "").strip().lower()
        objective_type = explicit_type or legacy_kind
        role = str(raw_spec.get("role") or "").strip().lower()

        if (
            not explicit_type
            and raw_spec.get("term")
            and legacy_kind in EVALUATOR_TERM_ROLES
        ):
            errors.append(
                f"legacy_evaluator_role_in_kind:{name}:{legacy_kind}"
            )
            runtime_bound_terms += 1
            continue
        if objective_type == EVALUATOR_TERM_OBJECTIVE_TYPE:
            runtime_bound_terms += 1
            term = str(raw_spec.get("term") or "").strip()
            if not term:
                errors.append(f"evaluator_term_missing_term:{name}")
            if role not in EVALUATOR_TERM_ROLES:
                errors.append(
                    f"evaluator_term_invalid_role:{name}:{role or 'missing'}"
                )
            if not evaluator_term_binding_available:
                errors.append(
                    f"evaluator_term_binding_unavailable:{name}:{term or 'missing'}"
                )
            continue
        if not objective_type:
            errors.append(f"objective_type_missing:{name}")
        elif objective_type not in supported_types:
            errors.append(f"unsupported_objective_type:{name}:{objective_type}")

    return {
        "valid": not errors,
        "errors": errors,
        "objective_count": len(specs),
        "runtime_bound_term_count": runtime_bound_terms,
        "evaluator_term_binding_available": bool(evaluator_term_binding_available),
    }


def evaluate_multistate_objectives(
    complex_summary: Dict[str, Any],
    objective_specs: Optional[Sequence[Dict[str, Any]]],
    compiled: Optional[Dict[str, Any]] = None,
    design_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:


    specs = [dict(spec) for spec in (objective_specs or []) if isinstance(spec, dict)]
    by_state = {
        str(state.get("name")): state
        for state in (complex_summary.get("states", []) or [])
        if isinstance(state, dict) and state.get("name")
    }
    if not specs:
        return {
            "enabled": False,
            "weighted_score": 0.0,
            "weight_sum": 0.0,
            "normalized_score": 0.0,
            "loss": 0.0,
            "objectives": {},
            "warnings": [],
            "supported_objective_types": supported_objective_types(),
        }

    design_state = design_state or ((compiled or {}).get("_design_state", {}) if compiled else {})
    objectives: Dict[str, Any] = {}
    warnings: List[str] = []
    weighted = 0.0
    weight_sum = 0.0

    for index, spec in enumerate(specs, start=1):
        name = str(spec.get("name") or f"{spec.get('type', 'objective')}_{index}")
        weight = float(spec.get("weight", 1.0))
        score, details, obj_warnings = _score_objective(by_state, spec, compiled, design_state if isinstance(design_state, dict) else {})
        score = _clamp01(score)
        weighted += weight * score
        weight_sum += abs(weight)
        objectives[name] = {
            "type": spec.get("type", spec.get("kind")),
            "weight": weight,
            "score": score,
            "details": details,
            "warnings": obj_warnings,
        }
        warnings.extend(f"{name}: {warning}" for warning in obj_warnings)

    normalized = _clamp01(weighted / weight_sum) if weight_sum > 0 else 0.0
    return {
        "enabled": True,
        "weighted_score": float(weighted),
        "weight_sum": float(weight_sum),
        "normalized_score": float(normalized),
        "loss": float(1.0 - normalized),
        "objectives": objectives,
        "warnings": warnings,
        "supported_objective_types": supported_objective_types(),
    }
