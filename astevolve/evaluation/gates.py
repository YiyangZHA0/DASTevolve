

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from astevolve.evaluation.contracts import ScoreTerm
from astevolve.evaluation.support import safe_float


def gate_status(terms: Sequence[ScoreTerm]) -> Dict[str, Any]:


    term_by_name = {term.name: term for term in terms}
    disqualification_reasons: List[str] = []
    details: List[Dict[str, Any]] = []
    hard_gate_margins: Dict[str, float] = {}

    for term in terms:
        if bool(term.details.get("required", False)) and not term.available:
            reason = f"required_backend_unavailable:{term.backend}"
            disqualification_reasons.append(reason)
            details.append({"reason": reason, "term": term.name, "backend": term.backend, "warnings": list(term.warnings), "details": term.details})

    mutation_guard = term_by_name.get("mutation_scope_guardrail")
    if mutation_guard and mutation_guard.score < 0.999:
        violation_reasons = {str(item.get("reason")) for item in mutation_guard.details.get("violations", []) if isinstance(item, Mapping)}
        if "fixed residue changed" in violation_reasons:
            reason = "fixed_residue_modified"
        elif "mutation outside editable mask" in violation_reasons:
            reason = "mutation_outside_editable_mask"
        else:
            reason = "mutation_scope_guardrail"
        disqualification_reasons.append(reason)
        details.append({"reason": reason, "term": mutation_guard.name, "score": mutation_guard.score, "details": mutation_guard.details})

    chain = term_by_name.get("chain_continuity")
    if chain and chain.score < 1.0:
        disqualification_reasons.append("severe_chain_break")
        details.append({"reason": "severe_chain_break", "term": chain.name, "score": chain.score, "details": chain.details})

    clash = term_by_name.get("severe_clash_filter")
    if clash and clash.score <= 0.05:
        disqualification_reasons.append("severe_clash")
        details.append({"reason": "severe_clash", "term": clash.name, "score": clash.score, "details": clash.details})

    target_confidence = term_by_name.get("target_complex_confidence_floor")
    if target_confidence and target_confidence.score < 0.35:
        disqualification_reasons.append("target_complex_confidence_extremely_low")
        details.append({"reason": "target_complex_confidence_extremely_low", "term": target_confidence.name, "score": target_confidence.score, "details": target_confidence.details})

    for term in terms:
        if not isinstance(term.details, Mapping) or not term.details.get("hard_gate"):
            continue
        threshold = safe_float(term.details.get("hard_gate_min_score"), 0.999)
        threshold = 0.999 if threshold is None else threshold
        hard_gate_margins[str(term.name)] = float(term.score) - float(threshold)
        if term.score >= threshold:
            continue
        raw_reasons = term.details.get("hard_gate_reasons")
        reasons = (
            [str(item) for item in raw_reasons if str(item).strip()]
            if isinstance(raw_reasons, (list, tuple, set))
            else [str(term.details.get("hard_gate_reason") or term.name)]
        )
        if not reasons:
            reasons = [str(term.details.get("hard_gate_reason") or term.name)]
        for reason in reasons:
            disqualification_reasons.append(reason)
            details.append({"reason": reason, "term": term.name, "score": term.score, "details": term.details})

    margin_values = list(hard_gate_margins.values())
    return {
        "hard_gate_pass": not disqualification_reasons,
        "passed": not disqualification_reasons,
        "disqualification_reasons": sorted(set(disqualification_reasons)),
        "hard_failures": sorted(set(disqualification_reasons)),
        "details": details,
        "feasibility_priority": {
            "hard_gate_pass_count": sum(value >= 0.0 for value in margin_values),
            "hard_gate_total": len(margin_values),
            "min_hard_margin": min(margin_values) if margin_values else 0.0,
            "joint_hard_margin": sum(margin_values),
            "hard_gate_margins": hard_gate_margins,
        },
    }
