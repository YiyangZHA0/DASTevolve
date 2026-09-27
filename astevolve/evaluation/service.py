

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from astevolve.domain import EvaluationReport
from astevolve.evaluation.backends.registry import optional_backend_terms
from astevolve.evaluation.case_feedback import backend_evidence_summary
from astevolve.evaluation.contracts import EvaluatorContext, ScoreTerm
from astevolve.evaluation.gates import gate_status
from astevolve.evaluation.plugins.runner import plugin_terms
from astevolve.evaluation.reporting.aggregation import (
    category_summary,
    design_energy,
    dimension_summary,
    overall_score,
    scorer_layers,
)
from astevolve.evaluation.reporting.evidence import residue_pair_distance_evidence
from astevolve.evaluation.reporting.recommendations import recommended_edit_targets
from astevolve.evaluation.scorers.builtin import append_builtin_terms


def _plugin_term_summaries(
    terms: list[ScoreTerm], plugin_status: Mapping[str, Any]
) -> Dict[str, Any]:


    result: Dict[str, Any] = {}
    for loaded in plugin_status.get("loaded", []) or []:
        if not isinstance(loaded, Mapping) or not loaded.get("name"):
            continue
        plugin_name = str(loaded["name"])
        rows = []
        for term in terms:
            details = term.details if isinstance(term.details, Mapping) else {}
            if details.get("_plugin_name") != plugin_name:
                continue
            report_details = details.get("report_details")
            if not isinstance(report_details, Mapping):
                report_details = {
                    key: details[key]
                    for key in (
                        "dimension",
                        "semantic_binding",
                        "semantic_bindings",
                        "residue_pair_evidence",
                        "hard_gate",
                        "hard_gate_reason",
                        "negative_design_position_plan",
                    )
                    if key in details
                }
            rows.append(
                {
                    "name": term.name,
                    "category": term.category,
                    "score": float(term.score),
                    "weight": float(term.weight),
                    "available": bool(term.available),
                    "backend": term.backend,
                    "details": dict(report_details),
                    "warnings": list(term.warnings)[:6],
                }
            )
        rows.sort(key=lambda item: (item["score"], -item["weight"], item["name"]))
        result[plugin_name] = {"available": bool(rows), "terms": rows[:16]}
    return result


def evaluate_candidate(
    out: Mapping[str, Any],
    *,
    compiled: Optional[Mapping[str, Any]] = None,
    design_state: Optional[Mapping[str, Any]] = None,
    masks: Optional[Mapping[str, Any]] = None,
    template_seqs: Optional[Mapping[str, str]] = None,
    fixed_residues: Optional[Mapping[str, Mapping[Any, str]]] = None,
    score_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:


    resolved_design_state = design_state or {}
    resolved_score_config = score_config or {}
    structure = out.get("structure_metrics", {}) if isinstance(out.get("structure_metrics"), Mapping) else {}
    context = EvaluatorContext(
        out=out,
        structure=structure,
        compiled=compiled,
        design_state=resolved_design_state,
        masks=masks,
        template_seqs=template_seqs,
        fixed_residues=fixed_residues,
        score_config=resolved_score_config,
    )
    terms: list[ScoreTerm] = []
    append_builtin_terms(context, terms)
    plugin_status = plugin_terms(context, terms)
    backend_status = optional_backend_terms(structure, terms, resolved_score_config)
    backend_summary = backend_evidence_summary(terms)
    plugin_summaries = _plugin_term_summaries(terms, plugin_status)

    soft_score = overall_score(terms)
    gate = gate_status(terms)
    normalized = soft_score if gate.get("hard_gate_pass", True) else 0.0
    energy = design_energy(terms)
    recommendations = recommended_edit_targets(terms, gate, resolved_design_state)
    residue_pair_evidence = residue_pair_distance_evidence(terms)
    weakest = sorted((term.to_dict() for term in terms), key=lambda item: (item["score"], -item["weight"]))[:8]
    warnings = []
    for term in terms:
        warnings.extend(f"{term.name}: {warning}" for warning in term.warnings)

    return {
        "schema_version": "ast_evaluator_report_v1",
        "normalized_score": float(normalized),
        "soft_score": float(soft_score),
        "loss": float(1.0 - normalized),
        **energy,
        "gate_status": gate,
        "feasibility_priority": dict(gate.get("feasibility_priority", {}) or {}),
        "hard_gate_pass": bool(gate.get("hard_gate_pass", True)),
        "disqualification_reasons": list(gate.get("disqualification_reasons", []) or []),
        "recommended_edit_targets": recommendations,
        "residue_pair_distance_evidence": residue_pair_evidence,
        "category_summary": category_summary(terms),
        "dimension_summary": dimension_summary(terms),
        "scorer_layers": scorer_layers(terms),
        "backend_evidence_summary": backend_summary,
        "plugin_term_summaries": plugin_summaries,
        "case_specific_terms": plugin_summaries,
        "terms": [term.to_dict() for term in terms],
        "weakest_terms": weakest,
        "warnings": sorted(set(warnings)),
        "plugins": plugin_status,
        "backends": {"fast_default": True, **backend_status},
    }


def evaluate_typed(
    out: Mapping[str, Any],
    *,
    compiled: Optional[Mapping[str, Any]] = None,
    design_state: Optional[Mapping[str, Any]] = None,
    masks: Optional[Mapping[str, Any]] = None,
    template_seqs: Optional[Mapping[str, str]] = None,
    fixed_residues: Optional[Mapping[str, Mapping[Any, str]]] = None,
    score_config: Optional[Mapping[str, Any]] = None,
) -> EvaluationReport:


    legacy_report = evaluate_candidate(
        out,
        compiled=compiled,
        design_state=design_state,
        masks=masks,
        template_seqs=template_seqs,
        fixed_residues=fixed_residues,
        score_config=score_config,
    )
    return EvaluationReport.from_legacy(legacy_report)
