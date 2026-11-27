

import importlib.util
import json
import time
import numpy as np
import traceback
import concurrent.futures
import contextvars
import hashlib
import os
from astevolve.evaluation.energy_objective import compute_outer_energy_objective
from astevolve.runtime.paths import artifact_root
from astevolve.evaluation.selection import (
    select_feasibility_first,
)
from engine.memory_lifecycle import (
    current_memory_execution_context,
    memory_execution_scope,
)
from engine.history_lifecycle import DuplicateEffectiveContractError


OUTER_INNER_SEED_VERSION = "astevolve.outer_inner_seed.v1"
_PORTABLE_SEARCH_SEED_MODULUS = 2**31
try:
    from outerloop.evaluation_result import EvaluationResult
    from outerloop.candidate_validation import CandidateValidationError
except ModuleNotFoundError:
    from dataclasses import dataclass
    from typing import Any, Dict

    @dataclass
    class EvaluationResult:


        metrics: Dict[str, float]
        artifacts: Dict[str, Any]

    class CandidateValidationError(ValueError):
        def __init__(self, code, message, *, details=None):
            self.code = str(code)
            self.message = str(message)
            self.details = dict(details or {})
            super().__init__(f"{self.code}: {self.message}")


def _run_output_root(program_path=None):
    raw = os.environ.get("ASTEVOLVE_RUN_ROOT") or os.environ.get("ASTEVOLVE_CASE_OUTPUT_ROOT")
    if raw:
        return os.path.abspath(raw)
    root = artifact_root() / "outerloop"
    if program_path:
        root /= _case_id_from_program_path(program_path)
    return str(root)


def _case_id_from_program_path(program_path):
    configured = os.environ.get("ASTEVOLVE_CASE_ID")
    if configured:
        return configured
    try:
        return os.path.basename(os.path.dirname(os.path.abspath(program_path))) or "default"
    except Exception:
        return "default"


def _best_sequence_dir(program_path):
    path = os.path.join(_run_output_root(program_path), "best_sequences")
    os.makedirs(path, exist_ok=True)
    return path


def run_with_timeout(func, args=(), kwargs=None, timeout_seconds=10):


    if kwargs is None: kwargs = {}
    execution_context = contextvars.copy_context()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(execution_context.run, func, *args, **kwargs)
        return fut.result(timeout=timeout_seconds)


def save_fasta(program_path, seqs, energy, combined_score, plddt, avg_plddt_delta, progen, structure_metrics=None):


    try:
        base_name = os.path.splitext(os.path.basename(program_path))[0]
        filename = f"{base_name}_E{energy:.2f}_S{combined_score:.2f}_P{plddt:.1f}.fasta"
        filepath = os.path.join(_best_sequence_dir(program_path), filename)

        with open(filepath, "w") as f:
            f.write(f"# Program: {base_name}\n")
            f.write(f"# Total Loss: {energy}\n")
            f.write(f"# Combined Score: {combined_score}\n")
            f.write(f"# pLDDT: {plddt}\n")
            f.write(f'# avg_plddt_delta: {avg_plddt_delta}\n')
            f.write(f"# ProGen loglik avg: {progen}\n")
            if structure_metrics:
                scalar = structure_metrics.get("scalar", {}) or {}
                interface = structure_metrics.get("interface", {}) or {}
                f.write(f"# ptm: {scalar.get('ptm')}\n")
                f.write(f"# iptm: {scalar.get('iptm')}\n")
                f.write(f"# ranking_score: {scalar.get('ranking_score')}\n")
                f.write(f"# interface_plddt_mean: {interface.get('interface_plddt_mean')}\n")
                f.write(f"# interface_contact_count: {interface.get('total_contact_count')}\n")
            for chain_id, seq in seqs.items():
                f.write(f">Chain_{chain_id}\n")
                f.write(f"{seq}\n")
        return filepath
    except Exception as e:
        print(f"Error saving FASTA: {e}")
        return None


def _clamp01(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _score_from_100(value):
    return _clamp01(float(value) / 100.0) if value is not None else 0.0


def _safe_float_or_none(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact_scalar(summary):
    scalar = (summary or {}).get("scalar", {}) or {}
    keep = ("plddt", "ptm", "iptm", "gpde", "ranking_score", "has_clash", "disorder", "num_recycles")
    return {
        key: _safe_float_or_none(scalar.get(key))
        for key in keep
        if _safe_float_or_none(scalar.get(key)) is not None
    }


def _compact_interface(summary, pair_limit=8):
    interface = (summary or {}).get("interface", {}) or {}
    pairs = interface.get("pairs", {}) or {}
    top_pairs = []
    if isinstance(pairs, dict):
        pair_items = sorted(
            pairs.items(),
            key=lambda item: float((item[1] or {}).get("contact_count") or 0.0),
            reverse=True,
        )[:pair_limit]
        for name, item in pair_items:
            if not isinstance(item, dict):
                continue
            top_pairs.append(
                {
                    "pair": str(name),
                    "contact_count": item.get("contact_count"),
                    "residue_pair_count": item.get("residue_pair_count"),
                    "clash_count": item.get("clash_count"),
                    "interface_plddt_mean": item.get("interface_plddt_mean"),
                    "interface_plddt_min": item.get("interface_plddt_min"),
                }
            )

    return {
        "available": bool(interface.get("available", False)),
        "reason": interface.get("reason"),
        "total_contact_count": interface.get("total_contact_count"),
        "total_residue_pair_count": interface.get("total_residue_pair_count"),
        "clash_count": interface.get("clash_count"),
        "interface_plddt_mean": interface.get("interface_plddt_mean"),
        "interface_plddt_min": interface.get("interface_plddt_min"),
        "top_pairs": top_pairs,
    }


def _compact_node_summary(summary):
    node_summary = (summary or {}).get("node_summary", {}) or {}
    return {
        "node_count": node_summary.get("node_count", 0),
        "node_plddt_mean": node_summary.get("node_plddt_mean"),
        "node_plddt_min": node_summary.get("node_plddt_min"),
        "low_confidence_nodes": list(node_summary.get("low_confidence_nodes", []) or [])[:10],
    }


def _compact_chain_plddt(summary):
    chain_plddt = (summary or {}).get("chain_plddt", {}) or {}
    out = {}
    if isinstance(chain_plddt, dict):
        for chain_id, value in list(chain_plddt.items())[:12]:
            out[str(chain_id)] = value
    return out


def _compact_node_plddt(node_plddt, limit=24):
    items = []
    if isinstance(node_plddt, dict):
        for key, item in node_plddt.items():
            if not isinstance(item, dict):
                continue
            items.append(
                {
                    "key": str(key),
                    "state": item.get("state"),
                    "chain_id": item.get("chain_id"),
                    "kind": item.get("kind"),
                    "name": item.get("name"),
                    "residue_count": item.get("residue_count"),
                    "plddt_mean": item.get("plddt_mean"),
                    "plddt_min": item.get("plddt_min"),
                    "plddt_max": item.get("plddt_max"),
                }
            )
    items.sort(
        key=lambda item: (
            _safe_float_or_none(item.get("plddt_mean")) is None,
            _safe_float_or_none(item.get("plddt_mean")) or 999.0,
        )
    )
    return items[:limit]


def _compact_objectives(multistate_pack):
    compact = {}
    warnings = list((multistate_pack or {}).get("warnings", []) or [])
    objectives = (multistate_pack or {}).get("objectives", {}) or {}
    if not isinstance(objectives, dict):
        return compact, warnings

    detail_keys = (
        "state",
        "states",
        "available",
        "contact_count",
        "residue_pair_count",
        "clash_count",
        "interface_plddt_mean",
        "interface_strength",
        "left_region_coverage",
        "right_region_coverage",
        "coverage",
        "coverage_target",
        "coverage_score",
        "full_contact_count",
        "full_residue_pair_count",
        "off_target_contact_count",
        "off_target_residue_pair_count",
        "off_target_score",
        "contact_count_delta",
        "target_contact_count",
        "target_strength",
        "specificity_ratio",
        "specificity_score",
        "positive_strength",
        "negative_strength",
        "strength_delta",
        "winner_strength",
        "loser_strength",
        "interface_strength",
        "chemistry_score",
        "class_scores",
        "failing_nodes",
        "hard_floor",
        "region_rmsd",
        "apo_path",
        "holo_path",
    )
    for name, item in objectives.items():
        if not isinstance(item, dict):
            continue
        details = item.get("details", {}) or {}
        obj_warnings = list(item.get("warnings", []) or [])
        warnings.extend(f"{name}: {warning}" for warning in obj_warnings)
        compact[str(name)] = {
            "type": item.get("type"),
            "weight": item.get("weight"),
            "score": item.get("score"),
            "details": {key: details.get(key) for key in detail_keys if key in details},
            "warnings": obj_warnings,
        }
    return compact, warnings


def _compact_evaluator_report(report):
    if not isinstance(report, dict) or not report:
        return {}
    category_summary = report.get("category_summary", {}) or {}
    weakest = []
    for item in report.get("weakest_terms", []) or []:
        if not isinstance(item, dict):
            continue
        weakest.append(
            {
                "name": item.get("name"),
                "category": item.get("category"),
                "score": item.get("score"),
                "weight": item.get("weight"),
                "warnings": item.get("warnings", []),
            }
        )
    return {
        "schema_version": report.get("schema_version"),
        "normalized_score": report.get("normalized_score"),
        "soft_score": report.get("soft_score"),
        "loss": report.get("loss"),
        "gate_status": report.get("gate_status", {}),
        "hard_gate_pass": report.get("hard_gate_pass"),
        "disqualification_reasons": report.get("disqualification_reasons", []),
        "category_summary": category_summary,
        "dimension_summary": report.get("dimension_summary", {}),
        "scorer_layers": report.get("scorer_layers", {}),
        "backends": report.get("backends", {}),
        "backend_terms": _compact_backend_terms(report.get("terms", [])),
        "plugins": report.get("plugins", {}),
        "backend_evidence_summary": report.get("backend_evidence_summary", {}),
        "case_specific_terms": report.get("case_specific_terms", {}),
        "weakest_terms": weakest[:8],
        "recommended_edit_targets": report.get("recommended_edit_targets", [])[:8],
        "residue_pair_distance_evidence": report.get("residue_pair_distance_evidence", {}),
        "warnings": report.get("warnings", [])[:20],
    }


def _compact_backend_terms(terms):

    backend_names = {"rosetta", "pyrosetta", "getcontacts", "ipsae", "foldx", "fpocket", "povme"}
    detail_keys = (
        "enabled",
        "required",
        "ignored_for_score",
        "ignored_for_score_reason",
        "reason",
        "method",
        "ok",
        "returncode",
        "command",
        "parsed_interface_dg",
        "parsed_shape_complementarity",
        "parsed_contact_counts",
        "targets",
        "output_path",
        "score_files",
        "row_count",
        "selected_row",
        "ipSAE",
        "pDockQ",
        "pDockQ2",
        "LIS",
        "ipTM_d0chn",
    )
    compact = {}
    for item in terms or []:
        if not isinstance(item, dict):
            continue
        backend = str(item.get("backend") or "")
        if backend not in backend_names:
            continue
        details = item.get("details", {}) if isinstance(item.get("details"), dict) else {}
        enabled = bool(details.get("enabled", True))
        available = bool(item.get("available"))
        warnings = item.get("warnings", [])
        if not enabled and not available and not warnings:
            continue
        kept_details = {key: details.get(key) for key in detail_keys if key in details}
        for key in ("stdout_tail", "stderr_tail", "error"):
            value = details.get(key)
            if value:
                kept_details[key] = str(value)[-500:]
        compact[backend] = {
            "term": item.get("name"),
            "category": item.get("category"),
            "score": item.get("score"),
            "weight": item.get("weight"),
            "available": available,
            "warnings": warnings,
            "details": kept_details,
        }
    return compact


def _objective_detail(details, *keys):
    if not isinstance(details, dict):
        return {}
    return {key: details.get(key) for key in keys if key in details}


def _build_mechanism_summary(out, metrics, objectives, states, round_summary):

    metrics = metrics or {}
    blueprint = out.get("blueprint_summary", {}) if isinstance(out, dict) else {}
    design_points = out.get("case_design_points", {}) if isinstance(out, dict) else {}
    candidate = (round_summary.get("candidate_comparison", {}) or {}) if isinstance(round_summary, dict) else {}
    semantic = (round_summary.get("semantic_coverage", {}) or {}) if isinstance(round_summary, dict) else {}
    functional = (round_summary.get("functional_node_coverage", {}) or {}) if isinstance(round_summary, dict) else {}
    audit = out.get("inner_loop_semantic_audit", {}) if isinstance(out, dict) else {}
    functional_scores = (
        functional.get("functional_node_scores")
        or (audit.get("functional_node_scores", {}) if isinstance(audit, dict) else {})
        or {}
    )
    missing = (
        semantic.get("missing_required_nodes_by_mutation")
        or semantic.get("missing_required_nodes_by_visit")
        or ((candidate.get("failed_constraints", {}) or {}).get("semantic_missing_required_nodes", {}) or {})
    )
    if isinstance(missing, dict):
        missing = sorted(missing)
    missing_functional = (
        functional.get("missing_required_nodes_by_mutation")
        or functional.get("missing_required_nodes_by_visit")
        or functional.get("unavailable_required_nodes")
        or []
    )

    objective_items = {}
    for name, item in (objectives or {}).items():
        if not isinstance(item, dict):
            continue
        details = item.get("details", {}) or {}
        objective_items[name] = {
            "type": item.get("type"),
            "score": item.get("score"),
            "weight": item.get("weight"),
            "key_details": _objective_detail(
                details,
                "state",
                "states",
                "positive_state",
                "negative_state",
                "positive_strength",
                "negative_strength",
                "strength_delta",
                "contact_count_delta",
                "contact_delta_target",
                "winner_strength",
                "loser_strength",
                "interface_strength",
                "contact_residue_count",
                "contact_residues",
                "chemistry_score",
                "class_scores",
                "failing_nodes",
                "region_scores",
                "kinetic_path_score",
            ),
            "warnings": list(item.get("warnings", []) or [])[:6],
        }

    state_confidence = {}
    for name, item in (states or {}).items():
        if not isinstance(item, dict):
            continue
        confidence = item.get("confidence", {}) or {}
        interface = item.get("interface", {}) or {}
        state_confidence[name] = {
            "role": item.get("role"),
            "available": item.get("state_available"),
            "plddt": confidence.get("plddt"),
            "ptm": confidence.get("ptm"),
            "iptm": confidence.get("iptm"),
            "ranking_score": confidence.get("ranking_score"),
            "interface_contact_count": interface.get("total_contact_count"),
            "interface_residue_pair_count": interface.get("total_residue_pair_count"),
            "clash_count": interface.get("clash_count"),
        }

    best_candidates = []
    for item in (candidate.get("best_candidates", []) or [])[:5]:
        if not isinstance(item, dict):
            continue
        best_candidates.append(
            {
                "variant_id": item.get("variant_id"),
                "node": item.get("node"),
                "nodes": item.get("nodes"),
                "op": item.get("op"),
                "tier": item.get("tier"),
                "num_changes": item.get("num_changes"),
                "fast_filter_pass": item.get("fast_filter_pass"),
                "semantic_coverage_pass": item.get("semantic_coverage_pass"),
                "physical_fast_improvement": item.get("physical_fast_improvement"),
                "semantic_final_joint_coverage_pass": item.get("semantic_final_joint_coverage_pass"),
                "joint_success": item.get("joint_success"),
                "raw_occurrence_count": item.get("raw_occurrence_count"),
                "unique_action_count": item.get("unique_action_count"),
                "changes": item.get("changes"),
            }
        )

    action_hints = [
        "If objective-specific scores are weak, rebalance layout_plan node priorities, anchors, residue classes, or motif candidates toward the weakest objective terms.",
        "If structure confidence or interface confidence is weak, reduce disruptive edits and strengthen fold/interface guardrail nodes before expanding the mutation budget.",
        "If inner-loop candidates pass filters but do not improve loss, narrow operator weights and mutation targets toward nodes with stronger recent rewards.",
        "If semantic coverage is weak, rebalance required node priority or split over-broad regions so inner MCTS touches missing structural nodes.",
    ]
    primary_nodes = design_points.get("primary_design_nodes") if isinstance(design_points, dict) else None
    if primary_nodes:
        action_hints.insert(
            0,
            "Prioritize the case_design_points.primary_design_nodes unless the objective report identifies a different bottleneck.",
        )

    return {
        "purpose": "Mechanism-first feedback for AST edits; case-specific target identity is provided in case_design_points.",
        "score": {
            "direction": metrics.get("direction", "minimize"),
            "combined_energy": metrics.get("combined_energy"),
            "final_energy": metrics.get("final_energy"),
            "combined_score": metrics.get("combined_score"),
            "struct_score": metrics.get("structure_score", metrics.get("struct_score")),
            "multistate_score": metrics.get("multistate_score"),
            "evaluator_score": metrics.get("evaluator_score"),
            "hard_gate_pass": metrics.get("hard_gate_pass"),
        },
        "state_confidence": state_confidence,
        "objectives": objective_items,
        "inner_loop_instruction_following": {
            "candidate_count": candidate.get("candidate_count"),
            "raw_candidate_count": candidate.get("raw_candidate_count"),
            "candidate_aggregation": candidate.get("aggregation", {}),
            "success_count": candidate.get("success_count"),
            "fast_filter_pass_count": (candidate.get("effective_constraints", {}) or {}).get("fast_filter_pass_count"),
            "semantic_coverage_pass_count": (candidate.get("effective_constraints", {}) or {}).get("semantic_coverage_pass_count"),
            "physical_fast_outcomes": candidate.get("physical_fast_outcomes", {}),
            "semantic_final_joint_coverage": candidate.get("semantic_final_joint_coverage", {}),
            "joint_outcomes": candidate.get("joint_outcomes", {}),
            "missing_required_nodes": missing,
            "functional_node_coverage_pass": functional.get("pass"),
            "missing_functional_nodes": missing_functional,
            "functional_node_scores": functional_scores,
            "functional_node_success_counts": candidate.get("functional_node_success_counts", {}),
            "functional_node_failure_counts": candidate.get("functional_node_failure_counts", {}),
            "node_success_counts": candidate.get("node_success_counts", {}),
            "node_failure_counts": candidate.get("node_failure_counts", {}),
            "node_proxy_improvement_counts": candidate.get("node_proxy_improvement_counts", {}),
            "node_non_improving_counts": candidate.get("node_non_improving_counts", {}),
            "node_semantic_incomplete_counts": candidate.get("node_semantic_incomplete_counts", {}),
            "best_candidates": best_candidates,
        },
        "outer_loop_action_hint": action_hints,
    }


def _evaluator_energy_residuals(report, limit=24):


    if not isinstance(report, dict):
        return []
    canonical_terms = report.get("term_energy_breakdown")
    if isinstance(canonical_terms, list):
        rows = []
        included_weight = _safe_float_or_none(
            (report.get("energy_coverage", {}) or {}).get("included_weight")
        )
        included_weight = max(0.0, float(included_weight or 0.0))
        for raw in canonical_terms:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            if not name:
                continue
            score = _clamp01(raw.get("score"))
            residual = _clamp01(raw.get("cost", 1.0 - score))
            weight = max(0.0, float(_safe_float_or_none(raw.get("weight")) or 0.0))
            included = bool(raw.get("included", True))
            weighted_residual = (
                max(
                    0.0,
                    float(
                        _safe_float_or_none(raw.get("weighted_cost"))
                        or 0.0
                    ),
                )
                if included
                else 0.0
            )
            rows.append(
                {
                    "term_key": raw.get("term_key"),
                    "name": name,
                    "category": raw.get("category"),
                    "backend": raw.get("backend"),
                    "provider": raw.get("provider"),
                    "state": raw.get("state"),
                    "available": bool(raw.get("available", True)),
                    "required": bool(raw.get("required", False)),
                    "availability_semantics": raw.get("availability_semantics"),
                    "included": included,
                    "score": float(score),
                    "residual": float(residual),
                    "weight": float(weight),
                    "weighted_residual": float(weighted_residual),
                    "normalized_residual_contribution": (
                        float(weighted_residual / included_weight)
                        if included_weight > 0.0 and included
                        else 0.0
                    ),
                }
            )
        rows.sort(key=lambda row: (-row["weighted_residual"], row["name"]))
        return rows[:limit]

    rows = []
    for raw in report.get("terms", []) or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        score = _clamp01(raw.get("score"))
        weight = _safe_float_or_none(raw.get("weight"))
        weight = max(0.0, float(weight or 0.0))
        residual = 1.0 - score
        rows.append(
            {
                "name": name,
                "category": raw.get("category"),
                "backend": raw.get("backend"),
                "provider": raw.get("provider"),
                "state": raw.get("state"),
                "available": bool(raw.get("available", True)),
                "score": float(score),
                "residual": float(residual),
                "weight": float(weight),
                "weighted_residual": float(weight * residual),
            }
        )
    active_weight_sum = sum(row["weight"] for row in rows if row["weight"] > 0.0)
    for row in rows:
        row["normalized_residual_contribution"] = (
            float(row["weighted_residual"] / active_weight_sum)
            if active_weight_sum > 0.0 and row["weight"] > 0.0
            else 0.0
        )
    rows.sort(key=lambda row: (-row["weighted_residual"], row["name"]))
    return rows[:limit]


def _node_structure_energy_residuals(out, limit=24):


    structure = out.get("structure_metrics", {}) if isinstance(out, dict) else {}
    raw = (structure or {}).get("node_plddt", {}) or out.get("node_plddt", {}) or {}
    rows = []
    for item in _compact_node_plddt(raw, limit=limit):
        value = _safe_float_or_none(item.get("plddt_mean"))
        if value is None:
            continue
        rows.append(
            {
                "evidence_key": item.get("key"),
                "node": item.get("name") or item.get("key"),
                "state": item.get("state"),
                "chain_id": item.get("chain_id"),
                "kind": item.get("kind"),
                "metric": "plddt_mean",
                "value": float(value),
                "residual": float(1.0 - _score_from_100(value)),
                "attribution_status": "direct_node_measurement",
                "included_in_combined_energy": False,
            }
        )
    rows.sort(
        key=lambda row: (
            -row["residual"],
            str(row.get("state") or ""),
            str(row.get("node") or ""),
            str(row.get("evidence_key") or ""),
        )
    )
    return rows[:limit]


def _recommended_target_energy_links(report, evaluator_residuals, limit=12):


    if not isinstance(report, dict):
        return []
    by_name = {row["name"]: row for row in evaluator_residuals}
    linked = []
    for raw in (report.get("recommended_edit_targets", []) or [])[:limit]:
        if not isinstance(raw, dict):
            continue
        source_terms = sorted(
            {
                str(name)
                for name in (raw.get("source_terms", []) or [])
                if str(name).strip()
            }
        )
        binding_status = str(raw.get("binding_status") or "").strip()
        explicit = bool(raw.get("node") and source_terms) and not binding_status.startswith(
            "unbound"
        )
        residual_links = []
        for name in source_terms:
            term = by_name.get(name)
            if term is None:
                continue
            residual_links.append(
                {
                    "term": name,
                    "residual": term["residual"],
                    "weight": term["weight"],
                    "weighted_residual": term["weighted_residual"],
                    "normalized_residual_contribution": term[
                        "normalized_residual_contribution"
                    ],
                }
            )
        linked.append(
            {
                "node": raw.get("node"),
                "action": raw.get("action"),
                "priority": raw.get("priority"),
                "reason": raw.get("reason"),
                "source_terms": source_terms,
                "linked_residuals": residual_links,
                "attribution_status": (
                    "declared_semantic_binding"
                    if explicit
                    else binding_status or "recommendation_only_unbound"
                ),
                "causal_attribution": explicit,
                "residual_allocation": "not_allocated_across_nodes",
            }
        )
    return linked


def _build_energy_feedback(out, metrics):


    out = out or {}
    metrics = metrics or {}
    components = []
    for raw in metrics.get("energy_components", []) or []:
        if not isinstance(raw, dict):
            continue
        components.append(
            {
                key: raw.get(key)
                for key in (
                    "name",
                    "source_metric",
                    "weight",
                    "active",
                    "residual",
                    "weighted_residual",
                    "normalized_contribution",
                )
            }
        )
    active_components = sorted(
        (item for item in components if item.get("active")),
        key=lambda item: (
            -float(item.get("normalized_contribution") or 0.0),
            str(item.get("name") or ""),
        ),
    )
    evaluator_report = out.get("evaluator_report", {}) or {}
    evaluator_residuals = _evaluator_energy_residuals(evaluator_report)
    target_links = _recommended_target_energy_links(
        evaluator_report, evaluator_residuals
    )
    direct_node_measurements = _node_structure_energy_residuals(out)
    declared_count = sum(
        1 for row in target_links if row.get("causal_attribution") is True
    )
    return {
        "schema_version": metrics.get(
            "energy_schema_version", "astevolve.outer_energy.v1"
        ),
        "direction": "minimize",
        "range": [0.0, 1.0],
        "aggregation": "positive_weighted_mean_of_bounded_residuals",
        "hard_gate_policy": "separate_feasibility_first",
        "combined_energy": metrics.get("combined_energy"),
        "final_energy": metrics.get("final_energy"),
        "raw_combined_energy": metrics.get("raw_combined_energy"),
        "active_weight_sum": metrics.get("active_energy_weight_sum"),
        "evaluator_energy_contract": {
            "schema_version": evaluator_report.get("energy_schema_version"),
            "direction": evaluator_report.get("direction"),
            "total_energy": evaluator_report.get("total_energy"),
            "soft_energy": evaluator_report.get("soft_energy"),
            "category_energy_breakdown": evaluator_report.get(
                "category_energy_breakdown", {}
            ),
            "coverage": evaluator_report.get("energy_coverage", {}),
        },
        "components": components,
        "highest_residual_components": active_components[:8],
        "evaluator_term_residuals": evaluator_residuals,
        "recommended_edit_targets": target_links,
        "node_attribution": {
            "status": (
                "declared_bindings_available"
                if declared_count
                else "no_declared_energy_to_node_binding"
            ),
            "declared_binding_count": declared_count,
            "policy": (
                "Only evaluator-declared semantic bindings are causal links; "
                "generic fallbacks are recommendations, and measured node pLDDT "
                "residuals are diagnostics rather than allocations of total energy."
            ),
        "direct_node_measurements": direct_node_measurements,
            "recommendation_links": target_links,
            "unattributed_combined_components": [
                str(item.get("name")) for item in active_components
            ],
        },
    }


def _build_objective_vector(out, metrics, trial_records=None):


    report = (out or {}).get("evaluator_report", {}) or {}
    raw_terms = [
        term
        for term in (report.get("terms", []) or [])
        if isinstance(term, dict) and str(term.get("name") or "").strip()
    ]
    term_name_counts = {}
    for term in raw_terms:
        name = str(term.get("name") or "").strip()
        term_name_counts[name] = int(term_name_counts.get(name, 0)) + 1

    terms = {
        str(term.get("name") or "").strip(): term
        for term in raw_terms
        if term_name_counts[str(term.get("name") or "").strip()] == 1
    }

    def details(name):
        raw = terms.get(name, {})
        value = raw.get("details", {}) if isinstance(raw, dict) else {}
        return value if isinstance(value, dict) else {}

    def finite(value):
        number = _safe_float_or_none(value)
        if number is None or not np.isfinite(number):
            return None
        return float(number)

    generated_metric_names = {
        "positive_A_iptm",
        "positive_A_interface_q",
        "positive_A_plddt",
        "apo_plddt",
        "iptm_margin",
        "gpde_margin",
        "interface_q_margin",
        "selectivity_proxy_score",
        "positive_noninferiority_min_ratio",
        "worst_case_score",
        "seed_score_std",
    }
    reserved_metric_names = set(metrics or {}) | generated_metric_names
    projected_terms = {}
    projection_skips = []
    for term in raw_terms:
        name = str(term.get("name") or "").strip()
        if term_name_counts.get(name, 0) != 1:
            projection_skips.append({"name": name, "reason": "duplicate_name"})
            continue
        if term.get("available") is not True:
            projection_skips.append({"name": name, "reason": "not_explicitly_available"})
            continue
        raw_score = term.get("score")
        if isinstance(raw_score, bool):
            projection_skips.append({"name": name, "reason": "non_numeric_score"})
            continue
        score = finite(raw_score)
        if score is None:
            projection_skips.append({"name": name, "reason": "non_finite_score"})
            continue
        if name in reserved_metric_names:
            projection_skips.append({"name": name, "reason": "reserved_metric_name"})
            continue
        projected_terms[name] = _clamp01(score)

    apo = details("tiam1_apo_fold_integrity")
    a_complex = details("tiam1_A_complex_integrity")
    a_iptm = details("tiam1_A_iptm_noninferiority")
    a_q = details("tiam1_A_interface_q_noninferiority")
    iptm_margin = details("tiam1_v2_iptm_margin")
    gpde_margin = details("tiam1_v2_gpde_margin")
    q_margin = details("tiam1_v2_interface_q_margin")

    values = {
        "positive_A_iptm": finite(a_iptm.get("A_iptm")),
        "positive_A_interface_q": finite(a_q.get("A_interface_q")),
        "positive_A_plddt": finite(a_complex.get("plddt")),
        "apo_plddt": finite(apo.get("plddt")),
        "iptm_margin": finite(iptm_margin.get("A_minus_B_iptm")),
        "gpde_margin": finite(gpde_margin.get("B_minus_A_gpde")),
        "interface_q_margin": finite(q_margin.get("A_minus_B_interface_q")),
    }

    floors = {
        "positive_A_iptm": finite(a_iptm.get("absolute_floor")),
        "positive_A_interface_q": finite(a_q.get("absolute_floor")),
        "positive_A_plddt": finite(a_complex.get("absolute_floor")),
        "apo_plddt": finite(apo.get("absolute_floor")),
    }
    floor_ratios = [
        values[name] / floor
        for name, floor in floors.items()
        if floor is not None and floor > 0.0 and values.get(name) is not None
    ]
    vector_metrics = dict(projected_terms)
    vector_metrics.update({
        name: value for name, value in values.items() if value is not None
    })
    if any(value is not None for value in values.values()):
        vector_metrics["selectivity_proxy_score"] = float(
            metrics.get("combined_score", 0.0) or 0.0
        )
    if floor_ratios:
        vector_metrics["positive_noninferiority_min_ratio"] = float(
            min(floor_ratios)
        )

    trial_scores = []
    for record in trial_records or []:
        score_pack = record.get("score_pack", {}) if isinstance(record, dict) else {}
        score = finite(score_pack.get("combined_score"))
        if score is not None:
            trial_scores.append(score)
    if trial_scores:
        vector_metrics["worst_case_score"] = float(min(trial_scores))
        vector_metrics["seed_score_std"] = float(np.std(trial_scores))

    if not vector_metrics:
        return {}

    semantics = {
        "evaluator_term_projection": (
            "unique explicitly-available finite term scores; higher is better"
        ),
        "robustness": "worst evaluated combined score and trial dispersion",
    }
    if any(value is not None for value in values.values()):
        semantics.update(
            {
                "positive_target_quality": "absolute A-state and apo evidence",
                "selectivity": "paired A-minus-B proxy evidence",
                "legacy_scalar": "selectivity_proxy_score aliases combined_score",
            }
        )

    return {
        "schema_version": "astevolve.objective_vector.v1",
        "scientific_semantics": semantics,
        "hard_gate_pass": bool(metrics.get("hard_gate_pass", False)),
        "metrics": vector_metrics,
        "floors": {name: value for name, value in floors.items() if value is not None},
        "evaluator_term_projection": {
            "projected_names": sorted(projected_terms),
            "skipped": projection_skips,
        },
    }


def _build_llm_feedback_summary(out, metrics):
    out = out or {}
    metrics = metrics or {}
    structure = out.get("structure_metrics", {}) or {}
    multistate_pack = out.get("multistate_objectives", {}) or structure.get("multistate_objectives", {}) or {}
    objectives, objective_warnings = _compact_objectives(multistate_pack)

    states = {}
    runtime_warnings = []
    for state in structure.get("states", []) or []:
        if not isinstance(state, dict):
            continue
        name = str(state.get("name") or f"state_{len(states) + 1}")
        summary = state.get("structure_metrics", {}) or {}
        cif_path = summary.get("cif_path") or state.get("cif_path")
        state_available = bool(cif_path or state.get("out_dir") or (summary.get("scalar") or {}))
        if not cif_path:
            runtime_warnings.append(f"{name}: no CIF path; state prediction may have failed")
        states[name] = {
            "role": state.get("role"),
            "objective": state.get("objective"),
            "state_available": state_available,
            "cif_available": bool(cif_path),
            "summary_available": bool(state.get("summary_json")),
            "confidence": _compact_scalar(summary),
            "interface": _compact_interface(summary),
            "chain_plddt": _compact_chain_plddt(summary),
            "node_summary": _compact_node_summary(summary),
        }

    aggregate_node_plddt = structure.get("node_plddt", {}) or out.get("node_plddt", {}) or {}
    search_artifacts = out.get("search_artifacts", {}) or {}
    round_summary = search_artifacts.get("round_summary", {}) or {}
    feedback = {
        "purpose": (
            "Compact ASTevolve feedback for outer-loop edits to node selectors, "
            "functional mappings, residue preferences, and search policy."
        ),
        "score_summary": {
            "direction": metrics.get("direction", "minimize"),
            "combined_energy": metrics.get("combined_energy"),
            "final_energy": metrics.get("final_energy"),
            "combined_score": metrics.get("combined_score"),
            "struct_score": metrics.get("structure_score", metrics.get("struct_score")),
            "total_loss": out.get("fast_loss"),
            "fast_loss": out.get("fast_loss"),
            "design_loss": _design_loss_from_out(out),
            "progen_loglik_avg": out.get("progen_loglik_avg"),
            "plddt": metrics.get("plddt"),
            "ptm": metrics.get("ptm"),
            "iptm": metrics.get("iptm"),
            "ranking_score": metrics.get("ranking_score"),
            "multistate_score": metrics.get("multistate_score"),
            "multistate_loss": out.get("multistate_loss"),
        },
        "energy_feedback": _build_energy_feedback(out, metrics),
        "mechanism_summary": _build_mechanism_summary(out, metrics, objectives, states, round_summary),
        "case_design_points": out.get("case_design_points", {}),
        "case_sheet_summary": out.get("case_sheet_summary", {}),
        "graph_ablation_mode": out.get("graph_ablation_mode", "full"),
        "node_sweep_summary": search_artifacts.get("node_sweep_summary", {}),
        "contract_response_report": out.get("contract_response_report", {}),
        "external_knowledge_policy": out.get("external_knowledge_policy", {}),
        "evaluator": _compact_evaluator_report(out.get("evaluator_report", {})),
        "functional_ast": out.get("functional_ast_summary", {}),
        "semantic_graph": out.get("semantic_graph_summary", {}),
        "semantic_graph_diagnosis": out.get("semantic_graph_diagnosis", {}),
        "edit_contract": out.get("edit_contract", {}),
        "ast_revision_report": out.get("ast_revision_report", {}),
        "parent_sequence_lineage": out.get("parent_sequence_lineage"),
        "parent_effective_ast_lineage": out.get(
            "parent_effective_ast_lineage"
        ),
        "residue_evidence_catalog": out.get(
            "residue_evidence_catalog_summary"
        ),
        "residue_evidence_digest": out.get("residue_evidence_prompt_digest"),
        "effective_structural_scope": (
            out.get("compiled_executable_node_plan", {}) or {}
        ).get("structural_nodes", []),
        "outer_loop_next_strategy_guidance": out.get("outer_loop_next_strategy_guidance", {}),
        "residue_pair_distance_evidence": (out.get("evaluator_report", {}) or {}).get("residue_pair_distance_evidence", {}),
        "backend_evidence_summary": (out.get("evaluator_report", {}) or {}).get("backend_evidence_summary", {}),
        "case_specific_terms": (out.get("evaluator_report", {}) or {}).get("case_specific_terms", {}),
        "inner_loop_candidate_comparison": round_summary.get("candidate_comparison", {}),
        "structure_finalist_feedback": out.get(
            "structure_finalist_feedback",
            search_artifacts.get("structure_finalist_feedback", {}),
        ),
        "functional_node_coverage": round_summary.get("functional_node_coverage", {}),
        "functional_node_scores": (out.get("inner_loop_semantic_audit", {}) or {}).get("functional_node_scores", {}),
        "experiment_analysis_report": round_summary.get("experiment_analysis_report", {}),
        "search_schedule": {
            "outer_loop_phase": out.get("outer_loop_phase"),
            "search_schedule": out.get("search_schedule", {}),
            "iterations": out.get("sa_config", {}).get("iterations") if isinstance(out.get("sa_config"), dict) else None,
            "proposal_tier_mode": out.get("sa_config", {}).get("proposal_tier_mode") if isinstance(out.get("sa_config"), dict) else None,
            "proposal_exploit_frac": out.get("sa_config", {}).get("proposal_exploit_frac") if isinstance(out.get("sa_config"), dict) else None,
            "proposal_explore_frac": out.get("sa_config", {}).get("proposal_explore_frac") if isinstance(out.get("sa_config"), dict) else None,
            "proposal_repair_frac": out.get("sa_config", {}).get("proposal_repair_frac") if isinstance(out.get("sa_config"), dict) else None,
            "max_total_mutations": out.get("sa_config", {}).get("max_total_mutations") if isinstance(out.get("sa_config"), dict) else None,
        },
        "residue_semantic_map": out.get("residue_semantic_map_summary", {}),
        "mutation_semantics": (out.get("inner_loop_semantic_audit", {}) or {}).get("mutation_semantic_summary", {}),
        "inner_loop_semantic_audit": out.get("inner_loop_semantic_audit", {}),
        "aggregate_structure": {
            "confidence": _compact_scalar(structure),
            "interface": _compact_interface(structure, pair_limit=0),
            "node_summary": _compact_node_summary(structure),
        },
        "states": states,
        "nodes_lowest_confidence": _compact_node_plddt(aggregate_node_plddt),
        "objectives": {
            "enabled": multistate_pack.get("enabled"),
            "weighted_score": multistate_pack.get("weighted_score"),
            "weight_sum": multistate_pack.get("weight_sum"),
            "normalized_score": multistate_pack.get("normalized_score"),
            "loss": multistate_pack.get("loss"),
            "items": objectives,
        },
        "objective_warnings": sorted(set(str(w) for w in objective_warnings if w)),
        "runtime_warnings": sorted(set(runtime_warnings)),
    }
    return feedback


def _design_loss_from_out(out):

    out = out or {}
    value = out.get("design_loss")
    if value is None:
        value = out.get("chai_combined_loss")
    if value is None:
        value = out.get("fast_loss")
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _top_dict_keys(values, limit=8):
    if not isinstance(values, dict):
        return []
    items = sorted(
        values.items(),
        key=lambda item: _safe_float_or_none(item[1]) or 0.0,
        reverse=True,
    )
    return [str(key) for key, _value in items[:limit]]


def _build_optimizer_memory_summary(out, metrics):

    out = out or {}
    metrics = metrics or {}
    search_artifacts = out.get("search_artifacts", {}) or {}
    round_summary = search_artifacts.get("round_summary", {}) or {}
    candidate = round_summary.get("candidate_comparison", {}) or {}
    functional = round_summary.get("functional_node_coverage", {}) or {}
    semantic = round_summary.get("semantic_coverage", {}) or {}
    audit = out.get("inner_loop_semantic_audit", {}) or {}
    functional_scores = (
        functional.get("functional_node_scores")
        or audit.get("functional_node_scores", {})
        or {}
    )
    best_mutated = round_summary.get("best_mutated_candidate", {}) or {}
    root_fast = _safe_float_or_none(round_summary.get("root_fast_loss"))
    best_fast = _safe_float_or_none(best_mutated.get("fast_loss"))
    evaluator_report = out.get("evaluator_report", {}) or {}
    guidance = out.get("outer_loop_next_strategy_guidance", {}) or {}
    energy_feedback = _build_energy_feedback(out, metrics)

    return {
        "schema_version": 1,
        "source": "astevolve.evaluation.outerloop",
        "score": {
            "direction": metrics.get("direction", "minimize"),
            "combined_energy": metrics.get("combined_energy"),
            "final_energy": metrics.get("final_energy"),
            "combined_score": metrics.get("combined_score"),
            "fast_loss": out.get("fast_loss"),
            "design_loss": _design_loss_from_out(out),
            "struct_score": metrics.get("structure_score", metrics.get("struct_score")),
            "multistate_score": metrics.get("multistate_score"),
            "evaluator_score": metrics.get("evaluator_score"),
            "hard_gate_pass": metrics.get("hard_gate_pass"),
            "inner_fast_loss_improvement_vs_root": (
                None if root_fast is None or best_fast is None else root_fast - best_fast
            ),
        },
        "functional_nodes": {
            "scores": functional_scores,
            "missing": functional.get("missing_required_nodes_by_mutation")
            or functional.get("missing_required_nodes_by_visit")
            or functional.get("unavailable_required_nodes")
            or [],
            "success_counts": candidate.get("functional_node_success_counts", {}),
            "failure_counts": candidate.get("functional_node_failure_counts", {}),
        },
        "structural_nodes": {
            "successful": _top_dict_keys(candidate.get("node_success_counts", {})),
            "failed": _top_dict_keys(candidate.get("node_failure_counts", {})),
            "missing": semantic.get("missing_required_nodes_by_mutation")
            or semantic.get("missing_required_nodes_by_visit")
            or [],
            "low_confidence": (
                (out.get("structure_metrics", {}) or {}).get("node_summary", {}) or {}
            ).get("low_confidence_nodes", []),
        },
        "best_candidates": (candidate.get("best_candidates", []) or [])[:5],
        "next_round": {
            "avoid": guidance.get("avoid", []),
            "strengthen": guidance.get("strengthen", []),
            "refine": guidance.get("refine", []),
            "action_hints": (
                guidance.get("action_hints")
                or guidance.get("outer_loop_action_hint")
                or []
            ),
            "recommended_edit_targets": evaluator_report.get("recommended_edit_targets", [])[:8],
            "energy_feedback": {
                "direction": energy_feedback.get("direction"),
                "final_energy": energy_feedback.get("final_energy"),
                "highest_residual_components": energy_feedback.get(
                    "highest_residual_components", []
                )[:5],
                "recommended_edit_targets": energy_feedback.get(
                    "recommended_edit_targets", []
                )[:8],
                "node_attribution": {
                    "status": (energy_feedback.get("node_attribution", {}) or {}).get(
                        "status"
                    ),
                    "direct_node_measurements": (
                        energy_feedback.get("node_attribution", {}) or {}
                    ).get("direct_node_measurements", [])[:8],
                },
            },
        },
    }


def _json_artifact(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


_HISTORY_CAUSAL_ARTIFACT_KEYS = (
    "graph_patch",
    "parent_effective_search_contract",
    "effective_search_contract",
    "contract_diff",
    "causal_trace",
    "compiled_graph_patch_hash",
    "mapping_execution",
    "executable_mapping_plan",
    "effective_mapping_schedule",
    "executable_dual_ast",
    "compiled_executable_node_plan",
    "measurement_intents",
    "ablation_control",
    "sequence_generator",
    "node_optimizer",
    "memory_lifecycle",
    "evaluator_plugin_resolution",
    "immutable_sequence_reference",
    "design_action",
    "compiled_design_action",
    "compiled_portfolio_optimization_request",
    "mutation_ownership_ledger",
    "candidate_wave_activation",
)


def _history_causal_artifacts(out):


    if not isinstance(out, dict):
        return {}
    return {key: out.get(key) for key in _HISTORY_CAUSAL_ARTIFACT_KEYS}


def _compute_combined_score(total_loss, out, score_cfg):


    return compute_outer_energy_objective(total_loss, out, score_cfg)


def _outer_trial_gate_sources(out, score_pack):


    report = out.get("evaluator_report", {}) if isinstance(out, dict) else {}
    if isinstance(report, dict) and report:
        evaluator_gate = report
    else:
        evaluator_gate = bool(score_pack.get("hard_gate_pass", 1.0))
    return {"evaluator": evaluator_gate}


def _trial_memory_scope_kwargs(parent_context, program_path, trial_index):


    trial_id = f"trial:{trial_index}"
    if parent_context is not None:
        base_output_dir = parent_context.output_dir
        scope_prefix = parent_context.scope_id or "/".join(
            value
            for value in (
                parent_context.generation_id,
                parent_context.proposal_id,
            )
            if value
        )
        generation_id = parent_context.generation_id
        proposal_id = parent_context.proposal_id
        logical_time = parent_context.logical_time
        snapshot = parent_context.snapshot
        target_path = parent_context.target_path
        edit_contract_envelope_json = parent_context.edit_contract_envelope_json
        edit_contract_response_json = parent_context.edit_contract_response_json
        proposal_causal_envelope_json = (
            parent_context.proposal_causal_envelope_json
        )
        parent_sequence_lineage_json = (
            parent_context.parent_sequence_lineage_json
        )
        parent_effective_ast_lineage_json = (
            parent_context.parent_effective_ast_lineage_json
        )
        design_action_json = parent_context.design_action_json
        design_action_parent_binding_json = (
            parent_context.design_action_parent_binding_json
        )
        memory_scope = parent_context.memory_scope
        memory_policy = parent_context.memory_policy
        history_registry_path = parent_context.history_registry_path
        history_scope = parent_context.history_scope
        history_owner_token = parent_context.history_owner_token
        history_lease_seconds = parent_context.history_lease_seconds
        history_replicate_policy = parent_context.history_replicate_policy
    else:
        base_output_dir = ""
        scope_prefix = "outer-evaluation"
        generation_id = ""
        proposal_id = ""
        logical_time = ""
        snapshot = None
        target_path = ""
        edit_contract_envelope_json = ""
        edit_contract_response_json = ""
        proposal_causal_envelope_json = ""
        parent_sequence_lineage_json = ""
        parent_effective_ast_lineage_json = ""
        design_action_json = ""
        design_action_parent_binding_json = ""
        memory_scope = None
        memory_policy = None
        history_registry_path = ""
        history_scope = ""
        history_owner_token = ""
        history_lease_seconds = 300.0
        history_replicate_policy = "reject"

    if not base_output_dir:
        output_root = _run_output_root(program_path)
        base_output_dir = os.path.join(
            output_root,
            "outer_trials",
            os.path.splitext(os.path.basename(program_path))[0],
        )
    trial_output_dir = os.path.abspath(
        os.path.join(str(base_output_dir), "trials", f"trial_{trial_index:03d}")
    )
    os.makedirs(trial_output_dir, exist_ok=True)
    scope_id = "/".join(value for value in (scope_prefix, trial_id) if value)
    return {
        "generation_id": generation_id,
        "proposal_id": proposal_id,
        "trial_id": trial_id,
        "scope_id": scope_id,
        "logical_time": logical_time,
        "commit_mode": "deferred",
        "snapshot": snapshot,
        "target_path": target_path,
        "output_dir": trial_output_dir,
        "edit_contract_envelope_json": edit_contract_envelope_json,
        "edit_contract_response_json": edit_contract_response_json,
        "proposal_causal_envelope_json": proposal_causal_envelope_json,
        "parent_sequence_lineage_json": parent_sequence_lineage_json,
        "parent_effective_ast_lineage_json": (
            parent_effective_ast_lineage_json
        ),
        "design_action_json": design_action_json,
        "design_action_parent_binding_json": (
            design_action_parent_binding_json
        ),
        "memory_scope": memory_scope,
        "memory_policy": memory_policy,
        "history_registry_path": history_registry_path,
        "history_scope": history_scope,
        "history_owner_token": history_owner_token,
        "history_lease_seconds": history_lease_seconds,
        "history_replicate_policy": history_replicate_policy,
    }


def _outer_inner_search_seed(parent_context, trial_index):


    index = int(trial_index)
    if parent_context is None:
        return index, {
            "schema_version": OUTER_INNER_SEED_VERSION,
            "mode": "direct_trial_index",
            "generation_id": None,
            "proposal_id": None,
            "trial_index": index,
            "derivation_input_hash": None,
            "seed": index,
        }
    generation_id = str(parent_context.generation_id or "").strip()
    proposal_id = str(parent_context.proposal_id or "").strip()
    if (generation_id, proposal_id) in {("", ""), ("initial", "initial")}:
        return index, {
            "schema_version": OUTER_INNER_SEED_VERSION,
            "mode": "initial_trial_index",
            "generation_id": generation_id or None,
            "proposal_id": proposal_id or None,
            "trial_index": index,
            "derivation_input_hash": None,
            "seed": index,
        }
    material = "\0".join(
        (
            OUTER_INNER_SEED_VERSION,
            generation_id,
            proposal_id,
            str(index),
        )
    ).encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    seed = int(digest[:16], 16) % _PORTABLE_SEARCH_SEED_MODULUS
    return seed, {
        "schema_version": OUTER_INNER_SEED_VERSION,
        "mode": "outer_scope_sha256",
        "generation_id": generation_id,
        "proposal_id": proposal_id,
        "trial_index": index,
        "derivation_input_hash": digest,
        "seed": seed,
    }


def _memory_proposal_from_output(out):
    update = out.get("memory_update", {}) if isinstance(out, dict) else {}
    if not isinstance(update, dict):
        return {}, None
    proposal = update.get("proposal")
    return update, proposal if isinstance(proposal, dict) else None


def _memory_proposal_summary(proposal):
    if not isinstance(proposal, dict):
        return None
    return {
        "base_content_hash": proposal.get("base_content_hash"),
        "base_raw_hash": proposal.get("base_raw_hash"),
        "proposed_content_hash": proposal.get("proposed_content_hash"),
        "proposal_hash": proposal.get("proposal_hash"),
        "commit_id": proposal.get("commit_id"),
    }


def _yaml_unchanged_for_trial(proposal, trial_scope):


    snapshot = trial_scope.snapshot
    target_path = (
        str(proposal.get("target_path") or "")
        if isinstance(proposal, dict)
        else ""
    ) or str(trial_scope.target_path or "")
    expected_raw_hash = (
        str(proposal.get("base_raw_hash") or "")
        if isinstance(proposal, dict)
        else ""
    ) or (str(snapshot.raw_hash) if snapshot is not None else "")
    source_exists = (
        bool(proposal.get("base_source_exists", True))
        if isinstance(proposal, dict)
        else bool(snapshot.source_exists)
        if snapshot is not None
        else None
    )
    if not target_path or not expected_raw_hash or source_exists is None:
        return None
    exists_now = os.path.exists(target_path)
    if not source_exists:
        return not exists_now
    if not exists_now or not os.path.isfile(target_path):
        return False
    with open(target_path, "rb") as handle:
        actual_raw_hash = hashlib.sha256(handle.read()).hexdigest()
    return actual_raw_hash == expected_raw_hash


def _validate_trial_memory_records(trial_records, parent_context):


    proposals = []
    for record in trial_records:
        update = record.get("memory_update", {}) or {}
        if bool(update.get("updated")) or isinstance(update.get("commit"), dict):
            raise RuntimeError("outer evaluation trials must not commit memory updates")
        if record.get("yaml_unchanged") is False:
            raise RuntimeError("outer evaluation trial changed shared memory YAML bytes")
        proposal = record.get("memory_proposal")
        if proposal is not None:
            proposals.append(proposal)

    if not proposals:
        return
    base_hashes = {
        (
            str(proposal.get("base_content_hash") or ""),
            str(proposal.get("base_raw_hash") or ""),
        )
        for proposal in proposals
    }
    if any(not content_hash or not raw_hash for content_hash, raw_hash in base_hashes):
        raise ValueError("trial memory proposal base hash is missing")
    if len(base_hashes) != 1:
        raise ValueError(
            "trial memory proposal base hash mismatch: sibling trials used different snapshots"
        )
    if parent_context is not None and parent_context.snapshot is not None:
        expected = (
            parent_context.snapshot.content_hash,
            parent_context.snapshot.raw_hash,
        )
        if next(iter(base_hashes)) != expected:
            raise ValueError(
                "trial memory proposal base hash mismatch: proposal did not inherit parent snapshot"
            )


def _selected_memory_lifecycle(selected_record):
    proposal = selected_record.get("memory_proposal")
    summary = _memory_proposal_summary(proposal) or {}
    scope = selected_record["memory_scope"]
    snapshot = scope.snapshot
    return {
        "schema_version": "astevolve.outer_selected_memory_lifecycle.v1",
        "selected_candidate_id": selected_record["candidate_id"],
        "selected_trial_id": scope.trial_id,
        "input_content_hash": summary.get("base_content_hash")
        or (snapshot.content_hash if snapshot is not None else None),
        "input_raw_hash": summary.get("base_raw_hash")
        or (snapshot.raw_hash if snapshot is not None else None),
        "proposed_content_hash": summary.get("proposed_content_hash"),
        "proposal_hash": summary.get("proposal_hash"),
        "commit_id": summary.get("commit_id"),
        "commit_status": "deferred",
        "yaml_unchanged": selected_record.get("yaml_unchanged"),
    }


def _trial_artifact_row(record, decision_row):


    score_pack = record["score_pack"]
    return {
        "candidate_id": record["candidate_id"],
        "trial_index": record["trial_index"],
        "seed": record["seed"],
        "seed_derivation": record["seed_derivation"],
        "direction": "minimize",
        "raw_combined_energy": float(score_pack["raw_combined_energy"]),
        "combined_energy": float(score_pack["combined_energy"]),
        "final_energy": float(score_pack["final_energy"]),
        "raw_combined_score": float(score_pack["raw_combined_score"]),
        "combined_score": float(score_pack["combined_score"]),
        "hard_gate_pass": bool(decision_row["feasible"]),
        "disqualified": not bool(decision_row["feasible"]),
        "eligible": bool(decision_row["eligible"]),
        "gate_reasons": list(decision_row.get("gate_reasons", []) or []),
        "gate_sources": list(decision_row.get("gate_sources", []) or []),
        "total_loss": float(record["total_loss"]),
        "design_loss": float(record["design_loss"]),
        "progen_loglik_avg": float(record["progen"]),
        "plddt_delta": float(record["plddt_delta"] or 0.0),
        "memory_scope": record["memory_scope"].to_artifact(),
        "memory_proposal": _memory_proposal_summary(record.get("memory_proposal")),
        "yaml_unchanged": record.get("yaml_unchanged"),
    }


def _failed_evaluation(error, *, traceback_text=None):


    artifacts = {
        "error": str(error),
        "direction": "minimize",
    }
    if traceback_text is not None:
        artifacts["traceback"] = traceback_text
    return EvaluationResult(
        metrics={
            "combined_energy": 1.0,
            "final_energy": 1.0,
            "combined_score": 0.0,
            "hard_gate_pass": False,
            "disqualified": True,
        },
        artifacts=artifacts,
    )


def validate_candidate(program_path: str):


    module_name = "candidate_preflight_" + hashlib.sha256(
        os.path.abspath(program_path).encode("utf-8")
    ).hexdigest()[:16]
    try:
        spec = importlib.util.spec_from_file_location(module_name, program_path)
        if spec is None or spec.loader is None:
            raise ImportError("could not create a module spec")
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
    except DuplicateEffectiveContractError:
        raise
    except Exception as exc:
        raise CandidateValidationError(
            "candidate_import_error",
            f"candidate program could not be imported: {type(exc).__name__}: {exc}",
        ) from exc

    preview = getattr(program, "preview_case", None)
    if not callable(preview):
        return {"status": "validation_hook_not_supported"}
    try:
        preview()
    except DuplicateEffectiveContractError:
        raise
    except Exception as exc:
        raise CandidateValidationError(
            "candidate_semantic_compile_error",
            f"{type(exc).__name__}: {exc}",
            details={"exception_type": type(exc).__name__},
        ) from exc
    return {"status": "valid"}


def evaluate(program_path: str):


    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "run_search"):
            return _failed_evaluation("Missing run_search")

        try:
            num_trials = max(1, int(getattr(program, "OUTER_EVALUATION_TRIALS", 1)))
        except (TypeError, ValueError):
            return _failed_evaluation("OUTER_EVALUATION_TRIALS must be an integer")
        parent_memory_context = current_memory_execution_context()
        trial_records = []

        for t in range(num_trials):
            try:
                search_seed, seed_derivation = _outer_inner_search_seed(
                    parent_memory_context,
                    t,
                )
                scope_kwargs = _trial_memory_scope_kwargs(
                    parent_memory_context,
                    program_path,
                    t,
                )
                with memory_execution_scope(**scope_kwargs) as trial_scope:
                    out = run_with_timeout(
                        program.run_search,
                        kwargs={"seed": search_seed},
                        timeout_seconds=None,
                    )
                if not out:
                    print(f"[trial {t}] out is None/False")
                    continue
                if "seqs" not in out:
                    print(f"[trial {t}] missing 'seqs' keys: {list(out.keys())}")
                    continue

                total_loss = float(out.get("fast_loss", 0.0))
                design_loss = _design_loss_from_out(out)
                progen = float(out.get("progen_loglik_avg", 0.0))
                chai_results = out.get("chai_results", [])
                plddt_delta = out.get("plddt_delta", None)
                if len(chai_results) > 0:
                    plddt_delta = chai_results[0].get("plddt_delta", plddt_delta)

                score_cfg = out.get("score_config", {})
                score_pack = _compute_combined_score(total_loss, out, score_cfg)
                memory_update, memory_proposal = _memory_proposal_from_output(out)
                trial_records.append(
                    {
                        "candidate_id": f"trial:{t}",
                        "trial_index": t,
                        "seed": search_seed,
                        "seed_derivation": seed_derivation,
                        "out": out,
                        "score_pack": score_pack,
                        "total_loss": total_loss,
                        "design_loss": design_loss,
                        "progen": progen,
                        "plddt_delta": None if plddt_delta is None else float(plddt_delta),
                        "gate_sources": _outer_trial_gate_sources(out, score_pack),
                        "memory_scope": trial_scope,
                        "memory_update": memory_update,
                        "memory_proposal": memory_proposal,
                        "yaml_unchanged": _yaml_unchanged_for_trial(
                            memory_proposal,
                            trial_scope,
                        ),
                    }
                )

            except DuplicateEffectiveContractError:


                raise
            except Exception as e:
                print(f"[trial {t}] Exception: {e}")
                traceback.print_exc()
                continue

        if not trial_records:
            return _failed_evaluation("All trials failed")
        _validate_trial_memory_records(trial_records, parent_memory_context)

        trial_selection_decision = select_feasibility_first(
            [
                {
                    "candidate_id": record["candidate_id"],
                    "raw_objective": record["score_pack"]["final_energy"],
                    "gate_sources": record["gate_sources"],
                }
                for record in trial_records
            ],
            direction="minimize",
        )
        selected_id = trial_selection_decision["selected_candidate_id"]
        record_by_id = {record["candidate_id"]: record for record in trial_records}
        decision_by_id = {
            row["candidate_id"]: row
            for row in trial_selection_decision["candidates"]
        }
        selected_record = record_by_id[selected_id]
        selected_decision_row = decision_by_id[selected_id]
        best_trial_out = selected_record["out"]
        selected_memory_update_proposal = selected_record.get("memory_proposal")
        selected_memory_lifecycle = _selected_memory_lifecycle(selected_record)
        selected_scores = dict(selected_record["score_pack"])
        selected_scores.update(
            {
                "total_loss": float(selected_record["total_loss"]),
                "fast_loss": float(selected_record["total_loss"]),
                "design_loss": float(selected_record["design_loss"]),
                "progen_loglik_avg": float(selected_record["progen"]),
                "plddt_delta": float(selected_record["plddt_delta"] or 0.0),
                "hard_gate_pass": bool(selected_decision_row["feasible"]),
                "disqualified": not bool(selected_decision_row["feasible"]),
            }
        )
        selected_loss = selected_scores["total_loss"]
        selected_design_loss = selected_scores["design_loss"]
        selected_progen = selected_scores["progen_loglik_avg"]
        selected_plddt_delta = selected_scores["plddt_delta"]
        combined = selected_scores["combined_score"]

        trial_rows = [
            _trial_artifact_row(record, decision_by_id[record["candidate_id"]])
            for record in trial_records
        ]
        numeric_score_keys = (
            "raw_combined_energy",
            "combined_energy",
            "final_energy",
            "combined_score",
            "raw_combined_score",
            "fast_score",
            "structure_score",
            "plddt_score",
            "iptm_score",
            "ptm_score",
            "multistate_score",
            "evaluator_score",
            "evaluator_energy",
        )
        mean_numeric_scores = {
            key: float(np.mean([record["score_pack"].get(key, 0.0) for record in trial_records]))
            for key in numeric_score_keys
        }
        trial_aggregate = {
            "schema_version": "astevolve.outer_trial_aggregate.v1",
            "direction": "minimize",
            "trial_count": num_trials,
            "successful_trial_count": len(trial_records),
            "selected_candidate_id": selected_id,
            "selected_metrics": dict(selected_scores),
            "mean_numeric_scores": mean_numeric_scores,
            "feasible_trial_count": trial_selection_decision["counts"]["feasible"],
            "all_infeasible": trial_selection_decision["all_infeasible"],
            "trials": trial_rows,
        }

        saved_path = "N/A"
        if best_trial_out is not None:
            saved_path = save_fasta(
                program_path,
                best_trial_out["seqs"],
                selected_loss,
                combined,
                selected_scores.get("plddt", 0.0),
                selected_plddt_delta,
                selected_progen,
                best_trial_out.get("structure_metrics"),
            )

        energy_feedback = (
            _build_energy_feedback(best_trial_out, selected_scores)
            if best_trial_out is not None
            else None
        )
        objective_vector = (
            _build_objective_vector(
                best_trial_out,
                selected_scores,
                trial_records=trial_records,
            )
            if best_trial_out is not None
            else {}
        )
        selected_scores.update(objective_vector.get("metrics", {}))
        llm_feedback_summary = (
            _json_artifact(_build_llm_feedback_summary(best_trial_out, selected_scores))
            if best_trial_out is not None
            else None
        )
        optimizer_memory_summary = (
            _json_artifact(_build_optimizer_memory_summary(best_trial_out, selected_scores))
            if best_trial_out is not None
            else None
        )

        return EvaluationResult(
            metrics={
                "raw_combined_energy": selected_scores.get(
                    "raw_combined_energy", selected_scores["final_energy"]
                ),
                "combined_energy": selected_scores["combined_energy"],
                "final_energy": selected_scores["final_energy"],
                "combined_score": combined,
                "raw_combined_score": selected_scores.get("raw_combined_score", combined),
                "struct_score": selected_scores.get("structure_score", 0.0),
                "total_loss": selected_loss,
                "fast_loss": selected_loss,
                "design_loss": selected_design_loss,
                "fast_score": selected_scores.get("fast_score", 0.0),
                "plddt_score": selected_scores.get("plddt_score", 0.0),
                "progen_loglik_avg": selected_progen,
                "plddt": selected_scores.get("plddt", 0.0),
                "plddt_delta": selected_plddt_delta,
                "ptm": selected_scores.get("ptm", 0.0),
                "iptm": selected_scores.get("iptm", 0.0),
                "ranking_score": selected_scores.get("ranking_score", 0.0),
                "interface_plddt_mean": selected_scores.get("interface_plddt_mean", 0.0),
                "interface_contact_count": selected_scores.get("interface_contact_count", 0.0),
                "interface_residue_pair_count": selected_scores.get("interface_residue_pair_count", 0.0),
                "clash_count": selected_scores.get("clash_count", 0.0),
                "node_plddt_mean": selected_scores.get("node_plddt_mean", 0.0),
                "node_plddt_min": selected_scores.get("node_plddt_min", 0.0),
                "multistate_score": selected_scores.get("multistate_score", 0.0),
                "multistate_loss": float(best_trial_out.get("multistate_loss", 0.0) if best_trial_out else 0.0),
                "evaluator_score": selected_scores.get("evaluator_score", 0.0),
                "evaluator_energy": selected_scores.get("evaluator_energy", 1.0),
                "evaluator_loss": selected_scores.get("evaluator_loss", 1.0),
                "hard_gate_pass": bool(selected_scores["hard_gate_pass"]),
                "disqualified": bool(selected_scores["disqualified"]),
                **objective_vector.get("metrics", {}),
            },
            artifacts={
                **_history_causal_artifacts(best_trial_out),
                "trial_selection_decision": trial_selection_decision,
                "trial_aggregate": trial_aggregate,
                "energy_feedback": (
                    _json_artifact(energy_feedback)
                    if energy_feedback is not None
                    else None
                ),
                "objective_vector": objective_vector or None,
                "selected_memory_update_proposal": selected_memory_update_proposal,
                "selected_memory_lifecycle": selected_memory_lifecycle,
                "llm_feedback_summary": llm_feedback_summary,
                "ast_revision_report": (
                    best_trial_out.get("ast_revision_report")
                    if best_trial_out
                    else None
                ),
                "parent_sequence_lineage": (
                    best_trial_out.get("parent_sequence_lineage")
                    if best_trial_out
                    else None
                ),
                "parent_effective_ast_lineage": (
                    best_trial_out.get("parent_effective_ast_lineage")
                    if best_trial_out
                    else None
                ),
                "residue_evidence_catalog_summary": (
                    best_trial_out.get("residue_evidence_catalog_summary")
                    if best_trial_out
                    else None
                ),
                "residue_evidence_prompt_digest": (
                    best_trial_out.get("residue_evidence_prompt_digest")
                    if best_trial_out
                    else None
                ),
                "migration_frontier": (
                    best_trial_out.get("migration_frontier")
                    if best_trial_out
                    else None
                ),
                "optimizer_memory_summary": optimizer_memory_summary,
                "best_seqs": best_trial_out.get("seqs") if best_trial_out else None,
                "segment_scores": best_trial_out.get("segment_scores") if best_trial_out else None,
                "mutation_history": best_trial_out.get("mutation_history") if best_trial_out else None,
                "inner_loop_semantic_audit": best_trial_out.get("inner_loop_semantic_audit") if best_trial_out else None,
                "progen_loglik_avg": best_trial_out.get("progen_loglik_avg") if best_trial_out else None,
                "progen_loglik_sum": best_trial_out.get("progen_loglik_sum") if best_trial_out else None,
                "chai_plddt": best_trial_out.get("chai_plddt") if best_trial_out else None,
                "confidence_metrics": best_trial_out.get("confidence_metrics") if best_trial_out else None,
                "structure_metrics": best_trial_out.get("structure_metrics") if best_trial_out else None,
                "chain_plddt": best_trial_out.get("chain_plddt") if best_trial_out else None,
                "node_plddt": best_trial_out.get("node_plddt") if best_trial_out else None,
                "evaluator_report": best_trial_out.get("evaluator_report") if best_trial_out else None,
                "multistate_objectives": best_trial_out.get("multistate_objectives") if best_trial_out else None,
                "chai_evaluated": best_trial_out.get("chai_evaluated") if best_trial_out else None,
                "chain_lengths": best_trial_out.get("chain_lengths") if best_trial_out else None,
                "blueprint_summary": best_trial_out.get("blueprint_summary") if best_trial_out else None,
                "layout_summary": best_trial_out.get("layout_summary") if best_trial_out else None,
                "strategy_schema_report": best_trial_out.get("strategy_schema_report") if best_trial_out else None,
                "external_knowledge_policy": best_trial_out.get("external_knowledge_policy") if best_trial_out else None,
                "graph_ablation_mode": best_trial_out.get("graph_ablation_mode") if best_trial_out else None,
                "contract_response_report": best_trial_out.get("contract_response_report") if best_trial_out else None,
                "last_contract_response": best_trial_out.get("last_contract_response") if best_trial_out else None,
                "semantic_graph_summary": best_trial_out.get("semantic_graph_summary") if best_trial_out else None,
                "functional_ast_summary": best_trial_out.get("functional_ast_summary") if best_trial_out else None,
                "semantic_graph_diagnosis": best_trial_out.get("semantic_graph_diagnosis") if best_trial_out else None,
                "edit_contract": best_trial_out.get("edit_contract") if best_trial_out else None,
                "applied_edit_contract": best_trial_out.get("applied_edit_contract") if best_trial_out else None,
                "edit_contract_lifecycle": best_trial_out.get("edit_contract_lifecycle") if best_trial_out else None,
                "outer_loop_next_strategy_guidance": (
                    _json_artifact(best_trial_out.get("outer_loop_next_strategy_guidance", {}))
                    if best_trial_out
                    else None
                ),
                "segments": best_trial_out.get("segments") if best_trial_out else None,
                "search_artifacts": best_trial_out.get("search_artifacts") if best_trial_out else None,
                "structure_finalist_feedback": (
                    best_trial_out.get("structure_finalist_feedback")
                    if best_trial_out
                    else None
                ),
                "saved_fasta_path": saved_path,
            }
        )

    except DuplicateEffectiveContractError:
        raise
    except Exception as e:
        return _failed_evaluation(e, traceback_text=traceback.format_exc())

if __name__ == "__main__":
    from astevolve.cases import resolve_case

    case = resolve_case()
    entry = case.root / "cases" / case.case_id / str(case.metadata.get("entry_program", "initial_program.py"))
    print(evaluate(str(entry)))
