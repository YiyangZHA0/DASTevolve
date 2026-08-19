

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict

from astevolve.evaluation.selection import normalize_gate_payload
from astevolve.metrics.structure import metric_value


OUTER_ENERGY_SCHEMA_VERSION = "astevolve.outer_energy.v1"


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _score_from_100(value: Any) -> float:
    return _clamp01(float(value) / 100.0) if value is not None else 0.0


def _safe_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _structure_value(out: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    summary = out.get("structure_metrics", {}) or {}
    if key == "plddt" and not summary:
        return float(out.get("chai_plddt") or default)
    return metric_value(summary, key, default=default)


def compute_outer_energy_objective(
    total_loss: Any,
    out: Mapping[str, Any],
    score_cfg: Mapping[str, Any] | None,
) -> Dict[str, Any]:


    score_cfg = score_cfg or {}
    w_fast = float(score_cfg.get("weight_fast", 0.5))
    w_plddt = float(score_cfg.get("weight_plddt", 0.5))
    w_iptm = float(score_cfg.get("weight_iptm", 0.0))
    w_ptm = float(score_cfg.get("weight_ptm", 0.0))
    w_ranking = float(score_cfg.get("weight_ranking_score", 0.0))
    w_interface_plddt = float(score_cfg.get("weight_interface_plddt", 0.0))
    w_node_min = float(score_cfg.get("weight_node_plddt_min", 0.0))
    w_clash = float(score_cfg.get("weight_clash", 0.0))
    w_multistate = float(score_cfg.get("weight_multistate", 0.0))
    w_evaluator = float(score_cfg.get("weight_evaluator", 0.0))
    plddt_scale = float(score_cfg.get("plddt_scale", 100.0))
    clash_scale = max(1.0, float(score_cfg.get("clash_scale", 10.0)))
    clamp_nonneg = bool(score_cfg.get("fast_loss_nonneg", True))

    loss = max(0.0, total_loss) if clamp_nonneg else float(total_loss)
    fast_score = 1.0 / (1.0 + max(0.0, loss))

    plddt = _structure_value(out, "plddt", out.get("chai_plddt") or 0.0)
    ptm = _structure_value(out, "ptm", 0.0)
    iptm = _structure_value(out, "iptm", 0.0)
    ranking_score = _structure_value(out, "ranking_score", 0.0)
    interface_plddt = _structure_value(out, "interface_plddt_mean", 0.0)
    node_plddt_min = _structure_value(out, "node_plddt_min", 0.0)
    clash_count = _structure_value(out, "clash_count", 0.0)
    has_clash = _structure_value(out, "has_clash", 0.0)
    multistate_pack = out.get("multistate_objectives", {}) or {}
    multistate_score = _clamp01(
        out.get(
            "multistate_score",
            multistate_pack.get("normalized_score", 0.0),
        )
    )

    plddt_score = (
        _clamp01(float(plddt) / plddt_scale) if plddt is not None else 0.0
    )
    iptm_score = _clamp01(iptm)
    ptm_score = _clamp01(ptm)
    ranking_score_component = _clamp01(ranking_score)
    interface_plddt_score = _score_from_100(interface_plddt)
    node_plddt_min_score = _score_from_100(node_plddt_min)
    clash_penalty = max(
        _clamp01(has_clash),
        _clamp01(float(clash_count) / clash_scale),
    )

    evaluator_report = out.get("evaluator_report", {}) or {}
    evaluator_score = _clamp01(
        out.get(
            "evaluator_score",
            evaluator_report.get("normalized_score", 0.0),
        )
    )
    evaluator_soft_score = _clamp01(
        out.get(
            "evaluator_soft_score",
            evaluator_report.get("soft_score", evaluator_score),
        )
    )
    evaluator_energy_value = _safe_float_or_none(
        evaluator_report.get("total_energy")
    )
    evaluator_energy = (
        _clamp01(evaluator_energy_value)
        if evaluator_energy_value is not None
        else 1.0 - evaluator_soft_score
    )
    gate_decision = normalize_gate_payload(evaluator_report)
    hard_gate_pass = bool(gate_decision["passed"])
    structure_score = (
        (w_plddt * plddt_score)
        + (w_iptm * iptm_score)
        + (w_ptm * ptm_score)
        + (w_ranking * ranking_score_component)
        + (w_interface_plddt * interface_plddt_score)
        + (w_node_min * node_plddt_min_score)
        + (w_multistate * multistate_score)
        + (w_evaluator * evaluator_score)
        - (w_clash * clash_penalty)
    )
    raw_combined = (w_fast * fast_score) + structure_score
    combined = raw_combined
    disqualified_score = float(
        score_cfg.get("hard_gate_disqualified_score", 0.0)
    )
    applied_gate_scale = 1.0
    if not hard_gate_pass:
        failure_scale = float(
            score_cfg.get("hard_gate_failure_score_scale", 0.25)
        )
        failure_scale = max(0.0, min(1.0, failure_scale))
        applied_gate_scale = failure_scale
        combined = max(
            disqualified_score,
            max(0.0, raw_combined) * failure_scale,
        )

    energy_components = [
        {
            "name": "fast",
            "source_metric": "fast_loss",
            "weight": w_fast,
            "residual": 1.0 - fast_score,
        },
        {
            "name": "plddt",
            "source_metric": "plddt_score",
            "weight": w_plddt,
            "residual": 1.0 - plddt_score,
        },
        {
            "name": "iptm",
            "source_metric": "iptm_score",
            "weight": w_iptm,
            "residual": 1.0 - iptm_score,
        },
        {
            "name": "ptm",
            "source_metric": "ptm_score",
            "weight": w_ptm,
            "residual": 1.0 - ptm_score,
        },
        {
            "name": "ranking_score",
            "source_metric": "ranking_score_component",
            "weight": w_ranking,
            "residual": 1.0 - ranking_score_component,
        },
        {
            "name": "interface_plddt",
            "source_metric": "interface_plddt_score",
            "weight": w_interface_plddt,
            "residual": 1.0 - interface_plddt_score,
        },
        {
            "name": "node_plddt_min",
            "source_metric": "node_plddt_min_score",
            "weight": w_node_min,
            "residual": 1.0 - node_plddt_min_score,
        },
        {
            "name": "multistate",
            "source_metric": "multistate_score",
            "weight": w_multistate,
            "residual": 1.0 - multistate_score,
        },
        {
            "name": "evaluator",
            "source_metric": "evaluator_report.total_energy",
            "weight": w_evaluator,
            "residual": evaluator_energy,
        },
        {
            "name": "clash",
            "source_metric": "clash_penalty",
            "weight": w_clash,
            "residual": clash_penalty,
        },
    ]
    active_weight_sum = sum(
        float(component["weight"])
        for component in energy_components
        if float(component["weight"]) > 0.0
    )
    raw_combined_energy = 0.0
    for component in energy_components:
        weight = float(component["weight"])
        residual = _clamp01(component["residual"])
        active = weight > 0.0
        weighted_residual = weight * residual if active else 0.0
        component["residual"] = float(residual)
        component["active"] = active
        component["weighted_residual"] = float(weighted_residual)
        component["normalized_contribution"] = (
            float(weighted_residual / active_weight_sum)
            if active_weight_sum > 0.0 and active
            else 0.0
        )
        raw_combined_energy += weighted_residual
    combined_energy = (
        _clamp01(raw_combined_energy / active_weight_sum)
        if active_weight_sum > 0.0
        else 1.0
    )
    return {
        "energy_schema_version": OUTER_ENERGY_SCHEMA_VERSION,
        "direction": "minimize",
        "raw_combined_energy": float(raw_combined_energy),
        "combined_energy": float(combined_energy),
        "final_energy": float(combined_energy),
        "active_energy_weight_sum": float(active_weight_sum),
        "energy_components": energy_components,
        "combined_score": float(combined),
        "raw_combined_score": float(raw_combined),
        "hard_gate_failure_score_scale": float(applied_gate_scale),
        "hard_gate_pass": 1.0 if hard_gate_pass else 0.0,
        "disqualified": 0.0 if hard_gate_pass else 1.0,
        "fast_score": float(fast_score),
        "plddt_score": float(plddt_score),
        "iptm_score": float(iptm_score),
        "ptm_score": float(ptm_score),
        "ranking_score_component": float(ranking_score_component),
        "interface_plddt_score": float(interface_plddt_score),
        "node_plddt_min_score": float(node_plddt_min_score),
        "multistate_score": float(multistate_score),
        "clash_penalty": float(clash_penalty),
        "evaluator_score": float(evaluator_score),
        "evaluator_soft_score": float(evaluator_soft_score),
        "evaluator_energy": float(evaluator_energy),
        "evaluator_loss": float(
            evaluator_report.get(
                "loss",
                out.get("evaluator_loss", 1.0),
            )
            or 1.0
        ),
        "structure_score": float(structure_score),
        "plddt": float(plddt or 0.0),
        "ptm": float(ptm or 0.0),
        "iptm": float(iptm or 0.0),
        "ranking_score": float(ranking_score or 0.0),
        "interface_plddt_mean": float(interface_plddt or 0.0),
        "interface_contact_count": float(
            _structure_value(out, "interface_contact_count", 0.0)
        ),
        "interface_residue_pair_count": float(
            _structure_value(out, "interface_residue_pair_count", 0.0)
        ),
        "clash_count": float(clash_count or 0.0),
        "node_plddt_mean": float(
            _structure_value(out, "node_plddt_mean", 0.0)
        ),
        "node_plddt_min": float(node_plddt_min or 0.0),
    }


__all__ = [
    "OUTER_ENERGY_SCHEMA_VERSION",
    "compute_outer_energy_objective",
]
