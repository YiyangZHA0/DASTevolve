

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from astevolve.evaluation.objectives.region_metrics import _ast_region_specs
from astevolve.evaluation.objectives.support import (
    _clamp01,
    _coverage,
    _mean,
    _normalize_metric,
    _one_letter_residue,
    _pair_key,
    _region_filters_by_asym,
    _residue_allowed,
    _safe_float,
    _selector_asym_ids,
)


def _filtered_pair_item(
    item: Dict[str, Any],
    left_ids: List[str],
    right_ids: List[str],
    left_filters: Dict[str, Optional[set[int]]],
    right_filters: Dict[str, Optional[set[int]]],
) -> Optional[Dict[str, Any]]:
    residue_pairs = item.get("residue_pairs")
    if not residue_pairs:
        if left_filters or right_filters:
            return None
        return dict(item)

    contact_count = 0
    clash_count = 0
    residue_pair_count = 0
    plddt_values: List[float] = []
    left_contacted: set[Tuple[str, int]] = set()
    right_contacted: set[Tuple[str, int]] = set()
    examples: List[Dict[str, Any]] = []

    for pair in residue_pairs:
        if not isinstance(pair, dict):
            continue
        raw_left = pair.get("left", {}) or {}
        raw_right = pair.get("right", {}) or {}
        left_chain = str(raw_left.get("chain") or "")
        right_chain = str(raw_right.get("chain") or "")
        if left_chain in left_ids and right_chain in right_ids:
            oriented_left, oriented_right = raw_left, raw_right
        elif left_chain in right_ids and right_chain in left_ids:
            oriented_left, oriented_right = raw_right, raw_left
        else:
            continue
        if not _residue_allowed(oriented_left, left_filters):
            continue
        if not _residue_allowed(oriented_right, right_filters):
            continue

        contact_count += int(pair.get("contact_count") or 0)
        clash_count += int(pair.get("clash_count") or 0)
        residue_pair_count += 1
        for residue, contacted in ((oriented_left, left_contacted), (oriented_right, right_contacted)):
            plddt = _safe_float(residue.get("plddt"))
            if plddt is not None:
                plddt_values.append(float(plddt))
            try:
                contacted.add((str(residue.get("chain") or ""), int(residue.get("residue") or 0) - 1))
            except (TypeError, ValueError):
                pass
        if len(examples) < 10:
            examples.append(
                {
                    "left": oriented_left,
                    "right": oriented_right,
                    "contact_count": pair.get("contact_count"),
                    "clash_count": pair.get("clash_count"),
                    "min_distance": pair.get("min_distance"),
                }
            )

    if contact_count <= 0 and clash_count <= 0:
        return None
    return {
        "contact_count": int(contact_count),
        "residue_pair_count": int(residue_pair_count),
        "clash_count": int(clash_count),
        "interface_plddt_mean": _mean(plddt_values),
        "interface_plddt_min": min(plddt_values) if plddt_values else None,
        "left_region_coverage": _coverage(left_filters, left_contacted),
        "right_region_coverage": _coverage(right_filters, right_contacted),
        "contact_examples": examples,
    }


def _sum_pair_metrics(
    state_result: Dict[str, Any],
    left_selector: Any,
    right_selector: Any,
    left_region_specs: Optional[List[Dict[str, Any]]] = None,
    right_region_specs: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[str]]:


    summary = state_result.get("structure_metrics", {}) or {}
    interface = summary.get("interface", {}) or {}
    pairs = interface.get("pairs", {}) or {}
    warnings: List[str] = []

    if left_selector is None and right_selector is None:
        return {
            "available": bool(interface.get("available", False)),
            "contact_count": int(interface.get("total_contact_count") or 0),
            "residue_pair_count": int(interface.get("total_residue_pair_count") or 0),
            "clash_count": int(interface.get("clash_count") or 0),
            "interface_plddt_mean": interface.get("interface_plddt_mean"),
        }, warnings

    left_ids = _selector_asym_ids(state_result, left_selector)
    right_ids = _selector_asym_ids(state_result, right_selector)
    if not left_ids or not right_ids:
        warnings.append(f"could not resolve interface selectors: {left_selector}, {right_selector}")
        return {"available": False}, warnings

    selected: List[Dict[str, Any]] = []
    for left in left_ids:
        for right in right_ids:
            if left == right:
                continue
            key = _pair_key(left, right, pairs)
            if key is not None and isinstance(pairs.get(key), dict):
                selected.append(pairs[key])

    if not selected:
        warnings.append(f"no interface pair metrics for selectors: {left_selector}, {right_selector}")
        return {"available": False}, warnings

    left_filters = _region_filters_by_asym(state_result, left_region_specs)
    right_filters = _region_filters_by_asym(state_result, right_region_specs)
    if left_region_specs and not left_filters:
        warnings.append(f"could not resolve left region for selector: {left_selector}")
    if right_region_specs and not right_filters:
        warnings.append(f"could not resolve right region for selector: {right_selector}")

    if left_filters or right_filters:
        filtered: List[Dict[str, Any]] = []
        for item in selected:
            filtered_item = _filtered_pair_item(item, left_ids, right_ids, left_filters, right_filters)
            if filtered_item is not None:
                filtered.append(filtered_item)
        selected = filtered
        if not selected:
            warnings.append("no contacts remained after interface region filters")
            return {
                "available": False,
                "reason": "no_region_filtered_contacts",
                "left_region_required": bool(left_region_specs),
                "right_region_required": bool(right_region_specs),
                "left_region_resolved": (not left_region_specs) or bool(left_filters),
                "right_region_resolved": (not right_region_specs) or bool(right_filters),
            }, warnings

    plddt_values = [
        float(item["interface_plddt_mean"])
        for item in selected
        if _safe_float(item.get("interface_plddt_mean")) is not None
    ]
    left_coverages = [
        float(item["left_region_coverage"])
        for item in selected
        if _safe_float(item.get("left_region_coverage")) is not None
    ]
    right_coverages = [
        float(item["right_region_coverage"])
        for item in selected
        if _safe_float(item.get("right_region_coverage")) is not None
    ]
    return {
        "available": True,
        "contact_count": int(sum(int(item.get("contact_count") or 0) for item in selected)),
        "residue_pair_count": int(sum(int(item.get("residue_pair_count") or 0) for item in selected)),
        "clash_count": int(sum(int(item.get("clash_count") or 0) for item in selected)),
        "interface_plddt_mean": _mean(plddt_values),
        "left_region_coverage": _mean(left_coverages),
        "right_region_coverage": _mean(right_coverages),
        "region_filtered": bool(left_filters or right_filters),
    }, warnings


def _interface_selectors(spec: Dict[str, Any]) -> Tuple[Any, Any]:
    pair = spec.get("pair")
    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
        return pair[0], pair[1]
    return (
        spec.get("left", spec.get("protein", spec.get("binder"))),
        spec.get("right", spec.get("target", spec.get("ligand"))),
    )


def _interface_region_specs(
    spec: Dict[str, Any],
    side: str,
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    regions = spec.get("regions")
    value = None
    if isinstance(regions, dict):
        value = regions.get(side)
    if value is None:
        if side == "left":
            for key in ("left_region", "binder_region", "protein_region"):
                if spec.get(key) is not None:
                    value = spec.get(key)
                    break
        else:
            for key in ("right_region", "target_region", "ligand_region"):
                if spec.get(key) is not None:
                    value = spec.get(key)
                    break
    return _ast_region_specs(value, compiled, design_state)


def _interface_strength(
    state_result: Dict[str, Any],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]] = None,
    design_state: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any], List[str]]:


    left, right = _interface_selectors(spec)
    design_state = design_state or {}
    left_region_specs = _interface_region_specs(spec, "left", compiled, design_state)
    right_region_specs = _interface_region_specs(spec, "right", compiled, design_state)
    metrics, warnings = _sum_pair_metrics(
        state_result,
        left,
        right,
        left_region_specs=left_region_specs,
        right_region_specs=right_region_specs,
    )
    full_metrics, full_warnings = _sum_pair_metrics(state_result, left, right)
    full_warnings = [warning for warning in full_warnings if warning not in warnings]
    warnings.extend(full_warnings)
    if not metrics.get("available"):
        details = dict(metrics)
        details["full_interface"] = full_metrics
        return 0.0, details, warnings


    default_target = 30.0
    contact_target = max(1.0, float(spec.get("contact_target", default_target)))
    residue_target = max(1.0, float(spec.get("residue_pair_target", max(3.0, contact_target / 5.0))))
    clash_target = max(1.0, float(spec.get("clash_tolerance", 3.0)))

    contact_score = _clamp01(math.log1p(float(metrics.get("contact_count") or 0)) / math.log1p(contact_target))
    residue_score = _clamp01(math.log1p(float(metrics.get("residue_pair_count") or 0)) / math.log1p(residue_target))
    plddt_score = _normalize_metric("plddt", metrics.get("interface_plddt_mean"))
    clash_score = 1.0 - _clamp01(float(metrics.get("clash_count") or 0) / clash_target)
    coverage_values = [
        float(value)
        for value in (metrics.get("left_region_coverage"), metrics.get("right_region_coverage"))
        if _safe_float(value) is not None
    ]
    coverage = _mean(coverage_values)
    coverage_target = _safe_float(spec.get("coverage_target"))
    coverage_score = None
    if coverage is not None and coverage_target is not None:
        coverage_score = _clamp01(float(coverage) / max(1e-6, float(coverage_target)))

    if coverage_score is None:
        score = (0.35 * contact_score) + (0.20 * residue_score) + (0.30 * plddt_score) + (0.15 * clash_score)
    else:
        score = (
            (0.25 * contact_score)
            + (0.15 * residue_score)
            + (0.25 * plddt_score)
            + (0.20 * coverage_score)
            + (0.15 * clash_score)
        )

    off_target_contact_count = max(0.0, float(full_metrics.get("contact_count") or 0) - float(metrics.get("contact_count") or 0))
    off_target_residue_pair_count = max(
        0.0,
        float(full_metrics.get("residue_pair_count") or 0) - float(metrics.get("residue_pair_count") or 0),
    )
    off_target_penalty_weight = float(spec.get("off_target_penalty_weight", 0.0) or 0.0)
    off_target_contact_tolerance = max(1.0, float(spec.get("off_target_contact_tolerance", contact_target) or contact_target))
    off_target_score = _clamp01(math.log1p(off_target_contact_count) / math.log1p(off_target_contact_tolerance))
    if off_target_penalty_weight > 0.0:
        score -= off_target_penalty_weight * off_target_score

    details = dict(metrics)
    details.update(
        {
            "contact_score": contact_score,
            "residue_pair_score": residue_score,
            "interface_plddt_score": plddt_score,
            "coverage": coverage,
            "coverage_target": coverage_target,
            "coverage_score": coverage_score,
            "clash_score": clash_score,
            "full_contact_count": full_metrics.get("contact_count"),
            "full_residue_pair_count": full_metrics.get("residue_pair_count"),
            "off_target_contact_count": off_target_contact_count,
            "off_target_residue_pair_count": off_target_residue_pair_count,
            "off_target_score": off_target_score,
            "off_target_penalty_weight": off_target_penalty_weight,
        }
    )
    return _clamp01(score), details, warnings


def _interface_left_residue_names(
    state_result: Dict[str, Any],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any], List[str]]:
    left, right = _interface_selectors(spec)
    left_region_specs = _interface_region_specs(spec, "left", compiled, design_state)
    right_region_specs = _interface_region_specs(spec, "right", compiled, design_state)
    metrics, warnings = _sum_pair_metrics(
        state_result,
        left,
        right,
        left_region_specs=left_region_specs,
        right_region_specs=right_region_specs,
    )
    if not metrics.get("available"):
        return [], metrics, warnings

    pairs = ((state_result.get("structure_metrics", {}) or {}).get("interface", {}) or {}).get("pairs", {}) or {}
    left_ids = _selector_asym_ids(state_result, left)
    right_ids = _selector_asym_ids(state_result, right)
    left_filters = _region_filters_by_asym(state_result, left_region_specs)
    right_filters = _region_filters_by_asym(state_result, right_region_specs)
    residues: List[str] = []
    for left_id in left_ids:
        for right_id in right_ids:
            key = _pair_key(left_id, right_id, pairs)
            if key is None or not isinstance(pairs.get(key), dict):
                continue
            for pair in pairs[key].get("residue_pairs", []) or []:
                raw_left = pair.get("left", {}) or {}
                raw_right = pair.get("right", {}) or {}
                left_chain = str(raw_left.get("chain") or "")
                right_chain = str(raw_right.get("chain") or "")
                if left_chain in left_ids and right_chain in right_ids:
                    oriented_left, oriented_right = raw_left, raw_right
                elif left_chain in right_ids and right_chain in left_ids:
                    oriented_left, oriented_right = raw_right, raw_left
                else:
                    continue
                if not _residue_allowed(oriented_left, left_filters):
                    continue
                if not _residue_allowed(oriented_right, right_filters):
                    continue
                resname = _one_letter_residue(oriented_left.get("resname"))
                if resname:
                    residues.append(resname)
    return residues, metrics, warnings
