

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from astevolve.evaluation.mechanistic_transition_evaluator import evaluate_mechanistic_transition
from astevolve.evaluation.objectives.region_metrics import _ast_region_specs
from astevolve.evaluation.objectives.support import (
    _clamp01,
    _expand_entity_units,
    _state_names,
)


def _state_cif_path(state: Optional[Dict[str, Any]]) -> Optional[str]:
    if not state:
        return None
    summary = state.get("structure_metrics", {}) or {}
    return summary.get("cif_path") or state.get("cif_path")


def _canonical_source_chain(unit: Dict[str, Any]) -> Optional[str]:
    source_chain = str(unit.get("source_chain") or "").strip()
    asym_id = str(unit.get("asym_id") or "").strip()
    if not source_chain or not asym_id:
        return None
    try:
        copy_index = int(unit.get("copy_index") or 1)
    except (TypeError, ValueError):
        copy_index = 1
    return f"{source_chain}:{max(1, copy_index)}"


def _transition_chain_map(state: Optional[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not state:
        return out
    for unit in _expand_entity_units(state):
        if not isinstance(unit, dict):
            continue
        asym_id = str(unit.get("asym_id") or "").strip()
        canonical = _canonical_source_chain(unit)
        if asym_id and canonical:
            out[asym_id] = canonical
    return out


def _transition_structure_input(state: Dict[str, Any], path: str) -> Any:
    chain_map = _transition_chain_map(state)
    if not chain_map:
        return path
    return {
        "path": path,
        "chain_map": chain_map,
        "source": str(state.get("name") or path),
    }


def _score_mechanistic_transition(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:


    states = _state_names(spec)
    if len(states) < 2:
        states = [str(spec.get("state_a") or ""), str(spec.get("state_b") or "")]
    if len(states) < 2 or not states[0] or not states[1]:
        return 0.0, {}, ["mechanistic_transition needs states, or state_a/state_b"]
    apo_state = by_state.get(states[0])
    holo_state = by_state.get(states[1])
    apo_path = _state_cif_path(apo_state)
    holo_path = _state_cif_path(holo_state)
    if not apo_path or not holo_path:
        return 0.0, {"states": states, "apo_path": apo_path, "holo_path": holo_path}, [
            "mechanistic_transition needs CIF/PDB paths for both states"
        ]

    moving_value = spec.get("moving_regions", spec.get("moving_region", spec.get("region")))
    preserved_value = spec.get("preserved_regions", spec.get("preserved_region"))
    moving_regions = _ast_region_specs(moving_value, compiled, design_state)
    preserved_regions = _ast_region_specs(preserved_value, compiled, design_state)
    apo_structure = _transition_structure_input(apo_state, apo_path)
    holo_structure = _transition_structure_input(holo_state, holo_path)
    report = evaluate_mechanistic_transition(
        apo_structure,
        holo_structure,
        moving_regions=moving_regions,
        preserved_regions=preserved_regions,
        forbid=spec.get("forbid", ["chain_break", "severe_clash", "complete_unfolding"]),
        intermediate_states=spec.get("intermediate_states"),
        config=spec.get("config", spec),
    )
    details = {
        "states": states,
        "moving_region_count": len(moving_regions),
        "preserved_region_count": len(preserved_regions),
        "apo_chain_map": _transition_chain_map(apo_state),
        "holo_chain_map": _transition_chain_map(holo_state),
        "kinetic_path_score": report.get("kinetic_path_score", 0.0),
        "interpretability_report": report.get("interpretability_report", {}),
        "region_scores": report.get("region_scores", {}),
    }
    warnings = list((details["interpretability_report"] or {}).get("warnings", []) or [])
    return _clamp01(report.get("kinetic_path_score", 0.0)), details, warnings
