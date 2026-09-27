

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from astevolve.evaluation.objectives.support import (
    _clamp01,
    _expand_entity_units,
    _mean,
    _normalize_metric,
    _safe_float,
    _state_names,
)
from astevolve.metrics.structure import _load_atoms, metric_value


def _score_confidence(by_state: Dict[str, Dict[str, Any]], spec: Dict[str, Any]) -> Tuple[float, Dict[str, Any], List[str]]:
    names = _state_names(spec) or list(by_state)
    metric = str(spec.get("metric") or "plddt")
    values: Dict[str, float] = {}
    warnings: List[str] = []
    for name in names:
        state = by_state.get(name)
        if not state:
            warnings.append(f"unknown state: {name}")
            continue
        summary = state.get("structure_metrics", {}) or {}
        raw = metric_value(summary, metric, default=0.0)
        values[name] = _normalize_metric(metric, raw)
    if not values:
        return 0.0, {"metric": metric, "states": names, "values": values}, warnings
    return float(sum(values.values()) / len(values)), {"metric": metric, "states": names, "values": values}, warnings


def _node_metric_value(item: Dict[str, Any], metric: str) -> Optional[float]:
    metric = str(metric or "plddt_mean")
    aliases = {
        "plddt": "plddt_mean",
        "mean": "plddt_mean",
        "min": "plddt_min",
        "max": "plddt_max",
    }
    key = aliases.get(metric, metric)
    return _safe_float(item.get(key))


def _score_region_confidence(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:


    names = _state_names(spec) or list(by_state)
    metric = str(spec.get("metric") or "plddt_mean")
    region_value = spec.get("region", spec.get("regions", spec.get("nodes", spec.get("node"))))
    nodes = _expand_region_names(region_value, design_state)
    if not nodes:
        return 0.0, {"metric": metric, "states": names, "nodes": []}, ["region_confidence needs region, regions, node, or nodes"]

    target_raw = _safe_float(spec.get("target"), None)
    target = _normalize_metric(metric, target_raw) if target_raw is not None else None
    values: Dict[str, Dict[str, Dict[str, float]]] = {}
    scores: List[float] = []
    warnings: List[str] = []

    for name in names:
        state = by_state.get(name)
        if not state:
            warnings.append(f"unknown state: {name}")
            continue
        summary = state.get("structure_metrics", {}) or {}
        node_plddt = summary.get("node_plddt", {}) or state.get("node_plddt", {}) or {}
        state_values: Dict[str, Dict[str, float]] = {}
        for node in nodes:
            item = node_plddt.get(node)
            if not isinstance(item, dict):
                continue
            raw = _node_metric_value(item, metric)
            if raw is None:
                continue
            normalized = _normalize_metric(metric, raw)
            score = _clamp01(normalized / target) if target and target > 0 else normalized
            state_values[str(node)] = {"raw": float(raw), "normalized": float(normalized), "score": float(score)}
            scores.append(float(score))
        if not state_values:
            warnings.append(f"{name}: no node confidence values for region/nodes {nodes}")
        values[name] = state_values

    if not scores:
        return 0.0, {"metric": metric, "states": names, "nodes": nodes, "values": values}, warnings
    return float(sum(scores) / len(scores)), {
        "metric": metric,
        "states": names,
        "nodes": nodes,
        "target": target_raw,
        "values": values,
    }, warnings


def _score_region_confidence_floor(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    names = _state_names(spec) or list(by_state)
    metric = str(spec.get("metric") or "plddt_min")
    region_value = spec.get("region", spec.get("regions", spec.get("nodes", spec.get("node"))))
    nodes = _expand_region_names(region_value, design_state)
    if not nodes:
        return 0.0, {"metric": metric, "states": names, "nodes": []}, [
            "region_confidence_floor needs region, regions, node, or nodes"
        ]

    target_raw = _safe_float(spec.get("target"), 70.0) or 70.0
    hard_floor_raw = _safe_float(spec.get("hard_floor"), None)
    if hard_floor_raw is None:
        hard_floor_raw = min(target_raw, _safe_float(spec.get("floor"), 50.0) or 50.0)
    target = _normalize_metric(metric, target_raw)
    hard_floor = _normalize_metric(metric, hard_floor_raw)

    values: Dict[str, Dict[str, Dict[str, float]]] = {}
    scores: List[float] = []
    failing: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for name in names:
        state = by_state.get(name)
        if not state:
            warnings.append(f"unknown state: {name}")
            continue
        summary = state.get("structure_metrics", {}) or {}
        node_plddt = summary.get("node_plddt", {}) or state.get("node_plddt", {}) or {}
        state_values: Dict[str, Dict[str, float]] = {}
        for node in nodes:
            item = node_plddt.get(node)
            if not isinstance(item, dict):
                continue
            raw = _node_metric_value(item, metric)
            if raw is None:
                continue
            normalized = _normalize_metric(metric, raw)
            if normalized < hard_floor:
                score = 0.25 * _clamp01(normalized / max(1e-6, hard_floor))
                failing.append({"state": name, "node": node, "raw": float(raw), "hard_floor": hard_floor_raw})
            else:
                score = _clamp01(normalized / max(1e-6, target))
            state_values[str(node)] = {"raw": float(raw), "normalized": float(normalized), "score": float(score)}
            scores.append(float(score))
        if not state_values:
            warnings.append(f"{name}: no node confidence values for region/nodes {nodes}")
        values[name] = state_values

    if not scores:
        return 0.0, {
            "metric": metric,
            "states": names,
            "nodes": nodes,
            "target": target_raw,
            "hard_floor": hard_floor_raw,
            "values": values,
        }, warnings

    aggregate = str(spec.get("aggregate", "min")).lower()
    score = min(scores) if aggregate in {"min", "floor", "strict"} else float(sum(scores) / len(scores))
    return _clamp01(score), {
        "metric": metric,
        "states": names,
        "nodes": nodes,
        "target": target_raw,
        "hard_floor": hard_floor_raw,
        "aggregate": aggregate,
        "values": values,
        "failing_nodes": failing,
    }, warnings


def _region_names(spec: Dict[str, Any], design_state: Dict[str, Any]) -> Optional[List[str]]:
    region = spec.get("region", spec.get("regions"))
    if region is None:
        return None
    aliases = design_state.get("multistate_regions", {}) if isinstance(design_state, dict) else {}
    if isinstance(region, str):
        value = aliases.get(region, region)
    else:
        value = region
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return None


def _region_indices_by_chain(compiled: Optional[Dict[str, Any]], names: Optional[List[str]]) -> Dict[str, Optional[set[int]]]:
    if not compiled or not names:
        return {}
    wanted = {str(name) for name in names}
    out: Dict[str, set[int]] = {}
    for seg in compiled.get("segments", []) or []:
        if getattr(seg, "name", None) not in wanted:
            continue
        chain_id = str(getattr(seg, "chain_id", ""))
        out.setdefault(chain_id, set()).update(int(i) for i in seg.indices())
    return out


def _expand_region_names(value: Any, design_state: Dict[str, Any]) -> List[str]:
    aliases = design_state.get("multistate_regions", {}) if isinstance(design_state, dict) else {}
    if value is None:
        return []
    if isinstance(value, str):
        expanded = aliases.get(value, value)
        if isinstance(expanded, list):
            return [str(item) for item in expanded]
        return [str(expanded)]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                out.extend(_expand_region_names(item, design_state))
        return out
    return []


def _ast_region_specs(
    value: Any,
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> List[Dict[str, Any]]:


    if value is None:
        return []
    if isinstance(value, dict):
        if any(key in value for key in ("indices", "residues", "spans", "ranges", "residue_ranges")):
            return [dict(value)]
        if value.get("region") or value.get("regions"):
            return _ast_region_specs(value.get("region", value.get("regions")), compiled, design_state)
    if isinstance(value, list) and any(isinstance(item, dict) for item in value):
        out: List[Dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                out.extend(_ast_region_specs(item, compiled, design_state))
            elif isinstance(item, str):
                out.extend(_ast_region_specs(item, compiled, design_state))
        return out

    names = _expand_region_names(value, design_state)
    if not names:
        return []
    if not compiled:
        return [{"name": name} for name in names]
    wanted = set(names)
    out: List[Dict[str, Any]] = []
    for seg in compiled.get("segments", []) or []:
        if getattr(seg, "name", None) not in wanted:
            continue
        out.append(
            {
                "name": getattr(seg, "name", "region"),
                "chain_id": str(getattr(seg, "chain_id", "A")),
                "indices": [int(i) for i in seg.indices()],
                "zero_based": True,
            }
        )
    if isinstance(design_state, dict):
        target = design_state.get("target", {}) or {}
        target_name = str(target.get("epitope_name") or "")
        existing_names = {str(spec.get("name") or "") for spec in out if isinstance(spec, dict)}
        if target_name and target_name in wanted and target_name not in existing_names:
            indices: List[int] = []
            for span in target.get("epitope_spans", []) or []:
                if not isinstance(span, (list, tuple)) or len(span) < 2:
                    continue
                try:
                    start = int(span[0])
                    end = int(span[1])
                except (TypeError, ValueError):
                    continue
                indices.extend(range(max(0, start), max(0, end)))
            out.append(
                {
                    "name": target_name,
                    "chain_id": str(target.get("chain_id", "T")),
                    "indices": sorted(set(indices)),
                    "zero_based": True,
                }
            )
    if isinstance(design_state, dict):
        constraints = design_state.get("design_constraints", {})
        region_tables: List[Any] = [design_state.get("entity_regions")]
        if isinstance(constraints, dict):
            region_tables.extend([constraints.get("target_epitopes"), constraints.get("decoy_epitopes")])
        named: Dict[str, Dict[str, Any]] = {}
        for table in region_tables:
            if isinstance(table, dict):
                for name, spec in table.items():
                    if isinstance(spec, dict):
                        named[str(name)] = dict(spec)
            elif isinstance(table, list):
                for item in table:
                    if isinstance(item, dict) and item.get("name"):
                        named[str(item["name"])] = dict(item)
        existing_names = {str(spec.get("name") or "") for spec in out if isinstance(spec, dict)}
        for name in names:
            if str(name) in existing_names:
                continue
            spec = named.get(str(name))
            if not isinstance(spec, dict):
                continue
            resolved = dict(spec)
            resolved.setdefault("name", str(name))
            if "entity" not in resolved and resolved.get("id"):
                resolved["entity"] = resolved.get("id")
            if "chain_id" not in resolved and resolved.get("source_chain"):
                resolved["chain_id"] = resolved.get("source_chain")
            resolved.setdefault("zero_based", True)
            out.append(resolved)
    return out


def _unit_key(unit: Dict[str, Any]) -> Tuple[str, int]:
    return (str(unit.get("source_chain") or unit.get("base_label") or unit.get("kind") or ""), int(unit.get("copy_index") or 1))


def _protein_ca_by_seq(cif_path: Optional[str], asym_id: str) -> Dict[int, np.ndarray]:
    if not cif_path:
        return {}
    coords: Dict[int, np.ndarray] = {}
    for atom in _load_atoms(str(cif_path)):
        if str(atom.get("asym") or "") != str(asym_id):
            continue
        atom_name = str(atom.get("atom") or "").strip("'\"")
        if atom_name != "CA":
            continue
        coords[int(atom.get("seq_id") or 1) - 1] = np.asarray(atom.get("xyz"), dtype=float)
    return coords


def _kabsch_rmsd(ref: np.ndarray, mob: np.ndarray, eval_ref: np.ndarray, eval_mob: np.ndarray) -> Optional[float]:
    if len(ref) < 3 or len(mob) < 3 or len(eval_ref) == 0 or len(eval_mob) == 0:
        return None
    ref_center = ref.mean(axis=0)
    mob_center = mob.mean(axis=0)
    ref0 = ref - ref_center
    mob0 = mob - mob_center
    cov = mob0.T @ ref0
    try:
        u, _, vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None
    det = np.linalg.det(u @ vt)
    corr = np.eye(3)
    corr[2, 2] = 1.0 if det >= 0 else -1.0
    rot = u @ corr @ vt
    moved = (eval_mob - mob_center) @ rot + ref_center
    diff = moved - eval_ref
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _region_rmsd(
    state_a: Dict[str, Any],
    state_b: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
    spec: Dict[str, Any],
) -> Optional[float]:


    region_by_chain = _region_indices_by_chain(compiled, _region_names(spec, design_state))
    units_a = {_unit_key(unit): unit for unit in _expand_entity_units(state_a) if unit.get("source_chain")}
    units_b = {_unit_key(unit): unit for unit in _expand_entity_units(state_b) if unit.get("source_chain")}
    values: List[float] = []
    for key in sorted(set(units_a).intersection(units_b)):
        source_chain = key[0]
        asym_a = units_a[key].get("asym_id")
        asym_b = units_b[key].get("asym_id")
        if not asym_a or not asym_b:
            continue
        coords_a = _protein_ca_by_seq((state_a.get("structure_metrics") or {}).get("cif_path"), str(asym_a))
        coords_b = _protein_ca_by_seq((state_b.get("structure_metrics") or {}).get("cif_path"), str(asym_b))
        common = sorted(set(coords_a).intersection(coords_b))
        if len(common) < 3:
            continue
        region_indices = region_by_chain.get(source_chain)
        eval_indices = [idx for idx in common if region_indices is None or idx in region_indices]
        if not eval_indices:
            continue
        ref = np.asarray([coords_a[idx] for idx in common], dtype=float)
        mob = np.asarray([coords_b[idx] for idx in common], dtype=float)
        eval_ref = np.asarray([coords_a[idx] for idx in eval_indices], dtype=float)
        eval_mob = np.asarray([coords_b[idx] for idx in eval_indices], dtype=float)
        rmsd = _kabsch_rmsd(ref, mob, eval_ref, eval_mob)
        if rmsd is not None:
            values.append(float(rmsd))
    return _mean(values)


def _score_conf_change(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    states = _state_names(spec)
    if len(states) < 2:
        states = [str(spec.get("state_a") or ""), str(spec.get("state_b") or "")]
    if len(states) < 2 or not states[0] or not states[1]:
        return 0.0, {}, ["conf_change needs states, or state_a/state_b"]
    state_a = by_state.get(states[0])
    state_b = by_state.get(states[1])
    if not state_a or not state_b:
        return 0.0, {"states": states}, [f"unknown conf_change states: {states}"]

    rmsd = _region_rmsd(state_a, state_b, compiled, design_state, spec)
    if rmsd is not None:
        min_rmsd = float(spec.get("min_rmsd", 0.5))
        target_rmsd = max(min_rmsd + 1e-6, float(spec.get("target_rmsd", 3.0)))
        score = _clamp01((float(rmsd) - min_rmsd) / (target_rmsd - min_rmsd))
        return score, {"states": states, "region_rmsd": float(rmsd), "min_rmsd": min_rmsd, "target_rmsd": target_rmsd}, []

    interface_a = (state_a.get("structure_metrics", {}) or {}).get("interface", {}) or {}
    interface_b = (state_b.get("structure_metrics", {}) or {}).get("interface", {}) or {}
    delta = abs(float(interface_a.get("total_contact_count") or 0) - float(interface_b.get("total_contact_count") or 0))
    score = _clamp01(math.log1p(delta) / math.log1p(float(spec.get("contact_delta_target", 50.0))))
    return score, {"states": states, "contact_count_delta": delta, "fallback": "interface_contact_delta"}, [
        "conf_change used contact-delta fallback because region RMSD was unavailable"
    ]
