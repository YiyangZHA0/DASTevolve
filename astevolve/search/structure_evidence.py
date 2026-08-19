

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _safe_stat(values: List[float], kind: str) -> Optional[float]:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    if kind == "min":
        return float(arr.min())
    if kind == "max":
        return float(arr.max())
    return float(arr.mean())


def compute_node_plddt(
    compiled: Dict[str, Any],
    residue_plddt: Optional[Dict[str, List[float]]],
) -> Dict[str, Dict[str, Any]]:


    if not residue_plddt:
        return {}

    name_counts: Dict[str, int] = {}
    for seg in compiled.get("segments", []):
        name_counts[seg.name] = name_counts.get(seg.name, 0) + 1

    out: Dict[str, Dict[str, Any]] = {}
    for seg in compiled.get("segments", []):
        chain_vals = residue_plddt.get(seg.chain_id)
        if not chain_vals:
            continue
        vals: List[float] = []
        for idx in seg.indices():
            if 0 <= int(idx) < len(chain_vals):
                vals.append(float(chain_vals[int(idx)]))
        if not vals:
            continue
        key = seg.name if name_counts.get(seg.name, 0) == 1 else f"{seg.chain_id}:{seg.name}"
        out[key] = {
            "chain_id": seg.chain_id,
            "kind": seg.kind,
            "name": seg.name,
            "spans": seg.spans,
            "residue_count": len(vals),
            "plddt_mean": _safe_stat(vals, "mean"),
            "plddt_min": _safe_stat(vals, "min"),
            "plddt_max": _safe_stat(vals, "max"),
        }
    return out


def _mean_indexed_plddt(copies: List[List[float]]) -> List[float]:
    if not copies:
        return []
    max_len = max((len(copy) for copy in copies), default=0)
    out: List[float] = []
    for idx in range(max_len):
        vals = [float(copy[idx]) for copy in copies if idx < len(copy)]
        if vals:
            out.append(float(sum(vals) / len(vals)))
    return out


def _residue_plddt_by_source_chain(
    confidence: Dict[str, Any],
    entity_units: List[Dict[str, Any]],
) -> Dict[str, List[float]]:
    residue_plddt = confidence.get("residue_plddt", {}) or {}
    if not residue_plddt or not entity_units:
        return {}

    by_source: Dict[str, List[List[float]]] = {}
    for unit in entity_units:
        source_chain = unit.get("source_chain")
        label = unit.get("label")
        if not source_chain or not label:
            continue
        vals = residue_plddt.get(str(label))
        if not vals:
            continue
        cleaned: List[float] = []
        for value in vals:
            try:
                cleaned.append(float(value))
            except (TypeError, ValueError):
                continue
        if cleaned:
            by_source.setdefault(str(source_chain), []).append(cleaned)

    return {
        source_chain: _mean_indexed_plddt(copies)
        for source_chain, copies in by_source.items()
        if copies
    }


def _resolve_complex_entities(
    raw_entities: List[Dict[str, Any]],
    seqs: Dict[str, str],
) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    for raw in raw_entities:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        source_chain = item.pop("source_chain", None)
        if source_chain:
            seq = seqs.get(str(source_chain), "")
            if not seq:
                continue
            item["sequence"] = seq
        entities.append(item)
    return entities


def _complex_entities_to_multichain(
    raw_entities: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]], List[Dict[str, Any]]]:

    chains: List[Tuple[str, str]] = []
    entity_units: List[Dict[str, Any]] = []
    report_entities: List[Dict[str, Any]] = []
    unit_index = 0
    for entity_index, entity in enumerate(entities, start=1):
        if not isinstance(entity, dict):
            continue
        seq = str(entity.get("sequence") or "").strip()
        if not seq:
            continue
        kind = str(entity.get("type") or entity.get("kind") or "protein")
        if kind.lower() not in {"protein", "peptide", "polymer"}:
            continue
        raw = raw_entities[entity_index - 1] if entity_index - 1 < len(raw_entities) and isinstance(raw_entities[entity_index - 1], dict) else {}
        label = str(entity.get("id") or entity.get("name") or raw.get("id") or raw.get("name") or f"entity{entity_index}")
        source_chain = str(raw.get("source_chain") or entity.get("source_chain") or label)
        chains.append((label, seq))
        report_entities.append({"label": label, "kind": kind, "count": 1})
        entity_units.append(
            {
                "entity_index": entity_index,
                "label": label,
                "base_label": label,
                "kind": kind,
                "copy_index": 1,


                "asym_id": label,
                "asym_alias": _asym_id(unit_index),
                "source_chain": source_chain,
            }
        )
        if label != _asym_id(unit_index):
            alias_unit = dict(entity_units[-1])
            alias_unit["asym_id"] = _asym_id(unit_index)
            alias_unit["alias_for"] = label
            entity_units.append(alias_unit)
        unit_index += 1
    return chains, entity_units, report_entities


def _asym_id(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(alphabet):
        return alphabet[index]
    return f"X{index + 1}"


def _infer_complex_entity_units(
    raw_entities: List[Dict[str, Any]],
    report_entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    unit_index = 0
    for entity_index, report in enumerate(report_entities, start=1):
        if not isinstance(report, dict):
            continue
        raw = raw_entities[entity_index - 1] if entity_index - 1 < len(raw_entities) and isinstance(raw_entities[entity_index - 1], dict) else {}
        label = str(report.get("label") or raw.get("id") or raw.get("name") or f"entity{entity_index}")
        try:
            count = max(1, int(report.get("count", raw.get("count", 1))))
        except (TypeError, ValueError):
            count = 1
        for copy_index in range(1, count + 1):
            unit_label = label if count == 1 else f"{label}_{copy_index}"
            asym_id = _asym_id(unit_index)
            unit = {
                "entity_index": entity_index,
                "label": unit_label,
                "base_label": label,
                "kind": report.get("kind"),
                "copy_index": copy_index,
                "asym_id": asym_id,
                "source_chain": raw.get("source_chain"),
            }
            units.append(unit)
            if unit_label != asym_id:
                alias_unit = dict(unit)
                alias_unit["asym_id"] = unit_label
                alias_unit["alias_for"] = asym_id
                units.append(alias_unit)
            unit_index += 1
    return units


def _mean_numeric(values: List[Any]) -> Optional[float]:
    vals: List[float] = []
    for value in values:
        try:
            vals.append(float(value))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _aggregate_state_node_plddt(state_results: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:


    node_items: Dict[str, Dict[str, Any]] = {}
    means: List[float] = []
    mins: List[float] = []
    low_nodes: List[Dict[str, Any]] = []

    for state_result in state_results:
        state_name = str(state_result.get("name") or "state")
        summary = state_result.get("structure_metrics", {}) or {}
        for node_name, item in (summary.get("node_plddt", {}) or {}).items():
            if not isinstance(item, dict):
                continue
            key = f"{state_name}:{node_name}"
            record = dict(item)
            record["state"] = state_name
            node_items[key] = record

            mean_val = _float_or_none(record.get("plddt_mean"))
            min_val = _float_or_none(record.get("plddt_min"))
            if mean_val is not None:
                means.append(mean_val)
                if mean_val < 70.0:
                    low_nodes.append(
                        {
                            "state": state_name,
                            "node": str(node_name),
                            "plddt_mean": mean_val,
                            "plddt_min": min_val,
                        }
                    )
            if min_val is not None:
                mins.append(min_val)

    low_nodes = sorted(low_nodes, key=lambda x: float(x.get("plddt_mean", 0.0)))[:10]
    return node_items, {
        "node_count": len(node_items),
        "node_plddt_mean": _mean_numeric(means),
        "node_plddt_min": min(mins) if mins else None,
        "low_confidence_nodes": low_nodes,
    }


def _aggregate_complex_state_metrics(state_results: List[Dict[str, Any]]) -> Dict[str, Any]:


    scalars: Dict[str, List[float]] = {}
    interface_contact_count = 0.0
    interface_residue_pair_count = 0.0
    clash_count = 0.0
    interface_plddts: List[float] = []

    for result in state_results:
        summary = result.get("structure_metrics", {}) or {}
        for key, value in (summary.get("scalar", {}) or {}).items():
            try:
                scalars.setdefault(str(key), []).append(float(value))
            except (TypeError, ValueError):
                continue
        interface = summary.get("interface", {}) or {}
        interface_contact_count += float(interface.get("total_contact_count") or 0.0)
        interface_residue_pair_count += float(interface.get("total_residue_pair_count") or 0.0)
        clash_count += float(interface.get("clash_count") or 0.0)
        if interface.get("interface_plddt_mean") is not None:
            try:
                interface_plddts.append(float(interface["interface_plddt_mean"]))
            except (TypeError, ValueError):
                pass

    node_plddt, node_summary = _aggregate_state_node_plddt(state_results)
    scalar_mean = {
        key: float(sum(vals) / len(vals))
        for key, vals in scalars.items()
        if vals
    }
    interface_summary = {
        "available": bool(state_results),
        "total_contact_count": interface_contact_count,
        "total_residue_pair_count": interface_residue_pair_count,
        "clash_count": clash_count,
        "interface_plddt_mean": _mean_numeric(interface_plddts),
        "pairs": {},
    }
    first_structure_path = None
    first_summary_json = None
    first_out_dir = None
    for state in state_results:
        first_structure_path = first_structure_path or state.get("cif_path") or state.get("structure_path")
        first_summary_json = first_summary_json or state.get("summary_json")
        first_out_dir = first_out_dir or state.get("out_dir")

    return {
        "scalar": scalar_mean,
        "chain_plddt": {},
        "node_plddt": node_plddt,
        "node_summary": node_summary,
        "interface": interface_summary,
        "dockq": {
            "available": False,
            "dockq": None,
            "reason": "true DockQ requires a native/reference complex",
        },
        "states": state_results,
        "cif_path": first_structure_path,
        "structure_path": first_structure_path,
        "summary_json": first_summary_json,
        "out_dir": first_out_dir,
    }
