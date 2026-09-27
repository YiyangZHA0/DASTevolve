

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from .support import (
    as_str_list,
    clamp01,
    get_nested,
    normalize_name,
    safe_float,
    safe_int,
)


def segment_indices(segment: Any) -> List[int]:


    if hasattr(segment, "indices"):
        try:
            return [int(index) for index in segment.indices()]
        except Exception:
            pass
    indices: List[int] = []
    for span in getattr(segment, "spans", []) or []:
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            continue
        start = safe_int(span[0], 0)
        end = safe_int(span[1], start)
        indices.extend(range(start, max(start, end)))
    if not indices:
        start = safe_int(getattr(segment, "start", 0), 0)
        end = safe_int(getattr(segment, "end", start), start)
        indices.extend(range(start, max(start, end)))
    return sorted(set(indices))


def segments(compiled: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:


    if not compiled:
        return []
    result: List[Dict[str, Any]] = []
    for segment in compiled.get("segments", []) or []:
        name = normalize_name(getattr(segment, "name", ""))
        if not name:
            continue
        result.append(
            {
                "name": name,
                "kind": normalize_name(getattr(segment, "kind", "")),
                "chain_id": normalize_name(getattr(segment, "chain_id", "")),
                "indices": segment_indices(segment),
                "spans": getattr(segment, "spans", None),
            }
        )
    return result


def segments_by_name(compiled: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:


    return {segment["name"]: segment for segment in segments(compiled)}


def node_names_from_design_state(state: Mapping[str, Any], key: str) -> List[str]:


    design_points = state.get("design_points", {}) if isinstance(state.get("design_points"), Mapping) else {}
    constraints = state.get("design_constraints", {}) if isinstance(state.get("design_constraints"), Mapping) else {}
    policy = state.get("mutation_policy", {}) if isinstance(state.get("mutation_policy"), Mapping) else {}

    names: List[str] = []
    if key == "preserved":
        names.extend(as_str_list(design_points.get("preserved_nodes")))
        names.extend(as_str_list(constraints.get("frozen_nodes")))
        names.extend(as_str_list(policy.get("generally_frozen")))
        graph = state.get("semantic_graph", {}) if isinstance(state.get("semantic_graph"), Mapping) else {}
        functional = get_nested(graph, "functional_graph", "nodes") or {}
        if isinstance(functional, Mapping):
            for function_name, node in functional.items():
                if "stability" in str(function_name).lower() and isinstance(node, Mapping):
                    names.extend(as_str_list(node.get("maps_to")))
    elif key == "primary":
        names.extend(as_str_list(design_points.get("primary_design_nodes")))
        names.extend(as_str_list(design_points.get("default_open_nodes")))
        names.extend(as_str_list(policy.get("always_open_segments")))
    elif key == "secondary":
        names.extend(as_str_list(design_points.get("secondary_design_nodes")))
        names.extend(as_str_list(policy.get("conditionally_open_segments")))

    seen = set()
    result = []
    for name in names:
        if name and name not in seen:
            result.append(name)
            seen.add(name)
    return result


def node_plddt_map(
    out: Mapping[str, Any],
    structure: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:


    direct = out.get("node_plddt")
    if isinstance(direct, Mapping) and direct:
        return {str(key): dict(value) for key, value in direct.items() if isinstance(value, Mapping)}
    nested = structure.get("node_plddt")
    if isinstance(nested, Mapping) and nested:
        return {str(key): dict(value) for key, value in nested.items() if isinstance(value, Mapping)}
    return {}


def node_value(item: Mapping[str, Any], metric: str) -> Optional[float]:


    candidates = {
        "plddt_mean": ("plddt_mean", "mean", "avg"),
        "plddt_min": ("plddt_min", "min"),
        "plddt_max": ("plddt_max", "max"),
    }.get(metric, (metric,))
    for key in candidates:
        value = safe_float(item.get(key))
        if value is not None:
            return value
    return None


def sequence_changes(
    sequences: Mapping[str, str],
    template_sequences: Optional[Mapping[str, str]],
) -> Dict[str, List[int]]:


    template_sequences = template_sequences or {}
    changes: Dict[str, List[int]] = {}
    for chain_id, sequence in sequences.items():
        template = template_sequences.get(chain_id)
        if not template:
            continue
        chain_changes = [
            int(index)
            for index, (new, old) in enumerate(zip(str(sequence), str(template)))
            if new != old
        ]
        if len(sequence) != len(template):
            chain_changes.extend(
                range(min(len(sequence), len(template)), max(len(sequence), len(template)))
            )
        changes[str(chain_id)] = sorted(set(chain_changes))
    return changes


def fixed_lookup(
    fixed_residues: Optional[Mapping[str, Mapping[Any, str]]],
) -> Dict[str, Dict[int, str]]:


    result: Dict[str, Dict[int, str]] = {}
    for chain_id, items in (fixed_residues or {}).items():
        if not isinstance(items, Mapping):
            continue
        result[str(chain_id)] = {}
        for index, amino_acid in items.items():
            try:
                result[str(chain_id)][int(index)] = str(amino_acid)
            except (TypeError, ValueError):
                continue
    return result


def mask_allows(masks: Optional[Mapping[str, Any]], chain_id: str, index: int) -> bool:


    if not masks or chain_id not in masks:
        return True
    mask = masks.get(chain_id)
    try:
        if index < 0 or index >= len(mask):
            return False
        return bool(mask[index])
    except Exception:

        return True


def all_interface_pairs(
    structure: Mapping[str, Any],
) -> List[Tuple[str, str, Mapping[str, Any]]]:


    result: List[Tuple[str, str, Mapping[str, Any]]] = []
    states = structure.get("states") if isinstance(structure.get("states"), list) else []
    if states:
        for state in states:
            if not isinstance(state, Mapping):
                continue
            state_name = str(state.get("name") or "state")
            asymmetry_metadata: Dict[str, Dict[str, Any]] = {}
            for unit in state.get("entity_units", []) or []:
                if not isinstance(unit, Mapping):
                    continue
                asymmetry_id = str(unit.get("asym_id") or unit.get("chain") or "")
                if not asymmetry_id:
                    continue
                asymmetry_metadata[asymmetry_id] = {
                    "asym": asymmetry_id,
                    "source_chain": str(unit.get("source_chain") or ""),
                    "entity": str(unit.get("base_label") or unit.get("label") or ""),
                    "entity_label": str(unit.get("label") or ""),
                }
            summary = state.get("structure_metrics", {}) if isinstance(state.get("structure_metrics"), Mapping) else {}
            interface = summary.get("interface", {}) if isinstance(summary.get("interface"), Mapping) else {}
            pairs = interface.get("pairs", {}) if isinstance(interface.get("pairs"), Mapping) else {}
            for pair_name, pair in pairs.items():
                if not isinstance(pair, Mapping):
                    continue
                annotated_pair = dict(pair)
                residue_pairs = []
                for residue_pair in pair.get("residue_pairs", []) or []:
                    if not isinstance(residue_pair, Mapping):
                        continue
                    annotated_residue_pair = dict(residue_pair)
                    for side in ("left", "right"):
                        endpoint = residue_pair.get(side)
                        if not isinstance(endpoint, Mapping):
                            continue
                        annotated_endpoint = dict(endpoint)
                        chain = str(
                            endpoint.get("chain")
                            or endpoint.get("asym")
                            or endpoint.get("chain_id")
                            or ""
                        )
                        if chain in asymmetry_metadata:
                            for key, value in asymmetry_metadata[chain].items():
                                if value and key not in annotated_endpoint:
                                    annotated_endpoint[key] = value
                        annotated_residue_pair[side] = annotated_endpoint
                    residue_pairs.append(annotated_residue_pair)
                annotated_pair["residue_pairs"] = residue_pairs
                result.append((state_name, str(pair_name), annotated_pair))

    interface = structure.get("interface", {}) if isinstance(structure.get("interface"), Mapping) else {}
    pairs = interface.get("pairs", {}) if isinstance(interface.get("pairs"), Mapping) else {}
    for pair_name, pair in pairs.items():
        if isinstance(pair, Mapping):
            result.append(("aggregate", str(pair_name), pair))
    return result


def objective_items(
    out: Mapping[str, Any],
    structure: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:


    pack = out.get("multistate_objectives") or structure.get("multistate_objectives") or {}
    if not isinstance(pack, Mapping):
        return {}
    objectives = pack.get("objectives", {}) if isinstance(pack.get("objectives"), Mapping) else {}
    return {str(key): dict(value) for key, value in objectives.items() if isinstance(value, Mapping)}


def multistate_score(out: Mapping[str, Any], structure: Mapping[str, Any]) -> Optional[float]:


    explicitly_disabled = False
    for source in (out.get("multistate_objectives"), structure.get("multistate_objectives")):
        if isinstance(source, Mapping):
            if source.get("enabled") is False:
                explicitly_disabled = True
                continue
            value = safe_float(source.get("normalized_score"))
            if value is not None:
                return clamp01(value)
    if explicitly_disabled:


        return None
    return safe_float(out.get("multistate_score"))
