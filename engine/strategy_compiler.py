

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, List, Optional, Tuple

from astevolve.search.operator_registry import (
    executable_operator_names,
    validate_operator_weights,
)

from .external_knowledge_policy import EXTERNAL_KB_FIELDS

from .design_state import (
    binder_domain_order,
    flatten_binder_parts,
    state_domain_aliases,
    state_domain_segment_keys,
)
from .strategy_effect_report import build_strategy_effect_report

AA_CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
AA_NO_CYS = set("ADEFGHIKLMNPQRSTVWY")
MUTATION_OPS = executable_operator_names("node")
MAX_DESIGN_REGIONS = 8
MAX_REGION_TARGETS = 6
MAX_REGION_RESIDUES = 16

DIAGNOSTIC_ONLY_SEMANTIC_FIELDS = frozenset(
    {"functional_nodes", "coupling_edges", "semantic_focus"}
)
OPERATOR_PHASE_ALIASES = {
    "explore": "explore",
    "exploit": "exploit",
    "repair": "repair",
    "refine": "exploit",
    "stabilize": "repair",
}
POSITION_RULE_FIELD_ALIASES = {
    "favored": "favored_residues",
    "disfavored": "disfavored_residues",
}
POSITION_RULE_FIELDS = frozenset(
    {
        "node",
        "position",
        "favored_residues",
        "disfavored_residues",
        "favored_residue_classes",
        "disfavored_residue_classes",
        "policy_weight",
        "intent",
        *POSITION_RULE_FIELD_ALIASES,
    }
)

KIND_LENGTH_RANGES: Dict[str, Tuple[int, int]] = {
    "cdr": (4, 24),
    "linker": (1, 20),
    "framework": (1, 80),
}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _strategy_tree(strategy: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("strategy_tree", "design_tree", "node_tree"):
        tree = strategy.get(key)
        if isinstance(tree, dict):
            return tree
    return {}


def _layout_plan(strategy: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("layout_plan", "domain_layout", "node_layout"):
        plan = strategy.get(key)
        if isinstance(plan, dict):
            return plan
    return {}


def _clamp_float(value: Any, default: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, _safe_float(value, default)))


def _clamp_int(value: Any, default: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, _safe_int(value, default)))


def _canonical_residue_list(value: Any, *, allow_cys: bool = False) -> List[str]:
    allowed = AA_CANONICAL if allow_cys else AA_NO_CYS
    out: List[str] = []
    for aa in _name_list(value):
        aa = aa.upper()
        if len(aa) == 1 and aa in allowed and aa not in out:
            out.append(aa)
        if len(out) >= MAX_REGION_RESIDUES:
            break
    return out


def _canonical_position_list(value: Any, limit: int = 64) -> List[int]:
    out: List[int] = []
    raw = value if isinstance(value, list) else []
    for item in raw:
        try:
            pos = int(round(float(item)))
        except (TypeError, ValueError):
            continue
        if pos >= 0 and pos not in out:
            out.append(pos)
        if len(out) >= limit:
            break
    return out


def _canonical_motif_list(value: Any, limit: int = 12) -> List[str]:
    raw = value if isinstance(value, list) else [value] if isinstance(value, str) else []
    out: List[str] = []
    for item in raw:
        motif = "".join(ch for ch in str(item).upper() if ch in AA_CANONICAL)
        if 2 <= len(motif) <= 48 and motif not in out:
            out.append(motif)
        if len(out) >= limit:
            break
    return out


def _canonical_domain_order(value: Any, state: Dict[str, Any]) -> List[str]:
    allowed = list(state_domain_segment_keys(state).keys())
    fallback = binder_domain_order(state)
    aliases = state_domain_aliases(state)
    if not isinstance(value, list):
        return fallback

    out: List[str] = []
    for item in value:
        text = str(item)
        canonical = aliases.get(text, text)
        if canonical in allowed and canonical not in out:
            out.append(canonical)
    for item in fallback:
        if item not in out:
            out.append(item)
    return out or fallback


def _node_length_range(node_name: str, kind: str) -> Tuple[int, int]:
    return KIND_LENGTH_RANGES.get(kind, (1, 80))


def _sanitize_length_range(value: Any, node_name: str, kind: str) -> Optional[List[int]]:
    if not (isinstance(value, list) and len(value) == 2):
        return None
    lo_bound, hi_bound = _node_length_range(node_name, kind)
    lo = _clamp_int(value[0], lo_bound, lo_bound, hi_bound)
    hi = _clamp_int(value[1], hi_bound, lo_bound, hi_bound)
    if lo > hi:
        lo, hi = hi, lo
    return [lo, hi]


def _sanitize_mutation_ops(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, dict):
        return None
    validated = validate_operator_weights(
        value,
        mode="node",
        context="layout_plan.design_regions[].mutation_ops",
    )
    return {
        op: _clamp_float(weight, 0.0, 0.0, 1.0)
        for op, weight in validated.items()
    }


def _ss_code(value: Any) -> Optional[str]:
    text = str(value).strip().lower()
    if text in {"h", "helix", "alpha", "alpha_helix"}:
        return "H"
    if text in {"e", "beta", "strand", "sheet", "beta_strand"}:
        return "E"
    if text in {"l", "loop", "coil", "turn", "flexible_loop"}:
        return "L"
    return None


def _sanitize_site_anchor(value: Any, node_name: str, kind: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Any] = {}
    rel_positions = []
    for item in value.get("relative_positions", value.get("positions", [])) or []:
        pos = _safe_int(item, -1)
        if 0 <= pos < _node_length_range(node_name, kind)[1] and pos not in rel_positions:
            rel_positions.append(pos)
    if rel_positions:
        out["relative_positions"] = rel_positions[:8]

    if isinstance(value.get("relative_ranges"), list):
        ranges = []
        for raw in value["relative_ranges"]:
            if isinstance(raw, list) and len(raw) == 2:
                start = _clamp_int(raw[0], 0, 0, _node_length_range(node_name, kind)[1])
                end = _clamp_int(raw[1], start + 1, start + 1, _node_length_range(node_name, kind)[1])
                ranges.append([start, end])
        if ranges:
            out["relative_ranges"] = ranges[:4]

    out["weight"] = _clamp_float(value.get("weight", value.get("priority_boost", 2.0)), 2.0, 0.25, 6.0)
    residues = _canonical_residue_list(value.get("favored_residues", []))
    if residues:
        out["favored_residues"] = residues
    classes = _name_list(value.get("favored_residue_classes", []))[:8]
    if classes:
        out["favored_residue_classes"] = classes
    return out


def _unique_names(value: Any) -> List[str]:
    raw = value if isinstance(value, (list, tuple, set)) else [value]
    out: List[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _strict_residue_list(value: Any, *, path: str) -> List[str]:
    raw = value if isinstance(value, (list, tuple, set)) else [value]
    out: List[str] = []
    for item in raw:
        aa = str(item).strip().upper()
        if len(aa) != 1 or aa not in AA_CANONICAL:
            raise ValueError(f"{path} must contain only canonical amino acid codes")
        if aa not in out:
            out.append(aa)
    if not out:
        raise ValueError(f"{path} must contain at least one canonical amino acid")
    return out[:MAX_REGION_RESIDUES]


def _position_rule_targets(
    raw_position: int,
    rule: Dict[str, Any],
    targets: List[str],
    segment_meta: Dict[str, Dict[str, Any]],
    *,
    path: str,
) -> List[str]:
    explicit_node = str(rule.get("node") or "").strip()
    if explicit_node:
        if explicit_node not in targets:
            raise ValueError(
                f"{path}.node {explicit_node!r} must name one of region targets {targets}"
            )
        candidates = [explicit_node]
    else:


        candidates = [
            node
            for node in targets
            if int(segment_meta[node]["start"]) <= raw_position < int(segment_meta[node]["end"])
        ]
        if not candidates:
            candidates = [
                node
                for node in targets
                if 0 <= raw_position < int(segment_meta[node]["length"])
            ]
    if not candidates:
        raise ValueError(
            f"{path} position {raw_position} does not belong to target node(s) {targets}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"{path} position {raw_position} is ambiguous across nodes {candidates}; add a node field"
        )
    return candidates


def _sanitize_position_residue_rules(
    value: Any,
    targets: List[str],
    segment_meta: Dict[str, Dict[str, Any]],
    *,
    path: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{path} must be a non-empty position-to-rule mapping")
    canonical: Dict[str, Dict[str, Any]] = {}
    by_node: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for raw_key, raw_rule in value.items():
        try:
            position = int(str(raw_key).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} position key {raw_key!r} must be an integer") from exc
        if position < 0:
            raise ValueError(f"{path} position {position} must be non-negative")
        if not isinstance(raw_rule, dict):
            raise ValueError(f"{path}.{raw_key} must be a rule mapping")
        unknown = sorted(set(raw_rule) - POSITION_RULE_FIELDS)
        if unknown:
            raise ValueError(
                f"{path}.{raw_key} has unknown fields {unknown}; allowed fields are "
                f"{sorted(POSITION_RULE_FIELDS)}"
            )
        if "position" in raw_rule and _safe_int(raw_rule.get("position"), -1) != position:
            raise ValueError(f"{path}.{raw_key}.position must equal its mapping key {position}")
        nodes = _position_rule_targets(
            position,
            raw_rule,
            targets,
            segment_meta,
            path=f"{path}.{raw_key}",
        )
        rule: Dict[str, Any] = {}
        if "node" in raw_rule:
            rule["node"] = nodes[0]
        if "position" in raw_rule:
            rule["position"] = position
        for requested, canonical_name in (
            ("favored_residues", "favored_residues"),
            ("favored", "favored_residues"),
            ("disfavored_residues", "disfavored_residues"),
            ("disfavored", "disfavored_residues"),
        ):
            if requested not in raw_rule:
                continue
            if canonical_name in rule:
                raise ValueError(
                    f"{path}.{raw_key} mixes aliases for {canonical_name}"
                )
            rule[canonical_name] = _strict_residue_list(
                raw_rule[requested],
                path=f"{path}.{raw_key}.{requested}",
            )
        for field in ("favored_residue_classes", "disfavored_residue_classes"):
            if field in raw_rule:
                values = _unique_names(raw_rule[field])
                if not values:
                    raise ValueError(f"{path}.{raw_key}.{field} must be non-empty")
                rule[field] = values[:8]
        if "policy_weight" in raw_rule:
            try:
                weight = float(raw_rule["policy_weight"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}.{raw_key}.policy_weight must be numeric") from exc
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(f"{path}.{raw_key}.policy_weight must be finite and positive")
            rule["policy_weight"] = max(0.05, min(6.0, weight))
        if "intent" in raw_rule:
            rule["intent"] = str(raw_rule["intent"])[:240]
        if not any(
            field in rule
            for field in (
                "favored_residues",
                "disfavored_residues",
                "favored_residue_classes",
                "disfavored_residue_classes",
            )
        ):
            raise ValueError(f"{path}.{raw_key} has no executable residue-prior field")
        key = str(position)
        canonical[key] = rule
        for node in nodes:
            by_node.setdefault(node, {})[key] = dict(rule)
    return canonical, by_node


def _strip_diagnostic_semantic_fields(value: Any) -> None:
    if isinstance(value, dict):
        for field in DIAGNOSTIC_ONLY_SEMANTIC_FIELDS:
            value.pop(field, None)
        for child in value.values():
            _strip_diagnostic_semantic_fields(child)
    elif isinstance(value, list):
        for child in value:
            _strip_diagnostic_semantic_fields(child)


def _sanitize_tree_execution_fields(
    value: Any,
    segment_meta: Dict[str, Dict[str, Any]],
    *,
    path: str,
) -> None:
    if not isinstance(value, dict):
        return
    node_name = str(value.get("name") or value.get("node") or "")
    containers = [(value, path)]
    if isinstance(value.get("edit_policy"), dict):
        containers.append((value["edit_policy"], f"{path}.edit_policy"))
    for container, container_path in containers:
        if "operator_phase" in container:
            raw_phase = str(container.get("operator_phase") or "").strip().lower()
            if raw_phase not in OPERATOR_PHASE_ALIASES:
                raise ValueError(
                    f"{container_path}.operator_phase must be one of explore, exploit, repair "
                    "(legacy aliases: refine, stabilize)"
                )
            container["operator_phase"] = OPERATOR_PHASE_ALIASES[raw_phase]
        if "position_residue_rules" in container:
            if node_name not in segment_meta:
                raise ValueError(
                    f"{container_path}.position_residue_rules requires a known structural node name"
                )
            rules, _by_node = _sanitize_position_residue_rules(
                container["position_residue_rules"],
                [node_name],
                segment_meta,
                path=f"{container_path}.position_residue_rules",
            )
            container["position_residue_rules"] = rules
    for index, child in enumerate(value.get("children", []) or []):
        if isinstance(child, dict):
            _sanitize_tree_execution_fields(
                child,
                segment_meta,
                path=f"{path}.children[{index}]",
            )


def sanitize_strategy_for_ast(state: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:

    if not isinstance(strategy, dict):
        strategy = {}
    cleaned = deepcopy(strategy)


    for field in EXTERNAL_KB_FIELDS:
        cleaned.pop(field, None)


    for field in (
        "adaptive_prior_mode",
        "inner_state_scope",
        "mcts_memory_enabled",
        "memory_auto_update_enabled",
        "memory_update_max_recent_runs",
        "memory_update_max_residues_per_node",
    ):
        cleaned.pop(field, None)
    plan = dict(_layout_plan(cleaned))
    segment_meta = _segment_metadata(state)


    for field in DIAGNOSTIC_ONLY_SEMANTIC_FIELDS:
        cleaned.pop(field, None)
        plan.pop(field, None)
    for tree_name in ("strategy_tree", "design_tree", "node_tree"):
        if tree_name in cleaned:
            _strip_diagnostic_semantic_fields(cleaned[tree_name])
            _sanitize_tree_execution_fields(
                cleaned[tree_name],
                segment_meta,
                path=tree_name,
            )

    if "semantic_required_nodes" in strategy:
        required_nodes = _unique_names(strategy.get("semantic_required_nodes"))
        cleaned["semantic_required_nodes"] = required_nodes
        known_nodes = [node for node in required_nodes if node in segment_meta]
        unknown_nodes = [node for node in required_nodes if node not in segment_meta]
        cleaned["semantic_required_node_resolution"] = {
            "schema_version": "ast_semantic_required_node_resolution_v1",
            "requested": required_nodes,
            "known_structural_nodes": known_nodes,
            "unknown_structural_nodes": unknown_nodes,
            "unknown_policy": "retained_as_unavailable_coverage_requirement",
        }

    plan["binder_domain_order"] = _canonical_domain_order(plan.get("binder_domain_order"), state)
    regions = plan.get("design_regions", plan.get("regions", []))
    if not isinstance(regions, list):
        regions = []

    clean_regions: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for idx, region in enumerate(regions[:MAX_DESIGN_REGIONS], start=1):
        if not isinstance(region, dict):
            continue

        targets = _region_targets(state, region, segment_meta)[:MAX_REGION_TARGETS]
        if not targets:
            rejected.append({"name": str(region.get("name", f"region_{idx}")), "reason": "no valid target nodes"})
            continue

        out: Dict[str, Any] = {
            "name": str(region.get("name") or f"design_region_{idx}")[:80],
            "role": str(region.get("role") or region.get("intent") or "")[:240],
            "position": _clamp_int(region.get("position"), idx, 1, MAX_DESIGN_REGIONS),
            "bind_to": targets,
            "enabled": _as_bool(region.get("enabled"), True),
            "mutable": _as_bool(region.get("mutable"), True),
            "priority_boost": _clamp_float(region.get("priority_boost", 1.0), 1.0, 0.01, 5.0),
            "mutation_rate": _clamp_float(region.get("mutation_rate", 0.05), 0.05, 0.0, 0.30),
            "max_mutations_per_step": _clamp_int(region.get("max_mutations_per_step"), 2, 1, 16),
            "policy_weight": _clamp_float(region.get("policy_weight", 0.7), 0.7, 0.0, 1.0),
        }

        mut_ops = _sanitize_mutation_ops(region.get("mutation_ops"))
        if mut_ops is not None:
            out["mutation_ops"] = mut_ops

        for field in ("hotspot_positions", "anchor_positions", "mutable_positions", "protected_positions"):
            positions = _canonical_position_list(region.get(field))
            if positions:
                out[field] = positions

        for field in ("graft_motifs", "motif_candidates"):
            motifs = _canonical_motif_list(region.get(field))
            if motifs:
                out[field] = motifs

        if "operator_phase" in region:
            phase = str(region.get("operator_phase") or "").strip().lower()
            if phase not in OPERATOR_PHASE_ALIASES:
                raise ValueError(
                    "layout_plan.design_regions"
                    f"[{idx - 1}].operator_phase must be one of explore, exploit, repair "
                    "(legacy aliases: refine, stabilize)"
                )
            out["operator_phase"] = OPERATOR_PHASE_ALIASES[phase]
        if "large_jump" in region:
            out["large_jump"] = _as_bool(region.get("large_jump"), False)
        if isinstance(region.get("design_points"), dict):
            out["design_points"] = dict(region["design_points"])

        favored = _canonical_residue_list(region.get("favored_residues", []))
        if favored:
            out["favored_residues"] = favored
        disfavored = _canonical_residue_list(region.get("disfavored_residues", []), allow_cys=True)
        if "C" not in disfavored:
            disfavored.append("C")
        out["disfavored_residues"] = disfavored[:MAX_REGION_RESIDUES]

        for field in ("favored_residue_classes", "disfavored_residue_classes"):
            values = _name_list(region.get(field, []))[:8]
            if values:
                out[field] = values

        length_budget = region.get("length_budget")
        if length_budget is not None:
            out["length_budget"] = _clamp_int(length_budget, 0, 1, 72)

        for field in ("target_lengths", "length_deltas", "length_ranges", "node_weights"):
            raw = region.get(field)
            if not isinstance(raw, dict):
                continue
            vals: Dict[str, Any] = {}
            for node_name in targets:
                if node_name not in raw:
                    continue
                kind = segment_meta[node_name]["kind"]
                if field == "length_ranges":
                    sanitized = _sanitize_length_range(raw[node_name], node_name, kind)
                    if sanitized is not None:
                        vals[node_name] = sanitized
                elif field == "target_lengths":
                    lo, hi = _node_length_range(node_name, kind)
                    vals[node_name] = _clamp_int(raw[node_name], segment_meta[node_name]["length"], lo, hi)
                elif field == "length_deltas":
                    vals[node_name] = _clamp_int(raw[node_name], 0, -8, 8)
                elif field == "node_weights":
                    vals[node_name] = _clamp_float(raw[node_name], 1.0, 0.05, 5.0)
            if vals:
                out[field] = vals

        if "length_range" in region and len(targets) == 1:
            sanitized = _sanitize_length_range(region["length_range"], targets[0], segment_meta[targets[0]]["kind"])
            if sanitized is not None:
                out["length_range"] = sanitized
        if "target_length" in region and len(targets) == 1:
            lo, hi = _node_length_range(targets[0], segment_meta[targets[0]]["kind"])
            out["target_length"] = _clamp_int(region["target_length"], segment_meta[targets[0]]["length"], lo, hi)
        if "length_mutable" in region:
            out["length_mutable"] = _as_bool(region.get("length_mutable"), False)
        if "allow_framework_length_change" in region:
            out["allow_framework_length_change"] = False

        ss = _ss_code(region.get("secondary_structure", region.get("ss", None)))
        if ss is not None:
            out["secondary_structure"] = ss

        site_anchors = region.get("site_anchors", {})
        if isinstance(site_anchors, dict):
            clean_anchors = {}
            for node_name in targets:
                anchor = _sanitize_site_anchor(site_anchors.get(node_name), node_name, segment_meta[node_name]["kind"])
                if anchor:
                    clean_anchors[node_name] = anchor
            if clean_anchors:
                out["site_anchors"] = clean_anchors

        if "position_residue_rules" in region:
            canonical_rules, rules_by_node = _sanitize_position_residue_rules(
                region.get("position_residue_rules"),
                targets,
                segment_meta,
                path=f"layout_plan.design_regions[{idx - 1}].position_residue_rules",
            )
            out["position_residue_rules"] = canonical_rules
            out["_compiled_position_residue_rules_by_node"] = rules_by_node

        clean_regions.append(out)

    plan["design_regions"] = clean_regions
    plan.pop("regions", None)

    ss_priors = plan.get("secondary_structure_priors", {})
    if isinstance(ss_priors, dict):
        clean_ss = {}
        for node_name, ss_value in ss_priors.items():
            if node_name in segment_meta:
                code = _ss_code(ss_value)
                if code:
                    clean_ss[node_name] = code
        plan["secondary_structure_priors"] = clean_ss
    else:
        plan["secondary_structure_priors"] = {}

    cleaned["layout_plan"] = plan
    legacy_summary = {
        "active_region_count": len(clean_regions),
        "rejected_regions": rejected,
        "allowed_nodes": list(segment_meta.keys()),
        "domain_order": plan["binder_domain_order"],
    }
    cleaned["strategy_schema_report"] = build_strategy_effect_report(
        strategy,
        cleaned,
        legacy_summary=legacy_summary,
    )
    return cleaned


def _iter_strategy_nodes(
    node: Dict[str, Any],
    inherited_mutable: bool = True,
    path: Tuple[str, ...] = (),
):
    order = 0

    def walk(cur: Dict[str, Any], parent_mutable: bool, cur_path: Tuple[str, ...]):
        nonlocal order
        if not isinstance(cur, dict):
            return

        name = str(cur.get("name") or cur.get("node") or "")
        if "mutable" in cur:
            mutable = _as_bool(cur.get("mutable"), parent_mutable)
        elif "enabled" in cur:
            mutable = _as_bool(cur.get("enabled"), parent_mutable)
        else:
            mutable = parent_mutable

        order += 1
        next_path = cur_path + ((name,) if name else ())
        yield cur, mutable, order, next_path

        for child in cur.get("children", []) or []:
            if isinstance(child, dict):
                yield from walk(child, mutable, next_path)

    yield from walk(node, inherited_mutable, path)


def _segment_metadata(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    cursor = 0
    for name, kind, seq in flatten_binder_parts(state):
        length = len(seq)
        meta[str(name)] = {
            "kind": str(kind),
            "length": length,
            "start": cursor,
            "end": cursor + length,
        }
        cursor += length
    return meta


def _copy_node_field(policy: Dict[str, Any], node: Dict[str, Any], field: str) -> None:
    if field in node and field not in policy:
        policy[field] = node[field]


def _name_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(x) for x in value if str(x)]
    return []


def _segment_names_by_domain(state: Dict[str, Any], domain_name: str) -> List[str]:
    key = state_domain_segment_keys(state).get(str(domain_name), "")
    if not key:
        return []
    return [str(x[0]) for x in state["binder"].get(key, []) if len(x) >= 3]


def _region_targets(
    state: Dict[str, Any],
    region: Dict[str, Any],
    segment_meta: Dict[str, Dict[str, Any]],
) -> List[str]:
    explicit: List[str] = []
    for key in ("bind_to", "nodes", "segments", "target_nodes", "covers"):
        explicit.extend(_name_list(region.get(key)))

    targets: List[str] = []
    for name in explicit:
        if name in segment_meta and name not in targets:
            targets.append(name)

    if not targets:
        domain = region.get("domain") or region.get("parent_domain")
        kind_filter = region.get("kind_filter") or region.get("node_kind") or region.get("kind")
        domain_names = _segment_names_by_domain(state, str(domain)) if domain else list(segment_meta)
        for name in domain_names:
            if name not in segment_meta:
                continue
            if kind_filter and segment_meta[name]["kind"] != str(kind_filter):
                continue
            targets.append(name)

    max_nodes = _safe_int(region.get("max_nodes"), 0)
    if max_nodes > 0:
        targets = targets[:max_nodes]
    return targets


def _combine_unique(old: Any, new: Any) -> List[str]:
    out: List[str] = []
    for value in _name_list(old) + _name_list(new):
        if value not in out:
            out.append(value)
    return out


def _augment_large_step_ops_for_policy(policy: Dict[str, Any], kind: str, node_name: str) -> None:
    if not _as_bool(policy.get("large_jump"), False):
        return
    ops = policy.get("mutation_ops")
    if not isinstance(ops, dict):
        ops = {}
    kind_text = str(kind or "").lower()
    name_text = str(node_name or "").lower()
    has_explicit_motif = bool(
        policy.get("graft_motifs") or policy.get("motif_candidates")
    )
    if kind_text == "cdr" or "cdr" in name_text:
        additions = {"cdr_resample": 0.12, "segment_mutagenesis": 0.10}
    elif kind_text == "linker" or "linker" in name_text:
        additions = {"segment_resample": 0.20, "region_shuffle": 0.04}
    elif kind_text in {"pocket", "ligand", "dna_contact"} or any(token in name_text for token in ("pocket", "groove", "efhand", "loop")):
        additions = {"segment_mutagenesis": 0.08}
    elif kind_text in {"hinge", "relay", "framework", "helix"}:
        additions = {"segment_mutagenesis": 0.14}
    else:
        additions = {"segment_mutagenesis": 0.08}
    if has_explicit_motif:
        if kind_text == "cdr" or "cdr" in name_text:
            additions["motif_graft"] = 0.10
        elif kind_text in {"pocket", "ligand", "dna_contact"} or any(
            token in name_text for token in ("pocket", "groove", "efhand", "loop")
        ):
            additions["motif_graft"] = 0.20
        else:
            additions["motif_graft"] = 0.04
    for op, weight in additions.items():
        ops.setdefault(op, weight)
    policy["mutation_ops"] = ops


def _augment_large_step_ops(
    policies: Dict[str, Dict[str, Any]],
    segment_meta: Dict[str, Dict[str, Any]],
) -> None:
    for node_name, policy in policies.items():
        if not isinstance(policy, dict):
            continue
        kind = segment_meta.get(node_name, {}).get("kind", policy.get("kind", ""))
        _augment_large_step_ops_for_policy(policy, str(kind), str(node_name))


def _merge_region_policy(
    base: Dict[str, Any],
    region: Dict[str, Any],
    node_name: str,
    node_weight: float,
    region_order: int,
) -> None:
    region_name = str(region.get("name") or f"layout_region_{region_order}")
    role = region.get("role") or region.get("edit_intent") or region.get("intent")
    base["enabled"] = _as_bool(region.get("enabled"), _as_bool(base.get("enabled"), True))
    base["mutable"] = _as_bool(region.get("mutable"), _as_bool(base.get("mutable"), True))
    base["layout_position"] = min(_safe_int(base.get("layout_position"), region_order), region_order)

    old_priority = _safe_float(base.get("priority_boost", base.get("priority", 1.0)), 1.0)
    region_priority = _safe_float(region.get("priority_boost", region.get("priority", 1.0)), 1.0)
    base["priority_boost"] = max(0.01, old_priority * region_priority * max(0.05, node_weight))

    if role and "edit_intent" not in base:
        base["edit_intent"] = str(role)

    for field in (
        "target_length",
        "length",
        "length_delta",
        "length_range",
        "min_length",
        "max_length",
        "length_mutable",
        "mutation_rate",
        "mutation_ops",
        "max_mutations_per_step",
        "aa_weights",
        "fill_residues",
        "policy_weight",
        "confidence",
        "allow_framework_length_change",
        "secondary_structure",
        "position_weights",
        "hotspot_positions",
        "anchor_positions",
        "mutable_positions",
        "protected_positions",
        "graft_motifs",
        "motif_candidates",
        "operator_phase",
        "large_jump",
        "design_points",
    ):
        if field in region:
            base[field] = region[field]

    position_rules_by_node = region.get("_compiled_position_residue_rules_by_node")
    if isinstance(position_rules_by_node, dict):
        node_rules = position_rules_by_node.get(node_name)
        if isinstance(node_rules, dict) and node_rules:
            base["position_residue_rules"] = deepcopy(node_rules)

    for field in (
        "favored_residues",
        "disfavored_residues",
        "favored_residue_classes",
        "disfavored_residue_classes",
    ):
        if field in region:
            base[field] = _combine_unique(base.get(field, []), region.get(field, []))

    target_lengths = region.get("target_lengths")
    if isinstance(target_lengths, dict) and node_name in target_lengths:
        base["target_length"] = target_lengths[node_name]
        base["length_mutable"] = True

    length_deltas = region.get("length_deltas")
    if isinstance(length_deltas, dict) and node_name in length_deltas:
        base["length_delta"] = length_deltas[node_name]
        base["length_mutable"] = True

    node_ranges = region.get("length_ranges")
    if isinstance(node_ranges, dict) and node_name in node_ranges:
        base["length_range"] = node_ranges[node_name]
        base["length_mutable"] = True

    if "length_bias" in region:
        base["length_bias"] = region["length_bias"]

    site_anchors = region.get("site_anchors")
    if isinstance(site_anchors, dict):
        node_anchor = site_anchors.get(node_name)
        if isinstance(node_anchor, dict):
            base["site_anchors"] = node_anchor
            base["favored_residues"] = _combine_unique(
                base.get("favored_residues", []),
                node_anchor.get("favored_residues", []),
            )
            base["favored_residue_classes"] = _combine_unique(
                base.get("favored_residue_classes", []),
                node_anchor.get("favored_residue_classes", []),
            )

    regions = list(base.get("layout_regions", []))
    if region_name not in regions:
        regions.append(region_name)
    base["layout_regions"] = regions


def _apply_region_length_budget(
    region: Dict[str, Any],
    targets: List[str],
    policies: Dict[str, Dict[str, Any]],
    segment_meta: Dict[str, Dict[str, Any]],
) -> None:
    if not targets or "length_budget" not in region:
        return
    budget = _safe_int(region.get("length_budget"), 0)
    if budget <= 0:
        return

    node_weights = region.get("node_weights", {})
    if not isinstance(node_weights, dict):
        node_weights = {}
    raw_weights = []
    for name in targets:
        raw_weights.append(max(0.01, _safe_float(node_weights.get(name), segment_meta[name]["length"])))
    total_weight = sum(raw_weights) or 1.0

    assigned: Dict[str, int] = {}
    remaining = int(budget)
    for idx, name in enumerate(targets):
        if idx == len(targets) - 1:
            target_len = max(1, remaining)
        else:
            target_len = max(1, int(round(budget * raw_weights[idx] / total_weight)))
            remaining -= target_len
        assigned[name] = target_len

    node_ranges = region.get("length_ranges", {})
    if not isinstance(node_ranges, dict):
        node_ranges = {}

    for name, target_len in assigned.items():
        kind = segment_meta[name]["kind"]
        lo, hi = _node_length_range(name, kind)
        if name in node_ranges:
            sanitized_range = _sanitize_length_range(node_ranges[name], name, kind)
            if sanitized_range is not None:
                lo, hi = sanitized_range
        target_len = max(lo, min(hi, int(target_len)))
        policies.setdefault(name, {})["target_length"] = target_len
        policies[name]["length_mutable"] = True


def _apply_layout_plan_to_policies(
    state: Dict[str, Any],
    strategy: Dict[str, Any],
    policies: Dict[str, Dict[str, Any]],
    memory_bias: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:


    plan = _layout_plan(strategy)
    regions = plan.get("design_regions", plan.get("regions", [])) if plan else []
    if not isinstance(regions, list):
        regions = []

    segment_meta = _segment_metadata(state)
    active_regions: List[Dict[str, Any]] = []

    for order, region in enumerate(regions, start=1):
        if not isinstance(region, dict) or not _as_bool(region.get("enabled"), True):
            continue
        targets = _region_targets(state, region, segment_meta)
        if not targets:
            continue

        node_weights = region.get("node_weights", {})
        if not isinstance(node_weights, dict):
            node_weights = {}

        _apply_region_length_budget(region, targets, policies, segment_meta)
        for name in targets:
            base = policies.setdefault(
                name,
                {
                    "node_name": name,
                    "kind": segment_meta[name]["kind"],
                    "current_length": segment_meta[name]["length"],
                    "mutable": True,
                    "enabled": True,
                    "priority_boost": 1.0,
                },
            )
            _merge_region_policy(
                base,
                region,
                name,
                _safe_float(node_weights.get(name), 1.0),
                _safe_int(region.get("position"), order),
            )

        active_regions.append(
            {
                "name": str(region.get("name") or f"layout_region_{order}"),
                "role": region.get("role", ""),
                "targets": targets,
                "position": _safe_int(region.get("position"), order),
            }
        )

    if policies:
        edit_entries = []
        for name, policy in policies.items():
            if not _as_bool(policy.get("enabled"), _as_bool(policy.get("mutable"), False)):
                continue
            edit_entries.append(
                (
                    _safe_int(policy.get("layout_position", policy.get("tree_order", 9999)), 9999),
                    -_safe_float(policy.get("priority_boost", 1.0), 1.0),
                    name,
                )
            )
        edit_entries.sort()
        edit_order = [name for _pos, _priority, name in edit_entries]
    else:
        edit_order = list(memory_bias.get("preferred_edit_order", []))

    return {"active_regions": active_regions}, edit_order


def _collect_strategy_tree_policies(
    state: Dict[str, Any],
    strategy: Dict[str, Any],
    memory_bias: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:


    tree = _strategy_tree(strategy)
    if not tree:
        return {}, []

    segment_meta = _segment_metadata(state)
    policies: Dict[str, Dict[str, Any]] = {}
    edit_entries: List[Tuple[float, int, str]] = []

    for node, inherited_mutable, order, path in _iter_strategy_nodes(tree):
        name = str(node.get("name") or node.get("node") or "")
        if name not in segment_meta:
            continue

        policy = dict(node.get("edit_policy") or {})
        for field in (
            "target_length",
            "length",
            "length_delta",
            "length_range",
            "min_length",
            "max_length",
            "length_mutable",
            "mutable",
            "enabled",
            "priority",
            "priority_boost",
            "mutation_rate",
            "mutation_ops",
            "max_mutations_per_step",
            "favored_residues",
            "disfavored_residues",
            "favored_residue_classes",
            "disfavored_residue_classes",
            "aa_weights",
            "fill_residues",
            "seed_sequence",
            "template_sequence",
            "policy_weight",
            "confidence",
            "edit_intent",
            "allow_framework_length_change",
            "secondary_structure",
            "position_weights",
            "position_residue_rules",
            "hotspot_positions",
            "anchor_positions",
            "mutable_positions",
            "protected_positions",
            "graft_motifs",
            "motif_candidates",
            "operator_phase",
            "large_jump",
            "design_points",
            "site_anchors",
        ):
            _copy_node_field(policy, node, field)

        mutable = _as_bool(policy.get("mutable"), inherited_mutable)
        enabled = _as_bool(policy.get("enabled"), mutable)
        priority = _safe_float(
            policy.get("priority_boost", policy.get("priority", 1.0)),
            1.0,
        )

        policy.update(
            {
                "node_name": name,
                "kind": segment_meta[name]["kind"],
                "current_length": segment_meta[name]["length"],
                "mutable": mutable,
                "enabled": enabled,
                "priority_boost": priority,
                "tree_order": order,
                "tree_path": list(path),
            }
        )
        policies[name] = policy

        if enabled and mutable:
            edit_entries.append((priority, order, name))

    if edit_entries:
        edit_entries.sort(key=lambda x: (-x[0], x[1]))
        edit_order = [name for _priority, _order, name in edit_entries]
    else:
        edit_order = list(memory_bias.get("preferred_edit_order", []))

    return policies, edit_order


def normalize_strategy_tree(
    state: Dict[str, Any],
    strategy: Dict[str, Any],
    memory_bias: Dict[str, Any],
) -> Dict[str, Any]:


    normalized = dict(strategy)
    policies, edit_order = _collect_strategy_tree_policies(state, strategy, memory_bias)
    layout_summary, layout_edit_order = _apply_layout_plan_to_policies(
        state,
        strategy,
        policies,
        memory_bias,
    )
    _augment_large_step_ops(policies, _segment_metadata(state))
    if layout_edit_order:
        edit_order = layout_edit_order
    if not policies:
        normalized.setdefault("node_edit_policies", {})
        return normalized

    normalized["_tree_policy_active"] = True
    normalized["_layout_plan_active"] = bool(layout_summary.get("active_regions"))
    normalized["layout_summary"] = layout_summary
    normalized["node_edit_policies"] = policies
    if not normalized.get("preferred_edit_order"):
        normalized["preferred_edit_order"] = edit_order
    return normalized


def _sanitize_aa_sequence(seq: Any) -> str:
    return "".join(aa for aa in str(seq).upper() if aa in AA_CANONICAL)


def _fill_residues(kind: str, policy: Dict[str, Any]) -> str:
    explicit = _sanitize_aa_sequence(policy.get("fill_residues", ""))
    if explicit:
        return explicit
    if kind == "linker":
        return "GGGGS"
    if kind == "cdr":
        return "YSGNQ"
    return "S"


def _repeat_to_length(seed: str, length: int) -> str:
    if length <= 0:
        return ""
    seed = seed or "S"
    repeats = (length + len(seed) - 1) // len(seed)
    return (seed * repeats)[:length]


def _resize_segment_sequence(seq: str, target_len: int, kind: str, policy: Dict[str, Any]) -> str:
    seed = _sanitize_aa_sequence(policy.get("seed_sequence") or policy.get("template_sequence") or "")
    if seed:
        seq = seed
    seq = _sanitize_aa_sequence(seq)
    if len(seq) == target_len:
        return seq
    if target_len <= 0:
        return ""

    if len(seq) > target_len:
        if kind == "cdr" and target_len >= 2:
            left = target_len // 2
            right = target_len - left
            return seq[:left] + seq[-right:]
        return seq[:target_len]

    insert = _repeat_to_length(_fill_residues(kind, policy), target_len - len(seq))
    if kind == "cdr" and len(seq) >= 2:
        mid = len(seq) // 2
        return seq[:mid] + insert + seq[mid:]
    return seq + insert


def _length_bounds(kind: str, current_len: int, policy: Dict[str, Any]) -> Tuple[int, int]:
    if isinstance(policy.get("length_range"), list) and len(policy["length_range"]) == 2:
        lo = _safe_int(policy["length_range"][0], current_len)
        hi = _safe_int(policy["length_range"][1], current_len)
    else:
        if kind == "cdr":
            lo, hi = 4, 24
        elif kind == "linker":
            lo, hi = 1, 20
        else:
            lo, hi = current_len, current_len
        lo = _safe_int(policy.get("min_length"), lo)
        hi = _safe_int(policy.get("max_length"), hi)
    lo = max(1, min(lo, hi))
    hi = max(lo, hi)
    return lo, hi


def _target_length(kind: str, current_len: int, policy: Dict[str, Any]) -> int:
    if "target_length" in policy:
        raw_target = _safe_int(policy.get("target_length"), current_len)
    elif "length" in policy:
        raw_target = _safe_int(policy.get("length"), current_len)
    elif "length_delta" in policy:
        raw_target = current_len + _safe_int(policy.get("length_delta"), 0)
    elif str(policy.get("length_bias", "")).lower() in {"extend", "longer", "expand"}:
        raw_target = current_len + 1
    elif str(policy.get("length_bias", "")).lower() in {"compact", "shorter", "shrink"}:
        raw_target = current_len - 1
    else:
        raw_target = current_len
    lo, hi = _length_bounds(kind, current_len, policy)
    return max(lo, min(hi, raw_target))


def _apply_layout_domain_order(state: Dict[str, Any], strategy: Dict[str, Any]) -> None:
    plan = _layout_plan(strategy)
    order = None
    for key in ("binder_domain_order", "domain_order", "binder_order"):
        if isinstance(plan.get(key), list):
            order = plan[key]
            break
        if isinstance(strategy.get(key), list):
            order = strategy[key]
            break
    if order is not None:
        state["binder"]["domain_order"] = order


def apply_strategy_tree_to_state(state: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:


    policies = strategy.get("node_edit_policies", {})
    updated = deepcopy(state)
    _apply_layout_domain_order(updated, strategy)

    if not isinstance(policies, dict) or not policies:
        return updated

    fixed_linker = set(updated["mutation_policy"].get("fixed_linker_segments", []))
    for group in state_domain_segment_keys(updated).values():
        for segment in updated["binder"].get(group, []):
            if len(segment) < 3:
                continue
            name, kind, seq = str(segment[0]), str(segment[1]), str(segment[2])
            policy = policies.get(name)
            if not isinstance(policy, dict):
                continue
            if not _as_bool(policy.get("enabled"), _as_bool(policy.get("mutable"), True)):
                continue
            if not _as_bool(policy.get("length_mutable"), False):
                continue
            if kind == "framework" and not _as_bool(policy.get("allow_framework_length_change"), False):
                continue
            if name in fixed_linker:
                continue

            target_len = _target_length(kind, len(seq), policy)
            if target_len != len(seq):
                segment[2] = _resize_segment_sequence(seq, target_len, kind, policy)

    return updated
