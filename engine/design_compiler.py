

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from astevolve.core.protein_lang import Blueprint, Node

from .design_state import (
    binder_domain_order,
    binder_sequence,
    flatten_binder_parts,
    has_target,
    segment_spans,
    state_domain_segment_keys,
)
from .strategy_compiler import (
    _as_bool,
    _canonical_residue_list,
    _clamp_float,
    _layout_plan,
    _name_list,
    _region_targets,
    _segment_metadata,
    _ss_code,
)

def _make_group_node(name: str, children: List[List[str]]) -> Node:
    child_kinds = {str(segment[1]) for segment in children if len(segment) >= 2}
    group_kind = "linker" if child_kinds == {"linker"} else "domain"
    return Node(
        kind=group_kind,
        name=name,
        children=[
            Node(kind=kind, name=segment_name, length=len(seq))
            for segment_name, kind, seq in children
        ],
    )


def make_binder_chain(state: Dict[str, Any]) -> Node:


    binder = state["binder"]
    children: List[Node] = []
    segment_keys = state_domain_segment_keys(state)
    for domain_name in binder_domain_order(state):
        key = segment_keys.get(domain_name)
        if key:
            children.append(_make_group_node(domain_name, binder[key]))

    return Node(
        kind="chain",
        name=binder.get("name", "Binder"),
        props={"chain_id": binder.get("chain_id", "BB")},
        children=children,
    )


def make_target_chain(state: Dict[str, Any]) -> Node:


    target = state["target"]
    return Node(
        kind="chain",
        name=target.get("name", "Target"),
        length=len(target["sequence"]),
        props={"chain_id": target.get("chain_id", "T")},
        children=[
            Node(
                kind=target.get("feature_kind", "epitope"),
                name=target.get("epitope_name", "target_feature"),
                residue_spans=[tuple(x) for x in target.get("epitope_spans", [])],
                props={
                    "source": target.get("epitope_source", ""),
                    "hotspot_motif": target.get("hotspot_motif", ""),
                },
            )
        ],
    )


def build_blueprint(state: Dict[str, Any]) -> Blueprint:


    children = [make_binder_chain(state)]
    if has_target(state):
        children.append(make_target_chain(state))

    return Blueprint(
        root=Node(
            kind="complex",
            name=state.get("task_name", "ASTevolve_Task"),
            children=children,
        )
    )


def _mask_from_spans(length: int, allowed_spans: List[Tuple[int, int]]) -> List[bool]:
    mask = [False] * length
    for start, end in allowed_spans:
        for idx in range(max(0, start), min(length, end)):
            mask[idx] = True
    return mask


def _spans_from_constraint_entries(entries: Any, default_chain: str) -> Dict[str, List[Tuple[int, int]]]:
    out: Dict[str, List[Tuple[int, int]]] = {}
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        chain_id = str(entry.get("chain_id") or entry.get("chain") or default_chain)
        raw_spans = entry.get("spans", entry.get("ranges", entry.get("residue_ranges", [])))
        spans: List[Tuple[int, int]] = []
        if isinstance(raw_spans, list):
            for raw in raw_spans:
                if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                    continue
                try:
                    start, end = int(raw[0]), int(raw[1])
                except (TypeError, ValueError):
                    continue
                spans.append((start, end))
        raw_residues = entry.get("residues", entry.get("indices", []))
        if isinstance(raw_residues, list):
            for raw in raw_residues:
                try:
                    idx = int(raw)
                except (TypeError, ValueError):
                    continue
                spans.append((idx, idx + 1))
        if spans:
            out.setdefault(chain_id, []).extend(spans)
    return out


def _constraint_spans_by_chain(
    state: Dict[str, Any],
    span_key: str,
    node_key: str,
) -> Dict[str, List[Tuple[int, int]]]:
    constraints = state.get("design_constraints", {})
    if not isinstance(constraints, dict):
        return {}
    binder_chain = state["binder"].get("chain_id", "BB")
    spans = segment_spans(flatten_binder_parts(state))
    out: Dict[str, List[Tuple[int, int]]] = {}
    for node_name in _name_list(constraints.get(node_key, [])):
        if node_name in spans:
            out.setdefault(binder_chain, []).append(spans[node_name])
    explicit = _spans_from_constraint_entries(constraints.get(span_key, []), binder_chain)
    for chain_id, chain_spans in explicit.items():
        out.setdefault(chain_id, []).extend(chain_spans)
    return out


def _apply_closed_spans(mask: List[bool], closed_spans: List[Tuple[int, int]]) -> None:
    for start, end in closed_spans:
        for idx in range(max(0, start), min(len(mask), end)):
            mask[idx] = False


def _global_ast_spans(
    state: Dict[str, Any], field: str, chain_id: str
) -> List[Tuple[int, int]]:
    policy = state.get("global_ast_evolution_policy")
    if not isinstance(policy, dict) or not bool(policy.get("enabled")):
        return []
    if policy.get("schema_version") != "astevolve.ast_evolution_policy.v2":
        return []
    by_chain = policy.get(field)
    if not isinstance(by_chain, dict):
        return []
    raw_spans = by_chain.get(chain_id)
    if not isinstance(raw_spans, list):
        return []
    spans: List[Tuple[int, int]] = []
    for raw_span in raw_spans:
        if not isinstance(raw_span, (list, tuple)) or len(raw_span) != 2:
            continue
        try:
            start, end = int(raw_span[0]), int(raw_span[1])
        except (TypeError, ValueError):
            continue
        if 0 <= start < end:
            spans.append((start, end))
    return spans


def build_masks(state: Dict[str, Any], memory_bias: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, List[bool]]:


    parts = flatten_binder_parts(state)
    spans = segment_spans(parts)
    policy = state["mutation_policy"]
    always_open = set(policy.get("always_open_segments", []))
    conditionally_open = set(policy.get("conditionally_open_segments", []))
    edit_order = list(strategy.get("preferred_edit_order") or memory_bias.get("preferred_edit_order", []))
    design_points = state.get("design_points", {}) if isinstance(state.get("design_points"), dict) else {}
    for key in ("primary_design_nodes", "secondary_design_nodes", "default_open_nodes"):
        for name in _name_list(design_points.get(key, [])):
            if name not in edit_order:
                edit_order.append(name)
    tree_policy_active = bool(strategy.get("_tree_policy_active"))
    node_policies = strategy.get("node_edit_policies", {})

    selected: List[Tuple[int, int]] = []
    for name in edit_order:
        if tree_policy_active:
            node_policy = node_policies.get(name, {}) if isinstance(node_policies, dict) else {}
            if not _as_bool(node_policy.get("enabled"), _as_bool(node_policy.get("mutable"), False)):
                continue
        if name in always_open or name in conditionally_open:
            if name in spans and spans[name] not in selected:
                selected.append(spans[name])
        elif name in spans and name in set(_name_list(design_points.get("default_open_nodes", []))):
            selected.append(spans[name])
    if not tree_policy_active:
        for name in sorted(always_open):
            if name in spans and spans[name] not in selected:
                selected.append(spans[name])

    binder_chain = state["binder"].get("chain_id", "BB")
    for span in _global_ast_spans(
        state, "allowed_chain_spans", str(binder_chain)
    ):
        if span not in selected:
            selected.append(span)
    for span in _constraint_spans_by_chain(state, "mutable_residue_spans", "mutable_nodes").get(binder_chain, []):
        if span not in selected:
            selected.append(span)
    binder_mask = _mask_from_spans(len(binder_sequence(state)), selected)
    frozen = _constraint_spans_by_chain(state, "frozen_residue_spans", "frozen_nodes")
    _apply_closed_spans(binder_mask, frozen.get(binder_chain, []))
    _apply_closed_spans(
        binder_mask,
        _global_ast_spans(state, "protected_chain_spans", str(binder_chain)),
    )

    masks = {binder_chain: binder_mask}
    if has_target(state):
        target = state["target"]
        masks[target.get("chain_id", "T")] = [False] * len(target["sequence"])
    return masks


def build_fixed_residues(state: Dict[str, Any], memory_bias: Dict[str, Any]) -> Dict[str, Dict[int, str]]:


    parts = flatten_binder_parts(state)
    spans = segment_spans(parts)
    fixed: Dict[str, Dict[int, str]] = {
        state["binder"].get("chain_id", "BB"): {},
    }
    if has_target(state):
        fixed[state["target"].get("chain_id", "T")] = {}

    defaults_by_segment = memory_bias.get("fixed_segment_sequences", {})
    if not isinstance(defaults_by_segment, dict):
        defaults_by_segment = {}
    shared_default = str(memory_bias.get("linker_default_sequence", ""))
    shared_offset = 0
    part_sequences = {
        str(name): str(sequence)
        for name, _kind, sequence in parts
    }
    for segment_name in state["mutation_policy"].get("fixed_linker_segments", []):
        if segment_name not in spans:
            continue
        start, end = spans[segment_name]
        length = max(0, end - start)
        configured = defaults_by_segment.get(segment_name)
        if isinstance(configured, str):
            segment_default = configured
        elif shared_offset + length <= len(shared_default):
            segment_default = shared_default[shared_offset : shared_offset + length]
        else:
            segment_default = part_sequences.get(str(segment_name), "")
        for i, pos in enumerate(range(start, end)):
            if i < len(segment_default):
                fixed[state["binder"].get("chain_id", "BB")][pos] = segment_default[i]
        shared_offset += length

    binder_chain = state["binder"].get("chain_id", "BB")
    binder_seq = binder_sequence(state)
    for start, end in _constraint_spans_by_chain(state, "frozen_residue_spans", "frozen_nodes").get(binder_chain, []):
        for pos in range(max(0, start), min(len(binder_seq), end)):
            fixed[binder_chain][pos] = binder_seq[pos]
    for start, end in _global_ast_spans(
        state, "protected_chain_spans", str(binder_chain)
    ):
        for pos in range(max(0, start), min(len(binder_seq), end)):
            fixed[binder_chain][pos] = binder_seq[pos]

    if has_target(state):
        target = state["target"]
        for i, aa in enumerate(target["sequence"]):
            fixed[target.get("chain_id", "T")][i] = aa

    return fixed


def _secondary_structure_constraint_specs(
    state: Dict[str, Any],
    strategy: Dict[str, Any],
) -> List[Dict[str, Any]]:
    plan = _layout_plan(strategy)
    binder_chain = state["binder"].get("chain_id", "BB")
    segment_meta = _segment_metadata(state)
    target_map: Dict[str, str] = {}

    ss_priors = plan.get("secondary_structure_priors", {})
    if isinstance(ss_priors, dict):
        for node_name, ss_value in ss_priors.items():
            if node_name in segment_meta:
                code = _ss_code(ss_value)
                if code:
                    target_map[f"{binder_chain}:{node_name}"] = code

    regions = plan.get("design_regions", [])
    if isinstance(regions, list):
        for region in regions:
            if not isinstance(region, dict):
                continue
            code = _ss_code(region.get("secondary_structure", region.get("ss", None)))
            if not code:
                continue
            for node_name in _region_targets(state, region, segment_meta):
                target_map[f"{binder_chain}:{node_name}"] = code

    if not target_map:
        return []
    return [
        {
            "kind": "ss_proxy",
            "weight": float(strategy.get("secondary_structure_weight", 0.35)),
            "params": {"target_map": target_map},
        }
    ]


def _site_anchor_constraint_specs(
    state: Dict[str, Any],
    strategy: Dict[str, Any],
) -> List[Dict[str, Any]]:
    plan = _layout_plan(strategy)
    binder_chain = state["binder"].get("chain_id", "BB")
    segment_meta = _segment_metadata(state)
    specs: List[Dict[str, Any]] = []

    regions = plan.get("design_regions", [])
    if not isinstance(regions, list):
        return specs

    for region in regions:
        if not isinstance(region, dict):
            continue
        site_anchors = region.get("site_anchors", {})
        if not isinstance(site_anchors, dict):
            continue
        for node_name in _region_targets(state, region, segment_meta):
            anchor = site_anchors.get(node_name)
            if not isinstance(anchor, dict):
                continue
            favored = _canonical_residue_list(anchor.get("favored_residues", []))
            if not favored:
                continue
            specs.append(
                {
                    "kind": "segment_composition",
                    "weight": _clamp_float(anchor.get("constraint_weight", 0.25), 0.25, 0.0, 1.0),
                    "params": {
                        "aa_set": "".join(favored),
                        "min_frac": 0.05,
                        "max_frac": 0.90,
                        "segment_filter": {
                            "chain_id": binder_chain,
                            "name": node_name,
                        },
                    },
                }
            )
    return specs


def build_constraint_specs(
    state: Dict[str, Any],
    memory_bias: Dict[str, Any],
    strategy: Dict[str, Any],
) -> List[Dict[str, Any]]:


    binder_chain = state["binder"].get("chain_id", "BB")
    allow_cysteine = bool(state.get("mutation_policy", {}).get("allow_cysteine"))
    allowed_alphabet = set("ACDEFGHIKLMNPQRSTVWY") if allow_cysteine else set(
        "ADEFGHIKLMNPQRSTVWY"
    )
    segment_kinds = {
        str(kind).strip().lower()
        for _name, kind, _sequence in flatten_binder_parts(state)
    }
    specs: List[Dict[str, Any]] = [
        {"kind": "alphabet", "weight": 1.0, "params": {"allowed": allowed_alphabet, "chain_ids": [binder_chain]}},
        {"kind": "hydrophobic_pattern", "weight": 0.8, "params": {"domain_min_hydro": 0.20, "linker_max_hydro": float(strategy.get("linker_hydrophobic_max", memory_bias.get("linker_hydrophobic_max", 0.35)))}},
        {"kind": "max_run", "weight": 1.2, "params": {"aa_set": "AILMFWVY", "max_run": int(strategy.get("max_hydrophobic_run", 3)), "segment_filter": {"chain_id": binder_chain}}},
        {"kind": "max_run", "weight": 0.8, "params": {"aa_set": "KRDE", "max_run": int(strategy.get("max_charged_run", 3)), "segment_filter": {"chain_id": binder_chain}}},
    ]
    if not allow_cysteine:
        specs.append(
            {"kind": "segment_composition", "weight": 0.7, "params": {"aa_set": "C", "min_frac": 0.0, "max_frac": 0.0, "segment_filter": {"chain_id": binder_chain}}}
        )
    if has_target(state):
        target = state["target"]
        specs.insert(
            1,
            {"kind": "fixed_chain_sequence", "weight": 1.0, "params": {"chain_id": target.get("chain_id", "T"), "sequence": target["sequence"]}},
        )
    if "pocket" in segment_kinds:
        specs.extend([
            {"kind": "segment_composition", "weight": 0.7, "params": {"aa_set": "AILMFWVY", "min_frac": 0.05, "max_frac": float(strategy.get("pocket_hydrophobic_max", 0.55)), "segment_filter": {"chain_id": binder_chain, "kind": "pocket"}}},
            {"kind": "segment_composition", "weight": 0.6, "params": {"aa_set": "YHSTNQDEKR", "min_frac": 0.15, "max_frac": 0.90, "segment_filter": {"chain_id": binder_chain, "kind": "pocket"}}},
        ])
    if "hinge" in segment_kinds:
        specs.append({"kind": "segment_composition", "weight": 0.5, "params": {"aa_set": "GSTNQAP", "min_frac": 0.12, "max_frac": 0.85, "segment_filter": {"chain_id": binder_chain, "kind": "hinge"}}})
    if "linker" in segment_kinds:
        linker_gs_min = float(strategy.get("linker_gs_min", memory_bias.get("linker_gs_min", 0.60)))
        linker_hydrophobic_max = float(strategy.get("linker_hydrophobic_max", memory_bias.get("linker_hydrophobic_max", 0.15)))
        linker_charged_max = float(strategy.get("linker_charged_max", memory_bias.get("linker_charged_max", 0.20)))
        specs.extend([
            {"kind": "segment_composition", "weight": 2.0, "params": {"aa_set": "GSAT", "min_frac": linker_gs_min, "max_frac": 1.0, "segment_filter": {"chain_id": binder_chain, "kind": "linker"}}},
            {"kind": "segment_composition", "weight": 1.3, "params": {"aa_set": "AILMFWVY", "min_frac": 0.0, "max_frac": linker_hydrophobic_max, "segment_filter": {"chain_id": binder_chain, "kind": "linker"}}},
            {"kind": "segment_composition", "weight": 1.0, "params": {"aa_set": "KRDE", "min_frac": 0.0, "max_frac": linker_charged_max, "segment_filter": {"chain_id": binder_chain, "kind": "linker"}}},
        ])
    if "cdr" in segment_kinds:
        cdr_favored = set(strategy.get("cdr_favored_residues") or memory_bias.get("cdr_favored_sparse", [])) or set("YWHNQSTRDE")
        specs.extend([
            {"kind": "segment_composition", "weight": 1.2, "params": {"aa_set": "".join(sorted(cdr_favored)), "min_frac": 0.20, "max_frac": 0.85, "segment_filter": {"chain_id": binder_chain, "kind": "cdr"}}},
            {"kind": "segment_composition", "weight": 1.0, "params": {"aa_set": "AILMFWVY", "min_frac": 0.05, "max_frac": float(strategy.get("cdr_hydrophobic_max", 0.40)), "segment_filter": {"chain_id": binder_chain, "kind": "cdr"}}},
            {"kind": "segment_composition", "weight": 0.8, "params": {"aa_set": "KRDE", "min_frac": 0.05, "max_frac": float(strategy.get("cdr_charged_max", 0.45)), "segment_filter": {"chain_id": binder_chain, "kind": "cdr"}}},
        ])
    if "framework" in segment_kinds:
        specs.extend([
            {"kind": "segment_composition", "weight": 1.0, "params": {"aa_set": "AILMFWVY", "min_frac": 0.10, "max_frac": 0.38, "segment_filter": {"chain_id": binder_chain, "kind": "framework"}}},
            {"kind": "segment_composition", "weight": 0.8, "params": {"aa_set": "KRDE", "min_frac": 0.05, "max_frac": 0.35, "segment_filter": {"chain_id": binder_chain, "kind": "framework"}}},
        ])

    constraints = state.get("design_constraints", {})
    explicit_interface = constraints.get("interface_proxy") if isinstance(constraints, dict) else None
    if isinstance(explicit_interface, dict):
        required = {"binder_segment", "target_segment"}
        if not required.issubset(explicit_interface):
            raise ValueError("design_constraints.interface_proxy requires binder_segment and target_segment")
        specs.append({
            "kind": "interface_proxy",
            "weight": float(explicit_interface.get("weight", 0.8)),
            "params": {
                "binder_chain": str(explicit_interface.get("binder_chain", binder_chain)),
                "binder_segment": str(explicit_interface["binder_segment"]),
                "target_chain": str(explicit_interface.get("target_chain", target_chain)),
                "target_segment": str(explicit_interface["target_segment"]),
                "desired_binder_hydro": float(explicit_interface.get("desired_binder_hydro", 0.30)),
            },
        })
    explicit_specs = constraints.get("fast_constraint_specs", []) if isinstance(constraints, dict) else []
    if explicit_specs:
        if not isinstance(explicit_specs, list) or not all(isinstance(item, dict) and item.get("kind") for item in explicit_specs):
            raise ValueError("design_constraints.fast_constraint_specs must be a list of constraint mappings")
        specs.extend(dict(item) for item in explicit_specs)
    specs.extend(_secondary_structure_constraint_specs(state, strategy))
    specs.extend(_site_anchor_constraint_specs(state, strategy))
    return specs
