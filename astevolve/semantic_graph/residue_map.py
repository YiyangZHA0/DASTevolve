

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .summary import build_semantic_graph_summary


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _listify(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _span_positions(spans: Any) -> List[int]:
    out: List[int] = []
    for span in _listify(spans):
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        try:
            start = int(span[0])
            end = int(span[1])
        except (TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        out.extend(range(max(0, start), max(0, end)))
    return sorted(set(out))


def _compiled_segments(compiled: Optional[Mapping[str, Any]]) -> List[Any]:
    if not compiled:
        return []
    return list(compiled.get("segments", []) or [])


def _segment_positions(seg: Any) -> List[int]:
    spans = getattr(seg, "spans", None)
    if spans:
        return _span_positions(spans)
    try:
        start = int(getattr(seg, "start"))
        end = int(getattr(seg, "end"))
    except (TypeError, ValueError):
        return []
    return list(range(max(0, start), max(0, end)))


def _sequence_for_chain(
    chain_id: str,
    sequences: Optional[Mapping[str, str]],
    compiled: Optional[Mapping[str, Any]],
) -> str:
    if sequences and chain_id in sequences:
        return str(sequences[chain_id] or "")
    lengths = _as_dict((compiled or {}).get("chain_lengths") if compiled else {})
    try:
        return "X" * int(lengths.get(chain_id, 0) or 0)
    except (TypeError, ValueError):
        return ""


def _functional_reverse_map(summary: Mapping[str, Any]) -> Dict[str, List[str]]:
    mapping = _as_dict(summary.get("functional_to_structural"))
    reverse: Dict[str, List[str]] = defaultdict(list)
    for fn_id, nodes in mapping.items():
        for node in _listify(nodes):
            node_id = str(node)
            if node_id:
                reverse[node_id].append(str(fn_id))
    return {key: sorted(set(values)) for key, values in reverse.items()}


def _executable_functional_reverse_map(
    summary: Mapping[str, Any],
) -> Dict[str, List[str]]:
    mapping = _as_dict(summary.get("executable_functional_to_structural"))
    reverse: Dict[str, List[str]] = defaultdict(list)
    for fn_id, nodes in mapping.items():
        for node in _listify(nodes):
            node_id = str(node)
            if node_id:
                reverse[node_id].append(str(fn_id))
    return {key: sorted(set(values)) for key, values in reverse.items()}


def _executable_position_map(
    summary: Mapping[str, Any],
) -> Dict[tuple[str, int], str]:


    nodes = _as_dict(summary.get("executable_structural_nodes"))
    out: Dict[tuple[str, int], str] = {}
    for node_id in sorted(nodes):
        node = _as_dict(nodes.get(node_id))
        chain_id = str(node.get("chain_id") or "")
        if not chain_id:
            continue
        selected = node.get("selected_positions")
        positions = (
            sorted({int(value) for value in selected})
            if isinstance(selected, (list, tuple))
            else _span_positions(node.get("spans"))
        )
        for position in positions:


            out.setdefault((chain_id, int(position)), str(node_id))
    return out


def _structural_flags(
    structural: Mapping[str, Any],
    *,
    position: int,
    frozen_by_case: bool,
) -> tuple[bool, bool]:
    frozen = bool(structural.get("frozen")) or bool(frozen_by_case)
    editable = bool(structural.get("editable"))
    if "legal_positions" in structural:
        legal = {
            int(value) for value in _listify(structural.get("legal_positions"))
        }
        editable = editable and int(position) in legal
    return bool(editable and not frozen), frozen


def _collect_case_residue_sets(
    case_sheet: Optional[Mapping[str, Any]],
    state: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:

    case_sheet = case_sheet or {}
    state = state or {}
    constraints = _as_dict(case_sheet.get("residue_level_constraints"))
    state_constraints = _as_dict(state.get("residue_level_constraints"))
    if not constraints:
        constraints = state_constraints

    mutable: Dict[tuple[str, int], List[str]] = defaultdict(list)
    frozen: Dict[tuple[str, int], List[str]] = defaultdict(list)
    protected: Dict[tuple[str, int], List[str]] = defaultdict(list)
    hotspot_by_node: Dict[str, List[str]] = defaultdict(list)
    hotspot_by_pos: Dict[tuple[str, int], List[str]] = defaultdict(list)

    def add_span_rows(rows: Any, target: Dict[tuple[str, int], List[str]], label_field: str = "node") -> None:
        for row in _listify(rows):
            if not isinstance(row, Mapping):
                continue
            chain_id = str(row.get("chain_id") or row.get("chain") or "")
            label = str(row.get(label_field) or row.get("node") or row.get("name") or "")
            if not chain_id:
                continue
            for pos in _span_positions(row.get("spans")):
                target[(chain_id, int(pos))].append(label)

    add_span_rows(constraints.get("mutable_residue_spans"), mutable)
    add_span_rows(constraints.get("frozen_residue_spans"), frozen)
    add_span_rows(constraints.get("protected_residue_spans"), protected)
    add_span_rows(state_constraints.get("mutable_residue_spans"), mutable)
    add_span_rows(state_constraints.get("frozen_residue_spans"), frozen)
    add_span_rows(state_constraints.get("protected_residue_spans"), protected)

    for hotspot in _listify(constraints.get("binder_hotspots")) + _listify(state_constraints.get("binder_hotspots")):
        if not isinstance(hotspot, Mapping):
            continue
        name = str(hotspot.get("name") or "hotspot")
        chain_id = str(hotspot.get("chain_id") or hotspot.get("chain") or "")
        for node in _listify(hotspot.get("nodes")):
            if str(node):
                hotspot_by_node[str(node)].append(name)
        if chain_id:
            for span in _listify(hotspot.get("spans")):
                for pos in _span_positions([span]):
                    hotspot_by_pos[(chain_id, int(pos))].append(name)

    target_epitopes: Dict[str, Dict[str, Any]] = {}
    for key in ("target_epitopes", "decoy_epitopes"):
        for epitope in _listify(constraints.get(key)) + _listify(state_constraints.get(key)):
            if not isinstance(epitope, Mapping):
                continue
            name = str(epitope.get("name") or "")
            if not name:
                continue
            target_epitopes[name] = {
                "entity": epitope.get("entity"),
                "spans": epitope.get("spans", []),
                "role": epitope.get("role"),
                "kind": "target" if key == "target_epitopes" else "decoy",
            }

    return {
        "mutable_positions": mutable,
        "frozen_positions": frozen,
        "protected_positions": protected,
        "hotspot_by_node": {k: sorted(set(v)) for k, v in hotspot_by_node.items()},
        "hotspot_by_pos": hotspot_by_pos,
        "epitopes": target_epitopes,
    }


def build_residue_semantic_map(
    state: Optional[Mapping[str, Any]],
    compiled: Optional[Mapping[str, Any]] = None,
    sequences: Optional[Mapping[str, str]] = None,
    semantic_graph_summary: Optional[Mapping[str, Any]] = None,
    case_sheet: Optional[Mapping[str, Any]] = None,
    node_plddt: Any = None,
) -> Dict[str, Any]:

    state = state or {}
    compiled = compiled or {}
    summary = dict(semantic_graph_summary or build_semantic_graph_summary(state, compiled, node_plddt))
    structural_nodes = _as_dict(summary.get("structural_nodes"))
    functional_by_structural = _functional_reverse_map(summary)
    executable_functional_by_structural = _executable_functional_reverse_map(
        summary
    )
    executable_by_position = _executable_position_map(summary)
    case_sets = _collect_case_residue_sets(case_sheet or _as_dict(state.get("_case_sheet")), state)
    mutable_positions = case_sets["mutable_positions"]
    frozen_positions = case_sets["frozen_positions"]
    protected_positions = case_sets["protected_positions"]
    hotspot_by_node = case_sets["hotspot_by_node"]
    hotspot_by_pos = case_sets["hotspot_by_pos"]

    residues: List[Dict[str, Any]] = []
    lookup: Dict[str, Dict[str, Any]] = {}
    covered: set[tuple[str, int]] = set()
    for seg in _compiled_segments(compiled):
        segment_node_id = str(getattr(seg, "name", "") or "")
        chain_id = str(getattr(seg, "chain_id", "") or "")
        if not segment_node_id or not chain_id:
            continue
        seq = _sequence_for_chain(chain_id, sequences, compiled)
        positions = _segment_positions(seg)
        for pos in positions:
            aa = seq[pos] if 0 <= pos < len(seq) else "X"
            pos_key = (chain_id, int(pos))
            executable_node_id = executable_by_position.get(pos_key)
            node_id = executable_node_id or segment_node_id
            structural = _as_dict(structural_nodes.get(node_id))
            functional_nodes = (
                executable_functional_by_structural.get(node_id, [])
                if executable_node_id
                else functional_by_structural.get(node_id, [])
            )
            node_hotspots = list(hotspot_by_node.get(node_id, []))
            hotspot_names = sorted(set(node_hotspots + list(hotspot_by_pos.get(pos_key, []))))
            editable, frozen = _structural_flags(
                structural,
                position=int(pos),
                frozen_by_case=pos_key in frozen_positions,
            )
            row = {
                "chain_id": chain_id,
                "position0": int(pos),
                "position1": int(pos) + 1,
                "aa": aa,
                "structural_node": node_id,
                "compiled_segment": segment_node_id,
                "structural_kind": structural.get("kind") or str(getattr(seg, "kind", "")),
                "structural_role": structural.get("role"),
                "functional_nodes": functional_nodes,
                "editable": editable,
                "frozen": frozen,
                "mutable_by_case": pos_key in mutable_positions,
                "protected": pos_key in protected_positions,
                "hotspot_names": hotspot_names,
                "semantic_labels": sorted(
                    set(
                        [str(structural.get("kind") or ""), node_id]
                        + functional_nodes
                        + hotspot_names
                    )
                    - {""}
                ),
            }
            residues.append(row)
            lookup[f"{chain_id}:{int(pos)}"] = row
            covered.add(pos_key)


    for epitope_name, epitope in case_sets["epitopes"].items():
        entity = str(epitope.get("entity") or "")
        chain_id = "T" if entity == "target_peptide" else entity
        if not chain_id or (sequences and chain_id not in sequences):
            continue
        seq = _sequence_for_chain(chain_id, sequences, compiled)
        for pos in _span_positions(epitope.get("spans")):
            pos_key = (chain_id, int(pos))
            if pos_key in covered:
                continue
            row = {
                "chain_id": chain_id,
                "position0": int(pos),
                "position1": int(pos) + 1,
                "aa": seq[pos] if 0 <= pos < len(seq) else "X",
                "structural_node": epitope_name,
                "compiled_segment": None,
                "structural_kind": "Interface",
                "structural_role": epitope.get("role"),
                "functional_nodes": functional_by_structural.get(epitope_name, []),
                "editable": False,
                "frozen": False,
                "mutable_by_case": False,
                "protected": False,
                "hotspot_names": [str(epitope.get("kind") or "epitope")],
                "semantic_labels": sorted(
                    set(["Interface", epitope_name, str(epitope.get("role") or "")])
                    - {""}
                ),
            }
            residues.append(row)
            lookup[f"{chain_id}:{int(pos)}"] = row

    residues.sort(key=lambda row: (str(row.get("chain_id", "")), int(row.get("position0", 0))))
    return {
        "schema_version": "ast_residue_semantic_map_v1",
        "enabled": bool(summary.get("enabled")),
        "indexing": "zero_based_positions; position1 is one_based",
        "residue_count": len(residues),
        "residues": residues,
        "lookup": lookup,
        "summary": summarize_residue_semantic_map({"residues": residues}),
    }


def summarize_residue_semantic_map(residue_map: Optional[Mapping[str, Any]], limit: int = 40) -> Dict[str, Any]:


    residue_map = residue_map or {}
    residues = list(residue_map.get("residues", []) or [])
    structural_counts: Counter[str] = Counter()
    functional_counts: Counter[str] = Counter()
    editable_counts: Counter[str] = Counter()
    hotspot_counts: Counter[str] = Counter()
    node_ranges: Dict[str, Dict[str, Any]] = {}
    for row in residues:
        if not isinstance(row, Mapping):
            continue
        node = str(row.get("structural_node") or "")
        chain_id = str(row.get("chain_id") or "")
        pos = int(row.get("position0") or 0)
        if node:
            structural_counts[node] += 1
            entry = node_ranges.setdefault(
                node,
                {
                    "chain_id": chain_id,
                    "start": pos,
                    "end_exclusive": pos + 1,
                    "kind": row.get("structural_kind"),
                    "functional_nodes": set(),
                    "editable_residues": 0,
                    "frozen_residues": 0,
                    "hotspots": set(),
                },
            )
            entry["start"] = min(int(entry["start"]), pos)
            entry["end_exclusive"] = max(int(entry["end_exclusive"]), pos + 1)
            if row.get("editable") or row.get("mutable_by_case"):
                entry["editable_residues"] += 1
                editable_counts[node] += 1
            if row.get("frozen"):
                entry["frozen_residues"] += 1
            for fn in _listify(row.get("functional_nodes")):
                entry["functional_nodes"].add(str(fn))
        for fn in _listify(row.get("functional_nodes")):
            if str(fn):
                functional_counts[str(fn)] += 1
        for hot in _listify(row.get("hotspot_names")):
            if str(hot):
                hotspot_counts[str(hot)] += 1
                if node:
                    node_ranges[node]["hotspots"].add(str(hot))
    nodes = []
    for node, entry in node_ranges.items():
        item = dict(entry)
        item["node"] = node
        item["functional_nodes"] = sorted(item["functional_nodes"])
        item["hotspots"] = sorted(item["hotspots"])
        nodes.append(item)
    nodes.sort(key=lambda item: (str(item.get("chain_id", "")), int(item.get("start", 0))))
    return {
        "schema_version": "ast_residue_semantic_map_summary_v1",
        "residue_count": len(residues),
        "structural_node_counts": dict(structural_counts),
        "functional_node_counts": dict(functional_counts),
        "editable_node_counts": dict(editable_counts),
        "hotspot_counts": dict(hotspot_counts),
        "nodes": nodes[:limit],
    }


def annotate_mutations_with_residue_map(
    diffs: Mapping[str, Sequence[Mapping[str, Any]]],
    residue_map: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:


    lookup = _as_dict((residue_map or {}).get("lookup"))
    out: List[Dict[str, Any]] = []
    for chain_id, chain_diffs in (diffs or {}).items():
        for diff in chain_diffs or []:
            if not isinstance(diff, Mapping):
                continue
            try:
                pos = int(diff.get("position"))
            except (TypeError, ValueError):
                continue
            row = _as_dict(lookup.get(f"{chain_id}:{pos}"))
            out.append(
                {
                    "chain_id": str(chain_id),
                    "position0": pos,
                    "position1": pos + 1,
                    "from": diff.get("from"),
                    "to": diff.get("to"),
                    "structural_node": row.get("structural_node"),
                    "compiled_segment": row.get("compiled_segment"),
                    "structural_kind": row.get("structural_kind"),
                    "functional_nodes": list(row.get("functional_nodes", []) or []),
                    "editable": row.get("editable"),
                    "frozen": row.get("frozen"),
                    "mutable_by_case": row.get("mutable_by_case"),
                    "protected": row.get("protected"),
                    "hotspot_names": list(row.get("hotspot_names", []) or []),
                    "semantic_labels": list(row.get("semantic_labels", []) or []),
                }
            )
    out.sort(key=lambda item: (item["chain_id"], item["position0"]))
    return out


def summarize_mutation_semantics(annotations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:


    structural = Counter()
    functional = Counter()
    hotspots = Counter()
    frozen = 0
    protected = 0
    unassigned = 0
    for item in annotations or []:
        node = str(item.get("structural_node") or "")
        if node:
            structural[node] += 1
        else:
            unassigned += 1
        for fn in _listify(item.get("functional_nodes")):
            if str(fn):
                functional[str(fn)] += 1
        for hot in _listify(item.get("hotspot_names")):
            if str(hot):
                hotspots[str(hot)] += 1
        if item.get("frozen"):
            frozen += 1
        if item.get("protected"):
            protected += 1
    return {
        "schema_version": "ast_mutation_semantic_summary_v1",
        "total_mutations": len(list(annotations or [])),
        "mutations_by_structural_node": dict(structural),
        "mutations_by_functional_node": dict(functional),
        "mutations_by_hotspot": dict(hotspots),
        "frozen_mutation_count": int(frozen),
        "protected_mutation_count": int(protected),
        "unassigned_mutation_count": int(unassigned),
    }
