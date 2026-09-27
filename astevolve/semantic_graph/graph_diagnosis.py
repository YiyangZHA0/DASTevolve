

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .model import normalize_graph_summary


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _strings(value: Any) -> List[str]:
    return [str(item) for item in _as_list(value) if str(item)]


def _score(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _term_weight(term: Mapping[str, Any]) -> float:
    try:
        return max(0.0, float(term.get("weight") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _term_details(term: Mapping[str, Any]) -> Mapping[str, Any]:
    details = term.get("details")
    return details if isinstance(details, Mapping) else {}


def _is_actionable_term(term: Mapping[str, Any]) -> bool:
    if _term_weight(term) <= 0.0:
        return False
    if bool(term.get("available", True)):
        return True
    details = _term_details(term)
    return bool(
        details.get("hard_gate")
        or details.get("semantic_binding")
        or details.get("semantic_bindings")
    )


def _binding_matches(binding: Any, term: Mapping[str, Any]) -> bool:


    if isinstance(binding, str):
        return str(term.get("name") or "") == binding
    if not isinstance(binding, Mapping):
        return False
    conditions = []
    names = _strings(
        binding.get("terms")
        or binding.get("evaluator_bindings")
        or binding.get("term")
        or binding.get("evaluator")
        or binding.get("evaluator_binding")
    )
    if names:
        conditions.append(str(term.get("name") or "") in names)
    categories = _strings(binding.get("categories") or binding.get("category"))
    if categories:
        conditions.append(str(term.get("category") or "") in categories)
    backends = _strings(binding.get("backends") or binding.get("backend"))
    if backends:
        conditions.append(str(term.get("backend") or "") in backends)
    return bool(conditions) and all(conditions)


def _node_bindings(node: Mapping[str, Any]) -> List[Any]:
    raw = (
        node.get("evaluator_bindings")
        if "evaluator_bindings" in node
        else node.get("evaluator_binding")
    )
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    return [raw]


def _mapped_structural_nodes(
    graph: Mapping[str, Any], functional_node: str
) -> List[str]:
    mapping = (
        graph.get("functional_to_structural")
        if isinstance(graph.get("functional_to_structural"), Mapping)
        else {}
    )
    mapped = mapping.get(functional_node)
    if isinstance(mapped, list):
        return [str(item) for item in mapped]
    functional = (
        graph.get("functional_nodes")
        if isinstance(graph.get("functional_nodes"), Mapping)
        else {}
    )
    node = functional.get(functional_node)
    if isinstance(node, Mapping):
        return _strings(node.get("maps_to"))
    return []


def _default_failure_action(node: Mapping[str, Any]) -> str:
    explicit = node.get("failure_action") or node.get("recommended_action")
    if explicit:
        return str(explicit)
    state = str(node.get("state") or "").strip().lower()
    kind = str(node.get("kind") or "").strip().lower()
    if state in {"negative", "avoid", "reject", "minimize"}:
        return "increase_negative_design_weight"
    if state in {"preserve", "constraint"} or kind in {
        "constraint",
        "guardrail",
        "invariant",
    }:
        return "repair_node"
    return "optimize_node"


def _detail_bindings(term: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    details = _term_details(term)
    raw = details.get("semantic_bindings")
    if raw is None:
        raw = details.get("semantic_binding")
    return [item for item in _as_list(raw) if isinstance(item, Mapping)]


def _term_impacts(
    term: Mapping[str, Any], graph: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    if not _is_actionable_term(term):
        return []
    name = str(term.get("name") or "")
    score = _score(term.get("score"))
    weight = _term_weight(term)
    severity = max(0.0, 1.0 - score) * weight
    functional = (
        graph.get("functional_nodes")
        if isinstance(graph.get("functional_nodes"), Mapping)
        else {}
    )
    structural = (
        graph.get("structural_nodes")
        if isinstance(graph.get("structural_nodes"), Mapping)
        else {}
    )
    impacts: List[Dict[str, Any]] = []

    for functional_name, raw_node in functional.items():
        node = raw_node if isinstance(raw_node, Mapping) else {}
        matched = [
            binding
            for binding in _node_bindings(node)
            if _binding_matches(binding, term)
        ]
        if not matched:
            continue
        binding = matched[0] if isinstance(matched[0], Mapping) else {}
        mapped = [
            item
            for item in _mapped_structural_nodes(graph, str(functional_name))
            if item in structural
        ]
        impacts.append(
            {
                "term": name,
                "functional_node": str(functional_name),
                "structural_nodes": mapped,
                "severity": severity,
                "term_score": score,
                "term_weight": weight,
                "recommended_action": str(
                    binding.get("failure_action")
                    or binding.get("recommended_action")
                    or _default_failure_action(node)
                ),
                "binding_source": "functional_node.evaluator_binding",
            }
        )

    for binding in _detail_bindings(term):
        functional_names = _strings(
            binding.get("functional_nodes") or binding.get("functional_node")
        )
        if not functional_names:
            continue
        for functional_name in functional_names:
            if functional_name not in functional:
                continue
            node = functional[functional_name]
            node = node if isinstance(node, Mapping) else {}
            declared_structural = _strings(
                binding.get("structural_nodes") or binding.get("structural_node")
            )
            mapped = declared_structural or _mapped_structural_nodes(
                graph, functional_name
            )
            mapped = [item for item in mapped if item in structural]
            impacts.append(
                {
                    "term": name,
                    "functional_node": functional_name,
                    "structural_nodes": mapped,
                    "severity": severity,
                    "term_score": score,
                    "term_weight": weight,
                    "recommended_action": str(
                        binding.get("failure_action")
                        or binding.get("recommended_action")
                        or _default_failure_action(node)
                    ),
                    "binding_source": "score_term.details.semantic_binding",
                }
            )

    deduplicated: Dict[tuple, Dict[str, Any]] = {}
    for impact in impacts:
        key = (
            impact["term"],
            impact["functional_node"],
            tuple(impact["structural_nodes"]),
            impact["recommended_action"],
        )
        deduplicated.setdefault(key, impact)
    return list(deduplicated.values())


def _collect_negative_design_plan(
    evaluator_report: Mapping[str, Any], source_terms: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    plans: List[Dict[str, Any]] = []
    for recommendation in _as_list(evaluator_report.get("recommended_edit_targets")):
        if isinstance(recommendation, Mapping):
            plans.extend(
                dict(item)
                for item in _as_list(
                    recommendation.get("negative_design_position_plan")
                )
                if isinstance(item, Mapping)
            )
    for term in source_terms:
        plans.extend(
            dict(item)
            for item in _as_list(
                _term_details(term).get("negative_design_position_plan")
            )
            if isinstance(item, Mapping)
        )
    deduplicated: Dict[tuple, Dict[str, Any]] = {}
    for item in plans:
        key = (
            str(item.get("chain_id") or ""),
            str(item.get("position") or ""),
            str(item.get("node") or ""),
        )
        deduplicated.setdefault(key, item)
    return list(deduplicated.values())[:20]


def diagnose_semantic_graph(
    evaluator_report: Mapping[str, Any],
    semantic_graph_summary: Mapping[str, Any],
    design_state: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:


    del design_state
    graph = normalize_graph_summary(semantic_graph_summary or {})
    terms = (
        evaluator_report.get("terms")
        if isinstance(evaluator_report.get("terms"), list)
        else []
    )
    weakest = (
        evaluator_report.get("weakest_terms")
        if isinstance(evaluator_report.get("weakest_terms"), list)
        else []
    )
    source_terms = [
        item for item in list(terms) + list(weakest) if isinstance(item, Mapping)
    ]

    functional_scores: Dict[str, Dict[str, Any]] = {}
    structural_scores: Dict[str, Dict[str, Any]] = {}
    impacts: List[Dict[str, Any]] = []
    seen_terms = set()
    for term in source_terms:
        term_key = (
            str(term.get("name") or ""),
            str(term.get("category") or ""),
            str(term.get("backend") or ""),
        )
        if term_key in seen_terms:
            continue
        seen_terms.add(term_key)
        for impact in _term_impacts(term, graph):
            impacts.append(impact)
            functional_name = impact["functional_node"]
            functional_scores.setdefault(
                functional_name,
                {"severity": 0.0, "terms": [], "recommended_actions": {}},
            )
            functional_scores[functional_name]["severity"] += impact["severity"]
            functional_scores[functional_name]["terms"].append(impact["term"])
            action = impact["recommended_action"]
            action_counts = functional_scores[functional_name][
                "recommended_actions"
            ]
            action_counts[action] = action_counts.get(action, 0) + 1
            for structural_name in impact["structural_nodes"]:
                structural_scores.setdefault(
                    structural_name,
                    {"severity": 0.0, "terms": [], "functional_nodes": []},
                )
                structural_scores[structural_name]["severity"] += impact[
                    "severity"
                ]
                structural_scores[structural_name]["terms"].append(impact["term"])
                structural_scores[structural_name]["functional_nodes"].append(
                    functional_name
                )

    for table in (functional_scores, structural_scores):
        for item in table.values():
            item["terms"] = sorted(set(item["terms"]))[:10]
            if "functional_nodes" in item:
                item["functional_nodes"] = sorted(
                    set(item["functional_nodes"])
                )[:10]
            item["score"] = max(
                0.0,
                min(1.0, 1.0 - min(1.0, item["severity"] / 3.0)),
            )

    bottleneck_functional = sorted(
        ({"node": key, **value} for key, value in functional_scores.items()),
        key=lambda item: item["severity"],
        reverse=True,
    )[:5]
    bottleneck_structural = sorted(
        ({"node": key, **value} for key, value in structural_scores.items()),
        key=lambda item: item["severity"],
        reverse=True,
    )[:8]
    gate = (
        evaluator_report.get("gate_status")
        if isinstance(evaluator_report.get("gate_status"), Mapping)
        else {}
    )
    hard_gate_pass = gate.get(
        "hard_gate_pass", evaluator_report.get("hard_gate_pass", True)
    )
    disqualification_reasons = gate.get(
        "disqualification_reasons",
        evaluator_report.get("disqualification_reasons", []),
    )
    return {
        "schema_version": "ast_semantic_graph_diagnosis_v1",
        "enabled": bool(graph.get("enabled")),
        "binding_policy": "explicit_evaluator_bindings_only",
        "functional_node_scores": functional_scores,
        "structural_node_scores": structural_scores,
        "bottleneck_functional_nodes": bottleneck_functional,
        "bottleneck_structural_nodes": bottleneck_structural,
        "term_node_impacts": impacts[:30],
        "unbound_actionable_terms": sorted(
            {
                str(term.get("name") or "")
                for term in source_terms
                if _is_actionable_term(term) and not _term_impacts(term, graph)
            }
        ),
        "negative_design_position_plan": _collect_negative_design_plan(
            evaluator_report, source_terms
        ),
        "hard_gate_pass": bool(hard_gate_pass),
        "disqualification_reasons": list(disqualification_reasons or []),
    }
