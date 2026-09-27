

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from astevolve.evaluation.contracts import ScoreTerm
from astevolve.evaluation.support import get_nested
from astevolve.evaluation.views import node_names_from_design_state


def semantic_nodes(design_state: Mapping[str, Any]) -> List[str]:
    graph = (
        design_state.get("semantic_graph")
        if isinstance(design_state.get("semantic_graph"), Mapping)
        else {}
    )
    structural = get_nested(graph, "structural_graph", "nodes") or {}
    if isinstance(structural, Mapping):
        return [str(name) for name in structural]
    nodes: List[str] = []
    for key in ("preserved", "primary", "secondary"):
        nodes.extend(node_names_from_design_state(design_state, key))
    return sorted(set(nodes))


def pick_nodes(
    design_state: Mapping[str, Any],
    roles: Sequence[str],
    fallback: Sequence[str] = (),
) -> List[str]:


    graph = (
        design_state.get("semantic_graph")
        if isinstance(design_state.get("semantic_graph"), Mapping)
        else {}
    )
    structural = get_nested(graph, "structural_graph", "nodes") or {}
    role_set = {str(role).strip().lower() for role in roles}
    selected: List[str] = []
    if isinstance(structural, Mapping):
        for name, raw_node in structural.items():
            node = raw_node if isinstance(raw_node, Mapping) else {}
            declared = node.get("roles", node.get("role", []))
            values = declared if isinstance(declared, list) else [declared]
            if role_set & {str(value).strip().lower() for value in values}:
                selected.append(str(name))
    if not selected:
        selected = [str(node) for node in fallback if str(node)]
    return selected[:4]


def add_recommendation(
    recommendations: List[Dict[str, Any]],
    *,
    node: str,
    action: str,
    reason: str,
    source_terms: Sequence[str],
    priority: str = "medium",
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    if not node:
        return
    key = (node, action, reason)
    for existing in recommendations:
        if (
            existing.get("node"),
            existing.get("action"),
            existing.get("reason"),
        ) == key:
            existing["source_terms"] = sorted(
                set(existing.get("source_terms", []) + list(source_terms))
            )
            return
    recommendations.append(
        {
            "node": node,
            "action": action,
            "reason": reason,
            "priority": priority,
            "source_terms": list(source_terms),
            **{
                name: value
                for name, value in dict(metadata or {}).items()
                if value not in (None, [], {})
            },
        }
    )


def term_is_actionable_for_recommendation(term: ScoreTerm) -> bool:
    if float(term.weight) <= 0.0:
        return False
    if term.available:
        return True
    details = term.details if isinstance(term.details, Mapping) else {}
    return bool(
        details.get("hard_gate")
        or details.get("semantic_binding")
        or details.get("semantic_bindings")
        or details.get("edit_recommendations")
    )


def _as_mappings(value: Any) -> List[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _declared_recommendations(term: ScoreTerm) -> List[Dict[str, Any]]:
    details = term.details if isinstance(term.details, Mapping) else {}
    declared = _as_mappings(details.get("edit_recommendations"))
    bindings = _as_mappings(
        details.get("semantic_bindings", details.get("semantic_binding"))
    )
    rows: List[Dict[str, Any]] = [dict(item) for item in declared]
    for binding in bindings:
        nodes = binding.get("structural_nodes", binding.get("structural_node", []))
        nodes = nodes if isinstance(nodes, list) else [nodes]
        for node in nodes:
            if node:
                rows.append(
                    {
                        "node": str(node),
                        "action": str(
                            binding.get("failure_action")
                            or binding.get("recommended_action")
                            or "optimize_node"
                        ),
                        "reason": str(
                            binding.get("reason")
                            or f"declared semantic binding for {term.name} is weak"
                        ),
                        "priority": str(binding.get("priority") or "high"),
                    }
                )
    return rows


def recommended_edit_targets(
    terms: Sequence[ScoreTerm],
    gate: Mapping[str, Any],
    design_state: Mapping[str, Any],
) -> List[Dict[str, Any]]:


    recommendations: List[Dict[str, Any]] = []
    actionable = sorted(
        (term for term in terms if term_is_actionable_for_recommendation(term)),
        key=lambda item: (item.score, -item.weight, item.name),
    )
    for term in actionable[:10]:
        for row in _declared_recommendations(term):
            add_recommendation(
                recommendations,
                node=str(row.get("node") or ""),
                action=str(row.get("action") or "optimize_node"),
                reason=str(row.get("reason") or f"weak evaluator term: {term.name}"),
                source_terms=[term.name],
                priority=str(row.get("priority") or "medium"),
                metadata={
                    key: value
                    for key, value in row.items()
                    if key not in {"node", "action", "reason", "priority"}
                },
            )

    if not recommendations and actionable:
        primary = node_names_from_design_state(design_state, "primary")[:2]
        weakest = actionable[0]
        dimension = (
            weakest.details.get("dimension")
            if isinstance(weakest.details, Mapping)
            else None
        )
        action = "repair_node" if dimension == "quality" else "optimize_node"
        for node in primary:
            add_recommendation(
                recommendations,
                node=node,
                action=action,
                reason="weak evaluator evidence has no more specific declared binding",
                source_terms=[weakest.name],
                priority="low",
                metadata={"binding_status": "unbound_generic_fallback"},
            )

    if not gate.get("hard_gate_pass", True) and not recommendations:
        add_recommendation(
            recommendations,
            node="mutation_policy",
            action="redesign_constraints",
            reason="hard gate failed without a declared node binding",
            source_terms=[str(item) for item in gate.get("disqualification_reasons", [])],
            priority="critical",
            metadata={"binding_status": "unbound_gate_fallback"},
        )

    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return sorted(
        recommendations,
        key=lambda item: (
            priority_order.get(str(item.get("priority")), 9),
            str(item.get("node")),
            str(item.get("action")),
        ),
    )[:10]
