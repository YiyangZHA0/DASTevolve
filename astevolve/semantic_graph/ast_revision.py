

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import re
from typing import Any, Dict, Mapping, Sequence, Tuple

from astevolve.domain.dual_ast import (
    ExecutableDualAST,
    MappingEdgeSpec,
    ResidueSelector,
    StructuralNodeSpec,
)


AST_REVISION_PLAN_VERSION = "astevolve.ast_revision_plan.v1"
AST_EVOLUTION_POLICY_VERSION = "astevolve.ast_evolution_policy.v1"
AST_REVISION_REPORT_VERSION = "astevolve.ast_revision_report.v1"
GLOBAL_AST_REVISION_PLAN_VERSION = "astevolve.ast_revision_plan.v2"
GLOBAL_AST_EVOLUTION_POLICY_VERSION = "astevolve.ast_evolution_policy.v2"
GLOBAL_AST_REVISION_REPORT_VERSION = "astevolve.ast_revision_report.v2"

_GLOBAL_PLAN_FIELDS = frozenset(
    {"schema_version", "structural_nodes", "mapping_edges", "decision_record"}
)
_GLOBAL_NODE_FIELDS = frozenset(
    {
        "node_id",
        "selector",
        "action_profile",
        "intent",
        "evidence_refs",
        "residue_policy",
    }
)
_GLOBAL_EDGE_FIELDS = frozenset(
    {
        "edge_id",
        "functional_node_id",
        "structural_node_id",
        "action_operator",
        "evidence_refs",
    }
)
_GLOBAL_DECISION_FIELDS = frozenset(
    {
        "action",
        "diagnosis",
        "hypothesis",
        "evidence_refs",
        "expected_effects",
        "failure_condition",
        "confidence",
        "rationale",
        "rollback_condition",
    }
)
_GLOBAL_DECISION_REQUIRED_FIELDS = _GLOBAL_DECISION_FIELDS - {
    "rationale",
    "rollback_condition",
}
_GLOBAL_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "enabled",
        "allowed_chain_spans",
        "protected_chain_spans",
        "min_total_editable_positions",
        "max_total_editable_positions",
        "min_active_structural_nodes",
        "max_active_structural_nodes",
        "max_positions_per_node",
        "max_spans_per_node",
        "max_mapping_edges",
        "immutable_structural_node_ids",
        "allowed_functional_node_ids",
        "action_profiles",
        "require_catalog_eligibility",
        "require_position_evidence_refs",
        "min_position_safety_score",
        "node_id_pattern",
        "edge_id_pattern",
        "bootstrap_required_spans",
        "bootstrap_release_callable",
    }
)
GLOBAL_AST_DECISION_ACTIONS = (
    "keep",
    "create",
    "delete",
    "migrate",
    "resize",
    "revise",
    "rewire",
    "mixed",
    "revert",
)
_GLOBAL_DECISION_ACTIONS = frozenset(GLOBAL_AST_DECISION_ACTIONS)
_ACTION_PROFILE_FIELDS = frozenset({"allowed_actions"})
_ACTION_FIELDS = frozenset({"operator", "budget"})
_BUDGET_FIELDS = frozenset({"min", "max"})
_RESIDUE_POLICY_FIELDS = frozenset(
    {
        "favored_residues",
        "disfavored_residues",
        "position_residue_rules",
        "policy_weight",
    }
)
_POSITION_RESIDUE_RULE_FIELDS = frozenset(
    {"favored_residues", "disfavored_residues", "policy_weight", "intent"}
)
_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "structural_node_edits",
        "mapping_edge_edits",
        "rationale",
    }
)
_STRUCTURAL_EDIT_FIELDS = frozenset({"node_id", "selector"})
_MAPPING_EDIT_FIELDS = frozenset(
    {"edge_id", "enabled", "structural_node_id", "action_operator"}
)
_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "enabled",
        "max_structural_node_edits",
        "max_mapping_edge_edits",
        "max_added_positions",
        "max_removed_positions",
        "max_total_editable_positions",
        "structural_node_envelopes",
        "mapping_edge_allowlist",
    }
)
_ENVELOPE_FIELDS = frozenset(
    {"chain_id", "allowed_spans", "min_positions", "max_positions"}
)
_MAPPING_ALLOWLIST_FIELDS = frozenset(
    {"structural_node_ids", "action_operators", "allow_disable"}
)


class ASTRevisionError(ValueError):
    pass


def _closed_mapping(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ASTRevisionError(f"{label} must be a mapping")
    keys = {str(key) for key in value}
    unknown = sorted(keys - fields)
    missing = sorted(fields - keys)
    if unknown:
        raise ASTRevisionError(f"{label} contains unknown fields: {', '.join(unknown)}")
    if missing:
        raise ASTRevisionError(f"{label} is missing fields: {', '.join(missing)}")
    return value


def _items(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ASTRevisionError(f"{label} must be a list")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ASTRevisionError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _names(value: Any, label: str) -> Tuple[str, ...]:
    raw = _items(value, label)
    names = tuple(str(item).strip() for item in raw)
    if any(not item for item in names):
        raise ASTRevisionError(f"{label} must contain non-empty strings")
    if len(names) != len(set(names)):
        raise ASTRevisionError(f"{label} contains duplicates")
    return names


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ASTRevisionError(f"AST revision value is not canonical JSON: {error}") from error


def _digest(domain: str, value: Any) -> str:
    payload = f"{domain}\0{_canonical_json(value)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _selector(chain_id: Any, spans: Any, label: str) -> ResidueSelector:
    try:
        return ResidueSelector.from_mapping(
            {
                "schema_version": "astevolve.residue_selector.v1",
                "chain_id": chain_id,
                "spans": spans,
            }
        )
    except (TypeError, ValueError) as error:
        raise ASTRevisionError(f"{label} is invalid: {error}") from error


def _positions(selector: ResidueSelector) -> Tuple[int, ...]:
    return tuple(
        position
        for start, end in selector.spans
        for position in range(int(start), int(end))
    )


def _node_positions(ast: ExecutableDualAST) -> Dict[str, Tuple[int, ...]]:
    return {
        node.node_id: _positions(node.selector)
        for node in ast.structural_nodes
        if node.kind == "editable"
    }


def _normalize_policy(raw: Any) -> Dict[str, Any]:
    policy = _closed_mapping(raw, _POLICY_FIELDS, "ast_evolution_policy")
    if policy.get("schema_version") != AST_EVOLUTION_POLICY_VERSION:
        raise ASTRevisionError(
            "unsupported ast_evolution_policy schema_version: "
            f"{policy.get('schema_version')!r}"
        )
    if not isinstance(policy.get("enabled"), bool):
        raise ASTRevisionError("ast_evolution_policy.enabled must be boolean")
    normalized: Dict[str, Any] = {
        "schema_version": AST_EVOLUTION_POLICY_VERSION,
        "enabled": bool(policy["enabled"]),
    }
    for field in (
        "max_structural_node_edits",
        "max_mapping_edge_edits",
        "max_added_positions",
        "max_removed_positions",
        "max_total_editable_positions",
    ):
        normalized[field] = _integer(policy[field], f"ast_evolution_policy.{field}")

    raw_envelopes = policy.get("structural_node_envelopes")
    if not isinstance(raw_envelopes, Mapping):
        raise ASTRevisionError(
            "ast_evolution_policy.structural_node_envelopes must be a mapping"
        )
    envelopes: Dict[str, Any] = {}
    for raw_node_id, raw_envelope in raw_envelopes.items():
        node_id = str(raw_node_id).strip()
        if not node_id or node_id in envelopes:
            raise ASTRevisionError("structural node envelope IDs must be unique and non-empty")
        envelope = _closed_mapping(
            raw_envelope,
            _ENVELOPE_FIELDS,
            f"ast_evolution_policy.structural_node_envelopes.{node_id}",
        )
        selector = _selector(
            envelope["chain_id"],
            envelope["allowed_spans"],
            f"structural node envelope {node_id!r}",
        )
        minimum = _integer(envelope["min_positions"], f"{node_id}.min_positions", minimum=1)
        maximum = _integer(envelope["max_positions"], f"{node_id}.max_positions", minimum=1)
        capacity = len(_positions(selector))
        if minimum > maximum or maximum > capacity:
            raise ASTRevisionError(
                f"structural node envelope {node_id!r} has invalid position bounds "
                f"[{minimum}, {maximum}] for capacity {capacity}"
            )
        envelopes[node_id] = {
            "chain_id": selector.chain_id,
            "allowed_spans": [list(span) for span in selector.spans],
            "allowed_positions": list(_positions(selector)),
            "min_positions": minimum,
            "max_positions": maximum,
        }
    normalized["structural_node_envelopes"] = envelopes

    raw_allowlist = policy.get("mapping_edge_allowlist")
    if not isinstance(raw_allowlist, Mapping):
        raise ASTRevisionError(
            "ast_evolution_policy.mapping_edge_allowlist must be a mapping"
        )
    allowlist: Dict[str, Any] = {}
    for raw_edge_id, raw_rule in raw_allowlist.items():
        edge_id = str(raw_edge_id).strip()
        if not edge_id or edge_id in allowlist:
            raise ASTRevisionError("mapping edge allowlist IDs must be unique and non-empty")
        rule = _closed_mapping(
            raw_rule,
            _MAPPING_ALLOWLIST_FIELDS,
            f"ast_evolution_policy.mapping_edge_allowlist.{edge_id}",
        )
        if not isinstance(rule.get("allow_disable"), bool):
            raise ASTRevisionError(f"mapping edge allowlist {edge_id!r} allow_disable must be boolean")
        allowlist[edge_id] = {
            "structural_node_ids": list(
                _names(rule["structural_node_ids"], f"{edge_id}.structural_node_ids")
            ),
            "action_operators": list(
                _names(rule["action_operators"], f"{edge_id}.action_operators")
            ),
            "allow_disable": bool(rule["allow_disable"]),
        }
    normalized["mapping_edge_allowlist"] = allowlist
    return normalized


def _normalize_plan(raw: Any) -> Dict[str, Any]:
    plan = _closed_mapping(raw, _PLAN_FIELDS, "ast_revision_plan")
    if plan.get("schema_version") != AST_REVISION_PLAN_VERSION:
        raise ASTRevisionError(
            f"unsupported ast_revision_plan schema_version: {plan.get('schema_version')!r}"
        )
    rationale = plan.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ASTRevisionError("ast_revision_plan.rationale must be a non-empty string")
    if len(rationale) > 4000:
        raise ASTRevisionError("ast_revision_plan.rationale exceeds 4000 characters")
    return {
        "schema_version": AST_REVISION_PLAN_VERSION,
        "structural_node_edits": list(
            _items(plan["structural_node_edits"], "ast_revision_plan.structural_node_edits")
        ),
        "mapping_edge_edits": list(
            _items(plan["mapping_edge_edits"], "ast_revision_plan.mapping_edge_edits")
        ),
        "rationale": rationale.strip(),
    }


def _text(value: Any, label: str, *, maximum: int = 4000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ASTRevisionError(f"{label} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ASTRevisionError(f"{label} exceeds {maximum} characters")
    return normalized


def _string_list(value: Any, label: str, *, allow_empty: bool = True) -> list[str]:
    items = list(_items(value, label))
    normalized = [_text(item, f"{label}[{index}]", maximum=512) for index, item in enumerate(items)]
    if not allow_empty and not normalized:
        raise ASTRevisionError(f"{label} must not be empty")
    if len(normalized) != len(set(normalized)):
        raise ASTRevisionError(f"{label} contains duplicates")
    return normalized


def _span_policy(value: Any, label: str) -> Dict[str, list[int]]:
    if not isinstance(value, Mapping):
        raise ASTRevisionError(f"{label} must be a mapping")
    normalized: Dict[str, list[int]] = {}
    for raw_chain_id, raw_spans in value.items():
        chain_id = _text(raw_chain_id, f"{label} chain", maximum=64)
        selector = _selector(chain_id, raw_spans, f"{label}.{chain_id}")
        normalized[chain_id] = list(_positions(selector))
    return normalized


def _amino_acid_list(value: Any, label: str) -> list[str]:
    residues = _string_list(value, label)
    invalid = sorted(set(residues) - _AMINO_ACIDS)
    if invalid:
        raise ASTRevisionError(
            f"{label} contains non-canonical amino acids: {', '.join(invalid)}"
        )
    return residues


def _policy_weight(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ASTRevisionError(f"{label} must be numeric")
    normalized = float(value)
    if not 0.0 < normalized <= 10.0:
        raise ASTRevisionError(f"{label} must be in (0, 10]")
    return normalized


def _normalize_residue_policy(
    value: Any,
    *,
    label: str,
    selector_positions: set[int],
) -> Dict[str, Any]:
    raw = _closed_mapping(value, _RESIDUE_POLICY_FIELDS, label)
    favored = _amino_acid_list(raw["favored_residues"], f"{label}.favored_residues")
    disfavored = _amino_acid_list(
        raw["disfavored_residues"], f"{label}.disfavored_residues"
    )
    overlap = sorted(set(favored) & set(disfavored))
    if overlap:
        raise ASTRevisionError(
            f"{label} favors and disfavors the same residues: {', '.join(overlap)}"
        )
    raw_rules = raw["position_residue_rules"]
    if not isinstance(raw_rules, Mapping):
        raise ASTRevisionError(f"{label}.position_residue_rules must be a mapping")
    rules: Dict[str, Any] = {}
    for raw_position, raw_rule in raw_rules.items():
        try:
            position = int(raw_position)
        except (TypeError, ValueError) as error:
            raise ASTRevisionError(
                f"{label}.position_residue_rules keys must be integer positions"
            ) from error
        if str(position) in rules:
            raise ASTRevisionError(
                f"{label}.position_residue_rules repeats position {position}"
            )
        if position not in selector_positions:
            raise ASTRevisionError(
                f"{label}.position_residue_rules position {position} is outside the node selector"
            )
        rule = _closed_mapping(
            raw_rule,
            _POSITION_RESIDUE_RULE_FIELDS,
            f"{label}.position_residue_rules.{position}",
        )
        rule_favored = _amino_acid_list(
            rule["favored_residues"],
            f"{label}.position_residue_rules.{position}.favored_residues",
        )
        rule_disfavored = _amino_acid_list(
            rule["disfavored_residues"],
            f"{label}.position_residue_rules.{position}.disfavored_residues",
        )
        rule_overlap = sorted(set(rule_favored) & set(rule_disfavored))
        if rule_overlap:
            raise ASTRevisionError(
                f"{label}.position_residue_rules.{position} favors and disfavors "
                f"the same residues: {', '.join(rule_overlap)}"
            )
        rules[str(position)] = {
            "favored_residues": rule_favored,
            "disfavored_residues": rule_disfavored,
            "policy_weight": _policy_weight(
                rule["policy_weight"],
                f"{label}.position_residue_rules.{position}.policy_weight",
            ),
            "intent": _text(
                rule["intent"],
                f"{label}.position_residue_rules.{position}.intent",
                maximum=1000,
            ),
        }
    return {
        "favored_residues": favored,
        "disfavored_residues": disfavored,
        "position_residue_rules": rules,
        "policy_weight": _policy_weight(raw["policy_weight"], f"{label}.policy_weight"),
    }


def _normalize_action_profiles(value: Any) -> Dict[str, list[Dict[str, Any]]]:
    if not isinstance(value, Mapping) or not value:
        raise ASTRevisionError("global AST action_profiles must be a non-empty mapping")
    profiles: Dict[str, list[Dict[str, Any]]] = {}
    for raw_name, raw_profile in value.items():
        name = _text(raw_name, "action profile name", maximum=128)
        profile = _closed_mapping(
            raw_profile,
            _ACTION_PROFILE_FIELDS,
            f"action_profiles.{name}",
        )
        raw_actions = list(
            _items(profile["allowed_actions"], f"action_profiles.{name}.allowed_actions")
        )
        if not raw_actions:
            raise ASTRevisionError(f"action profile {name!r} has no allowed actions")
        actions: list[Dict[str, Any]] = []
        operators: set[str] = set()
        for index, raw_action in enumerate(raw_actions):
            action = _closed_mapping(
                raw_action,
                _ACTION_FIELDS,
                f"action_profiles.{name}.allowed_actions[{index}]",
            )
            operator = _text(
                action["operator"],
                f"action_profiles.{name}.allowed_actions[{index}].operator",
                maximum=64,
            )
            if operator in operators:
                raise ASTRevisionError(f"action profile {name!r} repeats operator {operator!r}")
            operators.add(operator)
            budget = _closed_mapping(
                action["budget"],
                _BUDGET_FIELDS,
                f"action_profiles.{name}.allowed_actions[{index}].budget",
            )
            minimum = _integer(
                budget["min"],
                f"action_profiles.{name}.allowed_actions[{index}].budget.min",
                minimum=1,
            )
            maximum = _integer(
                budget["max"],
                f"action_profiles.{name}.allowed_actions[{index}].budget.max",
                minimum=1,
            )
            if minimum > maximum:
                raise ASTRevisionError(
                    f"action profile {name!r} budget min cannot exceed max"
                )
            actions.append(
                {
                    "schema_version": "astevolve.allowed_action.v1",
                    "operator": operator,
                    "budget": {"min": minimum, "max": maximum},
                }
            )
        profiles[name] = actions
    return profiles


def _normalize_global_policy(raw: Any) -> Dict[str, Any]:
    normalized_raw = deepcopy(dict(raw)) if isinstance(raw, Mapping) else raw
    if isinstance(normalized_raw, dict):
        normalized_raw.setdefault("bootstrap_required_spans", {})
        normalized_raw.setdefault("bootstrap_release_callable", "")
    policy = _closed_mapping(
        normalized_raw,
        _GLOBAL_POLICY_FIELDS,
        "global_ast_evolution_policy",
    )
    if policy.get("schema_version") != GLOBAL_AST_EVOLUTION_POLICY_VERSION:
        raise ASTRevisionError(
            "unsupported global_ast_evolution_policy schema_version: "
            f"{policy.get('schema_version')!r}"
        )
    if not isinstance(policy.get("enabled"), bool):
        raise ASTRevisionError("global_ast_evolution_policy.enabled must be boolean")
    minimum_positions = _integer(
        policy["min_total_editable_positions"],
        "global_ast_evolution_policy.min_total_editable_positions",
        minimum=1,
    )
    maximum_positions = _integer(
        policy["max_total_editable_positions"],
        "global_ast_evolution_policy.max_total_editable_positions",
        minimum=1,
    )
    minimum_nodes = _integer(
        policy["min_active_structural_nodes"],
        "global_ast_evolution_policy.min_active_structural_nodes",
        minimum=1,
    )
    maximum_nodes = _integer(
        policy["max_active_structural_nodes"],
        "global_ast_evolution_policy.max_active_structural_nodes",
        minimum=1,
    )
    if minimum_positions > maximum_positions:
        raise ASTRevisionError("global AST minimum editable positions exceeds maximum")
    if minimum_nodes > maximum_nodes:
        raise ASTRevisionError("global AST minimum active nodes exceeds maximum")
    safety_score = policy["min_position_safety_score"]
    if isinstance(safety_score, bool) or not isinstance(safety_score, (int, float)):
        raise ASTRevisionError("min_position_safety_score must be numeric")
    safety_score = float(safety_score)
    if not 0.0 <= safety_score <= 1.0:
        raise ASTRevisionError("min_position_safety_score must be in [0, 1]")
    for field in ("require_catalog_eligibility", "require_position_evidence_refs"):
        if not isinstance(policy.get(field), bool):
            raise ASTRevisionError(f"global_ast_evolution_policy.{field} must be boolean")
    node_pattern = _text(policy["node_id_pattern"], "node_id_pattern", maximum=256)
    edge_pattern = _text(policy["edge_id_pattern"], "edge_id_pattern", maximum=256)
    try:
        re.compile(node_pattern)
        re.compile(edge_pattern)
    except re.error as error:
        raise ASTRevisionError(f"invalid global AST ID pattern: {error}") from error
    allowed = _span_policy(
        policy["allowed_chain_spans"], "global_ast_evolution_policy.allowed_chain_spans"
    )
    if not allowed:
        raise ASTRevisionError(
            "global_ast_evolution_policy.allowed_chain_spans must be non-empty"
        )
    protected = _span_policy(
        policy["protected_chain_spans"],
        "global_ast_evolution_policy.protected_chain_spans",
    )
    unknown_protected = sorted(set(protected) - set(allowed))
    if unknown_protected:
        raise ASTRevisionError(
            "protected_chain_spans contains chain(s) absent from allowed_chain_spans: "
            + ", ".join(unknown_protected)
        )
    bootstrap_required = _span_policy(
        policy.get("bootstrap_required_spans") or {},
        "global_ast_evolution_policy.bootstrap_required_spans",
    )
    bootstrap_release_callable = str(
        policy.get("bootstrap_release_callable") or ""
    ).strip()
    if bootstrap_required and ":" not in bootstrap_release_callable:
        raise ASTRevisionError(
            "bootstrap_required_spans requires bootstrap_release_callable in "
            "module:function form"
        )
    return {
        "schema_version": GLOBAL_AST_EVOLUTION_POLICY_VERSION,
        "enabled": bool(policy["enabled"]),
        "allowed_positions": allowed,
        "protected_positions": protected,
        "min_total_editable_positions": minimum_positions,
        "max_total_editable_positions": maximum_positions,
        "min_active_structural_nodes": minimum_nodes,
        "max_active_structural_nodes": maximum_nodes,
        "max_positions_per_node": _integer(
            policy["max_positions_per_node"],
            "global_ast_evolution_policy.max_positions_per_node",
            minimum=1,
        ),
        "max_spans_per_node": _integer(
            policy["max_spans_per_node"],
            "global_ast_evolution_policy.max_spans_per_node",
            minimum=1,
        ),
        "max_mapping_edges": _integer(
            policy["max_mapping_edges"],
            "global_ast_evolution_policy.max_mapping_edges",
            minimum=1,
        ),
        "immutable_structural_node_ids": _string_list(
            policy["immutable_structural_node_ids"],
            "global_ast_evolution_policy.immutable_structural_node_ids",
        ),
        "allowed_functional_node_ids": _string_list(
            policy["allowed_functional_node_ids"],
            "global_ast_evolution_policy.allowed_functional_node_ids",
            allow_empty=False,
        ),
        "action_profiles": _normalize_action_profiles(policy["action_profiles"]),
        "require_catalog_eligibility": bool(policy["require_catalog_eligibility"]),
        "require_position_evidence_refs": bool(
            policy["require_position_evidence_refs"]
        ),
        "min_position_safety_score": safety_score,
        "node_id_pattern": node_pattern,
        "edge_id_pattern": edge_pattern,
        "bootstrap_required_positions": bootstrap_required,
        "bootstrap_release_callable": bootstrap_release_callable,
    }


def _normalize_global_decision(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ASTRevisionError("decision_record must be a mapping")
    keys = {str(key) for key in raw}
    unknown = sorted(keys - _GLOBAL_DECISION_FIELDS)
    missing = sorted(_GLOBAL_DECISION_REQUIRED_FIELDS - keys)
    if unknown:
        raise ASTRevisionError(
            "decision_record contains unknown fields: " + ", ".join(unknown)
        )
    if missing:
        raise ASTRevisionError(
            "decision_record is missing fields: " + ", ".join(missing)
        )
    decision = raw
    action = _text(decision["action"], "decision_record.action", maximum=32).lower()
    if action not in _GLOBAL_DECISION_ACTIONS:
        raise ASTRevisionError(
            "decision_record.action must be one of "
            + ", ".join(sorted(_GLOBAL_DECISION_ACTIONS))
        )
    confidence = decision["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ASTRevisionError("decision_record.confidence must be numeric")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ASTRevisionError("decision_record.confidence must be in [0, 1]")
    expected = _string_list(
        decision["expected_effects"], "decision_record.expected_effects", allow_empty=False
    )
    normalized = {
        "action": action,
        "diagnosis": _text(decision["diagnosis"], "decision_record.diagnosis"),
        "hypothesis": _text(decision["hypothesis"], "decision_record.hypothesis"),
        "evidence_refs": _string_list(
            decision["evidence_refs"], "decision_record.evidence_refs", allow_empty=False
        ),
        "expected_effects": expected,
        "failure_condition": _text(
            decision["failure_condition"], "decision_record.failure_condition"
        ),
        "confidence": confidence,
    }
    if "rationale" in decision:
        normalized["rationale"] = _text(
            decision["rationale"], "decision_record.rationale", maximum=1500
        )
    if "rollback_condition" in decision:
        normalized["rollback_condition"] = _text(
            decision["rollback_condition"],
            "decision_record.rollback_condition",
            maximum=1500,
        )
    return normalized


def _normalize_global_plan(raw: Any) -> Dict[str, Any]:
    plan = _closed_mapping(raw, _GLOBAL_PLAN_FIELDS, "ast_revision_plan")
    if plan.get("schema_version") != GLOBAL_AST_REVISION_PLAN_VERSION:
        raise ASTRevisionError(
            f"unsupported global ast_revision_plan schema_version: {plan.get('schema_version')!r}"
        )
    return {
        "schema_version": GLOBAL_AST_REVISION_PLAN_VERSION,
        "structural_nodes": list(
            _items(plan["structural_nodes"], "ast_revision_plan.structural_nodes")
        ),
        "mapping_edges": list(
            _items(plan["mapping_edges"], "ast_revision_plan.mapping_edges")
        ),
        "decision_record": _normalize_global_decision(plan["decision_record"]),
    }


def normalize_global_ast_revision_plan(raw: Any) -> Dict[str, Any]:


    return deepcopy(_normalize_global_plan(raw))


def _catalog_records(
    state: Mapping[str, Any],
) -> tuple[
    Dict[str, Dict[str, Any]],
    Dict[tuple[str, int], Dict[str, Any]],
    str | None,
]:
    raw_catalog = state.get("_residue_evidence_catalog") or state.get(
        "residue_evidence_catalog"
    )
    if not isinstance(raw_catalog, Mapping):
        return {}, {}, None
    records: Dict[str, Dict[str, Any]] = {}
    positions: Dict[tuple[str, int], Dict[str, Any]] = {}
    for raw in raw_catalog.get("residues", []) or []:
        if not isinstance(raw, Mapping):
            continue
        evidence_id = str(raw.get("evidence_id") or "").strip()
        if evidence_id:
            records[evidence_id] = dict(raw)
        chain_id = str(raw.get("chain_id") or "").strip()
        position = raw.get("position")
        if chain_id and isinstance(position, int) and not isinstance(position, bool):
            positions[(chain_id, int(position))] = dict(raw)
    for raw in raw_catalog.get("evidence", []) or []:
        if not isinstance(raw, Mapping):
            continue
        evidence_id = str(raw.get("evidence_id") or "").strip()
        if evidence_id:
            records[evidence_id] = dict(raw)
    catalog_hash = str(raw_catalog.get("catalog_hash") or "").strip()
    if not catalog_hash:
        catalog_hash = _digest(
            "astevolve.residue_evidence_catalog.v1",
            {
                key: value
                for key, value in raw_catalog.items()
                if str(key) != "catalog_hash"
            },
        )
    return records, positions, catalog_hash


def _global_policy_report(policy: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in policy.items()
        if key
        not in {
            "allowed_positions",
            "protected_positions",
            "action_profiles",
        }
    } | {
        "allowed_positions": deepcopy(dict(policy.get("allowed_positions", {}))),
        "protected_positions": deepcopy(dict(policy.get("protected_positions", {}))),
        "action_profiles": deepcopy(dict(policy.get("action_profiles", {}))),
    }


def _report(
    *,
    policy: Mapping[str, Any] | None,
    plan: Mapping[str, Any] | None,
    before: ExecutableDualAST | None,
    after: ExecutableDualAST | None,
    applied: bool,
    added: Sequence[Tuple[str, int]] = (),
    removed: Sequence[Tuple[str, int]] = (),
    mapping_changes: Sequence[Mapping[str, Any]] = (),
    report_version: str = AST_REVISION_REPORT_VERSION,
    policy_version: str = AST_EVOLUTION_POLICY_VERSION,
    plan_version: str = AST_REVISION_PLAN_VERSION,
    policy_limits: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    before_value = before.to_dict() if before is not None else None
    after_value = after.to_dict() if after is not None else None
    return {
        "schema_version": report_version,
        "enabled": bool(policy and policy.get("enabled")),
        "applied": bool(applied),
        "base_ast_hash": _digest("astevolve.executable_dual_ast.v1", before_value),
        "effective_ast_hash": _digest("astevolve.executable_dual_ast.v1", after_value),
        "policy_hash": _digest(policy_version, policy) if policy else None,
        "policy_limits": (
            deepcopy(dict(policy_limits))
            if policy_limits is not None
            else {
                "max_structural_node_edits": policy.get("max_structural_node_edits"),
                "max_mapping_edge_edits": policy.get("max_mapping_edge_edits"),
                "max_added_positions": policy.get("max_added_positions"),
                "max_removed_positions": policy.get("max_removed_positions"),
                "max_total_editable_positions": policy.get(
                    "max_total_editable_positions"
                ),
                "structural_node_envelopes": deepcopy(
                    dict(policy.get("structural_node_envelopes", {}))
                ),
                "mapping_edge_allowlist": deepcopy(
                    dict(policy.get("mapping_edge_allowlist", {}))
                ),
            }
            if policy
            else {}
        ),
        "plan_hash": _digest(plan_version, plan) if plan else None,
        "base_revision": before.revision if before is not None else None,
        "effective_revision": after.revision if after is not None else None,
        "rationale": str((plan or {}).get("rationale") or ""),
        "base_positions_by_node": {
            node: list(positions) for node, positions in _node_positions(before).items()
        } if before is not None else {},
        "effective_positions_by_node": {
            node: list(positions) for node, positions in _node_positions(after).items()
        } if after is not None else {},
        "added_positions": [
            {"node_id": node, "position": int(position)} for node, position in added
        ],
        "removed_positions": [
            {"node_id": node, "position": int(position)} for node, position in removed
        ],
        "mapping_edge_changes": [deepcopy(dict(item)) for item in mapping_changes],
    }


def _catalog_position_is_eligible(record: Mapping[str, Any]) -> bool:
    for field in ("editable_eligible", "eligible_for_edit", "eligible"):
        if field in record:
            return record[field] is True
    safety = record.get("safety")
    if isinstance(safety, Mapping):
        for field in ("editable_eligible", "eligible_for_edit", "eligible", "approved"):
            if field in safety:
                return safety[field] is True
    return False


def _catalog_position_safety_score(record: Mapping[str, Any]) -> float | None:
    value = record.get("safety_score")
    if value is None and isinstance(record.get("safety"), Mapping):
        value = record["safety"].get("score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _catalog_position_compiled_segment(
    state: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    chain_id: str,
    position: int,
) -> str | None:
    raw_map = state.get("_compiled_segment_by_position")
    if isinstance(raw_map, Mapping):
        value = str(raw_map.get(f"{chain_id}:{position}") or "").strip()
        if value:
            return value
    for container_name in ("structure", "metadata"):
        container = record.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for field in ("compiled_segment", "structural_node", "segment_id"):
            value = str(container.get(field) or "").strip()
            if value:
                return value
    return None


def _global_node_change_records(
    before: ExecutableDualAST,
    after: ExecutableDualAST,
) -> list[Dict[str, Any]]:
    before_nodes = {
        node.node_id: node
        for node in before.structural_nodes
        if node.kind == "editable"
    }
    after_nodes = {
        node.node_id: node
        for node in after.structural_nodes
        if node.kind == "editable"
    }
    records: list[Dict[str, Any]] = []
    for node_id in sorted(set(before_nodes) | set(after_nodes)):
        old = before_nodes.get(node_id)
        new = after_nodes.get(node_id)
        if old is None and new is not None:
            change = "created"
        elif old is not None and new is None:
            change = "deleted"
        else:
            assert old is not None and new is not None
            old_positions = set(_positions(old.selector))
            new_positions = set(_positions(new.selector))
            if old.to_dict() == new.to_dict():
                continue
            if old.selector.chain_id != new.selector.chain_id or not (
                old_positions & new_positions
            ):
                change = "migrated"
            elif len(old_positions) != len(new_positions):
                change = "resized"
            elif old_positions != new_positions:
                change = "shifted"
            else:
                change = "action_profile_changed"
        records.append(
            {
                "node_id": node_id,
                "change": change,
                "before": old.to_dict() if old is not None else None,
                "after": new.to_dict() if new is not None else None,
            }
        )
    return records


def _global_mapping_change_records(
    before: ExecutableDualAST,
    after: ExecutableDualAST,
) -> list[Dict[str, Any]]:
    before_edges = {edge.edge_id: edge.to_dict() for edge in before.mapping_edges}
    after_edges = {edge.edge_id: edge.to_dict() for edge in after.mapping_edges}
    return [
        {
            "edge_id": edge_id,
            "before": deepcopy(before_edges.get(edge_id)),
            "after": deepcopy(after_edges.get(edge_id)),
        }
        for edge_id in sorted(set(before_edges) | set(after_edges))
        if before_edges.get(edge_id) != after_edges.get(edge_id)
    ]


def _apply_global_ast_revision_plan(
    updated: Dict[str, Any],
    strategy: Mapping[str, Any],
    before: ExecutableDualAST,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    raw_policy = updated.get("global_ast_evolution_policy")
    if raw_policy is None:
        raise ASTRevisionError(
            "global ast_revision_plan requires global_ast_evolution_policy"
        )
    policy = _normalize_global_policy(raw_policy)
    raw_plan = strategy.get("ast_revision_plan")
    if raw_plan is None:
        report = _report(
            policy=policy,
            plan=None,
            before=before,
            after=before,
            applied=False,
            report_version=GLOBAL_AST_REVISION_REPORT_VERSION,
            policy_version=GLOBAL_AST_EVOLUTION_POLICY_VERSION,
            plan_version=GLOBAL_AST_REVISION_PLAN_VERSION,
            policy_limits=_global_policy_report(policy),
        )
        report.update(
            {
                "decision_record": None,
                "node_changes": [],
                "catalog_hash": None,
                "approved_positions": [],
                "node_metadata": {},
            }
        )
        return updated, report
    if not policy["enabled"]:
        raise ASTRevisionError(
            "global ast_revision_plan is present but global AST evolution is disabled"
        )
    plan = _normalize_global_plan(raw_plan)

    base_structural = {node.node_id: node for node in before.structural_nodes}
    missing_immutable = sorted(
        set(policy["immutable_structural_node_ids"]) - set(base_structural)
    )
    if missing_immutable:
        raise ASTRevisionError(
            "global AST policy references unknown immutable structural nodes: "
            + ", ".join(missing_immutable)
        )
    immutable_ids = set(policy["immutable_structural_node_ids"]) | {
        node.node_id for node in before.structural_nodes if node.kind == "frozen"
    }
    immutable_nodes = [
        node for node in before.structural_nodes if node.node_id in immutable_ids
    ]
    base_functional_ids = {node.node_id for node in before.functional_nodes}
    allowed_functional_ids = set(policy["allowed_functional_node_ids"])
    unknown_functional = sorted(allowed_functional_ids - base_functional_ids)
    if unknown_functional:
        raise ASTRevisionError(
            "global AST policy references unknown functional nodes: "
            + ", ".join(unknown_functional)
        )
    if allowed_functional_ids != base_functional_ids:
        missing = sorted(base_functional_ids - allowed_functional_ids)
        raise ASTRevisionError(
            "global AST policy must preserve every case-owned functional node; missing: "
            + ", ".join(missing)
        )

    raw_nodes = plan["structural_nodes"]
    if not policy["min_active_structural_nodes"] <= len(raw_nodes) <= policy[
        "max_active_structural_nodes"
    ]:
        raise ASTRevisionError(
            "global ast_revision_plan structural node count is outside policy bounds"
        )
    catalog_by_id, catalog_by_position, catalog_hash = _catalog_records(updated)
    if policy["require_catalog_eligibility"] and not catalog_by_position:
        raise ASTRevisionError(
            "global AST policy requires a loaded residue evidence catalog"
        )
    node_pattern = re.compile(policy["node_id_pattern"])
    desired_nodes: list[StructuralNodeSpec] = []
    node_metadata: Dict[str, Any] = {}
    selected_owner: Dict[tuple[str, int], str] = {}
    seen_node_ids: set[str] = set()
    approved_positions: list[Dict[str, Any]] = []
    for index, raw_node in enumerate(raw_nodes):
        label = f"ast_revision_plan.structural_nodes[{index}]"
        node = _closed_mapping(raw_node, _GLOBAL_NODE_FIELDS, label)
        node_id = _text(node["node_id"], f"{label}.node_id", maximum=128)
        if node_id in seen_node_ids:
            raise ASTRevisionError(f"global AST repeats structural node ID {node_id!r}")
        if node_id in immutable_ids:
            raise ASTRevisionError(
                f"global AST structural node {node_id!r} collides with an immutable node"
            )
        if node_pattern.fullmatch(node_id) is None:
            raise ASTRevisionError(
                f"global AST structural node ID {node_id!r} violates "
                f"node_id_pattern {policy['node_id_pattern']!r}"
            )
        seen_node_ids.add(node_id)
        try:
            selector = ResidueSelector.from_mapping(node["selector"])
        except (TypeError, ValueError) as error:
            raise ASTRevisionError(f"{label}.selector is invalid: {error}") from error
        if len(selector.spans) > policy["max_spans_per_node"]:
            raise ASTRevisionError(f"global AST node {node_id!r} exceeds max_spans_per_node")
        positions = _positions(selector)
        if len(positions) > policy["max_positions_per_node"]:
            raise ASTRevisionError(
                f"global AST node {node_id!r} exceeds max_positions_per_node"
            )
        allowed_positions = set(policy["allowed_positions"].get(selector.chain_id, ()))
        if not allowed_positions:
            raise ASTRevisionError(
                f"global AST node {node_id!r} selects disallowed chain {selector.chain_id!r}"
            )
        escaped = sorted(set(positions) - allowed_positions)
        if escaped:
            raise ASTRevisionError(
                f"global AST node {node_id!r} escapes allowed_chain_spans at {escaped}"
            )
        protected = set(policy["protected_positions"].get(selector.chain_id, ()))
        protected_hits = sorted(set(positions) & protected)
        if protected_hits:
            raise ASTRevisionError(
                f"global AST node {node_id!r} selects protected residues {protected_hits}"
            )
        overlap = sorted(
            position
            for position in positions
            if (selector.chain_id, position) in selected_owner
        )
        if overlap:
            raise ASTRevisionError(
                f"global AST node {node_id!r} overlaps another editable node at {overlap}"
            )
        action_profile = _text(
            node["action_profile"], f"{label}.action_profile", maximum=128
        )
        allowed_actions = policy["action_profiles"].get(action_profile)
        if allowed_actions is None:
            raise ASTRevisionError(
                f"global AST node {node_id!r} uses unknown action profile {action_profile!r}"
            )
        evidence_refs = _string_list(
            node["evidence_refs"], f"{label}.evidence_refs", allow_empty=False
        )
        unknown_refs = sorted(set(evidence_refs) - set(catalog_by_id))
        if unknown_refs:
            raise ASTRevisionError(
                f"global AST node {node_id!r} references unknown evidence IDs: "
                + ", ".join(unknown_refs)
            )
        compiled_segments: set[str] = set()
        for position in positions:
            key = (selector.chain_id, position)
            record = catalog_by_position.get(key)
            if policy["require_catalog_eligibility"]:
                if record is None:
                    raise ASTRevisionError(
                        f"global AST node {node_id!r} has no residue evidence for "
                        f"{selector.chain_id}:{position}"
                    )
                if not _catalog_position_is_eligible(record):
                    raise ASTRevisionError(
                        f"global AST node {node_id!r} selects ineligible residue "
                        f"{selector.chain_id}:{position}"
                    )
                score = _catalog_position_safety_score(record)
                if score is None or score < policy["min_position_safety_score"]:
                    raise ASTRevisionError(
                        f"global AST node {node_id!r} selects residue "
                        f"{selector.chain_id}:{position} below min_position_safety_score"
                    )
            if policy["require_position_evidence_refs"]:
                evidence_id = str((record or {}).get("evidence_id") or "")
                if not evidence_id or evidence_id not in evidence_refs:
                    raise ASTRevisionError(
                        f"global AST node {node_id!r} omits position evidence ref "
                        f"for {selector.chain_id}:{position}"
                    )
            compiled_segment = _catalog_position_compiled_segment(
                updated,
                record or {},
                chain_id=selector.chain_id,
                position=position,
            )
            if compiled_segment:
                compiled_segments.add(compiled_segment)
            selected_owner[key] = node_id
            approved_positions.append(
                {
                    "node_id": node_id,
                    "chain_id": selector.chain_id,
                    "position": position,
                    "evidence_id": str((record or {}).get("evidence_id") or "") or None,
                    "safety_score": _catalog_position_safety_score(record or {}),
                    "compiled_segment": compiled_segment,
                }
            )
        if len(compiled_segments) > 1:
            raise ASTRevisionError(
                f"global AST node {node_id!r} crosses compiled segments: "
                + ", ".join(sorted(compiled_segments))
                + "; create separate nodes for separate compiled segments"
            )
        residue_policy = _normalize_residue_policy(
            node["residue_policy"],
            label=f"{label}.residue_policy",
            selector_positions=set(positions),
        )
        desired_nodes.append(
            StructuralNodeSpec.from_mapping(
                {
                    "schema_version": "astevolve.structural_node.v1",
                    "node_id": node_id,
                    "kind": "editable",
                    "selector": selector.to_dict(),
                    "invariants": [],
                    "allowed_actions": deepcopy(allowed_actions),
                }
            )
        )
        node_metadata[node_id] = {
            "action_profile": action_profile,
            "intent": _text(node["intent"], f"{label}.intent"),
            "evidence_refs": evidence_refs,
            "residue_policy": residue_policy,
        }
        if compiled_segments:
            node_metadata[node_id]["compiled_segment"] = next(
                iter(compiled_segments)
            )

    total_positions = len(selected_owner)
    required_bootstrap_positions = policy.get("bootstrap_required_positions")
    bootstrap_release_callable = str(
        policy.get("bootstrap_release_callable") or ""
    )
    parent_sequences = updated.get("_current_parent_sequences")
    bootstrap_released = False
    if isinstance(parent_sequences, Mapping) and bootstrap_release_callable:
        try:
            module_name, function_name = bootstrap_release_callable.split(":", 1)
            release_function = getattr(
                importlib.import_module(module_name), function_name
            )
            release_result = release_function(
                str(parent_sequences.get("A") or "")
            )
            bootstrap_released = bool(
                isinstance(release_result, Mapping)
                and release_result.get("pass")
            )
        except Exception as error:
            raise ASTRevisionError(
                "bootstrap release gate failed: "
                f"{type(error).__name__}: {error}"
            ) from error
    if isinstance(required_bootstrap_positions, Mapping) and not bootstrap_released:
        missing_required: Dict[str, list[int]] = {}
        for chain_id, required_positions in required_bootstrap_positions.items():
            selected_positions = {
                position
                for selected_chain, position in selected_owner
                if selected_chain == str(chain_id)
            }
            missing = sorted(set(required_positions) - selected_positions)
            if missing:
                missing_required[str(chain_id)] = missing
        if missing_required:
            raise ASTRevisionError(
                "global ast_revision_plan cannot shrink bootstrap-required spans: "
                + repr(missing_required)
            )
    if not policy["min_total_editable_positions"] <= total_positions <= policy[
        "max_total_editable_positions"
    ]:
        raise ASTRevisionError(
            f"global ast_revision_plan selects {total_positions} total positions; allowed "
            f"[{policy['min_total_editable_positions']}, "
            f"{policy['max_total_editable_positions']}]"
        )

    raw_edges = plan["mapping_edges"]
    if len(raw_edges) > policy["max_mapping_edges"]:
        raise ASTRevisionError("global ast_revision_plan exceeds max_mapping_edges")
    edge_pattern = re.compile(policy["edge_id_pattern"])
    desired_by_id = {node.node_id: node for node in desired_nodes}
    desired_edges: list[MappingEdgeSpec] = []
    seen_edge_ids: set[str] = set()
    for index, raw_edge in enumerate(raw_edges):
        label = f"ast_revision_plan.mapping_edges[{index}]"
        edge = _closed_mapping(raw_edge, _GLOBAL_EDGE_FIELDS, label)
        edge_id = _text(edge["edge_id"], f"{label}.edge_id", maximum=128)
        if edge_id in seen_edge_ids:
            raise ASTRevisionError(f"global AST repeats mapping edge ID {edge_id!r}")
        if edge_pattern.fullmatch(edge_id) is None:
            raise ASTRevisionError(
                f"global AST edge ID {edge_id!r} violates "
                f"edge_id_pattern {policy['edge_id_pattern']!r}"
            )
        seen_edge_ids.add(edge_id)
        functional_node_id = _text(
            edge["functional_node_id"], f"{label}.functional_node_id", maximum=128
        )
        if functional_node_id not in allowed_functional_ids:
            raise ASTRevisionError(
                f"global AST edge {edge_id!r} references immutable/unknown functional "
                f"node {functional_node_id!r}; allowed functional nodes are "
                + ", ".join(sorted(allowed_functional_ids))
            )
        structural_node_id = _text(
            edge["structural_node_id"], f"{label}.structural_node_id", maximum=128
        )
        target = desired_by_id.get(structural_node_id)
        if target is None:
            raise ASTRevisionError(
                f"global AST edge {edge_id!r} references unknown desired structural "
                f"node {structural_node_id!r}"
            )
        operator = _text(
            edge["action_operator"], f"{label}.action_operator", maximum=64
        )
        if operator not in {action.operator for action in target.allowed_actions}:
            allowed_operators = sorted(
                action.operator for action in target.allowed_actions
            )
            raise ASTRevisionError(
                f"global AST edge {edge_id!r} operator {operator!r} is unavailable on "
                f"node {structural_node_id!r}; allowed operators are "
                + ", ".join(allowed_operators)
            )
        evidence_refs = _string_list(
            edge["evidence_refs"], f"{label}.evidence_refs"
        )
        unknown_refs = sorted(set(evidence_refs) - set(catalog_by_id))
        if unknown_refs:
            raise ASTRevisionError(
                f"global AST edge {edge_id!r} references unknown evidence IDs: "
                + ", ".join(unknown_refs)
            )
        desired_edges.append(
            MappingEdgeSpec.from_mapping(
                {
                    "schema_version": "astevolve.mapping_edge.v1",
                    "edge_id": edge_id,
                    "relation": "realizes",
                    "functional_node_id": functional_node_id,
                    "structural_node_id": structural_node_id,
                    "action_operator": operator,
                    "evidence_refs": evidence_refs,
                }
            )
        )

    decision_refs = set(plan["decision_record"]["evidence_refs"])
    unknown_decision_refs = sorted(decision_refs - set(catalog_by_id))
    if unknown_decision_refs:
        raise ASTRevisionError(
            "decision_record references unknown evidence IDs: "
            + ", ".join(unknown_decision_refs)
        )

    try:
        provisional = ExecutableDualAST(
            ast_id=before.ast_id,
            revision=before.revision + 1,
            structural_nodes=tuple(
                sorted(immutable_nodes, key=lambda item: item.node_id)
                + sorted(desired_nodes, key=lambda item: item.node_id)
            ),
            functional_nodes=before.functional_nodes,
            mapping_edges=tuple(sorted(desired_edges, key=lambda item: item.edge_id)),
        )
    except (TypeError, ValueError) as error:
        raise ASTRevisionError(
            f"global revised executable_dual_ast is invalid: {error}"
        ) from error
    old_metadata = updated.get("_global_ast_node_metadata")
    node_metadata_changed = old_metadata != node_metadata
    provisional_ast = provisional.to_dict()
    provisional_ast["revision"] = before.revision
    ast_body_changed = provisional_ast != (
        before.to_dict() | {"revision": before.revision}
    )
    changed = bool(ast_body_changed or node_metadata_changed)
    if changed:
        content_token = int(
            _digest(
                "astevolve.global_ast_revision_content.v2",
                {"ast": provisional_ast, "node_metadata": node_metadata},
            )[:12],
            16,
        )
        revised = provisional.to_dict()
        revised["revision"] = before.revision + 1 + content_token
        after = ExecutableDualAST.from_mapping(revised)
        updated["executable_dual_ast"] = after.to_dict()
        updated["_global_ast_node_metadata"] = deepcopy(node_metadata)
    else:
        after = before

    before_positions = _node_positions(before)
    after_positions = _node_positions(after)
    added = sorted(
        (node_id, position)
        for node_id, positions in after_positions.items()
        for position in set(positions) - set(before_positions.get(node_id, ()))
    )
    removed = sorted(
        (node_id, position)
        for node_id, positions in before_positions.items()
        for position in set(positions) - set(after_positions.get(node_id, ()))
    )
    mapping_changes = _global_mapping_change_records(before, after)
    report = _report(
        policy=policy,
        plan=plan,
        before=before,
        after=after,
        applied=changed,
        added=added,
        removed=removed,
        mapping_changes=mapping_changes,
        report_version=GLOBAL_AST_REVISION_REPORT_VERSION,
        policy_version=GLOBAL_AST_EVOLUTION_POLICY_VERSION,
        plan_version=GLOBAL_AST_REVISION_PLAN_VERSION,
        policy_limits=_global_policy_report(policy),
    )
    report.update(
        {
            "rationale": plan["decision_record"]["hypothesis"],
            "decision_record": deepcopy(plan["decision_record"]),
            "node_changes": _global_node_change_records(before, after),
            "catalog_hash": catalog_hash,
            "approved_positions": approved_positions,
            "node_metadata": deepcopy(node_metadata),
        }
    )


    from .residue_design_context import refresh_migration_frontier_context

    updated = refresh_migration_frontier_context(updated)
    return updated, report


def apply_ast_revision_plan(
    state: Mapping[str, Any], strategy: Mapping[str, Any]
) -> tuple[Dict[str, Any], Dict[str, Any]]:


    updated = deepcopy(dict(state))
    raw_ast = updated.get("executable_dual_ast")
    if raw_ast is None:
        if strategy.get("ast_revision_plan") is not None:
            raise ASTRevisionError("ast_revision_plan requires executable_dual_ast")
        report = _report(
            policy=None,
            plan=None,
            before=None,
            after=None,
            applied=False,
        )
        return updated, report
    try:
        before = ExecutableDualAST.from_mapping(raw_ast)
    except (TypeError, ValueError) as error:
        raise ASTRevisionError(f"base executable_dual_ast is invalid: {error}") from error

    raw_plan = strategy.get("ast_revision_plan")
    raw_plan_version = (
        raw_plan.get("schema_version") if isinstance(raw_plan, Mapping) else None
    )
    if raw_plan_version == GLOBAL_AST_REVISION_PLAN_VERSION or (
        raw_plan is None and updated.get("global_ast_evolution_policy") is not None
    ):
        return _apply_global_ast_revision_plan(updated, strategy, before)

    raw_policy = updated.get("ast_evolution_policy")
    if raw_policy is None:
        if strategy.get("ast_revision_plan") is not None:
            raise ASTRevisionError("case does not declare an ast_evolution_policy")
        report = _report(
            policy=None,
            plan=None,
            before=before,
            after=before,
            applied=False,
        )
        return updated, report
    policy = _normalize_policy(raw_policy)

    base_nodes = {node.node_id: node for node in before.structural_nodes}
    for node_id, envelope in policy["structural_node_envelopes"].items():
        node = base_nodes.get(node_id)
        if node is None or node.kind != "editable":
            raise ASTRevisionError(
                f"AST evolution envelope references non-editable node {node_id!r}"
            )
        if node.selector.chain_id != envelope["chain_id"]:
            raise ASTRevisionError(
                f"AST evolution envelope chain differs from node {node_id!r}"
            )
        base_selected = set(_positions(node.selector))
        if not base_selected.issubset(envelope["allowed_positions"]):
            raise ASTRevisionError(
                f"base selector for node {node_id!r} is outside its evolution envelope"
            )
        if not envelope["min_positions"] <= len(base_selected) <= envelope["max_positions"]:
            raise ASTRevisionError(
                f"base selector for node {node_id!r} violates its evolution size bounds"
            )
    if sum(len(positions) for positions in _node_positions(before).values()) > policy[
        "max_total_editable_positions"
    ]:
        raise ASTRevisionError(
            "base executable_dual_ast exceeds max_total_editable_positions"
        )
    base_edges = {edge.edge_id: edge for edge in before.mapping_edges}
    for edge_id, rule in policy["mapping_edge_allowlist"].items():
        edge = base_edges.get(edge_id)
        if edge is None:
            raise ASTRevisionError(
                f"mapping edge allowlist references unknown edge {edge_id!r}"
            )
        for node_id in rule["structural_node_ids"]:
            node = base_nodes.get(node_id)
            if node is None or node.kind != "editable":
                raise ASTRevisionError(
                    f"mapping edge allowlist {edge_id!r} references non-editable "
                    f"node {node_id!r}"
                )
            allowed_operators = {action.operator for action in node.allowed_actions}
            unsupported = set(rule["action_operators"]) - allowed_operators
            if unsupported:
                raise ASTRevisionError(
                    f"mapping edge allowlist {edge_id!r} permits unsupported operators "
                    f"for node {node_id!r}: {sorted(unsupported)}"
                )

    if raw_plan is None:
        return updated, _report(
            policy=policy,
            plan=None,
            before=before,
            after=before,
            applied=False,
        )
    if not policy["enabled"]:
        raise ASTRevisionError("ast_revision_plan is present but AST evolution is disabled")
    plan = _normalize_plan(raw_plan)
    structural_edits = plan["structural_node_edits"]
    mapping_edits = plan["mapping_edge_edits"]
    if len(structural_edits) > policy["max_structural_node_edits"]:
        raise ASTRevisionError("ast_revision_plan exceeds max_structural_node_edits")
    if len(mapping_edits) > policy["max_mapping_edge_edits"]:
        raise ASTRevisionError("ast_revision_plan exceeds max_mapping_edge_edits")

    patched = before.to_dict()
    raw_nodes = {str(node["node_id"]): node for node in patched["structural_nodes"]}
    seen_nodes: set[str] = set()
    for index, raw_edit in enumerate(structural_edits):
        edit = _closed_mapping(
            raw_edit,
            _STRUCTURAL_EDIT_FIELDS,
            f"ast_revision_plan.structural_node_edits[{index}]",
        )
        node_id = str(edit["node_id"]).strip()
        if not node_id or node_id in seen_nodes:
            raise ASTRevisionError("structural_node_edits node IDs must be unique and non-empty")
        seen_nodes.add(node_id)
        node = raw_nodes.get(node_id)
        envelope = policy["structural_node_envelopes"].get(node_id)
        if node is None or node.get("kind") != "editable" or envelope is None:
            raise ASTRevisionError(f"structural node {node_id!r} is not policy-editable")
        try:
            selector = ResidueSelector.from_mapping(edit["selector"])
        except (TypeError, ValueError) as error:
            raise ASTRevisionError(
                f"selector edit for node {node_id!r} is invalid: {error}"
            ) from error
        if selector.chain_id != envelope["chain_id"]:
            raise ASTRevisionError(f"selector edit for node {node_id!r} changes chain")
        selected = set(_positions(selector))
        allowed = set(envelope["allowed_positions"])
        if not selected.issubset(allowed):
            escaped = sorted(selected - allowed)
            raise ASTRevisionError(
                f"selector edit for node {node_id!r} escapes its envelope at {escaped}"
            )
        if not envelope["min_positions"] <= len(selected) <= envelope["max_positions"]:
            raise ASTRevisionError(
                f"selector edit for node {node_id!r} selects {len(selected)} positions; "
                f"allowed [{envelope['min_positions']}, {envelope['max_positions']}]"
            )
        node["selector"] = selector.to_dict()

    raw_edges = {str(edge["edge_id"]): edge for edge in patched["mapping_edges"]}
    disabled_edges: set[str] = set()
    seen_edges: set[str] = set()
    mapping_changes: list[Dict[str, Any]] = []
    for index, raw_edit in enumerate(mapping_edits):
        edit = _closed_mapping(
            raw_edit,
            _MAPPING_EDIT_FIELDS,
            f"ast_revision_plan.mapping_edge_edits[{index}]",
        )
        edge_id = str(edit["edge_id"]).strip()
        if not edge_id or edge_id in seen_edges:
            raise ASTRevisionError("mapping_edge_edits edge IDs must be unique and non-empty")
        seen_edges.add(edge_id)
        edge = raw_edges.get(edge_id)
        rule = policy["mapping_edge_allowlist"].get(edge_id)
        if edge is None or rule is None:
            raise ASTRevisionError(f"mapping edge {edge_id!r} is not policy-editable")
        enabled = edit["enabled"]
        if not isinstance(enabled, bool):
            raise ASTRevisionError(f"mapping edge {edge_id!r} enabled must be boolean")
        structural_node_id = str(edit["structural_node_id"]).strip()
        action_operator = str(edit["action_operator"]).strip()
        if structural_node_id not in rule["structural_node_ids"]:
            raise ASTRevisionError(
                f"mapping edge {edge_id!r} cannot target {structural_node_id!r}"
            )
        if action_operator not in rule["action_operators"]:
            raise ASTRevisionError(
                f"mapping edge {edge_id!r} cannot use operator {action_operator!r}"
            )
        before_edge = deepcopy(edge)
        if not enabled:
            if not rule["allow_disable"]:
                raise ASTRevisionError(f"mapping edge {edge_id!r} may not be disabled")
            disabled_edges.add(edge_id)
        else:
            edge["structural_node_id"] = structural_node_id
            edge["action_operator"] = action_operator
        after_edge = None if not enabled else deepcopy(edge)
        if before_edge != after_edge:
            mapping_changes.append(
                {"edge_id": edge_id, "before": before_edge, "after": after_edge}
            )
    if disabled_edges:
        patched["mapping_edges"] = [
            edge for edge in patched["mapping_edges"]
            if str(edge.get("edge_id")) not in disabled_edges
        ]

    before_positions = _node_positions(before)
    provisional = deepcopy(patched)

    provisional["revision"] = before.revision + 1
    try:
        after_candidate = ExecutableDualAST.from_mapping(provisional)
    except (TypeError, ValueError) as error:
        raise ASTRevisionError(f"revised executable_dual_ast is invalid: {error}") from error
    after_positions = _node_positions(after_candidate)
    added = sorted(
        (node_id, position)
        for node_id, positions in after_positions.items()
        for position in set(positions) - set(before_positions.get(node_id, ()))
    )
    removed = sorted(
        (node_id, position)
        for node_id, positions in before_positions.items()
        for position in set(positions) - set(after_positions.get(node_id, ()))
    )
    if len(added) > policy["max_added_positions"]:
        raise ASTRevisionError("ast_revision_plan exceeds max_added_positions")
    if len(removed) > policy["max_removed_positions"]:
        raise ASTRevisionError("ast_revision_plan exceeds max_removed_positions")
    total_positions = sum(len(positions) for positions in after_positions.values())
    if total_positions > policy["max_total_editable_positions"]:
        raise ASTRevisionError(
            "ast_revision_plan exceeds max_total_editable_positions"
        )

    changed = bool(added or removed or mapping_changes)
    if changed:


        revision_basis = deepcopy(patched)
        revision_basis["revision"] = before.revision
        content_token = int(
            _digest("astevolve.legacy_ast_revision_content.v1", revision_basis)[:12],
            16,
        )
        patched["revision"] = before.revision + 1 + content_token
        try:
            after = ExecutableDualAST.from_mapping(patched)
        except (TypeError, ValueError) as error:
            raise ASTRevisionError(
                f"content-addressed executable_dual_ast is invalid: {error}"
            ) from error
    else:
        after = before
    if changed:
        updated["executable_dual_ast"] = after.to_dict()
    report = _report(
        policy=policy,
        plan=plan,
        before=before,
        after=after,
        applied=changed,
        added=added,
        removed=removed,
        mapping_changes=mapping_changes,
    )
    return updated, report


def validate_ast_revision_permissions(
    raw_ast: Mapping[str, Any] | None,
    *,
    base_masks: Mapping[str, Sequence[bool]],
    base_fixed_residues: Mapping[str, Mapping[Any, str]],
) -> None:


    if raw_ast is None:
        return
    try:
        ast = ExecutableDualAST.from_mapping(raw_ast)
    except (TypeError, ValueError) as error:
        raise ASTRevisionError(f"executable_dual_ast is invalid: {error}") from error
    for node in ast.structural_nodes:
        if node.kind != "editable":
            continue
        chain_id = node.selector.chain_id
        mask = base_masks.get(chain_id)
        if mask is None:
            raise ASTRevisionError(
                f"editable node {node.node_id!r} references a chain without a base mask"
            )
        fixed = {
            int(position) for position in base_fixed_residues.get(chain_id, {})
        }
        denied = [
            position
            for position in _positions(node.selector)
            if position >= len(mask) or not bool(mask[position]) or position in fixed
        ]
        if denied:
            raise ASTRevisionError(
                f"editable node {node.node_id!r} selects residue positions denied by "
                f"the case base permissions: {denied}"
            )


__all__ = [
    "AST_EVOLUTION_POLICY_VERSION",
    "AST_REVISION_PLAN_VERSION",
    "AST_REVISION_REPORT_VERSION",
    "GLOBAL_AST_EVOLUTION_POLICY_VERSION",
    "GLOBAL_AST_DECISION_ACTIONS",
    "GLOBAL_AST_REVISION_PLAN_VERSION",
    "GLOBAL_AST_REVISION_REPORT_VERSION",
    "ASTRevisionError",
    "apply_ast_revision_plan",
    "normalize_global_ast_revision_plan",
    "validate_ast_revision_permissions",
]
