

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from astevolve.search.config import SAConfig


PORTFOLIO_MAPPING_COMPONENT_VERSION = (
    "astevolve.portfolio_mapping_component.v1"
)
PORTFOLIO_MAPPING_COMPONENT_SET_VERSION = (
    "astevolve.portfolio_mapping_component_set.v1"
)

_IDENTITY_FIELDS = (
    "ast_id",
    "ast_revision",
    "edge_id",
    "functional_node_id",
    "structural_node_id",
    "action_id",
    "measurement_id",
)
_COMPONENT_FIELDS = frozenset(
    {
        "schema_version",
        "component_id",
        "owner_node_id",
        "op",
        "chain_id",
        "node",
        "target_nodes",
        "attempted_positions",
        "positions",
        "changes",
        "actual_delta",
        "mapping_attribution",
        "mutation_plan",
        "selected_action_budget",
        "mapping_component_hash",
    }
)
_PLAN_FIELDS = frozenset(
    {
        *_IDENTITY_FIELDS,
        "node",
        "mapping_execution",
        "op_weights",
        "budget",
        "allowed_action_budgets",
        "legal_positions",
    }
)
_OPERATOR_PRIORITY = {
    "point": 0,
    "point_substitution": 0,
    "site_resample": 1,
    "segment_resample": 2,
    "segment_mutagenesis": 3,
    "cdr_resample": 4,
}


class PortfolioMappingComponentError(ValueError):


    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)


def _fail(code: str, detail: str = "") -> None:
    raise PortfolioMappingComponentError(code, detail)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        _fail("portfolio_mapping_not_canonical_json", str(error))


def _domain_hash(prefix: str, domain: str, value: Any) -> str:
    digest = hashlib.sha256(
        (domain + "\0" + _canonical_json(value)).encode("utf-8")
    ).hexdigest()
    return prefix + digest


def _action_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    _fail("portfolio_mapping_action_invalid")


def _normalized_actions(values: Sequence[Any]) -> List[Dict[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _fail("portfolio_mapping_actions_invalid")
    actions: List[Dict[str, Any]] = []
    action_ids = set()
    for raw in values:
        action = _action_mapping(raw)
        missing = sorted(
            {
                *_IDENTITY_FIELDS,
                "operator",
                "chain_id",
                "compiled_segment_name",
                "legal_positions",
                "budget",
            }
            - set(action)
        )
        if missing:
            _fail("portfolio_mapping_action_fields_missing", ",".join(missing))
        action_id = str(action.get("action_id") or "")
        if not action_id or action_id in action_ids:
            _fail("portfolio_mapping_action_id_invalid", action_id)
        action_ids.add(action_id)
        try:
            legal = sorted({int(position) for position in action["legal_positions"]})
            budget = action["budget"]
            budget_min = int(budget["min"])
            budget_max = int(budget["max"])
        except (KeyError, TypeError, ValueError) as error:
            _fail("portfolio_mapping_action_shape_invalid", str(error))
        if (
            legal != list(action["legal_positions"])
            or budget_min < 1
            or budget_min > budget_max
            or budget_min > len(legal)
        ):
            _fail("portfolio_mapping_action_shape_invalid", action_id)
        action["legal_positions"] = legal
        action["budget"] = {"min": budget_min, "max": budget_max}
        actions.append(action)
    return actions


def _outer_changes(move: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = move.get("changes")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        _fail("portfolio_mapping_outer_changes_missing")
    changes: List[Dict[str, Any]] = []
    seen = set()
    for item in raw:
        if not isinstance(item, Mapping):
            _fail("portfolio_mapping_outer_change_invalid")
        try:
            change = {
                "chain_id": str(item["chain_id"]),
                "position": int(item["position"]),
                "from": item["from"],
                "to": item["to"],
            }
        except (KeyError, TypeError, ValueError) as error:
            _fail("portfolio_mapping_outer_change_invalid", str(error))
        key = (change["chain_id"], change["position"])
        if not key[0] or key in seen or change["from"] == change["to"]:
            _fail("portfolio_mapping_outer_change_invalid", f"{key[0]}:{key[1]}")
        seen.add(key)
        changes.append(change)
    return sorted(changes, key=lambda row: (row["chain_id"], row["position"]))


def _trusted_position_owners(
    move: Mapping[str, Any],
    changes: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, int], str]:
    summary = move.get("portfolio_realization_summary")
    if not isinstance(summary, Mapping):
        summary = move.get("mapping_realization_summary")
    if not isinstance(summary, Mapping):
        _fail("portfolio_mapping_owner_summary_missing")
    raw = summary.get("position_owners")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        _fail("portfolio_mapping_position_owners_missing")
    owners: Dict[Tuple[str, int], str] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            _fail("portfolio_mapping_position_owner_invalid")
        try:
            key = (str(item["chain_id"]), int(item["position"]))
        except (KeyError, TypeError, ValueError) as error:
            _fail("portfolio_mapping_position_owner_invalid", str(error))
        owner = str(item.get("owner_node_id") or "")
        if not key[0] or not owner or key in owners:
            _fail("portfolio_mapping_position_owner_invalid", f"{key[0]}:{key[1]}")
        owners[key] = owner
    expected = {(str(row["chain_id"]), int(row["position"])) for row in changes}
    if set(owners) != expected:
        _fail(
            "portfolio_mapping_position_owner_coverage_mismatch",
            f"expected={sorted(expected)} observed={sorted(owners)}",
        )
    return owners


def _action_for_group(
    actions: Sequence[Mapping[str, Any]],
    *,
    owner: str,
    chain_id: str,
    positions: Sequence[int],
) -> Dict[str, Any]:
    wanted = set(int(position) for position in positions)
    matches = []
    for action in actions:
        budget = action["budget"]
        if (
            str(action.get("structural_node_id") or "") == owner
            and str(action.get("chain_id") or "") == chain_id
            and wanted.issubset(set(action["legal_positions"]))
            and int(budget["min"]) <= len(wanted) <= int(budget["max"])
        ):
            matches.append(dict(action))
    if not matches:
        _fail(
            "portfolio_mapping_action_unavailable",
            f"owner={owner} chain={chain_id} positions={sorted(wanted)}",
        )
    matches.sort(
        key=lambda action: (
            _OPERATOR_PRIORITY.get(str(action.get("operator") or ""), 100),
            len(action["legal_positions"]),
            str(action["action_id"]),
        )
    )
    return matches[0]


def _component(
    *,
    action: Mapping[str, Any],
    owner: str,
    chain_id: str,
    changes: Sequence[Mapping[str, Any]],
    outer_length_delta: Mapping[str, Any],
) -> Dict[str, Any]:
    positions = sorted(int(change["position"]) for change in changes)
    segment = str(action["compiled_segment_name"])
    operator = str(action["operator"])
    attribution = {field: action[field] for field in _IDENTITY_FIELDS}
    budget = {
        "min": int(action["budget"]["min"]),
        "max": int(action["budget"]["max"]),
    }
    component_changes = [
        {
            "chain_id": str(change["chain_id"]),
            "position": int(change["position"]),
            "from": change["from"],
            "to": change["to"],
            "node": segment,
        }
        for change in sorted(changes, key=lambda row: int(row["position"]))
    ]
    identity = {
        "owner_node_id": owner,
        "chain_id": chain_id,
        "action_id": str(action["action_id"]),
        "changes": [
            {key: row[key] for key in ("chain_id", "position", "from", "to")}
            for row in component_changes
        ],
    }
    component_id = _domain_hash(
        "portfolio_mapping_component_id_sha256:",
        "astevolve.portfolio_mapping_component_id.v1",
        identity,
    )
    semantic = {
        "schema_version": PORTFOLIO_MAPPING_COMPONENT_VERSION,
        "component_id": component_id,
        "owner_node_id": owner,
        "op": operator,
        "chain_id": chain_id,
        "node": segment,
        "target_nodes": [segment],
        "attempted_positions": {chain_id: positions},
        "positions": {chain_id: positions},
        "changes": component_changes,
        "actual_delta": {
            "residue_change_count": len(component_changes),
            "length_delta_by_chain": {
                str(chain): int(delta) for chain, delta in outer_length_delta.items()
            },
        },
        "mapping_attribution": attribution,
        "mutation_plan": {
            **attribution,
            "node": segment,
            "mapping_execution": "full",
            "op_weights": {operator: 1.0},
            "budget": budget,
            "allowed_action_budgets": {operator: budget},
            "legal_positions": list(action["legal_positions"]),
        },
        "selected_action_budget": budget,
    }
    return {
        **semantic,
        "mapping_component_hash": _domain_hash(
            "portfolio_mapping_component_sha256:",
            PORTFOLIO_MAPPING_COMPONENT_VERSION,
            semantic,
        ),
    }


def _component_set_semantic(
    move: Mapping[str, Any], components: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    return {
        "schema_version": PORTFOLIO_MAPPING_COMPONENT_SET_VERSION,
        "compiled_portfolio_request_hash": move.get(
            "compiled_portfolio_request_hash"
        ),
        "portfolio_slot_id": move.get("portfolio_slot_id"),
        "outer_changes": _outer_changes(move),
        "mapping_component_hashes": [
            str(component["mapping_component_hash"]) for component in components
        ],
    }


def bind_portfolio_mapping_components(
    prebuilt_proposals: Sequence[Tuple[Mapping[str, str], Mapping[str, Any]]],
    mapping_actions: Sequence[Any],
) -> List[Tuple[Dict[str, str], Dict[str, Any]]]:


    if isinstance(prebuilt_proposals, (str, bytes)) or not isinstance(
        prebuilt_proposals, Sequence
    ):
        _fail("portfolio_mapping_proposals_invalid")
    actions = _normalized_actions(mapping_actions)
    bound: List[Tuple[Dict[str, str], Dict[str, Any]]] = []
    for item in prebuilt_proposals:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            _fail("portfolio_mapping_proposal_invalid")
        sequences, raw_move = item
        if not isinstance(sequences, Mapping) or not isinstance(raw_move, Mapping):
            _fail("portfolio_mapping_proposal_invalid")
        move = deepcopy(dict(raw_move))
        if str(move.get("op") or "") not in {
            "portfolio_exact",
            "case_bootstrap_exact",
        }:
            bound.append((dict(sequences), move))
            continue
        if "mapping_components" in move or "mapping_component_set_hash" in move:
            _fail("portfolio_mapping_components_already_bound")
        changes = _outer_changes(move)
        owners = _trusted_position_owners(move, changes)
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for change in changes:
            key = (str(change["chain_id"]), int(change["position"]))
            grouped.setdefault((owners[key], key[0]), []).append(change)
        outer_delta = move.get("actual_delta")
        outer_delta = outer_delta if isinstance(outer_delta, Mapping) else {}
        raw_length = outer_delta.get("length_delta_by_chain")
        raw_length = raw_length if isinstance(raw_length, Mapping) else {}
        try:
            length_delta = {str(chain): int(delta) for chain, delta in raw_length.items()}
        except (TypeError, ValueError) as error:
            _fail("portfolio_mapping_outer_length_delta_invalid", str(error))
        if any(delta != 0 for delta in length_delta.values()):
            _fail("portfolio_mapping_length_delta_forbidden")
        components = []
        for (owner, chain_id), group in sorted(grouped.items()):
            action = _action_for_group(
                actions,
                owner=owner,
                chain_id=chain_id,
                positions=[int(row["position"]) for row in group],
            )
            components.append(
                _component(
                    action=action,
                    owner=owner,
                    chain_id=chain_id,
                    changes=group,
                    outer_length_delta=length_delta,
                )
            )
        move["mapping_components"] = components
        move["mapping_component_set_hash"] = _domain_hash(
            "portfolio_mapping_component_set_sha256:",
            PORTFOLIO_MAPPING_COMPONENT_SET_VERSION,
            _component_set_semantic(move, components),
        )
        validate_portfolio_mapping_components(move, mapping_actions=actions)
        bound.append((dict(sequences), move))
    return bound


def validate_portfolio_mapping_components(
    move: Mapping[str, Any],
    mapping_actions: Sequence[Any] | None = None,
) -> List[Dict[str, Any]]:


    if not isinstance(move, Mapping) or str(move.get("op") or "") not in {
        "portfolio_exact",
        "case_bootstrap_exact",
    }:
        _fail("portfolio_mapping_outer_move_invalid")
    changes = _outer_changes(move)
    owners = _trusted_position_owners(move, changes)
    outer_actual = move.get("actual_delta")
    if not isinstance(outer_actual, Mapping):
        _fail("portfolio_mapping_outer_actual_delta_invalid")
    try:
        outer_residue_count = int(outer_actual.get("residue_change_count", -1))
        raw_outer_length = outer_actual.get("length_delta_by_chain")
        if not isinstance(raw_outer_length, Mapping):
            raise TypeError("length_delta_by_chain must be a mapping")
        outer_length_delta = {
            str(chain): int(delta) for chain, delta in raw_outer_length.items()
        }
    except (TypeError, ValueError) as error:
        _fail("portfolio_mapping_outer_actual_delta_invalid", str(error))
    if outer_residue_count != len(changes) or any(
        delta != 0 for delta in outer_length_delta.values()
    ):
        _fail("portfolio_mapping_outer_actual_delta_invalid")
    raw_components = move.get("mapping_components")
    if isinstance(raw_components, (str, bytes)) or not isinstance(
        raw_components, Sequence
    ) or not raw_components:
        _fail("portfolio_mapping_components_missing")
    actions = _normalized_actions(mapping_actions) if mapping_actions is not None else []
    action_by_id = {str(action["action_id"]): action for action in actions}
    canonical: List[Dict[str, Any]] = []
    covered: Dict[Tuple[str, int], Dict[str, Any]] = {}
    component_ids = set()
    component_hashes = set()
    observed_groups: List[Tuple[str, str]] = []
    for raw in raw_components:
        if not isinstance(raw, Mapping) or set(raw) != _COMPONENT_FIELDS:
            _fail("portfolio_mapping_component_schema_invalid")
        component = deepcopy(dict(raw))
        if component.get("schema_version") != PORTFOLIO_MAPPING_COMPONENT_VERSION:
            _fail("portfolio_mapping_component_schema_invalid")
        semantic = {
            key: component[key] for key in component if key != "mapping_component_hash"
        }
        expected_hash = _domain_hash(
            "portfolio_mapping_component_sha256:",
            PORTFOLIO_MAPPING_COMPONENT_VERSION,
            semantic,
        )
        if component.get("mapping_component_hash") != expected_hash:
            _fail("portfolio_mapping_component_hash_mismatch")
        component_id = str(component.get("component_id") or "")
        component_hash = str(component.get("mapping_component_hash") or "")
        if not component_id or component_id in component_ids:
            _fail("portfolio_mapping_component_id_duplicate", component_id)
        if not component_hash or component_hash in component_hashes:
            _fail("portfolio_mapping_component_hash_duplicate", component_hash)
        component_ids.add(component_id)
        component_hashes.add(component_hash)
        attribution = component.get("mapping_attribution")
        plan = component.get("mutation_plan")
        if (
            not isinstance(attribution, Mapping)
            or set(attribution) != set(_IDENTITY_FIELDS)
            or not isinstance(plan, Mapping)
            or set(plan) != _PLAN_FIELDS
        ):
            _fail("portfolio_mapping_component_attribution_invalid", component_id)
        owner = str(component.get("owner_node_id") or "")
        if owner != str(attribution.get("structural_node_id") or ""):
            _fail("portfolio_mapping_component_owner_mismatch", component_id)
        for field in _IDENTITY_FIELDS:
            if plan.get(field) != attribution.get(field):
                _fail("portfolio_mapping_component_plan_identity_mismatch", field)
        raw_group = component.get("changes")
        if isinstance(raw_group, (str, bytes)) or not isinstance(raw_group, Sequence) or not raw_group:
            _fail("portfolio_mapping_component_changes_invalid", component_id)
        group_keys = []
        for row in raw_group:
            if not isinstance(row, Mapping) or set(row) != {
                "chain_id", "position", "from", "to", "node"
            }:
                _fail("portfolio_mapping_component_change_invalid", component_id)
            key = (str(row["chain_id"]), int(row["position"]))
            if key in covered:
                _fail("portfolio_mapping_component_overlap", f"{key[0]}:{key[1]}")
            expected_owner = owners.get(key)
            if expected_owner != owner:
                _fail("portfolio_mapping_component_owner_mismatch", f"{key[0]}:{key[1]}")
            outer = next(
                (item for item in changes if (item["chain_id"], item["position"]) == key),
                None,
            )
            if outer is None or any(row[field] != outer[field] for field in ("from", "to")):
                _fail("portfolio_mapping_component_delta_mismatch", f"{key[0]}:{key[1]}")
            if str(row.get("node") or "") != str(component.get("node") or ""):
                _fail("portfolio_mapping_component_change_node_mismatch", component_id)
            covered[key] = dict(row)
            group_keys.append(key)
        chain_id = str(component.get("chain_id") or "")
        positions = sorted(position for chain, position in group_keys if chain == chain_id)
        if len(positions) != len(group_keys):
            _fail("portfolio_mapping_component_chain_mismatch", component_id)
        group_key = (owner, chain_id)
        if group_key in observed_groups:
            _fail(
                "portfolio_mapping_component_group_duplicate",
                f"{owner}:{chain_id}",
            )
        observed_groups.append(group_key)
        if component.get("positions") != {chain_id: positions} or component.get(
            "attempted_positions"
        ) != {chain_id: positions}:
            _fail("portfolio_mapping_component_positions_mismatch", component_id)
        if component.get("target_nodes") != [component.get("node")]:
            _fail("portfolio_mapping_component_target_nodes_mismatch", component_id)
        actual = component.get("actual_delta")
        if not isinstance(actual, Mapping) or int(actual.get("residue_change_count", -1)) != len(group_keys):
            _fail("portfolio_mapping_component_actual_delta_mismatch", component_id)
        length_delta = actual.get("length_delta_by_chain")
        if not isinstance(length_delta, Mapping):
            _fail("portfolio_mapping_component_length_delta_invalid", component_id)
        try:
            normalized_length_delta = {
                str(chain): int(delta) for chain, delta in length_delta.items()
            }
        except (TypeError, ValueError) as error:
            _fail("portfolio_mapping_component_length_delta_invalid", str(error))
        if normalized_length_delta != outer_length_delta:
            _fail("portfolio_mapping_component_length_delta_invalid", component_id)
        if actions:
            declared_action_id = str(attribution.get("action_id") or "")
            if declared_action_id not in action_by_id:
                _fail(
                    "portfolio_mapping_component_action_inactive",
                    declared_action_id,
                )


            action = _action_for_group(
                actions,
                owner=owner,
                chain_id=chain_id,
                positions=positions,
            )
            if declared_action_id != str(action["action_id"]):
                _fail(
                    "portfolio_mapping_component_action_rebound",
                    f"expected={action['action_id']} observed={declared_action_id}",
                )
            expected = _component(
                action=action,
                owner=owner,
                chain_id=chain_id,
                changes=[
                    {key: row[key] for key in ("chain_id", "position", "from", "to")}
                    for row in raw_group
                ],
                outer_length_delta=length_delta,
            )
            if component != expected:
                _fail("portfolio_mapping_component_action_mismatch", component_id)
        canonical.append(component)
    expected_keys = {(row["chain_id"], row["position"]) for row in changes}
    if set(covered) != expected_keys:
        _fail("portfolio_mapping_component_coverage_mismatch")
    expected_groups = sorted(
        {
            (owners[(row["chain_id"], row["position"])], row["chain_id"])
            for row in changes
        }
    )
    if observed_groups != expected_groups:
        _fail(
            "portfolio_mapping_component_order_mismatch",
            f"expected={expected_groups} observed={observed_groups}",
        )
    expected_set_hash = _domain_hash(
        "portfolio_mapping_component_set_sha256:",
        PORTFOLIO_MAPPING_COMPONENT_SET_VERSION,
        _component_set_semantic(move, canonical),
    )
    if move.get("mapping_component_set_hash") != expected_set_hash:
        _fail("portfolio_mapping_component_set_hash_mismatch")
    return canonical


def active_mapping_actions(
    compiled: Mapping[str, Any],
    cfg: SAConfig,
) -> List[Dict[str, Any]]:


    _ = cfg
    raw_schedule = compiled.get("effective_mapping_schedule")
    schedule = raw_schedule if isinstance(raw_schedule, Mapping) else {}
    if not bool(schedule.get("execution_enabled")):
        return []
    active = []
    for raw_action in schedule.get("active_action_specs", []) or []:
        if not isinstance(raw_action, Mapping):
            raise ValueError("effective mapping schedule action must be a mapping")
        active.append(dict(raw_action))
    return active


def require_mapping_search_compatibility(
    compiled: Mapping[str, Any],
    cfg: SAConfig,
) -> None:


    raw_plan = compiled.get("executable_mapping_plan")
    plan = raw_plan if isinstance(raw_plan, Mapping) else {}
    if not bool(plan.get("execution_enabled")):
        return
    raw_schedule = compiled.get("effective_mapping_schedule")
    schedule = raw_schedule if isinstance(raw_schedule, Mapping) else {}
    if not bool(schedule.get("execution_enabled")):
        raise ValueError(
            "full executable mapping has no pre-inner effective mapping schedule"
        )
    if str(cfg.search_method).lower() != "mcts":
        raise ValueError(
            "full executable mapping is supported only by search_method='mcts' "
            "in AST-01B v1; use flat_mask for a controlled non-mapping baseline"
        )
    if not active_mapping_actions(compiled, cfg):
        raise ValueError(
            "no mapping edge action matches the final effective node mutation_ops"
        )


__all__ = [
    "PORTFOLIO_MAPPING_COMPONENT_SET_VERSION",
    "PORTFOLIO_MAPPING_COMPONENT_VERSION",
    "PortfolioMappingComponentError",
    "active_mapping_actions",
    "bind_portfolio_mapping_components",
    "require_mapping_search_compatibility",
    "validate_portfolio_mapping_components",
]
