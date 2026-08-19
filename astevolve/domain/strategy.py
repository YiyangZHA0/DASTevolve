

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Mapping, Tuple

from ._mapping import copied_mapping


@dataclass(frozen=True)
class ContractResponse:


    accepted: Tuple[str, ...] = ()
    rejected: Tuple[str, ...] = ()
    reason: str = ""

    @classmethod
    def from_mapping(cls, value: Any) -> "ContractResponse":
        data = value if isinstance(value, Mapping) else {}
        return cls(
            accepted=tuple(str(item) for item in data.get("accepted", ()) or ()),
            rejected=tuple(str(item) for item in data.get("rejected", ()) or ()),
            reason=str(data.get("reason") or data.get("rationale") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": list(self.accepted),
            "rejected": list(self.rejected),
            "reason": self.reason,
        }


EDIT_CONTRACT_SCHEMA_V1 = "ast_edit_contract_v1"
EDIT_CONTRACT_SCHEMA_V2 = "ast_edit_contract_v2"
EDIT_CONTRACT_ACTIONS = frozenset(
    {
        "optimize_node",
        "repair_node",
        "freeze_node",
        "expand_edit_scope",
        "increase_negative_design_weight",
        "modify_ast_node_definitions",
        "redesign_constraints",
        "adjust_search_space",
    }
)

_EDIT_CONTRACT_V2_FIELDS = frozenset(
    {
        "schema_version",
        "action",
        "required_nodes",
        "forbidden_nodes",
        "mutation_budget",
        "rationale",
        "metadata",
    }
)
_EDIT_CONTRACT_V2_REQUIRED_FIELDS = _EDIT_CONTRACT_V2_FIELDS - {"metadata"}
_EDIT_CONTRACT_V1_FIELDS = frozenset(
    {
        "schema_version",
        "action",
        "actions",
        "recommended_actions",
        "allowed_structural_nodes",
        "frozen_nodes",
        "mutation_budget",
        "rationale",
        "reason",
        "metadata",
    }
)


def _contract_nodes(value: Any, field_name: str) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"edit contract {field_name} must be a list of node names")
    nodes = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"edit contract {field_name} must contain non-empty strings"
            )
        nodes.append(item.strip())
    if len(nodes) != len(set(nodes)):
        raise ValueError(f"edit contract {field_name} contains duplicate nodes")
    return tuple(nodes)


def _contract_budget(value: Any, action: str) -> Dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {"min", "max"}:
        raise ValueError("edit contract mutation_budget must contain exactly min and max")
    minimum = value["min"]
    maximum = value["max"]
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, int)
        or not isinstance(maximum, int)
    ):
        raise ValueError("edit contract mutation_budget min and max must be integer values")
    if minimum < 0 or maximum < 0:
        raise ValueError("edit contract mutation_budget values must be non-negative")
    if minimum > maximum:
        raise ValueError("edit contract mutation_budget min cannot exceed max")
    if action == "freeze_node" and maximum != 0:
        raise ValueError("freeze_node requires mutation_budget max to be 0")
    return {"min": minimum, "max": maximum}


def _contract_action(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("edit contract must contain exactly one action")
    action = value.strip()
    if action not in EDIT_CONTRACT_ACTIONS:
        raise ValueError(f"unknown action in edit contract: {action!r}")
    return action


def _contract_rationale(value: Any) -> Any:
    if not isinstance(value, (str, Mapping)):
        raise ValueError("edit contract rationale must be a string or mapping")
    return deepcopy(value)


def _single_v1_action(data: Mapping[str, Any]) -> Any:
    action_sources = [
        name
        for name in ("action", "actions", "recommended_actions")
        if name in data
    ]
    if len(action_sources) != 1:
        raise ValueError("v1 edit contract must contain exactly one action representation")
    field_name = action_sources[0]
    value = data[field_name]
    if field_name == "action":
        return value
    if isinstance(value, Mapping):
        values = list(value)
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise ValueError("v1 edit contract must contain exactly one action")
    if len(values) != 1:
        raise ValueError("v1 edit contract must contain exactly one action")
    return values[0]


def _canonical_v2_data(value: Mapping[str, Any]) -> Dict[str, Any]:
    data = copied_mapping(value)
    legacy_fields = {
        "allowed_structural_nodes",
        "frozen_nodes",
        "actions",
        "recommended_actions",
        "required_structural_nodes",
    }
    present_legacy = sorted(legacy_fields & set(data))
    if present_legacy:
        raise ValueError(
            "ast_edit_contract_v2 contains legacy execution fields: "
            + ", ".join(present_legacy)
        )
    unknown = sorted(set(data) - _EDIT_CONTRACT_V2_FIELDS)
    if unknown:
        raise ValueError(
            "ast_edit_contract_v2 diagnostics must be nested under metadata; "
            f"unexpected fields: {', '.join(unknown)}"
        )
    missing = sorted(_EDIT_CONTRACT_V2_REQUIRED_FIELDS - set(data))
    if missing:
        raise ValueError(
            "ast_edit_contract_v2 is missing canonical fields: "
            + ", ".join(missing)
        )
    return data


def _migrate_v1_data(value: Mapping[str, Any]) -> Dict[str, Any]:
    data = copied_mapping(value)
    mixed_fields = {
        "required_nodes",
        "forbidden_nodes",
        "required_structural_nodes",
    } & set(data)
    if mixed_fields:
        raise ValueError(
            "ast_edit_contract_v1 mixes v1 and v2 node fields: "
            + ", ".join(sorted(mixed_fields))
        )
    if "rationale" in data and "reason" in data:
        raise ValueError("ast_edit_contract_v1 mixes rationale and reason fields")
    if "mutation_budget" not in data:
        raise ValueError("ast_edit_contract_v1 is missing mutation_budget")
    if "rationale" not in data and "reason" not in data:
        raise ValueError("ast_edit_contract_v1 is missing rationale")

    metadata = data.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("edit contract metadata must be a mapping")
    migrated_metadata = {
        **copied_mapping(metadata),
        "migrated_from": EDIT_CONTRACT_SCHEMA_V1,
    }
    for key, item in data.items():
        if key not in _EDIT_CONTRACT_V1_FIELDS:
            migrated_metadata[key] = deepcopy(item)

    return {
        "schema_version": EDIT_CONTRACT_SCHEMA_V2,
        "action": _single_v1_action(data),
        "required_nodes": data.get("allowed_structural_nodes", []),
        "forbidden_nodes": data.get("frozen_nodes", []),
        "mutation_budget": data["mutation_budget"],
        "rationale": data.get("rationale", data.get("reason")),
        "metadata": migrated_metadata,
    }


@dataclass(frozen=True)
class EditContract:


    action: str
    required_nodes: Tuple[str, ...]
    forbidden_nodes: Tuple[str, ...]
    mutation_budget: Mapping[str, int]
    rationale: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = EDIT_CONTRACT_SCHEMA_V2
    _emit_metadata: bool = field(default=False, repr=False, compare=False)

    @classmethod
    def from_mapping(cls, value: Any) -> "EditContract":
        if not isinstance(value, Mapping):
            raise TypeError("edit contract must be a mapping")
        if "schema_version" not in value:
            raise ValueError("edit contract requires an explicit schema_version")
        version = value.get("schema_version")
        if version == EDIT_CONTRACT_SCHEMA_V1:
            data = _migrate_v1_data(value)
            emit_metadata = True
        elif version == EDIT_CONTRACT_SCHEMA_V2:
            data = _canonical_v2_data(value)
            emit_metadata = "metadata" in data
        else:
            raise ValueError(f"unsupported edit contract schema_version: {version!r}")

        action = _contract_action(data["action"])
        required_nodes = _contract_nodes(data["required_nodes"], "required_nodes")
        forbidden_nodes = _contract_nodes(data["forbidden_nodes"], "forbidden_nodes")
        overlap = sorted(set(required_nodes) & set(forbidden_nodes))
        if overlap:
            raise ValueError(
                "edit contract required_nodes and forbidden_nodes overlap: "
                + ", ".join(overlap)
            )
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("edit contract metadata must be a mapping")
        return cls(
            action=action,
            required_nodes=required_nodes,
            forbidden_nodes=forbidden_nodes,
            mutation_budget=_contract_budget(data["mutation_budget"], action),
            rationale=_contract_rationale(data["rationale"]),
            metadata=copied_mapping(metadata),
            _emit_metadata=emit_metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "schema_version": EDIT_CONTRACT_SCHEMA_V2,
            "action": self.action,
            "required_nodes": list(self.required_nodes),
            "forbidden_nodes": list(self.forbidden_nodes),
            "mutation_budget": dict(self.mutation_budget),
            "rationale": deepcopy(self.rationale),
        }
        if self._emit_metadata or self.metadata:
            data["metadata"] = copied_mapping(self.metadata)
        return data


@dataclass(frozen=True)
class DesignStrategy:


    outer_loop_phase: str | None = None
    graph_ablation_mode: str | None = None
    layout_plan: Mapping[str, Any] | None = None
    search_schedule: Mapping[str, Any] | None = None
    edit_contract: EditContract | None = None
    last_contract_response: ContractResponse | None = None
    extensions: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str | None = "astevolve.strategy.v1"
    _emit_schema_version: bool = field(default=True, repr=False, compare=False)
    _present_fields: FrozenSet[str] = field(default_factory=frozenset, repr=False, compare=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DesignStrategy":
        data = copied_mapping(value)
        known = {
            "schema_version",
            "outer_loop_phase",
            "graph_ablation_mode",
            "layout_plan",
            "search_schedule",
            "edit_contract",
            "last_contract_response",
        }
        return cls(
            outer_loop_phase=(
                None
                if data.get("outer_loop_phase") is None
                else str(data["outer_loop_phase"])
            ),
            graph_ablation_mode=(
                None
                if data.get("graph_ablation_mode") is None
                else str(data["graph_ablation_mode"])
            ),
            layout_plan=(
                None
                if data.get("layout_plan") is None
                else copied_mapping(data.get("layout_plan"))
            ),
            search_schedule=(
                None
                if data.get("search_schedule") is None
                else copied_mapping(data.get("search_schedule"))
            ),
            edit_contract=(
                EditContract.from_mapping(data.get("edit_contract"))
                if data.get("edit_contract") is not None
                else None
            ),
            last_contract_response=(
                ContractResponse.from_mapping(data.get("last_contract_response"))
                if data.get("last_contract_response") is not None
                else None
            ),
            extensions={key: item for key, item in data.items() if key not in known},
            schema_version=(
                str(data["schema_version"])
                if data.get("schema_version") is not None
                else None
            ),
            _emit_schema_version="schema_version" in data,
            _present_fields=frozenset(data),
        )

    def to_legacy_dict(self) -> Dict[str, Any]:


        data = copied_mapping(self.extensions)
        if self._emit_schema_version:
            data["schema_version"] = self.schema_version
        if "outer_loop_phase" in self._present_fields or self.outer_loop_phase is not None:
            data["outer_loop_phase"] = self.outer_loop_phase
        if "graph_ablation_mode" in self._present_fields or self.graph_ablation_mode is not None:
            data["graph_ablation_mode"] = self.graph_ablation_mode
        if "layout_plan" in self._present_fields or self.layout_plan is not None:
            data["layout_plan"] = (
                copied_mapping(self.layout_plan) if self.layout_plan is not None else None
            )
        if "search_schedule" in self._present_fields or self.search_schedule is not None:
            data["search_schedule"] = (
                copied_mapping(self.search_schedule)
                if self.search_schedule is not None
                else None
            )
        if "last_contract_response" in self._present_fields or self.last_contract_response is not None:
            data["last_contract_response"] = (
                self.last_contract_response.to_dict()
                if self.last_contract_response is not None
                else None
            )
        if "edit_contract" in self._present_fields or self.edit_contract is not None:
            data["edit_contract"] = (
                self.edit_contract.to_dict() if self.edit_contract is not None else None
            )
        return data
