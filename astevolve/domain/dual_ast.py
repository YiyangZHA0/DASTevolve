

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


RESIDUE_SELECTOR_VERSION = "astevolve.residue_selector.v1"
INVARIANT_SPEC_VERSION = "astevolve.invariant_spec.v1"
ALLOWED_ACTION_VERSION = "astevolve.allowed_action.v1"
STRUCTURAL_NODE_VERSION = "astevolve.structural_node.v1"
EVALUATOR_BINDING_VERSION = "astevolve.evaluator_binding.v1"
FUNCTIONAL_NODE_VERSION = "astevolve.functional_node.v1"
MAPPING_EDGE_VERSION = "astevolve.mapping_edge.v1"
EXECUTABLE_DUAL_AST_VERSION = "astevolve.executable_dual_ast.v1"

STRUCTURAL_NODE_KINDS = frozenset({"editable", "frozen"})
FUNCTIONAL_NODE_KINDS = frozenset({"objective", "hard_constraint"})
FUNCTIONAL_NODE_STATES = frozenset({"positive", "negative", "preserve"})
OBJECTIVE_DIRECTIONS = frozenset({"maximize", "minimize"})
MISSING_POLICIES = frozenset({"fail", "abstain"})


class DualASTSchemaError(ValueError):
    pass


def _strict_mapping(
    value: Any,
    *,
    name: str,
    allowed: frozenset[str],
    required: Optional[frozenset[str]] = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DualASTSchemaError(f"{name} must be a mapping")
    keys = {str(key) for key in value}
    unknown = sorted(keys - allowed)
    if unknown:
        raise DualASTSchemaError(
            f"{name} contains unknown field(s): {', '.join(unknown)}"
        )
    missing = sorted((required or allowed) - keys)
    if missing:
        raise DualASTSchemaError(
            f"{name} is missing required field(s): {', '.join(missing)}"
        )
    return value


def _require_version(value: Any, expected: str, *, name: str) -> None:
    if value != expected:
        raise DualASTSchemaError(
            f"unsupported {name} schema_version {value!r}; expected {expected!r}"
        )


def _nonempty(value: Any, *, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise DualASTSchemaError(f"{name} must be non-empty")
    return text


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DualASTSchemaError(f"{name} must be a sequence")
    return value


def _normalize_spans(value: Any) -> Tuple[Tuple[int, int], ...]:
    raw_spans = _sequence(value, name="spans")
    if not raw_spans:
        raise DualASTSchemaError("spans must contain at least one half-open span")
    spans = []
    previous_end: Optional[int] = None
    for index, raw_span in enumerate(raw_spans):
        span = _sequence(raw_span, name=f"spans[{index}]")
        if len(span) != 2:
            raise DualASTSchemaError(
                f"spans[{index}] must contain exactly start and end"
            )
        start, end = span
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise DualASTSchemaError("span boundaries must be integer indices")
        if start < 0 or end < 0:
            raise DualASTSchemaError("span boundaries must be non-negative")
        if start >= end:
            raise DualASTSchemaError(
                "each span must be a non-empty 0-based half-open interval"
            )
        if previous_end is not None:
            if start < spans[-1][0]:
                raise DualASTSchemaError("spans must be ordered by start index")
            if start < previous_end:
                raise DualASTSchemaError("spans overlap or are duplicated")
        spans.append((start, end))
        previous_end = end
    return tuple(spans)


@dataclass(frozen=True)
class ResidueSelector:


    chain_id: str
    spans: Tuple[Tuple[int, int], ...]
    schema_version: str = RESIDUE_SELECTOR_VERSION

    def __post_init__(self) -> None:
        _require_version(
            self.schema_version, RESIDUE_SELECTOR_VERSION, name="residue selector"
        )
        object.__setattr__(self, "chain_id", _nonempty(self.chain_id, name="chain_id"))
        object.__setattr__(self, "spans", _normalize_spans(self.spans))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResidueSelector":
        raw = _strict_mapping(
            value,
            name="residue selector",
            allowed=frozenset({"schema_version", "chain_id", "spans"}),
        )
        return cls(
            chain_id=raw["chain_id"],
            spans=raw["spans"],
            schema_version=raw["schema_version"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "chain_id": self.chain_id,
            "spans": [[start, end] for start, end in self.spans],
        }


@dataclass(frozen=True)
class InvariantSpec:


    kind: str
    schema_version: str = INVARIANT_SPEC_VERSION

    def __post_init__(self) -> None:
        _require_version(
            self.schema_version, INVARIANT_SPEC_VERSION, name="invariant spec"
        )
        if self.kind != "preserve_parent":
            raise DualASTSchemaError(
                "invariant kind must be exactly 'preserve_parent' in v1"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InvariantSpec":
        raw = _strict_mapping(
            value,
            name="invariant spec",
            allowed=frozenset({"schema_version", "kind"}),
        )
        return cls(kind=str(raw["kind"]), schema_version=raw["schema_version"])

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": self.schema_version, "kind": self.kind}


def _normalize_budget(value: Any) -> Tuple[int, int]:
    if isinstance(value, tuple) and len(value) == 2:
        minimum, maximum = value
    else:
        raw = _strict_mapping(
            value,
            name="action budget",
            allowed=frozenset({"min", "max"}),
        )
        minimum, maximum = raw["min"], raw["max"]
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, int)
        or not isinstance(maximum, int)
    ):
        raise DualASTSchemaError("action budget min and max must be integers")
    if minimum < 0 or maximum < 0:
        raise DualASTSchemaError("action budget values must be non-negative")
    if minimum > maximum:
        raise DualASTSchemaError("action budget min cannot exceed max")
    return (minimum, maximum)


@dataclass(frozen=True)
class AllowedAction:


    operator: str
    budget: Tuple[int, int]
    schema_version: str = ALLOWED_ACTION_VERSION

    def __post_init__(self) -> None:
        _require_version(
            self.schema_version, ALLOWED_ACTION_VERSION, name="allowed action"
        )
        object.__setattr__(self, "operator", _nonempty(self.operator, name="operator"))
        object.__setattr__(self, "budget", _normalize_budget(self.budget))

    @property
    def budget_min(self) -> int:
        return self.budget[0]

    @property
    def budget_max(self) -> int:
        return self.budget[1]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AllowedAction":
        raw = _strict_mapping(
            value,
            name="allowed action",
            allowed=frozenset({"schema_version", "operator", "budget"}),
        )
        return cls(
            operator=raw["operator"],
            budget=raw["budget"],
            schema_version=raw["schema_version"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operator": self.operator,
            "budget": {"min": self.budget_min, "max": self.budget_max},
        }


@dataclass(frozen=True)
class StructuralNodeSpec:


    node_id: str
    kind: str
    selector: ResidueSelector
    invariants: Tuple[InvariantSpec, ...] = ()
    allowed_actions: Tuple[AllowedAction, ...] = ()
    schema_version: str = STRUCTURAL_NODE_VERSION

    def __post_init__(self) -> None:
        _require_version(
            self.schema_version, STRUCTURAL_NODE_VERSION, name="structural node"
        )
        object.__setattr__(self, "node_id", _nonempty(self.node_id, name="node_id"))
        if self.kind not in STRUCTURAL_NODE_KINDS:
            raise DualASTSchemaError(
                f"structural node kind must be one of {sorted(STRUCTURAL_NODE_KINDS)}"
            )
        if not isinstance(self.selector, ResidueSelector):
            raise DualASTSchemaError("structural node selector must be ResidueSelector")
        invariants = tuple(self.invariants)
        actions = tuple(self.allowed_actions)
        if any(not isinstance(item, InvariantSpec) for item in invariants):
            raise DualASTSchemaError("structural node invariants must be InvariantSpec")
        if any(not isinstance(item, AllowedAction) for item in actions):
            raise DualASTSchemaError("structural node actions must be AllowedAction")
        invariant_kinds = [item.kind for item in invariants]
        action_operators = [item.operator for item in actions]
        if len(invariant_kinds) != len(set(invariant_kinds)):
            raise DualASTSchemaError("duplicate structural node invariant")
        if len(action_operators) != len(set(action_operators)):
            raise DualASTSchemaError("duplicate structural node allowed action")
        if self.kind == "editable":
            if not actions:
                raise DualASTSchemaError(
                    "editable structural node requires at least one allowed action"
                )
            if "preserve_parent" in invariant_kinds:
                raise DualASTSchemaError(
                    "editable structural node cannot declare preserve_parent"
                )
        else:
            if actions:
                raise DualASTSchemaError(
                    "frozen structural node cannot declare allowed actions"
                )
            if invariant_kinds != ["preserve_parent"]:
                raise DualASTSchemaError(
                    "frozen structural node requires one preserve_parent invariant"
                )
        object.__setattr__(self, "invariants", invariants)
        object.__setattr__(self, "allowed_actions", actions)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StructuralNodeSpec":
        raw = _strict_mapping(
            value,
            name="structural node",
            allowed=frozenset(
                {
                    "schema_version",
                    "node_id",
                    "kind",
                    "selector",
                    "invariants",
                    "allowed_actions",
                }
            ),
        )
        invariants = _sequence(raw["invariants"], name="invariants")
        actions = _sequence(raw["allowed_actions"], name="allowed_actions")
        return cls(
            node_id=raw["node_id"],
            kind=str(raw["kind"]),
            selector=ResidueSelector.from_mapping(raw["selector"]),
            invariants=tuple(InvariantSpec.from_mapping(item) for item in invariants),
            allowed_actions=tuple(AllowedAction.from_mapping(item) for item in actions),
            schema_version=raw["schema_version"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "node_id": self.node_id,
            "kind": self.kind,
            "selector": self.selector.to_dict(),
            "invariants": [item.to_dict() for item in self.invariants],
            "allowed_actions": [item.to_dict() for item in self.allowed_actions],
        }


@dataclass(frozen=True)
class EvaluatorBinding:


    evaluator_id: str
    term_name: str
    schema_version: str = EVALUATOR_BINDING_VERSION

    def __post_init__(self) -> None:
        _require_version(
            self.schema_version, EVALUATOR_BINDING_VERSION, name="evaluator binding"
        )
        object.__setattr__(
            self, "evaluator_id", _nonempty(self.evaluator_id, name="evaluator_id")
        )
        object.__setattr__(
            self, "term_name", _nonempty(self.term_name, name="term_name")
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluatorBinding":
        raw = _strict_mapping(
            value,
            name="evaluator binding",
            allowed=frozenset({"schema_version", "evaluator_id", "term_name"}),
        )
        return cls(
            evaluator_id=raw["evaluator_id"],
            term_name=raw["term_name"],
            schema_version=raw["schema_version"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluator_id": self.evaluator_id,
            "term_name": self.term_name,
        }


def _normalize_threshold(value: Any, *, required: bool) -> Optional[float]:
    if value is None:
        if required:
            raise DualASTSchemaError("hard_constraint functional node requires threshold")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DualASTSchemaError("threshold must be a finite numeric value or None")
    threshold = float(value)
    if not math.isfinite(threshold):
        raise DualASTSchemaError("threshold must be finite")
    return threshold


@dataclass(frozen=True)
class FunctionalNodeSpec:


    node_id: str
    kind: str
    state: str
    direction: str
    evaluator_binding: EvaluatorBinding
    threshold: Optional[float]
    missing_policy: str
    evidence_refs: Tuple[str, ...] = ()
    schema_version: str = FUNCTIONAL_NODE_VERSION

    def __post_init__(self) -> None:
        _require_version(
            self.schema_version, FUNCTIONAL_NODE_VERSION, name="functional node"
        )
        object.__setattr__(self, "node_id", _nonempty(self.node_id, name="node_id"))
        if self.kind not in FUNCTIONAL_NODE_KINDS:
            raise DualASTSchemaError(
                f"functional node kind must be one of {sorted(FUNCTIONAL_NODE_KINDS)}"
            )
        if self.state not in FUNCTIONAL_NODE_STATES:
            raise DualASTSchemaError(
                f"functional node state must be one of {sorted(FUNCTIONAL_NODE_STATES)}"
            )
        if self.direction not in OBJECTIVE_DIRECTIONS:
            raise DualASTSchemaError(
                f"functional node direction must be one of {sorted(OBJECTIVE_DIRECTIONS)}"
            )
        if self.missing_policy not in MISSING_POLICIES:
            raise DualASTSchemaError(
                f"missing_policy must be one of {sorted(MISSING_POLICIES)}"
            )
        if not isinstance(self.evaluator_binding, EvaluatorBinding):
            raise DualASTSchemaError(
                "functional node evaluator_binding must be EvaluatorBinding"
            )
        refs = tuple(_nonempty(item, name="evidence_ref") for item in self.evidence_refs)
        if len(refs) != len(set(refs)):
            raise DualASTSchemaError("duplicate functional node evidence_refs")
        object.__setattr__(
            self,
            "threshold",
            _normalize_threshold(
                self.threshold, required=self.kind == "hard_constraint"
            ),
        )
        object.__setattr__(self, "evidence_refs", refs)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FunctionalNodeSpec":
        raw = _strict_mapping(
            value,
            name="functional node",
            allowed=frozenset(
                {
                    "schema_version",
                    "node_id",
                    "kind",
                    "state",
                    "direction",
                    "evaluator_binding",
                    "threshold",
                    "missing_policy",
                    "evidence_refs",
                }
            ),
        )
        evidence_refs = _sequence(raw["evidence_refs"], name="evidence_refs")
        return cls(
            node_id=raw["node_id"],
            kind=str(raw["kind"]),
            state=str(raw["state"]),
            direction=str(raw["direction"]),
            evaluator_binding=EvaluatorBinding.from_mapping(
                raw["evaluator_binding"]
            ),
            threshold=raw["threshold"],
            missing_policy=str(raw["missing_policy"]),
            evidence_refs=tuple(evidence_refs),
            schema_version=raw["schema_version"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "node_id": self.node_id,
            "kind": self.kind,
            "state": self.state,
            "direction": self.direction,
            "evaluator_binding": self.evaluator_binding.to_dict(),
            "threshold": self.threshold,
            "missing_policy": self.missing_policy,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class MappingEdgeSpec:


    edge_id: str
    functional_node_id: str
    structural_node_id: str
    action_operator: str
    evidence_refs: Tuple[str, ...] = ()
    relation: str = "realizes"
    schema_version: str = MAPPING_EDGE_VERSION

    def __post_init__(self) -> None:
        _require_version(
            self.schema_version, MAPPING_EDGE_VERSION, name="mapping edge"
        )
        object.__setattr__(self, "edge_id", _nonempty(self.edge_id, name="edge_id"))
        object.__setattr__(
            self,
            "functional_node_id",
            _nonempty(self.functional_node_id, name="functional_node_id"),
        )
        object.__setattr__(
            self,
            "structural_node_id",
            _nonempty(self.structural_node_id, name="structural_node_id"),
        )
        object.__setattr__(
            self,
            "action_operator",
            _nonempty(self.action_operator, name="action_operator"),
        )
        if self.relation != "realizes":
            raise DualASTSchemaError(
                "mapping edge relation must be exactly 'realizes'"
            )
        refs = tuple(_nonempty(item, name="evidence_ref") for item in self.evidence_refs)
        if len(refs) != len(set(refs)):
            raise DualASTSchemaError("duplicate mapping edge evidence_refs")
        object.__setattr__(self, "evidence_refs", refs)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MappingEdgeSpec":
        allowed = frozenset(
            {
                "schema_version",
                "edge_id",
                "relation",
                "functional_node_id",
                "structural_node_id",
                "action_operator",
                "evidence_refs",
            }
        )
        raw = _strict_mapping(
            value,
            name="mapping edge",
            allowed=allowed,
            required=frozenset(
                {
                    "schema_version",
                    "edge_id",
                    "functional_node_id",
                    "structural_node_id",
                    "action_operator",
                }
            ),
        )
        evidence_refs = _sequence(
            raw.get("evidence_refs", ()), name="evidence_refs"
        )
        return cls(
            edge_id=raw["edge_id"],
            relation=str(raw.get("relation", "realizes")),
            functional_node_id=raw["functional_node_id"],
            structural_node_id=raw["structural_node_id"],
            action_operator=raw["action_operator"],
            evidence_refs=tuple(evidence_refs),
            schema_version=raw["schema_version"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "edge_id": self.edge_id,
            "relation": self.relation,
            "functional_node_id": self.functional_node_id,
            "structural_node_id": self.structural_node_id,
            "action_operator": self.action_operator,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class ExecutableDualAST:


    ast_id: str
    revision: int
    structural_nodes: Tuple[StructuralNodeSpec, ...]
    functional_nodes: Tuple[FunctionalNodeSpec, ...]
    mapping_edges: Tuple[MappingEdgeSpec, ...]
    schema_version: str = EXECUTABLE_DUAL_AST_VERSION

    def __post_init__(self) -> None:
        _require_version(
            self.schema_version,
            EXECUTABLE_DUAL_AST_VERSION,
            name="executable dual AST",
        )
        object.__setattr__(self, "ast_id", _nonempty(self.ast_id, name="ast_id"))
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 1
        ):
            raise DualASTSchemaError("executable dual AST revision must be an integer >= 1")

        structural_nodes = tuple(self.structural_nodes)
        functional_nodes = tuple(self.functional_nodes)
        mapping_edges = tuple(self.mapping_edges)
        if not structural_nodes:
            raise DualASTSchemaError(
                "executable dual AST requires at least one structural node"
            )
        if not functional_nodes:
            raise DualASTSchemaError(
                "executable dual AST requires at least one functional node"
            )
        if any(not isinstance(item, StructuralNodeSpec) for item in structural_nodes):
            raise DualASTSchemaError(
                "structural_nodes must contain StructuralNodeSpec values"
            )
        if any(not isinstance(item, FunctionalNodeSpec) for item in functional_nodes):
            raise DualASTSchemaError(
                "functional_nodes must contain FunctionalNodeSpec values"
            )
        if any(not isinstance(item, MappingEdgeSpec) for item in mapping_edges):
            raise DualASTSchemaError(
                "mapping_edges must contain MappingEdgeSpec values"
            )

        structural_ids = [item.node_id for item in structural_nodes]
        functional_ids = [item.node_id for item in functional_nodes]
        edge_ids = [item.edge_id for item in mapping_edges]
        if len(structural_ids) != len(set(structural_ids)):
            raise DualASTSchemaError("duplicate structural node_id")
        if len(functional_ids) != len(set(functional_ids)):
            raise DualASTSchemaError("duplicate functional node_id")
        if len(edge_ids) != len(set(edge_ids)):
            raise DualASTSchemaError("duplicate mapping edge_id")

        structural_by_id = {item.node_id: item for item in structural_nodes}
        functional_id_set = set(functional_ids)
        semantic_edges = set()
        covered_functional = set()
        for edge in mapping_edges:
            if edge.functional_node_id not in functional_id_set:
                raise DualASTSchemaError(
                    f"mapping edge {edge.edge_id!r} has dangling functional node reference"
                )
            target = structural_by_id.get(edge.structural_node_id)
            if target is None:
                raise DualASTSchemaError(
                    f"mapping edge {edge.edge_id!r} has dangling structural node reference"
                )
            if target.kind == "frozen":
                raise DualASTSchemaError(
                    f"mapping edge {edge.edge_id!r} cannot target frozen structural node"
                )
            allowed_operators = {
                action.operator for action in target.allowed_actions
            }
            if edge.action_operator not in allowed_operators:
                raise DualASTSchemaError(
                    f"mapping edge {edge.edge_id!r} operator is not an allowed action "
                    f"of structural node {target.node_id!r}"
                )
            semantic_key = (
                edge.functional_node_id,
                edge.structural_node_id,
                edge.action_operator,
            )
            if semantic_key in semantic_edges:
                raise DualASTSchemaError("duplicate semantic mapping edge")
            semantic_edges.add(semantic_key)
            covered_functional.add(edge.functional_node_id)

        uncovered = sorted(functional_id_set - covered_functional)
        if uncovered:
            raise DualASTSchemaError(
                "every executable functional node requires at least one mapping edge; "
                "uncovered: " + ", ".join(uncovered)
            )

        object.__setattr__(self, "structural_nodes", structural_nodes)
        object.__setattr__(self, "functional_nodes", functional_nodes)
        object.__setattr__(self, "mapping_edges", mapping_edges)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExecutableDualAST":
        raw = _strict_mapping(
            value,
            name="executable dual AST",
            allowed=frozenset(
                {
                    "schema_version",
                    "ast_id",
                    "revision",
                    "structural_nodes",
                    "functional_nodes",
                    "mapping_edges",
                }
            ),
        )
        structural = _sequence(raw["structural_nodes"], name="structural_nodes")
        functional = _sequence(raw["functional_nodes"], name="functional_nodes")
        edges = _sequence(raw["mapping_edges"], name="mapping_edges")
        return cls(
            ast_id=raw["ast_id"],
            revision=raw["revision"],
            structural_nodes=tuple(
                StructuralNodeSpec.from_mapping(item) for item in structural
            ),
            functional_nodes=tuple(
                FunctionalNodeSpec.from_mapping(item) for item in functional
            ),
            mapping_edges=tuple(MappingEdgeSpec.from_mapping(item) for item in edges),
            schema_version=raw["schema_version"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ast_id": self.ast_id,
            "revision": self.revision,
            "structural_nodes": [item.to_dict() for item in self.structural_nodes],
            "functional_nodes": [item.to_dict() for item in self.functional_nodes],
            "mapping_edges": [item.to_dict() for item in self.mapping_edges],
        }


__all__ = [
    "ALLOWED_ACTION_VERSION",
    "EVALUATOR_BINDING_VERSION",
    "EXECUTABLE_DUAL_AST_VERSION",
    "FUNCTIONAL_NODE_KINDS",
    "FUNCTIONAL_NODE_STATES",
    "FUNCTIONAL_NODE_VERSION",
    "INVARIANT_SPEC_VERSION",
    "MAPPING_EDGE_VERSION",
    "MISSING_POLICIES",
    "OBJECTIVE_DIRECTIONS",
    "RESIDUE_SELECTOR_VERSION",
    "STRUCTURAL_NODE_KINDS",
    "STRUCTURAL_NODE_VERSION",
    "AllowedAction",
    "DualASTSchemaError",
    "ExecutableDualAST",
    "EvaluatorBinding",
    "FunctionalNodeSpec",
    "InvariantSpec",
    "MappingEdgeSpec",
    "ResidueSelector",
    "StructuralNodeSpec",
]
