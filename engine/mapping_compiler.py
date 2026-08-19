

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

from astevolve.domain.dual_ast import ExecutableDualAST

from .mapping_execution import (
    COMPILED_MAPPING_ACTION_VERSION,
    COMPILED_MEASUREMENT_VERSION,
)
from .node_compiler import (
    EXECUTABLE_NODE_SET_VERSION,
    ExecutableNodePlan,
    compile_executable_ast_nodes,
)


MAPPING_PLAN_VERSION = "astevolve.executable_mapping_plan.v1"
EFFECTIVE_MAPPING_SCHEDULE_VERSION = "astevolve.effective_mapping_schedule.v1"
MAPPING_EXECUTION_MODES = frozenset({"full", "flat_mask"})


class MappingCompileError(ValueError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(domain: str, value: Any) -> str:
    payload = domain.encode("utf-8") + b"\0" + _canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ActionSpec:


    ast_id: str
    ast_revision: int
    edge_id: str
    functional_node_id: str
    structural_node_id: str
    action_id: str
    measurement_id: str
    operator: str
    chain_id: str
    compiled_segment_name: str
    compiled_segment_kind: str
    legal_positions: Tuple[int, ...]
    budget_min: int
    budget_max: int
    evidence_refs: Tuple[str, ...]

    def to_projector_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": COMPILED_MAPPING_ACTION_VERSION,
            "ast_id": self.ast_id,
            "ast_revision": self.ast_revision,
            "edge_id": self.edge_id,
            "functional_node_id": self.functional_node_id,
            "structural_node_id": self.structural_node_id,
            "action_id": self.action_id,
            "measurement_id": self.measurement_id,
            "operator": self.operator,
            "chain_id": self.chain_id,
            "legal_positions": list(self.legal_positions),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.to_projector_dict(),
            "measurement_id": self.measurement_id,
            "compiled_segment_name": self.compiled_segment_name,
            "compiled_segment_kind": self.compiled_segment_kind,
            "budget": {"min": self.budget_min, "max": self.budget_max},
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class MeasurementSpec:


    ast_id: str
    ast_revision: int
    edge_id: str
    functional_node_id: str
    structural_node_id: str
    action_id: str
    measurement_id: str
    evaluator_id: str
    term_name: str
    kind: str
    state: str
    direction: str
    comparator: str
    threshold: Optional[float]
    aspiration_threshold: Optional[float]
    missing_policy: str
    evidence_refs: Tuple[str, ...]

    def to_projector_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": COMPILED_MEASUREMENT_VERSION,
            "ast_id": self.ast_id,
            "ast_revision": self.ast_revision,
            "edge_id": self.edge_id,
            "functional_node_id": self.functional_node_id,
            "structural_node_id": self.structural_node_id,
            "action_id": self.action_id,
            "measurement_id": self.measurement_id,
            "evaluator_id": self.evaluator_id,
            "term_name": self.term_name,
            "state": self.state,
            "direction": self.direction,
            "threshold": self.threshold,
            "missing_policy": self.missing_policy,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.to_projector_dict(),
            "measurement_id": self.measurement_id,
            "kind": self.kind,
            "state": self.state,
            "comparator": self.comparator,
            "aspiration_threshold": self.aspiration_threshold,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class ExecutableMappingPlan:


    enabled: bool
    execution_mode: str
    ast_id: str
    ast_revision: int
    action_specs: Tuple[ActionSpec, ...]
    measurement_specs: Tuple[MeasurementSpec, ...]
    control_contract: Mapping[str, Any]
    attribution_hash: str
    schema_version: str = MAPPING_PLAN_VERSION

    @property
    def execution_enabled(self) -> bool:
        return bool(self.enabled and self.execution_mode == "full")

    def control_fingerprint(self, *, seed: Optional[int]) -> str:
        return _digest(
            "astevolve.mapping_control_fingerprint.v1",
            {
                "control_contract": self.control_contract,
                "seed": None if seed is None else int(seed),
            },
        )

    def to_dict(self, *, seed: Optional[int] = None) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "execution_mode": self.execution_mode,
            "execution_enabled": self.execution_enabled,
            "ast_id": self.ast_id,
            "ast_revision": self.ast_revision,
            "action_specs": [item.to_dict() for item in self.action_specs],
            "measurement_specs": [item.to_dict() for item in self.measurement_specs],
            "control_contract": json.loads(_canonical_json(self.control_contract)),
            "control_fingerprint": self.control_fingerprint(seed=seed),
            "attribution_hash": self.attribution_hash,
        }

    def action_for_step(self, step: int) -> Optional[ActionSpec]:
        if not self.execution_enabled or not self.action_specs:
            return None
        return self.action_specs[int(step) % len(self.action_specs)]

    def measurement_for_action(self, action_id: str) -> MeasurementSpec:
        matches = [
            item for item in self.measurement_specs if item.action_id == str(action_id)
        ]
        if len(matches) != 1:
            raise MappingCompileError(
                f"action {action_id!r} must resolve to exactly one measurement; "
                f"resolved {len(matches)}"
            )
        return matches[0]


def validate_executable_mapping_plan(
    mapping_plan: ExecutableMappingPlan,
) -> ExecutableMappingPlan:


    if mapping_plan.schema_version != MAPPING_PLAN_VERSION:
        raise MappingCompileError("executable mapping plan schema version mismatch")
    if mapping_plan.execution_mode not in MAPPING_EXECUTION_MODES:
        raise MappingCompileError("executable mapping plan execution mode invalid")
    if mapping_plan.enabled:
        if not mapping_plan.ast_id or mapping_plan.ast_revision < 1:
            raise MappingCompileError("enabled mapping plan has invalid AST identity")
        if not mapping_plan.action_specs or not mapping_plan.measurement_specs:
            raise MappingCompileError("enabled mapping plan has no executable edges")
    elif mapping_plan.action_specs or mapping_plan.measurement_specs:
        raise MappingCompileError("disabled mapping plan contains edge projections")

    action_by_id: Dict[str, ActionSpec] = {}
    edge_ids = set()
    for action in mapping_plan.action_specs:
        if action.action_id in action_by_id:
            raise MappingCompileError("mapping plan contains duplicate action ids")
        if action.edge_id in edge_ids:
            raise MappingCompileError("mapping plan contains duplicate edge ids")
        action_by_id[action.action_id] = action
        edge_ids.add(action.edge_id)
        if action.ast_id != mapping_plan.ast_id or (
            action.ast_revision != mapping_plan.ast_revision
        ):
            raise MappingCompileError("mapping action AST identity mismatch")
        if not action.legal_positions or tuple(sorted(set(action.legal_positions))) != (
            action.legal_positions
        ):
            raise MappingCompileError("mapping action legal positions invalid")
        if (
            action.budget_min < 1
            or action.budget_min > action.budget_max
            or action.budget_min > len(action.legal_positions)
        ):
            raise MappingCompileError("mapping action budget invalid")

    measurement_by_action: Dict[str, MeasurementSpec] = {}
    for measurement in mapping_plan.measurement_specs:
        if measurement.action_id in measurement_by_action:
            raise MappingCompileError(
                "mapping plan contains duplicate measurements for an action"
            )
        action = action_by_id.get(measurement.action_id)
        if action is None:
            raise MappingCompileError("mapping measurement has no declared action")
        measurement_by_action[measurement.action_id] = measurement
        for field in (
            "ast_id",
            "ast_revision",
            "edge_id",
            "functional_node_id",
            "structural_node_id",
            "action_id",
            "measurement_id",
        ):
            if getattr(measurement, field) != getattr(action, field):
                raise MappingCompileError(
                    f"mapping action/measurement {field} mismatch"
                )
    if set(measurement_by_action) != set(action_by_id):
        raise MappingCompileError("mapping plan action/measurement coverage mismatch")

    attribution_payload: Any
    if mapping_plan.enabled:
        attribution_payload = {
            "ast_id": mapping_plan.ast_id,
            "ast_revision": mapping_plan.ast_revision,
            "actions": [item.to_dict() for item in mapping_plan.action_specs],
            "measurements": [
                item.to_dict() for item in mapping_plan.measurement_specs
            ],
        }
    else:
        attribution_payload = []
    expected_hash = _digest(
        "astevolve.mapping_attribution.v1", attribution_payload
    )
    if mapping_plan.attribution_hash != expected_hash:
        raise MappingCompileError("executable mapping plan attribution hash mismatch")
    return mapping_plan


@dataclass(frozen=True)
class DisabledMappingAction:
    edge_id: str
    functional_node_id: str
    structural_node_id: str
    action_id: str
    measurement_id: str
    operator: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "functional_node_id": self.functional_node_id,
            "structural_node_id": self.structural_node_id,
            "action_id": self.action_id,
            "measurement_id": self.measurement_id,
            "operator": self.operator,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EffectiveMappingSchedule:


    enabled: bool
    execution_mode: str
    ast_id: str
    ast_revision: int
    mapping_attribution_hash: str
    active_action_specs: Tuple[ActionSpec, ...]
    active_measurement_specs: Tuple[MeasurementSpec, ...]
    disabled_actions: Tuple[DisabledMappingAction, ...]
    functional_action_coverage: Mapping[str, Tuple[str, ...]]
    functional_measurement_coverage: Mapping[str, Tuple[str, ...]]
    schedule_hash: str
    schema_version: str = EFFECTIVE_MAPPING_SCHEDULE_VERSION

    @property
    def execution_enabled(self) -> bool:
        return bool(self.enabled and self.execution_mode == "full")

    def action_for_step(self, step: int) -> Optional[ActionSpec]:
        if not self.execution_enabled or not self.active_action_specs:
            return None
        return self.active_action_specs[int(step) % len(self.active_action_specs)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "execution_mode": self.execution_mode,
            "execution_enabled": self.execution_enabled,
            "ast_id": self.ast_id,
            "ast_revision": self.ast_revision,
            "mapping_attribution_hash": self.mapping_attribution_hash,
            "active_action_specs": [
                item.to_dict() for item in self.active_action_specs
            ],
            "active_measurement_specs": [
                item.to_dict() for item in self.active_measurement_specs
            ],
            "disabled_actions": [
                item.to_dict() for item in self.disabled_actions
            ],
            "functional_action_coverage": {
                node_id: list(action_ids)
                for node_id, action_ids in sorted(
                    self.functional_action_coverage.items()
                )
            },
            "functional_measurement_coverage": {
                node_id: list(measurement_ids)
                for node_id, measurement_ids in sorted(
                    self.functional_measurement_coverage.items()
                )
            },
            "schedule_hash": self.schedule_hash,
        }


@dataclass(frozen=True)
class ExecutableDualASTCompilation:


    ast: Optional[ExecutableDualAST]
    node_plan: ExecutableNodePlan
    mapping_plan: ExecutableMappingPlan

    @property
    def enabled(self) -> bool:
        return self.ast is not None

    def to_dict(self, *, seed: Optional[int] = None) -> Dict[str, Any]:
        return {
            "ast": self.ast.to_dict() if self.ast is not None else None,
            "node_plan": self.node_plan.to_dict(),
            "mapping_plan": self.mapping_plan.to_dict(seed=seed),
        }


def compile_effective_mapping_schedule(
    mapping_plan: ExecutableMappingPlan,
    *,
    node_policies: Mapping[str, Any],
) -> EffectiveMappingSchedule:


    validate_executable_mapping_plan(mapping_plan)
    active: List[ActionSpec] = []
    disabled: List[DisabledMappingAction] = []
    for action in mapping_plan.action_specs:
        if not mapping_plan.execution_enabled:
            reason = "flat_mask_control"
        else:
            raw_policy = node_policies.get(action.compiled_segment_name)
            policy = raw_policy if isinstance(raw_policy, Mapping) else {}
            raw_ops = policy.get("mutation_ops")
            ops = raw_ops if isinstance(raw_ops, Mapping) else {}
            try:
                weight = float(ops.get(action.operator, 0.0))
            except (TypeError, ValueError):
                weight = 0.0
            reason = "" if weight > 0.0 else "operator_not_enabled_by_effective_policy"
        if reason:
            disabled.append(
                DisabledMappingAction(
                    edge_id=action.edge_id,
                    functional_node_id=action.functional_node_id,
                    structural_node_id=action.structural_node_id,
                    action_id=action.action_id,
                    measurement_id=action.measurement_id,
                    operator=action.operator,
                    reason=reason,
                )
            )
        else:
            active.append(action)

    active_ids = {item.action_id for item in active}
    active_measurements = tuple(
        item for item in mapping_plan.measurement_specs if item.action_id in active_ids
    )
    declared_functional = sorted(
        {item.functional_node_id for item in mapping_plan.measurement_specs}
    )
    action_coverage = {
        functional_id: tuple(
            item.action_id
            for item in active
            if item.functional_node_id == functional_id
        )
        for functional_id in declared_functional
    }
    measurement_coverage = {
        functional_id: tuple(
            dict.fromkeys(
                item.measurement_id
                for item in mapping_plan.measurement_specs
                if item.functional_node_id == functional_id
            )
        )
        for functional_id in declared_functional
    }
    if mapping_plan.execution_enabled:
        objective_ids = {
            item.functional_node_id
            for item in mapping_plan.measurement_specs
            if item.kind == "objective"
        }
        uncovered_objectives = sorted(
            node_id for node_id in objective_ids if not action_coverage[node_id]
        )
        if uncovered_objectives:
            raise MappingCompileError(
                "effective mapping schedule leaves objective node(s) without an "
                "active action: " + ", ".join(uncovered_objectives)
            )
        if not active:
            raise MappingCompileError(
                "effective mapping schedule contains no active actions"
            )
    payload = {
        "schema_version": EFFECTIVE_MAPPING_SCHEDULE_VERSION,
        "enabled": mapping_plan.enabled,
        "execution_mode": mapping_plan.execution_mode,
        "execution_enabled": mapping_plan.execution_enabled,
        "ast_id": mapping_plan.ast_id,
        "ast_revision": mapping_plan.ast_revision,
        "mapping_attribution_hash": mapping_plan.attribution_hash,
        "active_action_specs": [item.to_dict() for item in active],
        "active_measurement_specs": [
            item.to_dict() for item in active_measurements
        ],
        "disabled_actions": [item.to_dict() for item in disabled],
        "functional_action_coverage": {
            node_id: list(action_ids)
            for node_id, action_ids in sorted(action_coverage.items())
        },
        "functional_measurement_coverage": {
            node_id: list(measurement_ids)
            for node_id, measurement_ids in sorted(measurement_coverage.items())
        },
    }
    schedule = EffectiveMappingSchedule(
        enabled=mapping_plan.enabled,
        execution_mode=mapping_plan.execution_mode,
        ast_id=mapping_plan.ast_id,
        ast_revision=mapping_plan.ast_revision,
        mapping_attribution_hash=mapping_plan.attribution_hash,
        active_action_specs=tuple(active),
        active_measurement_specs=active_measurements,
        disabled_actions=tuple(disabled),
        functional_action_coverage=action_coverage,
        functional_measurement_coverage=measurement_coverage,
        schedule_hash=_digest("astevolve.effective_mapping_schedule.v1", payload),
    )
    return validate_effective_mapping_schedule(mapping_plan, schedule)


def validate_effective_mapping_schedule(
    mapping_plan: ExecutableMappingPlan,
    schedule: EffectiveMappingSchedule,
) -> EffectiveMappingSchedule:


    validate_executable_mapping_plan(mapping_plan)
    if (
        schedule.schema_version != EFFECTIVE_MAPPING_SCHEDULE_VERSION
        or schedule.enabled != mapping_plan.enabled
        or schedule.execution_mode != mapping_plan.execution_mode
        or schedule.execution_enabled != mapping_plan.execution_enabled
        or schedule.ast_id != mapping_plan.ast_id
        or schedule.ast_revision != mapping_plan.ast_revision
        or schedule.mapping_attribution_hash != mapping_plan.attribution_hash
    ):
        raise MappingCompileError("effective mapping schedule header/plan mismatch")

    plan_actions = {item.action_id: item for item in mapping_plan.action_specs}
    if len(plan_actions) != len(mapping_plan.action_specs):
        raise MappingCompileError("mapping plan contains duplicate action ids")
    active_ids = [item.action_id for item in schedule.active_action_specs]
    disabled_ids = [item.action_id for item in schedule.disabled_actions]
    if len(active_ids) != len(set(active_ids)) or len(disabled_ids) != len(
        set(disabled_ids)
    ):
        raise MappingCompileError("effective mapping schedule repeats an action")
    if set(active_ids).intersection(disabled_ids):
        raise MappingCompileError("effective mapping schedule action is both active and disabled")
    if set(active_ids).union(disabled_ids) != set(plan_actions):
        raise MappingCompileError("effective mapping schedule action coverage is not exhaustive")
    expected_active = tuple(
        item for item in mapping_plan.action_specs if item.action_id in set(active_ids)
    )
    if tuple(item.to_dict() for item in schedule.active_action_specs) != tuple(
        item.to_dict() for item in expected_active
    ):
        raise MappingCompileError("effective mapping schedule active actions mismatch plan")
    disabled_by_id = {item.action_id: item for item in schedule.disabled_actions}
    expected_disabled_ids = [
        item.action_id
        for item in mapping_plan.action_specs
        if item.action_id not in set(active_ids)
    ]
    if disabled_ids != expected_disabled_ids:
        raise MappingCompileError("effective mapping schedule disabled order mismatch plan")
    expected_reason = (
        "operator_not_enabled_by_effective_policy"
        if mapping_plan.execution_enabled
        else "flat_mask_control"
    )
    for action_id in expected_disabled_ids:
        action = plan_actions[action_id]
        disabled = disabled_by_id[action_id]
        expected = DisabledMappingAction(
            edge_id=action.edge_id,
            functional_node_id=action.functional_node_id,
            structural_node_id=action.structural_node_id,
            action_id=action.action_id,
            measurement_id=action.measurement_id,
            operator=action.operator,
            reason=expected_reason,
        )
        if disabled.to_dict() != expected.to_dict():
            raise MappingCompileError("effective mapping schedule disabled action mismatch plan")

    active_id_set = set(active_ids)
    expected_measurements = tuple(
        item
        for item in mapping_plan.measurement_specs
        if item.action_id in active_id_set
    )
    if tuple(item.to_dict() for item in schedule.active_measurement_specs) != tuple(
        item.to_dict() for item in expected_measurements
    ):
        raise MappingCompileError("effective mapping schedule measurements mismatch plan")
    declared_functional = sorted(
        {item.functional_node_id for item in mapping_plan.measurement_specs}
    )
    expected_action_coverage = {
        functional_id: tuple(
            item.action_id
            for item in expected_active
            if item.functional_node_id == functional_id
        )
        for functional_id in declared_functional
    }
    if mapping_plan.execution_enabled:
        if not expected_active:
            raise MappingCompileError(
                "effective mapping schedule contains no active actions"
            )
        objective_ids = {
            item.functional_node_id
            for item in mapping_plan.measurement_specs
            if item.kind == "objective"
        }
        uncovered_objectives = sorted(
            node_id
            for node_id in objective_ids
            if not expected_action_coverage[node_id]
        )
        if uncovered_objectives:
            raise MappingCompileError(
                "effective mapping schedule leaves objective node(s) without an "
                "active action: " + ", ".join(uncovered_objectives)
            )
    expected_measurement_coverage = {
        functional_id: tuple(
            dict.fromkeys(
                item.measurement_id
                for item in mapping_plan.measurement_specs
                if item.functional_node_id == functional_id
            )
        )
        for functional_id in declared_functional
    }
    if dict(schedule.functional_action_coverage) != expected_action_coverage:
        raise MappingCompileError("effective mapping schedule action coverage mismatch")
    if dict(schedule.functional_measurement_coverage) != expected_measurement_coverage:
        raise MappingCompileError("effective mapping schedule measurement coverage mismatch")

    payload = {
        "schema_version": EFFECTIVE_MAPPING_SCHEDULE_VERSION,
        "enabled": schedule.enabled,
        "execution_mode": schedule.execution_mode,
        "execution_enabled": schedule.execution_enabled,
        "ast_id": schedule.ast_id,
        "ast_revision": schedule.ast_revision,
        "mapping_attribution_hash": schedule.mapping_attribution_hash,
        "active_action_specs": [
            item.to_dict() for item in schedule.active_action_specs
        ],
        "active_measurement_specs": [
            item.to_dict() for item in schedule.active_measurement_specs
        ],
        "disabled_actions": [item.to_dict() for item in schedule.disabled_actions],
        "functional_action_coverage": {
            node_id: list(action_ids)
            for node_id, action_ids in sorted(
                schedule.functional_action_coverage.items()
            )
        },
        "functional_measurement_coverage": {
            node_id: list(measurement_ids)
            for node_id, measurement_ids in sorted(
                schedule.functional_measurement_coverage.items()
            )
        },
    }
    expected_hash = _digest("astevolve.effective_mapping_schedule.v1", payload)
    if schedule.schedule_hash != expected_hash:
        raise MappingCompileError("effective mapping schedule hash mismatch")
    return schedule


def _control_contract(node_plan: ExecutableNodePlan) -> Dict[str, Any]:


    return {
        "effective_masks": {
            chain: list(mask)
            for chain, mask in sorted(node_plan.effective_masks.items())
        },
        "effective_fixed_residues": {
            chain: {
                str(position): residue
                for position, residue in sorted(residues.items())
            }
            for chain, residues in sorted(node_plan.effective_fixed_residues.items())
        },
        "structural_capabilities": [
            {
                "structural_node_id": item.node_id,
                "chain_id": item.chain_id,
                "legal_positions": list(item.legal_positions),
                "allowed_operators": list(item.allowed_operators),
                "action_budgets": {
                    name: dict(budget)
                    for name, budget in sorted(item.action_budgets.items())
                },
            }
            for item in node_plan.structural_nodes
        ],
        "evaluator_capabilities": sorted(
            {
                (item.evaluator_id, item.term_name)
                for item in node_plan.measurement_intents
            }
        ),
    }


def _disabled_mapping_plan(
    node_plan: ExecutableNodePlan,
    *,
    execution_mode: str,
) -> ExecutableMappingPlan:
    return ExecutableMappingPlan(
        enabled=False,
        execution_mode=execution_mode,
        ast_id="",
        ast_revision=0,
        action_specs=(),
        measurement_specs=(),
        control_contract=_control_contract(node_plan),
        attribution_hash=_digest("astevolve.mapping_attribution.v1", []),
    )


def compile_executable_dual_ast(
    raw: Mapping[str, Any] | None,
    *,
    compiled: Mapping[str, Any],
    template_sequences: Mapping[str, str],
    base_masks: Mapping[str, Any],
    base_fixed_residues: Mapping[str, Mapping[int, str]],
    evaluator_capabilities: Mapping[str, Any] | None,
    execution_mode: str = "full",
) -> ExecutableDualASTCompilation:


    mode = str(execution_mode).strip().lower()
    if mode not in MAPPING_EXECUTION_MODES:
        raise MappingCompileError(
            f"unsupported mapping execution mode {execution_mode!r}; "
            f"expected one of {sorted(MAPPING_EXECUTION_MODES)}"
        )
    if raw is None:
        node_plan = compile_executable_ast_nodes(
            None,
            compiled=compiled,
            template_sequences=template_sequences,
            base_masks=base_masks,
            base_fixed_residues=base_fixed_residues,
            evaluator_capabilities=evaluator_capabilities,
        )
        return ExecutableDualASTCompilation(
            ast=None,
            node_plan=node_plan,
            mapping_plan=_disabled_mapping_plan(node_plan, execution_mode=mode),
        )

    ast = ExecutableDualAST.from_mapping(raw)
    node_plan = compile_executable_ast_nodes(
        {
            "schema_version": EXECUTABLE_NODE_SET_VERSION,
            "structural_nodes": [item.to_dict() for item in ast.structural_nodes],
            "functional_nodes": [item.to_dict() for item in ast.functional_nodes],
        },
        compiled=compiled,
        template_sequences=template_sequences,
        base_masks=base_masks,
        base_fixed_residues=base_fixed_residues,
        evaluator_capabilities=evaluator_capabilities,
    )
    structural = {item.node_id: item for item in node_plan.structural_nodes}
    measurements = {
        item.functional_node_id: item for item in node_plan.measurement_intents
    }
    functional = {item.node_id: item for item in ast.functional_nodes}
    action_specs = []
    measurement_specs = []
    runtime_keys: Dict[Tuple[str, str], str] = {}
    for edge in ast.mapping_edges:
        structural_contract = structural[edge.structural_node_id]
        measurement_intent = measurements[edge.functional_node_id]
        functional_spec = functional[edge.functional_node_id]
        if edge.action_operator not in structural_contract.allowed_operators:
            raise MappingCompileError(
                f"mapping edge {edge.edge_id!r} operator {edge.action_operator!r} "
                "has no executable capability after mask/invariant compilation"
            )
        if not structural_contract.legal_positions:
            raise MappingCompileError(
                f"mapping edge {edge.edge_id!r} targets no legal residue positions"
            )
        runtime_key = (
            structural_contract.compiled_segment_name,
            edge.action_operator,
        )
        if runtime_key in runtime_keys:
            raise MappingCompileError(
                "AST-01B v1 requires unambiguous (structural node, operator) routing; "
                f"edge {edge.edge_id!r} conflicts with {runtime_keys[runtime_key]!r}"
            )
        runtime_keys[runtime_key] = edge.edge_id
        budget = structural_contract.action_budgets[edge.action_operator]
        minimum_arity = {
            "block": 2,
            "swap": 2,
            "region_shuffle": 2,
        }.get(edge.action_operator, 1)
        legal_capacity = len(structural_contract.legal_positions)
        if int(budget["min"]) > legal_capacity:
            raise MappingCompileError(
                f"mapping edge {edge.edge_id!r} budget min {budget['min']} exceeds "
                f"legal position capacity {legal_capacity}"
            )
        if minimum_arity > legal_capacity:
            raise MappingCompileError(
                f"mapping edge {edge.edge_id!r} operator "
                f"{edge.action_operator!r} minimum arity {minimum_arity} exceeds "
                f"legal position capacity {legal_capacity}"
            )
        if int(budget["max"]) < minimum_arity:
            raise MappingCompileError(
                f"mapping edge {edge.edge_id!r} budget max {budget['max']} cannot "
                f"realize operator {edge.action_operator!r} minimum arity "
                f"{minimum_arity}"
            )
        fixed_arity = {"swap": 2}.get(edge.action_operator)
        if fixed_arity is not None and int(budget["min"]) > fixed_arity:
            raise MappingCompileError(
                f"mapping edge {edge.edge_id!r} budget min {budget['min']} cannot "
                f"realize operator {edge.action_operator!r} fixed arity "
                f"{fixed_arity}"
            )
        action_id = f"{ast.ast_id}@{ast.revision}:action:{edge.edge_id}"
        measurement_id = (
            f"{ast.ast_id}@{ast.revision}:functional-measurement:"
            f"{edge.functional_node_id}"
        )
        action_specs.append(
            ActionSpec(
                ast_id=ast.ast_id,
                ast_revision=ast.revision,
                edge_id=edge.edge_id,
                functional_node_id=edge.functional_node_id,
                structural_node_id=edge.structural_node_id,
                action_id=action_id,
                measurement_id=measurement_id,
                operator=edge.action_operator,
                chain_id=structural_contract.chain_id,
                compiled_segment_name=structural_contract.compiled_segment_name,
                compiled_segment_kind=structural_contract.compiled_segment_kind,
                legal_positions=structural_contract.legal_positions,
                budget_min=int(budget["min"]),
                budget_max=int(budget["max"]),
                evidence_refs=tuple(edge.evidence_refs),
            )
        )
        measurement_specs.append(
            MeasurementSpec(
                ast_id=ast.ast_id,
                ast_revision=ast.revision,
                edge_id=edge.edge_id,
                functional_node_id=edge.functional_node_id,
                structural_node_id=edge.structural_node_id,
                action_id=action_id,
                measurement_id=measurement_id,
                evaluator_id=measurement_intent.evaluator_id,
                term_name=measurement_intent.term_name,
                kind=functional_spec.kind,
                state=functional_spec.state,
                direction=measurement_intent.direction,
                comparator=measurement_intent.comparator,
                threshold=(
                    float(measurement_intent.gate["threshold"])
                    if measurement_intent.gate is not None
                    else None
                ),
                aspiration_threshold=(
                    float(functional_spec.threshold)
                    if functional_spec.kind == "objective"
                    and functional_spec.threshold is not None
                    else None
                ),
                missing_policy=measurement_intent.missing_policy,
                evidence_refs=tuple(
                    dict.fromkeys((*functional_spec.evidence_refs, *edge.evidence_refs))
                ),
            )
        )

    attribution_payload = {
        "ast_id": ast.ast_id,
        "ast_revision": ast.revision,
        "actions": [item.to_dict() for item in action_specs],
        "measurements": [item.to_dict() for item in measurement_specs],
    }
    mapping_plan = ExecutableMappingPlan(
        enabled=True,
        execution_mode=mode,
        ast_id=ast.ast_id,
        ast_revision=ast.revision,
        action_specs=tuple(action_specs),
        measurement_specs=tuple(measurement_specs),
        control_contract=_control_contract(node_plan),
        attribution_hash=_digest(
            "astevolve.mapping_attribution.v1", attribution_payload
        ),
    )
    validate_executable_mapping_plan(mapping_plan)
    return ExecutableDualASTCompilation(
        ast=ast,
        node_plan=node_plan,
        mapping_plan=mapping_plan,
    )


__all__ = [
    "EFFECTIVE_MAPPING_SCHEDULE_VERSION",
    "MAPPING_EXECUTION_MODES",
    "MAPPING_PLAN_VERSION",
    "ActionSpec",
    "ExecutableDualASTCompilation",
    "EffectiveMappingSchedule",
    "ExecutableMappingPlan",
    "MappingCompileError",
    "MeasurementSpec",
    "compile_effective_mapping_schedule",
    "compile_executable_dual_ast",
    "validate_effective_mapping_schedule",
    "validate_executable_mapping_plan",
]
