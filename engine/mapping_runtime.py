

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Tuple

from astevolve.search.mutation_move import MoveContractError, validate_move_contract

from .causal_flow import SequenceRecord
from .mapping_compiler import (
    EffectiveMappingSchedule,
    ExecutableMappingPlan,
    MappingCompileError,
    MeasurementSpec,
    validate_effective_mapping_schedule,
)
from .mapping_execution import (
    REALIZED_MAPPING_MOVE_VERSION,
    MappingExecutionError,
    MappingExecutionTrace,
    project_mapping_execution,
)


MAPPING_RUNTIME_VERSION = "astevolve.mapping_runtime.v1"


@dataclass(frozen=True)
class MappingRuntimeProjection:
    evaluator_report: Mapping[str, Any]
    traces: Tuple[MappingExecutionTrace, ...]
    final_measurements: Tuple[Mapping[str, Any], ...]
    artifact: Mapping[str, Any]


def _candidate_index(runtime: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    rows = [runtime.get("root_candidate"), *(runtime.get("candidates") or [])]
    index: Dict[str, Mapping[str, Any]] = {}
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        variant_id = str(item.get("variant_id") or "")
        if not variant_id:
            raise MappingExecutionError("selected_mapping_variant_id_missing")
        if variant_id in index:
            raise MappingExecutionError("selected_mapping_variant_id_duplicate")
        index[variant_id] = item
    return index


def _selected_lineage(runtime: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    selected = runtime.get("selected_candidate")
    if not isinstance(selected, Mapping):
        return []
    by_id = _candidate_index(runtime)
    selected_id = str(selected.get("variant_id") or "")
    indexed_selected = by_id.get(selected_id)
    if indexed_selected is None:
        raise MappingExecutionError("selected_mapping_candidate_unregistered")
    selected_sequences = selected.get("seqs")
    indexed_sequences = indexed_selected.get("seqs")
    if not isinstance(selected_sequences, Mapping) or not isinstance(
        indexed_sequences, Mapping
    ):
        raise MappingExecutionError("selected_mapping_sequences_missing")
    if SequenceRecord.create(selected_sequences).semantic_id != (
        SequenceRecord.create(indexed_sequences).semantic_id
    ):
        raise MappingExecutionError("selected_mapping_candidate_registry_mismatch")
    lineage: List[Mapping[str, Any]] = []
    current = indexed_selected
    seen = set()
    while True:
        variant_id = str(current.get("variant_id") or "")
        if not variant_id or variant_id in seen:
            raise MappingExecutionError("selected_mapping_lineage_invalid")
        seen.add(variant_id)
        lineage.append(current)
        parent_id = current.get("parent_id")
        if parent_id in (None, ""):
            break
        parent = by_id.get(str(parent_id))
        if parent is None:
            raise MappingExecutionError("selected_mapping_lineage_parent_missing")
        current = parent
    lineage.reverse()
    declared_root = runtime.get("root_candidate")
    if not isinstance(declared_root, Mapping):
        raise MappingExecutionError("selected_mapping_root_missing")
    declared_root_sequences = declared_root.get("seqs")
    lineage_root_sequences = lineage[0].get("seqs")
    if (
        str(lineage[0].get("variant_id") or "")
        != str(declared_root.get("variant_id") or "")
        or not isinstance(declared_root_sequences, Mapping)
        or not isinstance(lineage_root_sequences, Mapping)
        or SequenceRecord.create(declared_root_sequences).semantic_id
        != SequenceRecord.create(lineage_root_sequences).semantic_id
    ):
        raise MappingExecutionError("selected_mapping_lineage_root_mismatch")
    return lineage


def _validated_lineage_moves(
    runtime: Mapping[str, Any],
) -> List[Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]]:
    lineage = _selected_lineage(runtime)
    moves = []
    for parent, child in zip(lineage, lineage[1:]):
        parent_seqs = parent.get("seqs")
        child_seqs = child.get("seqs")
        move = child.get("move")
        if not isinstance(parent_seqs, Mapping) or not isinstance(child_seqs, Mapping):
            raise MappingExecutionError("selected_mapping_transition_missing")
        if not isinstance(move, Mapping) or str(move.get("outcome")) != "executed":
            raise MappingExecutionError("selected_mapping_action_missing")
        try:
            validate_move_contract(parent_seqs, child_seqs, move)
        except MoveContractError as error:
            raise MappingExecutionError(
                "selected_mapping_move_contract_invalid", str(error)
            ) from error
        moves.append((parent, child, move))
    return moves


def _realized_move(
    move: Mapping[str, Any],
    action: Any,
    *,
    evaluated_sequence_id: str,
) -> Dict[str, Any]:
    attribution = move.get("mapping_attribution")
    if not isinstance(attribution, Mapping):
        raise MappingExecutionError(
            "selected_mapping_attribution_missing",
            "full mapping selected a move without edge attribution",
        )
    identity_fields = {
        "ast_id",
        "ast_revision",
        "edge_id",
        "functional_node_id",
        "structural_node_id",
        "action_id",
        "measurement_id",
    }
    if set(attribution) != identity_fields:
        raise MappingExecutionError("selected_mapping_attribution_fields_invalid")
    for field in (
        "ast_id",
        "ast_revision",
        "edge_id",
        "functional_node_id",
        "structural_node_id",
        "action_id",
        "measurement_id",
    ):
        if attribution.get(field) != getattr(action, field):
            raise MappingExecutionError(
                f"selected_mapping_{field}_mismatch",
                f"expected {getattr(action, field)!r}, found "
                f"{attribution.get(field)!r}",
            )
    if str(move.get("op") or "") != action.operator:
        raise MappingExecutionError("selected_mapping_operator_mismatch")
    if str(move.get("chain_id") or "") != action.chain_id:
        raise MappingExecutionError("selected_mapping_chain_id_mismatch")
    if str(move.get("node") or "") != action.compiled_segment_name:
        raise MappingExecutionError("selected_mapping_compiled_segment_mismatch")
    target_nodes = [str(item) for item in move.get("target_nodes", []) or []]
    if target_nodes != [action.compiled_segment_name]:
        raise MappingExecutionError("selected_mapping_target_nodes_mismatch")
    mutation_plan = move.get("mutation_plan")
    if not isinstance(mutation_plan, Mapping):
        raise MappingExecutionError("selected_mapping_mutation_plan_missing")
    for field in (
        "ast_id",
        "ast_revision",
        "edge_id",
        "functional_node_id",
        "structural_node_id",
        "action_id",
        "measurement_id",
    ):
        if mutation_plan.get(field) != getattr(action, field):
            raise MappingExecutionError(
                f"selected_mapping_plan_{field}_mismatch"
            )
    if str(mutation_plan.get("node") or "") != action.compiled_segment_name:
        raise MappingExecutionError("selected_mapping_plan_segment_mismatch")
    if str(mutation_plan.get("mapping_execution") or "") != "full":
        raise MappingExecutionError("selected_mapping_plan_execution_mode_mismatch")
    op_weights = mutation_plan.get("op_weights")
    if not isinstance(op_weights, Mapping) or set(op_weights) != {action.operator}:
        raise MappingExecutionError("selected_mapping_plan_operator_mismatch")
    try:
        operator_weight = float(op_weights[action.operator])
    except (TypeError, ValueError):
        operator_weight = 0.0
    if not math.isfinite(operator_weight) or operator_weight <= 0.0:
        raise MappingExecutionError("selected_mapping_plan_operator_mismatch")
    expected_budget = {"min": action.budget_min, "max": action.budget_max}
    if mutation_plan.get("budget") != expected_budget:
        raise MappingExecutionError("selected_mapping_plan_budget_mismatch")
    allowed_budgets = mutation_plan.get("allowed_action_budgets")
    if allowed_budgets != {action.operator: expected_budget}:
        raise MappingExecutionError("selected_mapping_plan_budget_mismatch")
    if move.get("selected_action_budget") != expected_budget:
        raise MappingExecutionError("selected_mapping_selected_budget_mismatch")
    try:
        declared_legal = sorted(
            int(position) for position in mutation_plan.get("legal_positions", [])
        )
    except (TypeError, ValueError):
        raise MappingExecutionError("selected_mapping_plan_legal_positions_mismatch")
    if declared_legal != list(action.legal_positions):
        raise MappingExecutionError("selected_mapping_plan_legal_positions_mismatch")
    changes = [
        change
        for change in move.get("changes", []) or []
        if isinstance(change, Mapping)
    ]
    extra_chains = sorted(
        {
            str(change.get("chain_id") or "")
            for change in changes
            if str(change.get("chain_id") or "") != action.chain_id
        }
    )
    if extra_chains:
        raise MappingExecutionError(
            "realized_move_extra_chain",
            f"mapped action wrote undeclared chain(s): {', '.join(extra_chains)}",
        )
    wrong_change_nodes = sorted(
        {
            str(change.get("node") or "")
            for change in changes
            if str(change.get("node") or "") != action.compiled_segment_name
        }
    )
    if wrong_change_nodes:
        raise MappingExecutionError("selected_mapping_change_node_mismatch")
    actual_delta = move.get("actual_delta")
    actual_delta = actual_delta if isinstance(actual_delta, Mapping) else {}
    length_delta = actual_delta.get("length_delta_by_chain")
    length_delta = length_delta if isinstance(length_delta, Mapping) else {}
    nonzero_length = {
        str(chain): int(delta)
        for chain, delta in length_delta.items()
        if int(delta) != 0
    }
    if nonzero_length:
        raise MappingExecutionError(
            "realized_move_length_delta_forbidden",
            f"mapped substitution action changed length: {nonzero_length}",
        )
    positions = sorted(
        {int(change["position"]) for change in changes if change.get("position") is not None}
    )
    outside = sorted(set(positions) - set(action.legal_positions))
    if outside:
        raise MappingExecutionError(
            "realized_positions_outside_legal_positions",
            f"mapped action wrote illegal position(s): {outside}",
        )
    realized_count = len(changes)
    if realized_count < action.budget_min or realized_count > action.budget_max:
        raise MappingExecutionError(
            "realized_move_budget_violation",
            f"mapped action realized {realized_count} residue changes outside "
            f"declared budget [{action.budget_min}, {action.budget_max}]",
        )
    return {
        "schema_version": REALIZED_MAPPING_MOVE_VERSION,
        "ast_id": attribution.get("ast_id"),
        "ast_revision": attribution.get("ast_revision"),
        "edge_id": attribution.get("edge_id"),
        "functional_node_id": attribution.get("functional_node_id"),
        "structural_node_id": attribution.get("structural_node_id"),
        "action_id": attribution.get("action_id"),
        "measurement_id": attribution.get("measurement_id"),
        "operator": move.get("op"),
        "chain_id": move.get("chain_id"),
        "positions": positions,
        "evaluated_sequence_id": evaluated_sequence_id,
    }


def _validated_composite_actions(
    move: Mapping[str, Any],
    active_actions: Mapping[str, Any],
    *,
    evaluated_sequence_id: str,
) -> List[Tuple[Mapping[str, Any], Any, Dict[str, Any]]]:


    from astevolve.search.mapping_schedule_runtime import (
        validate_portfolio_mapping_components,
    )

    try:
        validated = validate_portfolio_mapping_components(
            move,
            mapping_actions=[
                action.to_dict()
                for _action_id, action in sorted(active_actions.items())
            ],
        )
    except (TypeError, ValueError) as error:
        raise MappingExecutionError(
            "selected_mapping_components_invalid", str(error)
        ) from error


    if (
        isinstance(validated, (str, bytes))
        or not isinstance(validated, (list, tuple))
        or not validated
        or not all(isinstance(component, Mapping) for component in validated)
    ):
        raise MappingExecutionError("selected_mapping_components_invalid")

    resolved: List[Tuple[Mapping[str, Any], Any, Dict[str, Any]]] = []
    for component in validated:
        attribution = component.get("mapping_attribution")
        if not isinstance(attribution, Mapping):
            raise MappingExecutionError("selected_mapping_attribution_missing")
        action_id = str(attribution.get("action_id") or "")
        action = active_actions.get(action_id)
        if action is None:
            raise MappingExecutionError(
                "selected_mapping_component_action_inactive", action_id
            )
        resolved.append(
            (
                component,
                action,
                _realized_move(
                    component,
                    action,
                    evaluated_sequence_id=evaluated_sequence_id,
                ),
            )
        )
    return resolved


def _unique_functional_measurements(
    mapping_plan: ExecutableMappingPlan,
) -> Tuple[MeasurementSpec, ...]:
    grouped: Dict[str, List[MeasurementSpec]] = {}
    for item in mapping_plan.measurement_specs:
        grouped.setdefault(item.functional_node_id, []).append(item)
    unique = []
    for functional_id, values in grouped.items():
        signatures = {
            (
                item.measurement_id,
                item.evaluator_id,
                item.term_name,
                item.kind,
                item.state,
                item.direction,
                item.threshold,
                item.aspiration_threshold,
                item.missing_policy,
            )
            for item in values
        }
        if len(signatures) != 1:
            raise MappingExecutionError(
                "functional_measurement_signature_conflict", functional_id
            )
        unique.append(values[0])
    return tuple(unique)


def _exact_term(
    report: Mapping[str, Any],
    measurement: MeasurementSpec,
) -> Mapping[str, Any] | None:
    terms = report.get("terms")
    if not isinstance(terms, list):
        raise MappingExecutionError("evaluator_report_terms_not_list")
    matches = [
        term
        for term in terms
        if isinstance(term, Mapping)
        and str(term.get("provider") or "") == measurement.evaluator_id
        and str(term.get("name") or "") == measurement.term_name
        and str(term.get("state") or "") == measurement.state
    ]
    if len(matches) > 1:
        raise MappingExecutionError("measurement_term_duplicate")
    return matches[0] if matches else None


def _final_measurement(
    measurement: MeasurementSpec,
    evaluator_report: Mapping[str, Any],
    *,
    evaluated_sequence_id: str,
) -> Dict[str, Any]:
    term = _exact_term(evaluator_report, measurement)
    measured = bool(
        term is not None
        and term.get("available") is True
        and term.get("score") is not None
    )
    if not measured:
        missing_fail = measurement.missing_policy == "fail"
        return {
            "measurement_id": measurement.measurement_id,
            "functional_node_id": measurement.functional_node_id,
            "kind": measurement.kind,
            "state": measurement.state,
            "evaluator_id": measurement.evaluator_id,
            "term_name": measurement.term_name,
            "status": "missing_fail" if missing_fail else "abstain",
            "term_provider": None,
            "term_value": None,
            "direction": measurement.direction,
            "directional_value": None,
            "threshold": measurement.threshold,
            "aspiration_threshold": measurement.aspiration_threshold,
            "missing_policy": measurement.missing_policy,
            "gate": (
                {"passed": False, "reason": "exact_term_missing"}
                if missing_fail
                else None
            ),
            "evaluated_sequence_id": evaluated_sequence_id,
        }
    raw_score = term.get("score")
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raise MappingExecutionError("measurement_term_value_not_numeric")
    score = float(raw_score)
    if not math.isfinite(score):
        raise MappingExecutionError("measurement_term_value_not_finite")
    directional = score if measurement.direction == "maximize" else -score
    gate = None
    if measurement.kind == "hard_constraint":
        if measurement.threshold is None:
            raise MappingExecutionError("hard_constraint_threshold_missing")
        gate = {
            "operator": ">=" if measurement.direction == "maximize" else "<=",
            "threshold": measurement.threshold,
            "passed": (
                score >= measurement.threshold
                if measurement.direction == "maximize"
                else score <= measurement.threshold
            ),
        }
    return {
        "measurement_id": measurement.measurement_id,
        "functional_node_id": measurement.functional_node_id,
        "kind": measurement.kind,
        "state": measurement.state,
        "evaluator_id": measurement.evaluator_id,
        "term_name": measurement.term_name,
        "status": "measured",
        "term_provider": str(term.get("provider")),
        "term_value": score,
        "direction": measurement.direction,
        "directional_value": directional,
        "threshold": measurement.threshold,
        "aspiration_threshold": measurement.aspiration_threshold,
        "missing_policy": measurement.missing_policy,
        "gate": gate,
        "evaluated_sequence_id": evaluated_sequence_id,
    }


def _apply_final_measurement_gates(
    evaluator_report: Mapping[str, Any],
    measurements: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Any]:
    report = deepcopy(dict(evaluator_report))
    failed = []
    for item in measurements:
        gate = item.get("gate")
        if isinstance(gate, Mapping) and gate.get("passed") is False:
            failed.append(
                "mapping_measurement_failed:"
                f"{item['functional_node_id']}:"
                f"{item['evaluator_id']}/{item['term_name']}"
            )
    if not failed:
        return report
    reasons = list(
        dict.fromkeys(
            [*(report.get("disqualification_reasons", []) or []), *failed]
        )
    )
    report.update(
        {
            "hard_gate_pass": False,
            "disqualification_reasons": reasons,
            "normalized_score": 0.0,
            "loss": 1.0,
        }
    )
    gate_status = dict(report.get("gate_status") or {})
    gate_status.update(
        {
            "passed": False,
            "hard_gate_pass": False,
            "disqualification_reasons": reasons,
            "hard_failures": reasons,
        }
    )
    report["gate_status"] = gate_status
    return report


def project_selected_mapping_runtime(
    *,
    runtime: Mapping[str, Any],
    mapping_plan: ExecutableMappingPlan,
    effective_mapping_schedule: EffectiveMappingSchedule,
    evaluator_report: Mapping[str, Any],
) -> MappingRuntimeProjection:


    try:
        validate_effective_mapping_schedule(
            mapping_plan, effective_mapping_schedule
        )
    except MappingCompileError as error:
        raise MappingExecutionError(
            "effective_mapping_schedule_plan_mismatch", str(error)
        ) from error
    active_action_ids = {
        item.action_id for item in effective_mapping_schedule.active_action_specs
    }

    base = {
        "schema_version": MAPPING_RUNTIME_VERSION,
        "execution_mode": mapping_plan.execution_mode,
        "execution_enabled": mapping_plan.execution_enabled,
        "ast_id": mapping_plan.ast_id or None,
        "ast_revision": mapping_plan.ast_revision or None,
        "attribution_hash": mapping_plan.attribution_hash,
        "projected_action_scope": "selected_immediate_action",
    }
    lineage = _selected_lineage(runtime)
    selected = lineage[-1] if lineage else None
    selected_seqs = selected.get("seqs") if isinstance(selected, Mapping) else None
    if not isinstance(selected_seqs, Mapping):
        raise MappingExecutionError("selected_mapping_sequences_missing")
    selected_sequence_id = SequenceRecord.create(selected_seqs).semantic_id
    report_sequence_id = str(evaluator_report.get("evaluated_sequence_id") or "")
    if report_sequence_id != selected_sequence_id:
        raise MappingExecutionError(
            "evaluated_sequence_id_mismatch",
            f"selected {selected_sequence_id!r}, report {report_sequence_id!r}",
        )

    functional_specs = _unique_functional_measurements(mapping_plan)
    final_measurements = tuple(
        _final_measurement(
            item,
            evaluator_report,
            evaluated_sequence_id=selected_sequence_id,
        )
        for item in functional_specs
    )
    report = _apply_final_measurement_gates(evaluator_report, final_measurements)
    if not mapping_plan.execution_enabled:
        flat_transitions = _validated_lineage_moves(runtime)
        for _parent, _child, move in flat_transitions:
            if "mapping_components" in move:
                raise MappingExecutionError(
                    "flat_mask_mapping_components_forbidden"
                )
            attribution = move.get("mapping_attribution")
            if isinstance(attribution, Mapping) and attribution:
                raise MappingExecutionError("flat_mask_mapping_attribution_forbidden")
            mutation_plan = move.get("mutation_plan")
            if (
                isinstance(mutation_plan, Mapping)
                and str(mutation_plan.get("mapping_execution") or "") == "full"
            ):
                raise MappingExecutionError("flat_mask_mapping_plan_forbidden")
        artifact = {
            **base,
            "status": "flat_mask_control",
            "selected_lineage_action_count": len(flat_transitions),
            "unobserved_lineage_action_ids": [],
            "traces": [],
            "action_observations": [],
            "observations": [dict(item) for item in final_measurements],
        }
        return MappingRuntimeProjection(report, (), final_measurements, artifact)

    transitions = _validated_lineage_moves(runtime)
    lineage_action_ids = []


    validated_actions: List[
        List[Tuple[Mapping[str, Any], Any, Dict[str, Any]]]
    ] = []
    active_actions = {
        item.action_id: item
        for item in effective_mapping_schedule.active_action_specs
    }
    if len(active_actions) != len(
        effective_mapping_schedule.active_action_specs
    ):
        raise MappingExecutionError("selected_mapping_active_action_duplicate")
    for _parent, child, move in transitions:
        child_seqs = child.get("seqs")
        if not isinstance(child_seqs, Mapping):
            raise MappingExecutionError("selected_mapping_transition_missing")
        evaluated_sequence_id = SequenceRecord.create(child_seqs).semantic_id
        if "mapping_components" in move:
            transition_actions = _validated_composite_actions(
                move,
                active_actions,
                evaluated_sequence_id=evaluated_sequence_id,
            )
        else:

            attribution = move.get("mapping_attribution")
            if not isinstance(attribution, Mapping):
                raise MappingExecutionError("selected_mapping_attribution_missing")
            action_id = str(attribution.get("action_id") or "")
            actions = [
                item
                for item in mapping_plan.action_specs
                if item.action_id == action_id
            ]
            if len(actions) != 1:
                raise MappingExecutionError("selected_mapping_action_unresolved")
            action = actions[0]
            if action.action_id not in active_action_ids:
                raise MappingExecutionError("selected_mapping_action_inactive")
            transition_actions = [
                (
                    move,
                    action,
                    _realized_move(
                        move,
                        action,
                        evaluated_sequence_id=evaluated_sequence_id,
                    ),
                )
            ]
        lineage_action_ids.extend(
            action.action_id
            for _component, action, _realized in transition_actions
        )
        validated_actions.append(transition_actions)

    traces: Tuple[MappingExecutionTrace, ...] = ()
    action_observation_error = None
    immediate_unobserved: List[str] = []
    if transitions:
        _parent, _child, _move = transitions[-1]
        projected: List[MappingExecutionTrace] = []
        for component, action, realized in validated_actions[-1]:
            attribution = component["mapping_attribution"]
            measurement = mapping_plan.measurement_for_action(action.action_id)
            if (
                str(attribution.get("measurement_id") or "")
                != measurement.measurement_id
            ):
                raise MappingExecutionError(
                    "selected_mapping_measurement_id_mismatch"
                )
            try:
                projected.append(
                    project_mapping_execution(
                        action.to_projector_dict(),
                        measurement.to_projector_dict(),
                        realized,
                        evaluator_report,
                    )
                )
            except MappingExecutionError as error:
                if error.code != "measurement_missing_fail":
                    raise
                action_observation_error = error.code
                immediate_unobserved.append(action.action_id)
        traces = tuple(projected)


    immediate_action_count = len(validated_actions[-1]) if validated_actions else 0
    unobserved = list(
        lineage_action_ids[: len(lineage_action_ids) - immediate_action_count]
    )
    unobserved.extend(immediate_unobserved)
    artifact = {
        **base,
        "status": "executed" if transitions else "parent_selected_no_action",
        "selected_lineage_action_count": len(lineage_action_ids),
        "unobserved_lineage_action_ids": unobserved,
        "action_observation_error": action_observation_error,
        "traces": [trace.to_dict() for trace in traces],
        "action_observations": [trace.observation for trace in traces],
        "observations": [dict(item) for item in final_measurements],
    }
    return MappingRuntimeProjection(report, traces, final_measurements, artifact)


__all__ = [
    "MAPPING_RUNTIME_VERSION",
    "MappingRuntimeProjection",
    "project_selected_mapping_runtime",
]
