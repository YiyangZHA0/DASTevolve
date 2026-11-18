

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

from astevolve.evaluation.selection import (
    FEASIBILITY_SELECTION_VERSION,
    FeasibilitySelectionError,
    select_feasibility_first,
)
from engine.causal_flow import (
    CausalFlowContractError,
    CausalTrace,
    validate_causal_trace,
)
from engine.experiment_identity import (
    CodeIdentity,
    EffectiveContractIdentity,
    ExperimentIdentityError,
    SequenceBundleIdentity,
)
from engine.mapping_compiler import EFFECTIVE_MAPPING_SCHEDULE_VERSION
from engine.mapping_execution import (
    COMPILED_MAPPING_ACTION_VERSION,
    COMPILED_MEASUREMENT_VERSION,
    MappingExecutionError,
    validate_mapping_execution_trace,
)
from engine.mapping_runtime import MAPPING_RUNTIME_VERSION


RUNTIME_ACCEPTANCE_VERSION = "astevolve.runtime_acceptance.v1"
ACCEPTED_RUNTIME_ARTIFACT_VERSION = "astevolve.accepted_runtime_artifact.v1"
EFFECTIVE_PHENOTYPE_IDENTITY_VERSION = "astevolve.effective_phenotype_identity.v1"
PHENOTYPE_DESCRIPTOR_CONFIG_VERSION = "astevolve.phenotype_descriptor_config.v1"
EFFECTIVE_PHENOTYPE_DESCRIPTOR_VERSION = "astevolve.effective_phenotype_descriptor.v1"

DESCRIPTOR_COMPONENTS = (
    "node_action_coverage",
    "mutation_topology",
    "feasibility",
    "strategy_novelty",
    "sequence_novelty",
)

_RAW_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_SEQUENCE_HASH_RE = re.compile(r"sequence_sha256:[0-9a-f]{64}\Z")


class EffectivePhenotypeError(ValueError):


    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)


def _fail(code: str, detail: str = "") -> None:
    raise EffectivePhenotypeError(code, detail)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        _fail("not_canonical_json", str(exc))


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _digest(domain: str, value: Any) -> str:
    preimage = domain.encode("utf-8") + b"\0" + _canonical_json(value).encode("utf-8")
    return hashlib.sha256(preimage).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("mapping_required", label)
    return value


def _closed(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    observed = {str(key) for key in value}
    unknown = sorted(observed - fields)
    missing = sorted(fields - observed)
    if unknown:
        _fail(f"{label}_unknown_fields", ",".join(unknown))
    if missing:
        _fail(f"{label}_fields_missing", ",".join(missing))


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label}_required")
    return value.strip()


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{label}_not_boolean", repr(value))
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{label}_invalid", repr(value))
    return int(value)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label}_invalid", repr(value))
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{label}_invalid", repr(value))
    return number


def _string_list(value: Any, label: str, *, unique: bool = True) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        _fail(f"{label}_invalid")
    result = [_nonempty(item, label) for item in value]
    if unique and len(result) != len(set(result)):
        _fail(f"{label}_duplicate")
    return result


def _positions(value: Any, label: str, *, allow_empty: bool = False) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        _fail(f"{label}_invalid")
    positions = [_nonnegative_int(item, label) for item in value]
    if positions != sorted(set(positions)):
        _fail(f"{label}_not_canonical")
    if not allow_empty and not positions:
        _fail(f"{label}_empty")
    return positions


def _validate_selection(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _mapping(value, "trial_selection_decision")
    if raw.get("schema_version") != FEASIBILITY_SELECTION_VERSION:
        _fail("selection_decision_invalid", "schema_version")
    candidates = raw.get("candidates")
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, list):
        _fail("selection_decision_invalid", "candidates")
    inputs = []
    try:
        for candidate in candidates:
            row = _mapping(candidate, "selection_candidate")
            inputs.append(
                {
                    "candidate_id": row.get("candidate_id"),
                    "raw_objective": row.get("raw_objective"),
                    "gate_sources": {
                        source["source"]: {
                            "passed": source["passed"],
                            "reasons": source["reasons"],
                        }
                        for source in row.get("gate_sources") or []
                    },
                }
            )
        rebuilt = select_feasibility_first(
            inputs,
            direction=raw.get("direction"),
        )
    except (EffectivePhenotypeError, FeasibilitySelectionError, KeyError, TypeError) as exc:
        _fail("selection_decision_invalid", str(exc))
    if _canonical_json(raw) != _canonical_json(rebuilt):
        _fail("selection_decision_invalid", "decision does not reproduce")
    selected_id = rebuilt["selected_candidate_id"]
    selected = next(
        (row for row in rebuilt["candidates"] if row["candidate_id"] == selected_id),
        None,
    )
    if selected is None:
        _fail("selection_decision_invalid", "selected candidate missing")
    return rebuilt, selected


_ACTION_IDENTITY_FIELDS = (
    "ast_id",
    "ast_revision",
    "edge_id",
    "functional_node_id",
    "structural_node_id",
    "action_id",
    "measurement_id",
)
_SCHEDULE_FIELDS = {
    "schema_version",
    "enabled",
    "execution_mode",
    "execution_enabled",
    "ast_id",
    "ast_revision",
    "mapping_attribution_hash",
    "active_action_specs",
    "active_measurement_specs",
    "disabled_actions",
    "functional_action_coverage",
    "functional_measurement_coverage",
    "schedule_hash",
}
_ACTION_FIELDS = {
    "schema_version",
    *_ACTION_IDENTITY_FIELDS,
    "operator",
    "chain_id",
    "legal_positions",
    "compiled_segment_name",
    "compiled_segment_kind",
    "budget",
    "evidence_refs",
}
_MEASUREMENT_FIELDS = {
    "schema_version",
    *_ACTION_IDENTITY_FIELDS,
    "evaluator_id",
    "term_name",
    "state",
    "direction",
    "threshold",
    "missing_policy",
    "kind",
    "comparator",
    "aspiration_threshold",
    "evidence_refs",
}
_DISABLED_FIELDS = {
    "edge_id",
    "functional_node_id",
    "structural_node_id",
    "action_id",
    "measurement_id",
    "operator",
    "reason",
}


def _normalize_action_spec(value: Any) -> dict[str, Any]:
    raw = _mapping(value, "active_action_spec")
    _closed(raw, _ACTION_FIELDS, "active_action_spec")
    if raw.get("schema_version") != COMPILED_MAPPING_ACTION_VERSION:
        _fail("mapping_schedule_invalid", "action schema_version")
    normalized = _json_copy(raw)
    for field in _ACTION_IDENTITY_FIELDS:
        if field == "ast_revision":
            if _nonnegative_int(raw[field], field) < 1:
                _fail("mapping_schedule_invalid", "ast_revision")
        else:
            _nonempty(raw[field], field)
    _nonempty(raw["operator"], "operator")
    _nonempty(raw["chain_id"], "chain_id")
    _nonempty(raw["compiled_segment_name"], "compiled_segment_name")
    _nonempty(raw["compiled_segment_kind"], "compiled_segment_kind")
    positions = _positions(raw["legal_positions"], "legal_positions")
    budget = _mapping(raw["budget"], "budget")
    _closed(budget, {"min", "max"}, "budget")
    minimum = _nonnegative_int(budget["min"], "budget_min")
    maximum = _nonnegative_int(budget["max"], "budget_max")
    if minimum < 1 or minimum > maximum or minimum > len(positions):
        _fail("mapping_schedule_invalid", "budget")
    _string_list(raw["evidence_refs"], "evidence_refs")
    return normalized


def _normalize_measurement_spec(value: Any) -> dict[str, Any]:
    raw = _mapping(value, "active_measurement_spec")
    _closed(raw, _MEASUREMENT_FIELDS, "active_measurement_spec")
    if raw.get("schema_version") != COMPILED_MEASUREMENT_VERSION:
        _fail("mapping_schedule_invalid", "measurement schema_version")
    for field in _ACTION_IDENTITY_FIELDS:
        if field == "ast_revision":
            if _nonnegative_int(raw[field], field) < 1:
                _fail("mapping_schedule_invalid", "ast_revision")
        else:
            _nonempty(raw[field], field)
    for field in (
        "evaluator_id",
        "term_name",
        "state",
        "direction",
        "missing_policy",
        "kind",
        "comparator",
    ):
        _nonempty(raw[field], field)
    for field in ("threshold", "aspiration_threshold"):
        if raw[field] is not None:
            _finite_number(raw[field], field)
    _string_list(raw["evidence_refs"], "evidence_refs")
    return _json_copy(raw)


def _coverage_mapping(value: Any, label: str) -> dict[str, list[str]]:
    raw = _mapping(value, label)
    result: dict[str, list[str]] = {}
    for raw_key in sorted(raw, key=str):
        key = _nonempty(raw_key, f"{label}_node")
        result[key] = _string_list(raw[raw_key], f"{label}_{key}")
    return result


def _normalize_mapping_schedule(value: Any) -> dict[str, Any]:
    try:
        raw = _mapping(value, "effective_mapping_schedule")
        _closed(raw, _SCHEDULE_FIELDS, "mapping_schedule")
        if raw.get("schema_version") != EFFECTIVE_MAPPING_SCHEDULE_VERSION:
            _fail("mapping_schedule_invalid", "schema_version")
        enabled = _strict_bool(raw["enabled"], "mapping_enabled")
        execution_enabled = _strict_bool(
            raw["execution_enabled"], "mapping_execution_enabled"
        )
        mode = _nonempty(raw["execution_mode"], "mapping_execution_mode")
        if mode not in {"full", "flat_mask"}:
            _fail("mapping_schedule_invalid", "execution_mode")
        if execution_enabled != bool(enabled and mode == "full"):
            _fail("mapping_schedule_invalid", "execution_enabled")
        ast_id = _nonempty(raw["ast_id"], "ast_id") if enabled else str(raw["ast_id"])
        revision = _nonnegative_int(raw["ast_revision"], "ast_revision")
        if enabled and revision < 1:
            _fail("mapping_schedule_invalid", "ast_revision")
        attribution_hash = _nonempty(
            raw["mapping_attribution_hash"], "mapping_attribution_hash"
        )
        if _RAW_HASH_RE.fullmatch(attribution_hash) is None:
            _fail("mapping_schedule_invalid", "mapping_attribution_hash")

        actions = [_normalize_action_spec(item) for item in raw["active_action_specs"]]
        measurements = [
            _normalize_measurement_spec(item)
            for item in raw["active_measurement_specs"]
        ]
        action_ids = [item["action_id"] for item in actions]
        measurement_action_ids = [item["action_id"] for item in measurements]
        if len(action_ids) != len(set(action_ids)):
            _fail("mapping_schedule_invalid", "duplicate active action")
        if len(measurement_action_ids) != len(set(measurement_action_ids)):
            _fail("mapping_schedule_invalid", "duplicate active measurement")
        if measurement_action_ids != action_ids:
            _fail("mapping_schedule_invalid", "action/measurement coverage")
        action_by_id = {item["action_id"]: item for item in actions}
        for measurement in measurements:
            action = action_by_id[measurement["action_id"]]
            if any(measurement[field] != action[field] for field in _ACTION_IDENTITY_FIELDS):
                _fail("mapping_schedule_invalid", "action/measurement identity")

        disabled = []
        for item in raw["disabled_actions"]:
            entry = _mapping(item, "disabled_action")
            _closed(entry, _DISABLED_FIELDS, "disabled_action")
            for field in _DISABLED_FIELDS:
                _nonempty(entry[field], f"disabled_{field}")
            disabled.append(_json_copy(entry))
        disabled_ids = [item["action_id"] for item in disabled]
        if len(disabled_ids) != len(set(disabled_ids)) or set(disabled_ids) & set(action_ids):
            _fail("mapping_schedule_invalid", "disabled action identity")
        if execution_enabled and not actions:
            _fail("mapping_schedule_invalid", "no active action")

        action_coverage = _coverage_mapping(
            raw["functional_action_coverage"], "functional_action_coverage"
        )
        measurement_coverage = _coverage_mapping(
            raw["functional_measurement_coverage"],
            "functional_measurement_coverage",
        )
        expected_action_coverage = {
            node: [
                action["action_id"]
                for action in actions
                if action["functional_node_id"] == node
            ]
            for node in sorted(action_coverage)
        }
        if action_coverage != expected_action_coverage:
            _fail("mapping_schedule_invalid", "functional action coverage")
        for measurement in measurements:
            node = measurement["functional_node_id"]
            if measurement["measurement_id"] not in measurement_coverage.get(node, []):
                _fail("mapping_schedule_invalid", "functional measurement coverage")

        payload = {key: _json_copy(raw[key]) for key in raw if key != "schedule_hash"}
        expected_hash = _digest("astevolve.effective_mapping_schedule.v1", payload)
        if raw["schedule_hash"] != expected_hash:
            _fail("mapping_schedule_invalid", "schedule_hash")
        return {**payload, "schedule_hash": expected_hash}
    except EffectivePhenotypeError as exc:
        if exc.code == "mapping_schedule_invalid":
            raise
        _fail("mapping_schedule_invalid", str(exc))


_MAPPING_EXECUTION_REQUIRED = {
    "schema_version",
    "execution_mode",
    "execution_enabled",
    "ast_id",
    "ast_revision",
    "attribution_hash",
    "projected_action_scope",
    "status",
    "selected_lineage_action_count",
    "unobserved_lineage_action_ids",
    "traces",
    "action_observations",
    "observations",
}


def _normalize_mapping_execution(
    value: Any,
    *,
    schedule: Mapping[str, Any],
    final_sequence_id: str,
) -> dict[str, Any]:
    try:
        raw = _mapping(value, "mapping_execution")
        fields = set(raw)
        unknown = fields - (_MAPPING_EXECUTION_REQUIRED | {"action_observation_error"})
        missing = _MAPPING_EXECUTION_REQUIRED - fields
        if unknown or missing:
            _fail(
                "mapping_execution_invalid",
                f"unknown={sorted(unknown)},missing={sorted(missing)}",
            )
        if raw.get("schema_version") != MAPPING_RUNTIME_VERSION:
            _fail("mapping_execution_invalid", "schema_version")
        execution_enabled = _strict_bool(
            raw["execution_enabled"], "mapping_execution_enabled"
        )
        for field in ("execution_mode", "ast_id", "ast_revision"):
            if raw[field] != schedule[field]:
                _fail("mapping_execution_invalid", f"schedule_{field}_mismatch")
        if execution_enabled != schedule["execution_enabled"]:
            _fail("mapping_execution_invalid", "schedule_enabled_mismatch")
        if raw["attribution_hash"] != schedule["mapping_attribution_hash"]:
            _fail("mapping_execution_invalid", "attribution_hash_mismatch")
        if raw["projected_action_scope"] != "selected_immediate_action":
            _fail("mapping_execution_invalid", "projected_action_scope")
        status = raw["status"]
        if status not in {"executed", "parent_selected_no_action", "flat_mask_control"}:
            _fail("mapping_execution_invalid", "status")
        count = _nonnegative_int(
            raw["selected_lineage_action_count"], "selected_lineage_action_count"
        )
        active_ids = {
            item["action_id"] for item in schedule["active_action_specs"]
        }
        unobserved = _string_list(
            raw["unobserved_lineage_action_ids"],
            "unobserved_lineage_action_ids",
            unique=False,
        )
        if any(action_id not in active_ids for action_id in unobserved):
            _fail("mapping_execution_invalid", "unobserved action inactive")

        traces = []
        for item in raw["traces"]:
            try:
                trace = validate_mapping_execution_trace(item)
            except (MappingExecutionError, TypeError, ValueError) as exc:
                _fail("mapping_execution_invalid", str(exc))
            if trace.evaluated_sequence_id != final_sequence_id:
                _fail("mapping_execution_invalid", "trace final sequence mismatch")
            if trace.action_id not in active_ids:
                _fail("mapping_execution_invalid", "trace action inactive")
            traces.append(trace.to_dict())
        if not execution_enabled and traces:
            _fail("mapping_execution_invalid", "flat trace forbidden")
        if execution_enabled and count != len(unobserved) + len(traces):
            _fail("mapping_execution_invalid", "lineage action count")
        expected_observations = [item["observation"] for item in traces]
        if _canonical_json(raw["action_observations"]) != _canonical_json(
            expected_observations
        ):
            _fail("mapping_execution_invalid", "action observations mismatch")

        observations = []
        for item in raw["observations"]:
            observation = _mapping(item, "mapping_observation")
            if observation.get("evaluated_sequence_id") != final_sequence_id:
                _fail("mapping_execution_invalid", "observation sequence mismatch")
            _nonempty(observation.get("functional_node_id"), "functional_node_id")
            observations.append(_json_copy(observation))
        if status == "executed" and count < 1:
            _fail("mapping_execution_invalid", "executed without action")
        if status == "parent_selected_no_action" and count != 0:
            _fail("mapping_execution_invalid", "parent status with action")
        if status == "flat_mask_control" and execution_enabled:
            _fail("mapping_execution_invalid", "flat status in full mode")
        return _json_copy(raw)
    except EffectivePhenotypeError as exc:
        if exc.code == "mapping_execution_invalid":
            raise
        _fail("mapping_execution_invalid", str(exc))


def _validated_metrics(value: Any) -> dict[str, Any]:
    raw = _mapping(value, "metrics")
    if "hard_gate_pass" not in raw:
        _fail("feasibility_unknown", "metrics.hard_gate_pass")
    gate = _strict_bool(raw["hard_gate_pass"], "hard_gate_pass")
    if "disqualified" in raw and _strict_bool(raw["disqualified"], "disqualified") == gate:
        _fail("feasibility_mismatch", "disqualified")
    return _json_copy(raw)


def _raw_runtime_evidence(
    runtime_artifacts: Mapping[str, Any], metrics: Mapping[str, Any]
) -> tuple[
    dict[str, Any],
    EffectiveContractIdentity,
    SequenceBundleIdentity,
    dict[str, Any],
]:
    required = {
        "effective_search_contract",
        "causal_trace",
        "effective_mapping_schedule",
        "mapping_execution",
        "best_seqs",
        "trial_selection_decision",
    }
    missing = sorted(key for key in required if key not in runtime_artifacts)
    if missing:
        _fail("runtime_evidence_missing", ",".join(missing))
    if runtime_artifacts.get("error"):
        _fail("runtime_not_accepted", str(runtime_artifacts.get("error")))

    try:
        contract_identity = EffectiveContractIdentity.create(
            _mapping(
                runtime_artifacts["effective_search_contract"],
                "effective_search_contract",
            )
        )
    except (EffectivePhenotypeError, ExperimentIdentityError, CausalFlowContractError) as exc:
        _fail("effective_contract_invalid", str(exc))
    try:
        causal_trace = validate_causal_trace(
            _mapping(runtime_artifacts["causal_trace"], "causal_trace")
        )
    except (EffectivePhenotypeError, CausalFlowContractError, TypeError, ValueError) as exc:
        _fail("causal_trace_invalid", str(exc))
    if causal_trace.effective_contract.contract_hash != contract_identity.effective_contract_hash:
        _fail("effective_contract_invalid", "causal trace mismatch")

    final_records = [
        record
        for record in causal_trace.sequences
        if record.semantic_id == causal_trace.final_sequence_id
    ]
    if len(final_records) != 1:
        _fail("final_sequence_invalid", "causal trace final sequence")
    trace_sequence = SequenceBundleIdentity.create(final_records[0])
    try:
        selected_sequence = SequenceBundleIdentity.create(
            _mapping(runtime_artifacts["best_seqs"], "best_seqs")
        )
    except (EffectivePhenotypeError, ExperimentIdentityError) as exc:
        _fail("final_sequence_invalid", str(exc))
    if trace_sequence.sequence_bundle_hash != selected_sequence.sequence_bundle_hash:
        _fail("final_sequence_mismatch", "best_seqs does not match causal trace")

    schedule = _normalize_mapping_schedule(
        runtime_artifacts["effective_mapping_schedule"]
    )
    mapping_execution = _normalize_mapping_execution(
        runtime_artifacts["mapping_execution"],
        schedule=schedule,
        final_sequence_id=trace_sequence.sequence_bundle_hash,
    )
    selection, selected_row = _validate_selection(
        runtime_artifacts["trial_selection_decision"]
    )
    clean_metrics = _validated_metrics(metrics)
    feasible = bool(selected_row["feasible"])
    if clean_metrics["hard_gate_pass"] != feasible:
        _fail("feasibility_mismatch", "selection/metrics")

    actions_by_id = {
        action["action_id"]: action for action in schedule["active_action_specs"]
    }
    causal_mapping_ids: list[str] = []
    if schedule["execution_enabled"]:
        for action in causal_trace.actions:
            parameters = action.parameters
            if any(field not in parameters for field in _ACTION_IDENTITY_FIELDS):
                _fail("mapping_execution_invalid", "causal action attribution missing")
            action_id = str(parameters["action_id"])
            spec = actions_by_id.get(action_id)
            if spec is None:
                _fail("mapping_execution_invalid", "causal action inactive")
            for field in _ACTION_IDENTITY_FIELDS:
                if parameters[field] != spec[field]:
                    _fail("mapping_execution_invalid", f"causal {field} mismatch")
            if action.operator != spec["operator"] or action.node_id != spec["structural_node_id"]:
                _fail("mapping_execution_invalid", "causal action semantic mismatch")
            if not set(action.positions).issubset(set(spec["legal_positions"])):
                _fail("mapping_execution_invalid", "causal positions out of bounds")
            causal_mapping_ids.append(action_id)
        traces = mapping_execution["traces"]
        trace_ids = [item["action_id"] for item in traces]
        immediate_actions = []
        if causal_trace.actions:
            final_action = causal_trace.actions[-1]
            for action in reversed(causal_trace.actions):
                if (
                    action.parent_sequence_id != final_action.parent_sequence_id
                    or action.child_sequence_id != causal_trace.final_sequence_id
                ):
                    break
                immediate_actions.append(action)
            immediate_actions.reverse()
        history_count = len(causal_trace.actions) - len(immediate_actions)
        immediate_ids = [
            str(action.parameters["action_id"]) for action in immediate_actions
        ]
        missing_immediate: list[str] = []
        trace_cursor = 0
        for action_id in immediate_ids:
            if trace_cursor < len(trace_ids) and action_id == trace_ids[trace_cursor]:
                trace_cursor += 1
            else:
                missing_immediate.append(action_id)
        expected_unobserved = (
            causal_mapping_ids[:history_count] + missing_immediate
        )
        if trace_cursor != len(trace_ids) or (
            mapping_execution["unobserved_lineage_action_ids"] != expected_unobserved
        ):
            _fail("mapping_execution_invalid", "causal lineage attribution mismatch")
    else:
        forbidden = set(_ACTION_IDENTITY_FIELDS)
        if any(forbidden.intersection(action.parameters) for action in causal_trace.actions):
            _fail("mapping_execution_invalid", "flat causal attribution forbidden")
    if mapping_execution["selected_lineage_action_count"] != len(causal_trace.actions):
        _fail("mapping_execution_invalid", "causal lineage action count mismatch")

    functional_nodes = set(schedule["functional_measurement_coverage"])
    for observation in mapping_execution["observations"]:
        if observation["functional_node_id"] not in functional_nodes:
            _fail("mapping_execution_invalid", "functional node not scheduled")
    if not any(
        observation.sequence_id == causal_trace.final_sequence_id
        for observation in causal_trace.observations
    ):
        _fail("causal_trace_invalid", "final sequence was not observed")

    evidence = {
        "effective_search_contract": contract_identity.effective_contract,
        "causal_trace": causal_trace.to_dict(),
        "effective_mapping_schedule": schedule,
        "mapping_execution": mapping_execution,
        "trial_selection_decision": selection,
        "metrics": clean_metrics,
    }
    feasibility = {
        "feasible": feasible,
        "reasons": list(selected_row.get("gate_reasons") or []),
    }
    return evidence, contract_identity, trace_sequence, feasibility


def _sealed_runtime_evidence(
    value: Any,
) -> tuple[
    dict[str, Any],
    EffectiveContractIdentity,
    SequenceBundleIdentity,
    dict[str, Any],
]:
    raw = _mapping(value, "runtime_evidence")
    _closed(
        raw,
        {
            "effective_search_contract",
            "causal_trace",
            "effective_mapping_schedule",
            "mapping_execution",
            "trial_selection_decision",
            "metrics",
        },
        "runtime_evidence",
    )
    try:
        causal_trace = validate_causal_trace(raw["causal_trace"])
    except (CausalFlowContractError, TypeError, ValueError) as exc:
        _fail("causal_trace_invalid", str(exc))
    final = next(
        (
            record
            for record in causal_trace.sequences
            if record.semantic_id == causal_trace.final_sequence_id
        ),
        None,
    )
    if final is None:
        _fail("final_sequence_invalid")
    synthetic = {
        "effective_search_contract": raw["effective_search_contract"],
        "causal_trace": raw["causal_trace"],
        "effective_mapping_schedule": raw["effective_mapping_schedule"],
        "mapping_execution": raw["mapping_execution"],
        "trial_selection_decision": raw["trial_selection_decision"],
        "best_seqs": final.chains,
    }
    return _raw_runtime_evidence(synthetic, raw["metrics"])


@dataclass(frozen=True)
class AcceptedRuntimeArtifact:


    code_identity: CodeIdentity
    effective_contract_identity: EffectiveContractIdentity
    sequence_bundle_identity: SequenceBundleIdentity
    _acceptance_json: str
    _runtime_evidence_json: str
    _feasibility_json: str
    artifact_hash: str
    schema_version: str = ACCEPTED_RUNTIME_ARTIFACT_VERSION

    @property
    def acceptance(self) -> dict[str, Any]:
        return json.loads(self._acceptance_json)

    @property
    def runtime_evidence(self) -> dict[str, Any]:
        return json.loads(self._runtime_evidence_json)

    @property
    def feasibility(self) -> dict[str, Any]:
        return json.loads(self._feasibility_json)

    @classmethod
    def create(
        cls,
        *,
        source_code: str | bytes | bytearray | memoryview,
        runtime_artifacts: Mapping[str, Any],
        metrics: Mapping[str, Any],
    ) -> "AcceptedRuntimeArtifact":
        if isinstance(source_code, str):
            code_identity = CodeIdentity.from_text(source_code)
        else:
            try:
                code_identity = CodeIdentity.create(source_code)
            except ExperimentIdentityError as exc:
                _fail("source_code_invalid", str(exc))
        evidence, contract_identity, sequence_identity, feasibility = (
            _raw_runtime_evidence(
                _mapping(runtime_artifacts, "runtime_artifacts"),
                _mapping(metrics, "metrics"),
            )
        )
        selection = evidence["trial_selection_decision"]
        acceptance = {
            "schema_version": RUNTIME_ACCEPTANCE_VERSION,
            "status": "accepted",
            "selected_candidate_id": selection["selected_candidate_id"],
            "selection_decision_hash": "selection_sha256:"
            + _digest("astevolve.accepted_selection.v1", selection),
        }
        core = {
            "schema_version": ACCEPTED_RUNTIME_ARTIFACT_VERSION,
            "acceptance": acceptance,
            "code_identity": code_identity.to_dict(),
            "effective_contract_identity": contract_identity.to_dict(),
            "sequence_bundle_identity": sequence_identity.to_dict(),
            "runtime_evidence": evidence,
            "feasibility": feasibility,
        }
        return cls(
            code_identity=code_identity,
            effective_contract_identity=contract_identity,
            sequence_bundle_identity=sequence_identity,
            _acceptance_json=_canonical_json(acceptance),
            _runtime_evidence_json=_canonical_json(evidence),
            _feasibility_json=_canonical_json(feasibility),
            artifact_hash="accepted_runtime_sha256:"
            + _digest("astevolve.accepted_runtime_artifact.v1", core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "acceptance": self.acceptance,
            "code_identity": self.code_identity.to_dict(),
            "effective_contract_identity": self.effective_contract_identity.to_dict(),
            "sequence_bundle_identity": self.sequence_bundle_identity.to_dict(),
            "runtime_evidence": self.runtime_evidence,
            "feasibility": self.feasibility,
            "artifact_hash": self.artifact_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AcceptedRuntimeArtifact":
        try:
            raw = _mapping(value, "accepted_runtime_artifact")
            _closed(
                raw,
                {
                    "schema_version",
                    "acceptance",
                    "code_identity",
                    "effective_contract_identity",
                    "sequence_bundle_identity",
                    "runtime_evidence",
                    "feasibility",
                    "artifact_hash",
                },
                "accepted_artifact",
            )
        except EffectivePhenotypeError as exc:
            _fail("accepted_artifact_invalid", str(exc))
        if raw.get("schema_version") != ACCEPTED_RUNTIME_ARTIFACT_VERSION:
            _fail("accepted_artifact_invalid", "schema_version")
        acceptance = _mapping(raw["acceptance"], "acceptance")
        _closed(
            acceptance,
            {
                "schema_version",
                "status",
                "selected_candidate_id",
                "selection_decision_hash",
            },
            "acceptance",
        )
        if acceptance.get("schema_version") != RUNTIME_ACCEPTANCE_VERSION:
            _fail("accepted_artifact_invalid", "acceptance schema_version")
        if acceptance.get("status") != "accepted":
            _fail("runtime_not_accepted", str(acceptance.get("status")))
        try:
            code_identity = CodeIdentity.from_mapping(raw["code_identity"])
            declared_contract = EffectiveContractIdentity.from_mapping(
                raw["effective_contract_identity"]
            )
            declared_sequence = SequenceBundleIdentity.from_mapping(
                raw["sequence_bundle_identity"]
            )
        except ExperimentIdentityError as exc:
            _fail("accepted_artifact_invalid", str(exc))
        evidence, contract_identity, sequence_identity, feasibility = (
            _sealed_runtime_evidence(raw["runtime_evidence"])
        )
        if declared_contract.to_dict() != contract_identity.to_dict():
            _fail("effective_contract_invalid", "declared/evidence mismatch")
        if declared_sequence.to_dict() != sequence_identity.to_dict():
            _fail("final_sequence_mismatch", "declared/evidence mismatch")
        selection = evidence["trial_selection_decision"]
        expected_selection_hash = "selection_sha256:" + _digest(
            "astevolve.accepted_selection.v1", selection
        )
        if (
            acceptance.get("selected_candidate_id")
            != selection["selected_candidate_id"]
            or acceptance.get("selection_decision_hash") != expected_selection_hash
        ):
            _fail("selection_decision_invalid", "acceptance mismatch")
        if _canonical_json(raw["feasibility"]) != _canonical_json(feasibility):
            _fail("feasibility_mismatch", "declared/evidence mismatch")
        core = {key: _json_copy(raw[key]) for key in raw if key != "artifact_hash"}
        expected_hash = "accepted_runtime_sha256:" + _digest(
            "astevolve.accepted_runtime_artifact.v1", core
        )
        if raw.get("artifact_hash") != expected_hash:
            _fail("accepted_artifact_hash_mismatch")
        return cls(
            code_identity=code_identity,
            effective_contract_identity=contract_identity,
            sequence_bundle_identity=sequence_identity,
            _acceptance_json=_canonical_json(acceptance),
            _runtime_evidence_json=_canonical_json(evidence),
            _feasibility_json=_canonical_json(feasibility),
            artifact_hash=expected_hash,
        )


def _accepted(value: AcceptedRuntimeArtifact | Mapping[str, Any]) -> AcceptedRuntimeArtifact:
    if isinstance(value, AcceptedRuntimeArtifact):
        return AcceptedRuntimeArtifact.from_mapping(value.to_dict())
    if not isinstance(value, Mapping) or value.get("schema_version") != (
        ACCEPTED_RUNTIME_ARTIFACT_VERSION
    ):
        _fail("accepted_artifact_invalid", "sealed accepted runtime artifact required")
    return AcceptedRuntimeArtifact.from_mapping(value)


@dataclass(frozen=True)
class EffectivePhenotypeIdentity:


    code_identity: CodeIdentity
    effective_contract_identity: EffectiveContractIdentity
    sequence_bundle_identity: SequenceBundleIdentity
    accepted_artifact_hash: str
    identity_hash: str
    archive_niche_hash: str
    schema_version: str = EFFECTIVE_PHENOTYPE_IDENTITY_VERSION

    @classmethod
    def create(
        cls, value: AcceptedRuntimeArtifact | Mapping[str, Any]
    ) -> "EffectivePhenotypeIdentity":
        artifact = _accepted(value)
        layers = {
            "code_identity_hash": artifact.code_identity.identity_hash,
            "effective_contract_identity_hash": (
                artifact.effective_contract_identity.identity_hash
            ),
            "sequence_bundle_hash": artifact.sequence_bundle_identity.sequence_bundle_hash,
        }
        effective_layers = {
            "effective_contract_identity_hash": layers[
                "effective_contract_identity_hash"
            ],
            "sequence_bundle_hash": layers["sequence_bundle_hash"],
        }
        return cls(
            code_identity=artifact.code_identity,
            effective_contract_identity=artifact.effective_contract_identity,
            sequence_bundle_identity=artifact.sequence_bundle_identity,
            accepted_artifact_hash=artifact.artifact_hash,
            identity_hash="effective_phenotype_sha256:"
            + _digest("astevolve.effective_phenotype_identity.v1", layers),
            archive_niche_hash="effective_niche_sha256:"
            + _digest("astevolve.effective_phenotype_niche.v1", effective_layers),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "code_identity": self.code_identity.to_dict(),
            "effective_contract_identity": self.effective_contract_identity.to_dict(),
            "sequence_bundle_identity": self.sequence_bundle_identity.to_dict(),
            "accepted_artifact_hash": self.accepted_artifact_hash,
            "identity_hash": self.identity_hash,
            "archive_niche_hash": self.archive_niche_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EffectivePhenotypeIdentity":
        raw = _mapping(value, "effective_phenotype_identity")
        _closed(
            raw,
            {
                "schema_version",
                "code_identity",
                "effective_contract_identity",
                "sequence_bundle_identity",
                "accepted_artifact_hash",
                "identity_hash",
                "archive_niche_hash",
            },
            "effective_phenotype_identity",
        )
        if raw.get("schema_version") != EFFECTIVE_PHENOTYPE_IDENTITY_VERSION:
            _fail("phenotype_identity_invalid", "schema_version")
        try:
            code = CodeIdentity.from_mapping(raw["code_identity"])
            contract = EffectiveContractIdentity.from_mapping(
                raw["effective_contract_identity"]
            )
            sequence = SequenceBundleIdentity.from_mapping(
                raw["sequence_bundle_identity"]
            )
        except ExperimentIdentityError as exc:
            _fail("phenotype_identity_invalid", str(exc))
        layers = {
            "code_identity_hash": code.identity_hash,
            "effective_contract_identity_hash": contract.identity_hash,
            "sequence_bundle_hash": sequence.sequence_bundle_hash,
        }
        effective_layers = {
            "effective_contract_identity_hash": contract.identity_hash,
            "sequence_bundle_hash": sequence.sequence_bundle_hash,
        }
        expected_identity = "effective_phenotype_sha256:" + _digest(
            "astevolve.effective_phenotype_identity.v1", layers
        )
        expected_niche = "effective_niche_sha256:" + _digest(
            "astevolve.effective_phenotype_niche.v1", effective_layers
        )
        if raw["identity_hash"] != expected_identity or raw["archive_niche_hash"] != expected_niche:
            _fail("phenotype_identity_hash_mismatch")
        return cls(
            code,
            contract,
            sequence,
            _nonempty(raw["accepted_artifact_hash"], "accepted_artifact_hash"),
            expected_identity,
            expected_niche,
        )


@dataclass(frozen=True)
class PhenotypeDescriptorConfig:


    components: tuple[str, ...]
    strategy_reference_hashes: tuple[str, ...]
    sequence_reference_hashes: tuple[str, ...]
    config_hash: str
    schema_version: str = PHENOTYPE_DESCRIPTOR_CONFIG_VERSION

    @classmethod
    def create(
        cls,
        *,
        components: Optional[Iterable[str]] = None,
        strategy_reference_hashes: Iterable[str] = (),
        sequence_reference_hashes: Iterable[str] = (),
    ) -> "PhenotypeDescriptorConfig":
        requested = list(DESCRIPTOR_COMPONENTS if components is None else components)
        if not requested or len(requested) != len(set(requested)):
            _fail("descriptor_components_invalid")
        unknown = sorted(set(requested) - set(DESCRIPTOR_COMPONENTS))
        if unknown:
            _fail("descriptor_components_invalid", ",".join(unknown))
        ordered = tuple(item for item in DESCRIPTOR_COMPONENTS if item in requested)
        strategy_refs = tuple(sorted({_nonempty(item, "strategy_reference_hash") for item in strategy_reference_hashes}))
        sequence_refs = tuple(sorted({_nonempty(item, "sequence_reference_hash") for item in sequence_reference_hashes}))
        if any(_RAW_HASH_RE.fullmatch(item) is None for item in strategy_refs):
            _fail("strategy_reference_hash_invalid")
        if any(_SEQUENCE_HASH_RE.fullmatch(item) is None for item in sequence_refs):
            _fail("sequence_reference_hash_invalid")
        payload = {
            "schema_version": PHENOTYPE_DESCRIPTOR_CONFIG_VERSION,
            "components": list(ordered),
            "strategy_reference_hashes": list(strategy_refs),
            "sequence_reference_hashes": list(sequence_refs),
        }
        return cls(
            ordered,
            strategy_refs,
            sequence_refs,
            "phenotype_descriptor_config_sha256:"
            + _digest("astevolve.phenotype_descriptor_config.v1", payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "components": list(self.components),
            "strategy_reference_hashes": list(self.strategy_reference_hashes),
            "sequence_reference_hashes": list(self.sequence_reference_hashes),
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PhenotypeDescriptorConfig":
        raw = _mapping(value, "phenotype_descriptor_config")
        _closed(
            raw,
            {
                "schema_version",
                "components",
                "strategy_reference_hashes",
                "sequence_reference_hashes",
                "config_hash",
            },
            "phenotype_descriptor_config",
        )
        if raw.get("schema_version") != PHENOTYPE_DESCRIPTOR_CONFIG_VERSION:
            _fail("descriptor_config_invalid", "schema_version")
        restored = cls.create(
            components=raw["components"],
            strategy_reference_hashes=raw["strategy_reference_hashes"],
            sequence_reference_hashes=raw["sequence_reference_hashes"],
        )
        if raw["config_hash"] != restored.config_hash:
            _fail("descriptor_config_hash_mismatch")
        return restored


def _mutation_sites(action: Any) -> list[str]:
    changes = action.parameters.get("changes")
    sites = []
    if isinstance(changes, list):
        for change in changes:
            if not isinstance(change, Mapping) or "position" not in change:
                continue
            position = _nonnegative_int(change["position"], "mutation_position")
            chain = str(change.get("chain_id") or "_").strip() or "_"
            sites.append(f"{chain}:{position}")
    if not sites:
        sites = [f"_:{position}" for position in action.positions]
    return sorted(set(sites))


def _descriptor_components(
    artifact: AcceptedRuntimeArtifact,
    config: PhenotypeDescriptorConfig,
) -> dict[str, Any]:
    evidence = artifact.runtime_evidence
    trace: CausalTrace = validate_causal_trace(evidence["causal_trace"])
    schedule = evidence["effective_mapping_schedule"]
    mapping_execution = evidence["mapping_execution"]
    planned_action_ids = [
        item["action_id"] for item in schedule["active_action_specs"]
    ]
    executed_action_ids = [
        str(action.parameters["action_id"])
        for action in trace.actions
        if "action_id" in action.parameters
    ]
    structural_nodes = sorted({action.node_id for action in trace.actions})
    functional_nodes = sorted(
        {
            str(item["functional_node_id"])
            for item in mapping_execution["observations"]
        }
    )
    operators = sorted({action.operator for action in trace.actions})
    planned_functional = sorted(schedule["functional_action_coverage"])
    coverage_fraction = (
        len(set(executed_action_ids) & set(planned_action_ids))
        / len(set(planned_action_ids))
        if planned_action_ids
        else 0.0
    )
    sites = sorted({site for action in trace.actions for site in _mutation_sites(action)})
    transitions = [
        {
            "node_id": action.node_id,
            "operator": action.operator,
            "sites": _mutation_sites(action),
        }
        for action in trace.actions
    ]
    topology = {
        "transition_count": len(trace.actions),
        "mutation_event_count": sum(len(action.positions) for action in trace.actions),
        "unique_mutated_sites": sites,
        "operators": operators,
        "topology_hash": "mutation_topology_sha256:"
        + _digest("astevolve.mutation_topology.v1", transitions),
    }
    all_values = {
        "node_action_coverage": {
            "planned_functional_nodes": planned_functional,
            "executed_functional_nodes": functional_nodes,
            "executed_structural_nodes": structural_nodes,
            "planned_action_ids": planned_action_ids,
            "executed_action_ids": executed_action_ids,
            "operators": operators,
            "action_coverage_fraction": coverage_fraction,
        },
        "mutation_topology": topology,
        "feasibility": artifact.feasibility,
        "strategy_novelty": {
            "effective_contract_hash": (
                artifact.effective_contract_identity.effective_contract_hash
            ),
            "is_novel": (
                artifact.effective_contract_identity.effective_contract_hash
                not in config.strategy_reference_hashes
            ),
        },
        "sequence_novelty": {
            "sequence_bundle_hash": (
                artifact.sequence_bundle_identity.sequence_bundle_hash
            ),
            "is_novel": (
                artifact.sequence_bundle_identity.sequence_bundle_hash
                not in config.sequence_reference_hashes
            ),
        },
    }
    return {component: all_values[component] for component in config.components}


@dataclass(frozen=True)
class EffectivePhenotypeDescriptor:


    config: PhenotypeDescriptorConfig
    accepted_artifact_hash: str
    archive_niche_hash: str
    _components_json: str
    descriptor_hash: str
    schema_version: str = EFFECTIVE_PHENOTYPE_DESCRIPTOR_VERSION

    @property
    def components(self) -> dict[str, Any]:
        return json.loads(self._components_json)

    @classmethod
    def create(
        cls,
        value: AcceptedRuntimeArtifact | Mapping[str, Any],
        *,
        config: Optional[PhenotypeDescriptorConfig | Mapping[str, Any]] = None,
    ) -> "EffectivePhenotypeDescriptor":
        artifact = _accepted(value)
        if config is None:
            resolved_config = PhenotypeDescriptorConfig.create()
        elif isinstance(config, PhenotypeDescriptorConfig):
            resolved_config = PhenotypeDescriptorConfig.from_mapping(config.to_dict())
        else:
            resolved_config = PhenotypeDescriptorConfig.from_mapping(config)
        identity = EffectivePhenotypeIdentity.create(artifact)
        components = _descriptor_components(artifact, resolved_config)
        payload = {
            "schema_version": EFFECTIVE_PHENOTYPE_DESCRIPTOR_VERSION,
            "config": resolved_config.to_dict(),
            "accepted_artifact_hash": artifact.artifact_hash,
            "archive_niche_hash": identity.archive_niche_hash,
            "components": components,
        }
        return cls(
            config=resolved_config,
            accepted_artifact_hash=artifact.artifact_hash,
            archive_niche_hash=identity.archive_niche_hash,
            _components_json=_canonical_json(components),
            descriptor_hash="phenotype_descriptor_sha256:"
            + _digest("astevolve.effective_phenotype_descriptor.v1", payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "accepted_artifact_hash": self.accepted_artifact_hash,
            "archive_niche_hash": self.archive_niche_hash,
            "components": self.components,
            "descriptor_hash": self.descriptor_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EffectivePhenotypeDescriptor":
        raw = _mapping(value, "effective_phenotype_descriptor")
        _closed(
            raw,
            {
                "schema_version",
                "config",
                "accepted_artifact_hash",
                "archive_niche_hash",
                "components",
                "descriptor_hash",
            },
            "effective_phenotype_descriptor",
        )
        if raw.get("schema_version") != EFFECTIVE_PHENOTYPE_DESCRIPTOR_VERSION:
            _fail("descriptor_invalid", "schema_version")
        config = PhenotypeDescriptorConfig.from_mapping(raw["config"])
        components = _mapping(raw["components"], "descriptor_components")
        if set(components) != set(config.components):
            _fail("descriptor_components_mismatch")
        payload = {key: _json_copy(raw[key]) for key in raw if key != "descriptor_hash"}
        expected = "phenotype_descriptor_sha256:" + _digest(
            "astevolve.effective_phenotype_descriptor.v1", payload
        )
        if raw["descriptor_hash"] != expected:
            _fail("descriptor_hash_mismatch")
        return cls(
            config=config,
            accepted_artifact_hash=_nonempty(
                raw["accepted_artifact_hash"], "accepted_artifact_hash"
            ),
            archive_niche_hash=_nonempty(raw["archive_niche_hash"], "archive_niche_hash"),
            _components_json=_canonical_json(components),
            descriptor_hash=expected,
        )


__all__ = [
    "ACCEPTED_RUNTIME_ARTIFACT_VERSION",
    "DESCRIPTOR_COMPONENTS",
    "EFFECTIVE_PHENOTYPE_DESCRIPTOR_VERSION",
    "EFFECTIVE_PHENOTYPE_IDENTITY_VERSION",
    "PHENOTYPE_DESCRIPTOR_CONFIG_VERSION",
    "RUNTIME_ACCEPTANCE_VERSION",
    "AcceptedRuntimeArtifact",
    "EffectivePhenotypeDescriptor",
    "EffectivePhenotypeError",
    "EffectivePhenotypeIdentity",
    "PhenotypeDescriptorConfig",
]
