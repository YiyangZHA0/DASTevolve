

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence

from outerloop.structured_ast_proposal import STRUCTURED_AST_PROPOSAL_MODE


REASONING_PREDICTION_VERSION = "astevolve.reasoning_prediction.v1"
REASONING_OBSERVATION_VERSION = "astevolve.reasoning_observation.v1"
REASONING_LEDGER_VERSION = "astevolve.reasoning_audit_ledger.v1"
_STAGES = ("planned", "compiled", "executed", "sequence_changed", "accepted")


class ReasoningAuditError(ValueError):
    pass


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def seal_reasoning_prediction(
    *,
    proposal_id: str,
    parent_program_id: str,
    iteration: int,
    island_id: int,
    audit: Mapping[str, Any],
    ast_revision_plan: Mapping[str, Any],
    hierarchical_design_hash: str = "",
) -> Dict[str, Any]:


    if str(audit.get("schema_version")) != "astevolve.structured_ast_audit.v2":
        return {}
    core = {
        "schema_version": REASONING_PREDICTION_VERSION,
        "proposal_id": str(proposal_id),
        "parent_program_id": str(parent_program_id),
        "iteration": int(iteration),
        "island_id": int(island_id),
        "hierarchical_design_hash": str(hierarchical_design_hash or ""),
        "hypothesis_id": str(audit.get("hypothesis_id") or ""),
        "strategy_epoch_id": str(audit.get("strategy_epoch_id") or ""),
        "prediction": {
            "target_node_ids": list(audit.get("target_node_ids") or []),
            "target_segments": list(audit.get("target_segments") or []),
            "target_residue_refs": list(audit.get("target_residue_refs") or []),
            "coordinate_system": deepcopy(dict(audit.get("coordinate_system") or {})),
            "evidence_refs": list(audit.get("evidence_refs") or []),
            "expected_effects": deepcopy(list(audit.get("expected_effects") or [])),
            "absolute_quality_floors": deepcopy(
                list(audit.get("absolute_quality_floors") or [])
            ),
            "failure_condition": str(audit.get("failure_condition") or ""),
            "falsification_test": str(audit.get("falsification_test") or ""),
            "rollback_condition": str(audit.get("rollback_condition") or ""),
            "rollback_target": str(audit.get("rollback_target") or ""),
            "planned_ablation": str(audit.get("planned_ablation") or ""),
        },
        "ast_revision_plan_hash": _digest(ast_revision_plan),
        "sealed_before_evaluation": True,
    }
    return {**core, "prediction_hash": "reasoning_prediction_sha256:" + _digest(core)}


def structured_reasoning_artifacts(
    structured_application: Any,
    prediction: Mapping[str, Any] | None,
) -> Dict[str, Any]:


    artifacts: Dict[str, Any] = {}
    if structured_application is not None:
        artifacts["structured_ast_proposal_audit"] = {
            "schema_version": "astevolve.structured_ast_proposal_audit_artifact.v1",
            "proposal_mode": STRUCTURED_AST_PROPOSAL_MODE,
            "parent_evolve_hash": str(
                getattr(structured_application, "parent_evolve_hash", "") or ""
            ),
            "audit": deepcopy(
                dict(getattr(structured_application, "audit", {}) or {})
            ),
            "ast_revision_plan": deepcopy(
                dict(getattr(structured_application, "plan", {}) or {})
            ),
            "contains_hidden_chain_of_thought": False,
        }
    if isinstance(prediction, Mapping) and prediction:
        artifacts["sealed_reasoning_prediction"] = deepcopy(dict(prediction))
    return artifacts


def validate_reasoning_prediction(value: Mapping[str, Any]) -> Dict[str, Any]:
    raw = deepcopy(dict(value))
    supplied = raw.pop("prediction_hash", None)
    if raw.get("schema_version") != REASONING_PREDICTION_VERSION:
        raise ReasoningAuditError("unsupported reasoning prediction schema")
    expected = "reasoning_prediction_sha256:" + _digest(raw)
    if supplied != expected:
        raise ReasoningAuditError("reasoning prediction hash mismatch")
    return {**raw, "prediction_hash": supplied}


def _descriptor_components(artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    descriptor = _mapping(artifacts.get("effective_phenotype_descriptor"))
    return _mapping(descriptor.get("components"))


def _numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _residue_ref(value: str) -> str:
    fields = str(value).split(":")
    return ":".join(fields[-2:]) if len(fields) >= 2 else str(value)


def _parse_residue_ref(value: Any) -> tuple[str, int] | None:
    fields = str(value).rsplit(":", 1)
    if len(fields) != 2:
        return None
    try:
        position = int(fields[1])
    except ValueError:
        return None
    return fields[0], position


def _shift_residue_ref(value: Any, delta: int) -> str:
    parsed = _parse_residue_ref(value)
    if parsed is None:
        return _residue_ref(str(value))
    return f"{parsed[0]}:{parsed[1] + int(delta)}"


def _migration_frontier(
    hierarchical_design: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    found: Dict[str, Any] = {}

    def visit(item: Any) -> None:
        nonlocal found
        if found:
            return
        if isinstance(item, Mapping):
            if item.get("schema_version") == "astevolve.migration_frontier.v1":
                found = deepcopy(dict(item))
                return
            for child in item.values():
                visit(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for child in item:
                visit(child)

    if isinstance(hierarchical_design, Mapping):
        visit(hierarchical_design)
    return found


def build_reasoning_observation(
    prediction: Mapping[str, Any],
    *,
    parent_program: Any,
    child_program: Any,
    artifacts: Mapping[str, Any],
    hierarchical_design: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:


    sealed = validate_reasoning_prediction(prediction)
    public = _mapping(sealed.get("prediction"))
    proposal_artifact = _mapping(artifacts.get("structured_ast_proposal_audit"))
    plan = _mapping(proposal_artifact.get("ast_revision_plan"))
    plan_nodes = {
        str(item.get("node_id"))
        for item in plan.get("structural_nodes", [])
        if isinstance(item, Mapping) and item.get("node_id")
    }
    components = _descriptor_components(artifacts)
    coverage = _mapping(components.get("node_action_coverage"))
    topology = _mapping(components.get("mutation_topology"))
    transitions = topology.get("transitions") or []
    changed_nodes = {
        str(item.get("node_id"))
        for item in transitions
        if isinstance(item, Mapping) and item.get("node_id")
    }


    mutated_sites = {
        _residue_ref(item) for item in topology.get("unique_mutated_sites", [])
    }
    frontier = _migration_frontier(hierarchical_design)
    incumbent_by_position: Dict[str, bool] = {}
    segment_by_position: Dict[str, str] = {}
    frontier_indexing = str(frontier.get("indexing") or "zero_based")
    frontier_to_internal = -1 if frontier_indexing == "one_based" else 0
    for segment in frontier.get("segments", []) if frontier else []:
        if not isinstance(segment, Mapping):
            continue
        chain = str(segment.get("chain_id") or "")
        segment_id = str(segment.get("compiled_segment") or "")
        for row in segment.get("positions", []):
            if not isinstance(row, Mapping) or not isinstance(row.get("position"), int):
                continue
            reference = f"{chain}:{int(row['position']) + frontier_to_internal}"
            incumbent_by_position[reference] = bool(row.get("incumbent_node_id"))
            segment_by_position[reference] = segment_id
    mutated_incumbent = sorted(
        reference for reference in mutated_sites if incumbent_by_position.get(reference) is True
    )
    mutated_non_incumbent = sorted(
        reference for reference in mutated_sites if incumbent_by_position.get(reference) is False
    )
    target_nodes = set(map(str, public.get("target_node_ids", [])))
    coordinate = _mapping(public.get("coordinate_system"))
    audit_index_base = int(coordinate.get("index_base", 1) or 1)
    target_residue_refs = {
        _residue_ref(item) for item in public.get("target_residue_refs", [])
    }
    target_residues = {
        _shift_residue_ref(item, -audit_index_base)
        for item in target_residue_refs
    }
    compiled = target_nodes & plan_nodes
    executed = compiled & set(map(str, coverage.get("executed_structural_nodes", [])))
    changed = executed & changed_nodes
    if executed and target_residues & mutated_sites and not changed_nodes:
        changed = set(executed)
    child_metrics = deepcopy(dict(getattr(child_program, "metrics", {}) or {}))
    parent_metrics = deepcopy(dict(getattr(parent_program, "metrics", {}) or {}))
    accepted_runtime = _mapping(artifacts.get("accepted_runtime_artifact"))
    feasibility = _mapping(accepted_runtime.get("feasibility"))
    hard_gate = child_metrics.get("hard_gate_pass")
    runtime_feasible = feasibility.get("feasible")
    accepted = set(changed) if hard_gate is not False and runtime_feasible is not False else set()
    stage_values = {
        "planned": sorted(target_nodes),
        "compiled": sorted(compiled),
        "executed": sorted(executed),
        "sequence_changed": sorted(changed),
        "accepted": sorted(accepted),
    }
    previous: set[str] | None = None
    for stage in _STAGES:
        current = set(stage_values[stage])
        if previous is not None and not current <= previous:
            raise ReasoningAuditError(f"reasoning scope stage {stage} is not a subset")
        previous = current

    calibrations = []
    for expectation in public.get("expected_effects", []):
        if not isinstance(expectation, Mapping):
            continue
        metric_id = str(expectation.get("metric_id") or "")
        before = _numeric(parent_metrics.get(metric_id))
        after = _numeric(child_metrics.get(metric_id))
        observed = "unmeasured"
        if before is not None and after is not None:
            observed = "increase" if after > before else "decrease" if after < before else "maintain"
        expected = str(expectation.get("direction") or "")
        confidence = _numeric(expectation.get("confidence"))
        matched = observed == expected
        calibrations.append(
            {
                "metric_id": metric_id,
                "expected_direction": expected,
                "observed_direction": observed,
                "parent_value": before,
                "child_value": after,
                "matched": matched,
                "confidence": confidence,
                "brier": (
                    (float(confidence) - (1.0 if matched else 0.0)) ** 2
                    if confidence is not None and observed != "unmeasured"
                    else None
                ),
            }
        )

    floor_results = []
    for floor in public.get("absolute_quality_floors", []):
        if not isinstance(floor, Mapping):
            continue
        metric_id = str(floor.get("metric_id") or "")
        before = _numeric(parent_metrics.get(metric_id))
        after = _numeric(child_metrics.get(metric_id))
        threshold = _numeric(floor.get("threshold"))
        comparison = str(floor.get("comparison") or "")
        direction = str(floor.get("direction") or "")
        boundary = threshold
        if comparison == "relative_to_parent" and before is not None and threshold is not None:
            boundary = before * threshold
        passed = None
        if after is not None and boundary is not None:
            passed = after >= boundary if direction == "min" else after <= boundary
        floor_results.append(
            {
                "metric_id": metric_id,
                "comparison": comparison,
                "direction": direction,
                "threshold": threshold,
                "effective_boundary": boundary,
                "observed_value": after,
                "passed": passed,
            }
        )
    rollback = bool(
        hard_gate is False
        or runtime_feasible is False
        or any(item["passed"] is False for item in floor_results)
    )
    classified_sites = set(mutated_incumbent) | set(mutated_non_incumbent)
    public_mutated_sites = {
        _shift_residue_ref(item, audit_index_base) for item in mutated_sites
    }
    public_incumbent = [
        _shift_residue_ref(item, audit_index_base) for item in mutated_incumbent
    ]
    public_non_incumbent = [
        _shift_residue_ref(item, audit_index_base) for item in mutated_non_incumbent
    ]
    public_unclassified = sorted(
        _shift_residue_ref(item, audit_index_base)
        for item in mutated_sites - classified_sites
    )
    core = {
        "schema_version": REASONING_OBSERVATION_VERSION,
        "prediction_hash": sealed["prediction_hash"],
        "proposal_id": sealed["proposal_id"],
        "hypothesis_id": sealed["hypothesis_id"],
        "island_id": sealed["island_id"],
        "parent_program_id": getattr(parent_program, "id", None),
        "child_program_id": getattr(child_program, "id", None),
        "scope_trace": stage_values,
        "target_residue_refs": sorted(target_residue_refs),
        "mutated_residue_refs": sorted(public_mutated_sites),
        "mutated_incumbent_residue_refs": sorted(public_incumbent),
        "mutated_non_incumbent_residue_refs": sorted(public_non_incumbent),
        "mutated_unclassified_residue_refs": public_unclassified,
        "incumbent_classification_available": bool(frontier),
        "non_incumbent_mutation_fraction": (
            len(mutated_non_incumbent) / len(classified_sites)
            if frontier and classified_sites
            else None
        ),
        "mutated_segments": sorted(
            {
                segment_by_position[reference]
                for reference in mutated_sites
                if reference in segment_by_position
            }
        ),
        "metric_calibration": calibrations,
        "quality_floor_results": floor_results,
        "hard_gate_pass": hard_gate,
        "runtime_feasible": runtime_feasible,
        "rollback_recommended": rollback,
        "rollback_target": sealed["parent_program_id"] if rollback else None,
    }
    return {**core, "observation_hash": "reasoning_observation_sha256:" + _digest(core)}


def build_failed_reasoning_observation(
    prediction: Mapping[str, Any],
    *,
    parent_program: Any,
    artifacts: Mapping[str, Any],
    error: Any,
    hierarchical_design: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:


    base = build_reasoning_observation(
        prediction,
        parent_program=parent_program,
        child_program=None,
        artifacts=artifacts,
        hierarchical_design=hierarchical_design,
    )
    core = deepcopy(dict(base))
    core.pop("observation_hash", None)
    scope = _mapping(core.get("scope_trace"))
    core["attempt_outcome"] = {
        "status": "failed_without_child",
        "planned": bool(scope.get("planned")),
        "compiled": bool(scope.get("compiled")),
        "executed": False,
        "sequence_changed": False,
        "accepted": False,
        "error": str(error or "proposal_failed")[:1000],
    }
    core["rollback_recommended"] = True
    core["rollback_target"] = validate_reasoning_prediction(prediction)[
        "parent_program_id"
    ]
    return {
        **core,
        "observation_hash": "reasoning_observation_sha256:" + _digest(core),
    }


class ReasoningAuditLedger:


    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: Dict[str, Dict[str, Any]] = {}
        if self.path.is_file():
            self.load(self.path)

    def record(
        self,
        prediction: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> None:
        sealed = validate_reasoning_prediction(prediction)
        key = sealed["prediction_hash"]
        existing = self.records.get(key)
        value = {"prediction": sealed, "observation": deepcopy(dict(observation))}
        if existing is not None and existing != value:
            raise ReasoningAuditError("sealed reasoning record cannot be overwritten")
        self.records[key] = value
        self.save()

    def aggregate(self) -> Dict[str, Any]:
        rows = [item["observation"] for item in self.records.values()]
        calibration = [
            metric
            for row in rows
            for metric in row.get("metric_calibration", [])
            if isinstance(metric, Mapping) and metric.get("observed_direction") != "unmeasured"
        ]
        briers = [item["brier"] for item in calibration if _numeric(item.get("brier")) is not None]
        mutated_total = sum(len(row.get("mutated_residue_refs", [])) for row in rows)
        classified_total = sum(
            len(row.get("mutated_incumbent_residue_refs", []))
            + len(row.get("mutated_non_incumbent_residue_refs", []))
            for row in rows
        )
        non_incumbent_total = sum(
            len(row.get("mutated_non_incumbent_residue_refs", [])) for row in rows
        )
        segments = sorted(
            {
                str(segment)
                for row in rows
                for segment in row.get("mutated_segments", [])
            }
        )
        return {
            "record_count": len(rows),
            "hard_gate_pass_count": sum(item.get("hard_gate_pass") is True for item in rows),
            "rollback_recommended_count": sum(bool(item.get("rollback_recommended")) for item in rows),
            "prediction_match_rate": (
                sum(bool(item.get("matched")) for item in calibration) / len(calibration)
                if calibration
                else None
            ),
            "mean_brier": sum(briers) / len(briers) if briers else None,
            "mutated_residue_count": mutated_total,
            "incumbent_classified_mutated_residue_count": classified_total,
            "non_incumbent_mutated_residue_count": non_incumbent_total,
            "non_incumbent_mutation_fraction": (
                non_incumbent_total / classified_total if classified_total else None
            ),
            "mutated_segment_count": len(segments),
            "by_segment": {
                segment: sum(segment in row.get("mutated_segments", []) for row in rows)
                for segment in segments
            },
            "stage_reach": {
                stage: sum(bool(row.get("scope_trace", {}).get(stage)) for row in rows)
                for stage in _STAGES
            },
            "by_island": {
                str(island): sum(row.get("island_id") == island for row in rows)
                for island in sorted({row.get("island_id") for row in rows if row.get("island_id") is not None})
            },
        }
    def snapshot(self) -> Dict[str, Any]:
        material = {
            "schema_version": REASONING_LEDGER_VERSION,
            "records": [self.records[key] for key in sorted(self.records)],
            "aggregate": self.aggregate(),
        }
        return {**deepcopy(material), "ledger_hash": "reasoning_ledger_sha256:" + _digest(material)}

    def save(self, path: str | Path | None = None) -> None:
        _atomic(Path(path) if path else self.path, self.snapshot())

    def load(self, path: str | Path) -> None:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping) or value.get("schema_version") != REASONING_LEDGER_VERSION:
            raise ReasoningAuditError("invalid reasoning ledger schema")
        supplied = value.get("ledger_hash")
        material = {key: value[key] for key in ("schema_version", "records", "aggregate")}
        if supplied != "reasoning_ledger_sha256:" + _digest(material):
            raise ReasoningAuditError("reasoning ledger hash mismatch")
        records = value.get("records")
        if not isinstance(records, list):
            raise ReasoningAuditError("reasoning ledger records must be a list")
        self.records = {}
        for item in records:
            if not isinstance(item, Mapping):
                raise ReasoningAuditError("reasoning ledger record must be an object")
            prediction = validate_reasoning_prediction(item.get("prediction") or {})
            self.records[prediction["prediction_hash"]] = deepcopy(dict(item))
        if self.aggregate() != value.get("aggregate"):
            raise ReasoningAuditError("reasoning ledger aggregate mismatch")


__all__ = [
    "REASONING_LEDGER_VERSION",
    "REASONING_OBSERVATION_VERSION",
    "REASONING_PREDICTION_VERSION",
    "ReasoningAuditError",
    "ReasoningAuditLedger",
    "build_failed_reasoning_observation",
    "build_reasoning_observation",
    "seal_reasoning_prediction",
    "structured_reasoning_artifacts",
    "validate_reasoning_prediction",
]
