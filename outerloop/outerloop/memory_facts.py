

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from engine.experiment_identity import CodeIdentity, SequenceBundleIdentity
from engine.memory_policy import MemoryPolicyError, MemoryScope
from outerloop.utils.metric_semantics import compare_metrics, summarize_comparisons


OUTER_OBSERVATION_FACT_VERSION = "astevolve.outer_observation_fact.v1"
OUTER_FACT_LEDGER_VERSION = "astevolve.outer_memory_fact_ledger.v1"
OUTER_MEMORY_PROJECTION_VERSION = "astevolve.outer_memory_projection.v1"


_FACT_LEDGER_JSON_PREFIX = b'{"facts":['
_FACT_LEDGER_JSON_SUFFIX = (
    b'],"schema_version":"' + OUTER_FACT_LEDGER_VERSION.encode("utf-8") + b'"}'
)


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"unsupported fact value: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"memory fact is not canonically serializable: {exc}") from exc


def _safe_mapping(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return deepcopy({str(key): item for key, item in value.items()})


def _safe_metric_mapping(value: Any) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, item in (_safe_mapping(value)).items():
        if isinstance(item, bool) or item is None or isinstance(item, str):
            output[key] = item
        elif isinstance(item, (int, float)):
            numeric = float(item)
            if math.isfinite(numeric):
                output[key] = numeric
        elif isinstance(item, Mapping):
            output[key] = _safe_mapping(item)
    return output


@dataclass(frozen=True)
class OuterObservationFact:
    scope: MemoryScope
    generation_id: str
    proposal_id: str
    logical_time: str
    iteration: int
    program_id: Optional[str]
    parent_id: Optional[str]
    hypothesis: Optional[str]
    effective_contract: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    parent_metrics: Mapping[str, Any] = field(default_factory=dict)
    gate: Mapping[str, Any] = field(default_factory=dict)
    failure: Optional[Mapping[str, Any]] = None
    sequence_fingerprint: Optional[str] = None
    evaluator_provenance: Mapping[str, Any] = field(default_factory=dict)
    code_hash: Optional[str] = None
    effective_contract_hash: Optional[str] = None
    sequence_bundle_hash: Optional[str] = None
    applied_action: Mapping[str, Any] = field(default_factory=dict)
    explored_nodes: tuple[str, ...] = ()
    explored_operators: tuple[str, ...] = ()
    trial_id: Optional[str] = None
    schema_version: str = OUTER_OBSERVATION_FACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OUTER_OBSERVATION_FACT_VERSION:
            raise ValueError(
                f"unsupported outer observation fact version: {self.schema_version!r}"
            )
        if not isinstance(self.scope, MemoryScope):
            raise TypeError("outer observation fact scope must be MemoryScope")
        for name in ("generation_id", "proposal_id", "logical_time"):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"outer observation fact {name} must be non-empty")
        if int(self.iteration) < 0:
            raise ValueError("outer observation fact iteration cannot be negative")
        _canonical_json(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OuterObservationFact":
        if not isinstance(value, Mapping):
            raise ValueError("outer observation fact must be a mapping")
        allowed = {
            "schema_version",
            "scope",
            "generation_id",
            "proposal_id",
            "logical_time",
            "iteration",
            "program_id",
            "parent_id",
            "hypothesis",
            "effective_contract",
            "metrics",
            "parent_metrics",
            "gate",
            "failure",
            "sequence_fingerprint",
            "evaluator_provenance",
            "code_hash",
            "effective_contract_hash",
            "sequence_bundle_hash",
            "applied_action",
            "explored_nodes",
            "explored_operators",
            "trial_id",
        }
        unknown = sorted(str(key) for key in set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown outer observation fact field(s): {', '.join(unknown)}")
        failure = value.get("failure")
        return cls(
            scope=MemoryScope.from_mapping(value.get("scope") or {}),
            generation_id=str(value.get("generation_id") or ""),
            proposal_id=str(value.get("proposal_id") or ""),
            logical_time=str(value.get("logical_time") or ""),
            iteration=int(value.get("iteration") or 0),
            program_id=(str(value["program_id"]) if value.get("program_id") else None),
            parent_id=(str(value["parent_id"]) if value.get("parent_id") else None),
            hypothesis=(str(value["hypothesis"]) if value.get("hypothesis") else None),
            effective_contract=_safe_mapping(value.get("effective_contract")),
            metrics=_safe_metric_mapping(value.get("metrics")),
            parent_metrics=_safe_metric_mapping(value.get("parent_metrics")),
            gate=_safe_mapping(value.get("gate")),
            failure=_safe_mapping(failure) if isinstance(failure, Mapping) else None,
            sequence_fingerprint=(
                str(value["sequence_fingerprint"])
                if value.get("sequence_fingerprint")
                else None
            ),
            evaluator_provenance=_safe_mapping(value.get("evaluator_provenance")),
            code_hash=(str(value["code_hash"]) if value.get("code_hash") else None),
            effective_contract_hash=(
                str(value["effective_contract_hash"])
                if value.get("effective_contract_hash")
                else None
            ),
            sequence_bundle_hash=(
                str(value["sequence_bundle_hash"])
                if value.get("sequence_bundle_hash")
                else None
            ),
            applied_action=_safe_mapping(value.get("applied_action")),
            explored_nodes=tuple(
                str(item) for item in (value.get("explored_nodes") or [])
                if str(item).strip()
            ),
            explored_operators=tuple(
                str(item) for item in (value.get("explored_operators") or [])
                if str(item).strip()
            ),
            trial_id=(str(value["trial_id"]) if value.get("trial_id") else None),
            schema_version=str(
                value.get("schema_version") or OUTER_OBSERVATION_FACT_VERSION
            ),
        )

    @property
    def chronology_key(self) -> tuple[int, str, str, str, str, str]:


        return (
            int(self.iteration),
            self.logical_time,
            self.generation_id,
            self.proposal_id,
            self.trial_id or "",
            self.program_id or "",
        )

    @property
    def stable_key(self) -> tuple[int, str, str, str, str, str]:


        return self.chronology_key

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.to_artifact(),
            "generation_id": self.generation_id,
            "proposal_id": self.proposal_id,
            "trial_id": self.trial_id,
            "logical_time": self.logical_time,
            "iteration": int(self.iteration),
            "program_id": self.program_id,
            "parent_id": self.parent_id,
            "hypothesis": self.hypothesis,
            "effective_contract": _safe_mapping(self.effective_contract),
            "metrics": _safe_metric_mapping(self.metrics),
            "parent_metrics": _safe_metric_mapping(self.parent_metrics),
            "gate": _safe_mapping(self.gate),
            "failure": _safe_mapping(self.failure) if self.failure is not None else None,
            "sequence_fingerprint": self.sequence_fingerprint,
            "evaluator_provenance": _safe_mapping(self.evaluator_provenance),
            "code_hash": self.code_hash,
            "effective_contract_hash": self.effective_contract_hash,
            "sequence_bundle_hash": self.sequence_bundle_hash,
            "applied_action": _safe_mapping(self.applied_action),
            "explored_nodes": list(self.explored_nodes),
            "explored_operators": list(self.explored_operators),
        }


def _program_mapping(program: Any) -> Dict[str, Any]:
    if program is None:
        return {}
    if hasattr(program, "to_dict"):
        value = program.to_dict()
    elif isinstance(program, Mapping):
        value = dict(program)
    else:
        value = vars(program)
    return _safe_mapping(value)


def _evaluator_provenance(artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    explicit = artifacts.get("evaluator_provenance")
    if isinstance(explicit, Mapping):
        return _safe_mapping(explicit)
    report = artifacts.get("evaluator_report")
    if isinstance(report, Mapping):
        provenance = report.get("provenance")
        if isinstance(provenance, Mapping):
            return _safe_mapping(provenance)
    resolution = artifacts.get("evaluator_plugin_resolution")
    if isinstance(resolution, Mapping):
        return {"plugin_resolution": _safe_mapping(resolution)}
    return {"status": "unavailable", "reason": "evaluator_provenance_not_recorded"}


def _mapping_artifact(value: Any) -> Dict[str, Any]:


    if isinstance(value, Mapping):
        return _safe_mapping(value)
    if isinstance(value, (str, bytes)):
        try:
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            parsed = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return _safe_mapping(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _effective_contract_artifact(artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    for key in ("effective_search_contract", "effective_contract"):
        value = _mapping_artifact(artifacts.get(key))
        if value:
            return value
    trace = _mapping_artifact(artifacts.get("causal_trace"))
    return _mapping_artifact(trace.get("effective_contract"))


def _code_hash(program_data: Mapping[str, Any], artifacts: Mapping[str, Any]) -> Optional[str]:
    explicit = artifacts.get("code_hash")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    identity = _mapping_artifact(artifacts.get("code_identity"))
    if identity.get("code_hash"):
        return str(identity["code_hash"])
    code = program_data.get("code")
    if isinstance(code, str):
        return CodeIdentity.from_text(code).code_hash
    return None


def _sequence_bundle_hash(artifacts: Mapping[str, Any]) -> Optional[str]:
    for key in ("sequence_bundle_hash", "sequence_hash", "sequence_fingerprint"):
        value = artifacts.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    identity = _mapping_artifact(artifacts.get("sequence_bundle_identity"))
    if identity.get("sequence_bundle_hash"):
        return str(identity["sequence_bundle_hash"])
    trace = _mapping_artifact(artifacts.get("causal_trace"))
    if trace.get("final_sequence_id"):
        return str(trace["final_sequence_id"])
    for key in ("best_seqs", "seqs"):
        seqs = artifacts.get(key)
        if isinstance(seqs, Mapping) and seqs:
            try:
                return SequenceBundleIdentity.create(seqs).sequence_bundle_hash
            except (TypeError, ValueError):
                pass
    lifecycle = _mapping_artifact(artifacts.get("selected_memory_lifecycle"))
    if lifecycle.get("sequence_hash"):
        return str(lifecycle["sequence_hash"])
    return None


_ACTION_FIELDS = (
    "action_id",
    "semantic_id",
    "edge_id",
    "functional_node_id",
    "structural_node_id",
    "node_id",
    "operator",
    "positions",
    "measurement_id",
)


def _action_rows(artifacts: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    trace = _mapping_artifact(artifacts.get("causal_trace"))
    for value in trace.get("actions") or []:
        if not isinstance(value, Mapping):
            continue
        row = _safe_mapping(value)
        parameters = _mapping_artifact(row.get("parameters"))
        for key in _ACTION_FIELDS:
            if parameters.get(key) not in (None, "", [], {}) and not row.get(key):
                row[key] = deepcopy(parameters[key])
        rows.append(row)
    mapping_execution = _mapping_artifact(artifacts.get("mapping_execution"))
    for value in mapping_execution.get("traces") or []:
        if isinstance(value, Mapping):
            row = _safe_mapping(value)
            nested_action = _mapping_artifact(row.get("action"))
            nested_parameters = _mapping_artifact(nested_action.get("parameters"))
            for source in (nested_action, nested_parameters):
                for key in _ACTION_FIELDS:
                    if source.get(key) not in (None, "", [], {}) and not row.get(key):
                        row[key] = deepcopy(source[key])
            rows.append(row)
    return rows


def _gate_from_artifacts(
    artifacts: Mapping[str, Any],
    *,
    fallback: Mapping[str, Any],
) -> Dict[str, Any]:
    report = _mapping_artifact(artifacts.get("evaluator_report"))
    status = _mapping_artifact(report.get("gate_status"))
    passed = status.get("hard_gate_pass")
    if not isinstance(passed, bool):
        passed = status.get("passed")
    if not isinstance(passed, bool):
        passed = report.get("hard_gate_pass")
    reasons: list[str] = []
    for value in (
        status.get("hard_failures"),
        status.get("reasons"),
        report.get("disqualification_reasons"),
        report.get("gate_reasons"),
    ):
        if isinstance(value, (list, tuple)):
            reasons.extend(str(item) for item in value if str(item).strip())
    if isinstance(passed, bool):
        return {
            "known": True,
            "passed": passed,
            "reasons": sorted(set(reasons)) if not passed else [],
        }
    return _safe_mapping(fallback)


def _applied_action(artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    rows = _action_rows(artifacts)
    if not rows:
        return {
            "status": "unavailable",
            "reason": "executed_action_not_recorded",
        }
    selected = rows[-1]
    return {
        key: deepcopy(selected[key])
        for key in _ACTION_FIELDS
        if selected.get(key) not in (None, "", [], {})
    }


def _explored_actions(artifacts: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    nodes = set()
    operators = set()
    for row in _action_rows(artifacts):
        for key in ("structural_node_id", "node_id"):
            if row.get(key):
                nodes.add(str(row[key]))
        if row.get("operator"):
            operators.add(str(row["operator"]))
    return tuple(sorted(nodes)), tuple(sorted(operators))


def build_observation_fact(
    *,
    scope: MemoryScope,
    generation_id: str,
    proposal_id: str,
    logical_time: str,
    iteration: int,
    program: Any,
    parent: Any,
    artifacts: Optional[Mapping[str, Any]] = None,
    error: Optional[str] = None,
) -> OuterObservationFact:


    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    program_data = _program_mapping(program)
    parent_data = _program_mapping(parent)
    metrics = _safe_metric_mapping(program_data.get("metrics"))
    parent_metrics = _safe_metric_mapping(parent_data.get("metrics"))
    hard_gate = metrics.get("hard_gate_pass")
    disqualified = metrics.get("disqualified")
    if isinstance(hard_gate, bool):
        gate = {
            "known": True,
            "passed": hard_gate,
            "reasons": [] if hard_gate else ["hard_gate_failed"],
        }
    elif isinstance(disqualified, bool):
        gate = {
            "known": True,
            "passed": not disqualified,
            "reasons": [] if not disqualified else ["disqualified"],
        }
    else:
        gate = {
            "known": False,
            "passed": None,
            "reasons": ["gate_not_recorded"],
        }
    gate = _gate_from_artifacts(artifacts, fallback=gate)
    raw_failure = None
    artifact_error = artifacts.get("error")
    if error or program is None or artifact_error:
        raw_failure = {
            "reason": str(error or artifact_error or "candidate_failed"),
            "stage": "outer_candidate",
        }
    contract = _effective_contract_artifact(artifacts)
    if not contract:
        contract = _mapping_artifact(artifacts.get("applied_edit_contract"))
    if not contract:
        contract = {
            "status": "unavailable",
            "reason": "effective_contract_not_recorded",
        }
    sequence_fingerprint = _sequence_bundle_hash(artifacts)
    graph_patch = _mapping_artifact(artifacts.get("graph_patch"))
    hypothesis = (
        graph_patch.get("hypothesis")
        or artifacts.get("hypothesis")
        or program_data.get("changes_description")
        or (program_data.get("metadata") or {}).get("hypothesis")
        or (program_data.get("metadata") or {}).get("changes")
    )
    trial_id = None
    lifecycle = artifacts.get("selected_memory_lifecycle")
    if isinstance(lifecycle, Mapping) and lifecycle.get("selected_trial_id"):
        trial_id = str(lifecycle["selected_trial_id"])
    explored_nodes, explored_operators = _explored_actions(artifacts)
    effective_contract_hash = (
        artifacts.get("effective_contract_hash")
        or (contract.get("contract_hash") if contract else None)
    )
    return OuterObservationFact(
        scope=scope,
        generation_id=str(generation_id),
        proposal_id=str(proposal_id),
        trial_id=trial_id,
        logical_time=str(logical_time),
        iteration=int(iteration),
        program_id=str(program_data.get("id")) if program_data.get("id") else None,
        parent_id=(
            str(program_data.get("parent_id") or parent_data.get("id"))
            if (program_data.get("parent_id") or parent_data.get("id"))
            else None
        ),
        hypothesis=str(hypothesis) if hypothesis else None,
        effective_contract=_safe_mapping(contract),
        metrics=metrics,
        parent_metrics=parent_metrics,
        gate=gate,
        failure=raw_failure,
        sequence_fingerprint=str(sequence_fingerprint) if sequence_fingerprint else None,
        evaluator_provenance=_evaluator_provenance(artifacts),
        code_hash=_code_hash(program_data, artifacts),
        effective_contract_hash=(
            str(effective_contract_hash) if effective_contract_hash else None
        ),
        sequence_bundle_hash=(
            str(sequence_fingerprint) if sequence_fingerprint else None
        ),
        applied_action=_applied_action(artifacts),
        explored_nodes=explored_nodes,
        explored_operators=explored_operators,
    )


def _ordered_facts(facts: Iterable[OuterObservationFact]) -> list[OuterObservationFact]:
    values = list(facts)
    if any(not isinstance(fact, OuterObservationFact) for fact in values):
        raise TypeError("fact ledger accepts only OuterObservationFact values")
    return sorted(values, key=lambda fact: fact.stable_key)


@dataclass(frozen=True)
class _PreparedFact:


    fact: OuterObservationFact
    data: Mapping[str, Any]
    payload_json: str


@dataclass(frozen=True)
class _PreparedFactLedger:
    items: tuple[_PreparedFact, ...]
    source_fact_ledger_hash: str
    open_hasher: Any = field(repr=False, compare=False)


def _finish_fact_ledger_hash(open_hasher: Any) -> str:
    finished = open_hasher.copy()
    finished.update(_FACT_LEDGER_JSON_SUFFIX)
    return finished.hexdigest()


def _prepare_ordered_fact_ledger(
    facts: Iterable[OuterObservationFact],
) -> _PreparedFactLedger:


    return _prepare_fact_ledger_values(_ordered_facts(facts))


def _prepare_fact_ledger_values(
    ordered_facts: Sequence[OuterObservationFact],
) -> _PreparedFactLedger:


    open_hasher = hashlib.sha256()
    open_hasher.update(_FACT_LEDGER_JSON_PREFIX)
    prepared = []
    for index, fact in enumerate(ordered_facts):
        data = fact.to_dict()
        payload_json = _canonical_json(data)
        if index:
            open_hasher.update(b",")
        open_hasher.update(payload_json.encode("utf-8"))
        prepared.append(
            _PreparedFact(
                fact=fact,
                data=data,
                payload_json=payload_json,
            )
        )
    return _PreparedFactLedger(
        items=tuple(prepared),
        source_fact_ledger_hash=_finish_fact_ledger_hash(open_hasher),
        open_hasher=open_hasher,
    )


def _prepare_scoped_fact_ledger(
    facts: Iterable[OuterObservationFact], scope: MemoryScope
) -> _PreparedFactLedger:
    return _prepare_fact_ledger_values(_fact_scope_filter(facts, scope))


def fact_ledger_hash(facts: Iterable[OuterObservationFact]) -> str:
    return _prepare_ordered_fact_ledger(facts).source_fact_ledger_hash


def _fact_scope_filter(
    facts: Iterable[OuterObservationFact], scope: MemoryScope
) -> list[OuterObservationFact]:
    selected = []
    for fact in facts:
        try:
            scope.require_compatible(fact.scope, level="run")
        except MemoryPolicyError:
            continue
        selected.append(fact)
    return _ordered_facts(selected)


def _trajectory_entry(
    fact: OuterObservationFact,
    *,
    fact_data: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:


    data = fact_data if isinstance(fact_data, Mapping) else fact.to_dict()
    parent_metrics = _safe_metric_mapping(data.get("parent_metrics"))
    metrics = _safe_metric_mapping(data.get("metrics"))
    comparisons = compare_metrics(
        parent_metrics,
        metrics,
        names=set(parent_metrics) | set(metrics),
    )
    comparison = summarize_comparisons(comparisons)
    return {
        "generation_id": fact.generation_id,
        "proposal_id": fact.proposal_id,
        "trial_id": fact.trial_id,
        "logical_time": fact.logical_time,
        "iteration": fact.iteration,
        "program_id": fact.program_id,
        "parent_id": fact.parent_id,
        "lineage_id": fact.scope.lineage_id,
        "hypothesis": fact.hypothesis,
        "identity": {
            "code_hash": fact.code_hash,
            "effective_contract_hash": fact.effective_contract_hash,
            "sequence_bundle_hash": fact.sequence_bundle_hash,
        },
        "applied_action": _safe_mapping(data.get("applied_action")),
        "explored": {
            "nodes": list(fact.explored_nodes),
            "operators": list(fact.explored_operators),
        },
        "effective_contract": _safe_mapping(data.get("effective_contract")),
        "metrics": metrics,
        "metric_comparison": comparison,
        "gate": _safe_mapping(data.get("gate")),
        "failure": (
            _safe_mapping(data.get("failure"))
            if data.get("failure") is not None
            else None
        ),
        "sequence_fingerprint": fact.sequence_fingerprint,
        "evaluator_provenance": _safe_mapping(data.get("evaluator_provenance")),
    }


def _primary_score(
    fact: OuterObservationFact,
    *,
    fact_data: Optional[Mapping[str, Any]] = None,
) -> Optional[float]:
    metrics = (
        fact_data.get("metrics")
        if isinstance(fact_data, Mapping)
        else fact.metrics
    )
    if not isinstance(metrics, Mapping):
        return None
    for name in ("raw_combined_score", "combined_score", "evaluator_score", "multistate_score"):
        value = metrics.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                return numeric
    return None


def _candidate_entry(item: _PreparedFact) -> Optional[Dict[str, Any]]:
    fact = item.fact
    score = _primary_score(fact, fact_data=item.data)
    if fact.program_id is None or fact.failure is not None or score is None:
        return None
    return {
        "program_id": fact.program_id,
        "proposal_id": fact.proposal_id,
        "lineage_id": fact.scope.lineage_id,
        "iteration": fact.iteration,
        "combined_score": score,
        "metrics": _safe_metric_mapping(item.data.get("metrics")),
        "gate": _safe_mapping(item.data.get("gate")),
        "sequence_fingerprint": fact.sequence_fingerprint,
    }


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[float, str]:
    return (
        -(row["combined_score"] if row["combined_score"] is not None else -1e18),
        str(row["program_id"]),
    )


class _OptimizerMemoryProjectionCache:


    def __init__(
        self,
        prepared: _PreparedFactLedger,
        *,
        scope: MemoryScope,
        recent_limit: int,
        best_limit: int,
    ) -> None:
        self.scope = scope
        self.recent_limit = max(1, int(recent_limit))
        self.best_limit = max(1, int(best_limit))
        self._open_hasher = prepared.open_hasher.copy()
        self.source_fact_ledger_hash = prepared.source_fact_ledger_hash
        self.source_fact_count = len(prepared.items)
        self._last_key = (
            prepared.items[-1].fact.stable_key if prepared.items else None
        )
        self._recent_attempts: list[Dict[str, Any]] = []
        self._top_programs: list[Dict[str, Any]] = []
        self._program_summaries: Dict[str, Dict[str, Any]] = {}
        self._consume_prepared(prepared.items, initializing=True)

    def matches(
        self,
        prepared: _PreparedFactLedger,
        *,
        scope: MemoryScope,
        recent_limit: int,
        best_limit: int,
    ) -> bool:
        return bool(
            self.scope == scope
            and self.recent_limit == max(1, int(recent_limit))
            and self.best_limit == max(1, int(best_limit))
            and self.source_fact_count == len(prepared.items)
            and self.source_fact_ledger_hash
            == prepared.source_fact_ledger_hash
        )

    def _consume_prepared(
        self,
        items: Sequence[_PreparedFact],
        *,
        initializing: bool,
    ) -> None:
        candidates = [] if initializing else list(self._top_programs)
        for item in items:
            entry = _trajectory_entry(item.fact, fact_data=item.data)
            self._recent_attempts.append(entry)
            if len(self._recent_attempts) > self.recent_limit:
                del self._recent_attempts[: -self.recent_limit]
            if item.fact.program_id is not None:


                self._program_summaries[item.fact.program_id] = deepcopy(entry)
            candidate = _candidate_entry(item)
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(key=_candidate_sort_key)
        self._top_programs = candidates[: self.best_limit]

    def extend(
        self,
        facts: Iterable[OuterObservationFact],
        *,
        expected_source_fact_ledger_hash: str,
    ) -> None:
        if expected_source_fact_ledger_hash != self.source_fact_ledger_hash:
            raise ValueError("optimizer-memory ledger cache hash mismatch")
        prepared = _prepare_scoped_fact_ledger(facts, self.scope)
        if not prepared.items:
            return
        first_key = prepared.items[0].fact.stable_key
        if self._last_key is not None and first_key <= self._last_key:
            raise ValueError(
                "optimizer-memory ledger extension is not strictly append-only"
            )


        derived = []
        for item in prepared.items:
            derived.append(
                (
                    item,
                    _trajectory_entry(item.fact, fact_data=item.data),
                    _candidate_entry(item),
                )
            )

        open_hasher = self._open_hasher.copy()
        count = self.source_fact_count
        for item, _entry, _candidate in derived:
            if count:
                open_hasher.update(b",")
            open_hasher.update(item.payload_json.encode("utf-8"))
            count += 1

        candidates = list(self._top_programs)
        for item, entry, candidate in derived:
            self._recent_attempts.append(entry)
            if len(self._recent_attempts) > self.recent_limit:
                del self._recent_attempts[: -self.recent_limit]
            if item.fact.program_id is not None:
                self._program_summaries[item.fact.program_id] = deepcopy(entry)
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(key=_candidate_sort_key)
        self._top_programs = candidates[: self.best_limit]
        self._open_hasher = open_hasher
        self.source_fact_count = count
        self.source_fact_ledger_hash = _finish_fact_ledger_hash(open_hasher)
        self._last_key = prepared.items[-1].fact.stable_key

    def projection(self) -> Dict[str, Any]:


        recent_attempts = deepcopy(self._recent_attempts)
        top_programs = deepcopy(self._top_programs)
        summaries = deepcopy(self._program_summaries)
        mandatory_prompt_capsule = _mandatory_prompt_capsule(
            recent_attempts,
            source_fact_ledger_hash=self.source_fact_ledger_hash,
        )
        return {
            "schema_version": OUTER_MEMORY_PROJECTION_VERSION,
            "scope": self.scope.to_artifact(),
            "source_fact_ledger_hash": self.source_fact_ledger_hash,
            "source_fact_count": self.source_fact_count,
            "recent_attempts": recent_attempts,
            "top_programs": top_programs,
            "mandatory_prompt_capsule": mandatory_prompt_capsule,
            "trajectory_memory": recent_attempts,
            "best_candidate_memory": top_programs,
            "program_summaries": summaries,
            "projection_only": True,
            "fact_source": "program_database",
        }


def project_optimizer_memory(
    facts: Sequence[OuterObservationFact],
    *,
    scope: MemoryScope,
    recent_limit: int = 20,
    best_limit: int = 5,
) -> Dict[str, Any]:


    prepared = _prepare_scoped_fact_ledger(facts, scope)
    return _OptimizerMemoryProjectionCache(
        prepared,
        scope=scope,
        recent_limit=recent_limit,
        best_limit=best_limit,
    ).projection()


def _bounded_text(value: Any, limit: int = 180) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value)
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _capsule_entry(row: Mapping[str, Any]) -> Dict[str, Any]:
    comparison = _safe_mapping(row.get("metric_comparison"))
    action = _safe_mapping(row.get("applied_action"))
    compact_action = {
        key: deepcopy(action[key])
        for key in (
            "action_id",
            "edge_id",
            "functional_node_id",
            "structural_node_id",
            "node_id",
            "operator",
            "positions",
        )
        if action.get(key) not in (None, "", [], {})
    }
    failure = _safe_mapping(row.get("failure"))
    return {
        "iteration": row.get("iteration"),
        "logical_time": row.get("logical_time"),
        "proposal_id": row.get("proposal_id"),
        "program_id": row.get("program_id"),
        "parent_id": row.get("parent_id"),
        "identity": _safe_mapping(row.get("identity")),
        "hypothesis": _bounded_text(row.get("hypothesis")),
        "applied_action": compact_action,
        "directional_deltas": _safe_mapping(comparison.get("directional_deltas")),
        "outcome": comparison.get("overall_outcome"),
        "gate": _safe_mapping(row.get("gate")),
        "failure": (
            {
                "reason": _bounded_text(failure.get("reason")),
                "stage": failure.get("stage"),
            }
            if failure
            else None
        ),
        "explored": _safe_mapping(row.get("explored")),
    }


def _mandatory_prompt_capsule(
    recent_attempts: Sequence[Mapping[str, Any]],
    *,
    source_fact_ledger_hash: str,
) -> Dict[str, Any]:


    rows = list(recent_attempts)
    selected: list[Mapping[str, Any]] = []
    if rows:
        selected.append(rows[-1])
    for row in reversed(rows):
        if row in selected or not row.get("failure"):
            continue
        selected.append(row)
        if len(selected) >= 3:
            break
    for row in reversed(rows):
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= 3:
            break
    selected.sort(
        key=lambda row: (
            int(row.get("iteration") or 0),
            str(row.get("logical_time") or ""),
            str(row.get("proposal_id") or ""),
        )
    )
    return {
        "schema_version": "astevolve.outer_prompt_capsule.v1",
        "source_fact_ledger_hash": source_fact_ledger_hash,
        "recent_attempts": [_capsule_entry(row) for row in selected],
    }


__all__ = [
    "OUTER_FACT_LEDGER_VERSION",
    "OUTER_MEMORY_PROJECTION_VERSION",
    "OUTER_OBSERVATION_FACT_VERSION",
    "OuterObservationFact",
    "build_observation_fact",
    "fact_ledger_hash",
    "project_optimizer_memory",
]
