

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping


COMPILED_MAPPING_ACTION_VERSION = "astevolve.compiled_mapping_action.v1"
COMPILED_MEASUREMENT_VERSION = "astevolve.compiled_mapping_measurement.v1"
REALIZED_MAPPING_MOVE_VERSION = "astevolve.realized_mapping_move.v1"
MAPPING_EXECUTION_TRACE_VERSION = "astevolve.mapping_execution_trace.v1"

_IDENTITY_FIELDS = (
    "ast_id",
    "ast_revision",
    "edge_id",
    "functional_node_id",
    "structural_node_id",
    "action_id",
    "measurement_id",
)
_ACTION_FIELDS = frozenset(
    {
        "schema_version",
        *_IDENTITY_FIELDS,
        "operator",
        "chain_id",
        "legal_positions",
    }
)
_MEASUREMENT_FIELDS = frozenset(
    {
        "schema_version",
        *_IDENTITY_FIELDS,
        "evaluator_id",
        "term_name",
        "state",
        "direction",
        "threshold",
        "missing_policy",
    }
)
_MOVE_FIELDS = frozenset(
    {
        "schema_version",
        *_IDENTITY_FIELDS,
        "operator",
        "chain_id",
        "positions",
        "evaluated_sequence_id",
    }
)
_TRACE_FIELDS = frozenset(
    {
        "schema_version",
        *_IDENTITY_FIELDS,
        "action",
        "observation",
        "evaluated_sequence_id",
        "trace_hash",
    }
)
_ACTION_ATTRIBUTION_FIELDS = frozenset(
    {
        *_IDENTITY_FIELDS,
        "operator",
        "chain_id",
        "legal_positions",
        "realized_positions",
    }
)
_OBSERVATION_ATTRIBUTION_FIELDS = frozenset(
    {
        "status",
        "evaluator_id",
        "term_name",
        "state",
        "term_provider",
        "term_value",
        "direction",
        "directional_value",
        "threshold",
        "missing_policy",
        "gate",
        "evaluated_sequence_id",
        *_IDENTITY_FIELDS,
    }
)


class MappingExecutionError(ValueError):


    def __init__(self, code: str, message: str = "") -> None:
        self.code = str(code)
        super().__init__(f"{self.code}: {message or self.code}")


def _fail(code: str, message: str = "") -> None:
    raise MappingExecutionError(code, message)


def _strict_mapping(
    value: Any,
    *,
    name: str,
    allowed: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name}_not_mapping")
    keys = {str(key) for key in value}
    unknown = sorted(keys - allowed)
    if unknown:
        _fail(
            f"{name}_unknown_fields",
            ", ".join(unknown),
        )
    missing = sorted(allowed - keys)
    if missing:
        _fail(
            f"{name}_missing_fields",
            ", ".join(missing),
        )
    return value


def _require_version(value: Any, expected: str, *, name: str) -> None:
    if value != expected:
        _fail(f"{name}_schema_version_invalid", repr(value))


def _nonempty(value: Any, *, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        _fail(f"{name}_empty")
    return text


def _revision(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail("ast_revision_invalid")
    return int(value)


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name}_not_numeric")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{name}_not_finite")
    return number


def _positions(value: Any, *, name: str, allow_empty: bool) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        _fail(f"{name}_not_sequence")
    positions: list[int] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            _fail(f"{name}_invalid")
        positions.append(int(raw))
    if len(positions) != len(set(positions)):
        _fail(f"{name}_duplicate")
    if not positions and not allow_empty:
        _fail(f"{name}_empty")
    return sorted(positions)


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
        raise MappingExecutionError(
            "mapping_trace_not_json_safe", str(error)
        ) from error


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _hash_payload(value: Mapping[str, Any]) -> str:
    preimage = (
        MAPPING_EXECUTION_TRACE_VERSION.encode("utf-8")
        + b"\0"
        + _canonical_json(value).encode("utf-8")
    )
    return hashlib.sha256(preimage).hexdigest()


def _normalized_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ast_id": _nonempty(value.get("ast_id"), name="ast_id"),
        "ast_revision": _revision(value.get("ast_revision")),
        "edge_id": _nonempty(value.get("edge_id"), name="edge_id"),
        "functional_node_id": _nonempty(
            value.get("functional_node_id"), name="functional_node_id"
        ),
        "structural_node_id": _nonempty(
            value.get("structural_node_id"), name="structural_node_id"
        ),
        "action_id": _nonempty(value.get("action_id"), name="action_id"),
        "measurement_id": _nonempty(
            value.get("measurement_id"), name="measurement_id"
        ),
    }


def _require_same_identity(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    source: str,
) -> None:
    for field in _IDENTITY_FIELDS:
        if observed[field] != expected[field]:
            _fail(
                f"{source}_{field}_mismatch",
                f"expected {expected[field]!r}, found {observed[field]!r}",
            )


def _normalize_action(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = _strict_mapping(value, name="compiled_action", allowed=_ACTION_FIELDS)
    _require_version(
        raw["schema_version"],
        COMPILED_MAPPING_ACTION_VERSION,
        name="compiled_action",
    )
    return {
        **_normalized_identity(raw),
        "operator": _nonempty(raw["operator"], name="operator"),
        "chain_id": _nonempty(raw["chain_id"], name="chain_id"),
        "legal_positions": _positions(
            raw["legal_positions"], name="legal_positions", allow_empty=False
        ),
    }


def _normalize_measurement(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = _strict_mapping(
        value, name="compiled_measurement", allowed=_MEASUREMENT_FIELDS
    )
    _require_version(
        raw["schema_version"],
        COMPILED_MEASUREMENT_VERSION,
        name="compiled_measurement",
    )
    direction = str(raw["direction"] or "").strip().lower()
    if direction not in {"maximize", "minimize"}:
        _fail("measurement_direction_invalid")
    missing_policy = str(raw["missing_policy"] or "").strip().lower()
    if missing_policy not in {"fail", "abstain"}:
        _fail("measurement_missing_policy_invalid")
    threshold = raw["threshold"]
    if threshold is not None:
        threshold = _finite(threshold, name="measurement_threshold")
    state = str(raw["state"] or "").strip().lower()
    if state not in {"positive", "negative", "preserve"}:
        _fail("measurement_state_invalid")
    return {
        **_normalized_identity(raw),
        "evaluator_id": _nonempty(raw["evaluator_id"], name="evaluator_id"),
        "term_name": _nonempty(raw["term_name"], name="term_name"),
        "state": state,
        "direction": direction,
        "threshold": threshold,
        "missing_policy": missing_policy,
    }


def _normalize_move(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = _strict_mapping(value, name="realized_move", allowed=_MOVE_FIELDS)
    _require_version(
        raw["schema_version"],
        REALIZED_MAPPING_MOVE_VERSION,
        name="realized_move",
    )
    return {
        **_normalized_identity(raw),
        "operator": _nonempty(raw["operator"], name="operator"),
        "chain_id": _nonempty(raw["chain_id"], name="chain_id"),
        "positions": _positions(
            raw["positions"], name="realized_positions", allow_empty=False
        ),
        "evaluated_sequence_id": _nonempty(
            raw["evaluated_sequence_id"], name="evaluated_sequence_id"
        ),
    }


def _exact_term(
    report: Mapping[str, Any],
    *,
    evaluator_id: str,
    term_name: str,
    state: str,
) -> Mapping[str, Any] | None:
    if not isinstance(report, Mapping):
        _fail("evaluator_report_not_mapping")
    terms = report.get("terms")
    if not isinstance(terms, list):
        _fail("evaluator_report_terms_not_list")
    matches = [
        item
        for item in terms
        if isinstance(item, Mapping)
        and str(item.get("provider") or "") == evaluator_id
        and str(item.get("name") or "") == term_name
        and str(item.get("state") or "") == state
    ]
    if len(matches) > 1:
        _fail("measurement_term_duplicate")
    return matches[0] if matches else None


def _observation(
    identity: Mapping[str, Any],
    measurement: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    evaluated_sequence_id: str,
) -> dict[str, Any]:
    term = _exact_term(
        report,
        evaluator_id=measurement["evaluator_id"],
        term_name=measurement["term_name"],
        state=measurement["state"],
    )
    report_sequence_id = _nonempty(
        report.get("evaluated_sequence_id"), name="report_evaluated_sequence_id"
    )
    if report_sequence_id != evaluated_sequence_id:
        _fail(
            "evaluated_sequence_id_mismatch",
            f"expected {evaluated_sequence_id!r}, found {report_sequence_id!r}",
        )
    available = bool(term is not None and term.get("available") is True)
    score = term.get("score") if term is not None else None
    measured = available and score is not None
    if measured:
        score = _finite(score, name="measurement_term_value")
    if not measured:
        if measurement["missing_policy"] == "fail":
            _fail(
                "measurement_missing_fail",
                f"missing {measurement['evaluator_id']}/{measurement['term_name']}",
            )
        return {
            "status": "abstain",
            "evaluator_id": measurement["evaluator_id"],
            "term_name": measurement["term_name"],
            "state": measurement["state"],
            "term_provider": None,
            "term_value": None,
            "direction": measurement["direction"],
            "directional_value": None,
            "threshold": measurement["threshold"],
            "missing_policy": measurement["missing_policy"],
            "gate": None,
            "evaluated_sequence_id": evaluated_sequence_id,
            **identity,
        }

    directional = score if measurement["direction"] == "maximize" else -score
    gate = None
    if measurement["threshold"] is not None:
        maximize = measurement["direction"] == "maximize"
        gate = {
            "operator": ">=" if maximize else "<=",
            "threshold": measurement["threshold"],
            "passed": (
                score >= measurement["threshold"]
                if maximize
                else score <= measurement["threshold"]
            ),
        }
    return {
        "status": "measured",
        "evaluator_id": measurement["evaluator_id"],
        "term_name": measurement["term_name"],
        "state": measurement["state"],
        "term_provider": str(term.get("provider")),
        "term_value": score,
        "direction": measurement["direction"],
        "directional_value": directional,
        "threshold": measurement["threshold"],
        "missing_policy": measurement["missing_policy"],
        "gate": gate,
        "evaluated_sequence_id": evaluated_sequence_id,
        **identity,
    }


def _validate_action_attribution(
    value: Any, identity: Mapping[str, Any]
) -> dict[str, Any]:
    raw = _strict_mapping(
        value, name="trace_action", allowed=_ACTION_ATTRIBUTION_FIELDS
    )
    normalized_identity = _normalized_identity(raw)
    _require_same_identity(identity, normalized_identity, source="trace_action")
    legal = _positions(raw["legal_positions"], name="legal_positions", allow_empty=False)
    realized = _positions(
        raw["realized_positions"], name="realized_positions", allow_empty=False
    )
    if not set(realized).issubset(set(legal)):
        _fail("realized_positions_outside_legal_positions")
    return {
        **normalized_identity,
        "operator": _nonempty(raw["operator"], name="operator"),
        "chain_id": _nonempty(raw["chain_id"], name="chain_id"),
        "legal_positions": legal,
        "realized_positions": realized,
    }


def _validate_observation_attribution(
    value: Any, identity: Mapping[str, Any]
) -> dict[str, Any]:
    raw = _strict_mapping(
        value,
        name="trace_observation",
        allowed=_OBSERVATION_ATTRIBUTION_FIELDS,
    )
    normalized_identity = _normalized_identity(raw)
    _require_same_identity(
        identity, normalized_identity, source="trace_observation"
    )
    status = str(raw["status"] or "")
    if status not in {"measured", "abstain"}:
        _fail("trace_observation_status_invalid")
    evaluator_id = _nonempty(raw["evaluator_id"], name="evaluator_id")
    term_name = _nonempty(raw["term_name"], name="term_name")
    state = str(raw["state"] or "")
    if state not in {"positive", "negative", "preserve"}:
        _fail("trace_observation_state_invalid")
    evaluated_sequence_id = _nonempty(
        raw["evaluated_sequence_id"], name="evaluated_sequence_id"
    )
    direction = str(raw["direction"] or "")
    if direction not in {"maximize", "minimize"}:
        _fail("trace_observation_direction_invalid")
    missing_policy = str(raw["missing_policy"] or "")
    if missing_policy not in {"fail", "abstain"}:
        _fail("trace_observation_missing_policy_invalid")
    threshold = raw["threshold"]
    if threshold is not None:
        threshold = _finite(threshold, name="trace_observation_threshold")

    if status == "abstain":
        if missing_policy != "abstain" or any(
            raw[field] is not None
            for field in ("term_provider", "term_value", "directional_value", "gate")
        ):
            _fail("trace_observation_abstain_invalid")
        return {
            "status": status,
            "evaluator_id": evaluator_id,
            "term_name": term_name,
            "state": state,
            "term_provider": None,
            "term_value": None,
            "direction": direction,
            "directional_value": None,
            "threshold": threshold,
            "missing_policy": missing_policy,
            "gate": None,
            "evaluated_sequence_id": evaluated_sequence_id,
            **normalized_identity,
        }

    provider = _nonempty(raw["term_provider"], name="term_provider")
    if provider != evaluator_id:
        _fail("trace_observation_provider_mismatch")
    score = _finite(raw["term_value"], name="trace_observation_term_value")
    expected_directional = score if direction == "maximize" else -score
    directional = _finite(
        raw["directional_value"], name="trace_observation_directional_value"
    )
    if directional != expected_directional:
        _fail("trace_observation_directional_value_mismatch")
    expected_gate = None
    if threshold is not None:
        maximize = direction == "maximize"
        expected_gate = {
            "operator": ">=" if maximize else "<=",
            "threshold": threshold,
            "passed": score >= threshold if maximize else score <= threshold,
        }
    if raw["gate"] != expected_gate:
        _fail("trace_observation_gate_mismatch")
    return {
        "status": status,
        "evaluator_id": evaluator_id,
        "term_name": term_name,
        "state": state,
        "term_provider": provider,
        "term_value": score,
        "direction": direction,
        "directional_value": directional,
        "threshold": threshold,
        "missing_policy": missing_policy,
        "gate": _json_copy(expected_gate),
        "evaluated_sequence_id": evaluated_sequence_id,
        **normalized_identity,
    }


@dataclass(frozen=True)
class MappingExecutionTrace:


    ast_id: str
    ast_revision: int
    edge_id: str
    functional_node_id: str
    structural_node_id: str
    action_id: str
    measurement_id: str
    evaluated_sequence_id: str
    _action_json: str
    _observation_json: str
    trace_hash: str
    schema_version: str = MAPPING_EXECUTION_TRACE_VERSION

    def __post_init__(self) -> None:
        _require_version(
            self.schema_version,
            MAPPING_EXECUTION_TRACE_VERSION,
            name="mapping_execution_trace",
        )
        identity = _normalized_identity(self.to_dict(include_hash=False))
        if self.observation.get("evaluated_sequence_id") != self.evaluated_sequence_id:
            _fail("trace_evaluated_sequence_id_mismatch")
        action = _validate_action_attribution(json.loads(self._action_json), identity)
        observation = _validate_observation_attribution(
            json.loads(self._observation_json), identity
        )
        if self._action_json != _canonical_json(action):
            _fail("trace_action_not_canonical")
        if self._observation_json != _canonical_json(observation):
            _fail("trace_observation_not_canonical")
        expected = _hash_payload(self.to_dict(include_hash=False))
        if self.trace_hash != expected:
            _fail("mapping_execution_trace_hash_mismatch")

    @property
    def action(self) -> dict[str, Any]:
        return json.loads(self._action_json)

    @property
    def observation(self) -> dict[str, Any]:
        return json.loads(self._observation_json)

    @classmethod
    def _create(
        cls,
        *,
        identity: Mapping[str, Any],
        action: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> "MappingExecutionTrace":
        core = {
            "schema_version": MAPPING_EXECUTION_TRACE_VERSION,
            **dict(identity),
            "evaluated_sequence_id": _nonempty(
                observation.get("evaluated_sequence_id"),
                name="evaluated_sequence_id",
            ),
            "action": _json_copy(action),
            "observation": _json_copy(observation),
        }
        return cls(
            **dict(identity),
            evaluated_sequence_id=_nonempty(
                observation.get("evaluated_sequence_id"),
                name="evaluated_sequence_id",
            ),
            _action_json=_canonical_json(core["action"]),
            _observation_json=_canonical_json(core["observation"]),
            trace_hash=_hash_payload(core),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MappingExecutionTrace":
        raw = _strict_mapping(value, name="mapping_trace", allowed=_TRACE_FIELDS)
        _require_version(
            raw["schema_version"],
            MAPPING_EXECUTION_TRACE_VERSION,
            name="mapping_execution_trace",
        )
        identity = _normalized_identity(raw)
        action = _validate_action_attribution(raw["action"], identity)
        observation = _validate_observation_attribution(raw["observation"], identity)
        if raw["evaluated_sequence_id"] != observation["evaluated_sequence_id"]:
            _fail("trace_evaluated_sequence_id_mismatch")
        rebuilt = cls._create(
            identity=identity,
            action=action,
            observation=observation,
        )
        if str(raw["trace_hash"] or "") != rebuilt.trace_hash:
            _fail("mapping_execution_trace_hash_mismatch")
        return rebuilt

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "ast_id": self.ast_id,
            "ast_revision": self.ast_revision,
            "edge_id": self.edge_id,
            "functional_node_id": self.functional_node_id,
            "structural_node_id": self.structural_node_id,
            "action_id": self.action_id,
            "measurement_id": self.measurement_id,
            "evaluated_sequence_id": self.evaluated_sequence_id,
            "action": self.action,
            "observation": self.observation,
        }
        if include_hash:
            value["trace_hash"] = self.trace_hash
        return value


def project_mapping_execution(
    compiled_action: Mapping[str, Any],
    compiled_measurement: Mapping[str, Any],
    realized_move: Mapping[str, Any],
    evaluator_report: Mapping[str, Any],
) -> MappingExecutionTrace:


    action = _normalize_action(compiled_action)
    measurement = _normalize_measurement(compiled_measurement)
    move = _normalize_move(realized_move)
    identity = {field: action[field] for field in _IDENTITY_FIELDS}
    _require_same_identity(identity, measurement, source="measurement")
    _require_same_identity(identity, move, source="realized_move")
    if move["operator"] != action["operator"]:
        _fail("realized_move_operator_mismatch")
    if move["chain_id"] != action["chain_id"]:
        _fail("realized_move_chain_id_mismatch")
    if not set(move["positions"]).issubset(set(action["legal_positions"])):
        _fail("realized_positions_outside_legal_positions")
    action_attribution = {
        **identity,
        "operator": action["operator"],
        "chain_id": action["chain_id"],
        "legal_positions": action["legal_positions"],
        "realized_positions": move["positions"],
    }
    observation = _observation(
        identity,
        measurement,
        evaluator_report,
        evaluated_sequence_id=move["evaluated_sequence_id"],
    )
    return MappingExecutionTrace._create(
        identity=identity,
        action=action_attribution,
        observation=observation,
    )


def validate_mapping_execution_trace(
    value: Mapping[str, Any] | MappingExecutionTrace,
) -> MappingExecutionTrace:


    if isinstance(value, MappingExecutionTrace):
        return MappingExecutionTrace.from_mapping(value.to_dict())
    return MappingExecutionTrace.from_mapping(value)


__all__ = [
    "COMPILED_MAPPING_ACTION_VERSION",
    "COMPILED_MEASUREMENT_VERSION",
    "MAPPING_EXECUTION_TRACE_VERSION",
    "REALIZED_MAPPING_MOVE_VERSION",
    "MappingExecutionError",
    "MappingExecutionTrace",
    "project_mapping_execution",
    "validate_mapping_execution_trace",
]
