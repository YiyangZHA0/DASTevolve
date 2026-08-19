

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Optional

from engine.causal_flow import (
    CausalFlowContractError,
    EffectiveSearchContract,
    SequenceRecord,
    canonical_json,
)


ABLATION_CONTROL_VERSION = "astevolve.ablation_control.v1"

_SEMANTIC_CONTROL_FIELDS = (
    "masks",
    "fixed_residues",
    "node_policies",
    "operator_policy",
    "search_budget",
    "constraints",
    "evaluator_routing",
    "generator_policy",
    "state_context",
)
_MECHANISM_ROUTING_FIELDS = frozenset(
    {
        "executable_mapping_plan",
        "effective_mapping_schedule",
        "mapping_measurement_specs",
    }
)
_SERIALIZED_FIELDS = frozenset(
    {
        "schema_version",
        "semantic_controls",
        "parent_sequence_id",
        "memory_snapshot_hash",
        "seed",
        "control_hash",
    }
)
_SEQUENCE_ID_RE = re.compile(r"sequence_sha256:[0-9a-f]{64}\Z")


class AblationControlError(CausalFlowContractError):
    pass


def _fail(code: str, detail: str = "") -> None:
    raise AblationControlError(code, detail)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("mapping_required", label)
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    unknown_code: str = "unknown_fields",
    missing_code: str = "fields_missing",
) -> None:
    observed = set(value)
    unknown = sorted(str(item) for item in observed - expected)
    missing = sorted(str(item) for item in expected - observed)
    if unknown:
        _fail(unknown_code, ",".join(unknown))
    if missing:
        _fail(missing_code, ",".join(missing))


def _canonical_copy(value: Any) -> Any:
    try:
        return json.loads(canonical_json(value))
    except CausalFlowContractError as exc:
        _fail("not_canonical_json", str(exc))


def _seed(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("seed_invalid", repr(value))
    return value


def _required_text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(code)
    return value.strip()


def _parent_sequence_id(value: Any) -> str:
    sequence_id = _required_text(value, "parent_sequence_id_required")
    if _SEQUENCE_ID_RE.fullmatch(sequence_id) is None:
        _fail("parent_sequence_id_invalid", sequence_id)
    return sequence_id


def _validate_semantic_controls(value: Any) -> dict[str, Any]:
    controls = _mapping(value, "semantic_controls")
    _exact_fields(
        controls,
        set(_SEMANTIC_CONTROL_FIELDS),
        unknown_code="semantic_unknown_fields",
        missing_code="semantic_fields_missing",
    )
    clean = _canonical_copy(controls)
    routing = _mapping(clean["evaluator_routing"], "evaluator_routing")
    forbidden = sorted(_MECHANISM_ROUTING_FIELDS & set(routing))
    if forbidden:
        _fail("mechanism_field_forbidden", ",".join(forbidden))
    return clean


def _project_effective_contract(
    value: EffectiveSearchContract | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(value, EffectiveSearchContract):
        serialized: Any = value.to_dict()
    elif isinstance(value, Mapping):
        serialized = value
    else:
        _fail("effective_contract_invalid", type(value).__name__)

    try:
        effective = EffectiveSearchContract.from_mapping(serialized)
    except (CausalFlowContractError, TypeError, ValueError) as exc:
        _fail("effective_contract_invalid", str(exc))

    semantic = _canonical_copy(effective.semantic)
    routing = _mapping(semantic["evaluator_routing"], "evaluator_routing")
    for field in _MECHANISM_ROUTING_FIELDS:
        routing.pop(field, None)
    return _validate_semantic_controls(semantic)


def _control_digest(
    semantic_controls: Mapping[str, Any],
    parent_sequence_id: str,
    memory_snapshot_hash: str,
    seed: Optional[int],
) -> str:
    payload = {
        "semantic_controls": semantic_controls,
        "parent_sequence_id": parent_sequence_id,
        "memory_snapshot_hash": memory_snapshot_hash,
        "seed": seed,
    }
    encoded = (
        f"astevolve.ablation_control.v1\0{canonical_json(payload)}".encode("utf-8")
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AblationControlContract:


    _semantic_controls_json: str
    parent_sequence_id: str
    memory_snapshot_hash: str
    seed: Optional[int]
    control_hash: str
    schema_version: str = ABLATION_CONTROL_VERSION

    @property
    def semantic_controls(self) -> dict[str, Any]:


        return json.loads(self._semantic_controls_json)

    @classmethod
    def _assemble(
        cls,
        *,
        semantic_controls: Mapping[str, Any],
        parent_sequence_id: Any,
        memory_snapshot_hash: Any,
        seed: Any,
    ) -> "AblationControlContract":
        controls = _validate_semantic_controls(semantic_controls)
        sequence_id = _parent_sequence_id(parent_sequence_id)
        memory_hash = _required_text(
            memory_snapshot_hash, "memory_snapshot_hash_required"
        )
        resolved_seed = _seed(seed)
        return cls(
            _semantic_controls_json=canonical_json(controls),
            parent_sequence_id=sequence_id,
            memory_snapshot_hash=memory_hash,
            seed=resolved_seed,
            control_hash=_control_digest(
                controls, sequence_id, memory_hash, resolved_seed
            ),
        )

    @classmethod
    def create(
        cls,
        effective_contract: EffectiveSearchContract | Mapping[str, Any],
        *,
        parent_sequences: Mapping[str, str],
        memory_snapshot_hash: str,
        seed: Optional[int],
    ) -> "AblationControlContract":


        controls = _project_effective_contract(effective_contract)
        try:
            sequence_id = SequenceRecord.create(parent_sequences).semantic_id
        except (CausalFlowContractError, TypeError, ValueError) as exc:
            _fail("parent_sequences_invalid", str(exc))
        return cls._assemble(
            semantic_controls=controls,
            parent_sequence_id=sequence_id,
            memory_snapshot_hash=memory_snapshot_hash,
            seed=seed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "semantic_controls": self.semantic_controls,
            "parent_sequence_id": self.parent_sequence_id,
            "memory_snapshot_hash": self.memory_snapshot_hash,
            "seed": self.seed,
            "control_hash": self.control_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AblationControlContract":
        artifact = _mapping(value, "ablation_control")
        _exact_fields(artifact, _SERIALIZED_FIELDS)
        if artifact.get("schema_version") != ABLATION_CONTROL_VERSION:
            _fail("schema_version_invalid", str(artifact.get("schema_version")))
        contract = cls._assemble(
            semantic_controls=artifact.get("semantic_controls"),
            parent_sequence_id=artifact.get("parent_sequence_id"),
            memory_snapshot_hash=artifact.get("memory_snapshot_hash"),
            seed=artifact.get("seed"),
        )
        if artifact.get("control_hash") != contract.control_hash:
            _fail("hash_mismatch", "ablation_control")
        return contract


def build_ablation_control_contract(
    effective_contract: EffectiveSearchContract | Mapping[str, Any],
    *,
    parent_sequences: Mapping[str, str],
    memory_snapshot_hash: str,
    seed: Optional[int],
) -> AblationControlContract:


    return AblationControlContract.create(
        effective_contract,
        parent_sequences=parent_sequences,
        memory_snapshot_hash=memory_snapshot_hash,
        seed=seed,
    )


__all__ = [
    "ABLATION_CONTROL_VERSION",
    "AblationControlContract",
    "AblationControlError",
    "build_ablation_control_contract",
]
