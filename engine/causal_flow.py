

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Optional, Sequence


GRAPH_PATCH_VERSION = "astevolve.graph_patch.v1"
PATCH_FIELD_VERSION = "astevolve.patch_field_disposition.v1"
EFFECTIVE_SEARCH_CONTRACT_VERSION = "astevolve.effective_search_contract.v1"
CONTRACT_DIFF_VERSION = "astevolve.effective_contract_diff.v1"
SEQUENCE_RECORD_VERSION = "astevolve.causal_sequence.v1"
ACTION_RECORD_VERSION = "astevolve.causal_action.v1"
OBSERVATION_RECORD_VERSION = "astevolve.causal_observation.v1"
CAUSAL_TRACE_VERSION = "astevolve.llm_inner_causal_trace.v1"

FIELD_DISPOSITIONS = frozenset({"rejected", "no_op", "compiled", "executed"})


class CausalFlowContractError(ValueError):


    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)


def _fail(code: str, detail: str = "") -> None:
    raise CausalFlowContractError(code, detail)


def _reject_nonfinite(value: Any, path: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail("non_finite_value", path)
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_json(item) for item in value]
        return sorted(normalized, key=canonical_json)
    return value


def _normalize_contract_value(value: Any) -> Any:


    if isinstance(value, Mapping):
        return {
            deepcopy(key): _normalize_contract_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_contract_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_contract_value(item) for item in value]
        return sorted(normalized, key=canonical_json)
    return deepcopy(value)


def canonical_json(value: Any) -> str:
    value = _normalize_json(value)
    _reject_nonfinite(value)
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


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _digest(domain: str, value: Any) -> str:
    payload = f"{domain}\0{canonical_json(value)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("mapping_required", label)
    return value


def _closed(value: Mapping[str, Any], allowed: set[str], *, label: str = "") -> None:
    unknown = sorted(str(key) for key in set(value) - allowed)
    if unknown:
        prefix = f"{label}_" if label else ""
        _fail(f"{prefix}unknown_fields", ",".join(unknown))


def _required_text(value: Any, code: str) -> str:
    text = str(value or "").strip()
    if not text:
        _fail(code)
    return text


@dataclass(frozen=True)
class PatchFieldDisposition:
    path: str
    requested: Any
    disposition: str
    effective: Any
    reason: str
    consumer: Optional[str]
    action_ids: tuple[str, ...]
    schema_version: str = PATCH_FIELD_VERSION

    @classmethod
    def create(
        cls,
        *,
        path: str,
        requested: Any,
        disposition: str,
        effective: Any = None,
        reason: str = "",
        consumer: Optional[str] = None,
        action_ids: Iterable[str] = (),
    ) -> "PatchFieldDisposition":
        normalized = str(disposition or "").strip()
        if normalized not in FIELD_DISPOSITIONS:
            _fail("disposition_invalid", normalized)
        resolved_path = _required_text(path, "path_required")
        resolved_consumer = str(consumer).strip() if consumer not in (None, "") else None
        resolved_actions = tuple(str(item).strip() for item in action_ids if str(item).strip())
        if normalized in {"compiled", "executed"} and not resolved_consumer:
            _fail("consumer_required", resolved_path)
        if normalized == "executed" and not resolved_actions:
            _fail("action_ids_required", resolved_path)
        resolved_reason = _required_text(reason, "reason_required")
        if normalized == "rejected" and effective is not None:
            _fail("rejected_effective_forbidden", resolved_path)
        return cls(
            path=resolved_path,
            requested=_copy_json(requested),
            disposition=normalized,
            effective=_copy_json(effective),
            reason=resolved_reason,
            consumer=resolved_consumer,
            action_ids=resolved_actions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "requested": deepcopy(self.requested),
            "disposition": self.disposition,
            "effective": deepcopy(self.effective),
            "reason": self.reason,
            "consumer": self.consumer,
            "action_ids": list(self.action_ids),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PatchFieldDisposition":
        value = _require_mapping(value, "patch_field")
        _closed(
            value,
            {"schema_version", "path", "requested", "disposition", "effective", "reason", "consumer", "action_ids"},
        )
        if value.get("schema_version") != PATCH_FIELD_VERSION:
            _fail("schema_version_invalid", str(value.get("schema_version")))
        return cls.create(
            path=value.get("path", ""),
            requested=value.get("requested"),
            disposition=value.get("disposition", ""),
            effective=value.get("effective"),
            reason=value.get("reason", ""),
            consumer=value.get("consumer"),
            action_ids=value.get("action_ids") or (),
        )


@dataclass(frozen=True)
class GraphPatch:
    proposal_id: str
    parent_program_id: str
    hypothesis: str
    fields: tuple[PatchFieldDisposition, ...]
    patch_hash: str
    schema_version: str = GRAPH_PATCH_VERSION

    @classmethod
    def create(
        cls,
        *,
        proposal_id: str,
        parent_program_id: str = "",
        hypothesis: str = "",
        fields: Iterable[PatchFieldDisposition | Mapping[str, Any]] = (),
    ) -> "GraphPatch":
        records = tuple(
            item if isinstance(item, PatchFieldDisposition) else PatchFieldDisposition.from_mapping(item)
            for item in fields
        )
        paths = [item.path for item in records]
        if len(paths) != len(set(paths)):
            _fail("field_path_duplicate")
        semantic = {
            "proposal_id": _required_text(proposal_id, "proposal_id_required"),
            "parent_program_id": str(parent_program_id or ""),
            "hypothesis": str(hypothesis or ""),
            "fields": [item.to_dict() for item in records],
        }
        return cls(
            proposal_id=semantic["proposal_id"],
            parent_program_id=semantic["parent_program_id"],
            hypothesis=semantic["hypothesis"],
            fields=records,
            patch_hash=_digest("graph_patch.v1", semantic),
        )

    def _semantic(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "parent_program_id": self.parent_program_id,
            "hypothesis": self.hypothesis,
            "fields": [item.to_dict() for item in self.fields],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, **self._semantic(), "patch_hash": self.patch_hash}

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, require_hash: bool = True
    ) -> "GraphPatch":
        value = _require_mapping(value, "graph_patch")
        _closed(value, {"schema_version", "proposal_id", "parent_program_id", "hypothesis", "fields", "patch_hash"})
        if value.get("schema_version") != GRAPH_PATCH_VERSION:
            _fail("schema_version_invalid", str(value.get("schema_version")))
        patch = cls.create(
            proposal_id=value.get("proposal_id", ""),
            parent_program_id=value.get("parent_program_id", ""),
            hypothesis=value.get("hypothesis", ""),
            fields=value.get("fields") or (),
        )
        observed = str(value.get("patch_hash") or "")
        if require_hash and observed != patch.patch_hash:
            _fail("hash_mismatch", "graph_patch")
        return patch


_CONTRACT_FIELDS = (
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
_PROVENANCE_FIELDS = {
    "proposal_id",
    "parent_program_id",
    "run_id",
    "generation_id",
    "trial_id",
    "compilation_id",
    "artifact_path",
    "created_at",
    "source_code_hash",
}


@dataclass(frozen=True)
class EffectiveSearchContract:
    semantic: dict[str, Any]
    provenance: dict[str, Any]
    contract_hash: str
    schema_version: str = EFFECTIVE_SEARCH_CONTRACT_VERSION

    @classmethod
    def create(cls, *, provenance: Optional[Mapping[str, Any]] = None, **semantic: Any) -> "EffectiveSearchContract":
        unknown = sorted(set(semantic) - set(_CONTRACT_FIELDS))
        missing = sorted(set(_CONTRACT_FIELDS) - set(semantic))
        if unknown:
            _fail("unknown_fields", ",".join(unknown))
        if missing:
            _fail("contract_fields_missing", ",".join(missing))
        clean_semantic = {}
        for field in _CONTRACT_FIELDS:


            canonical_json(semantic[field])
            clean_semantic[field] = _normalize_contract_value(semantic[field])
        clean_provenance = _copy_json(dict(provenance or {}))
        provenance_unknown = sorted(set(clean_provenance) - _PROVENANCE_FIELDS)
        if provenance_unknown:
            _fail("provenance_unknown_fields", ",".join(provenance_unknown))
        return cls(
            semantic=clean_semantic,
            provenance=clean_provenance,
            contract_hash=_digest("effective_search_contract.v1", clean_semantic),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "semantic": deepcopy(self.semantic),
            "provenance": deepcopy(self.provenance),
            "contract_hash": self.contract_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EffectiveSearchContract":
        value = _require_mapping(value, "effective_contract")
        _closed(value, {"schema_version", "semantic", "provenance", "contract_hash"})
        if value.get("schema_version") != EFFECTIVE_SEARCH_CONTRACT_VERSION:
            _fail("schema_version_invalid", str(value.get("schema_version")))
        semantic = _require_mapping(value.get("semantic"), "semantic")
        contract = cls.create(provenance=value.get("provenance") or {}, **dict(semantic))
        if str(value.get("contract_hash") or "") != contract.contract_hash:
            _fail("hash_mismatch", "effective_search_contract")
        return contract


@dataclass(frozen=True)
class ContractChange:
    path: str
    before: Any
    after: Any

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "before": deepcopy(self.before), "after": deepcopy(self.after)}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ContractChange":
        _closed(value, {"path", "before", "after"})
        return cls(_required_text(value.get("path"), "path_required"), _copy_json(value.get("before")), _copy_json(value.get("after")))


@dataclass(frozen=True)
class EffectiveContractDiff:
    parent_contract_hash: str
    effective_contract_hash: str
    changes: tuple[ContractChange, ...]
    has_effect: bool
    schema_version: str = CONTRACT_DIFF_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "parent_contract_hash": self.parent_contract_hash,
            "effective_contract_hash": self.effective_contract_hash,
            "has_effect": self.has_effect,
            "changes": [item.to_dict() for item in self.changes],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EffectiveContractDiff":
        _closed(value, {"schema_version", "parent_contract_hash", "effective_contract_hash", "has_effect", "changes"})
        if value.get("schema_version") != CONTRACT_DIFF_VERSION:
            _fail("schema_version_invalid")
        changes = tuple(ContractChange.from_mapping(item) for item in value.get("changes") or ())
        has_effect = bool(value.get("has_effect"))
        if has_effect != bool(changes):
            _fail("contract_diff_inconsistent")
        return cls(str(value.get("parent_contract_hash") or ""), str(value.get("effective_contract_hash") or ""), changes, has_effect)


def _flatten_diff(before: Any, after: Any, path: str, output: list[ContractChange]) -> None:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        for key in sorted(set(before) | set(after), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in before:
                output.append(ContractChange(child, None, _copy_json(after[key])))
            elif key not in after:
                output.append(ContractChange(child, _copy_json(before[key]), None))
            else:
                _flatten_diff(before[key], after[key], child, output)
        return
    if canonical_json(before) != canonical_json(after):
        output.append(ContractChange(path, _copy_json(before), _copy_json(after)))


def diff_effective_contract(parent: EffectiveSearchContract, child: EffectiveSearchContract) -> EffectiveContractDiff:
    changes: list[ContractChange] = []
    _flatten_diff(parent.semantic, child.semantic, "", changes)
    return EffectiveContractDiff(parent.contract_hash, child.contract_hash, tuple(changes), bool(changes))


def require_effective_contract_delta(parent: EffectiveSearchContract, child: EffectiveSearchContract) -> EffectiveContractDiff:
    delta = diff_effective_contract(parent, child)
    if not delta.has_effect:
        _fail("accepted_patch_no_effect", child.contract_hash)
    return delta


@dataclass(frozen=True)
class SequenceRecord:
    _chain_items: tuple[tuple[str, str], ...]
    semantic_id: str
    schema_version: str = SEQUENCE_RECORD_VERSION

    @property
    def sequence_id(self) -> str:
        return self.semantic_id

    @property
    def chains(self) -> dict[str, str]:
        return dict(self._chain_items)

    @classmethod
    def create(cls, chains: Mapping[str, str]) -> "SequenceRecord":
        if not isinstance(chains, Mapping) or not chains:
            _fail("chains_required")
        items = tuple(sorted((str(key), str(value)) for key, value in chains.items()))
        if any(not key or not value for key, value in items):
            _fail("chain_or_sequence_empty")
        payload = [{"chain_id": key, "sequence": value} for key, value in items]
        return cls(items, f"sequence_sha256:{_digest('sequence_record.v1', payload)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "chains": [{"chain_id": key, "sequence": value} for key, value in self._chain_items],
            "semantic_id": self.semantic_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SequenceRecord":
        _closed(value, {"schema_version", "chains", "semantic_id"})
        if value.get("schema_version") != SEQUENCE_RECORD_VERSION:
            _fail("schema_version_invalid")
        chains: dict[str, str] = {}
        for item in value.get("chains") or ():
            _closed(item, {"chain_id", "sequence"})
            chain_id = str(item.get("chain_id") or "")
            if chain_id in chains:
                _fail("chain_duplicate", chain_id)
            chains[chain_id] = str(item.get("sequence") or "")
        record = cls.create(chains)
        if value.get("semantic_id") != record.semantic_id:
            _fail("hash_mismatch", "sequence")
        return record


@dataclass(frozen=True)
class ActionRecord:
    contract_hash: str
    node_id: str
    operator: str
    positions: tuple[int, ...]
    parent_sequence_id: str
    child_sequence_id: str
    parameters: dict[str, Any]
    field_paths: tuple[str, ...]
    semantic_id: str
    schema_version: str = ACTION_RECORD_VERSION

    @classmethod
    def create(cls, *, contract_hash: str, node_id: str, operator: str, positions: Iterable[int], parent_sequence_id: str, child_sequence_id: str, parameters: Optional[Mapping[str, Any]] = None, field_paths: Iterable[str] = ()) -> "ActionRecord":
        data = {
            "contract_hash": _required_text(contract_hash, "contract_hash_required"),
            "node_id": _required_text(node_id, "node_id_required"),
            "operator": _required_text(operator, "operator_required"),
            "positions": sorted({int(item) for item in positions}),
            "parent_sequence_id": _required_text(parent_sequence_id, "parent_sequence_id_required"),
            "child_sequence_id": _required_text(child_sequence_id, "child_sequence_id_required"),
            "parameters": _copy_json(dict(parameters or {})),
            "field_paths": sorted({str(item) for item in field_paths if str(item)}),
        }
        semantic_id = f"action_sha256:{_digest('action_record.v1', data)}"
        return cls(data["contract_hash"], data["node_id"], data["operator"], tuple(data["positions"]), data["parent_sequence_id"], data["child_sequence_id"], data["parameters"], tuple(data["field_paths"]), semantic_id)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "contract_hash": self.contract_hash, "node_id": self.node_id, "operator": self.operator, "positions": list(self.positions), "parent_sequence_id": self.parent_sequence_id, "child_sequence_id": self.child_sequence_id, "parameters": deepcopy(self.parameters), "field_paths": list(self.field_paths), "semantic_id": self.semantic_id}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ActionRecord":
        _closed(value, {"schema_version", "contract_hash", "node_id", "operator", "positions", "parent_sequence_id", "child_sequence_id", "parameters", "field_paths", "semantic_id"})
        if value.get("schema_version") != ACTION_RECORD_VERSION:
            _fail("schema_version_invalid")
        record = cls.create(contract_hash=value.get("contract_hash", ""), node_id=value.get("node_id", ""), operator=value.get("operator", ""), positions=value.get("positions") or (), parent_sequence_id=value.get("parent_sequence_id", ""), child_sequence_id=value.get("child_sequence_id", ""), parameters=value.get("parameters") or {}, field_paths=value.get("field_paths") or ())
        if value.get("semantic_id") != record.semantic_id:
            _fail("hash_mismatch", "action")
        return record


@dataclass(frozen=True)
class ObservationRecord:
    sequence_id: str
    evaluator: dict[str, Any]
    state: dict[str, Any]
    seed: Optional[int]
    metrics: dict[str, Any]
    gate: dict[str, Any]
    semantic_id: str
    schema_version: str = OBSERVATION_RECORD_VERSION

    @classmethod
    def create(cls, *, sequence_id: str, evaluator: Mapping[str, Any], state: Mapping[str, Any], seed: Optional[int], metrics: Mapping[str, Any], gate: Mapping[str, Any]) -> "ObservationRecord":
        payload = {"sequence_id": _required_text(sequence_id, "sequence_id_required"), "evaluator": _copy_json(dict(evaluator)), "state": _copy_json(dict(state)), "seed": None if seed is None else int(seed), "metrics": _copy_json(dict(metrics)), "gate": _copy_json(dict(gate))}
        semantic_id = f"observation_sha256:{_digest('observation_record.v1', payload)}"
        return cls(payload["sequence_id"], payload["evaluator"], payload["state"], payload["seed"], payload["metrics"], payload["gate"], semantic_id)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "sequence_id": self.sequence_id, "evaluator": deepcopy(self.evaluator), "state": deepcopy(self.state), "seed": self.seed, "metrics": deepcopy(self.metrics), "gate": deepcopy(self.gate), "semantic_id": self.semantic_id}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ObservationRecord":
        _closed(value, {"schema_version", "sequence_id", "evaluator", "state", "seed", "metrics", "gate", "semantic_id"})
        if value.get("schema_version") != OBSERVATION_RECORD_VERSION:
            _fail("schema_version_invalid")
        record = cls.create(sequence_id=value.get("sequence_id", ""), evaluator=value.get("evaluator") or {}, state=value.get("state") or {}, seed=value.get("seed"), metrics=value.get("metrics") or {}, gate=value.get("gate") or {})
        if value.get("semantic_id") != record.semantic_id:
            _fail("hash_mismatch", "observation")
        return record


@dataclass(frozen=True)
class CausalTrace:
    proposal_id: str
    graph_patch: GraphPatch
    parent_contract: EffectiveSearchContract
    effective_contract: EffectiveSearchContract
    sequences: tuple[SequenceRecord, ...]
    actions: tuple[ActionRecord, ...]
    observations: tuple[ObservationRecord, ...]
    parent_sequence_id: str
    final_sequence_id: str
    integrity_hash: str
    schema_version: str = CAUSAL_TRACE_VERSION

    @classmethod
    def create(cls, *, proposal_id: str, graph_patch: GraphPatch, parent_contract: EffectiveSearchContract, effective_contract: EffectiveSearchContract, sequences: Iterable[SequenceRecord], actions: Iterable[ActionRecord], observations: Iterable[ObservationRecord], parent_sequence_id: str, final_sequence_id: str) -> "CausalTrace":
        seqs, acts, obs = tuple(sequences), tuple(actions), tuple(observations)
        payload = {"proposal_id": _required_text(proposal_id, "proposal_id_required"), "graph_patch": graph_patch.to_dict(), "parent_contract": parent_contract.to_dict(), "effective_contract": effective_contract.to_dict(), "sequences": [item.to_dict() for item in seqs], "actions": [item.to_dict() for item in acts], "observations": [item.to_dict() for item in obs], "parent_sequence_id": str(parent_sequence_id), "final_sequence_id": str(final_sequence_id)}
        trace = cls(payload["proposal_id"], graph_patch, parent_contract, effective_contract, seqs, acts, obs, payload["parent_sequence_id"], payload["final_sequence_id"], _digest("causal_trace.v1", payload))
        _validate_trace_links(trace)
        return trace

    def _payload(self) -> dict[str, Any]:
        return {"proposal_id": self.proposal_id, "graph_patch": self.graph_patch.to_dict(), "parent_contract": self.parent_contract.to_dict(), "effective_contract": self.effective_contract.to_dict(), "sequences": [item.to_dict() for item in self.sequences], "actions": [item.to_dict() for item in self.actions], "observations": [item.to_dict() for item in self.observations], "parent_sequence_id": self.parent_sequence_id, "final_sequence_id": self.final_sequence_id}

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, **self._payload(), "integrity_hash": self.integrity_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CausalTrace":
        value = _require_mapping(value, "causal_trace")
        _closed(value, {"schema_version", "proposal_id", "graph_patch", "parent_contract", "effective_contract", "sequences", "actions", "observations", "parent_sequence_id", "final_sequence_id", "integrity_hash"})
        if value.get("schema_version") != CAUSAL_TRACE_VERSION:
            _fail("schema_version_invalid")
        trace = cls.create(proposal_id=value.get("proposal_id", ""), graph_patch=GraphPatch.from_mapping(value.get("graph_patch") or {}), parent_contract=EffectiveSearchContract.from_mapping(value.get("parent_contract") or {}), effective_contract=EffectiveSearchContract.from_mapping(value.get("effective_contract") or {}), sequences=[SequenceRecord.from_mapping(item) for item in value.get("sequences") or ()], actions=[ActionRecord.from_mapping(item) for item in value.get("actions") or ()], observations=[ObservationRecord.from_mapping(item) for item in value.get("observations") or ()], parent_sequence_id=value.get("parent_sequence_id", ""), final_sequence_id=value.get("final_sequence_id", ""))
        if value.get("integrity_hash") != trace.integrity_hash:
            _fail("hash_mismatch", "causal_trace")
        return trace


def _validate_trace_links(trace: CausalTrace) -> None:
    if trace.graph_patch.proposal_id != trace.proposal_id:
        _fail("proposal_id_mismatch")
    sequence_ids = {item.semantic_id for item in trace.sequences}
    if len(sequence_ids) != len(trace.sequences):
        _fail("sequence_id_duplicate")
    if trace.parent_sequence_id not in sequence_ids or trace.final_sequence_id not in sequence_ids:
        _fail("sequence_reference_missing")
    action_ids = {item.semantic_id for item in trace.actions}
    if len(action_ids) != len(trace.actions):
        _fail("action_id_duplicate")
    for action in trace.actions:
        if action.contract_hash != trace.effective_contract.contract_hash:
            _fail("action_contract_mismatch")
        if action.parent_sequence_id not in sequence_ids or action.child_sequence_id not in sequence_ids:
            _fail("action_sequence_reference_missing")
    for field in trace.graph_patch.fields:
        if any(action_id not in action_ids for action_id in field.action_ids):
            _fail("patch_action_reference_missing", field.path)
    for observation in trace.observations:
        if observation.sequence_id not in sequence_ids:
            _fail("observation_sequence_reference_missing")
    if trace.parent_sequence_id != trace.final_sequence_id:
        parents = {item.child_sequence_id: item.parent_sequence_id for item in trace.actions}
        cursor, visited = trace.final_sequence_id, set()
        while cursor != trace.parent_sequence_id:
            if cursor in visited or cursor not in parents:
                _fail("final_sequence_not_reachable")
            visited.add(cursor)
            cursor = parents[cursor]


def validate_causal_trace(value: CausalTrace | Mapping[str, Any]) -> CausalTrace:
    if isinstance(value, CausalTrace):
        return CausalTrace.from_mapping(value.to_dict())
    return CausalTrace.from_mapping(value)


__all__ = [
    "ACTION_RECORD_VERSION", "CAUSAL_TRACE_VERSION", "CONTRACT_DIFF_VERSION",
    "EFFECTIVE_SEARCH_CONTRACT_VERSION", "GRAPH_PATCH_VERSION",
    "OBSERVATION_RECORD_VERSION", "PATCH_FIELD_VERSION", "SEQUENCE_RECORD_VERSION",
    "ActionRecord", "CausalFlowContractError", "CausalTrace", "ContractChange",
    "EffectiveContractDiff", "EffectiveSearchContract", "GraphPatch",
    "ObservationRecord", "PatchFieldDisposition", "SequenceRecord",
    "canonical_json", "diff_effective_contract", "require_effective_contract_delta",
    "validate_causal_trace",
]
