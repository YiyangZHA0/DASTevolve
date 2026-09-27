

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from astevolve.search.run_memory import InnerRunMemory
from engine.memory_lifecycle import MemorySnapshot, ScopedAdaptivePriorSnapshot
from engine.memory_policy import MemoryPolicyConfig, MemoryScope
from outerloop.memory_facts import (
    OuterObservationFact,
    fact_ledger_hash,
)


MEMORY_REPLAY_VERSION = "astevolve.memory_replay_bundle.v1"


class MemoryReplayError(ValueError):
    pass


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
        raise MemoryReplayError(f"memory replay payload is not JSON-safe: {exc}") from exc


def _payload_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _snapshot_payload(value: Optional[ScopedAdaptivePriorSnapshot]) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    snapshot = value.snapshot
    return {
        "schema_version": value.schema_version,
        "scope": value.scope.to_artifact(),
        "snapshot": {
            **snapshot.to_artifact(),
            "canonical_content": snapshot.canonical_content,
        },
    }


def _snapshot_from_payload(value: Any, expected_scope: MemoryScope) -> Optional[ScopedAdaptivePriorSnapshot]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise MemoryReplayError("scoped adaptive snapshot must be a mapping")
    scope = MemoryScope.from_mapping(value.get("scope") or {})
    expected_scope.require_compatible(scope, level="lineage")
    raw = value.get("snapshot")
    if not isinstance(raw, Mapping):
        raise MemoryReplayError("scoped adaptive snapshot is missing snapshot payload")
    try:
        snapshot = MemorySnapshot(
            canonical_content=str(raw.get("canonical_content") or ""),
            content_hash=str(raw.get("content_hash") or ""),
            raw_hash=str(raw.get("raw_hash") or ""),
            source_path=str(raw.get("source_path") or ""),
            raw_size=int(raw.get("raw_size") or 0),
            source_exists=bool(raw.get("source_exists", False)),
            schema_version=str(raw.get("schema_version") or ""),
        )
        return ScopedAdaptivePriorSnapshot(
            scope=scope,
            snapshot=snapshot,
            schema_version=str(value.get("schema_version") or ""),
        )
    except (TypeError, ValueError) as exc:
        raise MemoryReplayError(f"invalid scoped adaptive snapshot: {exc}") from exc


@dataclass(frozen=True)
class MemoryReplayBundle:
    scope: MemoryScope
    policy: MemoryPolicyConfig
    outer_facts: Tuple[OuterObservationFact, ...]
    outer_projection: Dict[str, Any]
    adaptive_input: Optional[ScopedAdaptivePriorSnapshot]
    adaptive_output: Optional[ScopedAdaptivePriorSnapshot]
    inner_runs: Tuple[InnerRunMemory, ...]
    generation: Dict[str, Any]
    commit: Dict[str, Any]
    payload_hash: str
    schema_version: str = MEMORY_REPLAY_VERSION


def build_memory_replay_bundle(
    *,
    scope: MemoryScope,
    policy: MemoryPolicyConfig,
    outer_facts: Sequence[OuterObservationFact],
    outer_projection: Mapping[str, Any],
    adaptive_input: Optional[ScopedAdaptivePriorSnapshot],
    adaptive_output: Optional[ScopedAdaptivePriorSnapshot],
    inner_runs: Sequence[InnerRunMemory],
    generation: Mapping[str, Any],
    commit: Mapping[str, Any],
) -> Dict[str, Any]:
    if not isinstance(scope, MemoryScope):
        raise TypeError("scope must be MemoryScope")
    if not isinstance(policy, MemoryPolicyConfig):
        raise TypeError("policy must be MemoryPolicyConfig")
    facts = tuple(outer_facts)
    for fact in facts:
        scope.require_compatible(fact.scope, level="run")
    runs = tuple(inner_runs)
    for run in runs:
        if not isinstance(run, InnerRunMemory):
            raise TypeError("inner_runs accepts only InnerRunMemory values")
        scope.require_compatible(run.scope, level="lineage")
    for snapshot in (adaptive_input, adaptive_output):
        if snapshot is not None:
            scope.require_compatible(snapshot.scope, level="lineage")
    projection = json.loads(_canonical_json(dict(outer_projection)))
    if projection.get("source_fact_ledger_hash") != fact_ledger_hash(facts):
        raise MemoryReplayError("outer projection does not match source fact ledger hash")
    payload = {
        "scope": scope.to_artifact(),
        "policy": policy.to_artifact(),
        "outer_facts": [fact.to_dict() for fact in facts],
        "outer_projection": projection,
        "adaptive_input": _snapshot_payload(adaptive_input),
        "adaptive_output": _snapshot_payload(adaptive_output),
        "inner_runs": [run.to_replay_state() for run in runs],
        "generation": json.loads(_canonical_json(dict(generation))),
        "commit": json.loads(_canonical_json(dict(commit))),
        "boundaries": {
            "persistent_evaluator_cache": False,
            "cross_round_sequence_identity": False,
            "scope": "memory_state_only",
        },
    }
    return {
        "schema_version": MEMORY_REPLAY_VERSION,
        "payload": payload,
        "payload_hash": _payload_hash(payload),
    }


def load_memory_replay_bundle(
    value: Mapping[str, Any],
    *,
    expected_scope: MemoryScope,
) -> MemoryReplayBundle:
    if not isinstance(value, Mapping):
        raise MemoryReplayError("memory replay bundle must be a mapping")
    if value.get("schema_version") != MEMORY_REPLAY_VERSION:
        raise MemoryReplayError(
            f"unsupported memory replay version: {value.get('schema_version')!r}"
        )
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise MemoryReplayError("memory replay bundle is missing payload")
    actual_hash = _payload_hash(payload)
    if str(value.get("payload_hash") or "") != actual_hash:
        raise MemoryReplayError("memory replay payload hash mismatch")
    try:
        scope = MemoryScope.from_mapping(payload.get("scope") or {})
        expected_scope.require_compatible(scope, level="lineage")
        policy_payload = payload.get("policy") or {}
        policy = MemoryPolicyConfig.from_mapping(
            {
                key: policy_payload[key]
                for key in ("schema_version", "adaptive_prior_mode", "inner_state_scope")
                if key in policy_payload
            }
        )
        facts = tuple(
            OuterObservationFact.from_mapping(item)
            for item in (payload.get("outer_facts") or [])
        )
        for fact in facts:
            scope.require_compatible(fact.scope, level="run")
        projection = json.loads(_canonical_json(payload.get("outer_projection") or {}))
        if projection.get("source_fact_ledger_hash") != fact_ledger_hash(facts):
            raise MemoryReplayError("outer projection fact ledger hash mismatch")
        adaptive_input = _snapshot_from_payload(payload.get("adaptive_input"), scope)
        adaptive_output = _snapshot_from_payload(payload.get("adaptive_output"), scope)
        inner_runs = tuple(
            InnerRunMemory.from_replay_state(item, expected_scope=scope)
            for item in (payload.get("inner_runs") or [])
        )
        return MemoryReplayBundle(
            scope=scope,
            policy=policy,
            outer_facts=facts,
            outer_projection=projection,
            adaptive_input=adaptive_input,
            adaptive_output=adaptive_output,
            inner_runs=inner_runs,
            generation=json.loads(_canonical_json(payload.get("generation") or {})),
            commit=json.loads(_canonical_json(payload.get("commit") or {})),
            payload_hash=actual_hash,
        )
    except MemoryReplayError:
        raise
    except (TypeError, ValueError) as exc:
        raise MemoryReplayError(f"invalid memory replay bundle: {exc}") from exc


__all__ = [
    "MEMORY_REPLAY_VERSION",
    "MemoryReplayBundle",
    "MemoryReplayError",
    "build_memory_replay_bundle",
    "load_memory_replay_bundle",
]
