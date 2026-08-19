

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from threading import Condition, RLock
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from engine.memory_policy import MemoryPolicyError, MemoryScope


INNER_RUN_MEMORY_VERSION = "astevolve.inner_run_memory.v1"
INNER_RUN_REPLAY_VERSION = "astevolve.inner_run_memory_replay.v1"
SequenceKey = Tuple[Tuple[str, str], ...]


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
        raise ValueError(f"inner run memory contains a non-JSON value: {exc}") from exc


def _payload_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def canonical_sequence_key(seqs: Mapping[str, str]) -> SequenceKey:


    if not isinstance(seqs, Mapping) or not seqs:
        raise ValueError("sequence bundle must be a non-empty mapping")
    items = []
    for chain_id, sequence in seqs.items():
        cid = str(chain_id or "").strip()
        seq = str(sequence or "").strip().replace(" ", "").replace("\n", "")
        if not cid or not seq:
            raise ValueError("sequence bundle contains an empty chain or sequence")
        items.append((cid, seq))
    items.sort(key=lambda item: item[0])
    if len({item[0] for item in items}) != len(items):
        raise ValueError("sequence bundle contains duplicate chain identifiers")
    return tuple(items)


def _key_to_json(key: SequenceKey) -> list[list[str]]:
    return [[chain_id, sequence] for chain_id, sequence in key]


def _key_from_json(value: Any) -> SequenceKey:
    if not isinstance(value, list):
        raise ValueError("serialized sequence key must be a list")
    mapping: Dict[str, str] = {}
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("serialized sequence key entry must contain chain and sequence")
        mapping[str(item[0])] = str(item[1])
    return canonical_sequence_key(mapping)


@dataclass(frozen=True)
class SequenceClaim:
    is_new: bool
    transposition_node_id: Optional[str]
    visits: int


@dataclass(frozen=True)
class CacheLookup:
    value: Any
    cache_hit: bool


class InnerRunMemory:


    def __init__(self, *, scope: MemoryScope, run_instance_id: str) -> None:
        if not isinstance(scope, MemoryScope):
            raise TypeError("scope must be a MemoryScope")
        instance = str(run_instance_id or "").strip()
        if not instance:
            raise ValueError("run_instance_id must be non-empty")
        self.scope = scope
        self.run_instance_id = instance
        self._seen: Dict[SequenceKey, Dict[str, Any]] = {}
        self._transpositions: Dict[SequenceKey, Dict[str, Any]] = {}
        self._fast_cache: Dict[Tuple[str, SequenceKey], Any] = {}
        self._structure_cache: Dict[Tuple[str, SequenceKey], Any] = {}
        self._cache_hits = {"fast": 0, "structure": 0}
        self._cache_misses = {"fast": 0, "structure": 0}


        self._cache_condition = Condition(RLock())
        self._cache_inflight: set[Tuple[str, str, SequenceKey]] = set()
        self._action_stats: Dict[str, Dict[str, Any]] = {}

    def __getstate__(self) -> Dict[str, Any]:


        with self._cache_condition:
            if self._cache_inflight:
                raise RuntimeError(
                    "cannot serialize InnerRunMemory while cache evaluations are active"
                )
            state = dict(self.__dict__)
            state.pop("_cache_condition", None)
            state.pop("_cache_inflight", None)
            return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(dict(state))
        self._cache_condition = Condition(RLock())
        self._cache_inflight = set()

    def claim_sequence(
        self,
        seqs: Mapping[str, str],
        *,
        node_id: Optional[str] = None,
    ) -> SequenceClaim:
        key = canonical_sequence_key(seqs)
        record = self._seen.get(key)
        if record is None:
            first_node = str(node_id) if node_id not in (None, "") else None
            record = {"first_node_id": first_node, "visits": 1}
            self._seen[key] = record
            if first_node:
                self._transpositions[key] = {
                    "node_id": first_node,
                    "visits": 1,
                }
            return SequenceClaim(True, first_node, 1)
        record["visits"] = int(record.get("visits", 0)) + 1
        transposition = self._transpositions.get(key, {})
        if transposition:
            transposition["visits"] = int(transposition.get("visits", 0)) + 1
        return SequenceClaim(
            False,
            str(transposition.get("node_id")) if transposition.get("node_id") else record.get("first_node_id"),
            int(record["visits"]),
        )

    def record_transposition(
        self,
        seqs: Mapping[str, str],
        *,
        node_id: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        key = canonical_sequence_key(seqs)
        node = str(node_id or "").strip()
        if not node:
            raise ValueError("transposition node_id must be non-empty")
        existing = self._transpositions.get(key)
        if existing is not None and existing.get("node_id") != node:

            return
        self._transpositions[key] = {
            "node_id": node,
            "visits": int((existing or {}).get("visits", 1)),
            "payload": deepcopy(dict(payload or {})),
        }
        self._seen.setdefault(key, {"first_node_id": node, "visits": 1})

    def lookup_transposition(self, seqs: Mapping[str, str]) -> Optional[Dict[str, Any]]:
        value = self._transpositions.get(canonical_sequence_key(seqs))
        return deepcopy(value) if value is not None else None

    def _get_or_compute(
        self,
        cache_name: str,
        cache: Dict[Tuple[str, SequenceKey], Any],
        seqs: Mapping[str, str],
        evaluator_key: str,
        compute: Callable[[], Any],
    ) -> CacheLookup:
        descriptor = str(evaluator_key or "").strip()
        if not descriptor:
            raise ValueError("evaluator_key must be non-empty")
        key = (descriptor, canonical_sequence_key(seqs))
        inflight_key = (cache_name, key[0], key[1])
        with self._cache_condition:
            while inflight_key in self._cache_inflight:
                self._cache_condition.wait()
            if key in cache:
                self._cache_hits[cache_name] += 1
                return CacheLookup(deepcopy(cache[key]), True)
            self._cache_inflight.add(inflight_key)

        try:
            value = compute()


            _canonical_json(value)
            stored = deepcopy(value)
        except BaseException:
            with self._cache_condition:
                self._cache_inflight.discard(inflight_key)
                self._cache_condition.notify_all()
            raise

        with self._cache_condition:
            cache[key] = stored
            self._cache_misses[cache_name] += 1
            self._cache_inflight.discard(inflight_key)
            self._cache_condition.notify_all()
            return CacheLookup(deepcopy(stored), False)

    def get_or_compute_fast(
        self,
        seqs: Mapping[str, str],
        evaluator_key: str,
        compute: Callable[[], Any],
    ) -> CacheLookup:
        return self._get_or_compute("fast", self._fast_cache, seqs, evaluator_key, compute)

    def get_or_compute_structure(
        self,
        seqs: Mapping[str, str],
        evaluator_key: str,
        compute: Callable[[], Any],
    ) -> CacheLookup:
        return self._get_or_compute(
            "structure", self._structure_cache, seqs, evaluator_key, compute
        )

    def record_action(self, action: str, *, accepted: bool, reward: Any = None) -> None:
        name = str(action or "").strip()
        if not name:
            raise ValueError("action must be non-empty")
        block = self._action_stats.setdefault(
            name,
            {"attempted": 0, "accepted": 0, "reward_sum": 0.0, "reward_count": 0},
        )
        block["attempted"] += 1
        if accepted:
            block["accepted"] += 1
        if isinstance(reward, (int, float)) and not isinstance(reward, bool):
            numeric = float(reward)
            if math.isfinite(numeric):
                block["reward_sum"] += numeric
                block["reward_count"] += 1

    def cache_summary(self) -> Dict[str, Any]:
        return {
            "schema_version": INNER_RUN_MEMORY_VERSION,
            "scope": self.scope.to_artifact(),
            "run_instance_id": self.run_instance_id,
            "unique_sequences": len(self._seen),
            "transposition_entries": len(self._transpositions),
            "fast": {
                "entries": len(self._fast_cache),
                "hits": self._cache_hits["fast"],
                "misses": self._cache_misses["fast"],
            },
            "structure": {
                "entries": len(self._structure_cache),
                "hits": self._cache_hits["structure"],
                "misses": self._cache_misses["structure"],
            },
            "action_stats": deepcopy(self._action_stats),
            "persistence": "single_inner_run_only",
        }

    @staticmethod
    def _serialize_cache(cache: Dict[Tuple[str, SequenceKey], Any]) -> list[dict[str, Any]]:
        return [
            {
                "evaluator_key": descriptor,
                "sequence_key": _key_to_json(sequence_key),
                "value": deepcopy(value),
            }
            for (descriptor, sequence_key), value in sorted(
                cache.items(), key=lambda item: (item[0][0], item[0][1])
            )
        ]

    def to_replay_state(self) -> Dict[str, Any]:
        payload = {
            "scope": self.scope.to_artifact(),
            "run_instance_id": self.run_instance_id,
            "seen": [
                {
                    "sequence_key": _key_to_json(key),
                    "first_node_id": value.get("first_node_id"),
                    "visits": int(value.get("visits", 0)),
                }
                for key, value in sorted(self._seen.items())
            ],
            "transpositions": [
                {
                    "sequence_key": _key_to_json(key),
                    "node_id": value.get("node_id"),
                    "visits": int(value.get("visits", 0)),
                    "payload": deepcopy(value.get("payload", {})),
                }
                for key, value in sorted(self._transpositions.items())
            ],
            "fast_cache": self._serialize_cache(self._fast_cache),
            "structure_cache": self._serialize_cache(self._structure_cache),
            "cache_hits": deepcopy(self._cache_hits),
            "cache_misses": deepcopy(self._cache_misses),
            "action_stats": deepcopy(self._action_stats),
        }
        return {
            "schema_version": INNER_RUN_REPLAY_VERSION,
            "payload": payload,
            "payload_hash": _payload_hash(payload),
        }

    @classmethod
    def from_replay_state(
        cls,
        value: Mapping[str, Any],
        *,
        expected_scope: MemoryScope,
    ) -> "InnerRunMemory":
        if not isinstance(value, Mapping):
            raise ValueError("inner run replay state must be a mapping")
        if value.get("schema_version") != INNER_RUN_REPLAY_VERSION:
            raise ValueError(
                f"unsupported inner run replay version: {value.get('schema_version')!r}"
            )
        payload = value.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("inner run replay state is missing payload")
        if str(value.get("payload_hash") or "") != _payload_hash(payload):
            raise ValueError("inner run replay payload hash mismatch")
        scope = MemoryScope.from_mapping(payload.get("scope") or {})
        expected_scope.require_compatible(scope, level="lineage")
        memory = cls(
            scope=scope,
            run_instance_id=str(payload.get("run_instance_id") or ""),
        )
        for item in payload.get("seen", []) or []:
            key = _key_from_json(item.get("sequence_key"))
            memory._seen[key] = {
                "first_node_id": item.get("first_node_id"),
                "visits": int(item.get("visits", 0)),
            }
        for item in payload.get("transpositions", []) or []:
            key = _key_from_json(item.get("sequence_key"))
            memory._transpositions[key] = {
                "node_id": item.get("node_id"),
                "visits": int(item.get("visits", 0)),
                "payload": deepcopy(item.get("payload") or {}),
            }

        def restore_cache(items: Any) -> Dict[Tuple[str, SequenceKey], Any]:
            restored: Dict[Tuple[str, SequenceKey], Any] = {}
            for item in items or []:
                descriptor = str(item.get("evaluator_key") or "")
                key = _key_from_json(item.get("sequence_key"))
                restored[(descriptor, key)] = deepcopy(item.get("value"))
            return restored

        memory._fast_cache = restore_cache(payload.get("fast_cache"))
        memory._structure_cache = restore_cache(payload.get("structure_cache"))
        memory._cache_hits = {
            key: int((payload.get("cache_hits") or {}).get(key, 0))
            for key in ("fast", "structure")
        }
        memory._cache_misses = {
            key: int((payload.get("cache_misses") or {}).get(key, 0))
            for key in ("fast", "structure")
        }
        memory._action_stats = deepcopy(dict(payload.get("action_stats") or {}))

        if memory.to_replay_state() != dict(value):
            raise ValueError("inner run replay state is not canonical")
        return memory


__all__ = [
    "CacheLookup",
    "INNER_RUN_MEMORY_VERSION",
    "INNER_RUN_REPLAY_VERSION",
    "InnerRunMemory",
    "SequenceClaim",
    "canonical_sequence_key",
]
