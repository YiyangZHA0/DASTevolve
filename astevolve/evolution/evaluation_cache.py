

from __future__ import annotations

from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Dict, Iterator, Mapping, Optional, Protocol, Tuple

from astevolve.domain import DesignStrategy, EvidenceBundle, EvidenceRecord
from astevolve.domain.dual_ast import ExecutableDualAST

from .domain import (
    DUAL_AST_REVISION,
    EVALUATION_FAILED,
    STRATEGY_REVISION,
    Proposal,
    SealedEvaluation,
)
from .orchestrator import (
    EvaluationTask,
    EvolutionInputSnapshot,
    GenerationInput,
    ProposalEvaluator,
)


EVALUATION_CACHE_KEY_SCHEMA = "astevolve.evolution.evaluation_cache_key.v1"
EVALUATION_CACHE_RECORD_SCHEMA = "astevolve.evolution.evaluation_cache_record.v1"
CONTENT_DETERMINISM_SCHEMA = "astevolve.evolution.content_determinism.v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_INSENSITIVE_FIELDS = (
    "parent_id",
    "provenance",
    "proposal_id",
    "seed",
    "slot",
)


class EvaluationCacheError(RuntimeError):
    pass


class EvaluationCacheContractError(EvaluationCacheError):
    pass


class EvaluationCacheCorruption(EvaluationCacheError):
    pass


class FailureCachePolicy(str, Enum):


    RETRY = "retry"
    CACHE = "cache"


class EvaluationIdentityPolicy(str, Enum):


    EXACT_TASK = "exact_task"
    CONTENT_DETERMINISTIC = "content_deterministic"


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
        raise EvaluationCacheContractError(
            "evaluation cache values must be finite JSON data"
        ) from exc


def _digest(namespace: str, value: Any) -> str:
    payload = f"{namespace}\0{_canonical_json(value)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _required_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise EvaluationCacheContractError(f"{name} must be a lowercase SHA-256")
    return value


def _required_namespace(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EvaluationCacheContractError(
            "evaluator_namespace must be a non-empty normalized string"
        )
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationCacheContractError(f"{name} must be a non-negative integer")
    return value


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EvaluationCacheContractError(f"{name} must be normalized non-empty text")
    return value


def _exact_mapping(value: Any, *, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EvaluationCacheContractError(f"{name} has invalid fields")
    return value


@dataclass(frozen=True)
class ContentDeterminismDeclaration:


    evaluator_namespace: str
    rationale: str
    insensitive_to: Tuple[str, ...] = _CONTENT_INSENSITIVE_FIELDS
    schema_version: str = CONTENT_DETERMINISM_SCHEMA

    def __post_init__(self) -> None:
        self.verify()

    @classmethod
    def create(
        cls, *, evaluator_namespace: str, rationale: str
    ) -> "ContentDeterminismDeclaration":
        return cls(
            evaluator_namespace=_required_namespace(evaluator_namespace),
            rationale=_required_text(rationale, "content determinism rationale"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ContentDeterminismDeclaration":
        raw = _exact_mapping(
            value,
            fields={
                "schema_version",
                "evaluator_namespace",
                "rationale",
                "insensitive_to",
            },
            name="content determinism declaration",
        )
        if not isinstance(raw["insensitive_to"], list):
            raise EvaluationCacheContractError(
                "content determinism insensitive_to must be a list"
            )
        return cls(
            evaluator_namespace=raw["evaluator_namespace"],
            rationale=raw["rationale"],
            insensitive_to=tuple(raw["insensitive_to"]),
            schema_version=raw["schema_version"],
        )

    def verify(self) -> None:
        if self.schema_version != CONTENT_DETERMINISM_SCHEMA:
            raise EvaluationCacheContractError(
                f"unsupported content determinism schema: {self.schema_version!r}"
            )
        _required_namespace(self.evaluator_namespace)
        _required_text(self.rationale, "content determinism rationale")
        if self.insensitive_to != _CONTENT_INSENSITIVE_FIELDS:
            raise EvaluationCacheContractError(
                "content determinism must explicitly declare insensitivity to "
                + ", ".join(_CONTENT_INSENSITIVE_FIELDS)
            )

    def to_dict(self) -> Dict[str, Any]:
        self.verify()
        return {
            "schema_version": self.schema_version,
            "evaluator_namespace": self.evaluator_namespace,
            "rationale": self.rationale,
            "insensitive_to": list(self.insensitive_to),
        }


def _validated_task(task: EvaluationTask) -> EvaluationTask:
    if not isinstance(task, EvaluationTask):
        raise TypeError("task must be EvaluationTask")


    try:
        task.input_snapshot.verify()
        task.generation_input.verify()
        task.proposal.verify()
        return EvaluationTask(
            run_id=task.run_id,
            generation_index=task.generation_index,
            generation_id=task.generation_id,
            slot=task.slot,
            seed=task.seed,
            input_snapshot=EvolutionInputSnapshot.from_mapping(
                task.input_snapshot.to_dict()
            ),
            generation_input=GenerationInput.create(
                generation_index=task.generation_input.generation_index,
                initial_snapshot_hash=task.generation_input.initial_snapshot_hash,
                prior_commit_hashes=task.generation_input.prior_commit_hashes,
            ),
            proposal=Proposal.from_mapping(task.proposal.to_dict()),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise EvaluationCacheContractError("evaluation task is invalid") from exc


@dataclass(frozen=True)
class EvaluationCacheKey:


    descriptor_json: str
    digest: str
    schema_version: str = EVALUATION_CACHE_KEY_SCHEMA

    def __post_init__(self) -> None:
        self.verify()

    @classmethod
    def create(
        cls,
        task: EvaluationTask,
        *,
        evaluator_namespace: str,
        failure_policy: FailureCachePolicy = FailureCachePolicy.RETRY,
        identity_policy: EvaluationIdentityPolicy = (
            EvaluationIdentityPolicy.EXACT_TASK
        ),
        content_determinism: Optional[ContentDeterminismDeclaration] = None,
    ) -> "EvaluationCacheKey":
        trusted = _validated_task(task)
        return cls._from_validated_task(
            trusted,
            evaluator_namespace=evaluator_namespace,
            failure_policy=failure_policy,
            identity_policy=identity_policy,
            content_determinism=content_determinism,
        )

    @classmethod
    def _from_validated_task(
        cls,
        task: EvaluationTask,
        *,
        evaluator_namespace: str,
        failure_policy: FailureCachePolicy,
        identity_policy: EvaluationIdentityPolicy,
        content_determinism: Optional[ContentDeterminismDeclaration],
    ) -> "EvaluationCacheKey":
        namespace = _required_namespace(evaluator_namespace)
        if not isinstance(failure_policy, FailureCachePolicy):
            raise TypeError("failure_policy must be FailureCachePolicy")
        if not isinstance(identity_policy, EvaluationIdentityPolicy):
            raise TypeError("identity_policy must be EvaluationIdentityPolicy")
        if identity_policy is EvaluationIdentityPolicy.CONTENT_DETERMINISTIC:
            if not isinstance(content_determinism, ContentDeterminismDeclaration):
                raise EvaluationCacheContractError(
                    "content-deterministic identity requires an explicit declaration"
                )
            content_determinism.verify()
            if content_determinism.evaluator_namespace != namespace:
                raise EvaluationCacheContractError(
                    "content determinism declaration belongs to another evaluator"
                )
            slot: Optional[int] = None
            seed: Optional[int] = None
            proposal: Dict[str, Any] = {
                "kind": task.proposal.kind,
                "payload": task.proposal.payload(),
                "payload_hash": task.proposal.payload_hash,
            }
            declaration = content_determinism.to_dict()
        else:
            if content_determinism is not None:
                raise EvaluationCacheContractError(
                    "content determinism declaration requires content identity policy"
                )
            slot = task.slot
            seed = task.seed
            proposal = task.proposal.to_dict()
            declaration = None
        descriptor = {
            "schema_version": EVALUATION_CACHE_KEY_SCHEMA,
            "evaluator_namespace": namespace,
            "failure_policy": failure_policy.value,
            "identity_policy": identity_policy.value,
            "content_determinism": declaration,
            "run_id": task.run_id,
            "generation_index": task.generation_index,
            "generation_id": task.generation_id,
            "slot": slot,
            "seed": seed,
            "input_snapshot_hash": task.input_snapshot.snapshot_hash,
            "generation_input": task.generation_input.to_dict(),
            "proposal": proposal,
        }
        return cls(
            descriptor_json=_canonical_json(descriptor),
            digest=_digest(EVALUATION_CACHE_KEY_SCHEMA, descriptor),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationCacheKey":
        raw = _exact_mapping(
            value,
            fields={"schema_version", "descriptor", "digest"},
            name="evaluation cache key",
        )
        if not isinstance(raw["descriptor"], Mapping):
            raise EvaluationCacheContractError("cache-key descriptor must be a mapping")
        return cls(
            descriptor_json=_canonical_json(dict(raw["descriptor"])),
            digest=raw["digest"],
            schema_version=raw["schema_version"],
        )

    def descriptor(self) -> Dict[str, Any]:
        try:
            value = json.loads(self.descriptor_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise EvaluationCacheContractError(
                "cache-key descriptor is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise EvaluationCacheContractError(
                "cache-key descriptor must be a JSON object"
            )
        return value

    def proposal(self) -> Proposal:
        if self.identity_policy is not EvaluationIdentityPolicy.EXACT_TASK:
            raise EvaluationCacheContractError(
                "content cache keys do not contain a proposal identity"
            )
        descriptor = self.descriptor()
        proposal = descriptor.get("proposal")
        if not isinstance(proposal, Mapping):
            raise EvaluationCacheContractError(
                "cache-key descriptor has no sealed proposal"
            )
        try:
            return Proposal.from_mapping(proposal)
        except (TypeError, ValueError) as exc:
            raise EvaluationCacheContractError("cache-key proposal is invalid") from exc

    @property
    def identity_policy(self) -> EvaluationIdentityPolicy:
        descriptor = self.descriptor()
        try:
            return EvaluationIdentityPolicy(descriptor["identity_policy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationCacheContractError(
                "cache-key identity policy is invalid"
            ) from exc

    def matches_proposal(self, proposal: Proposal) -> bool:
        if not isinstance(proposal, Proposal):
            return False
        proposal.verify()
        descriptor = self.descriptor()
        if proposal.generation_id != descriptor["generation_id"]:
            return False
        if self.identity_policy is EvaluationIdentityPolicy.EXACT_TASK:
            return proposal == self.proposal()
        content = descriptor["proposal"]
        return (
            proposal.kind == content["kind"]
            and proposal.payload_hash == content["payload_hash"]
            and proposal.payload() == content["payload"]
        )

    def verify(self) -> None:
        if self.schema_version != EVALUATION_CACHE_KEY_SCHEMA:
            raise EvaluationCacheContractError(
                f"unsupported evaluation cache key schema: {self.schema_version!r}"
            )
        descriptor = self.descriptor()
        expected_fields = {
            "schema_version",
            "evaluator_namespace",
            "failure_policy",
            "identity_policy",
            "content_determinism",
            "run_id",
            "generation_index",
            "generation_id",
            "slot",
            "seed",
            "input_snapshot_hash",
            "generation_input",
            "proposal",
        }
        if set(descriptor) != expected_fields:
            raise EvaluationCacheContractError(
                "cache-key descriptor has invalid fields"
            )
        if descriptor["schema_version"] != EVALUATION_CACHE_KEY_SCHEMA:
            raise EvaluationCacheContractError("cache-key descriptor schema mismatch")
        _required_namespace(descriptor["evaluator_namespace"])
        try:
            FailureCachePolicy(descriptor["failure_policy"])
        except (TypeError, ValueError) as exc:
            raise EvaluationCacheContractError(
                "cache-key failure policy is invalid"
            ) from exc
        try:
            identity_policy = EvaluationIdentityPolicy(descriptor["identity_policy"])
        except (TypeError, ValueError) as exc:
            raise EvaluationCacheContractError(
                "cache-key identity policy is invalid"
            ) from exc
        _required_text(descriptor["run_id"], "cache-key run_id")
        generation_index = _non_negative_int(
            descriptor["generation_index"], "cache-key generation_index"
        )
        generation_id = _required_text(
            descriptor["generation_id"], "cache-key generation_id"
        )
        snapshot_hash = _required_digest(
            descriptor["input_snapshot_hash"], "cache-key input snapshot hash"
        )
        raw_generation_input = _exact_mapping(
            descriptor["generation_input"],
            fields={
                "schema_version",
                "generation_index",
                "initial_snapshot_hash",
                "prior_commit_hashes",
                "input_hash",
            },
            name="cache-key generation input",
        )
        prior_hashes = raw_generation_input["prior_commit_hashes"]
        if not isinstance(prior_hashes, list):
            raise EvaluationCacheContractError(
                "cache-key prior commit hashes must be a list"
            )
        try:
            generation_input = GenerationInput.create(
                generation_index=raw_generation_input["generation_index"],
                initial_snapshot_hash=raw_generation_input["initial_snapshot_hash"],
                prior_commit_hashes=prior_hashes,
            )
        except (TypeError, ValueError) as exc:
            raise EvaluationCacheContractError(
                "cache-key generation input is invalid"
            ) from exc
        if generation_input.to_dict() != dict(raw_generation_input):
            raise EvaluationCacheContractError(
                "cache-key generation input seal mismatch"
            )
        if generation_input.generation_index != generation_index:
            raise EvaluationCacheContractError(
                "cache-key generation index does not match generation input"
            )
        if generation_input.initial_snapshot_hash != snapshot_hash:
            raise EvaluationCacheContractError(
                "cache-key initial snapshot does not match generation input"
            )
        if identity_policy is EvaluationIdentityPolicy.EXACT_TASK:
            if descriptor["content_determinism"] is not None:
                raise EvaluationCacheContractError(
                    "exact-task cache key cannot contain content declaration"
                )
            slot = _non_negative_int(descriptor["slot"], "cache-key slot")
            _non_negative_int(descriptor["seed"], "cache-key seed")

            proposal = self.proposal()
            if proposal.generation_id != generation_id or proposal.slot != slot:
                raise EvaluationCacheContractError(
                    "cache-key proposal does not match generation identity and slot"
                )
        else:
            if descriptor["slot"] is not None or descriptor["seed"] is not None:
                raise EvaluationCacheContractError(
                    "content cache key must omit slot and seed"
                )
            declaration_value = descriptor["content_determinism"]
            if not isinstance(declaration_value, Mapping):
                raise EvaluationCacheContractError(
                    "content cache key requires a determinism declaration"
                )
            declaration = ContentDeterminismDeclaration.from_mapping(declaration_value)
            if declaration.evaluator_namespace != descriptor["evaluator_namespace"]:
                raise EvaluationCacheContractError(
                    "content declaration evaluator namespace mismatch"
                )
            proposal_content = _exact_mapping(
                descriptor["proposal"],
                fields={"kind", "payload", "payload_hash"},
                name="content cache proposal",
            )
            kind = _required_text(proposal_content["kind"], "proposal kind")
            payload_hash = _required_digest(
                proposal_content["payload_hash"], "proposal payload hash"
            )
            if not isinstance(proposal_content["payload"], Mapping):
                raise EvaluationCacheContractError(
                    "content cache proposal payload must be a mapping"
                )
            payload = dict(proposal_content["payload"])
            try:
                if kind == DUAL_AST_REVISION:
                    normalized = ExecutableDualAST.from_mapping(payload).to_dict()
                elif kind == STRATEGY_REVISION:
                    normalized = DesignStrategy.from_mapping(payload).to_legacy_dict()
                else:
                    raise EvaluationCacheContractError(
                        f"unsupported content proposal kind: {kind!r}"
                    )
            except EvaluationCacheContractError:
                raise
            except (TypeError, ValueError) as exc:
                raise EvaluationCacheContractError(
                    "content cache proposal payload is invalid"
                ) from exc
            if _canonical_json(normalized) != _canonical_json(payload):
                raise EvaluationCacheContractError(
                    "content cache proposal payload is not canonical domain data"
                )
            expected_payload_hash = _digest(
                "astevolve.evolution.proposal_payload.v1", payload
            )
            if payload_hash != expected_payload_hash:
                raise EvaluationCacheContractError(
                    "content cache proposal payload hash mismatch"
                )
        expected = _digest(EVALUATION_CACHE_KEY_SCHEMA, descriptor)
        if _required_digest(self.digest, "cache key digest") != expected:
            raise EvaluationCacheContractError("evaluation cache key digest mismatch")
        if _canonical_json(descriptor) != self.descriptor_json:
            raise EvaluationCacheContractError(
                "cache-key descriptor is not canonical JSON"
            )

    def to_dict(self) -> Dict[str, Any]:
        self.verify()
        return {
            "schema_version": self.schema_version,
            "descriptor": self.descriptor(),
            "digest": self.digest,
        }


@dataclass(frozen=True)
class EvaluationCacheRecord:


    key: EvaluationCacheKey
    source_proposal: Proposal
    evaluation: SealedEvaluation
    record_hash: str
    schema_version: str = EVALUATION_CACHE_RECORD_SCHEMA

    def __post_init__(self) -> None:
        self.verify()

    @classmethod
    def create(
        cls,
        *,
        key: EvaluationCacheKey,
        source_proposal: Proposal,
        evaluation: SealedEvaluation,
    ) -> "EvaluationCacheRecord":
        if not isinstance(key, EvaluationCacheKey):
            raise TypeError("key must be EvaluationCacheKey")
        if not isinstance(evaluation, SealedEvaluation):
            raise TypeError("evaluation must be SealedEvaluation")
        if not isinstance(source_proposal, Proposal):
            raise TypeError("source_proposal must be Proposal")
        key.verify()
        source_proposal.verify()
        if not key.matches_proposal(source_proposal):
            raise EvaluationCacheContractError(
                "source proposal does not match evaluation cache key"
            )
        evaluation.verify_for(source_proposal)
        core = {
            "schema_version": EVALUATION_CACHE_RECORD_SCHEMA,
            "key": key.to_dict(),
            "source_proposal": source_proposal.to_dict(),
            "evaluation": evaluation.to_dict(),
        }
        return cls(
            key=key,
            source_proposal=source_proposal,
            evaluation=evaluation,
            record_hash=_digest(EVALUATION_CACHE_RECORD_SCHEMA, core),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationCacheRecord":
        raw = _exact_mapping(
            value,
            fields={
                "schema_version",
                "key",
                "source_proposal",
                "evaluation",
                "record_hash",
            },
            name="evaluation cache record",
        )
        if (
            not isinstance(raw["key"], Mapping)
            or not isinstance(raw["source_proposal"], Mapping)
            or not isinstance(raw["evaluation"], Mapping)
        ):
            raise EvaluationCacheContractError(
                "cache record key, source proposal, and evaluation must be mappings"
            )
        try:
            return cls(
                key=EvaluationCacheKey.from_mapping(raw["key"]),
                source_proposal=Proposal.from_mapping(raw["source_proposal"]),
                evaluation=SealedEvaluation.from_mapping(raw["evaluation"]),
                record_hash=raw["record_hash"],
                schema_version=raw["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise EvaluationCacheContractError("cache record is invalid") from exc

    def verify(self) -> None:
        if self.schema_version != EVALUATION_CACHE_RECORD_SCHEMA:
            raise EvaluationCacheContractError(
                f"unsupported evaluation cache record schema: {self.schema_version!r}"
            )
        if not isinstance(self.key, EvaluationCacheKey):
            raise EvaluationCacheContractError("cache record has an invalid key")
        if not isinstance(self.evaluation, SealedEvaluation):
            raise EvaluationCacheContractError("cache record has an invalid evaluation")
        if not isinstance(self.source_proposal, Proposal):
            raise EvaluationCacheContractError(
                "cache record has an invalid source proposal"
            )
        self.key.verify()
        self.source_proposal.verify()
        if not self.key.matches_proposal(self.source_proposal):
            raise EvaluationCacheContractError(
                "cache record source proposal does not match key"
            )
        self.evaluation.verify_for(self.source_proposal)
        core = {
            "schema_version": self.schema_version,
            "key": self.key.to_dict(),
            "source_proposal": self.source_proposal.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }
        expected = _digest(EVALUATION_CACHE_RECORD_SCHEMA, core)
        if _required_digest(self.record_hash, "cache record hash") != expected:
            raise EvaluationCacheContractError("evaluation cache record hash mismatch")

    def to_dict(self) -> Dict[str, Any]:
        self.verify()
        return {
            "schema_version": self.schema_version,
            "key": self.key.to_dict(),
            "source_proposal": self.source_proposal.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "record_hash": self.record_hash,
        }


@dataclass(frozen=True)
class EvaluationCachePutResult:
    record: EvaluationCacheRecord
    inserted: bool


class EvaluationCacheStore(Protocol):


    def get(self, key: EvaluationCacheKey) -> Optional[EvaluationCacheRecord]: ...

    def put_if_absent(
        self, record: EvaluationCacheRecord
    ) -> EvaluationCachePutResult: ...

    def exclusive(self, key: EvaluationCacheKey) -> Any: ...


class InMemoryEvaluationCacheStore:


    def __init__(self) -> None:
        self._records: Dict[str, EvaluationCacheRecord] = {}
        self._guard = threading.RLock()
        self._key_locks: Dict[str, threading.RLock] = {}

    def __getstate__(self) -> Dict[str, Any]:
        with self._guard:
            return {"records": dict(self._records)}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self._records = dict(state.get("records", {}))
        self._guard = threading.RLock()
        self._key_locks = {}

    def get(self, key: EvaluationCacheKey) -> Optional[EvaluationCacheRecord]:
        key.verify()
        with self._guard:
            record = self._records.get(key.digest)
        if record is not None:
            record.verify()
            if record.key != key:
                raise EvaluationCacheCorruption("cache digest resolved to another key")
        return record

    def put_if_absent(self, record: EvaluationCacheRecord) -> EvaluationCachePutResult:
        record.verify()
        with self._guard:
            existing = self._records.get(record.key.digest)
            if existing is None:
                self._records[record.key.digest] = record
                return EvaluationCachePutResult(record=record, inserted=True)
        existing.verify()
        if existing.key != record.key:
            raise EvaluationCacheCorruption("cache digest collision")
        return EvaluationCachePutResult(record=existing, inserted=False)

    @contextmanager
    def exclusive(self, key: EvaluationCacheKey) -> Iterator[None]:
        key.verify()
        with self._guard:
            lock = self._key_locks.setdefault(key.digest, threading.RLock())
        with lock:
            yield

    def iter_records(self) -> Tuple[EvaluationCacheRecord, ...]:
        with self._guard:
            records = tuple(self._records[key] for key in sorted(self._records))
        for record in records:
            record.verify()
        return records


class FileEvaluationCacheStore:


    def __init__(self, root: os.PathLike[str] | str) -> None:
        self._root = Path(root)
        self._objects = self._root / "objects"
        self._locks = self._root / "locks"
        try:
            self._objects.mkdir(parents=True, exist_ok=True)
            self._locks.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise EvaluationCacheError(
                f"cannot initialize evaluation cache at {self._root}"
            ) from exc

    @property
    def root(self) -> Path:
        return self._root

    def _path(self, digest: str) -> Path:
        return self._objects / f"{_required_digest(digest, 'cache digest')}.json"

    def _lock_path(self, digest: str) -> Path:
        return self._locks / f"{_required_digest(digest, 'cache digest')}.lock"

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor: Optional[int] = None
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            os.fsync(descriptor)
        except OSError as exc:
            raise EvaluationCacheError(
                f"cannot durably publish evaluation cache record in {path}"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _load_path(self, path: Path) -> Optional[EvaluationCacheRecord]:
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise EvaluationCacheError(f"cannot read cache record {path}") from exc
        try:
            decoded = json.loads(payload.decode("utf-8"))
            if not isinstance(decoded, Mapping):
                raise EvaluationCacheContractError(
                    "cache record must contain a JSON object"
                )
            record = EvaluationCacheRecord.from_mapping(decoded)
            canonical = _canonical_json(record.to_dict()).encode("utf-8")
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            EvaluationCacheContractError,
            TypeError,
            ValueError,
        ) as exc:
            raise EvaluationCacheCorruption(
                f"visible evaluation cache record is invalid: {path}"
            ) from exc
        if payload != canonical:
            raise EvaluationCacheCorruption(
                f"visible evaluation cache record is not canonical: {path}"
            )
        if path.stem != record.key.digest:
            raise EvaluationCacheCorruption(
                f"cache record does not match its content-addressed path: {path}"
            )
        return record

    def get(self, key: EvaluationCacheKey) -> Optional[EvaluationCacheRecord]:
        key.verify()
        record = self._load_path(self._path(key.digest))
        if record is not None and record.key != key:
            raise EvaluationCacheCorruption("cache digest resolved to another key")
        return record

    def put_if_absent(self, record: EvaluationCacheRecord) -> EvaluationCachePutResult:
        record.verify()
        path = self._path(record.key.digest)
        payload = _canonical_json(record.to_dict()).encode("utf-8")
        descriptor: Optional[int] = None
        temporary: Optional[Path] = None
        inserted = False
        try:
            descriptor, raw_path = tempfile.mkstemp(
                prefix=f".{record.key.digest}.",
                suffix=".tmp",
                dir=self._objects,
            )
            temporary = Path(raw_path)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = None
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
                inserted = True
                self._fsync_directory(self._objects)
            except FileExistsError:
                inserted = False
        except EvaluationCacheError:
            raise
        except OSError as exc:
            raise EvaluationCacheError(
                f"cannot atomically publish evaluation cache record: {path}"
            ) from exc
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
        stored = self.get(record.key)
        if stored is None:
            raise EvaluationCacheError("cache record disappeared after publication")
        return EvaluationCachePutResult(record=stored, inserted=inserted)

    @contextmanager
    def exclusive(self, key: EvaluationCacheKey) -> Iterator[None]:
        key.verify()
        path = self._lock_path(key.digest)
        descriptor: Optional[int] = None
        try:
            descriptor = os.open(
                path,
                os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                descriptor = None
            raise EvaluationCacheError(
                f"cannot lock evaluation cache key {key.digest}"
            ) from exc
        try:
            yield
        finally:
            if descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)

    def iter_records(self) -> Tuple[EvaluationCacheRecord, ...]:
        records = []
        for path in sorted(self._objects.glob("*.json")):
            record = self._load_path(path)
            if record is not None:
                records.append(record)
        return tuple(records)


@dataclass(frozen=True)
class EvaluationCacheMetrics:
    hits: int
    misses: int
    singleflight_waits: int
    stores: int
    failure_bypasses: int
    errors: int


class CachedProposalEvaluator:


    def __init__(
        self,
        delegate: ProposalEvaluator,
        *,
        store: EvaluationCacheStore,
        evaluator_namespace: str,
        failure_policy: FailureCachePolicy = FailureCachePolicy.RETRY,
        identity_policy: EvaluationIdentityPolicy = (
            EvaluationIdentityPolicy.EXACT_TASK
        ),
        content_determinism: Optional[ContentDeterminismDeclaration] = None,
    ) -> None:
        evaluate = getattr(delegate, "evaluate", None)
        if not callable(evaluate):
            raise TypeError("delegate must provide evaluate(task)")
        if not isinstance(failure_policy, FailureCachePolicy):
            raise TypeError("failure_policy must be FailureCachePolicy")
        if not isinstance(identity_policy, EvaluationIdentityPolicy):
            raise TypeError("identity_policy must be EvaluationIdentityPolicy")
        namespace = _required_namespace(evaluator_namespace)
        if identity_policy is EvaluationIdentityPolicy.CONTENT_DETERMINISTIC:
            if not isinstance(content_determinism, ContentDeterminismDeclaration):
                raise EvaluationCacheContractError(
                    "content-deterministic identity requires an explicit declaration"
                )
            content_determinism.verify()
            if content_determinism.evaluator_namespace != namespace:
                raise EvaluationCacheContractError(
                    "content determinism declaration belongs to another evaluator"
                )
        elif content_determinism is not None:
            raise EvaluationCacheContractError(
                "content determinism declaration requires content identity policy"
            )
        for method in ("get", "put_if_absent", "exclusive"):
            if not callable(getattr(store, method, None)):
                raise TypeError(f"store must provide {method}()")
        self._delegate = delegate
        self._store = store
        self._namespace = namespace
        self._failure_policy = failure_policy
        self._identity_policy = identity_policy
        self._content_determinism = content_determinism
        self._guard = threading.RLock()
        self._inflight: Dict[str, Future[EvaluationCacheRecord]] = {}
        self._counters = {
            "hits": 0,
            "misses": 0,
            "singleflight_waits": 0,
            "stores": 0,
            "failure_bypasses": 0,
            "errors": 0,
        }

    def __getstate__(self) -> Dict[str, Any]:
        with self._guard:
            if self._inflight:
                raise RuntimeError(
                    "cannot serialize CachedProposalEvaluator during evaluation"
                )
            return {
                "delegate": self._delegate,
                "store": self._store,
                "namespace": self._namespace,
                "failure_policy": self._failure_policy,
                "identity_policy": self._identity_policy,
                "content_determinism": self._content_determinism,
                "counters": dict(self._counters),
            }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self._delegate = state["delegate"]
        self._store = state["store"]
        self._namespace = state["namespace"]
        self._failure_policy = state["failure_policy"]
        self._identity_policy = state["identity_policy"]
        self._content_determinism = state["content_determinism"]
        self._counters = dict(state.get("counters", {}))
        self._guard = threading.RLock()
        self._inflight = {}

    @property
    def evaluator_namespace(self) -> str:
        return self._namespace

    @property
    def failure_policy(self) -> FailureCachePolicy:
        return self._failure_policy

    @property
    def identity_policy(self) -> EvaluationIdentityPolicy:
        return self._identity_policy

    def metrics(self) -> EvaluationCacheMetrics:
        with self._guard:
            return EvaluationCacheMetrics(**self._counters)

    def _increment(self, name: str) -> None:
        with self._guard:
            self._counters[name] += 1

    @staticmethod
    def _checked_record(
        record: EvaluationCacheRecord,
        key: EvaluationCacheKey,
        proposal: Proposal,
    ) -> EvaluationCacheRecord:
        if not isinstance(record, EvaluationCacheRecord):
            raise EvaluationCacheContractError(
                "cache store returned a non-record value"
            )
        record.verify()
        if record.key != key:
            raise EvaluationCacheCorruption("cache store returned another key")
        if not key.matches_proposal(proposal):
            raise EvaluationCacheContractError(
                "target proposal does not match evaluation cache key"
            )
        return record

    @staticmethod
    def _evaluation_for_target(
        record: EvaluationCacheRecord,
        target: Proposal,
    ) -> SealedEvaluation:
        source = record.source_proposal
        evaluation = record.evaluation
        if source == target:
            evaluation.verify_for(target)
            return evaluation
        if not record.key.matches_proposal(target):
            raise EvaluationCacheContractError(
                "cached evaluation content does not match target proposal"
            )
        source_evidence = evaluation.evidence()
        descriptor = record.key.descriptor()
        replay_receipt = EvidenceRecord(
            source="astevolve-evaluation-cache",
            kind="content_cache_replay",
            value={"source_evaluation_hash": evaluation.evaluation_hash},
            details={
                "cache_key_digest": record.key.digest,
                "evaluator_namespace": descriptor["evaluator_namespace"],
                "source_proposal_id": source.proposal_id,
                "target_proposal_id": target.proposal_id,
            },
        )
        evidence = EvidenceBundle.of((*source_evidence.records, replay_receipt))


        candidate_id = evaluation.candidate_id
        if evaluation.status == EVALUATION_FAILED:
            rebound = SealedEvaluation.failure(
                proposal=target,
                reason=evaluation.failure_reason,
                evidence=evidence,
                candidate_id=candidate_id,
            )
        else:
            report = evaluation.report()
            if report is None:
                raise EvaluationCacheCorruption(
                    "successful cached evaluation has no report"
                )
            rebound = SealedEvaluation.success(
                proposal=target,
                candidate_id=candidate_id,
                report=report,
                evidence=evidence,
            )
        rebound.verify_for(target)
        return rebound

    def _load(
        self, key: EvaluationCacheKey, proposal: Proposal
    ) -> Optional[EvaluationCacheRecord]:
        record = self._store.get(key)
        if record is None:
            return None
        return self._checked_record(record, key, proposal)

    def evaluate(self, task: EvaluationTask) -> SealedEvaluation:
        trusted = _validated_task(task)
        key = EvaluationCacheKey._from_validated_task(
            trusted,
            evaluator_namespace=self._namespace,
            failure_policy=self._failure_policy,
            identity_policy=self._identity_policy,
            content_determinism=self._content_determinism,
        )
        cached = self._load(key, trusted.proposal)
        if cached is not None:
            self._increment("hits")
            return self._evaluation_for_target(cached, trusted.proposal)

        with self._guard:
            flight = self._inflight.get(key.digest)
            if flight is None:
                flight = Future()
                self._inflight[key.digest] = flight
                leader = True
            else:
                self._counters["singleflight_waits"] += 1
                leader = False
        if not leader:
            return self._evaluation_for_target(flight.result(), trusted.proposal)

        try:
            with self._store.exclusive(key):


                cached = self._load(key, trusted.proposal)
                if cached is not None:
                    self._increment("hits")
                    record = cached
                else:
                    self._increment("misses")
                    result = self._delegate.evaluate(trusted)
                    if not isinstance(result, SealedEvaluation):
                        raise EvaluationCacheContractError(
                            "delegate must return SealedEvaluation"
                        )
                    result.verify_for(trusted.proposal)
                    record = EvaluationCacheRecord.create(
                        key=key,
                        source_proposal=trusted.proposal,
                        evaluation=result,
                    )
                    if (
                        result.status == EVALUATION_FAILED
                        and self._failure_policy is FailureCachePolicy.RETRY
                    ):
                        self._increment("failure_bypasses")
                    else:
                        put = self._store.put_if_absent(record)
                        record = self._checked_record(put.record, key, trusted.proposal)
                        if put.inserted:
                            self._increment("stores")
            flight.set_result(record)
            return self._evaluation_for_target(record, trusted.proposal)
        except BaseException as exc:
            self._increment("errors")
            flight.set_exception(exc)
            raise
        finally:
            with self._guard:
                if self._inflight.get(key.digest) is flight:
                    self._inflight.pop(key.digest, None)


__all__ = [
    "CachedProposalEvaluator",
    "CONTENT_DETERMINISM_SCHEMA",
    "ContentDeterminismDeclaration",
    "EVALUATION_CACHE_KEY_SCHEMA",
    "EVALUATION_CACHE_RECORD_SCHEMA",
    "EvaluationCacheContractError",
    "EvaluationCacheCorruption",
    "EvaluationCacheError",
    "EvaluationCacheKey",
    "EvaluationCacheMetrics",
    "EvaluationCachePutResult",
    "EvaluationCacheRecord",
    "EvaluationCacheStore",
    "EvaluationIdentityPolicy",
    "FailureCachePolicy",
    "FileEvaluationCacheStore",
    "InMemoryEvaluationCacheStore",
]
