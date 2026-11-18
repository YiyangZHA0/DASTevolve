

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Optional

from .causal_flow import CausalFlowContractError, canonical_json
from .experiment_identity import (
    CodeIdentity,
    EvaluatorDescriptor,
    ExactEvaluationKey,
    ExperimentIdentityError,
    SequenceBundleIdentity,
)


REGISTRY_SCHEMA_VERSION = "astevolve.experiment_registry.v1"
REGISTRY_METRICS_VERSION = "astevolve.registry_metrics.v2"
LEASE_CLAIM_VERSION = "astevolve.registry_lease_claim.v1"
SEQUENCE_OCCURRENCE_VERSION = "astevolve.sequence_occurrence_result.v1"
EVALUATION_CACHE_ENTRY_VERSION = "astevolve.evaluation_cache_entry.v1"

_CONTRACT_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_CODE_HASH_RE = re.compile(r"code_sha256:[0-9a-f]{64}\Z")
_SEQUENCE_HASH_RE = re.compile(r"sequence_sha256:[0-9a-f]{64}\Z")
_CACHE_KEY_RE = re.compile(r"evaluation_cache_sha256:[0-9a-f]{64}\Z")
_CLAIM_STATES = frozenset({"pending", "completed", "failed"})


class RegistryContractError(ValueError):


    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)


class RegistryLeaseError(RegistryContractError):
    pass


def _fail(code: str, detail: str = "") -> None:
    raise RegistryContractError(code, detail)


def _lease_fail(code: str, detail: str = "") -> None:
    raise RegistryLeaseError(code, detail)


def _required_text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(code)
    return value.strip()


def _json(value: Any, label: str) -> str:
    try:
        return canonical_json(value)
    except (CausalFlowContractError, TypeError, ValueError) as exc:
        _fail("not_canonical_json", f"{label}:{exc}")


def _json_load(value: Optional[str]) -> Any:
    return None if value is None else json.loads(value)


def _digest(domain: str, payload: Any) -> str:
    encoded = f"{domain}\0{canonical_json(payload)}".encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_number(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(code, repr(value))
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        _fail(code, repr(value))
    return number


def _nonnegative_number(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(code, repr(value))
    number = float(value)
    if not math.isfinite(number) or number < 0:
        _fail(code, repr(value))
    return number


def _contract_hash(value: Any) -> str:
    text = _required_text(value, "contract_hash_required")
    if _CONTRACT_HASH_RE.fullmatch(text) is None:
        _fail("contract_hash_invalid", text)
    return text


def _code_identity(value: CodeIdentity | Mapping[str, Any]) -> CodeIdentity:
    try:
        identity = value if isinstance(value, CodeIdentity) else CodeIdentity.from_mapping(value)
        return CodeIdentity.from_mapping(identity.to_dict())
    except (ExperimentIdentityError, TypeError, ValueError) as exc:
        _fail("code_identity_invalid", str(exc))


def _sequence_hash(value: Any) -> str:
    text = _required_text(value, "sequence_bundle_hash_required")
    if _SEQUENCE_HASH_RE.fullmatch(text) is None:
        _fail("sequence_bundle_hash_invalid", text)
    return text


def _exact_key(value: ExactEvaluationKey | Mapping[str, Any]) -> ExactEvaluationKey:
    try:
        key = (
            value
            if isinstance(value, ExactEvaluationKey)
            else ExactEvaluationKey.from_mapping(value)
        )
        return ExactEvaluationKey.from_mapping(key.to_dict())
    except (ExperimentIdentityError, TypeError, ValueError) as exc:
        _fail("exact_evaluation_key_invalid", str(exc))


@dataclass(frozen=True)
class LeaseClaim:


    acquired: bool
    outcome: str
    status: str
    fencing_token: int
    lease_expires_at: Optional[float]
    attempt_count: int
    cached_result: Any = None
    cached_error: Any = None
    schema_version: str = LEASE_CLAIM_VERSION


@dataclass(frozen=True)
class SequenceOccurrenceResult:
    inserted: bool
    occurrence_hash: str
    sequence_bundle_hash: str
    sequence_occurrence_count: int
    unique_sequence_count: int
    schema_version: str = SEQUENCE_OCCURRENCE_VERSION


@dataclass(frozen=True)
class EvaluationCacheEntry:
    cache_key: str
    sequence_bundle_hash: str
    evaluator_descriptor_hash: str
    status: str
    fencing_token: int
    attempt_count: int
    result: Any
    error: Any
    actual_cost: Optional[float]
    schema_version: str = EVALUATION_CACHE_ENTRY_VERSION


_METRIC_FIELDS = (
    "code_claim_attempts",
    "unique_codes",
    "duplicate_code_claims",
    "effective_contract_claim_attempts",
    "unique_effective_contracts",
    "duplicate_effective_contract_claims",
    "sequence_occurrence_attempts",
    "sequence_occurrences_recorded",
    "unique_sequences",
    "evaluation_claim_attempts",
    "evaluation_executions_claimed",
    "duplicate_expensive_eval_avoided",
    "evaluation_cache_hits",
    "reclaimed_leases",
    "estimated_cache_cost_saved",
    "actual_evaluation_cost",
)


@dataclass(frozen=True)
class RegistryMetrics:


    scope: str
    code_claim_attempts: int
    unique_codes: int
    duplicate_code_claims: int
    effective_contract_claim_attempts: int
    unique_effective_contracts: int
    duplicate_effective_contract_claims: int
    sequence_occurrence_attempts: int
    sequence_occurrences_recorded: int
    unique_sequences: int
    evaluation_claim_attempts: int
    evaluation_executions_claimed: int
    duplicate_expensive_eval_avoided: int
    evaluation_cache_hits: int
    reclaimed_leases: int
    estimated_cache_cost_saved: float
    actual_evaluation_cost: float
    metrics_hash: str
    schema_version: str = REGISTRY_METRICS_VERSION

    @property
    def unique_inner_sequence_rate(self) -> float:
        if self.sequence_occurrence_attempts == 0:
            return 0.0
        return self.unique_sequences / self.sequence_occurrence_attempts

    @property
    def duplicate_proposal_rate(self) -> float:


        if self.code_claim_attempts:
            return self.duplicate_code_claims / self.code_claim_attempts
        if self.effective_contract_claim_attempts:
            return (
                self.duplicate_effective_contract_claims
                / self.effective_contract_claim_attempts
            )
        return 0.0

    @property
    def duplicate_effective_contract_rate(self) -> float:
        if not self.effective_contract_claim_attempts:
            return 0.0
        return (
            self.duplicate_effective_contract_claims
            / self.effective_contract_claim_attempts
        )

    @classmethod
    def create(cls, *, scope: str, **values: Any) -> "RegistryMetrics":
        unknown = sorted(set(values) - set(_METRIC_FIELDS))
        missing = sorted(set(_METRIC_FIELDS) - set(values))
        if unknown:
            _fail("unknown_fields", ",".join(unknown))
        if missing:
            _fail("fields_missing", ",".join(missing))
        clean: dict[str, int | float] = {}
        for name in _METRIC_FIELDS:
            raw = values[name]
            if name in {"estimated_cache_cost_saved", "actual_evaluation_cost"}:
                clean[name] = _nonnegative_number(raw, f"{name}_invalid")
            else:
                if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                    _fail(f"{name}_invalid", repr(raw))
                clean[name] = raw
        resolved_scope = _required_text(scope, "scope_required")
        attempts = int(clean["sequence_occurrence_attempts"])
        unique_rate = float(clean["unique_sequences"]) / attempts if attempts else 0.0
        code_attempts = int(clean["code_claim_attempts"])
        contract_attempts = int(clean["effective_contract_claim_attempts"])
        duplicate_rate = (
            float(clean["duplicate_code_claims"]) / code_attempts
            if code_attempts
            else float(clean["duplicate_effective_contract_claims"]) / contract_attempts
            if contract_attempts
            else 0.0
        )
        duplicate_contract_rate = (
            float(clean["duplicate_effective_contract_claims"]) / contract_attempts
            if contract_attempts
            else 0.0
        )
        semantic = {
            "scope": resolved_scope,
            **clean,
            "unique_inner_sequence_rate": unique_rate,
            "duplicate_proposal_rate": duplicate_rate,
            "duplicate_effective_contract_rate": duplicate_contract_rate,
        }
        metrics_hash = "registry_metrics_sha256:" + _digest(
            "registry_metrics.v1", semantic
        )
        return cls(
            scope=resolved_scope,
            **clean,
            metrics_hash=metrics_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            **{name: getattr(self, name) for name in _METRIC_FIELDS},
            "unique_inner_sequence_rate": self.unique_inner_sequence_rate,
            "duplicate_proposal_rate": self.duplicate_proposal_rate,
            "duplicate_effective_contract_rate": self.duplicate_effective_contract_rate,
            "metrics_hash": self.metrics_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RegistryMetrics":
        if not isinstance(value, Mapping):
            _fail("mapping_required", "registry_metrics")
        fields = {
            "schema_version",
            "scope",
            *_METRIC_FIELDS,
            "unique_inner_sequence_rate",
            "duplicate_proposal_rate",
            "duplicate_effective_contract_rate",
            "metrics_hash",
        }
        unknown = sorted(str(key) for key in set(value) - fields)
        missing = sorted(str(key) for key in fields - set(value))
        if unknown:
            _fail("unknown_fields", ",".join(unknown))
        if missing:
            _fail("fields_missing", ",".join(missing))
        if value.get("schema_version") != REGISTRY_METRICS_VERSION:
            _fail("schema_version_invalid", str(value.get("schema_version")))
        metrics = cls.create(
            scope=value.get("scope"),
            **{name: value.get(name) for name in _METRIC_FIELDS},
        )
        if value.get("unique_inner_sequence_rate") != metrics.unique_inner_sequence_rate:
            _fail("derived_metric_mismatch", "unique_inner_sequence_rate")
        if value.get("duplicate_proposal_rate") != metrics.duplicate_proposal_rate:
            _fail("derived_metric_mismatch", "duplicate_proposal_rate")
        if (
            value.get("duplicate_effective_contract_rate")
            != metrics.duplicate_effective_contract_rate
        ):
            _fail("derived_metric_mismatch", "duplicate_effective_contract_rate")
        if value.get("metrics_hash") != metrics.metrics_hash:
            _fail("hash_mismatch", "registry_metrics")
        return metrics


class ExperimentRegistry:


    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        clock: Callable[[], float] = time.time,
        timeout_seconds: float = 30.0,
        default_lease_seconds: float = 300.0,
    ) -> None:
        self.path = str(path)
        if not self.path:
            _fail("registry_path_required")
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self._clock = clock
        self._default_lease_seconds = _positive_number(
            default_lease_seconds, "default_lease_seconds_invalid"
        )
        timeout = _positive_number(timeout_seconds, "timeout_seconds_invalid")
        self._lock = threading.RLock()
        self._closed = False
        self._connection = sqlite3.connect(
            self.path,
            timeout=timeout,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._journal_mode = str(
            self._connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        ).lower()


        if self.path != ":memory:" and self._journal_mode != "wal":
            self.close()
            _fail("wal_unavailable", self._journal_mode)
        self._initialize_schema()

    @property
    def journal_mode(self) -> str:
        return self._journal_mode

    def __enter__(self) -> "ExperimentRegistry":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            _fail("registry_closed")

    def _initialize_schema(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS registry_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS effective_contract_claims (
                    scope TEXT NOT NULL,
                    contract_hash TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','completed','failed')),
                    owner_token TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL CHECK(fencing_token >= 1),
                    lease_expires_at REAL,
                    attempt_count INTEGER NOT NULL CHECK(attempt_count >= 1),
                    error_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(scope, contract_hash)
                );

                CREATE TABLE IF NOT EXISTS code_claims (
                    scope TEXT NOT NULL,
                    code_hash TEXT NOT NULL,
                    code_identity_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','completed','failed')),
                    owner_token TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL CHECK(fencing_token >= 1),
                    lease_expires_at REAL,
                    attempt_count INTEGER NOT NULL CHECK(attempt_count >= 1),
                    error_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(scope, code_hash)
                );

                CREATE TABLE IF NOT EXISTS sequence_occurrences (
                    scope TEXT NOT NULL,
                    occurrence_hash TEXT NOT NULL,
                    sequence_bundle_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    context_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(scope, occurrence_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_sequence_identity
                    ON sequence_occurrences(scope, sequence_bundle_hash);

                CREATE TABLE IF NOT EXISTS sequence_identities (
                    scope TEXT NOT NULL,
                    sequence_bundle_hash TEXT NOT NULL,
                    occurrence_count INTEGER NOT NULL CHECK(occurrence_count >= 1),
                    first_seen_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(scope, sequence_bundle_hash)
                );

                CREATE TABLE IF NOT EXISTS evaluation_cache (
                    scope TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    sequence_bundle_hash TEXT NOT NULL,
                    evaluator_descriptor_hash TEXT NOT NULL,
                    exact_key_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','completed','failed')),
                    owner_token TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL CHECK(fencing_token >= 1),
                    lease_expires_at REAL,
                    attempt_count INTEGER NOT NULL CHECK(attempt_count >= 1),
                    result_json TEXT,
                    error_json TEXT,
                    actual_cost REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(scope, cache_key)
                );
                CREATE INDEX IF NOT EXISTS idx_evaluation_sequence
                    ON evaluation_cache(scope, sequence_bundle_hash);

                CREATE TABLE IF NOT EXISTS registry_counters (
                    scope TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    PRIMARY KEY(scope, name)
                );
                """
            )
            row = self._connection.execute(
                "SELECT value FROM registry_metadata WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO registry_metadata(key,value) VALUES('schema_version',?)",
                    (REGISTRY_SCHEMA_VERSION,),
                )
            elif row["value"] != REGISTRY_SCHEMA_VERSION:
                self.close()
                _fail("database_schema_invalid", str(row["value"]))


            self._connection.execute("BEGIN IMMEDIATE")
            try:
                migrated = self._connection.execute(
                    "SELECT 1 FROM registry_metadata "
                    "WHERE key='sequence_identity_index_v1'"
                ).fetchone()
                if migrated is None:
                    self._connection.execute(
                        """
                        INSERT OR IGNORE INTO sequence_identities(
                            scope,sequence_bundle_hash,occurrence_count,
                            first_seen_at,updated_at
                        )
                        SELECT scope,sequence_bundle_hash,COUNT(*),MIN(created_at),MAX(created_at)
                        FROM sequence_occurrences
                        GROUP BY scope,sequence_bundle_hash
                        """
                    )
                    self._connection.execute(
                        "DELETE FROM registry_counters WHERE name='unique_sequences'"
                    )
                    self._connection.execute(
                        """
                        INSERT INTO registry_counters(scope,name,value)
                        SELECT scope,'unique_sequences',COUNT(*)
                        FROM sequence_identities
                        GROUP BY scope
                        """
                    )
                    self._connection.execute(
                        "INSERT INTO registry_metadata(key,value) "
                        "VALUES('sequence_identity_index_v1','complete')"
                    )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        self._require_open()
        with self._lock:
            cursor = self._connection.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                yield cursor
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
            finally:
                cursor.close()

    def _now(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail("clock_value_invalid", repr(value))
        value = float(value)
        if not math.isfinite(value):
            _fail("clock_value_invalid", repr(value))
        return value

    @staticmethod
    def _increment(
        cursor: sqlite3.Cursor, scope: str, name: str, amount: float = 1.0
    ) -> None:
        cursor.execute(
            """
            INSERT INTO registry_counters(scope,name,value) VALUES(?,?,?)
            ON CONFLICT(scope,name) DO UPDATE SET value=value+excluded.value
            """,
            (scope, name, float(amount)),
        )

    @staticmethod
    def _claim_result(
        *,
        acquired: bool,
        outcome: str,
        status: str,
        fencing_token: int,
        lease_expires_at: Optional[float],
        attempt_count: int,
        result_json: Optional[str] = None,
        error_json: Optional[str] = None,
    ) -> LeaseClaim:
        return LeaseClaim(
            acquired=acquired,
            outcome=outcome,
            status=status,
            fencing_token=int(fencing_token),
            lease_expires_at=(
                float(lease_expires_at) if lease_expires_at is not None else None
            ),
            attempt_count=int(attempt_count),
            cached_result=_json_load(result_json),
            cached_error=_json_load(error_json),
        )

    def claim_code(
        self,
        identity: CodeIdentity | Mapping[str, Any],
        *,
        owner_token: str,
        scope: str = "default",
        lease_seconds: Optional[float] = None,
        retry_failed: bool = False,
    ) -> LeaseClaim:


        code = _code_identity(identity)
        owner = _required_text(owner_token, "owner_token_required")
        resolved_scope = _required_text(scope, "scope_required")
        lease = (
            self._default_lease_seconds
            if lease_seconds is None
            else _positive_number(lease_seconds, "lease_seconds_invalid")
        )
        if not isinstance(retry_failed, bool):
            _fail("retry_failed_invalid")
        now = self._now()
        expires = now + lease
        identity_json = canonical_json(code.to_dict())
        with self._transaction() as cursor:
            self._increment(cursor, resolved_scope, "code_claim_attempts")
            row = cursor.execute(
                "SELECT * FROM code_claims WHERE scope=? AND code_hash=?",
                (resolved_scope, code.code_hash),
            ).fetchone()
            if row is None:
                cursor.execute(
                    """
                    INSERT INTO code_claims(
                        scope,code_hash,code_identity_json,status,owner_token,
                        fencing_token,lease_expires_at,attempt_count,error_json,
                        created_at,updated_at
                    ) VALUES(?,?,?,'pending',?,1,?,1,NULL,?,?)
                    """,
                    (
                        resolved_scope,
                        code.code_hash,
                        identity_json,
                        owner,
                        expires,
                        now,
                        now,
                    ),
                )
                return self._claim_result(
                    acquired=True,
                    outcome="acquired",
                    status="pending",
                    fencing_token=1,
                    lease_expires_at=expires,
                    attempt_count=1,
                )

            if row["code_identity_json"] != identity_json:
                _fail("code_hash_collision", code.code_hash)
            attempts = int(row["attempt_count"]) + 1
            status = str(row["status"])
            fence = int(row["fencing_token"])
            if status == "pending" and float(row["lease_expires_at"]) <= now:
                fence += 1
                cursor.execute(
                    """
                    UPDATE code_claims
                    SET owner_token=?,fencing_token=?,lease_expires_at=?,
                        attempt_count=?,error_json=NULL,updated_at=?
                    WHERE scope=? AND code_hash=?
                    """,
                    (
                        owner,
                        fence,
                        expires,
                        attempts,
                        now,
                        resolved_scope,
                        code.code_hash,
                    ),
                )
                self._increment(cursor, resolved_scope, "reclaimed_leases")
                return self._claim_result(
                    acquired=True,
                    outcome="reclaimed",
                    status="pending",
                    fencing_token=fence,
                    lease_expires_at=expires,
                    attempt_count=attempts,
                )
            if status == "failed" and retry_failed:
                fence += 1
                cursor.execute(
                    """
                    UPDATE code_claims
                    SET status='pending',owner_token=?,fencing_token=?,
                        lease_expires_at=?,attempt_count=?,error_json=NULL,updated_at=?
                    WHERE scope=? AND code_hash=?
                    """,
                    (
                        owner,
                        fence,
                        expires,
                        attempts,
                        now,
                        resolved_scope,
                        code.code_hash,
                    ),
                )
                return self._claim_result(
                    acquired=True,
                    outcome="retried",
                    status="pending",
                    fencing_token=fence,
                    lease_expires_at=expires,
                    attempt_count=attempts,
                )

            outcome = {
                "pending": "duplicate_pending",
                "completed": "duplicate_completed",
                "failed": "duplicate_failed",
            }[status]
            cursor.execute(
                "UPDATE code_claims SET attempt_count=?,updated_at=? "
                "WHERE scope=? AND code_hash=?",
                (attempts, now, resolved_scope, code.code_hash),
            )
            self._increment(cursor, resolved_scope, "duplicate_code_claims")
            return self._claim_result(
                acquired=False,
                outcome=outcome,
                status=status,
                fencing_token=fence,
                lease_expires_at=row["lease_expires_at"],
                attempt_count=attempts,
                error_json=row["error_json"],
            )

    def _finish_code(
        self,
        identity: CodeIdentity | Mapping[str, Any],
        *,
        owner_token: str,
        fencing_token: int,
        status: str,
        error: Any = None,
        scope: str = "default",
    ) -> None:
        code = _code_identity(identity)
        owner = _required_text(owner_token, "owner_token_required")
        resolved_scope = _required_text(scope, "scope_required")
        if status not in {"completed", "failed"}:
            _fail("claim_status_invalid", status)
        if isinstance(fencing_token, bool) or not isinstance(fencing_token, int):
            _fail("fencing_token_invalid", repr(fencing_token))
        error_json = _json(error, "error") if status == "failed" else None
        now = self._now()
        identity_json = canonical_json(code.to_dict())
        with self._transaction() as cursor:
            row = cursor.execute(
                "SELECT status,owner_token,fencing_token,code_identity_json "
                "FROM code_claims WHERE scope=? AND code_hash=?",
                (resolved_scope, code.code_hash),
            ).fetchone()
            if row is not None and row["code_identity_json"] != identity_json:
                _fail("code_hash_collision", code.code_hash)
            if (
                row is None
                or row["status"] != "pending"
                or row["owner_token"] != owner
                or int(row["fencing_token"]) != fencing_token
            ):
                _lease_fail("stale_lease", f"code:{code.code_hash}")
            cursor.execute(
                """
                UPDATE code_claims
                SET status=?,lease_expires_at=NULL,error_json=?,updated_at=?
                WHERE scope=? AND code_hash=?
                """,
                (status, error_json, now, resolved_scope, code.code_hash),
            )

    def complete_code(
        self,
        identity: CodeIdentity | Mapping[str, Any],
        *,
        owner_token: str,
        fencing_token: int,
        scope: str = "default",
    ) -> None:
        self._finish_code(
            identity,
            owner_token=owner_token,
            fencing_token=fencing_token,
            status="completed",
            scope=scope,
        )

    def fail_code(
        self,
        identity: CodeIdentity | Mapping[str, Any],
        *,
        owner_token: str,
        fencing_token: int,
        error: Any,
        scope: str = "default",
    ) -> None:
        self._finish_code(
            identity,
            owner_token=owner_token,
            fencing_token=fencing_token,
            status="failed",
            error=error,
            scope=scope,
        )

    def renew_code_lease(
        self,
        identity: CodeIdentity | Mapping[str, Any],
        *,
        owner_token: str,
        fencing_token: int,
        scope: str = "default",
        lease_seconds: Optional[float] = None,
    ) -> float:
        code = _code_identity(identity)
        owner = _required_text(owner_token, "owner_token_required")
        resolved_scope = _required_text(scope, "scope_required")
        if isinstance(fencing_token, bool) or not isinstance(fencing_token, int):
            _fail("fencing_token_invalid", repr(fencing_token))
        lease = (
            self._default_lease_seconds
            if lease_seconds is None
            else _positive_number(lease_seconds, "lease_seconds_invalid")
        )
        now = self._now()
        expires = now + lease
        with self._transaction() as cursor:
            row = cursor.execute(
                "SELECT status,owner_token,fencing_token,lease_expires_at "
                "FROM code_claims WHERE scope=? AND code_hash=?",
                (resolved_scope, code.code_hash),
            ).fetchone()
            if (
                row is None
                or row["status"] != "pending"
                or row["owner_token"] != owner
                or int(row["fencing_token"]) != fencing_token
                or row["lease_expires_at"] is None
                or float(row["lease_expires_at"]) <= now
            ):
                _lease_fail("stale_lease", f"code:{code.code_hash}")
            cursor.execute(
                "UPDATE code_claims SET lease_expires_at=?,updated_at=? "
                "WHERE scope=? AND code_hash=?",
                (expires, now, resolved_scope, code.code_hash),
            )
        return expires

    def claim_effective_contract(
        self,
        contract_hash: str,
        *,
        owner_token: str,
        scope: str = "default",
        lease_seconds: Optional[float] = None,
        retry_failed: bool = False,
    ) -> LeaseClaim:
        identity = _contract_hash(contract_hash)
        owner = _required_text(owner_token, "owner_token_required")
        resolved_scope = _required_text(scope, "scope_required")
        lease = (
            self._default_lease_seconds
            if lease_seconds is None
            else _positive_number(lease_seconds, "lease_seconds_invalid")
        )
        if not isinstance(retry_failed, bool):
            _fail("retry_failed_invalid")
        now = self._now()
        expires = now + lease
        with self._transaction() as cursor:
            self._increment(
                cursor, resolved_scope, "effective_contract_claim_attempts"
            )
            row = cursor.execute(
                "SELECT * FROM effective_contract_claims WHERE scope=? AND contract_hash=?",
                (resolved_scope, identity),
            ).fetchone()
            if row is None:
                cursor.execute(
                    """
                    INSERT INTO effective_contract_claims(
                        scope,contract_hash,status,owner_token,fencing_token,
                        lease_expires_at,attempt_count,error_json,created_at,updated_at
                    ) VALUES(?,?,'pending',?,1,?,1,NULL,?,?)
                    """,
                    (resolved_scope, identity, owner, expires, now, now),
                )
                return self._claim_result(
                    acquired=True,
                    outcome="acquired",
                    status="pending",
                    fencing_token=1,
                    lease_expires_at=expires,
                    attempt_count=1,
                )

            attempts = int(row["attempt_count"]) + 1
            status = str(row["status"])
            fence = int(row["fencing_token"])
            if status == "pending" and float(row["lease_expires_at"]) <= now:
                fence += 1
                cursor.execute(
                    """
                    UPDATE effective_contract_claims
                    SET owner_token=?,fencing_token=?,lease_expires_at=?,
                        attempt_count=?,error_json=NULL,updated_at=?
                    WHERE scope=? AND contract_hash=?
                    """,
                    (owner, fence, expires, attempts, now, resolved_scope, identity),
                )
                self._increment(cursor, resolved_scope, "reclaimed_leases")
                return self._claim_result(
                    acquired=True,
                    outcome="reclaimed",
                    status="pending",
                    fencing_token=fence,
                    lease_expires_at=expires,
                    attempt_count=attempts,
                )
            if status == "failed" and retry_failed:
                fence += 1
                cursor.execute(
                    """
                    UPDATE effective_contract_claims
                    SET status='pending',owner_token=?,fencing_token=?,lease_expires_at=?,
                        attempt_count=?,error_json=NULL,updated_at=?
                    WHERE scope=? AND contract_hash=?
                    """,
                    (owner, fence, expires, attempts, now, resolved_scope, identity),
                )
                return self._claim_result(
                    acquired=True,
                    outcome="retried",
                    status="pending",
                    fencing_token=fence,
                    lease_expires_at=expires,
                    attempt_count=attempts,
                )

            outcome = {
                "pending": "duplicate_pending",
                "completed": "duplicate_completed",
                "failed": "duplicate_failed",
            }[status]
            cursor.execute(
                """
                UPDATE effective_contract_claims
                SET attempt_count=?,updated_at=? WHERE scope=? AND contract_hash=?
                """,
                (attempts, now, resolved_scope, identity),
            )
            self._increment(
                cursor, resolved_scope, "duplicate_effective_contract_claims"
            )
            return self._claim_result(
                acquired=False,
                outcome=outcome,
                status=status,
                fencing_token=fence,
                lease_expires_at=row["lease_expires_at"],
                attempt_count=attempts,
                error_json=row["error_json"],
            )

    def _finish_effective_contract(
        self,
        contract_hash: str,
        *,
        owner_token: str,
        fencing_token: int,
        status: str,
        error: Any = None,
        scope: str = "default",
    ) -> None:
        identity = _contract_hash(contract_hash)
        owner = _required_text(owner_token, "owner_token_required")
        resolved_scope = _required_text(scope, "scope_required")
        if status not in {"completed", "failed"}:
            _fail("claim_status_invalid", status)
        if isinstance(fencing_token, bool) or not isinstance(fencing_token, int):
            _fail("fencing_token_invalid", repr(fencing_token))
        error_json = _json(error, "error") if status == "failed" else None
        now = self._now()
        with self._transaction() as cursor:
            row = cursor.execute(
                "SELECT status,owner_token,fencing_token FROM effective_contract_claims WHERE scope=? AND contract_hash=?",
                (resolved_scope, identity),
            ).fetchone()
            if (
                row is None
                or row["status"] != "pending"
                or row["owner_token"] != owner
                or int(row["fencing_token"]) != fencing_token
            ):
                _lease_fail("stale_lease", f"effective_contract:{identity}")
            cursor.execute(
                """
                UPDATE effective_contract_claims
                SET status=?,lease_expires_at=NULL,error_json=?,updated_at=?
                WHERE scope=? AND contract_hash=?
                """,
                (status, error_json, now, resolved_scope, identity),
            )

    def complete_effective_contract(
        self,
        contract_hash: str,
        *,
        owner_token: str,
        fencing_token: int,
        scope: str = "default",
    ) -> None:
        self._finish_effective_contract(
            contract_hash,
            owner_token=owner_token,
            fencing_token=fencing_token,
            status="completed",
            scope=scope,
        )

    def fail_effective_contract(
        self,
        contract_hash: str,
        *,
        owner_token: str,
        fencing_token: int,
        error: Any,
        scope: str = "default",
    ) -> None:
        self._finish_effective_contract(
            contract_hash,
            owner_token=owner_token,
            fencing_token=fencing_token,
            status="failed",
            error=error,
            scope=scope,
        )

    def renew_effective_contract_lease(
        self,
        contract_hash: str,
        *,
        owner_token: str,
        fencing_token: int,
        scope: str = "default",
        lease_seconds: Optional[float] = None,
    ) -> float:


        identity = _contract_hash(contract_hash)
        owner = _required_text(owner_token, "owner_token_required")
        resolved_scope = _required_text(scope, "scope_required")
        if isinstance(fencing_token, bool) or not isinstance(fencing_token, int):
            _fail("fencing_token_invalid", repr(fencing_token))
        lease = (
            self._default_lease_seconds
            if lease_seconds is None
            else _positive_number(lease_seconds, "lease_seconds_invalid")
        )
        now = self._now()
        expires = now + lease
        with self._transaction() as cursor:
            row = cursor.execute(
                "SELECT status,owner_token,fencing_token,lease_expires_at FROM effective_contract_claims WHERE scope=? AND contract_hash=?",
                (resolved_scope, identity),
            ).fetchone()
            if (
                row is None
                or row["status"] != "pending"
                or row["owner_token"] != owner
                or int(row["fencing_token"]) != fencing_token
                or row["lease_expires_at"] is None
                or float(row["lease_expires_at"]) <= now
            ):
                _lease_fail("stale_lease", f"effective_contract:{identity}")
            cursor.execute(
                "UPDATE effective_contract_claims SET lease_expires_at=?,updated_at=? WHERE scope=? AND contract_hash=?",
                (expires, now, resolved_scope, identity),
            )
        return expires

    def register_sequence_occurrence(
        self,
        sequence: Mapping[str, str] | SequenceBundleIdentity | str,
        *,
        role: str,
        context_id: str,
        scope: str = "default",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SequenceOccurrenceResult:
        if isinstance(sequence, str):
            sequence_bundle_hash = _sequence_hash(sequence)
        else:
            try:
                sequence_bundle_hash = SequenceBundleIdentity.create(
                    sequence
                ).sequence_bundle_hash
            except (ExperimentIdentityError, TypeError, ValueError) as exc:
                _fail("sequence_bundle_invalid", str(exc))
        resolved_role = _required_text(role, "sequence_role_required")
        resolved_context = _required_text(context_id, "context_id_required")
        resolved_scope = _required_text(scope, "scope_required")
        metadata_json = _json(dict(metadata or {}), "sequence_metadata")
        occurrence_hash = "sequence_occurrence_sha256:" + _digest(
            "sequence_occurrence.v1",
            {
                "scope": resolved_scope,
                "sequence_bundle_hash": sequence_bundle_hash,
                "role": resolved_role,
                "context_id": resolved_context,
            },
        )
        now = self._now()
        with self._transaction() as cursor:
            self._increment(cursor, resolved_scope, "sequence_occurrence_attempts")
            row = cursor.execute(
                "SELECT * FROM sequence_occurrences WHERE scope=? AND occurrence_hash=?",
                (resolved_scope, occurrence_hash),
            ).fetchone()
            inserted = row is None
            if inserted:
                cursor.execute(
                    """
                    INSERT INTO sequence_occurrences(
                        scope,occurrence_hash,sequence_bundle_hash,role,context_id,
                        metadata_json,created_at
                    ) VALUES(?,?,?,?,?,?,?)
                    """,
                    (
                        resolved_scope,
                        occurrence_hash,
                        sequence_bundle_hash,
                        resolved_role,
                        resolved_context,
                        metadata_json,
                        now,
                    ),
                )
                identity_row = cursor.execute(
                    """
                    SELECT occurrence_count FROM sequence_identities
                    WHERE scope=? AND sequence_bundle_hash=?
                    """,
                    (resolved_scope, sequence_bundle_hash),
                ).fetchone()
                if identity_row is None:
                    cursor.execute(
                        """
                        INSERT INTO sequence_identities(
                            scope,sequence_bundle_hash,occurrence_count,
                            first_seen_at,updated_at
                        ) VALUES(?,?,1,?,?)
                        """,
                        (resolved_scope, sequence_bundle_hash, now, now),
                    )
                    self._increment(
                        cursor, resolved_scope, "unique_sequences"
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE sequence_identities
                        SET occurrence_count=occurrence_count+1,updated_at=?
                        WHERE scope=? AND sequence_bundle_hash=?
                        """,
                        (now, resolved_scope, sequence_bundle_hash),
                    )
            elif row["metadata_json"] != metadata_json:
                _fail("sequence_occurrence_conflict", occurrence_hash)
            occurrence_count = int(
                cursor.execute(
                    """
                    SELECT occurrence_count FROM sequence_identities
                    WHERE scope=? AND sequence_bundle_hash=?
                    """,
                    (resolved_scope, sequence_bundle_hash),
                ).fetchone()[0]
            )
            unique_count = int(
                cursor.execute(
                    """
                    SELECT COALESCE(value,0) FROM registry_counters
                    WHERE scope=? AND name='unique_sequences'
                    """,
                    (resolved_scope,),
                ).fetchone()[0]
            )
            return SequenceOccurrenceResult(
                inserted,
                occurrence_hash,
                sequence_bundle_hash,
                occurrence_count,
                unique_count,
            )

    def has_sequence(
        self, sequence_bundle_hash: str, *, scope: str = "default"
    ) -> bool:
        identity = _sequence_hash(sequence_bundle_hash)
        resolved_scope = _required_text(scope, "scope_required")
        self._require_open()
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM sequence_identities "
                "WHERE scope=? AND sequence_bundle_hash=? LIMIT 1",
                (resolved_scope, identity),
            ).fetchone()
        return row is not None

    def claim_evaluation(
        self,
        key: ExactEvaluationKey | Mapping[str, Any],
        *,
        owner_token: str,
        scope: str = "default",
        lease_seconds: Optional[float] = None,
        retry_failed: bool = False,
        estimated_cost: float = 0.0,
    ) -> LeaseClaim:
        exact = _exact_key(key)
        owner = _required_text(owner_token, "owner_token_required")
        resolved_scope = _required_text(scope, "scope_required")
        lease = (
            self._default_lease_seconds
            if lease_seconds is None
            else _positive_number(lease_seconds, "lease_seconds_invalid")
        )
        if not isinstance(retry_failed, bool):
            _fail("retry_failed_invalid")
        cost = _nonnegative_number(estimated_cost, "estimated_cost_invalid")
        now = self._now()
        expires = now + lease
        key_json = canonical_json(exact.to_dict())
        with self._transaction() as cursor:
            self._increment(cursor, resolved_scope, "evaluation_claim_attempts")
            row = cursor.execute(
                "SELECT * FROM evaluation_cache WHERE scope=? AND cache_key=?",
                (resolved_scope, exact.cache_key),
            ).fetchone()
            if row is None:
                cursor.execute(
                    """
                    INSERT INTO evaluation_cache(
                        scope,cache_key,sequence_bundle_hash,evaluator_descriptor_hash,
                        exact_key_json,status,owner_token,fencing_token,lease_expires_at,
                        attempt_count,result_json,error_json,actual_cost,created_at,updated_at
                    ) VALUES(?,?,?,?,?,'pending',?,1,?,1,NULL,NULL,NULL,?,?)
                    """,
                    (
                        resolved_scope,
                        exact.cache_key,
                        exact.sequence_bundle_hash,
                        exact.evaluator_descriptor_hash,
                        key_json,
                        owner,
                        expires,
                        now,
                        now,
                    ),
                )
                self._increment(
                    cursor, resolved_scope, "evaluation_executions_claimed"
                )
                return self._claim_result(
                    acquired=True,
                    outcome="acquired",
                    status="pending",
                    fencing_token=1,
                    lease_expires_at=expires,
                    attempt_count=1,
                )

            if row["exact_key_json"] != key_json:
                _fail("evaluation_cache_key_collision", exact.cache_key)
            attempts = int(row["attempt_count"]) + 1
            status = str(row["status"])
            fence = int(row["fencing_token"])
            if status == "pending" and float(row["lease_expires_at"]) <= now:
                fence += 1
                cursor.execute(
                    """
                    UPDATE evaluation_cache
                    SET owner_token=?,fencing_token=?,lease_expires_at=?,attempt_count=?,
                        result_json=NULL,error_json=NULL,actual_cost=NULL,updated_at=?
                    WHERE scope=? AND cache_key=?
                    """,
                    (
                        owner,
                        fence,
                        expires,
                        attempts,
                        now,
                        resolved_scope,
                        exact.cache_key,
                    ),
                )
                self._increment(cursor, resolved_scope, "reclaimed_leases")
                self._increment(
                    cursor, resolved_scope, "evaluation_executions_claimed"
                )
                return self._claim_result(
                    acquired=True,
                    outcome="reclaimed",
                    status="pending",
                    fencing_token=fence,
                    lease_expires_at=expires,
                    attempt_count=attempts,
                )
            if status == "failed" and retry_failed:
                fence += 1
                cursor.execute(
                    """
                    UPDATE evaluation_cache
                    SET status='pending',owner_token=?,fencing_token=?,lease_expires_at=?,
                        attempt_count=?,result_json=NULL,error_json=NULL,actual_cost=NULL,updated_at=?
                    WHERE scope=? AND cache_key=?
                    """,
                    (
                        owner,
                        fence,
                        expires,
                        attempts,
                        now,
                        resolved_scope,
                        exact.cache_key,
                    ),
                )
                self._increment(
                    cursor, resolved_scope, "evaluation_executions_claimed"
                )
                return self._claim_result(
                    acquired=True,
                    outcome="retried",
                    status="pending",
                    fencing_token=fence,
                    lease_expires_at=expires,
                    attempt_count=attempts,
                )

            outcome = {
                "pending": "duplicate_pending",
                "completed": "cache_hit",
                "failed": "duplicate_failed",
            }[status]
            cursor.execute(
                "UPDATE evaluation_cache SET attempt_count=?,updated_at=? WHERE scope=? AND cache_key=?",
                (attempts, now, resolved_scope, exact.cache_key),
            )


            self._increment(
                cursor, resolved_scope, "duplicate_expensive_eval_avoided"
            )
            self._increment(
                cursor, resolved_scope, "estimated_cache_cost_saved", cost
            )
            if status == "completed":
                self._increment(cursor, resolved_scope, "evaluation_cache_hits")
            return self._claim_result(
                acquired=False,
                outcome=outcome,
                status=status,
                fencing_token=fence,
                lease_expires_at=row["lease_expires_at"],
                attempt_count=attempts,
                result_json=row["result_json"],
                error_json=row["error_json"],
            )

    def _finish_evaluation(
        self,
        key: ExactEvaluationKey | Mapping[str, Any],
        *,
        owner_token: str,
        fencing_token: int,
        status: str,
        result: Any = None,
        error: Any = None,
        actual_cost: Optional[float] = None,
        scope: str = "default",
    ) -> None:
        exact = _exact_key(key)
        owner = _required_text(owner_token, "owner_token_required")
        resolved_scope = _required_text(scope, "scope_required")
        if isinstance(fencing_token, bool) or not isinstance(fencing_token, int):
            _fail("fencing_token_invalid", repr(fencing_token))
        if status not in {"completed", "failed"}:
            _fail("claim_status_invalid", status)
        result_json = _json(result, "evaluation_result") if status == "completed" else None
        error_json = _json(error, "evaluation_error") if status == "failed" else None
        cost = (
            None
            if actual_cost is None
            else _nonnegative_number(actual_cost, "actual_cost_invalid")
        )
        now = self._now()
        with self._transaction() as cursor:
            row = cursor.execute(
                "SELECT status,owner_token,fencing_token FROM evaluation_cache WHERE scope=? AND cache_key=?",
                (resolved_scope, exact.cache_key),
            ).fetchone()
            if (
                row is None
                or row["status"] != "pending"
                or row["owner_token"] != owner
                or int(row["fencing_token"]) != fencing_token
            ):
                _lease_fail("stale_lease", f"evaluation:{exact.cache_key}")
            cursor.execute(
                """
                UPDATE evaluation_cache
                SET status=?,lease_expires_at=NULL,result_json=?,error_json=?,
                    actual_cost=?,updated_at=?
                WHERE scope=? AND cache_key=?
                """,
                (
                    status,
                    result_json,
                    error_json,
                    cost,
                    now,
                    resolved_scope,
                    exact.cache_key,
                ),
            )
            if cost is not None:
                self._increment(
                    cursor, resolved_scope, "actual_evaluation_cost", cost
                )

    def complete_evaluation(
        self,
        key: ExactEvaluationKey | Mapping[str, Any],
        *,
        owner_token: str,
        fencing_token: int,
        result: Any,
        actual_cost: Optional[float] = None,
        scope: str = "default",
    ) -> None:
        self._finish_evaluation(
            key,
            owner_token=owner_token,
            fencing_token=fencing_token,
            status="completed",
            result=result,
            actual_cost=actual_cost,
            scope=scope,
        )

    def fail_evaluation(
        self,
        key: ExactEvaluationKey | Mapping[str, Any],
        *,
        owner_token: str,
        fencing_token: int,
        error: Any,
        actual_cost: Optional[float] = None,
        scope: str = "default",
    ) -> None:
        self._finish_evaluation(
            key,
            owner_token=owner_token,
            fencing_token=fencing_token,
            status="failed",
            error=error,
            actual_cost=actual_cost,
            scope=scope,
        )

    def renew_evaluation_lease(
        self,
        key: ExactEvaluationKey | Mapping[str, Any],
        *,
        owner_token: str,
        fencing_token: int,
        scope: str = "default",
        lease_seconds: Optional[float] = None,
    ) -> float:


        exact = _exact_key(key)
        owner = _required_text(owner_token, "owner_token_required")
        resolved_scope = _required_text(scope, "scope_required")
        if isinstance(fencing_token, bool) or not isinstance(fencing_token, int):
            _fail("fencing_token_invalid", repr(fencing_token))
        lease = (
            self._default_lease_seconds
            if lease_seconds is None
            else _positive_number(lease_seconds, "lease_seconds_invalid")
        )
        now = self._now()
        expires = now + lease
        with self._transaction() as cursor:
            row = cursor.execute(
                "SELECT status,owner_token,fencing_token,lease_expires_at FROM evaluation_cache WHERE scope=? AND cache_key=?",
                (resolved_scope, exact.cache_key),
            ).fetchone()
            if (
                row is None
                or row["status"] != "pending"
                or row["owner_token"] != owner
                or int(row["fencing_token"]) != fencing_token
                or row["lease_expires_at"] is None
                or float(row["lease_expires_at"]) <= now
            ):
                _lease_fail("stale_lease", f"evaluation:{exact.cache_key}")
            cursor.execute(
                "UPDATE evaluation_cache SET lease_expires_at=?,updated_at=? WHERE scope=? AND cache_key=?",
                (expires, now, resolved_scope, exact.cache_key),
            )
        return expires

    def lookup_evaluation(
        self,
        key: ExactEvaluationKey | Mapping[str, Any],
        *,
        scope: str = "default",
    ) -> Optional[EvaluationCacheEntry]:
        exact = _exact_key(key)
        resolved_scope = _required_text(scope, "scope_required")
        self._require_open()
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM evaluation_cache WHERE scope=? AND cache_key=?",
                (resolved_scope, exact.cache_key),
            ).fetchone()
        if row is None:
            return None
        return EvaluationCacheEntry(
            cache_key=row["cache_key"],
            sequence_bundle_hash=row["sequence_bundle_hash"],
            evaluator_descriptor_hash=row["evaluator_descriptor_hash"],
            status=row["status"],
            fencing_token=int(row["fencing_token"]),
            attempt_count=int(row["attempt_count"]),
            result=_json_load(row["result_json"]),
            error=_json_load(row["error_json"]),
            actual_cost=(
                float(row["actual_cost"]) if row["actual_cost"] is not None else None
            ),
        )

    def _counter(self, *, scope: Optional[str], name: str) -> float:
        if scope is None:
            row = self._connection.execute(
                "SELECT COALESCE(SUM(value),0) FROM registry_counters WHERE name=?",
                (name,),
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT COALESCE(value,0) FROM registry_counters WHERE scope=? AND name=?",
                (scope, name),
            ).fetchone()
        return float(row[0]) if row is not None else 0.0

    def _count(self, table: str, *, scope: Optional[str], distinct: Optional[str] = None) -> int:

        expression = f"COUNT(DISTINCT {distinct})" if distinct else "COUNT(*)"
        if scope is None:
            row = self._connection.execute(
                f"SELECT {expression} FROM {table}"
            ).fetchone()
        else:
            row = self._connection.execute(
                f"SELECT {expression} FROM {table} WHERE scope=?", (scope,)
            ).fetchone()
        return int(row[0])

    def metrics(self, *, scope: Optional[str] = None) -> RegistryMetrics:
        resolved_scope = (
            None if scope is None else _required_text(scope, "scope_required")
        )
        self._require_open()
        with self._lock:
            values = {
                "code_claim_attempts": int(
                    self._counter(scope=resolved_scope, name="code_claim_attempts")
                ),
                "unique_codes": self._count(
                    "code_claims", scope=resolved_scope, distinct="code_hash"
                ),
                "duplicate_code_claims": int(
                    self._counter(scope=resolved_scope, name="duplicate_code_claims")
                ),
                "effective_contract_claim_attempts": int(
                    self._counter(
                        scope=resolved_scope,
                        name="effective_contract_claim_attempts",
                    )
                ),
                "unique_effective_contracts": self._count(
                    "effective_contract_claims",
                    scope=resolved_scope,
                    distinct="contract_hash",
                ),
                "duplicate_effective_contract_claims": int(
                    self._counter(
                        scope=resolved_scope,
                        name="duplicate_effective_contract_claims",
                    )
                ),
                "sequence_occurrence_attempts": int(
                    self._counter(
                        scope=resolved_scope, name="sequence_occurrence_attempts"
                    )
                ),
                "sequence_occurrences_recorded": self._count(
                    "sequence_occurrences", scope=resolved_scope
                ),
                "unique_sequences": (
                    self._count(
                        "sequence_identities",
                        scope=None,
                        distinct="sequence_bundle_hash",
                    )
                    if resolved_scope is None
                    else int(
                        self._counter(
                            scope=resolved_scope,
                            name="unique_sequences",
                        )
                    )
                ),
                "evaluation_claim_attempts": int(
                    self._counter(
                        scope=resolved_scope, name="evaluation_claim_attempts"
                    )
                ),
                "evaluation_executions_claimed": int(
                    self._counter(
                        scope=resolved_scope,
                        name="evaluation_executions_claimed",
                    )
                ),
                "duplicate_expensive_eval_avoided": int(
                    self._counter(
                        scope=resolved_scope,
                        name="duplicate_expensive_eval_avoided",
                    )
                ),
                "evaluation_cache_hits": int(
                    self._counter(
                        scope=resolved_scope, name="evaluation_cache_hits"
                    )
                ),
                "reclaimed_leases": int(
                    self._counter(scope=resolved_scope, name="reclaimed_leases")
                ),
                "estimated_cache_cost_saved": self._counter(
                    scope=resolved_scope, name="estimated_cache_cost_saved"
                ),
                "actual_evaluation_cost": self._counter(
                    scope=resolved_scope, name="actual_evaluation_cost"
                ),
            }
        return RegistryMetrics.create(
            scope=resolved_scope if resolved_scope is not None else "*", **values
        )

    def backup_to(
        self, destination: str | os.PathLike[str], *, overwrite: bool = False
    ) -> Path:
        target = Path(destination).expanduser().resolve()
        if self.path != ":memory:" and target == Path(self.path).expanduser().resolve():
            _fail("backup_destination_is_source")
        if target.exists() and not overwrite:
            _fail("backup_destination_exists", str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        self._require_open()
        destination_connection = sqlite3.connect(str(target), isolation_level=None)
        try:
            with self._lock:
                self._connection.backup(destination_connection)
        finally:
            destination_connection.close()
        return target

    @classmethod
    def restore_from_backup(
        cls,
        backup: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool = False,
    ) -> Path:
        source = Path(backup).expanduser().resolve()
        target = Path(destination).expanduser().resolve()
        if not source.is_file():
            _fail("backup_source_missing", str(source))
        if source == target:
            _fail("restore_destination_is_source")
        if target.exists() and not overwrite:
            _fail("restore_destination_exists", str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        source_connection = sqlite3.connect(str(source), isolation_level=None)
        destination_connection = sqlite3.connect(str(target), isolation_level=None)
        try:
            row = source_connection.execute(
                "SELECT value FROM registry_metadata WHERE key='schema_version'"
            ).fetchone()
            if row is None or row[0] != REGISTRY_SCHEMA_VERSION:
                _fail("backup_schema_invalid", str(row[0] if row else None))
            source_connection.backup(destination_connection)
        finally:
            source_connection.close()
            destination_connection.close()


        restored = cls(target)
        restored.close()
        return target


__all__ = [
    "EVALUATION_CACHE_ENTRY_VERSION",
    "LEASE_CLAIM_VERSION",
    "REGISTRY_METRICS_VERSION",
    "REGISTRY_SCHEMA_VERSION",
    "SEQUENCE_OCCURRENCE_VERSION",
    "EvaluationCacheEntry",
    "ExperimentRegistry",
    "LeaseClaim",
    "RegistryContractError",
    "RegistryLeaseError",
    "RegistryMetrics",
    "SequenceOccurrenceResult",
]
