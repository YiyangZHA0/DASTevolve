

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
import threading
from typing import Any, Iterable, Mapping, Optional, Tuple

from .archive import OBJECTIVE_DIRECTIONS
from .domain import (
    EVALUATION_FAILED,
    EVALUATION_SUCCEEDED,
    GenerationCommit,
    SealedEvaluation,
)


EVOLUTION_FACT_VERSION = "astevolve.evolution.fact.v3"
PROMPT_MEMORY_VERSION = "astevolve.evolution.prompt_memory.v3"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_LABEL = re.compile(r"[^a-zA-Z0-9_.:@+-]+")
_FEEDBACK_TERM_LIMIT = 16
_FEEDBACK_TARGET_LIMIT = 8
_FEEDBACK_EVIDENCE_LIMIT = 16
_TARGET_FIELDS = frozenset(
    {
        "action",
        "chain",
        "chain_id",
        "category",
        "node",
        "node_id",
        "position",
        "positions",
        "priority",
        "reason",
        "residue",
        "residues",
        "score",
        "state",
        "type",
    }
)


class MemoryProjectionError(ValueError):
    pass


class MemoryProjectionConflict(MemoryProjectionError):
    pass


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise MemoryProjectionError(f"{name} must be a non-empty trimmed string")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MemoryProjectionError(f"{name} must be a non-negative integer")
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MemoryProjectionError("memory projection requires finite JSON") from exc


def _digest(namespace: str, value: Any) -> str:
    return hashlib.sha256(
        f"{namespace}\0{_canonical_json(value)}".encode("utf-8")
    ).hexdigest()


def _finite_optional(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MemoryProjectionError(f"{name} must be a finite number or null")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise MemoryProjectionError(f"{name} must be a finite number or null")
    return resolved


def _objective_path(value: Any) -> Tuple[str, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise MemoryProjectionError(
            "objective_path must be a non-empty tuple of strings"
        )
    return value


def _objective_direction(value: Any) -> str:
    if not isinstance(value, str) or value not in OBJECTIVE_DIRECTIONS:
        raise MemoryProjectionError(
            "objective_direction must be 'maximize' or 'minimize'"
        )
    return str(value)


def _path_value(root: Mapping[str, Any], path: Tuple[str, ...]) -> Any:
    current: Any = root
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            raise MemoryProjectionError(
                "report field is missing: " + ".".join(path)
            )
        current = current[component]
    return current


def _objective_rank_value(value: float, direction: str) -> float:
    return -value if direction == "maximize" else value


def _safe_label(value: Any, *, fallback: str = "unspecified") -> str:


    raw = str(value or "").strip()[:256]
    lowered = raw.lower()
    if any(
        marker in lowered
        for marker in (
            "api_key",
            "authorization",
            "bearer ",
            "credential",
            "password",
            "private_key",
            "secret",
            "token",
            "://",
            "/",
            "\\",
            "=",
        )
    ):
        return (
            "redacted_" + _digest("astevolve.evolution.diagnostic_label.v1", raw)[:16]
        )
    normalized = _SAFE_LABEL.sub("_", raw).strip("_.")[:80]
    return normalized or fallback


def _failure_projection(reason: str) -> tuple[str, str]:
    if not reason:
        return "", ""
    category = _safe_label(reason.split(":", 1)[0].split(None, 1)[0])
    return category, _digest("astevolve.evolution.failure_text.v1", reason)


def _safe_target_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, str):
        return _safe_label(value)
    if isinstance(value, (list, tuple)):
        return [_safe_target_value(item) for item in value[:32]]
    return None


def _feedback_projection(evaluation: SealedEvaluation) -> dict[str, Any]:


    if evaluation.status != EVALUATION_SUCCEEDED:
        category, failure_hash = _failure_projection(evaluation.failure_reason)
        return {
            "outcome_category": category or "semantic_failure",
            "failure_hash": failure_hash,
            "soft_energy": None,
            "total_energy": None,
            "energy_terms": [],
            "gate_reasons": [],
            "score_terms": [],
            "recommended_edits": [],
            "evidence_kinds": [],
        }
    report = evaluation.report()
    if report is None:
        raise MemoryProjectionError("successful evaluation has no report")
    terms = [
        {
            "name": _safe_label(term.name),
            "category": _safe_label(term.category),
            "score": _finite_optional(term.score, "score term"),
            "weight": _finite_optional(term.weight, "score weight"),
            "available": bool(term.available),
            "provider": _safe_label(term.provider),
            "state": (_safe_label(term.state) if term.state is not None else None),
        }
        for term in report.terms[:_FEEDBACK_TERM_LIMIT]
    ]
    energy_terms = []
    for raw in report.term_energy_breakdown[:_FEEDBACK_TERM_LIMIT]:
        if not isinstance(raw, Mapping):
            continue
        energy_terms.append(
            {
                "name": _safe_label(raw.get("name")),
                "category": _safe_label(raw.get("category")),
                "provider": _safe_label(raw.get("provider")),
                "state": (
                    _safe_label(raw.get("state"))
                    if raw.get("state") is not None
                    else None
                ),
                "available": bool(raw.get("available", False)),
                "required": bool(raw.get("required", False)),
                "included": bool(raw.get("included", False)),
                "cost": _finite_optional(raw.get("cost"), "energy term cost"),
                "weight": _finite_optional(
                    raw.get("weight"), "energy term weight"
                ),
                "weighted_cost": _finite_optional(
                    raw.get("weighted_cost"), "energy term weighted cost"
                ),
            }
        )
    targets = []
    for raw in report.recommended_edit_targets[:_FEEDBACK_TARGET_LIMIT]:
        target = {
            str(key): _safe_target_value(value)
            for key, value in raw.items()
            if isinstance(key, str) and key in _TARGET_FIELDS
        }
        if target:
            targets.append(target)
    evidence_kinds = []
    for record in evaluation.evidence().records[:_FEEDBACK_EVIDENCE_LIMIT]:
        evidence_kinds.append(
            {
                "source": _safe_label(record.source),
                "kind": _safe_label(record.kind),
                "available": bool(record.available),
            }
        )
    return {
        "outcome_category": (
            "feasible_success" if report.gate.passed else "infeasible_success"
        ),
        "failure_hash": "",
        "soft_energy": _finite_optional(report.soft_energy, "soft_energy"),
        "total_energy": _finite_optional(report.total_energy, "total_energy"),
        "energy_terms": energy_terms,
        "gate_reasons": [
            _safe_label(reason)
            for reason in report.gate.failures[:_FEEDBACK_TERM_LIMIT]
        ],
        "score_terms": terms,
        "recommended_edits": targets,
        "evidence_kinds": evidence_kinds,
        "truncated": {
            "score_terms": max(0, len(report.terms) - len(terms)),
            "energy_terms": max(
                0, len(report.term_energy_breakdown) - len(energy_terms)
            ),
            "recommended_edits": max(
                0, len(report.recommended_edit_targets) - len(targets)
            ),
            "evidence_kinds": max(
                0, len(evaluation.evidence().records) - len(evidence_kinds)
            ),
        },
    }


@dataclass(frozen=True)
class MemoryProjectionConfig:
    scope_id: str
    recent_limit: int = 20
    best_limit: int = 5
    objective_path: Tuple[str, ...] = ("normalized_score",)
    objective_direction: str = "maximize"

    def __post_init__(self) -> None:
        _required_text(self.scope_id, "scope_id")
        _non_negative_int(self.recent_limit, "recent_limit")
        _non_negative_int(self.best_limit, "best_limit")
        _objective_path(self.objective_path)
        _objective_direction(self.objective_direction)


@dataclass(frozen=True)
class EvolutionFact:


    scope_id: str
    occurrence_index: int
    generation_id: str
    commit_hash: str
    proposal_id: str
    evaluation_hash: str
    status: str
    candidate_id: str
    objective: Optional[float]
    objective_path: Tuple[str, ...]
    objective_direction: str
    feasible: Optional[bool]
    outcome_category: str
    failure_reason: str
    failure_hash: str
    feedback_json: str
    feedback_hash: str
    report_hash: str
    evidence_hash: str
    fact_hash: str
    schema_version: str = EVOLUTION_FACT_VERSION

    @classmethod
    def create(
        cls,
        *,
        scope_id: str,
        occurrence_index: int,
        commit: GenerationCommit,
        evaluation: SealedEvaluation,
        objective_path: Tuple[str, ...] = ("normalized_score",),
        objective_direction: str = "maximize",
    ) -> "EvolutionFact":
        commit.to_dict()
        if evaluation not in commit.evaluations:
            raise MemoryProjectionError("evaluation is not part of the commit")
        return cls._create_from_verified_commit_member(
            scope_id=scope_id,
            occurrence_index=occurrence_index,
            commit=commit,
            evaluation=evaluation,
            objective_path=objective_path,
            objective_direction=objective_direction,
        )

    @classmethod
    def _create_from_verified_commit_member(
        cls,
        *,
        scope_id: str,
        occurrence_index: int,
        commit: GenerationCommit,
        evaluation: SealedEvaluation,
        objective_path: Tuple[str, ...],
        objective_direction: str,
    ) -> "EvolutionFact":


        resolved_objective_path = _objective_path(objective_path)
        resolved_objective_direction = _objective_direction(objective_direction)
        objective: Optional[float] = None
        feasible: Optional[bool] = None
        if evaluation.status == EVALUATION_SUCCEEDED:
            report = evaluation.report()
            if report is None:
                raise MemoryProjectionError("successful evaluation has no report")
            objective = _finite_optional(
                _path_value(report.to_legacy_dict(), resolved_objective_path),
                "objective",
            )
            if not isinstance(report.gate.passed, bool):
                raise MemoryProjectionError("feasible must be boolean")
            feasible = report.gate.passed
        elif evaluation.status != EVALUATION_FAILED:
            raise MemoryProjectionError("fact status must be terminal")
        feedback = _feedback_projection(evaluation)
        feedback_json = _canonical_json(feedback)
        outcome_category = _safe_label(feedback["outcome_category"])
        failure_reason, failure_hash = _failure_projection(evaluation.failure_reason)
        resolved_scope = _required_text(scope_id, "scope_id")
        resolved_occurrence = _non_negative_int(occurrence_index, "occurrence_index")
        core: dict[str, Any] = {
            "schema_version": EVOLUTION_FACT_VERSION,
            "scope_id": resolved_scope,
            "occurrence_index": resolved_occurrence,
            "generation_id": evaluation.generation_id,
            "commit_hash": commit.commit_hash,
            "proposal_id": evaluation.proposal_id,
            "evaluation_hash": evaluation.evaluation_hash,
            "status": evaluation.status,
            "candidate_id": evaluation.candidate_id,
            "objective": objective,
            "objective_path": list(resolved_objective_path),
            "objective_direction": resolved_objective_direction,
            "feasible": feasible,
            "outcome_category": outcome_category,


            "failure_reason": failure_reason,
            "failure_hash": failure_hash,
            "feedback_hash": _digest("astevolve.evolution.feedback.v1", feedback),
            "report_hash": evaluation.report_hash,
            "evidence_hash": evaluation.evidence_hash,
        }
        return cls(
            scope_id=resolved_scope,
            occurrence_index=resolved_occurrence,
            generation_id=evaluation.generation_id,
            commit_hash=commit.commit_hash,
            proposal_id=evaluation.proposal_id,
            evaluation_hash=evaluation.evaluation_hash,
            status=evaluation.status,
            candidate_id=evaluation.candidate_id,
            objective=objective,
            objective_path=resolved_objective_path,
            objective_direction=resolved_objective_direction,
            feasible=feasible,
            outcome_category=outcome_category,
            failure_reason=failure_reason,
            failure_hash=failure_hash,
            feedback_json=feedback_json,
            feedback_hash=core["feedback_hash"],
            report_hash=evaluation.report_hash,
            evidence_hash=evaluation.evidence_hash,
            fact_hash=_digest(EVOLUTION_FACT_VERSION, core),
        )

    def verify(self) -> None:
        if self.schema_version != EVOLUTION_FACT_VERSION:
            raise MemoryProjectionError("unsupported evolution fact schema")
        _required_text(self.scope_id, "scope_id")
        _non_negative_int(self.occurrence_index, "occurrence_index")
        _required_text(self.generation_id, "generation_id")
        _required_text(self.commit_hash, "commit_hash")
        _required_text(self.proposal_id, "proposal_id")
        _required_text(self.evaluation_hash, "evaluation_hash")
        if self.status not in {EVALUATION_SUCCEEDED, EVALUATION_FAILED}:
            raise MemoryProjectionError("fact status must be terminal")
        if not isinstance(self.candidate_id, str) or not isinstance(
            self.failure_reason, str
        ):
            raise MemoryProjectionError("fact text fields are invalid")
        _required_text(self.outcome_category, "outcome_category")
        _objective_path(self.objective_path)
        _objective_direction(self.objective_direction)
        try:
            feedback = json.loads(self.feedback_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise MemoryProjectionError("fact feedback is invalid JSON") from exc
        if (
            not isinstance(feedback, dict)
            or _canonical_json(feedback) != self.feedback_json
        ):
            raise MemoryProjectionError("fact feedback is not a canonical mapping")
        expected_feedback_hash = _digest("astevolve.evolution.feedback.v1", feedback)
        if self.feedback_hash != expected_feedback_hash:
            raise MemoryProjectionError("fact feedback hash mismatch")
        objective = _finite_optional(self.objective, "objective")
        if self.status == EVALUATION_SUCCEEDED:
            if objective is None or not isinstance(self.feasible, bool):
                raise MemoryProjectionError(
                    "successful fact requires objective and feasible"
                )
            if self.failure_reason:
                raise MemoryProjectionError("successful fact cannot have a failure")
            if self.failure_hash:
                raise MemoryProjectionError(
                    "successful fact cannot have a failure hash"
                )
        elif objective is not None or self.feasible is not None:
            raise MemoryProjectionError(
                "failed fact cannot contain objective or feasible"
            )
        elif not _SHA256.fullmatch(self.failure_hash):
            raise MemoryProjectionError("failed fact requires a failure hash")
        expected = _digest(EVOLUTION_FACT_VERSION, self._core())
        if self.fact_hash != expected:
            raise MemoryProjectionError("evolution fact hash mismatch")

    def _core(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope_id": self.scope_id,
            "occurrence_index": self.occurrence_index,
            "generation_id": self.generation_id,
            "commit_hash": self.commit_hash,
            "proposal_id": self.proposal_id,
            "evaluation_hash": self.evaluation_hash,
            "status": self.status,
            "candidate_id": self.candidate_id,
            "objective": self.objective,
            "objective_path": list(self.objective_path),
            "objective_direction": self.objective_direction,
            "feasible": self.feasible,
            "outcome_category": self.outcome_category,
            "failure_reason": self.failure_reason,
            "failure_hash": self.failure_hash,
            "feedback_hash": self.feedback_hash,
            "report_hash": self.report_hash,
            "evidence_hash": self.evidence_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        self.verify()
        feedback = json.loads(self.feedback_json)
        return {**self._core(), "feedback": feedback, "fact_hash": self.fact_hash}


@dataclass(frozen=True)
class PromptMemorySnapshot:
    scope_id: str
    ledger_hash: str
    objective_path: Tuple[str, ...]
    objective_direction: str
    applied_commits: Tuple[Tuple[str, str], ...]
    recent: Tuple[EvolutionFact, ...]
    best_feasible: Tuple[EvolutionFact, ...]
    failure_counts: Tuple[Tuple[str, int], ...]
    outcome_category_counts: Tuple[Tuple[str, int], ...]
    total_occurrences: int
    succeeded: int
    failed: int
    snapshot_hash: str
    schema_version: str = PROMPT_MEMORY_VERSION

    def _semantic(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope_id": self.scope_id,
            "ledger_hash": self.ledger_hash,
            "objective_path": list(self.objective_path),
            "objective_direction": self.objective_direction,
            "applied_commits": [
                {"generation_id": generation_id, "commit_hash": commit_hash}
                for generation_id, commit_hash in self.applied_commits
            ],
            "recent": [fact.to_dict() for fact in self.recent],
            "best_feasible": [fact.to_dict() for fact in self.best_feasible],
            "failure_counts": {reason: count for reason, count in self.failure_counts},
            "outcome_category_counts": {
                category: count for category, count in self.outcome_category_counts
            },
            "total_occurrences": self.total_occurrences,
            "succeeded": self.succeeded,
            "failed": self.failed,
        }

    def to_dict(self) -> dict[str, Any]:
        semantic = self._semantic()
        expected = _digest(PROMPT_MEMORY_VERSION, semantic)
        if expected != self.snapshot_hash:
            raise MemoryProjectionError("prompt memory snapshot hash mismatch")
        return {**semantic, "snapshot_hash": self.snapshot_hash}

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


class EvolutionMemoryProjection:


    def __init__(self, config: MemoryProjectionConfig) -> None:
        if not isinstance(config, MemoryProjectionConfig):
            raise TypeError("config must be MemoryProjectionConfig")
        self._config = config
        self._facts: list[EvolutionFact] = []
        self._best_by_candidate: dict[str, EvolutionFact] = {}
        self._failure_counts: dict[str, int] = {}
        self._outcome_category_counts: dict[str, int] = {}
        self._succeeded = 0
        self._failed = 0
        self._applied: dict[str, str] = {}
        self._applied_order: list[tuple[str, str]] = []
        self._ledger_hash = _digest(
            "astevolve.evolution.fact_ledger.empty.v2",
            {
                "scope_id": config.scope_id,
                "objective_path": list(config.objective_path),
                "objective_direction": config.objective_direction,
            },
        )
        self._lock = threading.RLock()

    @property
    def config(self) -> MemoryProjectionConfig:
        return self._config

    def apply(self, commit: GenerationCommit) -> PromptMemorySnapshot:
        if not isinstance(commit, GenerationCommit):
            raise TypeError("memory projection accepts only GenerationCommit")


        commit.to_dict()
        with self._lock:
            existing = self._applied.get(commit.generation_id)
            if existing is not None:
                if existing != commit.commit_hash:
                    raise MemoryProjectionConflict(
                        f"generation {commit.generation_id!r} has a conflicting commit"
                    )
                return self.snapshot()

            offset = len(self._facts)


            facts = [
                EvolutionFact._create_from_verified_commit_member(
                    scope_id=self._config.scope_id,
                    occurrence_index=offset + index,
                    commit=commit,
                    evaluation=evaluation,
                    objective_path=self._config.objective_path,
                    objective_direction=self._config.objective_direction,
                )
                for index, evaluation in enumerate(commit.evaluations)
            ]
            next_ledger_hash = _digest(
                "astevolve.evolution.fact_ledger_step.v1",
                {
                    "previous_ledger_hash": self._ledger_hash,
                    "generation_id": commit.generation_id,
                    "commit_hash": commit.commit_hash,
                    "fact_hashes": [fact.fact_hash for fact in facts],
                },
            )
            best_by_candidate = dict(self._best_by_candidate)
            failure_counts = dict(self._failure_counts)
            outcome_category_counts = dict(self._outcome_category_counts)
            succeeded = self._succeeded
            failed = self._failed
            for fact in facts:
                outcome_category_counts[fact.outcome_category] = (
                    outcome_category_counts.get(fact.outcome_category, 0) + 1
                )
                if fact.status == EVALUATION_SUCCEEDED:
                    succeeded += 1
                    if not fact.feasible:
                        continue
                    if fact.objective is None:
                        raise MemoryProjectionError(
                            "successful feasible fact has no objective"
                        )
                    existing_fact = best_by_candidate.get(fact.candidate_id)
                    existing_objective = (
                        None if existing_fact is None else existing_fact.objective
                    )
                    if (
                        existing_fact is None
                        or existing_objective is None
                        or (
                            _objective_rank_value(
                                fact.objective,
                                self._config.objective_direction,
                            ),
                            fact.occurrence_index,
                        )
                        < (
                            _objective_rank_value(
                                existing_objective,
                                self._config.objective_direction,
                            ),
                            existing_fact.occurrence_index,
                        )
                    ):
                        best_by_candidate[fact.candidate_id] = fact
                else:
                    failed += 1
                    reason = fact.failure_reason or "unspecified"
                    failure_counts[reason] = failure_counts.get(reason, 0) + 1
            self._facts.extend(facts)
            self._best_by_candidate = best_by_candidate
            self._failure_counts = failure_counts
            self._outcome_category_counts = outcome_category_counts
            self._succeeded = succeeded
            self._failed = failed
            self._applied[commit.generation_id] = commit.commit_hash
            self._applied_order.append((commit.generation_id, commit.commit_hash))
            self._ledger_hash = next_ledger_hash
            return self.snapshot()

    def snapshot(self) -> PromptMemorySnapshot:
        with self._lock:
            recent = tuple(
                self._facts[-self._config.recent_limit :]
                if self._config.recent_limit
                else ()
            )

            def best_key(fact: EvolutionFact) -> tuple[float, str, int]:
                if fact.objective is None:
                    raise MemoryProjectionError(
                        "successful feasible fact has no objective"
                    )
                return (
                    _objective_rank_value(
                        fact.objective, self._config.objective_direction
                    ),
                    fact.candidate_id,
                    fact.occurrence_index,
                )

            ranked = sorted(
                self._best_by_candidate.values(),
                key=best_key,
            )
            best = tuple(ranked[: self._config.best_limit])
            semantic = {
                "schema_version": PROMPT_MEMORY_VERSION,
                "scope_id": self._config.scope_id,
                "ledger_hash": self._ledger_hash,
                "objective_path": list(self._config.objective_path),
                "objective_direction": self._config.objective_direction,
                "applied_commits": [
                    {"generation_id": generation_id, "commit_hash": commit_hash}
                    for generation_id, commit_hash in self._applied_order
                ],
                "recent": [fact.to_dict() for fact in recent],
                "best_feasible": [fact.to_dict() for fact in best],
                "failure_counts": dict(sorted(self._failure_counts.items())),
                "outcome_category_counts": dict(
                    sorted(self._outcome_category_counts.items())
                ),
                "total_occurrences": len(self._facts),
                "succeeded": self._succeeded,
                "failed": self._failed,
            }
            return PromptMemorySnapshot(
                scope_id=self._config.scope_id,
                ledger_hash=self._ledger_hash,
                objective_path=self._config.objective_path,
                objective_direction=self._config.objective_direction,
                applied_commits=tuple(self._applied_order),
                recent=recent,
                best_feasible=best,
                failure_counts=tuple(sorted(self._failure_counts.items())),
                outcome_category_counts=tuple(
                    sorted(self._outcome_category_counts.items())
                ),
                total_occurrences=len(self._facts),
                succeeded=self._succeeded,
                failed=self._failed,
                snapshot_hash=_digest(PROMPT_MEMORY_VERSION, semantic),
            )

    @classmethod
    def rebuild(
        cls,
        config: MemoryProjectionConfig,
        commits: Iterable[GenerationCommit],
    ) -> "EvolutionMemoryProjection":
        projection = cls(config)
        for commit in commits:
            projection.apply(commit)
        return projection


__all__ = [
    "EVOLUTION_FACT_VERSION",
    "PROMPT_MEMORY_VERSION",
    "EvolutionFact",
    "EvolutionMemoryProjection",
    "MemoryProjectionConfig",
    "MemoryProjectionConflict",
    "MemoryProjectionError",
    "PromptMemorySnapshot",
]
