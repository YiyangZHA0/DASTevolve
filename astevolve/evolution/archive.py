

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import threading
from typing import Any, Mapping, Sequence, Tuple

from .domain import (
    EVALUATION_FAILED,
    EVALUATION_SUCCEEDED,
    GenerationCommit,
    SealedEvaluation,
)


class ArchiveProjectionError(ValueError):
    pass


class ArchiveProjectionConflict(ArchiveProjectionError):
    pass


OBJECTIVE_DIRECTIONS = frozenset({"maximize", "minimize"})


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArchiveProjectionError(f"{name} must be non-empty")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArchiveProjectionError(f"{name} must be a positive integer")
    return value


def _finite_number(value: Any, name: str) -> float:


    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArchiveProjectionError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ArchiveProjectionError(f"{name} must be a finite number")
    return result


def _path_value(root: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = root
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            raise ArchiveProjectionError("report field is missing: " + ".".join(path))
        current = current[component]
    return current


@dataclass(frozen=True)
class FeatureDimension:


    name: str
    report_path: Tuple[str, ...]
    minimum: float
    maximum: float
    bins: int

    def __post_init__(self) -> None:
        _required_text(self.name, "feature name")
        if (
            not isinstance(self.report_path, tuple)
            or not self.report_path
            or any(not isinstance(item, str) or not item for item in self.report_path)
        ):
            raise ArchiveProjectionError(
                "feature report_path must be a non-empty tuple of strings"
            )
        minimum = _finite_number(self.minimum, f"{self.name} minimum")
        maximum = _finite_number(self.maximum, f"{self.name} maximum")
        if minimum >= maximum:
            raise ArchiveProjectionError(
                f"{self.name} minimum must be smaller than maximum"
            )
        _positive_int(self.bins, f"{self.name} bins")

    def locate(self, report: Mapping[str, Any]) -> tuple[float, int]:
        value = _finite_number(_path_value(report, self.report_path), self.name)
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        clipped = min(max(value, minimum), maximum)
        if clipped == maximum:
            cell = self.bins - 1
        else:
            cell = int(((clipped - minimum) / (maximum - minimum)) * self.bins)
        return value, cell


@dataclass(frozen=True)
class ArchiveProjectionConfig:


    feature_dimensions: Tuple[FeatureDimension, ...]
    objective_path: Tuple[str, ...] = ("normalized_score",)
    objective_direction: str = "maximize"
    num_islands: int = 1
    migration_interval: int = 10
    migrants_per_island: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.feature_dimensions, tuple)
            or not self.feature_dimensions
            or any(
                not isinstance(item, FeatureDimension)
                for item in self.feature_dimensions
            )
        ):
            raise ArchiveProjectionError(
                "feature_dimensions must be a non-empty FeatureDimension tuple"
            )
        names = tuple(item.name for item in self.feature_dimensions)
        if len(names) != len(set(names)):
            raise ArchiveProjectionError("feature dimension names must be unique")
        if (
            not isinstance(self.objective_path, tuple)
            or not self.objective_path
            or any(
                not isinstance(item, str) or not item for item in self.objective_path
            )
        ):
            raise ArchiveProjectionError(
                "objective_path must be a non-empty tuple of strings"
            )
        if (
            not isinstance(self.objective_direction, str)
            or self.objective_direction not in OBJECTIVE_DIRECTIONS
        ):
            raise ArchiveProjectionError(
                "objective_direction must be 'maximize' or 'minimize'"
            )
        _positive_int(self.num_islands, "num_islands")
        _positive_int(self.migration_interval, "migration_interval")
        _positive_int(self.migrants_per_island, "migrants_per_island")


@dataclass(frozen=True)
class ArchivedCandidate:


    candidate_id: str
    generation_id: str
    proposal_id: str
    evaluation_hash: str
    objective: float
    feasible: bool
    features: Tuple[float, ...]
    cell: Tuple[int, ...]
    primary_island: int


@dataclass(frozen=True)
class ArchiveCell:
    coordinates: Tuple[int, ...]
    candidate: ArchivedCandidate


@dataclass(frozen=True)
class ProjectionRejection:
    generation_id: str
    proposal_id: str
    candidate_id: str
    reason: str


@dataclass(frozen=True)
class ArchiveSnapshot:


    revision: int
    applied_commits: Tuple[Tuple[str, str], ...]
    cells: Tuple[ArchiveCell, ...]
    islands: Tuple[Tuple[str, ...], ...]
    tombstones: Tuple[str, ...]
    rejections: Tuple[ProjectionRejection, ...]
    migrations_completed: int

    @property
    def candidate_ids(self) -> Tuple[str, ...]:
        return tuple(cell.candidate.candidate_id for cell in self.cells)


def deterministic_island(candidate_id: str, num_islands: int) -> int:


    resolved_id = _required_text(candidate_id, "candidate_id")
    count = _positive_int(num_islands, "num_islands")
    digest = hashlib.sha256(
        f"astevolve.evolution.primary_island.v1\0{resolved_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % count


def _candidate_order(
    candidate: ArchivedCandidate, objective_direction: str
) -> tuple[int, float, str, str, str]:


    primary = (
        -candidate.objective
        if objective_direction == "maximize"
        else candidate.objective
    )
    return (
        -int(candidate.feasible),
        primary,
        candidate.candidate_id,
        candidate.proposal_id,
        candidate.evaluation_hash,
    )


def _challenger_wins(
    challenger: ArchivedCandidate,
    incumbent: ArchivedCandidate,
    objective_direction: str,
) -> bool:
    return _candidate_order(challenger, objective_direction) < _candidate_order(
        incumbent, objective_direction
    )


class ArchiveProjection:


    def __init__(self, config: ArchiveProjectionConfig) -> None:
        if not isinstance(config, ArchiveProjectionConfig):
            raise TypeError("config must be ArchiveProjectionConfig")
        self._config = config
        self._applied: dict[str, str] = {}
        self._applied_order: list[tuple[str, str]] = []
        self._cells: dict[tuple[int, ...], ArchivedCandidate] = {}
        self._islands: list[set[str]] = [set() for _ in range(config.num_islands)]
        self._seen_candidates: dict[str, str] = {}
        self._tombstones: set[str] = set()
        self._rejections: list[ProjectionRejection] = []
        self._migrations_completed = 0
        self._lock = threading.RLock()

    @property
    def config(self) -> ArchiveProjectionConfig:
        return self._config

    def candidate_order_key(
        self, candidate: ArchivedCandidate
    ) -> tuple[int, float, str, str, str]:


        if not isinstance(candidate, ArchivedCandidate):
            raise TypeError("candidate must be ArchivedCandidate")
        return _candidate_order(candidate, self._config.objective_direction)

    def snapshot(self) -> ArchiveSnapshot:
        with self._lock:
            cells = tuple(
                ArchiveCell(coordinates=coordinates, candidate=candidate)
                for coordinates, candidate in sorted(self._cells.items())
            )
            return ArchiveSnapshot(
                revision=len(self._applied_order),
                applied_commits=tuple(self._applied_order),
                cells=cells,
                islands=tuple(tuple(sorted(values)) for values in self._islands),
                tombstones=tuple(sorted(self._tombstones)),
                rejections=tuple(self._rejections),
                migrations_completed=self._migrations_completed,
            )

    def apply(self, commit: GenerationCommit) -> ArchiveSnapshot:
        if not isinstance(commit, GenerationCommit):
            raise TypeError(
                "archive projection accepts only a complete GenerationCommit"
            )
        with self._lock:
            return self._apply_locked(commit)

    def _apply_locked(self, commit: GenerationCommit) -> ArchiveSnapshot:


        commit.to_dict()

        existing_commit = self._applied.get(commit.generation_id)
        if existing_commit is not None:
            if existing_commit != commit.commit_hash:
                raise ArchiveProjectionConflict(
                    f"generation {commit.generation_id!r} has a conflicting commit"
                )
            return self.snapshot()

        evaluations = tuple(commit.evaluations)
        for evaluation in evaluations:
            if not evaluation.candidate_id:
                continue
            if (
                evaluation.status == EVALUATION_SUCCEEDED
                and evaluation.candidate_id in self._tombstones
            ):
                raise ArchiveProjectionConflict(
                    f"candidate {evaluation.candidate_id!r} is tombstoned"
                )


        cells = dict(self._cells)
        islands = [set(values) for values in self._islands]
        seen_candidates = dict(self._seen_candidates)
        tombstones = set(self._tombstones)
        rejections = list(self._rejections)

        grouped: dict[str, list[ArchivedCandidate]] = {}
        for evaluation in evaluations:
            candidate_id = evaluation.candidate_id
            if candidate_id:
                seen_candidates[candidate_id] = evaluation.evaluation_hash
            if evaluation.status == EVALUATION_FAILED:
                if candidate_id:
                    grouped.setdefault(candidate_id, [])
                continue
            if evaluation.status != EVALUATION_SUCCEEDED:


                raise ArchiveProjectionError("commit contains a non-terminal status")

            assert candidate_id
            try:
                candidate = self._candidate_from_evaluation(evaluation)
            except ArchiveProjectionError as exc:
                grouped.setdefault(candidate_id, [])
                rejections.append(
                    ProjectionRejection(
                        generation_id=evaluation.generation_id,
                        proposal_id=evaluation.proposal_id,
                        candidate_id=candidate_id,
                        reason=str(exc),
                    )
                )
                continue
            grouped.setdefault(candidate_id, []).append(candidate)

        for candidate_id, occurrences in grouped.items():


            if not occurrences:
                self._retire(candidate_id, cells, islands)
                tombstones.add(candidate_id)
                continue
            candidate = min(occurrences, key=self.candidate_order_key)
            seen_candidates[candidate_id] = candidate.evaluation_hash


            existing_occurrence = next(
                (
                    archived
                    for archived in cells.values()
                    if archived.candidate_id == candidate.candidate_id
                ),
                None,
            )
            if existing_occurrence is not None:
                if not _challenger_wins(
                    candidate,
                    existing_occurrence,
                    self._config.objective_direction,
                ):
                    continue
                self._retire(candidate.candidate_id, cells, islands)

            incumbent = cells.get(candidate.cell)
            if incumbent is not None and not _challenger_wins(
                candidate,
                incumbent,
                self._config.objective_direction,
            ):
                continue
            if incumbent is not None:
                self._retire(incumbent.candidate_id, cells, islands)
            cells[candidate.cell] = candidate
            islands[candidate.primary_island].add(candidate.candidate_id)

        next_revision = len(self._applied_order) + 1
        migrations_completed = self._migrations_completed
        if (
            self._config.num_islands > 1
            and next_revision % self._config.migration_interval == 0
        ):
            self._migrate(cells, islands)
            migrations_completed += 1


        self._cells = cells
        self._islands = islands
        self._seen_candidates = seen_candidates
        self._tombstones = tombstones
        self._rejections = rejections
        self._migrations_completed = migrations_completed
        self._applied[commit.generation_id] = commit.commit_hash
        self._applied_order.append((commit.generation_id, commit.commit_hash))
        return self.snapshot()

    def _candidate_from_evaluation(
        self, evaluation: SealedEvaluation
    ) -> ArchivedCandidate:
        payload = evaluation.to_dict()
        report = payload.get("report")
        if not isinstance(report, Mapping):
            raise ArchiveProjectionError("successful evaluation report is missing")
        objective = _finite_number(
            _path_value(report, self._config.objective_path), "objective"
        )
        gate_value = report.get("hard_gate_pass")
        if not isinstance(gate_value, bool):
            raise ArchiveProjectionError("hard_gate_pass must be boolean")
        located = tuple(
            dimension.locate(report) for dimension in self._config.feature_dimensions
        )
        features = tuple(item[0] for item in located)
        cell = tuple(item[1] for item in located)
        return ArchivedCandidate(
            candidate_id=evaluation.candidate_id,
            generation_id=evaluation.generation_id,
            proposal_id=evaluation.proposal_id,
            evaluation_hash=evaluation.evaluation_hash,
            objective=objective,
            feasible=gate_value,
            features=features,
            cell=cell,
            primary_island=deterministic_island(
                evaluation.candidate_id, self._config.num_islands
            ),
        )

    @staticmethod
    def _retire(
        candidate_id: str,
        cells: dict[tuple[int, ...], ArchivedCandidate],
        islands: list[set[str]],
    ) -> None:
        for coordinates, candidate in tuple(cells.items()):
            if candidate.candidate_id == candidate_id:
                del cells[coordinates]
        for members in islands:
            members.discard(candidate_id)

    def _migrate(
        self,
        cells: Mapping[tuple[int, ...], ArchivedCandidate],
        islands: list[set[str]],
    ) -> None:
        active = {candidate.candidate_id: candidate for candidate in cells.values()}


        sources = tuple(frozenset(values) & active.keys() for values in islands)
        additions: list[set[str]] = [set() for _ in islands]
        for source_index, source_ids in enumerate(sources):
            ranked = sorted(
                (active[candidate_id] for candidate_id in source_ids),
                key=self.candidate_order_key,
            )
            selected = ranked[: self._config.migrants_per_island]
            target_index = (source_index + 1) % self._config.num_islands
            additions[target_index].update(item.candidate_id for item in selected)
        for index, values in enumerate(additions):
            islands[index].update(values)


        for members in islands:
            members.intersection_update(active)


__all__ = [
    "ArchiveCell",
    "ArchiveProjection",
    "ArchiveProjectionConfig",
    "ArchiveProjectionConflict",
    "ArchiveProjectionError",
    "ArchiveSnapshot",
    "ArchivedCandidate",
    "FeatureDimension",
    "OBJECTIVE_DIRECTIONS",
    "ProjectionRejection",
    "deterministic_island",
]
