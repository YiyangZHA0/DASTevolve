

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Tuple

from .domain import GenerationCommit


class CommitProjection(Protocol):
    def apply(self, commit: GenerationCommit) -> Any: ...


@dataclass(frozen=True)
class CompositeProjectionResult:
    generation_id: str
    commit_hash: str
    results: Tuple[Any, ...]


class CompositeCommitProjection:


    def __init__(self, *projections: CommitProjection) -> None:
        if not projections:
            raise ValueError("at least one commit projection is required")
        if any(not callable(getattr(item, "apply", None)) for item in projections):
            raise TypeError("each projection must provide apply(commit)")
        self._projections = tuple(projections)

    @property
    def projections(self) -> Tuple[CommitProjection, ...]:
        return self._projections

    def apply(self, commit: GenerationCommit) -> CompositeProjectionResult:
        if not isinstance(commit, GenerationCommit):
            raise TypeError("composite projection accepts only GenerationCommit")
        commit.to_dict()
        results = tuple(projection.apply(commit) for projection in self._projections)
        return CompositeProjectionResult(
            generation_id=commit.generation_id,
            commit_hash=commit.commit_hash,
            results=results,
        )


__all__ = [
    "CommitProjection",
    "CompositeCommitProjection",
    "CompositeProjectionResult",
]
