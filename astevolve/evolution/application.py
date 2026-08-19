

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
from typing import Dict, Optional, Tuple

from .domain import (
    GenerationCommit,
    GenerationManifest,
    Proposal,
    SealedEvaluation,
)
from .ports import GenerationPublisher


class GenerationState(str, Enum):
    COLLECTING = "collecting"
    READY_TO_PUBLISH = "ready_to_publish"
    PUBLISHED = "published"


class GenerationStateError(RuntimeError):
    pass


class UnknownProposalError(GenerationStateError):
    pass


class DuplicateEvaluationError(GenerationStateError):
    pass


class IncompleteGenerationError(GenerationStateError):
    pass


class PublicationConflictError(GenerationStateError):
    pass


@dataclass(frozen=True)
class SubmissionReceipt:
    proposal_id: str
    accepted: bool
    idempotent: bool
    terminal_count: int
    logical_budget: int
    state: GenerationState


@dataclass(frozen=True)
class GenerationProgress:
    state: GenerationState
    terminal_count: int
    logical_budget: int
    pending_proposal_ids: Tuple[str, ...]


class GenerationEngine:


    def __init__(
        self,
        manifest: GenerationManifest,
        publisher: GenerationPublisher,
    ) -> None:
        if not isinstance(manifest, GenerationManifest):
            raise TypeError("manifest must be GenerationManifest")
        manifest.verify()
        self._manifest = manifest


        self._proposal_by_id: Dict[str, Proposal] = {
            proposal.proposal_id: proposal for proposal in manifest.proposals
        }
        self._ordered_proposal_ids = manifest.ordered_proposal_ids
        self._logical_budget = manifest.logical_budget
        self._publisher = publisher
        self._evaluations: Dict[str, SealedEvaluation] = {}
        self._state = GenerationState.COLLECTING
        self._published_commit: Optional[GenerationCommit] = None
        self._lock = threading.RLock()

    @classmethod
    def restore(
        cls,
        manifest: GenerationManifest,
        publisher: GenerationPublisher,
    ) -> "GenerationEngine":


        engine = cls(manifest, publisher)
        stored = publisher.get(manifest.generation_id)
        if stored is None:
            return engine
        if not isinstance(stored, GenerationCommit):
            raise PublicationConflictError(
                "publisher did not restore a GenerationCommit"
            )
        stored.verify(manifest)
        engine._evaluations = {
            evaluation.proposal_id: evaluation for evaluation in stored.evaluations
        }
        engine._published_commit = stored
        engine._state = GenerationState.PUBLISHED
        return engine

    @property
    def manifest(self) -> GenerationManifest:
        return self._manifest

    @property
    def state(self) -> GenerationState:
        with self._lock:
            return self._state

    @property
    def published_commit(self) -> Optional[GenerationCommit]:
        with self._lock:
            return self._published_commit

    def progress(self) -> GenerationProgress:
        with self._lock:
            pending = tuple(
                proposal_id
                for proposal_id in self._ordered_proposal_ids
                if proposal_id not in self._evaluations
            )
            return GenerationProgress(
                state=self._state,
                terminal_count=len(self._evaluations),
                logical_budget=self._logical_budget,
                pending_proposal_ids=pending,
            )

    def submit(self, evaluation: SealedEvaluation) -> SubmissionReceipt:
        if not isinstance(evaluation, SealedEvaluation):
            raise TypeError("evaluation must be SealedEvaluation")
        with self._lock:
            proposal = self._proposal_by_id.get(evaluation.proposal_id)
            if proposal is None:
                raise UnknownProposalError(
                    f"proposal {evaluation.proposal_id!r} is not reserved"
                )


            evaluation.verify_for(proposal)

            existing = self._evaluations.get(evaluation.proposal_id)
            if existing is not None:
                if existing != evaluation:
                    raise DuplicateEvaluationError(
                        f"conflicting outcome for {evaluation.proposal_id!r}"
                    )
                return self._receipt(evaluation.proposal_id, idempotent=True)

            if self._state == GenerationState.PUBLISHED:
                committed = {
                    item.proposal_id: item
                    for item in (
                        self._published_commit.evaluations
                        if self._published_commit
                        else ()
                    )
                }
                if committed.get(evaluation.proposal_id) == evaluation:
                    return self._receipt(evaluation.proposal_id, idempotent=True)
                raise DuplicateEvaluationError(
                    f"generation {self._manifest.generation_id!r} is already published"
                )

            self._evaluations[evaluation.proposal_id] = evaluation
            if len(self._evaluations) == self._logical_budget:
                self._state = GenerationState.READY_TO_PUBLISH
            return self._receipt(evaluation.proposal_id, idempotent=False)

    def _receipt(self, proposal_id: str, *, idempotent: bool) -> SubmissionReceipt:
        return SubmissionReceipt(
            proposal_id=proposal_id,
            accepted=not idempotent,
            idempotent=idempotent,
            terminal_count=len(self._evaluations),
            logical_budget=self._logical_budget,
            state=self._state,
        )

    def publish(self) -> GenerationCommit:
        with self._lock:
            if self._state == GenerationState.PUBLISHED:
                assert self._published_commit is not None
                return self._published_commit
            if self._state != GenerationState.READY_TO_PUBLISH:
                pending = self.progress().pending_proposal_ids
                raise IncompleteGenerationError(
                    "generation publication barrier is incomplete; pending="
                    + ", ".join(pending)
                )


            self._manifest.verify()
            for proposal in self._manifest.proposals:
                self._evaluations[proposal.proposal_id].verify_for(proposal)
            commit = GenerationCommit.create(self._manifest, self._evaluations.values())
            stored = self._publisher.publish_atomic(commit)
            if not isinstance(stored, GenerationCommit):
                raise PublicationConflictError(
                    "publisher did not return a GenerationCommit"
                )
            stored.verify(self._manifest)
            if stored.commit_hash != commit.commit_hash:
                raise PublicationConflictError(
                    "publisher returned a conflicting generation commit"
                )
            self._published_commit = stored
            self._state = GenerationState.PUBLISHED
            return stored


__all__ = [
    "DuplicateEvaluationError",
    "GenerationEngine",
    "GenerationProgress",
    "GenerationState",
    "GenerationStateError",
    "IncompleteGenerationError",
    "PublicationConflictError",
    "SubmissionReceipt",
    "UnknownProposalError",
]
