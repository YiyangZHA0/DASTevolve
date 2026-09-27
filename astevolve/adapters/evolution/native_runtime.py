

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Tuple, TypeAlias

from astevolve.application.ports.runner import DesignSearchRunner
from astevolve.domain import (
    DesignStrategy,
    EvidenceBundle,
    EvidenceRecord,
    ExperimentResult,
    RunContext,
)
from astevolve.domain.dual_ast import ExecutableDualAST
from astevolve.evolution.domain import STRATEGY_REVISION, SealedEvaluation
from astevolve.evolution.orchestrator import (
    EvaluationTask,
    ProposalContext,
    ProposalDraft,
)

from .candidate_artifact import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    CANDIDATE_RECORD_EVIDENCE_KIND,
    CANDIDATE_RECORD_SCHEMA_VERSION,
    EVALUATOR_DESCRIPTOR_SCHEMA_VERSION,
    SCIENTIFIC_DIAGNOSTIC_EVIDENCE_KIND,
    SCIENTIFIC_DIAGNOSTIC_SCHEMA_VERSION,
    CandidateArtifactError,
    build_candidate_record,
    build_scientific_diagnostic_record,
    recover_candidate_record,
    recover_scientific_diagnostic_record,
    verify_candidate_record,
    verify_scientific_diagnostic_record,
)


Revision: TypeAlias = DesignStrategy | ExecutableDualAST
RevisionFactory = Callable[[ProposalContext], ProposalDraft | Revision]
EvaluationAdapter = Callable[
    [EvaluationTask],
    SealedEvaluation | tuple[str, Any, EvidenceBundle],
]
PROPOSAL_ORIGIN_SCHEMA_VERSION = "astevolve.evolution.proposal_origin.v1"


def _callable_identity(value: Callable[..., Any]) -> Mapping[str, str]:
    return {
        "module": str(getattr(value, "__module__", value.__class__.__module__) or ""),
        "qualname": str(
            getattr(value, "__qualname__", value.__class__.__qualname__) or ""
        ),
    }


@dataclass(frozen=True)
class StaticRevisionProposalSource:


    revisions: Tuple[Revision, ...]
    parent_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.revisions:
            raise ValueError("at least one static revision is required")
        detached: list[Revision] = []
        for revision in self.revisions:
            if isinstance(revision, DesignStrategy):
                detached.append(DesignStrategy.from_mapping(revision.to_legacy_dict()))
            elif isinstance(revision, ExecutableDualAST):
                detached.append(ExecutableDualAST.from_mapping(revision.to_dict()))
            else:
                raise TypeError("static revisions must be typed domain revisions")
        if self.parent_ids and len(self.parent_ids) != len(detached):
            raise ValueError("parent_ids must align with revisions")
        if any(not str(value).strip() for value in self.parent_ids):
            raise ValueError("parent_ids must be non-empty")
        object.__setattr__(self, "revisions", tuple(detached))

    def propose(self, context: ProposalContext) -> ProposalDraft:
        index = context.slot % len(self.revisions)
        parent_id = (
            self.parent_ids[index] if self.parent_ids else f"static-parent:{index:04d}"
        )
        return ProposalDraft(
            parent_id=parent_id,
            revision=self.revisions[index],
            provenance={
                "schema_version": PROPOSAL_ORIGIN_SCHEMA_VERSION,
                "mechanism": "static_revision",
                "source_index": index,
            },
        )


@dataclass(frozen=True)
class CallableProposalSource:


    factory: RevisionFactory
    default_parent_id: str = "callable-parent"

    def __post_init__(self) -> None:
        if not callable(self.factory):
            raise TypeError("proposal factory must be callable")
        if not self.default_parent_id.strip():
            raise ValueError("default_parent_id must be non-empty")

    def propose(self, context: ProposalContext) -> ProposalDraft:
        result = self.factory(context)
        if isinstance(result, ProposalDraft):
            return result
        if isinstance(result, (DesignStrategy, ExecutableDualAST)):
            return ProposalDraft(
                parent_id=self.default_parent_id,
                revision=result,
                provenance={
                    "schema_version": PROPOSAL_ORIGIN_SCHEMA_VERSION,
                    "mechanism": "callable_revision",
                    "factory": dict(_callable_identity(self.factory)),
                },
            )
        raise TypeError(
            "proposal callable must return ProposalDraft, DesignStrategy, or "
            "ExecutableDualAST"
        )


@dataclass(frozen=True)
class CallableProposalEvaluator:


    operation: EvaluationAdapter

    def __post_init__(self) -> None:
        if not callable(self.operation):
            raise TypeError("evaluation operation must be callable")

    def evaluate(self, task: EvaluationTask) -> SealedEvaluation:
        value = self.operation(task)
        if isinstance(value, SealedEvaluation):
            value.verify_for(task.proposal)
            return value
        if not isinstance(value, tuple) or len(value) != 3:
            raise TypeError(
                "evaluation operation must return SealedEvaluation or "
                "(candidate_id, EvaluationReport, EvidenceBundle)"
            )
        candidate_id, report, evidence = value
        if not isinstance(evidence, EvidenceBundle):
            raise TypeError("evaluation evidence must be EvidenceBundle")
        return SealedEvaluation.success(
            proposal=task.proposal,
            candidate_id=str(candidate_id),
            report=report,
            evidence=evidence,
        )


@dataclass(frozen=True)
class DesignSearchProposalEvaluator:


    runner: DesignSearchRunner
    context_factory: Callable[[EvaluationTask], RunContext]

    def __post_init__(self) -> None:
        if not callable(getattr(self.runner, "run", None)):
            raise TypeError("runner must implement run(strategy, context)")
        if not callable(self.context_factory):
            raise TypeError("context_factory must be callable")

    def evaluate(self, task: EvaluationTask) -> SealedEvaluation:
        if task.proposal.kind != STRATEGY_REVISION:
            raise TypeError("DesignSearchProposalEvaluator accepts strategy revisions")
        strategy = DesignStrategy.from_mapping(task.proposal.payload())
        context = self.context_factory(task)
        if not isinstance(context, RunContext):
            raise TypeError("context_factory must return RunContext")
        raw = self.runner.run(strategy.to_legacy_dict(), context)
        if not isinstance(raw, Mapping):
            raise TypeError("design search runner must return a mapping")


        result = ExperimentResult.from_legacy(raw)
        candidate_record = build_candidate_record(
            adapter=self,
            runner=self.runner,
            context=context,
            result=result,
        )
        scientific_record = build_scientific_diagnostic_record(
            candidate_record=candidate_record,
            result=result,
        )
        evidence = EvidenceBundle.of(
            (
                EvidenceRecord(
                    source="astevolve-design-search-adapter",
                    kind=CANDIDATE_RECORD_EVIDENCE_KIND,
                    value=candidate_record,
                    details={
                        "run_id": task.run_id,
                        "generation_id": task.generation_id,
                        "proposal_id": task.proposal.proposal_id,
                        "candidate_record_hash": candidate_record[
                            "candidate_record_hash"
                        ],
                    },
                ),
                EvidenceRecord(
                    source="astevolve-design-search-adapter",
                    kind=SCIENTIFIC_DIAGNOSTIC_EVIDENCE_KIND,
                    value=scientific_record,
                    details={
                        "run_id": task.run_id,
                        "generation_id": task.generation_id,
                        "proposal_id": task.proposal.proposal_id,
                        "diagnostic_record_hash": scientific_record[
                            "diagnostic_record_hash"
                        ],
                    },
                ),
            )
        )
        return SealedEvaluation.success(
            proposal=task.proposal,
            candidate_id=candidate_record["candidate_id"],
            report=result.evaluation,
            evidence=evidence,
        )


__all__ = [
    "ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "CANDIDATE_RECORD_EVIDENCE_KIND",
    "CANDIDATE_RECORD_SCHEMA_VERSION",
    "EVALUATOR_DESCRIPTOR_SCHEMA_VERSION",
    "SCIENTIFIC_DIAGNOSTIC_EVIDENCE_KIND",
    "SCIENTIFIC_DIAGNOSTIC_SCHEMA_VERSION",
    "CandidateArtifactError",
    "CallableProposalEvaluator",
    "CallableProposalSource",
    "DesignSearchProposalEvaluator",
    "EvaluationAdapter",
    "PROPOSAL_ORIGIN_SCHEMA_VERSION",
    "RevisionFactory",
    "StaticRevisionProposalSource",
    "recover_candidate_record",
    "recover_scientific_diagnostic_record",
    "verify_candidate_record",
    "verify_scientific_diagnostic_record",
]
