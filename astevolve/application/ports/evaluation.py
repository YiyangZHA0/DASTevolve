

from __future__ import annotations

from typing import Protocol

from astevolve.domain import Candidate, CompiledDesign, EvaluationReport, EvidenceBundle, RunContext


class CandidateEvaluator(Protocol):


    def evaluate_for_ranking(
        self,
        candidate: Candidate,
        design: CompiledDesign,
        evidence: EvidenceBundle,
        context: RunContext,
    ) -> EvaluationReport:
        ...

    def evaluate_final(
        self,
        candidate: Candidate,
        design: CompiledDesign,
        evidence: EvidenceBundle,
        context: RunContext,
    ) -> EvaluationReport:
        ...
