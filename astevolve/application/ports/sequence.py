

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from astevolve.domain import Candidate, CompiledDesign, DesignStrategy, EvidenceBundle, RunContext


class SequenceProposer(Protocol):


    def propose(
        self,
        design: CompiledDesign,
        strategy: DesignStrategy,
        knowledge: EvidenceBundle,
        context: RunContext,
    ) -> Sequence[Candidate]:
        ...


class SequenceScorer(Protocol):


    def score(self, candidate: Candidate, context: RunContext) -> Mapping[str, float]:
        ...
