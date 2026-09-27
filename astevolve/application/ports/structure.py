

from __future__ import annotations

from typing import Protocol

from astevolve.domain import Candidate, EvidenceBundle, RunContext


class StructurePredictor(Protocol):


    def predict(self, candidate: Candidate, context: RunContext) -> EvidenceBundle:
        ...
