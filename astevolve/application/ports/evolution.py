

from __future__ import annotations

from typing import Protocol

from astevolve.domain import DesignStrategy, EditContract, EvaluationReport, RunContext


class StrategyEvolver(Protocol):


    def propose(
        self,
        strategy: DesignStrategy,
        evaluation: EvaluationReport,
        contract: EditContract,
        context: RunContext,
    ) -> DesignStrategy:
        ...
