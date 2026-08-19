

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from astevolve.application.ports.runner import DesignSearchRunner
from astevolve.domain import DesignStrategy, ExperimentResult, RunContext


@dataclass
class DesignRoundWorkflow:


    runner: DesignSearchRunner

    def run(
        self,
        strategy: DesignStrategy | Mapping[str, object],
        context: RunContext,
    ) -> ExperimentResult:
        typed_strategy = strategy if isinstance(strategy, DesignStrategy) else DesignStrategy.from_mapping(strategy)
        raw = self.runner.run(typed_strategy.to_legacy_dict(), context)
        return ExperimentResult.from_legacy(raw)


def run_design_round(
    strategy: DesignStrategy | Mapping[str, object],
    context: RunContext,
) -> ExperimentResult:


    from astevolve.adapters.legacy.design_search import LegacyDesignSearchRunner

    return DesignRoundWorkflow(LegacyDesignSearchRunner()).run(strategy, context)
