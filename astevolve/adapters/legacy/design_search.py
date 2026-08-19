

from __future__ import annotations

from typing import Mapping

from astevolve.domain import RunContext


class LegacyDesignSearchRunner:


    def run(self, strategy: Mapping[str, object], context: RunContext) -> Mapping[str, object]:

        from engine.case_builder import run_design_search

        return run_design_search(
            dict(strategy),
            seed=context.seed,
            design_state_path=str(context.design_state_path),
            memory_path=str(context.memory_path),
        )
