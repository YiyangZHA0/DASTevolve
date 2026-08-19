

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from astevolve.cases import resolve_case
from astevolve.domain import DesignStrategy, ExperimentResult, RunContext


@dataclass
class InnerLoopRunner:


    case_id: Optional[str] = None
    case_root: Optional[str | Path] = None
    manifest_path: Optional[str | Path] = None

    def _case(self):
        return resolve_case(
            self.case_id,
            case_root=self.case_root,
            manifest_path=self.manifest_path,
        )

    def run(
        self,
        strategy: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        from engine.case_builder import run_design_search

        case = self._case()
        return run_design_search(
            strategy,
            seed=seed,
            design_state_path=str(case.design_state_path),
            memory_path=str(case.memory_path),
        )

    def run_typed(
        self,
        strategy: DesignStrategy | Dict[str, Any],
        seed: Optional[int] = None,
    ) -> ExperimentResult:
        from astevolve.application import run_design_round

        case = self._case()
        context = RunContext(
            case_id=case.case_id,
            project_root=case.root,
            output_root=case.output_root,
            design_state_path=case.design_state_path,
            memory_path=case.memory_path,
            seed=seed,
        )
        return run_design_round(strategy, context)
