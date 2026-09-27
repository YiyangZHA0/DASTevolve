

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from astevolve.cases import resolve_case


@dataclass
class OuterLoopRun:


    project_root: Path
    case_id: Optional[str] = None
    case_root: Optional[str | Path] = None
    manifest_path: Optional[str | Path] = None
    initial_program: Optional[str] = None
    evaluator: str = "evaluator.py"
    config: Optional[str] = None
    output_dir: Optional[str] = None

    def command(self, iterations: Optional[int] = None) -> List[str]:


        case = resolve_case(
            self.case_id,
            case_root=self.case_root,
            manifest_path=self.manifest_path,
        )
        bundle_root = (
            case.manifest_path.parent if case.manifest_path else Path(case.root)
        )
        initial_program = self.initial_program or str(
            bundle_root
            / str(case.metadata.get("entry_program", "initial_program.py"))
        )
        config = self.config or str(
            bundle_root / str(case.metadata.get("config_path", "config.yaml"))
        )
        evaluator = self.evaluator
        if evaluator == "evaluator.py":
            evaluator = str(
                bundle_root
                / str(case.metadata.get("outer_evaluator_path", evaluator))
            ) if case.metadata.get("outer_evaluator_path") else evaluator
        command = [
            "python",
            "outerloop/outerloop-run.py",
            initial_program,
            evaluator,
            "--config",
            config,
        ]
        if iterations is not None:
            command.extend(["--iterations", str(iterations)])
        if self.output_dir:
            command.extend(["--output", self.output_dir])
        return command
