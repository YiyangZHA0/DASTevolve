

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping

from astevolve.evaluation.contracts import ScoreTerm
from astevolve.evaluation.support import evaluator_weight

from .base import (
    backend_command,
    backend_enabled,
    backend_extra_args,
    backend_timeout,
    parse_float_after,
    run_backend_command,
    score_energy,
    structure_input_for_backend,
)


def foldx_backend_term(
    structure: Mapping[str, Any],
    score_config: Mapping[str, Any],
) -> ScoreTerm:


    if not backend_enabled(score_config, "foldx"):
        return ScoreTerm("foldx_analyse_complex", "interface_energy", 0.0, 0.0, {"enabled": False}, backend="foldx", available=False)
    command = backend_command(score_config, "foldx", "ASTEVOLVE_FOLDX_BIN", ("foldx", "FoldX"))
    if not command:
        return ScoreTerm("foldx_analyse_complex", "interface_energy", 0.0, evaluator_weight(score_config, "eval_foldx", 0.0), {"enabled": True, "reason": "FoldX command not found"}, ["FoldX backend enabled but command not found"], backend="foldx", available=False)
    with tempfile.TemporaryDirectory(prefix="astevolve_foldx_") as tmp:
        work_dir = Path(tmp)
        pdb_path, input_details = structure_input_for_backend(structure, work_dir)
        if pdb_path is None:
            return ScoreTerm("foldx_analyse_complex", "interface_energy", 0.0, evaluator_weight(score_config, "eval_foldx", 0.0), input_details, [str(input_details.get("reason"))], backend="foldx", available=False)
        args = [command, "--command=AnalyseComplex", f"--pdb={pdb_path.name}", f"--output-dir={work_dir}"]
        args.extend(backend_extra_args(score_config, "foldx"))
        result = run_backend_command(args, backend_timeout(score_config, "foldx", 240), cwd=work_dir)
        files_text = []
        for path in work_dir.glob("*"):
            if path.is_file() and path.suffix.lower() in {".fxout", ".txt"}:
                try:
                    files_text.append(path.read_text(encoding="utf-8", errors="ignore")[-4000:])
                except OSError:
                    pass
        text = "\n".join(files_text) + f"\n{result.get('stdout_tail', '')}\n{result.get('stderr_tail', '')}"
        interaction_energy = parse_float_after(text, ("Interaction Energy", "interaction_energy", "summary"))
        score = score_energy(interaction_energy)
        details = {**input_details, **result, "parsed_interaction_energy": interaction_energy, "output_files": [path.name for path in work_dir.glob("*") if path.is_file()][:20]}
        return ScoreTerm("foldx_analyse_complex", "interface_energy", score, evaluator_weight(score_config, "eval_foldx", 0.0), details, [] if result.get("ok") else ["FoldX command failed or produced no parseable metrics"], backend="foldx", available=bool(result.get("ok")))
