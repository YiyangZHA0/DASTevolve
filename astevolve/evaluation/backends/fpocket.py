

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping

from astevolve.evaluation.contracts import ScoreTerm
from astevolve.evaluation.support import clamp01, evaluator_weight, mean, score_at_least

from .base import (
    backend_command,
    backend_enabled,
    backend_extra_args,
    backend_timeout,
    parse_float_after,
    run_backend_command,
    structure_input_for_backend,
)


def fpocket_backend_term(
    structure: Mapping[str, Any],
    score_config: Mapping[str, Any],
) -> ScoreTerm:


    if not backend_enabled(score_config, "fpocket"):
        return ScoreTerm("fpocket_pocket_geometry", "pocket_geometry", 0.0, 0.0, {"enabled": False}, backend="fpocket", available=False)
    command = backend_command(score_config, "fpocket", "ASTEVOLVE_FPOCKET_BIN", ("fpocket",))
    if not command:
        return ScoreTerm("fpocket_pocket_geometry", "pocket_geometry", 0.0, evaluator_weight(score_config, "eval_fpocket", 0.0), {"enabled": True, "reason": "fpocket command not found"}, ["fpocket backend enabled but command not found"], backend="fpocket", available=False)
    with tempfile.TemporaryDirectory(prefix="astevolve_fpocket_") as tmp:
        work_dir = Path(tmp)
        pdb_path, input_details = structure_input_for_backend(structure, work_dir)
        if pdb_path is None:
            return ScoreTerm("fpocket_pocket_geometry", "pocket_geometry", 0.0, evaluator_weight(score_config, "eval_fpocket", 0.0), input_details, [str(input_details.get("reason"))], backend="fpocket", available=False)
        args = [command, "-f", str(pdb_path)]
        args.extend(backend_extra_args(score_config, "fpocket"))
        result = run_backend_command(args, backend_timeout(score_config, "fpocket", 180), cwd=work_dir)
        info_text = ""
        for path in work_dir.rglob("*_info.txt"):
            try:
                info_text += "\n" + path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                pass
        pocket_score = parse_float_after(info_text, ("Pocket Score", "Druggability Score", "Score"))
        volume = parse_float_after(info_text, ("Pocket volume", "Volume"))
        score_parts = []
        if pocket_score is not None:
            score_parts.append(clamp01(pocket_score / 50.0 if pocket_score > 1.0 else pocket_score))
        if volume is not None:
            score_parts.append(score_at_least(volume, 120.0, 20.0))
        score = mean(score_parts) if score_parts else (0.0 if not result.get("ok") else 0.5)
        details = {**input_details, **result, "parsed_pocket_score": pocket_score, "parsed_pocket_volume": volume}
        return ScoreTerm("fpocket_pocket_geometry", "pocket_geometry", score or 0.0, evaluator_weight(score_config, "eval_fpocket", 0.0), details, [] if result.get("ok") else ["fpocket command failed or produced no parseable metrics"], backend="fpocket", available=bool(result.get("ok")))
