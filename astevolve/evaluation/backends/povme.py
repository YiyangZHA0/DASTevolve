

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from astevolve.evaluation.contracts import ScoreTerm
from astevolve.evaluation.support import evaluator_weight, score_at_least

from .base import (
    backend_command,
    backend_enabled,
    backend_extra_args,
    backend_item,
    backend_timeout,
    parse_float_after,
    run_backend_command,
    structure_input_for_backend,
)


def povme_backend_term(
    structure: Mapping[str, Any],
    score_config: Mapping[str, Any],
) -> ScoreTerm:


    if not backend_enabled(score_config, "povme"):
        return ScoreTerm("povme_pocket_volume", "pocket_geometry", 0.0, 0.0, {"enabled": False}, backend="povme", available=False)
    command = backend_command(score_config, "povme", "ASTEVOLVE_POVME_BIN", ("POVME3.py", "POVME3", "POVME", "povme"))
    if not command:
        return ScoreTerm("povme_pocket_volume", "pocket_geometry", 0.0, evaluator_weight(score_config, "eval_povme", 0.0), {"enabled": True, "reason": "POVME command not found"}, ["POVME backend enabled but command not found"], backend="povme", available=False)
    config_path = backend_item(score_config, "povme").get("config") or backend_item(score_config, "povme").get("config_path") or os.environ.get("ASTEVOLVE_POVME_CONFIG")
    if not config_path:
        return ScoreTerm("povme_pocket_volume", "pocket_geometry", 0.0, evaluator_weight(score_config, "eval_povme", 0.0), {"enabled": True, "reason": "POVME requires a config/config_path"}, ["POVME backend enabled but no config path provided"], backend="povme", available=False)
    config_path = Path(str(config_path))
    if not config_path.exists():
        return ScoreTerm("povme_pocket_volume", "pocket_geometry", 0.0, evaluator_weight(score_config, "eval_povme", 0.0), {"enabled": True, "config_path": str(config_path), "reason": "POVME config path does not exist"}, ["POVME config path does not exist"], backend="povme", available=False)
    with tempfile.TemporaryDirectory(prefix="astevolve_povme_") as tmp:
        work_dir = Path(tmp)
        pdb_path, input_details = structure_input_for_backend(structure, work_dir)
        if pdb_path is None:
            return ScoreTerm("povme_pocket_volume", "pocket_geometry", 0.0, evaluator_weight(score_config, "eval_povme", 0.0), input_details, [str(input_details.get("reason"))], backend="povme", available=False)
        args = [command, str(config_path)]
        args.extend(backend_extra_args(score_config, "povme"))
        result = run_backend_command(args, backend_timeout(score_config, "povme", 300), cwd=work_dir)
        text_parts = [str(result.get("stdout_tail", "")), str(result.get("stderr_tail", ""))]
        for path in work_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".csv", ".txt", ".log"}:
                try:
                    text_parts.append(path.read_text(encoding="utf-8", errors="ignore")[-4000:])
                except OSError:
                    pass
        text = "\n".join(text_parts)
        volume = parse_float_after(text, ("volume", "pocket volume", "avg_volume"))
        score = score_at_least(volume, 120.0, 20.0) if volume is not None else (0.0 if not result.get("ok") else 0.5)
        details = {**input_details, **result, "config_path": str(config_path), "parsed_volume": volume}
        return ScoreTerm("povme_pocket_volume", "pocket_geometry", score, evaluator_weight(score_config, "eval_povme", 0.0), details, [] if result.get("ok") else ["POVME command failed or produced no parseable metrics"], backend="povme", available=bool(result.get("ok")))
