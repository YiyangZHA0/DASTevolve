

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping

from astevolve.evaluation.contracts import ScoreTerm
from astevolve.evaluation.support import clamp01, evaluator_weight, mean

from .base import (
    backend_command,
    backend_enabled,
    backend_extra_args,
    backend_item,
    backend_required,
    backend_timeout,
    backend_unavailable_term,
    parse_float_after,
    run_backend_command,
    score_energy,
    structure_input_for_backend,
)


def rosetta_backend_term(
    structure: Mapping[str, Any],
    score_config: Mapping[str, Any],
) -> ScoreTerm:


    if not backend_enabled(score_config, "rosetta"):
        return ScoreTerm("rosetta_interface_analyzer", "interface_energy", 0.0, 0.0, {"enabled": False}, backend="rosetta", available=False)
    command = backend_command(
        score_config,
        "rosetta",
        "ASTEVOLVE_ROSETTA_INTERFACE_ANALYZER",
        ("InterfaceAnalyzer", "InterfaceAnalyzer.default.linuxgccrelease", "interface_analyzer.default.linuxgccrelease"),
    )
    if not command:
        return backend_unavailable_term(
            "rosetta",
            "interface_energy",
            score_config,
            "eval_rosetta",
            "Rosetta InterfaceAnalyzer command not found",
            "Rosetta backend enabled but InterfaceAnalyzer command not found",
        )
    with tempfile.TemporaryDirectory(prefix="astevolve_rosetta_") as tmp:
        work_dir = Path(tmp)
        pdb_path, input_details = structure_input_for_backend(structure, work_dir)
        if pdb_path is None:
            if backend_required(score_config, "rosetta"):
                input_details = {**input_details, "required": True, "dimension": "correctness"}
            return ScoreTerm("rosetta_interface_analyzer", "interface_energy", 0.0, evaluator_weight(score_config, "eval_rosetta", 0.0), input_details, [str(input_details.get("reason"))], backend="rosetta", available=False)
        args = [command, "-s", str(pdb_path), "-pack_separated", "false"]
        interface = backend_item(score_config, "rosetta").get("interface")
        if interface:
            args.extend(["-interface", str(interface)])
        args.extend(backend_extra_args(score_config, "rosetta"))
        result = run_backend_command(args, backend_timeout(score_config, "rosetta", 240), cwd=work_dir)
        text = f"{result.get('stdout_tail', '')}\n{result.get('stderr_tail', '')}"
        dg = parse_float_after(text, ("dG_separated", "interface_delta", "dG_cross", "dG"))
        shape_complementarity = parse_float_after(text, ("sc_value", "shape complementarity", "sc "))
        score_parts = []
        if dg is not None:
            score_parts.append(score_energy(dg))
        if shape_complementarity is not None:
            score_parts.append(clamp01(shape_complementarity))
        score = mean(score_parts) if score_parts else (0.0 if not result.get("ok") else 0.5)
        details = {**input_details, **result, "parsed_interface_dg": dg, "parsed_shape_complementarity": shape_complementarity}
        if backend_required(score_config, "rosetta"):
            details["required"] = True
        return ScoreTerm("rosetta_interface_analyzer", "interface_energy", score or 0.0, evaluator_weight(score_config, "eval_rosetta", 0.0), details, [] if result.get("ok") else ["Rosetta command failed or produced no parseable metrics"], backend="rosetta", available=bool(result.get("ok")))
