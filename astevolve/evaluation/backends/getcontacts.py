

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping

from astevolve.evaluation.contracts import ScoreTerm
from astevolve.evaluation.support import (
    as_list,
    evaluator_weight,
    mean,
    safe_float,
    score_at_least,
)
from astevolve.runtime.tools import resolve_tool_command

from .base import (
    backend_enabled,
    backend_extra_args,
    backend_item,
    backend_required,
    backend_timeout,
    backend_unavailable_term,
    run_backend_command,
    structure_input_for_backend,
)


def parse_getcontacts_text(text: str) -> Dict[str, Any]:


    counts = {
        "total": 0,
        "hbond": 0,
        "salt_bridge": 0,
        "hydrophobic": 0,
        "vdw": 0,
        "pi_cation": 0,
        "pi_stacking": 0,
        "t_stacking": 0,
        "other": 0,
    }
    examples: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2 or parts[0].lower() in {"frame", "frames"}:
            continue
        interaction_type = parts[1].lower()
        if not interaction_type or interaction_type in {"interaction_type", "type"}:
            continue
        counts["total"] += 1
        if (
            interaction_type.startswith("hb")
            or interaction_type.startswith("wb")
            or interaction_type.startswith("lwb")
        ):
            counts["hbond"] += 1
        elif interaction_type == "sb":
            counts["salt_bridge"] += 1
        elif interaction_type == "hp":
            counts["hydrophobic"] += 1
        elif interaction_type == "vdw":
            counts["vdw"] += 1
        elif interaction_type == "pc":
            counts["pi_cation"] += 1
        elif interaction_type == "ps":
            counts["pi_stacking"] += 1
        elif interaction_type == "ts":
            counts["t_stacking"] += 1
        else:
            counts["other"] += 1
        if len(examples) < 12:
            examples.append(line)
    counts["aromatic"] = (
        counts["pi_cation"] + counts["pi_stacking"] + counts["t_stacking"]
    )
    counts["polar_specific"] = counts["hbond"] + counts["salt_bridge"]
    counts["examples"] = examples
    return counts


def getcontacts_backend_term(
    structure: Mapping[str, Any],
    score_config: Mapping[str, Any],
) -> ScoreTerm:


    if not backend_enabled(score_config, "getcontacts"):
        return ScoreTerm(
            "getcontacts_interaction_graph",
            "interface_contact_graph",
            0.0,
            0.0,
            {"enabled": False},
            backend="getcontacts",
            available=False,
        )
    config = backend_item(score_config, "getcontacts")
    command = resolve_tool_command(
        configured=(config.get("command"),),
        env_names=(
            "ASTEVOLVE_GETCONTACTS_BIN",
            "ASTEVOLVE_GET_STATIC_CONTACTS_BIN",
        ),
        directory_env_candidates=(
            ("ASTEVOLVE_GETCONTACTS_ROOT", "get_static_contacts.py"),
            ("ASTEVOLVE_GETCONTACTS_ROOT", "get_static_contacts"),
        ),
        root_candidates=(
            "getcontacts/get_static_contacts.py",
            "getcontacts/get_static_contacts",
        ),
        path_candidates=("get_static_contacts.py", "get_static_contacts"),
    )
    if not command:
        return backend_unavailable_term(
            "getcontacts",
            "interface_contact_graph",
            score_config,
            "eval_getcontacts",
            "getContacts get_static_contacts.py command not found",
            "getContacts backend enabled but get_static_contacts.py command not found",
        )
    with tempfile.TemporaryDirectory(prefix="astevolve_getcontacts_") as tmp:
        work_dir = Path(tmp)
        pdb_path, input_details = structure_input_for_backend(structure, work_dir)
        if pdb_path is None:
            if backend_required(score_config, "getcontacts"):
                input_details = {
                    **input_details,
                    "required": True,
                    "dimension": "correctness",
                }
            return ScoreTerm(
                "getcontacts_interaction_graph",
                "interface_contact_graph",
                0.0,
                evaluator_weight(score_config, "eval_getcontacts", 0.0),
                input_details,
                [str(input_details.get("reason"))],
                backend="getcontacts",
                available=False,
            )

        output_path = work_dir / "getcontacts.tsv"
        interaction_types = (
            config.get("itypes")
            or os.environ.get("ASTEVOLVE_GETCONTACTS_ITYPES")
            or ["sb", "pc", "ps", "ts", "hb", "vdw", "hp"]
        )
        args = [command, "--structure", str(pdb_path), "--output", str(output_path)]
        if isinstance(interaction_types, str):
            args.extend(
                [
                    "--itypes",
                    *[
                        part
                        for part in interaction_types.replace(",", " ").split()
                        if part
                    ],
                ]
            )
        else:
            args.extend(
                [
                    "--itypes",
                    *[
                        str(part)
                        for part in as_list(interaction_types)
                        if str(part)
                    ],
                ]
            )
        selection = config.get("sele") or os.environ.get("ASTEVOLVE_GETCONTACTS_SELE")
        selection_two = config.get("sele2") or os.environ.get(
            "ASTEVOLVE_GETCONTACTS_SELE2"
        )
        ligand = config.get("ligand") or os.environ.get("ASTEVOLVE_GETCONTACTS_LIGAND")
        solvent = config.get("solv") or os.environ.get("ASTEVOLVE_GETCONTACTS_SOLV")
        if selection:
            args.extend(["--sele", str(selection)])
        if selection_two:
            args.extend(["--sele2", str(selection_two)])
        if ligand:
            args.extend(
                [
                    "--ligand",
                    *[
                        part
                        for part in str(ligand).replace(",", " ").split()
                        if part
                    ],
                ]
            )
        if solvent:
            args.extend(["--solv", str(solvent)])
        args.extend(backend_extra_args(score_config, "getcontacts"))

        executable = Path(str(command))
        if executable.suffix.lower() == ".py" and not os.access(
            str(executable), os.X_OK
        ):
            args = [sys.executable, *args]

        result = run_backend_command(
            args,
            backend_timeout(score_config, "getcontacts", 180),
            cwd=work_dir,
        )
        text_parts = [str(result.get("stdout_tail", ""))]
        if output_path.exists():
            try:
                text_parts.append(
                    output_path.read_text(encoding="utf-8", errors="ignore")
                )
            except OSError:
                pass
        counts = parse_getcontacts_text("\n".join(text_parts))
        total_target = safe_float(config.get("total_contact_target"), 16.0) or 16.0
        polar_target = safe_float(config.get("polar_contact_target"), 2.0) or 2.0
        hydrophobic_target = (
            safe_float(config.get("hydrophobic_contact_target"), 4.0) or 4.0
        )
        aromatic_target = (
            safe_float(config.get("aromatic_contact_target"), 1.0) or 1.0
        )
        score = mean(
            [
                score_at_least(counts["total"], total_target, 0.0),
                score_at_least(counts["polar_specific"], polar_target, 0.0),
                score_at_least(
                    counts["hydrophobic"] + counts["vdw"],
                    hydrophobic_target,
                    0.0,
                ),
                score_at_least(counts["aromatic"], aromatic_target, 0.0),
            ]
        )
        details = {
            **input_details,
            **result,
            "required": backend_required(score_config, "getcontacts"),
            "analysis_role": config.get("analysis_role"),
            "output_path": str(output_path),
            "parsed_contact_counts": counts,
            "targets": {
                "total": total_target,
                "polar_specific": polar_target,
                "hydrophobic_or_vdw": hydrophobic_target,
                "aromatic": aromatic_target,
            },
            "method": "getContacts static non-covalent interaction graph",
        }
        warnings = []
        if not result.get("ok"):
            warnings.append("getContacts command failed")
        if counts["total"] <= 0:
            warnings.append("getContacts produced no parseable interaction contacts")
        return ScoreTerm(
            "getcontacts_interaction_graph",
            "interface_contact_graph",
            score,
            evaluator_weight(score_config, "eval_getcontacts", 0.0),
            details,
            warnings,
            backend="getcontacts",
            available=bool(result.get("ok") and counts["total"] > 0),
        )
