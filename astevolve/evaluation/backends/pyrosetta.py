

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

from astevolve.evaluation.contracts import ScoreTerm
from astevolve.evaluation.support import clamp01, evaluator_weight, mean, safe_float
from astevolve.runtime.paths import artifact_root

from .base import (
    backend_enabled,
    backend_item,
    backend_required,
    backend_timeout,
    run_backend_command,
    score_energy,
    structure_input_for_backend,
)


PYROSETTA_EVAL_SCRIPT = r"""
from __future__ import annotations

import json
import os
import sys

pdb_path = sys.argv[1]
interface = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else ""
run_relax = len(sys.argv) > 3 and sys.argv[3].lower() in {"1", "true", "yes", "on"}
relax_repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 1
constrain_backbone = len(sys.argv) > 5 and sys.argv[5].lower() in {"1", "true", "yes", "on"}
relaxed_output = sys.argv[6] if len(sys.argv) > 6 else ""
init_opts = os.environ.get(
    "ASTEVOLVE_PYROSETTA_INIT_OPTS",
    "-mute all -ignore_unrecognized_res true -load_PDB_components false",
)

metrics = {}
try:
    import pyrosetta

    pyrosetta.init(init_opts)
    pose = pyrosetta.pose_from_pdb(pdb_path)
    scorefxn = pyrosetta.get_fa_scorefxn()
    reference_pose = pose.clone()
    metrics["pre_relax_total_score"] = float(scorefxn(pose))
    if run_relax:
        from pyrosetta.rosetta.protocols.relax import FastRelax
        relax = FastRelax(scorefxn, max(1, relax_repeats))
        if constrain_backbone:
            from pyrosetta.rosetta.core.kinematics import MoveMap
            movemap = MoveMap()
            movemap.set_bb(False)
            movemap.set_chi(True)
            movemap.set_jump(False)
            relax.set_movemap(movemap)
            if hasattr(relax, "constrain_relax_to_start_coords"):
                relax.constrain_relax_to_start_coords(True)
            if hasattr(relax, "coord_constrain_sidechains"):
                relax.coord_constrain_sidechains(False)
        relax.apply(pose)
        metrics["relaxed"] = True
        metrics["relax_repeats"] = max(1, relax_repeats)
        metrics["backbone_coordinate_constraint"] = bool(constrain_backbone)
        try:
            from pyrosetta.rosetta.core.scoring import CA_rmsd
            metrics["relax_ca_rmsd"] = float(CA_rmsd(reference_pose, pose))
        except Exception as exc:
            metrics["relax_ca_rmsd_error"] = str(exc)
        if relaxed_output:
            pose.dump_pdb(relaxed_output)
            metrics["relaxed_pdb_path"] = relaxed_output
    else:
        metrics["relaxed"] = False
    metrics["post_relax_total_score"] = float(scorefxn(pose))
    metrics["total_score"] = metrics["post_relax_total_score"]
    metrics["relax_total_score_delta"] = metrics["post_relax_total_score"] - metrics["pre_relax_total_score"]
    metrics["residue_count"] = int(pose.total_residue())

    if interface:
        try:
            from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

            mover = InterfaceAnalyzerMover(interface)
            if hasattr(mover, "set_compute_packstat"):
                mover.set_compute_packstat(True)
            if hasattr(mover, "set_pack_separated"):
                mover.set_pack_separated(False)
            mover.apply(pose)
            try:
                side_a, side_b = interface.split("_", 1)
                pdb_info = pose.pdb_info()
                group_a = [i for i in range(1, pose.total_residue() + 1) if pdb_info.chain(i) in set(side_a)]
                group_b = [i for i in range(1, pose.total_residue() + 1) if pdb_info.chain(i) in set(side_b)]
                cutoff_squared = 100.0
                interface_indices = set()
                for left in group_a:
                    left_xyz = pose.residue(left).nbr_atom_xyz()
                    for right in group_b:
                        if left_xyz.distance_squared(pose.residue(right).nbr_atom_xyz()) <= cutoff_squared:
                            interface_indices.add(left)
                            interface_indices.add(right)
                scorefxn(pose)
                metrics["interface_residue_energies"] = dict(
                    (str(index), float(pose.energies().residue_total_energy(index)))
                    for index in sorted(interface_indices)
                )
                metrics["interface_residue_count"] = len(interface_indices)
                metrics["interface_residue_cutoff_angstrom"] = 10.0
            except Exception as exc:
                metrics["interface_residue_energy_error"] = str(exc)
            accessors = {
                "interface_dg": ("get_interface_dG", "get_interface_delta"),
                "interface_delta_sasa": ("get_interface_delta_sasa",),
                "interface_packstat": ("get_interface_packstat",),
                "shape_complementarity": ("get_interface_sc", "get_sc_value"),
                "complex_energy": ("get_complex_energy",),
                "separated_interface_energy": ("get_separated_interface_energy",),
            }
            for key, names in accessors.items():
                for name in names:
                    fn = getattr(mover, name, None)
                    if not fn:
                        continue
                    try:
                        metrics[key] = float(fn())
                        break
                    except Exception:
                        pass
        except Exception as exc:
            metrics["interface_error"] = str(exc)

    print("ASTEVOLVE_PYROSETTA_JSON=" + json.dumps({"ok": True, "metrics": metrics}, sort_keys=True))
except Exception as exc:
    print("ASTEVOLVE_PYROSETTA_JSON=" + json.dumps({"ok": False, "error": str(exc), "metrics": metrics}, sort_keys=True))
    raise
"""


def pyrosetta_python(score_config: Mapping[str, Any]) -> str:


    config = backend_item(score_config, "pyrosetta")
    configured = config.get("python") or config.get("command") or os.environ.get("ASTEVOLVE_PYROSETTA_PYTHON")
    if configured:
        path = shutil.which(str(configured)) or str(configured)
        if Path(path).exists() or shutil.which(path):
            return path
        return str(configured)
    return sys.executable


def parse_pyrosetta_json(text: str) -> Dict[str, Any]:


    marker = "ASTEVOLVE_PYROSETTA_JSON="
    for line in reversed(text.splitlines()):
        if marker not in line:
            continue
        raw = line.split(marker, 1)[1].strip()
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def pyrosetta_backend_term(
    structure: Mapping[str, Any],
    score_config: Mapping[str, Any],
) -> ScoreTerm:


    if not backend_enabled(score_config, "pyrosetta"):
        return ScoreTerm("pyrosetta_interface_energy", "interface_energy", 0.0, 0.0, {"enabled": False}, backend="pyrosetta", available=False)
    with tempfile.TemporaryDirectory(prefix="astevolve_pyrosetta_") as tmp:
        work_dir = Path(tmp)
        pdb_path, input_details = structure_input_for_backend(structure, work_dir)
        if pdb_path is None:
            if backend_required(score_config, "pyrosetta"):
                input_details = {**input_details, "required": True, "dimension": "correctness"}
            return ScoreTerm("pyrosetta_interface_energy", "interface_energy", 0.0, evaluator_weight(score_config, "eval_pyrosetta", 0.0), input_details, [str(input_details.get("reason"))], backend="pyrosetta", available=False)

        config = backend_item(score_config, "pyrosetta")
        interface = config.get("interface") or os.environ.get("ASTEVOLVE_PYROSETTA_INTERFACE") or backend_item(score_config, "rosetta").get("interface") or ""
        python_bin = pyrosetta_python(score_config)
        run_relax = bool(config.get("fastrelax", False))
        relax_repeats = max(1, int(config.get("fastrelax_repeats", 1) or 1))
        constrain_backbone = bool(config.get("backbone_coordinate_constraint", True))
        relaxed_root = Path(
            str(
                config.get("relaxed_artifact_root")
                or os.environ.get("ASTEVOLVE_ARTIFACT_ROOT")
                or artifact_root()
            )
        ) / "pyrosetta_relaxed"
        relaxed_path = ""
        if run_relax:
            import hashlib
            relaxed_root.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(pdb_path.read_bytes()).hexdigest()[:20]
            relaxed_path = str((relaxed_root / f"{digest}.pdb").resolve())
        args = [
            python_bin, "-c", PYROSETTA_EVAL_SCRIPT, str(pdb_path), str(interface),
            "1" if run_relax else "0", str(relax_repeats),
            "1" if constrain_backbone else "0", relaxed_path,
        ]
        result = run_backend_command(args, backend_timeout(score_config, "pyrosetta", 300), cwd=work_dir)
        text = f"{result.get('stdout_tail', '')}\n{result.get('stderr_tail', '')}"
        parsed = parse_pyrosetta_json(text)
        metrics = parsed.get("metrics") if isinstance(parsed.get("metrics"), Mapping) else {}
        interface_dg = safe_float(metrics.get("interface_dg"))
        shape_complementarity = safe_float(metrics.get("shape_complementarity"))
        total_score = safe_float(metrics.get("total_score"))
        residue_count = safe_float(metrics.get("residue_count"))
        total_score_per_residue = None
        if total_score is not None and residue_count and residue_count > 0:
            total_score_per_residue = total_score / residue_count
        score_parts = []
        if interface_dg is not None:
            score_parts.append(
                score_energy(
                    interface_dg,
                    good=float(config.get("interface_dg_good", -8.0)),
                    bad=float(config.get("interface_dg_bad", 4.0)),
                )
            )
        if shape_complementarity is not None:
            score_parts.append(clamp01(shape_complementarity))
        score = mean(score_parts) if score_parts else (0.0 if not result.get("ok") else 0.5)
        interpretation = "pyrosetta_ran"
        if interface_dg is not None and interface_dg > 50:
            interpretation = "unfavorable_or_unrelaxed_interface; use as a negative sanity signal, not an affinity estimate"
        if total_score_per_residue is not None and total_score_per_residue > 100:
            interpretation = "extreme_positive_total_score; likely severe clashes, missing chemistry, or unrelaxed predicted structure"
        details = {
            **input_details,
            **result,
            "python": python_bin,
            "interface": str(interface),
            "analysis_role": config.get("analysis_role"),
            "parsed_metrics": dict(metrics),
            "parsed_interface_dg": interface_dg,
            "parsed_shape_complementarity": shape_complementarity,
            "parsed_total_score": total_score,
            "parsed_total_score_per_residue": total_score_per_residue,
            "interpretation": interpretation,
            "method": "PyRosetta fa_scorefxn with optional InterfaceAnalyzerMover",
        }
        if backend_required(score_config, "pyrosetta"):
            details["required"] = True
        warnings = []
        if not result.get("ok") or not parsed.get("ok"):
            warnings.append("PyRosetta command failed or import/init did not complete")
        if interface and interface_dg is None:
            warnings.append("PyRosetta InterfaceAnalyzerMover produced no parseable interface dG")
        if interface_dg is not None and interface_dg > 50:
            warnings.append("PyRosetta interface dG is strongly positive; treat as clash/unrelaxed-structure evidence, not binding affinity")
        if total_score_per_residue is not None and total_score_per_residue > 100:
            warnings.append("PyRosetta total_score per residue is extreme; structure likely needs relax/cleanup before quantitative Rosetta interpretation")
        if not interface:
            warnings.append("PyRosetta interface not configured; score uses total_score availability only")
        return ScoreTerm("pyrosetta_interface_energy", "interface_energy", score or 0.0, evaluator_weight(score_config, "eval_pyrosetta", 0.0), details, warnings, backend="pyrosetta", available=bool(result.get("ok") and parsed.get("ok")))
