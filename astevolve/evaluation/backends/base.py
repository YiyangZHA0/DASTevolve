

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from astevolve.evaluation.contracts import ScoreTerm
from astevolve.evaluation.support import (
    as_list,
    clamp01,
    evaluator_weight,
    safe_float,
    safe_int,
)

try:
    from astevolve.metrics.structure import _load_atoms
except Exception:
    _load_atoms = None
OPTIONAL_EXTERNAL_BACKENDS = {
    "rosetta",
    "pyrosetta",
    "getcontacts",
    "ipsae",
    "foldx",
    "fpocket",
    "povme",
}


def cif_atoms(cif_path: Optional[str]) -> List[Dict[str, Any]]:


    if not cif_path or _load_atoms is None:
        return []
    try:
        path = Path(str(cif_path))
        if not path.exists():
            return []
        return list(_load_atoms(str(path)))
    except Exception:
        return []


def backend_config(score_config: Mapping[str, Any]) -> Mapping[str, Any]:


    raw = score_config.get("evaluator_backends", {}) if isinstance(score_config, Mapping) else {}
    return raw if isinstance(raw, Mapping) else {}


def backend_enabled(score_config: Mapping[str, Any], name: str) -> bool:


    config = backend_config(score_config)
    raw = config.get(name)
    if isinstance(raw, Mapping) and "enabled" in raw:
        return str(raw.get("enabled")).lower() in {"1", "true", "yes", "on"}
    if raw is not None:
        return str(raw).lower() in {"1", "true", "yes", "on"}
    env_key = f"ASTEVOLVE_EVAL_{name.upper()}"
    return str(os.environ.get(env_key, "")).lower() in {"1", "true", "yes", "on"}


def backend_required(score_config: Mapping[str, Any], name: str) -> bool:


    raw = backend_item(score_config, name).get("required")
    if raw is None:
        raw = score_config.get(f"{name}_required")
    if raw is None:
        raw = os.environ.get(f"ASTEVOLVE_REQUIRE_{name.upper()}")
    return str(raw or "").lower() in {"1", "true", "yes", "on"}


def backend_timeout(score_config: Mapping[str, Any], name: str, default: int = 180) -> int:


    config = backend_config(score_config)
    raw = config.get(name) if isinstance(config, Mapping) else None
    if isinstance(raw, Mapping) and raw.get("timeout") is not None:
        return max(1, safe_int(raw.get("timeout"), default))
    return max(1, safe_int(score_config.get(f"{name}_timeout"), default))


def backend_item(score_config: Mapping[str, Any], name: str) -> Mapping[str, Any]:


    config = backend_config(score_config)
    raw = config.get(name) if isinstance(config, Mapping) else None
    return raw if isinstance(raw, Mapping) else {}


def backend_extra_args(score_config: Mapping[str, Any], name: str) -> List[str]:


    raw = backend_item(score_config, name).get("extra_args")
    if raw is None:
        raw = score_config.get(f"{name}_extra_args")
    if isinstance(raw, str):
        return [part for part in raw.split() if part]
    return [str(part) for part in as_list(raw) if str(part)]


def backend_command(
    score_config: Mapping[str, Any],
    name: str,
    env_key: str,
    candidates: Sequence[str],
) -> Optional[str]:


    config = backend_config(score_config)
    raw = config.get(name) if isinstance(config, Mapping) else None
    configured = raw.get("command") if isinstance(raw, Mapping) else None
    configured = configured or os.environ.get(env_key)
    if configured:
        path = shutil.which(str(configured)) or str(configured)
        if Path(path).exists() or shutil.which(path):
            return path
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    return None


def backend_env_command(
    score_config: Mapping[str, Any],
    name: str,
    env_keys: Sequence[str],
    candidates: Sequence[str],
) -> Optional[str]:


    for env_key in env_keys:
        command = backend_command(score_config, name, env_key, candidates)
        if command:
            return command
    return backend_command(score_config, name, "", candidates)


def run_backend_command(
    command: Sequence[str],
    timeout: int,
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:


    try:
        process = subprocess.run(
            [str(item) for item in command],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": process.returncode == 0,
            "returncode": process.returncode,
            "stdout_tail": process.stdout[-4000:],
            "stderr_tail": process.stderr[-4000:],
            "command": [str(item) for item in command],
        }
    except subprocess.TimeoutExpired as error:
        return {
            "ok": False,
            "returncode": None,
            "timeout": timeout,
            "stdout_tail": (error.stdout or "")[-4000:] if isinstance(error.stdout, str) else "",
            "stderr_tail": (error.stderr or "")[-4000:] if isinstance(error.stderr, str) else "",
            "command": [str(item) for item in command],
            "error": "timeout",
        }
    except Exception as error:
        return {"ok": False, "command": [str(item) for item in command], "error": str(error)}


def backend_unavailable_term(
    name: str,
    category: str,
    score_config: Mapping[str, Any],
    weight_key: str,
    reason: str,
    warning: str,
) -> ScoreTerm:


    required = backend_required(score_config, name)
    details = {"enabled": True, "required": required, "reason": reason}
    if required:
        details["dimension"] = "correctness"
    else:
        details["ignored_for_score"] = True
        details["ignored_for_score_reason"] = "optional_backend_unavailable"
    term_name = f"{name}_required_backend" if required else f"{name}_backend"
    weight = evaluator_weight(score_config, weight_key, 0.0) if required else 0.0
    return ScoreTerm(
        term_name,
        category,
        0.0,
        weight,
        details,
        [warning],
        backend=name,
        available=False,
    )

def neutralize_optional_unavailable_backend(term: ScoreTerm) -> ScoreTerm:


    if term.available or term.backend not in OPTIONAL_EXTERNAL_BACKENDS:
        return term
    details = dict(term.details) if isinstance(term.details, Mapping) else {}
    if details.get("enabled") is False:
        return term
    if bool(details.get("required", False)):
        return term
    if float(term.weight) == 0.0 and details.get("ignored_for_score"):
        return term
    details["ignored_for_score"] = True
    details["ignored_for_score_reason"] = "optional_backend_unavailable"
    return ScoreTerm(
        term.name,
        term.category,
        term.score,
        0.0,
        details,
        list(term.warnings),
        backend=term.backend,
        available=term.available,
    )


def _is_xyz(value: Any) -> bool:


    return isinstance(value, tuple) and len(value) == 3


def write_simple_pdb_from_cif(cif_path: str, pdb_path: Path) -> bool:


    atoms = cif_atoms(cif_path)
    if not atoms:
        return False
    chain_map: Dict[str, str] = {}
    chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    lines = []
    serial = 1
    for atom in atoms:
        if str(atom.get("group", "ATOM")).upper() not in {"ATOM", "HETATM"}:
            continue
        xyz = atom.get("xyz")
        if not _is_xyz(xyz):
            continue
        asymmetry = str(atom.get("asym") or "A")
        if asymmetry not in chain_map:
            chain_map[asymmetry] = chain_letters[len(chain_map) % len(chain_letters)]
        record = "HETATM" if str(atom.get("group")).upper() == "HETATM" else "ATOM  "
        atom_name = str(atom.get("atom") or "X")[:4]
        residue_name = str(atom.get("comp") or "UNK")[:3]
        chain_id = chain_map[asymmetry]
        residue_id = safe_int(atom.get("seq_id"), 1)
        b_factor = safe_float(atom.get("b"), 0.0) or 0.0
        lines.append(
            f"{record}{serial:5d} {atom_name:<4s} {residue_name:>3s} {chain_id:1s}{residue_id:4d}    "
            f"{float(xyz[0]):8.3f}{float(xyz[1]):8.3f}{float(xyz[2]):8.3f}"
            f"{1.00:6.2f}{b_factor:6.2f}           {atom_name[0]:>2s}"
        )
        serial += 1
    if not lines:
        return False
    pdb_path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    return True


def first_existing_structure_path(
    structure: Mapping[str, Any],
) -> Tuple[Optional[Path], Dict[str, Any]]:


    checked: List[str] = []
    for key in ("pdb_path", "cif_path", "structure_path"):
        raw = structure.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        checked.append(f"{key}={path}")
        if path.is_file() and path.suffix.lower() in {".pdb", ".ent", ".cif"}:
            return path, {"source_key": key, "checked": checked}

    for key in ("structure_out_dir", "protenix_out_dir", "out_dir", "protenix_output_dir"):
        raw = structure.get(key)
        if not raw:
            continue
        root = Path(str(raw))
        checked.append(f"{key}={root}")
        if not root.is_dir():
            continue
        for pattern in ("*.pdb", "*.ent", "*.cif"):
            matches = sorted(root.rglob(pattern))
            if matches:
                return matches[0], {"source_key": key, "checked": checked}

    for key in ("structure_summary_json", "protenix_summary_json", "summary_json"):
        raw = structure.get(key)
        if not raw:
            continue
        summary = Path(str(raw))
        checked.append(f"{key}={summary}")
        root = summary.parent
        if not root.is_dir():
            continue
        for pattern in ("*.pdb", "*.ent", "*.cif"):
            matches = sorted(root.rglob(pattern))
            if matches:
                return matches[0], {"source_key": key, "checked": checked}
    return None, {"reason": "no CIF/PDB path available", "checked": checked}


def structure_input_for_backend(
    structure: Mapping[str, Any],
    work_dir: Path,
) -> Tuple[Optional[Path], Dict[str, Any]]:


    path, details = first_existing_structure_path(structure)
    if path is None:
        return None, details


    path = path.expanduser().resolve()
    if path.suffix.lower() == ".pdb":
        return path, {**details, "input_path": str(path), "converted": False}
    pdb_path = work_dir / f"{path.stem}.pdb"
    if write_simple_pdb_from_cif(str(path), pdb_path):
        return pdb_path, {**details, "input_path": str(path), "converted": True, "pdb_path": str(pdb_path)}
    return None, {**details, "reason": "failed to convert CIF to simple PDB", "input_path": str(path)}


def parse_float_after(text: str, keys: Sequence[str]) -> Optional[float]:


    lower = text.lower()
    for key in keys:
        index = lower.find(key.lower())
        if index < 0:
            continue
        window = text[index : index + 240].replace("=", " ").replace(":", " ").replace(",", " ")
        for token in window.split():
            value = safe_float(token)
            if value is not None:
                return value
    return None


def score_energy(value: Optional[float], good: float = -8.0, bad: float = 4.0) -> float:


    if value is None:
        return 0.0
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return clamp01(1.0 - ((value - good) / max(1e-8, bad - good)))
