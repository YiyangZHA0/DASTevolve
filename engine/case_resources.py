

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .design_state import PROJECT_ROOT

def _safe_import_yaml():
    try:
        import yaml

        return yaml
    except Exception:
        return None


def load_memory_yaml(memory_path: Optional[str] = None) -> Dict[str, Any]:


    yaml = _safe_import_yaml()
    if yaml is None:
        return {}

    candidates: List[Path] = []
    if memory_path:
        p = Path(memory_path)
        candidates.append(p if p.is_absolute() else PROJECT_ROOT / p)
    candidates.append(PROJECT_ROOT / "memory.yaml")

    for path in candidates:
        try:
            if path.exists():
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
        except Exception:
            continue
    return {}


def resolve_memory_path(memory_path: Optional[str] = None) -> Path:


    candidates: List[Path] = []
    if memory_path:
        p = Path(memory_path)
        candidates.append(p if p.is_absolute() else PROJECT_ROOT / p)
    candidates.append(PROJECT_ROOT / "memory.yaml")

    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _resolve_case_path(raw_path: Any, design_state_path: Optional[str] = None) -> Optional[Path]:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    if design_state_path:
        base = Path(design_state_path)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        return base.parent / path
    return PROJECT_ROOT / path


def load_case_sheet(state: Dict[str, Any], design_state_path: Optional[str] = None) -> Dict[str, Any]:


    candidates: List[Path] = []
    configured = _resolve_case_path(state.get("case_sheet_path"), design_state_path)
    if configured:
        candidates.append(configured)
    default_sheet = _resolve_case_path("case_sheet.json", design_state_path)
    if default_sheet:
        candidates.append(default_sheet)

    for path in candidates:
        try:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            if path.suffix.lower() in {".yaml", ".yml"}:
                yaml = _safe_import_yaml()
                if yaml is None:
                    continue
                data = yaml.safe_load(text)
            else:
                data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            continue
    return {}


def compact_case_sheet(case_sheet: Dict[str, Any]) -> Dict[str, Any]:


    if not isinstance(case_sheet, dict) or not case_sheet:
        return {}
    return {
        "schema_version": case_sheet.get("schema_version"),
        "case_name": case_sheet.get("case_name"),
        "readiness": case_sheet.get("readiness", {}),
        "design_goal": case_sheet.get("design_goal", {}),
        "state_success_criteria": case_sheet.get("state_success_criteria", {}),
        "residue_level_constraints": case_sheet.get("residue_level_constraints", {}),
        "objective_thresholds": case_sheet.get("objective_thresholds", {}),
        "information_gaps": case_sheet.get("information_gaps", []),
        "provisional_assumptions": case_sheet.get("provisional_assumptions", []),
    }


def extract_memory_bias(memory: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:


    policy = state["mutation_policy"]
    linker_parts = state.get("binder", {}).get("linker_segments", [])
    out: Dict[str, Any] = {
        "preferred_edit_order": list(policy.get("preferred_edit_order", [])),
        "linker_default_sequence": "".join(p[2] for p in linker_parts if len(p) >= 3),
        "linker_gs_min": 0.60,
        "linker_hydrophobic_max": 0.15,
        "linker_charged_max": 0.20,
        "cdr_favored_sparse": ["Y", "W", "H", "N", "Q", "S", "T", "R", "D", "E"],
    }

    try:
        stable = memory.get("stable_priors", {})
        search_prior = stable.get("search_prior", {})
        aa_prior = stable.get("aa_prior", {})
        linker_prior = stable.get("linker_prior", {})
        segment_kind_priors = stable.get("segment_kind_priors", {})

        general = search_prior.get("general", {}) if isinstance(search_prior, dict) else {}
        if isinstance(general.get("preferred_edit_order"), list):
            out["preferred_edit_order"] = general["preferred_edit_order"]

        if isinstance(linker_prior.get("default_sequence"), str):
            out["linker_default_sequence"] = linker_prior["default_sequence"]

        comp_targets = linker_prior.get("composition_targets", {})
        if "gly_ser_fraction_preferred_min" in comp_targets:
            out["linker_gs_min"] = float(comp_targets["gly_ser_fraction_preferred_min"])
        if "hydrophobic_fraction_preferred_max" in comp_targets:
            out["linker_hydrophobic_max"] = float(comp_targets["hydrophobic_fraction_preferred_max"])
        if "charged_fraction_preferred_max" in comp_targets:
            out["linker_charged_max"] = float(comp_targets["charged_fraction_preferred_max"])

        cdr_design = segment_kind_priors.get("cdr", {}) if isinstance(segment_kind_priors, dict) else {}
        if isinstance(cdr_design.get("favored_residues_sparse"), list):
            out["cdr_favored_sparse"] = cdr_design["favored_residues_sparse"]

        subprefs = aa_prior.get("substitution_preferences", {})
        if isinstance(subprefs.get("globally_disfavored_new_positions"), list):
            out["globally_disfavored_new_positions"] = subprefs["globally_disfavored_new_positions"]
    except Exception:
        pass

    return out
