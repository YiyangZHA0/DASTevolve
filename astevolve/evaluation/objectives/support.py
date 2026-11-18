

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        if hasattr(value, "item"):
            value = value.item()
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _clamp01(value: Any) -> float:
    numeric = _safe_float(value, 0.0) or 0.0
    return float(max(0.0, min(1.0, numeric)))


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in values if _safe_float(x) is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _normalize_metric(metric: str, value: Any) -> float:
    val = _safe_float(value, 0.0) or 0.0
    key = str(metric or "").lower()
    if "plddt" in key:
        return _clamp01(val / 100.0 if abs(val) > 1.5 else val)
    if key in {"ptm", "iptm", "ranking_score", "dockq_proxy"}:
        return _clamp01(val)
    if key in {"has_clash", "clash", "clash_count"}:
        return 1.0 - _clamp01(val)
    return _clamp01(val)


_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _one_letter_residue(value: Any) -> str:
    text = str(value or "").upper().strip()
    if len(text) == 1:
        return text if text in "ACDEFGHIKLMNPQRSTVWY" else ""
    return _THREE_TO_ONE.get(text, "")


def _name_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item or "").strip()]
    return [str(value)]


def _aa_class_members(class_name: str) -> str:
    table = {
        "aromatic": "YWHF",
        "polar": "STNQY",
        "polar_uncharged": "STNQY",
        "contextual_charge": "RKHDE",
        "flexible_small": "GSA",
        "positive": "RKH",
        "negative": "DE",
        "acidic": "DE",
        "basic": "RKH",
        "hydrophobic": "AILMFWVY",
        "charged": "RKHDE",
        "small": "GAS",
        "turn_loop": "GSPNDT",
        "amine_near": "DENQ",
        "catechol_near": "YHSTNQDE",
        "aromatic_support": "FYWILV",
        "calcium_ligand": "DEQNST",
    }
    return table.get(str(class_name or "").lower(), "")


def _asym_id(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(alphabet):
        return alphabet[index]
    return f"X{index + 1}"


def _expand_entity_units(state_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    explicit = state_result.get("entity_units")
    if isinstance(explicit, list) and explicit:
        return [dict(unit) for unit in explicit if isinstance(unit, dict)]

    units: List[Dict[str, Any]] = []
    reports = state_result.get("entities", []) or []
    unit_index = 0
    for entity_index, report in enumerate(reports, start=1):
        if not isinstance(report, dict):
            continue
        label = str(report.get("label") or f"entity{entity_index}")
        count = int(_safe_float(report.get("count"), 1) or 1)
        kind = str(report.get("kind") or "")
        for copy_index in range(1, max(1, count) + 1):
            unit_label = label if count == 1 else f"{label}_{copy_index}"
            units.append(
                {
                    "label": unit_label,
                    "base_label": label,
                    "kind": kind,
                    "copy_index": copy_index,
                    "asym_id": _asym_id(unit_index),
                }
            )
            unit_index += 1
    return units


def _kind_aliases(kind: str) -> List[str]:
    value = str(kind or "").lower()
    aliases = [value]
    if "protein" in value:
        aliases.extend(["protein", "proteinchain"])
    if "dna" in value:
        aliases.extend(["dna", "dnasequence"])
    if "rna" in value:
        aliases.extend(["rna", "rnasequence"])
    if "ligand" in value:
        aliases.extend(["ligand", "small_molecule", "molecule"])
    if "ion" in value:
        aliases.append("ion")
    return aliases


def _selector_asym_ids(state_result: Dict[str, Any], selector: Any) -> List[str]:


    units = _expand_entity_units(state_result)
    if selector is None:
        return [str(unit.get("asym_id")) for unit in units if unit.get("asym_id")]
    if isinstance(selector, (list, tuple, set)):
        out: List[str] = []
        for item in selector:
            for asym in _selector_asym_ids(state_result, item):
                if asym not in out:
                    out.append(asym)
        return out

    raw = str(selector).strip()
    if not raw:
        return []
    low = raw.lower()
    out: List[str] = []
    for unit in units:
        asym = str(unit.get("asym_id") or "")
        candidates = {
            asym.lower(),
            str(unit.get("label") or "").lower(),
            str(unit.get("base_label") or "").lower(),
            str(unit.get("source_chain") or "").lower(),
        }
        candidates.update(_kind_aliases(str(unit.get("kind") or "")))
        label_text = " ".join(candidates)
        if low in candidates or (len(low) >= 3 and low in label_text):
            if asym and asym not in out:
                out.append(asym)
    if out:
        return out
    return [raw] if len(raw) <= 3 else []


def _pair_key(left: str, right: str, pairs: Dict[str, Any]) -> Optional[str]:
    direct = f"{left}:{right}"
    reverse = f"{right}:{left}"
    if direct in pairs:
        return direct
    if reverse in pairs:
        return reverse
    return None


def _region_indices_from_spec(spec: Dict[str, Any]) -> Optional[set[int]]:
    values = spec.get("indices")
    if values is None:
        values = spec.get("residues")
    if values is not None:
        try:
            out = {int(v) for v in values}
        except (TypeError, ValueError):
            out = set()
        if not bool(spec.get("zero_based", False)) and bool(spec.get("one_based", False)):
            out = {idx - 1 for idx in out if idx > 0}
        return out

    spans = spec.get("spans", spec.get("ranges", spec.get("residue_ranges")))
    if spans is None:
        return None
    out: set[int] = set()
    for raw in spans:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        try:
            start = int(raw[0])
            end = int(raw[1])
        except (TypeError, ValueError):
            continue
        if bool(spec.get("one_based", False)) and not bool(spec.get("zero_based", False)):
            start -= 1
            end -= 1
        for idx in range(max(0, start), max(0, end)):
            out.add(idx)
    return out


def _unit_region_candidates(unit: Dict[str, Any]) -> set[str]:
    candidates = {
        str(unit.get("asym_id") or "").lower(),
        str(unit.get("label") or "").lower(),
        str(unit.get("base_label") or "").lower(),
        str(unit.get("source_chain") or "").lower(),
        str(unit.get("kind") or "").lower(),
    }
    candidates.update(_kind_aliases(str(unit.get("kind") or "")))
    return {item for item in candidates if item}


def _region_filters_by_asym(
    state_result: Dict[str, Any],
    region_specs: Optional[List[Dict[str, Any]]],
) -> Dict[str, Optional[set[int]]]:


    if not region_specs:
        return {}
    units = _expand_entity_units(state_result)
    filters: Dict[str, Optional[set[int]]] = {}
    for spec in region_specs:
        if not isinstance(spec, dict):
            continue
        chain_id = str(
            spec.get("chain_id")
            or spec.get("source_chain")
            or spec.get("chain")
            or spec.get("asym_id")
            or spec.get("entity")
            or ""
        ).strip()
        wanted = chain_id.lower()
        indices = _region_indices_from_spec(spec)
        matched = False
        for unit in units:
            asym = str(unit.get("asym_id") or "")
            if not asym:
                continue
            if wanted and wanted not in _unit_region_candidates(unit):
                continue
            matched = True
            if indices is None:
                filters[asym] = None
            elif asym not in filters:
                filters[asym] = set(indices)
            elif filters.get(asym) is not None:
                filters.setdefault(asym, set()).update(indices)
        if not matched and chain_id and len(chain_id) <= 3:
            if indices is None:
                filters[chain_id] = None
            elif chain_id not in filters:
                filters[chain_id] = set(indices)
            elif filters.get(chain_id) is not None:
                filters.setdefault(chain_id, set()).update(indices)
    return filters


def _residue_allowed(residue: Dict[str, Any], filters: Dict[str, Optional[set[int]]]) -> bool:
    if not filters:
        return True
    chain = str(residue.get("chain") or "")
    if chain not in filters:
        return False
    allowed = filters[chain]
    if allowed is None:
        return True
    try:
        idx = int(residue.get("residue") or 0) - 1
    except (TypeError, ValueError):
        return False
    return idx in allowed


def _coverage(filters: Dict[str, Optional[set[int]]], contacted: set[Tuple[str, int]]) -> Optional[float]:
    if not filters:
        return None
    total = 0
    for residues in filters.values():
        if residues is not None:
            total += len(residues)
    if total <= 0:
        return None
    covered = 0
    for chain, zero_idx in contacted:
        allowed = filters.get(chain)
        if allowed is not None and zero_idx in allowed:
            covered += 1
    return _clamp01(float(covered) / float(total))


def _state_names(spec: Dict[str, Any]) -> List[str]:
    if spec.get("states") is not None:
        states = spec.get("states")
        return [str(s) for s in states] if isinstance(states, list) else [str(states)]
    if spec.get("state") is not None:
        return [str(spec.get("state"))]
    return []
