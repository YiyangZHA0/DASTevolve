

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from astevolve.metrics.structure import _load_structure_atoms


AA_NEGATIVE = {"ASP", "GLU"}
AA_POSITIVE = {"ARG", "LYS", "HIS"}
AA_AROMATIC = {"PHE", "TYR", "TRP", "HIS"}
AA_HYDROPHOBIC = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TYR", "TRP", "PRO"}
AA_POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "ASP", "GLU", "LYS", "ARG", "HIS", "TRP"}
NEGATIVE_ATOMS = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
}
POSITIVE_ATOMS = {
    "ARG": {"NE", "NH1", "NH2"},
    "LYS": {"NZ"},
    "HIS": {"ND1", "NE2"},
}


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
    number = _safe_float(value, 0.0) or 0.0
    return max(0.0, min(1.0, float(number)))


def _score_at_least(value: Any, good: float, bad: float = 0.0) -> Optional[float]:
    number = _safe_float(value)
    if number is None:
        return None
    if number >= good:
        return 1.0
    if number <= bad:
        return 0.0
    return _clamp01((number - bad) / max(1e-8, good - bad))


def _score_at_most(value: Any, good: float, bad: float) -> Optional[float]:
    number = _safe_float(value)
    if number is None:
        return None
    if number <= good:
        return 1.0
    if number >= bad:
        return 0.0
    return _clamp01(1.0 - ((number - good) / max(1e-8, bad - good)))


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(value) for value in values if value is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2 + (float(a[2]) - float(b[2])) ** 2)


def _atom_name(atom: Mapping[str, Any]) -> str:
    return str(atom.get("atom") or "").strip().upper()


def _resname(atom: Mapping[str, Any]) -> str:
    return str(atom.get("comp") or atom.get("resname") or "").strip().upper()


def _element(atom: Mapping[str, Any]) -> str:
    explicit = str(atom.get("element") or "").strip().upper()
    if explicit:
        return explicit[:2].strip()
    name = _atom_name(atom)
    while name and name[0].isdigit():
        name = name[1:]
    if name.startswith(("MG", "ZN", "CA", "MN", "FE", "CU", "CO", "NI")):
        return name[:2]
    return name[:1]


def _is_hydrogen(atom: Mapping[str, Any]) -> bool:
    return _element(atom) in {"H", "D"} or _atom_name(atom).startswith(("H", "D"))


def _is_heavy(atom: Mapping[str, Any]) -> bool:
    return bool(atom.get("xyz")) and not _is_hydrogen(atom)


def _is_protein_atom(atom: Mapping[str, Any]) -> bool:
    return str(atom.get("group") or "").upper() == "ATOM" and _is_heavy(atom)


def _is_ligand_atom(atom: Mapping[str, Any]) -> bool:
    return str(atom.get("group") or "").upper() == "HETATM" and _is_heavy(atom)


def _is_polar_atom(atom: Mapping[str, Any]) -> bool:
    return _element(atom) in {"N", "O", "S"}


def _is_negative_atom(atom: Mapping[str, Any]) -> bool:
    return _resname(atom) in NEGATIVE_ATOMS and _atom_name(atom) in NEGATIVE_ATOMS[_resname(atom)]


def _is_positive_atom(atom: Mapping[str, Any]) -> bool:
    return _resname(atom) in POSITIVE_ATOMS and _atom_name(atom) in POSITIVE_ATOMS[_resname(atom)]


def _unit_map(state: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for unit in state.get("entity_units", []) or []:
        if not isinstance(unit, Mapping):
            continue
        asym = str(unit.get("asym_id") or unit.get("chain") or unit.get("chain_id") or "").strip()
        if asym:
            out[asym] = dict(unit)
    return out


def _atom_haystack(atom: Mapping[str, Any], units: Mapping[str, Mapping[str, Any]]) -> str:
    asym = str(atom.get("asym") or "")
    unit = units.get(asym, {})
    fields = [
        atom.get("group"),
        atom.get("atom"),
        atom.get("comp"),
        atom.get("asym"),
        atom.get("entity"),
        unit.get("source_chain"),
        unit.get("base_label"),
        unit.get("label"),
        unit.get("kind"),
        unit.get("type"),
    ]
    return " ".join(str(item) for item in fields if item is not None).lower()


def _matches_tokens(text: str, tokens: Sequence[str]) -> bool:
    wanted = [str(token).lower() for token in tokens if str(token).strip()]
    return bool(wanted) and any(token in text for token in wanted)


def _state_name_blob(state: Mapping[str, Any]) -> str:
    return " ".join(str(state.get(key, "")) for key in ("name", "role", "objective")).lower()


def _state_structure_path(state: Mapping[str, Any]) -> str:
    summary = state.get("structure_metrics") if isinstance(state.get("structure_metrics"), Mapping) else {}
    for source in (summary, state):
        if not isinstance(source, Mapping):
            continue
        for key in ("cif_path", "pdb_path", "structure_path"):
            value = source.get(key)
            if value and Path(str(value)).exists():
                return str(value)
    return ""


def _iter_states(structure: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    states = structure.get("states")
    if isinstance(states, list):
        return [state for state in states if isinstance(state, Mapping)]
    return []


def _residue_label(atom: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "chain": str(atom.get("asym") or atom.get("chain") or ""),
        "position": atom.get("seq_id"),
        "resname": _resname(atom),
        "atom": _atom_name(atom),
    }


def ligand_state_geometry(
    structure: Mapping[str, Any],
    *,
    state_tokens: Sequence[str],
    ligand_tokens: Sequence[str],
    contact_cutoff: float = 4.5,
    hbond_cutoff: float = 3.6,
    salt_cutoff: float = 4.2,
    clash_cutoff: float = 2.0,
    max_examples: int = 10,
) -> Dict[str, Any]:


    state_reports: List[Dict[str, Any]] = []
    for state in _iter_states(structure):
        state_blob = _state_name_blob(state)
        if state_tokens and not _matches_tokens(state_blob, state_tokens):
            continue
        path = _state_structure_path(state)
        if not path:
            state_reports.append({"state": state.get("name"), "available": False, "reason": "no structure path"})
            continue
        units = _unit_map(state)
        atoms = _load_structure_atoms(path)
        ligand_atoms = [
            atom
            for atom in atoms
            if _is_ligand_atom(atom) and _matches_tokens(_atom_haystack(atom, units), ligand_tokens)
        ]
        protein_atoms = [atom for atom in atoms if _is_protein_atom(atom)]
        if not ligand_atoms or not protein_atoms:
            state_reports.append(
                {
                    "state": state.get("name"),
                    "available": False,
                    "reason": "no ligand or protein atoms matched token selectors",
                    "structure_path": path,
                    "ligand_tokens": list(ligand_tokens),
                }
            )
            continue
        residue_contacts: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
        hbond_like = 0
        acidic_amine_like = 0
        aromatic_support = 0
        hydrophobic_contacts = 0
        clashes = 0
        closest = None
        examples: List[Dict[str, Any]] = []
        ligand_polar_contacts = 0
        ligand_n_contacts = 0
        ligand_o_contacts = 0
        for lig in ligand_atoms:
            lig_element = _element(lig)
            for prot in protein_atoms:
                distance = _dist(lig["xyz"], prot["xyz"])
                if closest is None or distance < closest:
                    closest = distance
                if distance <= clash_cutoff:
                    clashes += 1
                if distance > contact_cutoff:
                    continue
                key = (str(prot.get("asym") or ""), int(prot.get("seq_id") or 0), _resname(prot))
                item = residue_contacts.setdefault(key, {"count": 0, "min_distance": distance})
                item["count"] += 1
                item["min_distance"] = min(float(item["min_distance"]), distance)
                if _is_polar_atom(lig) and _is_polar_atom(prot) and distance <= hbond_cutoff:
                    hbond_like += 1
                    ligand_polar_contacts += 1
                    if lig_element == "N":
                        ligand_n_contacts += 1
                    if lig_element == "O":
                        ligand_o_contacts += 1
                if lig_element == "N" and _is_negative_atom(prot) and distance <= salt_cutoff:
                    acidic_amine_like += 1
                if _resname(prot) in AA_AROMATIC and distance <= contact_cutoff:
                    aromatic_support += 1
                if _resname(prot) in AA_HYDROPHOBIC and _element(prot) in {"C", "S"} and distance <= contact_cutoff:
                    hydrophobic_contacts += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "ligand": _residue_label(lig),
                            "protein": _residue_label(prot),
                            "distance": round(float(distance), 3),
                            "classes": [
                                name
                                for name, ok in (
                                    ("polar_hbond_like", _is_polar_atom(lig) and _is_polar_atom(prot) and distance <= hbond_cutoff),
                                    ("acidic_amine_like", lig_element == "N" and _is_negative_atom(prot) and distance <= salt_cutoff),
                                    ("aromatic_support", _resname(prot) in AA_AROMATIC),
                                    ("hydrophobic_contact", _resname(prot) in AA_HYDROPHOBIC and _element(prot) in {"C", "S"}),
                                )
                                if ok
                            ],
                        }
                    )
        contact_residues = len(residue_contacts)
        score = _mean(
            [
                _score_at_least(contact_residues, 4.0, 0.0),
                _score_at_least(hbond_like, 2.0, 0.0),
                _score_at_least(acidic_amine_like, 1.0, 0.0),
                _score_at_least(aromatic_support, 2.0, 0.0),
                _score_at_most(clashes, 0.0, 6.0),
            ]
        )
        state_reports.append(
            {
                "state": state.get("name"),
                "available": True,
                "structure_path": path,
                "ligand_atom_count": len(ligand_atoms),
                "protein_atom_count": len(protein_atoms),
                "contact_residue_count": contact_residues,
                "atom_contact_count": sum(int(item["count"]) for item in residue_contacts.values()),
                "hbond_like_count": hbond_like,
                "ligand_polar_contact_count": ligand_polar_contacts,
                "ligand_n_contact_count": ligand_n_contacts,
                "ligand_o_contact_count": ligand_o_contacts,
                "acidic_amine_like_count": acidic_amine_like,
                "aromatic_support_count": aromatic_support,
                "hydrophobic_contact_count": hydrophobic_contacts,
                "ligand_clash_count": clashes,
                "closest_ligand_protein_distance": None if closest is None else round(float(closest), 3),
                "score": score,
                "examples": sorted(examples, key=lambda item: item["distance"])[:max_examples],
                "method": "ligand/protein heavy-atom distance proxy; Rosetta/PyRosetta backend should be preferred when available for energetic interpretation",
            }
        )
    available = [item for item in state_reports if item.get("available")]
    if not available:
        return {
            "available": False,
            "reason": "no matching ligand-state atom geometry available",
            "states": state_reports,
            "state_tokens": list(state_tokens),
            "ligand_tokens": list(ligand_tokens),
        }
    return {
        "available": True,
        "states": state_reports,
        "best_state_score": max(float(item.get("score") or 0.0) for item in available),
        "mean_state_score": _mean([_safe_float(item.get("score")) for item in available]),
        "total_contact_residue_count": sum(int(item.get("contact_residue_count") or 0) for item in available),
        "total_hbond_like_count": sum(int(item.get("hbond_like_count") or 0) for item in available),
        "total_acidic_amine_like_count": sum(int(item.get("acidic_amine_like_count") or 0) for item in available),
        "total_aromatic_support_count": sum(int(item.get("aromatic_support_count") or 0) for item in available),
        "total_ligand_clash_count": sum(int(item.get("ligand_clash_count") or 0) for item in available),
    }


def metal_coordination_geometry(
    structure: Mapping[str, Any],
    *,
    state_tokens: Sequence[str],
    metal_tokens: Sequence[str] = ("mg",),
    coordination_cutoff: float = 2.8,
    max_examples: int = 10,
) -> Dict[str, Any]:


    state_reports: List[Dict[str, Any]] = []
    for state in _iter_states(structure):
        if state_tokens and not _matches_tokens(_state_name_blob(state), state_tokens):
            continue
        path = _state_structure_path(state)
        if not path:
            state_reports.append({"state": state.get("name"), "available": False, "reason": "no structure path"})
            continue
        units = _unit_map(state)
        atoms = _load_structure_atoms(path)
        metals = [
            atom
            for atom in atoms
            if _is_ligand_atom(atom)
            and (_matches_tokens(_atom_haystack(atom, units), metal_tokens) or _element(atom) in {"MG", "ZN", "CA", "MN"})
        ]
        acceptors = [atom for atom in atoms if atom not in metals and _is_heavy(atom) and _element(atom) in {"O", "N", "S"}]
        examples: List[Dict[str, Any]] = []
        coordination_count = 0
        protein_coordination_count = 0
        ligand_coordination_count = 0
        for metal in metals:
            for atom in acceptors:
                distance = _dist(metal["xyz"], atom["xyz"])
                if distance > coordination_cutoff:
                    continue
                coordination_count += 1
                if _is_protein_atom(atom):
                    protein_coordination_count += 1
                elif _is_ligand_atom(atom):
                    ligand_coordination_count += 1
                if len(examples) < max_examples:
                    examples.append({"metal": _residue_label(metal), "partner": _residue_label(atom), "distance": round(distance, 3)})
        state_reports.append(
            {
                "state": state.get("name"),
                "available": bool(metals),
                "structure_path": path,
                "metal_atom_count": len(metals),
                "coordination_count": coordination_count,
                "protein_coordination_count": protein_coordination_count,
                "ligand_coordination_count": ligand_coordination_count,
                "coordination_cutoff": coordination_cutoff,
                "examples": examples,
            }
        )
    available = [item for item in state_reports if item.get("available")]
    if not available:
        return {"available": False, "reason": "no matching metal atoms available", "states": state_reports}
    return {
        "available": True,
        "states": state_reports,
        "coordination_count": sum(int(item.get("coordination_count") or 0) for item in available),
        "protein_coordination_count": sum(int(item.get("protein_coordination_count") or 0) for item in available),
        "ligand_coordination_count": sum(int(item.get("ligand_coordination_count") or 0) for item in available),
        "rejection_score": _score_at_most(sum(int(item.get("coordination_count") or 0) for item in available), 0.0, 8.0),
        "method": "Mg/metal to O/N/S heavy-atom distance proxy; low coordination is desired for CTC/Mg source rejection",
    }


def state_interface_summary(
    structure: Mapping[str, Any],
    *,
    state_tokens: Sequence[str],
    pair_tokens: Sequence[str],
    max_examples: int = 8,
) -> Dict[str, Any]:


    pairs_out: List[Dict[str, Any]] = []
    contact_count = 0
    residue_pair_count = 0
    clash_count = 0
    for state_name, pair_name, pair in _all_interface_pairs_like(structure):
        blob = f"{state_name} {pair_name}".lower()
        if state_tokens and not _matches_tokens(state_name.lower(), state_tokens) and not _matches_tokens(blob, state_tokens):
            continue
        if pair_tokens and not _matches_tokens(blob, pair_tokens):
            continue
        contacts = int(pair.get("contact_count") or 0)
        residue_pairs = int(pair.get("residue_pair_count") or len(pair.get("residue_pairs") or []))
        clashes = int(pair.get("clash_count") or 0)
        contact_count += contacts
        residue_pair_count += residue_pairs
        clash_count += clashes
        if len(pairs_out) < max_examples:
            pairs_out.append(
                {
                    "state": state_name,
                    "pair": pair_name,
                    "contact_count": contacts,
                    "residue_pair_count": residue_pairs,
                    "clash_count": clashes,
                    "interface_plddt_mean": pair.get("interface_plddt_mean"),
                }
            )
    return {
        "available": bool(pairs_out),
        "pair_count": len(pairs_out),
        "contact_count": contact_count,
        "residue_pair_count": residue_pair_count,
        "clash_count": clash_count,
        "examples": pairs_out,
    }


def _all_interface_pairs_like(structure: Mapping[str, Any]) -> Iterable[Tuple[str, str, Mapping[str, Any]]]:
    states = structure.get("states") if isinstance(structure.get("states"), list) else []
    for state in states:
        if not isinstance(state, Mapping):
            continue
        state_name = str(state.get("name") or "state")
        summary = state.get("structure_metrics", {}) if isinstance(state.get("structure_metrics"), Mapping) else {}
        interface = summary.get("interface", {}) if isinstance(summary.get("interface"), Mapping) else {}
        pairs = interface.get("pairs", {}) if isinstance(interface.get("pairs"), Mapping) else {}
        for pair_name, pair in pairs.items():
            if isinstance(pair, Mapping):
                yield state_name, str(pair_name), pair
    interface = structure.get("interface", {}) if isinstance(structure.get("interface"), Mapping) else {}
    pairs = interface.get("pairs", {}) if isinstance(interface.get("pairs"), Mapping) else {}
    for pair_name, pair in pairs.items():
        if isinstance(pair, Mapping):
            yield "aggregate", str(pair_name), pair


def backend_evidence_summary(terms: Sequence[Any]) -> Dict[str, Any]:


    backends: Dict[str, Any] = {}
    priority = ["rosetta", "pyrosetta", "getcontacts", "ipsae", "foldx", "fpocket", "povme"]
    for term in terms:
        backend = str(getattr(term, "backend", "") or "")
        if backend not in priority:
            continue
        details = getattr(term, "details", {}) if isinstance(getattr(term, "details", {}), Mapping) else {}
        item = {
            "term": getattr(term, "name", ""),
            "available": bool(getattr(term, "available", False)),
            "score": _safe_float(getattr(term, "score", None)),
            "weight": _safe_float(getattr(term, "weight", None)),
            "analysis_role": details.get("analysis_role"),
            "warnings": list(getattr(term, "warnings", []) or [])[:6],
        }
        if backend in {"rosetta", "pyrosetta"}:
            item["interface_dg"] = details.get("parsed_interface_dg")
            item["shape_complementarity"] = details.get("parsed_shape_complementarity")
            item["total_score"] = details.get("parsed_total_score")
            item["total_score_per_residue"] = details.get("parsed_total_score_per_residue")
            item["interface"] = details.get("interface")
            item["interpretation"] = details.get("interpretation")
        if backend == "getcontacts":
            item["contact_counts"] = details.get("parsed_contact_counts")
            item["targets"] = details.get("targets")
        if backend == "ipsae":
            item["ipSAE"] = details.get("ipSAE")
            item["pDockQ"] = details.get("pDockQ")
            item["pDockQ2"] = details.get("pDockQ2")
            item["LIS"] = details.get("LIS")
            item["selected_row"] = details.get("selected_row")
        if backend in {"fpocket", "povme"}:
            item["pocket_volume"] = details.get("parsed_pocket_volume") or details.get("parsed_volume")
            item["pocket_score"] = details.get("parsed_pocket_score")
        if not item["available"]:
            item["reason"] = details.get("reason")
        backends[backend] = item
    return {
        "available_backends": [name for name in priority if backends.get(name, {}).get("available")],
        "missing_enabled_backends": [
            name
            for name in priority
            if name in backends and not backends[name].get("available") and backends[name].get("weight", 0.0) != 0.0
        ],
        "preferred_energy_backend": "rosetta",
        "rosetta_usage_note": "Use Rosetta/PyRosetta interface energy and shape complementarity as the strongest optional physics evidence when available; absence means deployment missing, not biological failure unless marked required.",
        "backends": backends,
    }


def case_specific_terms_summary(terms: Sequence[Any], prefixes: Sequence[str], keep_detail_keys: Sequence[str], limit: int = 16) -> Dict[str, Any]:


    rows: List[Dict[str, Any]] = []
    for term in terms:
        name = str(getattr(term, "name", "") or "")
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        details = getattr(term, "details", {}) if isinstance(getattr(term, "details", {}), Mapping) else {}
        rows.append(
            {
                "name": name,
                "category": getattr(term, "category", ""),
                "score": _safe_float(getattr(term, "score", None)),
                "weight": _safe_float(getattr(term, "weight", None)),
                "available": bool(getattr(term, "available", False)),
                "backend": getattr(term, "backend", ""),
                "details": {key: details.get(key) for key in keep_detail_keys if key in details},
                "warnings": list(getattr(term, "warnings", []) or [])[:6],
            }
        )
    rows.sort(key=lambda item: (float(item.get("score") or 0.0), -float(item.get("weight") or 0.0)))
    return {"available": bool(rows), "terms": rows[:limit]}
