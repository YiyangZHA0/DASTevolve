

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

from astevolve.evaluation.backends.base import cif_atoms
from astevolve.evaluation.support import safe_float
from astevolve.evaluation.views import all_interface_pairs

from .fold import atom_name, dist2, element, is_hbond_heavy_atom, is_sidechain_heavy_atom, is_xyz


AA_POSITIVE = {"ARG", "LYS", "HIS"}
AA_NEGATIVE = {"ASP", "GLU"}
AA_HYDROPHOBIC = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR", "PRO"}
AA_POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "ASP", "GLU", "LYS", "ARG", "HIS", "TRP"}
CHARGED_ATOMS = {
    "ARG": {"NE", "NH1", "NH2"},
    "LYS": {"NZ"},
    "HIS": {"ND1", "NE2"},
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
}


def is_hydrophobic_atom(atom: Mapping[str, Any]) -> bool:


    return (
        str(atom.get("comp") or "").upper() in AA_HYDROPHOBIC
        and is_sidechain_heavy_atom(atom)
        and element(atom.get("atom")) in {"C", "S"}
    )


def is_positive_atom(atom: Mapping[str, Any]) -> bool:


    residue = str(atom.get("comp") or "").upper()
    return residue in AA_POSITIVE and atom_name(atom) in CHARGED_ATOMS.get(residue, set())


def is_negative_atom(atom: Mapping[str, Any]) -> bool:


    residue = str(atom.get("comp") or "").upper()
    return residue in AA_NEGATIVE and atom_name(atom) in CHARGED_ATOMS.get(residue, set())


def classify_pair(residue_pair: Mapping[str, Any]) -> Dict[str, int]:


    left = residue_pair.get("left", {}) if isinstance(residue_pair.get("left"), Mapping) else {}
    right = residue_pair.get("right", {}) if isinstance(residue_pair.get("right"), Mapping) else {}
    left_residue = str(left.get("resname") or "").upper()
    right_residue = str(right.get("resname") or "").upper()
    min_distance = safe_float(residue_pair.get("min_distance"), 999.0) or 999.0
    result = {"hbond": 0, "salt_bridge": 0, "hydrophobic": 0, "clash": int(residue_pair.get("clash_count") or 0)}
    if min_distance <= 3.6 and left_residue in AA_POLAR and right_residue in AA_POLAR:
        result["hbond"] = 1
    if min_distance <= 4.2 and ((left_residue in AA_POSITIVE and right_residue in AA_NEGATIVE) or (left_residue in AA_NEGATIVE and right_residue in AA_POSITIVE)):
        result["salt_bridge"] = 1
    if min_distance <= 4.8 and left_residue in AA_HYDROPHOBIC and right_residue in AA_HYDROPHOBIC:
        result["hydrophobic"] = 1
    return result


def interface_interaction_counts(structure: Mapping[str, Any]) -> Dict[str, Any]:


    counts = {
        "pair_count": 0,
        "residue_pair_count": 0,
        "contact_count": 0,
        "hbond_proxy_count": 0,
        "salt_bridge_proxy_count": 0,
        "hydrophobic_proxy_count": 0,
        "clash_count": 0,
        "state_pair_counts": {},
    }
    for state_name, pair_name, pair in all_interface_pairs(structure):
        pair_key = f"{state_name}/{pair_name}"
        residue_pairs = pair.get("residue_pairs", []) if isinstance(pair.get("residue_pairs"), list) else []
        pair_counts = {
            "contact_count": int(pair.get("contact_count") or 0),
            "residue_pair_count": int(pair.get("residue_pair_count") or len(residue_pairs)),
            "hbond_proxy_count": 0,
            "salt_bridge_proxy_count": 0,
            "hydrophobic_proxy_count": 0,
            "clash_count": int(pair.get("clash_count") or 0),
        }
        for residue_pair in residue_pairs:
            if not isinstance(residue_pair, Mapping):
                continue
            classified = classify_pair(residue_pair)
            pair_counts["hbond_proxy_count"] += classified["hbond"]
            pair_counts["salt_bridge_proxy_count"] += classified["salt_bridge"]
            pair_counts["hydrophobic_proxy_count"] += classified["hydrophobic"]
        counts["pair_count"] += 1
        counts["contact_count"] += pair_counts["contact_count"]
        counts["residue_pair_count"] += pair_counts["residue_pair_count"]
        counts["hbond_proxy_count"] += pair_counts["hbond_proxy_count"]
        counts["salt_bridge_proxy_count"] += pair_counts["salt_bridge_proxy_count"]
        counts["hydrophobic_proxy_count"] += pair_counts["hydrophobic_proxy_count"]
        counts["clash_count"] += pair_counts["clash_count"]
        counts["state_pair_counts"][pair_key] = pair_counts
    return counts


def atom_geometry_counts_from_cif(
    cif_path: Optional[str],
    max_atom_pairs: int = 2_000_000,
) -> Dict[str, Any]:


    atoms = [
        atom
        for atom in cif_atoms(cif_path)
        if str(atom.get("group", "ATOM")).upper() == "ATOM"
        and is_xyz(atom.get("xyz"))
        and not atom_name(atom).startswith("H")
    ]
    if not atoms:
        return {"available": False, "reason": "no atom coordinates parsed"}
    by_chain: Dict[str, List[Mapping[str, Any]]] = {}
    for atom in atoms:
        by_chain.setdefault(str(atom.get("asym")), []).append(atom)
    chain_ids = sorted(by_chain)
    if len(chain_ids) < 2:
        return {"available": False, "reason": "fewer than two chains in structure"}

    hbond = 0
    salt = 0
    hydrophobic = 0
    clashes = 0
    atom_pairs_scanned = 0
    examples: List[Dict[str, Any]] = []

    for index, left_id in enumerate(chain_ids):
        for right_id in chain_ids[index + 1 :]:
            left_atoms = by_chain[left_id]
            right_atoms = by_chain[right_id]
            for left_atom in left_atoms:
                for right_atom in right_atoms:
                    atom_pairs_scanned += 1
                    if atom_pairs_scanned > max_atom_pairs:
                        return {
                            "available": True,
                            "truncated": True,
                            "reason": "max atom pair scan reached",
                            "atom_pairs_scanned": atom_pairs_scanned,
                            "hbond_geometry_count": hbond,
                            "salt_bridge_geometry_count": salt,
                            "hydrophobic_geometry_count": hydrophobic,
                            "geometry_clash_count": clashes,
                            "examples": examples,
                        }
                    distance_squared = dist2(left_atom.get("xyz"), right_atom.get("xyz"))
                    if distance_squared is None:
                        continue
                    if distance_squared < 1.8 * 1.8:
                        clashes += 1
                    if distance_squared <= 3.6 * 3.6 and is_hbond_heavy_atom(left_atom) and is_hbond_heavy_atom(right_atom):
                        hbond += 1
                        if len(examples) < 12:
                            examples.append({"type": "hbond_geometry", "left": [left_id, left_atom.get("seq_id"), left_atom.get("comp"), left_atom.get("atom")], "right": [right_id, right_atom.get("seq_id"), right_atom.get("comp"), right_atom.get("atom")], "distance": round(math.sqrt(distance_squared), 3)})
                    if distance_squared <= 4.2 * 4.2 and ((is_positive_atom(left_atom) and is_negative_atom(right_atom)) or (is_negative_atom(left_atom) and is_positive_atom(right_atom))):
                        salt += 1
                        if len(examples) < 12:
                            examples.append({"type": "salt_bridge_geometry", "left": [left_id, left_atom.get("seq_id"), left_atom.get("comp"), left_atom.get("atom")], "right": [right_id, right_atom.get("seq_id"), right_atom.get("comp"), right_atom.get("atom")], "distance": round(math.sqrt(distance_squared), 3)})
                    if distance_squared <= 4.8 * 4.8 and is_hydrophobic_atom(left_atom) and is_hydrophobic_atom(right_atom):
                        hydrophobic += 1
                        if len(examples) < 12:
                            examples.append({"type": "hydrophobic_geometry", "left": [left_id, left_atom.get("seq_id"), left_atom.get("comp"), left_atom.get("atom")], "right": [right_id, right_atom.get("seq_id"), right_atom.get("comp"), right_atom.get("atom")], "distance": round(math.sqrt(distance_squared), 3)})
    return {
        "available": True,
        "atom_pairs_scanned": atom_pairs_scanned,
        "hbond_geometry_count": hbond,
        "salt_bridge_geometry_count": salt,
        "hydrophobic_geometry_count": hydrophobic,
        "geometry_clash_count": clashes,
        "examples": examples,
    }


def atom_geometry_counts(structure: Mapping[str, Any]) -> Dict[str, Any]:


    state_results = []
    states = structure.get("states") if isinstance(structure.get("states"), list) else []
    if states:
        for state in states:
            if not isinstance(state, Mapping):
                continue
            summary = state.get("structure_metrics", {}) if isinstance(state.get("structure_metrics"), Mapping) else {}
            cif_path = summary.get("cif_path") or state.get("cif_path")
            result = atom_geometry_counts_from_cif(str(cif_path) if cif_path else None)
            result["state"] = state.get("name")
            state_results.append(result)
    else:
        result = atom_geometry_counts_from_cif(str(structure.get("cif_path")) if structure.get("cif_path") else None)
        result["state"] = "aggregate"
        state_results.append(result)

    available = [item for item in state_results if item.get("available")]
    if not available:
        return {"available": False, "states": state_results, "reason": "no atom-level interaction geometry available"}
    return {
        "available": True,
        "states": state_results,
        "hbond_geometry_count": sum(int(item.get("hbond_geometry_count") or 0) for item in available),
        "salt_bridge_geometry_count": sum(int(item.get("salt_bridge_geometry_count") or 0) for item in available),
        "hydrophobic_geometry_count": sum(int(item.get("hydrophobic_geometry_count") or 0) for item in available),
        "geometry_clash_count": sum(int(item.get("geometry_clash_count") or 0) for item in available),
        "atom_pairs_scanned": sum(int(item.get("atom_pairs_scanned") or 0) for item in available),
    }
