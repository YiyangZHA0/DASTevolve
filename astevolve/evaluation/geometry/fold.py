

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from astevolve.evaluation.backends.base import cif_atoms
from astevolve.evaluation.support import clamp01


def ca_atoms(atoms: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:


    return [
        atom
        for atom in atoms
        if str(atom.get("atom", "")).strip().upper() == "CA"
        and str(atom.get("group", "ATOM")).upper() == "ATOM"
    ]


def chain_break_count(cif_path: Optional[str], cutoff: float = 4.5) -> int:


    atoms = ca_atoms(cif_atoms(cif_path))
    by_chain: Dict[str, List[Mapping[str, Any]]] = {}
    for atom in atoms:
        by_chain.setdefault(str(atom.get("asym")), []).append(atom)
    breaks = 0
    for chain_atoms in by_chain.values():
        chain_atoms = sorted(chain_atoms, key=lambda item: int(item.get("seq_id", 0)))
        for left, right in zip(chain_atoms, chain_atoms[1:]):
            if int(right.get("seq_id", 0)) != int(left.get("seq_id", 0)) + 1:
                continue
            distance_squared = dist2(left.get("xyz"), right.get("xyz"))
            if distance_squared is not None and math.sqrt(distance_squared) > cutoff:
                breaks += 1
    return int(breaks)


def radius_of_gyration(cif_path: Optional[str]) -> Optional[float]:


    atoms = ca_atoms(cif_atoms(cif_path))
    coords = [atom.get("xyz") for atom in atoms if is_xyz(atom.get("xyz"))]
    if len(coords) < 6:
        return None
    count = float(len(coords))
    centroid = (
        sum(float(coord[0]) for coord in coords) / count,
        sum(float(coord[1]) for coord in coords) / count,
        sum(float(coord[2]) for coord in coords) / count,
    )
    return math.sqrt(sum(dist2(coord, centroid) or 0.0 for coord in coords) / count)


def rg_score(cif_path: Optional[str]) -> Tuple[float, Dict[str, Any]]:


    atoms = ca_atoms(cif_atoms(cif_path))
    rg = radius_of_gyration(cif_path)
    if rg is None or len(atoms) < 6:
        return 0.0, {"available": False, "reason": "no sufficient CA atoms for radius of gyration"}
    count = len(atoms)
    expected = 2.2 * (float(count) ** (1.0 / 3.0))
    ratio = rg / max(expected, 1e-8)
    if 0.55 <= ratio <= 1.75:
        score = 1.0
    elif ratio < 0.35 or ratio > 2.4:
        score = 0.0
    elif ratio < 0.55:
        score = (ratio - 0.35) / 0.20
    else:
        score = 1.0 - (ratio - 1.75) / 0.65
    return clamp01(score), {"available": True, "ca_count": count, "rg": rg, "expected_rg": expected, "rg_ratio": ratio}


def is_xyz(value: Any) -> bool:


    return isinstance(value, tuple) and len(value) == 3


def dist2(left: Any, right: Any) -> Optional[float]:


    if not (is_xyz(left) and is_xyz(right)):
        return None
    return (float(left[0]) - float(right[0])) ** 2 + (float(left[1]) - float(right[1])) ** 2 + (float(left[2]) - float(right[2])) ** 2


def element(atom_name: Any) -> str:


    name = str(atom_name or "").strip().upper()
    while name and name[0].isdigit():
        name = name[1:]
    return name[:1]


def atom_name(atom: Mapping[str, Any]) -> str:


    return str(atom.get("atom") or "").strip().upper()


def is_sidechain_heavy_atom(atom: Mapping[str, Any]) -> bool:


    name = atom_name(atom)
    if not name or name.startswith("H"):
        return False
    return name not in {"N", "CA", "C", "O", "OXT"}


def is_hbond_heavy_atom(atom: Mapping[str, Any]) -> bool:


    return element(atom.get("atom")) in {"N", "O", "S"}
