

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from astevolve.evaluation.backends.base import cif_atoms
from astevolve.evaluation.support import clamp01
from astevolve.evaluation.views import node_names_from_design_state, segments_by_name

from .fold import ca_atoms


def ca_coord_map(cif_path: Optional[str]) -> Dict[Tuple[str, int], Tuple[float, float, float]]:


    result: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
    for atom in ca_atoms(cif_atoms(cif_path)):
        try:
            result[(str(atom.get("asym")), int(atom.get("seq_id")))] = atom["xyz"]
        except Exception:
            continue
    return result


def ca_coords_by_chain(
    cif_path: Optional[str],
) -> Dict[str, List[Tuple[int, Tuple[float, float, float]]]]:


    by_chain: Dict[str, List[Tuple[int, Tuple[float, float, float]]]] = {}
    for atom in ca_atoms(cif_atoms(cif_path)):
        try:
            by_chain.setdefault(str(atom.get("asym")), []).append((int(atom.get("seq_id")), atom["xyz"]))
        except Exception:
            continue
    for chain_id in list(by_chain):
        by_chain[chain_id] = sorted(by_chain[chain_id], key=lambda item: item[0])
    return by_chain


def kabsch_rmsd(ref_coords: Sequence[Any], mob_coords: Sequence[Any]) -> Optional[float]:


    if len(ref_coords) != len(mob_coords) or len(ref_coords) < 3:
        return None
    reference = np.asarray(ref_coords, dtype=float)
    mobile = np.asarray(mob_coords, dtype=float)
    reference_center = reference.mean(axis=0)
    mobile_center = mobile.mean(axis=0)
    reference_zero = reference - reference_center
    mobile_zero = mobile - mobile_center
    covariance = mobile_zero.T @ reference_zero
    try:
        left, _, right = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    sign = np.sign(np.linalg.det(left @ right))
    correction = np.diag([1.0, 1.0, sign if sign != 0 else 1.0])
    rotation = left @ correction @ right
    aligned = mobile_zero @ rotation
    difference = aligned - reference_zero
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))


def rmsd_score(rmsd: Optional[float], good: float = 1.5, bad: float = 4.0) -> float:


    if rmsd is None:
        return 0.0
    if rmsd <= good:
        return 1.0
    if rmsd >= bad:
        return 0.0
    return clamp01(1.0 - ((rmsd - good) / max(1e-8, bad - good)))


def resolve_reference_path(
    score_config: Mapping[str, Any],
    design_state: Mapping[str, Any],
) -> Optional[str]:


    candidates: List[Any] = [
        score_config.get("scaffold_reference_path"),
        score_config.get("reference_structure_path"),
        score_config.get("reference_cif_path"),
        design_state.get("scaffold_reference_path"),
        design_state.get("reference_structure_path"),
        design_state.get("reference_cif_path"),
    ]
    reference_inputs = design_state.get("reference_inputs")
    if isinstance(reference_inputs, Mapping):
        for item in reference_inputs.values():
            if isinstance(item, Mapping):
                candidates.extend(
                    [
                        item.get("path"),
                        item.get("structure_path"),
                        item.get("cif_path"),
                        item.get("pdb_path"),
                    ]
                )
    for raw in candidates:
        if not raw:
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            project_root = os.environ.get("ASTEVOLVE_PROJECT_ROOT")
            if project_root:
                path = Path(project_root) / path
        if path.exists():
            return str(path)
    return None


def global_ca_rmsd(
    reference_path: Optional[str],
    candidate_path: Optional[str],
) -> Tuple[Optional[float], Dict[str, Any]]:


    if not reference_path or not candidate_path:
        return None, {"available": False, "reason": "reference or candidate structure path missing"}
    reference_map = ca_coord_map(reference_path)
    candidate_map = ca_coord_map(candidate_path)
    common = sorted(set(reference_map) & set(candidate_map))
    method = "common_chain_residue_ids"
    if len(common) < 3:
        reference_chains = ca_coords_by_chain(reference_path)
        candidate_chains = ca_coords_by_chain(candidate_path)
        if not reference_chains or not candidate_chains:
            return None, {"available": False, "reason": "no CA atoms parsed"}
        reference_values = next(iter(reference_chains.values()))
        candidate_values = next(iter(candidate_chains.values()))
        count = min(len(reference_values), len(candidate_values))
        if count < 3:
            return None, {"available": False, "reason": "fewer than 3 matched CA atoms"}
        reference_coords = [coord for _, coord in reference_values[:count]]
        candidate_coords = [coord for _, coord in candidate_values[:count]]
        method = "first_chain_order_fallback"
    else:
        reference_coords = [reference_map[key] for key in common]
        candidate_coords = [candidate_map[key] for key in common]
        count = len(common)
    rmsd = kabsch_rmsd(reference_coords, candidate_coords)
    return rmsd, {
        "available": rmsd is not None,
        "method": method,
        "matched_ca_count": count,
        "reference_path": reference_path,
        "candidate_path": candidate_path,
    }


def preserved_node_ca_rmsd(
    reference_path: Optional[str],
    candidate_path: Optional[str],
    compiled: Optional[Mapping[str, Any]],
    design_state: Mapping[str, Any],
) -> Tuple[Optional[float], Dict[str, Any]]:


    if not reference_path or not candidate_path or not compiled:
        return None, {"available": False, "reason": "reference, candidate, or compiled segments missing"}
    preserved = node_names_from_design_state(design_state, "preserved")
    segment_map = segments_by_name(compiled)
    indices: List[int] = []
    for name in preserved:
        segment = segment_map.get(name)
        if segment and segment.get("chain_id") in {"BB", "A", ""}:
            indices.extend(segment.get("indices") or [])
    indices = sorted(set(index for index in indices if index >= 0))
    if not indices:
        return None, {"available": False, "reason": "no preserved node indices for RMSD"}
    reference_chains = ca_coords_by_chain(reference_path)
    candidate_chains = ca_coords_by_chain(candidate_path)
    if not reference_chains or not candidate_chains:
        return None, {"available": False, "reason": "no CA atoms parsed"}
    reference_values = next(iter(reference_chains.values()))
    candidate_values = next(iter(candidate_chains.values()))
    reference_by_order = [coord for _, coord in reference_values]
    candidate_by_order = [coord for _, coord in candidate_values]
    reference_coords = []
    candidate_coords = []
    for index in indices:
        if index < len(reference_by_order) and index < len(candidate_by_order):
            reference_coords.append(reference_by_order[index])
            candidate_coords.append(candidate_by_order[index])
    if len(reference_coords) < 3:
        return None, {"available": False, "reason": "fewer than 3 matched preserved-node CA atoms"}
    rmsd = kabsch_rmsd(reference_coords, candidate_coords)
    return rmsd, {
        "available": rmsd is not None,
        "method": "preserved_node_order_indices",
        "matched_ca_count": len(reference_coords),
        "node_count": len([name for name in preserved if name in segment_map]),
        "nodes": preserved[:20],
        "reference_path": reference_path,
        "candidate_path": candidate_path,
    }
