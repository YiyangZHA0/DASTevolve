

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from astevolve.metrics.structure import _load_atoms


CoordKey = Tuple[str, int]


@dataclass
class StructureView:


    ca_coords: Dict[CoordKey, np.ndarray]
    atoms: List[Dict[str, Any]]
    source: str = "unknown"


def _normalize_chain_map(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, str] = {}
    for key, mapped in value.items():
        if key is None or mapped is None:
            continue
        out[str(key)] = str(mapped)
    return out


def _mapped_chain(chain: Any, chain_map: Optional[Dict[str, str]]) -> str:
    chain_id = str(chain or "A")
    if not chain_map:
        return chain_id
    return str(chain_map.get(chain_id, chain_id))


def _map_atom_chains(atoms: Sequence[Dict[str, Any]], chain_map: Optional[Dict[str, str]]) -> List[Dict[str, Any]]:
    if not chain_map:
        return [dict(atom) for atom in atoms]
    mapped: List[Dict[str, Any]] = []
    for atom in atoms:
        item = dict(atom)
        item["asym"] = _mapped_chain(item.get("asym"), chain_map)
        mapped.append(item)
    return mapped


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


def _chain_id(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(alphabet):
        return alphabet[index]
    return f"X{index + 1}"


def _parse_pdb_atoms(path: Path) -> List[Dict[str, Any]]:
    atoms: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return atoms

    for line in lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        if not all(math.isfinite(v) for v in (x, y, z)):
            continue
        chain = line[21].strip() or "A"
        try:
            seq_id = int(line[22:26])
        except ValueError:
            seq_id = len(atoms) + 1
        atoms.append(
            {
                "group": line[:6].strip(),
                "atom": line[12:16].strip(),
                "comp": line[17:20].strip(),
                "asym": chain,
                "entity": "",
                "seq_id": seq_id,
                "b": _safe_float(line[60:66], 0.0) or 0.0,
                "xyz": (x, y, z),
            }
        )
    return atoms


def _coords_from_atoms(
    atoms: Sequence[Dict[str, Any]],
    chain_map: Optional[Dict[str, str]] = None,
) -> Dict[CoordKey, np.ndarray]:
    coords: Dict[CoordKey, np.ndarray] = {}
    for atom in atoms:
        name = str(atom.get("atom") or "").strip("'\"")
        if name != "CA":
            continue
        chain = _mapped_chain(atom.get("asym"), chain_map)

        index = int(atom.get("seq_id") or 1) - 1
        coords[(chain, index)] = np.asarray(atom.get("xyz"), dtype=float)
    return coords


def _coords_from_array(value: Any) -> Dict[CoordKey, np.ndarray]:
    arr = np.asarray(value, dtype=float)
    coords: Dict[CoordKey, np.ndarray] = {}
    if arr.ndim == 2 and arr.shape[1] == 3:
        for idx, xyz in enumerate(arr):
            if np.all(np.isfinite(xyz)):
                coords[("A", int(idx))] = np.asarray(xyz, dtype=float)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        for chain_idx in range(arr.shape[0]):
            chain = _chain_id(chain_idx)
            for residue_idx, xyz in enumerate(arr[chain_idx]):
                if np.all(np.isfinite(xyz)):
                    coords[(chain, int(residue_idx))] = np.asarray(xyz, dtype=float)
    return coords


def _coords_from_dict(
    value: Dict[str, Any],
    chain_map: Optional[Dict[str, str]] = None,
) -> Dict[CoordKey, np.ndarray]:
    for path_key in ("path", "pdb_path", "cif_path", "structure_path"):
        if value.get(path_key):
            return load_structure({"path": value[path_key], "chain_map": chain_map or {}}).ca_coords
    for coord_key in ("ca_coords", "coords", "coordinates"):
        if coord_key in value:
            raw = value[coord_key]
            if isinstance(raw, dict):
                coords: Dict[CoordKey, np.ndarray] = {}
                for chain, chain_coords in raw.items():
                    for idx, xyz in enumerate(chain_coords):
                        arr = np.asarray(xyz, dtype=float)
                        if arr.shape == (3,) and np.all(np.isfinite(arr)):
                            coords[(_mapped_chain(chain, chain_map), int(idx))] = arr
                return coords
            return _coords_from_array(raw)
    return {}


def _load_path_structure(path: Path, chain_map: Optional[Dict[str, str]] = None) -> StructureView:
    suffix = path.suffix.lower()
    raw_atoms = _parse_pdb_atoms(path) if suffix in {".pdb", ".ent"} else _load_atoms(str(path))
    atoms = _map_atom_chains(raw_atoms, chain_map)
    return StructureView(ca_coords=_coords_from_atoms(atoms), atoms=atoms, source=str(path))


def load_structure(structure: Any) -> StructureView:

    if isinstance(structure, StructureView):
        return structure
    if isinstance(structure, dict):
        chain_map = _normalize_chain_map(
            structure.get("chain_map")
            or structure.get("asym_to_chain")
            or structure.get("asym_id_map")
            or structure.get("chain_aliases")
        )
        for path_key in ("path", "pdb_path", "cif_path", "structure_path"):
            if structure.get(path_key):
                return _load_path_structure(Path(str(structure[path_key])), chain_map)
        atoms = list(structure.get("atoms", []) or [])
        mapped_atoms = _map_atom_chains(atoms, chain_map)
        coords = _coords_from_atoms(mapped_atoms) if mapped_atoms else _coords_from_dict(structure, chain_map)
        return StructureView(ca_coords=coords, atoms=mapped_atoms, source=str(structure.get("source", "dict")))
    if isinstance(structure, (str, Path)):
        path = Path(str(structure))
        return _load_path_structure(path)
    return StructureView(ca_coords=_coords_from_array(structure), atoms=[], source="array")


def _normalize_region(region: Any, default_name: str) -> Dict[str, Any]:
    if region is None:
        return {"name": default_name}
    if isinstance(region, dict):
        out = dict(region)
        out.setdefault("name", default_name)
        return out
    if isinstance(region, str):
        return {"name": region}
    if isinstance(region, (list, tuple)) and len(region) == 2 and all(isinstance(x, (int, float)) for x in region):
        return {"name": default_name, "spans": [[int(region[0]), int(region[1])]], "zero_based": True}
    return {"name": default_name}


def _region_chains(region: Dict[str, Any]) -> Optional[set[str]]:
    value = region.get("chain_id", region.get("chain", region.get("asym_id", region.get("source_chain"))))
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return {str(value)}


def _range_indices(region: Dict[str, Any]) -> Optional[set[int]]:
    if "indices" in region:
        return {int(x) for x in region.get("indices") or []}
    if "residues" in region:
        base = int(region.get("residue_index_base", 1))
        return {int(x) - base for x in region.get("residues") or []}

    spans = region.get("spans", region.get("ranges", region.get("residue_ranges")))
    if spans is None:
        return None
    zero_based = bool(region.get("zero_based", "spans" in region))
    out: set[int] = set()
    for span in spans:
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            continue
        start = int(span[0])
        end = int(span[1])
        if zero_based:
            out.update(range(start, end))
        else:
            out.update(range(start - 1, end))
    return out


def _chain_matches(chain: str, wanted_chains: Optional[set[str]]) -> bool:
    if wanted_chains is None:
        return True
    if chain in wanted_chains:
        return True
    for wanted in wanted_chains:
        if chain.startswith(f"{wanted}:"):
            return True
    return False


def _select_keys(coords: Dict[CoordKey, np.ndarray], region: Optional[Dict[str, Any]]) -> List[CoordKey]:
    if not region:
        return sorted(coords)
    chains = _region_chains(region)
    indices = _range_indices(region)
    selected = []
    for key in sorted(coords):
        chain, idx = key
        if not _chain_matches(chain, chains):
            continue
        if indices is not None and idx not in indices:
            continue
        selected.append(key)
    return selected


def _common_keys(*views: StructureView) -> List[CoordKey]:
    if not views:
        return []
    keys = set(views[0].ca_coords)
    for view in views[1:]:
        keys.intersection_update(view.ca_coords)
    return sorted(keys)


def _stack(view: StructureView, keys: Sequence[CoordKey]) -> np.ndarray:
    return np.asarray([view.ca_coords[key] for key in keys], dtype=float)


def _kabsch_transform(reference: np.ndarray, mobile: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_center = reference.mean(axis=0)
    mob_center = mobile.mean(axis=0)
    ref0 = reference - ref_center
    mob0 = mobile - mob_center
    cov = mob0.T @ ref0
    u, _, vt = np.linalg.svd(cov)
    det = np.linalg.det(u @ vt)
    corr = np.eye(3)
    corr[2, 2] = 1.0 if det >= 0 else -1.0
    rot = u @ corr @ vt
    return rot, ref_center, mob_center


def _apply_transform(coords: np.ndarray, rot: np.ndarray, ref_center: np.ndarray, mob_center: np.ndarray) -> np.ndarray:
    return (coords - mob_center) @ rot + ref_center


def _rmsd(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return None
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _movement_score(rmsd: Optional[float], min_rmsd: float, target_rmsd: float, max_rmsd: float) -> float:
    if rmsd is None:
        return 0.0
    if rmsd <= min_rmsd:
        return 0.0
    score = _clamp01((float(rmsd) - min_rmsd) / max(target_rmsd - min_rmsd, 1e-6))
    if rmsd > max_rmsd:
        score *= 1.0 - _clamp01((float(rmsd) - max_rmsd) / max(max_rmsd, 1.0))
    return float(score)


def _stability_score(rmsd: Optional[float], target_rmsd: float, max_rmsd: float) -> float:
    if rmsd is None:
        return 0.0
    if rmsd <= target_rmsd:
        return 1.0
    return float(1.0 - _clamp01((float(rmsd) - target_rmsd) / max(max_rmsd - target_rmsd, 1e-6)))


def _radius_of_gyration(coords: np.ndarray) -> Optional[float]:
    if len(coords) == 0:
        return None
    center = coords.mean(axis=0)
    diff = coords - center
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _chain_break_report(view: StructureView, threshold: float) -> Dict[str, Any]:
    breaks: List[Dict[str, Any]] = []
    by_chain: Dict[str, List[Tuple[int, np.ndarray]]] = {}
    for (chain, idx), xyz in view.ca_coords.items():
        by_chain.setdefault(chain, []).append((idx, xyz))
    for chain, items in by_chain.items():
        items = sorted(items, key=lambda x: x[0])
        for (idx_a, xyz_a), (idx_b, xyz_b) in zip(items, items[1:]):
            if idx_b != idx_a + 1:
                continue
            dist = float(np.linalg.norm(xyz_b - xyz_a))
            if dist > threshold:
                breaks.append({"chain": chain, "left": idx_a, "right": idx_b, "ca_distance": dist})
    return {"triggered": bool(breaks), "count": len(breaks), "examples": breaks[:10]}


def _severe_clash_report(
    view: StructureView,
    threshold: float,
    local_sequence_gap: int = 3,
    allowed_count: int = 2,
) -> Dict[str, Any]:
    if view.atoms:
        particles = [
            (str(a.get("asym") or "A"), int(a.get("seq_id") or 1) - 1, str(a.get("atom") or ""), np.asarray(a.get("xyz"), dtype=float))
            for a in view.atoms
        ]
    else:
        particles = [(chain, idx, "CA", xyz) for (chain, idx), xyz in view.ca_coords.items()]
    clashes: List[Dict[str, Any]] = []
    ignored_local_pairs = 0
    threshold2 = float(threshold) ** 2
    for i in range(len(particles)):
        chain_i, idx_i, atom_i, xyz_i = particles[i]
        for j in range(i + 1, len(particles)):
            chain_j, idx_j, atom_j, xyz_j = particles[j]
            d2 = float(np.sum((xyz_i - xyz_j) ** 2))
            if chain_i == chain_j and abs(idx_i - idx_j) <= int(local_sequence_gap):
                if d2 <= threshold2:
                    ignored_local_pairs += 1
                continue
            if d2 <= threshold2:
                clashes.append(
                    {
                        "left": {"chain": chain_i, "residue": idx_i, "atom": atom_i},
                        "right": {"chain": chain_j, "residue": idx_j, "atom": atom_j},
                        "distance": math.sqrt(d2),
                    }
                )
                if len(clashes) >= 20:
                    triggered = len(clashes) > int(allowed_count)
                    return {
                        "triggered": triggered,
                        "count": len(clashes),
                        "allowed_count": int(allowed_count),
                        "ignored_local_pairs": int(ignored_local_pairs),
                        "local_sequence_gap": int(local_sequence_gap),
                        "examples": clashes,
                    }
    triggered = len(clashes) > int(allowed_count)
    return {
        "triggered": triggered,
        "count": len(clashes),
        "allowed_count": int(allowed_count),
        "ignored_local_pairs": int(ignored_local_pairs),
        "local_sequence_gap": int(local_sequence_gap),
        "examples": clashes,
    }


def _path_continuity_score(
    apo: StructureView,
    holo: StructureView,
    intermediates: Sequence[StructureView],
    keys: Sequence[CoordKey],
    max_step_rmsd: float,
    max_path_ratio: float,
) -> Dict[str, Any]:
    if len(keys) < 3:
        return {"score": 0.0, "available": False, "reason": "fewer than 3 common CA coordinates"}
    if not intermediates:
        direct = _rmsd(_stack(apo, keys), _stack(holo, keys)) or 0.0
        score = 1.0 - (0.5 * _clamp01((direct - max_step_rmsd) / max(max_step_rmsd, 1.0)))
        return {
            "score": float(score),
            "available": False,
            "mode": "linear_interpolation_assumed",
            "direct_rmsd": float(direct),
        }

    views = [apo, *intermediates, holo]
    common = _common_keys(*views)
    key_set = set(keys)
    path_keys = [key for key in common if key in key_set]
    if len(path_keys) < 3:
        return {"score": 0.0, "available": False, "reason": "intermediates share fewer than 3 alignment residues"}

    step_rmsds: List[float] = []
    for left, right in zip(views, views[1:]):
        rmsd = _rmsd(_stack(left, path_keys), _stack(right, path_keys))
        if rmsd is not None:
            step_rmsds.append(float(rmsd))
    direct = _rmsd(_stack(apo, path_keys), _stack(holo, path_keys)) or 0.0
    path_length = float(sum(step_rmsds))
    path_ratio = path_length / max(float(direct), 1e-6)
    max_step = max(step_rmsds) if step_rmsds else 0.0
    step_score = 1.0 - _clamp01((max_step - max_step_rmsd) / max(max_step_rmsd, 1.0))
    ratio_score = 1.0 - _clamp01((path_ratio - 1.0) / max(max_path_ratio - 1.0, 1e-6))
    score = 0.55 * step_score + 0.45 * ratio_score
    return {
        "score": float(_clamp01(score)),
        "available": True,
        "direct_rmsd": float(direct),
        "path_length": path_length,
        "path_ratio": float(path_ratio),
        "max_step_rmsd": float(max_step),
        "step_rmsds": step_rmsds,
    }


def evaluate_mechanistic_transition(
    apo_structure: Any,
    holo_structure: Any,
    moving_regions: Optional[Sequence[Any]] = None,
    preserved_regions: Optional[Sequence[Any]] = None,
    forbid: Optional[Sequence[str]] = None,
    intermediate_states: Optional[Sequence[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    cfg = dict(config or {})
    apo = load_structure(apo_structure)
    holo = load_structure(holo_structure)
    intermediates = [load_structure(item) for item in (intermediate_states or [])]
    forbidden = {str(item).strip().lower() for item in (forbid or []) if str(item).strip()}

    common = _common_keys(apo, holo)
    moving = [_normalize_region(region, f"moving_region_{i}") for i, region in enumerate(moving_regions or [], start=1)]
    preserved = [_normalize_region(region, f"preserved_region_{i}") for i, region in enumerate(preserved_regions or [], start=1)]

    preserved_keys: List[CoordKey] = []
    for region in preserved:
        preserved_keys.extend(key for key in _select_keys(apo.ca_coords, region) if key in holo.ca_coords)
    alignment_keys = sorted(set(preserved_keys))
    if len(alignment_keys) < 3:
        alignment_keys = common

    report: Dict[str, Any] = {
        "inputs": {
            "apo_source": apo.source,
            "holo_source": holo.source,
            "common_ca_count": len(common),
            "intermediate_count": len(intermediates),
        },
        "alignment": {
            "used_ca_count": len(alignment_keys),
            "used_preserved_regions": len(preserved_keys) >= 3,
        },
        "moving_regions": {},
        "preserved_regions": {},
        "path_continuity": {},
        "forbidden_events": {},
        "components": {},
        "warnings": [],
    }

    if len(alignment_keys) < 3:
        report["warnings"].append("fewer than 3 common CA coordinates; transition score is unavailable")
        return {
            "kinetic_path_score": 0.0,
            "interpretability_report": report,
            "region_scores": {},
        }

    apo_align = _stack(apo, alignment_keys)
    holo_align = _stack(holo, alignment_keys)
    rot, ref_center, mob_center = _kabsch_transform(apo_align, holo_align)
    holo_aligned: Dict[CoordKey, np.ndarray] = {
        key: _apply_transform(np.asarray([xyz]), rot, ref_center, mob_center)[0]
        for key, xyz in holo.ca_coords.items()
    }
    alignment_rmsd = _rmsd(apo_align, _stack(StructureView(holo_aligned, [], "holo_aligned"), alignment_keys))
    report["alignment"]["rmsd"] = alignment_rmsd

    min_move = float(cfg.get("min_moving_rmsd", 0.5))
    target_move = float(cfg.get("target_moving_rmsd", 3.0))
    max_move = float(cfg.get("max_moving_rmsd", 12.0))
    stable_target = float(cfg.get("stable_region_rmsd", 1.5))
    stable_max = float(cfg.get("max_preserved_rmsd", 4.0))

    moving_scores: List[float] = []
    for index, region in enumerate(moving, start=1):
        name = str(region.get("name") or f"moving_region_{index}")
        keys = [key for key in _select_keys(apo.ca_coords, region) if key in holo_aligned]
        rmsd = _rmsd(_stack(apo, keys), np.asarray([holo_aligned[key] for key in keys], dtype=float)) if keys else None
        score = _movement_score(rmsd, min_move, target_move, max_move)
        moving_scores.append(score)
        report["moving_regions"][name] = {
            "ca_count": len(keys),
            "rmsd": rmsd,
            "score": score,
            "min_rmsd": min_move,
            "target_rmsd": target_move,
            "max_rmsd": max_move,
        }

    preserved_scores: List[float] = []
    for index, region in enumerate(preserved, start=1):
        name = str(region.get("name") or f"preserved_region_{index}")
        keys = [key for key in _select_keys(apo.ca_coords, region) if key in holo_aligned]
        rmsd = _rmsd(_stack(apo, keys), np.asarray([holo_aligned[key] for key in keys], dtype=float)) if keys else None
        score = _stability_score(rmsd, stable_target, stable_max)
        preserved_scores.append(score)
        report["preserved_regions"][name] = {
            "ca_count": len(keys),
            "rmsd": rmsd,
            "score": score,
            "target_rmsd": stable_target,
            "max_rmsd": stable_max,
        }

    if not moving_scores:
        global_rmsd = _rmsd(_stack(apo, common), np.asarray([holo_aligned[key] for key in common], dtype=float))
        moving_scores.append(_movement_score(global_rmsd, min_move, target_move, max_move))
        report["moving_regions"]["global_fallback"] = {"ca_count": len(common), "rmsd": global_rmsd, "score": moving_scores[-1]}

    if not preserved_scores:
        preserved_scores.append(_stability_score(alignment_rmsd, stable_target, stable_max))

    continuity_keys = alignment_keys
    path_report = _path_continuity_score(
        apo,
        holo,
        intermediates,
        continuity_keys,
        max_step_rmsd=float(cfg.get("max_step_rmsd", 4.0)),
        max_path_ratio=float(cfg.get("max_path_ratio", 2.5)),
    )
    report["path_continuity"] = path_report

    penalty = 0.0
    rg_apo = _radius_of_gyration(_stack(apo, common))
    rg_holo = _radius_of_gyration(_stack(StructureView(holo_aligned, [], "holo_aligned"), common))
    rg_ratio = float(rg_holo / rg_apo) if rg_apo and rg_holo else None
    unfolding_triggered = bool(rg_ratio is not None and (rg_ratio > float(cfg.get("max_rg_ratio", 2.5)) or rg_ratio < float(cfg.get("min_rg_ratio", 0.4))))
    if "complete_unfolding" in forbidden and unfolding_triggered:
        penalty += float(cfg.get("complete_unfolding_penalty", 0.35))
    report["forbidden_events"]["complete_unfolding"] = {
        "checked": "complete_unfolding" in forbidden,
        "triggered": unfolding_triggered,
        "apo_rg": rg_apo,
        "holo_rg": rg_holo,
        "rg_ratio": rg_ratio,
    }

    if "chain_break" in forbidden:
        apo_break = _chain_break_report(apo, float(cfg.get("chain_break_ca_distance", 5.0)))
        holo_break = _chain_break_report(StructureView(holo_aligned, holo.atoms, "holo_aligned"), float(cfg.get("chain_break_ca_distance", 5.0)))
        triggered = bool(apo_break["triggered"] or holo_break["triggered"])
        if triggered:
            penalty += float(cfg.get("chain_break_penalty", 0.35))
        report["forbidden_events"]["chain_break"] = {
            "checked": True,
            "triggered": triggered,
            "apo": apo_break,
            "holo": holo_break,
        }
    else:
        report["forbidden_events"]["chain_break"] = {"checked": False, "triggered": False}

    if "severe_clash" in forbidden:
        clash_threshold = float(cfg.get("severe_clash_distance", 1.8 if apo.atoms or holo.atoms else 2.2))
        clash_local_gap = int(cfg.get("severe_clash_ignore_sequence_separation", 3))
        clash_allowed_count = int(cfg.get("severe_clash_allowed_count", 2))
        apo_clash = _severe_clash_report(
            apo,
            clash_threshold,
            local_sequence_gap=clash_local_gap,
            allowed_count=clash_allowed_count,
        )
        holo_clash = _severe_clash_report(
            holo,
            clash_threshold,
            local_sequence_gap=clash_local_gap,
            allowed_count=clash_allowed_count,
        )
        triggered = bool(apo_clash["triggered"] or holo_clash["triggered"])
        if triggered:
            penalty += float(cfg.get("severe_clash_penalty", 0.35))
        report["forbidden_events"]["severe_clash"] = {
            "checked": True,
            "triggered": triggered,
            "threshold": clash_threshold,
            "local_sequence_gap": clash_local_gap,
            "allowed_count": clash_allowed_count,
            "apo": apo_clash,
            "holo": holo_clash,
        }
    else:
        report["forbidden_events"]["severe_clash"] = {"checked": False, "triggered": False}

    moving_score = float(_mean(moving_scores) or 0.0)
    preserved_score = float(_mean(preserved_scores) or 0.0)
    continuity_score = float(path_report.get("score") or 0.0)
    global_fold_score = 1.0 - (0.5 * _clamp01(abs((rg_ratio or 1.0) - 1.0)))

    weights = {
        "moving": float(cfg.get("moving_weight", 0.35)),
        "preserved": float(cfg.get("preserved_weight", 0.30)),
        "continuity": float(cfg.get("continuity_weight", 0.20)),
        "global_fold": float(cfg.get("global_fold_weight", 0.15)),
    }
    weight_sum = max(sum(abs(v) for v in weights.values()), 1e-6)
    raw_score = (
        weights["moving"] * moving_score
        + weights["preserved"] * preserved_score
        + weights["continuity"] * continuity_score
        + weights["global_fold"] * global_fold_score
    ) / weight_sum
    kinetic_path_score = _clamp01(raw_score - penalty)

    report["components"] = {
        "moving_score": moving_score,
        "preserved_score": preserved_score,
        "continuity_score": continuity_score,
        "global_fold_score": float(global_fold_score),
        "forbidden_penalty": float(penalty),
        "raw_score": float(raw_score),
        "kinetic_path_score": kinetic_path_score,
    }
    return {
        "kinetic_path_score": kinetic_path_score,
        "interpretability_report": report,
        "region_scores": {
            "moving_regions": {name: item.get("score", 0.0) for name, item in report["moving_regions"].items()},
            "preserved_regions": {name: item.get("score", 0.0) for name, item in report["preserved_regions"].items()},
        },
    }
