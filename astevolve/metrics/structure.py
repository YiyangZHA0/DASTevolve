

from __future__ import annotations

import math
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in values]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _first_structure_path(confidence: Dict[str, Any]) -> Optional[str]:
    explicit = confidence.get("cif_path")
    if explicit and Path(str(explicit)).exists():
        return str(explicit)
    explicit = confidence.get("pdb_path") or confidence.get("structure_path")
    if explicit and Path(str(explicit)).exists():
        return str(explicit)

    out_dir = confidence.get("out_dir")
    if not out_dir:
        return None
    paths = sorted(Path(str(out_dir)).rglob("*.cif"))
    if paths:
        return str(paths[0])
    paths = sorted(Path(str(out_dir)).rglob("*.pdb"))
    return str(paths[0]) if paths else None


def scalar_confidence_metrics(confidence: Dict[str, Any]) -> Dict[str, float]:


    metrics = confidence.get("metrics", {}) or {}
    keep = (
        "plddt",
        "ptm",
        "iptm",
        "gpde",
        "ranking_score",
        "has_clash",
        "disorder",
        "num_recycles",
    )
    out: Dict[str, float] = {}
    for key in keep:
        value = _safe_float(metrics.get(key))
        if value is not None:
            out[key] = value
    for key, value in metrics.items():
        if key in out:
            continue
        numeric = _safe_float(value)
        if numeric is not None:
            out[str(key)] = numeric
    return out


def summarize_node_plddt(node_plddt: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:


    node_plddt = node_plddt or {}
    means: List[float] = []
    mins: List[float] = []
    low_nodes: List[Dict[str, Any]] = []

    for name, item in node_plddt.items():
        mean_val = _safe_float(item.get("plddt_mean"))
        min_val = _safe_float(item.get("plddt_min"))
        if mean_val is not None:
            means.append(mean_val)
            if mean_val < 70.0:
                low_nodes.append({"node": name, "plddt_mean": mean_val, "plddt_min": min_val})
        if min_val is not None:
            mins.append(min_val)

    low_nodes = sorted(low_nodes, key=lambda x: float(x.get("plddt_mean", 0.0)))[:10]
    return {
        "node_count": len(node_plddt),
        "node_plddt_mean": _mean(means),
        "node_plddt_min": min(mins) if mins else None,
        "low_confidence_nodes": low_nodes,
    }


def _atom_site_rows_from_cif(path: Path) -> Iterable[Dict[str, str]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return

    headers: List[str] = []
    in_atom_loop = False
    data_started = False

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            if data_started:
                break
            continue
        if line == "loop_":
            if data_started:
                break
            headers = []
            in_atom_loop = False
            continue
        if line.startswith("_"):
            if data_started:
                break
            if line.startswith("_atom_site."):
                headers.append(line.split()[0])
                in_atom_loop = True
            elif in_atom_loop:
                break
            continue
        if not in_atom_loop or not headers:
            continue
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue

        data_started = True
        try:
            tokens = shlex.split(line, posix=False)
        except ValueError:
            tokens = line.split()
        if len(tokens) < len(headers):
            continue
        yield {headers[i]: tokens[i] for i in range(len(headers))}


def _load_atoms(cif_path: str) -> List[Dict[str, Any]]:
    atoms: List[Dict[str, Any]] = []
    for row in _atom_site_rows_from_cif(Path(cif_path)):
        try:
            x = float(row.get("_atom_site.Cartn_x", "nan"))
            y = float(row.get("_atom_site.Cartn_y", "nan"))
            z = float(row.get("_atom_site.Cartn_z", "nan"))
        except ValueError:
            continue
        if not all(math.isfinite(v) for v in (x, y, z)):
            continue

        seq_raw = row.get("_atom_site.label_seq_id") or row.get("_atom_site.auth_seq_id") or "."
        try:
            seq_id = int(float(seq_raw)) if seq_raw not in {".", "?"} else 1
        except ValueError:
            seq_id = 1

        atoms.append(
            {
                "group": row.get("_atom_site.group_PDB", ""),
                "atom": row.get("_atom_site.label_atom_id") or row.get("_atom_site.auth_atom_id") or "",
                "comp": row.get("_atom_site.label_comp_id") or row.get("_atom_site.auth_comp_id") or "",
                "asym": row.get("_atom_site.label_asym_id") or row.get("_atom_site.auth_asym_id") or "",
                "entity": row.get("_atom_site.label_entity_id", ""),
                "seq_id": seq_id,
                "b": _safe_float(row.get("_atom_site.B_iso_or_equiv"), 0.0) or 0.0,
                "xyz": (x, y, z),
            }
        )
    return atoms


def _load_pdb_atoms(pdb_path: str) -> List[Dict[str, Any]]:
    atoms: List[Dict[str, Any]] = []
    try:
        lines = Path(pdb_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return atoms

    in_first_model = True
    saw_model = False
    for line in lines:
        record = line[:6].strip()
        if record == "MODEL":
            saw_model = True
            model_id = line[10:14].strip()
            in_first_model = model_id in {"", "1"}
            continue
        if record == "ENDMDL" and saw_model and in_first_model:
            break
        if record not in {"ATOM", "HETATM"} or not in_first_model:
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        seq_raw = line[22:26].strip()
        try:
            seq_id = int(seq_raw) if seq_raw.lstrip("-").isdigit() else len(atoms) + 1
        except ValueError:
            seq_id = len(atoms) + 1
        atoms.append(
            {
                "group": record,
                "atom": line[12:16].strip(),
                "comp": line[17:20].strip().upper(),
                "asym": line[21].strip() or "A",
                "entity": "",
                "seq_id": seq_id,
                "b": _safe_float(line[60:66], 0.0) or 0.0,
                "xyz": xyz,
            }
        )
    return atoms


def _load_structure_atoms(structure_path: str) -> List[Dict[str, Any]]:
    path = Path(str(structure_path))
    if path.suffix.lower() in {".pdb", ".ent"}:
        return _load_pdb_atoms(str(path))
    return _load_atoms(str(path))


def _dist2(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def contact_metrics_from_cif(
    cif_path: Optional[str],
    contact_cutoff: float = 5.0,
    clash_cutoff: float = 2.0,
    max_reported_pairs: int = 50,
) -> Dict[str, Any]:


    if not cif_path or not Path(str(cif_path)).exists():
        return {
            "available": False,
            "reason": "no CIF/PDB structure available",
            "pairs": {},
        }

    atoms = _load_structure_atoms(str(cif_path))
    if not atoms:
        return {
            "available": False,
            "reason": "no atom coordinates parsed",
            "pairs": {},
        }

    by_asym: Dict[str, List[Dict[str, Any]]] = {}
    for atom in atoms:
        if atom["asym"]:
            by_asym.setdefault(atom["asym"], []).append(atom)

    cutoff2 = float(contact_cutoff) ** 2
    clash2 = float(clash_cutoff) ** 2
    pairs: Dict[str, Dict[str, Any]] = {}
    total_contacts = 0
    total_clashes = 0
    total_residue_pairs = 0
    all_interface_plddt: List[float] = []

    asym_ids = sorted(by_asym)
    for i, left in enumerate(asym_ids):
        for right in asym_ids[i + 1 :]:
            atom_contacts = 0
            atom_clashes = 0
            residue_pair_stats: Dict[
                Tuple[Tuple[str, int, str], Tuple[str, int, str]],
                Dict[str, Any],
            ] = {}
            contacted_residues = {}
            examples: List[Dict[str, Any]] = []

            for a in by_asym[left]:
                for b in by_asym[right]:
                    d2 = _dist2(a["xyz"], b["xyz"])
                    if d2 <= clash2:
                        atom_clashes += 1
                    if d2 > cutoff2:
                        continue
                    atom_contacts += 1
                    residue_a = (a["asym"], int(a["seq_id"]), a["comp"])
                    residue_b = (b["asym"], int(b["seq_id"]), b["comp"])
                    pair_key = (residue_a, residue_b)
                    pair_stat = residue_pair_stats.setdefault(
                        pair_key,
                        {
                            "left": {
                                "chain": a["asym"],
                                "residue": int(a["seq_id"]),
                                "resname": a["comp"],
                                "plddt": float(a["b"]),
                            },
                            "right": {
                                "chain": b["asym"],
                                "residue": int(b["seq_id"]),
                                "resname": b["comp"],
                                "plddt": float(b["b"]),
                            },
                            "contact_count": 0,
                            "clash_count": 0,
                            "min_distance": None,
                        },
                    )
                    pair_stat["contact_count"] = int(pair_stat["contact_count"]) + 1
                    if d2 <= clash2:
                        pair_stat["clash_count"] = int(pair_stat["clash_count"]) + 1
                    dist = round(math.sqrt(d2), 3)
                    old_min = pair_stat.get("min_distance")
                    pair_stat["min_distance"] = dist if old_min is None else min(float(old_min), dist)
                    contacted_residues[residue_a] = float(a["b"])
                    contacted_residues[residue_b] = float(b["b"])
                    if len(examples) < max_reported_pairs:
                        examples.append(
                            {
                                "left": {"chain": left, "residue": a["seq_id"], "resname": a["comp"], "atom": a["atom"]},
                                "right": {"chain": right, "residue": b["seq_id"], "resname": b["comp"], "atom": b["atom"]},
                                "distance": round(math.sqrt(d2), 3),
                            }
                        )

            if atom_contacts <= 0 and atom_clashes <= 0:
                continue
            plddt_values = list(contacted_residues.values())
            all_interface_plddt.extend(plddt_values)
            key = f"{left}:{right}"
            residue_pair_details = sorted(
                residue_pair_stats.values(),
                key=lambda item: (
                    str(item["left"]["chain"]),
                    int(item["left"]["residue"]),
                    str(item["right"]["chain"]),
                    int(item["right"]["residue"]),
                ),
            )
            pairs[key] = {
                "contact_count": int(atom_contacts),
                "residue_pair_count": int(len(residue_pair_details)),
                "clash_count": int(atom_clashes),
                "interface_plddt_mean": _mean(plddt_values),
                "interface_plddt_min": min(plddt_values) if plddt_values else None,
                "residue_pairs": residue_pair_details,
                "contact_examples": examples,
            }
            total_contacts += int(atom_contacts)
            total_clashes += int(atom_clashes)
            total_residue_pairs += int(len(residue_pair_details))

    return {
        "available": True,
        "cif_path": str(cif_path) if Path(str(cif_path)).suffix.lower() == ".cif" else None,
        "pdb_path": str(cif_path) if Path(str(cif_path)).suffix.lower() in {".pdb", ".ent"} else None,
        "structure_path": str(cif_path),
        "contact_cutoff": float(contact_cutoff),
        "clash_cutoff": float(clash_cutoff),
        "total_contact_count": int(total_contacts),
        "total_residue_pair_count": int(total_residue_pairs),
        "clash_count": int(total_clashes),
        "interface_plddt_mean": _mean(all_interface_plddt),
        "interface_plddt_min": min(all_interface_plddt) if all_interface_plddt else None,
        "pairs": pairs,
    }


def dockq_proxy_score(interface_metrics: Dict[str, Any], iptm: Optional[float]) -> Optional[float]:


    contacts = int(interface_metrics.get("total_contact_count") or 0)
    interface_plddt = _safe_float(interface_metrics.get("interface_plddt_mean"))
    if contacts <= 0 or interface_plddt is None:
        return None


    x = float(interface_plddt) * math.log(max(contacts, 1))
    pdockq = 0.724 / (1.0 + math.exp(-0.052 * (x - 152.611))) + 0.018
    if iptm is not None and 0.0 <= float(iptm) <= 1.0:
        pdockq = 0.5 * pdockq + 0.5 * float(iptm)
    return float(max(0.0, min(1.0, pdockq)))


def summarize_structure_metrics(
    confidence: Optional[Dict[str, Any]],
    node_plddt: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:


    confidence = confidence or {}
    scalar = scalar_confidence_metrics(confidence)
    chain_plddt = dict((confidence.get("chain_metrics", {}) or {}).get("plddt", {}) or {})
    structure_path = _first_structure_path(confidence)
    interface = contact_metrics_from_cif(structure_path)
    iptm = scalar.get("iptm")
    proxy = dockq_proxy_score(interface, iptm)

    return {
        "scalar": scalar,
        "chain_plddt": chain_plddt,
        "node_plddt": node_plddt or {},
        "node_summary": summarize_node_plddt(node_plddt),
        "interface": interface,
        "dockq_proxy": proxy,
        "dockq": {
            "available": False,
            "dockq": None,
            "reason": "true DockQ requires a native/reference complex; use dockq_proxy only as a no-reference heuristic",
        },
        "cif_path": structure_path if structure_path and Path(str(structure_path)).suffix.lower() == ".cif" else None,
        "pdb_path": structure_path if structure_path and Path(str(structure_path)).suffix.lower() in {".pdb", ".ent"} else None,
        "structure_path": structure_path,
    }


def metric_value(summary: Dict[str, Any], key: str, default: float = 0.0) -> float:


    if key in (summary.get("scalar") or {}):
        return float(summary["scalar"][key])
    if key == "interface_plddt_mean":
        value = _safe_float((summary.get("interface") or {}).get("interface_plddt_mean"))
        return float(value if value is not None else default)
    if key == "interface_contact_count":
        return float((summary.get("interface") or {}).get("total_contact_count") or default)
    if key == "interface_residue_pair_count":
        return float((summary.get("interface") or {}).get("total_residue_pair_count") or default)
    if key == "clash_count":
        return float((summary.get("interface") or {}).get("clash_count") or default)
    if key == "dockq_proxy":
        value = _safe_float(summary.get("dockq_proxy"))
        return float(value if value is not None else default)
    if key == "node_plddt_mean":
        value = _safe_float((summary.get("node_summary") or {}).get("node_plddt_mean"))
        return float(value if value is not None else default)
    if key == "node_plddt_min":
        value = _safe_float((summary.get("node_summary") or {}).get("node_plddt_min"))
        return float(value if value is not None else default)
    return float(default)
