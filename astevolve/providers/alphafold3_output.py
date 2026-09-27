

from __future__ import annotations

import json
from pathlib import Path
import shlex
import statistics
import string
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_SUMMARY_SUFFIX = "_summary_confidences.json"
_CONFIDENCE_SUFFIX = "_confidences.json"
_MODEL_SUFFIX = "_model.cif"


def _sanitise_job_name(value: str) -> str:
    spaceless = str(value).replace(" ", "_")
    allowed = set(string.ascii_letters + string.digits + "_-.")
    return "".join(char for char in spaceless if char in allowed)


def _artifact_rank(path: Path, root: Path, expected_name: Optional[str]) -> Tuple[Any, ...]:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    text = relative.as_posix().lower()
    is_sample = int("seed-" in text or "_sample-" in text)
    expected_match = 0
    if expected_name:
        expected_match = int(not path.name.startswith(f"{expected_name}_"))
    return expected_match, is_sample, len(relative.parts), relative.as_posix()


def _select_artifact(
    paths: Iterable[Path],
    *,
    root: Path,
    expected_name: Optional[str],
) -> Optional[Path]:
    candidates = [path for path in paths if path.is_file()]
    if not candidates:
        return None
    return min(candidates, key=lambda path: _artifact_rank(path, root, expected_name))


def find_alphafold3_artifacts(
    output_dir: Path | str,
    pred_name: Optional[str] = None,
) -> Dict[str, Any]:


    root = Path(output_dir)
    expected_name = _sanitise_job_name(pred_name) if pred_name else None
    if not root.exists():
        return {
            "out_dir": None,
            "summary_json": None,
            "sample_summary_jsons": [],
            "confidence_json": None,
            "cif_path": None,
        }

    summary = _select_artifact(
        root.rglob(f"*{_SUMMARY_SUFFIX}"),
        root=root,
        expected_name=expected_name,
    )
    job_dir = summary.parent if summary else root
    prefix = summary.name[: -len(_SUMMARY_SUFFIX)] if summary else expected_name

    confidence = None
    cif_path = None
    if prefix:
        direct_confidence = job_dir / f"{prefix}{_CONFIDENCE_SUFFIX}"
        direct_cif = job_dir / f"{prefix}{_MODEL_SUFFIX}"
        confidence = direct_confidence if direct_confidence.is_file() else None
        cif_path = direct_cif if direct_cif.is_file() else None
    if confidence is None:
        confidence = _select_artifact(
            (
                path
                for path in root.rglob(f"*{_CONFIDENCE_SUFFIX}")
                if not path.name.endswith(_SUMMARY_SUFFIX)
            ),
            root=root,
            expected_name=expected_name,
        )
    if cif_path is None:
        cif_path = _select_artifact(
            root.rglob(f"*{_MODEL_SUFFIX}"),
            root=root,
            expected_name=expected_name,
        )

    selected = summary or cif_path or confidence
    if selected is not None:
        job_dir = selected.parent
    sample_summaries = sorted(
        path
        for path in root.rglob(f"*{_SUMMARY_SUFFIX}")
        if path.is_file()
        and any(part.startswith("seed-") for part in path.relative_to(root).parts)
    )
    return {
        "out_dir": str(job_dir),
        "summary_json": str(summary) if summary else None,
        "sample_summary_jsons": [str(path) for path in sample_summaries],
        "confidence_json": str(confidence) if confidence else None,
        "cif_path": str(cif_path) if cif_path else None,
    }


def _load_json(path: Optional[str]) -> Tuple[Optional[Any], Optional[str]]:
    if not path:
        return None, None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Could not parse {path}: {type(exc).__name__}: {exc}"


def _atom_site_rows(path: Path) -> Iterable[Dict[str, str]]:


    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return

    headers: List[str] = []
    in_atom_loop = False
    data_started = False
    for raw_line in lines:
        line = raw_line.strip()
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
        if len(tokens) >= len(headers):
            yield {headers[index]: tokens[index] for index in range(len(headers))}


def _mean(values: Iterable[float]) -> Optional[float]:
    numbers = [float(value) for value in values]
    return float(sum(numbers) / len(numbers)) if numbers else None


def _scale_plddt(values: List[float]) -> List[float]:
    if values and max(abs(value) for value in values) <= 1.5:
        return [float(value) * 100.0 for value in values]
    return [float(value) for value in values]


def extract_residue_plddt_from_cif(
    cif_path: Path | str,
    chain_sequences: Optional[Mapping[str, str]] = None,
) -> Dict[str, List[float]]:


    expected = {str(key): str(value) for key, value in (chain_sequences or {}).items()}
    by_chain: Dict[str, Dict[int, List[float]]] = {}
    for row in _atom_site_rows(Path(cif_path)):
        chain_id = row.get("_atom_site.label_asym_id") or row.get("_atom_site.auth_asym_id")
        residue_raw = row.get("_atom_site.label_seq_id") or row.get("_atom_site.auth_seq_id")
        plddt_raw = row.get("_atom_site.B_iso_or_equiv")
        if not chain_id or not residue_raw or not plddt_raw or residue_raw in {".", "?"}:
            continue
        if expected and chain_id not in expected:
            continue
        try:
            residue_id = int(float(residue_raw))
            plddt = float(plddt_raw)
        except (TypeError, ValueError):
            continue
        if residue_id < 1:
            continue
        by_chain.setdefault(str(chain_id), {}).setdefault(residue_id, []).append(plddt)

    output: Dict[str, List[float]] = {}
    for chain_id, residues in by_chain.items():
        residue_means = {
            residue_id: float(sum(atom_values) / len(atom_values))
            for residue_id, atom_values in residues.items()
            if atom_values
        }
        if not residue_means:
            continue
        if chain_id in expected:
            chain_mean = _mean(residue_means.values()) or 0.0
            values = [
                residue_means.get(residue_id, chain_mean)
                for residue_id in range(1, len(expected[chain_id]) + 1)
            ]
        else:
            values = [residue_means[key] for key in sorted(residue_means)]
        output[chain_id] = _scale_plddt(values)
    return output


def _scalar_metrics(summary: Any) -> Dict[str, float]:
    if not isinstance(summary, Mapping):
        return {}
    metrics: Dict[str, float] = {}
    for key, value in summary.items():
        if isinstance(value, bool):
            metrics[str(key)] = float(value)
        elif isinstance(value, (int, float)):
            metrics[str(key)] = float(value)
    if "fraction_disordered" in metrics:
        metrics.setdefault("disorder", metrics["fraction_disordered"])
    return metrics


def _numeric_list(value: Any) -> Optional[List[float]]:
    if not isinstance(value, list):
        return None
    output: List[float] = []
    for item in value:
        if isinstance(item, bool):
            output.append(float(item))
        elif isinstance(item, (int, float)):
            output.append(float(item))
        else:
            return None
    return output


def _add_source_aliases(
    values: Mapping[str, Any],
    source_chain_map: Mapping[str, str],
) -> Dict[str, Any]:
    output = dict(values)
    for af3_id, source_label in source_chain_map.items():
        if af3_id in values and source_label:
            output.setdefault(str(source_label), values[af3_id])
    return output


def _summary_chain_ids(
    summary: Any,
    fallback_chain_ids: Sequence[str] = (),
) -> List[str]:


    raw_ids = summary.get("chain_ids", []) if isinstance(summary, Mapping) else []
    if raw_ids:
        return [str(value) for value in raw_ids]
    return [str(value) for value in fallback_chain_ids if str(value)]


def _chain_metrics(
    summary: Any,
    residue_plddt: Mapping[str, List[float]],
    source_chain_map: Mapping[str, str],
    fallback_chain_ids: Sequence[str] = (),
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    if isinstance(summary, Mapping):
        chain_ids = _summary_chain_ids(summary, fallback_chain_ids)
        for key, value in summary.items():
            if not str(key).startswith("chain_") or str(key).startswith("chain_pair_"):
                continue
            if key == "chain_ids":
                continue
            numbers = _numeric_list(value)
            if not numbers or len(numbers) != len(chain_ids):
                continue
            metric_name = str(key)[len("chain_") :]
            raw = {chain_id: numbers[index] for index, chain_id in enumerate(chain_ids)}
            metrics[metric_name] = _add_source_aliases(raw, source_chain_map)

    raw_plddt = {
        chain_id: float(sum(values) / len(values))
        for chain_id, values in residue_plddt.items()
        if values
    }
    if raw_plddt:
        metrics["plddt"] = _add_source_aliases(raw_plddt, source_chain_map)
    return metrics


def _chain_pair_metrics(
    summary: Any,
    fallback_chain_ids: Sequence[str] = (),
) -> Dict[str, Dict[str, float]]:
    if not isinstance(summary, Mapping):
        return {}
    chain_ids = _summary_chain_ids(summary, fallback_chain_ids)
    output: Dict[str, Dict[str, float]] = {}
    for key, value in summary.items():
        if not str(key).startswith("chain_pair_") or not isinstance(value, list):
            continue
        metric_name = str(key)[len("chain_pair_") :]
        pairs: Dict[str, float] = {}
        for left_index, row in enumerate(value):
            if left_index >= len(chain_ids) or not isinstance(row, list):
                continue
            for right_index, number in enumerate(row):
                if right_index >= len(chain_ids) or not isinstance(number, (int, float)):
                    continue
                pairs[f"{chain_ids[left_index]}:{chain_ids[right_index]}"] = float(number)
        if pairs:
            output[metric_name] = pairs
    return output


def empty_alphafold3_confidence(
    *,
    output_dir: Optional[Path | str] = None,
    preview: Optional[Mapping[str, Any]] = None,
    input_json: Optional[str] = None,
    status_json: Optional[str] = None,
) -> Dict[str, Any]:


    preview = preview or {}
    return {
        "provider": "alphafold3",
        "metrics": {},
        "chain_metrics": {},
        "chain_pair_metrics": {},
        "residue_plddt": {},
        "entities": list(preview.get("entities", []) or []),
        "polymer_units": list(preview.get("polymer_units", []) or []),
        "metric_units": list(preview.get("metric_units", []) or []),
        "source_chain_map": dict(preview.get("source_chain_map", {}) or {}),
        "chain_id_map": dict(preview.get("chain_id_map", {}) or {}),
        "input_json": input_json or preview.get("input_json"),
        "status_json": status_json or preview.get("status_json"),
        "summary_json": None,
        "sample_summary_jsons": [],
        "sample_metrics": [],
        "confidence_json": None,
        "cif_path": None,
        "structure_path": None,
        "out_dir": str(output_dir) if output_dir is not None else None,
        "warnings": [],
    }


def parse_alphafold3_output(
    output_dir: Path | str,
    *,
    pred_name: Optional[str] = None,
    preview: Optional[Mapping[str, Any]] = None,
    input_json: Optional[str] = None,
    status_json: Optional[str] = None,
) -> Dict[str, Any]:


    preview = preview or {}
    result = empty_alphafold3_confidence(
        output_dir=output_dir,
        preview=preview,
        input_json=input_json,
        status_json=status_json,
    )
    artifacts = find_alphafold3_artifacts(output_dir, pred_name=pred_name)
    result.update(artifacts)
    result["structure_path"] = artifacts.get("cif_path")

    summary, warning = _load_json(artifacts.get("summary_json"))
    if warning:
        result["warnings"].append(warning)
    if not artifacts.get("summary_json"):
        result["warnings"].append("AlphaFold 3 summary confidence JSON was not found")
    if not artifacts.get("cif_path"):
        result["warnings"].append("AlphaFold 3 top-ranked mmCIF was not found")

    source_chain_map = {
        str(key): str(value)
        for key, value in (preview.get("source_chain_map", {}) or {}).items()
    }
    fallback_chain_ids = tuple(source_chain_map)
    af3_polymer_units = preview.get("af3_polymer_units", []) or []
    chain_sequences = {
        str(chain_id): str(sequence)
        for chain_id, sequence in af3_polymer_units
    }
    raw_residue_plddt: Dict[str, List[float]] = {}
    if artifacts.get("cif_path"):
        raw_residue_plddt = extract_residue_plddt_from_cif(
            str(artifacts["cif_path"]),
            chain_sequences=chain_sequences or None,
        )

    metrics = _scalar_metrics(summary)
    sample_metrics: List[Dict[str, float]] = []
    for sample_path in artifacts.get("sample_summary_jsons", []) or []:
        sample_summary, sample_warning = _load_json(str(sample_path))
        if sample_warning:
            result["warnings"].append(sample_warning)
            continue
        parsed = _scalar_metrics(sample_summary)
        if parsed:
            sample_metrics.append(parsed)
    if sample_metrics:
        metrics["af3_sample_count"] = float(len(sample_metrics))
        for key in ("iptm", "ptm", "ranking_score"):
            values = [row[key] for row in sample_metrics if key in row]
            if not values:
                continue
            metrics[f"{key}_median"] = float(statistics.median(values))
            metrics[f"{key}_worst"] = float(min(values))
            metrics[f"{key}_std"] = float(statistics.pstdev(values))
    residue_values = [
        value
        for values in raw_residue_plddt.values()
        for value in values
    ]
    global_plddt = _mean(residue_values)
    if global_plddt is not None:
        metrics["plddt"] = float(global_plddt)

    result["metrics"] = metrics
    result["sample_metrics"] = sample_metrics
    result["chain_metrics"] = _chain_metrics(
        summary,
        raw_residue_plddt,
        source_chain_map,
        fallback_chain_ids,
    )
    result["chain_pair_metrics"] = _chain_pair_metrics(
        summary,
        fallback_chain_ids,
    )
    result["residue_plddt"] = _add_source_aliases(
        raw_residue_plddt,
        source_chain_map,
    )
    return result


__all__ = [
    "empty_alphafold3_confidence",
    "extract_residue_plddt_from_cif",
    "find_alphafold3_artifacts",
    "parse_alphafold3_output",
]
