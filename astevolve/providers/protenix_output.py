

from __future__ import annotations

import json
from pathlib import Path
import shlex
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .protenix_input import (
    _batch_pred_name,
    _cache_key,
    _json_cache_key,
    _normalise_constraint,
    _normalise_entities,
    _normalise_inputs,
)
from .protenix_runner import (
    _first_cif,
    _first_json,
    _run_protenix,
    _run_protenix_complex_batch,
    _run_protenix_complex,
)


_SCALAR_CACHE: Dict[str, float] = {}
_CONF_CACHE: Dict[str, Dict[str, Any]] = {}


def _complex_runtime_options(
    *,
    conda_env: Optional[str],
    use_msa: Optional[bool],
    cycle: Optional[int],
    step: Optional[int],
    sample: Optional[int],
    use_default_params: Optional[bool],
) -> Dict[str, Any]:


    return {
        "conda_env": conda_env,
        "use_msa": use_msa,
        "cycle": cycle,
        "step": step,
        "sample": sample,
        "use_default_params": use_default_params,
    }


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[protenix] failed to parse {path}: {exc}")
        return None


def _numeric_list(value: Any) -> Optional[List[float]]:
    if isinstance(value, list):
        out: List[float] = []
        for item in value:
            if isinstance(item, (int, float)):
                out.append(float(item))
            elif isinstance(item, list) and len(item) == 1 and isinstance(item[0], (int, float)):
                out.append(float(item[0]))
            else:
                return None
        return out
    return None


def _walk_values(obj: Any, keys: Iterable[str]) -> Optional[List[float]]:
    key_set = set(keys)
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in key_set:
                vals = _numeric_list(value)
                if vals:
                    return vals
        for value in obj.values():
            vals = _walk_values(value, key_set)
            if vals:
                return vals
    elif isinstance(obj, list):
        for value in obj:
            vals = _walk_values(value, key_set)
            if vals:
                return vals
    return None


def _npz_values(path: Path, keys: Iterable[str]) -> Optional[List[float]]:
    try:
        import numpy as np
    except Exception:
        return None

    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None

    for key in keys:
        if key not in data.files:
            continue
        arr = np.asarray(data[key]).reshape(-1)
        if arr.size == 0:
            continue
        try:
            return [float(x) for x in arr.tolist()]
        except (TypeError, ValueError):
            continue
    return None


def _maybe_scale_metric(metric_name: str, value: float) -> float:
    if "plddt" in metric_name.lower() and abs(value) <= 1.5:
        return float(value) * 100.0
    return float(value)


def _scalar_metrics(summary: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(summary, dict):
        return out
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
    return out


def _chain_metrics(summary: Any, chains: List[Tuple[str, str]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not isinstance(summary, dict) or not chains:
        return out

    for key, value in summary.items():
        if not str(key).startswith("chain_") or str(key).startswith("chain_pair_"):
            continue
        vals = _numeric_list(value)
        if not vals or len(vals) != len(chains):
            continue
        metric_name = str(key)[len("chain_"):]
        out[metric_name] = {
            cid: _maybe_scale_metric(metric_name, float(vals[i]))
            for i, (cid, _) in enumerate(chains)
        }
    return out


def _split_per_chain(
    values: Optional[List[float]],
    chains: List[Tuple[str, str]],
) -> Dict[str, List[float]]:
    if not values:
        return {}

    total_len = sum(len(seq) for _, seq in chains)
    if len(values) != total_len:
        if len(chains) == 1 and len(values) >= len(chains[0][1]):
            cid, seq = chains[0]
            return {cid: values[: len(seq)]}
        return {}

    out: Dict[str, List[float]] = {}
    offset = 0
    for cid, seq in chains:
        n = len(seq)
        out[cid] = values[offset : offset + n]
        offset += n
    return out


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


def _extract_residue_plddt_from_cif(
    out_dir: Path,
    chains: List[Tuple[str, str]],
) -> Dict[str, List[float]]:
    if not chains:
        return {}

    for path in sorted(out_dir.rglob("*.cif")):
        by_asym: Dict[str, Dict[int, List[float]]] = {}
        asym_order: List[str] = []

        for row in _atom_site_rows_from_cif(path):
            asym = row.get("_atom_site.label_asym_id") or row.get("_atom_site.auth_asym_id")
            seq_raw = row.get("_atom_site.label_seq_id") or row.get("_atom_site.auth_seq_id")
            b_raw = row.get("_atom_site.B_iso_or_equiv")
            if not asym or not seq_raw or not b_raw or seq_raw in {".", "?"}:
                continue
            try:
                seq_id = int(float(seq_raw))
                b_val = float(b_raw)
            except ValueError:
                continue
            if seq_id <= 0:
                continue
            if asym not in by_asym:
                by_asym[asym] = {}
                asym_order.append(asym)
            by_asym[asym].setdefault(seq_id, []).append(b_val)

        if not by_asym:
            continue

        per_chain: Dict[str, List[float]] = {}
        for chain_idx, (cid, seq) in enumerate(chains):
            if chain_idx >= len(asym_order):
                break
            residues = by_asym.get(asym_order[chain_idx], {})
            all_vals = [v for vals in residues.values() for v in vals]
            if not all_vals:
                continue
            fallback = float(sum(all_vals) / len(all_vals))
            values: List[float] = []
            for residue_idx in range(1, len(seq) + 1):
                vals = residues.get(residue_idx)
                if vals:
                    values.append(float(sum(vals) / len(vals)))
                else:
                    values.append(fallback)
            per_chain[cid] = values

        if per_chain:
            return per_chain

    return {}


def _extract_residue_plddt(
    out_dir: Path,
    chains: List[Tuple[str, str]],
) -> Dict[str, List[float]]:


    candidate_keys = (
        "residue_plddt",
        "residue_plddts",
        "token_plddt",
        "token_plddts",
        "plddt",
        "plddts",
    )

    for pattern in ("*confidence*.json", "*full*.json", "*.json"):
        for path in sorted(out_dir.rglob(pattern)):
            data = _load_json(path)
            vals = _walk_values(data, candidate_keys)
            per_chain = _split_per_chain(vals, chains)
            if per_chain:
                return per_chain
    for pattern in ("*confidence*.npz", "*full*.npz", "*.npz"):
        for path in sorted(out_dir.rglob(pattern)):
            vals = _npz_values(path, candidate_keys)
            per_chain = _split_per_chain(vals, chains)
            if per_chain:
                return per_chain
    per_chain = _extract_residue_plddt_from_cif(out_dir, chains)
    if per_chain:
        return per_chain
    return {}


def run_protenix_confidence_multichain(
    pred_name: Optional[str] = None,
    chains: Optional[List[Tuple[str, str]]] = None,
    metric: str = "plddt",
    device: Optional[str] = None,
    seed: int = 101,
    model_name: str = "protenix_mini_esm_v0.5.0",
    conda_env: Optional[str] = None,
    timeout: Optional[int] = None,
    use_msa: Optional[bool] = None,
    cycle: Optional[int] = None,
    step: Optional[int] = None,
    sample: Optional[int] = None,
    use_default_params: Optional[bool] = None,
) -> Dict[str, Any]:


    del device
    pred_name, chains = _normalise_inputs(pred_name, chains)
    if not chains:
        return {"metrics": {}, "residue_plddt": {}, "out_dir": None}

    cache_key = _cache_key(pred_name, chains, metric, seed, model_name)
    if cache_key in _CONF_CACHE:
        return _CONF_CACHE[cache_key]

    out_dir, preview = _run_protenix(
        pred_name=pred_name,
        chains=chains,
        seed=seed,
        model_name=model_name,
        conda_env=conda_env,
        timeout=timeout,
        use_msa=use_msa,
        cycle=cycle,
        step=step,
        sample=sample,
        use_default_params=use_default_params,
    )
    if out_dir is None:
        result = {"metrics": {}, "residue_plddt": {}, "out_dir": None}
        result["input_json"] = preview.get("input_json")
        result["status_json"] = preview.get("status_json")
        result["protenix_output_dir"] = preview.get("protenix_output_dir")
        _CONF_CACHE[cache_key] = result
        return result

    summary_json = _first_json(out_dir, "*summary_confidence*.json")
    cif_path = _first_cif(out_dir)
    summary = _load_json(summary_json) if summary_json else None
    metrics = _scalar_metrics(summary)
    chain_metrics = _chain_metrics(summary, chains)
    residue_plddt = _extract_residue_plddt(out_dir, chains)

    result = {
        "metrics": metrics,
        "chain_metrics": chain_metrics,
        "residue_plddt": residue_plddt,
        "input_json": preview.get("input_json"),
        "status_json": preview.get("status_json"),
        "summary_json": str(summary_json) if summary_json else None,
        "cif_path": str(cif_path) if cif_path else None,
        "out_dir": str(out_dir),
    }
    _CONF_CACHE[cache_key] = result
    return result


def run_protenix_confidence_complex(
    pred_name: Optional[str] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
    constraint: Optional[Dict[str, Any]] = None,
    covalent_bonds: Optional[List[Dict[str, Any]]] = None,
    metric: str = "plddt",
    device: Optional[str] = None,
    seed: int = 101,
    model_name: str = "protenix_mini_esm_v0.5.0",
    conda_env: Optional[str] = None,
    timeout: Optional[int] = None,
    use_msa: Optional[bool] = None,
    cycle: Optional[int] = None,
    step: Optional[int] = None,
    sample: Optional[int] = None,
    use_default_params: Optional[bool] = None,
) -> Dict[str, Any]:

    del device
    entities = list(entities or [])
    if not entities:
        return {
            "metrics": {},
            "chain_metrics": {},
            "residue_plddt": {},
            "entities": [],
            "input_json": None,
            "out_dir": None,
        }

    pred_name = str(pred_name or "protenix_complex")
    sequences, report, polymer_units, metric_units = _normalise_entities(entities)
    normalised_entities = [
        {"entity": sequence_entry, "report": report_entry}
        for sequence_entry, report_entry in zip(sequences, report)
    ]

    cache_key = _json_cache_key(
        pred_name,
        normalised_entities,
        _normalise_constraint(constraint),
        covalent_bonds,
        metric,
        seed,
        model_name,
        _complex_runtime_options(
            conda_env=conda_env,
            use_msa=use_msa,
            cycle=cycle,
            step=step,
            sample=sample,
            use_default_params=use_default_params,
        ),
    )
    if cache_key in _CONF_CACHE:
        return _CONF_CACHE[cache_key]

    out_dir, preview = _run_protenix_complex(
        pred_name=pred_name,
        entities=entities,
        seed=seed,
        model_name=model_name,
        conda_env=conda_env,
        constraint=constraint,
        covalent_bonds=covalent_bonds,
        timeout=timeout,
        use_msa=use_msa,
        cycle=cycle,
        step=step,
        sample=sample,
        use_default_params=use_default_params,
    )
    if out_dir is None:
        result = {
            "metrics": {},
            "chain_metrics": {},
            "residue_plddt": {},
            "entities": report,
            "polymer_units": polymer_units,
            "metric_units": metric_units,
            "input_json": preview.get("input_json"),
            "status_json": preview.get("status_json"),
            "protenix_output_dir": preview.get("protenix_output_dir"),
            "out_dir": None,
        }
        _CONF_CACHE[cache_key] = result
        return result

    summary_json = _first_json(out_dir, "*summary_confidence*.json")
    cif_path = _first_cif(out_dir)
    summary = _load_json(summary_json) if summary_json else None
    metrics = _scalar_metrics(summary)
    chain_metrics = _chain_metrics(summary, metric_units)
    residue_plddt = _extract_residue_plddt(out_dir, polymer_units)

    result = {
        "metrics": metrics,
        "chain_metrics": chain_metrics,
        "residue_plddt": residue_plddt,
        "entities": report,
        "polymer_units": polymer_units,
        "metric_units": metric_units,
        "input_json": preview.get("input_json"),
        "status_json": preview.get("status_json"),
        "summary_json": str(summary_json) if summary_json else None,
        "cif_path": str(cif_path) if cif_path else None,
        "out_dir": str(out_dir),
    }
    _CONF_CACHE[cache_key] = result
    return result


def run_protenix_confidence_complex_batch(
    jobs: Optional[List[Dict[str, Any]]] = None,
    metric: str = "plddt",
    device: Optional[str] = None,
    seed: int = 101,
    model_name: str = "protenix_mini_esm_v0.5.0",
    conda_env: Optional[str] = None,
    timeout: Optional[int] = None,
    use_msa: Optional[bool] = None,
    cycle: Optional[int] = None,
    step: Optional[int] = None,
    sample: Optional[int] = None,
    use_default_params: Optional[bool] = None,
) -> List[Dict[str, Any]]:


    del device
    raw_jobs = list(jobs or [])
    if not raw_jobs:
        return []

    pred_names: List[str] = []
    seen_names = set()
    for index, raw_job in enumerate(raw_jobs):
        if not isinstance(raw_job, dict):
            raise TypeError(f"Protenix batch job #{index} must be a dictionary")
        pred_name = _batch_pred_name(raw_job, index)
        if pred_name in seen_names:
            raise ValueError(f"Duplicate Protenix batch pred_name: {pred_name!r}")
        seen_names.add(pred_name)
        pred_names.append(pred_name)

    runtime_options = _complex_runtime_options(
        conda_env=conda_env,
        use_msa=use_msa,
        cycle=cycle,
        step=step,
        sample=sample,
        use_default_params=use_default_params,
    )
    results: List[Optional[Dict[str, Any]]] = [None] * len(raw_jobs)
    misses: List[Dict[str, Any]] = []
    miss_metadata: List[Dict[str, Any]] = []

    for index, (raw_job, pred_name) in enumerate(zip(raw_jobs, pred_names)):
        entities = raw_job.get("entities")
        if not isinstance(entities, list) or not entities:
            raise ValueError(
                f"Protenix batch job {pred_name!r} needs a non-empty entities list"
            )
        sequences, report, polymer_units, metric_units = _normalise_entities(entities)
        normalised_entities = [
            {"entity": sequence_entry, "report": report_entry}
            for sequence_entry, report_entry in zip(sequences, report)
        ]
        constraint = raw_job.get("constraint")
        covalent_bonds = raw_job.get("covalent_bonds")
        cache_key = _json_cache_key(
            pred_name,
            normalised_entities,
            _normalise_constraint(constraint),
            covalent_bonds,
            metric,
            seed,
            model_name,
            runtime_options,
        )
        cached = _CONF_CACHE.get(cache_key)


        if cached is not None and cached.get("out_dir") is not None:
            result = dict(cached)
            result.update({
                "pred_name": pred_name,
                "batch_index": index + 1,
                "cache_hit": True,
                "status": "ok",
                "runner_status": "cache_hit",
                "batch_status": "cache_hit",
            })
            results[index] = result
            continue

        misses.append({
            "pred_name": pred_name,
            "entities": entities,
            "constraint": constraint,
            "covalent_bonds": covalent_bonds,
        })
        miss_metadata.append({
            "result_index": index,
            "batch_index": index + 1,
            "pred_name": pred_name,
            "report": report,
            "polymer_units": polymer_units,
            "metric_units": metric_units,
            "cache_key": cache_key,
        })

    routed = _run_protenix_complex_batch(
        misses,
        seed=seed,
        model_name=model_name,
        conda_env=conda_env,
        timeout=timeout,
        use_msa=use_msa,
        cycle=cycle,
        step=step,
        sample=sample,
        use_default_params=use_default_params,
    ) if misses else []

    if len(routed) != len(miss_metadata):
        raise RuntimeError(
            "Protenix batch runner returned a different number of routed jobs"
        )

    for (out_dir, preview), metadata in zip(routed, miss_metadata):
        index = int(metadata["result_index"])
        pred_name = str(metadata["pred_name"])
        report = list(metadata["report"])
        polymer_units = list(metadata["polymer_units"])
        metric_units = list(metadata["metric_units"])
        if out_dir is None:
            result = {
                "metrics": {},
                "chain_metrics": {},
                "residue_plddt": {},
                "entities": report,
                "polymer_units": polymer_units,
                "metric_units": metric_units,
                "input_json": preview.get("input_json"),
                "status_json": preview.get("status_json"),
                "protenix_output_dir": preview.get("protenix_output_dir"),
                "batch_output_dir": preview.get("batch_output_dir"),
                "out_dir": None,
                "status": preview.get("runner_status", "failed"),
            }
        else:
            summary_json = _first_json(out_dir, "*summary_confidence*.json")
            cif_path = _first_cif(out_dir)
            summary = _load_json(summary_json) if summary_json else None
            result = {
                "metrics": _scalar_metrics(summary),
                "chain_metrics": _chain_metrics(summary, metric_units),
                "residue_plddt": _extract_residue_plddt(out_dir, polymer_units),
                "entities": report,
                "polymer_units": polymer_units,
                "metric_units": metric_units,
                "input_json": preview.get("input_json"),
                "status_json": preview.get("status_json"),
                "summary_json": str(summary_json) if summary_json else None,
                "cif_path": str(cif_path) if cif_path else None,
                "batch_output_dir": preview.get("batch_output_dir"),
                "out_dir": str(out_dir),
                "status": "ok",
            }


            _CONF_CACHE[str(metadata["cache_key"])] = dict(result)

        result.update({
            "pred_name": pred_name,
            "batch_index": int(metadata["batch_index"]),
            "cache_hit": False,
            "runner_status": preview.get("runner_status"),
            "batch_status": preview.get("batch_status"),
        })
        results[index] = result

    if any(result is None for result in results):
        raise RuntimeError("Protenix batch routing left an input without a result")
    return [dict(result) for result in results if result is not None]


def run_protenix_plddt_complex(
    pred_name: Optional[str] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
    metric: str = "plddt",
    **kwargs: Any,
) -> float:


    confidence = run_protenix_confidence_complex(
        pred_name=pred_name,
        entities=entities,
        metric=metric,
        **kwargs,
    )
    return float(confidence.get("metrics", {}).get(metric, 0.0))


def run_protenix_plddt_multichain(
    pred_name: Optional[str] = None,
    chains: Optional[List[Tuple[str, str]]] = None,
    metric: str = "plddt",
    device: Optional[str] = None,
    seed: int = 101,
    model_name: str = "protenix_mini_esm_v0.5.0",
    conda_env: Optional[str] = None,
    timeout: Optional[int] = None,
    use_msa: Optional[bool] = None,
    cycle: Optional[int] = None,
    step: Optional[int] = None,
    sample: Optional[int] = None,
    use_default_params: Optional[bool] = None,
) -> float:


    pred_name, chains = _normalise_inputs(pred_name, chains)
    if not chains:
        return 0.0

    cache_key = _cache_key(pred_name, chains, metric, seed, model_name)
    if cache_key in _SCALAR_CACHE:
        return _SCALAR_CACHE[cache_key]

    confidence = run_protenix_confidence_multichain(
        pred_name=pred_name,
        chains=chains,
        metric=metric,
        device=device,
        seed=seed,
        model_name=model_name,
        conda_env=conda_env,
        timeout=timeout,
        use_msa=use_msa,
        cycle=cycle,
        step=step,
        sample=sample,
        use_default_params=use_default_params,
    )
    value = float(confidence.get("metrics", {}).get(metric, 0.0))
    _SCALAR_CACHE[cache_key] = value
    return value
