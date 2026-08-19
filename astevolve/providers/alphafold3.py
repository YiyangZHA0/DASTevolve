

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from astevolve.runtime.conda import resolve_alphafold3_conda_env

from .alphafold3_input import (
    build_alphafold3_input,
    write_alphafold3_input_json,
)
from .alphafold3_output import (
    empty_alphafold3_confidence,
    extract_residue_plddt_from_cif,
    find_alphafold3_artifacts,
    parse_alphafold3_output,
)
from .alphafold3_runner import (
    alphafold3_database_dirs,
    alphafold3_model_dir,
    alphafold3_source_dir,
    alphafold3_tmp_dir,
    describe_alphafold3_runtime,
    run_alphafold3_job,
)


_CONFIDENCE_CACHE: Dict[str, Dict[str, Any]] = {}


def _bool_from_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_pipeline_mode(
    run_data_pipeline: Optional[bool],
    mode: Optional[str],
) -> bool:
    if run_data_pipeline is not None:
        return bool(run_data_pipeline)
    if mode:
        normalized = str(mode).strip().lower().replace("-", "_")
        if normalized in {"msa_free", "inference", "model_only", "fast"}:
            return False
        if normalized in {"full", "pipeline", "data_pipeline"}:
            return True
        raise ValueError("AF3 mode must be 'msa_free' or 'full'")
    return _bool_from_env("ASTEVOLVE_AF3_RUN_DATA_PIPELINE", False)


def _resolve_model_dir(
    model_dir: Optional[str | Path],
    model_name: Optional[str],
) -> Optional[str | Path]:
    if model_dir is not None:
        return model_dir
    if model_name:
        candidate = Path(str(model_name)).expanduser()
        if candidate.is_dir() or "/" in str(model_name) or "\\" in str(model_name):
            return candidate
    return None


def _integer_setting(
    explicit: Optional[int],
    env_name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    raw: Any = explicit
    if raw is None:
        raw = os.environ.get(env_name, default)
    try:
        return max(minimum, int(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be an integer, got {raw!r}") from exc


def _gpu_from_device(device: Optional[str], gpu_device: Optional[int]) -> int:
    if device and ":" in str(device):
        try:
            return max(0, int(str(device).rsplit(":", 1)[1]))
        except ValueError:
            pass
    return _integer_setting(
        gpu_device,
        "ASTEVOLVE_AF3_GPU_DEVICE",
        0,
        minimum=0,
    )


def _job_key(job: Dict[str, Any], settings: Dict[str, Any]) -> str:
    payload = json.dumps(
        {"job": job, "settings": settings},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    name = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(job.get("name") or "alphafold3")
    ).strip("_")
    return f"{name or 'alphafold3'}__{digest}"


def run_alphafold3_confidence_complex(
    pred_name: Optional[str] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
    constraint: Optional[Dict[str, Any]] = None,
    covalent_bonds: Optional[Sequence[Any]] = None,
    metric: str = "plddt",
    device: Optional[str] = None,
    seed: int = 101,
    model_name: Optional[str] = None,
    model_dir: Optional[str | Path] = None,
    conda_env: Optional[str] = None,
    timeout: Optional[int] = None,
    run_data_pipeline: Optional[bool] = None,
    db_dir: Any = None,
    num_recycles: Optional[int] = None,
    num_diffusion_samples: Optional[int] = None,
    flash_attention_implementation: Optional[str] = None,
    gpu_device: Optional[int] = None,
    mode: Optional[str] = None,
    **_: Any,
) -> Dict[str, Any]:


    del metric
    if constraint:
        raise NotImplementedError(
            "AlphaFold 3 does not support ASTevolve/Protenix constraint payloads"
        )
    entity_list = list(entities or [])
    if not entity_list:
        return empty_alphafold3_confidence()

    name = str(pred_name or "alphafold3_complex")
    use_pipeline = _resolve_pipeline_mode(run_data_pipeline, mode)
    job, preview = build_alphafold3_input(
        name,
        entity_list,
        seed=seed,
        covalent_bonds=covalent_bonds,
        run_data_pipeline=use_pipeline,
    )
    resolved_model_dir = _resolve_model_dir(model_dir, model_name)
    effective_model_dir = alphafold3_model_dir(resolved_model_dir)
    effective_conda_env = resolve_alphafold3_conda_env(conda_env)
    effective_db_dirs = alphafold3_database_dirs(db_dir)
    effective_num_recycles = _integer_setting(
        num_recycles,
        "ASTEVOLVE_AF3_NUM_RECYCLES",
        10,
        minimum=1,
    )
    effective_num_samples = _integer_setting(
        num_diffusion_samples,
        "ASTEVOLVE_AF3_NUM_DIFFUSION_SAMPLES",
        1,
        minimum=1,
    )
    effective_attention = str(
        flash_attention_implementation
        or os.environ.get("ASTEVOLVE_AF3_FLASH_ATTENTION", "triton")
    ).strip().lower()
    effective_gpu_device = _gpu_from_device(device, gpu_device)
    settings = {
        "model_dir": str(effective_model_dir),
        "conda_env": effective_conda_env,
        "run_data_pipeline": use_pipeline,
        "db_dir": [str(path) for path in effective_db_dirs],
        "num_recycles": effective_num_recycles,
        "num_diffusion_samples": effective_num_samples,
        "flash_attention_implementation": effective_attention,
        "gpu_device": effective_gpu_device,
    }
    cache_key = json.dumps(
        {"job": job, "settings": settings},
        sort_keys=True,
        default=str,
    )
    if cache_key in _CONFIDENCE_CACHE:
        return dict(_CONFIDENCE_CACHE[cache_key])

    job_key = _job_key(job, settings)
    input_path = alphafold3_tmp_dir() / job_key / "input.json"
    preview = write_alphafold3_input_json(
        input_path,
        name,
        entity_list,
        seed=seed,
        covalent_bonds=covalent_bonds,
        run_data_pipeline=use_pipeline,
    )
    output_dir, run_preview = run_alphafold3_job(
        input_json=input_path,
        job_key=job_key,
        model_dir=effective_model_dir,
        conda_env=effective_conda_env,
        timeout=timeout,
        run_data_pipeline=use_pipeline,
        db_dir=effective_db_dirs,
        num_recycles=effective_num_recycles,
        num_diffusion_samples=effective_num_samples,
        flash_attention_implementation=effective_attention,
        gpu_device=effective_gpu_device,
    )
    combined_preview = {**preview, **run_preview}
    if output_dir is None:
        result = empty_alphafold3_confidence(
            preview=combined_preview,
            input_json=str(input_path),
            status_json=run_preview.get("status_json"),
        )
        error = run_preview.get("error") or run_preview.get("stderr_tail")
        if error:
            result["warnings"].append(str(error)[-4000:])
        result["status"] = run_preview.get("status")
        result["alphafold3_output_dir"] = run_preview.get("alphafold3_output_dir")
    else:
        result = parse_alphafold3_output(
            output_dir,
            pred_name=name,
            preview=combined_preview,
            input_json=str(input_path),
            status_json=run_preview.get("status_json"),
        )
        result["status"] = run_preview.get("status")
        result["alphafold3_output_dir"] = str(output_dir)
    result["model_dir"] = str(effective_model_dir)
    result["run_data_pipeline"] = use_pipeline
    if result.get("status") == "ok" and result.get("cif_path"):
        _CONFIDENCE_CACHE[cache_key] = dict(result)
    return result


def run_alphafold3_confidence_multichain(
    pred_name: Optional[str] = None,
    chains: Optional[List[Tuple[str, str]]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:


    normalized = [(str(chain_id), str(sequence)) for chain_id, sequence in (chains or [])]
    if not normalized:
        return empty_alphafold3_confidence()
    return run_alphafold3_confidence_complex(
        pred_name=pred_name or "__".join(chain_id for chain_id, _ in normalized),
        entities=[
            {"type": "protein", "id": chain_id, "sequence": sequence}
            for chain_id, sequence in normalized
        ],
        **kwargs,
    )


def _scalar(result: Dict[str, Any], metric: str) -> float:
    metrics = result.get("metrics", {}) or {}
    value = metrics.get(metric)
    if value is None and metric != "plddt":
        value = metrics.get("plddt")
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def run_alphafold3_plddt_multichain(
    pred_name: Optional[str] = None,
    chains: Optional[List[Tuple[str, str]]] = None,
    metric: str = "plddt",
    **kwargs: Any,
) -> float:


    return _scalar(
        run_alphafold3_confidence_multichain(
            pred_name=pred_name,
            chains=chains,
            metric=metric,
            **kwargs,
        ),
        metric,
    )


def run_alphafold3_plddt_complex(
    pred_name: Optional[str] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
    metric: str = "plddt",
    **kwargs: Any,
) -> float:


    return _scalar(
        run_alphafold3_confidence_complex(
            pred_name=pred_name,
            entities=entities,
            metric=metric,
            **kwargs,
        ),
        metric,
    )


def describe_alphafold3_setup() -> Dict[str, Any]:


    report = describe_alphafold3_runtime()
    weight_path = Path(str(report.get("model_dir") or "")) / "af3.bin.zst"
    weight_bytes = weight_path.stat().st_size if weight_path.is_file() else None
    expected_bytes = 1_020_548_524
    expected_commit = "7b197fe859790fc3e04d03ea70dd0b9ba48881c9"
    actual_commit = report.get("source_commit_actual")
    source_revision_ok = actual_commit == expected_commit
    report.update(
        {
            "source_version": "v3.0.3",
            "source_commit": expected_commit,
            "source_revision_ok": source_revision_ok,
            "weight_filename": "af3.bin.zst",
            "weight_path": str(weight_path),
            "weight_bytes": weight_bytes,
            "weight_expected_bytes": expected_bytes,
            "weight_size_ok": weight_bytes == expected_bytes,
            "weight_expected_md5": "9d715a274286c5e4777067dec3910c04",
            "installation_ready": bool(
                report.get("source_available")
                and source_revision_ok
                and report.get("environment_ready")
                and weight_bytes == expected_bytes
            ),
            "msa_free_available": bool(
                report.get("source_available")
                and source_revision_ok
                and report.get("environment_ready")
                and report.get("gpu_dependencies_ready")
                and weight_bytes == expected_bytes
            ),
        }
    )
    return report


__all__ = [
    "alphafold3_database_dirs",
    "alphafold3_model_dir",
    "alphafold3_source_dir",
    "build_alphafold3_input",
    "describe_alphafold3_setup",
    "extract_residue_plddt_from_cif",
    "find_alphafold3_artifacts",
    "parse_alphafold3_output",
    "run_alphafold3_confidence_complex",
    "run_alphafold3_confidence_multichain",
    "run_alphafold3_plddt_complex",
    "run_alphafold3_plddt_multichain",
    "write_alphafold3_input_json",
]
