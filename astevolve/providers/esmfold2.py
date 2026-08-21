

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from threading import Lock
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_CONF_CACHE: Dict[str, Dict[str, Any]] = {}
_SCALAR_CACHE: Dict[str, float] = {}
_MODEL_CACHE: Dict[str, Tuple[Any, Any, Any, Any]] = {}
_MODEL_LOAD_LOCK = Lock()
_MODEL_INFERENCE_LOCK = Lock()

DEFAULT_LOCAL_MODEL = "biohub/ESMFold2"
DEFAULT_API_MODEL = "esmfold2-fast-2026-05"
DEFAULT_API_URL = "https://biohub.ai"


CONFIDENCE_UNITS = {
    "plddt": "0-100",
    "ptm": "0-1",
    "iptm": "0-1",
}


class ESMFold2WeightsRequired(RuntimeError):


    pass


class ESMFold2TokenRequired(RuntimeError):


    pass


def clear_esmfold2_model_cache(*, clear_confidence_cache: bool = False) -> None:


    with _MODEL_LOAD_LOCK:
        for model, _builder, _protein_input, _structure_input in _MODEL_CACHE.values():
            try:
                if hasattr(model, "cpu"):
                    model.cpu()
            except Exception:
                pass
        _MODEL_CACHE.clear()
    if clear_confidence_cache:
        _CONF_CACHE.clear()
        _SCALAR_CACHE.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _default_pred_name(chains: List[Tuple[str, str]]) -> str:
    names = [str(cid) for cid, _ in chains if cid]
    return "__".join(names) if names else "pred"


def _normalise_inputs(
    pred_name: Optional[str],
    chains: Optional[List[Tuple[str, str]]],
) -> Tuple[str, List[Tuple[str, str]]]:
    if chains is None and isinstance(pred_name, list):
        chains = pred_name
        pred_name = None
    norm_chains = [(str(cid), str(seq)) for cid, seq in (chains or [])]
    return str(pred_name or _default_pred_name(norm_chains)), norm_chains


def _tmp_root() -> Path:
    root = os.environ.get("ASTEVOLVE_ESMFOLD2_TMP")
    path = Path(root) if root else Path(tempfile.gettempdir()) / "astevolve_esmfold2"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _find_conda_exe() -> str:
    candidates = [
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
    ]

    prefix = Path(sys.prefix)
    for root in (prefix, prefix.parent.parent if prefix.parent.name == "envs" else None):
        if root is None:
            continue
        candidates.extend([
            str(root / "Scripts" / "conda.exe"),
            str(root / "bin" / "conda"),
        ])

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise RuntimeError("Cannot find conda executable. Set CONDA_EXE or run from a conda shell.")


def _cache_key(
    pred_name: str,
    chains: List[Tuple[str, str]],
    mode: str,
    model_name: str,
    metric: str,
    seed: int,
    num_loops: int,
    num_sampling_steps: int,
    num_diffusion_samples: int,
    msa_signature: str = "",
) -> str:
    seq_key = "|".join(f"{cid}:{seq}" for cid, seq in chains)
    return (
        f"{pred_name}|{seq_key}|{mode}|{model_name}|{metric}|{seed}|"
        f"{num_loops}|{num_sampling_steps}|{num_diffusion_samples}|msa={msa_signature}"
    )


def _msa_max_sequences(value: Optional[int]) -> Optional[int]:
    if value is not None:
        selected = int(value)
    else:
        raw = str(os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_MAX_SEQUENCES") or "").strip()
        if not raw:
            return None
        selected = int(raw)
    if selected < 1:
        raise ValueError("ESMFold2 MSA max_sequences must be positive")
    return selected


def _msa_signature(
    msa_mode: Optional[str],
    msa_paths: Optional[Dict[str, str]],
    msa_cache_dir: Optional[str],
    msa_max_sequences: Optional[int],
) -> str:
    return json.dumps(
        {
            "mode": str(msa_mode or os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_MODE", "off")).lower(),
            "paths": msa_paths if msa_paths is not None else os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_PATHS", ""),
            "cache_dir": msa_cache_dir or os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_CACHE", ""),
            "max_sequences": _msa_max_sequences(msa_max_sequences),
        },
        sort_keys=True,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return default


def _to_float_list(value: Any) -> List[float]:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "reshape"):
            value = value.reshape(-1)
        if hasattr(value, "tolist"):
            raw = value.tolist()
        else:
            raw = list(value)
        if raw and isinstance(raw[0], list):
            raw = [x for row in raw for x in row]
        return [float(x) for x in raw]
    except Exception:
        return []


def _split_residue_values(values: List[float], chains: List[Tuple[str, str]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    offset = 0
    for cid, seq in chains:
        length = len(seq)
        out[cid] = values[offset : offset + length]
        offset += length
    return out


def _chain_metric_from_residue_values(residue_values: Dict[str, List[float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for cid, values in residue_values.items():
        out[cid] = float(sum(values) / max(1, len(values))) if values else 0.0
    return out


def _unit_interval_scalar(value: Any, *, name: str) -> float:
    score = _safe_float(value, default=float("nan"))
    if not math.isfinite(score):
        raise ValueError(f"ESMFold2 {name} must be finite; received {score!r}")
    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"ESMFold2 {name} must be in the normalized 0-1 range; received {score!r}"
        )
    return score


def _normalized_residue_plddt(
    result: Any,
    chains: List[Tuple[str, str]],
) -> List[float]:
    values = _to_float_list(getattr(result, "plddt", []))
    expected_count = sum(len(sequence) for _chain_id, sequence in chains)
    if len(values) != expected_count:
        raise ValueError(
            "ESMFold2 residue pLDDT count does not match the input chains: "
            f"received {len(values)}, expected {expected_count}"
        )
    return [
        100.0 * _unit_interval_scalar(value, name=f"residue pLDDT[{index}]")
        for index, value in enumerate(values)
    ]


def _write_cif_if_possible(result: Any, path: Path) -> Optional[str]:
    try:
        complex_obj = getattr(result, "complex", None)
        if complex_obj is None or not hasattr(complex_obj, "to_mmcif"):
            return None
        path.write_text(complex_obj.to_mmcif(), encoding="utf-8")
        return str(path)
    except Exception:
        return None


def _result_to_confidence(
    result: Any,
    chains: List[Tuple[str, str]],
    metric: str,
    mode: str,
    model_name: str,
    out_dir: Path,
    pred_name: str,
    write_cif: bool,
    msa_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    normalized_plddt = _normalized_residue_plddt(result, chains)
    residue_values = _split_residue_values(normalized_plddt, chains)
    chain_plddt = _chain_metric_from_residue_values(residue_values)


    mean_plddt = (
        float(sum(normalized_plddt) / len(normalized_plddt))
        if normalized_plddt
        else 0.0
    )

    metrics = {
        "plddt": mean_plddt,
        "ptm": _unit_interval_scalar(getattr(result, "ptm", 0.0), name="pTM"),
        "iptm": _unit_interval_scalar(getattr(result, "iptm", 0.0), name="ipTM"),
    }
    metrics[metric] = metrics.get(metric, mean_plddt)

    out_dir.mkdir(parents=True, exist_ok=True)
    cif_path = None
    if write_cif:
        cif_path = _write_cif_if_possible(result, out_dir / f"{pred_name}.cif")

    summary = {
        "provider": "esmfold2",
        "mode": mode,
        "model_name": model_name,
        "metrics": metrics,
        "confidence_units": dict(CONFIDENCE_UNITS),
        "chain_metrics": {"plddt": chain_plddt},
        "residue_plddt": residue_values,
        "out_dir": str(out_dir),
        "cif_path": cif_path,
        "msa": msa_metadata or {},
    }
    (out_dir / f"{pred_name}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    summary["summary_json"] = str(out_dir / f"{pred_name}_summary.json")
    return summary


def _run_local_esmfold2(
    pred_name: str,
    chains: List[Tuple[str, str]],
    metric: str,
    seed: int,
    model_name: str,
    device: Optional[str],
    num_loops: int,
    num_sampling_steps: int,
    num_diffusion_samples: int,
    write_cif: bool,
    msa_mode: Optional[str] = None,
    msa_paths: Optional[Dict[str, str]] = None,
    msa_cache_dir: Optional[str] = None,
    msa_max_sequences: Optional[int] = None,
) -> Dict[str, Any]:
    allow_download = _as_bool_env("ASTEVOLVE_ESMFOLD2_ALLOW_DOWNLOAD", False)
    local_files_only = not allow_download
    requested_device = str(device or "auto").strip().lower()
    esmc_model_name = str(os.environ.get("ASTEVOLVE_ESMFOLD2_ESMC_MODEL", "")).strip()
    cache_key = (
        f"{model_name}|{requested_device}|local_files_only={local_files_only}|"
        f"esmc_model={esmc_model_name}"
    )
    with _MODEL_LOAD_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is None:
            try:
                import torch
                from esm.models.esmfold2 import (
                    ESMFold2InputBuilder,
                    ProteinInput,
                    StructurePredictionInput,
                )
                from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
            except Exception as exc:
                raise RuntimeError(
                    "ESMFold2 python dependencies are not available. Install the Biohub esm package "
                    "before running ESMFold2 inference."
                ) from exc

            try:
                load_kwargs: Dict[str, Any] = {"local_files_only": local_files_only}
                if esmc_model_name:
                    from transformers.models.esmfold2.configuration_esmfold2 import (
                        ESMFold2Config,
                    )

                    config = ESMFold2Config.from_pretrained(
                        model_name,
                        local_files_only=local_files_only,
                    )
                    config.esmc_id = esmc_model_name
                    load_kwargs["config"] = config
                model = ESMFold2Model.from_pretrained(model_name, **load_kwargs).eval()
            except Exception as exc:
                if not allow_download:
                    raise ESMFold2WeightsRequired(
                        "ESMFold2 weights are not available in the local Hugging Face cache. "
                        "No download was attempted. Set ASTEVOLVE_ESMFOLD2_ALLOW_DOWNLOAD=1 "
                        f"only after you are ready to download '{model_name}'."
                    ) from exc
                raise

            if requested_device not in {"auto", "none"}:
                model = model.to(requested_device)
            elif hasattr(torch, "cuda") and torch.cuda.is_available():
                model = model.cuda()
            cached = (
                model,
                ESMFold2InputBuilder,
                ProteinInput,
                StructurePredictionInput,
            )
            _MODEL_CACHE[cache_key] = cached
    model, input_builder_cls, protein_input_cls, structure_input_cls = cached

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("ESMFold2 local inference requires torch") from exc

    from .remote_msa import resolve_chain_msa_paths

    configured_msa_paths = bool(str(os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_PATHS") or "").strip())
    msa_requested = (
        bool(msa_paths)
        or configured_msa_paths
        or str(msa_mode or "off").lower() in {"remote", "file", "local"}
    )
    msa_encoder = getattr(getattr(model, "config", None), "msa_encoder", None)
    if msa_requested and not bool(getattr(msa_encoder, "enabled", False)):
        raise ValueError(
            f"Model {model_name!r} has no enabled MSA encoder; use full biohub/ESMFold2, "
            "not ESMFold2-Fast, when MSA input is enabled."
        )
    resolved_msa, msa_metadata = resolve_chain_msa_paths(
        chains,
        msa_mode=msa_mode,
        msa_paths=msa_paths,
        cache_dir=msa_cache_dir,
    )
    if resolved_msa:
        from esm.utils.msa import MSA

    protein_inputs = []
    for cid, seq in chains:
        protein_kwargs: Dict[str, Any] = {"id": cid, "sequence": seq}
        if cid in resolved_msa:
            msa = MSA.from_a3m(resolved_msa[cid], max_sequences=msa_max_sequences)
            msa_metadata[cid]["loaded_depth"] = int(msa.depth)
            protein_kwargs["msa"] = msa
        protein_inputs.append(protein_input_cls(**protein_kwargs))
    spi = structure_input_cls(sequences=protein_inputs)


    with _MODEL_INFERENCE_LOCK:
        with torch.inference_mode():
            result = input_builder_cls().fold(
                model,
                spi,
                num_loops=int(num_loops),
                num_sampling_steps=int(num_sampling_steps),
                num_diffusion_samples=int(num_diffusion_samples),
                seed=int(seed),
            )

    out_dir = _tmp_root() / pred_name
    return _result_to_confidence(
        result=result,
        chains=chains,
        metric=metric,
        mode="local",
        model_name=model_name,
        out_dir=out_dir,
        pred_name=pred_name,
        write_cif=write_cif,
        msa_metadata=msa_metadata,
    )


def _run_biohub_esmfold2(
    pred_name: str,
    chains: List[Tuple[str, str]],
    metric: str,
    seed: int,
    model_name: str,
    num_loops: int,
    num_sampling_steps: int,
    num_diffusion_samples: int,
    write_cif: bool,
    msa_mode: Optional[str] = None,
    msa_paths: Optional[Dict[str, str]] = None,
    msa_cache_dir: Optional[str] = None,
    msa_max_sequences: Optional[int] = None,
) -> Dict[str, Any]:
    token = os.environ.get("ASTEVOLVE_ESMFOLD2_TOKEN") or os.environ.get("BIOHUB_API_TOKEN")
    if not token:
        raise ESMFold2TokenRequired(
            "Biohub ESMFold2 mode needs ASTEVOLVE_ESMFOLD2_TOKEN or BIOHUB_API_TOKEN."
        )

    try:
        from esm.sdk import input_builder
        from esm.sdk.api import FoldingConfig
        from esm.sdk.forge import SequenceStructureForgeInferenceClient
    except Exception as exc:
        raise RuntimeError(
            "Biohub ESMFold2 API dependencies are not available. Install the Biohub esm package "
            "before running API inference."
        ) from exc

    url = os.environ.get("ASTEVOLVE_ESMFOLD2_URL", DEFAULT_API_URL)
    client = SequenceStructureForgeInferenceClient(model=model_name, url=url, token=token)
    configured_msa_paths = bool(str(os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_PATHS") or "").strip())
    msa_requested = (
        bool(msa_paths)
        or configured_msa_paths
        or str(msa_mode or "off").lower() in {"remote", "file", "local"}
    )
    if msa_requested and "fast" in model_name.lower():
        raise ValueError("Biohub ESMFold2-Fast does not support MSA input; select esmfold2-2026-05")

    resolved_msa: Dict[str, str] = {}
    msa_metadata: Dict[str, Dict[str, Any]] = {}
    if msa_requested:
        from .remote_msa import resolve_chain_msa_paths

        resolved_msa, msa_metadata = resolve_chain_msa_paths(
            chains,
            msa_mode=msa_mode,
            msa_paths=msa_paths,
            cache_dir=msa_cache_dir,
        )
    if resolved_msa:
        from esm.utils.msa import MSA

    sequences = []
    for cid, seq in chains:
        protein_cls = getattr(input_builder, "ProteinInput", None)
        if protein_cls is not None:
            protein_kwargs: Dict[str, Any] = {"id": cid, "sequence": seq}
            if cid in resolved_msa:
                msa = MSA.from_a3m(resolved_msa[cid], max_sequences=msa_max_sequences)
                msa_metadata[cid]["loaded_depth"] = int(msa.depth)
                protein_kwargs["msa"] = msa
            sequences.append(protein_cls(**protein_kwargs))
        else:
            if cid in resolved_msa:
                raise RuntimeError("Installed Biohub API client cannot attach an MSA to ProteinInput")
            sequences.append(seq)

    spi = input_builder.StructurePredictionInput(sequences=sequences)
    try:
        config = FoldingConfig(
            num_loops=int(num_loops),
            num_sampling_steps=int(num_sampling_steps),
            num_diffusion_samples=int(num_diffusion_samples),
            seed=int(seed),
        )
    except TypeError:
        config = FoldingConfig(
            num_loops=int(num_loops),
            num_sampling_steps=int(num_sampling_steps),
            seed=int(seed),
        )
    result = client.fold_all_atom(spi, config=config)

    out_dir = _tmp_root() / pred_name
    return _result_to_confidence(
        result=result,
        chains=chains,
        metric=metric,
        mode="biohub",
        model_name=model_name,
        out_dir=out_dir,
        pred_name=pred_name,
        write_cif=write_cif,
        msa_metadata=msa_metadata,
    )


def _run_esmfold2_in_conda_env(
    conda_env: str,
    pred_name: str,
    chains: List[Tuple[str, str]],
    metric: str,
    seed: int,
    model_name: str,
    mode: str,
    device: Optional[str],
    num_loops: int,
    num_sampling_steps: int,
    num_diffusion_samples: int,
    write_cif: bool,
    msa_mode: Optional[str] = None,
    msa_paths: Optional[Dict[str, str]] = None,
    msa_cache_dir: Optional[str] = None,
    msa_max_sequences: Optional[int] = None,
) -> Dict[str, Any]:
    run_dir = _tmp_root() / "subprocess"
    run_dir.mkdir(parents=True, exist_ok=True)
    key = abs(
        hash(
            (
                pred_name,
                tuple(chains),
                metric,
                seed,
                model_name,
                mode,
                msa_mode,
                json.dumps(msa_paths or {}, sort_keys=True),
                msa_max_sequences,
            )
        )
    )
    request_path = run_dir / f"esmfold2_request_{key}.json"
    response_path = run_dir / f"esmfold2_response_{key}.json"

    request = {
        "pred_name": pred_name,
        "chains": chains,
        "metric": metric,
        "seed": int(seed),
        "model_name": model_name,
        "mode": mode,
        "device": device,
        "num_loops": int(num_loops),
        "num_sampling_steps": int(num_sampling_steps),
        "num_diffusion_samples": int(num_diffusion_samples),
        "write_cif": bool(write_cif),
        "msa_mode": msa_mode,
        "msa_paths": msa_paths,
        "msa_cache_dir": msa_cache_dir,
        "msa_max_sequences": msa_max_sequences,
    }
    request_path.write_text(json.dumps(request), encoding="utf-8")

    env = os.environ.copy()
    env["ASTEVOLVE_ESMFOLD2_WORKER"] = "1"
    env.pop("ASTEVOLVE_ESMFOLD2_CONDA_ENV", None)
    project_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        _find_conda_exe(),
        "run",
        "-n",
        str(conda_env),
        "python",
        str(Path(__file__).resolve()),
        "--esmfold2-worker",
        str(request_path),
        str(response_path),
    ]
    timeout = int(os.environ.get("ASTEVOLVE_ESMFOLD2_TIMEOUT", "7200"))
    completed = subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(
            part[-3000:]
            for part in (completed.stdout, completed.stderr)
            if part
        )
        raise RuntimeError(f"ESMFold2 conda worker failed in env '{conda_env}'.\n{tail}")
    if not response_path.exists():
        raise RuntimeError(f"ESMFold2 conda worker did not create response: {response_path}")

    response = json.loads(response_path.read_text(encoding="utf-8"))
    if response.get("error"):
        raise RuntimeError(
            f"ESMFold2 conda worker error in env '{conda_env}': "
            f"{response.get('error_type')}: {response.get('error')}"
        )
    return response["result"]


def run_esmfold2_confidence_multichain(
    pred_name: Optional[str] = None,
    chains: Optional[List[Tuple[str, str]]] = None,
    metric: str = "plddt",
    device: Optional[str] = None,
    seed: int = 0,
    model_name: Optional[str] = None,
    mode: Optional[str] = None,
    conda_env: Optional[str] = None,
    num_loops: int = 3,
    num_sampling_steps: int = 32,
    num_diffusion_samples: int = 1,
    write_cif: bool = True,
    msa_mode: Optional[str] = None,
    msa_paths: Optional[Dict[str, str]] = None,
    msa_cache_dir: Optional[str] = None,
    msa_max_sequences: Optional[int] = None,
    **_: Any,
) -> Dict[str, Any]:


    pred_name, chains = _normalise_inputs(pred_name, chains)
    if not chains:
        return {"provider": "esmfold2", "metrics": {}, "residue_plddt": {}, "out_dir": None}

    selected_mode = str(mode or os.environ.get("ASTEVOLVE_ESMFOLD2_MODE", "local")).strip().lower()
    if selected_mode in {"api", "platform", "remote"}:
        selected_mode = "biohub"

    requested_model = str(model_name or "").strip()
    if "protenix" in requested_model.lower():
        requested_model = ""
    selected_model = str(
        requested_model
        or os.environ.get("ASTEVOLVE_ESMFOLD2_MODEL")
        or (DEFAULT_API_MODEL if selected_mode == "biohub" else DEFAULT_LOCAL_MODEL)
    )
    selected_msa_mode = str(
        msa_mode or os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_MODE", "off")
    ).strip().lower()
    selected_msa_cache_dir = msa_cache_dir or os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_CACHE")
    selected_msa_max_sequences = _msa_max_sequences(msa_max_sequences)
    msa_signature = _msa_signature(
        selected_msa_mode, msa_paths, selected_msa_cache_dir, selected_msa_max_sequences
    )
    key = _cache_key(
        pred_name,
        chains,
        selected_mode,
        selected_model,
        metric,
        int(seed),
        int(num_loops),
        int(num_sampling_steps),
        int(num_diffusion_samples),
        msa_signature,
    )
    if key in _CONF_CACHE:
        return _CONF_CACHE[key]

    selected_conda_env = str(conda_env or os.environ.get("ASTEVOLVE_ESMFOLD2_CONDA_ENV", "")).strip()
    if selected_conda_env and os.environ.get("ASTEVOLVE_ESMFOLD2_WORKER") != "1":
        out = _run_esmfold2_in_conda_env(
            conda_env=selected_conda_env,
            pred_name=pred_name,
            chains=chains,
            metric=metric,
            seed=int(seed),
            model_name=selected_model,
            mode=selected_mode,
            device=device,
            num_loops=int(num_loops),
            num_sampling_steps=int(num_sampling_steps),
            num_diffusion_samples=int(num_diffusion_samples),
            write_cif=write_cif,
            msa_mode=selected_msa_mode,
            msa_paths=msa_paths,
            msa_cache_dir=selected_msa_cache_dir,
            msa_max_sequences=selected_msa_max_sequences,
        )
        _CONF_CACHE[key] = out
        return out

    if selected_mode == "biohub":
        out = _run_biohub_esmfold2(
            pred_name=pred_name,
            chains=chains,
            metric=metric,
            seed=int(seed),
            model_name=selected_model,
            num_loops=int(num_loops),
            num_sampling_steps=int(num_sampling_steps),
            num_diffusion_samples=int(num_diffusion_samples),
            write_cif=write_cif,
            msa_mode=selected_msa_mode,
            msa_paths=msa_paths,
            msa_cache_dir=selected_msa_cache_dir,
            msa_max_sequences=selected_msa_max_sequences,
        )
    elif selected_mode == "local":
        out = _run_local_esmfold2(
            pred_name=pred_name,
            chains=chains,
            metric=metric,
            seed=int(seed),
            model_name=selected_model,
            device=device,
            num_loops=int(num_loops),
            num_sampling_steps=int(num_sampling_steps),
            num_diffusion_samples=int(num_diffusion_samples),
            write_cif=write_cif,
            msa_mode=selected_msa_mode,
            msa_paths=msa_paths,
            msa_cache_dir=selected_msa_cache_dir,
            msa_max_sequences=selected_msa_max_sequences,
        )
    else:
        raise ValueError(f"Unknown ESMFold2 mode: {selected_mode}")

    _CONF_CACHE[key] = out
    return out


def run_esmfold2_plddt_multichain(
    pred_name: Optional[str] = None,
    chains: Optional[List[Tuple[str, str]]] = None,
    metric: str = "plddt",
    **kwargs: Any,
) -> float:


    pred_name, chains = _normalise_inputs(pred_name, chains)
    requested_model = str(kwargs.get("model_name") or "").strip()
    if "protenix" in requested_model.lower():
        requested_model = ""
    key = _cache_key(
        pred_name,
        chains,
        str(kwargs.get("mode") or os.environ.get("ASTEVOLVE_ESMFOLD2_MODE", "local")),
        str(requested_model or os.environ.get("ASTEVOLVE_ESMFOLD2_MODEL") or DEFAULT_LOCAL_MODEL),
        metric,
        int(kwargs.get("seed", 0)),
        int(kwargs.get("num_loops", 3)),
        int(kwargs.get("num_sampling_steps", 32)),
        int(kwargs.get("num_diffusion_samples", 1)),
        _msa_signature(
            kwargs.get("msa_mode"),
            kwargs.get("msa_paths"),
            kwargs.get("msa_cache_dir"),
            kwargs.get("msa_max_sequences"),
        ),
    )
    if key in _SCALAR_CACHE:
        return _SCALAR_CACHE[key]

    confidence = run_esmfold2_confidence_multichain(
        pred_name=pred_name,
        chains=chains,
        metric=metric,
        **kwargs,
    )
    value = float(confidence.get("metrics", {}).get(metric, 0.0))
    _SCALAR_CACHE[key] = value
    return value


def _worker_main(request_path: str, response_path: str) -> int:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    try:
        result = run_esmfold2_confidence_multichain(**request)
        payload = {"result": result}
    except Exception as exc:
        payload = {
            "error": str(exc),
            "error_type": exc.__class__.__name__,
        }
    Path(response_path).write_text(json.dumps(payload), encoding="utf-8")
    return 0


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--esmfold2-worker", nargs=2, metavar=("REQUEST_JSON", "RESPONSE_JSON"))
    args = parser.parse_args()
    if args.esmfold2_worker:
        return _worker_main(args.esmfold2_worker[0], args.esmfold2_worker[1])
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


def describe_esmfold2_setup() -> Dict[str, Any]:


    mode = str(os.environ.get("ASTEVOLVE_ESMFOLD2_MODE", "local")).strip().lower()
    model = os.environ.get("ASTEVOLVE_ESMFOLD2_MODEL") or os.environ.get("ASTEVOLVE_STRUCTURE_MODEL_NAME") or (
        DEFAULT_API_MODEL if mode in {"api", "platform", "remote", "biohub"} else DEFAULT_LOCAL_MODEL
    )
    return {
        "provider": "esmfold2",
        "mode": mode,
        "model_name": model,
        "conda_env": os.environ.get("ASTEVOLVE_ESMFOLD2_CONDA_ENV"),
        "allow_download": _as_bool_env("ASTEVOLVE_ESMFOLD2_ALLOW_DOWNLOAD", False),
        "has_biohub_token": bool(os.environ.get("ASTEVOLVE_ESMFOLD2_TOKEN") or os.environ.get("BIOHUB_API_TOKEN")),
        "tmp_root": str(_tmp_root()),
        "msa_mode": os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_MODE", "off"),
        "msa_url": os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_URL", "https://api.colabfold.com"),
        "msa_cache": os.environ.get("ASTEVOLVE_ESMFOLD2_MSA_CACHE"),
    }
