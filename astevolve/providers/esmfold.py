

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import tempfile
import uuid
import atexit
import string
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

_MODEL: Any = None
_MODEL_LOCK = Lock()
_INFERENCE_LOCK = Lock()
_CACHE: Dict[str, Dict[str, Any]] = {}
_WORKER: Optional[subprocess.Popen[str]] = None
_WORKER_LOG: Any = None
_WORKER_LOCK = Lock()


def _bagel_chain_id_map(
    chains: List[Tuple[str, str]],
) -> Dict[str, str]:


    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits
    unique = list(dict.fromkeys(str(chain_id) for chain_id, _ in chains))
    if len(unique) > len(alphabet):
        raise ValueError(
            f"classic ESMFold supports at most {len(alphabet)} chains per call"
        )
    return {chain_id: alphabet[index] for index, chain_id in enumerate(unique)}


def _bagel_root() -> Path:


    configured = os.environ.get("ASTEVOLVE_BAGEL_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    from astevolve.runtime.paths import runtime_root

    return (runtime_root() / "tools" / "bagel").resolve()


def _worker_python() -> Optional[Path]:
    configured = str(os.environ.get("ASTEVOLVE_ESMFOLD_PYTHON") or "").strip()
    candidate = (
        Path(configured).expanduser()
        if configured
        else _bagel_root() / ".venv-local" / "bin" / "python"
    )


    if candidate.is_file():
        return candidate.absolute()
    return None


def _stop_worker() -> None:
    global _WORKER, _WORKER_LOG
    worker = _WORKER
    _WORKER = None
    if worker is not None and worker.poll() is None:
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=10)
    if _WORKER_LOG is not None:
        _WORKER_LOG.close()
        _WORKER_LOG = None


atexit.register(_stop_worker)


def _start_worker() -> subprocess.Popen[str]:
    global _WORKER, _WORKER_LOG
    python = _worker_python()
    if python is None:
        raise RuntimeError(
            "classic ESMFold requires ASTEVOLVE_ESMFOLD_PYTHON or the "
            "BAGEL .venv-local environment"
        )
    if _WORKER is not None and _WORKER.poll() is None:
        return _WORKER
    _stop_worker()
    root = _bagel_root()
    project_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    python_paths = [str(project_root), str(root / "src")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "ASTEVOLVE_BAGEL_ROOT": str(root),
            "ASTEVOLVE_ESMFOLD_IN_WORKER": "1",
            "MODEL_DIR": str(root / "models"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    log_dir = Path(
        os.environ.get(
            "ASTEVOLVE_ESMFOLD_TMP",
            str(Path(tempfile.gettempdir()) / "astevolve_esmfold"),
        )
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    _WORKER_LOG = (log_dir / "classic_esmfold_worker.log").open(
        "a", encoding="utf-8"
    )
    _WORKER = subprocess.Popen(
        [str(python), "-m", "astevolve.providers.esmfold_worker"],
        cwd=str(root),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=_WORKER_LOG,
        text=True,
        bufsize=1,
    )
    return _WORKER


def _worker_request(payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    with _WORKER_LOCK:
        for attempt in range(2):
            worker = _start_worker()
            request_id = uuid.uuid4().hex
            request = {**payload, "request_id": request_id}
            assert worker.stdin is not None and worker.stdout is not None
            try:
                worker.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                worker.stdin.flush()
                ready, _, _ = select.select([worker.stdout], [], [], timeout)
                if not ready:
                    raise TimeoutError(
                        f"classic ESMFold worker timed out after {timeout}s"
                    )
                line = worker.stdout.readline()
                if not line:
                    raise RuntimeError(
                        f"classic ESMFold worker exited with code {worker.poll()}"
                    )
                response = json.loads(line)
                if response.get("request_id") != request_id:
                    raise RuntimeError("classic ESMFold worker response ID mismatch")
                if response.get("status") != "ok":
                    raise RuntimeError(
                        str(response.get("error") or "classic ESMFold worker failed")
                    )
                result = response.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError("classic ESMFold worker returned no result")
                return result
            except Exception:
                _stop_worker()
                if attempt:
                    raise
        raise RuntimeError("classic ESMFold worker retry exhausted")


def _load_model() -> Any:
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        root = _bagel_root()
        source = root / "src"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        os.environ.setdefault("MODEL_DIR", str(root / "models"))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from bagel.oracles.folding.esmfold import ESMFold
        _MODEL = ESMFold(
            use_modal=False,
            config={
                "output_pdb": False,
                "output_cif": False,
                "output_atomarray": True,
                "glycine_linker": "",
                "position_ids_skip": 512,
            },
        )
        return _MODEL


def _cache_key(chains: List[Tuple[str, str]], metric: str) -> str:
    return metric + "|" + "|".join(f"{chain}:{sequence}" for chain, sequence in chains)


def run_esmfold_confidence_multichain(
    pred_name: Optional[str] = None,
    chains: Optional[List[Tuple[str, str]]] = None,
    metric: str = "plddt",
    model_name: Optional[str] = None,
    **_kwargs: Any,
) -> Dict[str, Any]:
    normalized = [(str(chain), str(sequence)) for chain, sequence in (chains or [])]
    if not normalized:
        return {"provider": "esmfold", "metrics": {}, "residue_plddt": {}, "out_dir": None}
    key = _cache_key(normalized, metric)
    if key in _CACHE:
        return dict(_CACHE[key])

    if os.environ.get("ASTEVOLVE_ESMFOLD_IN_WORKER") != "1" and _worker_python():
        result = _worker_request(
            {
                "pred_name": pred_name,
                "chains": normalized,
                "metric": metric,
                "model_name": model_name,
            },
            timeout=int(
                _kwargs.get("timeout")
                or os.environ.get("ASTEVOLVE_ESMFOLD_TIMEOUT", "1800")
            ),
        )
        _CACHE[key] = dict(result)
        return result

    root = _bagel_root()
    source = root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from bagel import Chain, Residue

    bagel_chain_ids = _bagel_chain_id_map(normalized)
    bagel_chains = []
    for chain_id, sequence in normalized:
        residues = [
            Residue(
                name=residue,
                chain_ID=bagel_chain_ids[chain_id],
                index=index,
                mutable=False,
            )
            for index, residue in enumerate(sequence)
        ]
        bagel_chains.append(Chain(residues=residues))

    with _INFERENCE_LOCK:
        result = _load_model().fold(bagel_chains)

    import numpy as np

    local = np.asarray(result.local_plddt, dtype=float).reshape(-1)
    if local.size and float(np.nanmax(local)) <= 1.0 + 1e-6:
        local = local * 100.0
    residue_plddt: Dict[str, List[float]] = {}
    chain_plddt: Dict[str, float] = {}
    offset = 0
    for chain_id, sequence in normalized:
        values = [float(value) for value in local[offset : offset + len(sequence)]]
        residue_plddt[chain_id] = values
        chain_plddt[chain_id] = float(np.mean(values)) if values else 0.0
        offset += len(sequence)
    plddt = float(np.mean(local)) if local.size else 0.0
    ptm_values = np.asarray(result.ptm, dtype=float).reshape(-1)
    ptm = float(np.mean(ptm_values)) if ptm_values.size else 0.0

    safe_name = str(pred_name or "esmfold").replace("/", "_")
    out_dir = Path(os.environ.get("ASTEVOLVE_ESMFOLD_TMP", tempfile.gettempdir())) / "astevolve_esmfold" / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cif_path = out_dir / f"{safe_name}.cif"
    result.to_cif(cif_path)
    summary_path = out_dir / f"{safe_name}_summary.json"
    summary = {
        "provider": "esmfold",
        "model_name": str(model_name or "facebook/esmfold_v1"),
        "metrics": {"plddt": plddt, "ptm": ptm, "iptm": 0.0, metric: {"plddt": plddt, "ptm": ptm}.get(metric, plddt)},
        "chain_metrics": {"plddt": chain_plddt},
        "residue_plddt": residue_plddt,
        "inference_chain_id_map": bagel_chain_ids,
        "out_dir": str(out_dir),
        "cif_path": str(cif_path),
        "summary_json": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _CACHE[key] = dict(summary)
    return summary


def run_esmfold_plddt_multichain(**kwargs: Any) -> float:
    result = run_esmfold_confidence_multichain(**kwargs)
    return float((result.get("metrics") or {}).get("plddt", 0.0))


def clear_esmfold_model_cache() -> None:
    global _MODEL
    with _MODEL_LOCK:
        _MODEL = None
    _CACHE.clear()


def release_esmfold_resources() -> None:


    _stop_worker()
    clear_esmfold_model_cache()
