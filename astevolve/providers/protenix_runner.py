

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple
import uuid

from astevolve.runtime.conda import resolve_protenix_conda_env
from astevolve.runtime.paths import model_path, tmp_root

from .protenix_input import (
    write_protenix_complex_batch_input_json,
    write_protenix_complex_input_json,
)


_DEFAULT_PROTENIX_ROOT = Path(
    os.environ.get("ASTEVOLVE_PROTENIX_ROOT", str(model_path("protenix")))
)


def _tmp_root() -> Path:
    root = os.environ.get("ASTEVOLVE_PROTENIX_TMP")
    if root:
        path = Path(root)
    else:
        path = tmp_root("protenix")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _protenix_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PROTENIX_ROOT_DIR", str(_DEFAULT_PROTENIX_ROOT))
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def _find_conda_exe() -> str:
    candidates = [
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
    ]

    prefix = Path(sys.prefix)
    roots = [prefix]
    if prefix.parent.name == "envs":
        roots.append(prefix.parent.parent)
    for root in roots:
        candidates.extend([
            str(root / "Scripts" / "conda.exe"),
            str(root / "bin" / "conda"),
        ])

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return "conda"


def _protenix_num_workers() -> int:
    try:
        return max(0, int(os.environ.get("ASTEVOLVE_PROTENIX_NUM_WORKERS", "1")))
    except (TypeError, ValueError):
        return 1


def _protenix_need_atom_confidence() -> bool:
    return str(os.environ.get("ASTEVOLVE_PROTENIX_NEED_ATOM_CONFIDENCE", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _protenix_command(
    *,
    input_json_path: Path,
    out_dir: Path,
    seed: int,
    model_name: str,
    conda_env: Optional[str],
    use_msa: Optional[bool],
    cycle: Optional[int],
    step: Optional[int],
    sample: Optional[int],
    use_default_params: Optional[bool],
) -> List[str]:


    resolved_conda_env = resolve_protenix_conda_env(conda_env)
    cmd = [
        _find_conda_exe(),
        "run",
        "-n",
        resolved_conda_env,
        "python",
        str(Path(__file__).with_name("protenix_worker.py")),
        "--input",
        str(input_json_path),
        "--out-dir",
        str(out_dir),
        "--model-name",
        model_name,
        "--seeds",
        str(seed),
        "--num-workers",
        str(_protenix_num_workers()),
        "--need-atom-confidence",
        "true" if _protenix_need_atom_confidence() else "false",
    ]
    if use_msa is not None:
        cmd.extend(["--use-msa", "true" if use_msa else "false"])
    if cycle is not None:
        cmd.extend(["--cycle", str(int(cycle))])
    if step is not None:
        cmd.extend(["--step", str(int(step))])
    if sample is not None:
        cmd.extend(["--sample", str(int(sample))])
    if use_default_params is not None:
        cmd.extend(["--use-default-params", "true" if use_default_params else "false"])
    return cmd


def _run_protenix(
    pred_name: str,
    chains: List[Tuple[str, str]],
    seed: int,
    model_name: str,
    conda_env: Optional[str],
    timeout: Optional[int] = None,
    use_msa: Optional[bool] = None,
    cycle: Optional[int] = None,
    step: Optional[int] = None,
    sample: Optional[int] = None,
    use_default_params: Optional[bool] = None,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    entities = [
        {"type": "protein", "id": cid, "sequence": seq, "count": 1}
        for cid, seq in chains
    ]
    return _run_protenix_complex(
        pred_name=pred_name,
        entities=entities,
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


def _run_protenix_complex(
    pred_name: str,
    entities: List[Dict[str, Any]],
    seed: int,
    model_name: str,
    conda_env: Optional[str],
    constraint: Optional[Dict[str, Any]] = None,
    covalent_bonds: Optional[List[Dict[str, Any]]] = None,
    timeout: Optional[int] = None,
    use_msa: Optional[bool] = None,
    cycle: Optional[int] = None,
    step: Optional[int] = None,
    sample: Optional[int] = None,
    use_default_params: Optional[bool] = None,
) -> Tuple[Optional[Path], Dict[str, Any]]:


    root = _tmp_root()
    run_dir = root / pred_name
    input_json_path = run_dir / "input.json"
    out_dir = run_dir / "output"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview = write_protenix_complex_input_json(
        input_json_path,
        pred_name,
        entities,
        constraint=constraint,
        covalent_bonds=covalent_bonds,
    )
    preview["status_json"] = str(run_dir / "protenix_status.json")
    preview["protenix_output_dir"] = str(out_dir)

    cmd = _protenix_command(
        input_json_path=input_json_path,
        out_dir=out_dir,
        seed=seed,
        model_name=model_name,
        conda_env=conda_env,
        use_msa=use_msa,
        cycle=cycle,
        step=step,
        sample=sample,
        use_default_params=use_default_params,
    )

    run_timeout = int(timeout or os.environ.get("ASTEVOLVE_PROTENIX_TIMEOUT", "600"))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=run_timeout,
            env=_protenix_env(),
            cwd=str(run_dir),
        )
    except subprocess.TimeoutExpired as exc:
        print(f"[protenix] prediction timed out ({run_timeout}s)")
        stdout = exc.stdout.decode(errors="ignore") if isinstance(exc.stdout, bytes) else exc.stdout
        stderr = exc.stderr.decode(errors="ignore") if isinstance(exc.stderr, bytes) else exc.stderr
        _write_run_status(
            run_dir,
            status="timeout",
            cmd=cmd,
            out_dir=out_dir,
            timeout=run_timeout,
            stdout=stdout,
            stderr=stderr,
            error="timeout",
        )
        if stdout:
            print(f"[protenix] stdout before timeout: {stdout[-1000:]}")
        if stderr:
            print(f"[protenix] stderr before timeout: {stderr[-1000:]}")
        if _prediction_available(out_dir):
            print("[protenix] salvaging completed prediction files after timeout")
            return out_dir, preview
        return None, preview
    except FileNotFoundError:
        print("[protenix] conda or protenix not found in PATH")
        _write_run_status(
            run_dir,
            status="command_not_found",
            cmd=cmd,
            out_dir=out_dir,
            timeout=run_timeout,
            error="conda or protenix not found in PATH",
        )
        return None, preview

    if result.returncode != 0:
        print(f"[protenix] failed (rc={result.returncode})")
        _write_run_status(
            run_dir,
            status="failed",
            cmd=cmd,
            out_dir=out_dir,
            timeout=run_timeout,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        if result.stdout:
            print(f"[protenix] stdout: {_head_tail_text(result.stdout)}")
        if result.stderr:
            print(f"[protenix] stderr: {_head_tail_text(result.stderr)}")
        if _prediction_available(out_dir):
            print("[protenix] salvaging prediction files despite nonzero return code")
            return out_dir, preview
        return None, preview

    if not _prediction_available(out_dir):
        _write_run_status(
            run_dir,
            status="missing_prediction",
            cmd=cmd,
            out_dir=out_dir,
            timeout=run_timeout,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        _report_missing_prediction(out_dir, result)
        return None, preview

    _write_run_status(
        run_dir,
        status="ok",
        cmd=cmd,
        out_dir=out_dir,
        timeout=run_timeout,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    return out_dir, preview


def _run_protenix_complex_batch(
    jobs: List[Dict[str, Any]],
    seed: int,
    model_name: str,
    conda_env: Optional[str],
    timeout: Optional[int] = None,
    use_msa: Optional[bool] = None,
    cycle: Optional[int] = None,
    step: Optional[int] = None,
    sample: Optional[int] = None,
    use_default_params: Optional[bool] = None,
) -> List[Tuple[Optional[Path], Dict[str, Any]]]:


    if not jobs:
        return []

    root = _tmp_root()
    run_dir = root / "batches" / uuid.uuid4().hex
    input_json_path = run_dir / "input.json"
    out_dir = run_dir / "output"
    run_dir.mkdir(parents=True, exist_ok=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_preview = write_protenix_complex_batch_input_json(input_json_path, jobs)
    job_previews = list(input_preview["jobs"])
    status_path = run_dir / "protenix_status.json"

    cmd = _protenix_command(
        input_json_path=input_json_path,
        out_dir=out_dir,
        seed=seed,
        model_name=model_name,
        conda_env=conda_env,
        use_msa=use_msa,
        cycle=cycle,
        step=step,
        sample=sample,
        use_default_params=use_default_params,
    )
    run_timeout = int(timeout or os.environ.get("ASTEVOLVE_PROTENIX_TIMEOUT", "600"))

    process_status = "ok"
    returncode: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    error: Optional[str] = None
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=run_timeout,
            env=_protenix_env(),
            cwd=str(run_dir),
        )
        returncode = result.returncode
        stdout = result.stdout
        stderr = result.stderr
        if result.returncode != 0:
            process_status = "failed"
            print(f"[protenix] batch failed (rc={result.returncode})")
            if result.stdout:
                print(f"[protenix] stdout: {_head_tail_text(result.stdout)}")
            if result.stderr:
                print(f"[protenix] stderr: {_head_tail_text(result.stderr)}")
    except subprocess.TimeoutExpired as exc:
        process_status = "timeout"
        error = "timeout"
        stdout = exc.stdout.decode(errors="ignore") if isinstance(exc.stdout, bytes) else exc.stdout
        stderr = exc.stderr.decode(errors="ignore") if isinstance(exc.stderr, bytes) else exc.stderr
        print(f"[protenix] batch prediction timed out ({run_timeout}s)")
    except FileNotFoundError:
        process_status = "command_not_found"
        error = "conda or protenix not found in PATH"
        print("[protenix] conda or protenix not found in PATH")

    routed: List[Tuple[Optional[Path], Dict[str, Any]]] = []
    available_count = 0
    for index, raw_preview in enumerate(job_previews):
        preview = dict(raw_preview)
        pred_name = str(preview["pred_name"])
        job_out_dir = out_dir / pred_name
        available = _prediction_available(job_out_dir)
        if available:
            available_count += 1
            job_status = "ok" if process_status == "ok" else "salvaged"
            routed_out_dir: Optional[Path] = job_out_dir
        else:
            job_status = "missing_prediction" if process_status == "ok" else process_status
            routed_out_dir = None
        preview.update({
            "batch_index": index + 1,
            "input_json": str(input_json_path),
            "status_json": str(status_path),
            "batch_output_dir": str(out_dir),
            "protenix_output_dir": str(job_out_dir),
            "runner_status": job_status,
        })
        routed.append((routed_out_dir, preview))

    if process_status == "ok":
        if available_count == len(job_previews):
            batch_status = "ok"
        elif available_count:
            batch_status = "partial"
        else:
            batch_status = "missing_prediction"
    elif available_count:
        batch_status = f"{process_status}_partial"
    else:
        batch_status = process_status

    for _, preview in routed:
        preview["batch_status"] = batch_status
    _write_batch_run_status(
        run_dir,
        status=batch_status,
        process_status=process_status,
        cmd=cmd,
        out_dir=out_dir,
        timeout=run_timeout,
        job_previews=[preview for _, preview in routed],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        error=error,
    )
    if batch_status in {"partial", "missing_prediction"}:
        print(
            "[protenix] batch completed with "
            f"{available_count}/{len(job_previews)} routed predictions"
        )
    return routed


def _first_json(out_dir: Path, pattern: str) -> Optional[Path]:
    files = sorted(out_dir.rglob(pattern))
    return files[0] if files else None


def _first_cif(out_dir: Path) -> Optional[Path]:
    files = sorted(out_dir.rglob("*.cif"))
    return files[0] if files else None


def _prediction_available(out_dir: Path) -> bool:
    return bool(_first_json(out_dir, "*summary_confidence*.json") or _first_cif(out_dir))


def _write_run_status(
    run_dir: Path,
    *,
    status: str,
    cmd: List[str],
    out_dir: Path,
    timeout: int,
    returncode: Optional[int] = None,
    stdout: Optional[str] = None,
    stderr: Optional[str] = None,
    error: Optional[str] = None,
) -> None:


    summary_json = _first_json(out_dir, "*summary_confidence*.json")
    cif_path = _first_cif(out_dir)
    payload = {
        "status": status,
        "returncode": returncode,
        "timeout": int(timeout),
        "command": [str(part) for part in cmd],
        "output_dir": str(out_dir),
        "summary_json": str(summary_json) if summary_json else None,
        "cif_path": str(cif_path) if cif_path else None,
        "prediction_available": bool(summary_json or cif_path),
        "stdout_tail": _tail_text(stdout),
        "stderr_tail": _tail_text(stderr),
    }
    if error:
        payload["error"] = str(error)
    try:
        (run_dir / "protenix_status.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError:
        pass


def _write_batch_run_status(
    run_dir: Path,
    *,
    status: str,
    process_status: str,
    cmd: List[str],
    out_dir: Path,
    timeout: int,
    job_previews: List[Dict[str, Any]],
    returncode: Optional[int] = None,
    stdout: Optional[str] = None,
    stderr: Optional[str] = None,
    error: Optional[str] = None,
) -> None:


    jobs: List[Dict[str, Any]] = []
    for preview in job_previews:
        job_out_dir = Path(str(preview["protenix_output_dir"]))
        summary_json = _first_json(job_out_dir, "*summary_confidence*.json")
        cif_path = _first_cif(job_out_dir)
        jobs.append({
            "batch_index": int(preview["batch_index"]),
            "pred_name": str(preview["pred_name"]),
            "status": str(preview["runner_status"]),
            "output_dir": str(job_out_dir),
            "summary_json": str(summary_json) if summary_json else None,
            "cif_path": str(cif_path) if cif_path else None,
            "prediction_available": bool(summary_json or cif_path),
        })

    payload: Dict[str, Any] = {
        "status": status,
        "process_status": process_status,
        "returncode": returncode,
        "timeout": int(timeout),
        "command": [str(part) for part in cmd],
        "input_json": str(run_dir / "input.json"),
        "output_dir": str(out_dir),
        "job_count": len(jobs),
        "successful_job_count": sum(
            1 for job in jobs if job["prediction_available"]
        ),
        "jobs": jobs,
        "stdout_tail": _tail_text(stdout),
        "stderr_tail": _tail_text(stderr),
    }
    if error:
        payload["error"] = str(error)
    try:
        (run_dir / "protenix_status.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError:
        pass


def _tail_text(value: Optional[str], limit: int = 12000) -> str:
    text = str(value or "")
    return text[-limit:] if len(text) > limit else text


def _head_tail_text(value: Optional[str], limit: int = 12000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    half = max(1000, limit // 2)
    return f"{text[:half]}\n... <trimmed {len(text) - (2 * half)} chars> ...\n{text[-half:]}"


def _read_error_files(out_dir: Path, limit: int = 12000) -> str:
    err_dir = out_dir / "ERR"
    if not err_dir.exists():
        return ""

    chunks: List[str] = []
    for path in sorted(err_dir.rglob("*.txt"))[:5]:
        try:
            chunks.append(f"{path}:\n{_tail_text(path.read_text(encoding='utf-8', errors='ignore'), limit)}")
        except OSError:
            continue
    return "\n\n".join(chunks)


def _report_missing_prediction(out_dir: Path, result: subprocess.CompletedProcess[str]) -> None:
    print("[protenix] finished without CIF or summary_confidence output")
    errors = _read_error_files(out_dir)
    if errors:
        print(f"[protenix] ERR files:\n{errors}")
    if result.stdout:
        print(f"[protenix] stdout tail: {_tail_text(result.stdout)}")
    if result.stderr:
        print(f"[protenix] stderr tail: {_tail_text(result.stderr)}")
