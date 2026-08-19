

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, Iterable, Optional, Tuple

from astevolve.runtime.conda import resolve_alphafold3_conda_env
from astevolve.runtime.paths import model_path, tmp_root
from astevolve.runtime.tools import resolve_tool_directory


DEFAULT_TIMEOUT_SECONDS = 7200


_ENVIRONMENT_PROBE = r"""
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

spec = importlib.util.find_spec("alphafold3")
package_root = None
if spec is not None:
    locations = list(spec.submodule_search_locations or [])
    if locations:
        package_root = Path(locations[0])
    elif spec.origin:
        package_root = Path(spec.origin).parent

dependencies = {
    name: importlib.util.find_spec(name) is not None
    for name in ("jax", "jaxlib", "haiku", "tokamax", "rdkit")
}
cuda_packages = {}
for distribution in ("jax-cuda12-plugin", "jax-cuda12-pjrt"):
    try:
        cuda_packages[distribution] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        cuda_packages[distribution] = None
jaxlib_build = None
for metadata_path in (Path(sys.prefix) / "conda-meta").glob("jaxlib-*.json"):
    try:
        jaxlib_build = json.loads(metadata_path.read_text()).get("build")
    except (OSError, json.JSONDecodeError):
        continue
    if jaxlib_build:
        break
conda_gpu_jaxlib = bool(
    jaxlib_build
    and any(marker in str(jaxlib_build).lower() for marker in ("cuda", "gpu"))
)
cpp_available = bool(package_root and list(package_root.glob("cpp*.so")))
ccd_pickle = bool(
    package_root
    and (package_root / "constants" / "converters" / "ccd.pickle").is_file()
)
component_sets_pickle = bool(
    package_root
    and (
        package_root
        / "constants"
        / "converters"
        / "chemical_component_sets.pickle"
    ).is_file()
)
try:
    package_version = importlib.metadata.version("alphafold3")
except importlib.metadata.PackageNotFoundError:
    package_version = None

print(json.dumps({
    "python_version": ".".join(str(value) for value in sys.version_info[:3]),
    "python_312_or_newer": sys.version_info >= (3, 12),
    "package_available": package_root is not None,
    "package_root": str(package_root) if package_root else None,
    "package_version": package_version,
    "dependencies": dependencies,
    "cuda_packages": cuda_packages,
    "jaxlib_build": jaxlib_build,
    "conda_gpu_jaxlib": conda_gpu_jaxlib,
    "cpp_extension_available": cpp_available,
    "ccd_pickle_available": ccd_pickle,
    "component_sets_pickle_available": component_sets_pickle,
}))
"""


def alphafold3_source_dir() -> Optional[Path]:


    return resolve_tool_directory(
        env_name="ASTEVOLVE_AF3_ROOT",
        relative_name="alphafold3",
    )


def alphafold3_model_dir(value: Optional[str | Path] = None) -> Path:


    configured = value or os.environ.get("ASTEVOLVE_AF3_MODEL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return model_path("alphafold3")


def alphafold3_database_dirs(value: Any = None) -> tuple[Path, ...]:


    configured = (
        value if value is not None else os.environ.get("ASTEVOLVE_AF3_DB_DIR")
    )
    if configured is None or configured == "":
        return ()
    if isinstance(configured, (str, Path)):
        items: Iterable[Any] = str(configured).split(os.pathsep)
    else:
        items = configured
    return tuple(
        Path(str(item)).expanduser().resolve()
        for item in items
        if str(item).strip()
    )


def alphafold3_tmp_dir() -> Path:


    configured = os.environ.get("ASTEVOLVE_AF3_TMP")
    if configured:
        path = Path(configured).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return tmp_root("alphafold3")


def _conda_executable() -> str:
    executable = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not executable:
        raise FileNotFoundError(
            "Conda is required to launch the AlphaFold 3 worker"
        )
    return executable


def _model_file_exists(directory: Path) -> bool:
    patterns = ("*.bin.zst", "*.bin.zst.*", "*.bin")
    return (
        any(any(directory.glob(pattern)) for pattern in patterns)
        if directory.is_dir()
        else False
    )


def _write_status(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def alphafold3_source_revision(
    source_dir: Optional[Path] = None,
) -> Optional[str]:


    source = source_dir if source_dir is not None else alphafold3_source_dir()
    if source is None or not (source / ".git").exists() or not shutil.which("git"):
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(source),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and len(revision) == 40 else None


def probe_alphafold3_environment(
    conda_env: Optional[str] = None,
    *,
    timeout: int = 30,
) -> Dict[str, Any]:


    env_name = resolve_alphafold3_conda_env(conda_env)
    report: Dict[str, Any] = {
        "conda_env": env_name,
        "probe_ok": False,
        "environment_ready": False,
        "gpu_dependencies_ready": False,
    }
    try:
        command = [
            _conda_executable(),
            "run",
            "-n",
            env_name,
            "python",
            "-c",
            _ENVIRONMENT_PROBE,
        ]
        probe_env = os.environ.copy()
        probe_env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout)),
            check=False,
            env=probe_env,
        )
    except Exception as exc:
        report["probe_error"] = f"{type(exc).__name__}: {exc}"
        return report

    report["returncode"] = int(result.returncode)
    if result.returncode != 0:
        report["probe_error"] = (
            result.stderr or result.stdout or "Conda probe failed"
        )[-4000:]
        return report

    payload: Optional[Dict[str, Any]] = None
    for line in reversed(result.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        report["probe_error"] = "AF3 environment probe returned no JSON payload"
        return report

    report.update(payload)
    dependencies = payload.get("dependencies", {}) or {}
    cuda_packages = payload.get("cuda_packages", {}) or {}
    report["probe_ok"] = True
    report["environment_ready"] = bool(
        payload.get("python_312_or_newer")
        and payload.get("package_available")
        and all(bool(value) for value in dependencies.values())
        and payload.get("cpp_extension_available")
        and payload.get("ccd_pickle_available")
        and payload.get("component_sets_pickle_available")
    )
    report["gpu_dependencies_ready"] = bool(
        report["environment_ready"]
        and (
            payload.get("conda_gpu_jaxlib")
            or (
                cuda_packages
                and all(bool(value) for value in cuda_packages.values())
            )
        )
    )
    return report


def run_alphafold3_job(
    *,
    input_json: Path,
    job_key: str,
    model_dir: Optional[str | Path] = None,
    conda_env: Optional[str] = None,
    timeout: Optional[int] = None,
    run_data_pipeline: bool = False,
    db_dir: Any = None,
    num_recycles: int = 10,
    num_diffusion_samples: int = 1,
    flash_attention_implementation: Optional[str] = None,
    gpu_device: int = 0,
) -> Tuple[Optional[Path], Dict[str, Any]]:


    input_json = Path(input_json).expanduser().resolve()
    source_dir = alphafold3_source_dir()
    resolved_model_dir = alphafold3_model_dir(model_dir)
    database_dirs = alphafold3_database_dirs(db_dir)
    env_name = resolve_alphafold3_conda_env(conda_env)
    job_dir = alphafold3_tmp_dir() / job_key
    output_dir = job_dir / "output"
    cache_dir = alphafold3_tmp_dir() / "jax_compilation_cache"
    status_path = job_dir / "alphafold3_status.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    preview: Dict[str, Any] = {
        "provider": "alphafold3",
        "source_dir": str(source_dir) if source_dir is not None else None,
        "model_dir": str(resolved_model_dir),
        "database_dirs": [str(path) for path in database_dirs],
        "conda_env": env_name,
        "input_json": str(input_json),
        "status_json": str(status_path),
        "alphafold3_output_dir": str(output_dir),
        "run_data_pipeline": bool(run_data_pipeline),
    }

    error: Optional[str] = None
    if not input_json.is_file():
        error = f"AlphaFold 3 input JSON is missing: {input_json}"
    elif source_dir is None:
        error = (
            "AlphaFold 3 source is not configured; set ASTEVOLVE_AF3_ROOT "
            "or ASTEVOLVE_TOOL_ROOT"
        )
    elif not (source_dir / "run_alphafold.py").is_file():
        error = f"AlphaFold 3 source is missing: {source_dir}"
    elif not _model_file_exists(resolved_model_dir):
        error = f"AlphaFold 3 parameters are missing from: {resolved_model_dir}"
    elif run_data_pipeline and not database_dirs:
        error = "Full AlphaFold 3 mode requires ASTEVOLVE_AF3_DB_DIR or db_dir"
    elif run_data_pipeline and any(
        not path.is_dir() for path in database_dirs
    ):
        error = "One or more AlphaFold 3 database directories do not exist"

    if error:
        status = {**preview, "status": "configuration_error", "error": error}
        _write_status(status_path, status)
        preview.update(status)
        return None, preview

    attention = str(
        flash_attention_implementation
        or os.environ.get("ASTEVOLVE_AF3_FLASH_ATTENTION", "triton")
    ).strip().lower()
    if attention not in {"triton", "cudnn", "xla"}:
        error = "AF3 flash attention must be one of: triton, cudnn, xla"
        status = {**preview, "status": "configuration_error", "error": error}
        _write_status(status_path, status)
        preview.update(status)
        return None, preview

    try:
        conda_executable = _conda_executable()
    except FileNotFoundError as exc:
        status = {
            **preview,
            "status": "configuration_error",
            "error": str(exc),
        }
        _write_status(status_path, status)
        preview.update(status)
        return None, preview

    assert source_dir is not None
    command = [
        conda_executable,
        "run",
        "--no-capture-output",
        "-n",
        env_name,
        "python",
        str(source_dir / "run_alphafold.py"),
        f"--json_path={input_json}",
        f"--model_dir={resolved_model_dir}",
        f"--output_dir={output_dir}",
        f"--run_data_pipeline={'true' if run_data_pipeline else 'false'}",
        "--run_inference=true",
        f"--num_recycles={max(1, int(num_recycles))}",
        f"--num_diffusion_samples={max(1, int(num_diffusion_samples))}",
        f"--flash_attention_implementation={attention}",
        f"--gpu_device={max(0, int(gpu_device))}",
        f"--jax_compilation_cache_dir={cache_dir}",
        "--force_output_dir=true",
    ]
    for directory in database_dirs:
        command.append(f"--db_dir={directory}")

    process_env = os.environ.copy()
    process_env.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
    process_env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    process_env.setdefault("XLA_CLIENT_MEM_FRACTION", "0.95")
    started = time.time()
    _write_status(
        status_path,
        {
            **preview,
            "status": "running",
            "command": command,
            "started_at": started,
        },
    )
    effective_timeout = int(
        timeout
        or os.environ.get("ASTEVOLVE_AF3_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)
    )
    try:
        result = subprocess.run(
            command,
            cwd=str(source_dir),
            env=process_env,
            capture_output=True,
            text=True,
            timeout=max(1, effective_timeout),
            check=False,
        )
        status = {
            **preview,
            "status": "ok" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "seconds": round(time.time() - started, 3),
            "command": command,
            "stdout_tail": result.stdout[-8000:],
            "stderr_tail": result.stderr[-8000:],
        }
    except subprocess.TimeoutExpired as exc:
        status = {
            **preview,
            "status": "timeout",
            "seconds": round(time.time() - started, 3),
            "command": command,
            "error": f"AlphaFold 3 exceeded {effective_timeout} seconds",
            "stdout_tail": str(exc.stdout or "")[-8000:],
            "stderr_tail": str(exc.stderr or "")[-8000:],
        }
    except Exception as exc:
        status = {
            **preview,
            "status": "error",
            "seconds": round(time.time() - started, 3),
            "command": command,
            "error": f"{type(exc).__name__}: {exc}",
        }

    _write_status(status_path, status)
    preview.update(status)
    return (output_dir if status.get("status") == "ok" else None), preview


def describe_alphafold3_runtime() -> Dict[str, Any]:


    source_dir = alphafold3_source_dir()
    model_dir = alphafold3_model_dir()
    database_dirs = alphafold3_database_dirs()
    conda_available = bool(os.environ.get("CONDA_EXE") or shutil.which("conda"))
    environment = (
        probe_alphafold3_environment()
        if conda_available
        else {
            "conda_env": resolve_alphafold3_conda_env(),
            "probe_ok": False,
            "environment_ready": False,
            "gpu_dependencies_ready": False,
            "probe_error": "Conda executable was not found",
        }
    )
    source_available = bool(
        source_dir is not None
        and (source_dir / "run_alphafold.py").is_file()
    )
    return {
        "provider": "alphafold3",
        "source_dir": str(source_dir) if source_dir is not None else None,
        "source_available": source_available,
        "source_commit_actual": alphafold3_source_revision(source_dir),
        "source_configuration": (
            "configured"
            if source_dir is not None
            else "set ASTEVOLVE_AF3_ROOT or ASTEVOLVE_TOOL_ROOT"
        ),
        "model_dir": str(model_dir),
        "model_available": _model_file_exists(model_dir),
        "database_dirs": [str(path) for path in database_dirs],
        "full_pipeline_available": bool(database_dirs)
        and all(path.is_dir() for path in database_dirs),
        "conda_env": resolve_alphafold3_conda_env(),
        "conda_available": conda_available,
        "environment_ready": bool(environment.get("environment_ready")),
        "gpu_dependencies_ready": bool(
            environment.get("gpu_dependencies_ready")
        ),
        "environment": environment,
        "default_mode": "full" if database_dirs else "msa_free",
    }


__all__ = [
    "alphafold3_database_dirs",
    "alphafold3_model_dir",
    "alphafold3_source_dir",
    "alphafold3_source_revision",
    "alphafold3_tmp_dir",
    "describe_alphafold3_runtime",
    "probe_alphafold3_environment",
    "run_alphafold3_job",
]
