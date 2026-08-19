

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Optional


def project_root() -> Path:


    raw = os.environ.get("ASTEVOLVE_PROJECT_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def runtime_root() -> Path:


    raw = os.environ.get("ASTEVOLVE_RUNTIME_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return (project_root().parent / "DASTevolve_runtime").resolve()


def _root_from_env(env_name: str, default_name: str) -> Path:
    raw = os.environ.get(env_name)
    if raw:
        return Path(raw).expanduser().resolve()
    return runtime_root() / default_name


def data_root() -> Path:


    return _root_from_env("ASTEVOLVE_DATA_ROOT", "datasets")


def artifact_root() -> Path:


    raw = os.environ.get("ASTEVOLVE_ARTIFACT_ROOT") or os.environ.get(
        "ASTEVOLVE_RUN_ROOT"
    )
    if raw:
        return Path(raw).expanduser().resolve()
    return runtime_root() / "runs"


def transient_artifact_dir() -> str:


    return (
        os.environ.get("ASTEVOLVE_TRANSIENT_ARTIFACT_DIR", "transient").strip()
        or "transient"
    )


def _looks_like_uuid(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{12}",
            str(value).strip(),
        )
    )


def model_root() -> Path:


    return _root_from_env("ASTEVOLVE_MODEL_ROOT", "models")


def tmp_root(name: str = "astevolve") -> Path:


    raw = os.environ.get("ASTEVOLVE_TMP_ROOT")
    root = (
        Path(raw).expanduser().resolve()
        if raw
        else Path(tempfile.gettempdir()).resolve()
    )
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(
    value: Optional[str | Path],
    *,
    base: Optional[Path] = None,
    fallback: Optional[Path] = None,
) -> Path:


    if value is None or str(value).strip() == "":
        if fallback is None:
            raise ValueError("resolve_path needs a value or fallback")
        return fallback.expanduser().resolve()
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((base or project_root()) / path).resolve()


def data_path(*parts: str) -> Path:


    return data_root().joinpath(*parts)


def artifact_path(*parts: str) -> Path:


    if len(parts) == 1 and _looks_like_uuid(str(parts[0])):
        raw = os.environ.get("ASTEVOLVE_RUN_ROOT")
        base = Path(raw).expanduser().resolve() if raw else artifact_root()
        return base.joinpath(transient_artifact_dir(), str(parts[0]))
    return artifact_root().joinpath(*parts)


def model_path(*parts: str) -> Path:


    return model_root().joinpath(*parts)


__all__ = [
    "artifact_path",
    "artifact_root",
    "data_path",
    "data_root",
    "model_path",
    "model_root",
    "project_root",
    "resolve_path",
    "runtime_root",
    "tmp_root",
    "transient_artifact_dir",
]
