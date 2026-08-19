

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from typing import Iterable, List, Optional

_AUTO_VALUES = {"", "auto", "default", "current", "detect"}
_FALLBACK_ENVS = ("ast", "pytorch")


def _clean_env_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in _AUTO_VALUES:
        return None
    return text


def _dedupe(values: Iterable[Optional[str]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        cleaned = _clean_env_name(value)
        if cleaned is None or cleaned in seen:
            continue
        out.append(cleaned)
        seen.add(cleaned)
    return out


@lru_cache(maxsize=1)
def available_conda_envs() -> tuple[str, ...]:


    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        return tuple()
    try:
        result = subprocess.run(
            [conda, "env", "list"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception:
        return tuple()
    if result.returncode != 0:
        return tuple()

    names: List[str] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "*" and len(parts) > 1:
            name = parts[1]
        else:
            name = parts[0].lstrip("*")
        if name and "/" not in name and "\\" not in name and name not in names:
            names.append(name)
    return tuple(names)


def candidate_conda_envs(preferred: Optional[str] = None) -> List[str]:


    active = _clean_env_name(os.environ.get("CONDA_DEFAULT_ENV"))
    active_non_base = active if active and active != "base" else None
    active_base = active if active == "base" else None
    return _dedupe(
        [
            preferred,
            os.environ.get("ASTEVOLVE_PROTENIX_CONDA_ENV"),
            os.environ.get("ASTEVOLVE_CONDA_ENV"),
            active_non_base,
            *_FALLBACK_ENVS,
            active_base,
        ]
    )


def resolve_conda_env(preferred: Optional[str] = None, *, default: str = "ast") -> str:


    explicit = _clean_env_name(preferred)
    if explicit is not None:
        return explicit

    candidates = candidate_conda_envs(None)
    available = set(available_conda_envs())
    if available:
        for candidate in candidates:
            if candidate in available:
                return candidate
    if candidates:
        return candidates[0]
    return default


def resolve_protenix_conda_env(preferred: Optional[str] = None) -> str:


    return resolve_conda_env(preferred, default="ast")


def resolve_alphafold3_conda_env(preferred: Optional[str] = None) -> str:


    explicit = _clean_env_name(preferred) or _clean_env_name(
        os.environ.get("ASTEVOLVE_AF3_CONDA_ENV")
    )
    if explicit is not None:
        return explicit


    return "alphafold3"
