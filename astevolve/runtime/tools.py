

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, Optional, Tuple


def tool_root() -> Optional[Path]:


    raw = str(os.environ.get("ASTEVOLVE_TOOL_ROOT") or "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def resolve_tool_directory(
    *,
    env_name: str,
    relative_name: str,
) -> Optional[Path]:


    configured = str(os.environ.get(env_name) or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    root = tool_root()
    return (root / relative_name).resolve() if root is not None else None


def _command_path(value: object) -> Optional[str]:
    candidate = str(value or "").strip()
    if not candidate:
        return None
    discovered = shutil.which(candidate)
    if discovered:
        return discovered
    path = Path(candidate).expanduser()
    return str(path.resolve()) if path.is_file() else None


def resolve_tool_command(
    *,
    configured: Iterable[object] = (),
    env_names: Iterable[str] = (),
    directory_env_candidates: Iterable[Tuple[str, str]] = (),
    root_candidates: Iterable[str] = (),
    path_candidates: Iterable[str] = (),
) -> Optional[str]:


    for raw in configured:
        resolved = _command_path(raw)
        if resolved:
            return resolved

    for env_name in env_names:
        resolved = _command_path(os.environ.get(env_name))
        if resolved:
            return resolved

    for env_name, relative in directory_env_candidates:
        configured_root = str(os.environ.get(env_name) or "").strip()
        if not configured_root:
            continue
        resolved = _command_path(Path(configured_root).expanduser() / relative)
        if resolved:
            return resolved

    root = tool_root()
    if root is not None:
        for relative in root_candidates:
            resolved = _command_path(root / relative)
            if resolved:
                return resolved

    for candidate in path_candidates:
        resolved = _command_path(candidate)
        if resolved:
            return resolved
    return None


__all__ = ["resolve_tool_command", "resolve_tool_directory", "tool_root"]
