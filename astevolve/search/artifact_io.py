

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np


def _seqs_hash(seqs: Mapping[str, str]) -> str:


    payload = "|".join(f"{key}:{seqs[key]}" for key in sorted(seqs))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _jsonable(obj: Any) -> Any:


    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


class _ArtifactJSONEncoder(json.JSONEncoder):


    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return str(obj)


def _atomic_text_write(path: Path, writer: Any) -> None:


    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, data: Any) -> None:


    def write(handle: Any) -> None:
        try:
            json.dump(
                data,
                handle,
                ensure_ascii=False,
                indent=2,
                cls=_ArtifactJSONEncoder,
            )
        except TypeError as exc:


            if "keys must be str" not in str(exc):
                raise
            handle.seek(0)
            handle.truncate()
            json.dump(
                _jsonable(data),
                handle,
                ensure_ascii=False,
                indent=2,
                cls=_ArtifactJSONEncoder,
            )

    _atomic_text_write(
        path,
        write,
    )


def _write_yaml(path: Path, data: Any) -> None:


    try:
        import yaml

        _atomic_text_write(
            path,
            lambda handle: yaml.safe_dump(
                _jsonable(data),
                handle,
                sort_keys=False,
                allow_unicode=True,
            ),
        )
    except Exception:
        _write_json(path, data)


__all__ = ["_jsonable", "_seqs_hash", "_write_json", "_write_yaml"]
