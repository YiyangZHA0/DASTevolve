

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping, Optional

from engine.memory_lifecycle import canonical_memory_content, capture_memory_snapshot
from engine.memory_policy import MemoryPolicyConfig, MemoryPolicyError


ADAPTIVE_MEMORY_ORIGIN_VERSION = "astevolve.adaptive_memory_origin.v1"
ADAPTIVE_MEMORY_CHECKPOINT_VERSION = "astevolve.adaptive_memory_checkpoint.v1"
RUNTIME_ADAPTIVE_MEMORY_NAME = "adaptive_inner_memory.yaml"
RUNTIME_ADAPTIVE_MEMORY_ORIGIN_NAME = "adaptive_inner_memory.origin.json"
CHECKPOINT_ADAPTIVE_MEMORY_NAME = "adaptive_inner_memory.yaml"
CHECKPOINT_ADAPTIVE_MEMORY_METADATA_NAME = (
    "adaptive_inner_memory.checkpoint.json"
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None


        directory_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


@contextmanager
def _origin_lock(output_dir: Path) -> Iterator[None]:
    lock_path = output_dir / ".adaptive_inner_memory.lock"
    output_dir.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _origin_document(source: Path) -> dict[str, Any]:
    snapshot = capture_memory_snapshot(source)
    if not snapshot.source_exists:
        raise MemoryPolicyError(
            "adaptive winner_commit requires an existing frozen case memory.yaml"
        )
    return {
        "schema_version": ADAPTIVE_MEMORY_ORIGIN_VERSION,
        "source_path": str(source),
        "source_content_hash": snapshot.content_hash,
        "source_raw_hash": snapshot.raw_hash,
        "source_byte_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "runtime_memory_name": RUNTIME_ADAPTIVE_MEMORY_NAME,
    }


def _load_origin(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MemoryPolicyError(
            f"adaptive memory origin is unreadable: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise MemoryPolicyError("adaptive memory origin must be a mapping")
    required = {
        "schema_version",
        "source_path",
        "source_content_hash",
        "source_raw_hash",
        "source_byte_sha256",
        "runtime_memory_name",
    }
    if set(value) != required:
        raise MemoryPolicyError("adaptive memory origin has an invalid closed schema")
    return dict(value)


def _require_checkpoint_policy(policy: MemoryPolicyConfig) -> bool:
    if not isinstance(policy, MemoryPolicyConfig):
        raise TypeError("policy must be MemoryPolicyConfig")
    return bool(policy.may_commit_adaptive_prior)


def _runtime_paths(
    source_path: str | Path | None,
    target_path: str | Path | None,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    if source_path is None or target_path is None:
        raise MemoryPolicyError(
            "adaptive checkpoint requires source and run-local target paths"
        )
    source = Path(source_path).expanduser().resolve()
    target = Path(target_path).expanduser().resolve()
    if not source.is_file():
        raise MemoryPolicyError(
            f"adaptive memory source does not exist: {source}"
        )
    if target.name != RUNTIME_ADAPTIVE_MEMORY_NAME:
        raise MemoryPolicyError(
            "adaptive checkpoint target is not the controller-owned run-local file"
        )
    origin_path = target.parent / RUNTIME_ADAPTIVE_MEMORY_ORIGIN_NAME
    expected_origin = _origin_document(source)
    return source, target, origin_path, expected_origin


def _validate_runtime_origin_locked(
    target: Path,
    origin_path: Path,
    expected_origin: Mapping[str, Any],
) -> None:
    if not target.is_file() or not origin_path.is_file():
        raise MemoryPolicyError(
            "adaptive memory resume state is incomplete: target/origin mismatch"
        )
    observed_origin = _load_origin(origin_path)
    if canonical_memory_content(observed_origin) != canonical_memory_content(
        expected_origin
    ):
        raise MemoryPolicyError(
            "adaptive memory origin no longer matches the frozen case input"
        )


def _checkpoint_document(
    *,
    iteration: int,
    expected_origin: Mapping[str, Any],
    runtime_snapshot: Any,
    runtime_bytes: bytes,
) -> dict[str, Any]:
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise MemoryPolicyError(
            "adaptive memory checkpoint iteration must be a non-negative integer"
        )
    return {
        "schema_version": ADAPTIVE_MEMORY_CHECKPOINT_VERSION,
        "iteration": iteration,
        "runtime_memory_name": CHECKPOINT_ADAPTIVE_MEMORY_NAME,
        "source_origin": dict(expected_origin),
        "runtime_snapshot": {
            "content_hash": runtime_snapshot.content_hash,
            "raw_hash": runtime_snapshot.raw_hash,
            "byte_sha256": hashlib.sha256(runtime_bytes).hexdigest(),
            "raw_size": len(runtime_bytes),
        },
    }


def _load_checkpoint_document(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MemoryPolicyError(
            f"adaptive memory checkpoint metadata is unreadable: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise MemoryPolicyError(
            "adaptive memory checkpoint metadata must be a mapping"
        )
    required = {
        "schema_version",
        "iteration",
        "runtime_memory_name",
        "source_origin",
        "runtime_snapshot",
    }
    if set(value) != required:
        raise MemoryPolicyError(
            "adaptive memory checkpoint metadata has an invalid closed schema"
        )
    if value.get("schema_version") != ADAPTIVE_MEMORY_CHECKPOINT_VERSION:
        raise MemoryPolicyError(
            "adaptive memory checkpoint metadata has an unsupported schema_version"
        )
    iteration = value.get("iteration")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise MemoryPolicyError(
            "adaptive memory checkpoint iteration must be a non-negative integer"
        )
    if value.get("runtime_memory_name") != CHECKPOINT_ADAPTIVE_MEMORY_NAME:
        raise MemoryPolicyError(
            "adaptive memory checkpoint references an unexpected runtime file"
        )
    source_origin = value.get("source_origin")
    runtime_snapshot = value.get("runtime_snapshot")
    if not isinstance(source_origin, Mapping) or set(source_origin) != {
        "schema_version",
        "source_path",
        "source_content_hash",
        "source_raw_hash",
        "source_byte_sha256",
        "runtime_memory_name",
    }:
        raise MemoryPolicyError(
            "adaptive memory checkpoint source origin is malformed"
        )
    if not isinstance(runtime_snapshot, Mapping) or set(runtime_snapshot) != {
        "content_hash",
        "raw_hash",
        "byte_sha256",
        "raw_size",
    }:
        raise MemoryPolicyError(
            "adaptive memory checkpoint runtime snapshot is malformed"
        )
    raw_size = runtime_snapshot.get("raw_size")
    if not isinstance(raw_size, int) or isinstance(raw_size, bool) or raw_size < 0:
        raise MemoryPolicyError(
            "adaptive memory checkpoint runtime size is invalid"
        )
    for field in ("content_hash", "raw_hash", "byte_sha256"):
        digest = runtime_snapshot.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise MemoryPolicyError(
                f"adaptive memory checkpoint {field} is invalid"
            )
    return dict(value)


def _validated_checkpoint_payload(
    checkpoint_root: Path,
    *,
    expected_origin: Mapping[str, Any],
    expected_iteration: Optional[int],
) -> tuple[bytes, dict[str, Any]]:
    checkpoint_memory = checkpoint_root / CHECKPOINT_ADAPTIVE_MEMORY_NAME
    checkpoint_metadata = (
        checkpoint_root / CHECKPOINT_ADAPTIVE_MEMORY_METADATA_NAME
    )
    memory_exists = checkpoint_memory.is_file()
    metadata_exists = checkpoint_metadata.is_file()
    if memory_exists != metadata_exists or not memory_exists:
        raise MemoryPolicyError(
            "adaptive memory checkpoint is incomplete: memory/metadata mismatch"
        )
    document = _load_checkpoint_document(checkpoint_metadata)
    if (
        expected_iteration is not None
        and document["iteration"] != expected_iteration
    ):
        raise MemoryPolicyError(
            "adaptive memory checkpoint iteration does not match the requested checkpoint"
        )
    if canonical_memory_content(document["source_origin"]) != canonical_memory_content(
        expected_origin
    ):
        raise MemoryPolicyError(
            "adaptive memory checkpoint origin does not match the frozen case input"
        )
    payload = checkpoint_memory.read_bytes()
    observed = capture_memory_snapshot(checkpoint_memory)
    expected_snapshot = document["runtime_snapshot"]
    if (
        len(payload) != expected_snapshot["raw_size"]
        or hashlib.sha256(payload).hexdigest() != expected_snapshot["byte_sha256"]
        or observed.raw_hash != expected_snapshot["raw_hash"]
        or observed.content_hash != expected_snapshot["content_hash"]
    ):
        raise MemoryPolicyError(
            "adaptive memory checkpoint bytes do not match their recorded hashes"
        )
    return payload, document


def save_adaptive_memory_checkpoint(
    source_path: str | Path | None,
    target_path: str | Path | None,
    checkpoint_dir: str | Path,
    policy: MemoryPolicyConfig,
    *,
    iteration: int,
) -> Optional[dict[str, Any]]:


    if not _require_checkpoint_policy(policy):
        return None
    _source, target, origin_path, expected_origin = _runtime_paths(
        source_path, target_path
    )
    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    with _origin_lock(target.parent):
        _validate_runtime_origin_locked(target, origin_path, expected_origin)
        runtime_bytes = target.read_bytes()
        runtime_snapshot = capture_memory_snapshot(target)
        document = _checkpoint_document(
            iteration=iteration,
            expected_origin=expected_origin,
            runtime_snapshot=runtime_snapshot,
            runtime_bytes=runtime_bytes,
        )
        _atomic_replace(
            checkpoint_root / CHECKPOINT_ADAPTIVE_MEMORY_NAME,
            runtime_bytes,
        )
        _atomic_replace(
            checkpoint_root / CHECKPOINT_ADAPTIVE_MEMORY_METADATA_NAME,
            _canonical_json(document),
        )


        _validated_checkpoint_payload(
            checkpoint_root,
            expected_origin=expected_origin,
            expected_iteration=iteration,
        )
    return document


def restore_adaptive_memory_checkpoint(
    source_path: str | Path | None,
    target_path: str | Path | None,
    checkpoint_dir: str | Path,
    policy: MemoryPolicyConfig,
    *,
    expected_iteration: Optional[int] = None,
) -> Optional[dict[str, Any]]:


    if not _require_checkpoint_policy(policy):
        return None
    _source, target, origin_path, expected_origin = _runtime_paths(
        source_path, target_path
    )
    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    with _origin_lock(target.parent):
        _validate_runtime_origin_locked(target, origin_path, expected_origin)
        payload, document = _validated_checkpoint_payload(
            checkpoint_root,
            expected_origin=expected_origin,
            expected_iteration=expected_iteration,
        )
        _atomic_replace(target, payload)
        restored = capture_memory_snapshot(target)
        recorded = document["runtime_snapshot"]
        if (
            restored.raw_hash != recorded["raw_hash"]
            or restored.content_hash != recorded["content_hash"]
            or target.read_bytes() != payload
        ):
            raise MemoryPolicyError(
                "adaptive memory checkpoint restore did not preserve checkpoint bytes"
            )
    return document


def prepare_adaptive_memory_target(
    source_path: str | Path | None,
    output_dir: str | Path,
    policy: MemoryPolicyConfig,
) -> Optional[str]:


    if not isinstance(policy, MemoryPolicyConfig):
        raise TypeError("policy must be MemoryPolicyConfig")
    if source_path is None:
        if policy.may_read_adaptive_prior:
            raise MemoryPolicyError(
                "adaptive memory policy requires a frozen case memory.yaml"
            )
        return None

    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        if policy.may_read_adaptive_prior:
            raise MemoryPolicyError(
                f"adaptive memory source does not exist: {source}"
            )
        return None
    if not policy.may_commit_adaptive_prior:
        return str(source)

    runtime_root = Path(output_dir).expanduser().resolve()
    target = runtime_root / RUNTIME_ADAPTIVE_MEMORY_NAME
    origin_path = runtime_root / RUNTIME_ADAPTIVE_MEMORY_ORIGIN_NAME
    expected_origin = _origin_document(source)

    with _origin_lock(runtime_root):
        target_exists = target.exists()
        origin_exists = origin_path.exists()
        if target_exists != origin_exists:
            raise MemoryPolicyError(
                "adaptive memory resume state is incomplete: target/origin mismatch"
            )
        if target_exists:
            observed_origin = _load_origin(origin_path)
            if canonical_memory_content(observed_origin) != canonical_memory_content(
                expected_origin
            ):
                raise MemoryPolicyError(
                    "adaptive memory origin no longer matches the frozen case input"
                )


            capture_memory_snapshot(target)
            return str(target)

        source_bytes = source.read_bytes()


        _atomic_replace(target, source_bytes)
        _atomic_replace(origin_path, _canonical_json(expected_origin))
        initialized = capture_memory_snapshot(target)
        source_snapshot = capture_memory_snapshot(source)
        if (
            initialized.content_hash != source_snapshot.content_hash
            or initialized.raw_hash != source_snapshot.raw_hash
            or target.read_bytes() != source_bytes
        ):
            raise MemoryPolicyError(
                "run-local adaptive memory initialization did not preserve the source"
            )
    return str(target)


__all__ = [
    "ADAPTIVE_MEMORY_CHECKPOINT_VERSION",
    "ADAPTIVE_MEMORY_ORIGIN_VERSION",
    "CHECKPOINT_ADAPTIVE_MEMORY_METADATA_NAME",
    "CHECKPOINT_ADAPTIVE_MEMORY_NAME",
    "RUNTIME_ADAPTIVE_MEMORY_NAME",
    "RUNTIME_ADAPTIVE_MEMORY_ORIGIN_NAME",
    "prepare_adaptive_memory_target",
    "restore_adaptive_memory_checkpoint",
    "save_adaptive_memory_checkpoint",
]
