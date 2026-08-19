

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any, Mapping, Optional


NATIVE_EXECUTION_CONTRACT_VERSION = "astevolve.native_execution_contract.v2"
NATIVE_EXECUTION_CONTRACT_KEY = "_astevolve_native_execution_contract"
PYTHON_CODE_IDENTITY_VERSION = "astevolve.python_module_code_identity.v1"
RESOURCE_IDENTITY_VERSION = "astevolve.execution_resource_identity.v1"
EXECUTION_PAYLOAD_IDENTITY_VERSION = "astevolve.execution_payload_identity.v1"
PYTHON_SOURCE_TREE_IDENTITY_VERSION = "astevolve.python_source_tree_identity.v1"

_IGNORED_PYTHON_SOURCE_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "artifacts",
        "cache",
        "caches",
        "outputs",
    }
)


class ExecutionIdentityError(ValueError):
    pass


def _canonical_clone(value: Any, *, label: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExecutionIdentityError(
            f"{label} must contain finite JSON-compatible data"
        ) from exc


def _canonical_bytes(value: Any, *, label: str) -> bytes:
    cloned = _canonical_clone(value, label=label)
    return json.dumps(
        cloned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(namespace: str, value: Any, *, label: str) -> str:
    payload = _canonical_bytes(value, label=label)
    return hashlib.sha256(namespace.encode("utf-8") + b"\0" + payload).hexdigest()


def _required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ExecutionIdentityError(f"{label} must be a non-empty normalized string")
    return value


def _non_negative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExecutionIdentityError(f"{label} must be a non-negative integer")
    return value


def canonical_payload_sha256(payload: Mapping[str, Any]) -> str:


    if not isinstance(payload, Mapping):
        raise ExecutionIdentityError("execution payload must be a mapping")
    return _digest(
        EXECUTION_PAYLOAD_IDENTITY_VERSION,
        dict(payload),
        label="execution payload",
    )


def _module_file(target: Any) -> Path:
    subject = (
        target if inspect.isclass(target) or inspect.isroutine(target) else type(target)
    )
    module_name = str(getattr(subject, "__module__", "") or "")
    if not module_name:
        raise ExecutionIdentityError("Python component has no defining module")
    try:
        raw_path = inspect.getsourcefile(subject) or inspect.getfile(subject)
    except (OSError, TypeError) as exc:
        raise ExecutionIdentityError(
            f"cannot resolve defining module file for {module_name!r}"
        ) from exc
    path = Path(raw_path)
    if path.suffix in {".pyc", ".pyo"}:
        try:
            source_path = Path(importlib.util.source_from_cache(str(path)))
        except (NotImplementedError, ValueError):
            source_path = path
        if source_path.is_file():
            path = source_path
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ExecutionIdentityError(
            f"cannot resolve defining module file for {module_name!r}"
        ) from exc
    if not resolved.is_file():
        raise ExecutionIdentityError(
            f"defining module for {module_name!r} is not a regular file"
        )
    return resolved


def python_code_identity(
    value: Any,
    *,
    requested_spec: Optional[str] = None,
) -> dict[str, Any]:


    subject = (
        value if inspect.isclass(value) or inspect.isroutine(value) else type(value)
    )
    module_name = _required_text(
        str(getattr(subject, "__module__", "") or ""),
        label="component module",
    )
    qualname = _required_text(
        str(getattr(subject, "__qualname__", "") or ""),
        label="component qualname",
    )
    path = _module_file(subject)
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ExecutionIdentityError(
            f"cannot read defining module for {module_name!r}"
        ) from exc
    core: dict[str, Any] = {
        "schema_version": PYTHON_CODE_IDENTITY_VERSION,
        "module": module_name,
        "qualname": qualname,
        "module_content_sha256": hashlib.sha256(content).hexdigest(),
        "module_size_bytes": len(content),
    }
    if requested_spec is not None:
        core["requested_spec"] = _required_text(
            requested_spec, label="component factory spec"
        )
    return {
        **core,
        "identity_hash": _digest(
            PYTHON_CODE_IDENTITY_VERSION,
            core,
            label="Python code identity",
        ),
    }


def factory_code_identity(spec: str) -> dict[str, Any]:


    normalized = _required_text(spec, label="component factory spec")
    module_name, separator, attribute = normalized.partition(":")
    if not separator or not module_name or not attribute:
        raise ExecutionIdentityError(
            "component factory spec must use module:attribute syntax"
        )
    try:
        value = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise ExecutionIdentityError(
            f"cannot resolve component factory {normalized!r} for identity"
        ) from exc
    if not callable(value):
        raise ExecutionIdentityError(
            f"component factory {normalized!r} is not callable"
        )
    return python_code_identity(value, requested_spec=normalized)


def python_source_tree_identity(
    roots: Mapping[str, Path | str],
    *,
    role: str,
) -> dict[str, Any]:


    resolved_role = _required_text(role, label="source tree role")
    if not isinstance(roots, Mapping) or not roots:
        raise ExecutionIdentityError("Python source roots must be a non-empty mapping")
    resolved_roots: list[tuple[str, Path]] = []
    for raw_label, raw_root in roots.items():
        label = _required_text(raw_label, label="Python source root label")
        if "/" in label or "\\" in label:
            raise ExecutionIdentityError(
                "Python source root labels cannot contain path separators"
            )
        try:
            root = Path(raw_root).expanduser().resolve(strict=True)
        except OSError as exc:
            raise ExecutionIdentityError(
                f"cannot resolve Python source root {label!r}"
            ) from exc
        if not root.is_dir():
            raise ExecutionIdentityError(
                f"Python source root {label!r} is not a directory"
            )
        resolved_roots.append((label, root))
    if len({label for label, _ in resolved_roots}) != len(resolved_roots):
        raise ExecutionIdentityError("Python source root labels must be unique")

    entries = []
    total_bytes = 0
    try:
        for label, root in sorted(resolved_roots, key=lambda item: item[0]):
            candidates = sorted(
                (
                    item
                    for item in root.rglob("*.py")
                    if item.is_file()
                    and not _IGNORED_PYTHON_SOURCE_PARTS.intersection(
                        item.relative_to(root).parts
                    )
                ),
                key=lambda item: item.relative_to(root).as_posix(),
            )
            for item in candidates:
                content = item.read_bytes()
                total_bytes += len(content)
                entries.append(
                    {
                        "source_root": label,
                        "relative_path": item.relative_to(root).as_posix(),
                        "content_sha256": hashlib.sha256(content).hexdigest(),
                        "size_bytes": len(content),
                    }
                )
    except OSError as exc:
        raise ExecutionIdentityError(
            "cannot read Python dependency source tree"
        ) from exc
    if not entries:
        raise ExecutionIdentityError(
            "Python dependency source tree contains no .py files"
        )
    core = {
        "schema_version": PYTHON_SOURCE_TREE_IDENTITY_VERSION,
        "role": resolved_role,
        "source_roots": sorted(label for label, _ in resolved_roots),
        "tree_sha256": _digest(
            PYTHON_SOURCE_TREE_IDENTITY_VERSION,
            entries,
            label="Python dependency source tree",
        ),
        "file_count": len(entries),
        "size_bytes": total_bytes,
    }
    return {
        **core,
        "identity_hash": _digest(
            PYTHON_SOURCE_TREE_IDENTITY_VERSION,
            core,
            label="Python source tree identity",
        ),
    }


def resource_content_identity(
    path: Path | str,
    *,
    role: str,
) -> dict[str, Any]:


    resolved_role = _required_text(role, label="resource role")
    try:
        target = Path(path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ExecutionIdentityError(
            f"cannot resolve execution resource {resolved_role!r}"
        ) from exc
    if target.is_file():
        try:
            content = target.read_bytes()
        except OSError as exc:
            raise ExecutionIdentityError(
                f"cannot read execution resource {resolved_role!r}"
            ) from exc
        core = {
            "schema_version": RESOURCE_IDENTITY_VERSION,
            "role": resolved_role,
            "kind": "file",
            "content_sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    elif target.is_dir():
        entries = []
        total_bytes = 0
        try:
            candidates = sorted(
                (item for item in target.rglob("*") if item.is_file()),
                key=lambda item: item.relative_to(target).as_posix(),
            )
            for item in candidates:
                content = item.read_bytes()
                total_bytes += len(content)
                entries.append(
                    {
                        "relative_path": item.relative_to(target).as_posix(),
                        "content_sha256": hashlib.sha256(content).hexdigest(),
                        "size_bytes": len(content),
                    }
                )
        except OSError as exc:
            raise ExecutionIdentityError(
                f"cannot read execution resource tree {resolved_role!r}"
            ) from exc
        tree_hash = _digest(
            RESOURCE_IDENTITY_VERSION,
            entries,
            label=f"resource tree {resolved_role}",
        )
        core = {
            "schema_version": RESOURCE_IDENTITY_VERSION,
            "role": resolved_role,
            "kind": "directory",
            "tree_sha256": tree_hash,
            "file_count": len(entries),
            "size_bytes": total_bytes,
        }
    else:
        raise ExecutionIdentityError(
            f"execution resource {resolved_role!r} is not a file or directory"
        )
    return {
        **core,
        "identity_hash": _digest(
            RESOURCE_IDENTITY_VERSION,
            core,
            label=f"resource identity {resolved_role}",
        ),
    }


def absent_resource_identity(*, role: str) -> dict[str, Any]:


    core = {
        "schema_version": RESOURCE_IDENTITY_VERSION,
        "role": _required_text(role, label="resource role"),
        "kind": "absent",
    }
    return {
        **core,
        "identity_hash": _digest(
            RESOURCE_IDENTITY_VERSION,
            core,
            label="absent resource identity",
        ),
    }


def seal_execution_contract(core: Mapping[str, Any]) -> dict[str, Any]:


    if not isinstance(core, Mapping):
        raise ExecutionIdentityError("execution contract core must be a mapping")
    cloned = _canonical_clone(dict(core), label="execution contract")
    if not isinstance(cloned, dict):
        raise ExecutionIdentityError("execution contract must be a JSON object")
    if "contract_hash" in cloned:
        raise ExecutionIdentityError(
            "execution contract core cannot define contract_hash"
        )
    if cloned.get("schema_version") != NATIVE_EXECUTION_CONTRACT_VERSION:
        raise ExecutionIdentityError(
            "execution contract has an unsupported schema version"
        )
    return {
        **cloned,
        "contract_hash": _digest(
            NATIVE_EXECUTION_CONTRACT_VERSION,
            cloned,
            label="execution contract",
        ),
    }


def validate_execution_contract(
    value: Mapping[str, Any],
    *,
    run_id: str,
    root_seed: int,
    proposal_budget: int,
) -> dict[str, Any]:


    if not isinstance(value, Mapping):
        raise ExecutionIdentityError("execution contract must be a mapping")
    cloned = _canonical_clone(dict(value), label="execution contract")
    if not isinstance(cloned, dict):
        raise ExecutionIdentityError("execution contract must be a JSON object")
    observed_hash = cloned.pop("contract_hash", None)
    sealed = seal_execution_contract(cloned)
    if observed_hash != sealed["contract_hash"]:
        raise ExecutionIdentityError("execution contract hash mismatch")
    expected_run = _required_text(run_id, label="run_id")
    expected_seed = _non_negative_int(root_seed, label="root_seed")
    expected_budget = _non_negative_int(proposal_budget, label="proposal_budget")
    if sealed.get("run_id") != expected_run:
        raise ExecutionIdentityError("execution contract belongs to another run_id")
    if sealed.get("root_seed") != expected_seed:
        raise ExecutionIdentityError("execution contract root_seed mismatch")
    if sealed.get("proposal_budget") != expected_budget:
        raise ExecutionIdentityError("execution contract proposal_budget mismatch")
    if not isinstance(sealed.get("components"), dict):
        raise ExecutionIdentityError("execution contract components must be an object")
    if not isinstance(sealed.get("inputs"), dict):
        raise ExecutionIdentityError("execution contract inputs must be an object")
    return sealed


__all__ = [
    "EXECUTION_PAYLOAD_IDENTITY_VERSION",
    "ExecutionIdentityError",
    "NATIVE_EXECUTION_CONTRACT_KEY",
    "NATIVE_EXECUTION_CONTRACT_VERSION",
    "PYTHON_CODE_IDENTITY_VERSION",
    "PYTHON_SOURCE_TREE_IDENTITY_VERSION",
    "RESOURCE_IDENTITY_VERSION",
    "absent_resource_identity",
    "canonical_payload_sha256",
    "factory_code_identity",
    "python_code_identity",
    "python_source_tree_identity",
    "resource_content_identity",
    "seal_execution_contract",
    "validate_execution_contract",
]
