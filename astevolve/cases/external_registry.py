

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from astevolve.runtime.paths import artifact_path, resolve_path

from .base import DesignCase


CASE_LIFECYCLE_SCHEMA = "astevolve.case_lifecycle.v1"
LEGACY_FROZEN_STATUS = "legacy_frozen"
_LIFECYCLE_FIELDS = {
    "schema_version",
    "status",
    "default_eligible",
    "smoke_eligible",
    "formal_eligible",
    "explicit_run_only",
}


def _as_path(value: str | os.PathLike[str] | Path) -> Path:
    return Path(value).expanduser().resolve()


def _read_manifest(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Case manifest does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON case manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Case manifest must be a JSON object: {path}")
    return value


def _manifest_case_id(path: Path, manifest: Mapping[str, Any]) -> str:
    case_id = str(manifest.get("case_id") or path.parent.name).strip()
    if not case_id:
        raise ValueError(f"Case manifest has no usable case_id: {path}")
    return case_id


def _manifest_lifecycle(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:


    raw = manifest.get("lifecycle")
    if raw is None:
        return {
            "schema_version": CASE_LIFECYCLE_SCHEMA,
            "status": "active",
            "default_eligible": True,
            "smoke_eligible": True,
            "formal_eligible": True,
            "explicit_run_only": False,
        }
    if raw == "test_fixture":
        return {
            "schema_version": CASE_LIFECYCLE_SCHEMA,
            "status": "test_fixture",
            "default_eligible": True,
            "smoke_eligible": True,
            "formal_eligible": False,
            "explicit_run_only": True,
        }
    location = str(manifest_path or "case manifest")
    if not isinstance(raw, Mapping):
        raise ValueError(f"Invalid lifecycle in {location}: expected an object")
    unknown = set(raw) - _LIFECYCLE_FIELDS
    missing = _LIFECYCLE_FIELDS - set(raw)
    if unknown or missing:
        raise ValueError(
            f"Invalid lifecycle fields in {location}: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    policy = dict(raw)
    if policy["schema_version"] != CASE_LIFECYCLE_SCHEMA:
        raise ValueError(
            f"Unsupported lifecycle schema in {location}: "
            f"{policy['schema_version']!r}"
        )
    if policy["status"] not in {"active", LEGACY_FROZEN_STATUS}:
        raise ValueError(
            f"Unsupported lifecycle status in {location}: {policy['status']!r}"
        )
    for name in (
        "default_eligible",
        "smoke_eligible",
        "formal_eligible",
        "explicit_run_only",
    ):
        if not isinstance(policy[name], bool):
            raise ValueError(f"Invalid lifecycle {name} in {location}: expected bool")
    if policy["status"] == LEGACY_FROZEN_STATUS:
        actual = tuple(
            policy[name]
            for name in (
                "default_eligible",
                "smoke_eligible",
                "formal_eligible",
                "explicit_run_only",
            )
        )
        if actual != (False, False, False, True):
            raise ValueError(
                f"Frozen lifecycle in {location} must be non-eligible and "
                "explicit-run-only"
            )
    return policy


def configured_case_root(
    case_root: str | os.PathLike[str] | Path | None = None,
) -> Optional[Path]:


    raw = (
        case_root
        or os.environ.get("ASTEVOLVE_CASE_ROOT")
        or os.environ.get("ASTEVOLVE_CASES_ROOT")
    )
    return _as_path(raw) if raw else None


def _configured_manifest(
    manifest_path: str | os.PathLike[str] | Path | None = None,
) -> Optional[Path]:
    raw = manifest_path or os.environ.get("ASTEVOLVE_CASE_MANIFEST")
    return _as_path(raw) if raw else None


def _manifest_under_root(root: Path, case_id: Optional[str]) -> Path:


    direct = root / "case.json"
    if direct.is_file():
        return direct
    if case_id:
        return root / str(case_id) / "case.json"
    eligible = _discover_manifests(root, include_legacy=False)
    if not eligible:
        raise FileNotFoundError(
            f"No default-eligible case manifests found under explicit root {root}"
        )
    return eligible[0]


def _discover_manifests(root: Path, *, include_legacy: bool) -> List[Path]:
    if not root.is_dir():
        return []
    direct = root / "case.json"
    candidates = [direct] if direct.is_file() else sorted(root.glob("*/case.json"))
    manifests: List[Path] = []
    for path in candidates:
        manifest = _read_manifest(path)
        lifecycle = _manifest_lifecycle(manifest, manifest_path=path)
        if include_legacy or (
            lifecycle["status"] != LEGACY_FROZEN_STATUS
            and lifecycle["default_eligible"]
        ):
            manifests.append(path)
    return manifests


def list_cases(
    *,
    case_root: str | os.PathLike[str] | Path | None = None,
    include_legacy: bool = False,
) -> List[str]:


    root = configured_case_root(case_root)
    if root is None:
        return []
    return sorted(
        _manifest_case_id(path, _read_manifest(path))
        for path in _discover_manifests(root, include_legacy=include_legacy)
    )


def default_case_id(
    *,
    case_root: str | os.PathLike[str] | Path | None = None,
) -> str:


    cases = list_cases(case_root=case_root)
    if not cases:
        root = configured_case_root(case_root)
        raise FileNotFoundError(
            "No default-eligible external case manifest is configured"
            + (f" under {root}" if root else "")
        )
    return cases[0]


def resolve_case(
    case_id: Optional[str] = None,
    *,
    case_root: str | os.PathLike[str] | Path | None = None,
    manifest_path: str | os.PathLike[str] | Path | None = None,
) -> DesignCase:


    exact_manifest = _configured_manifest(manifest_path)
    root = configured_case_root(case_root)
    requested = case_id or os.environ.get("ASTEVOLVE_CASE_ID")
    if exact_manifest is None:
        if root is None:
            raise FileNotFoundError(
                "No case is configured; pass manifest_path/case_root or set "
                "ASTEVOLVE_CASE_MANIFEST/ASTEVOLVE_CASE_ROOT"
            )
        if requested is None:
            requested = default_case_id(case_root=root)
        exact_manifest = _manifest_under_root(root, requested)

    manifest = _read_manifest(exact_manifest)
    _manifest_lifecycle(manifest, manifest_path=exact_manifest)
    selected = _manifest_case_id(exact_manifest, manifest)
    if requested and str(requested).strip() != selected:
        raise ValueError(
            f"Requested case_id {requested!r} does not match manifest case_id "
            f"{selected!r}: {exact_manifest}"
        )
    bundle_root = exact_manifest.parent

    design_state_path = resolve_path(
        os.environ.get("ASTEVOLVE_DESIGN_STATE_PATH")
        or manifest.get("design_state_path"),
        base=bundle_root,
        fallback=bundle_root / "design_state.json",
    )
    memory_path = resolve_path(
        os.environ.get("ASTEVOLVE_MEMORY_PATH") or manifest.get("memory_path"),
        base=bundle_root,
        fallback=bundle_root / "memory.yaml",
    )
    output_root = resolve_path(
        os.environ.get("ASTEVOLVE_CASE_OUTPUT_ROOT") or manifest.get("output_root"),
        base=bundle_root,
        fallback=artifact_path(selected),
    )
    metadata = {
        key: value
        for key, value in manifest.items()
        if key not in {"design_state_path", "memory_path", "output_root"}
    }
    return DesignCase(
        case_id=selected,
        root=bundle_root,
        manifest_path=exact_manifest,
        design_state_path=design_state_path,
        memory_path=memory_path,
        output_root=output_root,
        metadata=metadata,
    )


def current_case() -> DesignCase:


    return resolve_case()


__all__ = [
    "CASE_LIFECYCLE_SCHEMA",
    "LEGACY_FROZEN_STATUS",
    "configured_case_root",
    "current_case",
    "default_case_id",
    "list_cases",
    "resolve_case",
]
