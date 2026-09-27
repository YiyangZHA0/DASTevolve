

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from astevolve.application.ports.runner import DesignSearchRunner
from astevolve.domain import ExperimentResult, RunContext
from astevolve.evolution.domain import SealedEvaluation


CANDIDATE_RECORD_SCHEMA_VERSION = "astevolve.evolution.candidate_record.v1"
ARTIFACT_MANIFEST_SCHEMA_VERSION = "astevolve.evolution.artifact_manifest.v1"
EVALUATOR_DESCRIPTOR_SCHEMA_VERSION = "astevolve.evolution.evaluator_descriptor.v1"
SCIENTIFIC_DIAGNOSTIC_SCHEMA_VERSION = (
    "astevolve.evolution.scientific_candidate_diagnostics.v1"
)
CANDIDATE_RECORD_EVIDENCE_KIND = "content_addressed_candidate"
SCIENTIFIC_DIAGNOSTIC_EVIDENCE_KIND = "candidate_scientific_diagnostics"

_FILE_HASH_CHUNK_BYTES = 1024 * 1024
_MAX_ARTIFACT_REFERENCES = 128
_MAX_ARTIFACT_ROLES = 128
_MAX_ROLE_SEGMENT_CHARS = 64
_MAX_REFERENCE_DEPTH = 8
_MAX_DESCRIPTOR_DEPTH = 8
_MAX_DESCRIPTOR_ITEMS = 64
_MAX_DESCRIPTOR_AUDIT_ITEMS = 512
_MAX_DESCRIPTOR_STRING_CHARS = 1024
_MAX_DESCRIPTOR_BYTES = 128 * 1024
_MAX_SEQUENCE_CHAINS = 256
_MAX_SEQUENCE_BUNDLE_BYTES = 2 * 1024 * 1024
_MAX_CONTRACT_NODES = 256
_MAX_CONTRACT_NODE_CHARS = 256
_MAX_DIAGNOSTIC_RECORD_BYTES = 64 * 1024

_ARTIFACT_PATH_KEYS = frozenset(
    {
        "artifact_path",
        "cif_path",
        "out_dir",
        "output_dir",
        "protenix_out_dir",
        "protenix_summary_json",
        "structure_path",
        "summary_json",
    }
)
_RAW_ARTIFACT_FIELDS = (
    "cif_path",
    "protenix_out_dir",
    "protenix_summary_json",
    "structure_path",
)
_DESCRIPTOR_RESULT_FIELDS = (
    "evaluator_plugin_resolution",
    "sa_config",
    "score_config",
    "search_method",
    "sequence_generator",
)
_DESCRIPTOR_EVALUATION_FIELDS = ("backends", "plugins", "scorer_layers")
_SENSITIVE_KEY_NAMES = frozenset(
    {
        "access_key",
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "password",
        "passphrase",
        "private_key",
        "secret",
        "token",
    }
)
_SENSITIVE_KEY_SUFFIXES = (
    "access_key",
    "api_key",
    "authorization",
    "credential",
    "password",
    "private_key",
    "secret",
    "_token",
)
_PATH_KEY_FRAGMENTS = (
    "artifact_path",
    "cache_dir",
    "checkpoint_path",
    "file_path",
    "model_path",
    "output_dir",
    "output_path",
    "project_root",
    "root_dir",
)
_LOGICAL_ROLE_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_SEMANTIC_FIELDS = {
    "semantic_graph_summary": (
        "schema_version",
        "enabled",
        "ablation_mode",
        "edit_contract_enabled",
        "structural_node_count",
        "functional_node_count",
        "structural_edge_count",
        "functional_edge_count",
    ),
    "semantic_graph_diagnosis": (
        "schema_version",
        "enabled",
        "binding_policy",
        "hard_gate_pass",
    ),
}


class CandidateArtifactError(ValueError):
    pass


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CandidateArtifactError(
            "candidate evidence requires finite JSON values"
        ) from exc


def _canonical_hash(namespace: str, value: Any) -> str:
    payload = _canonical_json(value)
    return hashlib.sha256(f"{namespace}\0{payload}".encode("utf-8")).hexdigest()


def _content_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _detached_json(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _callable_identity(value: Callable[..., Any] | Any) -> Mapping[str, str]:
    return {
        "module": str(getattr(value, "__module__", value.__class__.__module__) or ""),
        "qualname": str(
            getattr(value, "__qualname__", value.__class__.__qualname__) or ""
        ),
    }


def _is_sensitive_key(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return bool(
        normalized in _SENSITIVE_KEY_NAMES
        or normalized.endswith(_SENSITIVE_KEY_SUFFIXES)
        or normalized.startswith(("authorization_", "credential_", "password_"))
        or "_client_secret" in normalized
    )


def _is_path_key(value: str) -> bool:
    normalized = value.strip().lower()
    return bool(
        normalized in {"dir", "directory", "path", "root"}
        or normalized.endswith(("_dir", "_directory", "_path", "_root"))
        or any(fragment in normalized for fragment in _PATH_KEY_FRAGMENTS)
    )


def _looks_like_absolute_path(value: str) -> bool:
    stripped = value.strip()
    return bool(
        stripped.startswith(("/", "~/", "./", "../", "\\\\"))
        or re.match(r"^[A-Za-z]:[\\/]", stripped)
        or (
            "://" not in stripped
            and ("/" in stripped or "\\" in stripped)
            and Path(stripped).suffix.lower()
            in {".bin", ".cif", ".json", ".npz", ".pt", ".pth", ".safetensors"}
        )
    )


def _descriptor_value(
    value: Any,
    *,
    path: str,
    redacted: list[str],
    omitted: list[str],
    depth: int = 0,
) -> Any:


    if depth > _MAX_DESCRIPTOR_DEPTH:
        omitted.append(f"{path}.__depth_limit__")
        return {"omitted": "depth_limit"}
    if isinstance(value, Mapping):
        projected: Dict[str, Any] = {}
        items = sorted(value.items(), key=lambda pair: str(pair[0]))
        for index, (raw_key, item) in enumerate(items[:_MAX_DESCRIPTOR_ITEMS]):
            if (
                not isinstance(raw_key, str)
                or not raw_key
                or len(raw_key) > _MAX_ROLE_SEGMENT_CHARS
                or not _LOGICAL_ROLE_SEGMENT.fullmatch(raw_key)
            ):
                omitted.append(f"{path}.__key_{index}_omitted__")
                continue
            key = raw_key
            child_path = f"{path}.{key}" if path else key
            if _is_sensitive_key(key):
                projected[key] = {"redacted": True}
                redacted.append(child_path)
                continue
            if _is_path_key(key):
                projected[key] = {"omitted": "path_value"}
                omitted.append(child_path)
                continue
            try:
                projected[key] = _descriptor_value(
                    item,
                    path=child_path,
                    redacted=redacted,
                    omitted=omitted,
                    depth=depth + 1,
                )
            except CandidateArtifactError:
                omitted.append(child_path)
        if len(items) > _MAX_DESCRIPTOR_ITEMS:
            omitted.append(f"{path}.__item_limit__")
        return projected
    if isinstance(value, (list, tuple)):
        projected_list: list[Any] = []
        for index, item in enumerate(value[:_MAX_DESCRIPTOR_ITEMS]):
            try:
                projected_list.append(
                    _descriptor_value(
                        item,
                        path=f"{path}[{index}]",
                        redacted=redacted,
                        omitted=omitted,
                        depth=depth + 1,
                    )
                )
            except CandidateArtifactError:
                omitted.append(f"{path}[{index}]")
        if len(value) > _MAX_DESCRIPTOR_ITEMS:
            omitted.append(f"{path}.__item_limit__")
        return projected_list
    if isinstance(value, Path):
        omitted.append(path)
        return {"omitted": "path_value"}
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CandidateArtifactError(
                f"non-finite evaluator descriptor value at {path or 'value'}"
            )
        return value
    if isinstance(value, str):
        if _looks_like_absolute_path(value):
            omitted.append(path)
            return {"omitted": "path_value"}
        if len(value) > _MAX_DESCRIPTOR_STRING_CHARS:
            omitted.append(path)
            return {
                "omitted": "string_limit",
                "content_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                "size_bytes": len(value.encode("utf-8")),
            }
        return value
    raise CandidateArtifactError(
        f"unsupported evaluator descriptor value at {path or 'value'}: "
        f"{type(value).__name__}"
    )


def _optional_descriptor_value(
    value: Any,
    *,
    path: str,
    redacted: list[str],
    omitted: list[str],
) -> Optional[Any]:
    try:
        return _descriptor_value(
            value,
            path=path,
            redacted=redacted,
            omitted=omitted,
        )
    except CandidateArtifactError:
        omitted.append(path)
        return None


def _build_evaluator_descriptor(
    *,
    adapter: Any,
    runner: DesignSearchRunner,
    context: RunContext,
    result: ExperimentResult,
) -> Dict[str, Any]:
    redacted: list[str] = []
    omitted: list[str] = []
    result_declarations: Dict[str, Any] = {}
    for field in _DESCRIPTOR_RESULT_FIELDS:
        if field not in result.raw:
            continue
        projected = _optional_descriptor_value(
            result.raw[field],
            path=f"result.{field}",
            redacted=redacted,
            omitted=omitted,
        )
        if projected is not None:
            result_declarations[field] = projected
    evaluation_declarations: Dict[str, Any] = {}
    for field in _DESCRIPTOR_EVALUATION_FIELDS:
        if field not in result.evaluation.metadata:
            continue
        projected = _optional_descriptor_value(
            result.evaluation.metadata[field],
            path=f"evaluation.{field}",
            redacted=redacted,
            omitted=omitted,
        )
        if projected is not None:
            evaluation_declarations[field] = projected
    settings = _optional_descriptor_value(
        context.settings,
        path="context.settings",
        redacted=redacted,
        omitted=omitted,
    )
    runner_declarations: Dict[str, Any] = {}
    for attribute in ("model", "model_name", "tool_name", "tool_version", "version"):
        value = getattr(runner, attribute, None)
        if isinstance(value, (str, int, float, bool)) and value != "":
            projected = _optional_descriptor_value(
                value,
                path=f"runner.{attribute}",
                redacted=redacted,
                omitted=omitted,
            )
            if projected is not None:
                runner_declarations[attribute] = projected
    case_id = _optional_descriptor_value(
        str(context.case_id),
        path="context.case_id",
        redacted=redacted,
        omitted=omitted,
    )
    report_schema = _optional_descriptor_value(
        result.evaluation.schema_version,
        path="evaluation.report_schema",
        redacted=redacted,
        omitted=omitted,
    )
    core = {
        "schema_version": EVALUATOR_DESCRIPTOR_SCHEMA_VERSION,
        "adapter": dict(_callable_identity(adapter)),
        "runner": {
            **dict(_callable_identity(runner)),
            "declared": runner_declarations,
        },
        "evaluation_report_schema": report_schema,
        "context": {
            "case_id": case_id,
            "seed": context.seed,
            "settings": settings if isinstance(settings, Mapping) else {},
        },
        "result_declarations": result_declarations,
        "evaluation_declarations": evaluation_declarations,
        "redacted_fields": sorted(set(redacted)),
        "omitted_fields": sorted(set(omitted)),
    }
    if len(_canonical_json(core).encode("utf-8")) > _MAX_DESCRIPTOR_BYTES:
        raise CandidateArtifactError("evaluator descriptor exceeds ledger size limit")
    return {
        **core,
        "descriptor_hash": _canonical_hash(EVALUATOR_DESCRIPTOR_SCHEMA_VERSION, core),
    }


def _logical_role_segment(value: Any) -> str:
    segment = str(value).strip()
    if (
        not segment
        or len(segment) > _MAX_ROLE_SEGMENT_CHARS
        or not _LOGICAL_ROLE_SEGMENT.fullmatch(segment)
    ):
        raise CandidateArtifactError(
            "artifact manifest role keys must be short logical identifiers"
        )
    return segment


def _is_artifact_path_key(key: str, *, under_artifact_paths: bool) -> bool:
    normalized = key.strip().lower()
    return bool(
        under_artifact_paths
        or normalized in _ARTIFACT_PATH_KEYS
        or normalized.endswith("_artifact_path")
    )


def _collect_artifact_references(
    value: Any,
    *,
    prefix: str,
    under_artifact_paths: bool = False,
    depth: int = 0,
) -> list[tuple[str, str]]:
    if depth > _MAX_REFERENCE_DEPTH:
        raise CandidateArtifactError("artifact declaration exceeds nesting limit")
    references: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for raw_key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            key = _logical_role_segment(raw_key)
            role = f"{prefix}.{key}" if prefix else key
            nested_artifact_paths = under_artifact_paths or key == "artifact_paths"
            if (
                _is_artifact_path_key(
                    key,
                    under_artifact_paths=under_artifact_paths,
                )
                and isinstance(item, (str, Path))
                and str(item).strip()
            ):
                references.append((role, str(item)))
            elif isinstance(item, (Mapping, list, tuple)):
                references.extend(
                    _collect_artifact_references(
                        item,
                        prefix=role,
                        under_artifact_paths=nested_artifact_paths,
                        depth=depth + 1,
                    )
                )
            if len(references) > _MAX_ARTIFACT_REFERENCES:
                raise CandidateArtifactError(
                    "artifact declaration exceeds manifest reference limit"
                )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            references.extend(
                _collect_artifact_references(
                    item,
                    prefix=f"{prefix}.{index}",
                    under_artifact_paths=under_artifact_paths,
                    depth=depth + 1,
                )
            )
            if len(references) > _MAX_ARTIFACT_REFERENCES:
                raise CandidateArtifactError(
                    "artifact declaration exceeds manifest reference limit"
                )
    return references


def _declared_path_fields(value: Any) -> Dict[str, Any]:


    if not isinstance(value, Mapping):
        return {}
    selected: Dict[str, Any] = {}
    for raw_key, item in value.items():
        key = str(raw_key)
        if key == "artifact_paths" or _is_artifact_path_key(
            key, under_artifact_paths=False
        ):
            selected[key] = item
    return selected


def _resolve_artifact_path(locator: str, context: RunContext) -> Path:
    if "://" in locator:
        raise CandidateArtifactError(
            "non-filesystem artifact references cannot be content-addressed"
        )
    raw = Path(locator).expanduser()
    if raw.is_absolute():
        return Path(os.path.abspath(raw))
    candidates = (
        Path(context.output_root) / raw,
        Path(context.project_root) / raw,
        raw,
    )
    for candidate in candidates:
        if candidate.exists() or candidate.is_symlink():
            return Path(os.path.abspath(candidate))
    return Path(os.path.abspath(candidates[0]))


def _stream_file_hash(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            total = 0
            while True:
                chunk = handle.read(_FILE_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise CandidateArtifactError("cannot hash candidate artifact") from exc
    if (
        before.st_size != total
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise CandidateArtifactError("candidate artifact changed while being hashed")
    return total, digest.hexdigest()


def _artifact_entry_identity(entry: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "roles": list(entry["roles"]),
        "exists": entry["exists"],
        "kind": entry["kind"],
        "size_bytes": entry["size_bytes"],
        "content_sha256": entry["content_sha256"],
        "file_count": entry["file_count"],
    }


def _artifact_content_identity(entry: Mapping[str, Any]) -> Dict[str, Any]:


    return {
        "exists": entry["exists"],
        "kind": entry["kind"],
        "size_bytes": entry["size_bytes"],
        "content_sha256": entry["content_sha256"],
        "file_count": entry["file_count"],
    }


def _artifact_entry(
    *,
    roles: Sequence[str],
    path: Path,
    file_hash_cache: Dict[str, tuple[int, str]],
) -> Dict[str, Any]:
    resolved_roles = sorted(set(str(role) for role in roles if str(role)))
    if not resolved_roles or len(resolved_roles) > _MAX_ARTIFACT_ROLES:
        raise CandidateArtifactError("artifact entry roles exceed manifest bounds")
    if path.is_symlink():
        raise CandidateArtifactError("artifact manifests do not follow symlinks")
    if not path.exists():
        core = {
            "roles": resolved_roles,
            "exists": False,
            "kind": "missing",
            "size_bytes": None,
            "content_sha256": None,
            "file_count": 0,
        }
    elif path.is_file():
        cache_key = str(path)
        cached = file_hash_cache.get(cache_key)
        if cached is None:
            cached = _stream_file_hash(path)
            file_hash_cache[cache_key] = cached
        size, content_hash = cached
        core = {
            "roles": resolved_roles,
            "exists": True,
            "kind": "file",
            "size_bytes": size,
            "content_sha256": content_hash,
            "file_count": 1,
        }
    elif path.is_dir():
        try:
            before_directory = path.stat()
            all_descendants = sorted(
                path.rglob("*"),
                key=lambda item: item.relative_to(path).as_posix(),
            )
        except OSError as exc:
            raise CandidateArtifactError(
                "cannot enumerate candidate artifact directory"
            ) from exc
        descendants: list[Path] = []
        for item in all_descendants:
            if item.is_symlink():
                raise CandidateArtifactError(
                    "artifact directories must not contain symlinks"
                )
            if item.is_file():
                descendants.append(item)
            elif not item.is_dir():
                raise CandidateArtifactError(
                    "artifact directory contains a non-regular entry"
                )
        member_identities: list[Dict[str, Any]] = []
        for item in descendants:
            cache_key = str(item)
            cached = file_hash_cache.get(cache_key)
            if cached is None:
                cached = _stream_file_hash(item)
                file_hash_cache[cache_key] = cached
            size, content_hash = cached
            member_identities.append(
                {"size_bytes": size, "content_sha256": content_hash}
            )
        try:
            after_directory = path.stat()
            final_files = sorted(
                item.relative_to(path).as_posix()
                for item in path.rglob("*")
                if item.is_file() and not item.is_symlink()
            )
        except OSError as exc:
            raise CandidateArtifactError(
                "cannot recheck candidate artifact directory"
            ) from exc
        initial_files = [item.relative_to(path).as_posix() for item in descendants]
        if (
            before_directory.st_mtime_ns != after_directory.st_mtime_ns
            or final_files != initial_files
        ):
            raise CandidateArtifactError(
                "candidate artifact directory changed while being hashed"
            )
        sorted_members = sorted(member_identities, key=_canonical_json)
        core = {
            "roles": resolved_roles,
            "exists": True,
            "kind": "directory",
            "size_bytes": sum(item["size_bytes"] for item in member_identities),
            "content_sha256": _canonical_hash(
                "astevolve.evolution.artifact_directory.v1", sorted_members
            ),
            "file_count": len(member_identities),
        }
    else:
        core = {
            "roles": resolved_roles,
            "exists": True,
            "kind": "other",
            "size_bytes": None,
            "content_sha256": None,
            "file_count": 0,
        }
    return {
        **core,
        "entry_hash": _canonical_hash(
            "astevolve.evolution.artifact_entry.v1",
            _artifact_entry_identity(core),
        ),
    }


def _build_artifact_manifest(
    *, result: ExperimentResult, context: RunContext
) -> Dict[str, Any]:
    references = [("context.output_root", str(context.output_root))]
    search_artifacts = _declared_path_fields(
        result.artifacts.get("search_artifacts", {})
    )
    references.extend(
        _collect_artifact_references(
            search_artifacts,
            prefix="artifacts.search_artifacts",
        )
    )
    selected_raw = {
        field: result.raw[field]
        for field in _RAW_ARTIFACT_FIELDS
        if field in result.raw
    }
    structure_metrics = _declared_path_fields(result.raw.get("structure_metrics"))
    if structure_metrics:
        selected_raw["structure_metrics"] = structure_metrics
    references.extend(_collect_artifact_references(selected_raw, prefix="result"))
    if len(references) > _MAX_ARTIFACT_REFERENCES:
        raise CandidateArtifactError(
            "artifact declaration exceeds manifest reference limit"
        )
    grouped: Dict[str, Dict[str, Any]] = {}
    for role, declared_locator in references:
        path = _resolve_artifact_path(declared_locator, context)
        key = str(path)
        block = grouped.setdefault(key, {"path": path, "roles": []})
        block["roles"].append(role)
    file_hash_cache: Dict[str, tuple[int, str]] = {}
    path_entries = [
        _artifact_entry(
            roles=block["roles"],
            path=block["path"],
            file_hash_cache=file_hash_cache,
        )
        for block in grouped.values()
    ]


    content_groups: Dict[str, Dict[str, Any]] = {}
    for entry in path_entries:
        content = _artifact_content_identity(entry)
        key = _canonical_json(content)
        grouped_entry = content_groups.setdefault(key, {**content, "roles": []})
        grouped_entry["roles"].extend(entry["roles"])
    entries = []
    for grouped_entry in content_groups.values():
        core = {
            "roles": sorted(set(grouped_entry.pop("roles"))),
            **grouped_entry,
        }
        entries.append(
            {
                **core,
                "entry_hash": _canonical_hash(
                    "astevolve.evolution.artifact_entry.v1",
                    _artifact_entry_identity(core),
                ),
            }
        )
    entries.sort(key=lambda entry: _canonical_json(_artifact_entry_identity(entry)))
    identity_entries = [_artifact_entry_identity(entry) for entry in entries]
    core = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "path_is_identity": False,
        "entries": entries,
    }
    return {
        **core,
        "artifact_manifest_hash": _canonical_hash(
            ARTIFACT_MANIFEST_SCHEMA_VERSION,
            {
                "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
                "path_is_identity": False,
                "entries": identity_entries,
            },
        ),
    }


def _verify_descriptor_projection(
    value: Any,
    *,
    path: str,
    depth: int = 0,
) -> None:
    if depth > _MAX_DESCRIPTOR_DEPTH + 4:
        raise CandidateArtifactError("evaluator descriptor exceeds nesting limit")
    if isinstance(value, Mapping):
        if len(value) > _MAX_DESCRIPTOR_ITEMS:
            raise CandidateArtifactError("evaluator descriptor mapping is too large")
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or not key
                or len(key) > _MAX_ROLE_SEGMENT_CHARS
                or not _LOGICAL_ROLE_SEGMENT.fullmatch(key)
            ):
                raise CandidateArtifactError("evaluator descriptor key is invalid")
            if _is_sensitive_key(key) and item != {"redacted": True}:
                raise CandidateArtifactError(
                    "evaluator descriptor exposes a sensitive value"
                )
            if _is_path_key(key) and item != {"omitted": "path_value"}:
                raise CandidateArtifactError(
                    "evaluator descriptor exposes a filesystem path"
                )
            _verify_descriptor_projection(
                item,
                path=f"{path}.{key}",
                depth=depth + 1,
            )
        return
    if isinstance(value, list):
        limit = (
            _MAX_DESCRIPTOR_AUDIT_ITEMS
            if path.endswith((".redacted_fields", ".omitted_fields"))
            else _MAX_DESCRIPTOR_ITEMS
        )
        if len(value) > limit:
            raise CandidateArtifactError("evaluator descriptor list is too large")
        for index, item in enumerate(value):
            _verify_descriptor_projection(
                item,
                path=f"{path}[{index}]",
                depth=depth + 1,
            )
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise CandidateArtifactError("evaluator descriptor contains non-finite data")
    if isinstance(value, str):
        if len(value) > _MAX_DESCRIPTOR_STRING_CHARS:
            raise CandidateArtifactError("evaluator descriptor string is too large")
        if _looks_like_absolute_path(value):
            raise CandidateArtifactError(
                "evaluator descriptor contains a filesystem path"
            )
        return
    raise CandidateArtifactError("evaluator descriptor value is invalid")


def _verify_evaluator_descriptor(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateArtifactError("evaluator descriptor must be a mapping")
    descriptor = _detached_json(value)
    if descriptor.get("schema_version") != EVALUATOR_DESCRIPTOR_SCHEMA_VERSION:
        raise CandidateArtifactError("unsupported evaluator descriptor schema")
    observed = descriptor.pop("descriptor_hash", None)
    if set(descriptor) != {
        "schema_version",
        "adapter",
        "runner",
        "evaluation_report_schema",
        "context",
        "result_declarations",
        "evaluation_declarations",
        "redacted_fields",
        "omitted_fields",
    }:
        raise CandidateArtifactError("evaluator descriptor fields are invalid")
    for identity_name in ("adapter", "runner"):
        identity = descriptor[identity_name]
        required = {"module", "qualname"}
        if identity_name == "runner":
            required.add("declared")
        if not isinstance(identity, Mapping) or set(identity) != required:
            raise CandidateArtifactError(
                f"evaluator {identity_name} identity is invalid"
            )
    context = descriptor["context"]
    if not isinstance(context, Mapping) or set(context) != {
        "case_id",
        "seed",
        "settings",
    }:
        raise CandidateArtifactError("evaluator context descriptor is invalid")
    for field in (
        "result_declarations",
        "evaluation_declarations",
    ):
        if not isinstance(descriptor[field], Mapping):
            raise CandidateArtifactError("evaluator declarations are invalid")
    for field in ("redacted_fields", "omitted_fields"):
        audit_values = descriptor[field]
        if (
            not isinstance(audit_values, list)
            or audit_values != sorted(set(audit_values))
            or any(not isinstance(item, str) or not item for item in audit_values)
        ):
            raise CandidateArtifactError(
                "evaluator descriptor audit fields are invalid"
            )
    _verify_descriptor_projection(descriptor, path="descriptor")
    expected = _canonical_hash(EVALUATOR_DESCRIPTOR_SCHEMA_VERSION, descriptor)
    if observed != expected:
        raise CandidateArtifactError("evaluator descriptor hash mismatch")
    if len(_canonical_json(descriptor).encode("utf-8")) > _MAX_DESCRIPTOR_BYTES:
        raise CandidateArtifactError("evaluator descriptor exceeds ledger size limit")
    return {**descriptor, "descriptor_hash": expected}


def _verify_artifact_manifest(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateArtifactError("artifact manifest must be a mapping")
    manifest = _detached_json(value)
    if set(manifest) != {
        "schema_version",
        "path_is_identity",
        "entries",
        "artifact_manifest_hash",
    }:
        raise CandidateArtifactError("artifact manifest fields are invalid")
    if manifest["schema_version"] != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise CandidateArtifactError("unsupported artifact manifest schema")
    if manifest["path_is_identity"] is not False:
        raise CandidateArtifactError("artifact paths must not participate in identity")
    entries = manifest["entries"]
    if not isinstance(entries, list) or len(entries) > _MAX_ARTIFACT_REFERENCES:
        raise CandidateArtifactError("artifact manifest entries are invalid")
    identity_entries = []
    observed_roles: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "roles",
            "exists",
            "kind",
            "size_bytes",
            "content_sha256",
            "file_count",
            "entry_hash",
        }:
            raise CandidateArtifactError("artifact manifest entry is invalid")
        roles = entry["roles"]
        if (
            not isinstance(roles, list)
            or not roles
            or len(roles) > _MAX_ARTIFACT_ROLES
            or any(
                not isinstance(role, str)
                or not role
                or len(role)
                > (_MAX_REFERENCE_DEPTH + 2) * (_MAX_ROLE_SEGMENT_CHARS + 1)
                or not _LOGICAL_ROLE_SEGMENT.fullmatch(role)
                for role in roles
            )
            or roles != sorted(set(roles))
            or observed_roles.intersection(roles)
        ):
            raise CandidateArtifactError("artifact entry roles are invalid")
        observed_roles.update(roles)
        if not isinstance(entry["exists"], bool):
            raise CandidateArtifactError("artifact entry existence is invalid")
        if entry["kind"] not in {"directory", "file", "missing", "other"}:
            raise CandidateArtifactError("artifact entry kind is invalid")
        size = entry["size_bytes"]
        if size is not None and (
            isinstance(size, bool) or not isinstance(size, int) or size < 0
        ):
            raise CandidateArtifactError("artifact entry size is invalid")
        file_count = entry["file_count"]
        if (
            isinstance(file_count, bool)
            or not isinstance(file_count, int)
            or file_count < 0
        ):
            raise CandidateArtifactError("artifact file count is invalid")
        content_hash = entry["content_sha256"]
        if content_hash is not None and not _is_sha256(content_hash):
            raise CandidateArtifactError("artifact content hash is invalid")
        if entry["kind"] == "file":
            consistent = bool(
                entry["exists"]
                and size is not None
                and content_hash is not None
                and file_count == 1
            )
        elif entry["kind"] == "directory":
            consistent = bool(
                entry["exists"] and size is not None and content_hash is not None
            )
        elif entry["kind"] == "missing":
            consistent = bool(
                not entry["exists"]
                and size is None
                and content_hash is None
                and file_count == 0
            )
        else:
            consistent = bool(
                entry["exists"]
                and size is None
                and content_hash is None
                and file_count == 0
            )
        if not consistent:
            raise CandidateArtifactError("artifact entry metadata is inconsistent")
        identity = _artifact_entry_identity(entry)
        if entry["entry_hash"] != _canonical_hash(
            "astevolve.evolution.artifact_entry.v1", identity
        ):
            raise CandidateArtifactError("artifact entry hash mismatch")
        identity_entries.append(identity)
    sorted_identities = sorted(identity_entries, key=_canonical_json)
    if identity_entries != sorted_identities:
        raise CandidateArtifactError("artifact entries are not canonically ordered")
    expected_manifest_hash = _canonical_hash(
        ARTIFACT_MANIFEST_SCHEMA_VERSION,
        {
            "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "path_is_identity": False,
            "entries": identity_entries,
        },
    )
    if manifest["artifact_manifest_hash"] != expected_manifest_hash:
        raise CandidateArtifactError("artifact manifest hash mismatch")
    return manifest


def verify_candidate_record(value: Any) -> Dict[str, Any]:


    if not isinstance(value, Mapping):
        raise CandidateArtifactError("candidate record must be a mapping")
    record = _detached_json(value)
    if set(record) != {
        "schema_version",
        "candidate_id",
        "sequence_bundle",
        "sequence_bundle_hash",
        "chain_lengths",
        "artifact_manifest",
        "evaluator_descriptor",
        "candidate_record_hash",
    }:
        raise CandidateArtifactError("candidate record fields are invalid")
    if record["schema_version"] != CANDIDATE_RECORD_SCHEMA_VERSION:
        raise CandidateArtifactError("unsupported candidate record schema")
    sequences = record["sequence_bundle"]
    if (
        not isinstance(sequences, Mapping)
        or not sequences
        or len(sequences) > _MAX_SEQUENCE_CHAINS
        or any(
            not isinstance(chain, str)
            or not chain
            or not isinstance(sequence, str)
            or not sequence
            for chain, sequence in sequences.items()
        )
    ):
        raise CandidateArtifactError("candidate sequence bundle is invalid")
    normalized_sequences = {chain: sequences[chain] for chain in sorted(sequences)}
    if (
        len(_canonical_json(normalized_sequences).encode("utf-8"))
        > _MAX_SEQUENCE_BUNDLE_BYTES
    ):
        raise CandidateArtifactError("candidate sequence bundle exceeds ledger limit")
    expected_sequence_hash = _canonical_hash(
        "astevolve.evolution.sequence_bundle.v1", normalized_sequences
    )
    if record["sequence_bundle_hash"] != expected_sequence_hash:
        raise CandidateArtifactError("candidate sequence hash mismatch")
    if record["candidate_id"] != f"sequence:{expected_sequence_hash}":
        raise CandidateArtifactError("candidate ID does not bind the sequence")
    expected_lengths = {
        chain: len(sequence) for chain, sequence in normalized_sequences.items()
    }
    if record["chain_lengths"] != expected_lengths:
        raise CandidateArtifactError("candidate chain lengths are inconsistent")
    manifest = _verify_artifact_manifest(record["artifact_manifest"])
    descriptor = _verify_evaluator_descriptor(record["evaluator_descriptor"])
    identity = {
        "schema_version": CANDIDATE_RECORD_SCHEMA_VERSION,
        "candidate_id": record["candidate_id"],
        "sequence_bundle_hash": expected_sequence_hash,
        "artifact_manifest_hash": manifest["artifact_manifest_hash"],
        "evaluator_descriptor_hash": descriptor["descriptor_hash"],
    }
    expected_record_hash = _canonical_hash(CANDIDATE_RECORD_SCHEMA_VERSION, identity)
    if record["candidate_record_hash"] != expected_record_hash:
        raise CandidateArtifactError("candidate record hash mismatch")
    record["sequence_bundle"] = normalized_sequences
    record["artifact_manifest"] = manifest
    record["evaluator_descriptor"] = descriptor
    return record


def build_candidate_record(
    *,
    adapter: Any,
    runner: DesignSearchRunner,
    context: RunContext,
    result: ExperimentResult,
) -> Dict[str, Any]:


    sequences = {
        str(chain): str(sequence)
        for chain, sequence in sorted(result.sequences.items())
    }
    if not sequences or any(
        not chain or not sequence for chain, sequence in sequences.items()
    ):
        raise CandidateArtifactError(
            "design search result must contain a non-empty sequence bundle"
        )
    sequence_hash = _canonical_hash("astevolve.evolution.sequence_bundle.v1", sequences)
    candidate_id = f"sequence:{sequence_hash}"
    artifact_manifest = _build_artifact_manifest(result=result, context=context)
    evaluator_descriptor = _build_evaluator_descriptor(
        adapter=adapter,
        runner=runner,
        context=context,
        result=result,
    )
    identity = {
        "schema_version": CANDIDATE_RECORD_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "sequence_bundle_hash": sequence_hash,
        "artifact_manifest_hash": artifact_manifest["artifact_manifest_hash"],
        "evaluator_descriptor_hash": evaluator_descriptor["descriptor_hash"],
    }
    record = {
        "schema_version": CANDIDATE_RECORD_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "sequence_bundle": sequences,
        "sequence_bundle_hash": sequence_hash,
        "chain_lengths": {
            chain: len(sequence) for chain, sequence in sequences.items()
        },
        "artifact_manifest": artifact_manifest,
        "evaluator_descriptor": evaluator_descriptor,
        "candidate_record_hash": _canonical_hash(
            CANDIDATE_RECORD_SCHEMA_VERSION, identity
        ),
    }
    return verify_candidate_record(record)


def _safe_semantic_summary(kind: str, value: Mapping[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for field in _SAFE_SEMANTIC_FIELDS[kind]:
        item = value.get(field)
        if isinstance(item, bool):
            summary[field] = item
        elif isinstance(item, int) and not isinstance(item, bool):
            summary[field] = item
        elif isinstance(item, str) and len(item) <= 128:
            summary[field] = item
    if kind == "semantic_graph_diagnosis":
        reasons = value.get("disqualification_reasons")
        if isinstance(reasons, (list, tuple)):
            summary["disqualification_reason_count"] = len(reasons)
    return summary


def _semantic_digest(kind: str, value: Any) -> Dict[str, Any]:
    if value is None:
        return {"available": False}
    if not isinstance(value, Mapping):
        raise CandidateArtifactError(f"{kind} must be a mapping when present")
    canonical = _canonical_json(value)
    return {
        "available": True,
        "content_encoding": "canonical-json",
        "content_sha256": _content_hash(canonical),
        "size_bytes": len(canonical.encode("utf-8")),
        "safe_summary": _safe_semantic_summary(kind, value),
    }


def _edit_contract_projection(result: ExperimentResult) -> Dict[str, Any]:
    contract = result.edit_contract
    required = list(contract.required_nodes)
    forbidden = list(contract.forbidden_nodes)
    if len(required) > _MAX_CONTRACT_NODES or len(forbidden) > _MAX_CONTRACT_NODES:
        raise CandidateArtifactError("edit contract exceeds node evidence limit")
    if any(
        not node or len(node) > _MAX_CONTRACT_NODE_CHARS
        for node in required + forbidden
    ):
        raise CandidateArtifactError("edit contract contains an invalid node label")
    full_contract = contract.to_dict()
    return {
        "schema_version": contract.schema_version,
        "action": contract.action,
        "required_nodes": required,
        "forbidden_nodes": forbidden,
        "mutation_budget": dict(contract.mutation_budget),
        "contract_hash": _canonical_hash(
            "astevolve.evolution.edit_contract.v1", full_contract
        ),
    }


def verify_scientific_diagnostic_record(value: Any) -> Dict[str, Any]:


    if not isinstance(value, Mapping):
        raise CandidateArtifactError("scientific diagnostic record must be a mapping")
    record = _detached_json(value)
    if set(record) != {
        "schema_version",
        "candidate_id",
        "candidate_record_hash",
        "edit_contract",
        "semantic_diagnostics",
        "diagnostic_record_hash",
    }:
        raise CandidateArtifactError("scientific diagnostic fields are invalid")
    if record["schema_version"] != SCIENTIFIC_DIAGNOSTIC_SCHEMA_VERSION:
        raise CandidateArtifactError("unsupported scientific diagnostic schema")
    if (
        not isinstance(record["candidate_id"], str)
        or not record["candidate_id"].startswith("sequence:")
        or not _is_sha256(record["candidate_id"][len("sequence:") :])
        or not _is_sha256(record["candidate_record_hash"])
    ):
        raise CandidateArtifactError(
            "scientific diagnostic candidate binding is invalid"
        )
    contract = record["edit_contract"]
    if not isinstance(contract, Mapping) or set(contract) != {
        "schema_version",
        "action",
        "required_nodes",
        "forbidden_nodes",
        "mutation_budget",
        "contract_hash",
    }:
        raise CandidateArtifactError("scientific edit-contract projection is invalid")
    if (
        not isinstance(contract["schema_version"], str)
        or not contract["schema_version"]
        or len(contract["schema_version"]) > 128
        or not isinstance(contract["action"], str)
        or not contract["action"]
        or len(contract["action"]) > 128
        or not _is_sha256(contract["contract_hash"])
    ):
        raise CandidateArtifactError("scientific edit-contract identity is invalid")
    required = contract["required_nodes"]
    forbidden = contract["forbidden_nodes"]
    for nodes in (required, forbidden):
        if (
            not isinstance(nodes, list)
            or len(nodes) > _MAX_CONTRACT_NODES
            or len(nodes) != len(set(nodes))
            or any(
                not isinstance(node, str)
                or not node
                or len(node) > _MAX_CONTRACT_NODE_CHARS
                for node in nodes
            )
        ):
            raise CandidateArtifactError("scientific edit-contract nodes are invalid")
    if set(required).intersection(forbidden):
        raise CandidateArtifactError("scientific edit-contract nodes overlap")
    budget = contract["mutation_budget"]
    if not isinstance(budget, Mapping) or set(budget) != {"min", "max"}:
        raise CandidateArtifactError("scientific mutation budget is invalid")
    minimum, maximum = budget["min"], budget["max"]
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, int)
        or not isinstance(maximum, int)
        or minimum < 0
        or maximum < minimum
    ):
        raise CandidateArtifactError("scientific mutation budget is invalid")
    diagnostics = record["semantic_diagnostics"]
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != set(
        _SAFE_SEMANTIC_FIELDS
    ):
        raise CandidateArtifactError("semantic diagnostics are invalid")
    for kind, digest in diagnostics.items():
        if not isinstance(digest, Mapping) or not isinstance(
            digest.get("available"), bool
        ):
            raise CandidateArtifactError(f"{kind} digest is invalid")
        if digest["available"]:
            if set(digest) != {
                "available",
                "content_encoding",
                "content_sha256",
                "size_bytes",
                "safe_summary",
            }:
                raise CandidateArtifactError(f"{kind} digest fields are invalid")
            allowed_summary_fields = set(_SAFE_SEMANTIC_FIELDS[kind])
            if kind == "semantic_graph_diagnosis":
                allowed_summary_fields.add("disqualification_reason_count")
            safe_summary = digest["safe_summary"]
            if (
                digest["content_encoding"] != "canonical-json"
                or not _is_sha256(digest["content_sha256"])
                or isinstance(digest["size_bytes"], bool)
                or not isinstance(digest["size_bytes"], int)
                or digest["size_bytes"] < 0
                or not isinstance(safe_summary, Mapping)
                or set(safe_summary) - allowed_summary_fields
            ):
                raise CandidateArtifactError(f"{kind} digest content is invalid")
            for field, item in safe_summary.items():
                if field in {
                    "enabled",
                    "edit_contract_enabled",
                    "hard_gate_pass",
                }:
                    valid = isinstance(item, bool)
                elif field.endswith("_count"):
                    valid = (
                        isinstance(item, int)
                        and not isinstance(item, bool)
                        and item >= 0
                    )
                else:
                    valid = isinstance(item, str) and bool(item) and len(item) <= 128
                if not valid:
                    raise CandidateArtifactError(
                        f"{kind} safe summary value is invalid"
                    )
        elif set(digest) != {"available"}:
            raise CandidateArtifactError(f"{kind} unavailable digest is invalid")
    core = dict(record)
    observed_hash = core.pop("diagnostic_record_hash", None)
    expected_hash = _canonical_hash(SCIENTIFIC_DIAGNOSTIC_SCHEMA_VERSION, core)
    if observed_hash != expected_hash:
        raise CandidateArtifactError("scientific diagnostic record hash mismatch")
    verified = {**core, "diagnostic_record_hash": expected_hash}
    if len(_canonical_json(verified).encode("utf-8")) > _MAX_DIAGNOSTIC_RECORD_BYTES:
        raise CandidateArtifactError(
            "scientific diagnostic record exceeds ledger limit"
        )
    return verified


def build_scientific_diagnostic_record(
    *, candidate_record: Mapping[str, Any], result: ExperimentResult
) -> Dict[str, Any]:


    candidate = verify_candidate_record(candidate_record)
    core = {
        "schema_version": SCIENTIFIC_DIAGNOSTIC_SCHEMA_VERSION,
        "candidate_id": candidate["candidate_id"],
        "candidate_record_hash": candidate["candidate_record_hash"],
        "edit_contract": _edit_contract_projection(result),
        "semantic_diagnostics": {
            kind: _semantic_digest(kind, result.artifacts.get(kind))
            for kind in _SAFE_SEMANTIC_FIELDS
        },
    }
    return verify_scientific_diagnostic_record(
        {
            **core,
            "diagnostic_record_hash": _canonical_hash(
                SCIENTIFIC_DIAGNOSTIC_SCHEMA_VERSION, core
            ),
        }
    )


def recover_candidate_record(evaluation: SealedEvaluation) -> Dict[str, Any]:


    if not isinstance(evaluation, SealedEvaluation):
        raise TypeError("evaluation must be SealedEvaluation")
    records = evaluation.evidence().available(CANDIDATE_RECORD_EVIDENCE_KIND)
    if len(records) != 1:
        raise CandidateArtifactError(
            "sealed evaluation must contain exactly one candidate record"
        )
    record = verify_candidate_record(records[0].value)
    if record["candidate_id"] != evaluation.candidate_id:
        raise CandidateArtifactError(
            "sealed evaluation candidate ID differs from candidate record"
        )
    return record


def recover_scientific_diagnostic_record(
    evaluation: SealedEvaluation,
) -> Dict[str, Any]:


    if not isinstance(evaluation, SealedEvaluation):
        raise TypeError("evaluation must be SealedEvaluation")
    records = evaluation.evidence().available(SCIENTIFIC_DIAGNOSTIC_EVIDENCE_KIND)
    if len(records) != 1:
        raise CandidateArtifactError(
            "sealed evaluation must contain exactly one scientific diagnostic record"
        )
    candidate = recover_candidate_record(evaluation)
    record = verify_scientific_diagnostic_record(records[0].value)
    if (
        record["candidate_id"] != candidate["candidate_id"]
        or record["candidate_record_hash"] != candidate["candidate_record_hash"]
    ):
        raise CandidateArtifactError(
            "scientific diagnostic record differs from candidate record"
        )
    return record


__all__ = [
    "ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "CANDIDATE_RECORD_EVIDENCE_KIND",
    "CANDIDATE_RECORD_SCHEMA_VERSION",
    "EVALUATOR_DESCRIPTOR_SCHEMA_VERSION",
    "SCIENTIFIC_DIAGNOSTIC_EVIDENCE_KIND",
    "SCIENTIFIC_DIAGNOSTIC_SCHEMA_VERSION",
    "CandidateArtifactError",
    "build_candidate_record",
    "build_scientific_diagnostic_record",
    "recover_candidate_record",
    "recover_scientific_diagnostic_record",
    "verify_candidate_record",
    "verify_scientific_diagnostic_record",
]
