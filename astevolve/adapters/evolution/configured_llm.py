

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Tuple
from urllib.parse import urlsplit

from astevolve.adapters.llm.registry import create_language_model
from astevolve.domain import DesignStrategy
from astevolve.domain.dual_ast import ExecutableDualAST
from astevolve.evolution.archive import ArchiveProjection
from astevolve.evolution.domain import DUAL_AST_REVISION, STRATEGY_REVISION
from astevolve.evolution.memory import EvolutionMemoryProjection
from astevolve.evolution.orchestrator import Revision
from astevolve.evolution.parent_selection import (
    PARENT_SELECTION_POLICY_VERSION,
    ArchiveParentSelector,
    ParentSelectionPolicy,
)

from .llm_proposal import (
    LLM_PROMPT_INPUT_CONTEXT_FIELD,
    LLM_PROPOSAL_POLICY_SCHEMA_VERSION,
    LLMProposalPolicy,
    StructuredLLMProposalSource,
)

if TYPE_CHECKING:
    from astevolve.evolution.cli import NativeRuntimeServices


NATIVE_LLM_RUNTIME_SCHEMA_VERSION = "astevolve.native_llm_runtime.v1"
NATIVE_LLM_SUPPLEMENTAL_CONTEXT_VERSION = "astevolve.native_llm_supplemental_context.v1"

_RUNTIME_FIELDS = frozenset(
    {
        "schema_version",
        "provider",
        "provider_config",
        "proposal_policy",
        "parent_selection_policy",
        "initial_revisions",
        "initial_parent_ids",
        "temperature",
        "top_p",
        "max_tokens",
        "timeout_seconds",
        "max_concurrency",
    }
)
_RUNTIME_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "provider",
        "provider_config",
        "proposal_policy",
        "parent_selection_policy",
        "initial_revisions",
        "initial_parent_ids",
    }
)
_PROPOSAL_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "policy_id",
        "kind",
        "mutable_fields",
        "max_context_bytes",
        "max_response_bytes",
    }
)
_PROPOSAL_POLICY_REQUIRED_FIELDS = frozenset(
    {"schema_version", "policy_id", "kind", "mutable_fields"}
)
_PARENT_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "policy_id",
        "elite_probability",
        "max_pool_size",
    }
)
_PARENT_POLICY_REQUIRED_FIELDS = frozenset({"schema_version", "policy_id"})
_INLINE_CREDENTIAL_FRAGMENTS = (
    "auth",
    "api_key",
    "bearer",
    "cookie",
    "credential",
    "key",
    "secret",
    "token",
    "password",
    "private_key",
    "access_key",
    "authorization",
)
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class NativeLLMRuntimeError(ValueError):
    pass


def _closed_mapping(
    value: Any,
    *,
    label: str,
    allowed: frozenset[str],
    required: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeLLMRuntimeError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise NativeLLMRuntimeError(f"{label} field names must be strings")
    keys = set(value)
    unknown = sorted(keys - allowed)
    if unknown:
        raise NativeLLMRuntimeError(f"unknown {label} fields: {unknown}")
    missing = sorted(required - keys)
    if missing:
        raise NativeLLMRuntimeError(f"missing {label} fields: {missing}")
    return value


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise NativeLLMRuntimeError(f"{label} must be a non-empty normalized string")
    return value


def _json_clone(value: Any, *, label: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise NativeLLMRuntimeError(f"{label} must be finite JSON data") from exc


def _normalized_field(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _is_environment_reference(field: str) -> bool:
    return field.endswith("_env") or field.endswith("_environment_variable")


def _contains_credential_field(field: str) -> bool:
    return any(
        field == fragment
        or field.startswith(f"{fragment}_")
        or field.endswith(f"_{fragment}")
        or f"_{fragment}_" in field
        for fragment in _INLINE_CREDENTIAL_FRAGMENTS
    )


def _reject_inline_credentials(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise NativeLLMRuntimeError(f"{path} field names must be strings")
            key = _normalized_field(raw_key)
            sensitive = _contains_credential_field(key)
            if sensitive and not _is_environment_reference(key):
                raise NativeLLMRuntimeError(
                    f"inline credential field is forbidden at {path}.{raw_key}"
                )
            if sensitive and _is_environment_reference(key):
                if not isinstance(item, str) or not _ENVIRONMENT_NAME.fullmatch(item):
                    raise NativeLLMRuntimeError(
                        f"credential environment reference at {path}.{raw_key} "
                        "must name an environment variable"
                    )
            if isinstance(item, str) and "://" in item:
                try:
                    parsed = urlsplit(item)
                except ValueError as exc:
                    raise NativeLLMRuntimeError(
                        f"provider URL is malformed at {path}.{raw_key}"
                    ) from exc
                if (
                    parsed.username is not None
                    or parsed.password is not None
                    or bool(parsed.query)
                    or bool(parsed.fragment)
                ):
                    raise NativeLLMRuntimeError(
                        f"inline credential in provider URL is forbidden at "
                        f"{path}.{raw_key}"
                    )
            _reject_inline_credentials(item, path=f"{path}.{raw_key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_inline_credentials(item, path=f"{path}[{index}]")


def _proposal_policy(value: Any) -> LLMProposalPolicy:
    raw = _closed_mapping(
        value,
        label="proposal_policy",
        allowed=_PROPOSAL_POLICY_FIELDS,
        required=_PROPOSAL_POLICY_REQUIRED_FIELDS,
    )
    if raw["schema_version"] != LLM_PROPOSAL_POLICY_SCHEMA_VERSION:
        raise NativeLLMRuntimeError(
            f"unsupported proposal_policy schema: {raw['schema_version']!r}"
        )
    mutable = raw["mutable_fields"]
    if not isinstance(mutable, list):
        raise NativeLLMRuntimeError("proposal_policy mutable_fields must be a list")
    try:
        return LLMProposalPolicy(
            policy_id=raw["policy_id"],
            kind=raw["kind"],
            mutable_fields=tuple(mutable),
            max_context_bytes=raw.get("max_context_bytes", 32_768),
            max_response_bytes=raw.get("max_response_bytes", 262_144),
            schema_version=raw["schema_version"],
        )
    except (TypeError, ValueError) as exc:
        raise NativeLLMRuntimeError("invalid proposal_policy") from exc


def _parent_policy(value: Any) -> ParentSelectionPolicy:
    raw = _closed_mapping(
        value,
        label="parent_selection_policy",
        allowed=_PARENT_POLICY_FIELDS,
        required=_PARENT_POLICY_REQUIRED_FIELDS,
    )
    if raw["schema_version"] != PARENT_SELECTION_POLICY_VERSION:
        raise NativeLLMRuntimeError(
            "unsupported parent_selection_policy schema: " f"{raw['schema_version']!r}"
        )
    try:
        return ParentSelectionPolicy(
            policy_id=raw["policy_id"],
            elite_probability=raw.get("elite_probability", 0.5),
            max_pool_size=raw.get("max_pool_size", 20),
            schema_version=raw["schema_version"],
        )
    except (TypeError, ValueError) as exc:
        raise NativeLLMRuntimeError("invalid parent_selection_policy") from exc


def _initial_revisions(value: Any, *, kind: str) -> Tuple[Revision, ...]:
    if not isinstance(value, list) or not value:
        raise NativeLLMRuntimeError("initial_revisions must be a non-empty list")
    revisions: list[Revision] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise NativeLLMRuntimeError(f"initial_revisions[{index}] must be a mapping")
        revision: Revision
        try:
            if kind == STRATEGY_REVISION:
                if raw.get("schema_version") != "astevolve.strategy.v1":
                    raise NativeLLMRuntimeError(
                        f"initial_revisions[{index}] has an unsupported strategy schema"
                    )
                strategy_revision = DesignStrategy.from_mapping(raw)
                if strategy_revision.to_legacy_dict() != dict(raw):
                    raise NativeLLMRuntimeError(
                        f"initial_revisions[{index}] is not canonical"
                    )
                revision = strategy_revision
            elif kind == DUAL_AST_REVISION:
                ast_revision = ExecutableDualAST.from_mapping(raw)
                if ast_revision.to_dict() != dict(raw):
                    raise NativeLLMRuntimeError(
                        f"initial_revisions[{index}] is not canonical"
                    )
                revision = ast_revision
            else:
                raise NativeLLMRuntimeError(f"unsupported proposal kind: {kind!r}")
        except NativeLLMRuntimeError:
            raise
        except (TypeError, ValueError) as exc:
            raise NativeLLMRuntimeError(
                f"initial_revisions[{index}] is not a valid {kind}"
            ) from exc
        revisions.append(revision)
    return tuple(revisions)


def _initial_parent_ids(value: Any, *, count: int) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise NativeLLMRuntimeError("initial_parent_ids must be a list")
    if value and len(value) != count:
        raise NativeLLMRuntimeError(
            "initial_parent_ids must be empty or align with initial_revisions"
        )
    return tuple(
        _text(item, label=f"initial_parent_ids[{index}]")
        for index, item in enumerate(value)
    )


def _services(
    services: Any,
) -> tuple[ArchiveProjection, EvolutionMemoryProjection]:
    archive = getattr(services, "archive", None)
    memory = getattr(services, "memory", None)
    if not isinstance(archive, ArchiveProjection):
        raise NativeLLMRuntimeError(
            "native LLM factory requires the CLI archive service"
        )
    if not isinstance(memory, EvolutionMemoryProjection):
        raise NativeLLMRuntimeError(
            "native LLM factory requires the CLI memory service"
        )
    return archive, memory


def create_configured_llm_proposal_source(
    payload: Mapping[str, Any], services: "NativeRuntimeServices"
) -> StructuredLLMProposalSource:


    if not isinstance(payload, Mapping):
        raise NativeLLMRuntimeError("native input payload must be a mapping")
    prompt_context = payload.get(LLM_PROMPT_INPUT_CONTEXT_FIELD, {})
    if not isinstance(prompt_context, Mapping):
        raise NativeLLMRuntimeError(
            f"{LLM_PROMPT_INPUT_CONTEXT_FIELD} must be a mapping"
        )
    case_id = payload.get("case_id")
    if isinstance(case_id, str) and case_id.strip() and not prompt_context:
        raise NativeLLMRuntimeError(
            f"case-scoped configured LLM input requires non-empty "
            f"{LLM_PROMPT_INPUT_CONTEXT_FIELD}"
        )
    raw_runtime = payload.get("native_llm_runtime")
    raw = _closed_mapping(
        _json_clone(raw_runtime, label="native_llm_runtime"),
        label="native_llm_runtime",
        allowed=_RUNTIME_FIELDS,
        required=_RUNTIME_REQUIRED_FIELDS,
    )
    if raw["schema_version"] != NATIVE_LLM_RUNTIME_SCHEMA_VERSION:
        raise NativeLLMRuntimeError(
            f"unsupported native_llm_runtime schema: {raw['schema_version']!r}"
        )
    _reject_inline_credentials(raw, path="native_llm_runtime")

    archive, memory = _services(services)
    proposal_policy = _proposal_policy(raw["proposal_policy"])
    parent_policy = _parent_policy(raw["parent_selection_policy"])
    revisions = _initial_revisions(raw["initial_revisions"], kind=proposal_policy.kind)
    parent_ids = _initial_parent_ids(raw["initial_parent_ids"], count=len(revisions))
    provider = _text(raw["provider"], label="provider")
    provider_config = raw["provider_config"]
    if not isinstance(provider_config, Mapping):
        raise NativeLLMRuntimeError("provider_config must be a mapping")
    try:
        model = create_language_model(provider, provider_config)
    except Exception as exc:
        raise NativeLLMRuntimeError(
            f"cannot create language-model provider {provider!r}"
        ) from exc

    selector = ArchiveParentSelector(
        archive=archive,
        initial_revisions=revisions,
        initial_parent_ids=parent_ids,
        policy=parent_policy,
    )

    def supplemental_context(_context: Any) -> Mapping[str, Any]:
        return {
            "schema_version": NATIVE_LLM_SUPPLEMENTAL_CONTEXT_VERSION,
            "memory": memory.snapshot().to_dict(),
        }

    try:
        return StructuredLLMProposalSource(
            model=model,
            parent_selector=selector,
            policy=proposal_policy,
            context_provider=supplemental_context,
            temperature=raw.get("temperature"),
            top_p=raw.get("top_p"),
            max_tokens=raw.get("max_tokens"),
            timeout_seconds=raw.get("timeout_seconds", 300.0),
            max_concurrency=raw.get("max_concurrency", 8),
        )
    except (TypeError, ValueError) as exc:
        raise NativeLLMRuntimeError(
            "invalid native_llm_runtime request settings"
        ) from exc


__all__ = [
    "NATIVE_LLM_RUNTIME_SCHEMA_VERSION",
    "NATIVE_LLM_SUPPLEMENTAL_CONTEXT_VERSION",
    "NativeLLMRuntimeError",
    "create_configured_llm_proposal_source",
]
