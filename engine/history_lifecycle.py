

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import math
import os
from pathlib import Path
import threading
from typing import Any, Iterator, Mapping, Optional


HISTORY_EXECUTION_CONTEXT_VERSION = "astevolve.history_execution_context.v1"
HISTORY_REPLICATE_POLICIES = frozenset({"reject", "allow", "retry_failed"})


class HistoryLifecycleError(ValueError):
    pass


class DuplicateEffectiveContractError(RuntimeError):


    def __init__(self, artifact: Mapping[str, Any]) -> None:
        self.artifact = dict(artifact)
        outcome = str(self.artifact.get("outcome") or "duplicate")
        contract_hash = str(self.artifact.get("contract_hash") or "")
        super().__init__(f"effective contract rejected: {outcome}:{contract_hash}")


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise HistoryLifecycleError(f"{field} must be non-empty")
    return text


@dataclass(frozen=True)
class HistoryExecutionContext:


    registry_path: str
    scope: str
    owner_token: str
    lease_seconds: float = 300.0
    replicate_policy: str = "reject"
    schema_version: str = HISTORY_EXECUTION_CONTEXT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != HISTORY_EXECUTION_CONTEXT_VERSION:
            raise HistoryLifecycleError(
                f"unsupported history context version: {self.schema_version!r}"
            )
        path = _required_text(self.registry_path, "registry_path")
        if path == ":memory:":
            raise HistoryLifecycleError(
                "persistent history context requires a file-backed registry"
            )
        _required_text(self.scope, "scope")
        _required_text(self.owner_token, "owner_token")
        if (
            isinstance(self.lease_seconds, bool)
            or not isinstance(self.lease_seconds, (int, float))
            or not math.isfinite(float(self.lease_seconds))
            or float(self.lease_seconds) <= 0
        ):
            raise HistoryLifecycleError("lease_seconds must be a finite positive number")
        policy = str(self.replicate_policy or "").strip().lower()
        if policy not in HISTORY_REPLICATE_POLICIES:
            raise HistoryLifecycleError(
                f"replicate_policy must be one of {sorted(HISTORY_REPLICATE_POLICIES)}"
            )

        object.__setattr__(self, "registry_path", str(Path(path)))
        object.__setattr__(self, "scope", str(self.scope).strip())
        object.__setattr__(self, "owner_token", str(self.owner_token).strip())
        object.__setattr__(self, "lease_seconds", float(self.lease_seconds))
        object.__setattr__(self, "replicate_policy", policy)

    @property
    def retry_failed(self) -> bool:
        return self.replicate_policy == "retry_failed"

    @property
    def allows_contract_replicate(self) -> bool:
        return self.replicate_policy == "allow"

    def to_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_path": self.registry_path,
            "scope": self.scope,
            "owner_token": self.owner_token,
            "lease_seconds": self.lease_seconds,
            "replicate_policy": self.replicate_policy,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HistoryExecutionContext":
        if not isinstance(value, Mapping):
            raise HistoryLifecycleError("history context must be a mapping")
        fields = {
            "schema_version",
            "registry_path",
            "scope",
            "owner_token",
            "lease_seconds",
            "replicate_policy",
        }
        unknown = sorted(str(key) for key in set(value) - fields)
        missing = sorted(str(key) for key in fields - set(value))
        if unknown:
            raise HistoryLifecycleError(f"unknown history context fields: {unknown}")
        if missing:
            raise HistoryLifecycleError(f"missing history context fields: {missing}")
        return cls(
            registry_path=value.get("registry_path"),
            scope=value.get("scope"),
            owner_token=value.get("owner_token"),
            lease_seconds=value.get("lease_seconds"),
            replicate_policy=value.get("replicate_policy"),
            schema_version=value.get("schema_version"),
        )


_HISTORY_EXECUTION_CONTEXT: ContextVar[Optional[HistoryExecutionContext]] = (
    ContextVar("astevolve_history_execution_context", default=None)
)
_HISTORY_REGISTRY_SESSION: ContextVar[Optional[Any]] = ContextVar(
    "astevolve_history_registry_session", default=None
)
_HISTORY_REGISTRY_SESSION_OWNER: ContextVar[Optional[tuple[int, int]]] = ContextVar(
    "astevolve_history_registry_session_owner", default=None
)


def current_history_execution_context() -> Optional[HistoryExecutionContext]:


    return _HISTORY_EXECUTION_CONTEXT.get()


def current_history_registry_session() -> Optional[Any]:


    registry = _HISTORY_REGISTRY_SESSION.get()
    owner = _HISTORY_REGISTRY_SESSION_OWNER.get()
    if owner != (os.getpid(), threading.get_ident()):


        return None
    return registry


@contextmanager
def history_execution_scope(
    *,
    context: Optional[HistoryExecutionContext] = None,
    registry_path: str | Path = "",
    scope: str = "default",
    owner_token: str = "",
    lease_seconds: float = 300.0,
    replicate_policy: str = "reject",
) -> Iterator[HistoryExecutionContext]:


    if context is not None and any(
        (
            str(registry_path),
            str(owner_token),
            scope != "default",
            lease_seconds != 300.0,
            replicate_policy != "reject",
        )
    ):
        raise HistoryLifecycleError(
            "pass either context or individual history context fields, not both"
        )
    resolved = context or HistoryExecutionContext(
        registry_path=str(registry_path),
        scope=scope,
        owner_token=owner_token,
        lease_seconds=lease_seconds,
        replicate_policy=replicate_policy,
    )
    if not isinstance(resolved, HistoryExecutionContext):
        raise TypeError("context must be a HistoryExecutionContext")
    token: Token[Optional[HistoryExecutionContext]] = (
        _HISTORY_EXECUTION_CONTEXT.set(resolved)
    )
    active_registry = current_history_registry_session()
    if (
        active_registry is not None
        and getattr(active_registry, "path", None) == resolved.registry_path
    ):
        try:
            yield resolved
        finally:
            _HISTORY_EXECUTION_CONTEXT.reset(token)
        return


    from .experiment_registry import ExperimentRegistry

    registry = ExperimentRegistry(
        resolved.registry_path,
        default_lease_seconds=resolved.lease_seconds,
    )
    registry_token: Token[Optional[Any]] = _HISTORY_REGISTRY_SESSION.set(registry)
    owner_state_token = _HISTORY_REGISTRY_SESSION_OWNER.set(
        (os.getpid(), threading.get_ident())
    )
    try:
        yield resolved
    finally:
        _HISTORY_REGISTRY_SESSION_OWNER.reset(owner_state_token)
        _HISTORY_REGISTRY_SESSION.reset(registry_token)
        registry.close()
        _HISTORY_EXECUTION_CONTEXT.reset(token)


__all__ = [
    "HISTORY_EXECUTION_CONTEXT_VERSION",
    "HISTORY_REPLICATE_POLICIES",
    "DuplicateEffectiveContractError",
    "HistoryExecutionContext",
    "HistoryLifecycleError",
    "current_history_execution_context",
    "current_history_registry_session",
    "history_execution_scope",
]
