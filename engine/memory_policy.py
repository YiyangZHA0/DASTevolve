

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


MEMORY_POLICY_VERSION = "astevolve.memory_policy.v1"
MEMORY_SCOPE_VERSION = "astevolve.memory_scope.v1"
ADAPTIVE_PRIOR_MODES = frozenset({"off", "read_only", "winner_commit"})
INNER_STATE_SCOPES = frozenset({"run", "lineage"})
SCOPE_LEVELS = ("case", "run", "lineage")


class MemoryPolicyError(ValueError):
    pass


def _required_identifier(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise MemoryPolicyError(f"memory scope {field_name} must be non-empty")
    if any(ord(char) < 32 for char in text):
        raise MemoryPolicyError(f"memory scope {field_name} contains a control character")
    return text


@dataclass(frozen=True)
class MemoryScope:


    case_id: str
    run_id: str
    lineage_id: str
    schema_version: str = MEMORY_SCOPE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_SCOPE_VERSION:
            raise MemoryPolicyError(
                f"unsupported memory scope version: {self.schema_version!r}"
            )
        object.__setattr__(self, "case_id", _required_identifier(self.case_id, "case_id"))
        object.__setattr__(self, "run_id", _required_identifier(self.run_id, "run_id"))
        object.__setattr__(
            self, "lineage_id", _required_identifier(self.lineage_id, "lineage_id")
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MemoryScope":
        if not isinstance(value, Mapping):
            raise MemoryPolicyError("memory scope must be a mapping")
        allowed = {"schema_version", "case_id", "run_id", "lineage_id"}
        unknown = sorted(str(key) for key in set(value) - allowed)
        if unknown:
            raise MemoryPolicyError(f"unknown memory scope field(s): {', '.join(unknown)}")
        return cls(
            case_id=str(value.get("case_id") or ""),
            run_id=str(value.get("run_id") or ""),
            lineage_id=str(value.get("lineage_id") or ""),
            schema_version=str(value.get("schema_version") or MEMORY_SCOPE_VERSION),
        )

    def to_artifact(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "run_id": self.run_id,
            "lineage_id": self.lineage_id,
        }

    def require_compatible(self, other: "MemoryScope", *, level: str = "lineage") -> None:


        normalized = str(level or "").strip().lower()
        if normalized not in SCOPE_LEVELS:
            raise MemoryPolicyError(f"unsupported scope level: {level!r}")
        if not isinstance(other, MemoryScope):
            raise MemoryPolicyError("memory scope comparison requires MemoryScope")
        fields = {
            "case": ("case_id",),
            "run": ("case_id", "run_id"),
            "lineage": ("case_id", "run_id", "lineage_id"),
        }[normalized]
        for field_name in fields:
            expected = getattr(self, field_name)
            observed = getattr(other, field_name)
            if expected != observed:
                label = field_name.removesuffix("_id")
                raise MemoryPolicyError(
                    f"memory {label} scope mismatch: expected {expected!r}, found {observed!r}"
                )


@dataclass(frozen=True)
class MemoryPolicyConfig:


    adaptive_prior_mode: str = "off"
    inner_state_scope: str = "run"
    schema_version: str = MEMORY_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_POLICY_VERSION:
            raise MemoryPolicyError(
                f"unsupported memory policy version: {self.schema_version!r}"
            )
        mode = str(self.adaptive_prior_mode or "").strip().lower()
        if mode not in ADAPTIVE_PRIOR_MODES:
            raise MemoryPolicyError(
                f"unsupported adaptive_prior_mode {self.adaptive_prior_mode!r}; "
                f"expected one of {sorted(ADAPTIVE_PRIOR_MODES)}"
            )
        state_scope = str(self.inner_state_scope or "").strip().lower()
        if state_scope not in INNER_STATE_SCOPES:
            raise MemoryPolicyError(
                f"unsupported inner_state_scope {self.inner_state_scope!r}; "
                f"expected one of {sorted(INNER_STATE_SCOPES)}"
            )
        object.__setattr__(self, "adaptive_prior_mode", mode)
        object.__setattr__(self, "inner_state_scope", state_scope)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "MemoryPolicyConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise MemoryPolicyError("memory policy must be a mapping")
        allowed = {"schema_version", "adaptive_prior_mode", "inner_state_scope"}
        unknown = sorted(str(key) for key in set(value) - allowed)
        if unknown:
            raise MemoryPolicyError(f"unknown memory policy field(s): {', '.join(unknown)}")
        return cls(
            adaptive_prior_mode=str(value.get("adaptive_prior_mode") or "off"),
            inner_state_scope=str(value.get("inner_state_scope") or "run"),
            schema_version=str(value.get("schema_version") or MEMORY_POLICY_VERSION),
        )

    @property
    def may_read_adaptive_prior(self) -> bool:
        return self.adaptive_prior_mode in {"read_only", "winner_commit"}

    @property
    def may_commit_adaptive_prior(self) -> bool:
        return self.adaptive_prior_mode == "winner_commit"

    def to_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "adaptive_prior_mode": self.adaptive_prior_mode,
            "inner_state_scope": self.inner_state_scope,
            "may_read_adaptive_prior": self.may_read_adaptive_prior,
            "may_commit_adaptive_prior": self.may_commit_adaptive_prior,
            "controller_locked": True,
            "evolvable": False,
        }


__all__ = [
    "ADAPTIVE_PRIOR_MODES",
    "INNER_STATE_SCOPES",
    "MEMORY_POLICY_VERSION",
    "MEMORY_SCOPE_VERSION",
    "MemoryPolicyConfig",
    "MemoryPolicyError",
    "MemoryScope",
]
