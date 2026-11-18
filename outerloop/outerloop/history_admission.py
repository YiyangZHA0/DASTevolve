

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from engine.experiment_identity import CodeIdentity
from engine.experiment_registry import ExperimentRegistry


class CodeAdmissionError(RuntimeError):


    def __init__(self, code: str, detail: str = "", *, artifact: Any = None) -> None:
        self.code = str(code)
        self.detail = str(detail)
        self.artifact = artifact
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "artifact": self.artifact,
        }


@dataclass(frozen=True)
class CodeAdmission:
    registry_path: str
    scope: str
    owner_token: str
    identity: CodeIdentity
    fencing_token: int
    outcome: str

    def to_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": "astevolve.code_admission.v1",
            "scope": self.scope,
            "owner_token": self.owner_token,
            "identity": self.identity.to_dict(),
            "fencing_token": self.fencing_token,
            "outcome": self.outcome,
        }

    def complete(self) -> None:
        with ExperimentRegistry(self.registry_path) as registry:
            registry.complete_code(
                self.identity,
                owner_token=self.owner_token,
                fencing_token=self.fencing_token,
                scope=self.scope,
            )

    def fail(self, error: Any) -> None:
        with ExperimentRegistry(self.registry_path) as registry:
            registry.fail_code(
                self.identity,
                owner_token=self.owner_token,
                fencing_token=self.fencing_token,
                error=_error_artifact(error),
                scope=self.scope,
            )


def _error_artifact(error: Any) -> Any:
    if isinstance(error, CodeAdmissionError):
        return error.to_dict()
    if isinstance(error, BaseException):
        return {"type": type(error).__name__, "message": str(error)}
    if isinstance(error, Mapping):
        return dict(error)
    return {"message": str(error)}


def claim_code_for_evaluation(
    source: str,
    database_config: Any,
    *,
    owner_token: str,
) -> Optional[CodeAdmission]:


    enabled = getattr(database_config, "experiment_registry_enabled", None)
    if enabled is None or enabled is False:
        return None
    if enabled is not True:
        raise CodeAdmissionError("experiment_registry_enabled_invalid", repr(enabled))
    path = getattr(database_config, "experiment_registry_path", None)
    if not isinstance(path, str) or not path.strip():
        raise CodeAdmissionError("experiment_registry_path_required")
    scope = getattr(database_config, "experiment_registry_scope", None) or "default"
    lease_seconds = getattr(
        database_config, "experiment_registry_lease_seconds", 3600.0
    )
    retry_failed = getattr(
        database_config, "experiment_registry_retry_failed", False
    )
    identity = CodeIdentity.from_text(source)
    with ExperimentRegistry(path) as registry:
        claim = registry.claim_code(
            identity,
            owner_token=owner_token,
            scope=scope,
            lease_seconds=lease_seconds,
            retry_failed=retry_failed,
        )
    if not claim.acquired:
        artifact = {
            "schema_version": "astevolve.code_admission_rejection.v1",
            "scope": scope,
            "identity": identity.to_dict(),
            "outcome": claim.outcome,
            "status": claim.status,
            "attempt_count": claim.attempt_count,
            "cached_error": claim.cached_error,
        }
        raise CodeAdmissionError(
            "duplicate_code_proposal",
            claim.outcome,
            artifact=artifact,
        )
    return CodeAdmission(
        registry_path=path,
        scope=str(scope),
        owner_token=str(owner_token),
        identity=identity,
        fencing_token=claim.fencing_token,
        outcome=claim.outcome,
    )


__all__ = [
    "CodeAdmission",
    "CodeAdmissionError",
    "claim_code_for_evaluation",
]
