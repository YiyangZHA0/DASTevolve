

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence, TypeVar

from .experiment_identity import EvaluatorDescriptor, ExactEvaluationKey
from .experiment_registry import ExperimentRegistry, SequenceOccurrenceResult
from .history_lifecycle import (
    DuplicateEffectiveContractError,
    HistoryExecutionContext,
    current_history_execution_context,
    current_history_registry_session,
)


PERSISTENT_EVALUATION_CACHE_VERSION = "astevolve.persistent_evaluation_cache.v1"
EFFECTIVE_CONTRACT_ADMISSION_VERSION = "astevolve.contract_admission.v1"

logger = logging.getLogger(__name__)


class ExactEvaluationUnavailableError(RuntimeError):


    def __init__(self, artifact: Mapping[str, Any]) -> None:
        self.artifact = dict(artifact)
        super().__init__(
            "exact evaluation unavailable: "
            f"{self.artifact.get('outcome')}:{self.artifact.get('cache_key')}"
        )


class ExactDescriptorError(ValueError):
    pass


@contextmanager
def _registry_for(context: HistoryExecutionContext):


    registry = current_history_registry_session()
    if registry is not None and getattr(registry, "path", None) == context.registry_path:
        yield registry
        return
    with ExperimentRegistry(
        context.registry_path,
        default_lease_seconds=context.lease_seconds,
    ) as opened:
        yield opened


@contextmanager
def _lease_heartbeat(
    context: HistoryExecutionContext,
    renew: Callable[[ExperimentRegistry], Any],
):


    interval = max(0.01, min(float(context.lease_seconds) / 3.0, 30.0))
    stopped = threading.Event()
    heartbeat_error: list[BaseException] = []

    def run() -> None:
        if stopped.wait(interval):
            return
        try:
            with ExperimentRegistry(
                context.registry_path,
                default_lease_seconds=context.lease_seconds,
            ) as registry:
                while not stopped.is_set():
                    renew(registry)
                    if stopped.wait(interval):
                        return
        except BaseException as exc:
            heartbeat_error.append(exc)

    thread = threading.Thread(
        target=run,
        name="astevolve-registry-heartbeat",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=1.0)
        if heartbeat_error:
            logger.warning(
                "Registry lease heartbeat stopped before completion: %s",
                heartbeat_error[0],
            )


@dataclass(frozen=True)
class EffectiveContractLease:
    context: HistoryExecutionContext
    contract_hash: str
    fencing_token: int
    outcome: str

    def to_artifact(self, *, status: str) -> dict[str, Any]:
        return {
            "schema_version": EFFECTIVE_CONTRACT_ADMISSION_VERSION,
            "contract_hash": self.contract_hash,
            "scope": self.context.scope,
            "owner_token": self.context.owner_token,
            "replicate_policy": self.context.replicate_policy,
            "outcome": self.outcome,
            "status": status,
            "fencing_token": self.fencing_token,
        }


def _class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def canonicalize_exact_value(value: Any, *, path: str = "value") -> Any:


    return _canonicalize_exact_value(value, path=path, active_ids=set())


def _canonicalize_exact_value(
    value: Any,
    *,
    path: str,
    active_ids: set[int],
) -> Any:


    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExactDescriptorError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Path):
        return {"__type__": "path", "value": str(value)}
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _canonicalize_exact_value(
                value.item(), path=path, active_ids=active_ids
            )
        if isinstance(value, np.ndarray):
            return {
                "__type__": "numpy.ndarray",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": _canonicalize_exact_value(
                    value.tolist(), path=f"{path}.data", active_ids=active_ids
                ),
            }
    except ImportError:
        pass
    recursive = (
        isinstance(value, (Mapping, list, tuple, set, frozenset))
        or (is_dataclass(value) and not isinstance(value, type))
        or callable(getattr(value, "to_dict", None))
        or isinstance(getattr(value, "__dict__", None), Mapping)
    )
    identity = id(value)
    if recursive:
        if identity in active_ids:
            raise ExactDescriptorError(f"{path} contains an object cycle")
        active_ids.add(identity)
    try:
        if isinstance(value, Mapping):
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                text = str(key)
                if text in normalized:
                    raise ExactDescriptorError(f"{path} has colliding mapping keys")
                normalized[text] = _canonicalize_exact_value(
                    item, path=f"{path}.{text}", active_ids=active_ids
                )
            return normalized
        if isinstance(value, (list, tuple)):
            return [
                _canonicalize_exact_value(
                    item, path=f"{path}[{index}]", active_ids=active_ids
                )
                for index, item in enumerate(value)
            ]
        if isinstance(value, (set, frozenset)):
            normalized = [
                _canonicalize_exact_value(
                    item, path=f"{path}[]", active_ids=active_ids
                )
                for item in value
            ]
            from .causal_flow import canonical_json

            return sorted(normalized, key=canonical_json)
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "__type__": _class_name(value),
                "fields": {
                    field.name: _canonicalize_exact_value(
                        getattr(value, field.name),
                        path=f"{path}.{field.name}",
                        active_ids=active_ids,
                    )
                    for field in fields(value)
                },
            }
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return {
                "__type__": _class_name(value),
                "value": _canonicalize_exact_value(
                    to_dict(), path=f"{path}.to_dict", active_ids=active_ids
                ),
            }
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, Mapping):
            return {
                "__type__": _class_name(value),
                "attributes": _canonicalize_exact_value(
                    dict(attributes),
                    path=f"{path}.__dict__",
                    active_ids=active_ids,
                ),
            }
    finally:
        if recursive:
            active_ids.remove(identity)
    raise ExactDescriptorError(f"{path} contains opaque {_class_name(value)}")


def claim_effective_contract(
    contract_hash: str,
) -> Optional[EffectiveContractLease]:


    context = current_history_execution_context()
    if context is None or context.allows_contract_replicate:
        return None
    with _registry_for(context) as registry:
        claim = registry.claim_effective_contract(
            contract_hash,
            owner_token=context.owner_token,
            scope=context.scope,
            lease_seconds=context.lease_seconds,
            retry_failed=context.retry_failed,
        )
    if not claim.acquired:
        raise DuplicateEffectiveContractError(
            {
                "schema_version": EFFECTIVE_CONTRACT_ADMISSION_VERSION,
                "contract_hash": contract_hash,
                "scope": context.scope,
                "owner_token": context.owner_token,
                "replicate_policy": context.replicate_policy,
                "outcome": claim.outcome,
                "status": claim.status,
                "fencing_token": claim.fencing_token,
                "attempt_count": claim.attempt_count,
                "cached_error": claim.cached_error,
            }
        )
    return EffectiveContractLease(
        context=context,
        contract_hash=contract_hash,
        fencing_token=claim.fencing_token,
        outcome=claim.outcome,
    )


def complete_effective_contract(lease: Optional[EffectiveContractLease]) -> None:
    if lease is None:
        return
    with _registry_for(lease.context) as registry:
        registry.complete_effective_contract(
            lease.contract_hash,
            owner_token=lease.context.owner_token,
            fencing_token=lease.fencing_token,
            scope=lease.context.scope,
        )


def fail_effective_contract(
    lease: Optional[EffectiveContractLease], exc: BaseException
) -> None:
    if lease is None:
        return
    with _registry_for(lease.context) as registry:
        registry.fail_effective_contract(
            lease.contract_hash,
            owner_token=lease.context.owner_token,
            fencing_token=lease.fencing_token,
            error={"type": type(exc).__name__, "message": str(exc)},
            scope=lease.context.scope,
        )


@contextmanager
def maintain_effective_contract_lease(
    lease: Optional[EffectiveContractLease],
):


    if lease is None:
        yield
        return
    with _lease_heartbeat(
        lease.context,
        lambda registry: registry.renew_effective_contract_lease(
            lease.contract_hash,
            owner_token=lease.context.owner_token,
            fencing_token=lease.fencing_token,
            scope=lease.context.scope,
            lease_seconds=lease.context.lease_seconds,
        ),
    ):
        yield


def register_sequence_occurrence(
    sequence: Mapping[str, str],
    *,
    role: str,
    context_id: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[SequenceOccurrenceResult]:


    context = current_history_execution_context()
    if context is None:
        return None
    resolved_context_id = f"{context.owner_token}:{str(context_id)}"
    with _registry_for(context) as registry:
        return registry.register_sequence_occurrence(
            sequence,
            role=role,
            context_id=resolved_context_id,
            scope=context.scope,
            metadata=metadata,
        )


def registry_metrics_artifact() -> Optional[dict[str, Any]]:
    context = current_history_execution_context()
    if context is None:
        return None
    with _registry_for(context) as registry:
        return registry.metrics(scope=context.scope).to_dict()


T = TypeVar("T")


def _cache_artifact(
    *,
    enabled: bool,
    key: Optional[ExactEvaluationKey],
    cache_hit: bool,
    evaluation_invoked: bool,
    outcome: str,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "schema_version": PERSISTENT_EVALUATION_CACHE_VERSION,
        "enabled": enabled,
        "exact": bool(enabled and key is not None),
        "cache_key": key.cache_key if key is not None else None,
        "evaluator_descriptor_hash": (
            key.evaluator_descriptor_hash if key is not None else None
        ),
        "cache_hit": cache_hit,
        "evaluation_invoked": evaluation_invoked,
        "outcome": outcome,
        "reason": reason,
    }


def evaluate_exact_persistently(
    sequence: Mapping[str, str],
    *,
    tool: str,
    tool_version: str,
    model: str,
    config: Any,
    state: Any,
    seed: Optional[int],
    compute: Callable[[], T],
    estimated_cost: float = 0.0,
    retry_failed: Optional[bool] = None,
    cache_scope: Optional[str] = None,
) -> tuple[T, dict[str, Any]]:


    context = current_history_execution_context()
    if context is None:
        return compute(), _cache_artifact(
            enabled=False,
            key=None,
            cache_hit=False,
            evaluation_invoked=True,
            outcome="disabled",
            reason="history_context_absent",
        )
    resolved_scope = context.scope if cache_scope is None else str(cache_scope).strip()
    if not resolved_scope:
        raise ValueError("cache_scope must be non-empty when supplied")
    try:
        descriptor = EvaluatorDescriptor.create(
            tool=tool,
            tool_version=tool_version,
            model=model,
            config=canonicalize_exact_value(config, path="config"),
            state=canonicalize_exact_value(state, path="state"),
            seed=seed,
        )
        key = ExactEvaluationKey.create(sequence, descriptor)
    except (ExactDescriptorError, TypeError, ValueError) as exc:
        return compute(), _cache_artifact(
            enabled=False,
            key=None,
            cache_hit=False,
            evaluation_invoked=True,
            outcome="bypassed",
            reason=f"non_exact_descriptor:{type(exc).__name__}:{exc}",
        )

    with _registry_for(context) as registry:
        claim = registry.claim_evaluation(
            key,
            owner_token=context.owner_token,
            scope=resolved_scope,
            lease_seconds=context.lease_seconds,
            retry_failed=(
                context.retry_failed
                if retry_failed is None
                else bool(retry_failed)
            ),
            estimated_cost=estimated_cost,
        )
    if claim.acquired:
        with _lease_heartbeat(
            context,
            lambda registry: registry.renew_evaluation_lease(
                key,
                owner_token=context.owner_token,
                fencing_token=claim.fencing_token,
                scope=resolved_scope,
                lease_seconds=context.lease_seconds,
            ),
        ):
            try:
                result = compute()
                stored = canonicalize_exact_value(result, path="result")
                with _registry_for(context) as registry:
                    registry.complete_evaluation(
                        key,
                        owner_token=context.owner_token,
                        fencing_token=claim.fencing_token,
                        result=stored,
                        scope=resolved_scope,
                    )
            except Exception as exc:
                with _registry_for(context) as registry:
                    registry.fail_evaluation(
                        key,
                        owner_token=context.owner_token,
                        fencing_token=claim.fencing_token,
                        error={"type": type(exc).__name__, "message": str(exc)},
                        scope=resolved_scope,
                    )
                raise
        return stored, _cache_artifact(
            enabled=True,
            key=key,
            cache_hit=False,
            evaluation_invoked=True,
            outcome=claim.outcome,
        )

    if claim.status == "completed":
        return claim.cached_result, _cache_artifact(
            enabled=True,
            key=key,
            cache_hit=True,
            evaluation_invoked=False,
            outcome=claim.outcome,
        )


    if claim.status == "pending":
        deadline = time.monotonic() + min(max(context.lease_seconds, 1.0), 30.0)
        poll_interval = 0.005
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            time.sleep(min(poll_interval, max(0.0, remaining)))
            poll_interval = min(poll_interval * 1.7, 0.25)
            with _registry_for(context) as registry:
                entry = registry.lookup_evaluation(key, scope=resolved_scope)
            if entry is not None and entry.status == "completed":
                return entry.result, _cache_artifact(
                    enabled=True,
                    key=key,
                    cache_hit=True,
                    evaluation_invoked=False,
                    outcome="singleflight_wait_hit",
                )
            if entry is not None and entry.status == "failed":
                break
    artifact = _cache_artifact(
        enabled=True,
        key=key,
        cache_hit=False,
        evaluation_invoked=False,
        outcome=claim.outcome,
        reason=f"exact_entry_{claim.status}",
    )
    raise ExactEvaluationUnavailableError(artifact)


__all__ = [
    "EFFECTIVE_CONTRACT_ADMISSION_VERSION",
    "PERSISTENT_EVALUATION_CACHE_VERSION",
    "EffectiveContractLease",
    "ExactDescriptorError",
    "ExactEvaluationUnavailableError",
    "canonicalize_exact_value",
    "claim_effective_contract",
    "complete_effective_contract",
    "evaluate_exact_persistently",
    "fail_effective_contract",
    "maintain_effective_contract_lease",
    "register_sequence_occurrence",
    "registry_metrics_artifact",
]
