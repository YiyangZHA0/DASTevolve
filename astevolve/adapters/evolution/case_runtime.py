

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from astevolve.adapters.legacy.design_search import LegacyDesignSearchRunner
from astevolve.application.ports.runner import DesignSearchRunner
from astevolve.cases import resolve_case
from astevolve.domain import RunContext
from astevolve.evolution.domain import SealedEvaluation
from astevolve.evolution.orchestrator import EvaluationTask
from engine.memory_lifecycle import (
    MemorySnapshot,
    capture_memory_snapshot,
    memory_execution_scope,
)
from engine.memory_policy import MemoryPolicyConfig, MemoryScope
from engine.history_lifecycle import history_execution_scope

from .native_runtime import DesignSearchProposalEvaluator

if TYPE_CHECKING:
    from astevolve.evolution.cli import NativeRuntimeServices


NATIVE_CASE_RUNTIME_SCHEMA_VERSION = "astevolve.native_case_runtime.v1"
_RUNTIME_FIELDS = {
    "schema_version",
    "output_root",
    "history_registry_path",
    "memory_policy",
}


class NativeCaseRuntimeError(ValueError):
    pass


def _safe_component(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"proposal-{digest}"


@dataclass(frozen=True)
class NativeCaseProposalEvaluator:


    case_id: str
    run_id: str
    project_root: Path
    design_state_path: Path
    design_state_hash: str
    output_root: Path
    history_registry_path: Path
    runner: DesignSearchRunner
    memory_snapshot: MemorySnapshot
    memory_target_path: Path
    memory_policy: MemoryPolicyConfig = MemoryPolicyConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise NativeCaseRuntimeError("case_id must be non-empty")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise NativeCaseRuntimeError("run_id must be non-empty")
        if not callable(getattr(self.runner, "run", None)):
            raise TypeError("runner must implement run(strategy, context)")
        if not isinstance(self.memory_snapshot, MemorySnapshot):
            raise TypeError("memory_snapshot must be MemorySnapshot")
        if not isinstance(self.memory_policy, MemoryPolicyConfig):
            raise TypeError("memory_policy must be MemoryPolicyConfig")
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "project_root", Path(self.project_root))
        object.__setattr__(self, "design_state_path", Path(self.design_state_path))
        object.__setattr__(
            self, "history_registry_path", Path(self.history_registry_path)
        )
        object.__setattr__(
            self, "memory_target_path", Path(self.memory_target_path)
        )

    def evaluate(self, task: EvaluationTask) -> SealedEvaluation:
        if not isinstance(task, EvaluationTask):
            raise TypeError("task must be EvaluationTask")
        try:
            current_design_hash = hashlib.sha256(
                self.design_state_path.read_bytes()
            ).hexdigest()
        except OSError as exc:
            raise NativeCaseRuntimeError("cannot read frozen case design state") from exc
        if current_design_hash != self.design_state_hash:
            raise NativeCaseRuntimeError(
                "case design state changed after native evaluator construction"
            )
        proposal_output = (
            self.output_root
            / f"generation-{task.generation_index:08d}"
            / _safe_component(task.proposal.proposal_id)
        )
        context = RunContext(
            case_id=self.case_id,
            project_root=self.project_root,
            output_root=proposal_output,
            design_state_path=self.design_state_path,
            memory_path=self.memory_target_path,
            seed=task.seed,
            settings={
                "generation_input_hash": task.generation_input.input_hash,
                "proposal_hash": task.proposal.proposal_hash,
            },
            run_id=task.run_id,
        )
        scope = MemoryScope(
            case_id=self.case_id,
            run_id=self.run_id,
            lineage_id=task.input_snapshot.snapshot_hash,
        )
        with memory_execution_scope(
            generation_id=task.generation_id,
            proposal_id=task.proposal.proposal_id,
            trial_id=f"seed-{task.seed}",
            scope_id=(
                f"{task.generation_id}/{task.proposal.proposal_id}/seed-{task.seed}"
            ),
            logical_time=f"generation:{task.generation_index}:slot:{task.slot}",
            commit_mode="deferred",
            snapshot=self.memory_snapshot,
            target_path=self.memory_target_path,
            output_dir=proposal_output,
            memory_scope=scope,
            memory_policy=self.memory_policy,
            history_registry_path=self.history_registry_path,
            history_scope=(
                f"native:{self.run_id}:{self.case_id}:"
                f"{task.input_snapshot.snapshot_hash[:16]}"
            ),
            history_owner_token=f"{task.proposal.proposal_id}:seed-{task.seed}",


            history_replicate_policy="allow",
        ) as installed_scope:
            history_context = installed_scope.history_execution_context()
            if history_context is None:
                raise NativeCaseRuntimeError(
                    "native case evaluator did not construct a history scope"
                )


            with history_execution_scope(context=history_context):
                return DesignSearchProposalEvaluator(
                    runner=self.runner,
                    context_factory=lambda _task: context,
                ).evaluate(task)


def create_case_design_evaluator(
    payload: Mapping[str, Any], services: NativeRuntimeServices
) -> NativeCaseProposalEvaluator:


    if not isinstance(payload, Mapping):
        raise NativeCaseRuntimeError("native input payload must be a mapping")
    case_id = payload.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise NativeCaseRuntimeError(
            "native case evaluator requires a non-empty input case_id"
        )
    raw_runtime = payload.get("native_case_runtime", {})
    if not isinstance(raw_runtime, Mapping):
        raise NativeCaseRuntimeError("native_case_runtime must be a mapping")
    unknown = sorted(set(raw_runtime) - _RUNTIME_FIELDS)
    if unknown:
        raise NativeCaseRuntimeError(
            f"unknown native_case_runtime fields: {unknown}"
        )
    schema = raw_runtime.get(
        "schema_version", NATIVE_CASE_RUNTIME_SCHEMA_VERSION
    )
    if schema != NATIVE_CASE_RUNTIME_SCHEMA_VERSION:
        raise NativeCaseRuntimeError(
            f"unsupported native_case_runtime schema: {schema!r}"
        )
    case = resolve_case(
            case_id.strip(),
            case_root=payload.get("case_root"),
            manifest_path=payload.get("case_manifest") or payload.get("manifest_path"),
        )
    output_root = Path(
        raw_runtime.get("output_root")
        or (
            case.output_root
            / "native_evolution"
            / (
                f"{services.run_config.run_id}-"
                f"{services.input_snapshot.snapshot_hash[:16]}"
            )
        )
    )
    history_path = Path(
        raw_runtime.get("history_registry_path")
        or (output_root / "experiment_registry.sqlite")
    )
    memory_policy = MemoryPolicyConfig.from_mapping(
        raw_runtime.get("memory_policy")
    )
    if memory_policy.may_commit_adaptive_prior:


        raise NativeCaseRuntimeError(
            "native case runtime does not yet support adaptive_prior_mode="
            "winner_commit; use off/read_only or the legacy generation "
            "coordinator"
        )
    return NativeCaseProposalEvaluator(
        case_id=case.case_id,
        run_id=services.run_config.run_id,
        project_root=case.root,
        design_state_path=case.design_state_path,
        design_state_hash=hashlib.sha256(
            case.design_state_path.read_bytes()
        ).hexdigest(),
        output_root=output_root,
        history_registry_path=history_path,
        runner=LegacyDesignSearchRunner(),
        memory_snapshot=capture_memory_snapshot(case.memory_path),
        memory_target_path=case.memory_path,
        memory_policy=memory_policy,
    )


__all__ = [
    "NATIVE_CASE_RUNTIME_SCHEMA_VERSION",
    "NativeCaseProposalEvaluator",
    "NativeCaseRuntimeError",
    "create_case_design_evaluator",
]
