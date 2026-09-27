

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
import threading
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Tuple, Union

from astevolve.domain import DesignStrategy
from astevolve.domain.dual_ast import ExecutableDualAST

from .application import GenerationEngine
from .domain import (
    DUAL_AST_REVISION,
    STRATEGY_REVISION,
    GenerationCommit,
    GenerationManifest,
    Proposal,
    SealedEvaluation,
)
from .execution_identity import (
    NATIVE_EXECUTION_CONTRACT_KEY,
    NATIVE_EXECUTION_CONTRACT_VERSION,
    python_code_identity,
    seal_execution_contract,
    validate_execution_contract,
)
from .persistence import FileGenerationLedger, PublishedGeneration


INPUT_SNAPSHOT_SCHEMA_VERSION = "astevolve.evolution.input_snapshot.v1"
GENERATION_INPUT_SCHEMA_VERSION = "astevolve.evolution.generation_input.v1"
_GENERATION_SUFFIX = re.compile(r"^[0-9]{8}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EvolutionOrchestrationError(RuntimeError):
    pass


class EvolutionRecoveryError(EvolutionOrchestrationError):
    pass


class GenerationIndexError(EvolutionOrchestrationError):
    pass


class ProposalSourceError(EvolutionOrchestrationError):
    pass


class EvaluationExecutionError(EvolutionOrchestrationError):
    pass


class ProjectionApplicationError(EvolutionOrchestrationError):
    pass


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty normalized string")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _non_negative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_mapping(
    value: Mapping[str, Any], name: str
) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        initial = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        decoded = json.loads(initial)
        if not isinstance(decoded, dict):
            raise TypeError(f"{name} must be a JSON object")
        canonical = json.dumps(
            decoded,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must contain finite JSON-compatible data") from exc
    return canonical, decoded


def _snapshot_hash(payload_json: str) -> str:
    return hashlib.sha256(
        f"{INPUT_SNAPSHOT_SCHEMA_VERSION}\0{payload_json}".encode("utf-8")
    ).hexdigest()


def _private_seed(
    *, root_seed: int, run_id: str, generation_index: int, slot: int, purpose: str
) -> int:
    payload = (
        f"astevolve.evolution.private_seed.v1\0{root_seed}\0{run_id}\0"
        f"{generation_index}\0{slot}\0{purpose}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class EvolutionInputSnapshot:


    payload_json: str
    snapshot_hash: str
    schema_version: str = INPUT_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.verify()

    @classmethod
    def freeze(cls, payload: Mapping[str, Any]) -> "EvolutionInputSnapshot":
        payload_json, _ = _canonical_mapping(payload, "evolution input")
        return cls(
            payload_json=payload_json,
            snapshot_hash=_snapshot_hash(payload_json),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvolutionInputSnapshot":
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "payload",
            "snapshot_hash",
        }:
            raise ValueError("input snapshot has invalid fields")
        payload_json, _ = _canonical_mapping(value["payload"], "evolution input")
        return cls(
            payload_json=payload_json,
            snapshot_hash=value["snapshot_hash"],
            schema_version=value["schema_version"],
        )

    def verify(self) -> None:
        if self.schema_version != INPUT_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported input snapshot schema: {self.schema_version!r}"
            )
        try:
            payload = json.loads(self.payload_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("input snapshot payload is invalid JSON") from exc
        canonical, _ = _canonical_mapping(payload, "evolution input")
        if canonical != self.payload_json:
            raise ValueError("input snapshot payload is not canonical JSON")
        if self.snapshot_hash != _snapshot_hash(canonical):
            raise ValueError("input snapshot hash mismatch")

    def payload(self) -> dict[str, Any]:


        self.verify()
        value = json.loads(self.payload_json)
        assert isinstance(value, dict)
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "payload": self.payload(),
            "snapshot_hash": self.snapshot_hash,
        }


@dataclass(frozen=True)
class GenerationInput:


    generation_index: int
    initial_snapshot_hash: str
    prior_commit_hashes: Tuple[str, ...]
    input_hash: str
    schema_version: str = GENERATION_INPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.verify()

    @classmethod
    def create(
        cls,
        *,
        generation_index: int,
        initial_snapshot_hash: str,
        prior_commit_hashes: Iterable[str],
    ) -> "GenerationInput":
        index = _non_negative_int(generation_index, "generation_index")
        initial = _sha256(initial_snapshot_hash, "initial_snapshot_hash")
        prior = tuple(
            _sha256(value, f"prior_commit_hashes[{offset}]")
            for offset, value in enumerate(prior_commit_hashes)
        )
        return cls(
            generation_index=index,
            initial_snapshot_hash=initial,
            prior_commit_hashes=prior,
            input_hash=cls._expected_hash(index, initial, prior),
        )

    @staticmethod
    def _expected_hash(index: int, initial: str, prior: Tuple[str, ...]) -> str:
        core = {
            "schema_version": GENERATION_INPUT_SCHEMA_VERSION,
            "generation_index": index,
            "initial_snapshot_hash": initial,
            "prior_commit_hashes": list(prior),
        }
        payload = json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(
            f"{GENERATION_INPUT_SCHEMA_VERSION}\0{payload}".encode("utf-8")
        ).hexdigest()

    def verify(self) -> None:
        if self.schema_version != GENERATION_INPUT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported generation input schema: {self.schema_version!r}"
            )
        index = _non_negative_int(self.generation_index, "generation_index")
        if not isinstance(self.prior_commit_hashes, tuple):
            raise ValueError("prior_commit_hashes must be a tuple")
        if len(self.prior_commit_hashes) != index:
            raise ValueError(
                "prior_commit_hashes must contain exactly one hash per prior generation"
            )
        initial = _sha256(self.initial_snapshot_hash, "initial_snapshot_hash")
        prior = tuple(
            _sha256(value, f"prior_commit_hashes[{offset}]")
            for offset, value in enumerate(self.prior_commit_hashes)
        )
        expected_hash = self._expected_hash(index, initial, prior)
        if _sha256(self.input_hash, "input_hash") != expected_hash:
            raise ValueError("generation input hash mismatch")

    def to_dict(self) -> dict[str, Any]:
        self.verify()
        return {
            "schema_version": self.schema_version,
            "generation_index": self.generation_index,
            "initial_snapshot_hash": self.initial_snapshot_hash,
            "prior_commit_hashes": list(self.prior_commit_hashes),
            "input_hash": self.input_hash,
        }


Revision = Union[ExecutableDualAST, DesignStrategy]


@dataclass(frozen=True)
class ProposalDraft:


    parent_id: str
    revision: Revision
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required_text(self.parent_id, "parent_id")
        revision = self.revision
        if isinstance(revision, ExecutableDualAST):
            detached: Revision = ExecutableDualAST.from_mapping(revision.to_dict())
        elif isinstance(revision, DesignStrategy):
            detached = DesignStrategy.from_mapping(revision.to_legacy_dict())
        else:
            raise TypeError("revision must be ExecutableDualAST or DesignStrategy")
        _, detached_provenance = _canonical_mapping(
            self.provenance, "proposal provenance"
        )
        object.__setattr__(self, "revision", detached)
        object.__setattr__(self, "provenance", detached_provenance)

    @property
    def kind(self) -> str:
        if isinstance(self.revision, ExecutableDualAST):
            return DUAL_AST_REVISION
        return STRATEGY_REVISION

    def seal(self, *, generation_id: str, slot: int) -> Proposal:
        if isinstance(self.revision, ExecutableDualAST):
            return Proposal.from_dual_ast(
                generation_id=generation_id,
                slot=slot,
                parent_id=self.parent_id,
                revision=self.revision,
                provenance=self.provenance,
            )
        return Proposal.from_strategy(
            generation_id=generation_id,
            slot=slot,
            parent_id=self.parent_id,
            revision=self.revision,
            provenance=self.provenance,
        )


@dataclass(frozen=True)
class ProposalContext:
    run_id: str
    generation_index: int
    generation_id: str
    slot: int
    generation_seed: int
    slot_seed: int
    input_snapshot: EvolutionInputSnapshot
    generation_input: GenerationInput
    prior_commits: Tuple[GenerationCommit, ...]
    prior_manifests: Tuple[GenerationManifest, ...] = ()

    def __post_init__(self) -> None:
        _non_negative_int(self.slot, "slot")
        _non_negative_int(self.generation_seed, "generation_seed")
        _non_negative_int(self.slot_seed, "slot_seed")
        _validated_generation_view(
            run_id=self.run_id,
            generation_index=self.generation_index,
            generation_id=self.generation_id,
            input_snapshot=self.input_snapshot,
            generation_input=self.generation_input,
            prior_commits=self.prior_commits,
            prior_manifests=self.prior_manifests,
        )


@dataclass(frozen=True)
class EvaluationTask:


    run_id: str
    generation_index: int
    generation_id: str
    slot: int
    seed: int
    input_snapshot: EvolutionInputSnapshot
    generation_input: GenerationInput
    proposal: Proposal

    def __post_init__(self) -> None:
        _non_negative_int(self.slot, "slot")
        _non_negative_int(self.seed, "evaluation seed")
        _validate_generation_identity(
            run_id=self.run_id,
            generation_index=self.generation_index,
            generation_id=self.generation_id,
            input_snapshot=self.input_snapshot,
            generation_input=self.generation_input,
        )
        _validate_task_proposal(self)


def _validate_task_proposal(task: EvaluationTask) -> None:
    if not isinstance(task.proposal, Proposal):
        raise TypeError("evaluation task proposal must be Proposal")
    task.proposal.verify()
    if task.proposal.generation_id != task.generation_id:
        raise ValueError("evaluation task proposal belongs to another generation")
    if task.proposal.slot != task.slot:
        raise ValueError("evaluation task proposal belongs to another slot")


_GENERATION_VIEW_NONCE = object()


@dataclass(frozen=True)
class _ValidatedGenerationView:


    run_id: str
    generation_index: int
    generation_id: str
    input_snapshot: EvolutionInputSnapshot
    generation_input: GenerationInput
    prior_commits: Tuple[GenerationCommit, ...]
    prior_manifests: Tuple[GenerationManifest, ...]
    _nonce: object = field(repr=False, compare=False)

    def _matches_base(
        self,
        *,
        run_id: str,
        generation_index: int,
        generation_id: str,
        input_snapshot: EvolutionInputSnapshot,
        generation_input: GenerationInput,
    ) -> bool:
        return (
            self._nonce is _GENERATION_VIEW_NONCE
            and self.run_id == run_id
            and self.generation_index == generation_index
            and self.generation_id == generation_id
            and self.input_snapshot is input_snapshot
            and self.generation_input is generation_input
        )

    def matches_context(self, context: ProposalContext) -> bool:
        return self._matches_base(
            run_id=context.run_id,
            generation_index=context.generation_index,
            generation_id=context.generation_id,
            input_snapshot=context.input_snapshot,
            generation_input=context.generation_input,
        ) and (
            self.prior_commits is context.prior_commits
            and self.prior_manifests is context.prior_manifests
        )

    def matches_task(self, task: EvaluationTask) -> bool:
        return self._matches_base(
            run_id=task.run_id,
            generation_index=task.generation_index,
            generation_id=task.generation_id,
            input_snapshot=task.input_snapshot,
            generation_input=task.generation_input,
        )

    def revalidate(self) -> None:
        _validated_generation_view(
            run_id=self.run_id,
            generation_index=self.generation_index,
            generation_id=self.generation_id,
            input_snapshot=self.input_snapshot,
            generation_input=self.generation_input,
            prior_commits=self.prior_commits,
            prior_manifests=self.prior_manifests,
        )


def _proposal_context_from_validated_view(
    view: _ValidatedGenerationView,
    *,
    slot: int,
    generation_seed: int,
    slot_seed: int,
) -> ProposalContext:


    _non_negative_int(slot, "slot")
    _non_negative_int(generation_seed, "generation_seed")
    _non_negative_int(slot_seed, "slot_seed")
    context = object.__new__(ProposalContext)
    for name, value in (
        ("run_id", view.run_id),
        ("generation_index", view.generation_index),
        ("generation_id", view.generation_id),
        ("slot", slot),
        ("generation_seed", generation_seed),
        ("slot_seed", slot_seed),
        ("input_snapshot", view.input_snapshot),
        ("generation_input", view.generation_input),
        ("prior_commits", view.prior_commits),
        ("prior_manifests", view.prior_manifests),
    ):
        object.__setattr__(context, name, value)
    if not view.matches_context(context):
        raise RuntimeError("validated proposal context construction failed")
    return context


def _evaluation_task_from_validated_view(
    view: _ValidatedGenerationView,
    *,
    slot: int,
    seed: int,
    proposal: Proposal,
) -> EvaluationTask:


    _non_negative_int(slot, "slot")
    _non_negative_int(seed, "evaluation seed")
    task = object.__new__(EvaluationTask)
    for name, value in (
        ("run_id", view.run_id),
        ("generation_index", view.generation_index),
        ("generation_id", view.generation_id),
        ("slot", slot),
        ("seed", seed),
        ("input_snapshot", view.input_snapshot),
        ("generation_input", view.generation_input),
        ("proposal", proposal),
    ):
        object.__setattr__(task, name, value)
    if not view.matches_task(task):
        raise RuntimeError("validated evaluation task construction failed")
    _validate_task_proposal(task)
    return task


def _validate_generation_identity(
    *,
    run_id: str,
    generation_index: int,
    generation_id: str,
    input_snapshot: EvolutionInputSnapshot,
    generation_input: GenerationInput,
) -> None:
    _required_text(run_id, "run_id")
    index = _non_negative_int(generation_index, "generation_index")
    _required_text(generation_id, "generation_id")
    if not isinstance(input_snapshot, EvolutionInputSnapshot):
        raise TypeError("input_snapshot must be EvolutionInputSnapshot")
    input_snapshot.verify()
    if not isinstance(generation_input, GenerationInput):
        raise TypeError("generation_input must be GenerationInput")
    generation_input.verify()
    if generation_input.generation_index != index:
        raise ValueError("generation input belongs to another generation index")
    if generation_input.initial_snapshot_hash != input_snapshot.snapshot_hash:
        raise ValueError("generation input belongs to another initial snapshot")


def _validated_generation_view(
    *,
    run_id: str,
    generation_index: int,
    generation_id: str,
    input_snapshot: EvolutionInputSnapshot,
    generation_input: GenerationInput,
    prior_commits: Tuple[GenerationCommit, ...],
    prior_manifests: Tuple[GenerationManifest, ...],
) -> _ValidatedGenerationView:
    _validate_generation_identity(
        run_id=run_id,
        generation_index=generation_index,
        generation_id=generation_id,
        input_snapshot=input_snapshot,
        generation_input=generation_input,
    )
    index = generation_index
    if not isinstance(prior_commits, tuple) or len(prior_commits) != index:
        raise ValueError("prior_commits must contain one commit per prior generation")
    if not isinstance(prior_manifests, tuple) or len(prior_manifests) != index:
        raise ValueError(
            "prior_manifests must contain one manifest per prior generation"
        )
    commit_hashes: list[str] = []
    for offset, (manifest, commit) in enumerate(zip(prior_manifests, prior_commits)):
        if not isinstance(commit, GenerationCommit):
            raise TypeError("prior_commits must contain GenerationCommit values")
        if not isinstance(manifest, GenerationManifest):
            raise TypeError("prior_manifests must contain GenerationManifest values")
        if manifest.generation_id != f"{run_id}:generation:{offset:08d}":
            raise ValueError("prior manifest generation chain is discontinuous")
        expected_input_hash = GenerationInput.create(
            generation_index=offset,
            initial_snapshot_hash=input_snapshot.snapshot_hash,
            prior_commit_hashes=tuple(commit_hashes),
        ).input_hash
        if manifest.input_snapshot_hash != expected_input_hash:
            raise ValueError("prior manifest input/commit chain is discontinuous")


        commit.verify(manifest)
        commit_hashes.append(commit.commit_hash)
    if tuple(commit_hashes) != generation_input.prior_commit_hashes:
        raise ValueError("prior commits do not match the sealed generation input")
    return _ValidatedGenerationView(
        run_id=run_id,
        generation_index=index,
        generation_id=generation_id,
        input_snapshot=input_snapshot,
        generation_input=generation_input,
        prior_commits=prior_commits,
        prior_manifests=prior_manifests,
        _nonce=_GENERATION_VIEW_NONCE,
    )


def _evaluate_operation(
    evaluator: "ProposalEvaluator", task: EvaluationTask
) -> SealedEvaluation:


    outcome = evaluator.evaluate(task)
    if not isinstance(outcome, SealedEvaluation):
        raise TypeError("evaluator must return SealedEvaluation")
    outcome.verify_for(task.proposal)
    return outcome


def _worker_operational_failure(evaluation: SealedEvaluation) -> bool:


    if evaluation.status != "failed":
        return False
    return any(
        record.source == "native-evolution-worker"
        and record.kind
        in {
            "evaluator_cancelled",
            "evaluator_exception",
            "evaluator_timeout",
        }
        for record in evaluation.evidence().records
    )


class ProposalSource(Protocol):


    def propose(self, context: ProposalContext) -> ProposalDraft: ...


class BatchedProposalSource(Protocol):


    def propose_many(
        self, contexts: Tuple[ProposalContext, ...]
    ) -> Tuple[ProposalDraft, ...]: ...


class ProposalEvaluator(Protocol):


    def evaluate(self, task: EvaluationTask) -> SealedEvaluation: ...


@dataclass(frozen=True)
class _SealingEvaluationOperation:


    evaluator: ProposalEvaluator

    def __call__(self, task: EvaluationTask) -> SealedEvaluation:
        return _evaluate_operation(self.evaluator, task)


EvaluationOperation = Callable[[EvaluationTask], SealedEvaluation]


class EvaluationExecutor(Protocol):


    def execute(
        self,
        tasks: Tuple[EvaluationTask, ...],
        operation: EvaluationOperation,
    ) -> Iterable[SealedEvaluation]: ...


class SequentialEvaluationExecutor:


    def execute(
        self,
        tasks: Tuple[EvaluationTask, ...],
        operation: EvaluationOperation,
    ) -> Iterable[SealedEvaluation]:
        return tuple(operation(task) for task in tasks)


def _executor_safety_identity(executor: EvaluationExecutor) -> Mapping[str, Any]:


    declared = getattr(executor, "execution_safety_identity", None)
    if callable(declared):
        value = declared()
        if not isinstance(value, Mapping):
            raise TypeError("execution_safety_identity() must return a mapping")
        return dict(value)
    timeout = getattr(executor, "_evaluation_timeout", None)
    executor_type = type(executor)
    hard_timeout = (
        executor_type.__module__ == "astevolve.evolution.workers"
        and executor_type.__qualname__ == "OwnedProcessEvaluationExecutor"
    )
    return {
        "hard_timeout": hard_timeout,
        "timeout_seconds": timeout,
    }


class CommitProjection(Protocol):


    def apply(self, commit: GenerationCommit) -> Any: ...


@dataclass(frozen=True)
class EvolutionRunConfig:
    run_id: str
    total_generations: int
    proposal_budget: int
    root_seed: int = 0

    def __post_init__(self) -> None:
        _required_text(self.run_id, "run_id")
        _non_negative_int(self.total_generations, "total_generations")
        _positive_int(self.proposal_budget, "proposal_budget")
        _non_negative_int(self.root_seed, "root_seed")
        if self.total_generations > 99_999_999:
            raise ValueError("total_generations exceeds generation ID capacity")


@dataclass(frozen=True)
class EvolutionProgress:
    run_id: str
    input_snapshot_hash: str
    target_generations: int
    committed_generations: int
    logical_budget_committed: int
    next_generation_index: int
    committed_generation_ids: Tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.committed_generations == self.target_generations


@dataclass(frozen=True)
class GenerationRunResult:
    generation_index: int
    manifest: GenerationManifest
    commit: GenerationCommit
    recovered: bool
    projection_applied: bool
    progress: EvolutionProgress


@dataclass(frozen=True)
class EvolutionRunResult:
    input_snapshot: EvolutionInputSnapshot
    generations: Tuple[GenerationRunResult, ...]
    progress: EvolutionProgress


class EvolutionOrchestrator:


    def __init__(
        self,
        *,
        config: EvolutionRunConfig,
        input_snapshot: Union[EvolutionInputSnapshot, Mapping[str, Any]],
        proposal_source: ProposalSource,
        evaluator: ProposalEvaluator,
        ledger: FileGenerationLedger,
        projection: Optional[CommitProjection] = None,
        executor: Optional[EvaluationExecutor] = None,
        execution_contract: Optional[Mapping[str, Any]] = None,
        execution_semantics: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not isinstance(config, EvolutionRunConfig):
            raise TypeError("config must be EvolutionRunConfig")
        original_snapshot = (
            input_snapshot
            if isinstance(input_snapshot, EvolutionInputSnapshot)
            else EvolutionInputSnapshot.freeze(input_snapshot)
        )
        original_snapshot.verify()
        resolved_executor = executor or SequentialEvaluationExecutor()
        raw_payload = original_snapshot.payload()
        has_embedded_contract = NATIVE_EXECUTION_CONTRACT_KEY in raw_payload
        embedded_contract = raw_payload.get(NATIVE_EXECUTION_CONTRACT_KEY)
        if execution_contract is None:
            if has_embedded_contract:
                raise ValueError(
                    "reserved native execution contract requires the matching "
                    "execution_contract argument"
                )
            if execution_semantics is not None and not isinstance(
                execution_semantics, Mapping
            ):
                raise TypeError("execution_semantics must be a mapping")
            semantics = dict(execution_semantics or {})
            direct_core = {
                "schema_version": NATIVE_EXECUTION_CONTRACT_VERSION,
                "mode": "direct_api",
                "run_id": config.run_id,
                "root_seed": config.root_seed,
                "proposal_budget": config.proposal_budget,
                "components": {
                    "proposal_source": python_code_identity(proposal_source),
                    "proposal_evaluator": python_code_identity(evaluator),
                    "commit_projection": (
                        python_code_identity(projection)
                        if projection is not None
                        else None
                    ),
                },
                "inputs": {
                    "original_input_snapshot_hash": (original_snapshot.snapshot_hash),
                    "declared_execution_semantics": semantics,
                },
                "evaluation_safety": _executor_safety_identity(resolved_executor),
                "seed_derivation": "astevolve.evolution.private_seed.v1",
            }
            resolved_contract = seal_execution_contract(direct_core)
            raw_payload[NATIVE_EXECUTION_CONTRACT_KEY] = resolved_contract
        else:
            if execution_semantics is not None:
                raise ValueError(
                    "execution_semantics cannot be combined with execution_contract"
                )
            resolved_contract = validate_execution_contract(
                execution_contract,
                run_id=config.run_id,
                root_seed=config.root_seed,
                proposal_budget=config.proposal_budget,
            )
            if has_embedded_contract and embedded_contract != resolved_contract:
                raise ValueError(
                    "input snapshot contains another native execution contract"
                )
            raw_payload[NATIVE_EXECUTION_CONTRACT_KEY] = resolved_contract
        snapshot = EvolutionInputSnapshot.freeze(raw_payload)
        self._config = config
        self._snapshot = snapshot
        self._execution_contract = resolved_contract
        self._proposal_source = proposal_source
        self._evaluator = evaluator
        self._ledger = ledger
        self._projection = projection
        self._executor = resolved_executor
        self._projected: dict[str, str] = {}
        self._lock = threading.RLock()

    @property
    def config(self) -> EvolutionRunConfig:
        return self._config

    @property
    def input_snapshot(self) -> EvolutionInputSnapshot:
        return self._snapshot

    @property
    def execution_contract(self) -> dict[str, Any]:


        return validate_execution_contract(
            self._execution_contract,
            run_id=self._config.run_id,
            root_seed=self._config.root_seed,
            proposal_budget=self._config.proposal_budget,
        )

    def _generation_id(self, index: int) -> str:
        _non_negative_int(index, "generation_index")
        return f"{self._config.run_id}:generation:{index:08d}"

    def _generation_index(self, generation_id: str) -> Optional[int]:
        prefix = f"{self._config.run_id}:generation:"
        if not generation_id.startswith(prefix):
            return None
        suffix = generation_id[len(prefix) :]
        if not _GENERATION_SUFFIX.fullmatch(suffix):
            raise EvolutionRecoveryError(
                f"run contains malformed generation identity: {generation_id!r}"
            )
        return int(suffix)

    def _validate_published(
        self,
        index: int,
        published: PublishedGeneration,
        prior: Tuple[PublishedGeneration, ...],
    ) -> None:
        manifest = published.manifest
        if manifest.generation_id != self._generation_id(index):
            raise EvolutionRecoveryError("generation identity/index mismatch")
        generation_input = self._generation_input(index=index, prior=prior)
        if manifest.input_snapshot_hash != generation_input.input_hash:
            raise EvolutionRecoveryError(
                f"generation {index} input snapshot hash / commit chain hash "
                "does not match this run"
            )
        if manifest.logical_budget != self._config.proposal_budget:
            raise EvolutionRecoveryError(
                f"generation {index} logical budget does not match this run"
            )
        published.commit.verify(manifest)

    def _recover_published(
        self, *, apply_projection: bool
    ) -> Tuple[PublishedGeneration, ...]:
        indexed: list[tuple[int, PublishedGeneration]] = []
        for published in self._ledger.iter_published():
            index = self._generation_index(published.manifest.generation_id)
            if index is not None:
                indexed.append((index, published))
        indexed.sort(key=lambda item: item[0])
        actual = tuple(index for index, _ in indexed)
        expected = tuple(range(len(indexed)))
        if actual != expected:
            raise EvolutionRecoveryError(
                f"committed generation index gap; expected={expected}, actual={actual}"
            )
        if len(indexed) > self._config.total_generations:
            raise EvolutionRecoveryError(
                "ledger contains more generations than the configured run target"
            )
        validated: list[PublishedGeneration] = []
        for index, published in indexed:
            self._validate_published(index, published, tuple(validated))
            validated.append(published)
            if apply_projection:
                self._apply_projection(published.commit)
        return tuple(published for _, published in indexed)

    def _progress(
        self, published: Tuple[PublishedGeneration, ...]
    ) -> EvolutionProgress:
        return EvolutionProgress(
            run_id=self._config.run_id,
            input_snapshot_hash=self._snapshot.snapshot_hash,
            target_generations=self._config.total_generations,
            committed_generations=len(published),
            logical_budget_committed=sum(
                item.commit.logical_budget_used for item in published
            ),
            next_generation_index=len(published),
            committed_generation_ids=tuple(
                item.manifest.generation_id for item in published
            ),
        )

    def progress(self) -> EvolutionProgress:


        with self._lock:
            return self._progress(self._recover_published(apply_projection=False))

    def _generation_input(
        self,
        *,
        index: int,
        prior: Tuple[PublishedGeneration, ...],
    ) -> GenerationInput:
        if len(prior) != index:
            raise EvolutionRecoveryError(
                "generation input cannot be built from a discontinuous prior chain"
            )
        return GenerationInput.create(
            generation_index=index,
            initial_snapshot_hash=self._snapshot.snapshot_hash,
            prior_commit_hashes=(item.commit.commit_hash for item in prior),
        )

    def _build_manifest(
        self,
        *,
        index: int,
        generation_id: str,
        prior: Tuple[PublishedGeneration, ...],
    ) -> tuple[GenerationManifest, _ValidatedGenerationView]:
        prior_commits = tuple(item.commit for item in prior)
        prior_manifests = tuple(item.manifest for item in prior)
        generation_input = self._generation_input(index=index, prior=prior)
        validated_view = _validated_generation_view(
            run_id=self._config.run_id,
            generation_index=index,
            generation_id=generation_id,
            input_snapshot=self._snapshot,
            generation_input=generation_input,
            prior_commits=prior_commits,
            prior_manifests=prior_manifests,
        )
        generation_seed = _private_seed(
            root_seed=self._config.root_seed,
            run_id=self._config.run_id,
            generation_index=index,
            slot=-1,
            purpose="generation",
        )
        contexts = tuple(
            _proposal_context_from_validated_view(
                validated_view,
                slot=slot,
                generation_seed=generation_seed,
                slot_seed=_private_seed(
                    root_seed=self._config.root_seed,
                    run_id=self._config.run_id,
                    generation_index=index,
                    slot=slot,
                    purpose="proposal",
                ),
            )
            for slot in range(self._config.proposal_budget)
        )
        batched = getattr(self._proposal_source, "propose_many", None)
        if callable(batched):
            try:
                drafts = tuple(batched(contexts))
            except Exception as exc:
                raise ProposalSourceError(
                    f"batched proposal source failed for generation {index}"
                ) from exc
            if len(drafts) != len(contexts):
                raise ProposalSourceError(
                    "batched proposal source must return exactly one draft per slot"
                )
        else:
            generated = []
            for context in contexts:
                try:
                    generated.append(self._proposal_source.propose(context))
                except Exception as exc:
                    raise ProposalSourceError(
                        "proposal source failed for generation "
                        f"{index}, slot {context.slot}"
                    ) from exc
            drafts = tuple(generated)
        proposals = []
        for context, draft in zip(contexts, drafts):
            try:
                if not isinstance(draft, ProposalDraft):
                    raise TypeError("proposal source must return ProposalDraft")
                proposals.append(
                    draft.seal(generation_id=generation_id, slot=context.slot)
                )
            except Exception as exc:
                raise ProposalSourceError(
                    "proposal source returned an invalid draft for generation "
                    f"{index}, slot {context.slot}"
                ) from exc
        try:
            validated_view.revalidate()
        except Exception as exc:
            raise ProposalSourceError(
                "proposal source mutated the sealed generation history"
            ) from exc
        return (
            GenerationManifest.create(
                generation_id=generation_id,
                input_snapshot_hash=generation_input.input_hash,
                proposals=tuple(proposals),
            ),
            validated_view,
        )

    def _load_or_register_manifest(
        self,
        *,
        index: int,
        prior: Tuple[PublishedGeneration, ...],
    ) -> tuple[GenerationManifest, Optional[_ValidatedGenerationView]]:
        generation_id = self._generation_id(index)
        generation_input = self._generation_input(index=index, prior=prior)
        existing = self._ledger.load_manifest(generation_id)
        if existing is not None:
            if existing.input_snapshot_hash != generation_input.input_hash:
                raise EvolutionRecoveryError(
                    f"registered generation {index} has another input/commit chain"
                )
            if existing.logical_budget != self._config.proposal_budget:
                raise EvolutionRecoveryError(
                    f"registered generation {index} has another logical budget"
                )
            return existing, None
        manifest, validated_view = self._build_manifest(
            index=index, generation_id=generation_id, prior=prior
        )

        return self._ledger.register_manifest(manifest), validated_view

    def _evaluate_task(self, task: EvaluationTask) -> SealedEvaluation:
        return _evaluate_operation(self._evaluator, task)

    def _evaluate_and_publish(
        self,
        *,
        index: int,
        manifest: GenerationManifest,
        prior: Tuple[PublishedGeneration, ...],
        validated_view: Optional[_ValidatedGenerationView] = None,
    ) -> GenerationCommit:
        engine = GenerationEngine.restore(manifest, self._ledger)
        if engine.published_commit is not None:
            return engine.published_commit
        if validated_view is None:
            validated_view = _validated_generation_view(
                run_id=self._config.run_id,
                generation_index=index,
                generation_id=manifest.generation_id,
                input_snapshot=self._snapshot,
                generation_input=self._generation_input(index=index, prior=prior),
                prior_commits=tuple(item.commit for item in prior),
                prior_manifests=tuple(item.manifest for item in prior),
            )
        else:
            if not validated_view._matches_base(
                run_id=self._config.run_id,
                generation_index=index,
                generation_id=manifest.generation_id,
                input_snapshot=self._snapshot,
                generation_input=validated_view.generation_input,
            ):
                raise EvaluationExecutionError(
                    "proposal wave returned another validated generation view"
                )
            if (
                manifest.input_snapshot_hash
                != validated_view.generation_input.input_hash
            ):
                raise EvaluationExecutionError(
                    "registered manifest does not match the validated generation view"
                )
            if len(prior) != len(validated_view.prior_commits) or any(
                published.commit is not commit
                or published.manifest is not prior_manifest
                for published, commit, prior_manifest in zip(
                    prior,
                    validated_view.prior_commits,
                    validated_view.prior_manifests,
                )
            ):
                raise EvaluationExecutionError(
                    "validated generation view does not match recovered history"
                )
        tasks = tuple(
            _evaluation_task_from_validated_view(
                validated_view,
                slot=proposal.slot,
                seed=_private_seed(
                    root_seed=self._config.root_seed,
                    run_id=self._config.run_id,
                    generation_index=index,
                    slot=proposal.slot,
                    purpose="evaluation",
                ),
                proposal=proposal,
            )
            for proposal in manifest.proposals
        )
        operation = _SealingEvaluationOperation(self._evaluator)
        try:
            outcomes = tuple(self._executor.execute(tasks, operation))
        except Exception as exc:
            raise EvaluationExecutionError(
                "evaluation executor failed before returning a complete terminal wave"
            ) from exc
        if len(outcomes) != manifest.logical_budget:
            raise EvaluationExecutionError(
                "evaluation executor did not return exactly one outcome per task"
            )
        try:


            _validate_generation_identity(
                run_id=validated_view.run_id,
                generation_index=validated_view.generation_index,
                generation_id=validated_view.generation_id,
                input_snapshot=validated_view.input_snapshot,
                generation_input=validated_view.generation_input,
            )
            outcome_ids = tuple(outcome.proposal_id for outcome in outcomes)
            if len(set(outcome_ids)) != len(outcome_ids) or set(outcome_ids) != set(
                manifest.ordered_proposal_ids
            ):
                raise ValueError(
                    "executor outcome identities do not match the reserved tasks"
                )
            for outcome in outcomes:
                if not isinstance(outcome, SealedEvaluation):
                    raise TypeError("executor returned a non-SealedEvaluation value")
                if _worker_operational_failure(outcome):
                    raise EvaluationExecutionError(
                        "evaluation wave contains an operational worker failure; "
                        "the registered manifest remains retryable"
                    )
                engine.submit(outcome)
            return engine.publish()
        except EvaluationExecutionError:
            raise
        except Exception as exc:
            raise EvaluationExecutionError(
                "evaluation executor returned invalid or duplicate task outcomes"
            ) from exc

    def _apply_projection(self, commit: GenerationCommit) -> bool:
        if self._projection is None:
            return False
        prior_hash = self._projected.get(commit.generation_id)
        if prior_hash == commit.commit_hash:
            return True
        if prior_hash is not None:
            raise EvolutionRecoveryError(
                f"generation {commit.generation_id!r} projection identity changed"
            )


        try:
            self._projection.apply(commit)
        except Exception as exc:
            raise ProjectionApplicationError(
                f"durable generation {commit.generation_id!r} projection failed; "
                "retry will replay the same commit"
            ) from exc
        self._projected[commit.generation_id] = commit.commit_hash
        return True

    def _run_generation_locked(
        self,
        *,
        index: int,
        prior: Tuple[PublishedGeneration, ...],
    ) -> GenerationRunResult:
        manifest, validated_view = self._load_or_register_manifest(
            index=index, prior=prior
        )


        commit = self._evaluate_and_publish(
            index=index,
            manifest=manifest,
            prior=prior,
            validated_view=validated_view,
        )
        projection_applied = self._apply_projection(commit)
        published = prior + (PublishedGeneration(manifest=manifest, commit=commit),)
        return GenerationRunResult(
            generation_index=index,
            manifest=manifest,
            commit=commit,
            recovered=False,
            projection_applied=projection_applied,
            progress=self._progress(published),
        )

    def run_generation(self, index: Optional[int] = None) -> GenerationRunResult:


        with self._lock:
            prior = self._recover_published(apply_projection=True)
            resolved = (
                len(prior) if index is None else _non_negative_int(index, "index")
            )
            if resolved < len(prior):
                published = prior[resolved]
                return GenerationRunResult(
                    generation_index=resolved,
                    manifest=published.manifest,
                    commit=published.commit,
                    recovered=True,
                    projection_applied=(self._projection is not None),
                    progress=self._progress(prior),
                )
            if resolved > len(prior):
                raise GenerationIndexError(
                    f"cannot skip generation {len(prior)} and run {resolved}"
                )
            if resolved >= self._config.total_generations:
                raise GenerationIndexError("configured generation target is complete")
            return self._run_generation_locked(index=resolved, prior=prior)

    def run(self) -> EvolutionRunResult:


        with self._lock:
            published = self._recover_published(apply_projection=True)
            results = [
                GenerationRunResult(
                    generation_index=index,
                    manifest=item.manifest,
                    commit=item.commit,
                    recovered=True,
                    projection_applied=(self._projection is not None),
                    progress=self._progress(published[: index + 1]),
                )
                for index, item in enumerate(published)
            ]
            while len(published) < self._config.total_generations:
                result = self._run_generation_locked(
                    index=len(published), prior=published
                )
                results.append(result)
                published = published + (
                    PublishedGeneration(manifest=result.manifest, commit=result.commit),
                )
            return EvolutionRunResult(
                input_snapshot=self._snapshot,
                generations=tuple(results),
                progress=self._progress(published),
            )


__all__ = [
    "BatchedProposalSource",
    "CommitProjection",
    "EvaluationExecutionError",
    "EvaluationExecutor",
    "EvaluationTask",
    "EvolutionInputSnapshot",
    "EvolutionOrchestrationError",
    "EvolutionOrchestrator",
    "EvolutionProgress",
    "EvolutionRecoveryError",
    "EvolutionRunConfig",
    "EvolutionRunResult",
    "GenerationInput",
    "GenerationIndexError",
    "GenerationRunResult",
    "ProjectionApplicationError",
    "ProposalContext",
    "ProposalDraft",
    "ProposalEvaluator",
    "ProposalSource",
    "ProposalSourceError",
    "SequentialEvaluationExecutor",
]
