

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from astevolve.evaluation.selection import select_feasibility_first
from astevolve.runtime.edit_contract_lifecycle import (
    EDIT_CONTRACT_RESPONSE_VERSION,
    EditContractEnvelope,
    edit_contract_prompt_projection,
)
from engine.causal_flow import EffectiveSearchContract, GraphPatch
from engine.memory_lifecycle import (
    MemoryExecutionContext,
    MemorySnapshot,
    ProposalCausalEnvelope,
    capture_memory_snapshot,
)
from engine.memory_policy import MemoryPolicyConfig, MemoryScope
from engine.memory_update import (
    MemoryProposalError,
    StaleMemorySnapshotError,
    commit_internal_memory_update,
)


GENERATION_LIFECYCLE_VERSION = "astevolve.outer_generation_lifecycle.v1"
GENERATION_PLAN_VERSION = "astevolve.outer_generation_plan.v6"

logger = logging.getLogger(__name__)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_hash(path: str | Path | None) -> Optional[str]:
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _semantic_outer_snapshot(snapshot: Mapping[str, Any]) -> Dict[str, Any]:


    value = deepcopy(dict(snapshot))
    value.pop("updated_at_unix", None)
    return value


def outer_memory_hash(snapshot: Mapping[str, Any]) -> str:


    return _hash_json(_semantic_outer_snapshot(snapshot))


def _safe_component(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))
    return text.strip("._") or "unnamed"


def proposal_causal_envelope_from_parent(
    *,
    parent_program_id: str,
    parent_code: str,
    parent_artifacts: Optional[Mapping[str, Any]],
) -> ProposalCausalEnvelope:


    artifacts = parent_artifacts if isinstance(parent_artifacts, Mapping) else {}
    effective_contract = _load_parent_causal_artifact(
        artifacts,
        key="effective_search_contract",
        validator=EffectiveSearchContract.from_mapping,
    )
    graph_patch = _load_parent_causal_artifact(
        artifacts,
        key="graph_patch",
        validator=GraphPatch.from_mapping,
    )
    return ProposalCausalEnvelope.create(
        parent_program_id=parent_program_id,
        parent_code_hash=hashlib.sha256(str(parent_code).encode("utf-8")).hexdigest(),
        parent_effective_contract=(
            effective_contract
        ),
        parent_patch=graph_patch,
    )


def _load_parent_causal_artifact(
    artifacts: Mapping[str, Any],
    *,
    key: str,
    validator: Callable[[Mapping[str, Any]], Any],
) -> Optional[Dict[str, Any]]:


    raw = artifacts.get(key)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"parent causal artifact {key!r} is not UTF-8 JSON") from error
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"parent causal artifact {key!r} is invalid JSON") from error
    if not isinstance(raw, Mapping):
        raise ValueError(f"parent causal artifact {key!r} must be a JSON object")
    try:
        validated = validator(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"parent causal artifact {key!r} failed validation") from error
    to_dict = getattr(validated, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(f"validator for parent causal artifact {key!r} is not serializable")
    value = to_dict()
    if not isinstance(value, dict):
        raise TypeError(f"validated parent causal artifact {key!r} is not a mapping")
    return value


@dataclass(frozen=True)
class GenerationReservation:


    iteration: int
    proposal_id: str
    target_island: Optional[int] = None
    parent_program_id: str = ""
    edit_contract_envelope_json: str = ""
    proposal_causal_envelope_json: str = ""
    proposal_causal_envelope_hash: str = ""

    def __post_init__(self) -> None:
        if bool(self.proposal_causal_envelope_json) != bool(
            self.proposal_causal_envelope_hash
        ):
            raise ValueError(
                "proposal causal envelope JSON and hash must be provided together"
            )
        if self.proposal_causal_envelope_json:
            envelope = ProposalCausalEnvelope.from_json(
                self.proposal_causal_envelope_json
            )
            if envelope.to_json() != self.proposal_causal_envelope_json:
                raise ValueError("proposal causal envelope JSON must be canonical")
            if envelope.envelope_hash != self.proposal_causal_envelope_hash:
                raise ValueError("proposal causal envelope hash mismatch")
            if self.parent_program_id and envelope.parent_program_id != self.parent_program_id:
                raise ValueError("proposal causal envelope parent mismatch")

    def edit_contract_envelope(self) -> Optional[Dict[str, Any]]:
        if not self.edit_contract_envelope_json:
            return None
        return EditContractEnvelope.from_json(
            self.edit_contract_envelope_json,
            expected_parent_program_id=self.parent_program_id,
        ).to_dict()

    def proposal_causal_envelope(self) -> Optional[ProposalCausalEnvelope]:
        if not self.proposal_causal_envelope_json:
            return None
        return ProposalCausalEnvelope.from_json(
            self.proposal_causal_envelope_json
        )


@dataclass(frozen=True)
class GenerationPlan:


    generation_id: str
    logical_time: str
    outer_memory_json: str
    outer_input_hash: str
    inner_memory: Optional[MemorySnapshot]
    inner_target_path: str
    output_root: str
    reservations: Tuple[GenerationReservation, ...]
    history_registry_path: str = ""
    history_scope: str = ""
    history_lease_seconds: float = 300.0
    history_replicate_policy: str = "reject"
    memory_scope: Optional[MemoryScope] = None
    memory_policy: MemoryPolicyConfig = MemoryPolicyConfig()
    hierarchical_design_json: str = ""
    hierarchical_design_hash: str = ""
    schema_version: str = GENERATION_PLAN_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_PLAN_VERSION:
            raise ValueError(f"unsupported generation plan version: {self.schema_version!r}")
        parsed = json.loads(self.outer_memory_json)
        if not isinstance(parsed, dict):
            raise ValueError("outer memory snapshot must contain a mapping")
        if outer_memory_hash(parsed) != self.outer_input_hash:
            raise ValueError("outer memory hash does not match its snapshot")
        if bool(self.hierarchical_design_json) != bool(
            self.hierarchical_design_hash
        ):
            raise ValueError(
                "hierarchical design JSON and hash must be provided together"
            )
        if self.hierarchical_design_json:
            design = json.loads(self.hierarchical_design_json)
            if not isinstance(design, dict):
                raise ValueError("hierarchical design snapshot must be a mapping")
            if _hash_json(design) != self.hierarchical_design_hash:
                raise ValueError("hierarchical design snapshot hash mismatch")
        if not isinstance(self.memory_policy, MemoryPolicyConfig):
            raise TypeError("generation memory_policy must be MemoryPolicyConfig")
        if self.memory_scope is not None and not isinstance(
            self.memory_scope, MemoryScope
        ):
            raise TypeError("generation memory_scope must be MemoryScope or None")
        if self.memory_policy.may_read_adaptive_prior:
            if self.inner_memory is None or not self.inner_target_path:
                raise ValueError(
                    "adaptive generation memory policy requires an inner-memory target"
                )
            if self.memory_scope is None:
                raise ValueError(
                    "adaptive generation memory policy requires a closed MemoryScope"
                )
        proposal_ids = [item.proposal_id for item in self.reservations]
        if len(proposal_ids) != len(set(proposal_ids)):
            raise ValueError("generation proposal IDs must be unique")
        for item in self.reservations:
            if item.edit_contract_envelope_json and not item.parent_program_id:
                raise ValueError("contract-bound reservation is missing parent_program_id")
            if item.edit_contract_envelope_json:
                EditContractEnvelope.from_json(
                    item.edit_contract_envelope_json,
                    expected_parent_program_id=item.parent_program_id,
                )
            if item.proposal_causal_envelope_json:
                envelope = item.proposal_causal_envelope()
                if envelope is None or envelope.parent_program_id != item.parent_program_id:
                    raise ValueError("reservation causal envelope parent mismatch")

    def prompt_memory(self) -> Dict[str, Any]:


        value = json.loads(self.outer_memory_json)
        return deepcopy(value)

    def prompt_memory_for(self, proposal_id: str) -> Dict[str, Any]:


        memory = self.prompt_memory()
        if self.hierarchical_design_json:
            memory["hierarchical_design"] = deepcopy(
                json.loads(self.hierarchical_design_json)
            )
        reservation = self.reservation(proposal_id)
        envelope = reservation.edit_contract_envelope()
        if envelope is not None:
            canonical_envelope = EditContractEnvelope.from_mapping(
                envelope,
                expected_parent_program_id=reservation.parent_program_id,
            )
            memory["contract_handoff"] = {
                "required_response": True,
                "response_schema": {
                    "schema_version": EDIT_CONTRACT_RESPONSE_VERSION,
                    "contract_id": envelope["contract_id"],
                    "decision": "accept|reject",
                    "reason": "non-empty string",
                },
                "envelope_projection": edit_contract_prompt_projection(
                    canonical_envelope
                ),
            }
        return memory

    def hierarchical_design(self) -> Dict[str, Any]:


        if not self.hierarchical_design_json:
            return {}
        return deepcopy(json.loads(self.hierarchical_design_json))

    def reservation(self, proposal_id: str) -> GenerationReservation:
        for reservation in self.reservations:
            if reservation.proposal_id == proposal_id:
                return reservation
        raise KeyError(f"proposal {proposal_id!r} is not reserved in {self.generation_id!r}")

    def with_proposal_causal_envelope(
        self,
        proposal_id: str,
        envelope: ProposalCausalEnvelope,
    ) -> "GenerationPlan":


        if not isinstance(envelope, ProposalCausalEnvelope):
            raise TypeError("envelope must be ProposalCausalEnvelope")
        reservation = self.reservation(proposal_id)
        if reservation.parent_program_id != envelope.parent_program_id:
            raise ValueError("proposal causal envelope parent mismatch")
        updated = tuple(
            replace(
                item,
                proposal_causal_envelope_json=envelope.to_json(),
                proposal_causal_envelope_hash=envelope.envelope_hash,
            )
            if item.proposal_id == proposal_id
            else item
            for item in self.reservations
        )
        return replace(self, reservations=updated)

    def runtime_context(
        self,
        proposal_id: str,
        *,
        trial_id: str = "",
    ) -> MemoryExecutionContext:


        reservation = self.reservation(proposal_id)
        proposal_dir = Path(self.output_root) / _safe_component(self.generation_id) / _safe_component(
            proposal_id
        )
        if trial_id:
            proposal_dir = proposal_dir / _safe_component(trial_id)
        island_id = reservation.target_island
        island_role = ""
        if island_id is not None:
            strategy = self.hierarchical_design().get("strategy_plan") or {}
            directives = (
                strategy.get("island_directives", [])
                if isinstance(strategy, Mapping)
                else []
            )
            for directive in directives:
                if not isinstance(directive, Mapping):
                    continue
                if directive.get("island") == island_id:
                    island_role = str(directive.get("role_id") or "").strip()
                    break
        return MemoryExecutionContext(
            generation_id=self.generation_id,
            proposal_id=proposal_id,
            trial_id=trial_id,
            island_id=island_id,
            island_role=island_role,
            scope_id="/".join(
                item for item in (self.generation_id, proposal_id, trial_id) if item
            ),
            logical_time=self.logical_time,
            commit_mode="deferred",
            snapshot=self.inner_memory,
            target_path=self.inner_target_path,
            output_dir=str(proposal_dir),
            edit_contract_envelope_json=reservation.edit_contract_envelope_json,
            proposal_causal_envelope_json=(
                reservation.proposal_causal_envelope_json
            ),
            history_registry_path=self.history_registry_path,
            history_scope=self.history_scope,
            history_owner_token=(
                "/".join(
                    item
                    for item in (
                        self.generation_id,
                        proposal_id,
                        trial_id or "candidate",
                    )
                    if item
                )
            ),
            history_lease_seconds=self.history_lease_seconds,
            history_replicate_policy=self.history_replicate_policy,
            memory_scope=self.memory_scope,
            memory_policy=self.memory_policy,
        )

    def to_artifact(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "logical_time": self.logical_time,
            "outer_input_hash": self.outer_input_hash,
            "inner_input": self.inner_memory.to_artifact() if self.inner_memory else None,
            "inner_target_path": self.inner_target_path or None,
            "history": {
                "enabled": bool(self.history_registry_path),
                "registry_path": self.history_registry_path or None,
                "scope": self.history_scope or None,
                "lease_seconds": self.history_lease_seconds,
                "replicate_policy": self.history_replicate_policy,
            },
            "memory_scope": (
                self.memory_scope.to_artifact() if self.memory_scope else None
            ),
            "memory_policy": self.memory_policy.to_artifact(),
            "reservations": [
                {
                    "iteration": item.iteration,
                    "proposal_id": item.proposal_id,
                    "target_island": item.target_island,
                    "parent_program_id": item.parent_program_id or None,
                    "edit_contract_envelope": item.edit_contract_envelope(),
                    "proposal_causal_envelope": (
                        item.proposal_causal_envelope().to_dict()
                        if item.proposal_causal_envelope() is not None
                        else None
                    ),
                    "proposal_causal_envelope_hash": (
                        item.proposal_causal_envelope_hash or None
                    ),
                }
                for item in self.reservations
            ],
        }


@dataclass
class GenerationObservation:


    proposal_id: str
    iteration: int
    child_program: Any = None
    parent: Any = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    target_island: Optional[int] = None
    prompt: Optional[Dict[str, Any]] = None
    llm_response: Optional[str] = None
    iteration_time: float = 0.0
    error: Optional[str] = None
    generation_id: str = ""
    trial_id: str = ""
    proposal_causal_envelope_json: str = ""
    proposal_causal_envelope_hash: str = ""

    def __post_init__(self) -> None:
        if bool(self.proposal_causal_envelope_json) != bool(
            self.proposal_causal_envelope_hash
        ):
            raise ValueError(
                "observation causal envelope JSON and hash must be provided together"
            )
        if self.proposal_causal_envelope_json:
            envelope = ProposalCausalEnvelope.from_json(
                self.proposal_causal_envelope_json
            )
            if envelope.envelope_hash != self.proposal_causal_envelope_hash:
                raise ValueError("observation causal envelope hash mismatch")
            if envelope.to_json() != self.proposal_causal_envelope_json:
                raise ValueError("observation causal envelope JSON must be canonical")

    @property
    def stable_key(self) -> Tuple[int, str, str]:
        program_id = str(getattr(self.child_program, "id", "") or "")
        return (int(self.iteration), str(self.proposal_id), program_id)


def _metric_objective(program: Any) -> float:
    metrics = getattr(program, "metrics", {}) or {}
    for key in ("raw_combined_score", "combined_score", "evaluator_score", "multistate_score"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                return numeric
    numeric = [
        float(value)
        for value in metrics.values()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    return sum(numeric) / len(numeric) if numeric else 0.0


def _gate_sources(program: Any) -> Dict[str, Any]:
    metrics = getattr(program, "metrics", {}) or {}
    if isinstance(metrics.get("hard_gate_pass"), bool):
        return {
            "outer_evaluator": {
                "passed": metrics["hard_gate_pass"],
                "reasons": [] if metrics["hard_gate_pass"] else ["hard_gate_failed"],
            }
        }
    if isinstance(metrics.get("disqualified"), bool):
        passed = not metrics["disqualified"]
        return {
            "outer_evaluator": {
                "passed": passed,
                "reasons": [] if passed else ["disqualified"],
            }
        }
    return {"outer_evaluator": {"passed": True, "reasons": []}}


def _selected_proposal(artifacts: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    for key in (
        "selected_memory_update_proposal",
        "selected_inner_memory_proposal",
        "inner_memory_proposal",
    ):
        value = artifacts.get(key)
        if isinstance(value, Mapping):
            return value
    memory_update = artifacts.get("memory_update")
    if isinstance(memory_update, Mapping) and isinstance(memory_update.get("proposal"), Mapping):
        return memory_update["proposal"]
    return None


def _has_high_fidelity_evidence(artifacts: Mapping[str, Any]) -> bool:


    search_artifacts = artifacts.get("search_artifacts")
    if isinstance(search_artifacts, Mapping):
        summary = search_artifacts.get("structure_evaluation_summary")
        feedback = search_artifacts.get("structure_finalist_feedback")
        if not isinstance(summary, Mapping) or not isinstance(feedback, Mapping):
            return False
        try:
            evaluated = int(summary.get("evaluated") or 0)
        except (TypeError, ValueError):
            return False
        selected_provider = str(
            summary.get("selected_provider") or ""
        ).strip().lower()
        selected_provider = {
            "af3": "alphafold3",
            "alpha-fold3": "alphafold3",
            "alpha_fold3": "alphafold3",
        }.get(selected_provider, selected_provider)
        if evaluated < 1 or selected_provider not in {
            "protenix",
            "alphafold3",
        }:
            return False
        rows = feedback.get("candidates")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            return False
        selected_rows = [
            row
            for row in rows
            if isinstance(row, Mapping) and bool(row.get("selected"))
        ]
        if len(selected_rows) != 1:
            return False
        selected = selected_rows[0]
        row_provider = str(selected.get("provider") or "").strip().lower()
        row_provider = {"af3": "alphafold3"}.get(row_provider, row_provider)
        return bool(selected.get("structure_signal_available")) and (
            row_provider == selected_provider
        )


    lifecycle = artifacts.get("selected_memory_lifecycle") or {}
    return bool(
        isinstance(lifecycle, Mapping)
        and lifecycle.get("high_fidelity") is True
    )


def _validate_selected_proposal(
    plan: GenerationPlan,
    observation: GenerationObservation,
    proposal: Mapping[str, Any],
) -> None:
    snapshot = plan.inner_memory
    if snapshot is None:
        raise MemoryProposalError("generation has no inner-memory snapshot")
    if str(proposal.get("base_content_hash") or "") != snapshot.content_hash:
        raise MemoryProposalError("selected proposal content base does not match generation snapshot")
    if str(proposal.get("base_raw_hash") or "") != snapshot.raw_hash:
        raise MemoryProposalError("selected proposal raw base does not match generation snapshot")
    target = str(proposal.get("target_path") or "")
    if target and Path(target).resolve() != Path(plan.inner_target_path).resolve():
        raise MemoryProposalError("selected proposal target does not match generation target")
    provenance = proposal.get("provenance") or {}
    if not isinstance(provenance, Mapping):
        raise MemoryProposalError("selected proposal provenance must be a mapping")
    if str(provenance.get("generation_id") or "") != plan.generation_id:
        raise MemoryProposalError("selected proposal generation provenance mismatch")
    if str(provenance.get("proposal_id") or "") != observation.proposal_id:
        raise MemoryProposalError("selected proposal sibling provenance mismatch")
    lifecycle = observation.artifacts.get("selected_memory_lifecycle") or {}
    if isinstance(lifecycle, Mapping):
        selected_trial = str(lifecycle.get("selected_trial_id") or "")
        proposal_trial = str(provenance.get("trial_id") or "")
        if selected_trial and selected_trial != proposal_trial:
            raise MemoryProposalError("selected proposal trial provenance mismatch")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


class GenerationCoordinator:


    def __init__(self, plan: GenerationPlan):
        self.plan = plan
        self._final_artifact: Optional[Dict[str, Any]] = None

    @classmethod
    def begin(
        cls,
        database: Any,
        *,
        iterations: Sequence[int],
        target_islands: Optional[Sequence[Optional[int]]] = None,
        generation_id: Optional[str] = None,
        logical_time: Optional[str] = None,
        inner_memory_path: Optional[str | Path] = None,
        output_root: str | Path = "generation_runtime",
        history_registry_path: str | Path = "",
        history_scope: str = "",
        history_lease_seconds: float = 300.0,
        history_replicate_policy: str = "reject",
        memory_scope: Optional[MemoryScope] = None,
        memory_policy: Optional[MemoryPolicyConfig] = None,
    ) -> "GenerationCoordinator":
        ordered_iterations = tuple(int(item) for item in iterations)
        if not ordered_iterations:
            raise ValueError("a generation must reserve at least one iteration")
        if target_islands is None:
            islands: Tuple[Optional[int], ...] = (None,) * len(ordered_iterations)
        else:
            islands = tuple(target_islands)
            if len(islands) != len(ordered_iterations):
                raise ValueError("target_islands must align with iterations")
        resolved_generation = generation_id or (
            f"generation-{min(ordered_iterations):08d}-{max(ordered_iterations):08d}"
        )
        reservations = tuple(
            GenerationReservation(
                iteration=iteration,
                proposal_id=f"{resolved_generation}:proposal:{index:04d}:iteration:{iteration:08d}",
                target_island=islands[index],
            )
            for index, iteration in enumerate(ordered_iterations)
        )
        outer_snapshot = (
            database.get_optimizer_memory_snapshot()
            if hasattr(database, "get_optimizer_memory_snapshot")
            else {}
        )
        if not isinstance(outer_snapshot, Mapping):
            outer_snapshot = {}
        target_path = str(inner_memory_path or "")
        inner_snapshot = capture_memory_snapshot(target_path) if target_path else None
        resolved_memory_policy = memory_policy or MemoryPolicyConfig()
        frozen_outer_snapshot = _semantic_outer_snapshot(outer_snapshot)
        hierarchical_design = (
            database.get_hierarchical_design_snapshot()
            if hasattr(database, "get_hierarchical_design_snapshot")
            else {}
        )
        if not isinstance(hierarchical_design, Mapping):
            hierarchical_design = {}
        frozen_hierarchical_design = deepcopy(dict(hierarchical_design))
        plan = GenerationPlan(
            generation_id=resolved_generation,
            logical_time=str(logical_time or resolved_generation),
            outer_memory_json=_canonical_json(frozen_outer_snapshot),
            outer_input_hash=outer_memory_hash(frozen_outer_snapshot),
            inner_memory=inner_snapshot,
            inner_target_path=target_path,
            output_root=str(output_root),
            reservations=reservations,
            history_registry_path=str(history_registry_path),
            history_scope=str(history_scope),
            history_lease_seconds=float(history_lease_seconds),
            history_replicate_policy=str(history_replicate_policy),
            memory_scope=memory_scope,
            memory_policy=resolved_memory_policy,
            hierarchical_design_json=(
                _canonical_json(frozen_hierarchical_design)
                if frozen_hierarchical_design
                else ""
            ),
            hierarchical_design_hash=(
                _hash_json(frozen_hierarchical_design)
                if frozen_hierarchical_design
                else ""
            ),
        )
        return cls(plan)

    def prompt_memory(self) -> Dict[str, Any]:
        return self.plan.prompt_memory()

    def bind_parents(
        self,
        bindings: Mapping[str, Tuple[Any, ...]],
    ) -> "GenerationCoordinator":


        expected = {item.proposal_id for item in self.plan.reservations}
        if set(bindings) != expected:
            missing = sorted(expected - set(bindings))
            extra = sorted(set(bindings) - expected)
            raise ValueError(
                f"parent bindings must cover all reservations; missing={missing}, extra={extra}"
            )
        bound = []
        for item in self.plan.reservations:
            binding = bindings[item.proposal_id]
            if not isinstance(binding, (tuple, list)) or len(binding) not in {2, 3}:
                raise ValueError(
                    "each parent binding must be (parent_id, edit_contract) or "
                    "(parent_id, edit_contract, proposal_causal_envelope)"
                )
            parent_program_id, envelope_value = binding[:2]
            causal_value = binding[2] if len(binding) == 3 else None
            parent_id = str(parent_program_id or "").strip()
            if not parent_id:
                raise ValueError(f"reservation {item.proposal_id} has no parent program")
            envelope_json = ""
            if envelope_value:
                if isinstance(envelope_value, str):
                    envelope = EditContractEnvelope.from_json(
                        envelope_value,
                        expected_parent_program_id=parent_id,
                    )
                else:
                    envelope = EditContractEnvelope.from_mapping(
                        envelope_value,
                        expected_parent_program_id=parent_id,
                    )
                envelope_json = envelope.to_json()
            causal_json = ""
            causal_hash = ""
            if causal_value:
                if isinstance(causal_value, ProposalCausalEnvelope):
                    causal_envelope = causal_value
                elif isinstance(causal_value, str):
                    causal_envelope = ProposalCausalEnvelope.from_json(causal_value)
                elif isinstance(causal_value, Mapping):
                    if "schema_version" in causal_value or "envelope_hash" in causal_value:
                        causal_envelope = ProposalCausalEnvelope.from_mapping(causal_value)
                    else:
                        causal_envelope = ProposalCausalEnvelope.create(
                            parent_program_id=str(
                                causal_value.get("parent_program_id") or parent_id
                            ),
                            parent_code_hash=str(
                                causal_value.get("parent_code_hash") or ""
                            ),
                            parent_effective_contract=(
                                causal_value.get("parent_effective_contract")
                                if isinstance(
                                    causal_value.get("parent_effective_contract"), Mapping
                                )
                                else None
                            ),
                            parent_patch=(
                                causal_value.get("parent_patch")
                                if isinstance(causal_value.get("parent_patch"), Mapping)
                                else None
                            ),
                        )
                else:
                    raise ValueError(
                        "proposal causal envelope must be a mapping, JSON string, or envelope"
                    )
                if causal_envelope.parent_program_id != parent_id:
                    raise ValueError("proposal causal envelope parent mismatch")
                causal_json = causal_envelope.to_json()
                causal_hash = causal_envelope.envelope_hash
            bound.append(
                replace(
                    item,
                    parent_program_id=parent_id,
                    edit_contract_envelope_json=envelope_json,
                    proposal_causal_envelope_json=causal_json,
                    proposal_causal_envelope_hash=causal_hash,
                )
            )
        return GenerationCoordinator(replace(self.plan, reservations=tuple(bound)))

    def runtime_context(self, proposal_id: str, *, trial_id: str = "") -> MemoryExecutionContext:
        return self.plan.runtime_context(proposal_id, trial_id=trial_id)

    def finalize(
        self,
        database: Any,
        observations: Iterable[GenerationObservation],
        *,
        commit_function: Callable[[str | Path, Mapping[str, Any]], Dict[str, Any]] = (
            commit_internal_memory_update
        ),
        write_artifact: bool = True,
    ) -> Dict[str, Any]:


        if self._final_artifact is not None:
            return deepcopy(self._final_artifact)
        ordered = sorted(list(observations), key=lambda item: item.stable_key)
        reserved = {item.proposal_id: item for item in self.plan.reservations}
        seen: set[str] = set()
        for observation in ordered:
            if observation.proposal_id not in reserved:
                raise ValueError(f"unreserved generation observation: {observation.proposal_id}")
            if observation.proposal_id in seen:
                raise ValueError(f"duplicate generation observation: {observation.proposal_id}")
            seen.add(observation.proposal_id)
            reservation = reserved[observation.proposal_id]
            if observation.generation_id and observation.generation_id != self.plan.generation_id:
                raise ValueError("generation observation identity mismatch")
            if (
                observation.proposal_causal_envelope_hash
                and observation.proposal_causal_envelope_hash
                != reservation.proposal_causal_envelope_hash
            ):
                raise ValueError("generation observation causal envelope mismatch")
            observation.generation_id = self.plan.generation_id
            if not observation.proposal_causal_envelope_json:
                observation.proposal_causal_envelope_json = (
                    reservation.proposal_causal_envelope_json
                )
                observation.proposal_causal_envelope_hash = (
                    reservation.proposal_causal_envelope_hash
                )
        missing = sorted(set(reserved) - seen)
        if missing:
            raise ValueError(
                "generation barrier is incomplete; missing observations for " + ", ".join(missing)
            )


        for observation in ordered:
            child = observation.child_program
            if child is None:
                continue
            if getattr(child, "id", None) not in getattr(database, "programs", {}):
                database.add(
                    child,
                    iteration=observation.iteration,
                    target_island=observation.target_island,
                )
            if observation.artifacts:
                database.store_artifacts(child.id, observation.artifacts)
            if observation.prompt and hasattr(database, "log_prompt"):
                database.log_prompt(
                    program_id=child.id,
                    template_key="generation_user",
                    prompt=observation.prompt,
                    responses=[observation.llm_response] if observation.llm_response else [],
                )

        batch_rows = [
            {
                "proposal_id": observation.proposal_id,
                "program": observation.child_program,
                "parent": observation.parent,
                "artifacts": observation.artifacts,
                "iteration": observation.iteration,
                "error": observation.error,
            }
            for observation in ordered
        ]
        fact_publication = None
        if hasattr(database, "publish_optimizer_memory_facts"):
            fact_publication = database.publish_optimizer_memory_facts(
                batch_rows,
                generation_id=self.plan.generation_id,
                logical_time=self.plan.logical_time,
            )
        if fact_publication is None and hasattr(database, "record_optimizer_memory_batch"):
            database.record_optimizer_memory_batch(
                batch_rows,
                logical_time=self.plan.logical_time,
            )

        selectable = [item for item in ordered if item.child_program is not None]
        selection: Optional[Dict[str, Any]] = None
        selected_observation: Optional[GenerationObservation] = None
        if selectable:
            selection = select_feasibility_first(
                [
                    {
                        "candidate_id": item.proposal_id,
                        "raw_objective": _metric_objective(item.child_program),
                        "gate_sources": _gate_sources(item.child_program),
                    }
                    for item in selectable
                ],
                direction="maximize",
            )
            selected_id = selection["selected_candidate_id"]
            selected_observation = next(
                item for item in selectable if item.proposal_id == selected_id
            )

        commit: Dict[str, Any] = {
            "commit_count": 0,
            "committed": False,
            "reason": "no_evaluated_candidate",
        }
        if selection is not None and selection["all_infeasible"]:
            commit["reason"] = "all_candidates_infeasible"
        elif selected_observation is not None and not self.plan.memory_policy.may_commit_adaptive_prior:
            commit["reason"] = "adaptive_prior_policy_disallows_commit"
        elif selected_observation is not None and not self.plan.inner_target_path:
            commit["reason"] = "inner_memory_unconfigured"
        elif selected_observation is not None and not _has_high_fidelity_evidence(
            selected_observation.artifacts
        ):
            commit["reason"] = "selected_candidate_not_high_fidelity"
        elif selected_observation is not None:
            proposal = _selected_proposal(selected_observation.artifacts)
            if proposal is None:
                commit["reason"] = "selected_inner_proposal_missing"
            else:
                try:
                    _validate_selected_proposal(self.plan, selected_observation, proposal)
                    commit = dict(commit_function(self.plan.inner_target_path, proposal))
                    commit.setdefault("commit_count", 1 if commit.get("committed") else 0)
                    commit.setdefault("reason", "committed" if commit.get("committed") else "not_committed")
                except StaleMemorySnapshotError as exc:
                    commit = {
                        "commit_count": 0,
                        "committed": False,
                        "reason": "stale_inner_memory_snapshot",
                        "error": str(exc),
                    }
                except (MemoryProposalError, TypeError, ValueError) as exc:
                    commit = {
                        "commit_count": 0,
                        "committed": False,
                        "reason": "invalid_selected_inner_proposal",
                        "error": str(exc),
                    }

        outer_output = (
            database.get_optimizer_memory_snapshot()
            if hasattr(database, "get_optimizer_memory_snapshot")
            else {}
        )
        inner_output = (
            capture_memory_snapshot(self.plan.inner_target_path)
            if self.plan.inner_target_path
            else None
        )
        optimizer_store = getattr(database, "optimizer_memory", None)
        optimizer_path = getattr(optimizer_store, "path", None)
        artifact: Dict[str, Any] = {
            "schema_version": GENERATION_LIFECYCLE_VERSION,
            "generation_id": self.plan.generation_id,
            "logical_time": self.plan.logical_time,
            "ordered_proposal_ids": [item.proposal_id for item in ordered],
            "prompt_snapshot_hashes": {
                item.proposal_id: self.plan.outer_input_hash for item in self.plan.reservations
            },
            "hierarchical_design_hash": (
                self.plan.hierarchical_design_hash or None
            ),
            "outer_memory": {
                "input_hash": self.plan.outer_input_hash,
                "output_hash": outer_memory_hash(outer_output if isinstance(outer_output, Mapping) else {}),
                "output_file_hash": _file_hash(optimizer_path),
            },
            "inner_memory": {
                "input_content_hash": self.plan.inner_memory.content_hash if self.plan.inner_memory else None,
                "input_raw_hash": self.plan.inner_memory.raw_hash if self.plan.inner_memory else None,
                "output_content_hash": inner_output.content_hash if inner_output else None,
                "output_raw_hash": inner_output.raw_hash if inner_output else None,
                "target_path": self.plan.inner_target_path or None,
            },
            "winner": {
                "proposal_id": selected_observation.proposal_id if selected_observation else None,
                "program_id": (
                    getattr(selected_observation.child_program, "id", None)
                    if selected_observation
                    else None
                ),
            },
            "selection": selection,
            "commit": commit,
            "observation_count": len(ordered),
        }
        hierarchical_publication: Dict[str, Any] = {"status": "not_configured"}
        if hasattr(database, "record_hierarchical_generation"):
            try:
                result = database.record_hierarchical_generation(self.plan, ordered)
                if isinstance(result, Mapping):
                    hierarchical_publication = deepcopy(dict(result))
                else:
                    hierarchical_publication = {"status": "published"}
            except Exception as exc:
                hierarchical_publication = {
                    "status": "failed_after_core_commit",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }
                logger.exception(
                    "Hierarchical generation publication failed after core wave commit"
                )
        artifact["hierarchical_publication"] = hierarchical_publication
        if selected_observation is not None and selected_observation.child_program is not None:
            if hasattr(database, "store_artifacts"):
                existing_artifacts = (
                    database.get_artifacts(selected_observation.child_program.id)
                    if hasattr(database, "get_artifacts")
                    else {}
                )
                database.store_artifacts(
                    selected_observation.child_program.id,
                    {
                        **(existing_artifacts or {}),
                        "generation_memory_lifecycle": artifact,
                    },
                )
        if write_artifact:
            artifact_path = (
                Path(self.plan.output_root)
                / _safe_component(self.plan.generation_id)
                / "generation_memory_lifecycle.json"
            )
            _atomic_write_json(artifact_path, artifact)
            artifact["artifact_path"] = str(artifact_path)
        self._final_artifact = deepcopy(artifact)
        return deepcopy(artifact)


def generation_chunks(
    start_iteration: int,
    max_iterations: int,
    wave_size: int,
) -> Tuple[Tuple[int, ...], ...]:


    size = max(1, int(wave_size))
    values = tuple(range(int(start_iteration), int(start_iteration) + int(max_iterations)))
    return tuple(values[index : index + size] for index in range(0, len(values), size))


def target_island_for_iteration(iteration: int, num_islands: int) -> int:


    count = int(num_islands)
    if count < 1:
        raise ValueError("num_islands must be positive")
    return (int(iteration) - 1) % count


__all__ = [
    "GENERATION_LIFECYCLE_VERSION",
    "GenerationCoordinator",
    "GenerationObservation",
    "GenerationPlan",
    "GenerationReservation",
    "generation_chunks",
    "outer_memory_hash",
    "target_island_for_iteration",
]
