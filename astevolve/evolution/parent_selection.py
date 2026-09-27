

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Tuple

from astevolve.domain import DesignStrategy
from astevolve.domain.dual_ast import ExecutableDualAST

from .archive import ArchiveProjection, ArchiveSnapshot
from .domain import DUAL_AST_REVISION, STRATEGY_REVISION, Proposal
from .orchestrator import ProposalContext, ProposalDraft, Revision


PARENT_SELECTION_POLICY_VERSION = "astevolve.evolution.parent_selection_policy.v1"
PARENT_SELECTION_PROVENANCE_VERSION = (
    "astevolve.evolution.parent_selection_provenance.v1"
)


class ParentSelectionError(ValueError):
    pass


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ParentSelectionError(f"{name} must be a non-empty normalized string")
    return value


def _kind(revision: Revision) -> str:
    if isinstance(revision, DesignStrategy):
        return STRATEGY_REVISION
    if isinstance(revision, ExecutableDualAST):
        return DUAL_AST_REVISION
    raise TypeError("parent revision must be DesignStrategy or ExecutableDualAST")


def _detach(revision: Revision) -> Revision:
    if isinstance(revision, DesignStrategy):
        return DesignStrategy.from_mapping(revision.to_legacy_dict())
    if isinstance(revision, ExecutableDualAST):
        return ExecutableDualAST.from_mapping(revision.to_dict())
    raise TypeError("parent revision must be DesignStrategy or ExecutableDualAST")


@dataclass(frozen=True)
class ParentSelectionPolicy:


    policy_id: str
    elite_probability: float = 0.5
    max_pool_size: int = 20
    schema_version: str = PARENT_SELECTION_POLICY_VERSION

    def __post_init__(self) -> None:
        _text(self.policy_id, "policy_id")
        if self.schema_version != PARENT_SELECTION_POLICY_VERSION:
            raise ParentSelectionError("unsupported parent-selection policy schema")
        if (
            isinstance(self.elite_probability, bool)
            or not isinstance(self.elite_probability, (int, float))
            or not math.isfinite(float(self.elite_probability))
            or not 0.0 <= float(self.elite_probability) <= 1.0
        ):
            raise ParentSelectionError("elite_probability must be within [0, 1]")
        if (
            isinstance(self.max_pool_size, bool)
            or not isinstance(self.max_pool_size, int)
            or self.max_pool_size <= 0
        ):
            raise ParentSelectionError("max_pool_size must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "elite_probability": float(self.elite_probability),
            "max_pool_size": self.max_pool_size,
        }


def _choice(seed: int, policy: ParentSelectionPolicy, count: int) -> int:
    if count <= 1:
        return 0
    digest = hashlib.sha256(
        (f"astevolve.evolution.parent_choice.v1\0{policy.policy_id}\0{seed}").encode(
            "utf-8"
        )
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    if unit < float(policy.elite_probability):
        return 0
    return 1 + (int.from_bytes(digest[8:16], "big") % (count - 1))


@dataclass(frozen=True)
class ArchiveParentSelector:


    archive: ArchiveProjection
    initial_revisions: Tuple[Revision, ...]
    policy: ParentSelectionPolicy
    initial_parent_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.archive, ArchiveProjection):
            raise TypeError("archive must be ArchiveProjection")
        if not isinstance(self.initial_revisions, tuple) or not self.initial_revisions:
            raise ParentSelectionError("initial_revisions must be a non-empty tuple")
        revisions = tuple(_detach(revision) for revision in self.initial_revisions)
        kinds = {_kind(revision) for revision in revisions}
        if len(kinds) != 1:
            raise ParentSelectionError("initial parent revisions must use one kind")
        if self.initial_parent_ids and len(self.initial_parent_ids) != len(revisions):
            raise ParentSelectionError(
                "initial_parent_ids must align with initial_revisions"
            )
        parent_ids = tuple(
            _text(value, "initial parent id") for value in self.initial_parent_ids
        )
        if not isinstance(self.policy, ParentSelectionPolicy):
            raise TypeError("policy must be ParentSelectionPolicy")
        object.__setattr__(self, "initial_revisions", revisions)
        object.__setattr__(self, "initial_parent_ids", parent_ids)

    @property
    def kind(self) -> str:
        return _kind(self.initial_revisions[0])

    def _initial(self, context: ProposalContext) -> ProposalDraft:
        index = context.slot % len(self.initial_revisions)
        parent_id = (
            self.initial_parent_ids[index]
            if self.initial_parent_ids
            else f"initial-parent:{index:04d}"
        )
        return ProposalDraft(
            parent_id=parent_id,
            revision=self.initial_revisions[index],
            provenance={
                "schema_version": PARENT_SELECTION_PROVENANCE_VERSION,
                "mechanism": "initial_revision",
                "policy": self.policy.to_dict(),
                "initial_index": index,
            },
        )

    def _proposal_index(self, context: ProposalContext) -> dict[str, Proposal]:
        proposals: dict[str, Proposal] = {}
        for manifest in context.prior_manifests:
            for proposal in manifest.proposals:
                existing = proposals.get(proposal.proposal_id)
                if (
                    existing is not None
                    and existing.proposal_hash != proposal.proposal_hash
                ):
                    raise ParentSelectionError("historical proposal identity conflict")
                proposals[proposal.proposal_id] = proposal
        return proposals

    def _select_from_view(
        self,
        context: ProposalContext,
        *,
        snapshot: ArchiveSnapshot,
        proposal_index: dict[str, Proposal],
    ) -> ProposalDraft:
        if snapshot.revision != context.generation_index:
            raise ParentSelectionError(
                "archive revision does not match the sealed prior generation count"
            )
        if not snapshot.cells:
            return self._initial(context)
        target_island = context.slot_seed % len(snapshot.islands)
        island_ids = set(snapshot.islands[target_island])
        candidates = [
            cell.candidate
            for cell in snapshot.cells
            if cell.candidate.candidate_id in island_ids
            and cell.candidate.candidate_id not in snapshot.tombstones
        ]
        if not candidates:
            candidates = [
                cell.candidate
                for cell in snapshot.cells
                if cell.candidate.candidate_id not in snapshot.tombstones
            ]
        compatible = []
        for candidate in candidates:
            proposal = proposal_index.get(candidate.proposal_id)
            if proposal is None:
                raise ParentSelectionError(
                    "archive candidate has no proposal in sealed history"
                )
            if proposal.kind == self.kind:
                compatible.append((candidate, proposal))
        if not compatible:
            return self._initial(context)
        ranked = sorted(
            compatible, key=lambda item: self.archive.candidate_order_key(item[0])
        )[
            : self.policy.max_pool_size
        ]
        candidate, proposal = ranked[
            _choice(context.slot_seed, self.policy, len(ranked))
        ]
        payload = proposal.payload()
        revision: Revision = (
            DesignStrategy.from_mapping(payload)
            if proposal.kind == STRATEGY_REVISION
            else ExecutableDualAST.from_mapping(payload)
        )
        return ProposalDraft(
            parent_id=candidate.candidate_id,
            revision=revision,
            provenance={
                "schema_version": PARENT_SELECTION_PROVENANCE_VERSION,
                "mechanism": "archive_island_selection",
                "policy": self.policy.to_dict(),
                "archive_revision": snapshot.revision,
                "target_island": target_island,
                "selected_candidate_id": candidate.candidate_id,
                "selected_proposal_id": proposal.proposal_id,
                "selected_evaluation_hash": candidate.evaluation_hash,
                "selected_objective": candidate.objective,
                "selected_objective_direction": (
                    self.archive.config.objective_direction
                ),
                "selected_feasible": candidate.feasible,
            },
        )

    def select_many(
        self, contexts: Tuple[ProposalContext, ...]
    ) -> Tuple[ProposalDraft, ...]:


        if not isinstance(contexts, tuple):
            raise TypeError("contexts must be a ProposalContext tuple")
        if not contexts:
            return ()
        for context in contexts:
            if not isinstance(context, ProposalContext):
                raise TypeError("contexts must contain ProposalContext values")
        first = contexts[0]
        signature = (
            first.run_id,
            first.generation_index,
            first.generation_id,
            first.input_snapshot.snapshot_hash,
            first.generation_input.input_hash,
            tuple(item.commit_hash for item in first.prior_commits),
            tuple(item.manifest_hash for item in first.prior_manifests),
        )
        for context in contexts[1:]:
            observed = (
                context.run_id,
                context.generation_index,
                context.generation_id,
                context.input_snapshot.snapshot_hash,
                context.generation_input.input_hash,
                tuple(item.commit_hash for item in context.prior_commits),
                tuple(item.manifest_hash for item in context.prior_manifests),
            )
            if observed != signature:
                raise ParentSelectionError(
                    "batched parent contexts do not share one sealed generation view"
                )
        if first.generation_index == 0:
            return tuple(self._initial(context) for context in contexts)
        snapshot = self.archive.snapshot()
        proposal_index = self._proposal_index(first)
        return tuple(
            self._select_from_view(
                context,
                snapshot=snapshot,
                proposal_index=proposal_index,
            )
            for context in contexts
        )

    def __call__(self, context: ProposalContext) -> ProposalDraft:
        if not isinstance(context, ProposalContext):
            raise TypeError("context must be ProposalContext")
        return self.select_many((context,))[0]


__all__ = [
    "ArchiveParentSelector",
    "PARENT_SELECTION_POLICY_VERSION",
    "PARENT_SELECTION_PROVENANCE_VERSION",
    "ParentSelectionError",
    "ParentSelectionPolicy",
]
