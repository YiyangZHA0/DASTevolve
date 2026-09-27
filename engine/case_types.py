

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, MutableMapping, Optional, Tuple

import numpy as np

from astevolve.core.protein_lang import Blueprint
from astevolve.domain import (
    CompiledDesign,
    CompiledDesignAction,
    CompiledPortfolioOptimizationRequest,
    DesignStrategy,
)
from astevolve.search.run_memory import InnerRunMemory
from .ablation_control import (
    AblationControlContract,
    build_ablation_control_contract,
)
from .causal_flow import (
    EffectiveContractDiff,
    EffectiveSearchContract,
    GraphPatch,
)
from .memory_lifecycle import MemorySnapshot, ScopedAdaptivePriorSnapshot
from .memory_policy import MemoryPolicyConfig, MemoryPolicyError, MemoryScope
from .mapping_compiler import (
    ExecutableDualASTCompilation,
    ExecutableMappingPlan,
    EffectiveMappingSchedule,
)
from .node_compiler import ExecutableNodePlan

if TYPE_CHECKING:
    from astevolve.search.config import SAConfig


ConstraintPayload = Dict[str, Any]
SearchConfigPayload = Dict[str, Any]
ScoreConfigPayload = Dict[str, Any]
DesignStatePayload = Dict[str, Any]
MaskPayload = Dict[str, List[bool]]
TemplateSequences = Dict[str, str]
FixedResidues = Dict[str, Dict[int, str]]

LegacyCaseInputs = Tuple[
    Blueprint,
    List[ConstraintPayload],
    SearchConfigPayload,
    MaskPayload,
    TemplateSequences,
    FixedResidues,
    ScoreConfigPayload,
    DesignStatePayload,
]


@dataclass(frozen=True)
class PreparedCaseInputs:


    blueprint: Blueprint
    constraint_specs: List[ConstraintPayload]
    search_config: SearchConfigPayload
    masks: MaskPayload
    immutable_reference_sequences: TemplateSequences
    template_sequences: TemplateSequences
    fixed_residues: FixedResidues
    score_config: ScoreConfigPayload
    design_state: DesignStatePayload
    strategy: DesignStrategy
    resolved_strategy: Dict[str, Any]
    memory_snapshot: MemorySnapshot
    dual_ast_compilation: ExecutableDualASTCompilation
    executable_node_plan: ExecutableNodePlan
    executable_mapping_plan: ExecutableMappingPlan
    effective_mapping_schedule: EffectiveMappingSchedule
    compiled_design_action: Optional[CompiledDesignAction]
    graph_patch: GraphPatch
    parent_effective_search_contract: EffectiveSearchContract
    effective_search_contract: EffectiveSearchContract
    contract_diff: EffectiveContractDiff
    accepted_patch_requires_effect: bool
    generation_id: str
    proposal_id: str
    trial_id: str
    compiled_portfolio_optimization_request: Optional[
        CompiledPortfolioOptimizationRequest
    ] = None

    def to_legacy_tuple(self) -> LegacyCaseInputs:


        return (
            self.blueprint,
            self.constraint_specs,
            self.search_config,
            self.masks,
            self.template_sequences,
            self.fixed_residues,
            self.score_config,
            self.design_state,
        )


@dataclass(frozen=True)
class SearchRequest:


    design: CompiledDesign
    legacy_compiled: MutableMapping[str, Any]
    constraint_specs: List[ConstraintPayload]
    config: "SAConfig"
    masks: Dict[str, np.ndarray]
    template_sequences: TemplateSequences
    fixed_residues: FixedResidues
    internal_memory: Dict[str, Any]
    adaptive_prior: Dict[str, Any]
    memory_snapshot: MemorySnapshot
    memory_scope: MemoryScope
    memory_policy: MemoryPolicyConfig
    inner_run_memory: InnerRunMemory
    score_config: ScoreConfigPayload
    design_state: DesignStatePayload
    dual_ast_compilation: ExecutableDualASTCompilation
    executable_node_plan: ExecutableNodePlan
    executable_mapping_plan: ExecutableMappingPlan
    effective_mapping_schedule: EffectiveMappingSchedule
    compiled_design_action: Optional[CompiledDesignAction]
    ablation_control: AblationControlContract
    graph_patch: GraphPatch
    parent_effective_search_contract: EffectiveSearchContract
    effective_search_contract: EffectiveSearchContract
    contract_diff: EffectiveContractDiff
    accepted_patch_requires_effect: bool
    generation_id: str
    proposal_id: str
    trial_id: str
    seed: Optional[int]
    compiled_portfolio_optimization_request: Optional[
        CompiledPortfolioOptimizationRequest
    ] = None

    @classmethod
    def from_prepared(
        cls,
        prepared: PreparedCaseInputs,
        *,
        config: "SAConfig",
        internal_memory: Optional[Dict[str, Any]] = None,
        memory_policy: Optional[MemoryPolicyConfig] = None,
        memory_scope: Optional[MemoryScope] = None,
        scoped_adaptive_snapshot: Optional[ScopedAdaptivePriorSnapshot] = None,
        inner_run_memory: Optional[InnerRunMemory] = None,
        run_instance_id: str = "direct-inner-run",
    ) -> "SearchRequest":


        legacy_compiled = prepared.blueprint.compile()
        legacy_compiled["_design_state"] = prepared.design_state
        if prepared.dual_ast_compilation.ast is not None:
            legacy_compiled["executable_dual_ast"] = (
                prepared.dual_ast_compilation.ast.to_dict()
            )
        legacy_compiled["compiled_executable_node_plan"] = (
            prepared.executable_node_plan.to_dict()
        )
        legacy_compiled["executable_mapping_plan"] = (
            prepared.executable_mapping_plan.to_dict(seed=config.seed)
        )
        legacy_compiled["effective_mapping_schedule"] = (
            prepared.effective_mapping_schedule.to_dict()
        )
        if prepared.compiled_design_action is not None:
            legacy_compiled["compiled_design_action"] = (
                prepared.compiled_design_action.to_artifact()
            )
        if prepared.compiled_portfolio_optimization_request is not None:
            legacy_compiled["compiled_portfolio_optimization_request"] = (
                prepared.compiled_portfolio_optimization_request.to_artifact()
            )
        ablation_control = build_ablation_control_contract(
            prepared.effective_search_contract,
            parent_sequences=prepared.template_sequences,
            memory_snapshot_hash=prepared.memory_snapshot.content_hash,
            seed=config.seed,
        )
        legacy_compiled["ablation_control"] = ablation_control.to_dict()
        design = CompiledDesign.from_legacy(
            legacy_compiled,
            template_sequences=prepared.template_sequences,
            masks=prepared.masks,
            fixed_residues=prepared.fixed_residues,
        )


        _ = internal_memory
        policy = memory_policy or MemoryPolicyConfig()
        scope = memory_scope or MemoryScope(
            case_id=str(
                prepared.design_state.get("case_id")
                or prepared.design_state.get("task_name")
                or "direct_case"
            ),
            run_id="direct_run",
            lineage_id="direct_lineage",
        )
        if scoped_adaptive_snapshot is not None:
            if (
                scoped_adaptive_snapshot.snapshot.content_hash
                != prepared.memory_snapshot.content_hash
            ):
                raise MemoryPolicyError(
                    "scoped adaptive snapshot does not match the prepared immutable snapshot"
                )
            adaptive_prior = scoped_adaptive_snapshot.effective_prior(policy, scope)
        elif policy.may_read_adaptive_prior:
            raise MemoryPolicyError(
                "read_only/winner_commit policy requires a scoped adaptive snapshot"
            )
        else:
            adaptive_prior = {}
        run_memory = inner_run_memory or InnerRunMemory(
            scope=scope,
            run_instance_id=str(run_instance_id or "direct-inner-run"),
        )
        scope.require_compatible(run_memory.scope, level="lineage")
        return cls(
            design=design,
            legacy_compiled=legacy_compiled,
            constraint_specs=prepared.constraint_specs,
            config=config,
            masks={
                chain_id: np.asarray(mask, dtype=bool)
                for chain_id, mask in prepared.masks.items()
            },
            template_sequences=prepared.template_sequences,
            fixed_residues=prepared.fixed_residues,
            internal_memory=deepcopy(adaptive_prior),
            adaptive_prior=deepcopy(adaptive_prior),
            memory_snapshot=prepared.memory_snapshot,
            memory_scope=scope,
            memory_policy=policy,
            inner_run_memory=run_memory,
            score_config=prepared.score_config,
            design_state=prepared.design_state,
            dual_ast_compilation=prepared.dual_ast_compilation,
            executable_node_plan=prepared.executable_node_plan,
            executable_mapping_plan=prepared.executable_mapping_plan,
            effective_mapping_schedule=prepared.effective_mapping_schedule,
            compiled_design_action=prepared.compiled_design_action,
            compiled_portfolio_optimization_request=(
                prepared.compiled_portfolio_optimization_request
            ),
            ablation_control=ablation_control,
            graph_patch=prepared.graph_patch,
            parent_effective_search_contract=prepared.parent_effective_search_contract,
            effective_search_contract=prepared.effective_search_contract,
            contract_diff=prepared.contract_diff,
            accepted_patch_requires_effect=prepared.accepted_patch_requires_effect,
            generation_id=prepared.generation_id,
            proposal_id=prepared.proposal_id,
            trial_id=prepared.trial_id,
            seed=config.seed,
        )
