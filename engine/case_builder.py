

from __future__ import annotations


del annotations

from copy import deepcopy as _deepcopy
from typing import Any as _Any, Dict as _Dict, Optional as _Optional

from astevolve import domain as _domain
from astevolve.evaluation.plugins import registry as _plugin_registry
from astevolve.runtime import edit_contract_lifecycle as _contract_lifecycle
from astevolve import semantic_graph as _semantic_graph
from astevolve.semantic_graph import residue_design_context as _residue_context

from . import case_resources as _case_resources
from . import case_types as _case_types
from . import causal_flow as _causal_flow
from . import causal_runtime as _causal_runtime
from . import design_compiler as _design_compiler
from . import design_action_compiler as _design_action_compiler
from . import design_state as _design_state
from . import experiment_identity as _experiment_identity
from . import memory_lifecycle as _memory_lifecycle
from . import mapping_compiler as _mapping_compiler
from . import node_compiler as _node_compiler
from . import parent_lineage as _parent_lineage
from . import runtime_profile as _runtime_profile
from . import strategy_compiler as _strategy_compiler
from . import strategy_effect_report as _strategy_effect_report


def _controller_protected_residues(
    state: _Dict[str, _Any],
    sequences: _Dict[str, str],
) -> _Dict[str, tuple[int, ...]]:


    raw_policy = state.get("global_ast_evolution_policy") or {}
    raw_spans = (
        raw_policy.get("protected_chain_spans", {})
        if isinstance(raw_policy, dict)
        else {}
    )
    if not isinstance(raw_spans, dict):
        raise _design_action_compiler.DesignActionCompileError(
            "protected_residues_invalid",
            "global_ast_evolution_policy.protected_chain_spans",
        )
    protected: _Dict[str, tuple[int, ...]] = {}
    for raw_chain, spans in raw_spans.items():
        chain = str(raw_chain)
        if chain not in sequences or not isinstance(spans, list):
            raise _design_action_compiler.DesignActionCompileError(
                "protected_residues_invalid", chain
            )
        positions: set[int] = set()
        for raw_span in spans:
            if (
                not isinstance(raw_span, list)
                or len(raw_span) != 2
                or isinstance(raw_span[0], bool)
                or isinstance(raw_span[1], bool)
                or not isinstance(raw_span[0], int)
                or not isinstance(raw_span[1], int)
            ):
                raise _design_action_compiler.DesignActionCompileError(
                    "protected_residues_invalid", f"{chain}:{raw_span!r}"
                )
            start, end = raw_span
            if start < 0 or end <= start or end > len(sequences[chain]):
                raise _design_action_compiler.DesignActionCompileError(
                    "protected_residues_invalid", f"{chain}:{start}:{end}"
                )
            overlap = positions.intersection(range(start, end))
            if overlap:
                raise _design_action_compiler.DesignActionCompileError(
                    "protected_residue_duplicate",
                    f"{chain}:{min(overlap)}",
                )
            positions.update(range(start, end))
        protected[chain] = tuple(sorted(positions))
    return dict(sorted(protected.items()))


def prepare_case_inputs(
    strategy: _Dict[str, _Any],
    design_state_path: _Optional[str] = None,
    memory_path: _Optional[str] = None,
    memory_snapshot: _Optional[_memory_lifecycle.MemorySnapshot] = None,
    mapping_execution_mode: str = "full",
    runtime_mcts_output_dir: _Optional[str] = None,
) -> _case_types.PreparedCaseInputs:


    execution_scope = _memory_lifecycle.current_memory_execution_context()
    design_action = (
        execution_scope.typed_design_action()
        if execution_scope is not None
        else None
    )
    design_action_parent_binding = (
        execution_scope.design_action_parent_binding()
        if execution_scope is not None
        else None
    )
    if (design_action is None) != (design_action_parent_binding is None):
        raise _design_action_compiler.DesignActionCompileError(
            "design_action_runtime_binding_incomplete"
        )
    strategy = _contract_lifecycle.inject_controller_contract_response(
        strategy, execution_scope.edit_contract_response() if execution_scope else None
    )
    causal_envelope = (
        execution_scope.proposal_causal_envelope()
        if execution_scope is not None
        else None
    )
    causal_envelope = causal_envelope if isinstance(causal_envelope, dict) else {}
    contract_lifecycle = None
    if execution_scope is not None and execution_scope.edit_contract_envelope_json:
        strategy, contract_lifecycle = _contract_lifecycle.resolve_parent_contract(
            strategy,
            execution_scope.edit_contract_envelope_json,
        )
    elif (
        execution_scope is not None
        and execution_scope.proposal_id
        and strategy.get("edit_contract") is not None
    ):
        raise _contract_lifecycle.EditContractLifecycleError(
            "child_supplied_edit_contract_forbidden",
            "outer proposals may apply only a contract offered by their selected parent",
        )
    requested_strategy = _deepcopy(strategy)
    (
        strategy,
        resume_template_seqs,
        parent_sequence_lineage,
        parent_effective_ast_lineage,
    ) = _parent_lineage.resolve_trusted_parent_inputs(
        execution_scope,
        causal_envelope,
        strategy,
    )
    state = _design_state.load_design_state(design_state_path)
    state = _parent_lineage.install_parent_effective_ast(
        state, parent_effective_ast_lineage
    )
    state = _residue_context.attach_residue_evidence_context(
        state,
        design_state_path=design_state_path,
        current_parent_sequences=resume_template_seqs,
    )
    case_sheet = _case_resources.load_case_sheet(state, design_state_path)
    if memory_snapshot is None:
        memory_snapshot = _memory_lifecycle.capture_memory_snapshot(
            _case_resources.resolve_memory_path(memory_path or state.get("memory_path"))
        )
    elif not isinstance(memory_snapshot, _memory_lifecycle.MemorySnapshot):
        raise TypeError("memory_snapshot must be a MemorySnapshot")


    memory_bias = _case_resources.extract_memory_bias({}, state)
    strategy = _strategy_compiler.sanitize_strategy_for_ast(state, strategy)
    strategy = _strategy_compiler.normalize_strategy_tree(state, strategy, memory_bias)
    policy_before_contract = _contract_lifecycle.policy_projection(strategy)
    strategy = _semantic_graph.apply_edit_contract_to_strategy(strategy)


    policy_after_contract = _contract_lifecycle.policy_projection(strategy)
    graph_ablation_mode = _runtime_profile.resolve_graph_ablation_mode(strategy)
    contract_response_report = _contract_lifecycle.summarize_contract_response(
        strategy
    )
    state = _strategy_compiler.apply_strategy_tree_to_state(state, strategy)
    state, ast_revision_report = _semantic_graph.apply_ast_revision_plan(
        state, strategy
    )
    ast_revision_report = _parent_lineage.bind_ast_revision_report(
        ast_revision_report, parent_effective_ast_lineage
    )
    state["_case_sheet"] = case_sheet
    state["_layout_summary"] = strategy.get("layout_summary", {})
    state["_node_edit_policies"] = strategy.get("node_edit_policies", {})
    state["_applied_edit_contract"] = strategy.get("edit_contract") if strategy.get("_edit_contract_applied") else None
    state["_last_contract_response"] = strategy.get("last_contract_response", {})
    state["_contract_response_report"] = contract_response_report
    state["_edit_contract_lifecycle"] = contract_lifecycle
    state["_graph_ablation_mode"] = graph_ablation_mode
    state["_adaptive_memory_compile_consumed"] = False
    state["_ast_revision_report"] = ast_revision_report
    state["_parent_sequence_lineage"] = _deepcopy(parent_sequence_lineage)

    bp = _design_compiler.build_blueprint(state)
    masks = _design_compiler.build_masks(state, memory_bias, strategy)
    templates = {
        state["binder"].get("chain_id", "BB"): _design_state.binder_sequence(state),
    }
    if _design_state.has_target(state):
        templates[state["target"].get("chain_id", "T")] = state["target"][
            "sequence"
        ]
    case_reference_templates = _deepcopy(templates)
    templates = _parent_lineage.apply_resume_template_sequences(
        templates, resume_template_seqs, state
    )
    fixed_residues = _design_compiler.build_fixed_residues(state, memory_bias)
    if (
        state.get("ast_evolution_policy") is not None
        or state.get("global_ast_evolution_policy") is not None
    ):
        _semantic_graph.validate_ast_revision_permissions(
            state.get("executable_dual_ast"),
            base_masks=masks,
            base_fixed_residues=fixed_residues,
        )
    dual_ast_compilation = _mapping_compiler.compile_executable_dual_ast(
        state.get("executable_dual_ast"),
        compiled=bp.compile(),
        template_sequences=templates,
        base_masks=masks,
        base_fixed_residues=fixed_residues,
        evaluator_capabilities=state.get("evaluator_capabilities"),
        execution_mode=mapping_execution_mode,
    )
    executable_node_plan = dual_ast_compilation.node_plan
    executable_mapping_plan = dual_ast_compilation.mapping_plan
    masks = {
        chain_id: list(mask)
        for chain_id, mask in executable_node_plan.effective_masks.items()
    }
    fixed_residues = {
        chain_id: dict(residues)
        for chain_id, residues in executable_node_plan.effective_fixed_residues.items()
    }
    retired_position_policy_audit = None
    if design_action is None:


        templates, retired_position_policy_audit = (
            _parent_lineage.apply_retired_position_policy(
                templates,
                case_reference_templates=case_reference_templates,
                masks=masks,
                raw_policy=state.get("case_owned_residue_policy"),
                current_node_owners=_parent_lineage.mutation_owners_from_dual_ast(
                    state.get("executable_dual_ast")
                ),
                parent_node_owners=_parent_lineage.mutation_owners_from_dual_ast(
                    parent_effective_ast_lineage.get("executable_dual_ast")
                    if isinstance(parent_effective_ast_lineage, dict)
                    else None
                ),
            )
        )
    if retired_position_policy_audit is not None:
        state["_retired_position_policy_audit"] = _deepcopy(
            retired_position_policy_audit
        )
    strategy = _node_compiler.merge_node_policy_patches(
        strategy,
        executable_node_plan,
        state.get("_global_ast_node_metadata"),
    )
    strategy = _node_compiler.apply_case_owned_residue_policy(
        strategy,
        executable_node_plan,
        state.get("case_owned_residue_policy"),
    )
    sa_config = _runtime_profile.build_sa_config(
        strategy,
        runtime_mcts_output_dir=runtime_mcts_output_dir,
    )
    compiled_design_action = None
    compiled_portfolio_optimization_request = None
    if design_action is not None:
        action_plan = design_action.to_payload()["ast_revision_plan"]
        runtime_plan = strategy.get("ast_revision_plan")
        if runtime_plan != action_plan:
            raise _design_action_compiler.DesignActionCompileError(
                "ast_revision_plan_runtime_mismatch"
            )
        raw_parent_contract = causal_envelope.get("parent_effective_contract")
        if not isinstance(raw_parent_contract, dict):
            raise _design_action_compiler.DesignActionCompileError(
                "parent_effective_contract_missing"
            )
        binding = dict(design_action_parent_binding or {})
        trusted_case_id = str(
            state.get("case_id") or state.get("task_name") or ""
        )
        trusted_parent_program_id = str(
            causal_envelope.get("parent_program_id") or ""
        )
        immutable_identity = _experiment_identity.SequenceBundleIdentity.create(
            case_reference_templates
        )
        compile_kwargs = dict(
            immutable_sequences=case_reference_templates,
            immutable_sequence_bundle_hash=(
                immutable_identity.sequence_bundle_hash
            ),
            parent_sequences=templates,
            parent_effective_contract=raw_parent_contract,
            trusted_case_id=trusted_case_id,
            trusted_parent_program_id=trusted_parent_program_id,
            trusted_parent_candidate_id=str(
                binding.get("parent_candidate_id") or ""
            ),
            trusted_parent_evolve_hash=str(
                binding.get("parent_evolve_hash") or ""
            ),
            executable_node_plan=executable_node_plan,
            hard_allowed_residues=(
                strategy.get("residue_mutation_contract") or {}
            ),
            fixed_residues=fixed_residues,
            protected_residues=_controller_protected_residues(
                state, case_reference_templates
            ),
            max_total_mutations=int(sa_config["max_total_mutations"]),
        )
        has_step3_portfolio_intent = bool(
            design_action.mutation_modules
            or design_action.sequence_seeds
            or design_action.candidate_portfolio
        )
        if has_step3_portfolio_intent:
            (
                compiled_design_action,
                compiled_portfolio_optimization_request,
            ) = _design_action_compiler.compile_design_action_with_portfolio(
                design_action,
                **compile_kwargs,
            )
        else:
            compiled_design_action = _design_action_compiler.compile_design_action(
                design_action,
                **compile_kwargs,
            )
        templates = dict(compiled_design_action.reconciled_root_sequences)
        if compiled_design_action.position_distributions:
            _design_action_compiler.validate_compiled_position_distribution_optimizer_support(
                compiled_design_action,
                top_k_per_position=int(
                    sa_config.get("node_optimizer_top_k_per_position", 4)
                ),
            )
            sa_config["compiled_position_distribution_policy"] = {
                "schema_version": (
                    "astevolve.compiled_position_distribution_policy.v1"
                ),
                "compiled_design_action_hash": (
                    compiled_design_action.compiled_design_action_hash
                ),
                "distribution_set_hash": (
                    compiled_design_action.position_distribution_set_hash
                ),
                "rows": [
                    item.to_dict()
                    for item in compiled_design_action.position_distributions
                ],
            }
        state["_design_action"] = design_action.to_dict()
        state["_compiled_design_action"] = (
            compiled_design_action.to_artifact()
        )
        state["_mutation_ownership_ledger"] = (
            compiled_design_action.ownership_ledger.to_dict()
        )
        if compiled_portfolio_optimization_request is not None:
            portfolio_artifact = (
                compiled_portfolio_optimization_request.to_artifact()
            )
            state["_compiled_portfolio_optimization_request"] = (
                portfolio_artifact
            )
            sa_config["compiled_portfolio_request_policy"] = {
                "schema_version": (
                    "astevolve.compiled_portfolio_request_policy.v1"
                ),
                "compiled_portfolio_request_hash": (
                    compiled_portfolio_optimization_request
                    .compiled_portfolio_request_hash
                ),
                "candidate_slot_hashes": sorted(
                    slot.slot_hash
                    for slot in compiled_portfolio_optimization_request
                    .candidate_slots
                ),
            }


    strategy.pop("executable_ast_nodes", None)
    effective_mapping_schedule = (
        _mapping_compiler.compile_effective_mapping_schedule(
            executable_mapping_plan,
            node_policies=strategy.get("node_edit_policies", {}),
        )
    )
    if contract_lifecycle is not None:
        contract_lifecycle = _contract_lifecycle.finalize_selected_parent_contract(
            contract_lifecycle,
            before_policy=policy_before_contract,
            after_policy=policy_after_contract,
            causal_envelope=causal_envelope,
            ast_revision_report=ast_revision_report,
        )
        state["_edit_contract_lifecycle"] = contract_lifecycle
        applied_record = contract_lifecycle.get("applied")
        if (
            isinstance(applied_record, dict)
            and applied_record.get("status") == "applied"
        ):
            state["_applied_edit_contract"] = strategy.get("edit_contract")
    state["_node_edit_policies"] = strategy.get("node_edit_policies", {})
    state["_executable_dual_ast"] = dual_ast_compilation.to_dict()
    state["_effective_mapping_schedule"] = effective_mapping_schedule.to_dict()
    if not strategy.get("progen_chains"):


        default_progen_chains = [
            str(chain_id)
            for chain_id, sequence in templates.items()
            if sequence and any(bool(value) for value in masks.get(chain_id, []))
        ]
        if not default_progen_chains and templates:
            default_progen_chains = [str(next(iter(templates.keys())))]
        strategy["progen_chains"] = default_progen_chains
    constraint_specs = _design_compiler.build_constraint_specs(
        state, memory_bias, strategy
    )
    score_config = _runtime_profile.build_score_config(strategy)
    if design_action is not None and compiled_design_action is not None:
        score_config["design_action"] = design_action.to_dict()
        score_config["compiled_design_action"] = (
            compiled_design_action.to_artifact()
        )
        score_config["design_action_validation_policy"] = {
            "schema_version": "astevolve.design_action_validation_policy.v1",
            "enforcement": "pre_search_and_pre_provider",
            "silent_clipping_allowed": False,
            "implicit_lineage_allowed": False,
        }
        if compiled_portfolio_optimization_request is not None:
            score_config["compiled_portfolio_optimization_request"] = (
                compiled_portfolio_optimization_request.to_artifact()
            )
            score_config["portfolio_validation_policy"] = {
                "schema_version": "astevolve.portfolio_validation_policy.v1",
                "enforcement": "pre_search_and_pre_provider",
                "atomic_partial_realization_allowed": False,
                "portfolio_role_text_trusted": False,
            }
    score_config["mutation_scope_contract"] = (
        _causal_runtime.compile_mutation_scope_contract(
            dual_ast_compilation=dual_ast_compilation,
            executable_node_plan=executable_node_plan,
            masks=masks,
            search_config=sa_config,
            ast_revision_report=ast_revision_report,
        )
    )


    score_config["case_owned_residue_policy_tier"] = str(
        strategy.get("case_owned_residue_policy_tier") or ""
    )
    score_config["case_owned_residue_policy_resolution"] = _deepcopy(
        strategy.get("case_owned_residue_policy_resolution") or {}
    )
    score_config["residue_mutation_contract"] = _deepcopy(
        strategy.get("residue_mutation_contract") or {}
    )
    if retired_position_policy_audit is not None:
        score_config["retired_position_policy_audit"] = _deepcopy(
            retired_position_policy_audit
        )
    score_config["measurement_intents"] = [
        intent.to_dict() for intent in executable_node_plan.measurement_intents
    ]
    score_config["mapping_measurement_specs"] = [
        item.to_dict() for item in executable_mapping_plan.measurement_specs
    ]
    score_config["executable_mapping_plan"] = executable_mapping_plan.to_dict(
        seed=None
    )
    score_config["effective_mapping_schedule"] = (
        effective_mapping_schedule.to_dict()
    )
    evaluator_plugin_resolution = _plugin_registry.preflight_evaluator_plugins(
        state, score_config
    )
    score_config["plugin_resolution"] = evaluator_plugin_resolution
    state["_evaluator_plugin_resolution"] = evaluator_plugin_resolution
    strategy["evaluator_plugin_resolution"] = evaluator_plugin_resolution
    strategy_report = _strategy_effect_report.build_strategy_effect_report(
        requested_strategy,
        strategy,
        search_config=sa_config,
        score_config=score_config,
        graph_ablation_mode=graph_ablation_mode,
        legacy_summary=strategy.get("strategy_schema_report", {}),
    )
    strategy["strategy_schema_report"] = strategy_report
    state["_strategy_schema_report"] = strategy_report
    if retired_position_policy_audit is not None:
        strategy["retired_position_policy_audit"] = _deepcopy(
            retired_position_policy_audit
        )

    generation_id = str(execution_scope.generation_id if execution_scope else "")
    proposal_id = str(
        (execution_scope.proposal_id if execution_scope else "")
        or f"direct::{state.get('case_id') or state.get('task_name') or 'case'}"
    )
    trial_id = str(execution_scope.trial_id if execution_scope else "")
    parent_program_id = str(
        causal_envelope.get("parent_program_id")
        or state.get("case_id")
        or state.get("task_name")
        or "direct-parent"
    )
    compiled_for_contract = bp.compile()
    effective_contract = _causal_runtime.build_effective_search_contract(
        masks=masks,
        fixed_residues=fixed_residues,
        search_config=sa_config,
        constraint_specs=constraint_specs,
        score_config=score_config,
        design_state=state,
        compiled=compiled_for_contract,
        proposal_id=proposal_id,
        parent_program_id=parent_program_id,
        run_id=str(execution_scope.scope_id if execution_scope else "direct-inner-run"),
        logical_time=str(execution_scope.logical_time if execution_scope else ""),
    )
    hypothesis = str(
        strategy.get("hypothesis")
        or ((state.get("_applied_edit_contract") or {}).get("rationale") if isinstance(state.get("_applied_edit_contract"), dict) else "")
        or ((strategy.get("last_contract_response") or {}).get("reason") if isinstance(strategy.get("last_contract_response"), dict) else "")
        or "compiled strategy changes inner search behavior"
    )
    graph_patch = _causal_runtime.build_graph_patch(
        strategy_report=strategy_report,
        proposal_id=proposal_id,
        parent_program_id=parent_program_id,
        hypothesis=hypothesis,
    )
    raw_parent_contract = causal_envelope.get("parent_effective_contract")
    is_bound_outer_proposal = bool(
        execution_scope is not None
        and str(execution_scope.generation_id or "").startswith("generation-")
        and ":proposal:" in str(execution_scope.proposal_id or "")
        and causal_envelope
    )
    if is_bound_outer_proposal and not isinstance(raw_parent_contract, dict):
        raise _causal_flow.CausalFlowContractError(
            "parent_effective_contract_missing",
            str(execution_scope.proposal_id),
        )
    accepted_patch_requires_effect = isinstance(raw_parent_contract, dict)
    parent_effective_contract = (
        _causal_flow.EffectiveSearchContract.from_mapping(raw_parent_contract)
        if accepted_patch_requires_effect
        else effective_contract
    )
    raw_parent_patch = causal_envelope.get("parent_patch")
    if isinstance(raw_parent_patch, dict):
        state["_parent_graph_patch"] = _causal_flow.GraphPatch.from_mapping(
            raw_parent_patch
        ).to_dict()
    contract_diff = _causal_flow.diff_effective_contract(
        parent_effective_contract, effective_contract
    )
    return _case_types.PreparedCaseInputs(
        blueprint=bp,
        constraint_specs=constraint_specs,
        search_config=sa_config,
        masks=masks,
        immutable_reference_sequences=case_reference_templates,
        template_sequences=templates,
        fixed_residues=fixed_residues,
        score_config=score_config,
        design_state=state,
        strategy=_domain.DesignStrategy.from_mapping(strategy),
        resolved_strategy=strategy,
        memory_snapshot=memory_snapshot,
        dual_ast_compilation=dual_ast_compilation,
        executable_node_plan=executable_node_plan,
        executable_mapping_plan=executable_mapping_plan,
        effective_mapping_schedule=effective_mapping_schedule,
        compiled_design_action=compiled_design_action,
        compiled_portfolio_optimization_request=(
            compiled_portfolio_optimization_request
        ),
        graph_patch=graph_patch,
        parent_effective_search_contract=parent_effective_contract,
        effective_search_contract=effective_contract,
        contract_diff=contract_diff,
        accepted_patch_requires_effect=accepted_patch_requires_effect,
        generation_id=generation_id,
        proposal_id=proposal_id,
        trial_id=trial_id,
    )


def build_case_inputs(
    strategy: _Dict[str, _Any],
    design_state_path: _Optional[str] = None,
    memory_path: _Optional[str] = None,
    memory_snapshot: _Optional[_memory_lifecycle.MemorySnapshot] = None,
    mapping_execution_mode: str = "full",
) -> _case_types.LegacyCaseInputs:


    return prepare_case_inputs(
        strategy,
        design_state_path=design_state_path,
        memory_path=memory_path,
        memory_snapshot=memory_snapshot,
        mapping_execution_mode=mapping_execution_mode,
    ).to_legacy_tuple()


def run_design_search(
    strategy: _Dict[str, _Any],
    seed: _Optional[int] = None,
    design_state_path: _Optional[str] = None,
    memory_path: _Optional[str] = None,
    memory_snapshot: _Optional[_memory_lifecycle.MemorySnapshot] = None,
    memory_commit_mode: _Optional[str] = None,
    mapping_execution_mode: str = "full",
) -> _Dict[str, _Any]:


    from .run_orchestrator import run_design_search as _run_design_search

    return _run_design_search(
        strategy,
        seed=seed,
        design_state_path=design_state_path,
        memory_path=memory_path,
        memory_snapshot=memory_snapshot,
        memory_commit_mode=memory_commit_mode,
        mapping_execution_mode=mapping_execution_mode,
    )


__all__ = [
    "prepare_case_inputs",
    "build_case_inputs",
    "run_design_search",
]
