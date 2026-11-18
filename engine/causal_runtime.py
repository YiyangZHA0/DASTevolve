

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .causal_flow import (
    ActionRecord,
    CausalFlowContractError,
    CausalTrace,
    EffectiveSearchContract,
    GraphPatch,
    ObservationRecord,
    PatchFieldDisposition,
    SequenceRecord,
    diff_effective_contract,
    validate_causal_trace,
)


def _string_keyed_fixed_residues(
    fixed_residues: Mapping[str, Mapping[int, str]],
) -> Dict[str, Dict[str, str]]:
    return {
        str(chain_id): {
            str(int(position)): str(residue)
            for position, residue in sorted(residues.items(), key=lambda item: int(item[0]))
        }
        for chain_id, residues in fixed_residues.items()
    }


def _string_keyed_residue_mutation_contract(value: Any) -> Any:


    if not isinstance(value, Mapping):
        return deepcopy(value)
    return {
        str(chain_id): (
            {
                str(position): deepcopy(residues)
                for position, residues in positions.items()
            }
            if isinstance(positions, Mapping)
            else deepcopy(positions)
        )
        for chain_id, positions in value.items()
    }


def canonical_semantic_node_ids(
    design_state: Mapping[str, Any],
    compiled: Mapping[str, Any],
) -> Dict[str, Any]:


    graph = design_state.get("semantic_graph")
    graph = graph if isinstance(graph, Mapping) else {}
    structural_graph = graph.get("structural_graph")
    structural_graph = structural_graph if isinstance(structural_graph, Mapping) else {}
    raw_structural = structural_graph.get("nodes")
    raw_structural = raw_structural if isinstance(raw_structural, Mapping) else {}

    structural: Dict[str, str] = {}
    for segment in compiled.get("segments", []) or []:
        name = str(getattr(segment, "name", "") or "")
        if not name:
            continue
        payload = raw_structural.get(name)
        payload = payload if isinstance(payload, Mapping) else {}
        chain_id = str(payload.get("chain_id") or getattr(segment, "chain_id", "") or "_")
        structural[name] = f"structural::{chain_id}::{name}"
    for name, raw in raw_structural.items():
        node = raw if isinstance(raw, Mapping) else {}
        raw_name = str(name)
        structural.setdefault(
            raw_name,
            f"structural::{str(node.get('chain_id') or '_')}::{raw_name}",
        )

    functional_graph = graph.get("functional_graph")
    functional_graph = functional_graph if isinstance(functional_graph, Mapping) else {}
    raw_functional = functional_graph.get("nodes")
    raw_functional = raw_functional if isinstance(raw_functional, Mapping) else {}
    functional = {
        str(name): f"functional::{str(name)}"
        for name in sorted(raw_functional, key=str)
    }

    raw_mappings = graph.get("mappings")
    raw_mappings = raw_mappings if isinstance(raw_mappings, Mapping) else {}
    mappings: Dict[str, List[str]] = {}
    for functional_name, structural_names in raw_mappings.items():
        fid = functional.get(str(functional_name), f"functional::{functional_name}")
        values = structural_names if isinstance(structural_names, Sequence) and not isinstance(structural_names, str) else []
        mappings[fid] = [
            structural.get(str(name), f"structural::_::{name}") for name in values
        ]
    return {
        "structural": structural,
        "functional": functional,
        "functional_to_structural": mappings,
    }


def compile_mutation_scope_contract(
    *,
    dual_ast_compilation: Any,
    executable_node_plan: Any,
    masks: Mapping[str, Iterable[bool]],
    search_config: Mapping[str, Any],
    ast_revision_report: Mapping[str, Any],
) -> Dict[str, Any]:


    ast = dual_ast_compilation.ast
    return {
        "schema_version": "astevolve.compiled_mutation_scope.v1",
        "ast_id": ast.ast_id if ast is not None else None,
        "ast_revision": ast.revision if ast is not None else None,
        "ast_revision_report_hash": ast_revision_report.get("effective_ast_hash"),
        "active_positions_by_chain": {
            str(chain_id): [
                int(position)
                for position, enabled in enumerate(mask)
                if bool(enabled)
            ]
            for chain_id, mask in masks.items()
        },
        "active_positions_by_node": {
            node.node_id: {
                "chain_id": node.chain_id,
                "positions": list(node.legal_positions),
            }
            for node in executable_node_plan.structural_nodes
            if node.legal_positions
        },
        "max_total_mutations": int(search_config.get("max_total_mutations", 0)),
    }


def build_effective_search_contract(
    *,
    masks: Mapping[str, Iterable[bool]],
    fixed_residues: Mapping[str, Mapping[int, str]],
    search_config: Mapping[str, Any],
    constraint_specs: Sequence[Mapping[str, Any]],
    score_config: Mapping[str, Any],
    design_state: Mapping[str, Any],
    compiled: Mapping[str, Any],
    proposal_id: str,
    parent_program_id: str,
    run_id: str,
    logical_time: str = "",
) -> EffectiveSearchContract:


    node_policies = deepcopy(dict(search_config.get("node_edit_policies") or {}))
    per_node_operators = {
        str(node): deepcopy(dict(policy.get("mutation_ops") or {}))
        for node, policy in node_policies.items()
        if isinstance(policy, Mapping)
    }
    budget_keys = (
        "iterations",
        "search_method",
        "mcts_max_depth",
        "mcts_tree_quality_required",
        "mcts_tree_min_root_children",
        "mcts_tree_min_branching_nodes",
        "mcts_tree_min_leaves",
        "mcts_tree_min_max_depth",
        "node_optimizer_candidate_count",
        "node_optimizer_beam_width",
        "node_optimizer_top_k_per_position",
        "history_size",
        "max_total_mutations",
        "exploit_max_mutations",
        "explore_max_mutations",
        "repair_max_mutations",
        "chai1_min_candidates",
        "chai1_max_candidates",
        "structure_screen_min_candidates",
        "structure_screen_max_candidates",
        "structure_screen_progen_batch_size",
        "structure_rerank_min_candidates",
        "structure_rerank_max_candidates",
        "structure_physics_max_candidates",
        "structure_multiseed_enabled",
        "structure_formal_funnel_enabled",
        "structure_protenix_seeds",
        "structure_af3_seeds",
        "structure_robust_top_candidates",
        "structure_pyrosetta_required",
        "structure_rerank_all_infeasible_rescue",
        "structure_screen_single_node_diagnostic_quota",
        "structure_position_distribution_engagement_quota",
        "structure_portfolio_contract_quota",
        "portfolio_seed_refinement_rounds",
        "candidate_wave_enabled",
        "candidate_wave_size",
        "candidate_wave_fail_on_underfill",
        "candidate_wave_protenix_mutant_quota",
        "candidate_wave_af3_mutant_quota",
        "candidate_wave_changed_node_min_generated_unique",
        "candidate_wave_changed_node_min_frozen_unique",
        "candidate_wave_changed_node_min_protenix_attempts",
        "structure_batch_size",
        "structure_parallel_workers",
        "executable_island_policy_enabled",
    )
    search_budget = {
        key: deepcopy(search_config.get(key))
        for key in budget_keys
        if key in search_config
    }
    score_weights = {
        str(key): deepcopy(value)
        for key, value in score_config.items()
        if str(key).startswith("weight_")
        or str(key).endswith("_weight")
        or str(key).endswith("_penalty")
    }
    evaluator_routing = {
        "plugin_resolution": deepcopy(score_config.get("plugin_resolution") or {}),
        "evaluator_backends": deepcopy(score_config.get("evaluator_backends") or {}),
        "measurement_intents": deepcopy(score_config.get("measurement_intents") or []),
        "mapping_measurement_specs": deepcopy(
            score_config.get("mapping_measurement_specs") or []
        ),
        "executable_mapping_plan": deepcopy(
            score_config.get("executable_mapping_plan") or {}
        ),
        "effective_mapping_schedule": deepcopy(
            score_config.get("effective_mapping_schedule") or {}
        ),
        "score_weights": score_weights,
        "structure": {
            key: deepcopy(search_config.get(key))
            for key in (
                "chai1_enabled",
                "structure_model",
                "structure_model_name",
                "structure_screen_enabled",
                "structure_screen_model",
                "structure_screen_model_name",
                "structure_rerank_enabled",
                "structure_rerank_model",
                "structure_rerank_model_name",
                "structure_shortlist_policy",
                "structure_position_distribution_engagement_quota",
                "structure_portfolio_contract_quota",
                "structure_selection_objective",
                "structure_stepping_stone_enabled",
                "structure_stepping_stone_max_energy_degradation",
                "structure_stepping_stone_metrics",
                "structure_stepping_stone_min_metric_gain",
                "structure_allow_low_fidelity_fallback",
                "structure_multiseed_enabled",
                "structure_formal_funnel_enabled",
                "structure_protenix_seeds",
                "structure_af3_seeds",
                "structure_robust_top_candidates",
                "structure_disagreement_threshold",
                "structure_pyrosetta_required",
                "af3_seed",
                "candidate_wave_enabled",
                "candidate_wave_size",
                "candidate_wave_fail_on_underfill",
                "candidate_wave_protenix_mutant_quota",
                "candidate_wave_af3_mutant_quota",
                "candidate_wave_changed_node_min_protenix_attempts",
                "structure_service_backend",
                "multistate_objectives_enabled",
            )
            if key in search_config
        },
    }
    semantic_ids = canonical_semantic_node_ids(design_state, compiled)
    state_context = {
        "case_id": str(
            design_state.get("case_id")
            or design_state.get("task_name")
            or "direct_case"
        ),
        "design_state_version": design_state.get("version"),
        "graph_ablation_mode": design_state.get("_graph_ablation_mode", "full"),
        "chain_order": [str(value) for value in compiled.get("chain_order", []) or []],
        "chain_lengths": {
            str(key): int(value)
            for key, value in (compiled.get("chain_lengths") or {}).items()
        },
        "semantic_node_ids": semantic_ids,
        "semantic_required_nodes": list(
            search_config.get("semantic_required_nodes") or []
        ),
        "executable_ast": {
            "ast_id": (
                (design_state.get("executable_dual_ast") or {}).get("ast_id")
                if isinstance(design_state.get("executable_dual_ast"), Mapping)
                else None
            ),
            "revision": (
                (design_state.get("executable_dual_ast") or {}).get("revision")
                if isinstance(design_state.get("executable_dual_ast"), Mapping)
                else None
            ),
            "effective_ast_hash": (
                (design_state.get("_ast_revision_report") or {}).get(
                    "effective_ast_hash"
                )
                if isinstance(design_state.get("_ast_revision_report"), Mapping)
                else None
            ),
            "positions_by_node": deepcopy(
                (design_state.get("_ast_revision_report") or {}).get(
                    "effective_positions_by_node", {}
                )
                if isinstance(design_state.get("_ast_revision_report"), Mapping)
                else {}
            ),
        },
    }
    raw_compiled_action = design_state.get("_compiled_design_action")
    if isinstance(raw_compiled_action, Mapping):
        state_context["design_action"] = {
            key: deepcopy(raw_compiled_action.get(key))
            for key in (
                "schema_version",
                "design_action_hash",
                "compiled_design_action_hash",
                "parent_candidate_id",
                "parent_sequence_bundle_hash",
                "parent_effective_contract_hash",
                "reconciled_root_sequence_bundle_hash",
                "executable_node_plan_hash",
            )
        }
        if raw_compiled_action.get("position_distributions"):
            state_context["design_action"]["position_distributions"] = deepcopy(
                raw_compiled_action["position_distributions"]
            )
    raw_portfolio_request = design_state.get(
        "_compiled_portfolio_optimization_request"
    )
    if isinstance(raw_portfolio_request, Mapping):
        state_context["portfolio_optimization_request"] = {
            "schema_version": raw_portfolio_request.get("schema_version"),
            "compiled_portfolio_request_hash": raw_portfolio_request.get(
                "compiled_portfolio_request_hash"
            ),
            "design_action_hash": raw_portfolio_request.get(
                "design_action_hash"
            ),
            "compiled_design_action_hash": raw_portfolio_request.get(
                "compiled_design_action_hash"
            ),
            "reconciled_root_sequence_bundle_hash": raw_portfolio_request.get(
                "reconciled_root_sequence_bundle_hash"
            ),
            "candidate_slot_hashes": [
                item.get("slot_hash")
                for item in raw_portfolio_request.get("candidate_slots", [])
                if isinstance(item, Mapping)
            ],
        }
    generator_policy = {
        key: deepcopy(search_config.get(key))
        for key in (
            "proposal_engine",
            "sequence_generator_id",
            "sequence_generator_structure_condition_refs",
            "sequence_generator_state_condition_refs",
            "sequence_prior_model",
            "residue_mutation_contract",
            "compiled_position_distribution_policy",
            "compiled_portfolio_request_policy",
            "node_optimizer_enabled",
            "node_optimizer_candidate_count",
            "node_optimizer_beam_width",
            "node_optimizer_top_k_per_position",
            "node_optimizer_temperature",
            "node_optimizer_diversity_weight",
            "node_optimizer_mutation_penalty",
            "node_optimizer_prior_model",
            "node_optimizer_model_path",
            "node_optimizer_device",
            "mcts_progressive_widening_c",
            "mcts_progressive_widening_alpha",
            "proposal_tier_mode",
            "progen_chains",
            "progen_weight",
            "proposal_exploit_frac",
            "proposal_explore_frac",
            "proposal_repair_frac",
            "candidate_wave_enabled",
            "candidate_wave_size",
            "candidate_wave_fail_on_underfill",
            "candidate_wave_changed_node_min_generated_unique",
            "candidate_wave_changed_node_min_frozen_unique",
            "executable_island_policy_enabled",
        )
        if key in search_config
    }
    if "residue_mutation_contract" in generator_policy:


        generator_policy["residue_mutation_contract"] = (
            _string_keyed_residue_mutation_contract(
                generator_policy["residue_mutation_contract"]
            )
        )
    return EffectiveSearchContract.create(
        masks={
            str(chain_id): [bool(value) for value in values]
            for chain_id, values in masks.items()
        },
        fixed_residues=_string_keyed_fixed_residues(fixed_residues),
        node_policies=node_policies,
        operator_policy={
            "global": deepcopy(dict(search_config.get("mutation_ops") or {})),
            "per_node": per_node_operators,
        },
        search_budget=search_budget,
        constraints=[deepcopy(dict(spec)) for spec in constraint_specs],
        evaluator_routing=evaluator_routing,
        generator_policy=generator_policy,
        state_context=state_context,
        provenance={
            "proposal_id": str(proposal_id),
            "parent_program_id": str(parent_program_id),
            "run_id": str(run_id),
            "compilation_id": str(design_state.get("version") or "design-state"),
            "artifact_path": str(search_config.get("mcts_output_dir") or ""),
            "created_at": str(logical_time or ""),
        },
    )


def build_graph_patch(
    *,
    strategy_report: Mapping[str, Any],
    proposal_id: str,
    parent_program_id: str,
    hypothesis: str = "",
) -> GraphPatch:


    fields: List[PatchFieldDisposition] = []
    for group in ("applied", "transformed"):
        for row in strategy_report.get(group, []) or []:
            if not isinstance(row, Mapping) or not row.get("path"):
                continue
            consumer = str(row.get("consumer") or "").strip()
            if not consumer:
                continue
            fields.append(
                PatchFieldDisposition.create(
                    path=str(row["path"]),
                    requested=deepcopy(row.get("requested")),
                    disposition="compiled",
                    effective=deepcopy(row.get("effective")),
                    reason=str(row.get("reason") or "compiled into an executable consumer"),
                    consumer=consumer,
                )
            )
    for group in ("rejected", "unknown"):
        for row in strategy_report.get(group, []) or []:
            if not isinstance(row, Mapping) or not row.get("path"):
                continue
            fields.append(
                PatchFieldDisposition.create(
                    path=str(row["path"]),
                    requested=deepcopy(row.get("requested")),
                    disposition="rejected",
                    effective=None,
                    reason=str(row.get("reason") or "no executable consumer"),
                )
            )
    if not fields:
        fields.append(
            PatchFieldDisposition.create(
                path="strategy",
                requested={},
                disposition="no_op",
                effective=None,
                reason="strategy contained no compiler-classified executable leaves",
            )
        )


    fields.sort(key=lambda item: item.path)
    return GraphPatch.create(
        proposal_id=str(proposal_id),
        parent_program_id=str(parent_program_id),
        fields=fields,
        hypothesis=str(hypothesis or "compiled strategy changes inner search behavior"),
    )


def causal_context_mapping(
    *,
    generation_id: str,
    proposal_id: str,
    trial_id: str,
    seed: Optional[int],
    graph_patch: GraphPatch,
    effective_contract: EffectiveSearchContract,
    island_id: Optional[int] = None,
    island_role: str = "",
    compiled_design_action: Any = None,
    compiled_portfolio_optimization_request: Any = None,
) -> Dict[str, Any]:
    context = {
        "generation_id": str(generation_id),
        "proposal_id": str(proposal_id),
        "trial_id": str(trial_id),
        "seed": int(seed) if seed is not None else None,
        "graph_patch_hash": graph_patch.patch_hash,
        "effective_contract_hash": effective_contract.contract_hash,
    }
    if island_id is not None:
        context["island_id"] = int(island_id)
    if str(island_role or "").strip():
        context["island_role"] = str(island_role).strip()
    if compiled_design_action is not None:
        artifact = (
            compiled_design_action.to_artifact()
            if hasattr(compiled_design_action, "to_artifact")
            else dict(compiled_design_action)
        )
        for key in (
            "design_action_hash",
            "compiled_design_action_hash",
            "case_id",
            "parent_program_id",
            "parent_candidate_id",
            "parent_sequence_bundle_hash",
            "parent_effective_contract_hash",
            "parent_evolve_hash",
        ):
            value = artifact.get(key)
            if value in (None, ""):
                raise ValueError(
                    f"compiled DesignAction causal identity is missing {key}"
                )
            context[key] = str(value)
    if compiled_portfolio_optimization_request is not None:
        portfolio_artifact = (
            compiled_portfolio_optimization_request.to_artifact()
            if hasattr(compiled_portfolio_optimization_request, "to_artifact")
            else dict(compiled_portfolio_optimization_request)
        )
        for key in (
            "compiled_portfolio_request_hash",
            "design_action_hash",
            "compiled_design_action_hash",
            "case_id",
            "parent_program_id",
            "parent_candidate_id",
        ):
            value = portfolio_artifact.get(key)
            if value in (None, ""):
                raise ValueError(
                    f"compiled portfolio causal identity is missing {key}"
                )
            if key in context and str(context[key]) != str(value):
                raise ValueError(
                    f"compiled portfolio causal identity disagrees on {key}"
                )
            context[key] = str(value)
    return context


def _operator_field_paths(patch: GraphPatch, operator: str) -> Tuple[str, ...]:
    suffix = f"mutation_ops.{operator}"
    paths = [
        field.path
        for field in patch.fields
        if field.disposition in {"compiled", "executed"}
        and (field.path == suffix or field.path.endswith("." + suffix))
    ]
    return tuple(sorted(set(paths)))


_MAPPING_IDENTITY_FIELDS = (
    "ast_id",
    "ast_revision",
    "edge_id",
    "functional_node_id",
    "structural_node_id",
    "action_id",
    "measurement_id",
)


def _canonical_mapping_components(
    move: Mapping[str, Any],
    effective_contract: EffectiveSearchContract,
) -> List[Dict[str, Any]]:


    routing = effective_contract.semantic.get("evaluator_routing", {})
    schedule = (
        routing.get("effective_mapping_schedule", {})
        if isinstance(routing, Mapping)
        else {}
    )
    if not isinstance(schedule, Mapping) or not bool(schedule.get("execution_enabled")):
        raise CausalFlowContractError(
            "selected_lineage_mapping_components_invalid",
            "composite mapping move has no enabled effective mapping schedule",
        )
    actions = schedule.get("active_action_specs")
    if (
        isinstance(actions, (str, bytes))
        or not isinstance(actions, Sequence)
        or not actions
    ):
        raise CausalFlowContractError(
            "selected_lineage_mapping_components_invalid",
            "effective mapping schedule has no active action specs",
        )
    from astevolve.search.mapping_schedule_runtime import (
        validate_portfolio_mapping_components,
    )

    try:
        return validate_portfolio_mapping_components(
            move,
            mapping_actions=actions,
        )
    except (TypeError, ValueError) as error:
        raise CausalFlowContractError(
            "selected_lineage_mapping_components_invalid", str(error)
        ) from error


def _validated_immediate_mapping_traces(
    actions: Sequence[ActionRecord],
    traces: Sequence[Any],
    *,
    final_sequence_id: str,
) -> None:


    if not traces:
        return
    if not actions:
        raise CausalFlowContractError("causal_mapping_trace_count_mismatch")
    from engine.mapping_execution import (
        MappingExecutionError,
        validate_mapping_execution_trace,
    )

    try:
        canonical_traces = [validate_mapping_execution_trace(item) for item in traces]
    except (MappingExecutionError, TypeError, ValueError) as error:
        raise CausalFlowContractError(
            "causal_mapping_trace_invalid", str(error)
        ) from error
    final_action = actions[-1]
    immediate = []
    for action in reversed(actions):
        if (
            action.parent_sequence_id != final_action.parent_sequence_id
            or action.child_sequence_id != final_sequence_id
        ):
            break
        immediate.append(action)
    immediate.reverse()
    cursor = 0
    for trace in canonical_traces:
        matched = False
        while cursor < len(immediate):
            action = immediate[cursor]
            cursor += 1
            parameters = action.parameters
            if (
                trace.evaluated_sequence_id == final_sequence_id
                and all(
                    parameters.get(field) == getattr(trace, field)
                    for field in _MAPPING_IDENTITY_FIELDS
                )
                and action.operator == trace.action["operator"]
                and action.node_id == trace.structural_node_id
                and list(action.positions) == trace.action["realized_positions"]
            ):
                matched = True
                break
        if not matched:
            raise CausalFlowContractError("causal_mapping_trace_order_mismatch")


def _finalize_executed_patch(
    patch: GraphPatch,
    actions: Sequence[ActionRecord],
) -> GraphPatch:


    action_ids_by_path: Dict[str, List[str]] = {}
    for action in actions:
        for path in action.field_paths:
            action_ids_by_path.setdefault(path, []).append(action.semantic_id)
    fields: List[PatchFieldDisposition] = []
    for field in patch.fields:
        action_ids = tuple(sorted(set(action_ids_by_path.get(field.path, []))))
        if field.disposition in {"compiled", "executed"} and action_ids:
            fields.append(
                PatchFieldDisposition.create(
                    path=field.path,
                    requested=field.requested,
                    disposition="executed",
                    effective=field.effective,
                    reason=field.reason,
                    consumer=field.consumer,
                    action_ids=action_ids,
                )
            )
        else:
            fields.append(field)
    return GraphPatch.create(
        proposal_id=patch.proposal_id,
        parent_program_id=patch.parent_program_id,
        hypothesis=patch.hypothesis,
        fields=fields,
    )


def _selected_lineage(runtime: Mapping[str, Any]) -> List[Dict[str, Any]]:
    root = runtime.get("root_candidate")
    selected = runtime.get("selected_candidate")
    if not isinstance(root, Mapping) or not isinstance(selected, Mapping):
        raise CausalFlowContractError(
            "causal_runtime_incomplete",
            "inner search did not return root and selected candidate records",
        )
    candidates = [root, *(runtime.get("candidates") or [])]
    by_id = {
        str(item.get("variant_id")): dict(item)
        for item in candidates
        if isinstance(item, Mapping) and item.get("variant_id") is not None
    }
    current = dict(selected)
    lineage: List[Dict[str, Any]] = []
    seen = set()
    while True:
        variant_id = str(current.get("variant_id") or "")
        if not variant_id or variant_id in seen:
            raise CausalFlowContractError(
                "selected_lineage_invalid",
                "selected candidate lineage is cyclic or has no variant identity",
            )
        seen.add(variant_id)
        lineage.append(current)
        parent_id = current.get("parent_id")
        if parent_id in (None, ""):
            break
        current = by_id.get(str(parent_id))
        if current is None:
            raise CausalFlowContractError(
                "selected_lineage_parent_missing",
                f"selected lineage references unknown parent {parent_id!r}",
            )
    lineage.reverse()
    if str(lineage[0].get("variant_id")) != str(root.get("variant_id")):
        raise CausalFlowContractError(
            "selected_lineage_root_mismatch",
            "selected lineage does not terminate at the initialized parent sequence",
        )
    return lineage


def build_selected_causal_trace(
    *,
    runtime: Mapping[str, Any],
    graph_patch: GraphPatch,
    parent_contract: EffectiveSearchContract,
    effective_contract: EffectiveSearchContract,
    evaluator_report: Mapping[str, Any],
    mapping_execution_traces: Sequence[Any] = (),
    exact_measurements: Sequence[Mapping[str, Any]] = (),
    seed: Optional[int],
) -> CausalTrace:


    lineage = _selected_lineage(runtime)
    sequence_by_id: Dict[str, SequenceRecord] = {}
    lineage_sequences: List[Tuple[Dict[str, Any], SequenceRecord]] = []
    for candidate in lineage:
        seqs = candidate.get("seqs")
        if not isinstance(seqs, Mapping):
            raise CausalFlowContractError(
                "candidate_sequences_missing", "lineage candidate has no sequence bundle"
            )
        record = SequenceRecord.create(seqs)
        sequence_by_id.setdefault(record.semantic_id, record)
        lineage_sequences.append((candidate, record))

    semantic_ids = effective_contract.semantic.get("state_context", {}).get(
        "semantic_node_ids", {}
    )
    structural_ids = (
        semantic_ids.get("structural", {})
        if isinstance(semantic_ids, Mapping)
        else {}
    )
    identity = dict(runtime.get("identity") or {})
    compiled_patch_hash = identity.pop("graph_patch_hash", None)
    if compiled_patch_hash:
        identity["compiled_graph_patch_hash"] = compiled_patch_hash
    actions: List[ActionRecord] = []
    for index in range(1, len(lineage_sequences)):
        child_candidate, child_sequence = lineage_sequences[index]
        _parent_candidate, parent_sequence = lineage_sequences[index - 1]
        if child_sequence.semantic_id == parent_sequence.semantic_id:
            continue
        move = child_candidate.get("move")
        if not isinstance(move, Mapping) or str(move.get("outcome")) != "executed":
            raise CausalFlowContractError(
                "selected_lineage_action_missing",
                "a selected sequence delta has no executed mutation move",
            )
        composite_claimed = (
            "mapping_components" in move or "mapping_component_set_hash" in move
        )
        move_records = (
            _canonical_mapping_components(move, effective_contract)
            if composite_claimed
            else [dict(move)]
        )
        for component_index, action_move in enumerate(move_records):
            operator = str(action_move.get("op") or "").strip()
            raw_node = str(
                action_move.get("node")
                or next(iter(action_move.get("target_nodes") or []), "search_space")
            )
            raw_mapping_attribution = action_move.get("mapping_attribution")
            mapping_attribution = (
                {
                    key: raw_mapping_attribution[key]
                    for key in _MAPPING_IDENTITY_FIELDS
                    if key in raw_mapping_attribution
                }
                if isinstance(raw_mapping_attribution, Mapping)
                else {}
            )
            node_id = str(
                mapping_attribution.get("structural_node_id")
                or structural_ids.get(raw_node)
                or f"structural::_::{raw_node}"
            )
            positions = sorted(
                {
                    int(change["position"])
                    for change in action_move.get("changes", []) or []
                    if isinstance(change, Mapping)
                    and change.get("position") is not None
                }
            )
            parameters = {
                **identity,
                "parent_variant_id": str(lineage[index - 1].get("variant_id") or ""),
                "child_variant_id": str(child_candidate.get("variant_id") or ""),
                "outcome": str(move.get("outcome") or ""),
                "changes": deepcopy(list(action_move.get("changes") or [])),
                **mapping_attribution,
            }
            if composite_claimed:
                parameters.update(
                    {
                        "mapping_component_schema_version": action_move.get(
                            "schema_version"
                        ),
                        "mapping_component_id": action_move.get("component_id"),
                        "mapping_component_hash": action_move.get(
                            "mapping_component_hash"
                        ),
                        "mapping_component_set_hash": move.get(
                            "mapping_component_set_hash"
                        ),
                        "mapping_component_index": component_index,
                        "mapping_component_count": len(move_records),
                    }
                )
                for field in (
                    "compiled_portfolio_request_hash",
                    "portfolio_id",
                    "portfolio_role",
                    "portfolio_slot_id",
                ):
                    if field in move:
                        parameters[field] = deepcopy(move[field])
            actions.append(ActionRecord.create(
                contract_hash=effective_contract.contract_hash,
                node_id=node_id,
                operator=operator,
                positions=positions,
                parent_sequence_id=parent_sequence.semantic_id,
                child_sequence_id=child_sequence.semantic_id,
                parameters=parameters,
                field_paths=_operator_field_paths(graph_patch, operator),
            ))

    parent_sequence_id = lineage_sequences[0][1].semantic_id
    final_sequence_id = lineage_sequences[-1][1].semantic_id
    _validated_immediate_mapping_traces(
        actions,
        mapping_execution_traces,
        final_sequence_id=final_sequence_id,
    )
    metrics: Dict[str, Any] = {}
    for key in (
        "normalized_score",
        "loss",
        "soft_energy",
        "total_energy",
        "hard_gate_pass",
    ):
        value = evaluator_report.get(key)
        if isinstance(value, (bool, int, float)):
            metrics[key] = value
    if exact_measurements:
        observations = []
        for observation in exact_measurements:
            raw_gate = observation.get("gate")
            observations.append(
                ObservationRecord.create(
                    sequence_id=final_sequence_id,
                    evaluator={
                        "tool": "exact_mapping_term",
                        "evaluator_id": observation.get("evaluator_id"),
                        "term_name": observation.get("term_name"),
                        "provider": observation.get("term_provider"),
                        "measurement_id": observation.get("measurement_id"),
                        "evaluated_sequence_id": observation.get(
                            "evaluated_sequence_id"
                        ),
                    },
                    state={
                        "functional_node_id": observation.get(
                            "functional_node_id"
                        ),
                        "kind": observation.get("kind"),
                        "functional_state": observation.get("state"),
                        "status": observation.get("status"),
                    },
                    seed=int(seed) if seed is not None else 0,
                    metrics={
                        "term_value": observation.get("term_value"),
                        "directional_value": observation.get("directional_value"),
                    },
                    gate=(
                        dict(raw_gate)
                        if isinstance(raw_gate, Mapping)
                        else {
                            "passed": None,
                            "status": (
                                "abstain"
                                if observation.get("status") == "abstain"
                                else "not_applicable"
                            ),
                            "reason": (
                                "exact_term_missing"
                                if observation.get("status") == "abstain"
                                else "no_hard_gate"
                            ),
                        }
                    ),
                )
            )
    else:
        observations = [ObservationRecord.create(
            sequence_id=final_sequence_id,
            evaluator={
                "tool": "astevolve.evaluation.evaluator_engine.evaluate_candidate",
                "plugin_resolution": deepcopy(
                    effective_contract.semantic.get("evaluator_routing", {}).get(
                        "plugin_resolution", {}
                    )
                ),
            },
            state={
                "name": str(
                    effective_contract.semantic.get("state_context", {}).get(
                        "case_id", "design_task"
                    )
                ),
                "selection_source": str(runtime.get("selection_source") or "unknown"),
            },
            seed=int(seed) if seed is not None else 0,
            metrics=metrics,
            gate={
                "passed": bool(evaluator_report.get("hard_gate_pass", True)),
                "reasons": list(
                    evaluator_report.get("disqualification_reasons", []) or []
                ),
            },
        )]
    finalized_patch = _finalize_executed_patch(graph_patch, actions)
    trace = CausalTrace.create(
        proposal_id=finalized_patch.proposal_id,
        graph_patch=finalized_patch,
        parent_contract=parent_contract,
        effective_contract=effective_contract,
        sequences=list(sequence_by_id.values()),
        actions=actions,
        observations=observations,
        parent_sequence_id=parent_sequence_id,
        final_sequence_id=final_sequence_id,
    )
    return validate_causal_trace(trace.to_dict())


def causal_artifacts(trace: CausalTrace) -> Dict[str, Any]:


    return {
        "graph_patch": trace.graph_patch.to_dict(),
        "parent_effective_search_contract": trace.parent_contract.to_dict(),
        "effective_search_contract": trace.effective_contract.to_dict(),
        "contract_diff": diff_effective_contract(
            trace.parent_contract, trace.effective_contract
        ).to_dict(),
        "causal_trace": trace.to_dict(),
    }


__all__ = [
    "build_effective_search_contract",
    "build_graph_patch",
    "build_selected_causal_trace",
    "canonical_semantic_node_ids",
    "causal_artifacts",
    "causal_context_mapping",
    "compile_mutation_scope_contract",
]
