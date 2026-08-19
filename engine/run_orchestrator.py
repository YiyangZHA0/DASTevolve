

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from astevolve.evaluation.evaluator_engine import evaluate_candidate
from astevolve.search.config import SAConfig
from astevolve.search.candidate_wave_runtime import (
    build_candidate_wave_request,
    build_free_slot_directives,
)
from astevolve.search.inner_opt import optimize_multichain
from astevolve.search.mutation_move import finalize_mutation_move
from astevolve.search.portfolio_runtime import (
    build_portfolio_prebuilt_proposals,
)
from astevolve.search.run_memory import InnerRunMemory
from astevolve.search.structure_pipeline import _score_config_for_structure_stage
from astevolve.semantic_graph import (
    apply_graph_ablation,
    build_residue_semantic_map,
    build_semantic_graph_summary,
    diagnose_semantic_graph,
    generate_edit_contract,
    summarize_residue_semantic_map,
)
from astevolve.semantic_graph.residue_design_context import (
    DEFAULT_RESIDUE_PROMPT_MAX_BYTES,
    build_migration_frontier,
    build_residue_prompt_digest,
)

from .case_builder import prepare_case_inputs
from .case_resources import compact_case_sheet, resolve_memory_path
from .case_types import PreparedCaseInputs, SearchRequest
from .causal_flow import SequenceRecord, require_effective_contract_delta
from .causal_runtime import (
    build_selected_causal_trace,
    causal_artifacts,
    causal_context_mapping,
)
from .design_state import binder_domain_order, has_target
from .evaluation_merge import merge_inner_semantic_audit
from .external_knowledge_policy import build_external_knowledge_policy
from .experiment_identity import SequenceBundleIdentity
from .memory_lifecycle import (
    MEMORY_COMMIT_MODES,
    MemoryExecutionContext,
    MemorySnapshot,
    ScopedAdaptivePriorSnapshot,
    current_memory_execution_context,
)
from .memory_policy import MemoryPolicyConfig, MemoryPolicyError, MemoryScope
from .memory_update import update_internal_memory
from .mapping_runtime import project_selected_mapping_runtime
from .history_runtime import (
    claim_effective_contract,
    complete_effective_contract,
    evaluate_exact_persistently,
    fail_effective_contract,
    maintain_effective_contract_lease,
    register_sequence_occurrence,
    registry_metrics_artifact,
)


def _execute_legacy_search(request: SearchRequest) -> Dict[str, Any]:


    scope = current_memory_execution_context()
    prebuilt_proposals = None
    portfolio_materialization_accounting = None
    portfolio_seed_refinement_directives = None
    candidate_wave_request = None
    candidate_wave_free_slot_directives = None
    candidate_wave_activation = None
    execution_config = request.config
    compiled_portfolio_request = getattr(
        request, "compiled_portfolio_optimization_request", None
    )
    compiled_design_action = getattr(request, "compiled_design_action", None)
    if compiled_portfolio_request is not None:
        (
            prebuilt_proposals,
            portfolio_materialization_accounting,
            portfolio_seed_refinement_directives,
        ) = build_portfolio_prebuilt_proposals(
            compiled_portfolio_request
        )
    if bool(getattr(request.config, "candidate_wave_enabled", False)):
        has_portfolio = compiled_portfolio_request is not None
        has_action = compiled_design_action is not None
        ast_revision_report = request.design_state.get("_ast_revision_report")
        is_initial_baseline = (
            str(request.proposal_id) == "initial"
            and not has_action
            and not has_portfolio
        )
        if is_initial_baseline:


            execution_config = deepcopy(request.config)
            execution_config.candidate_wave_enabled = False


            execution_config.structure_formal_funnel_enabled = False
            execution_config.structure_multiseed_enabled = False
            execution_config.structure_physics_max_candidates = 0
            execution_config.structure_pyrosetta_required = False
            execution_config.chai1_enabled = False
            execution_config.structure_screen_enabled = False
            execution_config.structure_rerank_enabled = False
            candidate_wave_activation = _candidate_wave_activation_receipt(
                requested=True,
                active=False,
                mode="initial_baseline",
                proposal_id=str(request.proposal_id),
                detail="no DesignAction exists before the first outer proposal",
            )
        elif not has_portfolio or not has_action:
            missing = []
            if not has_action:
                missing.append("compiled_design_action")
            if not has_portfolio:
                missing.append("compiled_portfolio_optimization_request")
            raise ValueError(
                "candidate_wave_enabled requires a complete child contract: "
                + ",".join(missing)
            )
        elif not isinstance(ast_revision_report, Mapping):
            raise ValueError(
                "candidate_wave_enabled requires an AST revision report"
            )
        else:
            candidate_wave_request = build_candidate_wave_request(
                compiled_portfolio_request,
                compiled_design_action,
                ast_revision_report,
            )
            candidate_wave_free_slot_directives = build_free_slot_directives(
                candidate_wave_request
            )
            candidate_wave_activation = _candidate_wave_activation_receipt(
                requested=True,
                active=True,
                mode="compiled_child",
                proposal_id=str(request.proposal_id),
                detail=candidate_wave_request["candidate_wave_request_hash"],
            )
    output = optimize_multichain(
        request.legacy_compiled,
        request.constraint_specs,
        execution_config,
        masks=request.masks,
        template_seqs=request.template_sequences,
        fixed_residues=request.fixed_residues,
        internal_memory=request.internal_memory,
        run_memory=request.inner_run_memory,
        score_config=request.score_config,
        design_state=request.design_state,
        causal_context=causal_context_mapping(
            generation_id=request.generation_id,
            proposal_id=request.proposal_id,
            trial_id=request.trial_id,
            seed=request.seed,
            graph_patch=request.graph_patch,
            effective_contract=request.effective_search_contract,
            island_id=scope.island_id if scope is not None else None,
            island_role=scope.island_role if scope is not None else "",
            compiled_design_action=compiled_design_action,
            compiled_portfolio_optimization_request=compiled_portfolio_request,
        ),
        prebuilt_proposals=prebuilt_proposals,
        portfolio_materialization_accounting=(
            portfolio_materialization_accounting
        ),
        portfolio_seed_refinement_directives=(
            portfolio_seed_refinement_directives
        ),
        candidate_wave_request=candidate_wave_request,
        candidate_wave_free_slot_directives=(
            candidate_wave_free_slot_directives
        ),
    )
    if candidate_wave_activation is not None:
        output["candidate_wave_activation"] = deepcopy(
            candidate_wave_activation
        )
        artifacts = output.setdefault("search_artifacts", {})
        if not isinstance(artifacts, dict):
            raise ValueError("search_artifacts must be a mapping")
        artifacts["candidate_wave_activation"] = deepcopy(
            candidate_wave_activation
        )
    return output


def _candidate_wave_activation_receipt(
    *,
    requested: bool,
    active: bool,
    mode: str,
    proposal_id: str,
    detail: str,
) -> Dict[str, Any]:


    semantic = {
        "schema_version": "astevolve.candidate_wave_activation.v1",
        "requested": bool(requested),
        "active": bool(active),
        "mode": str(mode),
        "proposal_id": str(proposal_id),
        "detail": str(detail),
    }
    canonical = json.dumps(
        semantic,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return {
        **semantic,
        "activation_receipt_hash": "candidate_wave_activation_sha256:"
        + hashlib.sha256(
            (
                "astevolve.candidate_wave_activation.v1\0" + canonical
            ).encode("utf-8")
        ).hexdigest(),
    }


def _compatibility_causal_runtime(
    output: Mapping[str, Any],
    prepared: PreparedCaseInputs,
    request: SearchRequest,
) -> Dict[str, Any]:


    parent_seqs = deepcopy(prepared.template_sequences)
    final_seqs = deepcopy(output.get("seqs") or parent_seqs)
    root = {
        "variant_id": "root",
        "parent_id": None,
        "seqs": parent_seqs,
        "candidate_role": "parent_baseline",
        "is_parent_baseline": True,
    }
    scope = current_memory_execution_context()
    compiled_design_action = getattr(request, "compiled_design_action", None)
    compiled_portfolio_request = getattr(
        request, "compiled_portfolio_optimization_request", None
    )
    identity = causal_context_mapping(
        generation_id=request.generation_id,
        proposal_id=request.proposal_id,
        trial_id=request.trial_id,
        seed=request.seed,
        graph_patch=request.graph_patch,
        effective_contract=request.effective_search_contract,
        island_id=scope.island_id if scope is not None else None,
        island_role=scope.island_role if scope is not None else "",
        compiled_design_action=compiled_design_action,
        compiled_portfolio_optimization_request=compiled_portfolio_request,
    )
    if final_seqs == parent_seqs:
        selected = root
        candidates = []
    else:
        move = finalize_mutation_move(
            parent_seqs,
            final_seqs,
            {
                "op": "recorded_search_adapter",
                "node": "compatibility_adapter",
                "positions": {},
                "segments": [],
                "causal_context": identity,
            },
        )
        selected = {
            "variant_id": "recorded_final",
            "parent_id": "root",
            "seqs": final_seqs,
            "move": move,
            "causal_context": identity,
        }
        candidates = [selected]
    return {
        "identity": identity,
        "root_candidate": root,
        "candidates": candidates,
        "selected_candidate": selected,
        "selection_source": "compatibility_adapter",
    }


def _attach_compilation_metadata(
    output: Dict[str, Any],
    prepared: PreparedCaseInputs,
    request: SearchRequest,
) -> None:


    state = prepared.design_state
    compiled = request.legacy_compiled
    search_config = prepared.search_config

    output["sa_config"] = search_config
    output["strategy_schema_report"] = state.get("_strategy_schema_report", {})
    output["evaluator_plugin_resolution"] = state.get(
        "_evaluator_plugin_resolution", {}
    )
    output["outer_loop_phase"] = search_config.get("outer_loop_phase")
    output["search_schedule"] = search_config.get("search_schedule", {})
    output["chain_lengths"] = compiled["chain_lengths"]
    output["segments"] = [
        {
            "chain_id": segment.chain_id,
            "kind": segment.kind,
            "name": segment.name,
            "spans": segment.spans,
            "total_length": segment.total_length,
            "is_contiguous": segment.is_contiguous,
            "start": segment.start,
            "end": segment.end,
            "props": segment.props,
        }
        for segment in compiled["segments"]
    ]
    target = state["target"] if has_target(state) else {}
    output["blueprint_summary"] = {
        "task_name": state.get("task_name", "ASTevolve_Task"),
        "chain_order": compiled["chain_order"],
        "chain_lengths": compiled["chain_lengths"],
        "binder_architecture": state["binder"].get("architecture", "unspecified"),
        "binder_domain_order": binder_domain_order(state),
        "target_name": target.get("name", target.get("chain_id")),
        "epitope_name": target.get("epitope_name"),
        "epitope_spans": target.get("epitope_spans", []),
    }

    design_points = state.get("design_points", {})
    if not isinstance(design_points, dict):
        design_points = {}
    output["case_design_points"] = {
        "design_intent": design_points.get("design_intent"),
        "primary_design_nodes": design_points.get("primary_design_nodes", []),
        "secondary_design_nodes": design_points.get("secondary_design_nodes", []),
        "preserved_nodes": design_points.get("preserved_nodes", []),
        "operator_policy": design_points.get("operator_policy", {}),
        "known_gap": (
            (design_points.get("epitope_focus", {}) or {}).get("known_gap")
            if isinstance(design_points.get("epitope_focus"), dict)
            else (design_points.get("state_logic", {}) or {}).get("known_gap")
            if isinstance(design_points.get("state_logic"), dict)
            else None
        ),
        "case_information_needed": state.get("case_information_needed", []),
    }
    output["case_sheet_summary"] = compact_case_sheet(state.get("_case_sheet", {}))
    output["graph_ablation_mode"] = state.get("_graph_ablation_mode", "full")
    output["contract_response_report"] = state.get("_contract_response_report", {})
    output["last_contract_response"] = state.get("_last_contract_response", {})
    output["edit_contract_lifecycle"] = state.get("_edit_contract_lifecycle")
    output["ast_revision_report"] = state.get("_ast_revision_report", {})
    output["parent_sequence_lineage"] = state.get("_parent_sequence_lineage")
    output["parent_effective_ast_lineage"] = state.get(
        "_parent_effective_ast_lineage"
    )
    residue_catalog = state.get("_residue_evidence_catalog")
    output["residue_evidence_catalog_summary"] = (
        {
            "schema_version": residue_catalog.get("schema_version"),
            "catalog_hash": residue_catalog.get("catalog_hash"),
            "chain_lengths": {
                str(chain_id): len(str(sequence))
                for chain_id, sequence in (residue_catalog.get("sequences") or {}).items()
            },
            "residue_count": len(residue_catalog.get("residues") or []),
        }
        if isinstance(residue_catalog, Mapping)
        else None
    )


    selected_sequences = output.get("seqs") or output.get("best_seqs")
    if isinstance(residue_catalog, Mapping) and isinstance(
        selected_sequences, Mapping
    ):
        output["residue_evidence_prompt_digest"] = build_residue_prompt_digest(
            residue_catalog,
            max_bytes=int(
                state.get(
                    "residue_evidence_prompt_max_bytes",
                    DEFAULT_RESIDUE_PROMPT_MAX_BYTES,
                )
            ),
            current_parent_sequences=selected_sequences,
        )
        output["migration_frontier"] = build_migration_frontier(
            state,
            residue_catalog,
            current_parent_sequences=selected_sequences,
        )
    else:
        output["residue_evidence_prompt_digest"] = state.get(
            "_residue_evidence_prompt_digest"
        )
        output["migration_frontier"] = state.get("_migration_frontier")
    output["executable_dual_ast"] = (
        prepared.dual_ast_compilation.ast.to_dict()
        if prepared.dual_ast_compilation.ast is not None
        else None
    )
    output["compiled_executable_node_plan"] = prepared.executable_node_plan.to_dict()
    output["executable_mapping_plan"] = prepared.executable_mapping_plan.to_dict(
        seed=getattr(request, "seed", None)
    )
    output["effective_mapping_schedule"] = (
        prepared.effective_mapping_schedule.to_dict()
    )
    ablation_control = getattr(request, "ablation_control", None)
    if ablation_control is not None:
        output["ablation_control"] = ablation_control.to_dict()
    generator_summary = (output.get("mutation_history") or {}).get(
        "sequence_generation"
    )
    output["sequence_generator"] = (
        dict(generator_summary)
        if isinstance(generator_summary, Mapping)
        else {
            "schema_version": "astevolve.sequence_generation_summary.v1",
            "selected_generator_id": getattr(
                getattr(request, "config", None),
                "sequence_generator_id",
                search_config.get("sequence_generator_id"),
            ),
            "attempted": 0,
            "validated": 0,
            "request_hashes": [],
            "result_hashes": [],
        }
    )
    node_optimizer_summary = (output.get("mutation_history") or {}).get(
        "node_optimization"
    )
    output["node_optimizer"] = (
        dict(node_optimizer_summary)
        if isinstance(node_optimizer_summary, Mapping)
        else {
            "schema_version": "astevolve.node_optimization_summary.v1",
            "enabled": bool(
                getattr(
                    getattr(request, "config", None),
                    "node_optimizer_enabled",
                    search_config.get("node_optimizer_enabled", False),
                )
            ),
            "selected_optimizer_id": None,
            "proposal_prior_role": "mcts_edge_prior_only",
            "fast_reward_includes_proposal_prior": False,
            "batches": 0,
            "generated": 0,
            "selected_for_fast_evaluation": 0,
            "request_hashes": [],
            "result_hashes": [],
            "prior_models": [],
        }
    )
    output["measurement_intents"] = [
        intent.to_dict()
        for intent in prepared.executable_node_plan.measurement_intents
    ]


def _attach_semantic_maps(
    output: Dict[str, Any],
    prepared: PreparedCaseInputs,
    request: SearchRequest,
) -> None:


    state = prepared.design_state
    compiled = request.legacy_compiled
    node_plddt = output.get("node_plddt") or (
        output.get("structure_metrics", {}) or {}
    ).get("node_plddt")
    graph_summary = apply_graph_ablation(
        build_semantic_graph_summary(state, compiled, node_plddt),
        output["graph_ablation_mode"],
    )
    output["semantic_graph_summary"] = graph_summary
    residue_map = build_residue_semantic_map(
        state,
        compiled,
        sequences=output.get("best_seqs") or prepared.template_sequences,
        semantic_graph_summary=graph_summary,
        case_sheet=state.get("_case_sheet", {}),
        node_plddt=node_plddt,
    )
    output["residue_semantic_map"] = residue_map
    output["residue_semantic_map_summary"] = summarize_residue_semantic_map(
        residue_map,
        limit=60,
    )
    output["layout_summary"] = state.get("_layout_summary", {})


_FINAL_EVALUATOR_INPUT_FIELDS = (
    "seqs",
    "best_seqs",
    "structure_metrics",
    "structure_provider_evidence",
    "chai_plddt",
    "node_plddt",
    "multistate_objectives",
    "multistate_score",
)


def _final_evaluator_input(output: Mapping[str, Any]) -> Dict[str, Any]:


    return {
        field: output[field]
        for field in _FINAL_EVALUATOR_INPUT_FIELDS
        if field in output
    }


def _final_evaluator_structure_stage(output: Mapping[str, Any]) -> str:


    runtime = output.get("_causal_runtime")
    selected = (
        runtime.get("selected_candidate")
        if isinstance(runtime, Mapping)
        else None
    )
    selected = selected if isinstance(selected, Mapping) else {}

    if output.get("physics_evaluated") is True or selected.get(
        "physics_evaluated"
    ) is True:
        return "physics"

    stage = str(
        selected.get("structure_stage")
        or output.get("structure_stage")
        or ""
    ).strip().lower()
    if stage:
        return stage

    artifacts = output.get("search_artifacts")
    summary = (
        artifacts.get("structure_evaluation_summary")
        if isinstance(artifacts, Mapping)
        else None
    )
    stage = str(
        summary.get("selected_stage") if isinstance(summary, Mapping) else ""
    ).strip().lower()
    return stage or "final"


def _evaluate_final_candidate(
    output: Dict[str, Any],
    prepared: PreparedCaseInputs,
    request: SearchRequest,
) -> Dict[str, Any]:


    sequences = output.get("seqs") or output.get("best_seqs")
    if not isinstance(sequences, Mapping):
        raise ValueError("final evaluator input has no sequence bundle")
    register_sequence_occurrence(
        sequences,
        role="final_evaluator_input",
        context_id=(
            f"{getattr(request.inner_run_memory, 'run_instance_id', request.proposal_id)}"
            ":final_evaluator"
        ),
        metadata={"effective_contract_hash": request.effective_search_contract.contract_hash},
    )
    evaluator_input = _final_evaluator_input(output)
    evaluation_stage = _final_evaluator_structure_stage(output)
    active_score_config = _score_config_for_structure_stage(
        prepared.score_config,
        evaluation_stage,
    )
    report, cache_artifact = evaluate_exact_persistently(
        sequences,
        tool="final_evaluator",
        tool_version="astevolve.evaluator_engine.v1",
        model="configured_plugin_graph",
        config={
            "compiled": request.legacy_compiled,
            "design_state": prepared.design_state,
            "masks": prepared.masks,
            "template_sequences": prepared.template_sequences,
            "fixed_residues": prepared.fixed_residues,
            "score_config": active_score_config,
        },
        state=evaluator_input,
        seed=request.seed,


        estimated_cost=1.0,
        compute=lambda: evaluate_candidate(
            evaluator_input,
            compiled=request.legacy_compiled,
            design_state=prepared.design_state,
            masks=prepared.masks,
            template_seqs=prepared.template_sequences,
            fixed_residues=prepared.fixed_residues,
            score_config=active_score_config,
        ),
    )
    merged = merge_inner_semantic_audit(
        report,
        output.get("inner_loop_semantic_audit", {}) or {},
    )
    merged["persistent_evaluation_cache"] = cache_artifact
    return merged


def _attach_feedback(
    output: Dict[str, Any],
    prepared: PreparedCaseInputs,
    evaluator_report: Mapping[str, Any],
) -> None:


    state = prepared.design_state
    round_summary = ((output.get("search_artifacts") or {}).get("round_summary") or {})
    candidate_comparison = (
        round_summary.get("candidate_comparison", {})
        if isinstance(round_summary, dict)
        else {}
    )
    experiment_analysis = (
        round_summary.get("experiment_analysis_report", {})
        if isinstance(round_summary, dict)
        else {}
    )
    graph_summary = output.get("semantic_graph_summary", {})
    contract_disabled = output.get("graph_ablation_mode") == "no_edit_contract" or not (
        graph_summary.get("edit_contract_enabled", True)
    )

    if contract_disabled:
        diagnosis = {
            "schema_version": "ast_semantic_graph_diagnosis_v1",
            "enabled": bool(graph_summary.get("enabled")),
            "ablation_mode": output.get("graph_ablation_mode"),
            "disabled_reason": "graph ablation disables edit_contract generation",
            "hard_gate_pass": bool(evaluator_report.get("hard_gate_pass", True)),
            "disqualification_reasons": list(
                evaluator_report.get("disqualification_reasons", []) or []
            ),
            "candidate_comparison": candidate_comparison,
            "experiment_analysis_report": experiment_analysis,
        }
        edit_contract = {
            "schema_version": "ast_edit_contract_v2",
            "action": "freeze_node",
            "required_nodes": [],
            "forbidden_nodes": [],
            "mutation_budget": {"min": 0, "max": 0},
            "rationale": "graph ablation disables edit_contract generation",
            "metadata": {
                "disabled": True,
                "ablation_mode": output.get("graph_ablation_mode"),
            },
        }
    else:
        diagnosis = dict(diagnose_semantic_graph(evaluator_report, graph_summary, state))
        diagnosis["candidate_comparison"] = candidate_comparison
        diagnosis["experiment_analysis_report"] = experiment_analysis
        edit_contract = generate_edit_contract(
            diagnosis,
            state,
            state.get("_case_sheet", {}),
        )

    output["semantic_graph_diagnosis"] = diagnosis
    output["edit_contract"] = edit_contract


def _persist_memory_update(
    output: Dict[str, Any],
    strategy: Mapping[str, Any],
    config: SAConfig,
    memory_path: Path,
    memory_snapshot: MemorySnapshot,
    commit_mode: str,
    scope: Optional[MemoryExecutionContext],
    memory_policy: MemoryPolicyConfig,
) -> None:


    if not memory_policy.may_read_adaptive_prior:
        output["memory_update"] = {
            "schema_version": "astevolve.adaptive_memory_update.v1",
            "mode": memory_policy.adaptive_prior_mode,
            "updated": False,
            "deferred": False,
            "proposal": None,
            "reason": "adaptive_prior_off",
        }
        return
    if not memory_policy.may_commit_adaptive_prior:
        output["memory_update"] = {
            "schema_version": "astevolve.adaptive_memory_update.v1",
            "mode": memory_policy.adaptive_prior_mode,
            "updated": False,
            "deferred": False,
            "proposal": None,
            "reason": "adaptive_prior_read_only",
        }
        return
    if memory_policy.may_commit_adaptive_prior:
        update = update_internal_memory(
            memory_path,
            output,
            max_recent_runs=10,
            max_residues_per_node=8,
            snapshot=memory_snapshot,
            commit_mode=commit_mode,
            scope=scope,
        )
        output["memory_update"] = {key: value for key, value in update.items() if key != "memory"}


def _target_memory_path(
    memory_path: Optional[str],
    state: Mapping[str, Any],
    scope: Optional[MemoryExecutionContext],
) -> Path:


    if scope is not None and scope.target_path:
        return Path(scope.target_path)
    configured = memory_path or state.get("memory_path")
    return resolve_memory_path(str(configured) if configured else None)


def _execute_prepared_design_search(
    *,
    request: SearchRequest,
    prepared: PreparedCaseInputs,
    config: SAConfig,
    effective_policy: MemoryPolicyConfig,
    effective_memory_scope: MemoryScope,
    external_knowledge_policy: Mapping[str, Any],
    memory_path: Optional[str],
    effective_mode: str,
    scope: Optional[MemoryExecutionContext],
) -> Dict[str, Any]:


    output = _execute_legacy_search(request)
    output["memory_policy"] = {
        **effective_policy.to_artifact(),
        "scope": effective_memory_scope.to_artifact(),
        "adaptive_prior_visible": bool(request.adaptive_prior),
    }
    output["external_knowledge_policy"] = dict(external_knowledge_policy)
    _attach_compilation_metadata(output, prepared, request)
    _attach_semantic_maps(output, prepared, request)
    evaluator_report = _evaluate_final_candidate(output, prepared, request)
    causal_runtime = output.pop("_causal_runtime", None)
    if not isinstance(causal_runtime, Mapping):
        causal_runtime = _compatibility_causal_runtime(output, prepared, request)
    evaluated_sequences = output.get("seqs") or output.get("best_seqs")
    if not isinstance(evaluated_sequences, Mapping):
        raise ValueError("final evaluator output has no sequence bundle")
    evaluated_sequence_id = SequenceRecord.create(evaluated_sequences).semantic_id
    existing_evaluated_id = evaluator_report.get("evaluated_sequence_id")
    if existing_evaluated_id not in (None, "", evaluated_sequence_id):
        raise ValueError(
            "evaluator report evaluated_sequence_id does not match the sequence "
            "bundle supplied by the orchestrator"
        )
    evaluator_report = dict(evaluator_report)
    evaluator_report["evaluated_sequence_id"] = evaluated_sequence_id
    mapping_projection = project_selected_mapping_runtime(
        runtime=causal_runtime,
        mapping_plan=request.executable_mapping_plan,
        effective_mapping_schedule=request.effective_mapping_schedule,
        evaluator_report=evaluator_report,
    )
    evaluator_report = dict(mapping_projection.evaluator_report)
    output["mapping_execution"] = dict(mapping_projection.artifact)
    output["evaluator_report"] = evaluator_report
    output["evaluator_score"] = float(
        evaluator_report.get("normalized_score", 0.0) or 0.0
    )
    raw_evaluator_loss = evaluator_report.get("loss", 1.0)
    output["evaluator_loss"] = float(
        1.0 if raw_evaluator_loss is None else raw_evaluator_loss
    )
    output["evaluator_energy"] = float(
        evaluator_report.get("total_energy", output["evaluator_loss"]) or 0.0
    )
    output["final_energy"] = output["evaluator_energy"]
    output["energy_direction"] = str(
        evaluator_report.get("direction")
        or evaluator_report.get("energy_direction")
        or "minimize"
    )
    output["energy_breakdown"] = {
        "terms": list(evaluator_report.get("term_energy_breakdown") or []),
        "categories": dict(
            evaluator_report.get("category_energy_breakdown") or {}
        ),
        "coverage": dict(evaluator_report.get("energy_coverage") or {}),
    }
    causal_trace = build_selected_causal_trace(
        runtime=causal_runtime,
        graph_patch=request.graph_patch,
        parent_contract=request.parent_effective_search_contract,
        effective_contract=request.effective_search_contract,
        evaluator_report=evaluator_report,
        mapping_execution_traces=mapping_projection.traces,
        exact_measurements=mapping_projection.final_measurements,
        seed=request.seed,
    )
    output.update(causal_artifacts(causal_trace))
    output["compiled_graph_patch_hash"] = request.graph_patch.patch_hash
    output["immutable_sequence_reference"] = SequenceBundleIdentity.create(
        prepared.immutable_reference_sequences
    ).to_dict()
    if prepared.compiled_design_action is not None:
        output["design_action"] = deepcopy(
            prepared.design_state.get("_design_action") or {}
        )
        output["compiled_design_action"] = (
            prepared.compiled_design_action.to_artifact()
        )
        output["mutation_ownership_ledger"] = (
            prepared.compiled_design_action.ownership_ledger.to_dict()
        )
    if prepared.compiled_portfolio_optimization_request is not None:
        output["compiled_portfolio_optimization_request"] = (
            prepared.compiled_portfolio_optimization_request.to_artifact()
        )
    _attach_feedback(output, prepared, evaluator_report)

    state = prepared.design_state
    output["applied_edit_contract"] = state.get("_applied_edit_contract")
    output["strategy_schema_report"] = state.get("_strategy_schema_report", {})
    output["score_config"] = prepared.score_config
    output["design_state_version"] = state.get("version")
    target_path = _target_memory_path(memory_path, state, scope)
    output["memory_lifecycle"] = {
        "schema_version": "astevolve.inner_memory_lifecycle.v1",
        "commit_mode": effective_mode,
        "input_snapshot": prepared.memory_snapshot.to_artifact(),
        "target_path": str(target_path),
        "scope": scope.to_artifact() if scope is not None else None,
        "memory_scope": effective_memory_scope.to_artifact(),
        "memory_policy": effective_policy.to_artifact(),
        "inner_run_memory": request.inner_run_memory.cache_summary(),
    }
    _persist_memory_update(
        output,
        prepared.resolved_strategy,
        config,
        target_path,
        prepared.memory_snapshot,
        effective_mode,
        scope,
        effective_policy,
    )
    memory_update = output.get("memory_update", {}) or {}
    output["memory_lifecycle"].update(
        {
            "reason": memory_update.get("reason"),
            "proposed_content_hash": memory_update.get("proposed_content_hash"),
            "output_content_hash": memory_update.get("output_content_hash"),
            "commit_id": (
                (memory_update.get("commit") or {}).get("commit_id")
                or (memory_update.get("proposal") or {}).get("commit_id")
            ),
        }
    )
    return output


def run_design_search(
    strategy: Dict[str, Any],
    seed: Optional[int] = None,
    design_state_path: Optional[str] = None,
    memory_path: Optional[str] = None,
    memory_snapshot: Optional[MemorySnapshot] = None,
    memory_commit_mode: Optional[str] = None,
    memory_policy: Optional[MemoryPolicyConfig] = None,
    memory_scope: Optional[MemoryScope] = None,
    mapping_execution_mode: str = "full",
) -> Dict[str, Any]:


    scope = current_memory_execution_context()
    if scope is not None and memory_policy is not None and memory_policy != scope.memory_policy:
        raise MemoryPolicyError("explicit memory policy conflicts with outer execution scope")
    if scope is not None and memory_scope is not None and scope.memory_scope is not None:
        memory_scope.require_compatible(scope.memory_scope, level="lineage")
    effective_policy = (
        scope.memory_policy
        if scope is not None
        else memory_policy
        if memory_policy is not None
        else MemoryPolicyConfig()
    )
    effective_snapshot = memory_snapshot or (scope.snapshot if scope else None)
    effective_mode = str(
        memory_commit_mode
        or (scope.commit_mode if scope is not None else "deferred")
    ).strip().lower()
    if effective_mode not in MEMORY_COMMIT_MODES:
        raise ValueError(
            f"unsupported memory commit mode {effective_mode!r}; "
            f"expected one of {sorted(MEMORY_COMMIT_MODES)}"
        )
    run_strategy = deepcopy(strategy)
    external_knowledge_policy = build_external_knowledge_policy(run_strategy)

    prepared = prepare_case_inputs(
        run_strategy,
        design_state_path=design_state_path,
        memory_path=memory_path,
        memory_snapshot=effective_snapshot,
        mapping_execution_mode=mapping_execution_mode,
        runtime_mcts_output_dir=(
            str(scope.output_dir)
            if scope is not None and scope.output_dir
            else None
        ),
    )
    effective_memory_scope = (
        scope.memory_scope
        if scope is not None and scope.memory_scope is not None
        else memory_scope
        if memory_scope is not None
        else MemoryScope(
            case_id=str(
                prepared.design_state.get("case_id")
                or prepared.design_state.get("task_name")
                or "direct_case"
            ),
            run_id="direct_run",
            lineage_id="direct_lineage",
        )
    )
    scoped_adaptive_snapshot = ScopedAdaptivePriorSnapshot(
        scope=effective_memory_scope,
        snapshot=prepared.memory_snapshot,
    )
    config = SAConfig(**prepared.search_config, seed=seed)
    island_directive_receipt = None
    if scope is not None and scope.island_role:
        from outerloop.island_runtime import apply_executable_island_directive

        island_directive_receipt = apply_executable_island_directive(
            config,
            scope.island_role,
            int(scope.island_id or 0),
        )


    artifact_mode = os.environ.get("ASTEVOLVE_MCTS_ARTIFACT_MODE")
    if artifact_mode:
        mode = str(artifact_mode).strip().lower()
        if mode not in {"normalized", "legacy_full"}:
            raise ValueError(
                "ASTEVOLVE_MCTS_ARTIFACT_MODE must be normalized or legacy_full"
            )
        config.mcts_artifact_mode = mode
    run_instance_id = (
        scope.scope_id
        if scope is not None and scope.scope_id
        else f"direct-inner-run:{seed if seed is not None else 'none'}"
    )
    request = SearchRequest.from_prepared(
        prepared,
        config=config,
        internal_memory=prepared.memory_snapshot.materialize(),
        memory_policy=effective_policy,
        memory_scope=effective_memory_scope,
        scoped_adaptive_snapshot=scoped_adaptive_snapshot,
        inner_run_memory=InnerRunMemory(
            scope=effective_memory_scope,
            run_instance_id=run_instance_id,
        ),
        run_instance_id=run_instance_id,
    )


    if request.accepted_patch_requires_effect:
        require_effective_contract_delta(
            request.parent_effective_search_contract,
            request.effective_search_contract,
        )


    contract_lease = claim_effective_contract(
        request.effective_search_contract.contract_hash
    )
    with maintain_effective_contract_lease(contract_lease):
        try:
            register_sequence_occurrence(
                prepared.template_sequences,
                role="template",
                context_id=f"{run_instance_id}:template",
                metadata={
                    "effective_contract_hash": request.effective_search_contract.contract_hash
                },
            )
            output = _execute_prepared_design_search(
                request=request,
                prepared=prepared,
                config=config,
                effective_policy=effective_policy,
                effective_memory_scope=effective_memory_scope,
                external_knowledge_policy=external_knowledge_policy,
                memory_path=memory_path,
                effective_mode=effective_mode,
                scope=scope,
            )
        except Exception as exc:
            fail_effective_contract(contract_lease, exc)
            raise
        complete_effective_contract(contract_lease)
    if contract_lease is not None:
        output["effective_contract_admission"] = contract_lease.to_artifact(
            status="completed"
        )
    if island_directive_receipt is not None:
        output["executable_island_directive"] = island_directive_receipt
    metrics = registry_metrics_artifact()
    if metrics is not None:
        output["experiment_registry_metrics"] = metrics
    return output
