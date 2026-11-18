

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .external_knowledge_policy import EXTERNAL_KB_FIELDS


REPORT_VERSION = "astevolve.strategy_effect_report.v1"
REPORT_MODE = "warning"
_MISSING = object()

DIAGNOSTIC_ONLY_FIELD_REASONS = {
    "functional_nodes": (
        "diagnostic_only: functional-node labels have no executable strategy consumer; "
        "functional-to-structural grounding is deferred to AST-01A."
    ),
    "coupling_edges": (
        "diagnostic_only: static coupling-edge labels have no executable search consumer; "
        "edge operation semantics are deferred to AST-01B."
    ),
    "semantic_focus": (
        "diagnostic_only: semantic_focus has no executable strategy consumer; "
        "functional-to-structural grounding is deferred to AST-01A."
    ),
}

CONTROLLER_MEMORY_POLICY_FIELDS = frozenset(
    {
        "adaptive_prior_mode",
        "inner_state_scope",
        "mcts_memory_enabled",
        "memory_auto_update_enabled",
        "memory_update_max_recent_runs",
        "memory_update_max_residues_per_node",
    }
)


@dataclass(frozen=True)
class FieldSpec:


    canonical: str
    consumer: Optional[str]


_SEARCH_CONFIG_FIELDS = (
    "iterations",
    "init_temp",
    "cooling",
    "mutation_rate",
    "resample_segment_prob",
    "progen_weight",
    "progen_chains",
    "sequence_prior_model",
    "inner_structure_enabled",
    "inner_structure_model",
    "inner_structure_model_name",
    "inner_structure_weight",
    "inner_structure_fail_closed",
    "inner_structure_hard_gate",
    "inner_structure_failure_penalty",
    "promote_inline_winner_structure_evidence",


    "inner_esmfold2_enabled",
    "inner_esmfold2_model_name",
    "inner_esmfold2_interval",
    "inner_esmfold2_weight",
    "mcts_node_sweep_enabled",
    "mcts_node_sweep_count",
    "mcts_node_sweep_parent_policy",
    "mcts_fidelity_upgrade_enabled",
    "mcts_fidelity_upgrade_provider",
    "mcts_fidelity_upgrade_interval",
    "mcts_fidelity_upgrade_candidates",
    "mcts_fidelity_upgrade_final_candidates",
    "mcts_fidelity_upgrade_required",
    "node_optimizer_device",
    "chai1_enabled",
    "chai1_top_frac",
    "chai1_min_candidates",
    "chai1_max_candidates",
    "protenix_model_name",
    "protenix_conda_env",
    "protenix_seed",
    "protenix_complex_use_msa",
    "protenix_complex_cycle",
    "protenix_complex_step",
    "protenix_complex_sample",
    "protenix_complex_use_default_params",
    "protenix_complex_timeout",
    "af3_model_dir",
    "af3_conda_env",
    "af3_run_data_pipeline",
    "af3_db_dir",
    "af3_num_recycles",
    "af3_num_diffusion_samples",
    "af3_timeout",
    "af3_flash_attention_implementation",
    "af3_gpu_device",
    "af3_seed",
    "structure_multiseed_enabled",
    "structure_formal_funnel_enabled",
    "structure_protenix_seeds",
    "structure_af3_seeds",
    "structure_robust_top_candidates",
    "structure_disagreement_threshold",
    "structure_pyrosetta_required",
    "structure_model",
    "structure_model_name",
    "structure_prescreen_enabled",
    "structure_prescreen_model",
    "structure_prescreen_model_name",
    "structure_prescreen_top_frac",
    "structure_prescreen_min_candidates",
    "structure_prescreen_max_candidates",
    "structure_prescreen_forward_all_to_screen",
    "structure_screen_model",
    "structure_screen_model_name",
    "structure_screen_enabled",
    "structure_screen_all_candidates",
    "structure_screen_top_frac",
    "structure_screen_min_candidates",
    "structure_screen_max_candidates",
    "structure_screen_progen_batch_size",
    "structure_rerank_model",
    "structure_rerank_model_name",
    "structure_rerank_enabled",
    "structure_rerank_top_frac",
    "structure_rerank_min_candidates",
    "structure_rerank_max_candidates",
    "structure_physics_max_candidates",
    "structure_rerank_all_infeasible_rescue",
    "structure_shortlist_policy",
    "structure_screen_single_node_diagnostic_quota",
    "structure_selection_objective",
    "structure_stepping_stone_enabled",
    "structure_stepping_stone_max_energy_degradation",
    "structure_stepping_stone_metrics",
    "structure_stepping_stone_min_metric_gain",
    "structure_allow_low_fidelity_fallback",
    "structure_batch_size",
    "structure_parallel_workers",
    "structure_service_backend",
    "esmfold2_mode",
    "esmfold2_conda_env",
    "esmfold2_num_loops",
    "esmfold2_num_sampling_steps",
    "esmfold2_num_diffusion_samples",
    "multistate_objectives_enabled",
    "multistate_objective_weight",
    "mutation_ops",
    "history_size",
    "search_method",
    "mcts_c_puct",
    "mcts_max_depth",
    "mcts_progressive_widening_c",
    "mcts_progressive_widening_alpha",
    "mcts_reward_scale",
    "mcts_iteration_unit",
    "mcts_candidate_budget_max_round_multiplier",
    "mcts_candidate_budget_fail_on_underfill",
    "mcts_output_dir",
    "mcts_save_tree",
    "mcts_save_variants",
    "mcts_artifact_mode",
    "mcts_tree_quality_required",
    "mcts_tree_min_root_children",
    "mcts_tree_min_branching_nodes",
    "mcts_tree_min_leaves",
    "mcts_tree_min_max_depth",
    "candidate_wave_enabled",
    "candidate_wave_size",
    "candidate_wave_fail_on_underfill",
    "candidate_wave_protenix_mutant_quota",
    "candidate_wave_af3_mutant_quota",
    "candidate_wave_changed_node_min_generated_unique",
    "candidate_wave_changed_node_min_frozen_unique",
    "candidate_wave_changed_node_min_protenix_attempts",
    "executable_island_policy_enabled",
    "node_edit_policies",
    "residue_mutation_contract",
    "proposal_engine",
    "sequence_generator_id",
    "sequence_generator_structure_condition_refs",
    "sequence_generator_state_condition_refs",
    "node_optimizer_enabled",
    "node_optimizer_candidate_count",
    "node_optimizer_beam_width",
    "node_optimizer_top_k_per_position",
    "node_optimizer_temperature",
    "node_optimizer_diversity_weight",
    "node_optimizer_mutation_penalty",
    "node_optimizer_prior_model",
    "proposal_tier_mode",
    "proposal_exploit_frac",
    "proposal_explore_frac",
    "proposal_repair_frac",
    "exploit_max_mutations",
    "explore_max_mutations",
    "repair_max_mutations",
    "max_total_mutations",
    "fast_filter_enabled",
    "sequence_prefilter_callable",
    "sequence_prefilter_config",
    "sequence_bootstrap_callable",
    "sequence_bootstrap_config",
    "semantic_required_nodes",
    "semantic_active_nodes",
    "semantic_anchor_nodes",
    "semantic_required_node_min_visits",
    "semantic_required_node_min_mutations",
    "semantic_coverage_mode",
    "semantic_missing_node_penalty",
    "semantic_required_node_round_robin",
    "semantic_required_node_force_steps",
    "outer_loop_phase",
    "search_schedule",
)

TOP_LEVEL_FIELD_SPECS: Dict[str, FieldSpec] = {
    name: FieldSpec(name, f"engine.runtime_profile.build_sa_config:{name}")
    for name in _SEARCH_CONFIG_FIELDS
}
TOP_LEVEL_FIELD_SPECS.update(
    {
        "preferred_edit_order": FieldSpec(
            "preferred_edit_order", "astevolve.search.proposal_sampling:node_order"
        ),
        "resume_template_seqs": FieldSpec(
            "resume_template_seqs", "engine.case_builder.prepare_case_inputs:templates"
        ),
        "graph_ablation_mode": FieldSpec(
            "graph_ablation_mode", "engine.runtime_profile.resolve_graph_ablation_mode"
        ),
        "edit_contract": FieldSpec(
            "edit_contract", "astevolve.semantic_graph.apply_edit_contract_to_strategy"
        ),
        "ast_revision_plan": FieldSpec(
            "ast_revision_plan",
            "astevolve.semantic_graph.apply_ast_revision_plan",
        ),
        "case_owned_residue_policy_tier": FieldSpec(
            "case_owned_residue_policy_tier",
            "engine.node_compiler.apply_case_owned_residue_policy:tier",
        ),
        "case_owned_residue_policy_resolution": FieldSpec(
            "case_owned_residue_policy_resolution",
            "engine.case_builder.prepare_case_inputs:hard_residue_audit",
        ),
        "last_contract_response": FieldSpec(
            "last_contract_response",
            "engine.case_builder.prepare_case_inputs:contract_response",
        ),
        "max_hydrophobic_run": FieldSpec(
            "max_hydrophobic_run", "engine.design_compiler.build_constraint_specs"
        ),
        "max_charged_run": FieldSpec(
            "max_charged_run", "engine.design_compiler.build_constraint_specs"
        ),
        "cdr_hydrophobic_max": FieldSpec(
            "cdr_hydrophobic_max", "engine.design_compiler.build_constraint_specs"
        ),
        "cdr_charged_max": FieldSpec(
            "cdr_charged_max", "engine.design_compiler.build_constraint_specs"
        ),
        "cdr_favored_residues": FieldSpec(
            "cdr_favored_residues", "engine.design_compiler.build_constraint_specs"
        ),
        "linker_hydrophobic_max": FieldSpec(
            "linker_hydrophobic_max", "engine.design_compiler.build_constraint_specs"
        ),
        "linker_charged_max": FieldSpec(
            "linker_charged_max", "engine.design_compiler.build_constraint_specs"
        ),
        "linker_gs_min": FieldSpec(
            "linker_gs_min", "engine.design_compiler.build_constraint_specs"
        ),
        "pocket_hydrophobic_max": FieldSpec(
            "pocket_hydrophobic_max", "engine.design_compiler.build_constraint_specs"
        ),
        "secondary_structure_weight": FieldSpec(
            "secondary_structure_weight", "engine.design_compiler.build_constraint_specs"
        ),
        "desired_cdr3_hydro": FieldSpec(
            "desired_cdr3_hydro", "engine.design_compiler.build_constraint_specs"
        ),
        "esmfold2_model_name": FieldSpec(
            "esmfold2_model_name", "engine.runtime_profile.build_sa_config:structure_model_name"
        ),
    }
)

PLAN_FIELD_SPECS: Dict[str, FieldSpec] = {
    "binder_domain_order": FieldSpec(
        "binder_domain_order", "engine.strategy_compiler._apply_layout_domain_order"
    ),
    "secondary_structure_priors": FieldSpec(
        "secondary_structure_priors", "engine.design_compiler._secondary_structure_constraint_specs"
    ),
    "design_regions": FieldSpec(
        "design_regions", "engine.strategy_compiler._apply_layout_plan_to_policies"
    ),
    "regions": FieldSpec(
        "design_regions", "engine.strategy_compiler._apply_layout_plan_to_policies"
    ),
}

_NODE_POLICY_CONSUMER = "astevolve.search.node_mutation:node_edit_policy"
_POLICY_FIELDS = (
    "name",
    "role",
    "position",
    "bind_to",
    "enabled",
    "mutable",
    "priority_boost",
    "mutation_rate",
    "max_mutations_per_step",
    "policy_weight",
    "mutation_ops",
    "hotspot_positions",
    "anchor_positions",
    "mutable_positions",
    "protected_positions",
    "graft_motifs",
    "motif_candidates",
    "operator_phase",
    "large_jump",
    "favored_residues",
    "disfavored_residues",
    "favored_residue_classes",
    "disfavored_residue_classes",
    "length_budget",
    "target_lengths",
    "length_deltas",
    "length_ranges",
    "node_weights",
    "length_range",
    "target_length",
    "length_mutable",
    "allow_framework_length_change",
    "secondary_structure",
    "site_anchors",
    "position_residue_rules",
)
REGION_FIELD_SPECS: Dict[str, FieldSpec] = {
    name: FieldSpec(name, _NODE_POLICY_CONSUMER) for name in _POLICY_FIELDS
}
REGION_FIELD_SPECS.update(
    {
        "intent": FieldSpec("role", _NODE_POLICY_CONSUMER),
        "nodes": FieldSpec("bind_to", _NODE_POLICY_CONSUMER),
        "segments": FieldSpec("bind_to", _NODE_POLICY_CONSUMER),
        "target_nodes": FieldSpec("bind_to", _NODE_POLICY_CONSUMER),
        "covers": FieldSpec("bind_to", _NODE_POLICY_CONSUMER),
        "domain": FieldSpec("bind_to", _NODE_POLICY_CONSUMER),
        "parent_domain": FieldSpec("bind_to", _NODE_POLICY_CONSUMER),
        "kind_filter": FieldSpec("bind_to", _NODE_POLICY_CONSUMER),
        "node_kind": FieldSpec("bind_to", _NODE_POLICY_CONSUMER),
        "kind": FieldSpec("bind_to", _NODE_POLICY_CONSUMER),
        "max_nodes": FieldSpec("bind_to", _NODE_POLICY_CONSUMER),
        "ss": FieldSpec("secondary_structure", _NODE_POLICY_CONSUMER),
    }
)


UNSUPPORTED_REGION_FIELDS = {
    "functional_nodes",
    "coupling_edges",
    "design_points",
    "target_tail_prior",
    "required_source_contact_positions_construct",
    "auxiliary_source_contact_positions_construct",
    "negative_design_targets",
    "positive_design_targets",
    "position_weights",
    "aa_weights",
    "fill_residues",
    "confidence",
    "length_bias",
    "length",
    "length_delta",
    "min_length",
    "max_length",
    "seed_sequence",
    "template_sequence",
    "edit_intent",
    "priority",
}

SITE_ANCHOR_FIELD_SPECS: Dict[str, FieldSpec] = {
    "relative_positions": FieldSpec("relative_positions", _NODE_POLICY_CONSUMER),
    "positions": FieldSpec("relative_positions", _NODE_POLICY_CONSUMER),
    "relative_ranges": FieldSpec("relative_ranges", _NODE_POLICY_CONSUMER),
    "weight": FieldSpec("weight", _NODE_POLICY_CONSUMER),
    "priority_boost": FieldSpec("weight", _NODE_POLICY_CONSUMER),
    "favored_residues": FieldSpec("favored_residues", _NODE_POLICY_CONSUMER),
    "favored_residue_classes": FieldSpec(
        "favored_residue_classes", _NODE_POLICY_CONSUMER
    ),
}

TREE_NODE_FIELD_SPECS: Dict[str, FieldSpec] = {
    name: FieldSpec(name, _NODE_POLICY_CONSUMER)
    for name in (
        "name",
        "node",
        "children",
        "edit_policy",
        "target_length",
        "length",
        "length_delta",
        "length_range",
        "min_length",
        "max_length",
        "length_mutable",
        "mutable",
        "enabled",
        "priority",
        "priority_boost",
        "mutation_rate",
        "mutation_ops",
        "max_mutations_per_step",
        "favored_residues",
        "disfavored_residues",
        "favored_residue_classes",
        "disfavored_residue_classes",
        "aa_weights",
        "fill_residues",
        "seed_sequence",
        "template_sequence",
        "policy_weight",
        "confidence",
        "edit_intent",
        "allow_framework_length_change",
        "secondary_structure",
        "position_weights",
        "position_residue_rules",
        "hotspot_positions",
        "anchor_positions",
        "mutable_positions",
        "protected_positions",
        "graft_motifs",
        "motif_candidates",
        "operator_phase",
        "large_jump",
        "site_anchors",
    )
}

_SCORE_FIELDS = {
    "weight_fast",
    "weight_plddt",
    "weight_iptm",
    "weight_ptm",
    "weight_ranking_score",
    "weight_interface_plddt",
    "weight_node_plddt_min",
    "weight_clash",
    "weight_multistate",
    "weight_evaluator",
    "plddt_scale",
    "clash_scale",
    "fast_loss_nonneg",
    "weight_eval_contract_response",
    "inner_evaluator_loss_weight",
    "inner_hard_gate_fail_penalty",
    "hard_gate_failure_score_scale",
    "hard_gate_disqualified_score",
}
_KNOWN_DROPPED_SCORE_FIELDS: set[str] = set()


def _path(tokens: Sequence[Any]) -> str:
    text = ""
    for token in tokens:
        if isinstance(token, int):
            text += f"[{token}]"
        else:
            text += ("." if text else "") + str(token)
    return text


def _iter_leaves(value: Any, tokens: Tuple[Any, ...] = ()) -> Iterable[Tuple[Tuple[Any, ...], Any]]:
    if isinstance(value, Mapping):
        if not value:
            if tokens:
                yield tokens, value
            return
        for key, child in value.items():
            yield from _iter_leaves(child, tokens + (str(key),))
        return
    if isinstance(value, list) and any(isinstance(item, Mapping) for item in value):
        if not value:
            if tokens:
                yield tokens, value
            return
        for index, child in enumerate(value):
            child_tokens = tokens + (index,)
            if isinstance(child, Mapping):
                yield from _iter_leaves(child, child_tokens)
            else:
                yield child_tokens, child
        return
    if tokens:
        yield tokens, value


def _get(value: Any, tokens: Sequence[Any], default: Any = _MISSING) -> Any:
    current = value
    for token in tokens:
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                return default
            current = current[token]
        else:
            if not isinstance(current, Mapping) or token not in current:
                return default
            current = current[token]
    return current


def _redact(path: str, value: Any) -> Any:
    lowered = path.lower()
    if any(word in lowered for word in ("password", "api_key", "apikey", "access_token", "secret")):
        return "<redacted>"
    return value


def _record(
    path: str,
    status: str,
    requested: Any,
    effective: Any,
    reason: str,
    consumer: Optional[str],
) -> Dict[str, Any]:
    return {
        "path": path,
        "status": status,
        "requested": _redact(path, requested),
        "effective": None if effective is _MISSING else _redact(path, effective),
        "reason": reason,
        "consumer": consumer,
    }


def _effective_status(
    path: str,
    requested: Any,
    effective: Any,
    consumer: str,
    *,
    alias: bool = False,
) -> Dict[str, Any]:
    if effective is _MISSING:
        return _record(
            path,
            "rejected",
            requested,
            _MISSING,
            "Known field was dropped before reaching its declared consumer.",
            None,
        )
    if alias:
        return _record(
            path,
            "transformed",
            requested,
            effective,
            "Alias was canonicalized before runtime consumption.",
            consumer,
        )
    if requested == effective:
        return _record(path, "applied", requested, effective, "Applied unchanged.", consumer)
    if isinstance(requested, (int, float)) and not isinstance(requested, bool):
        reason = "Value was clamped or normalized by compiler/runtime bounds."
    else:
        reason = "Value was normalized, filtered, canonicalized, or overridden before consumption."
    return _record(path, "transformed", requested, effective, reason, consumer)


def _find_compiled_region(
    requested_plan: Mapping[str, Any],
    resolved_plan: Mapping[str, Any],
    index: int,
) -> Any:
    raw_regions = requested_plan.get("design_regions", requested_plan.get("regions", []))
    if not isinstance(raw_regions, list) or index >= len(raw_regions) or index >= 8:
        return _MISSING
    raw = raw_regions[index]
    if not isinstance(raw, Mapping):
        return _MISSING
    expected_name = str(raw.get("name") or f"design_region_{index + 1}")[:80]
    compiled = resolved_plan.get("design_regions", [])
    if not isinstance(compiled, list):
        return _MISSING
    for region in compiled:
        if isinstance(region, Mapping) and region.get("name") == expected_name:
            return region
    return _MISSING


def _classify_region_leaf(
    path: str,
    tokens: Sequence[Any],
    requested: Any,
    requested_plan: Mapping[str, Any],
    resolved_plan: Mapping[str, Any],
    *,
    plan_alias: bool,
) -> Dict[str, Any]:
    if len(tokens) < 4 or not isinstance(tokens[2], int):
        return _record(path, "unknown", requested, _MISSING, "Unknown design-region field path.", None)
    index = int(tokens[2])
    compiled_region = _find_compiled_region(requested_plan, resolved_plan, index)
    if compiled_region is _MISSING:
        return _record(
            path,
            "rejected",
            requested,
            _MISSING,
            "Region was not compiled (invalid type, no valid target nodes, or region limit exceeded).",
            None,
        )

    field = str(tokens[3])
    if field in DIAGNOSTIC_ONLY_FIELD_REASONS:
        return _record(
            path,
            "rejected",
            requested,
            _MISSING,
            DIAGNOSTIC_ONLY_FIELD_REASONS[field],
            None,
        )
    if field in UNSUPPORTED_REGION_FIELDS:
        return _record(
            path,
            "rejected",
            requested,
            _MISSING,
            "Field has no executable consumer on the current layout-region path.",
            None,
        )
    spec = REGION_FIELD_SPECS.get(field)
    if spec is None:
        return _record(path, "unknown", requested, _MISSING, "Unknown layout-region field.", None)

    tail = list(tokens[4:])
    canonical = spec.canonical
    alias = plan_alias or canonical != field
    if canonical == "site_anchors" and tail:
        node = tail[0]
        if len(tail) < 2:
            effective = _get(compiled_region, (canonical, node), _MISSING)
        else:
            anchor_field = str(tail[1])
            anchor_spec = SITE_ANCHOR_FIELD_SPECS.get(anchor_field)
            if anchor_spec is None:
                return _record(path, "unknown", requested, _MISSING, "Unknown site-anchor field.", None)
            alias = alias or anchor_spec.canonical != anchor_field
            effective = _get(
                compiled_region,
                (canonical, node, anchor_spec.canonical, *tail[2:]),
                _MISSING,
            )
    elif canonical == "position_residue_rules" and tail:
        rule_tail = list(tail)
        if len(rule_tail) >= 2:
            aliases = {"favored": "favored_residues", "disfavored": "disfavored_residues"}
            requested_rule_field = str(rule_tail[1])
            canonical_rule_field = aliases.get(requested_rule_field, requested_rule_field)
            alias = alias or requested_rule_field != canonical_rule_field
            rule_tail[1] = canonical_rule_field
        effective = _get(compiled_region, (canonical, *rule_tail), _MISSING)
    elif canonical in {"mutation_ops", "target_lengths", "length_deltas", "length_ranges", "node_weights"}:
        effective = _get(compiled_region, (canonical, *tail), _MISSING)
    elif tail:
        effective = _get(compiled_region, (canonical, *tail), _MISSING)
    else:
        effective = _get(compiled_region, (canonical,), _MISSING)
    return _effective_status(path, requested, effective, spec.consumer or "", alias=alias)


def _classify_plan_leaf(
    path: str,
    tokens: Sequence[Any],
    requested: Any,
    requested_strategy: Mapping[str, Any],
    resolved_strategy: Mapping[str, Any],
) -> Dict[str, Any]:
    root = str(tokens[0])
    requested_plan = requested_strategy.get(root)
    if not isinstance(requested_plan, Mapping):
        return _record(path, "rejected", requested, _MISSING, "Layout plan must be a mapping.", None)
    resolved_plan = resolved_strategy.get("layout_plan")
    if not isinstance(resolved_plan, Mapping):
        resolved_plan = {}
    if len(tokens) < 2:
        return _effective_status(
            path,
            requested,
            resolved_plan,
            "engine.strategy_compiler._apply_layout_plan_to_policies",
            alias=root != "layout_plan",
        )
    field = str(tokens[1])
    if field in DIAGNOSTIC_ONLY_FIELD_REASONS:
        return _record(
            path,
            "rejected",
            requested,
            _MISSING,
            DIAGNOSTIC_ONLY_FIELD_REASONS[field],
            None,
        )
    if field in {"domain_order", "binder_order"}:
        return _record(
            path,
            "rejected",
            requested,
            _MISSING,
            "Legacy order alias is dropped by the current sanitizer before its consumer.",
            None,
        )
    spec = PLAN_FIELD_SPECS.get(field)
    if spec is None:
        return _record(path, "unknown", requested, _MISSING, "Unknown layout-plan field.", None)
    if spec.canonical == "design_regions":
        return _classify_region_leaf(
            path,
            tokens,
            requested,
            requested_plan,
            resolved_plan,
            plan_alias=root != "layout_plan" or field != "design_regions",
        )
    tail = tuple(tokens[2:])
    effective = _get(resolved_plan, (spec.canonical, *tail), _MISSING)
    return _effective_status(
        path,
        requested,
        effective,
        spec.consumer or "",
        alias=root != "layout_plan" or field != spec.canonical,
    )


def _classify_tree_leaf(
    path: str,
    tokens: Sequence[Any],
    requested: Any,
    resolved_strategy: Mapping[str, Any],
) -> Dict[str, Any]:
    root = str(tokens[0])
    field = ""
    for token in reversed(tokens[1:]):
        if isinstance(token, str) and token not in {"children", "edit_policy"}:
            field = token
            break
    diagnostic_field = next(
        (
            str(token)
            for token in tokens
            if isinstance(token, str) and token in DIAGNOSTIC_ONLY_FIELD_REASONS
        ),
        None,
    )
    if diagnostic_field is not None:
        field = diagnostic_field
    if field in DIAGNOSTIC_ONLY_FIELD_REASONS:
        return _record(
            path,
            "rejected",
            requested,
            _MISSING,
            DIAGNOSTIC_ONLY_FIELD_REASONS[field],
            None,
        )
    if "position_residue_rules" in tokens:
        spec = FieldSpec("position_residue_rules", _NODE_POLICY_CONSUMER)
    else:
        spec = TREE_NODE_FIELD_SPECS.get(field)
    if spec is None:
        return _record(path, "unknown", requested, _MISSING, "Unknown strategy-tree field.", None)
    effective_tokens = list(tokens)
    alias = root != "strategy_tree"
    if "position_residue_rules" in effective_tokens:
        rule_index = effective_tokens.index("position_residue_rules")
        if len(effective_tokens) > rule_index + 2:
            field_index = rule_index + 2
            rule_aliases = {
                "favored": "favored_residues",
                "disfavored": "disfavored_residues",
            }
            requested_rule_field = str(effective_tokens[field_index])
            canonical_rule_field = rule_aliases.get(
                requested_rule_field,
                requested_rule_field,
            )
            effective_tokens[field_index] = canonical_rule_field
            alias = alias or requested_rule_field != canonical_rule_field
    effective = _get(resolved_strategy, effective_tokens, _MISSING)
    if effective is _MISSING and root != "strategy_tree":
        effective = _get(resolved_strategy, (root, *effective_tokens[1:]), _MISSING)
    return _effective_status(
        path,
        requested,
        effective,
        spec.consumer or "",
        alias=alias,
    )


def _classify_score_leaf(
    path: str,
    tokens: Sequence[Any],
    requested: Any,
    resolved_strategy: Mapping[str, Any],
    score_config: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    requested_score = resolved_strategy.get("score_config")
    effective_score: Mapping[str, Any]
    if score_config is not None:
        effective_score = score_config
    elif isinstance(requested_score, Mapping):
        effective_score = requested_score
    else:
        effective_score = {}
    if len(tokens) == 1:
        return _effective_status(
            path, requested, effective_score, "engine.runtime_profile.build_score_config"
        )
    field = str(tokens[1])
    tail = tuple(tokens[2:])
    consumer = "astevolve.evaluation.service:score_config"
    if field in _KNOWN_DROPPED_SCORE_FIELDS:
        return _record(
            path,
            "rejected",
            requested,
            _MISSING,
            "Known score field is dropped by build_score_config before its downstream reader.",
            None,
        )
    if field in _SCORE_FIELDS:
        effective = _get(effective_score, (field, *tail), _MISSING)
    elif field.startswith("eval_"):
        effective = _get(effective_score, ("evaluator_weights", field, *tail), _MISSING)
    elif field in {
        "evaluator_weights",
        "evaluator_backends",
        "evaluator_plugins",
        "plugin_config",
    }:
        effective = _get(effective_score, (field, *tail), _MISSING)
    else:
        return _record(path, "unknown", requested, _MISSING, "Unknown score-config field.", None)
    return _effective_status(path, requested, effective, consumer)


def _classify_top_level_leaf(
    path: str,
    tokens: Sequence[Any],
    requested: Any,
    resolved_strategy: Mapping[str, Any],
    search_config: Optional[Mapping[str, Any]],
    graph_ablation_mode: Optional[str],
) -> Dict[str, Any]:
    root = str(tokens[0])
    if root in EXTERNAL_KB_FIELDS:
        return _record(
            path,
            "rejected",
            requested,
            _MISSING,
            "Deprecated and unsupported: external KB inputs have no formal runtime consumer.",
            None,
        )
    if root in CONTROLLER_MEMORY_POLICY_FIELDS:
        return _record(
            path,
            "rejected",
            requested,
            _MISSING,
            "Controller-locked: adaptive memory policy is not an evolvable strategy field.",
            None,
        )
    if root in DIAGNOSTIC_ONLY_FIELD_REASONS:
        return _record(
            path,
            "rejected",
            requested,
            _MISSING,
            DIAGNOSTIC_ONLY_FIELD_REASONS[root],
            None,
        )
    spec = TOP_LEVEL_FIELD_SPECS.get(root)
    if spec is None:
        return _record(path, "unknown", requested, _MISSING, "Unknown top-level strategy field.", None)
    tail = tuple(tokens[1:])
    if root == "graph_ablation_mode" and not tail and graph_ablation_mode is not None:
        effective = graph_ablation_mode
    elif search_config is not None and root in search_config:
        effective = _get(search_config, (root, *tail), _MISSING)
    else:
        effective = _get(resolved_strategy, (root, *tail), _MISSING)
    return _effective_status(path, requested, effective, spec.consumer or "")


def build_strategy_effect_report(
    requested_strategy: Mapping[str, Any],
    resolved_strategy: Mapping[str, Any],
    *,
    search_config: Optional[Mapping[str, Any]] = None,
    score_config: Optional[Mapping[str, Any]] = None,
    graph_ablation_mode: Optional[str] = None,
    legacy_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:


    report: Dict[str, Any] = {
        "schema_version": REPORT_VERSION,
        "mode": REPORT_MODE,
        "requested": [],
        "applied": [],
        "transformed": [],
        "rejected": [],
        "unknown": [],
    }
    layout_roots = {"layout_plan", "domain_layout", "node_layout"}
    tree_roots = {"strategy_tree", "design_tree", "node_tree"}
    for tokens, value in _iter_leaves(requested_strategy):
        path = _path(tokens)
        root = str(tokens[0])
        report["requested"].append(
            _record(path, "requested", value, _MISSING, "Field supplied to compiler.", None)
        )
        if root in layout_roots:
            outcome = _classify_plan_leaf(path, tokens, value, requested_strategy, resolved_strategy)
        elif root in tree_roots:
            outcome = _classify_tree_leaf(path, tokens, value, resolved_strategy)
        elif root == "score_config":
            outcome = _classify_score_leaf(path, tokens, value, resolved_strategy, score_config)
        else:
            outcome = _classify_top_level_leaf(
                path,
                tokens,
                value,
                resolved_strategy,
                search_config,
                graph_ablation_mode,
            )
        report[outcome["status"]].append(outcome)

    classified_count = sum(len(report[name]) for name in ("applied", "transformed", "rejected", "unknown"))
    applied_without_consumer = [
        item["path"]
        for name in ("applied", "transformed")
        for item in report[name]
        if not item.get("consumer")
    ]
    report["summary"] = {
        "requested_count": len(report["requested"]),
        "classified_count": classified_count,
        "unclassified_count": max(0, len(report["requested"]) - classified_count),
        "applied_without_consumer": applied_without_consumer,
        "counts": {
            name: len(report[name])
            for name in ("applied", "transformed", "rejected", "unknown")
        },
    }
    if legacy_summary:
        for key in ("active_region_count", "rejected_regions", "allowed_nodes", "domain_order"):
            if key in legacy_summary:
                report[key] = legacy_summary[key]
    return report


__all__ = [
    "FieldSpec",
    "PLAN_FIELD_SPECS",
    "REGION_FIELD_SPECS",
    "REPORT_MODE",
    "REPORT_VERSION",
    "SITE_ANCHOR_FIELD_SPECS",
    "TOP_LEVEL_FIELD_SPECS",
    "TREE_NODE_FIELD_SPECS",
    "UNSUPPORTED_REGION_FIELDS",
    "build_strategy_effect_report",
]
