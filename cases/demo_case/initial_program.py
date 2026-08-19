

# EVOLVE-BLOCK-START
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from engine.default_strategy import base_strategy


def propose_strategy() -> Dict[str, Any]:


    strategy = base_strategy()
    strategy["layout_plan"] = {
        "binder_domain_order": ["binder_domain"],
        "design_regions": [
            {
                "name": "target_edit_region",
                "bind_to": ["edit_loop"],
                "favored_residues": ["Y", "W", "N", "Q", "S", "T"],
                "disfavored_residues": ["C"],
                "position_residue_rules": {
                    "1": {"favored": ["Y"]},
                    "4": {"favored": ["W"]},
                },
                "operator_phase": "explore",
                "mutation_rate": 0.10,
                "mutation_ops": {"point": 0.70, "site_resample": 0.30},
            },
            {
                "name": "support_repair_region",
                "bind_to": ["support_helix"],
                "favored_residues": ["A", "E", "K"],
                "operator_phase": "repair",
                "mutation_rate": 0.03,
                "mutation_ops": {"point": 1.0},
            },
        ],
    }
    return strategy


# EVOLVE-BLOCK-END

from engine.case_builder import prepare_case_inputs, run_design_search
from astevolve.runtime.case_context import current_case_kwargs
from astevolve.runtime.case_program import merge_locked_runtime_defaults
from astevolve.runtime.paths import artifact_path


CASE_ROOT = Path(__file__).resolve().parent
_LOCKED_RUNTIME_DEFAULTS: Dict[str, Any] = {


    "iterations": 10,
    "search_method": "mcts",
    "mcts_iteration_unit": "evaluated_unique_candidates",
    "mcts_max_depth": 4,
    "mcts_progressive_widening_c": 0.5,
    "mcts_progressive_widening_alpha": 0.5,
    "mcts_candidate_budget_max_round_multiplier": 12,
    "mcts_candidate_budget_fail_on_underfill": True,
    "mcts_tree_quality_required": True,
    "mcts_tree_min_root_children": 2,
    "mcts_tree_min_branching_nodes": 1,
    "mcts_tree_min_leaves": 3,
    "mcts_tree_min_max_depth": 2,

    "progen_weight": 0.05,
    "progen_chains": ["B"],
    "sequence_prior_model": "progen",

    "inner_structure_enabled": True,
    "inner_structure_model": "esmfold",
    "inner_structure_model_name": os.environ.get(
        "ASTEVOLVE_ESMFOLD_MODEL", "facebook/esmfold_v1"
    ),
    "inner_structure_weight": 1.0,
    "inner_structure_fail_closed": True,
    "inner_structure_hard_gate": False,
    "promote_inline_winner_structure_evidence": True,
    "inner_esmfold2_enabled": False,


    "mcts_fidelity_upgrade_enabled": True,
    "mcts_fidelity_upgrade_provider": "protenix",
    "mcts_fidelity_upgrade_interval": 10,
    "mcts_fidelity_upgrade_candidates": 1,
    "mcts_fidelity_upgrade_final_candidates": 1,
    "mcts_fidelity_upgrade_required": True,
    "chai1_enabled": True,
    "structure_model": "protenix",
    "structure_model_name": os.environ.get(
        "ASTEVOLVE_PROTENIX_MODEL_NAME", "protenix_mini_esm_v0.5.0"
    ),
    "protenix_model_name": os.environ.get(
        "ASTEVOLVE_PROTENIX_MODEL_NAME", "protenix_mini_esm_v0.5.0"
    ),
    "protenix_complex_use_msa": False,
    "structure_screen_enabled": False,
    "structure_rerank_enabled": False,
    "multistate_objectives_enabled": False,
    "mcts_output_dir": str(artifact_path("demo_case", "inner")),
    "mcts_save_tree": True,
    "mcts_save_variants": True,
    "semantic_required_nodes": [],
    "score_config": {
        "weight_fast": 0.5,
        "weight_plddt": 1.0,
        "weight_iptm": 1.0,
        "weight_ptm": 0.0,
        "weight_interface_plddt": 1.0,
        "weight_node_plddt_min": 0.0,
        "weight_clash": 0.5,
        "weight_multistate": 0.0,
        "evaluator_backends": {},
        "evaluator_plugins": ["demo_gpu"],
        "plugin_config": {
            "demo_gpu": {
                "evaluator_weights": {
                    "eval_synthetic_goal_match": 1.0,
                    "eval_fixed_residue_integrity": 1.0,
                }
            }
        },
    },
}


def runtime_strategy() -> Dict[str, Any]:


    return merge_locked_runtime_defaults(
        propose_strategy(),
        _LOCKED_RUNTIME_DEFAULTS,
        base_strategy,
    )


def _case_paths() -> Dict[str, str]:
    case_id = os.environ.get("ASTEVOLVE_CASE_ID")
    has_case_root = bool(
        os.environ.get("ASTEVOLVE_CASE_ROOT")
        or os.environ.get("ASTEVOLVE_CASES_ROOT")
    )
    if os.environ.get("ASTEVOLVE_CASE_MANIFEST") or (case_id and has_case_root):
        return current_case_kwargs(case_id or "demo_case")
    return {
        "design_state_path": str(CASE_ROOT / "design_state.json"),
        "memory_path": str(CASE_ROOT / "memory.yaml"),
    }


def preview_case() -> Dict[str, Any]:


    prepared = prepare_case_inputs(runtime_strategy(), **_case_paths())
    compiled = prepared.blueprint.compile()
    return {
        "task_name": prepared.design_state["task_name"],
        "chain_order": compiled["chain_order"],
        "chain_lengths": compiled["chain_lengths"],
        "segments": [
            {
                "chain_id": segment.chain_id,
                "name": segment.name,
                "kind": segment.kind,
                "spans": segment.spans,
            }
            for segment in compiled["segments"]
        ],
        "mask_true_counts": {
            chain_id: int(sum(mask))
            for chain_id, mask in prepared.masks.items()
        },
        "fixed_residue_counts": {
            chain_id: len(residues)
            for chain_id, residues in prepared.fixed_residues.items()
        },
        "node_policy_names": sorted(
            prepared.resolved_strategy.get("node_edit_policies", {})
        ),
        "executable_structural_node_ids": [
            node.node_id for node in prepared.executable_node_plan.structural_nodes
        ],
        "measurement_intent_ids": [
            intent.functional_node_id
            for intent in prepared.executable_node_plan.measurement_intents
        ],
        "strategy_schema_report": prepared.resolved_strategy.get(
            "strategy_schema_report", {}
        ),
    }


def run_search(seed: Optional[int] = None) -> Dict[str, Any]:


    return run_design_search(
        runtime_strategy(),
        seed=seed,
        memory_commit_mode="deferred",
        **_case_paths(),
    )
