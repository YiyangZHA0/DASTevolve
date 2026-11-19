"""Blind, executable Dynamic Dual-AST baseline for TtgR."""

# EVOLVE-BLOCK-START
from __future__ import annotations

from typing import Any, Dict

from engine.default_strategy import base_strategy


def propose_strategy() -> Dict[str, Any]:
    """Return the full desired editable graph for this generation."""

    strategy = base_strategy()
    strategy.update(
        {
            "preferred_edit_order": [],
            "outer_loop_phase": "explore_ast",
            "mcts_c_puct": 1.5,
            "mcts_max_depth": 3,
            "mcts_progressive_widening_c": 1.8,
            "mcts_progressive_widening_alpha": 0.5,
            "ast_revision_plan": {
                "schema_version": "astevolve.ast_revision_plan.v2",
                "structural_nodes": [
                    {
                        "node_id": "llm_hinge_seed",
                        "selector": {"schema_version": "astevolve.residue_selector.v1", "chain_id": "T", "spans": [[66, 67], [69, 70], [77, 78]]},
                        "action_profile": "point_resample",
                        "intent": "Test a compact hinge/pocket scope while preserving the DNA-binding domain.",
                        "evidence_refs": ["ttgr:T:66", "ttgr:T:69", "ttgr:T:77"],
                        "residue_policy": {"favored_residues": [], "disfavored_residues": ["C", "P"], "position_residue_rules": {}, "policy_weight": 1.0}
                    },
                    {
                        "node_id": "llm_pocket_seed",
                        "selector": {"schema_version": "astevolve.residue_selector.v1", "chain_id": "T", "spans": [[88, 89], [91, 92], [95, 96]]},
                        "action_profile": "point_resample",
                        "intent": "Explore target-pocket chemistry without assuming a known successful sequence.",
                        "evidence_refs": ["ttgr:T:88", "ttgr:T:91", "ttgr:T:95"],
                        "residue_policy": {"favored_residues": [], "disfavored_residues": ["C", "P"], "position_residue_rules": {}, "policy_weight": 1.0}
                    },
                    {
                        "node_id": "llm_coupling_seed",
                        "selector": {"schema_version": "astevolve.residue_selector.v1", "chain_id": "T", "spans": [[112, 114]]},
                        "action_profile": "point_resample",
                        "intent": "Probe alpha6 pocket-to-core coupling with a small, reversible scope.",
                        "evidence_refs": ["ttgr:T:112", "ttgr:T:113"],
                        "residue_policy": {"favored_residues": [], "disfavored_residues": ["C", "P"], "position_residue_rules": {}, "policy_weight": 1.0}
                    }
                ],
                "mapping_edges": [
                    {"edge_id": "llm_target_hinge", "functional_node_id": "T_F01_naltrexone_target_on_proxy", "structural_node_id": "llm_hinge_seed", "action_operator": "point", "evidence_refs": ["ttgr:T:66", "ttgr:T:69", "ttgr:T:77"]},
                    {"edge_id": "llm_target_pocket", "functional_node_id": "T_F01_naltrexone_target_on_proxy", "structural_node_id": "llm_pocket_seed", "action_operator": "site_resample", "evidence_refs": ["ttgr:T:88", "ttgr:T:91", "ttgr:T:95"]},
                    {"edge_id": "llm_counter_pocket", "functional_node_id": "T_F02_counter_relative_margin_proxy", "structural_node_id": "llm_pocket_seed", "action_operator": "point", "evidence_refs": ["ttgr:T:88", "ttgr:T:91", "ttgr:T:95"]},
                    {"edge_id": "llm_coupling_hinge", "functional_node_id": "T_F05_ligand_to_DBD_transition_proxy", "structural_node_id": "llm_hinge_seed", "action_operator": "site_resample", "evidence_refs": ["ttgr:T:66", "ttgr:T:69", "ttgr:T:77"]},
                    {"edge_id": "llm_coupling_alpha6", "functional_node_id": "T_F05_ligand_to_DBD_transition_proxy", "structural_node_id": "llm_coupling_seed", "action_operator": "point", "evidence_refs": ["ttgr:T:112", "ttgr:T:113"]}
                ],
                "decision_record": {
                    "action": "keep",
                    "diagnosis": "No candidate-bound GPU evidence exists yet; begin from a balanced, non-oracle seed scope.",
                    "hypothesis": "Distributed hinge, pocket, and alpha6 probes will reveal whether target contacts and controlled apo-to-holo geometry can be improved without fold loss.",
                    "evidence_refs": ["ttgr:T:66", "ttgr:T:69", "ttgr:T:77", "ttgr:T:88", "ttgr:T:91", "ttgr:T:95", "ttgr:T:112", "ttgr:T:113"],
                    "expected_effects": ["finite target-pocket proxy", "finite transition proxy", "preserved fold and dimer proxies"],
                    "failure_condition": "Rollback or migrate a node when its edits violate sequence contracts or reduce worst-state structural quality.",
                    "confidence": 0.35,
                    "rationale": "Initial blind smoke hypothesis, not a biological success claim.",
                    "rollback_condition": "Return to the WT sequence and the smallest catalog-supported selector if required state evidence is missing."
                }
            }
        }
    )
    return strategy


# EVOLVE-BLOCK-END

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

from astevolve.runtime.case_context import current_case_kwargs
from astevolve.runtime.case_program import merge_locked_runtime_defaults
from astevolve.runtime.paths import artifact_path
from engine.case_builder import prepare_case_inputs, run_design_search


CASE_ROOT = Path(__file__).resolve().parent
if str(CASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CASE_ROOT))

from ttgr_evaluator import register_ttgr_plugin


register_ttgr_plugin()
OUTER_EVALUATION_TRIALS = 1
_WT = "MVRRTKEEAQETRAQIIEAAERAFYKRGVARTTLADIAELAGVTRGAIYWHFNNKAELVQALLDSLHETHDHLARASESEDEVDPLGCMRKLLLQVFNELVLDARTRRINEILHHKCEFTDDMCEIRQQRQSAVLDCHKGITLALANAVRRGQLPGELDAERAAVAMFAYVDGLIRRWLLLPDSVDLLGDVEKWVDTGLDMLRLSPALRK"
_AA = set("ACDEFGHIKLMNPQRSTVWY")
_ALLOWED = {
    66: set("EHKT"), 69: set("FHL"), 73: set("AILV"), 77: set("EQ"),
    88: set("FILMV"), 91: set("AILMTV"), 92: set("ILW"), 95: set("FILV"),
    109: set("FMN"), 112: set("AFL"), 113: set("FHY"),
}


def _positions(selector: Dict[str, Any]) -> list[int]:
    return [position for start, end in selector.get("spans", []) for position in range(int(start), int(end))]


def _apply_case_alphabet(strategy: Dict[str, Any]) -> Dict[str, Any]:
    """Clip LLM soft AA preferences to the case-owned hard alphabet."""

    updated = deepcopy(strategy)
    plan = updated.get("ast_revision_plan", {})
    for node in plan.get("structural_nodes", []) if isinstance(plan, dict) else []:
        policy = node.get("residue_policy", {})
        raw_rules = policy.get("position_residue_rules", {})
        raw_rules = raw_rules if isinstance(raw_rules, dict) else {}
        clipped_rules: Dict[str, Any] = {}
        for position in _positions(node.get("selector", {})):
            allowed = _ALLOWED.get(position)
            if not allowed:
                continue
            incoming = raw_rules.get(str(position), {})
            incoming = incoming if isinstance(incoming, dict) else {}
            favored = sorted(set(incoming.get("favored_residues", [])) & allowed)
            disfavored = sorted((_AA - allowed) | (set(incoming.get("disfavored_residues", [])) & allowed))
            clipped_rules[str(position)] = {
                "favored_residues": favored,
                "disfavored_residues": disfavored,
                "policy_weight": 10.0,
                "intent": str(incoming.get("intent") or "Rank residues only within the case-owned site alphabet."),
            }
        policy["favored_residues"] = sorted(set(policy.get("favored_residues", [])) & set().union(*(_ALLOWED[p] for p in _positions(node.get("selector", {})) if p in _ALLOWED))) if clipped_rules else []
        policy["disfavored_residues"] = []
        policy["position_residue_rules"] = clipped_rules
        policy["policy_weight"] = 10.0
    return updated


_LOCKED_RUNTIME_DEFAULTS: Dict[str, Any] = {
    "resume_template_seqs": {"T": _WT},
    "residue_mutation_contract": {
        "T": {position: sorted(residues) for position, residues in _ALLOWED.items()}
    },
    "iterations": int(os.environ.get("ASTEVOLVE_INNER_ITERATIONS", "3")),
    "search_method": "mcts",
    "mutation_ops": {"point": 0.70, "site_resample": 0.30},
    "max_total_mutations": 4,
    "search_schedule": {"enabled": True, "hard_max_total_mutations": 4, "phase_policy": "All phases retain the L1 four-substitution hard cap.", "phases": {"explore_ast": {"proposal_exploit_frac": 0.55, "proposal_explore_frac": 0.35, "proposal_repair_frac": 0.10}, "refine_ast": {"proposal_exploit_frac": 0.65, "proposal_explore_frac": 0.25, "proposal_repair_frac": 0.10}, "converge": {"proposal_exploit_frac": 0.80, "proposal_explore_frac": 0.10, "proposal_repair_frac": 0.10}}},
    "exploit_max_mutations": 2,
    "explore_max_mutations": 4,
    "repair_max_mutations": 1,
    "proposal_tier_mode": "mixed",
    "proposal_exploit_frac": 0.55,
    "proposal_explore_frac": 0.35,
    "proposal_repair_frac": 0.10,
    "fast_filter_enabled": True,
    "progen_weight": 0.0,
    "progen_chains": ["T"],
    "node_optimizer_enabled": True,
    "node_optimizer_candidate_count": 4,
    "node_optimizer_beam_width": 8,
    "node_optimizer_top_k_per_position": 3,
    "node_optimizer_prior_model": "heuristic",
    "semantic_anchor_nodes": [],
    "semantic_required_node_min_visits": 0,
    "semantic_required_node_min_mutations": 0,
    "semantic_coverage_mode": "soft",
    "chai1_enabled": True,
    "chai1_top_frac": 0.10,
    "chai1_min_candidates": 1,
    "chai1_max_candidates": 1,
    "structure_model": "protenix",
    "structure_model_name": "protenix_mini_esm_v0.5.0",
    "protenix_model_name": "protenix_mini_esm_v0.5.0",
    "protenix_conda_env": "protenix",
    "protenix_seed": 101,
    "protenix_complex_use_msa": False,
    "protenix_complex_cycle": 1,
    "protenix_complex_step": 20,
    "protenix_complex_sample": 1,
    "protenix_complex_use_default_params": False,
    "protenix_complex_timeout": 1800,
    "structure_batch_size": 0,
    "structure_parallel_workers": 1,
    "structure_allow_low_fidelity_fallback": False,
    "structure_screen_enabled": False,
    "structure_rerank_enabled": False,
    "multistate_objectives_enabled": True,
    "multistate_objective_weight": 0.0,
    "mcts_save_tree": True,
    "mcts_save_variants": True,
    "score_config": {
        "weight_fast": 0.0,
        "weight_plddt": 0.0,
        "weight_iptm": 0.0,
        "weight_ptm": 0.0,
        "weight_interface_plddt": 0.0,
        "weight_node_plddt_min": 0.0,
        "weight_clash": 0.0,
        "weight_multistate": 0.0,
        "weight_evaluator": 1.0,
        "inner_evaluator_loss_weight": 1.0,
        "inner_hard_gate_fail_penalty": 1000.0,
        "evaluator_plugins": ["ttgr_blind"],
        "plugin_config": {"ttgr_blind": {"evaluator_weights": {
            "eval_ttgr_target_pocket_proxy": 0.35,
            "eval_ttgr_counter_relative_margin_proxy": 0.0,
            "eval_ttgr_allosteric_transition_proxy": 0.0,
            "eval_ttgr_fold_guardrail": 0.20,
            "eval_ttgr_dimer_guardrail": 0.15,
            "eval_ttgr_mutation_burden": 0.10,
            "eval_ttgr_sequence_contract": 0.0,
            "eval_ttgr_state_evidence": 0.0
        }}}
    }
}


def runtime_strategy() -> Dict[str, Any]:
    merged = merge_locked_runtime_defaults(propose_strategy(), _LOCKED_RUNTIME_DEFAULTS, base_strategy)
    return _apply_case_alphabet(merged)


def _case_paths() -> Dict[str, str]:
    if os.environ.get("ASTEVOLVE_CASE_MANIFEST"):
        return current_case_kwargs()
    return {"design_state_path": str(CASE_ROOT / "design_state.json"), "memory_path": str(CASE_ROOT / "memory.yaml")}


def preview_case() -> Dict[str, Any]:
    prepared = prepare_case_inputs(runtime_strategy(), **_case_paths())
    compiled = prepared.blueprint.compile()
    return {
        "task_name": prepared.design_state["task_name"],
        "chain_lengths": compiled["chain_lengths"],
        "mask_true_counts": {chain: int(sum(mask)) for chain, mask in prepared.masks.items()},
        "executable_structural_nodes": [node.to_dict() for node in prepared.executable_node_plan.structural_nodes],
        "measurement_intents": [intent.to_dict() for intent in prepared.executable_node_plan.measurement_intents],
        "ast_revision_report": prepared.design_state.get("_ast_revision_report", {}),
        "mutation_scope_contract": prepared.score_config.get("mutation_scope_contract", {}),
        "evaluator_plugin_resolution": prepared.design_state.get("_evaluator_plugin_resolution", {}),
        "blind_runtime_degradations": {
            "executable_functional_nodes": ["T_F01_naltrexone_target_on_proxy", "T_F02_counter_relative_margin_proxy", "T_F05_ligand_to_DBD_transition_proxy"],
            "semantic_guardrails_are_case_owned": True,
            "counter_states_active": False,
            "generator_hard_alphabet_enforcement": True,
            "evaluator_hard_alphabet_enforcement": True
        }
    }


def run_search(seed: Optional[int] = None) -> Dict[str, Any]:
    strategy = deepcopy(runtime_strategy())
    strategy["mcts_output_dir"] = str(artifact_path("ttgr_naltrexone_l1_blind", "inner", f"seed_{int(seed or 0)}"))
    return run_design_search(strategy, seed=seed, memory_commit_mode="deferred", **_case_paths())


__all__ = ["OUTER_EVALUATION_TRIALS", "preview_case", "propose_strategy", "run_search", "runtime_strategy"]
