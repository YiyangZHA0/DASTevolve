

# EVOLVE-BLOCK-START
from __future__ import annotations

from typing import Any, Dict

from engine.default_strategy import base_strategy


def propose_strategy() -> Dict[str, Any]:


    strategy = base_strategy()
    strategy.update(
        {
            "preferred_edit_order": [],
            "outer_loop_phase": "explore_ast",
            "mcts_c_puct": 1.6,
            "mcts_max_depth": 8,
            "mcts_progressive_widening_c": 0.25,
            "mcts_progressive_widening_alpha": 0.5,
            "ast_revision_plan": {
                "schema_version": "astevolve.ast_revision_plan.v2",
                "structural_nodes": [
                    {
                        "node_id": "llm_alpha2_seed",
                        "selector": {
                            "schema_version": "astevolve.residue_selector.v1",
                            "chain_id": "P",
                            "spans": [[67, 77]],
                        },
                        "action_profile": "point_resample_3",
                        "intent": (
                            "Explore the complete evidence-eligible alpha2 specificity "
                            "helix while retaining conservative helix chemistry and the "
                            "experimentally grounded switch positions."
                        ),
                        "evidence_refs": [
                            "pdz:P:67", "pdz:P:68", "pdz:P:69", "pdz:P:70", "pdz:P:71",
                            "pdz:P:72", "pdz:P:73", "pdz:P:74", "pdz:P:75", "pdz:P:76",
                        ],
                        "residue_policy": {
                            "favored_residues": ["A", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "Q", "R", "S", "T", "V", "W", "Y"],
                            "disfavored_residues": ["C", "P"],
                            "position_residue_rules": {
                                "70": {
                                    "favored_residues": ["M"],
                                    "disfavored_residues": ["C", "P"],
                                    "policy_weight": 1.4,
                                    "intent": "Revisit the experimental L911M packing switch.",
                                },
                                "71": {
                                    "favored_residues": ["E", "D", "Q"],
                                    "disfavored_residues": ["C", "P"],
                                    "policy_weight": 1.5,
                                    "intent": "Test electrostatic discrimination at K912.",
                                },
                                "74": {
                                    "favored_residues": ["M", "F"],
                                    "disfavored_residues": ["C", "P"],
                                    "policy_weight": 1.6,
                                    "intent": "Test the reported L915M packing hypothesis from the natural parent.",
                                },
                            },
                            "policy_weight": 1.25,
                        },
                    },
                    {
                        "node_id": "llm_beta6_seed",
                        "selector": {
                            "schema_version": "astevolve.residue_selector.v1",
                            "chain_id": "P",
                            "spans": [[79, 81], [84, 85]],
                        },
                        "action_profile": "point_resample_3",
                        "intent": (
                            "Explore the evidence-eligible beta6 terminal-pocket and "
                            "solvent-rim positions while preserving the protected turn "
                            "and buried beta-strand core."
                        ),
                        "evidence_refs": ["pdz:P:79", "pdz:P:80", "pdz:P:84"],
                        "residue_policy": {
                            "favored_residues": ["A", "F", "I", "L", "M", "T", "V", "Y"],
                            "disfavored_residues": ["C", "P", "W"],
                            "position_residue_rules": {
                                "79": {
                                    "favored_residues": ["V", "I"],
                                    "disfavored_residues": ["C", "P", "W"],
                                    "policy_weight": 1.8,
                                    "intent": "Test the reported L920V pocket hypothesis from the natural parent.",
                                }
                            },
                            "policy_weight": 1.3,
                        },
                    },
                    {
                        "node_id": "llm_s05_layered_probe",
                        "selector": {
                            "schema_version": "astevolve.residue_selector.v1",
                            "chain_id": "P",
                            "spans": [[25, 31]],
                        },
                        "action_profile": "point_resample_3",
                        "intent": (
                            "Explore the complete solvent-exposed beta2-beta3 loop as an "
                            "optional negative-design layer without touching the protected "
                            "beta2 ligand-strand anchor."
                        ),
                        "evidence_refs": [
                            "pdz:P:25", "pdz:P:26", "pdz:P:27",
                            "pdz:P:28", "pdz:P:29", "pdz:P:30",
                        ],
                        "residue_policy": {
                            "favored_residues": ["A", "D", "E", "G", "H", "I", "K", "L", "M", "N", "Q", "R", "S", "T", "V"],
                            "disfavored_residues": ["C", "F", "P", "W", "Y"],
                            "position_residue_rules": {
                                "25": {
                                    "favored_residues": ["D", "E", "N", "Q", "K", "R"],
                                    "disfavored_residues": ["C", "F", "P", "W", "Y"],
                                    "policy_weight": 1.6,
                                    "intent": "Test the variant-confounded B-enriched contact as negative design.",
                                },
                                "30": {
                                    "favored_residues": ["G", "K", "N", "Q", "R", "S"],
                                    "disfavored_residues": ["C", "F", "P", "W", "Y"],
                                    "policy_weight": 1.5,
                                    "intent": "Guard the A-contact opportunity with conservative loop chemistry.",
                                },
                            },
                            "policy_weight": 1.3,
                        },
                    },
                    {
                        "node_id": "llm_beta3_interface_probe",
                        "selector": {
                            "schema_version": "astevolve.residue_selector.v1",
                            "chain_id": "P",
                            "spans": [[31, 32], [33, 34], [35, 38]],
                        },
                        "action_profile": "point_resample_3",
                        "intent": (
                            "Probe the evidence-eligible beta3 interface rim, including "
                            "the residue contacting different peptide positions in A and B, "
                            "while excluding the protected buried core positions."
                        ),
                        "evidence_refs": [
                            "pdz:P:31", "pdz:P:33", "pdz:P:35",
                            "pdz:P:36", "pdz:P:37",
                        ],
                        "residue_policy": {
                            "favored_residues": ["A", "D", "E", "G", "H", "I", "K", "L", "M", "N", "Q", "R", "S", "T", "V", "Y"],
                            "disfavored_residues": ["C", "P", "W"],
                            "position_residue_rules": {
                                "35": {
                                    "favored_residues": ["D", "E", "N", "Q", "K", "R"],
                                    "disfavored_residues": ["C", "P", "W"],
                                    "policy_weight": 1.7,
                                    "intent": "Test polar and charge discrimination at the A-P-3/B-P-6 contact.",
                                }
                            },
                            "policy_weight": 1.2,
                        },
                    },
                ],
                "mapping_edges": [
                    {
                        "edge_id": "llm_margin_alpha2_point",
                        "functional_node_id": "A_over_B_selectivity",
                        "structural_node_id": "llm_alpha2_seed",
                        "action_operator": "point",
                        "evidence_refs": [
                            "pdz:P:67", "pdz:P:68", "pdz:P:69", "pdz:P:70", "pdz:P:71",
                            "pdz:P:72", "pdz:P:73", "pdz:P:74", "pdz:P:75", "pdz:P:76",
                        ],
                    },
                    {
                        "edge_id": "llm_margin_beta6_point",
                        "functional_node_id": "A_over_B_selectivity",
                        "structural_node_id": "llm_beta6_seed",
                        "action_operator": "point",
                        "evidence_refs": ["pdz:P:79", "pdz:P:80", "pdz:P:84"],
                    },
                    {
                        "edge_id": "llm_bind_alpha2_resample",
                        "functional_node_id": "bind_A_Caspr4",
                        "structural_node_id": "llm_alpha2_seed",
                        "action_operator": "site_resample",
                        "evidence_refs": [
                            "pdz:P:67", "pdz:P:68", "pdz:P:69", "pdz:P:70", "pdz:P:71",
                            "pdz:P:72", "pdz:P:73", "pdz:P:74", "pdz:P:75", "pdz:P:76",
                        ],
                    },
                    {
                        "edge_id": "llm_bind_beta6_resample",
                        "functional_node_id": "bind_A_Caspr4",
                        "structural_node_id": "llm_beta6_seed",
                        "action_operator": "site_resample",
                        "evidence_refs": ["pdz:P:79", "pdz:P:80", "pdz:P:84"],
                    },
                    {
                        "edge_id": "llm_margin_s05_point",
                        "functional_node_id": "A_over_B_selectivity",
                        "structural_node_id": "llm_s05_layered_probe",
                        "action_operator": "point",
                        "evidence_refs": [
                            "pdz:P:25", "pdz:P:26", "pdz:P:27",
                            "pdz:P:28", "pdz:P:29", "pdz:P:30",
                        ],
                    },
                    {
                        "edge_id": "llm_bind_s05_resample",
                        "functional_node_id": "bind_A_Caspr4",
                        "structural_node_id": "llm_s05_layered_probe",
                        "action_operator": "site_resample",
                        "evidence_refs": [
                            "pdz:P:25", "pdz:P:26", "pdz:P:27",
                            "pdz:P:28", "pdz:P:29", "pdz:P:30",
                        ],
                    },
                    {
                        "edge_id": "llm_margin_beta3_point",
                        "functional_node_id": "A_over_B_selectivity",
                        "structural_node_id": "llm_beta3_interface_probe",
                        "action_operator": "point",
                        "evidence_refs": [
                            "pdz:P:31", "pdz:P:33", "pdz:P:35", "pdz:P:36", "pdz:P:37",
                        ],
                    },
                    {
                        "edge_id": "llm_bind_beta3_resample",
                        "functional_node_id": "bind_A_Caspr4",
                        "structural_node_id": "llm_beta3_interface_probe",
                        "action_operator": "site_resample",
                        "evidence_refs": [
                            "pdz:P:31", "pdz:P:33", "pdz:P:35", "pdz:P:36", "pdz:P:37",
                        ],
                    },
                ],
                "decision_record": {
                    "action": "mixed",
                    "diagnosis": (
                        "The narrow six-position executable scope produced a valid K72Q "
                        "selectivity signal but underused the evidence-eligible interface "
                        "and rim positions available to the global Dual-AST."
                    ),
                    "hypothesis": (
                        "Use four independently selectable nodes and 24 evidence-backed "
                        "positions so MCTS can compare alpha2, beta6, S05-loop, and beta3-rim "
                        "hypotheses under the same three-state protocol."
                    ),
                    "evidence_refs": [
                        "pdz:P:25", "pdz:P:26", "pdz:P:27", "pdz:P:28", "pdz:P:29", "pdz:P:30",
                        "pdz:P:31", "pdz:P:33", "pdz:P:35", "pdz:P:36", "pdz:P:37",
                        "pdz:P:67", "pdz:P:68", "pdz:P:69", "pdz:P:70", "pdz:P:71",
                        "pdz:P:72", "pdz:P:73", "pdz:P:74", "pdz:P:75", "pdz:P:76",
                        "pdz:P:79", "pdz:P:80", "pdz:P:84",
                    ],
                    "expected_effects": [
                        "retain positive-target interface quality",
                        "increase A-over-B raw and gPDE margins",
                        "separate four node-level effects from anchor preservation",
                        "avoid increasing interface clashes",
                    ],
                    "failure_condition": (
                        "Reduce or migrate any expanded node that fails fold/clash gates "
                        "or produces no selectivity gain across its dedicated candidates."
                    ),
                    "confidence": 0.82,
                    "rationale": (
                        "Broaden the searchable surface without broadening any single "
                        "candidate beyond the locked ten-mutation hard cap."
                    ),
                    "rollback_condition": (
                        "Return to the alpha2/beta6 anchors when the expanded S05 or beta3 "
                        "nodes are not beneficial in independently represented finalists."
                    ),
                },
            },
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


OUTER_EVALUATION_TRIALS = 1
_CASE_MANIFEST_OVERRIDE = str(os.environ.get("ASTEVOLVE_CASE_MANIFEST") or "").strip()
CASE_ROOT = (
    Path(_CASE_MANIFEST_OVERRIDE).expanduser().resolve().parent
    if _CASE_MANIFEST_OVERRIDE
    else Path(__file__).resolve().parent
)
if str(CASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CASE_ROOT))


from cases.tiam1_pdz_caspr4_vs_sdc1_selectivity.tiam1_evaluator import (
    register_tiam1_plugin,
)


register_tiam1_plugin()

_PDZ_V2_SEED_SEQUENCES = {
    "P": "KVTQSIHIEKSDTAADTYGFSLSSVEEDGIRRLYVNSVKETGLASKKGLKAGDEILEINNRAADALNSSMLKDFLSQPSLGLLVRTYPEL",
    "A": "ENQKEYFF",
}

_LOCKED_RUNTIME_DEFAULTS: Dict[str, Any] = {
    "resume_template_seqs": _PDZ_V2_SEED_SEQUENCES,
    "iterations": 100,
    "search_method": "mcts",


    "mcts_max_depth": 8,
    "mcts_progressive_widening_c": 0.25,
    "mcts_progressive_widening_alpha": 0.5,
    "mcts_iteration_unit": "evaluated_unique_candidates",
    "mcts_candidate_budget_max_round_multiplier": 12,
    "mcts_candidate_budget_fail_on_underfill": True,
    "mcts_node_sweep_enabled": False,
    "mcts_tree_quality_required": True,
    "mcts_tree_min_root_children": 2,
    "mcts_tree_min_branching_nodes": 2,
    "mcts_tree_min_leaves": 4,
    "mcts_tree_min_max_depth": 3,
    "mutation_ops": {"point": 0.65, "site_resample": 0.35},
    "max_total_mutations": 10,
    "exploit_max_mutations": 6,
    "explore_max_mutations": 10,
    "repair_max_mutations": 3,


    "proposal_tier_mode": "mixed",
    "proposal_exploit_frac": 0.55,
    "proposal_explore_frac": 0.35,
    "proposal_repair_frac": 0.10,
    "search_schedule": {
        "enabled": True,
        "hard_max_total_mutations": 10,
        "phase_policy": (
            "The outer LLM selects a discrete phase and may replace the complete "
            "8--24-position editable Dual-AST across the 90-residue construct. "
            "Each phase has a locked tier mixture and a ten-mutation hard cap."
        ),
        "phases": {
            "explore_ast": {
                "proposal_exploit_frac": 0.55,
                "proposal_explore_frac": 0.35,
                "proposal_repair_frac": 0.10,
            },
            "refine_ast": {
                "proposal_exploit_frac": 0.66,
                "proposal_explore_frac": 0.22,
                "proposal_repair_frac": 0.12,
            },
            "converge": {
                "proposal_exploit_frac": 0.78,
                "proposal_explore_frac": 0.08,
                "proposal_repair_frac": 0.14,
            },
        },
    },
    "fast_filter_enabled": True,
    "progen_weight": 0.05,
    "progen_chains": ["P"],
    "sequence_prior_model": "progen",
    "inner_structure_enabled": True,
    "inner_structure_model": "esmfold",
    "inner_structure_model_name": "facebook/esmfold_v1",
    "inner_structure_weight": 1.0,
    "inner_structure_fail_closed": True,
    "inner_structure_hard_gate": False,
    "promote_inline_winner_structure_evidence": True,
    "inner_esmfold2_enabled": False,
    "mcts_fidelity_upgrade_enabled": True,
    "mcts_fidelity_upgrade_provider": "protenix",
    "mcts_fidelity_upgrade_interval": 10,
    "mcts_fidelity_upgrade_candidates": 2,
    "mcts_fidelity_upgrade_final_candidates": 3,
    "mcts_fidelity_upgrade_required": True,
    "node_optimizer_enabled": True,
    "node_optimizer_candidate_count": 3,
    "node_optimizer_beam_width": 16,
    "node_optimizer_top_k_per_position": 4,
    "node_optimizer_temperature": 0.8,
    "node_optimizer_diversity_weight": 0.15,
    "node_optimizer_mutation_penalty": 0.20,
    "node_optimizer_prior_model": "masked_lm",
    "semantic_required_nodes": [
        "S05_beta2_beta3_loop",
        "S06_beta3",
        "S14_alpha2_specificity_helix",
        "S16_beta6",
    ],


    "semantic_anchor_nodes": [
        "S14_alpha2_specificity_helix",
        "S16_beta6",
    ],
    "semantic_required_node_min_visits": 1,
    "semantic_required_node_min_mutations": 1,
    "semantic_coverage_mode": "soft",
    "semantic_required_node_round_robin": True,
    "chai1_enabled": True,
    "chai1_top_frac": 0.02,


    "chai1_min_candidates": 3,
    "chai1_max_candidates": 3,
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
    "structure_shortlist_policy": "formal_layered_novel",
    "structure_screen_single_node_diagnostic_quota": 0,
    "structure_selection_objective": "outer_aligned",
    "structure_allow_low_fidelity_fallback": False,
    "structure_screen_enabled": False,
    "structure_rerank_enabled": False,
    "multistate_objectives_enabled": True,


    "multistate_objective_weight": 0.0,
    "mcts_save_tree": True,
    "mcts_save_variants": True,
    "mcts_artifact_mode": "normalized",
    "score_config": {


        "weight_fast": 0.0,
        "weight_plddt": 0.0,
        "weight_iptm": 0.0,
        "weight_ptm": 0.0,
        "weight_ranking_score": 0.0,
        "weight_interface_plddt": 0.0,
        "weight_node_plddt_min": 0.0,
        "weight_clash": 0.0,
        "weight_multistate": 0.0,
        "weight_evaluator": 1.0,
        "inner_evaluator_loss_weight": 1.0,
        "inner_hard_gate_fail_penalty": 1000.0,
        "plddt_scale": 100.0,
        "clash_scale": 10.0,
        "fast_loss_nonneg": True,


        "evaluator_weights": {
            "eval_global_plddt": 0.0,
            "eval_ptm": 0.0,
            "eval_iptm": 0.0,
            "eval_interface_plddt": 0.0,
            "eval_node_floor": 0.0,
            "eval_state_confidence": 0.0,
            "eval_target_confidence_floor": 0.0,
            "eval_clash": 0.0,
            "eval_chain_continuity": 0.0,
            "eval_compactness": 0.0,
            "eval_preserved_node_confidence": 0.0,
            "eval_preserved_sequence": 0.0,
            "eval_scaffold_rmsd": 0.0,
            "eval_preserved_node_rmsd": 0.0,
            "eval_contacts": 0.0,
            "eval_residue_pairs": 0.0,
            "eval_hbond_proxy": 0.0,
            "eval_salt_proxy": 0.0,
            "eval_hydrophobic_proxy": 0.0,
            "eval_hbond_geometry": 0.0,
            "eval_salt_geometry": 0.0,
            "eval_hydrophobic_geometry": 0.0,
            "eval_mutation_scope": 0.0,
            "eval_contract_response": 0.0,
            "eval_primary_engagement": 0.0,
            "eval_multistate": 0.0,
        },
        "evaluator_plugins": ["tiam1_selectivity"],
        "plugin_config": {
            "tiam1_selectivity": {
                "iptm_margin_target": 0.08,
                "iptm_margin_sigma": 0.04,
                "gpde_margin_target": 0.04,
                "gpde_margin_sigma": 0.02,
                "interface_q_margin_target": 0.03,
                "interface_q_margin_sigma": 0.02,
                "interface_q_contact_target": 400.0,
                "interface_q_residue_pair_target": 40.0,
                "interface_clash_budget": 3.0,
                "baseline_apo_plddt": 87.3060073852539,
                "baseline_positive_plddt": 92.97100067138672,
                "baseline_positive_iptm": 0.9367565512657166,
                "baseline_positive_interface_q": 0.9429525830682546,
                "apo_plddt_noninferiority_band": -5.0,
                "positive_plddt_noninferiority_band": -3.0,
                "positive_iptm_noninferiority_band": -0.03,
                "positive_interface_q_noninferiority_band": -0.05,
                "final_require_b_clash_free": False,
                "evaluator_weights": {
                    "eval_tiam1_state_evidence": 0.0,
                    "eval_tiam1_apo_fold": 0.0,
                    "eval_tiam1_A_complex": 0.0,
                    "eval_tiam1_A_anchor": 0.0,
                    "eval_tiam1_A_iptm_noninferiority": 0.0,
                    "eval_tiam1_A_interface_q_noninferiority": 0.0,
                    "eval_tiam1_A_clash_free": 0.0,
                    "eval_tiam1_B_clash_free_final": 0.0,
                    "eval_tiam1_A_interface": 0.0,
                    "eval_tiam1_B_weakening": 0.0,
                    "eval_tiam1_A_over_B_margin": 0.0,
                    "eval_tiam1_A_over_B_iptm_margin": 0.0,
                    "eval_tiam1_A_over_B_gpde_margin": 0.0,
                    "eval_tiam1_interface_clash_guard": 0.0,
                    "eval_tiam1_mutation_budget": 0.0,
                    "eval_tiam1_no_proline": 0.0,
                    "eval_tiam1_v2_iptm_margin": 0.50,
                    "eval_tiam1_v2_gpde_margin": 0.20,
                    "eval_tiam1_v2_interface_q_margin": 0.20,
                    "eval_tiam1_v2_positive_interface_q": 0.10,
                }
            }
        },
    },
}


def _effective_locked_runtime_defaults() -> Dict[str, Any]:


    defaults = deepcopy(_LOCKED_RUNTIME_DEFAULTS)
    budget = int(os.environ.get("ASTEVOLVE_INNER_ITERATIONS", defaults["iterations"]))
    if budget < 4:
        raise ValueError("TIAM1 true MCTS requires at least four evaluated candidates")
    defaults["iterations"] = budget
    if budget < 30:
        defaults.update(
            {
                "mcts_tree_min_root_children": 2,
                "mcts_tree_min_branching_nodes": 1,
                "mcts_tree_min_leaves": 4,
                "mcts_tree_min_max_depth": 2,
            }
        )
    return defaults


def runtime_strategy() -> Dict[str, Any]:


    return merge_locked_runtime_defaults(
        propose_strategy(),
        _effective_locked_runtime_defaults(),
        base_strategy,
    )


def _case_paths() -> Dict[str, str]:
    if os.environ.get("ASTEVOLVE_CASE_MANIFEST"):
        return current_case_kwargs()
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
        "mask_true_counts": {
            chain_id: int(sum(mask)) for chain_id, mask in prepared.masks.items()
        },
        "fixed_residue_counts": {
            chain_id: len(residues)
            for chain_id, residues in prepared.fixed_residues.items()
        },
        "node_policy_names": sorted(
            prepared.resolved_strategy.get("node_edit_policies", {})
        ),
        "executable_structural_nodes": [
            node.to_dict()
            for node in prepared.executable_node_plan.structural_nodes
        ],
        "measurement_intents": [
            intent.to_dict()
            for intent in prepared.executable_node_plan.measurement_intents
        ],
        "ast_revision_report": prepared.design_state.get(
            "_ast_revision_report", {}
        ),
        "mutation_scope_contract": prepared.score_config.get(
            "mutation_scope_contract", {}
        ),
        "evaluator_plugin_resolution": prepared.design_state.get(
            "_evaluator_plugin_resolution", {}
        ),
        "score_config": prepared.score_config,
        "strategy_schema_report": prepared.resolved_strategy.get(
            "strategy_schema_report", {}
        ),
    }


def run_search(seed: Optional[int] = None) -> Dict[str, Any]:


    strategy = deepcopy(runtime_strategy())
    from engine.runtime_profile import build_sa_config

    runtime = build_sa_config(strategy)
    expected = {
        "search_method": "mcts",
        "mcts_iteration_unit": "evaluated_unique_candidates",
        "mcts_max_depth": 8,
        "mcts_node_sweep_enabled": False,
        "node_optimizer_enabled": True,
        "node_optimizer_candidate_count": 3,
        "mcts_progressive_widening_c": 0.25,
        "mcts_progressive_widening_alpha": 0.5,
        "mcts_tree_quality_required": True,
        "inner_structure_enabled": True,
        "inner_structure_model": "esmfold",
        "inner_esmfold2_enabled": False,
        "mcts_fidelity_upgrade_enabled": True,
        "mcts_fidelity_upgrade_provider": "protenix",
        "mcts_fidelity_upgrade_required": True,
        "chai1_enabled": True,
    }
    mismatches = {
        key: {"expected": value, "observed": runtime.get(key)}
        for key, value in expected.items()
        if runtime.get(key) != value
    }
    if float(runtime.get("progen_weight", 0.0) or 0.0) <= 0.0:
        mismatches["progen_weight"] = {
            "expected": "positive",
            "observed": runtime.get("progen_weight"),
        }
    if mismatches:
        raise RuntimeError(f"tiam1_true_mcts_runtime_mismatch:{mismatches}")
    evaluation_id = f"{Path(__file__).stem}__seed_{int(seed or 0)}"
    strategy["mcts_output_dir"] = str(
        artifact_path(
            "tiam1_pdz_caspr4_vs_sdc1_selectivity",
            "inner",
            evaluation_id,
        )
    )
    return run_design_search(
        strategy,
        seed=seed,
        memory_commit_mode="deferred",
        **_case_paths(),
    )


__all__ = ["OUTER_EVALUATION_TRIALS", "preview_case", "propose_strategy", "run_search", "runtime_strategy"]
