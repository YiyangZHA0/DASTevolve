

# EVOLVE-BLOCK-START
from __future__ import annotations

from typing import Any, Dict

from engine.default_strategy import base_strategy


_FAVORED_AMINO_ACIDS = [
    "A",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]


def _residue_policy(
    weight: float,
    position_rules: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    return {
        "favored_residues": list(_FAVORED_AMINO_ACIDS),
        "disfavored_residues": ["C"],
        "position_residue_rules": dict(position_rules or {}),
        "policy_weight": weight,
    }


def propose_strategy() -> Dict[str, Any]:


    l0_refs = ["cab:A:101", "cab:A:104", "cab:A:105"]
    l1_positions = [
        98,
        99,
        100,
        102,
        103,
        106,
        107,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
    ]
    l1_refs = [f"cab:A:{position}" for position in l1_positions]
    cdr3_refs = sorted(set(l0_refs + l1_refs))
    l2_cdr1_refs = ["cab:A:28", "cab:A:29", "cab:A:30"]
    l2_cdr2_refs = ["cab:A:52", "cab:A:56"]
    l2_refs = l2_cdr1_refs + l2_cdr2_refs
    cdr1_envelope_refs = [f"cab:A:{position}" for position in range(25, 32)]
    cdr2_envelope_refs = [f"cab:A:{position}" for position in range(52, 57)]
    search_refs = sorted(
        set(cdr3_refs + cdr1_envelope_refs + cdr2_envelope_refs)
    )
    strategy = base_strategy()
    strategy.update(
        {
            "ast_revision_plan": {
                "schema_version": "astevolve.ast_revision_plan.v2",
                "structural_nodes": [
                    {
                        "node_id": "llm_cdr3_l0_l1_stage",
                        "selector": {
                            "schema_version": "astevolve.residue_selector.v1",
                            "chain_id": "A",
                            "spans": [[98, 108], [109, 122]],
                        },
                        "action_profile": "cdr3_high_entropy",
                        "intent": "Keep the complete 23-position CDR3 window editable until the diversity gate passes.",
                        "evidence_refs": [
                            "cab:A:98", "cab:A:99", "cab:A:100", "cab:A:101",
                            "cab:A:102", "cab:A:103", "cab:A:104", "cab:A:105",
                            "cab:A:106", "cab:A:107", "cab:A:109", "cab:A:110",
                            "cab:A:111", "cab:A:112", "cab:A:113", "cab:A:114",
                            "cab:A:115", "cab:A:116", "cab:A:117", "cab:A:118",
                            "cab:A:119", "cab:A:120", "cab:A:121",
                        ],
                        "residue_policy": {
                            "favored_residues": [
                                "A", "D", "E", "F", "G", "H", "I", "K",
                                "L", "M", "N", "P", "Q", "R", "S", "T",
                                "V", "W", "Y",
                            ],
                            "disfavored_residues": ["C"],
                            "position_residue_rules": {},
                            "policy_weight": 1.0,
                        },
                    },
                    {
                        "node_id": "llm_l2_cdr1_stage",
                        "selector": {
                            "schema_version": "astevolve.residue_selector.v1",
                            "chain_id": "A",
                            "spans": [[25, 32]],
                        },
                        "action_profile": "cdr_short_high_entropy",
                        "intent": "Optimize the full reviewed CDR1 window while retaining typed recovery measurements.",
                        "evidence_refs": [
                            "cab:A:25", "cab:A:26", "cab:A:27", "cab:A:28",
                            "cab:A:29", "cab:A:30", "cab:A:31",
                        ],
                        "residue_policy": {
                            "favored_residues": [
                                "A", "D", "E", "F", "G", "H", "I", "K",
                                "L", "M", "N", "P", "Q", "R", "S", "T",
                                "V", "W", "Y",
                            ],
                            "disfavored_residues": ["C"],
                            "position_residue_rules": {},
                            "policy_weight": 1.0,
                        },
                    },
                    {
                        "node_id": "llm_l2_cdr2_stage",
                        "selector": {
                            "schema_version": "astevolve.residue_selector.v1",
                            "chain_id": "A",
                            "spans": [[52, 57]],
                        },
                        "action_profile": "cdr2_high_entropy",
                        "intent": "Optimize the full reviewed CDR2 window while retaining typed recovery measurements.",
                        "evidence_refs": [
                            "cab:A:52", "cab:A:53", "cab:A:54", "cab:A:55",
                            "cab:A:56",
                        ],
                        "residue_policy": {
                            "favored_residues": [
                                "A", "D", "E", "F", "G", "H", "I", "K",
                                "L", "M", "N", "P", "Q", "R", "S", "T",
                                "V", "W", "Y",
                            ],
                            "disfavored_residues": ["C"],
                            "position_residue_rules": {},
                            "policy_weight": 1.0,
                        },
                    },
                ],
                "mapping_edges": [
                    {
                        "edge_id": "llm_l0_recovery_point",
                        "functional_node_id": "l0_contact_recovery",
                        "structural_node_id": "llm_cdr3_l0_l1_stage",
                        "action_operator": "point",
                        "evidence_refs": ["cab:A:101", "cab:A:104", "cab:A:105"],
                    },
                    {
                        "edge_id": "llm_l1_recovery_resample",
                        "functional_node_id": "l1_cdr3_recovery",
                        "structural_node_id": "llm_cdr3_l0_l1_stage",
                        "action_operator": "site_resample",
                        "evidence_refs": [
                            "cab:A:98",
                            "cab:A:99",
                            "cab:A:100",
                            "cab:A:102",
                            "cab:A:103",
                            "cab:A:106",
                            "cab:A:107",
                            "cab:A:109",
                            "cab:A:110",
                            "cab:A:111",
                            "cab:A:112",
                            "cab:A:113",
                            "cab:A:114",
                            "cab:A:115",
                            "cab:A:116",
                            "cab:A:117",
                            "cab:A:118",
                            "cab:A:119",
                            "cab:A:120",
                            "cab:A:121",
                        ],
                    },
                    {
                        "edge_id": "llm_l2_cdr1_recovery_point",
                        "functional_node_id": "l2_maturation_recovery",
                        "structural_node_id": "llm_l2_cdr1_stage",
                        "action_operator": "point",
                        "evidence_refs": [
                            "cab:A:25",
                            "cab:A:26",
                            "cab:A:27",
                            "cab:A:28",
                            "cab:A:29",
                            "cab:A:30",
                            "cab:A:31",
                        ],
                    },
                    {
                        "edge_id": "llm_l2_cdr2_recovery_point",
                        "functional_node_id": "l2_maturation_recovery",
                        "structural_node_id": "llm_l2_cdr2_stage",
                        "action_operator": "point",
                        "evidence_refs": [
                            "cab:A:52",
                            "cab:A:53",
                            "cab:A:54",
                            "cab:A:55",
                            "cab:A:56",
                        ],
                    },
                    {
                        "edge_id": "llm_l2_cdr1_integrity_resample",
                        "functional_node_id": "sequence_integrity_guard",
                        "structural_node_id": "llm_l2_cdr1_stage",
                        "action_operator": "site_resample",
                        "evidence_refs": ["cab:A:28", "cab:A:29", "cab:A:30"],
                    },
                    {
                        "edge_id": "llm_l2_cdr2_integrity_resample",
                        "functional_node_id": "sequence_integrity_guard",
                        "structural_node_id": "llm_l2_cdr2_stage",
                        "action_operator": "site_resample",
                        "evidence_refs": ["cab:A:52", "cab:A:56"],
                    },
                    {
                        "edge_id": "llm_cdr3_diversification_segment",
                        "functional_node_id": "cdr_all_a_escape",
                        "structural_node_id": "llm_cdr3_l0_l1_stage",
                        "action_operator": "segment_mutagenesis",
                        "evidence_refs": [
                            "cab:A:98", "cab:A:99", "cab:A:100", "cab:A:101",
                            "cab:A:102", "cab:A:103", "cab:A:104", "cab:A:105",
                            "cab:A:106", "cab:A:107", "cab:A:109", "cab:A:110",
                            "cab:A:111", "cab:A:112", "cab:A:113", "cab:A:114",
                            "cab:A:115", "cab:A:116", "cab:A:117", "cab:A:118",
                            "cab:A:119", "cab:A:120", "cab:A:121",
                        ],
                    },
                    {
                        "edge_id": "llm_cdr1_diversification_segment",
                        "functional_node_id": "cdr_all_a_escape",
                        "structural_node_id": "llm_l2_cdr1_stage",
                        "action_operator": "segment_mutagenesis",
                        "evidence_refs": [
                            "cab:A:25", "cab:A:26", "cab:A:27", "cab:A:28",
                            "cab:A:29", "cab:A:30", "cab:A:31",
                        ],
                    },
                    {
                        "edge_id": "llm_cdr2_diversification_segment",
                        "functional_node_id": "cdr_all_a_escape",
                        "structural_node_id": "llm_l2_cdr2_stage",
                        "action_operator": "segment_mutagenesis",
                        "evidence_refs": [
                            "cab:A:52", "cab:A:53", "cab:A:54", "cab:A:55",
                            "cab:A:56",
                        ],
                    },
                ],
                "decision_record": {
                    "action": "create",
                    "diagnosis": "Define the initial 35-position all-alanine CDR search with fixed L0/L1/L2 recovery measurements.",
                    "hypothesis": "Search the three reviewed CDR windows under the cysteine and HEWL preservation constraints.",
                    "evidence_refs": [
                        "cab:A:25",
                        "cab:A:26",
                        "cab:A:27",
                        "cab:A:28",
                        "cab:A:29",
                        "cab:A:30",
                        "cab:A:31",
                        "cab:A:52",
                        "cab:A:53",
                        "cab:A:54",
                        "cab:A:55",
                        "cab:A:56",
                        "cab:A:98",
                        "cab:A:99",
                        "cab:A:100",
                        "cab:A:101",
                        "cab:A:102",
                        "cab:A:103",
                        "cab:A:104",
                        "cab:A:105",
                        "cab:A:106",
                        "cab:A:107",
                        "cab:A:109",
                        "cab:A:110",
                        "cab:A:111",
                        "cab:A:112",
                        "cab:A:113",
                        "cab:A:114",
                        "cab:A:115",
                        "cab:A:116",
                        "cab:A:117",
                        "cab:A:118",
                        "cab:A:119",
                        "cab:A:120",
                        "cab:A:121",
                    ],
                    "expected_effects": [
                        "compile 35 editable CDR positions and 28 fixed recovery measurement sites",
                        "make every reviewed CDR position immediately reachable by MCTS",
                        "permit high-entropy multi-residue jumps away from the all-alanine parent",
                        "attribute each mutation action to one staged recovery objective",
                        "preserve the VHH disulfide cysteines and fixed HEWL chain",
                    ],
                    "failure_condition": "Reject if the mask is not exactly 35 positions, a protected cysteine becomes editable, or a staged evaluator term is missing.",
                    "confidence": 0.0,
                },
            }
        }
    )
    return strategy


# EVOLVE-BLOCK-END

import os
from pathlib import Path
from typing import Optional

from astevolve.runtime.case_context import current_case_kwargs
from astevolve.runtime.case_program import merge_locked_runtime_defaults
from astevolve.runtime.paths import artifact_path, runtime_root
from engine.case_builder import prepare_case_inputs, run_design_search

from cases.cab_lys3_hewl_cdr_recovery.cab_lys3_evaluator import register_cab_lys3_plugin
from cases.cab_lys3_hewl_cdr_recovery.runtime_profiles import (
    locked_runtime_defaults,
    runtime_profile_summary,
)


register_cab_lys3_plugin()

CASE_ROOT = Path(__file__).resolve().parent
_LOCKED_RUNTIME_DEFAULTS: Dict[str, Any] = {
    "iterations": 6,
    "max_total_mutations": 35,
    "progen_weight": 0.0,
    "progen_chains": ["A"],
    "sequence_prior_model": "progen",
    "chai1_enabled": False,
    "structure_screen_enabled": False,
    "structure_rerank_enabled": False,
    "multistate_objectives_enabled": False,
    "mcts_output_dir": str(artifact_path("cab_lys3_hewl_cdr_recovery", "cpu_inner")),
    "mcts_save_tree": False,
    "mcts_save_variants": False,
    "semantic_required_nodes": [],
    "score_config": {
        "weight_fast": 1.0,
        "weight_plddt": 0.0,
        "weight_iptm": 0.0,
        "weight_ptm": 0.0,
        "weight_interface_plddt": 0.0,
        "weight_node_plddt_min": 0.0,
        "weight_clash": 0.0,
        "weight_multistate": 0.0,
        "evaluator_backends": {},
        "evaluator_weights": {
            "eval_l0_native_recovery": 0.0,
            "eval_l1_native_recovery": 0.0,
            "eval_l2_native_recovery": 0.0,
            "eval_all_nodes_native_recovery": 1.0,
            "eval_mutation_sparsity": 0.0,
            "eval_fixed_residue_integrity": 0.0,
        },
    },
}


def runtime_strategy(structure_mode: Optional[str] = None) -> Dict[str, Any]:


    locked = locked_runtime_defaults(_LOCKED_RUNTIME_DEFAULTS, structure_mode)
    return merge_locked_runtime_defaults(propose_strategy(), locked, base_strategy)


def _case_paths() -> Dict[str, str]:
    case_id = os.environ.get("ASTEVOLVE_CASE_ID")
    has_case_root = bool(
        os.environ.get("ASTEVOLVE_CASE_ROOT") or os.environ.get("ASTEVOLVE_CASES_ROOT")
    )
    if os.environ.get("ASTEVOLVE_CASE_MANIFEST") or (case_id and has_case_root):
        return current_case_kwargs(case_id or "cab_lys3_hewl_cdr_recovery")
    return {
        "design_state_path": str(CASE_ROOT / "design_state.json"),
        "memory_path": str(CASE_ROOT / "memory.yaml"),
    }


def preview_case(structure_mode: Optional[str] = None) -> Dict[str, Any]:


    strategy = runtime_strategy(structure_mode)
    prepared = prepare_case_inputs(strategy, **_case_paths())
    compiled = prepared.blueprint.compile()
    ast = prepared.dual_ast_compilation.ast
    mask_positions = {
        chain_id: [index for index, enabled in enumerate(mask) if bool(enabled)]
        for chain_id, mask in prepared.masks.items()
    }
    return {
        "task_name": prepared.design_state["task_name"],
        "chain_order": compiled["chain_order"],
        "chain_lengths": compiled["chain_lengths"],
        "mask_positions": mask_positions,
        "editable_position_count": sum(len(items) for items in mask_positions.values()),
        "parent_a_is_all_alanine_on_mask": all(
            prepared.template_sequences[chain_id][position] == "A"
            for chain_id, positions in mask_positions.items()
            for position in positions
        ),
        "runtime_output_root": str(runtime_root()),
        "runtime_profile": runtime_profile_summary(strategy),
        "fixed_residue_counts": {
            chain_id: len(residues)
            for chain_id, residues in prepared.fixed_residues.items()
        },
        "ast_revision": ast.revision if ast is not None else None,
        "structural_node_ids": [
            node.node_id for node in prepared.executable_node_plan.structural_nodes
        ],
        "mapping_action_ids": [
            action.action_id for action in prepared.executable_mapping_plan.action_specs
        ],
        "measurement_intent_ids": [
            intent.functional_node_id
            for intent in prepared.executable_node_plan.measurement_intents
        ],
        "ast_revision_report": prepared.design_state.get("_ast_revision_report", {}),
        "strategy_schema_report": prepared.resolved_strategy.get(
            "strategy_schema_report", {}
        ),
    }


def run_search(
    seed: Optional[int] = None,
    structure_mode: Optional[str] = None,
) -> Dict[str, Any]:


    return run_design_search(
        runtime_strategy(structure_mode),
        seed=seed,
        memory_commit_mode="deferred",
        **_case_paths(),
    )
