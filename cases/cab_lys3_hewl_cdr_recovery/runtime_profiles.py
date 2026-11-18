

from __future__ import annotations

from copy import deepcopy
import os
import shutil
from typing import Any, Dict, Mapping, Optional

from astevolve.runtime.paths import artifact_path, data_path, model_root


CPU_MODE = "cpu"
DUAL_GPU_MODE = "dual_gpu"
FORMAL_INNER_ITERATIONS = int(os.environ.get("ASTEVOLVE_FORMAL_INNER_ITERATIONS", "100"))
FORMAL_PROTENIX_CHECKPOINT_INTERVAL = 20
FORMAL_PROTENIX_CHECKPOINT_CANDIDATES = 4
FORMAL_PROTENIX_FINALISTS = 5


FORMAL_PROTENIX_CANDIDATES = FORMAL_PROTENIX_FINALISTS
FORMAL_AF3_FINALISTS = 0
FORMAL_AF3_DIFFUSION_SAMPLES = 0
FINAL_AF3_DIFFUSION_SAMPLES = 0
ROSETTA_INTERFACE = "A_B"
SABDAB_VHH_INDEX_DIR = os.environ.get(
    "ASTEVOLVE_CAB_LYS3_SABDAB_INDEX",
    str(
        data_path(
            "cab_lys3_hewl_cdr_recovery",
            "sabdab_vhh_reference",
            "embeddings",
            "esm2_t6_8M_UR50D",
        )
    ),
)
SABDAB_ESM2_MODEL_DIR = os.environ.get(
    "ASTEVOLVE_CAB_LYS3_ESM2_MODEL",
    str(model_root() / "esm2_t6_8M_UR50D"),
)
SABDAB_PRIOR_DEVICE = os.environ.get(
    "ASTEVOLVE_CAB_LYS3_SABDAB_DEVICE",
    "cpu",
)
BOOTSTRAP_SEQUENCE_ENV = "ASTEVOLVE_CAB_LYS3_BOOTSTRAP_SEQUENCE_A"
_CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_ALIASES = {
    "": DUAL_GPU_MODE,
    "full": DUAL_GPU_MODE,
    "cpu": CPU_MODE,
    "off": CPU_MODE,
    "none": CPU_MODE,
    "dual": DUAL_GPU_MODE,
    "dual_gpu": DUAL_GPU_MODE,
    "af3_protenix": DUAL_GPU_MODE,
}

_SEQUENCE_WEIGHT_FIELDS = (
    "eval_l0_native_recovery",
    "eval_l1_native_recovery",
    "eval_l2_native_recovery",
    "eval_all_nodes_native_recovery",
    "eval_cdr_all_a_escape",
    "eval_mutation_sparsity",
    "eval_fixed_residue_integrity",
    "eval_developability_score",
    "eval_sabdab_vhh_plausibility",
    "eval_sabdab_whole_vhh_similarity",
    "eval_sabdab_framework_similarity",
    "eval_sabdab_cdr1_similarity",
    "eval_sabdab_cdr2_similarity",
    "eval_sabdab_cdr3_similarity",
)

_STRUCTURE_WEIGHT_FIELDS = (
    "eval_af3_binder_confidence",
    "eval_af3_design_region_confidence",
    "eval_af3_interface_confidence",
    "eval_af3_interface_plddt",
    "eval_af3_interface_pae_quality",
    "eval_af3_contact_quality",
    "eval_af3_stage_iptm_gate",
    "eval_af3_structure_quality",
    "eval_protenix_binder_confidence",
    "eval_protenix_design_region_confidence",
    "eval_protenix_interface_confidence",
    "eval_protenix_interface_plddt",
    "eval_protenix_interface_pae_quality",
    "eval_protenix_contact_quality",
    "eval_protenix_structure_quality",
    "eval_dual_structure_consensus",
    "eval_dual_structure_worst_case",
    "eval_dual_structure_agreement",
    "eval_dual_structure_score",
)

_CASE_WEIGHT_FIELDS = frozenset((*_SEQUENCE_WEIGHT_FIELDS, *_STRUCTURE_WEIGHT_FIELDS))


_GENERIC_EVALUATOR_WEIGHT_FIELDS = (
    "eval_chain_continuity",
    "eval_clash",
    "eval_compactness",
    "eval_contacts",
    "eval_contract_response",
    "eval_global_plddt",
    "eval_hbond_geometry",
    "eval_hbond_proxy",
    "eval_hydrophobic_geometry",
    "eval_hydrophobic_proxy",
    "eval_interface_plddt",
    "eval_iptm",
    "eval_multistate",
    "eval_mutation_scope",
    "eval_node_floor",
    "eval_preserved_node_confidence",
    "eval_preserved_node_rmsd",
    "eval_preserved_sequence",
    "eval_primary_engagement",
    "eval_ptm",
    "eval_residue_pairs",
    "eval_salt_geometry",
    "eval_salt_proxy",
    "eval_scaffold_rmsd",
    "eval_state_confidence",
    "eval_target_confidence_floor",
)


def resolve_rosetta_command() -> Optional[str]:
    configured = str(os.environ.get("ASTEVOLVE_ROSETTA_INTERFACE_ANALYZER") or "").strip()
    if configured:
        resolved = shutil.which(configured)
        if resolved:
            return resolved
        if os.path.isfile(configured) and os.access(configured, os.X_OK):
            return configured
    for candidate in (
        "InterfaceAnalyzer",
        "InterfaceAnalyzer.default.linuxgccrelease",
        "interface_analyzer.default.linuxgccrelease",
    ):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def resolve_pyrosetta_python() -> Optional[str]:
    configured = str(os.environ.get("ASTEVOLVE_PYROSETTA_PYTHON") or "").strip()
    if not configured:
        return None
    resolved = shutil.which(configured)
    if resolved:
        return resolved
    if os.path.isfile(configured) and os.access(configured, os.X_OK):
        return configured
    return None


def resolve_structure_mode(requested: Optional[str] = None) -> str:


    raw = (
        requested
        if requested is not None
        else os.environ.get("ASTEVOLVE_STRUCTURE_MODE", DUAL_GPU_MODE)
    )
    key = str(raw or "").strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            "ASTEVOLVE_STRUCTURE_MODE must be cpu or dual_gpu " f"(received {raw!r})"
        )
    return _ALIASES[key]


def resolve_sabdab_prior_enabled() -> bool:
    """This full case always uses the external ESM-2 reference prior."""
    return True



def locked_runtime_defaults(
    cpu_defaults: Mapping[str, Any],
    requested_mode: Optional[str] = None,
) -> Dict[str, Any]:


    mode = resolve_structure_mode(requested_mode)
    defaults = deepcopy(dict(cpu_defaults))
    sabdab_enabled = resolve_sabdab_prior_enabled()
    for name, default in {
        "ASTEVOLVE_PROTENIX_MSA_TEMPLATE_ROOT": str(
            data_path("cab_lys3_hewl_cdr_recovery", "msa_cache")
        ),
        "ASTEVOLVE_PROTENIX_ROOT": str(model_root() / "protenix-v2"),
        "ASTEVOLVE_PROTENIX_CONDA_ENV": "protenix-v2",
    }.items():
        if not str(os.environ.get(name) or "").strip():
            os.environ[name] = default
    bootstrap_sequence = str(os.environ.get(BOOTSTRAP_SEQUENCE_ENV) or "").strip()
    if bootstrap_sequence:
        if len(bootstrap_sequence) != 133:
            raise ValueError(
                f"{BOOTSTRAP_SEQUENCE_ENV} must contain exactly 133 residues"
            )
        invalid = sorted(set(bootstrap_sequence) - _CANONICAL_AMINO_ACIDS)
        if invalid:
            raise ValueError(
                f"{BOOTSTRAP_SEQUENCE_ENV} contains invalid residues: {invalid}"
            )
        defaults["resume_template_seqs"] = {"A": bootstrap_sequence}
    score_config = deepcopy(dict(defaults.get("score_config") or {}))
    raw_evaluator_weights = deepcopy(
        dict(score_config.pop("evaluator_weights", {}) or {})
    )
    evaluator_weights = {
        key: value
        for key, value in raw_evaluator_weights.items()
        if key in _CASE_WEIGHT_FIELDS
    }
    generic_evaluator_weights = {
        key: value
        for key, value in raw_evaluator_weights.items()
        if key not in _CASE_WEIGHT_FIELDS
    }
    for field in _GENERIC_EVALUATOR_WEIGHT_FIELDS:
        generic_evaluator_weights.setdefault(field, 0.0)
    for field in _STRUCTURE_WEIGHT_FIELDS:
        evaluator_weights.setdefault(field, 0.0)
    score_config.update(
        {
            "evaluator_weights": generic_evaluator_weights,
            "evaluator_plugins": ["cab_lys3_cpu"],
            "plugin_config": {
                "cab_lys3_cpu": {
                    "dual_structure_required": False,
                    "dual_structure_af3_weight": 0.0,
                    "dual_structure_disagreement_weight": 0.0,
                    "dual_structure_worst_case_weight": 0.0,
                    "staged_interface_gate_required": False,
                    "staged_l0_af3_iptm_floor": 0.10,
                    "staged_l1_af3_iptm_floor": 0.18,
                    "staged_l2_af3_iptm_floor": 0.25,
                    "staged_l0_unlock_recovery": 1.0 / 3.0,
                    "staged_l1_unlock_recovery": 0.15,
                    "sabdab_prior_enabled": sabdab_enabled,
                    "sabdab_prior_required": sabdab_enabled,
                    "sabdab_prior_index_dir": SABDAB_VHH_INDEX_DIR if sabdab_enabled else "",
                    "sabdab_prior_model_dir": SABDAB_ESM2_MODEL_DIR if sabdab_enabled else "",
                    "sabdab_prior_device": "cpu",
                    "sabdab_prior_top_k": 20,
                    "evaluator_weights": evaluator_weights,
                }
            },
        }
    )
    defaults["score_config"] = score_config
    if mode == CPU_MODE:
        return defaults

    final_high_fidelity_candidates = (
        FORMAL_PROTENIX_FINALISTS
        if FORMAL_INNER_ITERATIONS >= 30
        else min(1, FORMAL_PROTENIX_FINALISTS)
    )


    defaults.update(
        {
            "iterations": FORMAL_INNER_ITERATIONS,
            "search_method": "mcts",


            "mcts_c_puct": 2.0,
            "mcts_max_depth": 30,
            "mcts_progressive_widening_c": 0.25,
            "mcts_progressive_widening_alpha": 0.5,
            "mcts_tree_quality_required": True,
            "mcts_tree_min_root_children": 2,
            "mcts_tree_min_branching_nodes": 2,
            "mcts_tree_min_leaves": 4,
            "mcts_tree_min_max_depth": 3,
            "mcts_iteration_unit": "evaluated_unique_candidates",
            "mcts_candidate_budget_max_round_multiplier": 30,
            "mcts_candidate_budget_fail_on_underfill": True,
            "mcts_fidelity_upgrade_enabled": False,
            "mcts_fidelity_upgrade_provider": "protenix",
            "mcts_fidelity_upgrade_interval": FORMAL_PROTENIX_CHECKPOINT_INTERVAL,
            "mcts_fidelity_upgrade_candidates": FORMAL_PROTENIX_CHECKPOINT_CANDIDATES,
            "mcts_fidelity_upgrade_final_candidates": final_high_fidelity_candidates,
            "mcts_fidelity_upgrade_required": False,
            "progen_weight": 0.20,
            "progen_chains": ["A"],
            "sequence_prior_model": "progen",
            "inner_structure_enabled": True,
            "inner_structure_model": "protenix",
            "inner_structure_model_name": "protenix-v2",
            "inner_structure_weight": 1.0,
            "inner_structure_fail_closed": True,


            "inner_structure_hard_gate": False,
            "promote_inline_winner_structure_evidence": True,
            "inner_esmfold2_enabled": False,
            "chai1_enabled": True,
            "chai1_top_frac": 0.05,
            "chai1_min_candidates": final_high_fidelity_candidates,
            "chai1_max_candidates": final_high_fidelity_candidates,
            "structure_model": "protenix",
            "structure_model_name": "protenix-v2",
            "structure_prescreen_enabled": False,
            "structure_screen_enabled": False,
            "structure_rerank_enabled": False,
            "structure_physics_max_candidates": 0,
            "node_optimizer_enabled": True,
            "node_optimizer_candidate_count": 8,
            "node_optimizer_beam_width": 16,
            "node_optimizer_top_k_per_position": 6,
            "node_optimizer_temperature": 1.60,
            "node_optimizer_diversity_weight": 0.35,
            "node_optimizer_mutation_penalty": 0.03,
            "node_optimizer_prior_model": "masked_lm",
            "mutation_rate": 0.45,
            "proposal_tier_mode": "mixed",
            "proposal_exploit_frac": 0.30,
            "proposal_explore_frac": 0.60,
            "proposal_repair_frac": 0.10,
            "exploit_max_mutations": 20,
            "explore_max_mutations": 30,
            "repair_max_mutations": 8,
            "max_total_mutations": 35,
            "sequence_prefilter_callable": (
                "cases.cab_lys3_hewl_cdr_recovery.cab_lys3_sequence_gate:"
                "evaluate_pre_model_gate"
            ),
            "sequence_prefilter_config": {
                "sabdab_prior_enabled": sabdab_enabled,
                "sabdab_prior_index_dir": SABDAB_VHH_INDEX_DIR if sabdab_enabled else "",
                "sabdab_prior_model_dir": SABDAB_ESM2_MODEL_DIR if sabdab_enabled else "",
                "sabdab_prior_device": "cpu",
                "sabdab_prior_top_k": 20,
            },
            "structure_allow_low_fidelity_fallback": False,
            "structure_selection_objective": "outer_aligned",
            "structure_batch_size": 1,
            "structure_parallel_workers": 1,
            "protenix_model_name": "protenix-v2",
            "protenix_complex_use_msa": True,
            "protenix_complex_cycle": 10,
            "protenix_complex_step": 200,
            "protenix_complex_sample": 1,
            "protenix_complex_use_default_params": True,
            "protenix_complex_timeout": 1800,
            "multistate_objectives_enabled": False,
            "mcts_save_tree": True,
            "mcts_save_variants": True,
            "mcts_artifact_mode": "normalized",
            "mcts_output_dir": str(
                artifact_path("cab_lys3_hewl_cdr_recovery", "dual_gpu_inner")
            ),
        }
    )
    if sabdab_enabled:
        defaults.update(
            {
                "sequence_bootstrap_callable": (
                    "cases.cab_lys3_hewl_cdr_recovery.cab_lys3_sequence_gate:"
                    "build_sabdab_bootstrap_proposals"
                ),
                "sequence_bootstrap_config": {
                    "sabdab_prior_enabled": True,
                    "sabdab_prior_index_dir": SABDAB_VHH_INDEX_DIR,
                    "sabdab_prior_model_dir": SABDAB_ESM2_MODEL_DIR,
                    "sabdab_prior_device": "cpu",
                    "sabdab_prior_top_k": 20,
                },
            }
        )
    if FORMAL_INNER_ITERATIONS < 30:
        defaults.update(
            {
                "mcts_tree_min_root_children": 2,
                "mcts_tree_min_branching_nodes": 1,
                "mcts_tree_min_leaves": 4,
                "mcts_tree_min_max_depth": 2,
            }
        )
    score_config = deepcopy(score_config)
    evaluator_weights = deepcopy(evaluator_weights)
    for field in _STRUCTURE_WEIGHT_FIELDS:
        evaluator_weights[field] = 0.0
    evaluator_weights.update(
        {
            "eval_l0_native_recovery": 0.35,
            "eval_l1_native_recovery": 0.25,
            "eval_l2_native_recovery": 0.15,
            "eval_all_nodes_native_recovery": 0.25,
            "eval_cdr_all_a_escape": 0.40,
            "eval_developability_score": 0.10,
            "eval_sabdab_vhh_plausibility": 0.35,
            "eval_sabdab_cdr1_similarity": 0.05,
            "eval_sabdab_cdr2_similarity": 0.05,
            "eval_sabdab_cdr3_similarity": 0.15,
            "eval_protenix_interface_confidence": 0.25,
            "eval_protenix_interface_plddt": 0.15,
            "eval_protenix_interface_pae_quality": 0.15,
            "eval_protenix_contact_quality": 0.10,
        }
    )
    if not sabdab_enabled:
        for field in (
            "eval_sabdab_vhh_plausibility", "eval_sabdab_whole_vhh_similarity",
            "eval_sabdab_framework_similarity", "eval_sabdab_cdr1_similarity",
            "eval_sabdab_cdr2_similarity", "eval_sabdab_cdr3_similarity",
        ):
            evaluator_weights[field] = 0.0
    plugin_config = deepcopy(score_config["plugin_config"])
    plugin_config["cab_lys3_cpu"].update(
        {
            "dual_structure_required": False,
            "dual_structure_af3_weight": 0.0,
            "dual_structure_disagreement_weight": 0.0,
            "dual_structure_worst_case_weight": 0.0,
            "staged_interface_gate_required": False,
            "staged_l0_af3_iptm_floor": 0.10,
            "staged_l1_af3_iptm_floor": 0.18,
            "staged_l2_af3_iptm_floor": 0.25,
            "staged_l0_unlock_recovery": 1.0 / 3.0,
            "staged_l1_unlock_recovery": 0.15,
            "sabdab_prior_enabled": sabdab_enabled,
            "sabdab_prior_required": sabdab_enabled,
            "sabdab_prior_index_dir": SABDAB_VHH_INDEX_DIR if sabdab_enabled else "",
            "sabdab_prior_model_dir": SABDAB_ESM2_MODEL_DIR if sabdab_enabled else "",
            "sabdab_prior_device": SABDAB_PRIOR_DEVICE,
            "sabdab_prior_top_k": 20,
            "evaluator_weights": evaluator_weights,
        }
    )
    score_config["plugin_config"] = plugin_config
    generic_weights = deepcopy(dict(score_config.get("evaluator_weights") or {}))
    generic_weights["eval_pyrosetta"] = 0.0
    score_config["evaluator_weights"] = generic_weights
    backends = deepcopy(dict(score_config.get("evaluator_backends") or {}))
    pyrosetta_python = resolve_pyrosetta_python()
    backends["pyrosetta"] = {
        "enabled": False,
        "required": False,
        "interface": ROSETTA_INTERFACE,
        "source_interface": "A_L",
        "timeout": 600,
        "analysis_role": "disabled_legacy_interface_physics",
        "defer_until_stage": "physics",
        "fastrelax": False,
        "fastrelax_repeats": 2,
        "backbone_coordinate_constraint": True,
        "interface_dg_good": -25.0,
        "interface_dg_bad": 10.0,
    }
    if pyrosetta_python is not None:
        backends["pyrosetta"]["python"] = pyrosetta_python
    score_config["evaluator_backends"] = backends
    defaults["score_config"] = score_config
    return defaults


def runtime_profile_summary(defaults: Mapping[str, Any]) -> Dict[str, Any]:


    score = dict(defaults.get("score_config") or {})
    plugin = dict((score.get("plugin_config") or {}).get("cab_lys3_cpu") or {})
    weights = dict(plugin.get("evaluator_weights") or {})
    return {
        "mode": (
            DUAL_GPU_MODE
            if bool(defaults.get("inner_structure_enabled"))
            else CPU_MODE
        ),
        "structure_enabled": bool(defaults.get("chai1_enabled")),
        "inner_iterations": int(defaults.get("iterations", 0) or 0),
        "fast_structure_provider": defaults.get("inner_structure_model"),
        "fidelity_checkpoints_enabled": bool(defaults.get("mcts_fidelity_upgrade_enabled")),
        "model_name": defaults.get("protenix_model_name"),
        "frozen_msa_enabled": bool(defaults.get("protenix_complex_use_msa")),
        "protenix_cycle": defaults.get("protenix_complex_cycle"),
        "protenix_step": defaults.get("protenix_complex_step"),
        "protenix_sample": defaults.get("protenix_complex_sample"),
        "final_high_fidelity_candidates": defaults.get(
            "mcts_fidelity_upgrade_final_candidates"
        ),
        "screen_provider": defaults.get("structure_model"),
        "rerank_provider": None,
        "screen_mutant_cap": defaults.get("chai1_max_candidates"),
        "rerank_mutant_cap": 0,
        "af3_diffusion_samples": 0,
        "final_af3_diffusion_samples": FINAL_AF3_DIFFUSION_SAMPLES,
        "dual_structure_required": bool(plugin.get("dual_structure_required")),
        "dual_structure_weight": float(
            weights.get("eval_dual_structure_score", 0.0) or 0.0
        ),
        "af3_consensus_weight": float(
            plugin.get("dual_structure_af3_weight", 0.0) or 0.0
        ),
        "disagreement_penalty_weight": float(
            plugin.get("dual_structure_disagreement_weight", 0.0) or 0.0
        ),
        "worst_case_weight": float(
            plugin.get("dual_structure_worst_case_weight", 0.0) or 0.0
        ),
        "staged_interface_gate_required": bool(
            plugin.get("staged_interface_gate_required")
        ),
        "sabdab_prior_enabled": bool(plugin.get("sabdab_prior_enabled")),
        "sabdab_prior_required": bool(plugin.get("sabdab_prior_required")),
        "sabdab_prior_device": plugin.get("sabdab_prior_device"),
        "sabdab_prior_weight": float(
            weights.get("eval_sabdab_vhh_plausibility", 0.0) or 0.0
        ),
        "sabdab_external_embedding_used": bool(
            plugin.get("sabdab_prior_enabled")
            and defaults.get("sequence_bootstrap_callable")
        ),
        "sabdab_prior_index_dir": plugin.get("sabdab_prior_index_dir"),
        "pyrosetta_enabled": bool(
            ((score.get("evaluator_backends") or {}).get("pyrosetta") or {}).get("enabled")
        ),
        "pyrosetta_required": bool(
            ((score.get("evaluator_backends") or {}).get("pyrosetta") or {}).get("required")
        ),
        "pyrosetta_python_available": resolve_pyrosetta_python() is not None,
        "af3_model_dir": defaults.get("af3_model_dir"),
        "mcts_output_dir": defaults.get("mcts_output_dir"),
        "mcts_tree_quality_required": bool(
            defaults.get("mcts_tree_quality_required")
        ),
        "mcts_tree_quality_thresholds": {
            "root_children": int(
                defaults.get("mcts_tree_min_root_children", 0) or 0
            ),
            "branching_nodes": int(
                defaults.get("mcts_tree_min_branching_nodes", 0) or 0
            ),
            "leaves": int(defaults.get("mcts_tree_min_leaves", 0) or 0),
            "max_depth": int(
                defaults.get("mcts_tree_min_max_depth", 0) or 0
            ),
        },
    }


__all__ = [
    "CPU_MODE",
    "DUAL_GPU_MODE",
    "FORMAL_INNER_ITERATIONS",
    "FORMAL_PROTENIX_CANDIDATES",
    "FORMAL_PROTENIX_CHECKPOINT_INTERVAL",
    "FORMAL_PROTENIX_CHECKPOINT_CANDIDATES",
    "FORMAL_PROTENIX_FINALISTS",
    "FORMAL_AF3_FINALISTS",
    "FORMAL_AF3_DIFFUSION_SAMPLES",
    "FINAL_AF3_DIFFUSION_SAMPLES",
    "BOOTSTRAP_SEQUENCE_ENV",
    "ROSETTA_INTERFACE",
    "SABDAB_PRIOR_ENABLED_ENV",
    "SABDAB_VHH_INDEX_DIR",
    "SABDAB_ESM2_MODEL_DIR",
    "SABDAB_PRIOR_DEVICE",
    "locked_runtime_defaults",
    "resolve_pyrosetta_python",
    "resolve_rosetta_command",
    "resolve_sabdab_prior_enabled",
    "resolve_structure_mode",
    "runtime_profile_summary",
]
