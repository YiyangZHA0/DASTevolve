

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from astevolve.runtime.conda import (
    resolve_alphafold3_conda_env,
    resolve_protenix_conda_env,
)
from astevolve.evaluation.plugins.registry import (
    PLUGIN_REGISTRY,
    PLUGIN_RESOLUTION_VERSION,
    PluginConfigError,
    normalize_plugin_config,
    normalize_plugin_requests,
)
from astevolve.search.config import (
    CANDIDATE_WAVE_CONFIG_FIELDS,
    normalize_residue_mutation_contract,
    validate_candidate_wave_config,
)
from astevolve.search.operator_registry import (
    default_operator_weights,
    validate_operator_weights,
)
from astevolve.search.sequence_generator import CONSTRAINT_AWARE_GENERATOR_ID

from .design_state import PROJECT_ROOT
from .strategy_compiler import _as_bool, _safe_float, _safe_int

def _apply_search_schedule(strategy: Dict[str, Any]) -> Dict[str, Any]:


    updated = dict(strategy)
    requested = str(
        os.environ.get("ASTEVOLVE_OUTER_LOOP_PHASE")
        or updated.get("outer_loop_phase")
        or "explore_ast"
    ).strip().lower()
    aliases = {
        "early": "explore_ast",
        "explore": "explore_ast",
        "explore_ast": "explore_ast",
        "mid": "refine_ast",
        "middle": "refine_ast",
        "refine": "refine_ast",
        "refine_ast": "refine_ast",
        "late": "converge",
        "converge": "converge",
        "exploit": "converge",
    }
    phase = aliases.get(requested, "explore_ast")
    defaults: Dict[str, Dict[str, Any]] = {
        "explore_ast": {
            "proposal_explore_frac": 0.38,
            "proposal_exploit_frac": 0.48,
            "proposal_repair_frac": 0.14,
            "max_total_mutations": max(int(updated.get("max_total_mutations", 12)), 12),
            "explore_max_mutations": max(int(updated.get("explore_max_mutations", 8)), 10),
            "mcts_c_puct": max(float(updated.get("mcts_c_puct", 1.4)), 1.35),
            "mcts_max_depth": max(int(updated.get("mcts_max_depth", 4)), 4),
        },
        "refine_ast": {
            "proposal_explore_frac": 0.22,
            "proposal_exploit_frac": 0.66,
            "proposal_repair_frac": 0.12,
            "max_total_mutations": min(max(int(updated.get("max_total_mutations", 10)), 8), 12),
            "explore_max_mutations": min(max(int(updated.get("explore_max_mutations", 7)), 6), 10),
            "mcts_c_puct": min(float(updated.get("mcts_c_puct", 1.25)), 1.25),
            "mcts_max_depth": max(int(updated.get("mcts_max_depth", 4)), 4),
        },
        "converge": {
            "proposal_explore_frac": 0.08,
            "proposal_exploit_frac": 0.78,
            "proposal_repair_frac": 0.14,
            "max_total_mutations": min(max(int(updated.get("max_total_mutations", 6)), 4), 8),
            "explore_max_mutations": min(max(int(updated.get("explore_max_mutations", 4)), 3), 6),
            "mcts_c_puct": min(float(updated.get("mcts_c_puct", 1.1)), 1.15),
            "mcts_max_depth": max(int(updated.get("mcts_max_depth", 5)), 5),
        },
    }
    schedule = updated.get("search_schedule", {})
    schedule_dict = dict(schedule) if isinstance(schedule, dict) else {}
    phase_overrides = schedule_dict.get("phases", {})
    patch = dict(defaults[phase])
    if isinstance(phase_overrides, dict) and isinstance(phase_overrides.get(phase), dict):
        patch.update(phase_overrides[phase])


    hard_cap_raw = schedule_dict.get("hard_max_total_mutations")
    if hard_cap_raw is not None:
        hard_cap = max(0, int(hard_cap_raw))
        for field_name in (
            "max_total_mutations",
            "explore_max_mutations",
            "exploit_max_mutations",
            "repair_max_mutations",
        ):
            if field_name in patch or field_name in updated:
                patch[field_name] = min(
                    int(patch.get(field_name, updated.get(field_name, hard_cap))),
                    hard_cap,
                )
    updated.update(patch)
    updated["outer_loop_phase"] = phase
    updated["search_schedule"] = {
        **schedule_dict,
        "enabled": bool(schedule_dict.get("enabled", True)),
        "active_phase": phase,
        "applied_patch": patch,
        "phase_policy": schedule_dict.get(
            "phase_policy",
            "explore_ast broadens AST/search-space edits; refine_ast balances repair and exploitation; converge narrows edits and increases inner-loop precision.",
        ),
    }
    return updated


def build_sa_config(
    strategy: Dict[str, Any],
    *,
    runtime_mcts_output_dir: str | Path | None = None,
) -> Dict[str, Any]:


    strategy = _apply_search_schedule(strategy)
    scoped_output_dir = (
        str(runtime_mcts_output_dir).strip()
        if runtime_mcts_output_dir is not None
        else ""
    )
    mcts_output_dir = Path(
        scoped_output_dir
        or str(os.environ.get("ASTEVOLVE_MCTS_OUTPUT_DIR") or "").strip()
        or str(strategy.get("mcts_output_dir", "inner_loop"))
    )
    if not mcts_output_dir.is_absolute():
        mcts_output_dir = PROJECT_ROOT / mcts_output_dir

    def as_string_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw = [item.strip() for item in value.replace(";", ",").split(",")]
        elif isinstance(value, (list, tuple, set)):
            raw = [str(item).strip() for item in value]
        else:
            raw = [str(value).strip()]
        out: List[str] = []
        for item in raw:
            if item and item not in out:
                out.append(item)
        return out

    def env_int_list(name: str, current: Any) -> List[int]:
        raw = os.environ.get(name)
        value = current if raw is None or raw == "" else raw
        items = (
            [item.strip() for item in value.replace(";", ",").split(",")]
            if isinstance(value, str)
            else list(value or [])
        )
        output: List[int] = []
        for item in items:
            if isinstance(item, bool):
                raise ValueError(f"{name} must contain integers")
            number = int(item)
            if number in output:
                raise ValueError(f"{name} contains duplicate seeds")
            output.append(number)
        return output

    def env_str(name: str, current: Any) -> Any:
        raw = os.environ.get(name)
        return current if raw is None or raw == "" else raw

    def env_bool(name: str, current: Any) -> bool:
        raw = os.environ.get(name)
        return _as_bool(current) if raw is None or raw == "" else _as_bool(raw)

    def env_int(name: str, current: Any) -> int:
        raw = os.environ.get(name)
        return _safe_int(current if raw is None or raw == "" else raw, _safe_int(current, 0))

    def env_float(name: str, current: Any) -> float:
        raw = os.environ.get(name)
        return _safe_float(current if raw is None or raw == "" else raw, _safe_float(current, 0.0))

    def env_optional_bool(name: str, current: Any) -> Any:
        raw = os.environ.get(name)
        return current if raw is None or raw == "" else _as_bool(raw)

    def env_optional_int(name: str, current: Any) -> Any:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return current
        return _safe_int(raw, _safe_int(current, 0))

    def env_strict_bool(name: str, current: Any) -> bool:
        raw = os.environ.get(name)
        value = current if raw is None or raw == "" else raw
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and not isinstance(value, bool) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"1", "true", "yes", "y", "on"}:
                return True
            if token in {"0", "false", "no", "n", "off"}:
                return False
        raise ValueError(f"{name} must be an explicit boolean")

    def env_strict_int(name: str, current: Any) -> int:
        raw = os.environ.get(name)
        value = current if raw is None or raw == "" else raw
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            token = value.strip()
            if token and (
                token.isdecimal()
                or (token[0] == "-" and token[1:].isdecimal())
            ):
                return int(token)
        raise ValueError(f"{name} must be an integer")

    search_method = str(
        env_str("ASTEVOLVE_SEARCH_METHOD", strategy.get("search_method", "mcts"))
    )
    operator_mode = "node" if search_method.lower() == "mcts" else "legacy"
    raw_mutation_ops = strategy.get("mutation_ops")
    if raw_mutation_ops is None:
        mutation_ops = default_operator_weights(operator_mode)
    else:
        mutation_ops = validate_operator_weights(
            raw_mutation_ops,
            mode=operator_mode,
            context="strategy.mutation_ops",
        )

    node_edit_policies: Dict[str, Dict[str, Any]] = {}
    raw_node_policies = strategy.get("node_edit_policies", {})
    if isinstance(raw_node_policies, dict):
        for node_name, raw_policy in raw_node_policies.items():
            if not isinstance(raw_policy, dict):
                node_edit_policies[str(node_name)] = raw_policy
                continue
            policy = dict(raw_policy)
            if "mutation_ops" in policy:
                policy["mutation_ops"] = validate_operator_weights(
                    policy["mutation_ops"],
                    mode="node",
                    context=f"strategy.node_edit_policies.{node_name}.mutation_ops",
                )
            node_edit_policies[str(node_name)] = policy

    config = {
        "iterations": env_int("ASTEVOLVE_INNER_ITERATIONS", strategy.get("iterations", 1200)),
        "init_temp": float(strategy.get("init_temp", 2.0)),
        "cooling": float(strategy.get("cooling", 0.995)),
        "mutation_rate": float(strategy.get("mutation_rate", 0.06)),
        "resample_segment_prob": float(strategy.get("resample_segment_prob", 0.08)),
        "progen_weight": env_float("ASTEVOLVE_PROGEN_WEIGHT", strategy.get("progen_weight", 1.0)),
        "progen_chains": as_string_list(strategy.get("progen_chains", [])),
        "progen_reduce": "length_weighted",
        "sequence_prior_model": str(strategy.get("sequence_prior_model", "progen")),
        "inner_structure_enabled": env_bool("ASTEVOLVE_INNER_STRUCTURE_ENABLED", strategy.get("inner_structure_enabled", False)),
        "inner_structure_model": str(env_str("ASTEVOLVE_INNER_STRUCTURE_MODEL", strategy.get("inner_structure_model", "esmfold2"))),
        "inner_structure_model_name": env_str("ASTEVOLVE_INNER_STRUCTURE_MODEL_NAME", strategy.get("inner_structure_model_name")),
        "inner_structure_weight": env_float("ASTEVOLVE_INNER_STRUCTURE_WEIGHT", strategy.get("inner_structure_weight", 1.0)),
        "inner_structure_fail_closed": env_bool("ASTEVOLVE_INNER_STRUCTURE_FAIL_CLOSED", strategy.get("inner_structure_fail_closed", True)),
        "inner_structure_hard_gate": env_bool("ASTEVOLVE_INNER_STRUCTURE_HARD_GATE", strategy.get("inner_structure_hard_gate", True)),
        "inner_structure_failure_penalty": env_float("ASTEVOLVE_INNER_STRUCTURE_FAILURE_PENALTY", strategy.get("inner_structure_failure_penalty", 1000.0)),
        "promote_inline_winner_structure_evidence": env_bool("ASTEVOLVE_PROMOTE_INLINE_WINNER_STRUCTURE_EVIDENCE", strategy.get("promote_inline_winner_structure_evidence", False)),
        "inner_esmfold2_enabled": env_bool("ASTEVOLVE_INNER_ESMFOLD2_ENABLED", strategy.get("inner_esmfold2_enabled", False)),
        "inner_esmfold2_model_name": env_str("ASTEVOLVE_INNER_ESMFOLD2_MODEL_NAME", strategy.get("inner_esmfold2_model_name")),
        "inner_esmfold2_interval": env_int("ASTEVOLVE_INNER_ESMFOLD2_INTERVAL", strategy.get("inner_esmfold2_interval", 10)),
        "inner_esmfold2_weight": env_float("ASTEVOLVE_INNER_ESMFOLD2_WEIGHT", strategy.get("inner_esmfold2_weight", 1.0)),
        "mcts_node_sweep_enabled": env_bool("ASTEVOLVE_MCTS_NODE_SWEEP_ENABLED", strategy.get("mcts_node_sweep_enabled", False)),
        "mcts_node_sweep_count": env_int("ASTEVOLVE_MCTS_NODE_SWEEP_COUNT", strategy.get("mcts_node_sweep_count", 0)),
        "mcts_node_sweep_parent_policy": str(env_str("ASTEVOLVE_MCTS_NODE_SWEEP_PARENT_POLICY", strategy.get("mcts_node_sweep_parent_policy", "incumbent"))),
        "mcts_fidelity_upgrade_enabled": env_bool(
            "ASTEVOLVE_MCTS_FIDELITY_UPGRADE_ENABLED",
            strategy.get("mcts_fidelity_upgrade_enabled", False),
        ),
        "mcts_fidelity_upgrade_provider": str(
            env_str(
                "ASTEVOLVE_MCTS_FIDELITY_UPGRADE_PROVIDER",
                strategy.get("mcts_fidelity_upgrade_provider", "protenix"),
            )
        ),
        "mcts_fidelity_upgrade_interval": env_int(
            "ASTEVOLVE_MCTS_FIDELITY_UPGRADE_INTERVAL",
            strategy.get("mcts_fidelity_upgrade_interval", 20),
        ),
        "mcts_fidelity_upgrade_candidates": env_int(
            "ASTEVOLVE_MCTS_FIDELITY_UPGRADE_CANDIDATES",
            strategy.get("mcts_fidelity_upgrade_candidates", 4),
        ),
        "mcts_fidelity_upgrade_final_candidates": env_int(
            "ASTEVOLVE_MCTS_FIDELITY_UPGRADE_FINAL_CANDIDATES",
            strategy.get("mcts_fidelity_upgrade_final_candidates", 5),
        ),
        "mcts_fidelity_upgrade_required": env_bool(
            "ASTEVOLVE_MCTS_FIDELITY_UPGRADE_REQUIRED",
            strategy.get("mcts_fidelity_upgrade_required", False),
        ),
        "chai1_enabled": env_bool("ASTEVOLVE_ENABLE_PROTENIX", strategy.get("chai1_enabled", True)),
        "chai1_top_frac": float(strategy.get("chai1_top_frac", 0.01)),
        "chai1_min_candidates": int(strategy.get("chai1_min_candidates", 1)),
        "chai1_max_candidates": int(strategy.get("chai1_max_candidates", 3)),
        "chai1_num_trunk_recycles": 3,
        "chai1_num_diffn_timesteps": 50,
        "protenix_model_name": str(
            env_str(
                "ASTEVOLVE_PROTENIX_MODEL_NAME",
                strategy.get("protenix_model_name", "protenix_mini_esm_v0.5.0"),
            )
        ),
        "protenix_conda_env": resolve_protenix_conda_env(
            env_str(
                "ASTEVOLVE_PROTENIX_CONDA_ENV",
                strategy.get("protenix_conda_env"),
            )
        ),
        "protenix_seed": int(strategy.get("protenix_seed", 101)),
        "protenix_complex_use_msa": env_optional_bool(
            "ASTEVOLVE_PROTENIX_COMPLEX_USE_MSA",
            strategy.get("protenix_complex_use_msa"),
        ),
        "protenix_complex_cycle": env_optional_int(
            "ASTEVOLVE_PROTENIX_COMPLEX_CYCLE",
            strategy.get("protenix_complex_cycle"),
        ),
        "protenix_complex_step": env_optional_int(
            "ASTEVOLVE_PROTENIX_COMPLEX_STEP",
            strategy.get("protenix_complex_step"),
        ),
        "protenix_complex_sample": env_optional_int(
            "ASTEVOLVE_PROTENIX_COMPLEX_SAMPLE",
            strategy.get("protenix_complex_sample"),
        ),
        "protenix_complex_use_default_params": env_optional_bool(
            "ASTEVOLVE_PROTENIX_COMPLEX_USE_DEFAULT_PARAMS",
            strategy.get("protenix_complex_use_default_params"),
        ),
        "protenix_complex_timeout": env_optional_int(
            "ASTEVOLVE_PROTENIX_TIMEOUT",
            strategy.get("protenix_complex_timeout"),
        ),
        "af3_model_dir": env_str("ASTEVOLVE_AF3_MODEL_DIR", strategy.get("af3_model_dir")),
        "af3_conda_env": resolve_alphafold3_conda_env(strategy.get("af3_conda_env")),
        "af3_run_data_pipeline": env_bool("ASTEVOLVE_AF3_RUN_DATA_PIPELINE", strategy.get("af3_run_data_pipeline", False)),
        "af3_db_dir": env_str("ASTEVOLVE_AF3_DB_DIR", strategy.get("af3_db_dir")),
        "af3_num_recycles": env_int("ASTEVOLVE_AF3_NUM_RECYCLES", strategy.get("af3_num_recycles", 10)),
        "af3_num_diffusion_samples": env_int("ASTEVOLVE_AF3_NUM_DIFFUSION_SAMPLES", strategy.get("af3_num_diffusion_samples", 1)),
        "af3_timeout": env_int("ASTEVOLVE_AF3_TIMEOUT", strategy.get("af3_timeout", 7200)),
        "af3_flash_attention_implementation": str(env_str("ASTEVOLVE_AF3_FLASH_ATTENTION", strategy.get("af3_flash_attention_implementation", "triton"))),
        "af3_gpu_device": env_int("ASTEVOLVE_AF3_GPU_DEVICE", strategy.get("af3_gpu_device", 0)),
        "af3_seed": env_int("ASTEVOLVE_AF3_SEED", strategy.get("af3_seed", 202)),
        "structure_multiseed_enabled": env_strict_bool("ASTEVOLVE_STRUCTURE_MULTISEED_ENABLED", strategy.get("structure_multiseed_enabled", False)),
        "structure_formal_funnel_enabled": env_strict_bool("ASTEVOLVE_STRUCTURE_FORMAL_FUNNEL_ENABLED", strategy.get("structure_formal_funnel_enabled", False)),
        "structure_protenix_seeds": env_int_list("ASTEVOLVE_STRUCTURE_PROTENIX_SEEDS", strategy.get("structure_protenix_seeds", [101])),
        "structure_af3_seeds": env_int_list("ASTEVOLVE_STRUCTURE_AF3_SEEDS", strategy.get("structure_af3_seeds", [202])),
        "structure_robust_top_candidates": env_int("ASTEVOLVE_STRUCTURE_ROBUST_TOP_CANDIDATES", strategy.get("structure_robust_top_candidates", 2)),
        "structure_disagreement_threshold": env_float("ASTEVOLVE_STRUCTURE_DISAGREEMENT_THRESHOLD", strategy.get("structure_disagreement_threshold", 0.05)),
        "structure_pyrosetta_required": env_strict_bool("ASTEVOLVE_STRUCTURE_PYROSETTA_REQUIRED", strategy.get("structure_pyrosetta_required", False)),
        "structure_model": str(env_str("ASTEVOLVE_STRUCTURE_MODEL", strategy.get("structure_model", "protenix"))),
        "structure_model_name": env_str("ASTEVOLVE_STRUCTURE_MODEL_NAME", strategy.get("structure_model_name", strategy.get("esmfold2_model_name"))),
        "structure_prescreen_enabled": env_bool("ASTEVOLVE_STRUCTURE_PRESCREEN_ENABLED", strategy.get("structure_prescreen_enabled", False)),
        "structure_prescreen_model": str(env_str("ASTEVOLVE_STRUCTURE_PRESCREEN_MODEL", strategy.get("structure_prescreen_model", "esmfold2"))),
        "structure_prescreen_model_name": env_str("ASTEVOLVE_STRUCTURE_PRESCREEN_MODEL_NAME", strategy.get("structure_prescreen_model_name")),
        "structure_prescreen_top_frac": env_float("ASTEVOLVE_STRUCTURE_PRESCREEN_TOP_FRAC", strategy.get("structure_prescreen_top_frac", 1.0)),
        "structure_prescreen_min_candidates": env_int("ASTEVOLVE_STRUCTURE_PRESCREEN_MIN_CANDIDATES", strategy.get("structure_prescreen_min_candidates", 1)),
        "structure_prescreen_max_candidates": env_int("ASTEVOLVE_STRUCTURE_PRESCREEN_MAX_CANDIDATES", strategy.get("structure_prescreen_max_candidates", 0)),
        "structure_prescreen_forward_all_to_screen": env_bool(
            "ASTEVOLVE_STRUCTURE_PRESCREEN_FORWARD_ALL_TO_SCREEN",
            strategy.get("structure_prescreen_forward_all_to_screen", False),
        ),
        "structure_screen_model": str(env_str("ASTEVOLVE_STRUCTURE_SCREEN_MODEL", strategy.get("structure_screen_model", "esmfold2"))),
        "structure_screen_model_name": env_str("ASTEVOLVE_STRUCTURE_SCREEN_MODEL_NAME", strategy.get("structure_screen_model_name")),
        "structure_screen_enabled": env_bool("ASTEVOLVE_STRUCTURE_SCREEN_ENABLED", strategy.get("structure_screen_enabled", False)),
        "structure_screen_all_candidates": env_bool("ASTEVOLVE_STRUCTURE_SCREEN_ALL_CANDIDATES", strategy.get("structure_screen_all_candidates", True)),
        "structure_screen_top_frac": env_float("ASTEVOLVE_STRUCTURE_SCREEN_TOP_FRAC", strategy.get("structure_screen_top_frac", 1.0)),
        "structure_screen_min_candidates": env_int("ASTEVOLVE_STRUCTURE_SCREEN_MIN_CANDIDATES", strategy.get("structure_screen_min_candidates", 1)),
        "structure_screen_max_candidates": env_int("ASTEVOLVE_STRUCTURE_SCREEN_MAX_CANDIDATES", strategy.get("structure_screen_max_candidates", 0)),
        "structure_screen_progen_batch_size": env_int("ASTEVOLVE_STRUCTURE_SCREEN_PROGEN_BATCH_SIZE", strategy.get("structure_screen_progen_batch_size", 0)),
        "structure_rerank_model": str(env_str("ASTEVOLVE_STRUCTURE_RERANK_MODEL", strategy.get("structure_rerank_model", "protenix"))),
        "structure_rerank_model_name": env_str("ASTEVOLVE_STRUCTURE_RERANK_MODEL_NAME", strategy.get("structure_rerank_model_name")),
        "structure_rerank_enabled": env_bool("ASTEVOLVE_STRUCTURE_RERANK_ENABLED", strategy.get("structure_rerank_enabled", True)),
        "structure_rerank_top_frac": env_float("ASTEVOLVE_STRUCTURE_RERANK_TOP_FRAC", strategy.get("structure_rerank_top_frac", 0.25)),
        "structure_rerank_min_candidates": env_int("ASTEVOLVE_STRUCTURE_RERANK_MIN_CANDIDATES", strategy.get("structure_rerank_min_candidates", 1)),
        "structure_rerank_max_candidates": env_int("ASTEVOLVE_STRUCTURE_RERANK_MAX_CANDIDATES", strategy.get("structure_rerank_max_candidates", 2)),
        "structure_rerank_all_infeasible_rescue": env_bool("ASTEVOLVE_STRUCTURE_RERANK_ALL_INFEASIBLE_RESCUE", strategy.get("structure_rerank_all_infeasible_rescue", False)),
        "structure_physics_max_candidates": env_int("ASTEVOLVE_STRUCTURE_PHYSICS_MAX_CANDIDATES", strategy.get("structure_physics_max_candidates", 0)),
        "structure_shortlist_policy": str(env_str("ASTEVOLVE_STRUCTURE_SHORTLIST_POLICY", strategy.get("structure_shortlist_policy", "legacy_diverse"))),
        "structure_screen_single_node_diagnostic_quota": env_int("ASTEVOLVE_STRUCTURE_SCREEN_SINGLE_NODE_DIAGNOSTIC_QUOTA", strategy.get("structure_screen_single_node_diagnostic_quota", 0)),
        "structure_position_distribution_engagement_quota": env_int("ASTEVOLVE_STRUCTURE_POSITION_DISTRIBUTION_ENGAGEMENT_QUOTA", strategy.get("structure_position_distribution_engagement_quota", 0)),
        "structure_portfolio_contract_quota": env_int("ASTEVOLVE_STRUCTURE_PORTFOLIO_CONTRACT_QUOTA", strategy.get("structure_portfolio_contract_quota", 0)),
        "structure_selection_objective": str(env_str("ASTEVOLVE_STRUCTURE_SELECTION_OBJECTIVE", strategy.get("structure_selection_objective", "legacy_additive"))),
        "structure_stepping_stone_enabled": env_bool("ASTEVOLVE_STRUCTURE_STEPPING_STONE_ENABLED", strategy.get("structure_stepping_stone_enabled", False)),
        "structure_stepping_stone_max_energy_degradation": env_float("ASTEVOLVE_STRUCTURE_STEPPING_STONE_MAX_ENERGY_DEGRADATION", strategy.get("structure_stepping_stone_max_energy_degradation", 0.0)),
        "structure_stepping_stone_metrics": as_string_list(strategy.get("structure_stepping_stone_metrics", [])),
        "structure_stepping_stone_min_metric_gain": env_float("ASTEVOLVE_STRUCTURE_STEPPING_STONE_MIN_METRIC_GAIN", strategy.get("structure_stepping_stone_min_metric_gain", 0.0)),
        "structure_allow_low_fidelity_fallback": env_bool("ASTEVOLVE_STRUCTURE_ALLOW_LOW_FIDELITY_FALLBACK", strategy.get("structure_allow_low_fidelity_fallback", True)),
        "structure_batch_size": env_int("ASTEVOLVE_STRUCTURE_BATCH_SIZE", strategy.get("structure_batch_size", 0)),
        "structure_parallel_workers": env_int("ASTEVOLVE_STRUCTURE_PARALLEL_WORKERS", strategy.get("structure_parallel_workers", 1)),


        "structure_service_url": env_str("ASTEVOLVE_STRUCTURE_SERVICE_URL", None),
        "structure_service_backend": str(env_str("ASTEVOLVE_STRUCTURE_SERVICE_BACKEND", strategy.get("structure_service_backend", "esmfold2"))),
        "structure_service_token": env_str("ASTEVOLVE_STRUCTURE_SERVICE_TOKEN", None),
        "structure_service_timeout": env_int("ASTEVOLVE_STRUCTURE_SERVICE_TIMEOUT", 7200),
        "esmfold2_mode": str(strategy.get("esmfold2_mode", "local")),
        "esmfold2_conda_env": strategy.get("esmfold2_conda_env"),
        "esmfold2_num_loops": int(strategy.get("esmfold2_num_loops", 3)),
        "esmfold2_num_sampling_steps": int(strategy.get("esmfold2_num_sampling_steps", 32)),
        "esmfold2_num_diffusion_samples": int(strategy.get("esmfold2_num_diffusion_samples", 1)),
        "multistate_objectives_enabled": bool(strategy.get("multistate_objectives_enabled", True)),
        "multistate_objective_weight": float(strategy.get("multistate_objective_weight", 1.0)),
        "mutation_ops": mutation_ops,
        "history_size": int(strategy.get("history_size", 50)),
        "portfolio_seed_refinement_rounds": env_int("ASTEVOLVE_PORTFOLIO_SEED_REFINEMENT_ROUNDS", strategy.get("portfolio_seed_refinement_rounds", 0)),
        "candidate_wave_enabled": env_strict_bool(
            "ASTEVOLVE_CANDIDATE_WAVE_ENABLED",
            strategy.get("candidate_wave_enabled", False),
        ),
        "candidate_wave_size": env_strict_int(
            "ASTEVOLVE_CANDIDATE_WAVE_SIZE",
            strategy.get("candidate_wave_size", 8),
        ),
        "candidate_wave_fail_on_underfill": env_strict_bool(
            "ASTEVOLVE_CANDIDATE_WAVE_FAIL_ON_UNDERFILL",
            strategy.get("candidate_wave_fail_on_underfill", True),
        ),
        "candidate_wave_protenix_mutant_quota": env_strict_int(
            "ASTEVOLVE_CANDIDATE_WAVE_PROTENIX_MUTANT_QUOTA",
            strategy.get("candidate_wave_protenix_mutant_quota", 8),
        ),
        "candidate_wave_af3_mutant_quota": env_strict_int(
            "ASTEVOLVE_CANDIDATE_WAVE_AF3_MUTANT_QUOTA",
            strategy.get("candidate_wave_af3_mutant_quota", 4),
        ),
        "candidate_wave_changed_node_min_generated_unique": env_strict_int(
            "ASTEVOLVE_CANDIDATE_WAVE_CHANGED_NODE_MIN_GENERATED_UNIQUE",
            strategy.get(
                "candidate_wave_changed_node_min_generated_unique", 2
            ),
        ),
        "candidate_wave_changed_node_min_frozen_unique": env_strict_int(
            "ASTEVOLVE_CANDIDATE_WAVE_CHANGED_NODE_MIN_FROZEN_UNIQUE",
            strategy.get("candidate_wave_changed_node_min_frozen_unique", 2),
        ),
        "candidate_wave_changed_node_min_protenix_attempts": env_strict_int(
            "ASTEVOLVE_CANDIDATE_WAVE_CHANGED_NODE_MIN_PROTENIX_ATTEMPTS",
            strategy.get(
                "candidate_wave_changed_node_min_protenix_attempts", 1
            ),
        ),
        "executable_island_policy_enabled": env_strict_bool(
            "ASTEVOLVE_EXECUTABLE_ISLAND_POLICY_ENABLED",
            strategy.get("executable_island_policy_enabled", False),
        ),
        "search_method": search_method,
        "mcts_c_puct": float(strategy.get("mcts_c_puct", 1.4)),
        "mcts_max_depth": int(strategy.get("mcts_max_depth", 4)),
        "mcts_reward_scale": float(strategy.get("mcts_reward_scale", 1.0)),
        "mcts_iteration_unit": str(env_str("ASTEVOLVE_MCTS_ITERATION_UNIT", strategy.get("mcts_iteration_unit", "expansion_rounds"))),
        "mcts_candidate_budget_max_round_multiplier": env_int("ASTEVOLVE_MCTS_CANDIDATE_BUDGET_MAX_ROUND_MULTIPLIER", strategy.get("mcts_candidate_budget_max_round_multiplier", 4)),
        "mcts_candidate_budget_fail_on_underfill": env_bool("ASTEVOLVE_MCTS_CANDIDATE_BUDGET_FAIL_ON_UNDERFILL", strategy.get("mcts_candidate_budget_fail_on_underfill", True)),
        "mcts_output_dir": str(mcts_output_dir),
        "mcts_save_tree": bool(strategy.get("mcts_save_tree", True)),
        "mcts_save_variants": bool(strategy.get("mcts_save_variants", True)),
        "mcts_tree_quality_required": bool(
            strategy.get("mcts_tree_quality_required", False)
        ),
        "mcts_tree_min_root_children": int(
            strategy.get("mcts_tree_min_root_children", 0)
        ),
        "mcts_tree_min_branching_nodes": int(
            strategy.get("mcts_tree_min_branching_nodes", 0)
        ),
        "mcts_tree_min_leaves": int(
            strategy.get("mcts_tree_min_leaves", 0)
        ),
        "mcts_tree_min_max_depth": int(
            strategy.get("mcts_tree_min_max_depth", 0)
        ),
        "node_edit_policies": node_edit_policies,
        "residue_mutation_contract": normalize_residue_mutation_contract(
            strategy.get("residue_mutation_contract", {})
        ),
        "proposal_engine": str(strategy.get("proposal_engine", "contract_guided")),
        "sequence_generator_id": str(
            strategy.get("sequence_generator_id", CONSTRAINT_AWARE_GENERATOR_ID)
        ),
        "sequence_generator_structure_condition_refs": as_string_list(
            strategy.get("sequence_generator_structure_condition_refs", [])
        ),
        "sequence_generator_state_condition_refs": as_string_list(
            strategy.get("sequence_generator_state_condition_refs", [])
        ),
        "node_optimizer_enabled": env_bool(
            "ASTEVOLVE_NODE_OPTIMIZER_ENABLED",
            strategy.get("node_optimizer_enabled", False),
        ),
        "node_optimizer_candidate_count": env_int(
            "ASTEVOLVE_NODE_OPTIMIZER_CANDIDATES",
            strategy.get("node_optimizer_candidate_count", 8),
        ),
        "node_optimizer_beam_width": env_int(
            "ASTEVOLVE_NODE_OPTIMIZER_BEAM_WIDTH",
            strategy.get("node_optimizer_beam_width", 16),
        ),
        "node_optimizer_top_k_per_position": env_int(
            "ASTEVOLVE_NODE_OPTIMIZER_TOP_K",
            strategy.get("node_optimizer_top_k_per_position", 4),
        ),
        "node_optimizer_temperature": env_float(
            "ASTEVOLVE_NODE_OPTIMIZER_TEMPERATURE",
            strategy.get("node_optimizer_temperature", 0.8),
        ),
        "node_optimizer_diversity_weight": env_float(
            "ASTEVOLVE_NODE_OPTIMIZER_DIVERSITY_WEIGHT",
            strategy.get("node_optimizer_diversity_weight", 0.15),
        ),
        "node_optimizer_mutation_penalty": env_float(
            "ASTEVOLVE_NODE_OPTIMIZER_MUTATION_PENALTY",
            strategy.get("node_optimizer_mutation_penalty", 0.25),
        ),
        "node_optimizer_prior_model": str(
            env_str(
                "ASTEVOLVE_NODE_OPTIMIZER_PRIOR_MODEL",
                strategy.get("node_optimizer_prior_model", "heuristic"),
            )
        ),


        "node_optimizer_model_path": env_str(
            "ASTEVOLVE_MASKED_LM_MODEL_DIR", None
        ),
        "node_optimizer_device": str(
            env_str("ASTEVOLVE_NODE_OPTIMIZER_DEVICE", "cuda")
        ),
        "mcts_progressive_widening_c": env_float(
            "ASTEVOLVE_MCTS_PROGRESSIVE_WIDENING_C",
            strategy.get("mcts_progressive_widening_c", 2.0),
        ),
        "mcts_progressive_widening_alpha": env_float(
            "ASTEVOLVE_MCTS_PROGRESSIVE_WIDENING_ALPHA",
            strategy.get("mcts_progressive_widening_alpha", 0.5),
        ),
        "proposal_tier_mode": str(
            strategy.get("proposal_tier_mode", "fixed_node")
        ).strip().lower(),
        "proposal_exploit_frac": float(strategy.get("proposal_exploit_frac", 0.70)),
        "proposal_explore_frac": float(strategy.get("proposal_explore_frac", 0.20)),
        "proposal_repair_frac": float(strategy.get("proposal_repair_frac", 0.10)),
        "exploit_max_mutations": int(strategy.get("exploit_max_mutations", 4)),
        "explore_max_mutations": int(strategy.get("explore_max_mutations", 8)),
        "repair_max_mutations": int(strategy.get("repair_max_mutations", 2)),
        "max_total_mutations": int(strategy.get("max_total_mutations", 12)),
        "fast_filter_enabled": bool(strategy.get("fast_filter_enabled", True)),
        "sequence_prefilter_callable": strategy.get(
            "sequence_prefilter_callable"
        ),
        "sequence_prefilter_config": dict(
            strategy.get("sequence_prefilter_config") or {}
        ),
        "sequence_bootstrap_callable": strategy.get(
            "sequence_bootstrap_callable"
        ),
        "sequence_bootstrap_config": dict(
            strategy.get("sequence_bootstrap_config") or {}
        ),
        "semantic_required_nodes": as_string_list(strategy.get("semantic_required_nodes", [])),
        "semantic_active_nodes": as_string_list(strategy.get("semantic_active_nodes", [])),
        "semantic_anchor_nodes": as_string_list(strategy.get("semantic_anchor_nodes", [])),
        "semantic_required_node_min_visits": int(strategy.get("semantic_required_node_min_visits", 1)),
        "semantic_required_node_min_mutations": int(strategy.get("semantic_required_node_min_mutations", 1)),
        "semantic_coverage_mode": str(strategy.get("semantic_coverage_mode", "soft")),
        "semantic_missing_node_penalty": float(strategy.get("semantic_missing_node_penalty", 250.0)),
        "semantic_required_node_round_robin": bool(strategy.get("semantic_required_node_round_robin", True)),
        "semantic_required_node_force_steps": env_int(
            "ASTEVOLVE_SEMANTIC_REQUIRED_NODE_FORCE_STEPS",
            strategy.get("semantic_required_node_force_steps", 0),
        ),
        "outer_loop_phase": str(strategy.get("outer_loop_phase", "explore_ast")),
        "search_schedule": dict(strategy.get("search_schedule", {})) if isinstance(strategy.get("search_schedule"), dict) else {},
    }
    validate_candidate_wave_config(
        {
            field_name: config[field_name]
            for field_name in CANDIDATE_WAVE_CONFIG_FIELDS
        }
    )
    return config


def build_score_config(strategy: Dict[str, Any]) -> Dict[str, Any]:


    score = dict(strategy.get("score_config", {}))
    out = {
        "weight_fast": float(score.get("weight_fast", 1.0)),
        "weight_plddt": float(score.get("weight_plddt", 5.0)),
        "weight_iptm": float(score.get("weight_iptm", 1.0)),
        "weight_ptm": float(score.get("weight_ptm", 0.5)),
        "weight_ranking_score": float(score.get("weight_ranking_score", 0.0)),
        "weight_interface_plddt": float(score.get("weight_interface_plddt", 1.0)),
        "weight_node_plddt_min": float(score.get("weight_node_plddt_min", 0.5)),
        "weight_clash": float(score.get("weight_clash", 1.0)),
        "weight_multistate": float(score.get("weight_multistate", 1.0)),
        "weight_evaluator": float(score.get("weight_evaluator", 1.0)),
        "plddt_scale": float(score.get("plddt_scale", 100.0)),
        "clash_scale": float(score.get("clash_scale", 10.0)),
        "fast_loss_nonneg": bool(score.get("fast_loss_nonneg", True)),
    }
    evaluator_weights = dict(score.get("evaluator_weights") or {}) if isinstance(score.get("evaluator_weights"), dict) else {}
    for key, value in score.items():
        if str(key).startswith("eval_"):
            try:
                evaluator_weights[str(key)] = float(value)
            except (TypeError, ValueError):
                evaluator_weights[str(key)] = value
    if evaluator_weights:
        out["evaluator_weights"] = evaluator_weights
    if isinstance(score.get("evaluator_backends"), dict):
        out["evaluator_backends"] = dict(score["evaluator_backends"])
    if "weight_eval_contract_response" in score:
        out["weight_eval_contract_response"] = float(score.get("weight_eval_contract_response", 0.2))
    for field, default, converter in (
        ("inner_evaluator_loss_weight", 1.0, float),
        ("inner_hard_gate_fail_penalty", 1000.0, float),
        ("hard_gate_failure_score_scale", 0.25, float),
        ("hard_gate_disqualified_score", 0.0, float),
    ):
        if field in score:
            out[field] = converter(score.get(field, default))

    plugin_declared = "evaluator_plugins" in score or "plugin_config" in score
    resolved_plugins: List[str] = []
    if plugin_declared:
        requested, resolved_plugins = normalize_plugin_requests(
            score.get("evaluator_plugins")
        )
        if "plugin_config" in score and not resolved_plugins:
            raise PluginConfigError(
                "score_config.plugin_config requires an explicit evaluator_plugins request"
            )
        plugin_config = normalize_plugin_config(
            score.get("plugin_config"), requested_plugins=resolved_plugins
        )
        missing_request = sorted(set(plugin_config) - set(resolved_plugins))
        if missing_request:
            raise PluginConfigError(
                "score_config.plugin_config contains unrequested plugin(s): "
                + ", ".join(missing_request)
            )
        registered_runtime_fields = {
            field.runtime_key
            for spec in PLUGIN_REGISTRY.values()
            for field in spec.config_fields.values()
        }
        legacy_plugin_fields = sorted(
            str(key) for key in score if str(key) in registered_runtime_fields
        )
        if legacy_plugin_fields:
            raise PluginConfigError(
                "Plugin fields must use score_config.plugin_config.<plugin_name>: "
                + ", ".join(legacy_plugin_fields)
            )
        registered_plugin_weights = {
            weight
            for spec in PLUGIN_REGISTRY.values()
            for weight in spec.weight_fields
        }
        plugin_weight_namespaces = {
            token
            for name, spec in PLUGIN_REGISTRY.items()
            for token in (name, name.split("_", 1)[0], *spec.aliases)
            if token
        }
        unregistered_plugin_weights = sorted(
            {
                str(key)
                for key in (*score.keys(), *evaluator_weights.keys())
                if str(key).startswith("eval_")
                and str(key) not in registered_plugin_weights
                and any(
                    str(key).startswith(f"eval_{namespace}_")
                    for namespace in plugin_weight_namespaces
                )
            }
        )
        if unregistered_plugin_weights:
            raise PluginConfigError(
                "Unregistered plugin evaluator weights must use a registered "
                "score_config.plugin_config.<plugin_name>.evaluator_weights field: "
                + ", ".join(unregistered_plugin_weights)
            )
        legacy_plugin_weights = sorted(
            {
                str(key)
                for key in score
                if str(key) in registered_plugin_weights
            }
            | {
                str(key)
                for key in evaluator_weights
                if str(key) in registered_plugin_weights
            }
        )
        if legacy_plugin_weights:
            raise PluginConfigError(
                "Plugin evaluator weights must use "
                "score_config.plugin_config.<plugin_name>.evaluator_weights: "
                + ", ".join(legacy_plugin_weights)
            )
        out["evaluator_plugins"] = resolved_plugins
        out["plugin_config"] = plugin_config
        out["plugin_resolution"] = {
            "schema_version": PLUGIN_RESOLUTION_VERSION,
            "source": "score_config.evaluator_plugins",
            "requested": requested,
            "resolved": resolved_plugins,
            "unknown": [],
            "strict": True,
        }
    return out


def resolve_graph_ablation_mode(strategy: Dict[str, Any]) -> str:


    raw = os.environ.get("ASTEVOLVE_GRAPH_ABLATION_MODE") or strategy.get("graph_ablation_mode") or "full"
    mode = str(raw).strip().lower()
    aliases = {
        "none": "full",
        "disabled": "no_semantic_graph",
        "no_graph": "no_semantic_graph",
        "structural": "structural_only",
        "no_contract": "no_edit_contract",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"full", "no_semantic_graph", "structural_only", "no_edit_contract"}:
        mode = "full"
    return mode
