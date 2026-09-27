

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional

from astevolve.core.amino_acids import AA
from astevolve.domain.compiled import (
    CompiledPositionDistribution,
    compiled_position_distribution_set_hash,
)
from astevolve.runtime.conda import (
    resolve_alphafold3_conda_env,
    resolve_protenix_conda_env,
)
from astevolve.search.operator_registry import default_operator_weights
from astevolve.search.sequence_generator import CONSTRAINT_AWARE_GENERATOR_ID


def normalize_residue_mutation_contract(
    value: Any,
) -> Dict[str, Dict[int, List[str]]]:


    if not isinstance(value, Mapping):
        raise ValueError("residue_mutation_contract must be a mapping")
    normalized: Dict[str, Dict[int, List[str]]] = {}
    for raw_chain, raw_positions in value.items():
        if not isinstance(raw_chain, str):
            raise ValueError(
                "residue_mutation_contract chain identifiers must be strings"
            )
        chain = raw_chain.strip()
        if not chain:
            raise ValueError(
                "residue_mutation_contract chain identifiers must be non-empty"
            )
        if chain in normalized:
            raise ValueError(
                f"residue_mutation_contract contains duplicate chain {chain!r}"
            )
        if not isinstance(raw_positions, Mapping) or not raw_positions:
            raise ValueError(
                "residue_mutation_contract chain entries must be non-empty mappings: "
                f"{chain!r}"
            )

        positions: Dict[int, List[str]] = {}
        for raw_position, raw_residues in raw_positions.items():
            if isinstance(raw_position, bool):
                raise ValueError(
                    "residue_mutation_contract positions must be non-negative "
                    f"integers: {chain}:{raw_position!r}"
                )
            if isinstance(raw_position, int):
                position = raw_position
            elif isinstance(raw_position, str):
                token = raw_position.strip()
                if not token.isdecimal() or str(int(token)) != token:
                    raise ValueError(
                        "residue_mutation_contract positions must be canonical "
                        f"non-negative integers: {chain}:{raw_position!r}"
                    )
                position = int(token)
            else:
                raise ValueError(
                    "residue_mutation_contract positions must be non-negative "
                    f"integers: {chain}:{raw_position!r}"
                )
            if position < 0:
                raise ValueError(
                    "residue_mutation_contract positions must be non-negative: "
                    f"{chain}:{position}"
                )
            if position in positions:
                raise ValueError(
                    "residue_mutation_contract contains duplicate position "
                    f"{chain}:{position}"
                )
            if (
                isinstance(raw_residues, (str, bytes))
                or not isinstance(raw_residues, Sequence)
                or not raw_residues
            ):
                raise ValueError(
                    "residue_mutation_contract residue lists must be non-empty: "
                    f"{chain}:{position}"
                )
            residue_set = set()
            for raw_residue in raw_residues:
                if not isinstance(raw_residue, str):
                    raise ValueError(
                        "residue_mutation_contract residues must be strings: "
                        f"{chain}:{position}:{raw_residue!r}"
                    )
                residue = raw_residue.strip().upper()
                if len(residue) != 1 or residue not in AA:
                    raise ValueError(
                        "residue_mutation_contract residues must be standard amino "
                        f"acids: {chain}:{position}:{raw_residue!r}"
                    )
                residue_set.add(residue)
            positions[position] = [residue for residue in AA if residue in residue_set]
        normalized[chain] = dict(sorted(positions.items()))
    return dict(sorted(normalized.items()))


def normalize_compiled_position_distribution_policy(value: Any) -> Dict[str, Any]:


    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("compiled_position_distribution_policy must be a mapping")
    expected = {
        "schema_version",
        "compiled_design_action_hash",
        "distribution_set_hash",
        "rows",
    }
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        raise ValueError(
            "compiled_position_distribution_policy fields mismatch: "
            f"unknown={sorted(unknown)} missing={sorted(missing)}"
        )
    if value["schema_version"] != (
        "astevolve.compiled_position_distribution_policy.v1"
    ):
        raise ValueError("compiled_position_distribution_policy schema_version invalid")
    action_hash = str(value["compiled_design_action_hash"] or "")
    if not action_hash.startswith("compiled_design_action_sha256:"):
        raise ValueError("compiled_position_distribution_policy action hash invalid")
    rows = value["rows"]
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("compiled_position_distribution_policy rows must be non-empty")
    compiled_rows = tuple(
        CompiledPositionDistribution.from_mapping(row) for row in rows
    )
    computed_set_hash = compiled_position_distribution_set_hash(compiled_rows)
    if value["distribution_set_hash"] != computed_set_hash:
        raise ValueError("compiled_position_distribution_policy set hash mismatch")
    return {
        "schema_version": "astevolve.compiled_position_distribution_policy.v1",
        "compiled_design_action_hash": action_hash,
        "distribution_set_hash": computed_set_hash,
        "rows": [item.to_dict() for item in compiled_rows],
    }


def normalize_compiled_portfolio_request_policy(value: Any) -> Dict[str, Any]:


    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("compiled_portfolio_request_policy must be a mapping")
    expected = {
        "schema_version",
        "compiled_portfolio_request_hash",
        "candidate_slot_hashes",
    }
    if set(value) != expected:
        raise ValueError("compiled_portfolio_request_policy fields mismatch")
    if value["schema_version"] != "astevolve.compiled_portfolio_request_policy.v1":
        raise ValueError("compiled_portfolio_request_policy schema_version invalid")
    request_hash = str(value["compiled_portfolio_request_hash"] or "")
    if not request_hash.startswith("compiled_portfolio_request_sha256:"):
        raise ValueError("compiled_portfolio_request_policy request hash invalid")
    raw_slots = value["candidate_slot_hashes"]
    if isinstance(raw_slots, (str, bytes)) or not isinstance(raw_slots, Sequence):
        raise ValueError("compiled_portfolio_request_policy slots invalid")
    slots = [str(item or "") for item in raw_slots]
    if (
        not slots
        or len(slots) != len(set(slots))
        or any(
            not item.startswith("compiled_candidate_slot_sha256:")
            for item in slots
        )
    ):
        raise ValueError("compiled_portfolio_request_policy slot hash invalid")
    return {
        "schema_version": "astevolve.compiled_portfolio_request_policy.v1",
        "compiled_portfolio_request_hash": request_hash,
        "candidate_slot_hashes": sorted(slots),
    }


CANDIDATE_WAVE_CONFIG_FIELDS = (
    "candidate_wave_enabled",
    "candidate_wave_size",
    "candidate_wave_fail_on_underfill",
    "candidate_wave_protenix_mutant_quota",
    "candidate_wave_af3_mutant_quota",
    "candidate_wave_changed_node_min_generated_unique",
    "candidate_wave_changed_node_min_frozen_unique",
    "candidate_wave_changed_node_min_protenix_attempts",
)


def validate_candidate_wave_config(value: Mapping[str, Any]) -> None:


    missing = set(CANDIDATE_WAVE_CONFIG_FIELDS) - set(value)
    if missing:
        raise ValueError(
            "candidate wave config fields missing: " + ", ".join(sorted(missing))
        )
    enabled = value["candidate_wave_enabled"]
    fail_on_underfill = value["candidate_wave_fail_on_underfill"]
    if not isinstance(enabled, bool):
        raise ValueError("candidate_wave_enabled must be a boolean")
    if not isinstance(fail_on_underfill, bool):
        raise ValueError("candidate_wave_fail_on_underfill must be a boolean")

    integer_fields = (
        CANDIDATE_WAVE_CONFIG_FIELDS[1:2]
        + CANDIDATE_WAVE_CONFIG_FIELDS[3:]
    )
    integers: Dict[str, int] = {}
    for field_name in integer_fields:
        raw = value[field_name]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"{field_name} must be an integer")
        integers[field_name] = int(raw)

    size = integers["candidate_wave_size"]
    protenix_quota = integers["candidate_wave_protenix_mutant_quota"]
    af3_quota = integers["candidate_wave_af3_mutant_quota"]
    generated_min = integers[
        "candidate_wave_changed_node_min_generated_unique"
    ]
    frozen_min = integers["candidate_wave_changed_node_min_frozen_unique"]
    provider_min = integers[
        "candidate_wave_changed_node_min_protenix_attempts"
    ]
    if size < 1:
        raise ValueError("candidate_wave_size must be positive")
    if protenix_quota < 0 or protenix_quota > size:
        raise ValueError(
            "candidate_wave_protenix_mutant_quota must be between zero and "
            "candidate_wave_size"
        )
    if af3_quota < 0 or af3_quota > protenix_quota:
        raise ValueError(
            "candidate_wave_af3_mutant_quota must be between zero and the "
            "Protenix mutant quota"
        )
    if generated_min < 0 or generated_min > size:
        raise ValueError(
            "candidate_wave_changed_node_min_generated_unique must be between "
            "zero and candidate_wave_size"
        )
    if frozen_min < 0 or frozen_min > generated_min:
        raise ValueError(
            "candidate_wave_changed_node_min_frozen_unique must be between zero "
            "and the generated-unique minimum"
        )
    if provider_min < 0 or provider_min > frozen_min:
        raise ValueError(
            "candidate_wave_changed_node_min_protenix_attempts must be between "
            "zero and the frozen-unique minimum"
        )

    if not enabled:
        return
    if size != 8 or protenix_quota != 8 or af3_quota != 4:
        raise ValueError(
            "enabled candidate wave requires fixed size/Protenix/AF3 mutant "
            "quotas of 8/8/4"
        )
    if not fail_on_underfill:
        raise ValueError(
            "enabled candidate wave requires candidate_wave_fail_on_underfill=true"
        )
    if generated_min < 2 or frozen_min < 2 or provider_min < 1:
        raise ValueError(
            "enabled candidate wave requires changed-node minima generated/frozen/"
            "Protenix of at least 2/2/1"
        )


@dataclass
class SAConfig:


    iterations: int = 1200
    init_temp: float = 2.0
    cooling: float = 0.995
    mutation_rate: float = 0.03
    resample_segment_prob: float = 0.05
    seed: Optional[int] = None


    progen_weight: float = 1.0
    progen_chains: Optional[List[str]] = None
    progen_reduce: str = "length_weighted"
    sequence_prior_model: str = "progen"


    inner_structure_enabled: bool = False
    inner_structure_model: str = "esmfold2"
    inner_structure_model_name: Optional[str] = None
    inner_structure_weight: float = 1.0
    inner_structure_fail_closed: bool = True
    inner_structure_hard_gate: bool = True
    inner_structure_failure_penalty: float = 1000.0


    promote_inline_winner_structure_evidence: bool = False

    inner_esmfold2_enabled: bool = False
    inner_esmfold2_model_name: Optional[str] = None
    inner_esmfold2_interval: int = 10
    inner_esmfold2_weight: float = 1.0


    chai1_enabled: bool = True
    chai1_top_frac: float = 0.01
    chai1_min_candidates: int = 1
    chai1_max_candidates: int = 5


    chai1_num_trunk_recycles: int = 3
    chai1_num_diffn_timesteps: int = 50
    chai1_use_esm_embeddings: bool = True


    protenix_model_name: str = "protenix_mini_esm_v0.5.0"
    protenix_conda_env: str = field(default_factory=resolve_protenix_conda_env)
    protenix_seed: int = 101
    protenix_complex_use_msa: Optional[bool] = None
    protenix_complex_cycle: Optional[int] = None
    protenix_complex_step: Optional[int] = None
    protenix_complex_sample: Optional[int] = None
    protenix_complex_use_default_params: Optional[bool] = None
    protenix_complex_timeout: Optional[int] = None


    af3_model_dir: Optional[str] = None
    af3_conda_env: str = field(default_factory=resolve_alphafold3_conda_env)
    af3_run_data_pipeline: bool = False
    af3_db_dir: Optional[str] = None
    af3_num_recycles: int = 10
    af3_num_diffusion_samples: int = 1
    af3_timeout: int = 7200
    af3_flash_attention_implementation: str = "triton"
    af3_gpu_device: int = 0


    af3_seed: int = 202
    structure_multiseed_enabled: bool = False
    structure_formal_funnel_enabled: bool = False
    structure_protenix_seeds: List[int] = field(default_factory=lambda: [101])
    structure_af3_seeds: List[int] = field(default_factory=lambda: [202])
    structure_robust_top_candidates: int = 2
    structure_disagreement_threshold: float = 0.05
    structure_pyrosetta_required: bool = False


    structure_model: str = "protenix"
    structure_model_name: Optional[str] = None


    structure_prescreen_enabled: bool = False
    structure_prescreen_model: str = "esmfold2"
    structure_prescreen_model_name: Optional[str] = None
    structure_prescreen_top_frac: float = 1.0
    structure_prescreen_min_candidates: int = 1
    structure_prescreen_max_candidates: int = 0


    structure_prescreen_forward_all_to_screen: bool = False
    structure_screen_model: str = "esmfold2"
    structure_screen_model_name: Optional[str] = None
    structure_screen_enabled: bool = False
    structure_screen_all_candidates: bool = True
    structure_screen_top_frac: float = 1.0
    structure_screen_min_candidates: int = 1

    structure_screen_max_candidates: int = 0


    structure_screen_progen_batch_size: int = 0
    structure_rerank_model: str = "protenix"
    structure_rerank_model_name: Optional[str] = None
    structure_rerank_enabled: bool = True
    structure_rerank_top_frac: float = 0.25
    structure_rerank_min_candidates: int = 1
    structure_rerank_max_candidates: int = 2


    structure_rerank_all_infeasible_rescue: bool = False

    structure_physics_max_candidates: int = 0


    structure_shortlist_policy: str = "legacy_diverse"


    structure_screen_single_node_diagnostic_quota: int = 0


    structure_position_distribution_engagement_quota: int = 0


    structure_portfolio_contract_quota: int = 0


    structure_selection_objective: str = "legacy_additive"
    structure_stepping_stone_enabled: bool = False
    structure_stepping_stone_max_energy_degradation: float = 0.0
    structure_stepping_stone_metrics: List[str] = field(default_factory=list)
    structure_stepping_stone_min_metric_gain: float = 0.0


    structure_allow_low_fidelity_fallback: bool = True


    structure_batch_size: int = 0
    structure_parallel_workers: int = 1
    structure_service_url: Optional[str] = None
    structure_service_backend: str = "esmfold2"
    structure_service_token: Optional[str] = None
    structure_service_timeout: int = 7200
    esmfold2_mode: str = "local"
    esmfold2_conda_env: Optional[str] = None
    esmfold2_num_loops: int = 3
    esmfold2_num_sampling_steps: int = 32
    esmfold2_num_diffusion_samples: int = 1
    multistate_objectives_enabled: bool = True
    multistate_objective_weight: float = 1.0


    mutation_ops: Dict[str, float] = field(
        default_factory=default_operator_weights
    )
    history_size: int = 50


    search_method: str = "mcts"
    mcts_c_puct: float = 1.4
    mcts_max_depth: int = 4
    mcts_reward_scale: float = 1.0


    mcts_iteration_unit: str = "expansion_rounds"
    mcts_candidate_budget_max_round_multiplier: int = 4
    mcts_candidate_budget_fail_on_underfill: bool = True

    mcts_node_sweep_enabled: bool = False
    mcts_node_sweep_count: int = 0
    mcts_node_sweep_parent_policy: str = "incumbent"


    mcts_fidelity_upgrade_enabled: bool = False
    mcts_fidelity_upgrade_provider: str = "protenix"
    mcts_fidelity_upgrade_interval: int = 20
    mcts_fidelity_upgrade_candidates: int = 4
    mcts_fidelity_upgrade_final_candidates: int = 5
    mcts_fidelity_upgrade_required: bool = False
    mcts_output_dir: str = "inner_loop"
    mcts_save_tree: bool = True
    mcts_save_variants: bool = True


    mcts_artifact_mode: str = "normalized"


    mcts_tree_quality_required: bool = False
    mcts_tree_min_root_children: int = 0
    mcts_tree_min_branching_nodes: int = 0
    mcts_tree_min_leaves: int = 0
    mcts_tree_min_max_depth: int = 0


    portfolio_seed_refinement_rounds: int = 0


    candidate_wave_enabled: bool = False
    candidate_wave_size: int = 8
    candidate_wave_fail_on_underfill: bool = True
    candidate_wave_protenix_mutant_quota: int = 8
    candidate_wave_af3_mutant_quota: int = 4
    candidate_wave_changed_node_min_generated_unique: int = 2
    candidate_wave_changed_node_min_frozen_unique: int = 2
    candidate_wave_changed_node_min_protenix_attempts: int = 1


    executable_island_policy_enabled: bool = False

    node_edit_policies: Dict[str, Dict[str, Any]] = field(default_factory=dict)


    residue_mutation_contract: Dict[str, Dict[int, List[str]]] = field(
        default_factory=dict
    )


    compiled_position_distribution_policy: Dict[str, Any] = field(
        default_factory=dict
    )
    compiled_portfolio_request_policy: Dict[str, Any] = field(
        default_factory=dict
    )


    proposal_engine: str = "contract_guided"


    sequence_generator_id: str = CONSTRAINT_AWARE_GENERATOR_ID
    sequence_generator_structure_condition_refs: List[str] = field(
        default_factory=list
    )
    sequence_generator_state_condition_refs: List[str] = field(
        default_factory=list
    )


    node_optimizer_enabled: bool = False
    node_optimizer_candidate_count: int = 8
    node_optimizer_beam_width: int = 16
    node_optimizer_top_k_per_position: int = 4
    node_optimizer_temperature: float = 0.8
    node_optimizer_diversity_weight: float = 0.15
    node_optimizer_mutation_penalty: float = 0.25
    node_optimizer_prior_model: str = "heuristic"
    node_optimizer_model_path: Optional[str] = None
    node_optimizer_device: str = "cuda"
    mcts_progressive_widening_c: float = 2.0
    mcts_progressive_widening_alpha: float = 0.5


    proposal_tier_mode: str = "fixed_node"
    proposal_exploit_frac: float = 0.70
    proposal_explore_frac: float = 0.20
    proposal_repair_frac: float = 0.10
    exploit_max_mutations: int = 4
    explore_max_mutations: int = 8
    repair_max_mutations: int = 2
    max_total_mutations: int = 12
    fast_filter_enabled: bool = True
    sequence_prefilter_callable: Optional[str] = None
    sequence_prefilter_config: Dict[str, Any] = field(default_factory=dict)
    sequence_bootstrap_callable: Optional[str] = None
    sequence_bootstrap_config: Dict[str, Any] = field(default_factory=dict)


    semantic_required_nodes: List[str] = field(default_factory=list)


    semantic_active_nodes: List[str] = field(default_factory=list)


    semantic_anchor_nodes: List[str] = field(default_factory=list)
    semantic_required_node_min_visits: int = 1
    semantic_required_node_min_mutations: int = 1
    semantic_coverage_mode: str = "soft"
    semantic_missing_node_penalty: float = 250.0
    semantic_required_node_round_robin: bool = True
    semantic_required_node_force_steps: int = 0
    outer_loop_phase: str = "explore"
    search_schedule: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.residue_mutation_contract = normalize_residue_mutation_contract(
            self.residue_mutation_contract
        )
        self.compiled_position_distribution_policy = (
            normalize_compiled_position_distribution_policy(
                self.compiled_position_distribution_policy
            )
        )
        self.compiled_portfolio_request_policy = (
            normalize_compiled_portfolio_request_policy(
                self.compiled_portfolio_request_policy
            )
        )
        validate_candidate_wave_config(
            {
                field_name: getattr(self, field_name)
                for field_name in CANDIDATE_WAVE_CONFIG_FIELDS
            }
        )
        iteration_unit = str(self.mcts_iteration_unit).strip().lower()
        if iteration_unit not in {
            "expansion_rounds",
            "evaluated_unique_candidates",
        }:
            raise ValueError(
                "mcts_iteration_unit must be 'expansion_rounds' or "
                "'evaluated_unique_candidates'"
            )
        self.mcts_iteration_unit = iteration_unit
        if int(self.mcts_candidate_budget_max_round_multiplier) < 1:
            raise ValueError(
                "mcts_candidate_budget_max_round_multiplier must be positive"
            )
        if bool(self.mcts_fidelity_upgrade_enabled):
            if str(self.search_method).strip().lower() != "mcts":
                raise ValueError("MCTS fidelity upgrades require search_method=\"mcts\"")
            if not bool(self.inner_structure_enabled):
                raise ValueError("MCTS fidelity upgrades require the inline structure evaluator")
            provider = str(self.mcts_fidelity_upgrade_provider or "").strip().lower()
            if provider not in {"protenix", "alphafold3"}:
                raise ValueError(
                    "mcts_fidelity_upgrade_provider must be protenix or alphafold3"
                )
            self.mcts_fidelity_upgrade_provider = provider
            if int(self.mcts_fidelity_upgrade_interval) < 1:
                raise ValueError("mcts_fidelity_upgrade_interval must be positive")
            if int(self.mcts_fidelity_upgrade_candidates) < 1:
                raise ValueError("mcts_fidelity_upgrade_candidates must be positive")
            if int(self.mcts_fidelity_upgrade_final_candidates) < 1:
                raise ValueError(
                    "mcts_fidelity_upgrade_final_candidates must be positive"
                )
        if bool(self.mcts_node_sweep_enabled):
            if str(self.search_method).strip().lower() != "mcts":
                raise ValueError("node sweeps require search_method=\"mcts\"")
            if iteration_unit != "evaluated_unique_candidates":
                raise ValueError("node sweeps require evaluated_unique_candidates budgeting")
            if int(self.mcts_node_sweep_count) < 1:
                raise ValueError("mcts_node_sweep_count must be positive")
            parent_policy = str(self.mcts_node_sweep_parent_policy).strip().lower()
            if parent_policy != "incumbent":
                raise ValueError("mcts_node_sweep_parent_policy must be incumbent")
            self.mcts_node_sweep_parent_policy = parent_policy
        if iteration_unit == "evaluated_unique_candidates":
            if str(self.search_method).strip().lower() != "mcts":
                raise ValueError(
                    "evaluated_unique_candidates budget requires MCTS"
                )
            if not bool(self.inner_structure_enabled):
                raise ValueError(
                    "evaluated_unique_candidates budget requires the inline "
                    "structure evaluator"
                )
        if int(self.inner_esmfold2_interval) < 1:
            raise ValueError("inner_esmfold2_interval must be positive")
        if float(self.inner_esmfold2_weight) < 0.0:
            raise ValueError("inner_esmfold2_weight must be non-negative")
        if int(self.structure_prescreen_min_candidates) < 0:
            raise ValueError("structure_prescreen_min_candidates must be non-negative")
        if int(self.structure_prescreen_max_candidates) < 0:
            raise ValueError("structure_prescreen_max_candidates must be non-negative")
        if int(self.structure_screen_progen_batch_size) < 0:
            raise ValueError("structure_screen_progen_batch_size must be non-negative")
        if bool(self.structure_prescreen_forward_all_to_screen):
            if not bool(self.structure_prescreen_enabled):
                raise ValueError(
                    "structure_prescreen_forward_all_to_screen requires the prescreen"
                )
            if not bool(self.structure_screen_enabled):
                raise ValueError(
                    "structure_prescreen_forward_all_to_screen requires the screen"
                )
            prescreen_cap = int(self.structure_prescreen_max_candidates)
            screen_cap = int(self.structure_screen_max_candidates)
            if prescreen_cap <= 0 or screen_cap < prescreen_cap:
                raise ValueError(
                    "structure_prescreen_forward_all_to_screen requires a positive "
                    "screen cap at least as large as the prescreen cap"
                )
        prescreen_fraction = float(self.structure_prescreen_top_frac)
        if not math.isfinite(prescreen_fraction) or prescreen_fraction < 0.0:
            raise ValueError(
                "structure_prescreen_top_frac must be finite and non-negative"
            )
        for name in ("structure_protenix_seeds", "structure_af3_seeds"):
            raw = getattr(self, name)
            if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
                raise ValueError(f"{name} must be a list of integers")
            if not raw or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw):
                raise ValueError(f"{name} must contain at least one integer seed")
            if len(raw) != len(set(raw)):
                raise ValueError(f"{name} must not contain duplicate seeds")
        if int(self.structure_robust_top_candidates) < 1:
            raise ValueError("structure_robust_top_candidates must be positive")
        threshold = float(self.structure_disagreement_threshold)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("structure_disagreement_threshold must be finite and non-negative")
        if self.structure_multiseed_enabled:
            if len(self.structure_protenix_seeds) < 1 or len(self.structure_af3_seeds) < 1:
                raise ValueError("formal multiseed mode requires both provider seed lists")
        if self.structure_formal_funnel_enabled:
            if not self.candidate_wave_enabled:
                raise ValueError("formal structure funnel requires the frozen 8/8/4 candidate wave")
            if not self.structure_multiseed_enabled:
                raise ValueError("formal structure funnel requires multiseed mode")
            if len(self.structure_protenix_seeds) != 3 or len(self.structure_af3_seeds) != 3:
                raise ValueError("formal structure funnel requires exactly three fixed seeds per provider")
            if str(self.structure_screen_model).lower() != "protenix":
                raise ValueError("formal structure funnel screen provider must be Protenix")
            if str(self.structure_rerank_model).lower() not in {"alphafold3", "af3"}:
                raise ValueError("formal structure funnel rerank provider must be AlphaFold3")
            if bool(self.structure_allow_low_fidelity_fallback):
                raise ValueError("formal structure funnel forbids low-fidelity fallback")
            if int(self.structure_physics_max_candidates) != int(self.structure_robust_top_candidates):
                raise ValueError("formal structure funnel physics quota must equal robust top-candidate quota")
            if not bool(self.structure_pyrosetta_required):
                raise ValueError("formal structure funnel requires PyRosetta receipts")
        mode = str(self.mcts_artifact_mode).strip().lower()
        if mode not in {"normalized", "legacy_full"}:
            raise ValueError(
                "mcts_artifact_mode must be 'normalized' or 'legacy_full'"
            )
        self.mcts_artifact_mode = mode
        for field_name in (
            "mcts_tree_min_root_children",
            "mcts_tree_min_branching_nodes",
            "mcts_tree_min_leaves",
            "mcts_tree_min_max_depth",
        ):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if int(self.node_optimizer_candidate_count) < 1:
            raise ValueError("node_optimizer_candidate_count must be positive")
        if self.sequence_prefilter_callable is not None:
            callable_spec = str(self.sequence_prefilter_callable).strip()
            if callable_spec and ":" not in callable_spec:
                raise ValueError(
                    "sequence_prefilter_callable must use module:function syntax"
                )
            self.sequence_prefilter_callable = callable_spec or None
        if not isinstance(self.sequence_prefilter_config, dict):
            raise ValueError("sequence_prefilter_config must be a dictionary")
        if self.sequence_bootstrap_callable is not None:
            callable_spec = str(self.sequence_bootstrap_callable).strip()
            if callable_spec and ":" not in callable_spec:
                raise ValueError(
                    "sequence_bootstrap_callable must use module:function syntax"
                )
            self.sequence_bootstrap_callable = callable_spec or None
        if not isinstance(self.sequence_bootstrap_config, dict):
            raise ValueError("sequence_bootstrap_config must be a dictionary")
        if int(self.node_optimizer_beam_width) < 1:
            raise ValueError("node_optimizer_beam_width must be positive")
        if int(self.node_optimizer_top_k_per_position) < 1:
            raise ValueError("node_optimizer_top_k_per_position must be positive")
        if float(self.node_optimizer_temperature) <= 0.0:
            raise ValueError("node_optimizer_temperature must be positive")
        if float(self.node_optimizer_diversity_weight) < 0.0:
            raise ValueError("node_optimizer_diversity_weight must be non-negative")
        if float(self.node_optimizer_mutation_penalty) < 0.0:
            raise ValueError("node_optimizer_mutation_penalty must be non-negative")
        if float(self.mcts_progressive_widening_c) <= 0.0:
            raise ValueError("mcts_progressive_widening_c must be positive")
        alpha = float(self.mcts_progressive_widening_alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(
                "mcts_progressive_widening_alpha must be in the interval (0, 1]"
            )
        tier_mode = str(self.proposal_tier_mode).strip().lower()
        if tier_mode not in {"fixed_node", "mixed"}:
            raise ValueError(
                "proposal_tier_mode must be 'fixed_node' or 'mixed'"
            )
        self.proposal_tier_mode = tier_mode
        shortlist_policy = str(self.structure_shortlist_policy).strip().lower()
        if shortlist_policy not in {
            "legacy_diverse",
            "formal_joint_novel",
            "formal_layered_novel",
        }:
            raise ValueError(
                "structure_shortlist_policy must be 'legacy_diverse' or "
                "'formal_joint_novel' or 'formal_layered_novel'"
            )
        self.structure_shortlist_policy = shortlist_policy
        if int(self.structure_physics_max_candidates) < 0:
            raise ValueError("structure_physics_max_candidates must be non-negative")
        if int(self.structure_screen_single_node_diagnostic_quota) < 0:
            raise ValueError(
                "structure_screen_single_node_diagnostic_quota must be non-negative"
            )
        if int(self.structure_position_distribution_engagement_quota) < 0:
            raise ValueError(
                "structure_position_distribution_engagement_quota must be non-negative"
            )
        if int(self.structure_portfolio_contract_quota) < 0:
            raise ValueError(
                "structure_portfolio_contract_quota must be non-negative"
            )
        if int(self.portfolio_seed_refinement_rounds) < 0:
            raise ValueError(
                "portfolio_seed_refinement_rounds must be non-negative"
            )
        selection_objective = str(self.structure_selection_objective).strip().lower()
        if selection_objective not in {"legacy_additive", "outer_aligned"}:
            raise ValueError(
                "structure_selection_objective must be 'legacy_additive' or "
                "'outer_aligned'"
            )
        self.structure_selection_objective = selection_objective


SearchConfig = SAConfig


__all__ = [
    "SAConfig",
    "SearchConfig",
    "normalize_residue_mutation_contract",
    "normalize_compiled_position_distribution_policy",
    "normalize_compiled_portfolio_request_policy",
    "CANDIDATE_WAVE_CONFIG_FIELDS",
    "validate_candidate_wave_config",
]
