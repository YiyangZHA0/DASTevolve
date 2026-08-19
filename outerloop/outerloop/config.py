

import os
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import dacite
import yaml

from engine.memory_policy import MemoryPolicyConfig

if TYPE_CHECKING:
    from outerloop.llm.base import LLMInterface


_ENV_VAR_PATTERN = re.compile(r"^\$\{([^}:]+)(?::-([^}]*))?\}$")


def _resolve_env_var(value: Optional[str]) -> Optional[str]:

    if value is None:
        return None

    match = _ENV_VAR_PATTERN.match(value)
    if not match:
        return value

    var_name = match.group(1)
    default_value = match.group(2)
    env_value = os.environ.get(var_name)
    if env_value is None:
        if default_value is not None:
            return default_value
        raise ValueError(f"Environment variable {var_name} is not set")
    return env_value


def _resolve_env_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return _resolve_env_var(value)
    return value


def _optional_int(value: Any) -> Optional[int]:
    value = _resolve_env_scalar(value)
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> Optional[float]:
    value = _resolve_env_scalar(value)
    if value is None or value == "":
        return None
    return float(value)


def _resolve_env_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env_refs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env_refs(item) for item in value]
    return _resolve_env_scalar(value)


def _coerce_mapping_fields(mapping: Dict[str, Any], int_fields: set[str], float_fields: set[str]) -> None:
    for key in int_fields:
        if key in mapping and mapping[key] not in (None, ""):
            mapping[key] = int(mapping[key])
    for key in float_fields:
        if key in mapping and mapping[key] not in (None, ""):
            mapping[key] = float(mapping[key])


def _coerce_config_scalars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    llm = config_dict.get("llm")
    if not isinstance(llm, dict):
        return config_dict

    int_fields = {"max_tokens", "timeout", "retries", "retry_delay"}
    float_fields = {
        "temperature",
        "top_p",
        "max_budget_usd",
        "primary_model_weight",
        "secondary_model_weight",
    }
    _coerce_mapping_fields(llm, int_fields, float_fields)

    for list_key in ("models", "evaluator_models"):
        models = llm.get(list_key)
        if isinstance(models, list):
            for model in models:
                if isinstance(model, dict):
                    _coerce_mapping_fields(model, int_fields, float_fields | {"weight"})
    return config_dict


@dataclass
class LLMModelConfig:


    api_base: str = None
    api_key: Optional[str] = None
    name: str = None


    provider: Optional[str] = None


    init_client: Optional[Callable] = None


    weight: float = 1.0


    system_message: Optional[str] = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = None


    timeout: int = None
    retries: int = None
    retry_delay: int = None


    random_seed: Optional[int] = None


    reasoning_effort: Optional[str] = None


    max_budget_usd: Optional[float] = None


    manual_mode: Optional[bool] = None
    _manual_queue_dir: Optional[str] = None

    def __post_init__(self):

        self.api_base = _resolve_env_var(self.api_base)
        self.api_key = _resolve_env_var(self.api_key)
        self.timeout = _optional_int(self.timeout)
        self.retries = _optional_int(self.retries)
        self.retry_delay = _optional_int(self.retry_delay)
        self.max_tokens = _optional_int(self.max_tokens)
        self.temperature = _optional_float(self.temperature)
        self.top_p = _optional_float(self.top_p)
        self.max_budget_usd = _optional_float(self.max_budget_usd)
        if self.max_budget_usd is not None and self.max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be positive")


@dataclass
class LLMConfig(LLMModelConfig):


    api_base: str = "https://api.openai.com/v1"


    system_message: Optional[str] = "system_message"
    temperature: float | None = 0.7
    top_p: float | None = None
    max_tokens: int = 4096


    timeout: int = 60
    retries: int = 3
    retry_delay: int = 5


    models: List[LLMModelConfig] = field(default_factory=list)


    evaluator_models: List[LLMModelConfig] = field(default_factory=lambda: [])


    primary_model: str = None
    primary_model_weight: float = None
    secondary_model: str = None
    secondary_model_weight: float = None


    reasoning_effort: Optional[str] = None


    manual_mode: bool = False

    def __post_init__(self):

        super().__post_init__()


        if self.primary_model:

            primary_model = LLMModelConfig(
                name=self.primary_model, weight=self.primary_model_weight or 1.0
            )
            self.models.append(primary_model)

        if self.secondary_model:

            if self.secondary_model_weight is None or self.secondary_model_weight > 0:
                secondary_model = LLMModelConfig(
                    name=self.secondary_model,
                    weight=(
                        self.secondary_model_weight
                        if self.secondary_model_weight is not None
                        else 0.2
                    ),
                )
                self.models.append(secondary_model)


        if (
            self.primary_model
            or self.secondary_model
            or self.primary_model_weight
            or self.secondary_model_weight
        ) and not self.models:
            raise ValueError(
                "No LLM models configured. Please specify 'models' array or "
                "'primary_model' in your configuration."
            )


        if not self.evaluator_models:
            self.evaluator_models = self.models.copy()


        shared_config = {


            "provider": self.provider,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "random_seed": self.random_seed,
            "reasoning_effort": self.reasoning_effort,
            "max_budget_usd": self.max_budget_usd,
            "manual_mode": self.manual_mode,
        }
        self.update_model_params(shared_config)

    def update_model_params(self, args: Dict[str, Any], overwrite: bool = False) -> None:

        for model in self.models + self.evaluator_models:
            for key, value in args.items():
                if overwrite or getattr(model, key, None) is None:
                    setattr(model, key, value)

    def rebuild_models(self) -> None:


        self.models = []
        self.evaluator_models = []


        if self.primary_model:

            primary_model = LLMModelConfig(
                name=self.primary_model, weight=self.primary_model_weight or 1.0
            )
            self.models.append(primary_model)

        if self.secondary_model:

            if self.secondary_model_weight is None or self.secondary_model_weight > 0:
                secondary_model = LLMModelConfig(
                    name=self.secondary_model,
                    weight=(
                        self.secondary_model_weight
                        if self.secondary_model_weight is not None
                        else 0.2
                    ),
                )
                self.models.append(secondary_model)


        if not self.evaluator_models:
            self.evaluator_models = self.models.copy()


        shared_config = {
            "provider": self.provider,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "random_seed": self.random_seed,
            "reasoning_effort": self.reasoning_effort,
            "max_budget_usd": self.max_budget_usd,
        }
        self.update_model_params(shared_config)


@dataclass
class PromptConfig:


    template_dir: Optional[str] = None
    system_message: str = "system_message"
    evaluator_system_message: str = "evaluator_system_message"


    programs_as_changes_description: bool = False
    system_message_changes_description: Optional[str] = None
    initial_changes_description: str = ""


    num_top_programs: int = 3
    num_diverse_programs: int = 2


    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = field(default_factory=dict)


    use_meta_prompting: bool = False
    meta_prompt_weight: float = 0.1


    include_artifacts: bool = True
    max_artifact_bytes: int = 20 * 1024


    max_artifact_total_bytes: Optional[int] = None


    artifact_priority: List[str] = field(default_factory=list)


    complete_artifacts: List[str] = field(default_factory=list)
    artifact_security_filter: bool = True
    optimizer_memory_max_chars: int = 12 * 1024
    hierarchical_design_max_chars: int = 32 * 1024


    candidate_diff_retries: int = 0


    proposal_mode: str = "legacy_diff"


    require_sequence_reconciliation: bool = False
    require_position_distributions: bool = False
    require_portfolio_capabilities: bool = False
    require_frozen_candidate_wave: bool = False


    hierarchical_audit_v2: bool = False


    suggest_simplification_after_chars: Optional[int] = (
        500
    )
    include_changes_under_chars: Optional[int] = (
        100
    )
    concise_implementation_max_lines: Optional[int] = (
        10
    )
    comprehensive_implementation_min_lines: Optional[int] = (
        50
    )


    diff_summary_max_line_len: int = 100
    diff_summary_max_lines: int = 30


    code_length_threshold: Optional[int] = (
        None
    )


@dataclass
class DatabaseConfig:


    db_path: Optional[str] = None
    in_memory: bool = True


    experiment_registry_enabled: Optional[bool] = None
    experiment_registry_path: Optional[str] = None
    experiment_registry_scope: Optional[str] = None
    experiment_registry_lease_seconds: float = 3600.0
    experiment_registry_retry_failed: bool = False
    experiment_registry_replicate_policy: str = "reject"


    log_prompts: bool = True


    population_size: int = 1000
    archive_size: int = 100
    num_islands: int = 5


    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7

    diversity_metric: str = "edit_distance"


    outer_effective_phenotype_enabled: Optional[bool] = None
    outer_parent_selection_mode: str = "weighted"
    outer_effective_descriptor_dimensions: List[str] = field(
        default_factory=lambda: [
            "node_action_coverage",
            "mutation_topology",
            "feasibility",
            "strategy_novelty",
            "sequence_novelty",
        ]
    )
    outer_decision_artifacts_enabled: bool = True


    outer_population_policy_version: str = "legacy"
    outer_parent_selection_mixture: Dict[str, float] = field(
        default_factory=lambda: {"elite": 0.15, "weighted": 0.55, "uniform": 0.30}
    )
    outer_behavior_bin_dimensions: List[str] = field(
        default_factory=lambda: [
            "feasibility",
            "functional_scope",
            "structural_scope",
            "mutation_burden",
            "action_coverage",
            "operator_scope",
        ]
    )
    outer_robustness_metrics: List[str] = field(
        default_factory=lambda: [
            "positive_A_iptm",
            "positive_A_interface_q",
            "positive_A_plddt",
            "apo_plddt",
            "iptm_margin",
            "gpde_margin",
            "interface_q_margin",
            "worst_case_score",
            "seed_score_std",
            "multistate_score",
            "clash_count",
            "node_plddt_min",
        ]
    )


    proposal_wave_size: int = 1
    strategy_epoch_candidate_interval: int = 8


    feature_dimensions: List[str] = field(
        default_factory=lambda: ["complexity", "diversity"],
        metadata={
            "help": "List of feature dimensions for MAP-Elites grid. "
            "Built-in dimensions: 'complexity', 'diversity', 'score'. "
            "Custom dimensions: Must match metric names from evaluator. "
            "IMPORTANT: Evaluators must return raw continuous values for custom dimensions, "
            "NOT pre-computed bin indices. OuterLoop handles all scaling and binning internally."
        },
    )
    feature_bins: Union[int, Dict[str, int]] = 10
    diversity_reference_size: int = 20


    migration_interval: int = 50
    migration_rate: float = 0.1
    migration_interval_scope: str = "max_island_generation"


    random_seed: Optional[int] = 42


    artifacts_base_path: Optional[str] = None
    artifact_size_threshold: int = 32 * 1024
    cleanup_old_artifacts: bool = True
    artifact_retention_days: int = 30
    max_snapshot_artifacts: Optional[int] = (
        100
    )


    optimizer_memory_enabled: bool = True
    optimizer_memory_path: Optional[str] = None
    optimizer_memory_recent_limit: int = 20
    optimizer_memory_best_limit: int = 5
    ast_scheduler_enabled: bool = True
    ast_parent_selection_weight: float = 0.35
    island_role_selection_weight: float = 0.35
    island_role_survivor_objective_tolerance: float = 0.0
    ast_scheduler_stagnation_window: int = 5
    ast_scheduler_stagnation_min_delta: float = 1e-6
    island_specialization_enabled: bool = True
    executable_island_directives_enabled: bool = False


    island_roles: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "role_id": "a_interface_fold",
                "name": "A-interface and fold integrity",
                "focus": "Improve Caspr4 A binding while preserving fold/interface quality.",
                "soft_objectives": ["A ipTM", "A interface quality", "pLDDT", "pTM"],
                "complementary_roles": ["selectivity_margin", "robustness_safety"],
            },
            {
                "role_id": "selectivity_margin",
                "name": "A/B selectivity margin",
                "focus": "Improve the Caspr4-over-SDC1 selectivity margin without violating A-state gates.",
                "soft_objectives": ["A/B ipTM margin", "GPDE margin", "positive A interface quality"],
                "complementary_roles": ["a_interface_fold", "region_exploration", "robustness_safety"],
            },
            {
                "role_id": "region_exploration",
                "name": "new-region exploration",
                "focus": "Probe evidence-supported structural regions and non-incumbent residue positions.",
                "soft_objectives": ["node-action coverage", "mutation topology", "sequence novelty"],
                "complementary_roles": ["selectivity_margin", "a_interface_fold"],
            },
            {
                "role_id": "robustness_safety",
                "name": "robustness and safety",
                "focus": "Prefer candidates that remain good across states/seeds with low clashes and variance.",
                "soft_objectives": ["multi-seed robustness", "clash-free interface", "worst-case quality"],
                "complementary_roles": ["a_interface_fold", "selectivity_margin"],
            },
        ]
    )

    novelty_llm: Optional["LLMInterface"] = None
    embedding_model: Optional[str] = None
    similarity_threshold: float = 0.99

    def __post_init__(self) -> None:
        if self.outer_population_policy_version not in {"legacy", "v9"}:
            raise ValueError("outer_population_policy_version must be legacy or v9")
        if self.executable_island_directives_enabled:
            if self.num_islands != 4 or len(self.island_roles) != 4:
                raise ValueError(
                    "executable island directives require exactly four configured islands and roles"
                )
            if self.outer_population_policy_version != "v9":
                raise ValueError("executable island directives require v9 population policy")
            if self.migration_interval > 20:
                raise ValueError(
                    "executable island directives require migration_interval <= 20"
                )
        if self.migration_interval_scope not in {
            "max_island_generation",
            "global_iteration",
            "per_island",
        }:
            raise ValueError("migration_interval_scope is invalid")
        for name, value in (
            ("population_size", self.population_size),
            ("archive_size", self.archive_size),
            ("num_islands", self.num_islands),
            ("migration_interval", self.migration_interval),
            ("proposal_wave_size", self.proposal_wave_size),
            ("strategy_epoch_candidate_interval", self.strategy_epoch_candidate_interval),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        expected_arms = {"elite", "weighted", "uniform"}
        if set(self.outer_parent_selection_mixture) != expected_arms:
            raise ValueError(
                "outer_parent_selection_mixture must contain exactly "
                "elite, weighted, and uniform"
            )
        mixture_total = 0.0
        for value in self.outer_parent_selection_mixture.values():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError("parent selection mixture weights must be finite and non-negative")
            mixture_total += float(value)
        if mixture_total <= 0.0:
            raise ValueError("parent selection mixture must have positive total weight")
        for name, value in (
            ("ast_parent_selection_weight", self.ast_parent_selection_weight),
            ("island_role_selection_weight", self.island_role_selection_weight),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        survivor_tolerance = self.island_role_survivor_objective_tolerance
        if (
            isinstance(survivor_tolerance, bool)
            or not isinstance(survivor_tolerance, (int, float))
            or not math.isfinite(float(survivor_tolerance))
            or float(survivor_tolerance) < 0.0
        ):
            raise ValueError(
                "island_role_survivor_objective_tolerance must be finite and non-negative"
            )
        allowed_bins = {
            "feasibility",
            "functional_scope",
            "structural_scope",
            "mutation_burden",
            "action_coverage",
            "operator_scope",
            "chain_scope",
            "strategy_novelty",
            "sequence_novelty",
        }
        if (
            not self.outer_behavior_bin_dimensions
            or len(self.outer_behavior_bin_dimensions)
            != len(set(self.outer_behavior_bin_dimensions))
            or not set(self.outer_behavior_bin_dimensions) <= allowed_bins
        ):
            raise ValueError("outer_behavior_bin_dimensions is invalid")


@dataclass
class HierarchicalDesignConfig:


    enabled: bool = False
    global_strategist_enabled: bool = True
    proposal_critic_enabled: bool = True
    llm_critic_enabled: bool = False
    reasoning_audit_enabled: bool = True
    strategy_refresh_on_stagnation: bool = True
    global_strategy_retries: int = 2
    evidence_max_chars: int = 32768
    node_tenure: int = 3
    exploration_window: int = 4
    outside_incumbent_execution_quota: float = 0.5
    min_segments_per_window: int = 2
    required_hard_gate_metrics: List[str] = field(
        default_factory=lambda: [
            "hard_gate_pass",
            "positive_A_plddt",
            "apo_plddt",
        ]
    )
    hypothesis_ledger_path: Optional[str] = None
    reasoning_audit_path: Optional[str] = None

    def __post_init__(self) -> None:
        for name, value in (
            ("global_strategy_retries", self.global_strategy_retries),
            ("evidence_max_chars", self.evidence_max_chars),
            ("node_tenure", self.node_tenure),
            ("exploration_window", self.exploration_window),
            ("min_segments_per_window", self.min_segments_per_window),
        ):
            minimum = 0 if name == "global_strategy_retries" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        quota = self.outside_incumbent_execution_quota
        if (
            isinstance(quota, bool)
            or not isinstance(quota, (int, float))
            or not 0.0 <= float(quota) <= 1.0
        ):
            raise ValueError("outside_incumbent_execution_quota must be in [0, 1]")
        if (
            not isinstance(self.required_hard_gate_metrics, list)
            or not self.required_hard_gate_metrics
            or any(
                not isinstance(item, str) or not item.strip()
                for item in self.required_hard_gate_metrics
            )
            or len(set(self.required_hard_gate_metrics))
            != len(self.required_hard_gate_metrics)
        ):
            raise ValueError(
                "required_hard_gate_metrics must be a non-empty unique string list"
            )


@dataclass
class EvaluatorConfig:


    timeout: int = 300
    max_retries: int = 3


    memory_limit_mb: Optional[int] = None
    cpu_limit: Optional[float] = None


    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])


    parallel_evaluations: int = 1

    distributed: bool = False


    use_llm_feedback: bool = False
    llm_feedback_weight: float = 0.1


    enable_artifacts: bool = True
    max_artifact_storage: int = 100 * 1024 * 1024


@dataclass
class EvolutionTraceConfig:


    enabled: bool = False
    format: str = "jsonl"
    include_code: bool = False
    include_prompts: bool = True
    output_path: Optional[str] = None
    buffer_size: int = 10
    compress: bool = False


@dataclass
class Config:


    max_iterations: int = 10000
    checkpoint_interval: int = 100
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    random_seed: Optional[int] = 42
    language: str = None
    file_suffix: str = ".py"


    llm: LLMConfig = field(default_factory=LLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    evolution_trace: EvolutionTraceConfig = field(default_factory=EvolutionTraceConfig)
    hierarchical_design: HierarchicalDesignConfig = field(
        default_factory=HierarchicalDesignConfig
    )


    memory_policy: MemoryPolicyConfig = field(default_factory=MemoryPolicyConfig)


    diff_based_evolution: bool = True
    max_code_length: int = 10000
    diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"


    early_stopping_patience: Optional[int] = None
    convergence_threshold: float = 0.001
    early_stopping_metric: str = "combined_score"


    max_tasks_per_child: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":

        config_path = Path(path).resolve()
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        config = cls.from_dict(config_dict)


        if config.prompt.template_dir:
            template_path = Path(config.prompt.template_dir)
            if not template_path.is_absolute():
                config.prompt.template_dir = str((config_path.parent / template_path).resolve())

        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        config_dict = _resolve_env_refs(config_dict)
        config_dict = _coerce_config_scalars(config_dict)

        if "diff_pattern" in config_dict:
            try:
                re.compile(config_dict["diff_pattern"])
            except re.error as e:
                raise ValueError(f"Invalid regex pattern in diff_pattern: {e}")


        if "llm" in config_dict:
            if "temperature" in config_dict["llm"] and config_dict["llm"]["temperature"] is None:
                del config_dict["llm"]["temperature"]
            if "top_p" in config_dict["llm"] and config_dict["llm"]["top_p"] is None:
                del config_dict["llm"]["top_p"]

        config: Config = dacite.from_dict(
            data_class=cls,
            data=config_dict,
            config=dacite.Config(
                cast=[List, Union],
                forward_references={"LLMInterface": Any},
            ),
        )

        if config.database.random_seed is None and config.random_seed is not None:
            config.database.random_seed = config.random_seed

        if config.prompt.programs_as_changes_description and not config.diff_based_evolution:
            raise ValueError(
                "prompt.programs_as_changes_description=true requires diff_based_evolution=true "
                "(full rewrites cannot reliably update code and changes_description together)"
            )

        supported_proposal_modes = {
            "legacy_diff",
            "structured_strategy_v1",
            "structured_design_action_v1",
        }
        structured_proposal_modes = {
            "structured_strategy_v1",
            "structured_design_action_v1",
        }
        if config.prompt.proposal_mode not in supported_proposal_modes:
            raise ValueError(
                "prompt.proposal_mode must be one of 'legacy_diff', "
                "'structured_strategy_v1', or 'structured_design_action_v1'"
            )
        if (
            config.prompt.proposal_mode in structured_proposal_modes
            and not config.diff_based_evolution
        ):
            raise ValueError(
                f"prompt.proposal_mode={config.prompt.proposal_mode} requires "
                "diff_based_evolution=true because the controller produces one "
                "bounded source mutation"
            )
        if (
            config.prompt.proposal_mode in structured_proposal_modes
            and config.prompt.programs_as_changes_description
        ):
            raise ValueError(
                f"{config.prompt.proposal_mode} does not support mutable changes descriptions"
            )

        return config

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: Union[str, Path]) -> None:

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:

    if config_path and os.path.exists(config_path):
        config = Config.from_yaml(config_path)
    else:
        config = Config()


        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

        config.llm.update_model_params({"api_key": api_key, "api_base": api_base})


    config.llm.update_model_params({"system_message": config.prompt.system_message})

    return config
