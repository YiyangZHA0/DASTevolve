

import base64
from copy import deepcopy
import hashlib
import json
import logging
import math
import os
import random
import shutil
import time
import threading
import uuid
from dataclasses import asdict, dataclass, field, fields


from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from outerloop.config import DatabaseConfig
from outerloop.memory_facts import OuterObservationFact, build_observation_fact
from outerloop.effective_phenotype import (
    AcceptedRuntimeArtifact,
    EffectivePhenotypeDescriptor,
    EffectivePhenotypeIdentity,
    PhenotypeDescriptorConfig,
)
from outerloop.evolution_policy import (
    EvolutionCandidate,
    EvolutionDecision,
    compare_replacement,
    decide_migration,
    decide_parent,
    derive_private_seed,
)
from engine.memory_policy import MemoryPolicyError, MemoryScope
from engine.experiment_identity import CodeIdentity
from astevolve.evaluation.selection import select_feasibility_first
from outerloop.utils.code_utils import calculate_edit_distance
from outerloop.utils.metric_semantics import get_metric_spec
from outerloop.utils.metrics_utils import (
    get_fitness_score,
    get_primary_objective,
    safe_numeric_average,
)

logger = logging.getLogger(__name__)


def _safe_sum_metrics(metrics: Dict[str, Any]) -> float:

    numeric_values = [
        v for v in metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return sum(numeric_values) if numeric_values else 0.0


def _safe_avg_metrics(metrics: Dict[str, Any]) -> float:

    numeric_values = [
        v for v in metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return sum(numeric_values) / max(1, len(numeric_values)) if numeric_values else 0.0


@dataclass
class Program:


    id: str
    code: str
    changes_description: str = ""
    language: str = "python"


    parent_id: Optional[str] = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    iteration_found: int = 0


    metrics: Dict[str, float] = field(default_factory=dict)


    complexity: float = 0.0
    diversity: float = 0.0


    metadata: Dict[str, Any] = field(default_factory=dict)


    prompts: Optional[Dict[str, Any]] = None


    artifacts_json: Optional[str] = None
    artifact_dir: Optional[str] = None


    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":


        if "changes_description" not in data:
            metadata = data.get("metadata") or {}
            if isinstance(metadata, dict):
                data = {
                    **data,
                    "changes_description": metadata.get("changes_description")
                    or metadata.get("changes")
                    or "empty",
                }
            else:
                data = {**data, "changes_description": "empty"}


        valid_fields = {f.name for f in fields(cls)}


        filtered_data = {k: v for k, v in data.items() if k in valid_fields}


        if len(filtered_data) != len(data):
            filtered_out = set(data.keys()) - set(filtered_data.keys())
            logger.debug(f"Filtered out unsupported fields when loading Program: {filtered_out}")

        return cls(**filtered_data)


class ProgramDatabase:


    def __init__(self, config: DatabaseConfig):
        self.config = config


        self.programs: Dict[str, Program] = {}


        self.memory_observation_facts: List[OuterObservationFact] = []
        self._memory_fact_index: Dict[
            Tuple[str, str, str, str, Optional[str]], OuterObservationFact
        ] = {}
        self.outer_decisions: List[Dict[str, Any]] = []
        self.outer_decision_counter: int = 0
        self.migration_receipts: List[Dict[str, Any]] = []
        self.map_cell_collision_count: int = 0
        self.map_cell_replacement_count: int = 0
        self.effective_phenotype_index: Dict[str, str] = {}
        self._outer_policy_lock = threading.RLock()


        self.island_feature_maps: List[Dict[str, str]] = [{} for _ in range(config.num_islands)]


        if isinstance(config.feature_bins, int):
            self.feature_bins = max(
                config.feature_bins,
                int(pow(config.archive_size, 1 / len(config.feature_dimensions)) + 0.99),
            )
        else:

            self.feature_bins = 10


        self.islands: List[Set[str]] = [set() for _ in range(config.num_islands)]


        self.current_island: int = 0
        self.island_generations: List[int] = [0] * config.num_islands
        self.last_migration_generation: int = 0
        self.last_migration_iteration: int = 0
        self.last_migration_island_generations: List[int] = [0] * config.num_islands
        self.migration_interval: int = getattr(config, "migration_interval", 10)
        self.migration_rate: float = getattr(config, "migration_rate", 0.1)


        self.archive: Set[str] = set()


        self.best_program_id: Optional[str] = None


        self.island_best_programs: List[Optional[str]] = [None] * config.num_islands


        self.last_iteration: int = 0


        if config.db_path and os.path.exists(config.db_path):
            self.load(config.db_path)


        self.prompts_by_program: Dict[str, Dict[str, Dict[str, str]]] = None
        self.optimizer_memory = None
        self._hierarchical_design_store = None


        if config.random_seed is not None and not self._effective_outer_enabled():
            import random

            random.seed(config.random_seed)
            logger.debug(f"Database: Set random seed to {config.random_seed}")


        self.diversity_cache: Dict[int, Dict[str, Union[float, float]]] = (
            {}
        )
        self.diversity_cache_size: int = 1000
        self.diversity_reference_set: List[str] = (
            []
        )
        self.diversity_reference_size: int = getattr(config, "diversity_reference_size", 20)


        self.feature_stats: Dict[str, Dict[str, Union[float, float, List[float]]]] = {}
        self.feature_scaling_method: str = "minmax"


        if hasattr(config, "feature_bins") and isinstance(config.feature_bins, dict):
            self.feature_bins_per_dim = config.feature_bins
        else:

            self.feature_bins_per_dim = {
                dim: self.feature_bins for dim in config.feature_dimensions
            }

        logger.info(f"Initialized program database with {len(self.programs)} programs")


        from outerloop.embedding import EmbeddingClient

        self.novelty_llm = config.novelty_llm
        self.embedding_client = (
            EmbeddingClient(config.embedding_model) if config.embedding_model else None
        )
        self.similarity_threshold = config.similarity_threshold

    def _effective_outer_enabled(self) -> bool:


        return getattr(self.config, "outer_effective_phenotype_enabled", None) is True

    def _v9_population_policy_enabled(self) -> bool:
        return self._effective_outer_enabled() and str(
            getattr(self.config, "outer_population_policy_version", "legacy")
        ) == "v9"

    def attach_hierarchical_design(self, store: Any) -> None:


        with self._outer_policy_lock:
            self._hierarchical_design_store = store

    def get_hierarchical_design_snapshot(self) -> Dict[str, Any]:


        with self._outer_policy_lock:
            store = self._hierarchical_design_store
            if store is None:
                return {}
            if callable(getattr(store, "snapshot", None)):
                raw = store.snapshot()
            elif callable(getattr(store, "to_dict", None)):
                raw = store.to_dict()
            else:
                raw = store
            if not isinstance(raw, Mapping):
                raise TypeError("hierarchical design store must expose a mapping")
            return json.loads(self._canonical_outer_json(raw))

    def record_hierarchical_generation(
        self, plan: Any, observations: Sequence[Any]
    ) -> Dict[str, Any]:


        with self._outer_policy_lock:
            store = self._hierarchical_design_store
            if store is None:
                return {"status": "not_configured"}
            observer = getattr(store, "observe_generation", None)
            if callable(observer):
                result = observer(self, plan, tuple(observations))
                return (
                    deepcopy(dict(result))
                    if isinstance(result, Mapping)
                    else {"status": "published"}
                )
            return {"status": "observer_unavailable"}

    def _effective_behavior_bins(
        self, descriptor: EffectivePhenotypeDescriptor
    ) -> Dict[str, Any]:


        components = descriptor.components
        coverage = components.get("node_action_coverage", {}) or {}
        topology = components.get("mutation_topology", {}) or {}

        def scope(count: int) -> str:
            return (
                "none"
                if count <= 0
                else "single"
                if count == 1
                else "few"
                if count <= 3
                else "broad"
            )

        def burden(count: int) -> str:
            return (
                "none"
                if count <= 0
                else "single"
                if count == 1
                else "low"
                if count <= 3
                else "medium"
                if count <= 8
                else "high"
            )

        def coverage_bin(value: float) -> str:
            return (
                "none"
                if value <= 0.0
                else "low"
                if value < 0.5
                else "partial"
                if value < 0.999999
                else "full"
            )

        sites = list(topology.get("unique_mutated_sites", []) or [])
        chains = {str(site).split(":", 1)[0] for site in sites}
        all_bins = {
            "feasibility": (
                "feasible"
                if (components.get("feasibility", {}) or {}).get("feasible")
                else "infeasible"
            ),
            "functional_scope": scope(
                len(coverage.get("executed_functional_nodes", []) or [])
            ),
            "structural_scope": scope(
                len(coverage.get("executed_structural_nodes", []) or [])
            ),
            "mutation_burden": burden(
                int(topology.get("mutation_event_count", len(sites)) or 0)
            ),
            "action_coverage": coverage_bin(
                float(coverage.get("action_coverage_fraction", 0.0) or 0.0)
            ),
            "operator_scope": scope(
                len(set(coverage.get("operators", []) or []))
            ),
            "chain_scope": scope(len(chains)),
            "strategy_novelty": (
                "novel"
                if (components.get("strategy_novelty", {}) or {}).get("is_novel")
                else "seen"
            ),
            "sequence_novelty": (
                "novel"
                if (components.get("sequence_novelty", {}) or {}).get("is_novel")
                else "seen"
            ),
        }
        dimensions = list(
            getattr(self.config, "outer_behavior_bin_dimensions", ()) or ()
        )
        return {
            "schema_version": "astevolve.biological_behavior_bins.v1",
            "dimensions": dimensions,
            "bins": {dimension: all_bins[dimension] for dimension in dimensions},
        }

    def get_island_role(self, island_idx: Optional[int]) -> Dict[str, Any]:

        roles = getattr(self.config, "island_roles", None) or []
        idx = self.current_island if island_idx is None else int(island_idx)
        role = dict(roles[idx % len(roles)]) if roles and isinstance(roles[idx % len(roles)], Mapping) else {}
        role.setdefault("role_id", f"island_{idx % max(1, len(self.islands))}")
        role.setdefault("name", role["role_id"])
        role.setdefault("focus", "Explore feasible improvements under the global objective.")
        role.setdefault("soft_objectives", [])
        role["island"] = idx % max(1, len(self.islands))
        role["soft_objectives"] = [str(item) for item in role.get("soft_objectives", [])]
        snapshot = self.get_hierarchical_design_snapshot()
        executable = snapshot.get("executable_island_set", {}) if snapshot else {}
        rows = executable.get("directives", []) if isinstance(executable, Mapping) else []
        for directive in rows:
            if isinstance(directive, Mapping) and directive.get("island") == role["island"]:
                role["executable_directive"] = deepcopy(dict(directive))
                role["effective_directive_hash"] = directive.get("directive_hash")
                break
        return role

    @staticmethod
    def _global_program_key(program: Program) -> Tuple[Any, ...]:
        metrics = program.metrics or {}
        def numeric(name: str, default: float) -> float:
            value = metrics.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return default
            return float(value)

        objective = get_primary_objective(metrics) or {}
        value = float(objective.get("value", float("inf")))
        direction = str(objective.get("direction", "minimize"))
        objective_key = value if direction == "minimize" else -value
        feasible = 1 if bool(metrics.get("hard_gate_pass", True)) else 0
        multistate = numeric("multistate_score", float("-inf"))
        clashes = numeric("clash_count", float("inf"))
        node_min = numeric("node_plddt_min", float("-inf"))
        return (-feasible, objective_key, -multistate, clashes, -node_min, program.id)

    @staticmethod
    def _v9_robust_dimensions(program: Program) -> Dict[str, Tuple[float, str]]:
        metrics = program.metrics or {}
        objective = get_primary_objective(metrics) or {}
        values: Dict[str, Tuple[Any, str]] = {
            "primary_objective": (
                objective.get("value"),
                str(objective.get("direction", "minimize")),
            ),
            "positive_noninferiority_min_ratio": (
                metrics.get("positive_noninferiority_min_ratio"),
                "maximize",
            ),
            "positive_A_iptm": (metrics.get("positive_A_iptm"), "maximize"),
            "positive_A_interface_q": (
                metrics.get("positive_A_interface_q"),
                "maximize",
            ),
            "positive_A_plddt": (metrics.get("positive_A_plddt"), "maximize"),
            "apo_plddt": (metrics.get("apo_plddt"), "maximize"),
            "worst_case_score": (metrics.get("worst_case_score"), "maximize"),
            "seed_score_std": (metrics.get("seed_score_std"), "minimize"),
            "clash_count": (metrics.get("clash_count"), "minimize"),
            "node_plddt_min": (metrics.get("node_plddt_min"), "maximize"),
            "hard_gate_pass_count": (
                metrics.get("hard_gate_pass_count"), "maximize"
            ),
            "min_hard_margin": (metrics.get("min_hard_margin"), "maximize"),
            "final_energy": (metrics.get("final_energy"), "minimize"),
            "combined_score": (metrics.get("combined_score"), "maximize"),
            "backend_disagreement": (
                metrics.get("backend_disagreement"), "minimize"
            ),
        }
        return {
            name: (float(value), direction)
            for name, (value, direction) in values.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        }

    @classmethod
    def _v9_pareto_front(cls, programs: Sequence[Program]) -> List[Program]:
        feasible = [
            program
            for program in programs
            if bool((program.metrics or {}).get("hard_gate_pass", True))
        ]
        pool = feasible or list(programs)

        def dominates(left: Program, right: Program) -> bool:
            a = cls._v9_robust_dimensions(left)
            b = cls._v9_robust_dimensions(right)
            if set(a) != set(b):


                return False
            shared = sorted(set(a) & set(b))
            if not shared:
                return False
            no_worse = True
            strictly_better = False
            for name in shared:
                left_value, direction = a[name]
                right_value, _ = b[name]
                if direction == "minimize":
                    no_worse &= left_value <= right_value
                    strictly_better |= left_value < right_value
                else:
                    no_worse &= left_value >= right_value
                    strictly_better |= left_value > right_value
                if not no_worse:
                    return False
            return strictly_better

        return [
            program
            for program in pool
            if not any(
                other.id != program.id and dominates(other, program)
                for other in pool
            )
        ]

    @staticmethod
    def _v9_robust_key(program: Program) -> Tuple[Any, ...]:
        metrics = program.metrics or {}

        def numeric(name: str, default: float) -> float:
            value = metrics.get(name)
            return (
                float(value)
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                else default
            )

        objective = get_primary_objective(metrics) or {}
        objective_value = objective.get("value")
        objective_key = (
            float(objective_value)
            if isinstance(objective_value, (int, float))
            and not isinstance(objective_value, bool)
            and math.isfinite(float(objective_value))
            else float("inf")
        )
        if str(objective.get("direction", "minimize")) == "maximize":
            objective_key = -objective_key
        return (
            -numeric("positive_noninferiority_min_ratio", float("-inf")),
            -numeric("worst_case_score", float("-inf")),
            objective_key,
            numeric("seed_score_std", float("inf")),
            numeric("clash_count", float("inf")),
            -numeric("node_plddt_min", float("-inf")),
            program.id,
        )

    def _v9_global_entry(
        self, program: Program, island_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        metrics = program.metrics or {}
        metadata = program.metadata or {}
        primary_island = metadata.get("island", 0)
        resolved_island = (
            int(primary_island or 0)
            if island_idx is None
            else int(island_idx)
        ) % max(1, len(self.islands))
        behavior = metadata.get("outer_behavior_descriptor", {}) or {}
        robust_names = list(
            getattr(self.config, "outer_robustness_metrics", ()) or ()
        )
        return {
            "program_id": program.id,
            "parent_id": program.parent_id,
            "island": resolved_island,
            "islands": list(metadata.get("islands", []) or []),
            "island_role": self.get_island_role(resolved_island),
            "behavior_vector": dict(behavior.get("bins", {}) or {}),
            "robustness_vector": {
                "hard_gate_pass": metrics.get("hard_gate_pass"),
                "primary_objective": get_primary_objective(metrics),
                "metrics": {
                    name: metrics[name]
                    for name in robust_names
                    if name in metrics
                },
            },
            "audit_identity": {
                "effective_descriptor_hash": metadata.get(
                    "effective_descriptor_audit_hash"
                ),
                "effective_phenotype_hash": metadata.get(
                    "effective_phenotype_audit_hash"
                ),
            },
        }

    def _get_v9_global_summary(self, top_n: int) -> Dict[str, Any]:
        selectable_ids = set(self.archive)
        selectable_ids.update(pid for island in self.islands for pid in island)
        if self.best_program_id in self.programs:


            selectable_ids.add(self.best_program_id)
        selectable = [
            self.programs[program_id]
            for program_id in selectable_ids
            if program_id in self.programs
        ]
        pareto = sorted(self._v9_pareto_front(selectable), key=self._v9_robust_key)
        pareto_ids = {program.id for program in pareto}
        ranked = pareto + sorted(
            (program for program in selectable if program.id not in pareto_ids),
            key=self._v9_robust_key,
        )
        selected = [
            self._v9_global_entry(program)
            for program in ranked[: max(1, int(top_n))]
        ]
        portfolios = []
        for island_idx, island in enumerate(self.islands):
            programs = [
                self.programs[program_id]
                for program_id in island
                if program_id in self.programs
            ]
            island_pareto = sorted(
                self._v9_pareto_front(programs), key=self._v9_robust_key
            )
            cell_owner_ids = {
                program_id
                for program_id in self.island_feature_maps[island_idx].values()
                if program_id in self.programs
            }
            coverage = [
                self._v9_global_entry(self.programs[program_id], island_idx)[
                    "behavior_vector"
                ]
                for program_id in sorted(cell_owner_ids)
            ]
            portfolios.append(
                {
                    "island": island_idx,
                    "island_role": self.get_island_role(island_idx),
                    "population_size": len(programs),
                    "behavior_cell_count": len(cell_owner_ids),
                    "behavior_coverage": coverage,
                    "robust_frontier": [
                        self._v9_global_entry(program, island_idx)
                        for program in island_pareto[: min(3, max(1, int(top_n)))]
                    ],
                }
            )
        return {
            "schema_version": "astevolve.global_island_summary.v2",
            "iteration": self.last_iteration,
            "island_count": len(self.islands),
            "islands": self.get_island_stats(),
            "role_portfolios": portfolios,
            "independent_parent_pools": [
                {
                    "island": index,
                    "scope": f"island:{index}:independent",
                    "program_ids": sorted(island),
                    "shared_migrant_ids": sorted(
                        program_id
                        for program_id in island
                        if program_id in self.programs
                        and bool((self.programs[program_id].metadata or {}).get("shared_migrant"))
                    ),
                }
                for index, island in enumerate(self.islands)
            ],
            "migration_receipts": deepcopy(self.migration_receipts[-64:]),
            "map_elites_diagnostics": {
                "schema_version": "astevolve.map_elites_diagnostics.v1",
                "cell_collisions": self.map_cell_collision_count,
                "cell_replacements": self.map_cell_replacement_count,
                "descriptor_overfine": bool(
                    self.last_iteration >= 80
                    and self.map_cell_collision_count == 0
                    and self.map_cell_replacement_count == 0
                ),
            },
            "global_pareto_front": [
                self._v9_global_entry(program) for program in pareto
            ],
            "global_selection": selected,
            "recommended_program_id": (
                selected[0]["program_id"] if selected else None
            ),
            "recommendation_reason": (
                "feasibility-first Pareto frontier over absolute target quality, "
                "selectivity objective, robustness, clashes, and node confidence"
            ),
            "selection_policy": (
                "hard gates, primary objective, multistate quality, clashes, "
                "node confidence; coarse biological behavior defines cells; "
                "exact AST and phenotype hashes are audit/dedup identities only"
            ),
        }

    def get_global_summary(self, top_n: int = 8) -> Dict[str, Any]:

        with self._outer_policy_lock:
            if self._v9_population_policy_enabled():
                return self._get_v9_global_summary(top_n)
            selectable_ids = set(self.archive)
            selectable_ids.update(pid for island in self.islands for pid in island)
            programs = [self.programs[pid] for pid in selectable_ids if pid in self.programs]
            ranked = sorted(programs, key=self._global_program_key)
            selected = []
            for program in ranked[: max(1, int(top_n))]:
                selected.append({
                    "program_id": program.id,
                    "island": program.metadata.get("island"),
                    "island_role": program.metadata.get("island_role") or self.get_island_role(program.metadata.get("island")),
                    "parent_id": program.parent_id,
                    "hard_gate_pass": program.metrics.get("hard_gate_pass"),
                    "primary_objective": get_primary_objective(program.metrics),
                    "multistate_score": program.metrics.get("multistate_score"),
                    "clash_count": program.metrics.get("clash_count"),
                    "node_plddt_min": program.metrics.get("node_plddt_min"),
                })
            return {
                "schema_version": "astevolve.global_island_summary.v1",
                "iteration": self.last_iteration,
                "island_count": len(self.islands),
                "islands": self.get_island_stats(),
                "global_selection": selected,
                "recommended_program_id": selected[0]["program_id"] if selected else None,
                "recommendation_reason": "global robust ordering across island candidates; legacy scalar best is retained separately",
                "selection_policy": "feasible, primary objective, multistate score, clashes, node pLDDT minimum",
            }

    @staticmethod
    def _canonical_outer_json(value: Any) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def _effective_reference_hashes(self) -> Tuple[Set[str], Set[str]]:


        strategy_hashes: Set[str] = set()
        sequence_hashes: Set[str] = set()
        selectable_ids = set(self.archive)
        selectable_ids.update(pid for island in self.islands for pid in island)
        for program_id in sorted(selectable_ids):
            program = self.programs.get(program_id)
            if program is None:
                continue
            raw = (program.metadata or {}).get("effective_phenotype_identity")
            if not isinstance(raw, Mapping):
                continue
            identity = EffectivePhenotypeIdentity.from_mapping(raw)
            strategy_hashes.add(
                identity.effective_contract_identity.effective_contract_hash
            )
            sequence_hashes.add(
                identity.sequence_bundle_identity.sequence_bundle_hash
            )
        return strategy_hashes, sequence_hashes

    def _build_effective_candidate(
        self, program: Program
    ) -> Tuple[EvolutionCandidate, EffectivePhenotypeIdentity, str]:


        if not isinstance(program.metadata, dict):
            raise ValueError("effective outer program metadata must be a dictionary")
        metadata = program.metadata
        required = (
            "accepted_runtime_artifact",
            "effective_phenotype_identity",
            "effective_phenotype_descriptor",
        )
        missing = [key for key in required if not isinstance(metadata.get(key), Mapping)]
        if missing:
            raise ValueError(
                "effective outer admission requires accepted runtime metadata: "
                + ", ".join(missing)
            )

        accepted = AcceptedRuntimeArtifact.from_mapping(
            metadata["accepted_runtime_artifact"]
        )
        if accepted.code_identity.to_dict() != CodeIdentity.from_text(program.code).to_dict():
            raise ValueError("accepted runtime code identity does not match Program.code")
        accepted_metrics = accepted.runtime_evidence["metrics"]
        if self._canonical_outer_json(program.metrics) != self._canonical_outer_json(
            accepted_metrics
        ):
            raise ValueError(
                "Program.metrics does not match the sealed accepted runtime evidence"
            )

        supplied_identity = EffectivePhenotypeIdentity.from_mapping(
            metadata["effective_phenotype_identity"]
        )
        identity = EffectivePhenotypeIdentity.create(accepted)
        if supplied_identity.to_dict() != identity.to_dict():
            raise ValueError("effective phenotype identity does not match accepted runtime")

        supplied_descriptor = EffectivePhenotypeDescriptor.from_mapping(
            metadata["effective_phenotype_descriptor"]
        )
        reproduced_descriptor = EffectivePhenotypeDescriptor.create(
            accepted,
            config=supplied_descriptor.config,
        )
        if supplied_descriptor.to_dict() != reproduced_descriptor.to_dict():
            raise ValueError("effective phenotype descriptor is not derived from accepted runtime")

        strategy_refs, sequence_refs = self._effective_reference_hashes()
        configured_components = getattr(
            self.config, "outer_effective_descriptor_dimensions", None
        )
        descriptor_config = PhenotypeDescriptorConfig.create(
            components=configured_components if configured_components else None,
            strategy_reference_hashes=strategy_refs,
            sequence_reference_hashes=sequence_refs,
        )
        descriptor = EffectivePhenotypeDescriptor.create(
            accepted,
            config=descriptor_config,
        )
        behavior = None
        if self._v9_population_policy_enabled():
            behavior = self._effective_behavior_bins(descriptor)
            cell_payload = {
                "schema_version": "astevolve.biological_behavior_cell.v1",
                "dimensions": behavior["dimensions"],
                "bins": behavior["bins"],
            }
        else:
            cell_payload = {
                "schema_version": "astevolve.effective_descriptor_cell.v1",
                "components": descriptor.components,
            }
        cell_hash = "effective_cell_sha256:" + hashlib.sha256(
            self._canonical_outer_json(cell_payload).encode("utf-8")
        ).hexdigest()
        feasibility = accepted.feasibility
        candidate = EvolutionCandidate.create(
            candidate_id=program.id,
            objective=get_fitness_score(
                program.metrics, self.config.feature_dimensions
            ),
            gate_sources={
                "accepted_runtime": {
                    "passed": bool(feasibility["feasible"]),
                    "reasons": list(feasibility.get("reasons") or []),
                }
            },
            phenotype_hash=identity.archive_niche_hash,
            effective_contract_hash=(
                identity.effective_contract_identity.effective_contract_hash
            ),
            sequence_bundle_hash=(
                identity.sequence_bundle_identity.sequence_bundle_hash
            ),
        )
        metadata["effective_phenotype_descriptor"] = descriptor.to_dict()
        metadata["outer_evolution_candidate"] = candidate.to_dict()
        metadata["effective_descriptor_cell_hash"] = cell_hash
        if behavior is not None:
            metadata["effective_descriptor_audit_hash"] = descriptor.descriptor_hash
            metadata["effective_phenotype_audit_hash"] = identity.archive_niche_hash
            metadata["outer_behavior_descriptor"] = behavior
            metadata["outer_population_policy_version"] = "v9"
        return candidate, identity, cell_hash

    def _effective_candidate_for_program(self, program: Program) -> EvolutionCandidate:
        raw = (program.metadata or {}).get("outer_evolution_candidate")
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"program {program.id!r} lacks a validated outer evolution candidate"
            )
        candidate = EvolutionCandidate.from_mapping(raw)
        if candidate.candidate_id != program.id:
            raise ValueError("outer evolution candidate/program ID mismatch")
        return candidate

    def _resolve_effective_island(
        self, program: Program, target_island: Optional[int]
    ) -> int:
        if target_island is not None:
            return int(target_island) % len(self.islands)
        if program.parent_id:
            parent = self.programs.get(program.parent_id)
            if parent is not None:
                memberships = (parent.metadata or {}).get("islands") or []
                if memberships:
                    return int(sorted(memberships)[0]) % len(self.islands)
                if "island" in (parent.metadata or {}):
                    return int(parent.metadata["island"]) % len(self.islands)
        return self.current_island % len(self.islands)

    def _refresh_program_membership(self, program_id: str) -> None:
        program = self.programs.get(program_id)
        if program is None:
            return
        memberships = [
            index for index, island in enumerate(self.islands) if program_id in island
        ]
        program.metadata["islands"] = memberships
        if memberships:
            primary = program.metadata.get("island")
            if primary not in memberships:
                program.metadata["island"] = memberships[0]
        program.metadata["outer_selectable"] = bool(
            memberships or program_id in self.archive
        )

    def _refresh_effective_phenotype_index(self) -> None:
        selectable = set(self.archive)
        selectable.update(pid for island in self.islands for pid in island)
        rebuilt: Dict[str, str] = {}
        for program_id in sorted(selectable):
            program = self.programs.get(program_id)
            if program is None:
                continue
            candidate = self._effective_candidate_for_program(program)
            previous = rebuilt.get(candidate.phenotype_hash)
            if previous is not None and previous != program_id:
                raise ValueError(
                    "multiple selectable programs share one effective phenotype"
                )
            rebuilt[candidate.phenotype_hash] = program_id
        self.effective_phenotype_index = rebuilt

    def _new_effective_admission_decision(
        self,
        candidate: EvolutionCandidate,
        *,
        namespace: str,
        decision_index: int,
        reason: str,
    ) -> EvolutionDecision:
        seed = derive_private_seed(
            self.config.random_seed,
            namespace=namespace,
            decision_index=decision_index,
            candidate_ids=(candidate.candidate_id,),
        )
        return EvolutionDecision.create(
            kind="cell_admission",
            namespace=namespace,
            decision_index=decision_index,
            derived_seed=seed,
            selected_candidate_id=candidate.candidate_id,
            reason=reason,
            candidate_ids=(candidate.candidate_id,),
            eligible_ids=(candidate.candidate_id,),
            add_ids=(candidate.candidate_id,),
            details={"candidate": candidate.to_dict()},
        )

    def _commit_effective_archive(self, candidate: EvolutionCandidate) -> None:
        valid_ids = [pid for pid in sorted(self.archive) if pid in self.programs]
        self.archive = set(valid_ids)
        index = self._next_outer_decision_index()
        namespace = "effective-archive"
        if len(valid_ids) < self.config.archive_size:
            seed = derive_private_seed(
                self.config.random_seed,
                namespace=namespace,
                decision_index=index,
                candidate_ids=(candidate.candidate_id,),
            )
            decision = EvolutionDecision.create(
                kind="archive_admission",
                namespace=namespace,
                decision_index=index,
                derived_seed=seed,
                selected_candidate_id=candidate.candidate_id,
                reason="archive_has_capacity",
                candidate_ids=(candidate.candidate_id,),
                eligible_ids=(candidate.candidate_id,),
                add_ids=(candidate.candidate_id,),
                details={"candidate": candidate.to_dict()},
            )
        else:
            archive_candidates = [
                self._effective_candidate_for_program(self.programs[pid])
                for pid in valid_ids
            ]
            ordering = select_feasibility_first(
                [item.selection_row() for item in archive_candidates]
            )
            worst_id = ordering["ordered_ids"][-1]
            incumbent = next(
                item for item in archive_candidates if item.candidate_id == worst_id
            )
            decision = compare_replacement(
                candidate,
                incumbent,
                base_seed=self.config.random_seed,
                namespace=namespace,
                decision_index=index,
            )

            decision = EvolutionDecision.create(
                kind="archive_replacement",
                namespace=decision.namespace,
                decision_index=decision.decision_index,
                derived_seed=decision.derived_seed,
                selected_candidate_id=decision.selected_candidate_id,
                reason=decision.reason,
                candidate_ids=decision.candidate_ids,
                eligible_ids=decision.eligible_ids,
                add_ids=decision.add_ids,
                remove_ids=decision.remove_ids,
                details=decision.details,
            )

        for program_id in decision.remove_ids:
            self.archive.discard(program_id)
        for program_id in decision.add_ids:
            if program_id != candidate.candidate_id:
                raise ValueError("archive decision attempted a foreign add")
            self.archive.add(program_id)
        self._record_outer_decision(decision.to_dict())
        self.programs[candidate.candidate_id].metadata[
            "outer_archive_decision"
        ] = decision.to_dict()
        for program_id in set(decision.remove_ids) | set(decision.add_ids):
            self._refresh_program_membership(program_id)

    def _add_effective_program(
        self,
        program: Program,
        *,
        iteration: Optional[int],
        target_island: Optional[int],
    ) -> str:


        with self._outer_policy_lock:
            if program.id in self.programs:
                raise ValueError(f"program ID already exists: {program.id}")
            if iteration is not None:
                program.iteration_found = iteration
                self.last_iteration = max(self.last_iteration, iteration)
            candidate, identity, cell_hash = self._build_effective_candidate(program)
            island_idx = self._resolve_effective_island(program, target_island)
            program.metadata["island_role"] = self.get_island_role(island_idx)
            program.metadata["island_role_id"] = program.metadata["island_role"]["role_id"]
            feature_key = f"effective:{cell_hash}"
            namespace = f"island:{island_idx}:cell:{cell_hash}"
            decision_index = self._next_outer_decision_index()

            duplicate_id = self.effective_phenotype_index.get(
                identity.archive_niche_hash
            )
            incumbent_id = duplicate_id or self.island_feature_maps[island_idx].get(
                feature_key
            )
            if self.island_feature_maps[island_idx].get(feature_key) is not None:
                self.map_cell_collision_count += 1
            if incumbent_id is not None and incumbent_id not in self.programs:
                incumbent_id = None
            if incumbent_id is None:
                decision = self._new_effective_admission_decision(
                    candidate,
                    namespace=namespace,
                    decision_index=decision_index,
                    reason=(
                        "new_biological_behavior_cell"
                        if self._v9_population_policy_enabled()
                        else "new_effective_descriptor_cell"
                    ),
                )
            else:
                incumbent = self._effective_candidate_for_program(
                    self.programs[incumbent_id]
                )
                role_scores = None
                role_label = ""
                if self._v9_population_policy_enabled():
                    role = self.get_island_role(island_idx)
                    role_label = str(role.get("role_id") or "island_role")
                    role_scores = {
                        program.id: self._island_role_sampling_bonus(program, role),
                        incumbent_id: self._island_role_sampling_bonus(
                            self.programs[incumbent_id], role
                        ),
                    }
                decision = compare_replacement(
                    candidate,
                    incumbent,
                    base_seed=self.config.random_seed,
                    namespace=namespace,
                    decision_index=decision_index,
                    reject_duplicate_phenotype=True,
                    secondary_scores=role_scores,
                    secondary_label=role_label,
                    secondary_objective_tolerance=float(
                        getattr(
                            self.config,
                            "island_role_survivor_objective_tolerance",
                            0.0,
                        )
                    ),
                )


            program.metadata["outer_cell_decision"] = decision.to_dict()
            program.metadata["outer_selectable"] = False
            program.metadata["islands"] = []
            self.programs[program.id] = program
            self._record_outer_decision(decision.to_dict())

            admitted = program.id in decision.add_ids
            if admitted and decision.remove_ids:
                self.map_cell_replacement_count += 1
            if admitted:
                for removed_id in decision.remove_ids:
                    self.islands[island_idx].discard(removed_id)
                    for key, mapped_id in list(
                        self.island_feature_maps[island_idx].items()
                    ):
                        if mapped_id == removed_id:
                            del self.island_feature_maps[island_idx][key]
                    self._refresh_program_membership(removed_id)
                self.island_feature_maps[island_idx][feature_key] = program.id
                self.islands[island_idx].add(program.id)
                program.metadata["island"] = island_idx
                self._refresh_program_membership(program.id)
                self.effective_phenotype_index[
                    candidate.phenotype_hash
                ] = program.id
                self._commit_effective_archive(candidate)
                self._update_best_program(program)
                self._update_island_best_program(program, island_idx)

            self._refresh_effective_phenotype_index()
            self._enforce_population_limit(


                exclude_program_id=(
                    program.id if admitted and candidate.feasible else None
                )
            )
            if self.config.db_path:
                self._save_program(program)
            return program.id

    def add(
        self, program: Program, iteration: int = None, target_island: Optional[int] = None
    ) -> str:

        if self._effective_outer_enabled():
            return self._add_effective_program(
                program,
                iteration=iteration,
                target_island=target_island,
            )


        if iteration is not None:
            program.iteration_found = iteration

            self.last_iteration = max(self.last_iteration, iteration)

        self.programs[program.id] = program


        feature_coords = self._calculate_feature_coords(program)


        if target_island is None and program.parent_id:
            parent = self.programs.get(program.parent_id)
            if parent and "island" in parent.metadata:

                island_idx = parent.metadata["island"]
                logger.debug(
                    f"Program {program.id} inheriting island {island_idx} from parent {program.parent_id}"
                )
            else:

                island_idx = self.current_island
                if parent:
                    logger.warning(
                        f"Parent {program.parent_id} has no island metadata, using current_island {island_idx}"
                    )
                else:
                    logger.warning(
                        f"Parent {program.parent_id} not found, using current_island {island_idx}"
                    )
        elif target_island is not None:

            island_idx = target_island
        else:

            island_idx = self.current_island

        island_idx = island_idx % len(self.islands)


        if not self._is_novel(program.id, island_idx):
            logger.debug(
                f"Program {program.id} failed in novelty check and won't be added in the island {island_idx}"
            )
            return program.id


        feature_key = self._feature_coords_to_key(feature_coords)
        island_feature_map = self.island_feature_maps[island_idx]
        should_replace = feature_key not in island_feature_map

        if not should_replace:

            existing_program_id = island_feature_map[feature_key]
            if existing_program_id not in self.programs:

                should_replace = True
                logger.debug(
                    f"Replacing stale program reference {existing_program_id} in island {island_idx} feature map"
                )
            else:

                should_replace = self._is_better(program, self.programs[existing_program_id])


        replaced_program_id = None

        if should_replace:

            coords_dict = {
                self.config.feature_dimensions[i]: feature_coords[i]
                for i in range(len(feature_coords))
            }

            if feature_key not in island_feature_map:

                logger.info(
                    "New MAP-Elites cell occupied in island %d: %s", island_idx, coords_dict
                )

                total_possible_cells = self.feature_bins ** len(self.config.feature_dimensions)
                island_coverage = (len(island_feature_map) + 1) / total_possible_cells
                if island_coverage in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    logger.info(
                        "Island %d MAP-Elites coverage reached %.1f%% (%d/%d cells)",
                        island_idx,
                        island_coverage * 100,
                        len(island_feature_map) + 1,
                        total_possible_cells,
                    )
            else:

                existing_program_id = island_feature_map[feature_key]
                if existing_program_id in self.programs:
                    existing_program = self.programs[existing_program_id]
                    new_objective = get_primary_objective(program.metrics)
                    existing_objective = get_primary_objective(existing_program.metrics)
                    if new_objective is not None and existing_objective is not None:
                        logger.info(
                            "Island %d MAP-Elites cell improved: %s "
                            "(%s=%.3f [%s] -> %s=%.3f [%s])",
                            island_idx,
                            coords_dict,
                            existing_objective["name"],
                            existing_objective["value"],
                            existing_objective["direction"],
                            new_objective["name"],
                            new_objective["value"],
                            new_objective["direction"],
                        )
                    else:
                        new_fitness = get_fitness_score(
                            program.metrics, self.config.feature_dimensions
                        )
                        existing_fitness = get_fitness_score(
                            existing_program.metrics, self.config.feature_dimensions
                        )
                        logger.info(
                            "Island %d MAP-Elites cell improved: %s "
                            "(compatibility fitness: %.3f -> %.3f)",
                            island_idx,
                            coords_dict,
                            existing_fitness,
                            new_fitness,
                        )


                    if existing_program_id in self.archive:
                        self.archive.discard(existing_program_id)
                        self.archive.add(program.id)


                self.islands[island_idx].discard(existing_program_id)
                replaced_program_id = existing_program_id

            island_feature_map[feature_key] = program.id


        self.islands[island_idx].add(program.id)


        program.metadata["island"] = island_idx


        self._update_archive(program)


        self._enforce_population_limit(exclude_program_id=program.id)


        self._update_best_program(program)


        self._update_island_best_program(program, island_idx)


        if (
            replaced_program_id is not None
            and replaced_program_id != program.id
            and replaced_program_id != self.best_program_id
        ):
            self._remove_program_if_orphaned(replaced_program_id)


        if self.config.db_path:
            self._save_program(program)

        logger.debug(f"Added program {program.id} to island {island_idx}")

        return program.id

    def get(self, program_id: str) -> Optional[Program]:

        return self.programs.get(program_id)

    def sample(self, num_inspirations: Optional[int] = None) -> Tuple[Program, List[Program]]:

        if self._effective_outer_enabled():
            return self._sample_effective_parent(
                self.current_island,
                num_inspirations=num_inspirations,
            )


        parent = self._sample_parent()
        parent = self._maybe_ast_scheduler_parent(
            parent,
            list(self.programs.values()),
            self.current_island,
            "global",
        )


        if num_inspirations is None:
            num_inspirations = 5
        inspirations = self._sample_inspirations(parent, n=num_inspirations)

        logger.debug(f"Sampled parent {parent.id} and {len(inspirations)} inspirations")
        return parent, inspirations

    def sample_from_island(
        self, island_id: int, num_inspirations: Optional[int] = None
    ) -> Tuple[Program, List[Program]]:

        if self._effective_outer_enabled():
            return self._sample_effective_parent(
                island_id,
                num_inspirations=num_inspirations,
            )


        island_id = island_id % len(self.islands)


        island_programs = list(self.islands[island_id])

        if not island_programs:

            logger.debug(f"Island {island_id} is empty, sampling from all programs")
            return self.sample(num_inspirations)


        rand_val = random.random()

        if rand_val < self.config.exploration_ratio:

            parent = self._sample_from_island_random(island_id)
            sampling_mode = "exploration"
        elif rand_val < self.config.exploration_ratio + self.config.exploitation_ratio:

            parent = self._sample_from_archive_for_island(island_id)
            sampling_mode = "exploitation"
        else:

            parent = self._sample_from_island_weighted(island_id)
            sampling_mode = "weighted"

        island_candidates = [self.programs[pid] for pid in island_programs if pid in self.programs]
        parent = self._maybe_ast_scheduler_parent(
            parent,
            island_candidates,
            island_id,
            sampling_mode,
        )


        if num_inspirations is None:
            num_inspirations = 5
        inspirations = self._sample_inspirations(
            parent,
            n=num_inspirations,
            island_id=island_id,
        )

        logger.debug(
            f"Sampled parent {parent.id} and {len(inspirations)} inspirations from island {island_id} "
            f"(mode: {sampling_mode}, rand_val: {rand_val:.3f})"
        )
        return parent, inspirations

    def _sample_effective_parent(
        self,
        island_id: int,
        *,
        num_inspirations: Optional[int],
    ) -> Tuple[Program, List[Program]]:


        with self._outer_policy_lock:
            island_id = int(island_id) % len(self.islands)
            candidate_ids = sorted(
                pid for pid in self.islands[island_id] if pid in self.programs
            )
            if not candidate_ids:
                candidate_ids = sorted(
                    {
                        pid
                        for island in self.islands
                        for pid in island
                        if pid in self.programs
                    }
                    | {pid for pid in self.archive if pid in self.programs}
                )
            if not candidate_ids:
                raise ValueError("cannot sample an empty effective phenotype population")
            programs = [self.programs[pid] for pid in candidate_ids]
            candidates = [
                self._effective_candidate_for_program(program)
                for program in programs
            ]
            decision_index = self._next_outer_decision_index()
            if self._v9_population_policy_enabled():
                factors, ast_audit = self._effective_ast_sampling_factors(
                    programs, island_id
                )
                raw_decision = decide_parent(
                    candidates,
                    base_seed=self.config.random_seed,
                    namespace=f"island:{island_id}:parent",
                    decision_index=decision_index,
                    mode="mixture",
                    mixture=getattr(
                        self.config, "outer_parent_selection_mixture", None
                    ),
                    candidate_sampling_factors=factors,
                )
                decision = EvolutionDecision.create(
                    kind=raw_decision.kind,
                    namespace=raw_decision.namespace,
                    decision_index=raw_decision.decision_index,
                    derived_seed=raw_decision.derived_seed,
                    selected_candidate_id=raw_decision.selected_candidate_id,
                    reason=raw_decision.reason,
                    candidate_ids=raw_decision.candidate_ids,
                    eligible_ids=raw_decision.eligible_ids,
                    add_ids=raw_decision.add_ids,
                    remove_ids=raw_decision.remove_ids,
                    details={**raw_decision.details, "ast_scheduler": ast_audit},
                )
            else:
                ast_audit = {"applied": False, "reason": "legacy_policy"}
                decision = decide_parent(
                    candidates,
                    base_seed=self.config.random_seed,
                    namespace=f"island:{island_id}:parent",
                    decision_index=decision_index,
                    mode=getattr(
                        self.config, "outer_parent_selection_mode", "weighted"
                    ),
                )
            self._record_outer_decision(decision.to_dict())
            parent = self.programs[decision.selected_candidate_id]
            parent.metadata["last_outer_parent_selection"] = decision.to_dict()
            parent.metadata["island_role"] = self.get_island_role(island_id)
            parent.metadata["island_role_id"] = parent.metadata["island_role"]["role_id"]
            if ast_audit.get("applied"):
                parent.metadata["last_scheduler_parent_selection"] = {
                    "decision_hash": decision.decision_hash,
                    "phase": ast_audit.get("phase"),
                    "sampling_factor": (
                        ast_audit.get("sampling_factors", {}).get(parent.id)
                    ),
                    "island": island_id,
                }

            requested = 5 if num_inspirations is None else max(
                0, int(num_inspirations)
            )
            available = sorted(pid for pid in candidate_ids if pid != parent.id)
            selected_ids: List[str] = []
            if requested and available:
                inspiration_index = self._next_outer_decision_index()
                namespace = f"island:{island_id}:inspirations"
                seed = derive_private_seed(
                    self.config.random_seed,
                    namespace=namespace,
                    decision_index=inspiration_index,
                    candidate_ids=available,
                )
                rng = random.Random(seed)
                shuffled = list(available)
                rng.shuffle(shuffled)
                selected_ids = shuffled[:requested]
                inspiration_decision = EvolutionDecision.create(
                    kind="inspiration_selection",
                    namespace=namespace,
                    decision_index=inspiration_index,
                    derived_seed=seed,
                    selected_candidate_id=(selected_ids[0] if selected_ids else None),
                    reason="private_uniform_without_replacement",
                    candidate_ids=available,
                    eligible_ids=available,
                    add_ids=selected_ids,
                    details={
                        "requested": requested,
                        "selected_ids": selected_ids,
                    },
                )
                self._record_outer_decision(inspiration_decision.to_dict())
            return parent, [self.programs[pid] for pid in selected_ids]

    def get_best_program(self, metric: Optional[str] = None) -> Optional[Program]:

        if self._effective_outer_enabled():
            candidates = self._effective_selectable_programs()
            ordered = self._order_effective_programs(candidates, metric=metric)
            if not ordered:
                return None
            if metric is None:
                self.best_program_id = ordered[0].id
            return ordered[0]

        if not self.programs:
            return None


        if metric is None and self.best_program_id:
            if self.best_program_id in self.programs:
                logger.debug(f"Using tracked best program: {self.best_program_id}")
                return self.programs[self.best_program_id]
            else:
                logger.warning(
                    f"Tracked best program {self.best_program_id} no longer exists, will recalculate"
                )
                self.best_program_id = None

        if metric:

            metric_direction = get_metric_spec(metric).direction
            sorted_programs = sorted(
                [p for p in self.programs.values() if metric in p.metrics],
                key=lambda p: p.metrics[metric],
                reverse=metric_direction != "minimize",
            )
            if sorted_programs:
                logger.debug(f"Found best program by metric '{metric}': {sorted_programs[0].id}")
        else:

            sorted_programs = sorted(
                self.programs.values(),
                key=lambda p: get_fitness_score(p.metrics, self.config.feature_dimensions),
                reverse=True,
            )
            if sorted_programs:
                logger.debug(f"Found best program by fitness score: {sorted_programs[0].id}")


        if sorted_programs and (
            self.best_program_id is None or sorted_programs[0].id != self.best_program_id
        ):
            old_id = self.best_program_id
            self.best_program_id = sorted_programs[0].id
            logger.info(f"Updated best program tracking from {old_id} to {self.best_program_id}")


            old_objective = (
                get_primary_objective(self.programs[old_id].metrics)
                if old_id and old_id in self.programs
                else None
            )
            new_objective = get_primary_objective(
                self.programs[self.best_program_id].metrics
            )
            if old_objective is not None and new_objective is not None:
                logger.info(
                    "Objective change: %s=%.4f [%s] → %s=%.4f [%s]",
                    old_objective["name"],
                    old_objective["value"],
                    old_objective["direction"],
                    new_objective["name"],
                    new_objective["value"],
                    new_objective["direction"],
                )

        return sorted_programs[0] if sorted_programs else None

    def get_top_programs(
        self, n: int = 10, metric: Optional[str] = None, island_idx: Optional[int] = None
    ) -> List[Program]:


        if island_idx is not None and (island_idx < 0 or island_idx >= len(self.islands)):
            raise IndexError(f"Island index {island_idx} is out of range (0-{len(self.islands)-1})")

        if self._effective_outer_enabled():
            candidates = self._effective_selectable_programs(island_idx=island_idx)
            return self._order_effective_programs(candidates, metric=metric)[:n]

        if not self.programs:
            return []


        if island_idx is not None:

            island_programs = [
                self.programs[pid] for pid in self.islands[island_idx] if pid in self.programs
            ]
            candidates = island_programs
        else:

            candidates = list(self.programs.values())

        if not candidates:
            return []

        if metric:

            metric_direction = get_metric_spec(metric).direction
            sorted_programs = sorted(
                [p for p in candidates if metric in p.metrics],
                key=lambda p: p.metrics[metric],
                reverse=metric_direction != "minimize",
            )
        else:

            sorted_programs = sorted(
                candidates,
                key=lambda p: get_fitness_score(p.metrics, self.config.feature_dimensions),
                reverse=True,
            )

        return sorted_programs[:n]

    def _effective_selectable_programs(
        self, *, island_idx: Optional[int] = None
    ) -> List[Program]:
        if island_idx is not None:
            ids = set(self.islands[island_idx])
        else:
            ids = set(self.archive)
            ids.update(pid for island in self.islands for pid in island)
        return [self.programs[pid] for pid in sorted(ids) if pid in self.programs]

    def _order_effective_programs(
        self,
        programs: List[Program],
        *,
        metric: Optional[str] = None,
    ) -> List[Program]:
        if metric is not None:
            programs = [program for program in programs if metric in program.metrics]
        if not programs:
            return []
        rows = []
        for program in programs:
            candidate = self._effective_candidate_for_program(program)
            rows.append(
                {
                    "candidate_id": program.id,
                    "raw_objective": (
                        float(program.metrics[metric])
                        if metric is not None
                        else candidate.objective
                    ),
                    "gate_sources": candidate.gate_sources,
                }
            )
        direction = (
            get_metric_spec(metric).direction
            if metric is not None
            else "maximize"
        )
        if direction not in {"maximize", "minimize"}:
            direction = "maximize"
        decision = select_feasibility_first(rows, direction=direction)
        by_id = {program.id: program for program in programs}
        return [by_id[program_id] for program_id in decision["ordered_ids"]]

    def attach_optimizer_memory(self, optimizer_memory) -> None:

        self.optimizer_memory = optimizer_memory

    def _next_outer_decision_index(self) -> int:
        with self._outer_policy_lock:
            value = self.outer_decision_counter
            self.outer_decision_counter += 1
            return value

    def _record_outer_decision(self, artifact: Mapping[str, Any]) -> None:
        if not getattr(self.config, "outer_decision_artifacts_enabled", True):
            return
        if not isinstance(artifact, Mapping):
            raise TypeError("outer decision artifact must be a mapping")
        with self._outer_policy_lock:
            self.outer_decisions.append(json.loads(json.dumps(dict(artifact))))

    def get_outer_decisions(self) -> List[Dict[str, Any]]:
        with self._outer_policy_lock:
            return json.loads(json.dumps(self.outer_decisions))

    def append_memory_facts_batch(
        self, facts: List[OuterObservationFact]
    ) -> List[OuterObservationFact]:


        incoming = list(facts or [])
        if any(not isinstance(fact, OuterObservationFact) for fact in incoming):
            raise TypeError("memory fact ledger accepts only OuterObservationFact values")


        ordered = sorted(
            (
                OuterObservationFact.from_mapping(fact.to_dict())
                for fact in incoming
            ),
            key=lambda fact: fact.stable_key,
        )
        existing = self._memory_fact_index
        pending: Dict[
            Tuple[str, str, str, str, Optional[str]], OuterObservationFact
        ] = {}
        additions: List[OuterObservationFact] = []
        for fact in ordered:
            key = (
                fact.scope.case_id,
                fact.scope.run_id,
                fact.generation_id,
                fact.proposal_id,
                fact.trial_id,
            )
            previous = pending.get(key) or existing.get(key)
            if previous is not None:
                if previous.to_dict() != fact.to_dict():
                    raise ValueError(
                        "conflicting outer memory fact for "
                        f"generation={fact.generation_id!r}, proposal={fact.proposal_id!r}"
                    )
                continue
            pending[key] = fact
            additions.append(fact)
        additions.sort(key=lambda fact: fact.chronology_key)
        if additions and self.memory_observation_facts:
            last_key = self.memory_observation_facts[-1].chronology_key
            regressed = next(
                (
                    fact
                    for fact in additions
                    if fact.chronology_key < last_key
                ),
                None,
            )
            if regressed is not None:
                raise ValueError(
                    "outer memory fact chronology regression: "
                    f"incoming={regressed.chronology_key!r}, last={last_key!r}"
                )
        self.memory_observation_facts.extend(additions)
        existing.update(pending)
        return [
            OuterObservationFact.from_mapping(fact.to_dict())
            for fact in additions
        ]

    def get_memory_facts(self, scope: MemoryScope) -> List[OuterObservationFact]:


        if not isinstance(scope, MemoryScope):
            raise TypeError("scope must be MemoryScope")
        selected = []
        for fact in self.memory_observation_facts:
            try:
                scope.require_compatible(fact.scope, level="run")
            except MemoryPolicyError:
                continue
            selected.append(OuterObservationFact.from_mapping(fact.to_dict()))
        return selected

    def get_optimizer_memory_snapshot(self) -> Dict[str, Any]:

        if self.optimizer_memory is None:
            return {}
        try:
            snapshot = self.optimizer_memory.snapshot()
            if self._ast_scheduler_enabled():
                snapshot["scheduler"] = self.get_scheduler_signal()
            return snapshot
        except Exception as exc:
            logger.warning("Failed to snapshot optimizer memory: %s", exc)
            return {}

    def record_optimizer_memory(
        self,
        program: Program,
        parent: Optional[Program] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        iteration: Optional[int] = None,
    ) -> None:

        published = self.publish_optimizer_memory_facts(
            [
                {
                    "proposal_id": program.id,
                    "program": program,
                    "parent": parent,
                    "artifacts": artifacts or {},
                    "iteration": iteration,
                    "error": None,
                }
            ],
            generation_id=str(
                (program.metadata or {}).get("generation_id") or "initial"
            ),
            logical_time=f"iteration:{int(iteration or 0):08d}",
        )
        if published is not None:
            return
        self.record_optimizer_memory_batch(
            [
                {
                    "proposal_id": program.id,
                    "program": program,
                    "parent": parent,
                    "artifacts": artifacts or {},
                    "iteration": iteration,
                    "error": None,
                }
            ]
        )

    def publish_optimizer_memory_facts(
        self,
        observations: List[Dict[str, Any]],
        *,
        generation_id: str,
        logical_time: str,
    ) -> Optional[List[OuterObservationFact]]:


        store = self.optimizer_memory
        scope = getattr(store, "scope", None) if store is not None else None
        if store is None or not isinstance(scope, MemoryScope):
            return None
        facts = [
            build_observation_fact(
                scope=scope,
                generation_id=str(generation_id),
                proposal_id=str(row.get("proposal_id") or ""),
                logical_time=str(logical_time),
                iteration=int(row.get("iteration") or 0),
                program=row.get("program"),
                parent=row.get("parent"),
                artifacts=row.get("artifacts") or {},
                error=(str(row["error"]) if row.get("error") else None),
            )
            for row in sorted(
                observations or [],
                key=lambda item: (
                    int(item.get("iteration") or 0),
                    str(item.get("proposal_id") or ""),
                ),
            )
        ]
        previous_hash = store.state.get("source_fact_ledger_hash")
        additions = self.append_memory_facts_batch(facts)
        extended = False
        if hasattr(store, "extend_from_facts"):
            extended = bool(
                store.extend_from_facts(
                    additions,
                    scope=scope,
                    expected_source_fact_ledger_hash=previous_hash,
                    persist=True,
                )
            )
        if not extended:
            store.rebuild_from_facts(
                self.get_memory_facts(scope),
                scope=scope,
                persist=True,
            )
        return additions

    def record_optimizer_memory_batch(
        self,
        observations: List[Dict[str, Any]],
        *,
        logical_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:


        if self.optimizer_memory is None:
            return []
        try:
            top_programs = self.get_top_programs(
                getattr(self.config, "optimizer_memory_best_limit", 5)
            )
            if hasattr(self.optimizer_memory, "record_generation_observations"):
                return self.optimizer_memory.record_generation_observations(
                    observations,
                    best_candidates=top_programs,
                    logical_time=logical_time,
                )


            entries = []
            for row in sorted(
                observations or [],
                key=lambda item: (
                    int(item.get("iteration") or 0),
                    str(item.get("proposal_id") or ""),
                ),
            ):
                if row.get("program") is None:
                    continue
                try:
                    entry = self.optimizer_memory.record_result(
                        program=row["program"],
                        parent=row.get("parent"),
                        artifacts=row.get("artifacts") or {},
                        iteration=row.get("iteration"),
                        persist=False,
                    )
                except TypeError:
                    entry = self.optimizer_memory.record_result(
                        program=row["program"],
                        parent=row.get("parent"),
                        artifacts=row.get("artifacts") or {},
                        iteration=row.get("iteration"),
                    )
                entries.append(entry)
            try:
                self.optimizer_memory.refresh_best_candidates(top_programs, persist=False)
            except TypeError:
                self.optimizer_memory.refresh_best_candidates(top_programs)
            self.optimizer_memory.save()
            return entries
        except Exception as exc:
            logger.warning("Failed to update optimizer memory: %s", exc)
            return []

    def get_scheduler_signal(self) -> Dict[str, Any]:

        if not self._ast_scheduler_enabled():
            return {}
        try:
            if hasattr(self.optimizer_memory, "scheduler_signal"):
                return self.optimizer_memory.scheduler_signal(
                    stagnation_window=getattr(self.config, "ast_scheduler_stagnation_window", 5),
                    stagnation_min_delta=getattr(
                        self.config, "ast_scheduler_stagnation_min_delta", 1e-6
                    ),
                )
            snapshot = self.optimizer_memory.snapshot()
            return snapshot.get("scheduler", {}) if isinstance(snapshot, dict) else {}
        except Exception as exc:
            logger.warning("Failed to build AST scheduler signal: %s", exc)
            return {}

    def _ast_scheduler_enabled(self) -> bool:
        return bool(getattr(self.config, "ast_scheduler_enabled", True)) and self.optimizer_memory is not None

    def _maybe_ast_scheduler_parent(
        self,
        fallback: Program,
        candidates: List[Program],
        island_idx: Optional[int],
        sampling_mode: str,
    ) -> Program:
        if not self._ast_scheduler_enabled() or not candidates:
            return fallback

        weight = max(0.0, min(1.0, float(getattr(self.config, "ast_parent_selection_weight", 0.35))))
        if sampling_mode == "exploration" and random.random() > weight:
            return fallback

        selected = self._sample_ast_aware_from_candidates(candidates, island_idx, sampling_mode)
        return selected or fallback

    def _sample_ast_aware_from_candidates(
        self,
        candidates: List[Program],
        island_idx: Optional[int],
        sampling_mode: str,
    ) -> Optional[Program]:
        signal = self.get_scheduler_signal()
        if not signal:
            return None

        unique: Dict[str, Program] = {
            program.id: program for program in candidates if program and program.id in self.programs
        }
        if not unique:
            return None

        island_profile = self._island_profile(island_idx, signal)
        scored = [
            (self._ast_scheduler_score(program, signal, island_profile), program)
            for program in unique.values()
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        pool_size = max(1, min(len(scored), max(3, int(len(scored) * 0.4))))
        pool = scored[:pool_size]
        min_score = min(score for score, _ in pool)
        weights = [max(score - min_score + 0.001, 0.001) for score, _ in pool]
        selected = random.choices([program for _, program in pool], weights=weights, k=1)[0]

        selected.metadata["last_scheduler_parent_selection"] = {
            "mode": sampling_mode,
            "island": island_idx,
            "island_profile": island_profile.get("name"),
            "scheduler_phase": signal.get("phase"),
        }
        logger.debug(
            "AST scheduler selected parent %s for island %s (%s)",
            selected.id[:8],
            island_idx,
            island_profile.get("name"),
        )
        return selected

    def _ast_scheduler_score(
        self,
        program: Program,
        signal: Dict[str, Any],
        island_profile: Dict[str, Any],
    ) -> float:
        base = get_fitness_score(program.metrics, self.config.feature_dimensions)
        weight = max(0.0, min(1.0, float(getattr(self.config, "ast_parent_selection_weight", 0.35))))
        return base + weight * self._ast_scheduler_bonus(
            program, signal, island_profile
        )

    def _ast_scheduler_bonus(
        self,
        program: Program,
        signal: Dict[str, Any],
        island_profile: Dict[str, Any],
    ) -> float:


        profile = self._program_ast_profile(program)
        categories = set(profile.get("categories", []))
        island_categories = set(island_profile.get("categories", []))

        bonus = 0.0
        if categories & island_categories:
            bonus += 0.45
        if signal.get("phase") == "escape_stagnation" and categories:
            bonus += 0.15

        weak_functional = self._scheduler_node_names(signal.get("weak_functional_nodes", []))
        strong_functional = set(map(str, profile.get("strong_functional_nodes", [])))
        covered_functional = set(map(str, profile.get("weak_functional_nodes", [])))
        bonus += 0.22 * len(strong_functional & weak_functional)
        bonus += 0.08 * len(covered_functional & weak_functional)

        weak_structural = self._scheduler_node_names(signal.get("weak_structural_nodes", []))
        successful_structural = set(map(str, profile.get("successful_structural_nodes", [])))
        touched_structural = set(map(str, profile.get("weak_structural_nodes", [])))
        bonus += 0.20 * len(successful_structural & weak_structural)
        bonus += 0.06 * len(touched_structural & weak_structural)

        if signal.get("contract_attention", {}).get("needs_response"):
            bonus += 0.10 if profile.get("contract_responsive", True) else -0.25

        specialist_score = profile.get("specialist_score", 0.0)
        if isinstance(specialist_score, (int, float)):
            bonus += min(max(float(specialist_score), 0.0), 1.0) * 0.10

        return bonus

    def _effective_ast_sampling_factors(
        self, programs: List[Program], island_idx: int
    ) -> Tuple[Optional[Dict[str, float]], Dict[str, Any]]:


        if not programs:
            return None, {"applied": False, "reason": "no_candidates"}

        role = self.get_island_role(island_idx)
        role_bonuses = {
            program.id: self._island_role_sampling_bonus(program, role)
            for program in programs
        }
        role_preferences = self._normalize_sampling_preferences(role_bonuses)
        role_strength = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(self.config, "island_role_selection_weight", 0.35)
                ),
            ),
        )

        signal = self.get_scheduler_signal() if self._ast_scheduler_enabled() else {}
        profile = self._island_profile(island_idx, signal or {})
        ast_bonuses = {
            program.id: self._ast_scheduler_bonus(program, signal, profile)
            for program in programs
        } if signal else {program.id: 0.0 for program in programs}
        ast_preferences = self._normalize_sampling_preferences(ast_bonuses)
        ast_strength = (
            max(
                0.0,
                min(
                    1.0,
                    float(
                        getattr(self.config, "ast_parent_selection_weight", 0.35)
                    ),
                ),
            )
            if signal
            else 0.0
        )


        total_strength = role_strength + ast_strength
        if total_strength > 1.0:
            role_strength /= total_strength
            ast_strength /= total_strength
            total_strength = 1.0
        base = 1.0 - total_strength
        factors = {
            program.id: max(
                1.0e-12,
                base
                + role_strength * role_preferences[program.id]
                + ast_strength * ast_preferences[program.id],
            )
            for program in programs
        }
        return factors, {
            "applied": bool(role_strength or ast_strength),
            "phase": signal.get("phase") if signal else None,
            "island_role_id": role.get("role_id"),
            "island_role": role,
            "role_strength": role_strength,
            "ast_strength": ast_strength,
            "raw_role_bonus": role_bonuses,
            "raw_ast_bonus": ast_bonuses,
            "sampling_factors": factors,
        }

    @staticmethod
    def _normalize_sampling_preferences(
        bonuses: Mapping[str, float]
    ) -> Dict[str, float]:
        if not bonuses:
            return {}
        low = min(bonuses.values())
        high = max(bonuses.values())
        if high == low:
            return {program_id: 1.0 for program_id in bonuses}
        return {
            program_id: (bonus - low) / (high - low)
            for program_id, bonus in bonuses.items()
        }

    @staticmethod
    def _island_role_sampling_bonus(
        program: Program, island_role: Mapping[str, Any]
    ) -> float:


        metrics = program.metrics or {}
        metadata = program.metadata or {}

        def metric(name: str, default: float = 0.0) -> float:
            value = metrics.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                return default
            return float(value)

        def unit(name: str) -> float:
            return max(0.0, min(1.0, metric(name)))

        def plddt(name: str) -> float:
            value = metric(name)
            if value > 1.0:
                value /= 100.0
            return max(0.0, min(1.0, value))

        def signed_margin(name: str) -> float:


            return 0.5 + 0.5 * math.tanh(5.0 * metric(name))

        bins = dict(
            ((metadata.get("outer_behavior_descriptor") or {}).get("bins") or {})
        )
        scope_scores = {
            "none": 0.0,
            "single": 0.25,
            "narrow": 0.25,
            "low": 0.35,
            "partial": 0.6,
            "moderate": 0.6,
            "medium": 0.7,
            "broad": 1.0,
            "high": 1.0,
            "full": 1.0,
            "seen": 0.0,
            "novel": 1.0,
        }

        def behavior(name: str) -> float:
            return scope_scores.get(str(bins.get(name, "none")).lower(), 0.0)

        metric_terms = island_role.get("metric_terms")
        if metric_terms:
            if (
                not isinstance(metric_terms, Sequence)
                or isinstance(metric_terms, (str, bytes))
            ):
                raise ValueError("island role metric_terms must be a sequence")

            expected_fields = {
                "metric",
                "direction",
                "transform",
                "weight",
                "missing_policy",
            }
            weighted_total = 0.0
            declared_weight = 0.0
            for index, raw_term in enumerate(metric_terms):
                if not isinstance(raw_term, Mapping):
                    raise ValueError(
                        f"island role metric_terms[{index}] must be a mapping"
                    )
                unexpected = set(raw_term) - expected_fields
                missing_fields = expected_fields - set(raw_term)
                if unexpected or missing_fields:
                    raise ValueError(
                        "island role metric term fields must be exactly "
                        "metric, direction, transform, weight, missing_policy"
                    )

                metric_name = str(raw_term["metric"]).strip()
                direction = str(raw_term["direction"]).strip().lower()
                transform = str(raw_term["transform"]).strip().lower()
                missing_policy = str(raw_term["missing_policy"]).strip().lower()
                raw_weight = raw_term["weight"]
                if not metric_name:
                    raise ValueError("island role metric name cannot be empty")
                if direction not in {"maximize", "minimize"}:
                    raise ValueError(
                        "island role metric direction must be maximize or minimize"
                    )
                if transform not in {
                    "identity",
                    "one_minus",
                    "clamped_inverse",
                }:
                    raise ValueError(
                        "island role metric transform must be identity, "
                        "one_minus, or clamped_inverse"
                    )
                if missing_policy not in {"fail", "neutral"}:
                    raise ValueError(
                        "island role metric missing_policy must be fail or neutral"
                    )
                if (
                    isinstance(raw_weight, bool)
                    or not isinstance(raw_weight, (int, float))
                    or not math.isfinite(float(raw_weight))
                    or not 0.0 <= float(raw_weight) <= 1.0
                ):
                    raise ValueError(
                        "island role metric weight must be finite and in [0, 1]"
                    )
                weight = float(raw_weight)
                if weight == 0.0:
                    continue

                raw_value = metrics.get(metric_name)
                available = (
                    not isinstance(raw_value, bool)
                    and isinstance(raw_value, (int, float))
                    and math.isfinite(float(raw_value))
                )
                if not available:
                    utility = 0.0 if missing_policy == "fail" else 0.5
                else:
                    value = float(raw_value)
                    if transform == "identity":
                        normalized = max(0.0, min(1.0, value))
                    elif transform == "one_minus":
                        normalized = max(0.0, min(1.0, 1.0 - value))
                    else:
                        normalized = 1.0 / (1.0 + max(0.0, value))
                    utility = (
                        normalized
                        if direction == "maximize"
                        else 1.0 - normalized
                    )
                weighted_total += weight * utility
                declared_weight += weight

            if declared_weight <= 0.0:
                raise ValueError(
                    "island role metric_terms must have positive total weight"
                )
            return max(
                0.0,
                min(1.0, weighted_total / declared_weight),
            )

        role_id = str(island_role.get("role_id", ""))
        if role_id == "a_interface_fold":
            values = (
                unit("positive_A_iptm"),
                unit("positive_A_interface_q"),
                plddt("positive_A_plddt"),
                plddt("apo_plddt"),
                unit("multistate_score"),
            )
        elif role_id == "selectivity_margin":
            values = (
                unit("selectivity_proxy_score"),
                signed_margin("iptm_margin"),
                signed_margin("gpde_margin"),
                signed_margin("interface_q_margin"),
                unit("positive_A_interface_q"),
            )
        elif role_id == "region_exploration":
            values = (
                behavior("structural_scope"),
                behavior("functional_scope"),
                behavior("action_coverage"),
                behavior("strategy_novelty"),
                behavior("sequence_novelty"),
            )
        elif role_id == "robustness_safety":
            values = (
                unit("worst_case_score"),
                max(0.0, 1.0 - unit("seed_score_std")),
                1.0 / (1.0 + max(0.0, metric("clash_count"))),
                plddt("node_plddt_min"),
                unit("positive_noninferiority_min_ratio"),
            )
        else:
            values = (unit("multistate_score"),)
        return sum(values) / max(1, len(values))

    def _order_island_role_programs(
        self, programs: Sequence[Program], island_idx: int
    ) -> List[Program]:


        values = list(programs)
        if not values:
            return []
        global_order = self._order_effective_programs(values)
        global_rank = {
            program.id: index for index, program in enumerate(global_order)
        }
        role = self.get_island_role(island_idx)
        return sorted(
            values,
            key=lambda program: (
                not self._effective_candidate_for_program(program).feasible,
                -self._island_role_sampling_bonus(program, role),
                global_rank[program.id],
                program.id,
            ),
        )

    def _program_ast_profile(self, program: Program) -> Dict[str, Any]:
        if self.optimizer_memory is not None and hasattr(self.optimizer_memory, "program_scheduler_profile"):
            try:
                return self.optimizer_memory.program_scheduler_profile(program.id, program.metrics)
            except Exception as exc:
                logger.debug("Could not read optimizer program profile for %s: %s", program.id, exc)

        categories = set()
        metric_names = {str(key).lower() for key in program.metrics.keys()}
        if metric_names & {"struct_score", "structure_score", "plddt", "ptm", "hard_gate_pass"}:
            categories.add("fold_stability")
        if any("interface" in name or "contact" in name or name == "iptm" for name in metric_names):
            categories.add("interface_contact")
        if any("specificity" in name or "negative" in name or "selectivity" in name for name in metric_names):
            categories.add("specificity_negative_design")
        if any("pocket" in name or "ligand" in name or "alloster" in name for name in metric_names):
            categories.add("allostery_pocket")
        return {
            "categories": sorted(categories),
            "strong_functional_nodes": [],
            "weak_functional_nodes": [],
            "successful_structural_nodes": [],
            "weak_structural_nodes": [],
            "contract_responsive": True,
            "specialist_score": get_fitness_score(program.metrics, self.config.feature_dimensions),
        }

    def _scheduler_node_names(self, nodes: Any) -> Set[str]:
        if not nodes:
            return set()
        if isinstance(nodes, dict):
            nodes = nodes.keys()
        names = set()
        for item in nodes:
            if isinstance(item, dict):
                value = item.get("name")
            else:
                value = item
            if value is not None:
                names.add(str(value))
        return names

    def _island_profile(
        self, island_idx: Optional[int], signal: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        signal = signal if isinstance(signal, dict) else self.get_scheduler_signal()
        configured = getattr(self.config, "island_roles", None) or []
        if configured:
            profile = self.get_island_role(island_idx)
            profile.setdefault("categories", [profile["role_id"]])
            return profile
        profiles = signal.get("island_profiles", []) if isinstance(signal, dict) else []
        if not profiles:
            profiles = [
                {"name": "fold_stability", "categories": ["fold_stability"]},
                {"name": "interface_contact", "categories": ["interface_contact"]},
                {"name": "specificity_negative_design", "categories": ["specificity_negative_design"]},
                {"name": "allostery_pocket", "categories": ["allostery_pocket"]},
            ]
        if island_idx is None:
            island_idx = self.current_island
        profile = dict(profiles[island_idx % len(profiles)])
        profile["island"] = island_idx
        return profile

    def save(self, path: Optional[str] = None, iteration: int = 0) -> None:

        save_path = path or self.config.db_path
        if not save_path:
            logger.warning("No database path specified, skipping save")
            return


        self._cleanup_old_artifacts(save_path)


        os.makedirs(save_path, exist_ok=True)


        for program in self.programs.values():
            prompts = None
            if (
                self.config.log_prompts
                and self.prompts_by_program
                and program.id in self.prompts_by_program
            ):
                prompts = self.prompts_by_program[program.id]
            self._save_program(program, save_path, prompts=prompts)


        metadata = {
            "island_feature_maps": self.island_feature_maps,
            "islands": [list(island) for island in self.islands],
            "archive": list(self.archive),
            "best_program_id": self.best_program_id,
            "island_best_programs": self.island_best_programs,
            "last_iteration": iteration or self.last_iteration,
            "current_island": self.current_island,
            "island_generations": self.island_generations,
            "last_migration_generation": self.last_migration_generation,
            "last_migration_iteration": self.last_migration_iteration,
            "last_migration_island_generations": self.last_migration_island_generations,
            "feature_stats": self._serialize_feature_stats(),
            "memory_observation_facts": [
                fact.to_dict() for fact in self.memory_observation_facts
            ],
            "outer_decisions": list(self.outer_decisions),
            "outer_decision_counter": self.outer_decision_counter,
            "effective_phenotype_index": dict(self.effective_phenotype_index),
            "migration_receipts": list(self.migration_receipts),
            "map_cell_collision_count": self.map_cell_collision_count,
            "map_cell_replacement_count": self.map_cell_replacement_count,
        }

        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        if self.optimizer_memory is not None:
            try:
                self.optimizer_memory.save(os.path.join(save_path, "optimizer_memory.json"))
            except Exception as exc:
                logger.warning("Failed to save optimizer memory: %s", exc)

        logger.info(f"Saved database with {len(self.programs)} programs to {save_path}")

    def load(self, path: str) -> None:

        if not os.path.exists(path):
            logger.warning(f"Database path {path} does not exist, skipping load")
            return


        metadata_path = os.path.join(path, "metadata.json")
        saved_islands = []
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.island_feature_maps = metadata.get(
                "island_feature_maps", [{} for _ in range(self.config.num_islands)]
            )
            saved_islands = metadata.get("islands", [])
            self.archive = set(metadata.get("archive", []))
            self.best_program_id = metadata.get("best_program_id")
            self.island_best_programs = metadata.get(
                "island_best_programs", [None] * len(saved_islands)
            )
            self.last_iteration = metadata.get("last_iteration", 0)
            self.current_island = metadata.get("current_island", 0)
            self.island_generations = metadata.get("island_generations", [0] * len(saved_islands))
            self.last_migration_generation = metadata.get("last_migration_generation", 0)
            self.last_migration_iteration = metadata.get(
                "last_migration_iteration", 0
            )
            self.last_migration_island_generations = metadata.get(
                "last_migration_island_generations",
                [0] * len(saved_islands),
            )
            self.migration_receipts = list(metadata.get("migration_receipts", []))
            self.map_cell_collision_count = int(
                metadata.get("map_cell_collision_count", 0) or 0
            )
            self.map_cell_replacement_count = int(
                metadata.get("map_cell_replacement_count", 0) or 0
            )
            self.memory_observation_facts = [
                OuterObservationFact.from_mapping(item)
                for item in metadata.get("memory_observation_facts", [])
            ]
            self.memory_observation_facts.sort(
                key=lambda fact: fact.chronology_key
            )
            self._memory_fact_index = {
                (
                    fact.scope.case_id,
                    fact.scope.run_id,
                    fact.generation_id,
                    fact.proposal_id,
                    fact.trial_id,
                ): fact
                for fact in self.memory_observation_facts
            }
            raw_outer_decisions = metadata.get("outer_decisions", [])
            raw_outer_counter = metadata.get(
                "outer_decision_counter", len(raw_outer_decisions)
            )
            if self._effective_outer_enabled():
                try:
                    validated_decisions = [
                        EvolutionDecision.from_mapping(item).to_dict()
                        for item in raw_outer_decisions
                    ]
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"outer decision ledger is invalid: {exc}"
                    ) from exc
                indexes = [
                    int(item["decision_index"]) for item in validated_decisions
                ]
                if indexes != sorted(indexes) or len(indexes) != len(set(indexes)):
                    raise ValueError(
                        "outer decision ledger has duplicate or regressed indexes"
                    )
                if (
                    isinstance(raw_outer_counter, bool)
                    or not isinstance(raw_outer_counter, int)
                    or raw_outer_counter < 0
                    or (indexes and raw_outer_counter <= indexes[-1])
                ):
                    raise ValueError(
                        "outer decision counter does not follow the persisted ledger"
                    )
                self.outer_decisions = validated_decisions
                self.outer_decision_counter = raw_outer_counter
            else:
                self.outer_decisions = list(raw_outer_decisions)
                self.outer_decision_counter = int(raw_outer_counter)
            self.effective_phenotype_index = {
                str(key): str(value)
                for key, value in (
                    metadata.get("effective_phenotype_index", {}) or {}
                ).items()
            }


            self.feature_stats = self._deserialize_feature_stats(metadata.get("feature_stats", {}))

            logger.info(f"Loaded database metadata with last_iteration={self.last_iteration}")
            if self.feature_stats:
                logger.info(f"Loaded feature_stats for {len(self.feature_stats)} dimensions")


        programs_dir = os.path.join(path, "programs")
        if os.path.exists(programs_dir):
            for program_file in os.listdir(programs_dir):
                if program_file.endswith(".json"):
                    program_path = os.path.join(programs_dir, program_file)
                    try:
                        with open(program_path, "r") as f:
                            program_data = json.load(f)

                        program = Program.from_dict(program_data)
                        self.programs[program.id] = program
                    except Exception as e:
                        logger.warning(f"Error loading program {program_file}: {str(e)}")


        self._reconstruct_islands(saved_islands)


        if len(self.island_generations) != len(self.islands):
            self.island_generations = [0] * len(self.islands)
        if len(self.last_migration_island_generations) != len(self.islands):
            self.last_migration_island_generations = [0] * len(self.islands)


        if len(self.island_best_programs) != len(self.islands):
            self.island_best_programs = [None] * len(self.islands)

        logger.info(f"Loaded database with {len(self.programs)} programs from {path}")

        if self.optimizer_memory is not None:
            memory_path = os.path.join(path, "optimizer_memory.json")
            if os.path.exists(memory_path):
                try:
                    self.optimizer_memory.load(memory_path)
                except Exception as exc:
                    logger.warning("Failed to load optimizer memory: %s", exc)


        self.log_island_status()

    def _reconstruct_islands(self, saved_islands: List[List[str]]) -> None:


        num_islands = max(len(saved_islands), self.config.num_islands)
        self.islands = [set() for _ in range(num_islands)]

        missing_programs = []
        restored_programs = 0


        for island_idx, program_ids in enumerate(saved_islands):
            if island_idx >= len(self.islands):
                continue

            for program_id in program_ids:
                if program_id in self.programs:

                    self.islands[island_idx].add(program_id)
                    restored_programs += 1
                else:

                    missing_programs.append((island_idx, program_id))


        original_archive_size = len(self.archive)
        self.archive = {pid for pid in self.archive if pid in self.programs}


        feature_keys_to_remove = []
        for island_idx, island_map in enumerate(self.island_feature_maps):
            island_keys_to_remove = []
            for key, program_id in island_map.items():
                if program_id not in self.programs:
                    island_keys_to_remove.append(key)
                    feature_keys_to_remove.append((island_idx, key))
            for key in island_keys_to_remove:
                del island_map[key]


        self._cleanup_stale_island_bests()


        if self.best_program_id and self.best_program_id not in self.programs:
            logger.warning(f"Best program {self.best_program_id} not found, will recalculate")
            self.best_program_id = None

        if self._effective_outer_enabled():
            membership_ids = set(self.archive)
            membership_ids.update(pid for island in self.islands for pid in island)
            for program_id in sorted(membership_ids):
                self._refresh_program_membership(program_id)
            self._refresh_effective_phenotype_index()


        if missing_programs:
            logger.warning(
                f"Found {len(missing_programs)} missing programs during island reconstruction:"
            )
            for island_idx, program_id in missing_programs[:5]:
                logger.warning(f"  Island {island_idx}: {program_id}")
            if len(missing_programs) > 5:
                logger.warning(f"  ... and {len(missing_programs) - 5} more")

        if original_archive_size > len(self.archive):
            logger.info(
                f"Removed {original_archive_size - len(self.archive)} missing programs from archive"
            )

        if feature_keys_to_remove:
            logger.info(
                f"Removed {len(feature_keys_to_remove)} missing programs from island feature maps"
            )

        logger.info(f"Reconstructed islands: restored {restored_programs} programs to islands")


        if self.programs and sum(len(island) for island in self.islands) == 0:
            logger.info("No island assignments found, distributing programs across islands")
            self._distribute_programs_to_islands()

    def _distribute_programs_to_islands(self) -> None:

        program_ids = list(self.programs.keys())


        for i, program_id in enumerate(program_ids):
            island_idx = i % len(self.islands)
            self.islands[island_idx].add(program_id)
            self.programs[program_id].metadata["island"] = island_idx

        logger.info(f"Distributed {len(program_ids)} programs across {len(self.islands)} islands")

    def _save_program(
        self,
        program: Program,
        base_path: Optional[str] = None,
        prompts: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:

        save_path = base_path or self.config.db_path
        if not save_path:
            return


        programs_dir = os.path.join(save_path, "programs")
        os.makedirs(programs_dir, exist_ok=True)


        program_dict = program.to_dict()
        if prompts:
            program_dict["prompts"] = prompts
        program_path = os.path.join(programs_dir, f"{program.id}.json")

        with open(program_path, "w") as f:
            json.dump(program_dict, f)

    def _calculate_feature_coords(self, program: Program) -> List[int]:

        coords = []

        for dim in self.config.feature_dimensions:


            if dim in program.metrics:

                score = program.metrics[dim]

                self._update_feature_stats(dim, score)
                scaled_value = self._scale_feature_value(dim, score)
                num_bins = self.feature_bins_per_dim.get(dim, self.feature_bins)
                bin_idx = int(scaled_value * num_bins)
                bin_idx = max(0, min(num_bins - 1, bin_idx))
                coords.append(bin_idx)

            elif dim == "complexity":

                complexity = len(program.code)
                bin_idx = self._calculate_complexity_bin(complexity)
                coords.append(bin_idx)
            elif dim == "diversity":

                if len(self.programs) < 2:
                    bin_idx = 0
                else:
                    diversity = self._get_cached_diversity(program)
                    bin_idx = self._calculate_diversity_bin(diversity)
                coords.append(bin_idx)
            elif dim == "score":

                if not program.metrics:
                    bin_idx = 0
                else:

                    avg_score = get_fitness_score(program.metrics, self.config.feature_dimensions)

                    self._update_feature_stats("score", avg_score)
                    scaled_value = self._scale_feature_value("score", avg_score)
                    num_bins = self.feature_bins_per_dim.get("score", self.feature_bins)
                    bin_idx = int(scaled_value * num_bins)
                    bin_idx = max(0, min(num_bins - 1, bin_idx))
                coords.append(bin_idx)
            else:

                raise ValueError(
                    f"Feature dimension '{dim}' specified in config but not found in program metrics. "
                    f"Available metrics: {list(program.metrics.keys())}. "
                    f"Built-in features: 'complexity', 'diversity', 'score'. "
                    f"Either remove '{dim}' from feature_dimensions or ensure your evaluator returns it."
                )

        logger.debug(
            "MAP-Elites coords: %s",
            str({self.config.feature_dimensions[i]: coords[i] for i in range(len(coords))}),
        )
        return coords

    def _calculate_complexity_bin(self, complexity: int) -> int:


        self._update_feature_stats("complexity", float(complexity))


        scaled_value = self._scale_feature_value("complexity", float(complexity))


        num_bins = self.feature_bins_per_dim.get("complexity", self.feature_bins)


        bin_idx = int(scaled_value * num_bins)


        bin_idx = max(0, min(num_bins - 1, bin_idx))

        return bin_idx

    def _calculate_diversity_bin(self, diversity: float) -> int:


        self._update_feature_stats("diversity", diversity)


        scaled_value = self._scale_feature_value("diversity", diversity)


        num_bins = self.feature_bins_per_dim.get("diversity", self.feature_bins)


        bin_idx = int(scaled_value * num_bins)


        bin_idx = max(0, min(num_bins - 1, bin_idx))

        return bin_idx

    def _feature_coords_to_key(self, coords: List[int]) -> str:

        return "-".join(str(c) for c in coords)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:

        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        arr1 = np.array(vec1, dtype=np.float32)
        arr2 = np.array(vec2, dtype=np.float32)

        norm_a = np.linalg.norm(arr1)
        norm_b = np.linalg.norm(arr2)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = np.dot(arr1, arr2) / (norm_a * norm_b)

        return float(similarity)

    def _llm_judge_novelty(self, program: Program, similar_program: Program) -> bool:

        import asyncio
        from outerloop.novelty_judge import NOVELTY_SYSTEM_MSG, NOVELTY_USER_MSG

        user_msg = NOVELTY_USER_MSG.format(
            language=program.language,
            existing_code=similar_program.code,
            proposed_code=program.code,
        )

        try:

            try:
                loop = asyncio.get_running_loop()

                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.novelty_llm.generate_with_context(
                            system_message=NOVELTY_SYSTEM_MSG,
                            messages=[{"role": "user", "content": user_msg}],
                        ),
                    )
                    content: str = future.result()
            except RuntimeError:

                content: str = asyncio.run(
                    self.novelty_llm.generate_with_context(
                        system_message=NOVELTY_SYSTEM_MSG,
                        messages=[{"role": "user", "content": user_msg}],
                    )
                )

            if content is None or content is None:
                logger.warning("Novelty LLM returned empty response")
                return True

            content = content.strip()


            NOVEL_i = content.upper().find("NOVEL")
            NOT_NOVEL_i = content.upper().find("NOT NOVEL")

            if NOVEL_i == -1 and NOT_NOVEL_i == -1:
                logger.warning(f"Unexpected novelty LLM response: {content}")
                return True

            if NOVEL_i != -1 and NOT_NOVEL_i != -1:

                is_novel = NOVEL_i < NOT_NOVEL_i
            elif NOVEL_i != -1:
                is_novel = True
            else:
                is_novel = False

            return is_novel

        except Exception as e:
            logger.error(f"Error in novelty LLM check: {e}")

        return True

    def _is_novel(self, program_id: int, island_idx: int) -> bool:

        if self.embedding_client is None or self.similarity_threshold <= 0.0:

            return True

        program = self.programs[program_id]
        embd = self.embedding_client.get_embedding(program.code)
        self.programs[program_id].embedding = embd

        max_smlty = float("-inf")
        max_smlty_pid = None

        for pid in self.islands[island_idx]:
            other = self.programs[pid]

            if other.embedding is None:
                logger.warning(
                    f"Program {other.id} has no embedding, skipping similarity check"
                )
                continue

            similarity = self._cosine_similarity(embd, other.embedding)

            if similarity >= max(max_smlty, self.similarity_threshold):
                max_smlty = similarity
                max_smlty_pid = pid

        if max_smlty_pid is None:

            return True

        return self._llm_judge_novelty(program, self.programs[max_smlty_pid])

    def _is_better(self, program1: Program, program2: Program) -> bool:

        if self._effective_outer_enabled():
            challenger = self._effective_candidate_for_program(program1)
            incumbent = self._effective_candidate_for_program(program2)
            if challenger.feasible != incumbent.feasible:
                return challenger.feasible

            return challenger.objective > incumbent.objective


        if not program1.metrics and not program2.metrics:
            return program1.timestamp > program2.timestamp


        if program1.metrics and not program2.metrics:
            return True
        if not program1.metrics and program2.metrics:
            return False


        fitness1 = get_fitness_score(program1.metrics, self.config.feature_dimensions)
        fitness2 = get_fitness_score(program2.metrics, self.config.feature_dimensions)

        return fitness1 > fitness2

    def _update_archive(self, program: Program) -> None:


        if len(self.archive) < self.config.archive_size:
            self.archive.add(program.id)
            return


        valid_archive_programs = []
        stale_ids = []

        for pid in self.archive:
            if pid in self.programs:
                valid_archive_programs.append(self.programs[pid])
            else:
                stale_ids.append(pid)


        for stale_id in stale_ids:
            self.archive.discard(stale_id)
            logger.debug(f"Removing stale program {stale_id} from archive")


        if len(self.archive) < self.config.archive_size:
            self.archive.add(program.id)
            return


        if valid_archive_programs:
            worst_program = min(
                valid_archive_programs,
                key=lambda p: get_fitness_score(p.metrics, self.config.feature_dimensions),
            )


            if self._is_better(program, worst_program):
                self.archive.remove(worst_program.id)
                self.archive.add(program.id)
        else:

            self.archive.add(program.id)

    def _update_best_program(self, program: Program) -> None:


        if self.best_program_id is None:
            self.best_program_id = program.id
            logger.debug(f"Set initial best program to {program.id}")
            return


        if self.best_program_id not in self.programs:
            logger.warning(
                f"Best program {self.best_program_id} no longer exists, clearing reference"
            )
            self.best_program_id = program.id
            logger.info(f"Set new best program to {program.id}")
            return

        current_best = self.programs[self.best_program_id]


        if self._is_better(program, current_best):
            old_id = self.best_program_id
            self.best_program_id = program.id

            old_objective = get_primary_objective(current_best.metrics)
            new_objective = get_primary_objective(program.metrics)
            if old_objective is not None and new_objective is not None:
                logger.info(
                    "New best program %s replaces %s "
                    "(%s=%.4f [%s] → %s=%.4f [%s])",
                    program.id,
                    old_id,
                    old_objective["name"],
                    old_objective["value"],
                    old_objective["direction"],
                    new_objective["name"],
                    new_objective["value"],
                    new_objective["direction"],
                )
            else:
                logger.info(f"New best program {program.id} replaces {old_id}")

    def _update_island_best_program(self, program: Program, island_idx: int) -> None:


        if island_idx >= len(self.island_best_programs):
            logger.warning(f"Invalid island index {island_idx}, skipping island best update")
            return

        if self._v9_population_policy_enabled():
            self._recompute_effective_island_best(island_idx)
            return


        current_island_best_id = self.island_best_programs[island_idx]
        if current_island_best_id is None:
            self.island_best_programs[island_idx] = program.id
            logger.debug(f"Set initial best program for island {island_idx} to {program.id}")
            return


        if current_island_best_id not in self.programs:
            logger.warning(
                f"Island {island_idx} best program {current_island_best_id} no longer exists, updating to {program.id}"
            )
            self.island_best_programs[island_idx] = program.id
            return

        current_island_best = self.programs[current_island_best_id]


        if self._is_better(program, current_island_best):
            old_id = current_island_best_id
            self.island_best_programs[island_idx] = program.id

            old_objective = get_primary_objective(current_island_best.metrics)
            new_objective = get_primary_objective(program.metrics)
            if old_objective is not None and new_objective is not None:
                logger.debug(
                    "Island %d: New best program %s replaces %s "
                    "(%s=%.4f [%s] → %s=%.4f [%s])",
                    island_idx,
                    program.id,
                    old_id,
                    old_objective["name"],
                    old_objective["value"],
                    old_objective["direction"],
                    new_objective["name"],
                    new_objective["value"],
                    new_objective["direction"],
                )
            else:
                logger.debug(
                    f"Island {island_idx}: New best program {program.id} replaces {old_id}"
                )

    def _sample_parent(self) -> Program:


        rand_val = random.random()

        if rand_val < self.config.exploration_ratio:

            return self._sample_exploration_parent()
        elif rand_val < self.config.exploration_ratio + self.config.exploitation_ratio:

            return self._sample_exploitation_parent()
        else:

            return self._sample_random_parent()

    def _sample_exploration_parent(self) -> Program:

        current_island_programs = self.islands[self.current_island]

        if not current_island_programs:

            if self.best_program_id and self.best_program_id in self.programs:

                best_program = self.programs[self.best_program_id]
                copy_program = Program(
                    id=str(uuid.uuid4()),
                    code=best_program.code,
                    changes_description=best_program.changes_description,
                    language=best_program.language,
                    parent_id=best_program.id,
                    generation=best_program.generation,
                    timestamp=time.time(),
                    iteration_found=self.last_iteration,
                    metrics=best_program.metrics.copy(),
                    complexity=best_program.complexity,
                    diversity=best_program.diversity,
                    metadata={"island": self.current_island},
                    artifacts_json=best_program.artifacts_json,
                    artifact_dir=best_program.artifact_dir,
                )
                self.programs[copy_program.id] = copy_program
                self.islands[self.current_island].add(copy_program.id)
                logger.debug(
                    f"Initialized empty island {self.current_island} with copy of best program"
                )
                return copy_program
            else:

                return next(iter(self.programs.values()))


        valid_programs = [pid for pid in current_island_programs if pid in self.programs]


        if len(valid_programs) < len(current_island_programs):
            stale_ids = current_island_programs - set(valid_programs)
            logger.debug(
                f"Removing {len(stale_ids)} stale program IDs from island {self.current_island}"
            )
            for stale_id in stale_ids:
                self.islands[self.current_island].discard(stale_id)


        if not valid_programs:
            logger.warning(
                f"Island {self.current_island} has no valid programs after cleanup, reinitializing"
            )
            if self.best_program_id and self.best_program_id in self.programs:

                best_program = self.programs[self.best_program_id]
                copy_program = Program(
                    id=str(uuid.uuid4()),
                    code=best_program.code,
                    changes_description=best_program.changes_description,
                    language=best_program.language,
                    parent_id=best_program.id,
                    generation=best_program.generation,
                    timestamp=time.time(),
                    iteration_found=self.last_iteration,
                    metrics=best_program.metrics.copy(),
                    complexity=best_program.complexity,
                    diversity=best_program.diversity,
                    metadata={"island": self.current_island},
                    artifacts_json=best_program.artifacts_json,
                    artifact_dir=best_program.artifact_dir,
                )
                self.programs[copy_program.id] = copy_program
                self.islands[self.current_island].add(copy_program.id)
                logger.debug(
                    f"Reinitialized empty island {self.current_island} with copy of best program"
                )
                return copy_program
            else:
                return next(iter(self.programs.values()))


        parent_id = random.choice(valid_programs)
        return self.programs[parent_id]

    def _sample_exploitation_parent(self) -> Program:

        if not self.archive:

            return self._sample_exploration_parent()


        valid_archive = [pid for pid in self.archive if pid in self.programs]


        if len(valid_archive) < len(self.archive):
            stale_ids = self.archive - set(valid_archive)
            logger.debug(f"Removing {len(stale_ids)} stale program IDs from archive")
            for stale_id in stale_ids:
                self.archive.discard(stale_id)


        if not valid_archive:
            logger.warning(
                "Archive has no valid programs after cleanup, falling back to exploration"
            )
            return self._sample_exploration_parent()


        archive_programs_in_island = [
            pid
            for pid in valid_archive
            if self.programs[pid].metadata.get("island") == self.current_island
        ]

        if archive_programs_in_island:
            parent_id = random.choice(archive_programs_in_island)
            return self.programs[parent_id]
        else:

            parent_id = random.choice(valid_archive)
            return self.programs[parent_id]

    def _sample_random_parent(self) -> Program:

        if not self.programs:
            raise ValueError("No programs available for sampling")


        program_id = random.choice(list(self.programs.keys()))
        return self.programs[program_id]

    def _sample_from_island_weighted(self, island_id: int) -> Program:

        island_id = island_id % len(self.islands)
        island_programs = list(self.islands[island_id])

        if not island_programs:

            logger.debug(f"Island {island_id} is empty, sampling from all programs")
            return self._sample_random_parent()


        if len(island_programs) == 1:
            parent_id = island_programs[0]
        else:

            island_program_objects = [
                self.programs[pid] for pid in island_programs if pid in self.programs
            ]

            if not island_program_objects:

                parent_id = random.choice(island_programs)
            else:


                fitness_values = [
                    get_fitness_score(prog.metrics, self.config.feature_dimensions)
                    for prog in island_program_objects
                ]
                has_energy_objective = any(
                    (get_primary_objective(prog.metrics) or {}).get("direction")
                    == "minimize"
                    for prog in island_program_objects
                )
                if has_energy_objective:
                    minimum_fitness = min(fitness_values)
                    weights = [
                        max(fitness - minimum_fitness + 0.001, 0.001)
                        for fitness in fitness_values
                    ]
                else:

                    weights = [max(fitness, 0.001) for fitness in fitness_values]


                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                else:
                    weights = [1.0 / len(island_program_objects)] * len(island_program_objects)


                parent = random.choices(island_program_objects, weights=weights, k=1)[0]
                parent_id = parent.id

        parent = self.programs.get(parent_id)
        if not parent:

            logger.error(f"Parent program {parent_id} not found in database")
            return self._sample_random_parent()

        return parent

    def _sample_from_island_random(self, island_id: int) -> Program:

        island_id = island_id % len(self.islands)
        island_programs = list(self.islands[island_id])

        if not island_programs:

            logger.debug(f"Island {island_id} is empty, sampling from all programs")
            return self._sample_random_parent()


        valid_programs = [pid for pid in island_programs if pid in self.programs]

        if not valid_programs:
            logger.warning(
                f"Island {island_id} has no valid programs, falling back to random sampling"
            )
            return self._sample_random_parent()


        parent_id = random.choice(valid_programs)
        return self.programs[parent_id]

    def _sample_from_archive_for_island(self, island_id: int) -> Program:

        if not self.archive:

            logger.debug(f"Archive is empty, falling back to weighted island sampling")
            return self._sample_from_island_weighted(island_id)


        valid_archive = [pid for pid in self.archive if pid in self.programs]

        if not valid_archive:
            logger.warning(
                "Archive has no valid programs, falling back to weighted island sampling"
            )
            return self._sample_from_island_weighted(island_id)

        island_id = island_id % len(self.islands)


        archive_programs_in_island = [
            pid for pid in valid_archive if self.programs[pid].metadata.get("island") == island_id
        ]

        if archive_programs_in_island:
            parent_id = random.choice(archive_programs_in_island)
            return self.programs[parent_id]
        else:

            parent_id = random.choice(valid_archive)
            return self.programs[parent_id]

    def _sample_inspirations(
        self,
        parent: Program,
        n: int = 5,
        island_id: Optional[int] = None,
    ) -> List[Program]:

        inspirations = []


        parent_island = (
            parent.metadata.get("island", self.current_island)
            if island_id is None
            else island_id
        )
        parent_island = int(parent_island) % len(self.islands)


        island_program_ids = list(self.islands[parent_island])
        island_programs = [self.programs[pid] for pid in island_program_ids if pid in self.programs]

        if not island_programs:
            logger.warning(f"Island {parent_island} has no programs for inspiration sampling")
            return []


        island_best_id = self.island_best_programs[parent_island]
        if (
            island_best_id is not None
            and island_best_id != parent.id
            and island_best_id in self.programs
        ):
            island_best = self.programs[island_best_id]
            inspirations.append(island_best)
            logger.debug(
                f"Including island {parent_island} best program {island_best_id} in inspirations"
            )
        elif island_best_id is not None and island_best_id not in self.programs:

            logger.warning(
                f"Island {parent_island} best program {island_best_id} no longer exists, clearing reference"
            )
            self.island_best_programs[parent_island] = None


        top_n = max(1, int(n * self.config.elite_selection_ratio))
        top_island_programs = self.get_top_programs(n=top_n, island_idx=parent_island)
        for program in top_island_programs:
            if program.id not in [p.id for p in inspirations] and program.id != parent.id:
                inspirations.append(program)


        if len(island_programs) > n and len(inspirations) < n:
            remaining_slots = n - len(inspirations)


            feature_coords = self._calculate_feature_coords(parent)
            nearby_programs = []


            island_feature_map = {}
            for prog_id in island_program_ids:
                if prog_id in self.programs:
                    prog = self.programs[prog_id]
                    prog_coords = self._calculate_feature_coords(prog)
                    cell_key = self._feature_coords_to_key(prog_coords)
                    island_feature_map[cell_key] = prog_id


            for _ in range(remaining_slots * 3):

                perturbed_coords = [
                    max(0, min(self.feature_bins - 1, c + random.randint(-2, 2)))
                    for c in feature_coords
                ]

                cell_key = self._feature_coords_to_key(perturbed_coords)
                if cell_key in island_feature_map:
                    program_id = island_feature_map[cell_key]
                    if (
                        program_id != parent.id
                        and program_id not in [p.id for p in inspirations]
                        and program_id not in [p.id for p in nearby_programs]
                        and program_id in self.programs
                    ):
                        nearby_programs.append(self.programs[program_id])
                        if len(nearby_programs) >= remaining_slots:
                            break


            if len(inspirations) + len(nearby_programs) < n:
                remaining = n - len(inspirations) - len(nearby_programs)


                excluded_ids = (
                    {parent.id}
                    .union(p.id for p in inspirations)
                    .union(p.id for p in nearby_programs)
                )
                available_island_ids = [
                    pid
                    for pid in island_program_ids
                    if pid not in excluded_ids and pid in self.programs
                ]

                if available_island_ids:
                    random_ids = random.sample(
                        available_island_ids, min(remaining, len(available_island_ids))
                    )
                    random_programs = [self.programs[pid] for pid in random_ids]
                    nearby_programs.extend(random_programs)

            inspirations.extend(nearby_programs)


        logger.debug(
            f"Sampled {len(inspirations)} inspirations from island {parent_island} "
            f"(island has {len(island_programs)} programs total)"
        )

        return inspirations[:n]

    def _remove_program_if_orphaned(self, program_id: str) -> None:


        if program_id not in self.programs:
            return

        if any(
            program_id in island_map.values()
            for island_map in self.island_feature_maps
        ):
            return

        if any(program_id in island for island in self.islands):
            return

        del self.programs[program_id]
        self.archive.discard(program_id)
        self._cleanup_stale_island_bests()
        logger.debug(
            "Removed orphaned program %s displaced from its legacy feature cell",
            program_id,
        )

    def _enforce_population_limit(self, exclude_program_id: Optional[str] = None) -> None:

        if self._effective_outer_enabled():
            self._enforce_effective_population_limit(
                exclude_program_id=exclude_program_id
            )
            return

        if len(self.programs) <= self.config.population_size:
            return


        num_to_remove = len(self.programs) - self.config.population_size

        logger.info(
            f"Population size ({len(self.programs)}) exceeds limit ({self.config.population_size}), removing {num_to_remove} programs"
        )


        elite_ids = set()
        for island_map in self.island_feature_maps:
            elite_ids.update(island_map.values())

        protected_ids = {self.best_program_id, exclude_program_id} - {None}
        all_programs = list(self.programs.values())

        non_elite = sorted(
            [
                program
                for program in all_programs
                if program.id not in elite_ids and program.id not in protected_ids
            ],
            key=lambda p: get_fitness_score(p.metrics, self.config.feature_dimensions),
        )
        elite = sorted(
            [
                program
                for program in all_programs
                if program.id in elite_ids and program.id not in protected_ids
            ],
            key=lambda p: get_fitness_score(p.metrics, self.config.feature_dimensions),
        )

        programs_to_remove = non_elite[:num_to_remove]
        if len(programs_to_remove) < num_to_remove:
            remaining = num_to_remove - len(programs_to_remove)
            programs_to_remove.extend(elite[:remaining])


        for program in programs_to_remove:
            program_id = program.id


            if program_id in self.programs:
                del self.programs[program_id]


            for island_idx, island_map in enumerate(self.island_feature_maps):
                keys_to_remove = []
                for key, pid in island_map.items():
                    if pid == program_id:
                        keys_to_remove.append(key)
                for key in keys_to_remove:
                    del island_map[key]


            for island in self.islands:
                island.discard(program_id)


            self.archive.discard(program_id)

            logger.debug(f"Removed program {program_id} due to population limit")

        logger.info(f"Population size after cleanup: {len(self.programs)}")


        self._cleanup_stale_island_bests()

    def _remove_effective_program(self, program_id: str) -> None:
        self.programs.pop(program_id, None)
        self.archive.discard(program_id)
        for island in self.islands:
            island.discard(program_id)
        for island_map in self.island_feature_maps:
            for key, mapped_id in list(island_map.items()):
                if mapped_id == program_id:
                    del island_map[key]
        for phenotype_hash, mapped_id in list(self.effective_phenotype_index.items()):
            if mapped_id == program_id:
                del self.effective_phenotype_index[phenotype_hash]

    def _enforce_effective_population_limit(
        self, *, exclude_program_id: Optional[str]
    ) -> None:
        excess = len(self.programs) - self.config.population_size
        if excess <= 0:
            return
        protected = {self.best_program_id, exclude_program_id} - {None}
        selectable_ids = set(self.archive)
        selectable_ids.update(pid for island in self.islands for pid in island)
        rejected = sorted(
            (
                program
                for program in self.programs.values()
                if program.id not in selectable_ids and program.id not in protected
            ),
            key=lambda program: (program.timestamp, program.id),
        )
        remove_ids = [program.id for program in rejected[:excess]]
        remaining = excess - len(remove_ids)
        if remaining > 0:
            selectable = [
                self.programs[pid]
                for pid in selectable_ids
                if pid in self.programs and pid not in protected
            ]

            ordered = self._order_effective_programs(selectable)
            remove_ids.extend(program.id for program in reversed(ordered) if program.id not in remove_ids)
            remove_ids = remove_ids[:excess]
        for program_id in remove_ids:
            self._remove_effective_program(program_id)
        self._refresh_effective_phenotype_index()
        self._cleanup_stale_island_bests()
        if self.best_program_id not in self.programs:
            best = self.get_best_program()
            self.best_program_id = best.id if best is not None else None


    def set_current_island(self, island_idx: int) -> None:

        self.current_island = island_idx % len(self.islands)
        logger.debug(f"Switched to evolving island {self.current_island}")

    def next_island(self) -> int:

        self.current_island = (self.current_island + 1) % len(self.islands)
        logger.debug(f"Advanced to island {self.current_island}")
        return self.current_island

    def increment_island_generation(self, island_idx: Optional[int] = None) -> None:

        idx = island_idx if island_idx is not None else self.current_island
        self.island_generations[idx] += 1
        logger.debug(f"Island {idx} generation incremented to {self.island_generations[idx]}")

    def _migration_interval_scope(self) -> str:
        if not self._v9_population_policy_enabled():
            return "max_island_generation"
        return str(
            getattr(
                self.config,
                "migration_interval_scope",
                "max_island_generation",
            )
        )

    def _migration_due_source_islands(self) -> List[int]:
        interval = max(1, int(self.migration_interval))
        scope = self._migration_interval_scope()
        if scope == "per_island":
            return [
                island_idx
                for island_idx, generation in enumerate(self.island_generations)
                if generation - self.last_migration_island_generations[island_idx]
                >= interval
            ]
        if scope == "global_iteration":
            return (
                list(range(len(self.islands)))
                if self.last_iteration - self.last_migration_iteration >= interval
                else []
            )
        return (
            list(range(len(self.islands)))
            if max(self.island_generations, default=0)
            - self.last_migration_generation
            >= interval
            else []
        )

    def _mark_migration_completed(self, source_islands: Iterable[int]) -> None:
        sources = sorted({int(item) % len(self.islands) for item in source_islands})
        self.last_migration_generation = max(self.island_generations, default=0)
        self.last_migration_iteration = self.last_iteration
        for island_idx in sources:
            self.last_migration_island_generations[island_idx] = (
                self.island_generations[island_idx]
            )

    def should_migrate(self) -> bool:

        return bool(self._migration_due_source_islands())

    def migrate_programs(self) -> None:

        if self._effective_outer_enabled():
            self._migrate_effective_programs()
            return

        if len(self.islands) < 2:
            return

        logger.info("Performing migration between islands")

        for i, island in enumerate(self.islands):
            if len(island) == 0:
                continue


            island_programs = [self.programs[pid] for pid in island if pid in self.programs]
            if not island_programs:
                continue


            island_programs.sort(
                key=lambda p: get_fitness_score(p.metrics, self.config.feature_dimensions),
                reverse=True,
            )


            num_to_migrate = max(1, int(len(island_programs) * self.migration_rate))
            migrants = island_programs[:num_to_migrate]

            for migrant in migrants:


                if migrant.metadata.get("migrant", False):
                    continue

                target_islands = self._migration_targets_for_island(i, migrant)
                for target_island in target_islands:


                    target_island_programs = [
                        self.programs[pid]
                        for pid in self.islands[target_island]
                        if pid in self.programs
                    ]
                    has_duplicate_code = any(p.code == migrant.code for p in target_island_programs)

                    if has_duplicate_code:
                        logger.debug(
                            f"Skipping migration of program {migrant.id[:8]} to island {target_island} "
                            f"(duplicate code already exists)"
                        )
                        continue

                    import uuid

                    migrant_copy = Program(
                        id=str(uuid.uuid4()),
                        code=migrant.code,
                        changes_description=migrant.changes_description,
                        language=migrant.language,
                        parent_id=migrant.id,
                        generation=migrant.generation,
                        metrics=migrant.metrics.copy(),
                        metadata={
                            **migrant.metadata,
                            "island": target_island,
                            "migrant": True,
                            "migration_reason": self._migration_reason(i, target_island, migrant),
                            "source_island_profile": self._island_profile(i).get("name"),
                            "target_island_profile": self._island_profile(target_island).get("name"),
                        },
                    )


                    self.add(migrant_copy, target_island=target_island)


                    logger.info(
                        "Program %s migrated to island %d",
                        migrant_copy.id[:8],
                        target_island,
                    )


        self._mark_migration_completed(range(len(self.islands)))
        logger.info(f"Migration completed at generation {self.last_migration_generation}")


        self._validate_migration_results()

    def _recompute_effective_island_best(self, island_idx: int) -> None:
        programs = self._effective_selectable_programs(island_idx=island_idx)
        ordered = (
            self._order_island_role_programs(programs, island_idx)
            if self._v9_population_policy_enabled()
            else self._order_effective_programs(programs)
        )
        self.island_best_programs[island_idx] = ordered[0].id if ordered else None

    def _migrate_effective_programs(self) -> None:


        if len(self.islands) < 2:
            return
        with self._outer_policy_lock:
            source_snapshots = [set(island) for island in self.islands]
            due_sources = self._migration_due_source_islands()
            source_indices = due_sources or list(range(len(self.islands)))
            capacity = max(1, self.config.population_size // len(self.islands))
            for source_idx in source_indices:
                source_ids = source_snapshots[source_idx]
                source_programs = [
                    self.programs[pid]
                    for pid in source_ids
                    if pid in self.programs
                ]
                if self._v9_population_policy_enabled():


                    source_programs = self._v9_pareto_front(source_programs)
                ordered = (
                    self._order_island_role_programs(source_programs, source_idx)
                    if self._v9_population_policy_enabled()
                    else self._order_effective_programs(source_programs)
                )
                if not ordered:
                    continue
                count = max(1, int(len(ordered) * self.migration_rate))
                for migrant_program in ordered[:count]:
                    migrant = self._effective_candidate_for_program(migrant_program)
                    targets: List[int] = []
                    for target in self._migration_targets_for_island(
                        source_idx, migrant_program
                    ):
                        target = int(target) % len(self.islands)
                        if target != source_idx and target not in targets:
                            targets.append(target)
                    for target_idx in targets:
                        if migrant_program.id in self.islands[target_idx]:
                            continue
                        residents = [
                            self.programs[pid]
                            for pid in sorted(self.islands[target_idx])
                            if pid in self.programs
                        ]
                        feature_key = "effective:" + str(
                            migrant_program.metadata["effective_descriptor_cell_hash"]
                        )
                        cell_incumbent_id = self.island_feature_maps[target_idx].get(
                            feature_key
                        )
                        if (
                            cell_incumbent_id is not None
                            and cell_incumbent_id in self.programs
                        ):
                            decision_programs = [
                                self.programs[cell_incumbent_id]
                            ]
                            decision_residents = [
                                self._effective_candidate_for_program(
                                    self.programs[cell_incumbent_id]
                                )
                            ]
                            decision_capacity = 1
                        else:
                            decision_programs = residents
                            decision_residents = [
                                self._effective_candidate_for_program(program)
                                for program in residents
                            ]
                            decision_capacity = capacity
                        target_role = self.get_island_role(target_idx)
                        role_scores = (
                            {
                                item.id: self._island_role_sampling_bonus(
                                    item, target_role
                                )
                                for item in [migrant_program, *decision_programs]
                            }
                            if self._v9_population_policy_enabled()
                            else None
                        )
                        decision_index = self._next_outer_decision_index()
                        decision = decide_migration(
                            migrant,
                            decision_residents,
                            capacity=decision_capacity,
                            base_seed=self.config.random_seed,
                            namespace=f"migration:{source_idx}:{target_idx}",
                            decision_index=decision_index,
                            secondary_scores=role_scores,
                            secondary_label=str(
                                target_role.get("role_id") or "island_role"
                            ),
                            secondary_objective_tolerance=float(
                                getattr(
                                    self.config,
                                    "island_role_survivor_objective_tolerance",
                                    0.0,
                                )
                            ),
                        )
                        if self._v9_population_policy_enabled():
                            source_role = self.get_island_role(source_idx)
                            decision = EvolutionDecision.create(
                                kind=decision.kind,
                                namespace=decision.namespace,
                                decision_index=decision.decision_index,
                                derived_seed=decision.derived_seed,
                                selected_candidate_id=decision.selected_candidate_id,
                                reason=decision.reason,
                                candidate_ids=decision.candidate_ids,
                                eligible_ids=decision.eligible_ids,
                                add_ids=decision.add_ids,
                                remove_ids=decision.remove_ids,
                                details={
                                    **decision.details,
                                    "source_island": source_idx,
                                    "target_island": target_idx,
                                    "source_role_id": source_role.get("role_id"),
                                    "target_role_id": target_role.get("role_id"),
                                    "migration_interval_scope": self._migration_interval_scope(),
                                    "source_selection": "pareto_non_dominated_only",
                                    "target_rescored": True,
                                },
                            )
                        self._record_outer_decision(decision.to_dict())
                        receipt_material = {
                            "schema_version": "astevolve.migration_receipt.v1",
                            "iteration": self.last_iteration,
                            "source_island": source_idx,
                            "target_island": target_idx,
                            "program_id": migrant_program.id,
                            "source_selection": (
                                "pareto_non_dominated_only"
                                if self._v9_population_policy_enabled()
                                else "legacy_ranked"
                            ),
                            "target_rescored": bool(
                                self._v9_population_policy_enabled()
                            ),
                            "admitted": migrant_program.id in decision.add_ids,
                            "decision_index": decision.decision_index,
                            "decision_reason": decision.reason,
                        }
                        receipt_hash = "migration_receipt_sha256:" + hashlib.sha256(
                            self._canonical_outer_json(receipt_material).encode("utf-8")
                        ).hexdigest()
                        self.migration_receipts.append(
                            {**receipt_material, "receipt_hash": receipt_hash}
                        )
                        migrant_program.metadata[
                            "last_outer_migration_decision"
                        ] = decision.to_dict()
                        if migrant_program.id not in decision.add_ids:
                            continue
                        for removed_id in decision.remove_ids:
                            self.islands[target_idx].discard(removed_id)
                            for key, mapped_id in list(
                                self.island_feature_maps[target_idx].items()
                            ):
                                if mapped_id == removed_id:
                                    del self.island_feature_maps[target_idx][key]
                            self._refresh_program_membership(removed_id)

                        self.islands[target_idx].add(migrant_program.id)
                        self.island_feature_maps[target_idx][
                            feature_key
                        ] = migrant_program.id
                        migrant_program.metadata["shared_migrant"] = True
                        self._refresh_program_membership(migrant_program.id)
                        self._recompute_effective_island_best(target_idx)
            self._mark_migration_completed(source_indices)
            self._refresh_effective_phenotype_index()
            self._validate_migration_results()

    def _migration_targets_for_island(self, source_island: int, program: Program) -> List[int]:
        if not getattr(self.config, "island_specialization_enabled", True) or (
            not self._v9_population_policy_enabled()
            and not self._ast_scheduler_enabled()
        ):
            return [(source_island + 1) % len(self.islands), (source_island - 1) % len(self.islands)]

        source_profile = self._island_profile(source_island)
        program_profile = self._program_ast_profile(program)
        categories = set(program_profile.get("categories", []))
        complement = {
            "fold_stability": {"interface_contact", "specificity_negative_design", "allostery_pocket"},
            "interface_contact": {"fold_stability", "specificity_negative_design"},
            "specificity_negative_design": {"interface_contact", "fold_stability"},
            "allostery_pocket": {"fold_stability", "interface_contact"},
        }

        role_complement = {
            "a_interface_fold": {"selectivity_margin", "robustness_safety"},
            "selectivity_margin": {"a_interface_fold", "region_exploration", "robustness_safety"},
            "region_exploration": {"selectivity_margin", "a_interface_fold"},
            "robustness_safety": {"a_interface_fold", "selectivity_margin"},
        }
        source_role_id = str(source_profile.get("role_id", ""))
        desired = {
            str(item)
            for item in source_profile.get("complementary_roles", []) or []
        }
        if not desired:
            desired = set(role_complement.get(source_role_id, set()))
        if not desired:
            for category in categories or {source_profile.get("name", "")}:
                desired.update(complement.get(category, set()))

        targets = []
        for idx in range(len(self.islands)):
            if idx == source_island:
                continue
            profile = self._island_profile(idx)
            profile_key = profile.get("role_id") or profile.get("name")
            if profile_key in desired or (not source_role_id and profile.get("name") in desired):
                targets.append(idx)

        if not targets:
            targets = [(source_island + 1) % len(self.islands), (source_island - 1) % len(self.islands)]

        deduped = []
        for target in targets:
            if target != source_island and target not in deduped:
                deduped.append(target)
        return deduped

    def _migration_reason(self, source_island: int, target_island: int, program: Program) -> str:
        source = self._island_profile(source_island).get("name")
        target = self._island_profile(target_island).get("name")
        categories = self._program_ast_profile(program).get("categories", [])
        if categories:
            return (
                f"migrate {', '.join(categories)} specialist from {source} "
                f"into complementary {target} island"
            )
        return f"migrate top candidate from {source} into complementary {target} island"

    def _validate_migration_results(self) -> None:

        if self._effective_outer_enabled():
            actual_memberships: Dict[str, List[int]] = {}
            for island_idx, island in enumerate(self.islands):
                for program_id in island:
                    actual_memberships.setdefault(program_id, []).append(island_idx)
            for program_id, memberships in actual_memberships.items():
                if program_id not in self.programs:
                    logger.warning(
                        "Island membership contains nonexistent program %s", program_id
                    )
                    continue
                declared = sorted(
                    int(item)
                    for item in (
                        self.programs[program_id].metadata.get("islands") or []
                    )
                )
                if declared != sorted(memberships):
                    logger.warning(
                        "Shared island membership mismatch for %s: actual=%s declared=%s",
                        program_id,
                        sorted(memberships),
                        declared,
                    )
            return

        seen_program_ids = set()

        for i, island in enumerate(self.islands):
            for program_id in island:

                if program_id in seen_program_ids:
                    logger.error(f"Program {program_id} assigned to multiple islands")
                    continue
                seen_program_ids.add(program_id)


                if program_id not in self.programs:
                    logger.warning(f"Island {i} contains nonexistent program {program_id}")
                    continue


                program = self.programs[program_id]
                stored_island = program.metadata.get("island")
                if stored_island != i:
                    logger.warning(
                        f"Island mismatch for program {program_id}: "
                        f"in island {i} but metadata says {stored_island}"
                    )


        for i, best_id in enumerate(self.island_best_programs):
            if best_id is not None:
                if best_id not in self.programs:
                    logger.warning(f"Island {i} best program {best_id} does not exist")
                elif best_id not in self.islands[i]:
                    logger.warning(f"Island {i} best program {best_id} not in island")

    def _cleanup_stale_island_bests(self) -> None:

        cleaned_count = 0

        for i, best_id in enumerate(self.island_best_programs):
            if best_id is not None:
                should_clear = False


                if best_id not in self.programs:
                    logger.debug(
                        f"Clearing stale island {i} best program {best_id} (program deleted)"
                    )
                    should_clear = True

                elif best_id not in self.islands[i]:
                    logger.debug(
                        f"Clearing stale island {i} best program {best_id} (not in island)"
                    )
                    should_clear = True

                if should_clear:
                    self.island_best_programs[i] = None
                    cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} stale island best program references")


            for i, best_id in enumerate(self.island_best_programs):
                if best_id is None and len(self.islands[i]) > 0:

                    island_programs = [
                        self.programs[pid] for pid in self.islands[i] if pid in self.programs
                    ]
                    if island_programs:

                        best_program = max(
                            island_programs,
                            key=lambda p: get_fitness_score(
                                p.metrics, self.config.feature_dimensions
                            ),
                        )
                        self.island_best_programs[i] = best_program.id
                        logger.debug(f"Recalculated island {i} best program: {best_program.id}")

    def get_island_stats(self) -> List[dict]:

        stats = []

        for i, island in enumerate(self.islands):
            island_programs = [self.programs[pid] for pid in island if pid in self.programs]

            if island_programs:
                objectives = [
                    objective
                    for objective in (
                        get_primary_objective(program.metrics)
                        for program in island_programs
                    )
                    if objective is not None
                ]
                if objectives and all(
                    objective["direction"] == "minimize"
                    for objective in objectives
                ):
                    values = [float(objective["value"]) for objective in objectives]
                    best_score = min(values)
                    avg_score = sum(values) / len(values)
                    score_direction = "minimize"
                    objective_names = {objective["name"] for objective in objectives}
                    score_metric = (
                        next(iter(objective_names))
                        if len(objective_names) == 1
                        else "energy"
                    )
                else:
                    values = [
                        float(objective["value"])
                        for objective in objectives
                        if objective["direction"] == "maximize"
                    ]
                    if not values:
                        values = [
                            get_fitness_score(
                                program.metrics, self.config.feature_dimensions
                            )
                            for program in island_programs
                        ]
                    best_score = max(values) if values else 0.0
                    avg_score = sum(values) / len(values) if values else 0.0
                    score_direction = "maximize"
                    score_metric = (
                        objectives[0]["name"]
                        if objectives
                        and len({objective["name"] for objective in objectives}) == 1
                        else "compatibility_fitness"
                    )
                diversity = self._calculate_island_diversity(island_programs)
            else:
                best_score = avg_score = diversity = 0.0
                score_direction = "maximize"
                score_metric = "unavailable"

            stats.append(
                {
                    "island": i,
                    "population_size": len(island_programs),
                    "best_score": best_score,
                    "average_score": avg_score,
                    "score_metric": score_metric,
                    "score_direction": score_direction,
                    "diversity": diversity,
                    "generation": self.island_generations[i],
                    "is_current": i == self.current_island,
                    "profile": self._island_profile(i).get("name"),
                }
            )

        return stats

    def _calculate_island_diversity(self, programs: List[Program]) -> float:

        if len(programs) < 2:
            return 0.0

        total_diversity = 0
        comparisons = 0


        sample_size = min(5, len(programs))


        sorted_programs = sorted(programs, key=lambda p: p.id)


        sample_programs = sorted_programs[:sample_size]


        max_comparisons = 6

        for i, prog1 in enumerate(sample_programs):
            for prog2 in sample_programs[i + 1 :]:
                if comparisons >= max_comparisons:
                    break


                diversity = self._fast_code_diversity(prog1.code, prog2.code)
                total_diversity += diversity
                comparisons += 1

            if comparisons >= max_comparisons:
                break

        return total_diversity / max(1, comparisons)

    def _fast_code_diversity(self, code1: str, code2: str) -> float:

        if code1 == code2:
            return 0.0


        len1, len2 = len(code1), len(code2)
        length_diff = abs(len1 - len2)


        lines1 = code1.count("\n")
        lines2 = code2.count("\n")
        line_diff = abs(lines1 - lines2)


        chars1 = set(code1)
        chars2 = set(code2)
        char_diff = len(chars1.symmetric_difference(chars2))


        diversity = length_diff * 0.1 + line_diff * 10 + char_diff * 0.5

        return diversity

    def _get_cached_diversity(self, program: Program) -> float:

        code_hash = hash(program.code)


        if code_hash in self.diversity_cache:
            return self.diversity_cache[code_hash]["value"]


        if (
            not self.diversity_reference_set
            or len(self.diversity_reference_set) < self.diversity_reference_size
        ):
            self._update_diversity_reference_set()


        diversity_scores = []
        for ref_code in self.diversity_reference_set:
            if ref_code != program.code:
                diversity_scores.append(self._fast_code_diversity(program.code, ref_code))

        diversity = (
            sum(diversity_scores) / max(1, len(diversity_scores)) if diversity_scores else 0.0
        )


        self._cache_diversity_value(code_hash, diversity)

        return diversity

    def _update_diversity_reference_set(self) -> None:

        if len(self.programs) == 0:
            return


        all_programs = list(self.programs.values())

        if len(all_programs) <= self.diversity_reference_size:
            self.diversity_reference_set = [p.code for p in all_programs]
        else:

            selected = []
            remaining = all_programs.copy()


            first_idx = random.randint(0, len(remaining) - 1)
            selected.append(remaining.pop(first_idx))


            while len(selected) < self.diversity_reference_size and remaining:
                max_diversity = -1
                best_idx = -1

                for i, candidate in enumerate(remaining):

                    min_div = float("inf")
                    for selected_prog in selected:
                        div = self._fast_code_diversity(candidate.code, selected_prog.code)
                        min_div = min(min_div, div)

                    if min_div > max_diversity:
                        max_diversity = min_div
                        best_idx = i

                if best_idx >= 0:
                    selected.append(remaining.pop(best_idx))

            self.diversity_reference_set = [p.code for p in selected]

        logger.debug(
            f"Updated diversity reference set with {len(self.diversity_reference_set)} programs"
        )

    def _cache_diversity_value(self, code_hash: int, diversity: float) -> None:


        if len(self.diversity_cache) >= self.diversity_cache_size:

            oldest_hash = min(self.diversity_cache.items(), key=lambda x: x[1]["timestamp"])[0]
            del self.diversity_cache[oldest_hash]


        self.diversity_cache[code_hash] = {"value": diversity, "timestamp": time.time()}

    def _invalidate_diversity_cache(self) -> None:

        self.diversity_cache.clear()
        self.diversity_reference_set = []
        logger.debug("Diversity cache invalidated")

    def _update_feature_stats(self, feature_name: str, value: float) -> None:

        if feature_name not in self.feature_stats:
            self.feature_stats[feature_name] = {
                "min": value,
                "max": value,
                "values": [],
            }

        stats = self.feature_stats[feature_name]
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)


        stats["values"].append(value)
        if len(stats["values"]) > 1000:
            stats["values"] = stats["values"][-1000:]

    def _scale_feature_value(self, feature_name: str, value: float) -> float:

        if feature_name not in self.feature_stats:

            return min(1.0, max(0.0, value))

        stats = self.feature_stats[feature_name]

        if self.feature_scaling_method == "minmax":

            min_val = stats["min"]
            max_val = stats["max"]

            if max_val == min_val:
                return 0.5

            scaled = (value - min_val) / (max_val - min_val)
            return min(1.0, max(0.0, scaled))

        elif self.feature_scaling_method == "percentile":

            values = stats["values"]
            if not values:
                return 0.5


            count = sum(1 for v in values if v <= value)
            percentile = count / len(values)
            return percentile

        else:

            return self._scale_feature_value_minmax(feature_name, value)

    def _scale_feature_value_minmax(self, feature_name: str, value: float) -> float:

        if feature_name not in self.feature_stats:
            return min(1.0, max(0.0, value))

        stats = self.feature_stats[feature_name]
        min_val = stats["min"]
        max_val = stats["max"]

        if max_val == min_val:
            return 0.5

        scaled = (value - min_val) / (max_val - min_val)
        return min(1.0, max(0.0, scaled))

    def _serialize_feature_stats(self) -> Dict[str, Any]:

        serialized = {}
        for feature_name, stats in self.feature_stats.items():

            serialized_stats = {}
            for key, value in stats.items():
                if key == "values":


                    if isinstance(value, list) and len(value) > 100:
                        serialized_stats[key] = value[-100:]
                    else:
                        serialized_stats[key] = value
                else:

                    if hasattr(value, "item"):
                        serialized_stats[key] = value.item()
                    else:
                        serialized_stats[key] = value
            serialized[feature_name] = serialized_stats
        return serialized

    def _deserialize_feature_stats(
        self, stats_dict: Dict[str, Any]
    ) -> Dict[str, Dict[str, Union[float, List[float]]]]:

        if not stats_dict:
            return {}

        deserialized = {}
        for feature_name, stats in stats_dict.items():
            if isinstance(stats, dict):

                deserialized_stats = {
                    "min": float(stats.get("min", 0.0)),
                    "max": float(stats.get("max", 1.0)),
                    "values": list(stats.get("values", [])),
                }
                deserialized[feature_name] = deserialized_stats
            else:
                logger.warning(
                    f"Skipping malformed feature_stats entry for '{feature_name}': {stats}"
                )

        return deserialized

    def log_island_status(self) -> None:

        stats = self.get_island_stats()
        logger.info("Island Status:")
        for stat in stats:
            current_marker = " *" if stat["is_current"] else "  "
            island_idx = stat["island"]
            island_best_id = (
                self.island_best_programs[island_idx]
                if island_idx < len(self.island_best_programs)
                else None
            )
            best_indicator = f" (best: {island_best_id})" if island_best_id else ""
            logger.info(
                f"{current_marker} Island {stat['island']}: {stat['population_size']} programs, "
                f"best={stat['best_score']:.4f}, avg={stat['average_score']:.4f}, "
                f"diversity={stat['diversity']:.2f}, gen={stat['generation']}{best_indicator}"
            )


    def store_artifacts(self, program_id: str, artifacts: Dict[str, Union[str, bytes]]) -> None:

        if not artifacts:
            return

        program = self.get(program_id)
        if not program:
            logger.warning(f"Cannot store artifacts: program {program_id} not found")
            return


        artifacts_enabled = os.environ.get("ENABLE_ARTIFACTS", "true").lower() == "true"
        if not artifacts_enabled:
            logger.debug("Artifacts disabled, skipping storage")
            return


        small_artifacts = {}
        large_artifacts = {}
        size_threshold = getattr(self.config, "artifact_size_threshold", 32 * 1024)

        for key, value in artifacts.items():
            size = self._get_artifact_size(value)
            if size <= size_threshold:
                small_artifacts[key] = value
            else:
                large_artifacts[key] = value


        if small_artifacts:
            program.artifacts_json = json.dumps(small_artifacts, default=self._artifact_serializer)
            logger.debug(f"Stored {len(small_artifacts)} small artifacts for program {program_id}")


        if large_artifacts:
            artifact_dir = self._create_artifact_dir(program_id)
            program.artifact_dir = artifact_dir
            for key, value in large_artifacts.items():
                self._write_artifact_file(artifact_dir, key, value)
            logger.debug(f"Stored {len(large_artifacts)} large artifacts for program {program_id}")

    def get_artifacts(self, program_id: str) -> Dict[str, Union[str, bytes]]:

        program = self.get(program_id)
        if not program:
            return {}

        artifacts = {}


        if program.artifacts_json:
            try:
                small_artifacts = json.loads(program.artifacts_json)
                artifacts.update(small_artifacts)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode artifacts JSON for program {program_id}: {e}")


        if program.artifact_dir and os.path.exists(program.artifact_dir):
            disk_artifacts = self._load_artifact_dir(program.artifact_dir)
            artifacts.update(disk_artifacts)

        return artifacts

    def _get_artifact_size(self, value: Union[str, bytes]) -> int:

        if isinstance(value, str):
            return len(value.encode("utf-8"))
        elif isinstance(value, bytes):
            return len(value)
        else:
            return len(str(value).encode("utf-8"))

    def _artifact_serializer(self, obj):

        if isinstance(obj, bytes):
            return {"__bytes__": base64.b64encode(obj).decode("utf-8")}
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _artifact_deserializer(self, dct):

        if "__bytes__" in dct:
            return base64.b64decode(dct["__bytes__"])
        return dct

    def _create_artifact_dir(self, program_id: str) -> str:

        base_path = getattr(self.config, "artifacts_base_path", None)
        if not base_path:
            base_path = (
                os.path.join(self.config.db_path or ".", "artifacts")
                if self.config.db_path
                else "./artifacts"
            )

        artifact_dir = os.path.join(base_path, program_id)
        os.makedirs(artifact_dir, exist_ok=True)
        return artifact_dir

    def _cleanup_old_artifacts(self, checkpoint_path: str) -> None:

        if not self.config.cleanup_old_artifacts:
            return

        artifacts_base_path = os.path.join(checkpoint_path, "artifacts")

        if not os.path.isdir(artifacts_base_path):
            return

        now = time.time()
        retention_seconds = self.config.artifact_retention_days * 24 * 60 * 60
        deleted_count = 0

        logger.debug(f"Starting artifact cleanup in {artifacts_base_path}...")

        for dirname in os.listdir(artifacts_base_path):
            dirpath = os.path.join(artifacts_base_path, dirname)
            if os.path.isdir(dirpath):
                try:
                    dir_mod_time = os.path.getmtime(dirpath)
                    if (now - dir_mod_time) > retention_seconds:
                        shutil.rmtree(dirpath)
                        deleted_count += 1
                        logger.debug(f"Removed old artifact directory: {dirpath}")
                except FileNotFoundError:

                    continue
                except Exception as e:
                    logger.error(f"Error removing artifact directory {dirpath}: {e}")

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old artifact directories.")

    def _write_artifact_file(self, artifact_dir: str, key: str, value: Union[str, bytes]) -> None:


        safe_key = "".join(c for c in key if c.isalnum() or c in "._-")
        if not safe_key:
            safe_key = "artifact"

        file_path = os.path.join(artifact_dir, safe_key)

        try:
            if isinstance(value, str):
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(value)
            elif isinstance(value, bytes):
                with open(file_path, "wb") as f:
                    f.write(value)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(value, f, default=self._artifact_serializer, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write artifact {key} to {file_path}: {e}")

    def _load_artifact_dir(self, artifact_dir: str) -> Dict[str, Union[str, bytes]]:

        artifacts = {}

        try:
            for filename in os.listdir(artifact_dir):
                file_path = os.path.join(artifact_dir, filename)
                if os.path.isfile(file_path):
                    try:

                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        artifacts[filename] = content
                    except UnicodeDecodeError:

                        with open(file_path, "rb") as f:
                            content = f.read()
                        artifacts[filename] = content
                    except Exception as e:
                        logger.warning(f"Failed to read artifact file {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to list artifact directory {artifact_dir}: {e}")

        return artifacts

    def log_prompt(
        self,
        program_id: str,
        template_key: str,
        prompt: Dict[str, str],
        responses: Optional[List[str]] = None,
    ) -> None:


        if not self.config.log_prompts:
            return

        if responses is None:
            responses = []
        prompt["responses"] = responses

        if self.prompts_by_program is None:
            self.prompts_by_program = {}

        if program_id not in self.prompts_by_program:
            self.prompts_by_program[program_id] = {}
        self.prompts_by_program[program_id][template_key] = prompt
