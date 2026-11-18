

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

from engine.memory_policy import MemoryScope
from outerloop.memory_facts import (
    OuterObservationFact,
    _OptimizerMemoryProjectionCache,
    _prepare_scoped_fact_ledger,
)
from outerloop.utils.metric_semantics import (
    METRIC_SEMANTICS_VERSION,
    compare_metric,
    compare_metrics,
    summarize_comparisons,
)

logger = logging.getLogger(__name__)


SCORE_METRICS = (
    "combined_score",
    "struct_score",
    "structure_score",
    "multistate_score",
    "evaluator_score",
    "fast_score",
    "plddt_score",
    "iptm_score",
    "ptm_score",
)

LOSS_METRICS = (
    "combined_energy",
    "final_energy",
    "design_loss",
    "total_loss",
    "fast_loss",
    "multistate_loss",
    "evaluator_loss",
    "evaluator_energy",
)

ISLAND_PROFILES = (
    {
        "name": "fold_stability",
        "mission": "Repair or preserve global fold confidence before expanding functional edits.",
        "categories": ("fold_stability",),
        "metrics": ("struct_score", "structure_score", "plddt", "ptm", "hard_gate_pass"),
    },
    {
        "name": "interface_contact",
        "mission": "Improve desired interface/contact geometry and epitope or peptide engagement.",
        "categories": ("interface_contact",),
        "metrics": ("interface_contact_count", "interface_plddt_mean", "iptm", "multistate_score"),
    },
    {
        "name": "specificity_negative_design",
        "mission": "Improve target specificity while penalizing off-target/source-state retention.",
        "categories": ("specificity_negative_design",),
        "metrics": ("specificity_score", "negative_design_score", "multistate_score"),
    },
    {
        "name": "allostery_pocket",
        "mission": "Improve ligand pocket, allosteric path, conformational switch, or binding-state coupling.",
        "categories": ("allostery_pocket",),
        "metrics": ("pocket_score", "ligand_score", "allostery_score", "multistate_score"),
    },
)

CATEGORY_KEYWORDS = {
    "fold_stability": (
        "fold",
        "stability",
        "plddt",
        "ptm",
        "clash",
        "hard_gate",
        "framework",
        "guardrail",
    ),
    "interface_contact": (
        "interface",
        "contact",
        "epitope",
        "binding",
        "peptide",
        "cdr",
        "iptm",
    ),
    "specificity_negative_design": (
        "specificity",
        "negative",
        "off_target",
        "counter",
        "source",
        "orthogonal",
        "selectivity",
    ),
    "allostery_pocket": (
        "allostery",
        "allosteric",
        "pocket",
        "ligand",
        "small_molecule",
        "switch",
        "hinge",
        "dna",
    ),
}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    except (TypeError, ValueError):
        return None


def _safe_json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _json_contains(value: Any, needles: Iterable[str]) -> bool:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str).lower()
    except TypeError:
        text = str(value).lower()
    return any(needle.lower() in text for needle in needles)


def _count_items(counts: Dict[str, float], values: Any, amount: float = 1.0) -> None:
    if values is None:
        return
    if isinstance(values, dict):
        items = values.items()
        for key, value in items:
            numeric = _safe_float(value)
            counts[str(key)] = counts.get(str(key), 0.0) + (numeric if numeric is not None else amount)
        return
    if not isinstance(values, list):
        values = [values]
    for value in values:
        if value is None:
            continue
        counts[str(value)] = counts.get(str(value), 0.0) + amount


def _rank_counts(counts: Dict[str, float], limit: int = 8) -> List[Dict[str, Any]]:
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [{"name": key, "count": value} for key, value in ranked[:limit]]


def _metric_value(metrics: Dict[str, Any], name: str) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(name)
    if isinstance(value, dict):
        value = value.get("value", value.get("score"))
    return _safe_float(value)


def _short_list(values: Any, limit: int = 12) -> List[Any]:
    if values is None:
        return []
    if isinstance(values, dict):
        values = list(values.keys())
    if not isinstance(values, list):
        values = [values]
    out = []
    for value in values:
        if value is None:
            continue
        out.append(value)
        if len(out) >= limit:
            break
    return out


def _load_yaml_or_text(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        logger.warning("Could not read case memory %s: %s", path, exc)
        return {}

    try:
        import yaml

        loaded = yaml.safe_load(text) or {}
        if isinstance(loaded, dict):
            return loaded
        return {"raw": loaded}
    except Exception:
        return {"raw_text": text[:12000]}


def _compact_case_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    metadata = memory.get("metadata", {}) if isinstance(memory.get("metadata"), dict) else {}
    stable_priors = (
        memory.get("stable_priors", {}) if isinstance(memory.get("stable_priors"), dict) else {}
    )
    compact = {
        "metadata": {
            key: metadata.get(key)
            for key in (
                "memory_name",
                "task_type",
                "species_context",
                "status",
                "last_manual_update",
                "last_auto_update",
            )
            if metadata.get(key) is not None
        },
        "llm_guidance": memory.get("llm_guidance", {}),
        "stable_priors": stable_priors,
        "adaptive_memory": memory.get("adaptive_memory", {}),
        "optimization_windows": memory.get("optimization_windows", {}),
        "known_failure": memory.get("known_failure", memory.get("known_failures", [])),
        "safety_and_guardrails": memory.get("safety_and_guardrails", {}),
    }
    return {key: value for key, value in compact.items() if value not in ({}, [], None)}


def _summary_from_artifacts(artifacts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    artifacts = artifacts or {}
    for key in ("optimizer_memory_summary", "llm_optimizer_memory_summary"):
        parsed = _safe_json_loads(artifacts.get(key))
        if isinstance(parsed, dict):
            _attach_contract_artifacts(parsed, artifacts)
            return parsed

    feedback = _safe_json_loads(artifacts.get("llm_feedback_summary"))
    if not isinstance(feedback, dict):
        return {}

    score_summary = feedback.get("score_summary", {}) or {}
    mechanism = feedback.get("mechanism_summary", {}) or {}
    instruction = mechanism.get("inner_loop_instruction_following", {}) or {}
    candidate = feedback.get("inner_loop_candidate_comparison", {}) or {}
    aggregate = feedback.get("aggregate_structure", {}) or {}
    summary = {
        "source": "llm_feedback_summary",
        "score": score_summary,
        "contract": feedback.get("contract_response_report", {}),
        "functional_nodes": {
            "scores": feedback.get("functional_node_scores", {}),
            "missing": instruction.get("missing_functional_nodes", []),
            "success_counts": instruction.get("functional_node_success_counts", {}),
            "failure_counts": instruction.get("functional_node_failure_counts", {}),
        },
        "structural_nodes": {
            "missing": instruction.get("missing_required_nodes", []),
            "success_counts": instruction.get("node_success_counts", {}),
            "failure_counts": instruction.get("node_failure_counts", {}),
            "low_confidence": (aggregate.get("node_summary", {}) or {}).get(
                "low_confidence_nodes", []
            ),
        },
        "best_candidates": candidate.get("best_candidates", []),
        "next_round": {
            "action_hints": mechanism.get("outer_loop_action_hint", []),
            "recommended_edit_targets": (feedback.get("evaluator", {}) or {}).get(
                "recommended_edit_targets", []
            ),
        },
    }
    _attach_contract_artifacts(summary, artifacts)
    return summary


def _attach_contract_artifacts(summary: Dict[str, Any], artifacts: Dict[str, Any]) -> None:
    contract = summary.setdefault("contract", {})
    if not isinstance(contract, dict):
        contract = {}
        summary["contract"] = contract
    for key in (
        "contract_response_report",
        "last_contract_response",
        "applied_edit_contract",
        "edit_contract",
        "edit_contract_envelope",
        "edit_contract_lifecycle",
    ):
        value = _safe_json_loads(artifacts.get(key))
        if value not in (None, "", {}, []):
            contract[key] = value


class OptimizerMemoryStore:


    def __init__(
        self,
        path: str,
        *,
        case_memory_path: Optional[str] = None,
        scope: Optional[MemoryScope] = None,
        recent_limit: int = 20,
        best_limit: int = 5,
    ) -> None:
        self.path = path
        self.scope = scope

        self.case_memory_path = case_memory_path
        self.recent_limit = max(1, int(recent_limit))
        self.best_limit = max(1, int(best_limit))


        self._projection_cache: Optional[_OptimizerMemoryProjectionCache] = None
        self._scheduler_signal_cache: Dict[
            tuple[str, int, float], Dict[str, Any]
        ] = {}
        self.state: Dict[str, Any] = {
            "schema_version": "astevolve.outer_memory_projection.v1",
            "created_at_unix": int(time.time()),
            "updated_at_unix": int(time.time()),
            "recent_attempts": [],
            "top_programs": [],
            "mandatory_prompt_capsule": {},
            "trajectory_memory": [],
            "best_candidate_memory": [],
            "program_summaries": {},
            "projection_only": True,
            "fact_source": "program_database",
        }
        self.load(path)

    def load(self, path: Optional[str] = None) -> None:


        self._projection_cache = None
        self._scheduler_signal_cache.clear()
        load_path = path or self.path
        if not load_path or not os.path.exists(load_path):
            return
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):


                loaded.pop("case_memory", None)
                loaded.pop("case_memory_path", None)
                loaded.setdefault(
                    "recent_attempts", loaded.get("trajectory_memory", [])
                )
                loaded.setdefault(
                    "top_programs", loaded.get("best_candidate_memory", [])
                )
                loaded.setdefault("mandatory_prompt_capsule", {})
                loaded.setdefault("trajectory_memory", [])
                loaded.setdefault("best_candidate_memory", [])
                loaded.setdefault("program_summaries", {})
                loaded.setdefault("projection_only", True)
                loaded.setdefault("fact_source", "program_database")
                self.state.update(loaded)
                logger.info("Loaded optimizer memory from %s", load_path)
        except Exception as exc:
            logger.warning("Failed to load optimizer memory from %s: %s", load_path, exc)

    def save(self, path: Optional[str] = None, *, touch_timestamp: bool = True) -> None:
        save_path = path or self.path
        if not save_path:
            return
        if touch_timestamp:
            self.state["updated_at_unix"] = int(time.time())
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{os.path.basename(save_path)}.",
                suffix=".tmp",
                dir=os.path.dirname(save_path) or ".",
                delete=False,
            ) as f:
                temp_name = f.name
                json.dump(self.state, f, ensure_ascii=False, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_name, save_path)
            temp_name = None
        finally:
            if temp_name:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass

    def snapshot(self) -> Dict[str, Any]:
        scheduler = self.scheduler_signal()
        snapshot = {
            "schema_version": self.state.get("schema_version"),
            "scope": deepcopy(self.state.get("scope")),
            "source_fact_ledger_hash": self.state.get("source_fact_ledger_hash"),
            "source_fact_count": self.state.get("source_fact_count", 0),
            "recent_attempts": deepcopy(
                self.state.get("recent_attempts")
                or self.state.get("trajectory_memory", [])
            ),
            "top_programs": deepcopy(
                self.state.get("top_programs")
                or self.state.get("best_candidate_memory", [])
            ),
            "mandatory_prompt_capsule": deepcopy(
                self.state.get("mandatory_prompt_capsule", {})
            ),
            "trajectory_memory": deepcopy(self.state.get("trajectory_memory", [])),
            "best_candidate_memory": deepcopy(self.state.get("best_candidate_memory", [])),
            "program_summaries": deepcopy(self.state.get("program_summaries", {})),
            "projection_only": True,
            "fact_source": "program_database",
            "scheduler": scheduler,
        }
        return {key: value for key, value in snapshot.items() if value is not None}

    def rebuild_from_facts(
        self,
        facts: Iterable[OuterObservationFact],
        *,
        scope: Optional[MemoryScope] = None,
        persist: bool = True,
    ) -> Dict[str, Any]:


        effective_scope = scope or self.scope
        if not isinstance(effective_scope, MemoryScope):
            raise ValueError("rebuild_from_facts requires an explicit MemoryScope")
        values = list(facts or [])
        prepared = _prepare_scoped_fact_ledger(values, effective_scope)
        cache = self._projection_cache
        if cache is None or not cache.matches(
            prepared,
            scope=effective_scope,
            recent_limit=self.recent_limit,
            best_limit=self.best_limit,
        ):
            cache = _OptimizerMemoryProjectionCache(
                prepared,
                scope=effective_scope,
                recent_limit=self.recent_limit,
                best_limit=self.best_limit,
            )
            self._projection_cache = cache
        projection = cache.projection()
        self.scope = effective_scope


        self.state = projection
        self._scheduler_signal_cache.clear()
        if persist:
            self.save(touch_timestamp=False)

            self.state.pop("updated_at_unix", None)
        return self.snapshot()

    def extend_from_facts(
        self,
        facts: Iterable[OuterObservationFact],
        *,
        scope: Optional[MemoryScope] = None,
        expected_source_fact_ledger_hash: Optional[str],
        persist: bool = True,
    ) -> bool:


        effective_scope = scope or self.scope
        cache = self._projection_cache
        expected_hash = str(expected_source_fact_ledger_hash or "")
        if (
            cache is None
            or not isinstance(effective_scope, MemoryScope)
            or cache.scope != effective_scope
            or cache.recent_limit != self.recent_limit
            or cache.best_limit != self.best_limit
            or not expected_hash
            or cache.source_fact_ledger_hash != expected_hash
            or self.state.get("source_fact_ledger_hash") != expected_hash
            or int(self.state.get("source_fact_count", -1))
            != cache.source_fact_count
        ):
            return False
        values = list(facts or [])
        if not values:
            return True
        cache.extend(
            values,
            expected_source_fact_ledger_hash=expected_hash,
        )
        self.scope = effective_scope
        self.state = cache.projection()
        self._scheduler_signal_cache.clear()
        if persist:
            self.save(touch_timestamp=False)
            self.state.pop("updated_at_unix", None)
        return True

    def scheduler_signal(
        self,
        *,
        stagnation_window: int = 5,
        stagnation_min_delta: float = 1e-6,
    ) -> Dict[str, Any]:

        window = max(2, int(stagnation_window or 5))
        min_delta = float(stagnation_min_delta)
        projection_cache = self._projection_cache
        source_hash = self.state.get("source_fact_ledger_hash")
        cache_key = None
        if (
            projection_cache is not None
            and source_hash == projection_cache.source_fact_ledger_hash
            and int(self.state.get("source_fact_count", -1))
            == projection_cache.source_fact_count
        ):
            cache_key = (str(source_hash), window, min_delta)
            cached = self._scheduler_signal_cache.get(cache_key)
            if cached is not None:
                return deepcopy(cached)
        trajectory = list(
            self.state.get("recent_attempts")
            or self.state.get("trajectory_memory", [])
            or []
        )
        program_summaries = self.state.get("program_summaries", {}) or {}
        scores = [score for score in (self._entry_score(entry) for entry in trajectory) if score is not None]

        latest_score = scores[-1] if scores else None
        best_score = max(scores) if scores else None
        best_index = scores.index(best_score) if best_score is not None else None
        iterations_since_improvement = (
            None if best_index is None else max(0, len(scores) - best_index - 1)
        )
        recent_scores = scores[-window:]
        prior_scores = scores[:-window]
        recent_best = max(recent_scores) if recent_scores else None
        prior_best = max(prior_scores) if prior_scores else None
        stagnated = bool(
            len(scores) >= window
            and prior_best is not None
            and recent_best is not None
            and recent_best <= prior_best + min_delta
        )

        weak_functional_nodes = self._aggregate_weak_functional(trajectory)
        weak_structural_nodes = self._aggregate_weak_structural(trajectory)
        contract_attention = self._contract_attention(trajectory, program_summaries)
        specialists = self._objective_specialists(program_summaries)

        if stagnated:
            phase = "escape_stagnation"
        elif weak_functional_nodes or weak_structural_nodes or contract_attention.get("needs_response"):
            phase = "repair_ast_coverage"
        elif specialists:
            phase = "refine_specialists"
        else:
            phase = "explore_ast_coverage"

        prompt_control = self._prompt_control(
            phase=phase,
            stagnated=stagnated,
            weak_functional_nodes=weak_functional_nodes,
            weak_structural_nodes=weak_structural_nodes,
            contract_attention=contract_attention,
        )

        signal = {
            "schema_version": 1,
            "phase": phase,
            "score_trend": {
                "latest_combined_score": latest_score,
                "best_combined_score": best_score,
                "iterations_since_improvement": iterations_since_improvement,
                "stagnated": stagnated,
                "stagnation_window": window,
            },
            "weak_functional_nodes": weak_functional_nodes,
            "weak_structural_nodes": weak_structural_nodes,
            "contract_attention": contract_attention,
            "objective_specialists": specialists[:8],
            "island_profiles": self.island_profiles(),
            "prompt_control": prompt_control,
        }
        if cache_key is not None:
            self._scheduler_signal_cache = {cache_key: deepcopy(signal)}
        return signal

    def island_profiles(self) -> List[Dict[str, Any]]:

        return [dict(profile) for profile in ISLAND_PROFILES]

    def program_scheduler_profile(
        self, program_id: str, metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        summary = (self.state.get("program_summaries", {}) or {}).get(program_id, {})
        return self._classify_summary(summary, metrics or {})

    def record_result(
        self,
        *,
        program: Any,
        parent: Any = None,
        artifacts: Optional[Dict[str, Any]] = None,
        iteration: Optional[int] = None,
        proposal_id: Optional[str] = None,
        persist: bool = True,
    ) -> Dict[str, Any]:


        self._projection_cache = None
        self._scheduler_signal_cache.clear()
        program_dict = program.to_dict() if hasattr(program, "to_dict") else dict(program or {})
        parent_dict = parent.to_dict() if hasattr(parent, "to_dict") else dict(parent or {})
        metrics = program_dict.get("metrics", {}) or {}
        parent_metrics = parent_dict.get("metrics", {}) or {}
        summary = _summary_from_artifacts(artifacts)

        entry = {
            "iteration": iteration if iteration is not None else program_dict.get("iteration_found"),
            "proposal_id": proposal_id,
            "program_id": program_dict.get("id"),
            "parent_id": program_dict.get("parent_id") or parent_dict.get("id"),
            "generation": program_dict.get("generation"),
            "edit": program_dict.get("changes_description")
            or (program_dict.get("metadata", {}) or {}).get("changes"),
            "hypothesis": self._extract_hypothesis(summary),
            "result": self._result_summary(metrics, parent_metrics, summary),
            "functional_nodes": self._functional_memory(summary),
            "structural_nodes": self._structural_memory(summary),
            "lesson": self._derive_lesson(metrics, parent_metrics, summary),
        }
        self.state.setdefault("trajectory_memory", []).append(entry)
        self.state["trajectory_memory"] = self.state["trajectory_memory"][-self.recent_limit :]
        self.state["recent_attempts"] = deepcopy(self.state["trajectory_memory"])
        if entry.get("program_id"):
            self.state.setdefault("program_summaries", {})[entry["program_id"]] = summary
        self._upsert_best_candidate(program_dict, summary)
        if persist:
            self.save()
        return entry

    def refresh_best_candidates(self, programs: Iterable[Any], *, persist: bool = True) -> None:
        self._projection_cache = None
        self._scheduler_signal_cache.clear()
        for program in programs or []:
            program_dict = program.to_dict() if hasattr(program, "to_dict") else dict(program or {})
            summary = self.state.get("program_summaries", {}).get(program_dict.get("id"), {})
            self._upsert_best_candidate(program_dict, summary)
        if persist:
            self.save()

    def record_generation_observations(
        self,
        observations: Iterable[Dict[str, Any]],
        *,
        best_candidates: Iterable[Any] = (),
        logical_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:


        self._projection_cache = None
        self._scheduler_signal_cache.clear()

        ordered = sorted(
            list(observations or []),
            key=lambda row: (
                int(row.get("iteration") or 0),
                str(row.get("proposal_id") or ""),
                str(getattr(row.get("program"), "id", "") or ""),
            ),
        )
        entries: List[Dict[str, Any]] = []
        for row in ordered:
            program = row.get("program")
            if program is not None:
                entry = self.record_result(
                    program=program,
                    parent=row.get("parent"),
                    artifacts=row.get("artifacts") or {},
                    iteration=row.get("iteration"),
                    proposal_id=str(row.get("proposal_id") or "") or None,
                    persist=False,
                )
            else:
                entry = {
                    "iteration": row.get("iteration"),
                    "proposal_id": str(row.get("proposal_id") or "") or None,
                    "program_id": None,
                    "parent_id": getattr(row.get("parent"), "id", None),
                    "generation": None,
                    "edit": None,
                    "hypothesis": None,
                    "result": {
                        "status": "failed",
                        "error": str(row.get("error") or "candidate_failed"),
                    },
                    "functional_nodes": {},
                    "structural_nodes": {},
                    "lesson": "Candidate did not produce an evaluable program.",
                }
                self.state.setdefault("trajectory_memory", []).append(entry)
                self.state["trajectory_memory"] = self.state["trajectory_memory"][
                    -self.recent_limit :
                ]
                self.state["recent_attempts"] = deepcopy(
                    self.state["trajectory_memory"]
                )
            entries.append(entry)

        self.refresh_best_candidates(best_candidates, persist=False)
        self.state["last_generation_observation"] = {
            "logical_time": logical_time,
            "proposal_ids": [str(row.get("proposal_id") or "") for row in ordered],
            "observation_count": len(ordered),
        }


        self.save(touch_timestamp=False)
        return entries

    def _entry_score(self, entry: Dict[str, Any]) -> Optional[float]:
        result = entry.get("result", {}) if isinstance(entry, dict) else {}
        score_metrics = result.get("score_metrics", {}) if isinstance(result, dict) else {}
        item = score_metrics.get("combined_score") if isinstance(score_metrics, dict) else None
        value = item.get("value") if isinstance(item, dict) else item
        score = _safe_float(value)
        if score is not None:
            return score


        metrics = entry.get("metrics", {}) if isinstance(entry, dict) else {}
        return _safe_float(metrics.get("combined_score")) if isinstance(metrics, dict) else None

    def _aggregate_weak_functional(self, trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        counts: Dict[str, float] = {}
        for entry in trajectory[-self.recent_limit :]:
            functional = entry.get("functional_nodes", {}) if isinstance(entry, dict) else {}
            _count_items(counts, functional.get("weak_or_failed", []), amount=2.0)
            _count_items(counts, functional.get("failure_counts", {}), amount=1.0)
        return _rank_counts(counts, limit=8)

    def _aggregate_weak_structural(self, trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        counts: Dict[str, float] = {}
        for entry in trajectory[-self.recent_limit :]:
            structural = entry.get("structural_nodes", {}) if isinstance(entry, dict) else {}
            _count_items(counts, structural.get("failed_or_missing", []), amount=2.0)
            _count_items(counts, structural.get("low_confidence", []), amount=1.0)
        return _rank_counts(counts, limit=8)

    def _contract_attention(
        self,
        trajectory: List[Dict[str, Any]],
        program_summaries: Dict[str, Any],
    ) -> Dict[str, Any]:
        recent_ids = [
            entry.get("program_id")
            for entry in trajectory[-self.recent_limit :]
            if isinstance(entry, dict) and entry.get("program_id")
        ]
        violations = []
        for program_id in recent_ids:
            summary = program_summaries.get(program_id, {})
            contract = summary.get("contract", {}) if isinstance(summary, dict) else {}
            report = contract.get("contract_response_report", contract)
            if isinstance(report, dict) and report.get("violation"):
                violations.append(
                    {
                        "program_id": program_id,
                        "warnings": _short_list(report.get("warnings", []), limit=4),
                    }
                )
            elif _json_contains(contract, ("last_contract_response is missing", "contract violation")) or _json_contains(
                summary, ("last_contract_response is missing", "contract violation")
            ):
                violations.append({"program_id": program_id, "warnings": ["contract response missing"]})
        return {
            "needs_response": bool(violations),
            "recent_violations": violations[-5:],
        }

    def _objective_specialists(self, program_summaries: Dict[str, Any]) -> List[Dict[str, Any]]:
        specialists = []
        for program_id, summary in (program_summaries or {}).items():
            profile = self._classify_summary(summary)
            if not profile.get("categories"):
                continue
            if not profile.get("strong_functional_nodes") and not profile.get("successful_structural_nodes"):
                if profile.get("specialist_score", 0.0) <= 0.0:
                    continue
            specialists.append(
                {
                    "program_id": program_id,
                    "categories": profile.get("categories", []),
                    "strong_functional_nodes": profile.get("strong_functional_nodes", [])[:4],
                    "successful_structural_nodes": profile.get("successful_structural_nodes", [])[:4],
                    "specialist_score": profile.get("specialist_score", 0.0),
                }
            )
        specialists.sort(key=lambda item: item.get("specialist_score", 0.0), reverse=True)
        return specialists

    def _classify_summary(
        self,
        summary: Optional[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        summary = summary if isinstance(summary, dict) else {}
        metrics = metrics if isinstance(metrics, dict) else {}
        score_summary = summary.get("score", {}) if isinstance(summary.get("score"), dict) else {}
        merged_metrics = dict(score_summary)
        merged_metrics.update(metrics)

        categories = set()
        metric_strengths = {}
        for profile in ISLAND_PROFILES:
            name = profile["name"]
            values = [
                _metric_value(merged_metrics, metric)
                for metric in profile.get("metrics", ())
                if _metric_value(merged_metrics, metric) is not None
            ]
            if values:
                metric_strengths[name] = max(values)
                if max(values) > 0.0:
                    categories.add(name)

        for name, keywords in CATEGORY_KEYWORDS.items():
            if _json_contains(summary, keywords):
                categories.add(name)

        functional = self._functional_memory(summary)
        structural = self._structural_memory(summary)
        contract = summary.get("contract", {}) if isinstance(summary.get("contract"), dict) else {}
        report = contract.get("contract_response_report", contract)
        contract_responsive = not (isinstance(report, dict) and report.get("violation"))
        specialist_score = max(metric_strengths.values()) if metric_strengths else 0.0
        return {
            "categories": sorted(categories),
            "metric_strengths": metric_strengths,
            "strong_functional_nodes": functional.get("strong", []),
            "weak_functional_nodes": functional.get("weak_or_failed", []),
            "successful_structural_nodes": structural.get("successful", []),
            "weak_structural_nodes": structural.get("failed_or_missing", []),
            "contract_responsive": contract_responsive,
            "specialist_score": specialist_score,
        }

    def _prompt_control(
        self,
        *,
        phase: str,
        stagnated: bool,
        weak_functional_nodes: List[Dict[str, Any]],
        weak_structural_nodes: List[Dict[str, Any]],
        contract_attention: Dict[str, Any],
    ) -> List[str]:
        controls = [f"Scheduler phase: {phase}."]
        if stagnated:
            controls.append(
                "Combined_score is stagnant; avoid another minor patch to the same strategy."
            )
            controls.append(
                "Change executable semantic_required_nodes or node_edit_policies before broadening generic mutation rates."
            )
            controls.append(
                "If the current branch is unstable, restart from a best candidate that preserved fold or interface terms."
            )
        if weak_functional_nodes:
            names = ", ".join(item["name"] for item in weak_functional_nodes[:5])
            controls.append(f"Must respond to weak functional nodes: {names}.")
        if weak_structural_nodes:
            names = ", ".join(item["name"] for item in weak_structural_nodes[:5])
            controls.append(f"Must cover or repair weak structural nodes: {names}.")
        if contract_attention.get("needs_response"):
            controls.append(
                "last_contract_response is required: explicitly accept/reject prior edit_contract items and list implemented_changes."
            )
        return controls

    def _upsert_best_candidate(self, program: Dict[str, Any], summary: Dict[str, Any]) -> None:
        program_id = program.get("id")
        if not program_id:
            return
        metrics = program.get("metrics", {}) or {}
        combined_score = _safe_float(metrics.get("combined_score"))
        design_loss = _safe_float(metrics.get("design_loss"))
        if design_loss is None:
            design_loss = _safe_float(metrics.get("total_loss"))
        candidate = {
            "program_id": program_id,
            "iteration": program.get("iteration_found"),
            "generation": program.get("generation"),
            "combined_score": combined_score,


            "fitness": combined_score,
            "fitness_source": "combined_score" if combined_score is not None else "unavailable",
            "design_loss": design_loss,
            "metrics": {
                key: metrics.get(key)
                for key in (
                    "combined_score",
                    "struct_score",
                    "multistate_score",
                    "evaluator_score",
                    "hard_gate_pass",
                    "design_loss",
                    "total_loss",
                    "plddt",
                    "iptm",
                    "ptm",
                )
                if key in metrics
            },
            "edit": program.get("changes_description")
            or (program.get("metadata", {}) or {}).get("changes"),
            "why_kept": self._best_candidate_reason(summary),
        }
        existing = [
            item
            for item in self.state.get("best_candidate_memory", [])
            if item.get("program_id") != program_id
        ]
        existing.append(candidate)
        existing.sort(
            key=lambda item: _safe_float(item.get("combined_score")) if _safe_float(item.get("combined_score")) is not None else -1e18,
            reverse=True,
        )
        self.state["best_candidate_memory"] = existing[: self.best_limit]
        self.state["top_programs"] = deepcopy(
            self.state["best_candidate_memory"]
        )

    def _result_summary(
        self,
        metrics: Dict[str, Any],
        parent_metrics: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        score_summary = summary.get("score", {}) if isinstance(summary.get("score"), dict) else {}
        observed_metrics = dict(score_summary)
        observed_metrics.update(metrics)
        names = set(parent_metrics) | set(observed_metrics) | {"combined_score"}
        comparisons = compare_metrics(parent_metrics, observed_metrics, names=names)
        comparison_summary = summarize_comparisons(comparisons)
        by_name = comparison_summary["comparisons"]


        score_delta = {}
        for key in SCORE_METRICS:
            item = by_name.get(key)
            if item and item["child"].get("value") is not None:
                score_delta[key] = {
                    "value": item["child"]["value"],
                    "delta": item["raw_delta"],
                    "improvement_delta": item["improvement_delta"],
                    "outcome": item["outcome"],
                }
        loss_delta = {}
        for key in LOSS_METRICS:
            item = by_name.get(key)
            if item and item["child"].get("value") is not None:
                loss_delta[key] = {
                    "value": item["child"]["value"],
                    "improvement": item["improvement_delta"],
                    "raw_delta": item["raw_delta"],
                    "outcome": item["outcome"],
                }
        return {
            "schema_version": METRIC_SEMANTICS_VERSION,
            "score_metrics": score_delta,
            "loss_metrics_lower_is_better": loss_delta,
            "overall_outcome": comparison_summary["overall_outcome"],
            "comparable_directional_count": comparison_summary["comparable_count"],
            "raw_delta": comparison_summary["raw_deltas"],
            "directional_improvement_delta": comparison_summary["directional_deltas"],
            "metric_comparisons": by_name,
            "hard_gate_pass": metrics.get("hard_gate_pass"),
        }

    def _functional_memory(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        functional = summary.get("functional_nodes", {}) if isinstance(summary, dict) else {}
        scores = functional.get("scores", {}) if isinstance(functional, dict) else {}
        weak = []
        strong = []
        if isinstance(scores, dict):
            for name, item in scores.items():
                if not isinstance(item, dict):
                    continue
                success_rate = _safe_float(item.get("success_rate"))
                if success_rate is not None and success_rate <= 0.0:
                    weak.append(name)
                elif success_rate is not None and success_rate > 0.0:
                    strong.append(name)
        return {
            "touched_or_scored": _short_list(scores, limit=16),
            "strong": _short_list(strong, limit=8),
            "weak_or_failed": _short_list(weak or functional.get("missing", []), limit=8),
            "success_counts": functional.get("success_counts", {}),
            "failure_counts": functional.get("failure_counts", {}),
        }

    def _structural_memory(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        structural = summary.get("structural_nodes", {}) if isinstance(summary, dict) else {}
        return {
            "successful": _short_list(structural.get("successful", structural.get("success_counts", {})), limit=10),
            "failed_or_missing": _short_list(
                structural.get("failed", structural.get("missing", structural.get("failure_counts", {}))),
                limit=10,
            ),
            "low_confidence": _short_list(structural.get("low_confidence", []), limit=10),
        }

    def _extract_hypothesis(self, summary: Dict[str, Any]) -> str:
        next_round = summary.get("next_round", {}) if isinstance(summary, dict) else {}
        hints = _short_list(next_round.get("action_hints", []), limit=1)
        if hints:
            return str(hints[0])
        return "Outer-loop edit is expected to improve combined_score while preserving AST guardrails."

    def _derive_lesson(
        self,
        metrics: Dict[str, Any],
        parent_metrics: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> str:
        combined = compare_metric(
            "combined_score",
            parent_metrics.get("combined_score") if "combined_score" in parent_metrics else None,
            metrics.get("combined_score") if "combined_score" in metrics else None,
        )
        hard_gate = metrics.get("hard_gate_pass")
        functional = self._functional_memory(summary)
        weak_nodes = functional.get("weak_or_failed", [])
        strong_nodes = functional.get("strong", [])
        if hard_gate == 0 or hard_gate is False:
            return "Hard gate failed; prioritize fold/interface guardrail repair before broadening mutations."
        if combined.outcome == "improved":
            suffix = f" Strong functional nodes: {', '.join(map(str, strong_nodes[:4]))}." if strong_nodes else ""
            return "Combined score improved; refine the touched strategy instead of resetting the case." + suffix
        if weak_nodes:
            return "No clear score gain; increase or repair coverage for weak functional nodes: " + ", ".join(map(str, weak_nodes[:5]))
        return "No clear score gain; use evaluator weakest terms and semantic coverage to narrow the next edit."

    def _best_candidate_reason(self, summary: Dict[str, Any]) -> str:
        functional = self._functional_memory(summary)
        structural = self._structural_memory(summary)
        pieces = []
        if functional.get("strong"):
            pieces.append("functional gains: " + ", ".join(map(str, functional["strong"][:4])))
        if structural.get("successful"):
            pieces.append("structural nodes touched: " + ", ".join(map(str, structural["successful"][:4])))
        return "; ".join(pieces) if pieces else "top combined_score candidate"
