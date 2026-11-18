

import asyncio
from copy import deepcopy
import inspect
import logging
import multiprocessing as mp
import os
import pickle
import signal
import time
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from outerloop.config import Config
from outerloop.candidate_validation import (
    CandidateValidationError,
    configured_proposal_seed,
    duplicate_proposal_rejection,
    is_duplicate_code_error,
    proposal_generation_seed,
    proposal_repair_instruction,
    repair_attempt,
    summarize_repairs,
)
from outerloop.database import Program, ProgramDatabase
from outerloop.evolution_policy import EvolutionCandidate
from outerloop.generation_lifecycle import (
    GenerationCoordinator,
    GenerationObservation,
    GenerationPlan,
    generation_chunks,
    proposal_causal_envelope_from_parent,
    target_island_for_iteration,
)
from outerloop.history_admission import (
    CodeAdmissionError,
    claim_code_for_evaluation,
)
from engine.history_lifecycle import DuplicateEffectiveContractError
from engine.parent_lineage import bind_selected_parent_lineages
from outerloop.design_action_parent_binding import (
    build_design_action_parent_binding,
    build_structured_design_action_prompt_context,
    design_action_execution_constraints,
)
from outerloop.phenotype_runtime import seal_effective_phenotype
from outerloop.reasoning_audit import (
    seal_reasoning_prediction,
    structured_reasoning_artifacts,
)
from astevolve.evaluation.selection import select_feasibility_first
from astevolve.runtime.edit_contract_lifecycle import (
    envelope_from_parent_artifacts,
    finalize_generated_contract_artifacts,
)
from outerloop.utils.metrics_utils import safe_numeric_average
from outerloop.utils.code_utils import CandidateDiffError
from outerloop.structured_ast_proposal import (
    STRUCTURED_AST_AUDIT_VERSION,
    STRUCTURED_AST_AUDIT_V2_VERSION,
    STRUCTURED_AST_PROPOSAL_MODE,
    apply_structured_ast_proposal,
    structured_ast_proposal_constraints,
    structured_ast_repair_instruction,
)
from outerloop.structured_design_action_proposal import (
    STRUCTURED_DESIGN_ACTION_CONTROLS,
    STRUCTURED_DESIGN_ACTION_PROPOSAL_MODE,
    apply_structured_design_action_proposal,
    structured_design_action_artifacts,
    structured_design_action_repair_instruction,
)

logger = logging.getLogger(__name__)


def logical_proposal_wave_size(config: Any) -> int:


    evaluator = getattr(config, "evaluator", None)
    physical_workers = max(
        1,
        int(getattr(evaluator, "parallel_evaluations", 1) or 1),
    )
    database = getattr(config, "database", None)
    if str(getattr(database, "outer_population_policy_version", "legacy")) != "v9":
        return physical_workers
    return max(1, int(getattr(database, "proposal_wave_size", 1) or 1))


def _queued_proposal_timeout_budget(
    single_proposal_timeout: float,
    queue_index: int,
    physical_workers: int,
) -> float:


    physical_batch = max(0, int(queue_index)) // max(1, int(physical_workers))
    return float(single_proposal_timeout) * (physical_batch + 1)


def _wait_for_processes(
    processes: Tuple[mp.Process, ...], timeout: float
) -> List[mp.Process]:


    deadline = time.monotonic() + timeout
    alive = list(processes)
    while alive:
        next_alive = []
        for process in alive:
            try:
                process.join(timeout=0)
                if process.is_alive():
                    next_alive.append(process)
            except (AssertionError, ValueError):


                continue
        alive = next_alive
        remaining = deadline - time.monotonic()
        if not alive or remaining <= 0:
            break
        time.sleep(min(0.001, remaining))
    return alive


def _terminate_process_pool(executor: ProcessPoolExecutor) -> None:


    process_map = getattr(executor, "_processes", None) or {}


    processes = tuple(process_map.copy().values())

    terminate_workers = getattr(executor, "terminate_workers", None)
    if callable(terminate_workers):

        terminate_workers()
    else:
        executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            try:
                if process.is_alive():
                    process.terminate()
            except (ProcessLookupError, ValueError):
                continue

    surviving_processes = _wait_for_processes(processes, timeout=1.0)
    for process in surviving_processes:
        try:
            process.kill()
        except (ProcessLookupError, ValueError):
            continue

    surviving_processes = _wait_for_processes(
        tuple(surviving_processes), timeout=1.0
    )
    if surviving_processes:
        logger.warning(
            "Process-pool workers did not exit: %s",
            [process.pid for process in surviving_processes],
        )


def _worker_context_island(
    parent: Program,
    db_snapshot: Mapping[str, Any],
    *,
    generation_plan: Optional[GenerationPlan] = None,
    proposal_id: Optional[str] = None,
) -> int:


    islands = db_snapshot.get("islands")
    if not isinstance(islands, list) or not islands:
        raise ValueError("worker snapshot must contain at least one island")

    reserved_island = None
    if generation_plan is not None:
        if not proposal_id:
            raise ValueError("proposal_id is required with generation_plan")
        reserved_island = generation_plan.reservation(proposal_id).target_island
    sampled_island = db_snapshot.get("sampling_island")
    if (
        reserved_island is not None
        and sampled_island is not None
        and int(reserved_island) != int(sampled_island)
    ):
        raise ValueError("reserved target island does not match worker snapshot")

    selected = (
        reserved_island
        if reserved_island is not None
        else sampled_island
        if sampled_island is not None
        else (parent.metadata or {}).get("island", db_snapshot.get("current_island", 0))
    )
    return int(selected) % len(islands)


def _order_worker_prompt_programs(
    programs: List[Program], config: Any
) -> List[Program]:


    database_config = getattr(config, "database", None)
    enabled = getattr(database_config, "outer_effective_phenotype_enabled", None)
    if enabled not in (None, False, True):
        raise ValueError("outer_effective_phenotype_enabled must be boolean")
    if enabled is not True:
        return sorted(
            programs,
            key=lambda program: program.metrics.get(
                "combined_score", safe_numeric_average(program.metrics)
            ),
            reverse=True,
        )

    candidates: List[EvolutionCandidate] = []
    by_id: Dict[str, Program] = {}
    for program in programs:
        raw_candidate = (program.metadata or {}).get("outer_evolution_candidate")
        if not isinstance(raw_candidate, Mapping):
            raise ValueError(
                f"program {program.id!r} lacks a validated outer evolution candidate"
            )
        candidate = EvolutionCandidate.from_mapping(raw_candidate)
        if candidate.candidate_id != program.id:
            raise ValueError("outer evolution candidate/program ID mismatch")
        candidates.append(candidate)
        by_id[program.id] = program
    if not candidates:
        return []
    ordering = select_feasibility_first(
        [candidate.selection_row() for candidate in candidates]
    )
    return [by_id[candidate_id] for candidate_id in ordering["ordered_ids"]]


@dataclass
class SerializableResult:


    child_program_dict: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    iteration_time: float = 0.0
    prompt: Optional[Dict[str, str]] = None
    llm_response: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None
    iteration: int = 0
    error: Optional[str] = None
    target_island: Optional[int] = None
    proposal_id: Optional[str] = None
    generation_id: str = ""
    trial_id: str = ""
    proposal_causal_envelope_json: str = ""
    proposal_causal_envelope_hash: str = ""

    @classmethod
    def failure(
        cls,
        error: Any,
        *,
        iteration: int,
        generation_plan: Optional[GenerationPlan] = None,
        proposal_id: Optional[str] = None,
        trial_id: str = "",
        parent_id: Optional[str] = None,
        target_island: Optional[int] = None,
        prompt: Optional[Dict[str, str]] = None,
        llm_response: Optional[str] = None,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> "SerializableResult":
        causal_json = ""
        causal_hash = ""
        if generation_plan is not None and proposal_id:
            reservation = generation_plan.reservation(proposal_id)
            causal_json = reservation.proposal_causal_envelope_json
            causal_hash = reservation.proposal_causal_envelope_hash
        return cls(
            parent_id=parent_id,
            iteration=int(iteration),
            prompt=prompt,
            llm_response=llm_response,
            error=str(error),
            artifacts=artifacts,
            target_island=target_island,
            proposal_id=proposal_id,
            generation_id=(generation_plan.generation_id if generation_plan else ""),
            trial_id=str(trial_id or ""),
            proposal_causal_envelope_json=causal_json,
            proposal_causal_envelope_hash=causal_hash,
        )


def _worker_init(config_dict: dict, evaluation_file: str, parent_env: dict = None) -> None:

    import os


    if parent_env:
        os.environ.update(parent_env)

    global _worker_config
    global _worker_evaluation_file
    global _worker_evaluator
    global _worker_llm_ensemble
    global _worker_prompt_sampler


    from outerloop.config import (
        Config,
        DatabaseConfig,
        EvaluatorConfig,
        EvolutionTraceConfig,
        LLMConfig,
        LLMModelConfig,
        PromptConfig,
    )
    from engine.memory_policy import MemoryPolicyConfig


    models = [LLMModelConfig(**m) for m in config_dict["llm"]["models"]]
    evaluator_models = [LLMModelConfig(**m) for m in config_dict["llm"]["evaluator_models"]]


    llm_dict = config_dict["llm"].copy()
    llm_dict["models"] = models
    llm_dict["evaluator_models"] = evaluator_models
    llm_config = LLMConfig(**llm_dict)


    prompt_config = PromptConfig(**config_dict["prompt"])
    database_config = DatabaseConfig(**config_dict["database"])
    evaluator_config = EvaluatorConfig(**config_dict["evaluator"])
    evolution_trace_config = EvolutionTraceConfig(
        **config_dict.get("evolution_trace", {})
    )
    memory_policy_config = MemoryPolicyConfig(
        **config_dict.get("memory_policy", {})
    )

    _worker_config = Config(
        llm=llm_config,
        prompt=prompt_config,
        database=database_config,
        evaluator=evaluator_config,
        evolution_trace=evolution_trace_config,
        memory_policy=memory_policy_config,
        **{
            k: v
            for k, v in config_dict.items()
            if k
            not in [
                "llm",
                "prompt",
                "database",
                "evaluator",
                "evolution_trace",
                "memory_policy",
            ]
        },
    )
    _worker_evaluation_file = evaluation_file


    _worker_evaluator = None
    _worker_llm_ensemble = None
    _worker_prompt_sampler = None


def _lazy_init_worker_components():

    global _worker_evaluator
    global _worker_llm_ensemble
    global _worker_prompt_sampler

    if _worker_llm_ensemble is None:
        from outerloop.llm.ensemble import LLMEnsemble

        _worker_llm_ensemble = LLMEnsemble(_worker_config.llm.models)

    if _worker_prompt_sampler is None:
        from outerloop.prompt.sampler import PromptSampler

        _worker_prompt_sampler = PromptSampler(_worker_config.prompt)

    if _worker_evaluator is None:
        from outerloop.evaluator import Evaluator

        evaluator_llm = None
        evaluator_prompt = None
        if _worker_config.evaluator.use_llm_feedback:
            from outerloop.llm.ensemble import LLMEnsemble
            from outerloop.prompt.sampler import PromptSampler

            evaluator_llm = LLMEnsemble(_worker_config.llm.evaluator_models)
            evaluator_prompt = PromptSampler(_worker_config.prompt)
            evaluator_prompt.set_templates("evaluator_system_message")

        _worker_evaluator = Evaluator(
            _worker_config.evaluator,
            _worker_evaluation_file,
            evaluator_llm,
            evaluator_prompt,
            database=None,
            suffix=getattr(_worker_config, "file_suffix", ".py"),
        )


def _run_iteration_worker(
    iteration: int,
    db_snapshot: Dict[str, Any],
    parent_id: str,
    inspiration_ids: List[str],
    generation_plan: Optional[GenerationPlan] = None,
    proposal_id: Optional[str] = None,
    trial_id: str = "",
) -> SerializableResult:

    sealed_attempt_artifacts: Dict[str, Any] = {}
    failure_prompt: Optional[Dict[str, str]] = None
    failure_llm_response: Optional[str] = None
    try:

        _lazy_init_worker_components()


        programs = {pid: Program(**prog_dict) for pid, prog_dict in db_snapshot["programs"].items()}

        parent = programs[parent_id]
        inspirations = [programs[pid] for pid in inspiration_ids if pid in programs]


        parent_artifacts = db_snapshot["artifacts"].get(parent_id)
        if generation_plan is not None:
            if not proposal_id:
                raise ValueError("proposal_id is required with generation_plan")
            reservation = generation_plan.reservation(proposal_id)
            if reservation.parent_program_id and reservation.parent_program_id != parent.id:
                raise ValueError("reserved parent does not match selected parent")
            envelope = reservation.edit_contract_envelope()
            if envelope is not None:
                parent_artifacts = {
                    **(parent_artifacts or {}),
                    "edit_contract_envelope": envelope,
                }


        parent_island = _worker_context_island(
            parent,
            db_snapshot,
            generation_plan=generation_plan,
            proposal_id=proposal_id,
        )
        island_programs = [
            programs[pid] for pid in db_snapshot["islands"][parent_island] if pid in programs
        ]


        island_programs = _order_worker_prompt_programs(
            island_programs, _worker_config
        )


        programs_for_prompt = island_programs[
            : _worker_config.prompt.num_top_programs + _worker_config.prompt.num_diverse_programs
        ]

        best_programs_only = island_programs[: _worker_config.prompt.num_top_programs]


        if _worker_config.prompt.programs_as_changes_description:
            parent_changes_desc = (
                parent.changes_description or _worker_config.prompt.initial_changes_description
            )
            child_changes_desc = parent_changes_desc
        else:
            parent_changes_desc = None
            child_changes_desc = None

        proposal_mode = str(
            getattr(
                _worker_config.prompt, "proposal_mode", "legacy_diff"
            )
            or "legacy_diff"
        )
        design_action_mode = (
            proposal_mode == STRUCTURED_DESIGN_ACTION_PROPOSAL_MODE
        )
        design_action_parent_binding = None
        structured_design_action_context = None
        design_action_execution_limits = {}
        if design_action_mode:
            if generation_plan is None or not proposal_id:
                raise ValueError(
                    "structured_design_action_v1 requires a reserved generation proposal"
                )
            design_action_parent_binding = build_design_action_parent_binding(
                parent_program_id=parent.id,
                parent_code=parent.code,
                parent_artifacts=parent_artifacts,
            )
            structured_design_action_context = (
                build_structured_design_action_prompt_context(
                    parent_code=parent.code,
                    parent_artifacts=parent_artifacts,
                    binding=design_action_parent_binding,
                    require_sequence_reconciliation=(
                        os.environ.get(
                            "ASTEVOLVE_STEP1_REQUIRE_SEQUENCE_RECONCILIATION",
                            (
                                "1"
                                if bool(getattr(_worker_config.prompt, "require_sequence_reconciliation", False))
                                else "0"
                            ),
                        )
                        == "1"
                    ),
                    require_position_distributions=(
                        os.environ.get(
                            "ASTEVOLVE_STEP2_REQUIRE_POSITION_DISTRIBUTIONS",
                            (
                                "1"
                                if bool(getattr(_worker_config.prompt, "require_position_distributions", False))
                                else "0"
                            ),
                        )
                        == "1"
                    ),
                    require_portfolio_capabilities=(
                        os.environ.get(
                            "ASTEVOLVE_STEP3_REQUIRE_PORTFOLIO_CAPABILITIES",
                            (
                                "1"
                                if bool(getattr(_worker_config.prompt, "require_portfolio_capabilities", False))
                                else "0"
                            ),
                        )
                        == "1"
                    ),
                    require_frozen_candidate_wave=(
                        os.environ.get(
                            "ASTEVOLVE_STEP4_REQUIRE_FROZEN_WAVE",
                            (
                                "1"
                                if bool(getattr(_worker_config.prompt, "require_frozen_candidate_wave", False))
                                else "0"
                            ),
                        )
                        == "1"
                    ),
                )
            )
            structured_design_action_context = {
                **structured_design_action_context,
                "parent_binding": deepcopy(design_action_parent_binding),
                "required_controls": deepcopy(
                    STRUCTURED_DESIGN_ACTION_CONTROLS
                ),
            }
            design_action_execution_limits = (
                design_action_execution_constraints(
                    structured_design_action_context
                )
            )

        prompt = _worker_prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=[p.to_dict() for p in best_programs_only],
            top_programs=[p.to_dict() for p in programs_for_prompt],
            inspirations=[p.to_dict() for p in inspirations],
            language=_worker_config.language,
            evolution_round=iteration,
            diff_based_evolution=_worker_config.diff_based_evolution,
            program_artifacts=parent_artifacts,
            feature_dimensions=db_snapshot.get("feature_dimensions", []),
            current_changes_description=parent_changes_desc,
            optimizer_memory=(
                generation_plan.prompt_memory_for(proposal_id)
                if generation_plan is not None and proposal_id
                else db_snapshot.get("optimizer_memory", {})
            ),
            island_role=db_snapshot.get("island_roles", {}).get(
                str(parent_island), {}
            ),
            global_summary=db_snapshot.get("global_summary", {}),
            structured_design_action_context=(
                structured_design_action_context if design_action_mode else None
            ),
        )
        failure_prompt = prompt

        iteration_start = time.time()
        runtime_memory_context = None
        if generation_plan is not None:
            if not proposal_id:
                raise ValueError("proposal_id is required with generation_plan")
            runtime_memory_context = generation_plan.runtime_context(
                proposal_id, trial_id=trial_id
            )
            runtime_memory_context = runtime_memory_context.with_island_identity(
                parent_island,
                db_snapshot.get("island_roles", {}).get(
                    str(parent_island), {}
                ),
            )
            runtime_memory_context = bind_selected_parent_lineages(
                runtime_memory_context,
                parent_program_id=parent.id,
                parent_artifacts=parent_artifacts,
            )

        retry_limit = max(
            0,
            int(
                getattr(
                    _worker_config.prompt, "candidate_diff_retries", 0
                )
                or 0
            ),
        )
        child_code = None
        child_id = None
        child_metrics = None
        code_admission = None
        llm_response = None
        last_candidate_error = None
        candidate_errors = []
        candidate_repair_attempts = []
        structured_application = None
        reasoning_prediction = {}
        ast_structured_mode = proposal_mode == STRUCTURED_AST_PROPOSAL_MODE
        structured_mode = ast_structured_mode or design_action_mode
        structured_constraints = (
            structured_ast_proposal_constraints(parent_artifacts)
            if ast_structured_mode
            else None
        )
        for attempt in range(retry_limit + 1):
            messages = [{"role": "user", "content": prompt["user"]}]
            if attempt and last_candidate_error is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            structured_design_action_repair_instruction(
                                last_candidate_error,
                                parent_code=parent.code,
                                expected_parent_binding=(
                                    design_action_parent_binding or {}
                                ),
                                edit_contract_envelope_json=(
                                    runtime_memory_context.edit_contract_envelope_json
                                    if runtime_memory_context is not None
                                    else ""
                                ),
                                previous_errors=tuple(candidate_errors),
                                **design_action_execution_limits,
                            )
                            if design_action_mode
                            else structured_ast_repair_instruction(
                                last_candidate_error,
                                parent_code=parent.code,
                                edit_contract_envelope_json=(
                                    runtime_memory_context.edit_contract_envelope_json
                                    if runtime_memory_context is not None
                                    else ""
                                ),
                                proposal_constraints=structured_constraints,
                                previous_errors=tuple(candidate_errors),
                                required_audit_version=(
                                    STRUCTURED_AST_AUDIT_V2_VERSION
                                    if bool(
                                        getattr(
                                            _worker_config.prompt,
                                            "hierarchical_audit_v2",
                                            False,
                                        )
                                    )
                                    else STRUCTURED_AST_AUDIT_VERSION
                                ),
                            )
                            if structured_mode
                            else proposal_repair_instruction(
                                last_candidate_error,
                                diff_based=bool(
                                    _worker_config.diff_based_evolution
                                ),
                            )
                        ),
                    }
                )
            try:
                generation_seed = proposal_generation_seed(
                    configured_proposal_seed(
                        getattr(_worker_config, "llm", None)
                    ),
                    attempt_index=attempt,
                )
                generation_kwargs = (
                    {"seed": generation_seed}
                    if structured_mode and generation_seed is not None
                    else {}
                )
                llm_response = asyncio.run(
                    _worker_llm_ensemble.generate_with_context(
                        system_message=prompt["system"],
                        messages=messages,
                        **generation_kwargs,
                    )
                )
                failure_llm_response = llm_response
            except Exception as exc:
                logger.error("LLM generation failed: %s", exc)
                return SerializableResult.failure(
                    f"LLM generation failed: {exc}",
                    iteration=iteration,
                    generation_plan=generation_plan,
                    proposal_id=proposal_id,
                    trial_id=trial_id,
                    parent_id=parent.id,
                    prompt=prompt,
                    llm_response=failure_llm_response,
                    artifacts=sealed_attempt_artifacts,
                )
            if llm_response is None:
                return SerializableResult.failure(
                    "LLM returned None response",
                    iteration=iteration,
                    generation_plan=generation_plan,
                    proposal_id=proposal_id,
                    trial_id=trial_id,
                    parent_id=parent.id,
                    prompt=prompt,
                    artifacts=sealed_attempt_artifacts,
                )

            if design_action_mode:
                try:
                    if runtime_memory_context is None:
                        raise ValueError(
                            "structured_design_action_v1 runtime context is missing"
                        )
                    candidate_runtime_memory_context = runtime_memory_context
                    structured_application = (
                        apply_structured_design_action_proposal(
                            parent.code,
                            llm_response,
                            expected_parent_binding=(
                                design_action_parent_binding or {}
                            ),
                            max_code_length=_worker_config.max_code_length,
                            edit_contract_envelope_json=(
                                runtime_memory_context.edit_contract_envelope_json
                            ),
                            **design_action_execution_limits,
                        )
                    )
                    sealed_attempt_artifacts = (
                        structured_design_action_artifacts(
                            structured_application,
                            expected_parent_binding=(
                                design_action_parent_binding or {}
                            ),
                        )
                    )
                    candidate_runtime_memory_context = (
                        runtime_memory_context.with_edit_contract_response(
                            structured_application.edit_contract_response
                        ).with_design_action(
                            structured_application.design_action,
                            parent_binding=(
                                design_action_parent_binding or {}
                            ),
                        )
                    )
                except CandidateDiffError as exc:
                    last_candidate_error = exc
                    candidate_errors.append(exc)
                    candidate_repair_attempts.append(
                        repair_attempt(exc, attempt + 1)
                    )
                    continue
                candidate_code = structured_application.code
                candidate_changes_desc = child_changes_desc
                candidate_changes_summary = (
                    structured_application.changes_summary
                )
            elif ast_structured_mode:
                try:
                    candidate_runtime_memory_context = runtime_memory_context
                    structured_application = apply_structured_ast_proposal(
                        parent.code,
                        llm_response,
                        max_code_length=_worker_config.max_code_length,
                        edit_contract_envelope_json=(
                            runtime_memory_context.edit_contract_envelope_json
                            if runtime_memory_context is not None
                            else ""
                        ),
                        hierarchical_design=(
                            generation_plan.hierarchical_design()
                            if generation_plan is not None
                            else {}
                        ),
                        expected_island_role_id=str(
                            (
                                db_snapshot.get("island_roles", {}).get(
                                    str(parent_island), {}
                                )
                                or {}
                            ).get("role_id", "")
                        ),
                        required_audit_version=(
                            STRUCTURED_AST_AUDIT_V2_VERSION
                            if bool(
                                getattr(
                                    _worker_config.prompt,
                                    "hierarchical_audit_v2",
                                    False,
                                )
                            )
                            else STRUCTURED_AST_AUDIT_VERSION
                        ),
                        proposal_critic_enabled=bool(
                            getattr(
                                getattr(
                                    _worker_config,
                                    "hierarchical_design",
                                    None,
                                ),
                                "proposal_critic_enabled",
                                True,
                            )
                        ),
                    )
                    reasoning_prediction = seal_reasoning_prediction(
                        proposal_id=str(proposal_id or f"iteration:{iteration}"),
                        parent_program_id=parent.id,
                        iteration=iteration,
                        island_id=parent_island,
                        audit=structured_application.audit,
                        ast_revision_plan=structured_application.plan,
                        hierarchical_design_hash=(
                            generation_plan.hierarchical_design_hash
                            if generation_plan is not None
                            else ""
                        ),
                    )
                    sealed_attempt_artifacts = structured_reasoning_artifacts(
                        structured_application,
                        reasoning_prediction,
                    )
                    if runtime_memory_context is not None:
                        candidate_runtime_memory_context = (
                            runtime_memory_context.with_edit_contract_response(
                                structured_application.edit_contract_response
                            )
                        )
                except CandidateDiffError as exc:
                    last_candidate_error = exc
                    candidate_errors.append(exc)
                    candidate_repair_attempts.append(
                        repair_attempt(exc, attempt + 1)
                    )
                    continue
                candidate_code = structured_application.code
                candidate_changes_desc = child_changes_desc
                candidate_changes_summary = (
                    structured_application.changes_summary
                )
            elif _worker_config.diff_based_evolution:
                candidate_runtime_memory_context = runtime_memory_context
                from outerloop.utils.code_utils import (
                    apply_candidate_diffs,
                    format_diff_summary,
                )

                try:
                    application = apply_candidate_diffs(
                        parent.code,
                        llm_response,
                        diff_pattern=_worker_config.diff_pattern,
                        changes_description=parent_changes_desc,
                        require_changes_description_update=(
                            _worker_config.prompt.programs_as_changes_description
                        ),
                    )
                except CandidateDiffError as exc:
                    last_candidate_error = exc
                    candidate_errors.append(exc)
                    candidate_repair_attempts.append(
                        repair_attempt(exc, attempt + 1)
                    )
                    continue
                candidate_code = application.code
                candidate_changes_desc = application.changes_description
                summary_diffs = (
                    application.code_diffs or application.description_diffs
                )
                candidate_changes_summary = format_diff_summary(
                    list(summary_diffs),
                    max_line_len=_worker_config.prompt.diff_summary_max_line_len,
                    max_lines=_worker_config.prompt.diff_summary_max_lines,
                )
            else:
                candidate_runtime_memory_context = runtime_memory_context
                from outerloop.utils.code_utils import parse_full_rewrite

                candidate_code = parse_full_rewrite(
                    llm_response, _worker_config.language
                )
                if not candidate_code:
                    last_candidate_error = CandidateDiffError(
                        "invalid_full_rewrite",
                        "No valid code found in response",
                    )
                    candidate_errors.append(last_candidate_error)
                    candidate_repair_attempts.append(
                        repair_attempt(last_candidate_error, attempt + 1)
                    )
                    continue
                candidate_changes_desc = child_changes_desc
                candidate_changes_summary = "Full rewrite"

            validator = getattr(
                _worker_evaluator, "validate_candidate_program", None
            )
            try:
                if callable(validator):
                    asyncio.run(
                        validator(
                            candidate_code,
                            runtime_memory_context=(
                                candidate_runtime_memory_context
                            ),
                        )
                    )
            except CandidateValidationError as exc:
                last_candidate_error = exc
                candidate_errors.append(exc)
                candidate_repair_attempts.append(
                    repair_attempt(exc, attempt + 1)
                )
                continue
            except DuplicateEffectiveContractError as exc:
                last_candidate_error = exc
                candidate_errors.append(exc)
                candidate_repair_attempts.append(
                    repair_attempt(exc, attempt + 1)
                )
                logger.info(
                    "Worker iteration %s duplicate effective contract rejected "
                    "by pre-evaluator validation (attempt %s/%s)",
                    iteration + 1,
                    attempt + 1,
                    retry_limit + 1,
                )
                continue

            if len(candidate_code) > _worker_config.max_code_length:
                return SerializableResult.failure(
                    "Generated code exceeds maximum length "
                    f"({len(candidate_code)} > "
                    f"{_worker_config.max_code_length})",
                    iteration=iteration,
                    generation_plan=generation_plan,
                    proposal_id=proposal_id,
                    trial_id=trial_id,
                    parent_id=parent.id,
                    prompt=prompt,
                    llm_response=llm_response,
                    artifacts=sealed_attempt_artifacts,
                )


            import uuid

            candidate_id = str(uuid.uuid4())
            candidate_admission = None
            try:
                candidate_admission = claim_code_for_evaluation(
                    candidate_code,
                    getattr(_worker_config, "database", None),
                    owner_token=(
                        f"worker:{generation_plan.generation_id}:{proposal_id}:"
                        f"{trial_id or candidate_id}"
                        if generation_plan is not None
                        else f"worker:{iteration}:{candidate_id}"
                    ),
                )
            except CodeAdmissionError as exc:
                if not is_duplicate_code_error(exc):
                    raise
                last_candidate_error = exc
                candidate_errors.append(exc)
                candidate_repair_attempts.append(
                    repair_attempt(exc, attempt + 1)
                )
                logger.info(
                    "Worker iteration %s duplicate code rejected before "
                    "evaluator (attempt %s/%s)",
                    iteration + 1,
                    attempt + 1,
                    retry_limit + 1,
                )
                continue

            try:
                candidate_metrics = asyncio.run(
                    _worker_evaluator.evaluate_program(
                        candidate_code,
                        candidate_id,
                        runtime_memory_context=candidate_runtime_memory_context,
                    )
                )
            except DuplicateEffectiveContractError as exc:
                if candidate_admission is not None:
                    candidate_admission.fail(exc)
                last_candidate_error = exc
                candidate_errors.append(exc)
                candidate_repair_attempts.append(
                    repair_attempt(exc, attempt + 1)
                )
                logger.info(
                    "Worker iteration %s duplicate effective contract rejected "
                    "before provider execution (attempt %s/%s)",
                    iteration + 1,
                    attempt + 1,
                    retry_limit + 1,
                )
                continue
            except Exception as exc:
                if candidate_admission is not None:
                    candidate_admission.fail(exc)
                raise

            if candidate_admission is not None:
                candidate_admission.complete()
            child_code = candidate_code
            child_changes_desc = candidate_changes_desc
            changes_summary = candidate_changes_summary
            child_id = candidate_id
            code_admission = candidate_admission
            child_metrics = candidate_metrics
            break

        if child_code is None:
            assert last_candidate_error is not None
            failure_error = last_candidate_error
            rejection_artifacts = {
                **sealed_attempt_artifacts,
                "candidate_repair_summary": summarize_repairs(
                    candidate_repair_attempts
                )
            }
            if isinstance(last_candidate_error, CandidateValidationError):
                rejection_artifacts["candidate_validation_rejection"] = (
                    last_candidate_error.to_dict()
                )
            elif isinstance(
                last_candidate_error, DuplicateEffectiveContractError
            ) or is_duplicate_code_error(last_candidate_error):
                compact_rejection = duplicate_proposal_rejection(
                    last_candidate_error
                )
                rejection_artifacts["duplicate_proposal_rejection"] = (
                    compact_rejection
                )
                failure_error = compact_rejection["error_code"]
                if is_duplicate_code_error(last_candidate_error):
                    rejection_artifacts["code_admission_rejection"] = (
                        compact_rejection
                    )
            return SerializableResult.failure(
                failure_error,
                iteration=iteration,
                generation_plan=generation_plan,
                proposal_id=proposal_id,
                trial_id=trial_id,
                parent_id=parent.id,
                target_island=db_snapshot.get("sampling_island"),
                prompt=prompt,
                llm_response=llm_response,
                artifacts=rejection_artifacts,
            )

        assert child_id is not None
        assert child_metrics is not None


        artifacts = _worker_evaluator.get_pending_artifacts(child_id)
        artifacts = {**(artifacts or {}), **sealed_attempt_artifacts}
        artifacts = finalize_generated_contract_artifacts(
            artifacts,
            parent_program_id=child_id,
            generated_generation_id=(
                generation_plan.generation_id if generation_plan is not None else ""
            ),
            generated_proposal_id=proposal_id or "",
        )
        if code_admission is not None:
            artifacts = {
                **(artifacts or {}),
                "code_admission": code_admission.to_artifact(),
            }
        phenotype_metadata = seal_effective_phenotype(
            child_code,
            artifacts or {},
            child_metrics,
            getattr(_worker_config, "database", None),
        )
        if phenotype_metadata:
            artifacts = {**(artifacts or {}), **phenotype_metadata}
        if candidate_repair_attempts:
            artifacts = {
                **(artifacts or {}),
                "candidate_repair_summary": summarize_repairs(
                    candidate_repair_attempts
                ),
            }


        child_program = Program(
            id=child_id,
            code=child_code,
            changes_description=child_changes_desc,
            language=_worker_config.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=child_metrics,
            iteration_found=iteration,
            metadata={
                **phenotype_metadata,
                "changes": changes_summary,
                "parent_metrics": parent.metrics,
                "island": parent_island,
                "generation_id": generation_plan.generation_id if generation_plan else None,
                "proposal_id": proposal_id,
                "trial_id": str(trial_id or ""),
                "proposal_causal_envelope_hash": (
                    generation_plan.reservation(
                        proposal_id
                    ).proposal_causal_envelope_hash
                    if generation_plan and proposal_id
                    else None
                ),
                "code_hash": (
                    code_admission.identity.code_hash if code_admission else None
                ),
                "outer_memory_input_hash": (
                    generation_plan.outer_input_hash if generation_plan else None
                ),
                "inner_memory_input_hash": (
                    generation_plan.inner_memory.content_hash
                    if generation_plan and generation_plan.inner_memory
                    else None
                ),
            },
        )

        iteration_time = time.time() - iteration_start


        target_island = db_snapshot.get("sampling_island")

        return SerializableResult(
            child_program_dict=child_program.to_dict(),
            parent_id=parent.id,
            iteration_time=iteration_time,
            prompt=prompt,
            llm_response=llm_response,
            artifacts=artifacts,
            iteration=iteration,
            target_island=target_island,
            proposal_id=proposal_id,
            generation_id=generation_plan.generation_id if generation_plan else "",
            trial_id=str(trial_id or ""),
            proposal_causal_envelope_json=(
                generation_plan.reservation(
                    proposal_id
                ).proposal_causal_envelope_json
                if generation_plan and proposal_id
                else ""
            ),
            proposal_causal_envelope_hash=(
                generation_plan.reservation(
                    proposal_id
                ).proposal_causal_envelope_hash
                if generation_plan and proposal_id
                else ""
            ),
        )

    except Exception as e:
        logger.exception(f"Error in worker iteration {iteration}")
        return SerializableResult.failure(
            str(e),
            iteration=iteration,
            generation_plan=generation_plan,
            proposal_id=proposal_id,
            trial_id=trial_id,
            parent_id=parent_id,
            prompt=failure_prompt,
            llm_response=failure_llm_response,
            artifacts=sealed_attempt_artifacts,
        )


class ProcessParallelController:


    def __init__(
        self,
        config: Config,
        evaluation_file: str,
        database: ProgramDatabase,
        evolution_tracer=None,
        file_suffix: str = ".py",
        inner_memory_path: Optional[str] = None,
        memory_scope: Any = None,
    ):
        self.config = config
        self.evaluation_file = evaluation_file
        self.database = database
        self.evolution_tracer = evolution_tracer
        self.file_suffix = file_suffix

        self.executor: Optional[ProcessPoolExecutor] = None
        self.shutdown_event = mp.Event()
        self.early_stopping_triggered = False


        self.num_workers = max(1, int(config.evaluator.parallel_evaluations or 1))
        self.logical_wave_size = logical_proposal_wave_size(config)
        self.num_islands = config.database.num_islands
        optimizer_memory = getattr(database, "optimizer_memory", None)
        self.inner_memory_path = inner_memory_path or getattr(
            optimizer_memory, "case_memory_path", None
        )
        self.memory_scope = memory_scope or getattr(optimizer_memory, "scope", None)
        artifacts_root = getattr(config.database, "artifacts_base_path", None)
        self.generation_output_root = str(
            Path(artifacts_root).parent / "generation_runtime"
            if artifacts_root
            else Path("generation_runtime")
        )

        logger.info(
            "Initialized process parallel controller with "
            f"{self.num_workers} workers and logical proposal waves of "
            f"{self.logical_wave_size}"
        )

    def _serialize_config(self, config: Config) -> dict:


        config.database.novelty_llm = None

        return {
            "llm": {
                "models": [asdict(m) for m in config.llm.models],
                "evaluator_models": [asdict(m) for m in config.llm.evaluator_models],
                "api_base": config.llm.api_base,
                "api_key": config.llm.api_key,
                "temperature": config.llm.temperature,
                "top_p": config.llm.top_p,
                "max_tokens": config.llm.max_tokens,
                "timeout": config.llm.timeout,
                "retries": config.llm.retries,
                "retry_delay": config.llm.retry_delay,
            },
            "prompt": asdict(config.prompt),
            "database": asdict(config.database),
            "evaluator": asdict(config.evaluator),
            "evolution_trace": asdict(config.evolution_trace),
            "hierarchical_design": asdict(config.hierarchical_design),
            "memory_policy": asdict(config.memory_policy),
            "max_iterations": config.max_iterations,
            "checkpoint_interval": config.checkpoint_interval,
            "log_level": config.log_level,
            "log_dir": config.log_dir,
            "random_seed": config.random_seed,
            "diff_based_evolution": config.diff_based_evolution,
            "diff_pattern": config.diff_pattern,
            "max_code_length": config.max_code_length,
            "language": config.language,
            "file_suffix": self.file_suffix,
        }

    def start(self) -> None:


        config_dict = self._serialize_config(self.config)


        import os
        import sys

        current_env = dict(os.environ)

        executor_kwargs = {
            "max_workers": self.num_workers,
            "initializer": _worker_init,
            "initargs": (config_dict, self.evaluation_file, current_env),
        }
        executor_kwargs["mp_context"] = mp.get_context("spawn")
        logger.info("Using spawn process context for CUDA-safe workers")
        if sys.version_info >= (3, 11):
            logger.info(f"Set max {self.config.max_tasks_per_child} tasks per child")
            executor_kwargs["max_tasks_per_child"] = self.config.max_tasks_per_child
        elif self.config.max_tasks_per_child is not None:
            logger.warn(
                "max_tasks_per_child is only supported in Python 3.11+. "
                "Ignoring max_tasks_per_child and using spawn start method."
            )


        self.executor = ProcessPoolExecutor(**executor_kwargs)
        logger.info(f"Started process pool with {self.num_workers} processes")

    def stop(self) -> None:

        self.shutdown_event.set()

        executor = self.executor
        self.executor = None
        if executor:
            _terminate_process_pool(executor)

        logger.info("Stopped process pool")

    def request_shutdown(self) -> None:

        logger.info("Graceful shutdown requested...")
        self.shutdown_event.set()

    def _prompt_context_program_ids(self, island_id: int) -> Tuple[str, ...]:


        resolved_island = int(island_id) % len(self.database.islands)
        island_programs = [
            self.database.programs[program_id]
            for program_id in self.database.islands[resolved_island]
            if program_id in self.database.programs
        ]
        ordered = _order_worker_prompt_programs(island_programs, self.config)
        limit = int(self.config.prompt.num_top_programs) + int(
            getattr(
                self.config.prompt,
                "num_diverse_programs",
                self.config.prompt.num_top_programs,
            )
        )
        return tuple(program.id for program in ordered[: max(0, limit)])

    def _create_database_snapshot(
        self,
        *,
        program_ids: Optional[Iterable[str]] = None,
        island_program_ids: Optional[Mapping[int, Iterable[str]]] = None,
        artifact_program_ids: Optional[Iterable[str]] = None,
        artifacts_by_program: Optional[Mapping[str, Any]] = None,
        include_optimizer_memory: bool = True,
    ) -> Dict[str, Any]:


        def stable_ids(values: Iterable[str]) -> List[str]:
            seen = set()
            ordered = []
            for value in values:
                program_id = str(value)
                if program_id not in seen:
                    seen.add(program_id)
                    ordered.append(program_id)
            return ordered

        if program_ids is None:
            selected_program_ids = list(self.database.programs)
        else:
            selected_program_ids = stable_ids(program_ids)

        if island_program_ids is None:
            selected_islands = [list(island) for island in self.database.islands]
        else:
            selected_islands = [[] for _ in self.database.islands]
            for raw_island_id, values in island_program_ids.items():
                island_id = int(raw_island_id)
                if not 0 <= island_id < len(selected_islands):
                    raise ValueError(f"snapshot island out of range: {island_id}")
                selected_islands[island_id] = stable_ids(values)
                selected_program_ids.extend(selected_islands[island_id])
            selected_program_ids = stable_ids(selected_program_ids)

        unknown_program_ids = [
            program_id
            for program_id in selected_program_ids
            if program_id not in self.database.programs
        ]
        if unknown_program_ids:
            raise ValueError(
                "snapshot contains unknown program IDs: "
                + ",".join(unknown_program_ids)
            )

        snapshot = {
            "programs": {
                program_id: self.database.programs[program_id].to_dict()
                for program_id in selected_program_ids
            },
            "islands": selected_islands,
            "current_island": self.database.current_island,
            "feature_dimensions": self.database.config.feature_dimensions,
            "island_roles": {
                str(island_id): self.database.get_island_role(island_id)
                for island_id in range(len(self.database.islands))
            }
            if hasattr(self.database, "get_island_role")
            else {},
            "global_summary": self.database.get_global_summary(top_n=6)
            if hasattr(self.database, "get_global_summary")
            else {},
            "optimizer_memory": (
                self.database.get_optimizer_memory_snapshot()
                if include_optimizer_memory
                and hasattr(self.database, "get_optimizer_memory_snapshot")
                else {}
            ),
            "artifacts": {},
        }

        if artifact_program_ids is None:
            selected_artifact_ids = list(self.database.programs)
            max_artifacts = self.database.config.max_snapshot_artifacts
            if max_artifacts is not None:
                selected_artifact_ids = selected_artifact_ids[:max_artifacts]
        else:


            selected_artifact_ids = stable_ids(artifact_program_ids)
        provided_artifacts = artifacts_by_program or {}
        for program_id in selected_artifact_ids:
            if program_id not in self.database.programs:
                raise ValueError(f"snapshot artifact program unknown: {program_id}")
            artifacts = (
                provided_artifacts[program_id]
                if program_id in provided_artifacts
                else self.database.get_artifacts(program_id)
            )
            if artifacts:
                snapshot["artifacts"][program_id] = artifacts

        return snapshot

    async def run_evolution(
        self,
        start_iteration: int,
        max_iterations: int,
        target_score: Optional[float] = None,
        checkpoint_callback=None,
        pre_wave_callback=None,
    ):

        if not self.executor:
            raise RuntimeError("Process pool not started")
        logger.info(
            f"Starting process-based evolution from iteration {start_iteration} "
            f"for {max_iterations} iterations in fixed logical waves of "
            f"{self.logical_wave_size} using {self.num_workers} workers"
        )
        early_stopping_enabled = self.config.early_stopping_patience is not None
        best_score = float("-inf")
        iterations_without_improvement = 0
        if early_stopping_enabled:
            if self.config.early_stopping_patience < 0:
                logger.info(
                    f"Early stopping patience is set to a negative value, running event-based early-stopping, "
                    f"Early stop when metric '{self.config.early_stopping_metric}' reaches {self.config.convergence_threshold}"
                )
            else:
                logger.info(
                    f"Early stopping enabled: patience={self.config.early_stopping_patience}, "
                    f"threshold={self.config.convergence_threshold}, "
                    f"metric={self.config.early_stopping_metric}"
                )
        else:
            logger.info("Early stopping disabled")
        for wave in generation_chunks(
            start_iteration,
            max_iterations,
            self.logical_wave_size,
        ):
            if self.shutdown_event.is_set() or self.early_stopping_triggered:
                break
            if pre_wave_callback is not None:
                callback_result = pre_wave_callback(min(wave))
                if inspect.isawaitable(callback_result):
                    await callback_result
            target_islands = tuple(
                target_island_for_iteration(iteration, self.num_islands)
                for iteration in wave
            )
            coordinator = GenerationCoordinator.begin(
                self.database,
                iterations=wave,
                target_islands=target_islands,
                inner_memory_path=self.inner_memory_path,
                output_root=self.generation_output_root,
                history_registry_path=(
                    getattr(self.config.database, "experiment_registry_path", "")
                    if getattr(
                        self.config.database, "experiment_registry_enabled", False
                    )
                    else ""
                ),
                history_scope=str(
                    getattr(
                        self.config.database, "experiment_registry_scope", ""
                    )
                    or ""
                ),
                history_lease_seconds=(
                    getattr(
                        self.config.database,
                        "experiment_registry_lease_seconds",
                        300.0,
                    )
                ),
                history_replicate_policy=(
                    getattr(
                        self.config.database,
                        "experiment_registry_replicate_policy",
                        "reject",
                    )
                ),
                memory_scope=self.memory_scope,
                memory_policy=self.config.memory_policy,
            )
            sampled_inputs = {}
            parent_bindings = {}
            parent_artifacts_by_id = {}
            for reservation in coordinator.plan.reservations:
                parent, inspirations = self.database.sample_from_island(
                    island_id=reservation.target_island or 0,
                    num_inspirations=getattr(
                        self.config.prompt,
                        "num_diverse_programs",
                        self.config.prompt.num_top_programs,
                    ),
                )
                sampled_inputs[reservation.proposal_id] = (parent, inspirations)
                if parent.id not in parent_artifacts_by_id:
                    parent_artifacts_by_id[parent.id] = self.database.get_artifacts(
                        parent.id
                    )
                parent_artifacts = parent_artifacts_by_id[parent.id]
                envelope = envelope_from_parent_artifacts(
                    parent_artifacts,
                    parent_program_id=parent.id,
                )
                causal_envelope = proposal_causal_envelope_from_parent(
                    parent_program_id=parent.id,
                    parent_code=parent.code,
                    parent_artifacts=parent_artifacts,
                )
                parent_bindings[reservation.proposal_id] = (
                    parent.id,
                    envelope.to_dict() if envelope is not None else None,
                    causal_envelope,
                )
            coordinator = coordinator.bind_parents(parent_bindings)

            reservations = {}
            pending: Dict[str, Future] = {}
            submitted_at: Dict[str, float] = {}
            timeout_budgets: Dict[str, float] = {}
            timeout_seconds = self.config.evaluator.timeout + 30
            for queue_index, reservation in enumerate(coordinator.plan.reservations):
                parent, inspirations = sampled_inputs[reservation.proposal_id]
                reservations[reservation.proposal_id] = (reservation, parent, inspirations)
                target_island = int(reservation.target_island or 0) % len(
                    self.database.islands
                )
                prompt_program_ids = self._prompt_context_program_ids(target_island)
                required_program_ids = (
                    parent.id,
                    *(item.id for item in inspirations),
                    *prompt_program_ids,
                )
                worker_snapshot = self._create_database_snapshot(
                    program_ids=required_program_ids,
                    island_program_ids={target_island: prompt_program_ids},
                    artifact_program_ids=(parent.id,),
                    artifacts_by_program=parent_artifacts_by_id,
                    include_optimizer_memory=False,
                )
                worker_snapshot["sampling_island"] = reservation.target_island
                future = self._submit_iteration(
                    reservation.iteration,
                    reservation.target_island,
                    generation_plan=coordinator.plan,
                    proposal_id=reservation.proposal_id,
                    db_snapshot=worker_snapshot,
                    parent_id=parent.id,
                    inspiration_ids=[item.id for item in inspirations],
                )
                if future is not None:
                    pending[reservation.proposal_id] = future
                    submitted_at[reservation.proposal_id] = time.monotonic()


                    timeout_budgets[reservation.proposal_id] = (
                        _queued_proposal_timeout_budget(
                            timeout_seconds,
                            queue_index,
                            self.num_workers,
                        )
                    )

            completed: Dict[str, SerializableResult] = {}
            while pending and not self.shutdown_event.is_set():
                made_progress = False
                for proposal_id, future in list(pending.items()):
                    elapsed = time.monotonic() - submitted_at[proposal_id]
                    timeout_budget = timeout_budgets[proposal_id]
                    if not future.done() and elapsed <= timeout_budget:
                        continue
                    made_progress = True
                    pending.pop(proposal_id)
                    reservation, parent, _inspirations = reservations[proposal_id]
                    if not future.done():
                        future.cancel()
                        completed[proposal_id] = SerializableResult.failure(
                            f"evaluation timed out after {timeout_budget}s",
                            iteration=reservation.iteration,
                            generation_plan=coordinator.plan,
                            proposal_id=proposal_id,
                            parent_id=parent.id,
                            target_island=reservation.target_island,
                        )
                        continue
                    try:
                        completed[proposal_id] = future.result()
                    except Exception as exc:
                        completed[proposal_id] = SerializableResult.failure(
                            str(exc),
                            iteration=reservation.iteration,
                            generation_plan=coordinator.plan,
                            proposal_id=proposal_id,
                            parent_id=parent.id,
                            target_island=reservation.target_island,
                        )
                if not made_progress:
                    await asyncio.sleep(0.01)

            if self.shutdown_event.is_set():
                logger.info("Shutdown requested; discarding the incomplete generation")
                for future in pending.values():
                    future.cancel()
                break

            observations: List[GenerationObservation] = []
            for proposal_id, (reservation, parent, _inspirations) in reservations.items():
                result = completed.get(proposal_id)
                if result is None:
                    result = SerializableResult.failure(
                        "worker submission failed",
                        iteration=reservation.iteration,
                        generation_plan=coordinator.plan,
                        proposal_id=proposal_id,
                        parent_id=parent.id,
                        target_island=reservation.target_island,
                    )
                child = (
                    Program.from_dict(result.child_program_dict)
                    if result.child_program_dict
                    else None
                )
                observations.append(
                    GenerationObservation(
                        proposal_id=proposal_id,
                        iteration=reservation.iteration,
                        child_program=child,
                        parent=parent,
                        artifacts=result.artifacts or {},
                        target_island=reservation.target_island,
                        prompt=result.prompt,
                        llm_response=result.llm_response,
                        iteration_time=result.iteration_time,
                        error=result.error,
                        generation_id=result.generation_id,
                        trial_id=result.trial_id,
                        proposal_causal_envelope_json=(
                            result.proposal_causal_envelope_json
                        ),
                        proposal_causal_envelope_hash=(
                            result.proposal_causal_envelope_hash
                        ),
                    )
                )

            lifecycle = coordinator.finalize(self.database, observations)

            for observation in sorted(observations, key=lambda item: item.stable_key):
                child = observation.child_program
                if child is None:
                    logger.warning(
                        f"Iteration {observation.iteration} error: {observation.error}"
                    )
                    if observation.prompt and observation.parent is not None:
                        self.database.log_prompt(
                            template_key=(
                                "structured_ast_user_failed"
                                if str(
                                    getattr(
                                        self.config.prompt,
                                        "proposal_mode",
                                        "legacy_diff",
                                    )
                                )
                                == STRUCTURED_AST_PROPOSAL_MODE
                                else "full_rewrite_user"
                                if not self.config.diff_based_evolution
                                else "diff_user_failed"
                            ),
                            program_id=observation.parent.id,
                            prompt=observation.prompt,
                            responses=[observation.llm_response]
                            if observation.llm_response
                            else [],
                        )
                    continue
                if self.evolution_tracer and observation.parent is not None:
                    self.evolution_tracer.log_trace(
                        iteration=observation.iteration,
                        parent_program=observation.parent,
                        child_program=child,
                        prompt=observation.prompt,
                        llm_response=observation.llm_response,
                        artifacts=observation.artifacts,
                        island_id=(
                            observation.target_island
                            if observation.target_island is not None
                            else self.database.current_island
                        ),
                        metadata={
                            "iteration_time": observation.iteration_time,
                            "changes": child.metadata.get("changes", ""),
                            "generation_id": coordinator.plan.generation_id,
                        },
                    )
                island_id = child.metadata.get("island", self.database.current_island)
                self.database.increment_island_generation(island_idx=island_id)
                logger.info(
                    f"Iteration {observation.iteration}: Program {child.id} "
                    f"(parent: {getattr(observation.parent, 'id', None)}) "
                    f"completed in {observation.iteration_time:.2f}s"
                )
                if child.metrics:
                    logger.info(
                        "Metrics: "
                        + ", ".join(
                            f"{key}={value:.4f}"
                            if isinstance(value, (int, float))
                            else f"{key}={value}"
                            for key, value in child.metrics.items()
                        )
                    )
                    if "combined_score" not in child.metrics and not getattr(
                        self, "_warned_about_combined_score", False
                    ):
                        avg_score = safe_numeric_average(child.metrics)
                        logger.warning(
                            "No 'combined_score' metric found; using numeric average "
                            f"({avg_score:.4f}) for evolution guidance."
                        )
                        self._warned_about_combined_score = True

                if early_stopping_enabled and child.metrics:
                    current_score = child.metrics.get(self.config.early_stopping_metric)
                    if current_score is None:
                        current_score = safe_numeric_average(child.metrics)
                    if isinstance(current_score, (int, float)):
                        if self.config.early_stopping_patience > 0:
                            improvement = current_score - best_score
                            if improvement >= self.config.convergence_threshold:
                                best_score = current_score
                                iterations_without_improvement = 0
                            else:
                                iterations_without_improvement += 1
                            if (
                                iterations_without_improvement
                                >= self.config.early_stopping_patience
                            ):
                                self.early_stopping_triggered = True
                        elif current_score == self.config.convergence_threshold:
                            best_score = current_score
                            self.early_stopping_triggered = True

            if self.database.should_migrate():
                logger.info(
                    f"Performing migration after generation {coordinator.plan.generation_id}"
                )
                self.database.migrate_programs()
                self.database.log_island_status()

            checkpoint_points = [
                iteration
                for iteration in wave
                if iteration > 0 and iteration % self.config.checkpoint_interval == 0
            ]
            if checkpoint_points and checkpoint_callback:
                checkpoint_iteration = max(wave)
                self.database.log_island_status()
                checkpoint_callback(checkpoint_iteration)

            winner_id = (lifecycle.get("winner") or {}).get("program_id")
            winner = self.database.get(winner_id) if winner_id else None
            if (
                target_score is not None
                and winner is not None
                and isinstance(winner.metrics.get("combined_score"), (int, float))
                and winner.metrics["combined_score"] >= target_score
            ):
                logger.info(
                    f"Target score {target_score} reached after generation "
                    f"{coordinator.plan.generation_id}"
                )
                break


        if self.early_stopping_triggered:
            logger.info("✅ Evolution completed - Early stopping triggered due to convergence")
        elif self.shutdown_event.is_set():
            logger.info("✅ Evolution completed - Shutdown requested")
        else:
            logger.info("✅ Evolution completed - Maximum iterations reached")

        return self.database.get_best_program()

    def _submit_iteration(
        self,
        iteration: int,
        island_id: Optional[int] = None,
        *,
        generation_plan: Optional[GenerationPlan] = None,
        proposal_id: Optional[str] = None,
        db_snapshot: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        inspiration_ids: Optional[List[str]] = None,
    ) -> Optional[Future]:

        try:

            target_island = island_id if island_id is not None else self.database.current_island

            if parent_id is None:


                parent, inspirations = self.database.sample_from_island(
                    island_id=target_island,
                    num_inspirations=getattr(
                        self.config.prompt,
                        "num_diverse_programs",
                        self.config.prompt.num_top_programs,
                    ),
                )
                parent_id = parent.id
                inspiration_ids = [item.id for item in inspirations]


            if db_snapshot is None:
                prompt_program_ids = self._prompt_context_program_ids(target_island)
                db_snapshot = self._create_database_snapshot(
                    program_ids=(
                        parent_id,
                        *(inspiration_ids or []),
                        *prompt_program_ids,
                    ),
                    island_program_ids={target_island: prompt_program_ids},
                    artifact_program_ids=(parent_id,),
                    include_optimizer_memory=generation_plan is None,
                )
            else:
                db_snapshot = dict(db_snapshot)
            db_snapshot["sampling_island"] = target_island
            if generation_plan is not None:
                if not proposal_id:
                    raise ValueError("proposal_id is required with generation_plan")


            future = self.executor.submit(
                _run_iteration_worker,
                iteration,
                db_snapshot,
                parent_id,
                list(inspiration_ids or []),
                generation_plan,
                proposal_id,
            )

            return future

        except Exception as e:
            logger.error(f"Error submitting iteration {iteration}: {e}")
            return None
