import asyncio
from copy import deepcopy
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional

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
from outerloop.evaluator import Evaluator
from outerloop.generation_lifecycle import (
    GenerationPlan,
    proposal_causal_envelope_from_parent,
)
from outerloop.history_admission import (
    CodeAdmissionError,
    claim_code_for_evaluation,
)
from engine.history_lifecycle import DuplicateEffectiveContractError
from engine.parent_lineage import bind_selected_parent_lineages
from astevolve.runtime.edit_contract_lifecycle import (
    finalize_generated_contract_artifacts,
)
from outerloop.llm.ensemble import LLMEnsemble
from outerloop.prompt.sampler import PromptSampler
from outerloop.reasoning_audit import (
    seal_reasoning_prediction,
    structured_reasoning_artifacts,
)
from outerloop.phenotype_runtime import seal_effective_phenotype
from outerloop.design_action_parent_binding import (
    build_design_action_parent_binding,
    build_structured_design_action_prompt_context,
    design_action_execution_constraints,
)
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
from outerloop.utils.code_utils import (
    CandidateDiffError,
    apply_candidate_diffs,
    format_diff_summary,
    parse_full_rewrite,
)


@dataclass
class Result:


    child_program: str = None
    parent: str = None
    child_metrics: str = None
    iteration_time: float = None
    prompt: str = None
    llm_response: str = None
    artifacts: dict = None
    proposal_id: str = None
    generation_id: str = ""
    trial_id: str = ""
    proposal_causal_envelope_json: str = ""
    proposal_causal_envelope_hash: str = ""
    error: str = None

    @classmethod
    def failure(
        cls,
        error: Any,
        *,
        generation_plan: Optional[GenerationPlan] = None,
        proposal_id: Optional[str] = None,
        trial_id: str = "",
        parent: Any = None,
        prompt: Any = None,
        llm_response: Any = None,
        artifacts: Optional[dict] = None,
    ) -> "Result":
        generation_id = generation_plan.generation_id if generation_plan else ""
        causal_json = ""
        causal_hash = ""
        if generation_plan is not None and proposal_id:
            reservation = generation_plan.reservation(proposal_id)
            causal_json = reservation.proposal_causal_envelope_json
            causal_hash = reservation.proposal_causal_envelope_hash
        return cls(
            parent=parent,
            prompt=prompt,
            llm_response=llm_response,
            artifacts=artifacts,
            proposal_id=proposal_id,
            generation_id=generation_id,
            trial_id=str(trial_id or ""),
            proposal_causal_envelope_json=causal_json,
            proposal_causal_envelope_hash=causal_hash,
            error=str(error),
        )


async def run_iteration_with_shared_db(
    iteration: int,
    config: Config,
    database: ProgramDatabase,
    evaluator: Evaluator,
    llm_ensemble: LLMEnsemble,
    prompt_sampler: PromptSampler,
    generation_plan: Optional[GenerationPlan] = None,
    proposal_id: Optional[str] = None,
    trial_id: str = "",
    preselected_parent: Optional[Program] = None,
    preselected_inspirations: Optional[List[Program]] = None,
):

    logger = logging.getLogger(__name__)
    sealed_attempt_artifacts: dict[str, Any] = {}
    failure_parent = preselected_parent
    failure_prompt = None
    failure_llm_response = None

    try:


        if preselected_parent is None:
            parent, inspirations = database.sample(
                num_inspirations=getattr(
                    config.prompt,
                    "num_diverse_programs",
                    config.prompt.num_top_programs,
                )
            )
        else:
            parent = preselected_parent
            inspirations = list(preselected_inspirations or [])
        failure_parent = parent


        parent_artifacts = database.get_artifacts(parent.id)
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
            if not reservation.proposal_causal_envelope_json:
                generation_plan = generation_plan.with_proposal_causal_envelope(
                    proposal_id,
                    proposal_causal_envelope_from_parent(
                        parent_program_id=parent.id,
                        parent_code=parent.code,
                        parent_artifacts=parent_artifacts,
                    ),
                )
                reservation = generation_plan.reservation(proposal_id)
        optimizer_memory = (
            generation_plan.prompt_memory_for(proposal_id)
            if generation_plan is not None
            else database.get_optimizer_memory_snapshot()
            if hasattr(database, "get_optimizer_memory_snapshot")
            else {}
        )


        reserved_island = (
            reservation.target_island
            if generation_plan is not None and reservation.target_island is not None
            else None
        )
        parent_island = (
            int(reserved_island) % len(database.islands)
            if reserved_island is not None
            else parent.metadata.get("island", database.current_island)
        )
        island_top_programs = database.get_top_programs(5, island_idx=parent_island)
        island_previous_programs = database.get_top_programs(3, island_idx=parent_island)


        if config.prompt.programs_as_changes_description:
            parent_changes_desc = (
                parent.changes_description or config.prompt.initial_changes_description
            )
            child_changes_desc = parent_changes_desc
        else:
            parent_changes_desc = None
            child_changes_desc = None

        island_role = (
            database.get_island_role(parent_island)
            if hasattr(database, "get_island_role")
            else {}
        )
        global_summary = (
            database.get_global_summary(top_n=6)
            if hasattr(database, "get_global_summary")
            else {}
        )
        proposal_mode = str(
            getattr(config.prompt, "proposal_mode", "legacy_diff")
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
                                if bool(getattr(config.prompt, "require_sequence_reconciliation", False))
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
                                if bool(getattr(config.prompt, "require_position_distributions", False))
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
                                if bool(getattr(config.prompt, "require_portfolio_capabilities", False))
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
                                if bool(getattr(config.prompt, "require_frozen_candidate_wave", False))
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
        prompt = prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=[p.to_dict() for p in island_previous_programs],
            top_programs=[p.to_dict() for p in island_top_programs],
            inspirations=[p.to_dict() for p in inspirations],
            language=config.language,
            evolution_round=iteration,
            diff_based_evolution=config.diff_based_evolution,
            program_artifacts=parent_artifacts if parent_artifacts else None,
            feature_dimensions=database.config.feature_dimensions,
            current_changes_description=parent_changes_desc,
            optimizer_memory=optimizer_memory,
            island_role=island_role,
            global_summary=global_summary,
            structured_design_action_context=(
                structured_design_action_context if design_action_mode else None
            ),
        )
        failure_prompt = prompt

        reservation = (
            generation_plan.reservation(proposal_id)
            if generation_plan is not None and proposal_id
            else None
        )
        result = Result(
            parent=parent,
            proposal_id=proposal_id,
            generation_id=generation_plan.generation_id if generation_plan else "",
            trial_id=str(trial_id or ""),
            proposal_causal_envelope_json=(
                reservation.proposal_causal_envelope_json if reservation else ""
            ),
            proposal_causal_envelope_hash=(
                reservation.proposal_causal_envelope_hash if reservation else ""
            ),
        )
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
                island_role,
            )
            runtime_memory_context = bind_selected_parent_lineages(
                runtime_memory_context,
                parent_program_id=parent.id,
                parent_artifacts=parent_artifacts,
            )


        retry_limit = max(
            0, int(getattr(config.prompt, "candidate_diff_retries", 0) or 0)
        )
        llm_response = None
        child_code = None
        child_id = None
        code_admission = None
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
                                            config.prompt,
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
                                diff_based=bool(config.diff_based_evolution),
                            )
                        ),
                    }
                )
            generation_seed = proposal_generation_seed(
                configured_proposal_seed(getattr(config, "llm", None)),
                attempt_index=attempt,
            )
            generation_kwargs = (
                {"seed": generation_seed}
                if structured_mode and generation_seed is not None
                else {}
            )
            llm_response = await llm_ensemble.generate_with_context(
                system_message=prompt["system"],
                messages=messages,
                **generation_kwargs,
            )
            failure_llm_response = llm_response

            try:
                candidate_runtime_memory_context = runtime_memory_context
                if design_action_mode:
                    if runtime_memory_context is None:
                        raise ValueError(
                            "structured_design_action_v1 runtime context is missing"
                        )
                    structured_application = (
                        apply_structured_design_action_proposal(
                            parent.code,
                            llm_response,
                            expected_parent_binding=(
                                design_action_parent_binding or {}
                            ),
                            max_code_length=config.max_code_length,
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
                    candidate_code = structured_application.code
                    candidate_changes_desc = child_changes_desc
                    candidate_changes_summary = (
                        structured_application.changes_summary
                    )
                elif ast_structured_mode:
                    structured_application = apply_structured_ast_proposal(
                        parent.code,
                        llm_response,
                        max_code_length=config.max_code_length,
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
                                database.get_island_role(parent_island)
                                if hasattr(database, "get_island_role")
                                else {}
                            ).get("role_id", "")
                        ),
                        required_audit_version=(
                            STRUCTURED_AST_AUDIT_V2_VERSION
                            if bool(
                                getattr(
                                    config.prompt,
                                    "hierarchical_audit_v2",
                                    False,
                                )
                            )
                            else STRUCTURED_AST_AUDIT_VERSION
                        ),
                        proposal_critic_enabled=bool(
                            getattr(
                                getattr(config, "hierarchical_design", None),
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
                    candidate_code = structured_application.code
                    candidate_changes_desc = child_changes_desc
                    candidate_changes_summary = (
                        structured_application.changes_summary
                    )
                elif config.diff_based_evolution:
                    application = apply_candidate_diffs(
                        parent.code,
                        llm_response,
                        diff_pattern=config.diff_pattern,
                        changes_description=parent_changes_desc,
                        require_changes_description_update=(
                            config.prompt.programs_as_changes_description
                        ),
                    )
                    candidate_code = application.code
                    candidate_changes_desc = application.changes_description
                    summary_diffs = (
                        application.code_diffs or application.description_diffs
                    )
                    candidate_changes_summary = format_diff_summary(
                        list(summary_diffs),
                        max_line_len=config.prompt.diff_summary_max_line_len,
                        max_lines=config.prompt.diff_summary_max_lines,
                    )
                else:
                    candidate_code = parse_full_rewrite(
                        llm_response, config.language
                    )
                    if not candidate_code:
                        raise CandidateDiffError(
                            "invalid_full_rewrite",
                            "No valid code found in response",
                        )
                    candidate_changes_desc = child_changes_desc
                    candidate_changes_summary = "Full rewrite"
            except CandidateDiffError as exc:
                last_candidate_error = exc
                candidate_errors.append(exc)
                candidate_repair_attempts.append(repair_attempt(exc, attempt + 1))
                logger.warning(
                    "Iteration %s candidate format rejected (attempt %s/%s): %s",
                    iteration + 1,
                    attempt + 1,
                    retry_limit + 1,
                    exc.to_dict(),
                )
                continue

            validator = getattr(evaluator, "validate_candidate_program", None)
            try:
                if callable(validator):
                    await validator(
                        candidate_code,
                        runtime_memory_context=candidate_runtime_memory_context,
                    )
            except CandidateValidationError as exc:
                last_candidate_error = exc
                candidate_errors.append(exc)
                candidate_repair_attempts.append(repair_attempt(exc, attempt + 1))
                logger.warning(
                    "Iteration %s candidate semantics rejected (attempt %s/%s): %s",
                    iteration + 1,
                    attempt + 1,
                    retry_limit + 1,
                    exc.to_dict(),
                )
                continue
            except DuplicateEffectiveContractError as exc:
                last_candidate_error = exc
                candidate_errors.append(exc)
                candidate_repair_attempts.append(repair_attempt(exc, attempt + 1))
                logger.info(
                    "Iteration %s duplicate effective contract rejected by "
                    "pre-evaluator validation (attempt %s/%s)",
                    iteration + 1,
                    attempt + 1,
                    retry_limit + 1,
                )
                continue

            if len(candidate_code) > config.max_code_length:
                message = (
                    "Generated code exceeds maximum length "
                    f"({len(candidate_code)} > {config.max_code_length})"
                )
                logger.warning("Iteration %s: %s", iteration + 1, message)
                if generation_plan is not None:
                    return Result.failure(
                        message,
                        generation_plan=generation_plan,
                        proposal_id=proposal_id,
                        trial_id=trial_id,
                        parent=parent,
                        prompt=prompt,
                        llm_response=llm_response,
                        artifacts=sealed_attempt_artifacts,
                    )
                return None

            candidate_id = str(uuid.uuid4())
            candidate_admission = None
            try:
                candidate_admission = claim_code_for_evaluation(
                    candidate_code,
                    getattr(config, "database", None),
                    owner_token=(
                        f"outer:{generation_plan.generation_id}:{proposal_id}:"
                        f"{trial_id or candidate_id}"
                        if generation_plan is not None
                        else f"outer:{iteration}:{candidate_id}"
                    ),
                )
            except CodeAdmissionError as exc:
                if not is_duplicate_code_error(exc):
                    raise
                last_candidate_error = exc
                candidate_errors.append(exc)
                candidate_repair_attempts.append(repair_attempt(exc, attempt + 1))
                logger.info(
                    "Iteration %s duplicate code rejected before evaluator "
                    "(attempt %s/%s)",
                    iteration + 1,
                    attempt + 1,
                    retry_limit + 1,
                )
                continue

            try:
                candidate_metrics = await evaluator.evaluate_program(
                    candidate_code,
                    candidate_id,
                    runtime_memory_context=candidate_runtime_memory_context,
                )
            except DuplicateEffectiveContractError as exc:
                if candidate_admission is not None:
                    candidate_admission.fail(exc)
                last_candidate_error = exc
                candidate_errors.append(exc)
                candidate_repair_attempts.append(repair_attempt(exc, attempt + 1))
                logger.info(
                    "Iteration %s duplicate effective contract rejected before "
                    "provider execution (attempt %s/%s)",
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
            result.child_metrics = candidate_metrics
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
            if generation_plan is not None:
                return Result.failure(
                    failure_error,
                    generation_plan=generation_plan,
                    proposal_id=proposal_id,
                    trial_id=trial_id,
                    parent=parent,
                    prompt=prompt,
                    llm_response=llm_response,
                    artifacts=rejection_artifacts,
                )
            return None

        assert child_id is not None


        artifacts = evaluator.get_pending_artifacts(child_id)
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
            result.child_metrics,
            getattr(config, "database", None),
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


        template_key = (
            "structured_design_action_user"
            if design_action_mode
            else "structured_ast_user"
            if ast_structured_mode
            else "full_rewrite_user"
            if not config.diff_based_evolution
            else "diff_user"
        )


        result.child_program = Program(
            id=child_id,
            code=child_code,
            changes_description=child_changes_desc,
            language=config.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=result.child_metrics,
            iteration_found=iteration,
            metadata={
                **phenotype_metadata,
                "changes": changes_summary,
                "parent_metrics": parent.metrics,
                "generation_id": generation_plan.generation_id if generation_plan else None,
                "proposal_id": proposal_id,
                "trial_id": str(trial_id or ""),
                "proposal_causal_envelope_hash": (
                    reservation.proposal_causal_envelope_hash if reservation else None
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
            prompts=(
                {
                    template_key: {
                        "system": prompt["system"],
                        "user": prompt["user"],
                        "responses": [llm_response] if llm_response is not None else [],
                    }
                }
                if database.config.log_prompts
                else None
            ),
        )

        result.prompt = prompt
        result.llm_response = llm_response
        result.artifacts = artifacts
        result.iteration_time = time.time() - iteration_start
        result.iteration = iteration

        return result

    except Exception as e:
        logger.exception(f"Error in iteration {iteration}: {e}")
        if generation_plan is not None:
            return Result.failure(
                str(e),
                generation_plan=generation_plan,
                proposal_id=proposal_id,
                trial_id=trial_id,
                parent=failure_parent,
                prompt=failure_prompt,
                llm_response=failure_llm_response,
                artifacts=sealed_attempt_artifacts,
            )
        return None
