

import asyncio
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from engine.memory_lifecycle import MemoryExecutionContext, install_memory_execution_context
from engine.history_lifecycle import DuplicateEffectiveContractError

from outerloop.config import EvaluatorConfig
from outerloop.candidate_validation import CandidateValidationError
from outerloop.evaluation_result import EvaluationResult
from outerloop.database import ProgramDatabase
from outerloop.llm.ensemble import LLMEnsemble
from outerloop.utils.async_utils import TaskPool, run_in_executor
from outerloop.prompt.sampler import PromptSampler
from outerloop.utils.format_utils import format_metrics_safe

logger = logging.getLogger(__name__)


class Evaluator:


    def __init__(
        self,
        config: EvaluatorConfig,
        evaluation_file: str,
        llm_ensemble: Optional[LLMEnsemble] = None,
        prompt_sampler: Optional[PromptSampler] = None,
        database: Optional[ProgramDatabase] = None,
        suffix: Optional[str] = ".py",
    ):
        self.config = config
        self.evaluation_file = evaluation_file
        self.program_suffix = suffix
        self.llm_ensemble = llm_ensemble
        self.prompt_sampler = prompt_sampler
        self.database = database


        self.task_pool = TaskPool(max_concurrency=config.parallel_evaluations)


        self._load_evaluation_function()


        self._pending_artifacts: Dict[str, Dict[str, Union[str, bytes]]] = {}

        logger.info(f"Initialized evaluator with {evaluation_file}")

    def _load_evaluation_function(self) -> None:

        if not os.path.exists(self.evaluation_file):
            raise ValueError(f"Evaluation file {self.evaluation_file} not found")

        try:

            eval_dir = os.path.dirname(os.path.abspath(self.evaluation_file))
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
                logger.debug(f"Added {eval_dir} to Python path for local imports")

            spec = importlib.util.spec_from_file_location("evaluation_module", self.evaluation_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load spec from {self.evaluation_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules["evaluation_module"] = module
            spec.loader.exec_module(module)

            if not hasattr(module, "evaluate"):
                raise AttributeError(
                    f"Evaluation file {self.evaluation_file} does not contain an 'evaluate' function"
                )

            self._evaluation_module = module
            self.evaluate_function = module.evaluate
            self.candidate_validation_function = getattr(
                module, "validate_candidate", None
            )
            logger.info(f"Successfully loaded evaluation function from {self.evaluation_file}")


            self._validate_cascade_configuration(module)
        except Exception as e:
            logger.error(f"Error loading evaluation function: {str(e)}")
            raise

    async def validate_candidate_program(
        self,
        program_code: str,
        runtime_memory_context: Optional[MemoryExecutionContext] = None,
    ) -> Any:


        validator = getattr(self, "candidate_validation_function", None)
        if not callable(validator):
            return None
        with tempfile.NamedTemporaryFile(
            suffix=self.program_suffix, delete=False
        ) as temp_file:
            temp_file.write(program_code.encode("utf-8"))
            temp_file_path = temp_file.name
        try:
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._invoke_with_runtime_memory_context,
                    validator,
                    temp_file_path,
                    runtime_memory_context,
                ),
                timeout=self.config.timeout,
            )
        except CandidateValidationError:
            raise
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def _validate_cascade_configuration(self, module) -> None:

        if self.config.cascade_evaluation:

            has_stage1 = hasattr(module, "evaluate_stage1")
            has_stage2 = hasattr(module, "evaluate_stage2")
            has_stage3 = hasattr(module, "evaluate_stage3")

            if not has_stage1:
                logger.warning(
                    f"Configuration has 'cascade_evaluation: true' but evaluator "
                    f"'{self.evaluation_file}' does not define 'evaluate_stage1' function. "
                    f"This will fall back to direct evaluation, making the cascade setting useless. "
                    f"Consider setting 'cascade_evaluation: false' or implementing cascade functions."
                )
            elif not (has_stage2 or has_stage3):
                logger.warning(
                    f"Evaluator '{self.evaluation_file}' defines 'evaluate_stage1' but no additional "
                    f"cascade stages (evaluate_stage2, evaluate_stage3). Consider implementing "
                    f"multi-stage evaluation for better cascade benefits."
                )
            else:
                logger.debug(
                    f"Cascade evaluation properly configured with available stage functions"
                )

    async def evaluate_program(
        self,
        program_code: str,
        program_id: str = "",
        runtime_memory_context: Optional[MemoryExecutionContext] = None,
    ) -> Dict[str, float]:

        start_time = time.time()
        program_id_str = f" {program_id}" if program_id else ""


        artifacts_enabled = os.environ.get("ENABLE_ARTIFACTS", "true").lower() == "true"


        last_exception = None
        for attempt in range(self.config.max_retries + 1):

            with tempfile.NamedTemporaryFile(suffix=self.program_suffix, delete=False) as temp_file:
                temp_file.write(program_code.encode("utf-8"))
                temp_file_path = temp_file.name

            try:

                if self.config.cascade_evaluation:

                    result = await self._cascade_evaluate(
                        temp_file_path,
                        runtime_memory_context=runtime_memory_context,
                    )
                else:

                    result = await self._direct_evaluate(
                        temp_file_path,
                        runtime_memory_context=runtime_memory_context,
                    )


                eval_result = self._process_evaluation_result(result)


                if artifacts_enabled and program_id and eval_result.metrics.get("timeout") is True:
                    if program_id not in self._pending_artifacts:
                        self._pending_artifacts[program_id] = {}

                    self._pending_artifacts[program_id].update(
                        {
                            "timeout": True,
                            "timeout_duration": self.config.timeout,
                            "failure_stage": "evaluation",
                            "error_type": "timeout",
                        }
                    )


                llm_eval_result = None
                if self.config.use_llm_feedback and self.llm_ensemble:
                    llm_result = await self._llm_evaluate(program_code, program_id=program_id)
                    llm_eval_result = self._process_evaluation_result(llm_result)


                    llm_scores = []
                    for name, value in llm_eval_result.metrics.items():
                        weighted_value = value * self.config.llm_feedback_weight
                        eval_result.metrics[f"llm_{name}"] = weighted_value
                        llm_scores.append(value)


                    if llm_scores:
                        llm_average = sum(llm_scores) / len(llm_scores)
                        eval_result.metrics["llm_average"] = (
                            llm_average * self.config.llm_feedback_weight
                        )


                        if "combined_score" in eval_result.metrics:

                            accuracy = eval_result.metrics["combined_score"]

                            eval_result.metrics["combined_score"] = (
                                accuracy * 0.7 + llm_average * 0.3
                            )


                if (
                    artifacts_enabled
                    and (
                        eval_result.has_artifacts()
                        or (llm_eval_result and llm_eval_result.has_artifacts())
                    )
                    and program_id
                ):
                    if program_id not in self._pending_artifacts:
                        self._pending_artifacts[program_id] = {}


                    if eval_result.has_artifacts():
                        self._pending_artifacts[program_id].update(eval_result.artifacts)
                        logger.debug(
                            f"Program{program_id_str} returned artifacts: "
                            f"{eval_result.artifacts}"
                        )

                    if llm_eval_result and llm_eval_result.has_artifacts():
                        self._pending_artifacts[program_id].update(llm_eval_result.artifacts)
                        logger.debug(
                            f"Program{program_id_str} returned LLM artifacts: "
                            f"{llm_eval_result.artifacts}"
                        )

                elapsed = time.time() - start_time
                logger.info(
                    f"Evaluated program{program_id_str} in {elapsed:.2f}s: "
                    f"{format_metrics_safe(eval_result.metrics)}"
                )


                return eval_result.metrics

            except DuplicateEffectiveContractError:


                if program_id:
                    self._pending_artifacts.pop(program_id, None)
                raise

            except asyncio.TimeoutError:

                logger.warning(f"Evaluation timed out after {self.config.timeout}s")


                if artifacts_enabled and program_id:
                    self._pending_artifacts[program_id] = {
                        "timeout": True,
                        "timeout_duration": self.config.timeout,
                        "failure_stage": "evaluation",
                        "error_type": "timeout",
                    }

                return {"error": 0.0, "timeout": True}

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Evaluation attempt {attempt + 1}/{self.config.max_retries + 1} failed for program{program_id_str}: {str(e)}"
                )
                traceback.print_exc()


                if artifacts_enabled and program_id:
                    self._pending_artifacts[program_id] = {
                        "stderr": str(e),
                        "traceback": traceback.format_exc(),
                        "failure_stage": "evaluation",
                        "attempt": attempt + 1,
                    }


                if attempt < self.config.max_retries:
                    await asyncio.sleep(1.0)

            finally:

                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)


        logger.error(
            f"All evaluation attempts failed for program{program_id_str}. Last error: {str(last_exception)}"
        )
        return {"error": 0.0}

    def _process_evaluation_result(self, result: Any) -> EvaluationResult:

        if isinstance(result, dict):

            return EvaluationResult.from_dict(result)
        elif isinstance(result, EvaluationResult):

            return result
        else:

            logger.warning(f"Unexpected evaluation result type: {type(result)}")
            return EvaluationResult(metrics={"error": 0.0})

    def get_pending_artifacts(self, program_id: str) -> Optional[Dict[str, Union[str, bytes]]]:

        return self._pending_artifacts.pop(program_id, None)

    @staticmethod
    def _invoke_with_runtime_memory_context(
        function: Callable[[str], Any],
        program_path: str,
        runtime_memory_context: Optional[MemoryExecutionContext],
    ) -> Any:


        if runtime_memory_context is None:
            return function(program_path)
        with install_memory_execution_context(runtime_memory_context):
            return function(program_path)

    async def _direct_evaluate(
        self,
        program_path: str,
        runtime_memory_context: Optional[MemoryExecutionContext] = None,
    ) -> Union[Dict[str, float], EvaluationResult]:


        async def run_evaluation():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._invoke_with_runtime_memory_context,
                self.evaluate_function,
                program_path,
                runtime_memory_context,
            )


        result = await asyncio.wait_for(run_evaluation(), timeout=self.config.timeout)


        return result

    async def _cascade_evaluate(
        self,
        program_path: str,
        runtime_memory_context: Optional[MemoryExecutionContext] = None,
    ) -> Union[Dict[str, float], EvaluationResult]:


        try:
            module = self._evaluation_module


            if not hasattr(module, "evaluate_stage1"):
                return await self._direct_evaluate(
                    program_path,
                    runtime_memory_context=runtime_memory_context,
                )


            try:

                async def run_stage1():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None,
                        self._invoke_with_runtime_memory_context,
                        module.evaluate_stage1,
                        program_path,
                        runtime_memory_context,
                    )

                stage1_result = await asyncio.wait_for(run_stage1(), timeout=self.config.timeout)
                stage1_eval_result = self._process_evaluation_result(stage1_result)
            except DuplicateEffectiveContractError:
                raise
            except asyncio.TimeoutError:
                logger.warning(f"Stage 1 evaluation timed out after {self.config.timeout}s")
                return EvaluationResult(
                    metrics={"stage1_passed": 0.0, "error": 0.0, "timeout": True},
                    artifacts={
                        "failure_stage": "stage1",
                        "timeout": True,
                    },
                )
            except Exception as e:
                logger.error(f"Error in stage 1 evaluation: {str(e)}")

                error_context = self._create_cascade_error_context("stage1", e)
                return EvaluationResult(
                    metrics={"stage1_passed": 0.0, "error": 0.0},
                    artifacts={
                        "stderr": str(e),
                        "traceback": traceback.format_exc(),
                        **error_context,
                    },
                )


            if not self._passes_threshold(
                stage1_eval_result.metrics, self.config.cascade_thresholds[0]
            ):
                return stage1_eval_result


            if not hasattr(module, "evaluate_stage2"):
                return stage1_eval_result


            try:

                async def run_stage2():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None,
                        self._invoke_with_runtime_memory_context,
                        module.evaluate_stage2,
                        program_path,
                        runtime_memory_context,
                    )

                stage2_result = await asyncio.wait_for(run_stage2(), timeout=self.config.timeout)
                stage2_eval_result = self._process_evaluation_result(stage2_result)
            except DuplicateEffectiveContractError:
                raise
            except asyncio.TimeoutError:
                logger.warning(f"Stage 2 evaluation timed out after {self.config.timeout}s")

                stage1_eval_result.artifacts.update(
                    {
                        "stage2_timeout": True,
                        "failure_stage": "stage2",
                    }
                )
                stage1_eval_result.metrics["stage2_passed"] = 0.0
                stage1_eval_result.metrics["timeout"] = True
                return stage1_eval_result
            except Exception as e:
                logger.error(f"Error in stage 2 evaluation: {str(e)}")

                stage1_eval_result.artifacts.update(
                    {
                        "stage2_stderr": str(e),
                        "stage2_traceback": traceback.format_exc(),
                        "failure_stage": "stage2",
                    }
                )
                stage1_eval_result.metrics["stage2_passed"] = 0.0
                return stage1_eval_result


            merged_metrics = {}

            for name, value in stage1_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_metrics[name] = float(value)

            for name, value in stage2_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_metrics[name] = float(value)


            merged_artifacts = {}
            merged_artifacts.update(stage1_eval_result.artifacts)
            merged_artifacts.update(stage2_eval_result.artifacts)

            merged_result = EvaluationResult(metrics=merged_metrics, artifacts=merged_artifacts)


            if len(self.config.cascade_thresholds) < 2 or not self._passes_threshold(
                merged_result.metrics, self.config.cascade_thresholds[1]
            ):
                return merged_result


            if not hasattr(module, "evaluate_stage3"):
                return merged_result


            try:

                async def run_stage3():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None,
                        self._invoke_with_runtime_memory_context,
                        module.evaluate_stage3,
                        program_path,
                        runtime_memory_context,
                    )

                stage3_result = await asyncio.wait_for(run_stage3(), timeout=self.config.timeout)
                stage3_eval_result = self._process_evaluation_result(stage3_result)
            except DuplicateEffectiveContractError:
                raise
            except asyncio.TimeoutError:
                logger.warning(f"Stage 3 evaluation timed out after {self.config.timeout}s")

                merged_result.artifacts.update(
                    {
                        "stage3_timeout": True,
                        "failure_stage": "stage3",
                    }
                )
                merged_result.metrics["stage3_passed"] = 0.0
                merged_result.metrics["timeout"] = True
                return merged_result
            except Exception as e:
                logger.error(f"Error in stage 3 evaluation: {str(e)}")

                merged_result.artifacts.update(
                    {
                        "stage3_stderr": str(e),
                        "stage3_traceback": traceback.format_exc(),
                        "failure_stage": "stage3",
                    }
                )
                merged_result.metrics["stage3_passed"] = 0.0
                return merged_result


            for name, value in stage3_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_result.metrics[name] = float(value)

            merged_result.artifacts.update(stage3_eval_result.artifacts)

            return merged_result

        except DuplicateEffectiveContractError:
            raise
        except Exception as e:
            logger.error(f"Error in cascade evaluation: {str(e)}")

            error_context = self._create_cascade_error_context("cascade_setup", e)
            return EvaluationResult(
                metrics={"stage1_passed": 0.0, "error": 0.0},
                artifacts={
                    "stderr": str(e),
                    "traceback": traceback.format_exc(),
                    **error_context,
                },
            )

    async def _llm_evaluate(self, program_code: str, program_id: str = "") -> Dict[str, float]:

        if not self.llm_ensemble:
            return {}

        try:

            feature_dimensions = self.database.config.feature_dimensions if self.database else []
            prompt = self.prompt_sampler.build_prompt(
                current_program=program_code,
                template_key="evaluation",
                feature_dimensions=feature_dimensions,
            )


            responses = await self.llm_ensemble.generate_all_with_context(
                prompt["system"], [{"role": "user", "content": prompt["user"]}]
            )


            if self.database and program_id:
                self.database.log_prompt(
                    program_id=program_id,
                    template_key="evaluation",
                    prompt=prompt,
                    responses=responses,
                )


            try:

                json_pattern = r"```json\n(.*?)\n```"
                import re

                artifacts = {}
                avg_metrics = {}
                for i, response in enumerate(responses):
                    json_match = re.search(json_pattern, response, re.DOTALL)

                    if json_match:
                        json_str = json_match.group(1)
                    else:

                        json_str = response

                        start_idx = json_str.find("{")
                        end_idx = json_str.rfind("}") + 1
                        if start_idx >= 0 and end_idx > start_idx:
                            json_str = json_str[start_idx:end_idx]


                    result = json.loads(json_str)


                    metrics = {}
                    for key, value in result.items():
                        if not isinstance(value, (int, float)):
                            artifacts[key] = value
                        else:
                            metrics[key] = float(value)


                    weight = self.llm_ensemble.weights[i] if self.llm_ensemble.weights else 1.0


                    for name, value in metrics.items():
                        if name in avg_metrics:
                            avg_metrics[name] += value * weight
                        else:
                            avg_metrics[name] = value * weight

                return EvaluationResult(
                    metrics=avg_metrics,
                    artifacts=artifacts,
                )

            except Exception as e:
                logger.warning(f"Error parsing LLM response: {str(e)}")
                return {}

        except Exception as e:
            logger.error(f"Error in LLM evaluation: {str(e)}")
            traceback.print_exc()
            return {}

    def _create_cascade_error_context(self, stage: str, error: Exception) -> dict:

        import time

        return {
            "failure_stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            "cascade_config": self.config.cascade_evaluation,
            "cascade_thresholds": getattr(self.config, "cascade_thresholds", []),
            "timeout_config": self.config.timeout,
            "evaluation_file": self.evaluation_file,
        }

    def _passes_threshold(self, metrics: Dict[str, float], threshold: float) -> bool:

        if not metrics:
            return False


        if "combined_score" in metrics:
            score = metrics.get("combined_score")
            if isinstance(score, (int, float)):
                return float(score) >= threshold


        valid_metrics = []
        for name, value in metrics.items():

            if name != "error" and isinstance(value, (int, float)):
                try:
                    valid_metrics.append(float(value))
                except (TypeError, ValueError):
                    logger.warning(f"Skipping non-numeric metric: {name}={value}")
                    continue

        if not valid_metrics:
            return False

        avg_score = sum(valid_metrics) / len(valid_metrics)
        return avg_score >= threshold

    async def evaluate_multiple(
        self,
        programs: List[Tuple[str, str]],
    ) -> List[Dict[str, float]]:

        tasks = [
            self.task_pool.create_task(self.evaluate_program, program_code, program_id)
            for program_code, program_id in programs
        ]

        return await asyncio.gather(*tasks)
