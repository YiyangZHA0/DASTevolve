

import asyncio
import tempfile
import os
import uuid
import inspect
from typing import Union, Callable, Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from outerloop.controller import OuterLoop
from outerloop.config import Config, load_config, LLMModelConfig
from outerloop.database import Program


@dataclass
class EvolutionResult:


    best_program: Optional[Program]
    best_score: float
    best_code: str
    metrics: Dict[str, Any]
    output_dir: Optional[str]

    def __repr__(self):
        return f"EvolutionResult(best_score={self.best_score:.4f})"


def run_evolution(
    initial_program: Union[str, Path, List[str]],
    evaluator: Union[str, Path, Callable],
    config: Union[str, Path, Config, None] = None,
    iterations: Optional[int] = None,
    output_dir: Optional[str] = None,
    cleanup: bool = True,
) -> EvolutionResult:

    return asyncio.run(
        _run_evolution_async(initial_program, evaluator, config, iterations, output_dir, cleanup)
    )


async def _run_evolution_async(
    initial_program: Union[str, Path, List[str]],
    evaluator: Union[str, Path, Callable],
    config: Union[str, Path, Config, None],
    iterations: Optional[int],
    output_dir: Optional[str],
    cleanup: bool,
) -> EvolutionResult:


    temp_dir = None
    temp_files = []

    try:

        if config is None:
            config_obj = Config()
        elif isinstance(config, Config):
            config_obj = config
        else:
            config_obj = load_config(str(config))


        if not config_obj.llm.models:
            raise ValueError(
                "No LLM models configured. Please provide a config with LLM models, or set up "
                "your configuration with models. For example:\n\n"
                "from outerloop.config import Config, LLMModelConfig\n"
                "config = Config()\n"
                "config.llm.models = [LLMModelConfig(name='gpt-4', api_key='your-key')]\n"
                "result = run_evolution(program, evaluator, config=config)"
            )


        if output_dir is None and cleanup:
            temp_dir = tempfile.mkdtemp(prefix="outerloop_")
            actual_output_dir = temp_dir
        else:
            actual_output_dir = output_dir or "outerloop_output"
            os.makedirs(actual_output_dir, exist_ok=True)


        program_path = _prepare_program(initial_program, temp_dir, temp_files)


        evaluator_path = _prepare_evaluator(evaluator, temp_dir, temp_files)


        if config_obj.evaluator.cascade_evaluation:
            with open(evaluator_path, "r") as f:
                eval_content = f.read()
            if "evaluate_stage1" not in eval_content:
                config_obj.evaluator.cascade_evaluation = False


        controller = OuterLoop(
            initial_program_path=program_path,
            evaluation_file=evaluator_path,
            config=config_obj,
            output_dir=actual_output_dir,
        )

        best_program = await controller.run(iterations=iterations)


        best_score = 0.0
        metrics = {}
        best_code = ""

        if best_program:
            best_code = best_program.code
            metrics = best_program.metrics or {}

            if "combined_score" in metrics:
                best_score = metrics["combined_score"]
            elif metrics:
                numeric_metrics = [v for v in metrics.values() if isinstance(v, (int, float))]
                if numeric_metrics:
                    best_score = sum(numeric_metrics) / len(numeric_metrics)

        return EvolutionResult(
            best_program=best_program,
            best_score=best_score,
            best_code=best_code,
            metrics=metrics,
            output_dir=actual_output_dir if not cleanup else None,
        )

    finally:

        if cleanup:
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            if temp_dir and os.path.exists(temp_dir):
                import shutil

                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass


def _prepare_program(
    initial_program: Union[str, Path, List[str]], temp_dir: Optional[str], temp_files: List[str]
) -> str:


    if isinstance(initial_program, (str, Path)):
        if os.path.exists(str(initial_program)):
            return str(initial_program)


    if isinstance(initial_program, list):
        code = "\n".join(initial_program)
    else:
        code = str(initial_program)


    if "EVOLVE-BLOCK-START" not in code:

        code = f"""# EVOLVE-BLOCK-START
{code}
# EVOLVE-BLOCK-END"""


    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    program_file = os.path.join(temp_dir, f"program_{uuid.uuid4().hex[:8]}.py")
    with open(program_file, "w") as f:
        f.write(code)
    temp_files.append(program_file)

    return program_file


def _prepare_evaluator(
    evaluator: Union[str, Path, Callable], temp_dir: Optional[str], temp_files: List[str]
) -> str:


    if isinstance(evaluator, (str, Path)):
        if os.path.exists(str(evaluator)):
            return str(evaluator)


    if callable(evaluator):


        try:
            func_source = inspect.getsource(evaluator)

            import textwrap

            func_source = textwrap.dedent(func_source)
            func_name = evaluator.__name__


            evaluator_code = f"""
import importlib.util
import sys
import os
import copy
import json
import time

{func_source}

def evaluate(program_path):
    return {func_name}(program_path)
"""
        except (OSError, TypeError):


            evaluator_id = f"_outerloop_evaluator_{uuid.uuid4().hex[:8]}"
            globals()[evaluator_id] = evaluator

            evaluator_code = f"""
import {__name__} as api_module

def evaluate(program_path):
    user_evaluator = getattr(api_module, '{evaluator_id}')
    return user_evaluator(program_path)
"""
    else:

        evaluator_code = str(evaluator)


        if "def evaluate" not in evaluator_code:
            raise ValueError("Evaluator code must contain an 'evaluate(program_path)' function")


    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    eval_file = os.path.join(temp_dir, f"evaluator_{uuid.uuid4().hex[:8]}.py")
    with open(eval_file, "w") as f:
        f.write(evaluator_code)
    temp_files.append(eval_file)

    return eval_file


def evolve_function(
    func: Callable, test_cases: List[Tuple[Any, Any]], iterations: int = 100, **kwargs
) -> EvolutionResult:


    func_source = inspect.getsource(func)
    func_name = func.__name__


    if "EVOLVE-BLOCK-START" not in func_source:

        lines = func_source.split("\n")
        func_def_line = next(i for i, line in enumerate(lines) if line.strip().startswith("def "))


        indent = len(lines[func_def_line]) - len(lines[func_def_line].lstrip())
        func_end = len(lines)
        for i in range(func_def_line + 1, len(lines)):
            if lines[i].strip() and (len(lines[i]) - len(lines[i].lstrip())) <= indent:
                func_end = i
                break


        lines.insert(func_def_line + 1, " " * (indent + 4) + "# EVOLVE-BLOCK-START")
        lines.insert(func_end + 1, " " * (indent + 4) + "# EVOLVE-BLOCK-END")
        func_source = "\n".join(lines)


    evaluator_code = f"""
import importlib.util
import copy

FUNC_NAME = {func_name!r}
TEST_CASES = {test_cases!r}

def evaluate(program_path):
    spec = importlib.util.spec_from_file_location("evolved", program_path)
    if spec is None or spec.loader is None:
        return {{"combined_score": 0.0, "score": 0.0, "error": "Failed to load program"}}

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return {{"combined_score": 0.0, "score": 0.0, "error": f"Failed to execute program: {{str(e)}}"}}

    if not hasattr(module, FUNC_NAME):
        return {{"combined_score": 0.0, "score": 0.0, "error": f"Function '{{FUNC_NAME}}' not found"}}

    evolved_func = getattr(module, FUNC_NAME)
    correct = 0
    total = len(TEST_CASES)
    errors = []

    for input_val, expected in TEST_CASES:
        try:
            if isinstance(input_val, list):
                test_input = input_val.copy()
            else:
                test_input = input_val

            result = evolved_func(test_input)
            if result == expected:
                correct += 1
            else:
                errors.append(f"Input {{input_val}}: expected {{expected}}, got {{result}}")
        except Exception as e:
            errors.append(f"Input {{input_val}}: {{str(e)}}")

    score = correct / total if total > 0 else 0.0
    return {{
        "combined_score": score,
        "score": score,
        "test_pass_rate": score,
        "tests_passed": correct,
        "total_tests": total,
        "errors": errors[:3],
    }}
"""

    return run_evolution(
        initial_program=func_source, evaluator=evaluator_code, iterations=iterations, **kwargs
    )


def evolve_algorithm(
    algorithm_class: type, benchmark: Callable, iterations: int = 100, **kwargs
) -> EvolutionResult:


    class_source = inspect.getsource(algorithm_class)


    if "EVOLVE-BLOCK-START" not in class_source:
        lines = class_source.split("\n")

        class_def_line = next(
            i for i, line in enumerate(lines) if line.strip().startswith("class ")
        )


        indent = len(lines[class_def_line]) - len(lines[class_def_line].lstrip())
        lines.insert(class_def_line + 1, " " * (indent + 4) + "# EVOLVE-BLOCK-START")
        lines.append(" " * (indent + 4) + "# EVOLVE-BLOCK-END")
        class_source = "\n".join(lines)


    import textwrap

    class_name = algorithm_class.__name__
    benchmark_source = textwrap.dedent(inspect.getsource(benchmark))

    evaluator_code = f"""
import importlib.util

CLASS_NAME = {class_name!r}

{benchmark_source}

def evaluate(program_path):
    spec = importlib.util.spec_from_file_location("evolved", program_path)
    if spec is None or spec.loader is None:
        return {{"combined_score": 0.0, "score": 0.0, "error": "Failed to load program"}}

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return {{"combined_score": 0.0, "score": 0.0, "error": f"Failed to execute program: {{str(e)}}"}}

    if not hasattr(module, CLASS_NAME):
        return {{"combined_score": 0.0, "score": 0.0, "error": f"Class '{{CLASS_NAME}}' not found"}}

    AlgorithmClass = getattr(module, CLASS_NAME)

    try:
        instance = AlgorithmClass()
        metrics = {benchmark.__name__}(instance)
        if not isinstance(metrics, dict):
            metrics = {{"score": metrics}}
        if "combined_score" not in metrics:
            metrics["combined_score"] = metrics.get("score", 0.0)
        return metrics
    except Exception as e:
        return {{"combined_score": 0.0, "score": 0.0, "error": str(e)}}
"""

    return run_evolution(
        initial_program=class_source, evaluator=evaluator_code, iterations=iterations, **kwargs
    )


def evolve_code(
    initial_code: str, evaluator: Callable[[str], Dict[str, Any]], iterations: int = 100, **kwargs
) -> EvolutionResult:

    return run_evolution(
        initial_program=initial_code, evaluator=evaluator, iterations=iterations, **kwargs
    )
