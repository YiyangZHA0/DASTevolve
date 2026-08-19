

import asyncio
from copy import deepcopy
import hashlib
import json
import logging
import os
import shutil
import signal
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from outerloop.config import Config, load_config
from outerloop.continuation_seed import ContinuationSeed
from outerloop.adaptive_memory_runtime import (
    prepare_adaptive_memory_target,
    restore_adaptive_memory_checkpoint,
    save_adaptive_memory_checkpoint,
)
from outerloop.database import Program, ProgramDatabase
from outerloop.evaluator import Evaluator
from outerloop.evolution_trace import EvolutionTracer
from outerloop.generation_lifecycle import (
    GenerationCoordinator,
    GenerationObservation,
    generation_chunks,
    target_island_for_iteration,
)
from outerloop.history_admission import claim_code_for_evaluation
from outerloop.hierarchical_design import HierarchicalDesignRuntime
from outerloop.iteration import run_iteration_with_shared_db
from outerloop.llm.ensemble import LLMEnsemble
from outerloop.memory import OptimizerMemoryStore
from outerloop.phenotype_runtime import seal_effective_phenotype
from outerloop.process_parallel import (
    ProcessParallelController,
    logical_proposal_wave_size,
)
from outerloop.prompt.sampler import PromptSampler
from outerloop.utils.code_utils import extract_code_language
from outerloop.utils.format_utils import (
    format_directional_improvement_safe,
    format_metrics_safe,
)
from astevolve.runtime.edit_contract_lifecycle import (
    envelope_from_parent_artifacts,
    finalize_generated_contract_artifacts,
)
from engine.experiment_registry import ExperimentRegistry
from engine.memory_policy import MemoryPolicyConfig, MemoryPolicyError, MemoryScope
from engine.memory_lifecycle import MemoryExecutionContext, capture_memory_snapshot

logger = logging.getLogger(__name__)


EVOLUTION_TRACE_CHECKPOINT_VERSION = (
    "astevolve.evolution_trace_checkpoint.v1"
)
EVOLUTION_TRACE_CHECKPOINT_NAME = "evolution_trace.checkpoint.jsonl"
EVOLUTION_TRACE_CHECKPOINT_METADATA_NAME = (
    "evolution_trace.checkpoint.json"
)
V9_CHECKPOINT_MANIFEST_VERSION = "astevolve.outer_checkpoint.v9.v1"
V9_CHECKPOINT_MANIFEST_NAME = "checkpoint_state.v9.json"


class CheckpointStateError(RuntimeError):
    pass


def _checkpoint_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_v9_checkpoint_manifest(
    checkpoint_path: str | Path,
    *,
    iteration: int,
    required_files: Sequence[str],
) -> Dict[str, Any]:
    root = Path(checkpoint_path)
    names = sorted(set(map(str, required_files)))
    hashes: Dict[str, str] = {}
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name != name:
            raise CheckpointStateError(f"unsafe v9 checkpoint manifest file: {name!r}")
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise CheckpointStateError(
                f"v9 checkpoint required file is missing or unsafe: {name}"
            )
        hashes[name] = _checkpoint_file_sha256(path)
    payload = {
        "schema_version": V9_CHECKPOINT_MANIFEST_VERSION,
        "iteration": int(iteration),
        "required_files": names,
        "sha256": hashes,
    }
    _atomic_checkpoint_replace(
        root / V9_CHECKPOINT_MANIFEST_NAME,
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8"),
    )
    return payload


def _validate_v9_checkpoint_manifest(
    checkpoint_path: str | Path,
    *,
    expected_iteration: int | None,
    required_files: Sequence[str],
) -> Dict[str, Any]:
    root = Path(checkpoint_path)
    manifest_path = root / V9_CHECKPOINT_MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise CheckpointStateError("v9 checkpoint completeness manifest is missing")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointStateError("v9 checkpoint manifest is unreadable") from exc
    expected_fields = {"schema_version", "iteration", "required_files", "sha256"}
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise CheckpointStateError("v9 checkpoint manifest has invalid fields")
    if payload["schema_version"] != V9_CHECKPOINT_MANIFEST_VERSION:
        raise CheckpointStateError("v9 checkpoint manifest version is unsupported")
    iteration = payload["iteration"]
    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise CheckpointStateError("v9 checkpoint manifest iteration is invalid")
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise CheckpointStateError("v9 checkpoint manifest iteration disagrees with path")
    names = payload["required_files"]
    hashes = payload["sha256"]
    if (
        not isinstance(names, list)
        or not isinstance(hashes, Mapping)
        or set(map(str, names)) != set(map(str, hashes))
    ):
        raise CheckpointStateError("v9 checkpoint manifest file table is invalid")
    missing_required = sorted(set(map(str, required_files)) - set(map(str, names)))
    if missing_required:
        raise CheckpointStateError(
            "v9 checkpoint manifest omits required state: "
            + ", ".join(missing_required)
        )
    for name in names:
        relative = Path(str(name))
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name != name:
            raise CheckpointStateError("v9 checkpoint manifest contains unsafe path")
        path = root / relative
        expected_hash = hashes.get(name)
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or not path.is_file()
            or path.is_symlink()
            or _checkpoint_file_sha256(path) != expected_hash
        ):
            raise CheckpointStateError(
                f"v9 checkpoint state hash mismatch: {name}"
            )
    return deepcopy(dict(payload))


def _atomic_checkpoint_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(
            str(path.parent), os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _trace_checkpoint_document(
    *,
    trace_path: Path,
    payload: bytes,
    iteration: int,
    compressed: bool,
) -> dict[str, Any]:
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise CheckpointStateError(
            "trace checkpoint iteration must be a non-negative integer"
        )
    return {
        "schema_version": EVOLUTION_TRACE_CHECKPOINT_VERSION,
        "iteration": iteration,
        "trace_format": "jsonl",
        "compressed": bool(compressed),
        "trace_output_path": str(trace_path.resolve()),
        "byte_sha256": hashlib.sha256(payload).hexdigest(),
        "raw_size": len(payload),
    }


def _load_trace_checkpoint_document(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointStateError(
            f"evolution trace checkpoint metadata is unreadable: {path}"
        ) from error
    required = {
        "schema_version",
        "iteration",
        "trace_format",
        "compressed",
        "trace_output_path",
        "byte_sha256",
        "raw_size",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise CheckpointStateError(
            "evolution trace checkpoint metadata has an invalid closed schema"
        )
    if value.get("schema_version") != EVOLUTION_TRACE_CHECKPOINT_VERSION:
        raise CheckpointStateError(
            "evolution trace checkpoint metadata has an unsupported schema_version"
        )
    iteration = value.get("iteration")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise CheckpointStateError(
            "trace checkpoint iteration must be a non-negative integer"
        )
    if value.get("trace_format") != "jsonl" or not isinstance(
        value.get("compressed"), bool
    ):
        raise CheckpointStateError(
            "evolution trace checkpoint format metadata is invalid"
        )
    raw_size = value.get("raw_size")
    if not isinstance(raw_size, int) or isinstance(raw_size, bool) or raw_size < 0:
        raise CheckpointStateError(
            "evolution trace checkpoint size is invalid"
        )
    digest = value.get("byte_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise CheckpointStateError(
            "evolution trace checkpoint SHA256 is invalid"
        )
    if not isinstance(value.get("trace_output_path"), str) or not value[
        "trace_output_path"
    ]:
        raise CheckpointStateError(
            "evolution trace checkpoint output path is invalid"
        )
    return dict(value)


def _save_evolution_trace_checkpoint(
    tracer: Optional[EvolutionTracer],
    checkpoint_path: str | Path,
    *,
    iteration: int,
) -> Optional[dict[str, Any]]:
    if tracer is None or not bool(getattr(tracer, "enabled", False)):
        return None
    if str(getattr(tracer, "format", "")) != "jsonl":


        return None
    tracer.flush()
    if getattr(tracer, "buffer", None):
        raise CheckpointStateError(
            "evolution trace buffer did not flush before checkpoint"
        )
    trace_path = Path(tracer.output_path).expanduser().resolve()
    payload = trace_path.read_bytes() if trace_path.is_file() else b""
    document = _trace_checkpoint_document(
        trace_path=trace_path,
        payload=payload,
        iteration=iteration,
        compressed=bool(getattr(tracer, "compress", False)),
    )
    checkpoint_root = Path(checkpoint_path).expanduser().resolve()
    _atomic_checkpoint_replace(
        checkpoint_root / EVOLUTION_TRACE_CHECKPOINT_NAME,
        payload,
    )
    _atomic_checkpoint_replace(
        checkpoint_root / EVOLUTION_TRACE_CHECKPOINT_METADATA_NAME,
        (
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    _validated_evolution_trace_checkpoint(
        tracer,
        checkpoint_root,
        expected_iteration=iteration,
    )
    return document


def _validated_evolution_trace_checkpoint(
    tracer: EvolutionTracer,
    checkpoint_path: str | Path,
    *,
    expected_iteration: Optional[int],
) -> tuple[bytes, dict[str, Any]]:
    checkpoint_root = Path(checkpoint_path).expanduser().resolve()
    trace_snapshot = checkpoint_root / EVOLUTION_TRACE_CHECKPOINT_NAME
    trace_metadata = (
        checkpoint_root / EVOLUTION_TRACE_CHECKPOINT_METADATA_NAME
    )
    snapshot_exists = trace_snapshot.is_file()
    metadata_exists = trace_metadata.is_file()
    if snapshot_exists != metadata_exists or not snapshot_exists:
        raise CheckpointStateError(
            "evolution trace checkpoint is incomplete: trace/metadata mismatch"
        )
    document = _load_trace_checkpoint_document(trace_metadata)
    if (
        expected_iteration is not None
        and document["iteration"] != expected_iteration
    ):
        raise CheckpointStateError(
            "evolution trace checkpoint iteration does not match the requested checkpoint"
        )
    trace_path = Path(tracer.output_path).expanduser().resolve()
    if document["trace_output_path"] != str(trace_path):
        raise CheckpointStateError(
            "evolution trace checkpoint belongs to a different run output"
        )
    if document["compressed"] != bool(getattr(tracer, "compress", False)):
        raise CheckpointStateError(
            "evolution trace checkpoint compression mode changed"
        )
    payload = trace_snapshot.read_bytes()
    if (
        len(payload) != document["raw_size"]
        or hashlib.sha256(payload).hexdigest() != document["byte_sha256"]
    ):
        raise CheckpointStateError(
            "evolution trace checkpoint bytes do not match their recorded hash"
        )
    return payload, document


def _restore_evolution_trace_checkpoint(
    tracer: Optional[EvolutionTracer],
    checkpoint_path: str | Path,
    *,
    expected_iteration: Optional[int],
) -> Optional[dict[str, Any]]:
    if tracer is None or not bool(getattr(tracer, "enabled", False)):
        return None
    if str(getattr(tracer, "format", "")) != "jsonl":
        return None
    if getattr(tracer, "buffer", None):
        raise CheckpointStateError(
            "cannot restore an evolution trace while its buffer is non-empty"
        )
    payload, document = _validated_evolution_trace_checkpoint(
        tracer,
        checkpoint_path,
        expected_iteration=expected_iteration,
    )
    trace_path = Path(tracer.output_path).expanduser().resolve()
    _atomic_checkpoint_replace(trace_path, payload)
    if trace_path.read_bytes() != payload:
        raise CheckpointStateError(
            "evolution trace restore did not preserve checkpoint bytes"
        )
    return document


def _format_metrics(metrics: Dict[str, Any]) -> str:

    formatted_parts = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                formatted_parts.append(f"{name}={value:.4f}")
            except (ValueError, TypeError):
                formatted_parts.append(f"{name}={value}")
        else:
            formatted_parts.append(f"{name}={value}")
    return ", ".join(formatted_parts)


def _format_improvement(improvement: Dict[str, Any]) -> str:

    formatted_parts = []
    for name, diff in improvement.items():
        if isinstance(diff, (int, float)) and not isinstance(diff, bool):
            try:
                formatted_parts.append(f"{name}={diff:+.4f}")
            except (ValueError, TypeError):
                formatted_parts.append(f"{name}={diff}")
        else:
            formatted_parts.append(f"{name}={diff}")
    return ", ".join(formatted_parts)


def _infer_case_memory_path(initial_program_path: str) -> Optional[str]:
    case_dir = os.path.dirname(os.path.abspath(initial_program_path))
    candidate = os.path.join(case_dir, "memory.yaml")
    return candidate if os.path.exists(candidate) else None


def _initial_runtime_output_dir(output_dir: str, initial_program_id: str) -> str:


    return str(
        Path(output_dir).expanduser().resolve()
        / "generation_runtime"
        / "initial"
        / initial_program_id
    )


class OuterLoop:


    def __init__(
        self,
        initial_program_path: str,
        evaluation_file: str,
        config: Config,
        output_dir: Optional[str] = None,
        continuation_seed_program_json: Optional[str] = None,
        continuation_seed_program_id: str = "",
        continuation_seed_program_sha256: str = "",
    ):

        self.config = config


        self.output_dir = output_dir or os.path.join(
            os.path.dirname(initial_program_path), "outerloop_output"
        )
        os.makedirs(self.output_dir, exist_ok=True)


        if self.config.database.experiment_registry_enabled is None:
            self.config.database.experiment_registry_enabled = True
        if self.config.database.outer_effective_phenotype_enabled is None:
            self.config.database.outer_effective_phenotype_enabled = True
        if (
            self.config.database.experiment_registry_retry_failed
            and self.config.database.experiment_registry_replicate_policy == "reject"
        ):
            self.config.database.experiment_registry_replicate_policy = "retry_failed"
        self.experiment_registry = None
        if self.config.database.experiment_registry_enabled:
            registry_path = self.config.database.experiment_registry_path
            if not registry_path:
                registry_path = os.path.join(self.output_dir, "experiment_registry.sqlite")
            elif not os.path.isabs(registry_path):
                registry_path = os.path.join(self.output_dir, registry_path)
            registry_path = str(Path(registry_path).expanduser().resolve())
            self.config.database.experiment_registry_path = registry_path
            if not self.config.database.experiment_registry_scope:
                scope_material = "\0".join(
                    (
                        str(Path(initial_program_path).expanduser().resolve()),
                        str(Path(evaluation_file).expanduser().resolve()),
                    )
                ).encode("utf-8")
                self.config.database.experiment_registry_scope = (
                    "outerloop:" + hashlib.sha256(scope_material).hexdigest()
                )
            self.experiment_registry = ExperimentRegistry(
                registry_path,
                default_lease_seconds=(
                    self.config.database.experiment_registry_lease_seconds
                ),
            )


        if not getattr(self.config.database, "artifacts_base_path", None):
            self.config.database.artifacts_base_path = os.path.join(self.output_dir, "artifacts")


        self._setup_logging()


        self._setup_manual_mode_queue()


        if self.config.random_seed is not None:
            import random

            import numpy as np


            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)


            base_seed = str(self.config.random_seed).encode("utf-8")
            llm_seed = int(hashlib.md5(base_seed + b"llm").hexdigest()[:8], 16) % (2**31)


            self.config.llm.random_seed = llm_seed
            for model_cfg in self.config.llm.models:
                if not hasattr(model_cfg, "random_seed") or model_cfg.random_seed is None:
                    model_cfg.random_seed = llm_seed
            for model_cfg in self.config.llm.evaluator_models:
                if not hasattr(model_cfg, "random_seed") or model_cfg.random_seed is None:
                    model_cfg.random_seed = llm_seed

            logger.info(f"Set random seed to {self.config.random_seed} for reproducibility")
            logger.debug(f"Generated LLM seed: {llm_seed}")


        self.initial_program_path = initial_program_path
        self.case_memory_source_path = _infer_case_memory_path(initial_program_path)
        self.initial_program_code = self._load_initial_program()
        self.continuation_seed: ContinuationSeed | None = None
        if continuation_seed_program_json:
            self.continuation_seed = ContinuationSeed.load(
                continuation_seed_program_json,
                expected_program_id=continuation_seed_program_id,
                expected_file_sha256=continuation_seed_program_sha256,
            )


            self.initial_program_code = self.continuation_seed.code
            logger.info(
                "Using verified continuation seed %s (AST %s, sequence %s)",
                self.continuation_seed.source_program_id,
                self.continuation_seed.effective_ast_hash,
                self.continuation_seed.sequence_identity.sequence_bundle_hash,
            )
        self.memory_scope = MemoryScope(
            case_id=Path(initial_program_path).resolve().parent.name or "case",
            run_id=str(self.config.database.experiment_registry_scope or "outerloop-run"),
            lineage_id=(
                "initial-code-"
                + hashlib.sha256(self.initial_program_code.encode("utf-8")).hexdigest()
            ),
        )
        self.inner_memory_path = prepare_adaptive_memory_target(
            self.case_memory_source_path,
            self.output_dir,
            self.config.memory_policy,
        )
        if not self.config.language:
            self.config.language = extract_code_language(self.initial_program_code)


        self.file_extension = os.path.splitext(initial_program_path)[1]
        if not self.file_extension:

            self.file_extension = ".py"
        else:

            if not self.file_extension.startswith("."):
                self.file_extension = f".{self.file_extension}"


        if not hasattr(self.config, "file_suffix") or self.config.file_suffix == ".py":
            self.config.file_suffix = self.file_extension


        self.llm_ensemble = LLMEnsemble(self.config.llm.models)
        self.prompt_sampler = PromptSampler(self.config.prompt)
        if self.config.evaluator.use_llm_feedback:
            self.llm_evaluator_ensemble = LLMEnsemble(
                self.config.llm.evaluator_models
            )
            self.evaluator_prompt_sampler = PromptSampler(self.config.prompt)
            self.evaluator_prompt_sampler.set_templates("evaluator_system_message")
        else:


            self.llm_evaluator_ensemble = None
            self.evaluator_prompt_sampler = None


        if self.config.random_seed is not None:
            self.config.database.random_seed = self.config.random_seed

        self.config.database.novelty_llm = self.llm_ensemble
        self.database = ProgramDatabase(self.config.database)
        self.hierarchical_design = None
        if bool(getattr(self.config.hierarchical_design, "enabled", False)):
            self.hierarchical_design = HierarchicalDesignRuntime(
                self.config.hierarchical_design,
                self.config.database,
                self.output_dir,
            )
            self.database.attach_hierarchical_design(self.hierarchical_design)
        self.optimizer_memory = None
        if getattr(self.config.database, "optimizer_memory_enabled", True):
            optimizer_memory_path = (
                self.config.database.optimizer_memory_path
                or os.path.join(self.output_dir, "optimizer_memory.json")
            )
            if not os.path.isabs(optimizer_memory_path):
                optimizer_memory_path = os.path.join(self.output_dir, optimizer_memory_path)
            self.optimizer_memory = OptimizerMemoryStore(
                optimizer_memory_path,
                case_memory_path=self.inner_memory_path,
                scope=self.memory_scope,
                recent_limit=getattr(self.config.database, "optimizer_memory_recent_limit", 20),
                best_limit=getattr(self.config.database, "optimizer_memory_best_limit", 5),
            )
            self.database.attach_optimizer_memory(self.optimizer_memory)

        self.evaluator = Evaluator(
            self.config.evaluator,
            evaluation_file,
            self.llm_evaluator_ensemble,
            self.evaluator_prompt_sampler,
            database=self.database,
            suffix=Path(self.initial_program_path).suffix,
        )
        self.evaluation_file = evaluation_file

        logger.info(f"Initialized OuterLoop with {initial_program_path}")


        if self.config.evolution_trace.enabled:
            trace_output_path = self.config.evolution_trace.output_path
            if not trace_output_path:

                trace_output_path = os.path.join(
                    self.output_dir, f"evolution_trace.{self.config.evolution_trace.format}"
                )

            self.evolution_tracer = EvolutionTracer(
                output_path=trace_output_path,
                format=self.config.evolution_trace.format,
                include_code=self.config.evolution_trace.include_code,
                include_prompts=self.config.evolution_trace.include_prompts,
                enabled=True,
                buffer_size=self.config.evolution_trace.buffer_size,
                compress=self.config.evolution_trace.compress,
            )
            logger.info(f"Evolution tracing enabled: {trace_output_path}")
        else:
            self.evolution_tracer = None


        self.parallel_controller = None
        self._shutdown_requested = False

    async def _prepare_hierarchical_strategy(self, iteration: int) -> None:


        runtime = getattr(self, "hierarchical_design", None)
        if runtime is None:
            return
        force = False
        if bool(
            getattr(
                self.config.hierarchical_design,
                "strategy_refresh_on_stagnation",
                True,
            )
        ):
            signal = (
                self.database.get_scheduler_signal()
                if hasattr(self.database, "get_scheduler_signal")
                else {}
            )
            trend = signal.get("score_trend", {}) if isinstance(signal, dict) else {}
            stagnated = bool(
                signal.get("phase") == "escape_stagnation"
                or (isinstance(trend, dict) and trend.get("stagnated"))
            )
            force = runtime.stagnation_refresh_required(stagnated)
        snapshot = await runtime.refresh(
            self.database,
            self.llm_ensemble,
            iteration=int(iteration),
            force=force,
        )
        logger.info(
            "Hierarchical strategy %s ready for iteration %s (%s)",
            snapshot.get("strategy_epoch_id"),
            iteration,
            snapshot.get("strategy_source"),
        )

    def _handle_shutdown_signal(self, signum, frame) -> None:


        del frame
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        already_requested = bool(getattr(self, "_shutdown_requested", False))
        self._shutdown_requested = True

        controller = getattr(self, "parallel_controller", None)
        if controller is not None and not already_requested:
            controller.request_shutdown()
        elif controller is None:
            logger.info(
                "No active parallel controller; serial evolution will stop "
                "after the current work item"
            )

        def force_exit_handler(force_signum, force_frame):
            del force_signum, force_frame
            logger.info("Force exit requested - terminating immediately")
            import sys

            sys.exit(0)


        signal.signal(signal.SIGINT, force_exit_handler)
        signal.signal(signal.SIGTERM, force_exit_handler)

    def _setup_logging(self) -> None:

        log_dir = self.config.log_dir or os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)


        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level))


        log_file = os.path.join(log_dir, f"outerloop_{time.strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(file_handler)


        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(console_handler)

        logger.info(f"Logging to {log_file}")

    def _setup_manual_mode_queue(self) -> None:

        if not bool(getattr(self.config.llm, "manual_mode", False)):
            return

        qdir = (Path(self.output_dir).expanduser().resolve() / "manual_tasks_queue")


        if qdir.exists():
            shutil.rmtree(qdir)
        qdir.mkdir(parents=True, exist_ok=True)


        self.config.llm._manual_queue_dir = str(qdir)
        for model_cfg in self.config.llm.models:
            model_cfg._manual_queue_dir = str(qdir)
        for model_cfg in self.config.llm.evaluator_models:
            model_cfg._manual_queue_dir = str(qdir)

        logger.info(f"Manual mode enabled. Queue dir: {qdir}")

    def _load_initial_program(self) -> str:

        with open(self.initial_program_path, "r") as f:
            return f.read()

    async def run(
        self,
        iterations: Optional[int] = None,
        target_score: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Optional[Program]:

        max_iterations = self.config.max_iterations if iterations is None else iterations


        start_iteration = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            start_iteration = self.database.last_iteration + 1
            logger.info(f"Resuming from checkpoint at iteration {start_iteration}")
        else:
            start_iteration = self.database.last_iteration


        should_add_initial = (
            start_iteration == 0
            and len(self.database.programs) == 0
            and not any(
                p.code == self.initial_program_code for p in self.database.programs.values()
            )
        )

        if should_add_initial:
            logger.info("Adding initial program to database")
            initial_program_id = str(uuid.uuid4())

            initial_admission = claim_code_for_evaluation(
                self.initial_program_code,
                self.config.database,
                owner_token=f"outer:initial:{initial_program_id}",
            )


            try:
                initial_context = MemoryExecutionContext(
                        generation_id="initial",
                        proposal_id="initial",
                        scope_id="initial/initial",
                        logical_time="initial",
                        commit_mode="deferred",
                        snapshot=(
                            capture_memory_snapshot(self.inner_memory_path)
                            if self.inner_memory_path
                            else None
                        ),
                        target_path=str(self.inner_memory_path or ""),
                        memory_scope=self.memory_scope,
                        memory_policy=self.config.memory_policy,
                        output_dir=_initial_runtime_output_dir(
                            self.output_dir,
                            initial_program_id,
                        ),
                        history_registry_path=(
                            self.config.database.experiment_registry_path
                            if self.config.database.experiment_registry_enabled
                            else ""
                        ),
                        history_scope=str(
                            self.config.database.experiment_registry_scope or ""
                        ),
                        history_owner_token=f"inner:initial:{initial_program_id}",
                        history_lease_seconds=(
                            self.config.database.experiment_registry_lease_seconds
                        ),
                        history_replicate_policy=(
                            self.config.database.experiment_registry_replicate_policy
                        ),
                    )
                if self.continuation_seed is not None:
                    initial_context = self.continuation_seed.bind_context(initial_context)
                initial_metrics = await self.evaluator.evaluate_program(
                    self.initial_program_code,
                    initial_program_id,
                    runtime_memory_context=initial_context,
                )
            except Exception as exc:
                if initial_admission is not None:
                    initial_admission.fail(exc)
                raise
            if initial_admission is not None:
                initial_admission.complete()

            initial_artifacts = self.evaluator.get_pending_artifacts(initial_program_id)
            if bool(initial_metrics.get("timeout")):
                duration = (initial_artifacts or {}).get(
                    "timeout_duration", self.config.evaluator.timeout
                )
                raise RuntimeError(
                    f"initial evaluation timed out after {duration}s; "
                    "runtime evidence was not sealed"
                )
            if bool(initial_metrics.get("error")) or (
                isinstance(initial_artifacts, dict)
                and initial_artifacts.get("failure_stage") == "evaluation"
            ):
                detail = (initial_artifacts or {}).get("stderr") or (
                    initial_artifacts or {}
                ).get("error_type") or "unknown evaluation failure"
                raise RuntimeError(
                    f"initial evaluation failed before runtime evidence sealing: {detail}"
                )
            initial_artifacts = finalize_generated_contract_artifacts(
                initial_artifacts,
                parent_program_id=initial_program_id,
                generated_generation_id="initial",
                generated_proposal_id="initial",
            )
            phenotype_metadata = seal_effective_phenotype(
                self.initial_program_code,
                initial_artifacts or {},
                initial_metrics,
                self.config.database,
            )
            if phenotype_metadata:
                initial_artifacts = {
                    **(initial_artifacts or {}),
                    **phenotype_metadata,
                }
            if initial_admission is not None:
                initial_artifacts = {
                    **(initial_artifacts or {}),
                    "code_admission": initial_admission.to_artifact(),
                }
            continuation_metadata: Dict[str, Any] = {}
            if self.continuation_seed is not None:
                continuation_metadata = {
                    "continuation_seed": self.continuation_seed.to_artifact()
                }
                initial_artifacts = {
                    **(initial_artifacts or {}),
                    **continuation_metadata,
                }
            initial_program = Program(
                id=initial_program_id,
                code=self.initial_program_code,
                changes_description=self.config.prompt.initial_changes_description,
                language=self.config.language,
                metrics=initial_metrics,
                iteration_found=start_iteration,
                metadata={**dict(phenotype_metadata), **continuation_metadata},
            )

            self.database.add(initial_program)
            if initial_artifacts:
                self.database.store_artifacts(initial_program_id, initial_artifacts)
            self.database.record_optimizer_memory(
                initial_program,
                parent=None,
                artifacts=initial_artifacts,
                iteration=start_iteration,
            )


            if "combined_score" not in initial_metrics:

                numeric_metrics = [
                    v
                    for v in initial_metrics.values()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                ]
                if numeric_metrics:
                    avg_score = sum(numeric_metrics) / len(numeric_metrics)
                    logger.warning(
                        f"⚠️  No 'combined_score' metric found in evaluation results. "
                        f"Using average of all numeric metrics ({avg_score:.4f}) for evolution guidance. "
                        f"For better evolution results, please modify your evaluator to return a 'combined_score' "
                        f"metric that properly weights different aspects of program performance."
                    )
        else:
            logger.info(
                f"Skipping initial program addition (resuming from iteration {start_iteration} "
                f"with {len(self.database.programs)} existing programs)"
            )

        if max_iterations <= 0:
            logger.info("No evolutionary iterations requested; returning initial evaluation result")
            best_program = None
            if self.database.best_program_id:
                best_program = self.database.get(self.database.best_program_id)
            self._save_registry_metrics()
            return best_program or self.database.get_best_program()


        try:
            self.parallel_controller = ProcessParallelController(
                self.config,
                self.evaluation_file,
                self.database,
                self.evolution_tracer,
                file_suffix=self.config.file_suffix,
                inner_memory_path=self.inner_memory_path,
                memory_scope=self.memory_scope,
            )


            signal.signal(signal.SIGINT, self._handle_shutdown_signal)
            signal.signal(signal.SIGTERM, self._handle_shutdown_signal)

            self.parallel_controller.start()


            evolution_start = start_iteration
            evolution_iterations = max_iterations


            if should_add_initial and start_iteration == 0:
                evolution_start = 1


            serial_mode = os.environ.get("ASTEVOLVE_OPENEVOLVE_SERIAL", "").lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
            if serial_mode:
                logger.info(
                    "Using serial in-process evolution because ASTEVOLVE_OPENEVOLVE_SERIAL is set"
                )
                self.parallel_controller.stop()
                self.parallel_controller = None
                await self._run_evolution_serial_with_checkpoints(
                    evolution_start, evolution_iterations, target_score
                )
            else:

                await self._run_evolution_with_checkpoints(
                    evolution_start, evolution_iterations, target_score
                )

        finally:

            if self.parallel_controller:
                self.parallel_controller.stop()
                self.parallel_controller = None


            if self.evolution_tracer:
                self.evolution_tracer.close()
                logger.info("Evolution tracer closed")


        best_program = None
        if self.database._v9_population_policy_enabled():
            recommended_id = self.database.get_global_summary(top_n=12).get(
                "recommended_program_id"
            )
            if recommended_id:
                best_program = self.database.get(recommended_id)
                logger.info("Using v9 global robust recommendation: %s", recommended_id)
        if best_program is None and self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
            logger.info(f"Using tracked best program: {self.database.best_program_id}")

        if best_program is None:
            best_program = self.database.get_best_program()
            logger.info("Using calculated best program (tracked program not found)")

        if best_program:
            if (
                hasattr(self, "parallel_controller")
                and self.parallel_controller
                and self.parallel_controller.early_stopping_triggered
            ):
                logger.info(
                    f"🛑 Evolution complete via early stopping. Best program has metrics: "
                    f"{format_metrics_safe(best_program.metrics)}"
                )
            else:
                logger.info(
                    f"Evolution complete. Best program has metrics: "
                    f"{format_metrics_safe(best_program.metrics)}"
                )
            self._save_best_program(best_program)
            self._save_registry_metrics()
            return best_program
        else:
            logger.warning("No valid programs found during evolution")
            self._save_registry_metrics()
            return None

    def _save_registry_metrics(self) -> None:
        if self.experiment_registry is None:
            return
        metrics = self.experiment_registry.metrics(
            scope=self.config.database.experiment_registry_scope
        )
        target = os.path.join(self.output_dir, "history_registry_metrics.json")
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(metrics.to_dict(), handle, indent=2, sort_keys=True)

    def _log_iteration(
        self,
        iteration: int,
        parent: Program,
        child: Program,
        elapsed_time: float,
    ) -> None:


        improvement_str = format_directional_improvement_safe(parent.metrics, child.metrics)

        logger.info(
            f"Iteration {iteration+1}: Child {child.id} from parent {parent.id} "
            f"in {elapsed_time:.2f}s. Metrics: "
            f"{format_metrics_safe(child.metrics)} "
            f"(Δ: {improvement_str})"
        )

    def _save_checkpoint(self, iteration: int) -> None:

        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)


        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
        os.makedirs(checkpoint_path, exist_ok=True)


        _save_evolution_trace_checkpoint(
            getattr(self, "evolution_tracer", None),
            checkpoint_path,
            iteration=iteration,
        )


        memory_policy = getattr(self.config, "memory_policy", None)
        if memory_policy is not None:
            if not isinstance(memory_policy, MemoryPolicyConfig):
                raise TypeError("controller memory_policy must be MemoryPolicyConfig")
            save_adaptive_memory_checkpoint(
                getattr(self, "case_memory_source_path", None),
                getattr(self, "inner_memory_path", None),
                checkpoint_path,
                memory_policy,
                iteration=iteration,
            )


        self.database.save(checkpoint_path, iteration)
        if hasattr(self.database, "get_global_summary"):
            global_summary = self.database.get_global_summary(top_n=12)
            global_summary_path = os.path.join(
                checkpoint_path, "global_island_summary.json"
            )
            _atomic_checkpoint_replace(
                Path(global_summary_path),
                (json.dumps(global_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"),
            )
            pareto_archive_path = os.path.join(checkpoint_path, "pareto_archive.json")
            archive_payload = {
                "schema_version": "astevolve.pareto_archive.v2",
                "iteration": iteration,
                "recommended_program_id": global_summary.get("recommended_program_id"),
                "global_pareto_front": global_summary.get("global_pareto_front", []),
                "role_portfolios": global_summary.get("role_portfolios", []),
                "independent_parent_pools": global_summary.get("independent_parent_pools", []),
                "migration_receipts": global_summary.get("migration_receipts", []),
                "map_elites_diagnostics": global_summary.get("map_elites_diagnostics", {}),
                "selection_policy": global_summary.get("selection_policy"),
            }
            _atomic_checkpoint_replace(
                Path(pareto_archive_path),
                (json.dumps(archive_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"),
            )
        if self.optimizer_memory is not None:
            self.optimizer_memory.save(os.path.join(checkpoint_path, "optimizer_memory.json"))
        hierarchical_design = getattr(self, "hierarchical_design", None)
        if hierarchical_design is not None:
            hierarchical_design.save(
                os.path.join(checkpoint_path, "hierarchical_design.json")
            )
            hierarchical_design.ledger.save(
                os.path.join(checkpoint_path, "hypothesis_ledger.json")
            )
            hierarchical_design.reasoning_ledger.save(
                os.path.join(checkpoint_path, "reasoning_audit.json")
            )
        if self.experiment_registry is not None:
            self.experiment_registry.backup_to(
                os.path.join(checkpoint_path, "experiment_registry.sqlite"),
                overwrite=True,
            )
            self._save_registry_metrics()


        best_program = None
        v9_enabled = bool(
            callable(getattr(self.database, "_v9_population_policy_enabled", None))
            and self.database._v9_population_policy_enabled()
        )
        if v9_enabled:
            recommended_id = self.database.get_global_summary(top_n=12).get(
                "recommended_program_id"
            )
            if recommended_id:
                best_program = self.database.get(recommended_id)
        if best_program is None and self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
        if best_program is None:
            best_program = self.database.get_best_program()

        if best_program:

            best_program_path = os.path.join(checkpoint_path, f"best_program{self.file_extension}")
            with open(best_program_path, "w") as f:
                f.write(best_program.code)


            best_program_info_path = os.path.join(checkpoint_path, "best_program_info.json")
            with open(best_program_info_path, "w") as f:
                json.dump(
                    {
                        "id": best_program.id,
                        "generation": best_program.generation,
                        "iteration": best_program.iteration_found,
                        "current_iteration": iteration,
                        "metrics": best_program.metrics,
                        "language": best_program.language,
                        "timestamp": best_program.timestamp,
                        "saved_at": time.time(),
                    },
                    f,
                    indent=2,
                )

            logger.info(
                f"Saved best program at checkpoint {iteration} with metrics: "
                f"{format_metrics_safe(best_program.metrics)}"
            )

        if v9_enabled:
            required_files = [
                "metadata.json",
                "global_island_summary.json",
                "pareto_archive.json",
            ]
            if self.optimizer_memory is not None:
                required_files.append("optimizer_memory.json")
            if hierarchical_design is not None:
                required_files.extend(
                    [
                        "hierarchical_design.json",
                        "hypothesis_ledger.json",
                        "reasoning_audit.json",
                    ]
                )
            _write_v9_checkpoint_manifest(
                checkpoint_path,
                iteration=iteration,
                required_files=required_files,
            )

        logger.info(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_path} not found")

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        memory_policy = getattr(self.config, "memory_policy", None)
        restored_adaptive_memory = None
        checkpoint_name = Path(checkpoint_path).name
        checkpoint_suffix = (
            checkpoint_name[len("checkpoint_") :]
            if checkpoint_name.startswith("checkpoint_")
            else ""
        )
        expected_iteration = (
            int(checkpoint_suffix) if checkpoint_suffix.isdigit() else None
        )
        v9_manifest = None
        v9_enabled = bool(
            callable(getattr(self.database, "_v9_population_policy_enabled", None))
            and self.database._v9_population_policy_enabled()
        )
        if v9_enabled:
            required_files = ["metadata.json", "global_island_summary.json"]
            if self.optimizer_memory is not None:
                required_files.append("optimizer_memory.json")
            if getattr(self, "hierarchical_design", None) is not None:
                required_files.extend(
                    [
                        "hierarchical_design.json",
                        "hypothesis_ledger.json",
                        "reasoning_audit.json",
                    ]
                )


            v9_manifest = _validate_v9_checkpoint_manifest(
                checkpoint_path,
                expected_iteration=expected_iteration,
                required_files=required_files,
            )
        restored_trace = _restore_evolution_trace_checkpoint(
            getattr(self, "evolution_tracer", None),
            checkpoint_path,
            expected_iteration=expected_iteration,
        )
        if memory_policy is not None:
            if not isinstance(memory_policy, MemoryPolicyConfig):
                raise TypeError("controller memory_policy must be MemoryPolicyConfig")


            restored_adaptive_memory = restore_adaptive_memory_checkpoint(
                getattr(self, "case_memory_source_path", None),
                getattr(self, "inner_memory_path", None),
                checkpoint_path,
                memory_policy,
                expected_iteration=expected_iteration,
            )
        self.database.load(checkpoint_path)
        if v9_manifest is not None and int(v9_manifest["iteration"]) != int(
            self.database.last_iteration
        ):
            raise CheckpointStateError(
                "v9 checkpoint manifest iteration disagrees with database state"
            )
        if restored_adaptive_memory is not None and int(
            restored_adaptive_memory["iteration"]
        ) != int(self.database.last_iteration):
            raise MemoryPolicyError(
                "adaptive memory checkpoint iteration disagrees with database state"
            )
        if restored_trace is not None and int(restored_trace["iteration"]) != int(
            self.database.last_iteration
        ):
            raise CheckpointStateError(
                "evolution trace checkpoint iteration disagrees with database state"
            )
        registry_backup = os.path.join(checkpoint_path, "experiment_registry.sqlite")
        if self.experiment_registry is not None and os.path.exists(registry_backup):
            registry_path = self.config.database.experiment_registry_path
            self.experiment_registry.close()
            ExperimentRegistry.restore_from_backup(
                registry_backup,
                registry_path,
                overwrite=True,
            )
            self.experiment_registry = ExperimentRegistry(
                registry_path,
                default_lease_seconds=(
                    self.config.database.experiment_registry_lease_seconds
                ),
            )
        if self.optimizer_memory is not None:
            memory_path = os.path.join(checkpoint_path, "optimizer_memory.json")
            if os.path.exists(memory_path):
                self.optimizer_memory.load(memory_path)
            self.optimizer_memory.rebuild_from_facts(
                self.database.get_memory_facts(self.memory_scope),
                scope=self.memory_scope,
                persist=True,
            )
        hierarchical_design = getattr(self, "hierarchical_design", None)
        if hierarchical_design is not None:
            design_path = os.path.join(checkpoint_path, "hierarchical_design.json")
            ledger_path = os.path.join(checkpoint_path, "hypothesis_ledger.json")
            reasoning_path = os.path.join(checkpoint_path, "reasoning_audit.json")
            if os.path.exists(design_path):
                hierarchical_design.load(design_path)
            if os.path.exists(ledger_path):
                hierarchical_design.ledger.load(ledger_path)
            if os.path.exists(reasoning_path):
                hierarchical_design.reasoning_ledger.load(reasoning_path)
        logger.info(f"Checkpoint loaded successfully (iteration {self.database.last_iteration})")

    async def _run_evolution_with_checkpoints(
        self, start_iteration: int, max_iterations: int, target_score: Optional[float]
    ) -> None:

        logger.info(f"Using island-based evolution with {self.config.database.num_islands} islands")
        self.database.log_island_status()


        await self.parallel_controller.run_evolution(
            start_iteration,
            max_iterations,
            target_score,
            checkpoint_callback=self._save_checkpoint,
            pre_wave_callback=self._prepare_hierarchical_strategy,
        )


        if self.parallel_controller.shutdown_event.is_set():
            logger.info("Evolution stopped due to shutdown request")
            return
        elif self.parallel_controller.early_stopping_triggered:
            logger.info("Evolution stopped due to early stopping - saving final checkpoint")


        final_iteration = start_iteration + max_iterations - 1
        if final_iteration > 0 and final_iteration % self.config.checkpoint_interval == 0:
            self._save_checkpoint(final_iteration)

    async def _run_evolution_serial_with_checkpoints(
        self, start_iteration: int, max_iterations: int, target_score: Optional[float]
    ) -> None:

        logger.info(f"Using island-based evolution with {self.config.database.num_islands} islands")
        self.database.log_island_status()
        logger.info(
            f"Starting serial evolution from iteration {start_iteration} "
            f"for {max_iterations} iterations"
        )

        wave_size = logical_proposal_wave_size(self.config)
        logger.info(
            f"Serial controller is using logical proposal waves of {wave_size}; "
            "evaluation remains in-process and sequential"
        )
        stopped_early = False
        last_completed_iteration = start_iteration - 1
        for wave in generation_chunks(start_iteration, max_iterations, wave_size):
            if bool(getattr(self, "_shutdown_requested", False)):
                logger.info("Serial evolution stopped at a graceful iteration boundary")
                stopped_early = True
                break
            await self._prepare_hierarchical_strategy(min(wave))
            target_islands = tuple(
                target_island_for_iteration(
                    iteration,
                    self.config.database.num_islands,
                )
                for iteration in wave
            )
            coordinator = GenerationCoordinator.begin(
                self.database,
                iterations=wave,
                target_islands=target_islands,
                inner_memory_path=self.inner_memory_path,
                output_root=os.path.join(self.output_dir, "generation_runtime"),
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


            reserved_inputs = {}
            parent_bindings = {}
            for reservation in coordinator.plan.reservations:
                parent, inspirations = self.database.sample_from_island(
                    island_id=reservation.target_island or 0,
                    num_inspirations=getattr(
                        self.config.prompt,
                        "num_diverse_programs",
                        self.config.prompt.num_top_programs,
                    ),
                )
                reserved_inputs[reservation.proposal_id] = (parent, inspirations)
                envelope = envelope_from_parent_artifacts(
                    self.database.get_artifacts(parent.id),
                    parent_program_id=parent.id,
                )
                parent_bindings[reservation.proposal_id] = (
                    parent.id,
                    envelope.to_dict() if envelope is not None else None,
                )
            coordinator = coordinator.bind_parents(parent_bindings)

            observations: List[GenerationObservation] = []
            for reservation in coordinator.plan.reservations:
                parent, inspirations = reserved_inputs[reservation.proposal_id]
                result = await run_iteration_with_shared_db(
                    iteration=reservation.iteration,
                    config=self.config,
                    database=self.database,
                    evaluator=self.evaluator,
                    llm_ensemble=self.llm_ensemble,
                    prompt_sampler=self.prompt_sampler,
                    generation_plan=coordinator.plan,
                    proposal_id=reservation.proposal_id,
                    preselected_parent=parent,
                    preselected_inspirations=inspirations,
                )
                if result is None or result.child_program is None:
                    logger.warning(
                        f"Iteration {reservation.iteration}: no valid child program produced"
                    )
                    observations.append(
                        GenerationObservation(
                            proposal_id=reservation.proposal_id,
                            iteration=reservation.iteration,
                            parent=(
                                result.parent
                                if result is not None and result.parent is not None
                                else parent
                            ),
                            artifacts=(result.artifacts or {}) if result is not None else {},
                            target_island=reservation.target_island,
                            prompt=result.prompt if result is not None else None,
                            llm_response=(
                                result.llm_response if result is not None else None
                            ),
                            iteration_time=(
                                result.iteration_time or 0.0
                                if result is not None
                                else 0.0
                            ),
                            error=(
                                result.error
                                if result is not None and result.error
                                else "no_valid_child_program"
                            ),
                            generation_id=coordinator.plan.generation_id,
                            trial_id=(result.trial_id if result is not None else ""),
                            proposal_causal_envelope_json=(
                                reservation.proposal_causal_envelope_json
                            ),
                            proposal_causal_envelope_hash=(
                                reservation.proposal_causal_envelope_hash
                            ),
                        )
                    )
                    continue
                observations.append(
                    GenerationObservation(
                        proposal_id=reservation.proposal_id,
                        iteration=reservation.iteration,
                        child_program=result.child_program,
                        parent=result.parent,
                        artifacts=result.artifacts or {},
                        target_island=reservation.target_island,
                        prompt=result.prompt,
                        llm_response=result.llm_response,
                        iteration_time=result.iteration_time or 0.0,
                    )
                )

            lifecycle = coordinator.finalize(self.database, observations)
            last_completed_iteration = max(wave)


            for observation in sorted(observations, key=lambda item: item.stable_key):
                child_program = observation.child_program
                parent_program = observation.parent
                if child_program is None:
                    continue
                if self.evolution_tracer and parent_program:
                    self.evolution_tracer.log_trace(
                        iteration=observation.iteration,
                        parent_program=parent_program,
                        child_program=child_program,
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
                            "changes": child_program.metadata.get("changes", ""),
                            "generation_id": coordinator.plan.generation_id,
                        },
                    )
                if observation.target_island is not None:
                    self.database.increment_island_generation(
                        island_idx=observation.target_island
                    )
                else:
                    self.database.increment_island_generation()
                if parent_program:
                    self._log_iteration(
                        iteration=observation.iteration,
                        parent=parent_program,
                        child=child_program,
                        elapsed_time=observation.iteration_time,
                    )

            if self.database.should_migrate():
                logger.info(f"Performing migration after generation {coordinator.plan.generation_id}")
                self.database.migrate_programs()
                self.database.log_island_status()

            checkpoint_points = [
                iteration
                for iteration in wave
                if iteration > 0 and iteration % self.config.checkpoint_interval == 0
            ]
            if checkpoint_points:
                checkpoint_iteration = max(wave)
                logger.info(
                    "Checkpoint interval crossed inside generation; saving the "
                    f"committed wave at iteration {checkpoint_iteration}"
                )
                self.database.log_island_status()
                self._save_checkpoint(checkpoint_iteration)

            winner_program_id = (lifecycle.get("winner") or {}).get("program_id")
            winner_program = self.database.get(winner_program_id) if winner_program_id else None
            if target_score is not None and winner_program and winner_program.metrics:
                current = winner_program.metrics.get("combined_score")
                if isinstance(current, (int, float)) and current >= target_score:
                    logger.info(
                        f"Target score {target_score} reached after generation "
                        f"{coordinator.plan.generation_id}"
                    )
                    stopped_early = True
                    break

        if (
            not stopped_early
            and last_completed_iteration > 0
            and last_completed_iteration % self.config.checkpoint_interval == 0
        ):
            self._save_checkpoint(last_completed_iteration)

    def _save_best_program(self, program: Optional[Program] = None) -> None:


        if program is None:
            if self.database._v9_population_policy_enabled():
                recommended_id = self.database.get_global_summary(top_n=12).get(
                    "recommended_program_id"
                )
                if recommended_id:
                    program = self.database.get(recommended_id)
            if program is None and self.database.best_program_id:
                program = self.database.get(self.database.best_program_id)
            if program is None:

                program = self.database.get_best_program()

        if not program:
            logger.warning("No best program found to save")
            return

        best_dir = os.path.join(self.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)


        filename = f"best_program{self.file_extension}"
        code_path = os.path.join(best_dir, filename)

        with open(code_path, "w") as f:
            f.write(program.code)


        info_path = os.path.join(best_dir, "best_program_info.json")
        with open(info_path, "w") as f:
            import json

            json.dump(
                {
                    "id": program.id,
                    "generation": program.generation,
                    "iteration": program.iteration_found,
                    "timestamp": program.timestamp,
                    "parent_id": program.parent_id,
                    "metrics": program.metrics,
                    "language": program.language,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
            )

        logger.info(f"Saved best program to {code_path} with program info to {info_path}")
