

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from outerloop.utils.trace_export_utils import (
    append_traces_jsonl,
    export_traces,
    export_traces_json,
)
from outerloop.utils.metric_semantics import compare_metrics, summarize_comparisons

logger = logging.getLogger(__name__)


@dataclass
class EvolutionTrace:


    iteration: int
    timestamp: float
    parent_id: str
    child_id: str
    parent_metrics: Dict[str, Any]
    child_metrics: Dict[str, Any]
    parent_code: Optional[str] = None
    child_code: Optional[str] = None
    parent_changes_description: Optional[str] = None
    child_changes_description: Optional[str] = None
    code_diff: Optional[str] = None
    prompt: Optional[Dict[str, str]] = None
    llm_response: Optional[str] = None
    raw_delta: Optional[Dict[str, float]] = None
    improvement_delta: Optional[Dict[str, float]] = None
    metric_comparisons: Optional[Dict[str, Dict[str, Any]]] = None
    island_id: Optional[int] = None
    generation: Optional[int] = None
    artifacts: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    schema_version: str = "astevolve.evolution_trace.v2"
    source_schema_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:

        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "EvolutionTrace":


        if not isinstance(value, dict):
            raise TypeError("EvolutionTrace payload must be a mapping")
        field_names = set(cls.__dataclass_fields__)
        payload = {key: item for key, item in value.items() if key in field_names}
        source_version = value.get("schema_version") or "legacy.v1"
        payload["schema_version"] = "astevolve.evolution_trace.v2"
        payload["source_schema_version"] = source_version
        if source_version == "legacy.v1":


            payload["raw_delta"] = None
            payload["improvement_delta"] = None
            payload["metric_comparisons"] = None
        trace = cls(**payload)
        if source_version == "legacy.v1" or trace.metric_comparisons is None:
            trace.populate_metric_comparisons()
        return trace

    def populate_metric_comparisons(self) -> Dict[str, Any]:


        summary = summarize_comparisons(
            compare_metrics(self.parent_metrics, self.child_metrics)
        )
        self.raw_delta = dict(summary["raw_deltas"])
        self.improvement_delta = dict(summary["directional_deltas"])
        self.metric_comparisons = dict(summary["comparisons"])
        return summary

    def calculate_improvement(self) -> Dict[str, float]:


        return dict(
            summarize_comparisons(
                compare_metrics(self.parent_metrics, self.child_metrics)
            )["directional_deltas"]
        )


class EvolutionTracer:


    def __init__(
        self,
        output_path: Optional[str] = None,
        format: str = "jsonl",
        include_code: bool = False,
        include_prompts: bool = True,
        enabled: bool = True,
        buffer_size: int = 10,
        compress: bool = False,
        include_changes_description: bool = True,
    ):

        self.enabled = enabled
        self.format = format
        self.include_code = include_code
        self.include_prompts = include_prompts
        self.include_changes_description = include_changes_description
        self.compress = compress
        self.buffer_size = buffer_size


        self.stats = {
            "total_traces": 0,
            "improvement_count": 0,
            "total_improvement": {},
            "best_improvement": {},
            "worst_decline": {},
        }

        if not self.enabled:
            logger.info("Evolution tracing is disabled")
            return


        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = Path(f"evolution_trace.{format}")


        if self.compress and format == "jsonl":
            self.output_path = self.output_path.with_suffix(".jsonl.gz")


        self.output_path.parent.mkdir(parents=True, exist_ok=True)


        self.buffer: List[EvolutionTrace] = []


        if format == "json":
            self.json_traces = []

        logger.info(f"Evolution tracer initialized: {self.output_path}")

    def log_trace(
        self,
        iteration: int,
        parent_program: Any,
        child_program: Any,
        prompt: Optional[Dict[str, str]] = None,
        llm_response: Optional[str] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        island_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:

        if not self.enabled:
            return

        try:

            trace = EvolutionTrace(
                iteration=iteration,
                timestamp=time.time(),
                parent_id=parent_program.id,
                child_id=child_program.id,
                parent_metrics=parent_program.metrics,
                child_metrics=child_program.metrics,
                island_id=island_id,
                generation=child_program.generation,
                artifacts=artifacts,
                metadata=metadata,
            )


            if self.include_code:
                trace.parent_code = parent_program.code
                trace.child_code = child_program.code


            if self.include_changes_description:
                trace.parent_changes_description = getattr(parent_program, "changes_description", None)
                trace.child_changes_description = getattr(child_program, "changes_description", None)


            if self.include_prompts:
                trace.prompt = prompt
                trace.llm_response = llm_response

            trace.populate_metric_comparisons()


            self._update_stats(trace)


            self.buffer.append(trace)


            if self.format == "json":
                self.json_traces.append(trace)


            if len(self.buffer) >= self.buffer_size:
                self.flush()

        except Exception as e:
            logger.error(f"Error logging evolution trace: {e}")

    def _update_stats(self, trace: EvolutionTrace):

        self.stats["total_traces"] += 1

        if trace.improvement_delta:

            if "combined_score" in trace.improvement_delta:
                delta = trace.improvement_delta["combined_score"]
                if delta > 0:
                    self.stats["improvement_count"] += 1


            for metric, delta in trace.improvement_delta.items():
                if metric not in self.stats["total_improvement"]:
                    self.stats["total_improvement"][metric] = 0
                    self.stats["best_improvement"][metric] = delta
                    self.stats["worst_decline"][metric] = delta

                self.stats["total_improvement"][metric] += delta

                if delta > self.stats["best_improvement"][metric]:
                    self.stats["best_improvement"][metric] = delta
                if delta < self.stats["worst_decline"][metric]:
                    self.stats["worst_decline"][metric] = delta

    def flush(self):

        if not self.enabled or not self.buffer:
            return

        try:
            if self.format == "jsonl":
                append_traces_jsonl(
                    self.buffer,
                    self.output_path,
                    compress=self.compress,
                )
            elif self.format == "json":

                pass
            elif self.format == "hdf5":


                pass


            if self.format == "jsonl":
                self.buffer.clear()

        except Exception as e:
            logger.error(f"Error flushing traces to file: {e}")

    def get_statistics(self) -> Dict[str, Any]:

        return {
            **self.stats,
            "improvement_rate": (
                self.stats["improvement_count"] / self.stats["total_traces"]
                if self.stats["total_traces"] > 0
                else 0
            ),
        }

    def close(self):

        if not self.enabled:
            return


        self.flush()


        if self.format == "json" and hasattr(self, "json_traces"):
            metadata = {
                "created_at": time.time(),
                "include_code": self.include_code,
                "include_prompts": self.include_prompts,
                "include_changes_description": getattr(self, "include_changes_description", True),
                "total_traces": len(self.json_traces),
            }
            export_traces_json(self.json_traces, self.output_path, metadata=metadata)
        elif self.format == "hdf5":

            all_traces = getattr(self, "json_traces", self.buffer)
            if all_traces:
                metadata = {
                    "created_at": time.time(),
                    "include_code": self.include_code,
                    "include_prompts": self.include_prompts,
                    "include_changes_description": getattr(self, "include_changes_description", True),
                }
                export_traces(all_traces, self.output_path, format="hdf5", metadata=metadata)


        stats = self.get_statistics()
        logger.info(f"Evolution tracing complete. Total traces: {stats['total_traces']}")
        logger.info(f"Improvement rate: {stats['improvement_rate']:.2%}")

        if stats["best_improvement"]:
            logger.info(f"Best improvements: {stats['best_improvement']}")
        if stats["worst_decline"]:
            logger.info(f"Worst declines: {stats['worst_decline']}")

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.close()


def extract_evolution_trace_from_checkpoint(
    checkpoint_dir: Union[str, Path],
    output_path: Optional[str] = None,
    format: str = "jsonl",
    include_code: bool = True,
    include_prompts: bool = True,
    include_changes_description: bool = True,
) -> List[EvolutionTrace]:

    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found")

    programs_dir = checkpoint_path / "programs"
    if not programs_dir.exists():
        raise FileNotFoundError(f"Programs directory not found in {checkpoint_dir}")

    logger.info(f"Extracting evolution traces from {checkpoint_dir}")


    programs = {}
    program_files = list(programs_dir.glob("*.json"))

    for prog_file in program_files:
        try:
            with open(prog_file, "r") as f:
                prog_data = json.load(f)
                programs[prog_data["id"]] = prog_data
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error loading program from {prog_file}: {e}")
            continue

    logger.info(f"Loaded {len(programs)} programs from checkpoint")


    traces = []
    for prog_id, prog in programs.items():

        parent_id = prog.get("parent_id")
        if not parent_id or parent_id not in programs:
            continue

        parent = programs[parent_id]


        parent_desc = parent.get("changes_description")
        if not parent_desc:
            parent_desc = (parent.get("metadata", {}) or {}).get("changes_description") or (
                parent.get("metadata", {}) or {}
            ).get("changes")

        child_desc = prog.get("changes_description")
        if not child_desc:
            child_desc = (prog.get("metadata", {}) or {}).get("changes_description") or (
                prog.get("metadata", {}) or {}
            ).get("changes")

        trace = EvolutionTrace(
            iteration=prog.get("iteration_found", 0),
            timestamp=prog.get("timestamp", 0),
            parent_id=parent_id,
            child_id=prog_id,
            parent_metrics=parent.get("metrics", {}),
            child_metrics=prog.get("metrics", {}),
            generation=prog.get("generation", 0),
            island_id=prog.get("metadata", {}).get("island"),
            parent_changes_description=parent_desc if include_changes_description else None,
            child_changes_description=child_desc if include_changes_description else None,
            metadata=prog.get("metadata", {}),
        )


        if include_code:
            trace.parent_code = parent.get("code", "")
            trace.child_code = prog.get("code", "")


        if include_prompts:

            if "prompts" in prog:

                trace.prompt = prog["prompts"].get("prompt")
                trace.llm_response = prog["prompts"].get("response")

        trace.populate_metric_comparisons()

        traces.append(trace)


    traces.sort(key=lambda x: (x.iteration, x.timestamp))

    logger.info(f"Extracted {len(traces)} evolution traces")


    if output_path:
        metadata = {
            "total_traces": len(traces),
            "extracted_at": time.time(),
            "source": "checkpoint",
        }
        export_traces(traces, output_path, format=format, metadata=metadata)
        logger.info(f"Saved {len(traces)} traces to {output_path}")

    return traces


def extract_full_lineage_traces(
    checkpoint_dir: Union[str, Path], output_path: Optional[str] = None, format: str = "json"
) -> List[Dict[str, Any]]:

    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found")

    programs_dir = checkpoint_path / "programs"
    if not programs_dir.exists():
        raise FileNotFoundError(f"Programs directory not found in {checkpoint_dir}")

    logger.info(f"Extracting full lineage traces from {checkpoint_dir}")


    programs = {}
    program_files = list(programs_dir.glob("*.json"))

    for prog_file in program_files:
        try:
            with open(prog_file, "r") as f:
                prog_data = json.load(f)
                programs[prog_data["id"]] = prog_data
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error loading program from {prog_file}: {e}")
            continue

    logger.info(f"Loaded {len(programs)} programs from checkpoint")


    traces = []

    for program_id, program in programs.items():

        lineage = []
        current = program

        while current:
            lineage.append(current)
            parent_id = current.get("parent_id")
            current = programs.get(parent_id) if parent_id else None


        lineage.reverse()


        improvements = []
        for i in range(len(lineage) - 1):
            parent = lineage[i]
            child = lineage[i + 1]


            prompts = child.get("prompts", {})
            action = None


            for template_key, prompt_data in prompts.items():
                action = {
                    "template": template_key,
                    "system_prompt": prompt_data.get("system", ""),
                    "user_prompt": prompt_data.get("user", ""),
                    "llm_response": (
                        prompt_data.get("responses", [""])[0]
                        if prompt_data.get("responses")
                        else ""
                    ),
                }
                break

            metric_summary = summarize_comparisons(
                compare_metrics(parent.get("metrics", {}), child.get("metrics", {}))
            )

            improvement = {
                "step": i,
                "parent_id": parent["id"],
                "child_id": child["id"],
                "parent_metrics": parent.get("metrics", {}),
                "child_metrics": child.get("metrics", {}),
                "schema_version": "astevolve.evolution_trace_step.v2",
                "raw_delta": metric_summary["raw_deltas"],
                "improvement": metric_summary["directional_deltas"],
                "metric_comparisons": metric_summary["comparisons"],
                "generation": child.get("generation", 0),
                "iteration_found": child.get("iteration_found", 0),
                "changes_description": (
                    child.get("changes_description")
                    or (child.get("metadata", {}) or {}).get("changes_description")
                    or (child.get("metadata", {}) or {}).get("changes")
                    or ""
                ),
                "island_id": child.get("metadata", {}).get("island"),
                "action": action,
            }
            improvements.append(improvement)


        if improvements:
            trace = {
                "schema_version": "astevolve.evolution_lineage_trace.v2",
                "final_program_id": program_id,
                "final_metrics": program.get("metrics", {}),
                "final_changes_description": (
                    program.get("changes_description")
                    or (program.get("metadata", {}) or {}).get("changes_description")
                    or (program.get("metadata", {}) or {}).get("changes")
                    or ""
                ),
                "generation_depth": len(lineage),
                "total_iterations": program.get("iteration_found", 0),
                "improvement_steps": improvements,
                "metadata": {
                    "language": program.get("language", ""),
                    "timestamp": program.get("timestamp", 0),
                },
            }
            traces.append(trace)


    traces.sort(key=lambda x: x["generation_depth"], reverse=True)

    logger.info(f"Extracted {len(traces)} lineage traces")


    if output_path:
        if format == "json":
            metadata = {
                "total_traces": len(traces),
                "extracted_at": time.time(),
                "source": "checkpoint",
                "type": "full_lineage",
            }
            export_traces_json(traces, output_path, metadata=metadata)
        elif format == "jsonl":

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for trace in traces:
                    json.dump(trace, f)
                    f.write("\n")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'jsonl'")

        logger.info(f"Saved {len(traces)} lineage traces to {output_path}")

    return traces
