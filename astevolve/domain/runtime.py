

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

from ._mapping import copied_mapping, copied_sequence_mapping, optional_float
from .evaluation import EvaluationReport
from .strategy import EditContract


@dataclass(frozen=True)
class RunContext:


    case_id: str
    project_root: Path
    output_root: Path
    design_state_path: Path
    memory_path: Path
    seed: int | None = None
    settings: Mapping[str, Any] = field(default_factory=dict)
    run_id: str = ""
    schema_version: str = "astevolve.run_context.v1"


@dataclass(frozen=True)
class SearchResult:


    sequences: Mapping[str, str]
    fast_loss: float | None
    constraint_penalty: float | None
    search_method: str
    raw: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "astevolve.search_result.v1"

    @classmethod
    def from_legacy(cls, value: Mapping[str, Any]) -> "SearchResult":


        data = copied_mapping(value)
        return cls(
            sequences=copied_sequence_mapping(data.get("seqs")),
            fast_loss=optional_float(data.get("fast_loss")),
            constraint_penalty=optional_float(data.get("constraint_penalty")),
            search_method=str(data.get("search_method") or ""),
            raw=data,
        )

    def to_legacy_dict(self) -> Dict[str, Any]:


        return copied_mapping(self.raw)


@dataclass(frozen=True)
class ExperimentResult:


    sequences: Mapping[str, str]
    evaluation: EvaluationReport
    edit_contract: EditContract
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "astevolve.experiment_result.v1"

    @classmethod
    def from_legacy(cls, value: Mapping[str, Any]) -> "ExperimentResult":
        data = copied_mapping(value)
        artifact_keys = {
            "search_artifacts",
            "semantic_graph_summary",
            "semantic_graph_diagnosis",
            "residue_semantic_map_summary",
            "memory_update",
        }
        return cls(
            sequences=copied_sequence_mapping(data.get("seqs") or data.get("best_seqs")),
            evaluation=EvaluationReport.from_legacy(data.get("evaluator_report")),
            edit_contract=EditContract.from_mapping(data.get("edit_contract")),
            artifacts={key: data[key] for key in artifact_keys if key in data},
            raw=data,
        )

    def to_legacy_dict(self) -> Dict[str, Any]:
        return copied_mapping(self.raw)
