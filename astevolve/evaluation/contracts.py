

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from astevolve.domain.evaluation import ScoreTermRecord

from .support import clamp01


@dataclass
class ScoreTerm:


    name: str
    category: str
    score: float
    weight: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    backend: str = "fast"
    available: bool = True

    @property
    def normalized_score(self) -> float:


        return clamp01(self.score)

    @property
    def cost(self) -> float:


        return float(1.0 - self.normalized_score)

    @property
    def required(self) -> bool:


        return bool(
            self.details.get("required", False)
            if isinstance(self.details, Mapping)
            else False
        )

    @property
    def explicitly_optional(self) -> bool:


        if not isinstance(self.details, Mapping):
            return False
        return (
            self.details.get("required") is False
            or self.details.get("enabled") is False
            or bool(self.details.get("ignored_for_score"))
        )

    @property
    def provider(self) -> str:


        plugin_name = (
            self.details.get("_plugin_name")
            if isinstance(self.details, Mapping)
            else None
        )
        return str(plugin_name or self.backend)

    @property
    def state(self) -> Optional[str]:
        value = self.details.get("state") if isinstance(self.details, Mapping) else None
        text = str(value or "").strip().lower()
        return text if text in {"positive", "negative", "preserve"} else None

    def to_dict(self) -> Dict[str, Any]:


        score = self.normalized_score
        return {
            "name": self.name,
            "category": self.category,
            "score": score,
            "weight": float(self.weight),
            "weighted_score": float(score * self.weight),
            "available": bool(self.available),
            "backend": self.backend,
            "provider": self.provider,
            "state": self.state,
            "details": self.details,
            "warnings": list(self.warnings),
        }

    def to_record(self) -> ScoreTermRecord:


        return ScoreTermRecord.from_mapping(self.to_dict())


@dataclass
class EvaluatorContext:


    out: Mapping[str, Any]
    structure: Mapping[str, Any]
    compiled: Optional[Mapping[str, Any]]
    design_state: Mapping[str, Any]
    masks: Optional[Mapping[str, Any]]
    template_seqs: Optional[Mapping[str, str]]
    fixed_residues: Optional[Mapping[str, Mapping[Any, str]]]
    score_config: Mapping[str, Any]
    plugin_name: Optional[str] = None
    plugin_config: Mapping[str, Any] = field(default_factory=dict)
