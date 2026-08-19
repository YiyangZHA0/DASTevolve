

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from ._mapping import copied_mapping, optional_float


@dataclass(frozen=True)
class ScoreTermRecord:


    name: str
    category: str
    score: float
    weight: float = 1.0
    backend: str = "fast"
    available: bool = True
    details: Mapping[str, Any] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()
    provider: str = "fast"
    state: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScoreTermRecord":
        raw_weight = optional_float(value.get("weight"))
        return cls(
            name=str(value.get("name") or "unnamed"),
            category=str(value.get("category") or "uncategorized"),
            score=float(optional_float(value.get("score")) or 0.0),
            weight=1.0 if raw_weight is None else float(raw_weight),
            backend=str(value.get("backend") or "fast"),
            available=bool(value.get("available", True)),
            details=copied_mapping(value.get("details")),
            warnings=tuple(str(item) for item in value.get("warnings", ()) or ()),
            provider=str(value.get("provider") or value.get("backend") or "fast"),
            state=(str(value["state"]) if value.get("state") is not None else None),
        )

    def to_dict(self) -> Dict[str, Any]:


        return {
            "name": self.name,
            "category": self.category,
            "score": self.score,
            "weight": self.weight,
            "weighted_score": self.score * self.weight,
            "backend": self.backend,
            "provider": self.provider,
            "state": self.state,
            "available": self.available,
            "details": copied_mapping(self.details),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class GateResult:


    passed: bool
    failures: Tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationReport:


    normalized_score: float
    loss: float
    gate: GateResult
    terms: Tuple[ScoreTermRecord, ...] = ()
    recommended_edit_targets: Tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "astevolve.evaluation.v1"
    energy_schema_version: str | None = None
    direction: str | None = None
    soft_energy: float | None = None
    total_energy: float | None = None
    term_energy_breakdown: Tuple[Mapping[str, Any], ...] = ()
    category_energy_breakdown: Mapping[str, Any] = field(default_factory=dict)
    energy_coverage: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:


        energy_values_present = any(
            value is not None
            for value in (self.direction, self.soft_energy, self.total_energy)
        ) or bool(
            self.term_energy_breakdown
            or self.category_energy_breakdown
            or self.energy_coverage
        )
        if self.energy_schema_version is None:
            if energy_values_present:
                raise ValueError(
                    "energy fields require energy_schema_version"
                )
            return
        if self.energy_schema_version != "astevolve.design_energy.v1":
            raise ValueError("unsupported evaluation energy schema")
        if self.direction != "minimize":
            raise ValueError("evaluation energy direction must be 'minimize'")
        for name, value in (
            ("soft_energy", self.soft_energy),
            ("total_energy", self.total_energy),
        ):
            if (
                value is None
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be a finite value in [0, 1]")
        if float(self.total_energy) != float(self.soft_energy):
            raise ValueError("total_energy must equal soft_energy in energy schema v1")

    @classmethod
    def from_legacy(cls, value: Mapping[str, Any] | None) -> "EvaluationReport":


        from astevolve.evaluation.selection import normalize_gate_payload

        data = copied_mapping(value)
        raw_gate = copied_mapping(data.get("gate_status"))
        normalized_gate = normalize_gate_payload(data)
        failures = tuple(str(item) for item in normalized_gate["reasons"])
        gate_details = {
            **raw_gate,
            "passed": bool(normalized_gate["passed"]),
            "hard_gate_pass": bool(normalized_gate["passed"]),
            "disqualification_reasons": list(failures),
            "hard_failures": list(failures),
        }
        raw_terms = data.get("terms") or data.get("score_terms") or ()
        energy_schema_version = data.get("energy_schema_version")
        energy_enabled = energy_schema_version is not None
        known = {
            "schema_version",
            "normalized_score",
            "loss",
            "gate_status",
            "hard_gate_pass",
            "disqualification_reasons",
            "terms",
            "score_terms",
            "recommended_edit_targets",
        }
        if energy_enabled:
            known.update(
                {
                    "energy_schema_version",
                    "direction",
                    "soft_energy",
                    "total_energy",
                    "term_energy_breakdown",
                    "category_energy_breakdown",
                    "energy_coverage",
                }
            )
        return cls(
            normalized_score=float(optional_float(data.get("normalized_score")) or 0.0),
            loss=float(optional_float(data.get("loss")) or 0.0),
            gate=GateResult(
                passed=bool(normalized_gate["passed"]),
                failures=failures,
                details=gate_details,
            ),
            terms=tuple(ScoreTermRecord.from_mapping(item) for item in raw_terms if isinstance(item, Mapping)),
            recommended_edit_targets=tuple(
                copied_mapping(item) for item in data.get("recommended_edit_targets", ()) or () if isinstance(item, Mapping)
            ),
            metadata={key: item for key, item in data.items() if key not in known},
            schema_version=str(data.get("schema_version") or "astevolve.evaluation.v1"),
            energy_schema_version=(
                str(energy_schema_version) if energy_enabled else None
            ),
            direction=(str(data.get("direction")) if energy_enabled else None),
            soft_energy=(
                optional_float(data.get("soft_energy")) if energy_enabled else None
            ),
            total_energy=(
                optional_float(data.get("total_energy")) if energy_enabled else None
            ),
            term_energy_breakdown=tuple(
                copied_mapping(item)
                for item in data.get("term_energy_breakdown", ()) or ()
                if isinstance(item, Mapping)
            ),
            category_energy_breakdown=copied_mapping(
                data.get("category_energy_breakdown")
            ),
            energy_coverage=copied_mapping(data.get("energy_coverage")),
        )

    def to_legacy_dict(self) -> Dict[str, Any]:
        data = copied_mapping(self.metadata)
        failures = list(self.gate.failures)
        gate_status = {
            **copied_mapping(self.gate.details),
            "passed": self.gate.passed,
            "hard_gate_pass": self.gate.passed,
            "disqualification_reasons": failures,
            "hard_failures": failures,
        }
        data.update(
            {
                "schema_version": self.schema_version,
                "normalized_score": self.normalized_score,
                "loss": self.loss,
                "hard_gate_pass": self.gate.passed,
                "disqualification_reasons": failures,
                "gate_status": gate_status,
                "terms": [term.to_dict() for term in self.terms],
                "recommended_edit_targets": [copied_mapping(item) for item in self.recommended_edit_targets],
            }
        )
        if self.energy_schema_version is not None:
            data.update(
                {
                    "energy_schema_version": self.energy_schema_version,
                    "direction": self.direction,
                    "soft_energy": self.soft_energy,
                    "total_energy": self.total_energy,
                    "term_energy_breakdown": [
                        copied_mapping(item)
                        for item in self.term_energy_breakdown
                    ],
                    "category_energy_breakdown": copied_mapping(
                        self.category_energy_breakdown
                    ),
                    "energy_coverage": copied_mapping(self.energy_coverage),
                }
            )
        return data
