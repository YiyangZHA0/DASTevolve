

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Tuple


METRIC_SEMANTICS_VERSION = "astevolve.metric_semantics.v1"

MetricDirection = Literal["maximize", "minimize", "diagnostic"]
MissingPolicy = Literal["not_comparable"]
ObservationStatus = Literal["valid", "missing", "invalid", "diagnostic"]
ComparisonOutcome = Literal["improved", "regressed", "unchanged", "not_comparable"]


@dataclass(frozen=True)
class MetricSpec:


    name: str
    direction: MetricDirection
    missing_policy: MissingPolicy = "not_comparable"
    registered: bool = True

    @property
    def directional(self) -> bool:
        return self.direction in {"maximize", "minimize"}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "missing_policy": self.missing_policy,
            "registered": self.registered,
        }


@dataclass(frozen=True)
class MetricObservation:


    metric: str
    spec: MetricSpec
    status: ObservationStatus
    value: Optional[float] = None
    reason: Optional[str] = None

    @property
    def comparable(self) -> bool:
        return self.status == "valid" and self.spec.directional

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "metric": self.metric,
            "direction": self.spec.direction,
            "registered": self.spec.registered,
            "status": self.status,
            "comparable": self.comparable,
        }
        if self.value is not None:
            payload["value"] = self.value
        if self.reason:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True)
class MetricComparison:


    metric: str
    spec: MetricSpec
    parent: MetricObservation
    child: MetricObservation
    outcome: ComparisonOutcome
    raw_delta: Optional[float] = None
    improvement_delta: Optional[float] = None
    reason: Optional[str] = None

    @property
    def comparable(self) -> bool:
        return self.outcome != "not_comparable"

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "metric": self.metric,
            "direction": self.spec.direction,
            "registered": self.spec.registered,
            "missing_policy": self.spec.missing_policy,
            "parent": self.parent.to_dict(),
            "child": self.child.to_dict(),
            "outcome": self.outcome,
            "comparable": self.comparable,
            "raw_delta": self.raw_delta,
            "improvement_delta": self.improvement_delta,
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


def _specs(direction: MetricDirection, names: Iterable[str]) -> Dict[str, MetricSpec]:
    return {name: MetricSpec(name=name, direction=direction) for name in names}


_REGISTRY: Dict[str, MetricSpec] = {}
_REGISTRY.update(
    _specs(
        "maximize",
        (
            "combined_score",
            "raw_combined_score",
            "score",
            "normalized_score",
            "struct_score",
            "structure_score",
            "multistate_score",
            "evaluator_score",
            "fast_score",
            "plddt",
            "plddt_score",
            "iptm",
            "iptm_score",
            "ptm",
            "ptm_score",
            "interface_plddt_mean",
            "interface_contact_count",
            "contact_count",
            "specificity_score",
            "negative_design_score",
            "pocket_score",
            "ligand_score",
            "allostery_score",
        ),
    )
)
_REGISTRY.update(
    _specs(
        "minimize",
        (
            "raw_combined_energy",
            "combined_energy",
            "final_energy",
            "loss",
            "combined_loss",
            "design_loss",
            "total_loss",
            "fast_loss",
            "multistate_loss",
            "evaluator_loss",
            "evaluator_energy",
            "clash",
            "clash_count",
            "steric_clash_count",
            "ligand_clash_count",
            "total_ligand_clash_count",
            "disqualified_count",
            "disqualification_count",
            "hard_gate_failure_count",
            "rmsd",
            "ca_rmsd",
        ),
    )
)
_REGISTRY.update(
    _specs(
        "diagnostic",
        (
            "error",
            "timeout",
            "has_clash",
            "feasible",
            "hard_gate_pass",
            "stage1_passed",
            "stage2_passed",
            "stage3_passed",
            "disqualified",
            "severe_clash_filter",
            "gate_status",
            "generation",
            "iteration",
            "execution_time",
            "prompt_length",
            "complexity",
            "diversity",
        ),
    )
)


METRIC_REGISTRY: Mapping[str, MetricSpec] = MappingProxyType(_REGISTRY)

_MISSING = object()


def get_metric_spec(name: str) -> MetricSpec:


    metric = str(name)
    spec = METRIC_REGISTRY.get(metric)
    if spec is not None:
        return spec
    return MetricSpec(name=metric, direction="diagnostic", registered=False)


def observe_metric(name: str, value: Any = _MISSING) -> MetricObservation:


    spec = get_metric_spec(name)
    if value is _MISSING or value is None:
        return MetricObservation(
            metric=spec.name,
            spec=spec,
            status="missing",
            reason=spec.missing_policy,
        )
    if isinstance(value, bool):
        return MetricObservation(
            metric=spec.name,
            spec=spec,
            status="invalid",
            reason="boolean_is_not_numeric_metric_evidence",
        )
    if not isinstance(value, (int, float)):
        return MetricObservation(
            metric=spec.name,
            spec=spec,
            status="invalid",
            reason="non_numeric_metric_value",
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        return MetricObservation(
            metric=spec.name,
            spec=spec,
            status="invalid",
            reason="non_finite_metric_value",
        )
    if not spec.directional:
        return MetricObservation(
            metric=spec.name,
            spec=spec,
            status="diagnostic",
            value=numeric,
            reason="metric_has_no_registered_optimization_direction",
        )
    return MetricObservation(metric=spec.name, spec=spec, status="valid", value=numeric)


def compare_metric(
    name: str,
    parent_value: Any = _MISSING,
    child_value: Any = _MISSING,
    *,
    tolerance: float = 1e-12,
) -> MetricComparison:


    parent = observe_metric(name, parent_value)
    child = observe_metric(name, child_value)
    spec = parent.spec
    if not parent.comparable or not child.comparable:
        reasons = []
        if not parent.comparable:
            reasons.append(f"parent:{parent.status}")
        if not child.comparable:
            reasons.append(f"child:{child.status}")
        return MetricComparison(
            metric=spec.name,
            spec=spec,
            parent=parent,
            child=child,
            outcome="not_comparable",
            reason=",".join(reasons) or spec.missing_policy,
        )

    assert parent.value is not None and child.value is not None
    raw_delta = child.value - parent.value
    improvement_delta = raw_delta if spec.direction == "maximize" else -raw_delta
    if improvement_delta > abs(float(tolerance)):
        outcome: ComparisonOutcome = "improved"
    elif improvement_delta < -abs(float(tolerance)):
        outcome = "regressed"
    else:
        outcome = "unchanged"
    return MetricComparison(
        metric=spec.name,
        spec=spec,
        parent=parent,
        child=child,
        outcome=outcome,
        raw_delta=raw_delta,
        improvement_delta=improvement_delta,
    )


def compare_metrics(
    parent_metrics: Optional[Mapping[str, Any]],
    child_metrics: Optional[Mapping[str, Any]],
    *,
    names: Optional[Iterable[str]] = None,
    tolerance: float = 1e-12,
) -> Tuple[MetricComparison, ...]:


    parent = parent_metrics if isinstance(parent_metrics, Mapping) else {}
    child = child_metrics if isinstance(child_metrics, Mapping) else {}
    metric_names = sorted(set(names) if names is not None else set(parent) | set(child))
    return tuple(
        compare_metric(
            name,
            parent[name] if name in parent else _MISSING,
            child[name] if name in child else _MISSING,
            tolerance=tolerance,
        )
        for name in metric_names
    )


def summarize_comparisons(comparisons: Iterable[MetricComparison]) -> Dict[str, Any]:


    ordered = tuple(comparisons)
    comparable = tuple(item for item in ordered if item.comparable)
    outcomes = {
        outcome: sum(item.outcome == outcome for item in comparable)
        for outcome in ("improved", "regressed", "unchanged")
    }
    if not comparable:
        overall = "no_directional_evidence"
    elif outcomes["improved"] == len(comparable):
        overall = "all_improved"
    elif outcomes["regressed"] == len(comparable):
        overall = "all_regressed"
    elif outcomes["unchanged"] == len(comparable):
        overall = "all_unchanged"
    else:
        overall = "mixed"

    return {
        "schema_version": METRIC_SEMANTICS_VERSION,
        "overall_outcome": overall,
        "comparable_count": len(comparable),
        "non_comparable_count": len(ordered) - len(comparable),
        "outcome_counts": outcomes,
        "raw_deltas": {
            item.metric: item.raw_delta
            for item in comparable
            if item.raw_delta is not None
        },
        "directional_deltas": {
            item.metric: item.improvement_delta
            for item in comparable
            if item.improvement_delta is not None
        },
        "comparisons": {item.metric: item.to_dict() for item in ordered},
    }


observe = observe_metric
compare = compare_metric
summarize_metric_comparisons = summarize_comparisons


__all__ = [
    "METRIC_SEMANTICS_VERSION",
    "METRIC_REGISTRY",
    "MetricSpec",
    "MetricObservation",
    "MetricComparison",
    "get_metric_spec",
    "observe_metric",
    "observe",
    "compare_metric",
    "compare",
    "compare_metrics",
    "summarize_comparisons",
    "summarize_metric_comparisons",
]
