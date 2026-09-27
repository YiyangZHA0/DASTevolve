

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from astevolve.evaluation.contracts import EvaluatorContext, ScoreTerm
from astevolve.evaluation.plugins.registry import (
    EvaluatorPluginSpec,
    register_plugin,
)


def _sequences(output: Mapping[str, Any]) -> Dict[str, str]:
    raw = output.get("seqs")
    if not isinstance(raw, Mapping):
        raw = output.get("best_seqs")
    if not isinstance(raw, Mapping):
        return {}
    return {str(chain): str(sequence) for chain, sequence in raw.items()}


def evaluate_candidate(
    out: Mapping[str, Any],
    *,
    compiled: Optional[Mapping[str, Any]] = None,
    design_state: Optional[Mapping[str, Any]] = None,
    masks: Optional[Mapping[str, Any]] = None,
    template_seqs: Optional[Mapping[str, str]] = None,
    fixed_residues: Optional[Mapping[str, Mapping[Any, str]]] = None,
    score_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:


    _ = compiled, masks, score_config
    state = design_state if isinstance(design_state, Mapping) else {}
    templates = template_seqs if isinstance(template_seqs, Mapping) else {}
    fixed = fixed_residues if isinstance(fixed_residues, Mapping) else {}
    sequences = _sequences(out)
    reasons = []

    for chain_id, template in sorted(templates.items(), key=lambda item: str(item[0])):
        chain = str(chain_id)
        if chain not in sequences:
            reasons.append(f"missing_chain:{chain}")
        elif len(sequences[chain]) != len(str(template)):
            reasons.append(f"length_mismatch:{chain}")

    for chain_id, assignments in sorted(fixed.items(), key=lambda item: str(item[0])):
        chain = str(chain_id)
        sequence = sequences.get(chain, "")
        if not isinstance(assignments, Mapping):
            continue
        for raw_position, expected in sorted(
            assignments.items(), key=lambda item: int(item[0])
        ):
            position = int(raw_position)
            actual = sequence[position] if 0 <= position < len(sequence) else None
            if actual != str(expected):
                reasons.append(f"fixed_residue_modified:{chain}:{position}")

    evaluator_config = state.get("synthetic_evaluator", {})
    if not isinstance(evaluator_config, Mapping):
        evaluator_config = {}
    binder_chain = str(evaluator_config.get("binder_chain_id") or "B")
    desired = evaluator_config.get("desired_residues", {})
    if not isinstance(desired, Mapping):
        desired = {}
    binder_sequence = sequences.get(binder_chain, "")
    matches = 0
    goal_rows = []
    for raw_position, expected in sorted(
        desired.items(), key=lambda item: int(item[0])
    ):
        position = int(raw_position)
        actual = (
            binder_sequence[position]
            if 0 <= position < len(binder_sequence)
            else None
        )
        matched = actual == str(expected)
        matches += int(matched)
        goal_rows.append(
            {
                "chain_id": binder_chain,
                "position": position,
                "expected": str(expected),
                "actual": actual,
                "matched": matched,
            }
        )

    soft_score = float(matches / len(desired)) if desired else 0.0
    reasons = sorted(set(reasons))
    passed = not reasons
    normalized = soft_score if passed else 0.0
    terms = [
        {
            "provider": "demo_gpu",
            "state": "positive",
            "name": "synthetic_goal_match",
            "category": "synthetic_objective",
            "score": soft_score,
            "weight": 1.0,
            "available": True,
            "details": {
                "matches": matches,
                "total": len(desired),
                "positions": goal_rows,
            },
            "warnings": [],
        },
        {
            "provider": "demo_gpu",
            "state": "preserve",
            "name": "fixed_residue_integrity",
            "category": "synthetic_guardrail",
            "score": 1.0 if passed else 0.0,
            "weight": 1.0,
            "available": True,
            "details": {"violations": reasons},
            "warnings": [],
        },
    ]
    return {
        "schema_version": "ast_evaluator_report_v1",
        "normalized_score": normalized,
        "soft_score": soft_score,
        "loss": 1.0 - normalized,
        "hard_gate_pass": passed,
        "disqualification_reasons": reasons,
        "gate_status": {
            "passed": passed,
            "hard_gate_pass": passed,
            "hard_failures": reasons,
            "disqualification_reasons": reasons,
        },
        "terms": terms,
        "weakest_terms": sorted(terms, key=lambda term: float(term["score"])),
        "warnings": [],
    }


class DemoGPUPlugin:


    name = "demo_gpu"

    def score_terms(self, context: EvaluatorContext) -> list[ScoreTerm]:
        report = evaluate_candidate(
            context.out,
            compiled=context.compiled,
            design_state=context.design_state,
            masks=context.masks,
            template_seqs=context.template_seqs,
            fixed_residues=context.fixed_residues,
            score_config=context.score_config,
        )
        configured_weights = context.score_config.get("evaluator_weights", {})
        if not isinstance(configured_weights, Mapping):
            configured_weights = {}
        terms: list[ScoreTerm] = []
        for raw in report.get("terms", []):
            if not isinstance(raw, Mapping):
                continue
            name = str(raw.get("name") or "unnamed")
            details = dict(raw.get("details") or {})
            if raw.get("state") is not None:
                details.setdefault("state", str(raw["state"]))
            if name == "fixed_residue_integrity":
                details.update(
                    {
                        "hard_gate": True,
                        "hard_gate_min_score": 1.0,
                        "dimension": "correctness",
                    }
                )
            elif name == "synthetic_goal_match":
                details.setdefault("dimension", "correctness")
            weight = configured_weights.get(
                f"eval_{name}", raw.get("weight", 1.0)
            )
            terms.append(
                ScoreTerm(
                    name=name,
                    category=str(raw.get("category") or "synthetic"),
                    score=float(raw.get("score", 0.0) or 0.0),
                    weight=float(weight),
                    details=details,
                    warnings=[str(item) for item in raw.get("warnings", [])],
                    backend=self.name,
                    available=bool(raw.get("available", True)),
                )
            )
        return terms


def register_demo_gpu_plugin() -> None:


    register_plugin(
        EvaluatorPluginSpec(
            name="demo_gpu",
            factory="cases.demo_case.demo_evaluator:DemoGPUPlugin",
            weight_fields=(
                "eval_fixed_residue_integrity",
                "eval_synthetic_goal_match",
            ),
        ),
        replace=True,
    )


__all__ = [
    "DemoGPUPlugin",
    "evaluate_candidate",
    "register_demo_gpu_plugin",
]
