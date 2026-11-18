

from __future__ import annotations

from typing import Any, Dict, Mapping

from astevolve.evaluation.selection import normalize_gate_payload


def _stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _write_canonical_gate(report: Dict[str, Any], *, passed: bool, reasons: list[str]) -> None:
    reasons = _stable_unique(reasons)
    gate_status = dict(report.get("gate_status", {}) or {})
    gate_status.update(
        {
            "hard_gate_pass": passed,
            "passed": passed,
            "disqualification_reasons": reasons,
            "hard_failures": reasons,
        }
    )
    report["gate_status"] = gate_status
    report["hard_gate_pass"] = passed
    report["disqualification_reasons"] = reasons


def merge_inner_semantic_audit(
    evaluator_report: Mapping[str, Any],
    semantic_audit: Any,
) -> Dict[str, Any]:


    report = dict(evaluator_report)
    evaluator_gate = normalize_gate_payload(report)
    evaluator_passed = bool(evaluator_gate["passed"])
    evaluator_reasons = list(evaluator_gate["reasons"])
    if not isinstance(semantic_audit, dict):
        _write_canonical_gate(
            report,
            passed=evaluator_passed,
            reasons=evaluator_reasons,
        )
        return report

    audit_gate = normalize_gate_payload(semantic_audit)
    audit_passed = bool(audit_gate["passed"])
    report["inner_loop_semantic_audit_hard_gate_pass"] = audit_passed

    reasons = [
        f"inner_loop_{reason}"
        for reason in audit_gate["reasons"]
        if str(reason).strip()
    ]
    if not audit_passed and not reasons:
        reasons = ["inner_loop_semantic_audit_failed"]
    combined_passed = evaluator_passed and audit_passed
    merged_reasons = evaluator_reasons + ([] if audit_passed else reasons)
    _write_canonical_gate(
        report,
        passed=combined_passed,
        reasons=merged_reasons,
    )
    if not audit_passed:
        report["normalized_score"] = 0.0
        report["loss"] = 1.0
        report["warnings"] = sorted(
            set(
                list(report.get("warnings", []) or [])
                + ["inner_loop_semantic_audit: final selected sequence failed semantic hard gate"]
            )
        )
    return report
