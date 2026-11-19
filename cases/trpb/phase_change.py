"""Shared frozen AST transitions for the TrpB closed-loop benchmark."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_phase_b_ast(
    reference_ast: Mapping[str, Any], runner: Any, allowed: Sequence[int]
) -> tuple[dict[str, Any], dict[str, float]]:
    """Apply a deterministic local structural patch and a two-edge functional remap."""
    structural_patch, achieved = runner.corrupt_ast(reference_ast, 0.25, 2026080507)
    functional = []
    for node in reference_ast["functional_ast"]["nodes"]:
        children = list(node["mapped_structural_nodes"])
        if node["id"] == "peripheral_tuning":
            children = ["segment_104_108", "core_left_182_186"]
        elif node["id"] == "core_chemistry":
            children = ["segment_117_119", "core_right_227_231"]
        functional.append({"id": str(node["id"]), "mapped_structural_nodes": children})
    phase_b = runner.normalize_ast({
        "structural_nodes": [
            {"id": str(node["id"]), "positions_1based": list(node["positions_1based"])}
            for node in structural_patch["structural_ast"]["nodes"]
        ],
        "functional_nodes": functional,
    }, allowed)
    before_runtime = runner.ast_runtime(reference_ast)
    after_runtime = runner.ast_runtime(phase_b)
    structural_locality = sum(
        before_runtime["position_to_structural"][position] != after_runtime["position_to_structural"][position]
        for position in allowed
    ) / len(allowed)
    structural_ids = before_runtime["structural_ids"]
    functional_locality = sum(
        before_runtime["structural_to_functional"][node] != after_runtime["structural_to_functional"][node]
        for node in structural_ids
    ) / len(structural_ids)
    return phase_b, {
        "structural_assignment_fraction_changed": float(structural_locality),
        "functional_mapping_fraction_changed": float(functional_locality),
        "structural_corruption_generator_achieved": float(achieved),
    }
