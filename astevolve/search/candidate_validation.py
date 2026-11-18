

from __future__ import annotations

from functools import lru_cache
import importlib
from typing import Any, Dict, List, Optional

from astevolve.search.config import SAConfig
from astevolve.search.proposal_engine import (
    _semantic_coverage_hard_enabled,
    _semantic_coverage_report,
    _semantic_min_mutations,
    _unique_strings,
)
from astevolve.semantic_graph import (
    annotate_mutations_with_residue_map,
    build_residue_semantic_map,
    summarize_mutation_semantics,
    summarize_residue_semantic_map,
)


def _sequence_diffs(
    seqs: Dict[str, str],
    template_seqs: Optional[Dict[str, str]],
) -> Dict[str, List[Dict[str, Any]]]:


    if not template_seqs:
        return {}
    diffs: Dict[str, List[Dict[str, Any]]] = {}
    for cid, seq in (seqs or {}).items():
        ref = str(template_seqs.get(cid, "") or "")
        chain_diffs: List[Dict[str, Any]] = []
        for i, aa in enumerate(str(seq or "")):
            old = ref[i] if i < len(ref) else None
            if old is not None and old != aa:
                chain_diffs.append({"position": int(i), "from": old, "to": aa})
        if len(seq or "") != len(ref):
            chain_diffs.append({
                "position": int(min(len(seq or ""), len(ref))),
                "from": "length",
                "to": f"{len(ref)}->{len(seq or '')}",
            })
        if chain_diffs:
            diffs[cid] = chain_diffs
    return diffs


def _fixed_residue_violations(
    seqs: Dict[str, str],
    fixed_residues: Optional[Dict[str, Dict[int, str]]],
) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    if not fixed_residues:
        return violations
    for cid, fixed in fixed_residues.items():
        seq = str((seqs or {}).get(cid, "") or "")
        for pos, expected in (fixed or {}).items():
            i = int(pos)
            actual = seq[i] if 0 <= i < len(seq) else None
            if actual != expected:
                violations.append({
                    "chain_id": cid,
                    "position": i,
                    "expected": expected,
                    "actual": actual,
                })
    return violations


def _residue_mutation_contract_status(
    seqs: Dict[str, str],
    template_seqs: Optional[Dict[str, str]],
    cfg: SAConfig,
) -> Dict[str, Any]:


    contract = dict(getattr(cfg, "residue_mutation_contract", {}) or {})
    if not contract:
        return {
            "schema_version": "astevolve.residue_mutation_contract_validation.v1",
            "enabled": False,
            "pass": True,
            "checked_mutation_count": 0,
            "violation_count": 0,
            "violations": [],
        }
    if template_seqs is None:
        violation = {
            "code": "template_sequences_unavailable",
            "chain_id": None,
            "position": None,
            "from": None,
            "to": None,
            "allowed_residues": [],
        }
        return {
            "schema_version": "astevolve.residue_mutation_contract_validation.v1",
            "enabled": True,
            "pass": False,
            "checked_mutation_count": 0,
            "violation_count": 1,
            "violations": [violation],
        }

    candidates = {str(chain): str(sequence) for chain, sequence in (seqs or {}).items()}
    templates = {
        str(chain): str(sequence) for chain, sequence in template_seqs.items()
    }
    violations: List[Dict[str, Any]] = []
    checked_mutations = 0
    for chain in sorted(set(templates) | set(candidates)):
        if chain not in candidates:
            violations.append({
                "code": "candidate_chain_missing",
                "chain_id": chain,
                "position": None,
                "from": templates[chain],
                "to": None,
                "allowed_residues": [],
            })
            continue
        if chain not in templates:
            violations.append({
                "code": "candidate_chain_extra",
                "chain_id": chain,
                "position": None,
                "from": None,
                "to": candidates[chain],
                "allowed_residues": [],
            })
            continue
        before = templates[chain]
        after = candidates[chain]
        if len(before) != len(after):
            violations.append({
                "code": "sequence_length_changed",
                "chain_id": chain,
                "position": min(len(before), len(after)),
                "from": len(before),
                "to": len(after),
                "allowed_residues": [],
            })
        chain_contract = contract.get(chain)
        for position, (old_residue, new_residue) in enumerate(zip(before, after)):
            if old_residue == new_residue:
                continue
            checked_mutations += 1
            if chain_contract is None:
                code = "chain_not_declared"
                allowed: List[str] = []
            elif position not in chain_contract:
                code = "position_not_declared"
                allowed = []
            else:
                allowed = list(chain_contract[position])
                code = "residue_not_allowed" if new_residue not in allowed else ""
            if code:
                violations.append({
                    "code": code,
                    "chain_id": chain,
                    "position": int(position),
                    "from": old_residue,
                    "to": new_residue,
                    "allowed_residues": allowed,
                })
    return {
        "schema_version": "astevolve.residue_mutation_contract_validation.v1",
        "enabled": True,
        "pass": not violations,
        "checked_mutation_count": int(checked_mutations),
        "violation_count": len(violations),
        "violations": violations,
    }


def _iter_compiled_segments(compiled: Dict[str, Any]) -> List[Any]:
    segments: List[Any] = []
    for key in ("segments", "compiled_segments", "node_segments"):
        value = compiled.get(key)
        if isinstance(value, list):
            segments.extend(value)
    bp = compiled.get("blueprint")
    if bp is not None:
        for attr in ("segments", "nodes"):
            value = getattr(bp, attr, None)
            if isinstance(value, list):
                segments.extend(value)
    return segments


def _segment_abs_positions(seg: Any) -> List[int]:
    out: List[int] = []
    for span in getattr(seg, "spans", []) or []:
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            continue
        start, end = int(span[0]), int(span[1])
        if end < start:
            start, end = end, start
        out.extend(range(start, end))
    return out


def _mutation_counts_by_node(
    seqs: Dict[str, str],
    template_seqs: Optional[Dict[str, str]],
    compiled: Dict[str, Any],
) -> Dict[str, Any]:
    diffs = _sequence_diffs(seqs, template_seqs)
    diff_sets = {cid: {int(d["position"]) for d in chain_diffs if isinstance(d.get("position"), int)} for cid, chain_diffs in diffs.items()}
    nodes: Dict[str, Dict[str, Any]] = {}
    assigned: Dict[str, set[int]] = {}
    for seg in _iter_compiled_segments(compiled):
        cid = getattr(seg, "chain_id", None)
        if cid is None or cid not in diff_sets:
            continue
        positions = set(_segment_abs_positions(seg))
        mutated = sorted(int(p) for p in (diff_sets[cid] & positions))
        if not mutated:
            continue
        name = str(getattr(seg, "name", "unknown_node"))
        node = nodes.setdefault(
            name,
            {
                "node": name,
                "node_kind": str(getattr(seg, "kind", "")),
                "chain_id": cid,
                "count": 0,
                "positions": [],
            },
        )
        node["count"] += len(mutated)
        node["positions"].extend(mutated)
        assigned.setdefault(cid, set()).update(mutated)
    unassigned = 0
    for cid, positions in diff_sets.items():
        unassigned += len(positions - assigned.get(cid, set()))
    for node in nodes.values():
        node["positions"] = sorted(set(int(x) for x in node["positions"]))
    return {
        "nodes": dict(sorted(nodes.items(), key=lambda item: (-item[1]["count"], item[0]))),
        "unassigned_mutations": int(unassigned),
    }


def _sequence_invariant_status(
    seqs: Dict[str, str],
    compiled: Dict[str, Any],
) -> Dict[str, Any]:


    state = compiled.get("_design_state") if isinstance(compiled, dict) else None
    constraints = state.get("design_constraints", {}) if isinstance(state, dict) else {}
    declarations = constraints.get("sequence_invariants", []) if isinstance(constraints, dict) else []
    if not isinstance(declarations, list):
        return {
            "schema_version": "declared_sequence_invariants.v1",
            "available": False,
            "pass": False,
            "checks": [],
            "configuration_error": "design_constraints.sequence_invariants must be a list",
        }
    segments = {
        str(getattr(seg, "name", "")): seg
        for seg in _iter_compiled_segments(compiled)
        if str(getattr(seg, "name", ""))
    }
    checks: List[Dict[str, Any]] = []
    for index, declaration in enumerate(declarations):
        if not isinstance(declaration, dict):
            continue
        invariant_id = str(declaration.get("id") or f"invariant_{index}")
        node_name = str(declaration.get("node") or "")
        segment = segments.get(node_name) if node_name else None
        cid = str(
            declaration.get("chain_id")
            or getattr(segment, "chain_id", "")
            or ""
        )
        if cid not in (seqs or {}):
            checks.append({
                "id": invariant_id,
                "chain_id": cid,
                "node": node_name or None,
                "positions": [],
                "observed": "",
                "allowed_sequences": [],
                "hard_gate": bool(declaration.get("hard_gate", True)),
                "pass": False,
                "reason": "declared chain is absent",
            })
            continue
        positions: List[int] = []
        raw_spans = declaration.get("spans", [])
        if isinstance(raw_spans, list):
            for raw_span in raw_spans:
                if isinstance(raw_span, (list, tuple)) and len(raw_span) == 2:
                    start, end = int(raw_span[0]), int(raw_span[1])
                    positions.extend(range(min(start, end), max(start, end)))
        if not positions and segment is not None:
            positions = _segment_abs_positions(segment)
        positions = sorted(set(positions))
        observed = "".join(str(seqs[cid])[p] for p in positions if 0 <= p < len(str(seqs[cid])))
        raw_allowed = declaration.get("allowed_sequences")
        if isinstance(raw_allowed, str):
            allowed = [raw_allowed]
        elif isinstance(raw_allowed, list):
            allowed = [str(value) for value in raw_allowed]
        elif declaration.get("expected_sequence") is not None:
            allowed = [str(declaration.get("expected_sequence"))]
        else:
            allowed = []
        checks.append({
            "id": invariant_id,
            "node": node_name or None,
            "chain_id": cid,
            "positions": positions,
            "observed": observed,
            "allowed_sequences": allowed,
            "hard_gate": bool(declaration.get("hard_gate", True)),
            "pass": bool(allowed) and observed in allowed,
            "reason": None if bool(allowed) and observed in allowed else "observed sequence is not declared as allowed",
        })
    if not checks:
        return {
            "schema_version": "declared_sequence_invariants.v1",
            "available": False,
            "pass": True,
            "checks": [],
        }
    return {
        "schema_version": "declared_sequence_invariants.v1",
        "available": True,
        "pass": all(
            bool(item.get("pass")) or not bool(item.get("hard_gate", True))
            for item in checks
        ),
        "checks": checks,
    }


@lru_cache(maxsize=16)
def _resolve_sequence_prefilter(spec: str) -> Any:
    module_name, function_name = str(spec).split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"sequence prefilter {spec!r} is not callable")
    return function


def _case_sequence_prefilter_status(
    seqs: Dict[str, str],
    template_seqs: Optional[Dict[str, str]],
    compiled: Dict[str, Any],
    cfg: SAConfig,
) -> Dict[str, Any]:
    spec = str(getattr(cfg, "sequence_prefilter_callable", "") or "").strip()
    if not spec:
        return {
            "schema_version": "astevolve.case_sequence_prefilter.v1",
            "enabled": False,
            "pass": True,
            "reasons": [],
        }
    try:
        result = _resolve_sequence_prefilter(spec)(
            seqs=seqs,
            template_seqs=template_seqs,
            compiled=compiled,
            config=dict(getattr(cfg, "sequence_prefilter_config", {}) or {}),
        )
        if not isinstance(result, dict):
            raise TypeError("sequence prefilter must return a dictionary")
        reasons = [str(value) for value in result.get("reasons", [])]
        return {
            **result,
            "enabled": True,
            "pass": bool(result.get("pass", False)) and not reasons,
            "reasons": reasons,
            "callable": spec,
        }
    except Exception as exc:
        return {
            "schema_version": "astevolve.case_sequence_prefilter.v1",
            "enabled": True,
            "pass": False,
            "reasons": ["case_sequence_prefilter_error"],
            "callable": spec,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _candidate_fast_filter(
    seqs: Dict[str, str],
    template_seqs: Optional[Dict[str, str]],
    fixed_residues: Optional[Dict[str, Dict[int, str]]],
    compiled: Dict[str, Any],
    cfg: SAConfig,
) -> Dict[str, Any]:


    diffs = _sequence_diffs(seqs, template_seqs)
    total_mutations = int(sum(len(v) for v in diffs.values()))
    fixed_violations = _fixed_residue_violations(seqs, fixed_residues)
    invariant_status = _sequence_invariant_status(seqs, compiled)
    contract_status = _residue_mutation_contract_status(seqs, template_seqs, cfg)
    case_prefilter = _case_sequence_prefilter_status(
        seqs,
        template_seqs,
        compiled,
        cfg,
    )
    design_action_status: Dict[str, Any] = {
        "schema_version": "astevolve.design_action_fast_validation.v1",
        "available": False,
        "pass": True,
        "error_code": None,
        "receipt": None,
    }
    raw_compiled_action = compiled.get("compiled_design_action")
    if raw_compiled_action is not None:
        from engine.design_action_compiler import (
            DesignActionCompileError,
            validate_candidate_against_compiled_action,
        )

        design_action_status["available"] = True
        try:
            compiled_hash = str(
                raw_compiled_action.get("compiled_design_action_hash") or ""
            )
            root_sequences = raw_compiled_action.get(
                "reconciled_root_sequences"
            )
            is_exact_root = isinstance(root_sequences, dict) and {
                str(chain): str(sequence)
                for chain, sequence in seqs.items()
            } == {
                str(chain): str(sequence)
                for chain, sequence in root_sequences.items()
            }
            design_action_status["receipt"] = (
                validate_candidate_against_compiled_action(
                    raw_compiled_action,
                    seqs,
                    compiled_hash,
                    allow_reconciled_root=is_exact_root,
                )
            )
        except DesignActionCompileError as error:
            design_action_status["pass"] = False
            design_action_status["error_code"] = error.code
            design_action_status["detail"] = error.detail
        except (TypeError, ValueError) as error:
            design_action_status["pass"] = False
            design_action_status["error_code"] = (
                "compiled_design_action_invalid"
            )
            design_action_status["detail"] = str(error)
    optional_reasons: List[str] = []
    if fixed_violations:
        optional_reasons.append("fixed_residue_modified")
    if int(cfg.max_total_mutations or 0) > 0 and total_mutations > int(cfg.max_total_mutations):
        optional_reasons.append("max_total_mutations_exceeded")
    if invariant_status.get("available") and not invariant_status.get("pass"):
        optional_reasons.append("declared_sequence_invariant_failed")
    mandatory_reasons = (
        [] if contract_status.get("pass")
        else ["residue_mutation_contract_violation"]
    )
    if not case_prefilter.get("pass"):
        mandatory_reasons.extend(
            "case_sequence_prefilter:" + str(reason)
            for reason in case_prefilter.get("reasons", [])
        )
    if not design_action_status.get("pass"):
        mandatory_reasons.append(
            "design_action_contract_violation:"
            + str(design_action_status.get("error_code") or "invalid")
        )
    reasons = mandatory_reasons + (
        optional_reasons if bool(cfg.fast_filter_enabled) else []
    )
    search_expandable = bool(case_prefilter.get("search_expandable", False)) and bool(
        contract_status.get("pass")
    ) and bool(design_action_status.get("pass")) and not optional_reasons
    return {
        "enabled": bool(cfg.fast_filter_enabled),
        "pass": bool(contract_status.get("pass")) and bool(
            design_action_status.get("pass")
        ) and bool(case_prefilter.get("pass")) and (
            (not bool(cfg.fast_filter_enabled)) or not optional_reasons
        ),
        "reasons": reasons,
        "search_expandable": search_expandable,
        "search_progress": float(case_prefilter.get("search_progress", 0.0) or 0.0),
        "total_mutations": total_mutations,
        "max_total_mutations": int(cfg.max_total_mutations or 0),
        "fixed_residue_violation_count": len(fixed_violations),
        "fixed_residue_violations": fixed_violations[:20],
        "sequence_invariants": invariant_status,
        "residue_mutation_contract": contract_status,
        "residue_mutation_contract_pass": bool(contract_status.get("pass")),
        "residue_mutation_contract_violation_count": int(
            contract_status.get("violation_count", 0) or 0
        ),
        "design_action_validation": design_action_status,
        "case_sequence_prefilter": case_prefilter,
    }


def _primary_design_nodes_from_policy(
    cfg: SAConfig,
    compiled: Optional[Dict[str, Any]] = None,
) -> List[str]:
    primary: List[str] = []
    design_state = (
        compiled.get("_design_state")
        if isinstance(compiled, dict)
        else None
    )
    design_points = (
        design_state.get("design_points")
        if isinstance(design_state, dict)
        else None
    )
    if isinstance(design_points, dict):
        primary.extend(
            _unique_strings(design_points.get("primary_design_nodes", []))
        )
    for node_name, policy in (cfg.node_edit_policies or {}).items():
        if not isinstance(policy, dict):
            continue
        action = str(policy.get("edit_contract_action", "") or "").lower()
        active = (
            action in {"optimize_node", "expand_edit_scope", "increase_negative_design_weight"}
            or bool(policy.get("primary"))
            or str(policy.get("selection_role") or "").lower() == "primary"
        )
        frozen = action in {"freeze_node", "repair_node"} and int(policy.get("max_mutations_per_step", 1) or 0) <= 0
        if active and not frozen:
            primary.append(str(node_name))
    return sorted(set(primary))


def _formal_layered_coverage_enabled(cfg: SAConfig) -> bool:


    return (
        str(getattr(cfg, "structure_shortlist_policy", "legacy_diverse") or "")
        .strip()
        .lower()
        == "formal_layered_novel"
    )


def _semantic_active_and_anchor_nodes(
    cfg: SAConfig,
) -> tuple[List[str], List[str], str]:


    required = _unique_strings(getattr(cfg, "semantic_required_nodes", []))
    declared_active = _unique_strings(getattr(cfg, "semantic_active_nodes", []))
    active = declared_active or list(required)
    declared_anchors = _unique_strings(getattr(cfg, "semantic_anchor_nodes", []))
    if declared_anchors:
        return active, declared_anchors, "semantic_anchor_nodes"
    if _formal_layered_coverage_enabled(cfg) and required:
        return active, list(required), "semantic_required_nodes_compatibility_fallback"
    return active, [], "none"


def _required_final_mutation_coverage(
    seqs: Dict[str, str],
    template_seqs: Optional[Dict[str, str]],
    compiled: Dict[str, Any],
    cfg: SAConfig,
) -> Dict[str, Any]:


    required = _unique_strings(getattr(cfg, "semantic_required_nodes", []))
    active, anchors, anchor_source = _semantic_active_and_anchor_nodes(cfg)
    layered = _formal_layered_coverage_enabled(cfg)
    min_mutations = _semantic_min_mutations(cfg)
    if not required and not active and not anchors:
        return {
            "schema_version": "ast_required_final_mutation_coverage_v1",
            "pass": True,
            "required_nodes": [],
            "min_mutations": int(min_mutations),
            "mutations_by_node": {},
            "missing_required_nodes_by_mutation": [],
        }
    counts = _mutation_counts_by_node(seqs, template_seqs, compiled)
    mutated_nodes = counts.get("nodes", {}) if isinstance(counts, dict) else {}
    all_audit_nodes = _unique_strings([*required, *active, *anchors])
    all_counts = {
        node: int((mutated_nodes.get(node, {}) or {}).get("count", 0) or 0)
        for node in all_audit_nodes
    }
    by_node = {node: int(all_counts.get(node, 0)) for node in required}
    active_by_node = {node: int(all_counts.get(node, 0)) for node in active}
    anchor_by_node = {node: int(all_counts.get(node, 0)) for node in anchors}
    missing = [
        node
        for node, count in by_node.items()
        if min_mutations > 0 and int(count) < min_mutations
    ]
    missing_active = [
        node
        for node, count in active_by_node.items()
        if min_mutations > 0 and int(count) < min_mutations
    ]
    missing_anchors = [
        node
        for node, count in anchor_by_node.items()
        if min_mutations > 0 and int(count) < min_mutations
    ]
    actual_pass = bool(not missing_active) if layered else bool(not missing)
    report = {
        "schema_version": (
            "ast_required_final_mutation_coverage_v2"
            if layered
            else "ast_required_final_mutation_coverage_v1"
        ),


        "pass": True if layered else actual_pass,
        "coverage_pass": actual_pass,
        "required_nodes": required,
        "min_mutations": int(min_mutations),
        "mutations_by_node": by_node,
        "missing_required_nodes_by_mutation": missing,
        "active_nodes": active,
        "active_mutations_by_node": active_by_node,
        "missing_active_nodes_by_mutation": missing_active,
        "anchor_nodes": anchors,
        "anchor_source": anchor_source,
        "anchor_mutations_by_node": anchor_by_node,
        "missing_anchor_nodes_by_mutation": missing_anchors,
        "anchor_coverage_pass": bool(not missing_anchors),
        "coverage_scope": "shortlist_set" if layered else "individual_sequence",
        "individual_hard_gate_enabled": bool(not layered),
        "individual_hard_gate_pass": True if layered else actual_pass,
    }
    return report


def _semantic_prefilter_structure_candidates(
    candidates: List[Dict[str, Any]],
    template_seqs: Optional[Dict[str, str]],
    compiled: Dict[str, Any],
    cfg: SAConfig,
) -> List[Dict[str, Any]]:
    if not candidates:
        return candidates
    required = _unique_strings(getattr(cfg, "semantic_required_nodes", []))
    active, anchors, _anchor_source = _semantic_active_and_anchor_nodes(cfg)
    if not required and not active and not anchors:
        return candidates
    passing: List[Dict[str, Any]] = []
    for item in candidates:
        final_coverage = _required_final_mutation_coverage(
            item.get("seqs", {}),
            template_seqs,
            compiled,
            cfg,
        )
        item["semantic_final_mutation_coverage"] = final_coverage
        if bool(final_coverage.get("pass", True)):
            passing.append(item)
    if (
        _semantic_coverage_hard_enabled(cfg)
        and not _formal_layered_coverage_enabled(cfg)
    ):
        return passing or candidates


    return candidates


def _inner_loop_semantic_audit(
    seqs: Dict[str, str],
    template_seqs: Optional[Dict[str, str]],
    fixed_residues: Optional[Dict[str, Dict[int, str]]],
    compiled: Dict[str, Any],
    cfg: SAConfig,
    history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:


    filter_report = _candidate_fast_filter(seqs, template_seqs, fixed_residues, compiled, cfg)
    counts = _mutation_counts_by_node(seqs, template_seqs, compiled)
    design_state = compiled.get("_design_state") if isinstance(compiled, dict) else {}
    declared_primary = _primary_design_nodes_from_policy(cfg, compiled)
    mutated_nodes = counts.get("nodes", {})
    total = int(filter_report.get("total_mutations", 0) or 0)
    visit_counts = dict((history or {}).get("node_visit_counts", {}) or {})
    total_visits = int(sum(int(v or 0) for v in visit_counts.values()))
    diffs = _sequence_diffs(seqs, template_seqs)
    residue_map = build_residue_semantic_map(
        design_state if isinstance(design_state, dict) else {},
        compiled,
        sequences=seqs,
        case_sheet=(design_state or {}).get("_case_sheet", {}) if isinstance(design_state, dict) else {},
    )
    mutation_annotations = annotate_mutations_with_residue_map(diffs, residue_map)
    mutation_semantic_summary = summarize_mutation_semantics(mutation_annotations)
    residue_map_summary = summarize_residue_semantic_map(residue_map, limit=40)
    semantic_coverage = _semantic_coverage_report(cfg, history or {})
    active_nodes, anchor_nodes, anchor_source = _semantic_active_and_anchor_nodes(cfg)


    primary = list(active_nodes or declared_primary)
    primary_mutations = sum(
        int(mutated_nodes.get(node, {}).get("count", 0) or 0)
        for node in primary
    )
    primary_visits = int(
        sum(int(visit_counts.get(node, 0) or 0) for node in primary)
    )
    primary_node_touch = {
        node: int((mutated_nodes.get(node, {}) or {}).get("count", 0) or 0)
        for node in primary
    }
    layered = _formal_layered_coverage_enabled(cfg)
    required_nodes = list(semantic_coverage.get("required_nodes", []) or [])
    min_required_mutations = int(semantic_coverage.get("min_mutations", 1) or 0)
    final_required_mutations = {
        node: int((mutated_nodes.get(node, {}) or {}).get("count", 0) or 0)
        for node in required_nodes
    }
    final_missing_required_nodes = [
        node
        for node, count in final_required_mutations.items()
        if min_required_mutations > 0 and int(count) < min_required_mutations
    ]
    final_active_mutations = {
        node: int((mutated_nodes.get(node, {}) or {}).get("count", 0) or 0)
        for node in active_nodes
    }
    final_anchor_mutations = {
        node: int((mutated_nodes.get(node, {}) or {}).get("count", 0) or 0)
        for node in anchor_nodes
    }
    final_missing_active_nodes = [
        node
        for node, count in final_active_mutations.items()
        if min_required_mutations > 0 and int(count) < min_required_mutations
    ]
    final_missing_anchor_nodes = [
        node
        for node, count in final_anchor_mutations.items()
        if min_required_mutations > 0 and int(count) < min_required_mutations
    ]
    coverage_pass = bool(semantic_coverage.get("pass", True)) and not final_missing_required_nodes
    hard_reasons = list(filter_report.get("reasons", []) or [])
    raw_joint_completion = (history or {}).get("joint_coverage_completion", {})
    joint_completion = (
        dict(raw_joint_completion)
        if isinstance(raw_joint_completion, dict)
        else {}
    )
    joint_completion_gate_failed = bool(joint_completion.get("enabled")) and not bool(
        joint_completion.get("pass", False)
    )
    individual_semantic_gate_enabled = bool(
        _semantic_coverage_hard_enabled(cfg) and not layered
    )
    if individual_semantic_gate_enabled and not coverage_pass:
        if semantic_coverage.get("missing_required_nodes_by_visit"):
            hard_reasons.append("semantic_required_node_visit_missing")
        if semantic_coverage.get("missing_required_nodes_by_mutation"):
            hard_reasons.append("semantic_required_node_mutation_missing")
        if semantic_coverage.get("unavailable_required_nodes"):
            hard_reasons.append("semantic_required_node_unavailable")
        if final_missing_required_nodes:
            hard_reasons.append("semantic_required_node_final_mutation_missing")
    if joint_completion_gate_failed:
        status = str(joint_completion.get("status") or "underfilled")
        hard_reasons.append(
            "joint_coverage_completion_impossible"
            if status == "impossible"
            else "joint_coverage_completion_underfilled"
        )
    return {
        "schema_version": "ast_inner_loop_semantic_audit_v1",
        "hard_gate_pass": bool(filter_report.get("pass")) and (
            coverage_pass or not individual_semantic_gate_enabled
        ) and not joint_completion_gate_failed,
        "hard_gate_reasons": hard_reasons,
        "primary_design_nodes": primary,
        "primary_design_node_source": (
            "semantic_active_nodes" if active_nodes else "case_primary_design_nodes"
        ),
        "declared_primary_design_nodes": declared_primary,
        "primary_design_node_touch_rate": float(primary_mutations / total) if total > 0 else 0.0,
        "primary_design_node_mutations": int(primary_mutations),
        "primary_design_node_mutations_by_node": primary_node_touch,
        "semantic_required_node_coverage": semantic_coverage,
        "semantic_required_node_coverage_pass": bool(coverage_pass),
        "semantic_required_nodes": required_nodes,
        "semantic_active_nodes": active_nodes,
        "semantic_anchor_nodes": anchor_nodes,
        "semantic_anchor_source": anchor_source,
        "semantic_final_coverage_scope": (
            "shortlist_set" if layered else "individual_sequence"
        ),
        "semantic_individual_hard_gate_enabled": individual_semantic_gate_enabled,
        "joint_coverage_completion": joint_completion,
        "joint_coverage_completion_pass": bool(
            not joint_completion_gate_failed
        ),
        "final_required_node_mutations_by_node": final_required_mutations,
        "final_missing_required_nodes_by_mutation": final_missing_required_nodes,
        "final_active_node_mutations_by_node": final_active_mutations,
        "final_missing_active_nodes_by_mutation": final_missing_active_nodes,
        "final_anchor_node_mutations_by_node": final_anchor_mutations,
        "final_missing_anchor_nodes_by_mutation": final_missing_anchor_nodes,
        "proposal_primary_node_touch_rate": float(primary_visits / total_visits) if total_visits > 0 else 0.0,
        "proposal_primary_node_visits": int(primary_visits),
        "proposal_total_node_visits": int(total_visits),
        "total_mutations": total,
        "max_total_mutations": int(cfg.max_total_mutations or 0),
        "mutation_budget_pass": bool(
            int(cfg.max_total_mutations or 0) <= 0 or total <= int(cfg.max_total_mutations or 0)
        ),
        "mutations_per_node": mutated_nodes,
        "unassigned_mutations": int(counts.get("unassigned_mutations", 0) or 0),
        "fixed_residue_violation_count": int(filter_report.get("fixed_residue_violation_count", 0) or 0),
        "fixed_residue_violations": list(filter_report.get("fixed_residue_violations", []) or []),
        "sequence_invariants": filter_report.get("sequence_invariants", {}),
        "residue_mutation_contract": filter_report.get(
            "residue_mutation_contract", {}
        ),
        "residue_mutation_contract_pass": bool(
            filter_report.get("residue_mutation_contract_pass", True)
        ),
        "residue_mutation_contract_violation_count": int(
            filter_report.get("residue_mutation_contract_violation_count", 0) or 0
        ),
        "mutation_semantic_annotations": mutation_annotations[:80],
        "mutation_semantic_summary": mutation_semantic_summary,
        "residue_semantic_map_summary": residue_map_summary,
    }
