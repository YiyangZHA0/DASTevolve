

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from astevolve.evaluation.contracts import ScoreTerm
from astevolve.evaluation.geometry.fold import chain_break_count, rg_score
from astevolve.evaluation.geometry.interface import atom_geometry_counts, interface_interaction_counts
from astevolve.evaluation.geometry.rmsd import (
    global_ca_rmsd,
    preserved_node_ca_rmsd,
    resolve_reference_path,
    rmsd_score,
)
from astevolve.evaluation.support import (
    clamp01,
    evaluator_weight,
    mean,
    safe_float,
    score_at_least,
    score_at_most,
    score_from_100,
)
from astevolve.evaluation.views import (
    fixed_lookup,
    mask_allows,
    multistate_score,
    node_names_from_design_state,
    node_plddt_map,
    node_value,
    objective_items,
    segments_by_name,
    sequence_changes,
)
from astevolve.metrics.structure import metric_value


def add_term(terms: List[ScoreTerm], term: ScoreTerm) -> None:


    terms.append(term)


def confidence_terms(out: Mapping[str, Any], structure: Mapping[str, Any], terms: List[ScoreTerm], score_config: Mapping[str, Any]) -> None:


    plddt = metric_value(dict(structure), "plddt", default=out.get("chai_plddt") or 0.0)
    ptm = metric_value(dict(structure), "ptm", default=0.0)
    iptm = metric_value(dict(structure), "iptm", default=0.0)
    interface_plddt = metric_value(dict(structure), "interface_plddt_mean", default=0.0)
    node_min = metric_value(dict(structure), "node_plddt_min", default=0.0)
    add_term(terms, ScoreTerm("global_plddt", "confidence", score_from_100(plddt), evaluator_weight(score_config, "eval_global_plddt", 1.0), {"plddt": plddt}))
    add_term(terms, ScoreTerm("global_ptm", "confidence", clamp01(ptm), evaluator_weight(score_config, "eval_ptm", 0.35), {"ptm": ptm}))
    add_term(terms, ScoreTerm("global_iptm", "confidence", clamp01(iptm), evaluator_weight(score_config, "eval_iptm", 0.45), {"iptm": iptm}))
    add_term(terms, ScoreTerm("interface_plddt", "confidence", score_from_100(interface_plddt), evaluator_weight(score_config, "eval_interface_plddt", 0.7), {"interface_plddt_mean": interface_plddt}))
    add_term(terms, ScoreTerm("node_confidence_floor", "confidence", score_from_100(node_min), evaluator_weight(score_config, "eval_node_floor", 0.65), {"node_plddt_min": node_min}))

    state_scores = []
    target_state_scores: List[Tuple[str, float]] = []
    for state in structure.get("states", []) or []:
        if not isinstance(state, Mapping):
            continue
        summary = state.get("structure_metrics", {}) if isinstance(state.get("structure_metrics"), Mapping) else {}
        value = metric_value(summary, "plddt", default=0.0)
        if value:
            score = score_from_100(value)
            state_scores.append(score)
            state_name = str(state.get("name") or "")
            state_role = str(state.get("role") or "")
            blob = f"{state_name} {state_role}".lower()
            if any(token in blob for token in ("target", "positive", "bound", "holo")):
                target_state_scores.append((state_name, score))
    if state_scores:
        add_term(terms, ScoreTerm("state_confidence_min", "confidence", min(state_scores), evaluator_weight(score_config, "eval_state_confidence", 0.6), {"state_count": len(state_scores), "state_confidence_min": min(state_scores)}))
    if target_state_scores:
        minimum_name, minimum_score = min(target_state_scores, key=lambda item: item[1])
        add_term(terms, ScoreTerm("target_complex_confidence_floor", "confidence", minimum_score, evaluator_weight(score_config, "eval_target_confidence_floor", 0.95), {"target_state_scores": dict(target_state_scores), "weakest_target_state": minimum_name}, warnings=["target complex confidence is extremely low"] if minimum_score < 0.35 else []))


def fold_validity_terms(structure: Mapping[str, Any], terms: List[ScoreTerm], score_config: Mapping[str, Any]) -> None:


    clash_count = metric_value(dict(structure), "clash_count", default=0.0)
    clash_bad = safe_float(score_config.get("evaluator_clash_bad"), 250.0) or 250.0
    add_term(terms, ScoreTerm("severe_clash_filter", "fold_validity", score_at_most(clash_count, 10.0, clash_bad), evaluator_weight(score_config, "eval_clash", 0.9), {"clash_count": clash_count, "bad_threshold": clash_bad}))
    cif_path = structure.get("cif_path")
    chain_breaks = chain_break_count(str(cif_path) if cif_path else None)
    add_term(terms, ScoreTerm("chain_continuity", "fold_validity", 1.0 if chain_breaks == 0 else 0.0, evaluator_weight(score_config, "eval_chain_continuity", 0.8), {"chain_break_count": chain_breaks, "ca_ca_break_cutoff": 4.5}))
    compactness_score, compactness_details = rg_score(str(cif_path) if cif_path else None)
    add_term(terms, ScoreTerm("compactness_rg", "fold_validity", compactness_score, evaluator_weight(score_config, "eval_compactness", 0.35), compactness_details, available=bool(compactness_details.get("available")), warnings=[] if compactness_details.get("available") else [str(compactness_details.get("reason"))]))


def scaffold_preservation_terms(
    out: Mapping[str, Any],
    structure: Mapping[str, Any],
    compiled: Optional[Mapping[str, Any]],
    design_state: Mapping[str, Any],
    template_seqs: Optional[Mapping[str, str]],
    terms: List[ScoreTerm],
    score_config: Mapping[str, Any],
) -> None:


    confidence_by_node = node_plddt_map(out, structure)
    preserved_nodes = node_names_from_design_state(design_state, "preserved")
    values = []
    for name in preserved_nodes:
        item = confidence_by_node.get(name) or {}
        value = node_value(item, "plddt_mean")
        if value is not None:
            values.append(value)
    if values:
        add_term(terms, ScoreTerm("preserved_node_confidence", "scaffold_preservation", score_from_100(mean(values)), evaluator_weight(score_config, "eval_preserved_node_confidence", 0.9), {"preserved_nodes_with_confidence": len(values), "plddt_mean": mean(values), "nodes": preserved_nodes[:20]}))
    else:
        add_term(terms, ScoreTerm("preserved_node_confidence", "scaffold_preservation", 0.0, evaluator_weight(score_config, "eval_preserved_node_confidence", 0.4), {"nodes": preserved_nodes[:20]}, warnings=["no node-level confidence available for preserved scaffold nodes"], available=False))

    sequences = out.get("seqs", {}) if isinstance(out.get("seqs"), Mapping) else {}
    changes = sequence_changes(sequences, template_seqs)
    segment_map = segments_by_name(compiled)
    total = 0
    changed = 0
    changed_nodes = {}
    for name in preserved_nodes:
        segment = segment_map.get(name)
        if not segment:
            continue
        chain_id = segment["chain_id"]
        chain_changes = set(changes.get(chain_id, []))
        indices = set(segment["indices"])
        node_changed = len(chain_changes & indices)
        if indices:
            total += len(indices)
            changed += node_changed
        if node_changed:
            changed_nodes[name] = int(node_changed)
    if total > 0:
        score = 1.0 - (float(changed) / float(total))
        warnings = [f"preserved nodes changed: {changed_nodes}"] if changed_nodes else []
        add_term(terms, ScoreTerm("preserved_scaffold_sequence", "scaffold_preservation", score, evaluator_weight(score_config, "eval_preserved_sequence", 0.65), {"preserved_positions": total, "changed_positions": changed, "changed_nodes": changed_nodes}, warnings=warnings))

    reference_path = resolve_reference_path(score_config, design_state)
    candidate_path = structure.get("cif_path")
    rmsd_good = safe_float(score_config.get("scaffold_rmsd_good"), 1.5) or 1.5
    rmsd_bad = safe_float(score_config.get("scaffold_rmsd_bad"), 4.0) or 4.0
    global_rmsd, global_details = global_ca_rmsd(reference_path, str(candidate_path) if candidate_path else None)
    add_term(terms, ScoreTerm("scaffold_global_ca_rmsd", "scaffold_preservation", rmsd_score(global_rmsd, rmsd_good, rmsd_bad), evaluator_weight(score_config, "eval_scaffold_rmsd", 0.0 if global_rmsd is None else 0.75), {**global_details, "rmsd": global_rmsd, "good_threshold": rmsd_good, "bad_threshold": rmsd_bad}, warnings=[] if global_rmsd is not None else [str(global_details.get("reason"))], available=global_rmsd is not None))
    node_rmsd, node_details = preserved_node_ca_rmsd(reference_path, str(candidate_path) if candidate_path else None, compiled, design_state)
    add_term(terms, ScoreTerm("preserved_node_ca_rmsd", "scaffold_preservation", rmsd_score(node_rmsd, rmsd_good, rmsd_bad), evaluator_weight(score_config, "eval_preserved_node_rmsd", 0.0 if node_rmsd is None else 0.85), {**node_details, "rmsd": node_rmsd, "good_threshold": rmsd_good, "bad_threshold": rmsd_bad}, warnings=[] if node_rmsd is not None else [str(node_details.get("reason"))], available=node_rmsd is not None))


def interface_terms(structure: Mapping[str, Any], terms: List[ScoreTerm], score_config: Mapping[str, Any]) -> None:


    counts = interface_interaction_counts(structure)
    contact_target = safe_float(score_config.get("evaluator_contact_target"), 120.0) or 120.0
    residue_pair_target = safe_float(score_config.get("evaluator_residue_pair_target"), 40.0) or 40.0
    add_term(terms, ScoreTerm("interface_contact_density", "interface", score_at_least(counts["contact_count"], contact_target), evaluator_weight(score_config, "eval_contacts", 0.65), {"contact_count": counts["contact_count"], "target": contact_target, "pair_count": counts["pair_count"]}))
    add_term(terms, ScoreTerm("interface_residue_pair_coverage", "interface", score_at_least(counts["residue_pair_count"], residue_pair_target), evaluator_weight(score_config, "eval_residue_pairs", 0.55), {"residue_pair_count": counts["residue_pair_count"], "target": residue_pair_target}))
    add_term(terms, ScoreTerm("interface_hbond_proxy", "interface", score_at_least(counts["hbond_proxy_count"], 4.0), evaluator_weight(score_config, "eval_hbond_proxy", 0.25), {"hbond_proxy_count": counts["hbond_proxy_count"], "method": "residue-pair polarity and distance proxy"}))
    add_term(terms, ScoreTerm("interface_salt_bridge_proxy", "interface", score_at_least(counts["salt_bridge_proxy_count"], 1.0), evaluator_weight(score_config, "eval_salt_proxy", 0.15), {"salt_bridge_proxy_count": counts["salt_bridge_proxy_count"], "method": "charged residue-pair distance proxy"}))
    add_term(terms, ScoreTerm("interface_hydrophobic_proxy", "interface", score_at_least(counts["hydrophobic_proxy_count"], 6.0), evaluator_weight(score_config, "eval_hydrophobic_proxy", 0.2), {"hydrophobic_proxy_count": counts["hydrophobic_proxy_count"], "method": "hydrophobic residue-pair distance proxy"}))
    geometry = atom_geometry_counts(structure)
    if geometry.get("available"):
        add_term(terms, ScoreTerm("interface_hbond_geometry", "interface_geometry", score_at_least(geometry.get("hbond_geometry_count"), 4.0), evaluator_weight(score_config, "eval_hbond_geometry", 0.45), {**geometry, "method": "atom-level heavy atom distance geometry; no hydrogens required"}, backend="atom_geometry"))
        add_term(terms, ScoreTerm("interface_salt_bridge_geometry", "interface_geometry", score_at_least(geometry.get("salt_bridge_geometry_count"), 1.0), evaluator_weight(score_config, "eval_salt_geometry", 0.25), {**geometry, "method": "charged side-chain atom distance geometry"}, backend="atom_geometry"))
        add_term(terms, ScoreTerm("interface_hydrophobic_geometry", "interface_geometry", score_at_least(geometry.get("hydrophobic_geometry_count"), 8.0), evaluator_weight(score_config, "eval_hydrophobic_geometry", 0.35), {**geometry, "method": "side-chain carbon/sulfur atom contact geometry"}, backend="atom_geometry"))
    else:
        add_term(terms, ScoreTerm("interface_geometry_unavailable", "interface_geometry", 0.0, 0.0, geometry, warnings=[str(geometry.get("reason", "atom-level geometry unavailable"))], backend="atom_geometry", available=False))


def semantic_constraint_terms(
    out: Mapping[str, Any],
    compiled: Optional[Mapping[str, Any]],
    design_state: Mapping[str, Any],
    masks: Optional[Mapping[str, Any]],
    template_seqs: Optional[Mapping[str, str]],
    fixed_residues: Optional[Mapping[str, Mapping[Any, str]]],
    terms: List[ScoreTerm],
    score_config: Mapping[str, Any],
) -> None:


    sequences = out.get("seqs", {}) if isinstance(out.get("seqs"), Mapping) else {}
    changes = sequence_changes(sequences, template_seqs)
    fixed = fixed_lookup(fixed_residues)
    violations: List[Dict[str, Any]] = []
    total_changes = 0
    for chain_id, indices in changes.items():
        total_changes += len(indices)
        for index in indices:
            if not mask_allows(masks, chain_id, index):
                violations.append({"chain_id": chain_id, "index": index, "reason": "mutation outside editable mask"})
            fixed_expected = fixed.get(chain_id, {}).get(index)
            if fixed_expected is not None:
                violations.append({"chain_id": chain_id, "index": index, "reason": "fixed residue changed", "expected": fixed_expected})
    denominator = max(1, total_changes)
    score = 1.0 - min(1.0, float(len(violations)) / float(denominator))
    add_term(terms, ScoreTerm("mutation_scope_guardrail", "semantic_constraint", score, evaluator_weight(score_config, "eval_mutation_scope", 1.0), {"total_mutations": total_changes, "violation_count": len(violations), "violations": violations[:20]}, warnings=[f"{len(violations)} mutation-scope violations"] if violations else []))
    declared_primary_nodes = node_names_from_design_state(design_state, "primary")
    coverage = out.get("semantic_final_mutation_coverage")
    active_nodes = (
        [str(node) for node in coverage.get("active_nodes", []) if str(node)]
        if isinstance(coverage, Mapping)
        else []
    )


    primary_nodes = list(dict.fromkeys(active_nodes or declared_primary_nodes))
    primary_node_source = (
        "semantic_final_mutation_coverage.active_nodes"
        if active_nodes
        else "design_state.primary_design_nodes"
    )
    segment_map = segments_by_name(compiled)
    changed_primary = []
    for name in primary_nodes:
        segment = segment_map.get(name)
        if not segment:
            continue
        if set(changes.get(segment["chain_id"], [])) & set(segment["indices"]):
            changed_primary.append(name)
    if primary_nodes:
        engagement = len(set(changed_primary)) / max(1, len(set(primary_nodes)))
        add_term(terms, ScoreTerm("primary_node_engagement", "semantic_constraint", clamp01(engagement), evaluator_weight(score_config, "eval_primary_engagement", 0.25), {"primary_nodes": primary_nodes, "changed_primary_nodes": sorted(set(changed_primary)), "primary_node_source": primary_node_source, "declared_primary_nodes": declared_primary_nodes}))
    contract_response = design_state.get("_contract_response_report") if isinstance(design_state.get("_contract_response_report"), Mapping) else {}
    if contract_response:
        if contract_response.get("usable"):
            response_score = 1.0
        elif contract_response.get("present"):
            response_score = 0.65
        else:
            response_score = 0.2
        add_term(terms, ScoreTerm("semantic_contract_response", "semantic_constraint", clamp01(response_score), evaluator_weight(score_config, "eval_contract_response", 0.2), {"present": bool(contract_response.get("present")), "usable": bool(contract_response.get("usable")), "accepted": contract_response.get("accepted", []), "rejected": contract_response.get("rejected", []), "reason": contract_response.get("reason", ""), "warnings": contract_response.get("warnings", [])}, warnings=[str(warning) for warning in contract_response.get("warnings", [])]))


def multistate_terms(out: Mapping[str, Any], structure: Mapping[str, Any], terms: List[ScoreTerm], score_config: Mapping[str, Any]) -> None:


    score = multistate_score(out, structure)
    if score is None:
        return
    objectives = objective_items(out, structure)
    add_term(terms, ScoreTerm("typed_multistate_objectives", "multistate", score, evaluator_weight(score_config, "eval_multistate", 1.2), {"objective_count": len(objectives), "normalized_score": score}))


def append_builtin_terms(context, terms: List[ScoreTerm]) -> None:


    confidence_terms(context.out, context.structure, terms, context.score_config)
    fold_validity_terms(context.structure, terms, context.score_config)
    scaffold_preservation_terms(context.out, context.structure, context.compiled, context.design_state, context.template_seqs, terms, context.score_config)
    interface_terms(context.structure, terms, context.score_config)
    semantic_constraint_terms(context.out, context.compiled, context.design_state, context.masks, context.template_seqs, context.fixed_residues, terms, context.score_config)
    multistate_terms(context.out, context.structure, terms, context.score_config)
