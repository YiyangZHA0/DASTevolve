

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from astevolve.evaluation.contracts import EvaluatorContext, ScoreTerm
from astevolve.evaluation.plugins.registry import (
    EvaluatorPluginSpec,
    PluginConfigField,
    register_plugin,
)


PLUGIN_NAME = "tiam1_selectivity"
STATE_APO = "candidate_apo"
STATE_A = "candidate_plus_A_Caspr4"
STATE_B = "candidate_plus_B_Syndecan1"


INTERFACE_Q_CONTACT_TARGET = 400.0
INTERFACE_Q_RESIDUE_PAIR_TARGET = 40.0
INTERFACE_Q_CONTACT_WEIGHT = 0.35
INTERFACE_Q_RESIDUE_PAIR_WEIGHT = 0.20
INTERFACE_Q_PLDDT_WEIGHT = 0.30
INTERFACE_Q_WEIGHT_SUM = (
    INTERFACE_Q_CONTACT_WEIGHT
    + INTERFACE_Q_RESIDUE_PAIR_WEIGHT
    + INTERFACE_Q_PLDDT_WEIGHT
)


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return float(max(0.0, min(1.0, number)))


def _weight(context: EvaluatorContext, name: str, default: float) -> float:
    weights = context.score_config.get("evaluator_weights", {})
    if not isinstance(weights, Mapping):
        return float(default)
    try:
        value = float(weights.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _config_float(context: EvaluatorContext, name: str, default: float) -> float:
    try:
        value = float(context.score_config.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _config_bool(context: EvaluatorContext, name: str, default: bool) -> bool:
    value = context.score_config.get(name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _compiled_mutation_scope(
    context: EvaluatorContext,
) -> Tuple[
    set[int],
    set[int],
    Tuple[str, ...],
    int,
    bool,
    str,
    Mapping[str, Any],
]:


    raw = context.score_config.get("mutation_scope_contract", {})
    if not isinstance(raw, Mapping):
        return set(), set(), (), 0, False, "mutation scope contract is missing", {}
    active_by_chain = raw.get("active_positions_by_chain", {})
    active_by_node = raw.get("active_positions_by_node", {})
    if not isinstance(active_by_chain, Mapping) or not isinstance(active_by_node, Mapping):
        return set(), set(), (), 0, False, "mutation scope contract is malformed", raw
    try:
        allowed = {int(position) for position in active_by_chain.get("P", [])}
        maximum = int(raw.get("max_total_mutations", 0))
    except (TypeError, ValueError):
        return set(), set(), (), 0, False, "mutation scope values are invalid", raw
    active_node_ids = tuple(sorted(str(node_id) for node_id in active_by_node))
    secondary_structure_sensitive = set()
    compiled = context.compiled if isinstance(context.compiled, Mapping) else {}
    for segment in compiled.get("segments", ()) or ():
        chain_id = str(
            segment.get("chain_id", "")
            if isinstance(segment, Mapping)
            else getattr(segment, "chain_id", "")
        )
        kind = str(
            segment.get("kind", "")
            if isinstance(segment, Mapping)
            else getattr(segment, "kind", "")
        )
        if chain_id != "P" or kind not in {"alpha_helix", "beta_strand"}:
            continue
        if hasattr(segment, "indices"):
            segment_positions = {int(position) for position in segment.indices()}
        else:
            spans = (
                segment.get("spans", ())
                if isinstance(segment, Mapping)
                else getattr(segment, "spans", ())
            )
            segment_positions = {
                position
                for start, end in spans or ()
                for position in range(int(start), int(end))
            }
        secondary_structure_sensitive.update(allowed & segment_positions)
    mask = context.masks.get("P") if isinstance(context.masks, Mapping) else None
    if mask is None:
        return allowed, secondary_structure_sensitive, active_node_ids, maximum, False, "runtime mask is missing", raw
    mask_allowed = {index for index, enabled in enumerate(mask) if bool(enabled)}
    node_allowed = set()
    try:
        for item in active_by_node.values():
            if isinstance(item, Mapping) and str(item.get("chain_id")) == "P":
                node_allowed.update(int(position) for position in item.get("positions", []))
    except (TypeError, ValueError):
        return allowed, secondary_structure_sensitive, active_node_ids, maximum, False, "node scope is invalid", raw
    if allowed != mask_allowed or allowed != node_allowed:
        return allowed, secondary_structure_sensitive, active_node_ids, maximum, False, "mask/AST scope mismatch", raw
    if maximum <= 0:
        return allowed, secondary_structure_sensitive, active_node_ids, maximum, False, "mutation budget is not positive", raw
    return allowed, secondary_structure_sensitive, active_node_ids, maximum, True, "", raw


def _states(structure: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(item.get("name")): item
        for item in (structure.get("states", []) or [])
        if isinstance(item, Mapping) and item.get("name")
    }


def _state_plddt(state: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not isinstance(state, Mapping):
        return None
    summary = state.get("structure_metrics", {})
    scalar = summary.get("scalar", {}) if isinstance(summary, Mapping) else {}
    value = scalar.get("plddt") if isinstance(scalar, Mapping) else None
    if value is None:
        confidence = state.get("confidence_metrics", {})
        value = confidence.get("plddt") if isinstance(confidence, Mapping) else None
    try:
        number = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return number if number is not None and math.isfinite(number) else None


def _state_scalar(state: Optional[Mapping[str, Any]], name: str) -> Optional[float]:
    if not isinstance(state, Mapping):
        return None
    summary = state.get("structure_metrics", {})
    scalar = summary.get("scalar", {}) if isinstance(summary, Mapping) else {}
    value = scalar.get(name) if isinstance(scalar, Mapping) else None
    if value is None:
        confidence = state.get("confidence_metrics", {})
        value = confidence.get(name) if isinstance(confidence, Mapping) else None
    try:
        number = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return number if number is not None and math.isfinite(number) else None


def _positive_margin_score(value: Optional[float], target: float) -> float:


    if value is None:
        return 0.0
    return _clamp01(float(value) / max(1e-8, float(target)))


def _sigmoid_margin_score(
    value: Optional[float],
    *,
    target: float,
    sigma: float,
) -> float:


    if value is None:
        return 0.0
    target = float(target)
    sigma = float(sigma)
    if not math.isfinite(target) or not math.isfinite(sigma) or sigma <= 0.0:
        return 0.0
    margin = float(value)
    if not math.isfinite(margin):
        return 0.0
    z = (margin - target) / sigma
    if z >= 0.0:
        return float(1.0 / (1.0 + math.exp(-min(z, 700.0))))
    exponential = math.exp(max(z, -700.0))
    return float(exponential / (1.0 + exponential))


def _detail_float(details: Mapping[str, Any], name: str) -> Optional[float]:
    try:
        value = details.get(name)
        number = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return number if number is not None and math.isfinite(number) else None


def _interface_q(
    details: Mapping[str, Any],
    *,
    contact_target: float,
    residue_pair_target: float,
) -> Tuple[Optional[float], Dict[str, Any]]:


    contact_count = _detail_float(details, "contact_count")
    residue_pair_count = _detail_float(details, "residue_pair_count")
    interface_plddt = _detail_float(details, "interface_plddt_mean")
    available = (
        details.get("available") is True
        and
        contact_count is not None
        and contact_count >= 0.0
        and residue_pair_count is not None
        and residue_pair_count >= 0.0
        and interface_plddt is not None
        and 0.0 <= interface_plddt <= 100.0
        and math.isfinite(contact_target)
        and contact_target > 0.0
        and math.isfinite(residue_pair_target)
        and residue_pair_target > 0.0
    )
    result: Dict[str, Any] = {
        "available": bool(available),
        "contact_count": contact_count,
        "residue_pair_count": residue_pair_count,
        "interface_plddt_mean": interface_plddt,
        "contact_target": contact_target,
        "residue_pair_target": residue_pair_target,
        "component_weights": {
            "log_contact": INTERFACE_Q_CONTACT_WEIGHT,
            "log_residue_pair": INTERFACE_Q_RESIDUE_PAIR_WEIGHT,
            "interface_plddt": INTERFACE_Q_PLDDT_WEIGHT,
        },
        "preclip": False,
        "includes_clash": False,
        "formula": (
            "q=(0.35*ln(1+C)/ln(1+400)+0.20*ln(1+R)/ln(1+40)"
            "+0.30*pLDDT_interface/100)/0.85"
        ),
    }
    if not available:
        result["reason"] = "finite contact, residue-pair, and interface-pLDDT evidence is required"
        return None, result
    contact_component = math.log1p(float(contact_count)) / math.log1p(contact_target)
    residue_component = math.log1p(float(residue_pair_count)) / math.log1p(
        residue_pair_target
    )
    plddt_component = float(interface_plddt) / 100.0
    q_value = (
        INTERFACE_Q_CONTACT_WEIGHT * contact_component
        + INTERFACE_Q_RESIDUE_PAIR_WEIGHT * residue_component
        + INTERFACE_Q_PLDDT_WEIGHT * plddt_component
    ) / INTERFACE_Q_WEIGHT_SUM
    if not math.isfinite(q_value):
        result["available"] = False
        result["reason"] = "computed q is non-finite"
        return None, result
    result.update(
        {
            "log_contact_component": contact_component,
            "log_residue_pair_component": residue_component,
            "interface_plddt_component": plddt_component,
            "q": q_value,
        }
    )
    return float(q_value), result


def _state_interface_clash_count(
    state: Optional[Mapping[str, Any]],
) -> Optional[float]:


    if not isinstance(state, Mapping):
        return None
    summary = state.get("structure_metrics", {})
    interface = summary.get("interface", {}) if isinstance(summary, Mapping) else {}
    if not isinstance(interface, Mapping) or not bool(interface.get("available")):
        return None
    if "clash_count" not in interface:
        return None
    try:
        number = float(interface.get("clash_count"))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _objective(structure: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    pack = structure.get("multistate_objectives", {})
    objectives = pack.get("objectives", {}) if isinstance(pack, Mapping) else {}
    item = objectives.get(name, {}) if isinstance(objectives, Mapping) else {}
    return item if isinstance(item, Mapping) else {}


def _objective_score(structure: Mapping[str, Any], name: str) -> Tuple[float, bool, Dict[str, Any]]:
    item = _objective(structure, name)
    available = bool(item) and not bool(item.get("warnings"))
    details = item.get("details", {}) if isinstance(item.get("details"), Mapping) else {}
    return _clamp01(item.get("score")), available, dict(details)


def _aliases(state: Mapping[str, Any], *, source_chain: str, label: str) -> set[str]:
    aliases = {label}
    for unit in state.get("entity_units", []) or []:
        if not isinstance(unit, Mapping):
            continue
        unit_source = str(unit.get("source_chain") or "")
        unit_label = str(unit.get("label") or "")
        base_label = str(unit.get("base_label") or "")
        if unit_source == source_chain or label in {unit_label, base_label}:
            for key in ("asym_id", "asym_alias", "label", "base_label"):
                value = str(unit.get(key) or "")
                if value:
                    aliases.add(value)
    return aliases


def _iter_residue_pairs(state: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    summary = state.get("structure_metrics", {})
    interface = summary.get("interface", {}) if isinstance(summary, Mapping) else {}
    pairs = interface.get("pairs", {}) if isinstance(interface, Mapping) else {}
    if not isinstance(pairs, Mapping):
        return []
    rows = []
    for pair_group in pairs.values():
        if not isinstance(pair_group, Mapping):
            continue
        rows.extend(
            item
            for item in (pair_group.get("residue_pairs", []) or [])
            if isinstance(item, Mapping)
        )
    return rows


def _residue_index(residue: Mapping[str, Any]) -> Optional[int]:
    try:
        return int(residue.get("residue"))
    except (TypeError, ValueError):
        return None


def _terminal_anchor_evidence(state: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(state, Mapping):
        return {"available": False, "passed": False, "reason": "A state is missing"}
    pdz_aliases = _aliases(state, source_chain="P", label="PDZ")
    peptide_aliases = _aliases(state, source_chain="A", label="A_peptide")
    matching = []
    pair_count = 0
    for pair in _iter_residue_pairs(state):
        pair_count += 1
        left = pair.get("left", {}) if isinstance(pair.get("left"), Mapping) else {}
        right = pair.get("right", {}) if isinstance(pair.get("right"), Mapping) else {}
        left_chain = str(left.get("chain") or "")
        right_chain = str(right.get("chain") or "")
        if left_chain in pdz_aliases and right_chain in peptide_aliases:
            protein, peptide = left, right
        elif right_chain in pdz_aliases and left_chain in peptide_aliases:
            protein, peptide = right, left
        else:
            continue
        protein_position = _residue_index(protein)
        peptide_position = _residue_index(peptide)
        if protein_position not in {18, 19} or peptide_position != 8:
            continue
        try:
            distance = float(pair.get("min_distance"))
        except (TypeError, ValueError):
            distance = None
        contacts = int(pair.get("contact_count") or 0)
        if contacts > 0 or (distance is not None and distance <= 4.5):
            matching.append(
                {
                    "protein_position": protein_position,
                    "peptide_position": peptide_position,
                    "min_distance": distance,
                    "contact_count": contacts,
                }
            )
    return {
        "available": pair_count > 0,
        "passed": bool(matching),
        "pair_count": pair_count,
        "anchor_contacts": matching,
        "pdz_aliases": sorted(pdz_aliases),
        "peptide_aliases": sorted(peptide_aliases),
    }


def _semantic_binding(*nodes: str, reason: str) -> Dict[str, Any]:
    return {
        "structural_nodes": list(nodes),
        "failure_action": "optimize_node",
        "reason": reason,
        "priority": "high",
    }


class Tiam1SelectivityPlugin:


    name = PLUGIN_NAME

    def score_terms(self, context: EvaluatorContext) -> list[ScoreTerm]:
        states = _states(context.structure)
        apo = states.get(STATE_APO)
        state_a = states.get(STATE_A)
        state_b = states.get(STATE_B)
        apo_plddt = _state_plddt(apo)
        a_plddt = _state_plddt(state_a)
        b_plddt = _state_plddt(state_b)
        a_iptm = _state_scalar(state_a, "iptm")
        b_iptm = _state_scalar(state_b, "iptm")
        a_gpde = _state_scalar(state_a, "gpde")
        b_gpde = _state_scalar(state_b, "gpde")
        state_files_complete = all(
            state is not None and _state_plddt(state) is not None
            and bool(state.get("cif_path") or state.get("structure_path"))
            for state in (apo, state_a, state_b)
        )

        a_score, a_available, a_details = _objective_score(
            context.structure, "A_canonical_interface"
        )
        b_off_score, b_available, b_details = _objective_score(
            context.structure, "B_interface_weakening"
        )
        delta_score, delta_available, delta_details = _objective_score(
            context.structure, "A_over_B_interface_delta"
        )
        iptm_delta = a_iptm - b_iptm if a_iptm is not None and b_iptm is not None else None
        gpde_delta = b_gpde - a_gpde if a_gpde is not None and b_gpde is not None else None
        iptm_target = max(1e-8, _config_float(context, "tiam1_iptm_margin_target", 0.08))
        gpde_target = max(1e-8, _config_float(context, "tiam1_gpde_margin_target", 0.04))
        iptm_sigma = max(1e-8, _config_float(context, "tiam1_iptm_margin_sigma", 0.04))
        gpde_sigma = max(1e-8, _config_float(context, "tiam1_gpde_margin_sigma", 0.02))
        q_target = _config_float(context, "tiam1_interface_q_margin_target", 0.03)
        q_sigma = max(1e-8, _config_float(context, "tiam1_interface_q_margin_sigma", 0.02))
        contact_target = max(
            1e-8,
            _config_float(
                context,
                "tiam1_interface_q_contact_target",
                INTERFACE_Q_CONTACT_TARGET,
            ),
        )
        residue_pair_target = max(
            1e-8,
            _config_float(
                context,
                "tiam1_interface_q_residue_pair_target",
                INTERFACE_Q_RESIDUE_PAIR_TARGET,
            ),
        )

        legacy_iptm_margin_score = _positive_margin_score(iptm_delta, iptm_target)
        legacy_gpde_margin_score = _positive_margin_score(gpde_delta, gpde_target)
        v2_iptm_margin_score = _sigmoid_margin_score(
            iptm_delta, target=iptm_target, sigma=iptm_sigma
        )
        v2_gpde_margin_score = _sigmoid_margin_score(
            gpde_delta, target=gpde_target, sigma=gpde_sigma
        )
        positive_details = (
            delta_details.get("positive_details", {})
            if isinstance(delta_details.get("positive_details"), Mapping)
            else {}
        )
        negative_details = (
            delta_details.get("negative_details", {})
            if isinstance(delta_details.get("negative_details"), Mapping)
            else {}
        )
        a_interface_q, a_q_details = _interface_q(
            positive_details,
            contact_target=contact_target,
            residue_pair_target=residue_pair_target,
        )
        b_interface_q, b_q_details = _interface_q(
            negative_details,
            contact_target=contact_target,
            residue_pair_target=residue_pair_target,
        )
        if not delta_available:
            a_interface_q = None
            b_interface_q = None
            a_q_details.update(
                available=False,
                reason="paired interface objective declared warnings or unavailable evidence",
            )
            b_q_details.update(
                available=False,
                reason="paired interface objective declared warnings or unavailable evidence",
            )
        q_delta = (
            a_interface_q - b_interface_q
            if a_interface_q is not None and b_interface_q is not None
            else None
        )
        v2_q_margin_score = _sigmoid_margin_score(
            q_delta, target=q_target, sigma=q_sigma
        )
        a_interface_clashes = _detail_float(positive_details, "clash_count")
        b_interface_clashes = _detail_float(negative_details, "clash_count")
        a_clash_source = "objective_pair_details"
        b_clash_source = "objective_pair_details"
        if a_interface_clashes is None:
            a_interface_clashes = _state_interface_clash_count(state_a)
            a_clash_source = "state_interface_summary"
        if b_interface_clashes is None:
            b_interface_clashes = _state_interface_clash_count(state_b)
            b_clash_source = "state_interface_summary"
        clash_evidence_available = a_interface_clashes is not None and b_interface_clashes is not None
        clash_budget = max(1e-8, _config_float(context, "tiam1_interface_clash_budget", 3.0))
        total_interface_clashes = (
            float(a_interface_clashes) + float(b_interface_clashes)
            if clash_evidence_available
            else None
        )
        clash_guard_score = (
            1.0 - _clamp01(float(total_interface_clashes) / clash_budget)
            if total_interface_clashes is not None
            else 0.0
        )
        baseline_apo_plddt = _config_float(
            context, "tiam1_baseline_apo_plddt", 87.3060073852539
        )
        baseline_a_plddt = _config_float(
            context, "tiam1_baseline_positive_plddt", 92.97100067138672
        )
        baseline_a_iptm = _config_float(
            context, "tiam1_baseline_positive_iptm", 0.9367565512657166
        )
        baseline_a_q = _config_float(
            context,
            "tiam1_baseline_positive_interface_q",
            0.9429525830682546,
        )
        apo_floor = baseline_apo_plddt + _config_float(
            context, "tiam1_apo_plddt_noninferiority_band", -5.0
        )
        a_plddt_floor = baseline_a_plddt + _config_float(
            context, "tiam1_positive_plddt_noninferiority_band", -3.0
        )
        a_iptm_floor = baseline_a_iptm + _config_float(
            context, "tiam1_positive_iptm_noninferiority_band", -0.03
        )
        a_q_floor = baseline_a_q + _config_float(
            context, "tiam1_positive_interface_q_noninferiority_band", -0.05
        )
        require_final_b_clash_free = _config_bool(
            context, "tiam1_final_require_b_clash_free", False
        )
        complete = bool(
            state_files_complete
            and iptm_delta is not None
            and gpde_delta is not None
            and a_interface_q is not None
            and b_interface_q is not None
            and clash_evidence_available
        )
        anchor = _terminal_anchor_evidence(state_a)

        sequences = context.out.get("seqs", {})
        if not isinstance(sequences, Mapping):
            sequences = context.out.get("best_seqs", {})
        templates = context.template_seqs if isinstance(context.template_seqs, Mapping) else {}
        candidate = str(sequences.get("P") or "") if isinstance(sequences, Mapping) else ""
        template = str(templates.get("P") or "")
        (
            open_positions,
            secondary_structure_sensitive_positions,
            active_node_ids,
            max_total_mutations,
            scope_available,
            scope_warning,
            scope_contract,
        ) = _compiled_mutation_scope(context)
        changes = [
            index
            for index, (before, after) in enumerate(zip(template, candidate))
            if before != after
        ] if len(candidate) == len(template) and candidate else []
        scope_pass = (
            scope_available
            and bool(candidate)
            and len(candidate) == len(template)
            and set(changes) <= open_positions
        )
        budget_pass = scope_pass and len(changes) <= max_total_mutations
        proline_positions = [
            index
            for index in secondary_structure_sensitive_positions
            if index < len(candidate)
            and index < len(template)
            and candidate[index] == "P"
            and template[index] != "P"
        ]

        task_binding = _semantic_binding(
            *active_node_ids,
            reason="paired A/B structural selectivity evidence is weak",
        )
        anchor_binding = _semantic_binding(
            *active_node_ids,
            reason="restore the canonical A-peptide terminal anchor without editing frozen anchor residues",
        )

        return [
            ScoreTerm(
                "tiam1_state_evidence_complete",
                "task_correctness",
                1.0 if complete else 0.0,
                _weight(context, "eval_tiam1_state_evidence", 0.0),
                {
                    "dimension": "correctness",
                    "state": "preserve",
                    "required": True,
                    "hard_gate": True,
                    "hard_gate_min_score": 0.999,
                    "hard_gate_reason": "tiam1_multistate_structure_evidence_missing",
                    "state_plddt": {STATE_APO: apo_plddt, STATE_A: a_plddt, STATE_B: b_plddt},
                    "state_files_complete": state_files_complete,
                    "paired_iptm_available": iptm_delta is not None,
                    "paired_gpde_available": gpde_delta is not None,
                    "paired_interface_q_available": (
                        a_interface_q is not None and b_interface_q is not None
                    ),
                    "paired_clash_evidence_available": clash_evidence_available,
                },
                warnings=[] if complete else ["required finite apo/A/B v2 evidence is incomplete"],
                backend=PLUGIN_NAME,
                available=complete,
            ),
            ScoreTerm(
                "tiam1_apo_fold_integrity",
                "task_specific",
                _clamp01((apo_plddt or 0.0) / 100.0),
                _weight(context, "eval_tiam1_apo_fold", 0.0),
                {
                    "dimension": "quality",
                    "state": "preserve",
                    "required": True,
                    "hard_gate": True,
                    "hard_gate_min_score": _clamp01(apo_floor / 100.0),
                    "hard_gate_reason": "tiam1_apo_baseline_noninferiority_failed",
                    "plddt": apo_plddt,
                    "locked_baseline_plddt": baseline_apo_plddt,
                    "noninferiority_band": apo_floor - baseline_apo_plddt,
                    "absolute_floor": apo_floor,
                    "semantic_binding": task_binding,
                },
                warnings=[] if apo_plddt is not None else ["apo confidence is missing"],
                backend=PLUGIN_NAME,
                available=apo_plddt is not None,
            ),
            ScoreTerm(
                "tiam1_A_complex_integrity",
                "task_specific",
                _clamp01((a_plddt or 0.0) / 100.0),
                _weight(context, "eval_tiam1_A_complex", 0.0),
                {
                    "dimension": "correctness",
                    "state": "positive",
                    "required": True,
                    "hard_gate": True,
                    "hard_gate_min_score": _clamp01(a_plddt_floor / 100.0),
                    "hard_gate_reason": "tiam1_A_complex_baseline_noninferiority_failed",
                    "plddt": a_plddt,
                    "locked_baseline_plddt": baseline_a_plddt,
                    "noninferiority_band": a_plddt_floor - baseline_a_plddt,
                    "absolute_floor": a_plddt_floor,
                    "semantic_binding": task_binding,
                },
                warnings=[] if a_plddt is not None else ["A-complex confidence is missing"],
                backend=PLUGIN_NAME,
                available=a_plddt is not None,
            ),
            ScoreTerm(
                "tiam1_A_terminal_anchor",
                "task_correctness",
                1.0 if anchor.get("passed") else 0.0,
                _weight(context, "eval_tiam1_A_anchor", 0.0),
                {
                    "dimension": "correctness",
                    "state": "positive",
                    "required": True,
                    "hard_gate": True,
                    "hard_gate_min_score": 0.999,
                    "hard_gate_reason": "tiam1_A_terminal_anchor_missing",
                    "semantic_binding": anchor_binding,
                    **anchor,
                },
                warnings=[] if anchor.get("passed") else ["canonical A terminal-anchor contact was not observed"],
                backend=PLUGIN_NAME,
                available=bool(anchor.get("available")),
            ),
            ScoreTerm(
                "tiam1_A_iptm_noninferiority",
                "task_correctness",
                _clamp01(a_iptm if a_iptm is not None else 0.0),
                _weight(context, "eval_tiam1_A_iptm_noninferiority", 0.0),
                {
                    "dimension": "correctness",
                    "state": "positive",
                    "required": True,
                    "hard_gate": True,
                    "hard_gate_min_score": _clamp01(a_iptm_floor),
                    "hard_gate_reason": "tiam1_A_iptm_baseline_noninferiority_failed",
                    "A_iptm": a_iptm,
                    "locked_baseline_A_iptm": baseline_a_iptm,
                    "noninferiority_band": a_iptm_floor - baseline_a_iptm,
                    "absolute_floor": a_iptm_floor,
                    "semantic_binding": task_binding,
                },
                warnings=[] if a_iptm is not None else ["A ipTM evidence is missing"],
                backend=PLUGIN_NAME,
                available=a_iptm is not None,
            ),
            ScoreTerm(
                "tiam1_A_interface_q_noninferiority",
                "task_correctness",
                _clamp01(a_interface_q if a_interface_q is not None else 0.0),
                _weight(context, "eval_tiam1_A_interface_q_noninferiority", 0.0),
                {
                    "dimension": "correctness",
                    "state": "positive",
                    "required": True,
                    "hard_gate": True,
                    "hard_gate_min_score": _clamp01(a_q_floor),
                    "hard_gate_reason": "tiam1_A_interface_q_baseline_noninferiority_failed",
                    "A_interface_q": a_interface_q,
                    "locked_baseline_A_interface_q": baseline_a_q,
                    "noninferiority_band": a_q_floor - baseline_a_q,
                    "absolute_floor": a_q_floor,
                    "q_evidence": a_q_details,
                    "semantic_binding": task_binding,
                },
                warnings=[] if a_interface_q is not None else ["A interface-q evidence is missing"],
                backend=PLUGIN_NAME,
                available=a_interface_q is not None,
            ),
            ScoreTerm(
                "tiam1_A_clash_free",
                "task_correctness",
                1.0 if a_interface_clashes == 0.0 else 0.0,
                _weight(context, "eval_tiam1_A_clash_free", 0.0),
                {
                    "dimension": "correctness",
                    "state": "positive",
                    "required": True,
                    "hard_gate": True,
                    "hard_gate_min_score": 0.999,
                    "hard_gate_reason": "tiam1_A_interface_clash_detected",
                    "A_interface_clash_count": a_interface_clashes,
                    "required_count": 0,
                    "soft_energy_weight": 0.0,
                    "semantic_binding": task_binding,
                },
                warnings=(
                    []
                    if a_interface_clashes == 0.0
                    else ["A interface must be clash-free"]
                ),
                backend=PLUGIN_NAME,
                available=a_interface_clashes is not None,
            ),
            ScoreTerm(
                "tiam1_B_clash_free_final",
                "task_correctness",
                1.0 if b_interface_clashes == 0.0 else 0.0,
                _weight(context, "eval_tiam1_B_clash_free_final", 0.0),
                {
                    "dimension": "correctness",
                    "state": "negative",
                    "required": bool(require_final_b_clash_free),
                    "hard_gate": bool(require_final_b_clash_free),
                    "hard_gate_min_score": 0.999,
                    "hard_gate_reason": "tiam1_B_interface_clash_detected_at_final_validation",
                    "B_interface_clash_count": b_interface_clashes,
                    "required_count": 0,
                    "gate_scope": "final_validation" if require_final_b_clash_free else "diagnostic_during_search",
                    "soft_energy_weight": 0.0,
                    "semantic_binding": task_binding,
                },
                warnings=(
                    []
                    if b_interface_clashes == 0.0 or not require_final_b_clash_free
                    else ["B interface must be clash-free at final validation"]
                ),
                backend=PLUGIN_NAME,
                available=b_interface_clashes is not None,
            ),
            ScoreTerm(
                "tiam1_A_interface_quality",
                "task_correctness",
                a_score,
                _weight(context, "eval_tiam1_A_interface", 0.0),
                {
                    "dimension": "correctness",
                    "state": "positive",
                    "diagnostic_only": True,
                    "energy_model_version": "v1",
                    "semantic_binding": task_binding,
                    "report_details": a_details,
                },
                warnings=[] if a_available else ["A interface objective is unavailable"],
                backend=PLUGIN_NAME,
                available=a_available,
            ),
            ScoreTerm(
                "tiam1_B_interface_weakening",
                "task_correctness",
                b_off_score,
                _weight(context, "eval_tiam1_B_weakening", 0.0),
                {
                    "dimension": "correctness",
                    "state": "negative",
                    "diagnostic_only": True,
                    "energy_model_version": "v1",
                    "semantic_binding": task_binding,
                    "report_details": b_details,
                },
                warnings=[] if b_available else ["B interface objective is unavailable"],
                backend=PLUGIN_NAME,
                available=b_available,
            ),
            ScoreTerm(
                "tiam1_A_over_B_margin",
                "task_correctness",
                delta_score,
                _weight(context, "eval_tiam1_A_over_B_margin", 0.0),
                {
                    "dimension": "correctness",
                    "state": "positive",
                    "diagnostic_only": True,
                    "energy_model_version": "v1",
                    "semantic_binding": task_binding,
                    "report_details": delta_details,
                },
                warnings=[] if delta_available else ["paired A/B margin objective is unavailable"],
                backend=PLUGIN_NAME,
                available=delta_available,
            ),
            ScoreTerm(
                "tiam1_A_over_B_iptm_margin",
                "task_correctness",
                legacy_iptm_margin_score,
                _weight(context, "eval_tiam1_A_over_B_iptm_margin", 0.0),
                {
                    "dimension": "correctness",
                    "state": "positive",
                    "required": True,
                    "diagnostic_only": True,
                    "energy_model_version": "v1",
                    "semantic_binding": task_binding,
                    "A_iptm": a_iptm,
                    "B_iptm": b_iptm,
                    "A_minus_B_iptm": iptm_delta,
                    "target_delta": iptm_target,
                    "formula": "clip((ipTM_A - ipTM_B) / target_delta, 0, 1)",
                },
                warnings=[] if iptm_delta is not None else ["paired A/B ipTM evidence is unavailable"],
                backend=PLUGIN_NAME,
                available=iptm_delta is not None,
            ),
            ScoreTerm(
                "tiam1_A_over_B_gpde_margin",
                "task_correctness",
                legacy_gpde_margin_score,
                _weight(context, "eval_tiam1_A_over_B_gpde_margin", 0.0),
                {
                    "dimension": "correctness",
                    "state": "positive",
                    "required": True,
                    "diagnostic_only": True,
                    "energy_model_version": "v1",
                    "semantic_binding": task_binding,
                    "A_gpde": a_gpde,
                    "B_gpde": b_gpde,
                    "B_minus_A_gpde": gpde_delta,
                    "target_delta": gpde_target,
                    "formula": "clip((gPDE_B - gPDE_A) / target_delta, 0, 1)",
                },
                warnings=[] if gpde_delta is not None else ["paired A/B gPDE evidence is unavailable"],
                backend=PLUGIN_NAME,
                available=gpde_delta is not None,
            ),
            ScoreTerm(
                "tiam1_v2_iptm_margin",
                "task_correctness",
                v2_iptm_margin_score,
                _weight(context, "eval_tiam1_v2_iptm_margin", 0.50),
                {
                    "dimension": "correctness",
                    "state": "positive",
                    "required": True,
                    "energy_model_version": "tiam1_proxy_energy_v2",
                    "A_iptm": a_iptm,
                    "B_iptm": b_iptm,
                    "A_minus_B_iptm": iptm_delta,
                    "target_tau": iptm_target,
                    "sigma": iptm_sigma,
                    "formula": "sigmoid(((ipTM_A-ipTM_B)-tau_i)/sigma_i)",
                    "semantic_binding": task_binding,
                },
                warnings=[] if iptm_delta is not None else ["paired A/B ipTM evidence is unavailable"],
                backend=PLUGIN_NAME,
                available=iptm_delta is not None,
            ),
            ScoreTerm(
                "tiam1_v2_gpde_margin",
                "task_correctness",
                v2_gpde_margin_score,
                _weight(context, "eval_tiam1_v2_gpde_margin", 0.20),
                {
                    "dimension": "correctness",
                    "state": "positive",
                    "required": True,
                    "energy_model_version": "tiam1_proxy_energy_v2",
                    "A_gpde": a_gpde,
                    "B_gpde": b_gpde,
                    "B_minus_A_gpde": gpde_delta,
                    "target_tau": gpde_target,
                    "sigma": gpde_sigma,
                    "formula": "sigmoid(((gPDE_B-gPDE_A)-tau_g)/sigma_g)",
                    "semantic_binding": task_binding,
                },
                warnings=[] if gpde_delta is not None else ["paired A/B gPDE evidence is unavailable"],
                backend=PLUGIN_NAME,
                available=gpde_delta is not None,
            ),
            ScoreTerm(
                "tiam1_v2_interface_q_margin",
                "task_correctness",
                v2_q_margin_score,
                _weight(context, "eval_tiam1_v2_interface_q_margin", 0.20),
                {
                    "dimension": "correctness",
                    "state": "positive",
                    "required": True,
                    "energy_model_version": "tiam1_proxy_energy_v2",
                    "A_interface_q": a_interface_q,
                    "B_interface_q": b_interface_q,
                    "A_minus_B_interface_q": q_delta,
                    "target_tau": q_target,
                    "sigma": q_sigma,
                    "formula": "sigmoid(((q_A-q_B)-tau_q)/sigma_q)",
                    "A_q_evidence": a_q_details,
                    "B_q_evidence": b_q_details,
                    "semantic_binding": task_binding,
                },
                warnings=[] if q_delta is not None else ["paired A/B interface-q evidence is unavailable"],
                backend=PLUGIN_NAME,
                available=q_delta is not None,
            ),
            ScoreTerm(
                "tiam1_v2_positive_interface_q",
                "task_correctness",
                a_interface_q if a_interface_q is not None else 0.0,
                _weight(context, "eval_tiam1_v2_positive_interface_q", 0.10),
                {
                    "dimension": "correctness",
                    "state": "positive",
                    "required": True,
                    "energy_model_version": "tiam1_proxy_energy_v2",
                    "A_interface_q": a_interface_q,
                    "formula": "S_A=q_A (bounded only at ScoreTerm aggregation)",
                    "q_preclip": False,
                    "q_evidence": a_q_details,
                    "semantic_binding": task_binding,
                },
                warnings=[] if a_interface_q is not None else ["A interface-q evidence is unavailable"],
                backend=PLUGIN_NAME,
                available=a_interface_q is not None,
            ),
            ScoreTerm(
                "tiam1_interface_clash_guard",
                "task_correctness",
                clash_guard_score,
                _weight(context, "eval_tiam1_interface_clash_guard", 0.0),
                {
                    "dimension": "correctness",
                    "state": "preserve",
                    "required": True,
                    "diagnostic_only": True,
                    "energy_model_version": "v1",
                    "semantic_binding": task_binding,
                    "A_interface_clash_count": a_interface_clashes,
                    "B_interface_clash_count": b_interface_clashes,
                    "A_interface_clash_source": a_clash_source,
                    "B_interface_clash_source": b_clash_source,
                    "total_interface_clash_count": total_interface_clashes,
                    "clash_budget": clash_budget,
                    "formula": "1 - clip((clash_A + clash_B) / clash_budget, 0, 1)",
                    "anti_gaming": "diagnostic only; clash cannot improve the v2 soft energy",
                    "v2_policy": "zero soft contribution; A is a search hard gate and both are final hard gates",
                },
                warnings=[] if clash_evidence_available else ["paired A/B interface clash evidence is unavailable"],
                backend=PLUGIN_NAME,
                available=clash_evidence_available,
            ),
            ScoreTerm(
                "tiam1_mutation_budget_integrity",
                "semantic_constraint",
                1.0 if budget_pass else 0.0,
                _weight(context, "eval_tiam1_mutation_budget", 0.0),
                {
                    "dimension": "quality",
                    "state": "preserve",
                    "hard_gate": True,
                    "hard_gate_min_score": 0.999,
                    "hard_gate_reasons": [
                        "tiam1_mutation_outside_active_ast_scope" if not scope_pass else "tiam1_mutation_budget_exceeded"
                    ],
                    "mutation_count": len(changes),
                    "max_total_mutations": max_total_mutations,
                    "changed_positions_zero_based": changes,
                    "allowed_positions_zero_based": sorted(open_positions),
                    "scope_contract": dict(scope_contract),
                    "scope_available": scope_available,
                },
                warnings=(
                    []
                    if budget_pass
                    else [
                        scope_warning
                        or "candidate violates the active AST mutation scope or budget"
                    ]
                ),
                backend=PLUGIN_NAME,
                available=bool(candidate and template and scope_available),
            ),
            ScoreTerm(
                "tiam1_no_new_alpha2_proline",
                "semantic_constraint",
                1.0 if not proline_positions else 0.0,
                _weight(context, "eval_tiam1_no_proline", 0.0),
                {
                    "dimension": "quality",
                    "state": "preserve",
                    "hard_gate": True,
                    "hard_gate_min_score": 0.999,
                    "hard_gate_reason": "tiam1_new_alpha2_proline",
                    "new_proline_positions_zero_based": sorted(proline_positions),
                    "secondary_structure_sensitive_positions_zero_based": sorted(
                        secondary_structure_sensitive_positions
                    ),
                },
                warnings=[] if not proline_positions else ["a mutable helix/strand site gained proline"],
                backend=PLUGIN_NAME,
                available=bool(candidate),
            ),
        ]


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
    """Deterministic CPU contract evaluator used only by case preflight."""

    _ = compiled, design_state, score_config
    raw_sequences = out.get("seqs", {}) if isinstance(out, Mapping) else {}
    sequences = {
        str(chain): str(sequence)
        for chain, sequence in (
            raw_sequences.items() if isinstance(raw_sequences, Mapping) else ()
        )
    }
    templates = {
        str(chain): str(sequence)
        for chain, sequence in (template_seqs or {}).items()
    }
    raw_masks = masks if isinstance(masks, Mapping) else {}
    fixed = fixed_residues if isinstance(fixed_residues, Mapping) else {}
    reasons: list[str] = []

    for chain in sorted(set(templates) - set(sequences)):
        reasons.append(f"missing_chain:{chain}")
    for chain in sorted(set(sequences) - set(templates)):
        reasons.append(f"unexpected_chain:{chain}")

    for chain, template in sorted(templates.items()):
        sequence = sequences.get(chain, "")
        if len(sequence) != len(template):
            reasons.append(f"length_mismatch:{chain}")
            continue
        mask = raw_masks.get(chain)
        if mask is None or len(mask) != len(template):
            reasons.append(f"mask_missing_or_malformed:{chain}")
            continue
        for position, (parent, candidate) in enumerate(zip(template, sequence)):
            if candidate != parent and not bool(mask[position]):
                reasons.append(f"outside_compiled_mask:{chain}:{position}")

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

    reasons = sorted(set(reasons))
    passed = not reasons
    return {
        "schema_version": "ast_evaluator_report_v1",
        "normalized_score": 1.0 if passed else 0.0,
        "soft_score": 1.0 if passed else 0.0,
        "loss": 0.0 if passed else 1.0,
        "hard_gate_pass": passed,
        "disqualification_reasons": reasons,
        "gate_status": {
            "passed": passed,
            "hard_gate_pass": passed,
            "hard_failures": reasons,
            "disqualification_reasons": reasons,
        },
        "terms": [
            {
                "provider": "tiam1_preflight_cpu",
                "state": "preserve",
                "name": "sequence_contract_integrity",
                "category": "contract_guardrail",
                "score": 1.0 if passed else 0.0,
                "weight": 1.0,
                "available": True,
                "details": {"violations": reasons},
                "warnings": [],
            }
        ],
        "warnings": [],
    }


def register_tiam1_plugin() -> None:


    register_plugin(
        EvaluatorPluginSpec(
            name=PLUGIN_NAME,
            factory="cases.tiam1_pdz_caspr4_vs_sdc1_selectivity.tiam1_evaluator:Tiam1SelectivityPlugin",
            weight_fields=(
                "eval_tiam1_state_evidence",
                "eval_tiam1_apo_fold",
                "eval_tiam1_A_complex",
                "eval_tiam1_A_anchor",
                "eval_tiam1_A_iptm_noninferiority",
                "eval_tiam1_A_interface_q_noninferiority",
                "eval_tiam1_A_clash_free",
                "eval_tiam1_B_clash_free_final",
                "eval_tiam1_A_interface",
                "eval_tiam1_B_weakening",
                "eval_tiam1_A_over_B_margin",
                "eval_tiam1_A_over_B_iptm_margin",
                "eval_tiam1_A_over_B_gpde_margin",
                "eval_tiam1_v2_iptm_margin",
                "eval_tiam1_v2_gpde_margin",
                "eval_tiam1_v2_interface_q_margin",
                "eval_tiam1_v2_positive_interface_q",
                "eval_tiam1_interface_clash_guard",
                "eval_tiam1_mutation_budget",
                "eval_tiam1_no_proline",
            ),
            config_fields={
                "iptm_margin_target": PluginConfigField(
                    kind="float", runtime_key="tiam1_iptm_margin_target"
                ),
                "gpde_margin_target": PluginConfigField(
                    kind="float", runtime_key="tiam1_gpde_margin_target"
                ),
                "iptm_margin_sigma": PluginConfigField(
                    kind="float", runtime_key="tiam1_iptm_margin_sigma"
                ),
                "gpde_margin_sigma": PluginConfigField(
                    kind="float", runtime_key="tiam1_gpde_margin_sigma"
                ),
                "interface_q_margin_target": PluginConfigField(
                    kind="float", runtime_key="tiam1_interface_q_margin_target"
                ),
                "interface_q_margin_sigma": PluginConfigField(
                    kind="float", runtime_key="tiam1_interface_q_margin_sigma"
                ),
                "interface_q_contact_target": PluginConfigField(
                    kind="float", runtime_key="tiam1_interface_q_contact_target"
                ),
                "interface_q_residue_pair_target": PluginConfigField(
                    kind="float", runtime_key="tiam1_interface_q_residue_pair_target"
                ),
                "interface_clash_budget": PluginConfigField(
                    kind="float", runtime_key="tiam1_interface_clash_budget"
                ),
                "baseline_apo_plddt": PluginConfigField(
                    kind="float", runtime_key="tiam1_baseline_apo_plddt"
                ),
                "baseline_positive_plddt": PluginConfigField(
                    kind="float", runtime_key="tiam1_baseline_positive_plddt"
                ),
                "baseline_positive_iptm": PluginConfigField(
                    kind="float", runtime_key="tiam1_baseline_positive_iptm"
                ),
                "baseline_positive_interface_q": PluginConfigField(
                    kind="float", runtime_key="tiam1_baseline_positive_interface_q"
                ),
                "apo_plddt_noninferiority_band": PluginConfigField(
                    kind="float", runtime_key="tiam1_apo_plddt_noninferiority_band"
                ),
                "positive_plddt_noninferiority_band": PluginConfigField(
                    kind="float", runtime_key="tiam1_positive_plddt_noninferiority_band"
                ),
                "positive_iptm_noninferiority_band": PluginConfigField(
                    kind="float", runtime_key="tiam1_positive_iptm_noninferiority_band"
                ),
                "positive_interface_q_noninferiority_band": PluginConfigField(
                    kind="float",
                    runtime_key="tiam1_positive_interface_q_noninferiority_band",
                ),
                "final_require_b_clash_free": PluginConfigField(
                    kind="bool", runtime_key="tiam1_final_require_b_clash_free"
                ),
            },
        ),
        replace=True,
    )


__all__ = [
    "Tiam1SelectivityPlugin",
    "evaluate_candidate",
    "register_tiam1_plugin",
]
