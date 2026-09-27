"""Case-owned evaluator for a 50-aa binder against fixed ASYN residues 99-140.

The evaluator deliberately does not reward high pLDDT for the intrinsically
disordered ASYN C terminus. It scores binder confidence inside the holo
complex, binder-ASYN interface evidence, hotspot localization, and sequence
developability while hard-gating the immutable target contract.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from astevolve.evaluation.contracts import EvaluatorContext, ScoreTerm
from astevolve.evaluation.plugins.registry import (
    EvaluatorPluginSpec,
    PluginConfigField,
    register_plugin,
)


PLUGIN_NAME = "asyn_hotspot_binder"
STATE_BOUND = "binder_plus_ASYN"
TARGET_SEQUENCE = (
    "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVT"
    "NVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSE"
    "EGYQDYEPEA"
)
CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC_AA = frozenset("AVILMFWY")
CHARGED_AA = frozenset("DEKR")


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return float(max(0.0, min(1.0, number)))


def _config_float(context: EvaluatorContext, name: str, default: float) -> float:
    try:
        value = float(context.score_config.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _weight(context: EvaluatorContext, name: str, default: float) -> float:
    weights = context.score_config.get("evaluator_weights", {})
    if not isinstance(weights, Mapping):
        return float(default)
    try:
        value = float(weights.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _sequences(out: Mapping[str, Any]) -> Dict[str, str]:
    raw = out.get("seqs")
    if not isinstance(raw, Mapping):
        raw = out.get("best_seqs")
    if not isinstance(raw, Mapping):
        return {}
    return {str(chain): str(sequence) for chain, sequence in raw.items()}


def _states(structure: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(item.get("name")): item
        for item in (structure.get("states", []) or [])
        if isinstance(item, Mapping) and item.get("name")
    }


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


def _binder_holo_confidence(state: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return full-chain and local binder confidence from the holo prediction.

    Full-chain confidence is the decision metric. Active node means are not a
    valid replacement after selectors are resized because they can omit fixed
    residues and weight differently sized nodes equally.
    """
    if not isinstance(state, Mapping):
        return {"plddt": None, "source": None, "node_means": {}, "node_min": None}
    summary = state.get("structure_metrics", {})
    chain_plddt = summary.get("chain_plddt", {}) if isinstance(summary, Mapping) else {}
    binder_plddt = None
    binder_source = None
    if isinstance(chain_plddt, Mapping):
        for alias in ("binder", "B", "A"):
            try:
                value = float(chain_plddt.get(alias))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                binder_plddt = value
                binder_source = f"chain_plddt.{alias}"
                break
    node_plddt = summary.get("node_plddt", {}) if isinstance(summary, Mapping) else {}
    node_means: Dict[str, float] = {}
    node_minima: list[float] = []
    if isinstance(node_plddt, Mapping):
        for node_name, item in node_plddt.items():
            if not str(node_name).startswith("B") or not isinstance(item, Mapping):
                continue
            try:
                value = float(item.get("plddt_mean"))
            except (TypeError, ValueError):
                value = float("nan")
            if math.isfinite(value):
                node_means[str(node_name)] = value
            try:
                minimum = float(item.get("plddt_min"))
            except (TypeError, ValueError):
                minimum = float("nan")
            if math.isfinite(minimum):
                node_minima.append(minimum)
    if binder_plddt is None and node_means:
        binder_plddt = float(sum(node_means.values()) / len(node_means))
        binder_source = "legacy_unweighted_node_mean_fallback"
    return {"plddt": binder_plddt, "source": binder_source,
            "node_means": node_means,
            "node_min": min(node_minima) if node_minima else None}


def _objective(structure: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    pack = structure.get("multistate_objectives", {})
    objectives = pack.get("objectives", {}) if isinstance(pack, Mapping) else {}
    item = objectives.get(name, {}) if isinstance(objectives, Mapping) else {}
    return item if isinstance(item, Mapping) else {}


def _objective_score(
    structure: Mapping[str, Any], name: str
) -> Tuple[float, bool, Dict[str, Any]]:
    item = _objective(structure, name)
    details = item.get("details", {}) if isinstance(item.get("details"), Mapping) else {}
    available = bool(item) and not bool(item.get("warnings"))
    return _clamp01(item.get("score")), available, dict(details)


def _aliases(state: Mapping[str, Any], *, source_chain: str, label: str) -> set[str]:
    aliases = {label, source_chain}
    for unit in state.get("entity_units", []) or []:
        if not isinstance(unit, Mapping):
            continue
        unit_source = str(unit.get("source_chain") or "")
        unit_label = str(unit.get("label") or "")
        base_label = str(unit.get("base_label") or "")
        if unit_source == source_chain or label in {unit_label, base_label}:
            for key in ("asym_id", "asym_alias", "label", "base_label", "source_chain"):
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


def _residue_number(residue: Mapping[str, Any]) -> Optional[int]:
    try:
        return int(residue.get("residue"))
    except (TypeError, ValueError):
        return None


def _hotspot_localization(
    state: Optional[Mapping[str, Any]],
    *,
    proximity_decay_residues: float = 12.0,
    hotspot_residue_target: float = 6.0,
) -> Dict[str, Any]:
    if not isinstance(state, Mapping):
        return {"available": False, "fraction": 0.0, "reason": "bound state missing"}
    binder_aliases = _aliases(state, source_chain="B", label="binder")
    target_aliases = _aliases(state, source_chain="T", label="ASYN")
    target_pairs: list[int] = []
    hotspot_pairs: list[int] = []
    binder_pairs: list[int] = []
    hotspot_binder_pairs: list[int] = []
    binder_target_contact_pairs: list[Dict[str, Any]] = []
    for pair in _iter_residue_pairs(state):
        left = pair.get("left", {}) if isinstance(pair.get("left"), Mapping) else {}
        right = pair.get("right", {}) if isinstance(pair.get("right"), Mapping) else {}
        left_chain = str(left.get("chain") or "")
        right_chain = str(right.get("chain") or "")
        if left_chain in binder_aliases and right_chain in target_aliases:
            binder_residue = _residue_number(left)
            target_residue = _residue_number(right)
        elif right_chain in binder_aliases and left_chain in target_aliases:
            binder_residue = _residue_number(right)
            target_residue = _residue_number(left)
        else:
            continue
        if binder_residue is None or target_residue is None:
            continue
        is_hotspot = 99 <= target_residue <= 140
        binder_pairs.append(binder_residue)
        target_pairs.append(target_residue)
        binder_target_contact_pairs.append(
            {
                "binder_residue": binder_residue,
                "target_residue": target_residue,
                "is_hotspot": is_hotspot,
                "contact_count": pair.get("contact_count"),
                "min_distance": pair.get("min_distance"),
            }
        )
        if is_hotspot:
            hotspot_binder_pairs.append(binder_residue)
            hotspot_pairs.append(target_residue)
    fraction = float(len(hotspot_pairs) / len(target_pairs)) if target_pairs else 0.0
    decay = max(1.0, float(proximity_decay_residues))
    proximity_values = [
        1.0
        if 99 <= residue <= 140
        else math.exp(-min(abs(residue - 99), abs(residue - 140)) / decay)
        for residue in target_pairs
    ]
    proximity_score = (
        float(sum(proximity_values) / len(proximity_values))
        if proximity_values
        else 0.0
    )
    unique_hotspot_residues = sorted(set(hotspot_pairs))
    hotspot_coverage_score = _clamp01(
        len(unique_hotspot_residues) / max(1.0, float(hotspot_residue_target))
    )
    # Direct contacts remain dominant. Proximity supplies a smooth gradient
    # before residue 99 is reached; unique coverage prevents one repeated
    # residue pair from dominating the localization objective.
    score = _clamp01(
        0.55 * fraction + 0.30 * hotspot_coverage_score + 0.15 * proximity_score
    )
    return {
        "available": bool(target_pairs),
        "fraction": fraction,
        "score": score,
        "direct_contact_fraction": fraction,
        "proximity_score": proximity_score,
        "proximity_decay_residues": decay,
        "hotspot_coverage_score": hotspot_coverage_score,
        "hotspot_residue_target": float(hotspot_residue_target),
        "target_residue_pair_count": len(target_pairs),
        "hotspot_residue_pair_count": len(hotspot_pairs),
        "unique_hotspot_residue_count": len(unique_hotspot_residues),
        "hotspot_one_based_inclusive": [99, 140],
        "contacted_binder_residues": sorted(set(binder_pairs)),
        "contacted_hotspot_binder_residues": sorted(set(hotspot_binder_pairs)),
        "contacted_target_residues": sorted(set(target_pairs)),
        "contacted_hotspot_residues": unique_hotspot_residues,
        "binder_target_contact_pairs": sorted(
            binder_target_contact_pairs,
            key=lambda item: (item["binder_residue"], item["target_residue"]),
        ),
        "binder_hotspot_contact_pairs": [
            item for item in binder_target_contact_pairs if item["is_hotspot"]
        ],
    }


def _max_homopolymer(sequence: str) -> int:
    maximum = 0
    current = 0
    previous = ""
    for residue in sequence:
        if residue == previous:
            current += 1
        else:
            previous = residue
            current = 1
        maximum = max(maximum, current)
    return maximum


def _developability(sequence: str, hydrophobic_max: float) -> Dict[str, Any]:
    length = max(1, len(sequence))
    hydrophobic_fraction = sum(aa in HYDROPHOBIC_AA for aa in sequence) / length
    charged_fraction = sum(aa in CHARGED_AA for aa in sequence) / length
    cysteine_count = sequence.count("C")
    max_homopolymer = _max_homopolymer(sequence)
    hydrophobic_score = _clamp01(
        (hydrophobic_max - hydrophobic_fraction) / max(0.01, hydrophobic_max - 0.30)
    )
    charge_score = 1.0 - _clamp01(abs(charged_fraction - 0.24) / 0.24)
    cysteine_score = 1.0 if cysteine_count <= 2 else _clamp01(1.0 - (cysteine_count - 2) / 4.0)
    repeat_score = 1.0 if max_homopolymer <= 3 else _clamp01(1.0 - (max_homopolymer - 3) / 4.0)
    score = 0.45 * hydrophobic_score + 0.25 * charge_score + 0.15 * cysteine_score + 0.15 * repeat_score
    return {
        "score": _clamp01(score),
        "hydrophobic_fraction": hydrophobic_fraction,
        "charged_fraction": charged_fraction,
        "cysteine_count": cysteine_count,
        "max_homopolymer": max_homopolymer,
        "hydrophobic_max": hydrophobic_max,
        "hydrophobic_pass": hydrophobic_fraction <= hydrophobic_max,
    }


def _term(
    context: EvaluatorContext,
    *,
    name: str,
    category: str,
    score: float,
    default_weight: float,
    details: Mapping[str, Any],
    available: bool = True,
    warnings: Optional[Sequence[str]] = None,
) -> ScoreTerm:
    return ScoreTerm(
        name=name,
        category=category,
        score=_clamp01(score),
        weight=_weight(context, f"eval_{name}", default_weight),
        details=dict(details),
        warnings=[str(item) for item in (warnings or [])],
        backend=PLUGIN_NAME,
        available=bool(available),
    )


class ASYNHotspotBinderPlugin:
    """Score fixed-target binder candidates using same-protocol structure states."""

    name = PLUGIN_NAME

    def score_terms(self, context: EvaluatorContext) -> list[ScoreTerm]:
        sequences = _sequences(context.out)
        target = sequences.get("T", "")
        binder = sequences.get("B", "")
        target_ok = target == TARGET_SEQUENCE
        binder_ok = len(binder) == 50 and set(binder).issubset(CANONICAL_AA)

        states = _states(context.structure)
        bound = states.get(STATE_BOUND)
        binder_confidence = _binder_holo_confidence(bound)
        binder_holo_plddt = binder_confidence["plddt"]
        binder_holo_floor = binder_confidence["node_min"]
        bound_iptm = _state_scalar(bound, "iptm")
        interface_score, interface_available, interface_details = _objective_score(
            context.structure, "binder_ASYN_interface"
        )
        evidence_complete = (
            bound is not None
            and binder_holo_plddt is not None
            and bound_iptm is not None
            and interface_available
        )

        holo_target = max(88.0, _config_float(context, "asyn_holo_binder_plddt_target", 88.0))
        holo_score = _clamp01((binder_holo_plddt or 0.0) / holo_target)
        holo_floor_target = max(1e-8, _config_float(context, "asyn_holo_binder_plddt_floor_target", 65.0))
        holo_floor_score = _clamp01((binder_holo_floor or 0.0) / holo_floor_target)
        iptm_target = max(1e-8, _config_float(context, "asyn_iptm_target", 0.65))
        iptm_score = _clamp01((bound_iptm or 0.0) / iptm_target)
        combined_interface_score = (
            0.70 * interface_score + 0.30 * iptm_score
            if interface_available and bound_iptm is not None
            else 0.0
        )

        localization = _hotspot_localization(
            bound,
            proximity_decay_residues=_config_float(
                context, "asyn_hotspot_proximity_decay_residues", 12.0
            ),
            hotspot_residue_target=_config_float(
                context, "asyn_hotspot_contact_residue_target", 6.0
            ),
        )
        localization_score = _clamp01(localization.get("score", 0.0))
        hotspot_contact_min_unique = max(
            1,
            int(_config_float(context, "asyn_hotspot_contact_min_unique", 1.0)),
        )
        hotspot_contact_count = int(localization.get("unique_hotspot_residue_count", 0))
        hotspot_contact_pass = hotspot_contact_count >= hotspot_contact_min_unique
        hydrophobic_max = _config_float(context, "asyn_hydrophobic_fraction_max", 0.50)
        developability = _developability(binder, hydrophobic_max) if binder else {
            "score": 0.0,
            "hydrophobic_fraction": 1.0,
            "hydrophobic_pass": False,
            "reason": "binder sequence missing",
        }

        clash_count = interface_details.get("clash_count")
        try:
            clash_number = float(clash_count)
        except (TypeError, ValueError):
            clash_number = None
        if clash_number is not None and not math.isfinite(clash_number):
            clash_number = None
        clash_budget = _config_float(context, "asyn_interface_clash_budget", 3.0)
        clash_available = clash_number is not None
        clash_pass = clash_available and float(clash_number) <= clash_budget
        clash_score = (
            _clamp01(1.0 - float(clash_number) / max(1.0, clash_budget + 1.0))
            if clash_available
            else 0.0
        )

        return [
            _term(
                context,
                name="asyn_target_sequence_integrity",
                category="target_guardrail",
                score=1.0 if target_ok else 0.0,
                default_weight=0.0,
                details={
                    "state": "preserve",
                    "hard_gate": True,
                    "hard_gate_min_score": 1.0,
                    "dimension": "correctness",
                    "expected_length": 140,
                    "observed_length": len(target),
                    "exact_match": target_ok,
                },
                available=bool(target),
                warnings=[] if target_ok else ["ASYN target sequence changed or is missing"],
            ),
            _term(
                context,
                name="asyn_binder_sequence_validity",
                category="binder_guardrail",
                score=1.0 if binder_ok else 0.0,
                default_weight=0.0,
                details={
                    "state": "preserve",
                    "hard_gate": True,
                    "hard_gate_min_score": 1.0,
                    "dimension": "correctness",
                    "expected_length": 50,
                    "observed_length": len(binder),
                    "noncanonical_residues": sorted(set(binder) - CANONICAL_AA),
                },
                available=bool(binder),
                warnings=[] if binder_ok else ["binder must contain exactly 50 canonical amino acids"],
            ),
            _term(
                context,
                name="asyn_state_evidence_complete",
                category="evidence_guardrail",
                score=1.0 if evidence_complete else 0.0,
                default_weight=0.0,
                details={
                    "state": "preserve",
                    "hard_gate": True,
                    "hard_gate_min_score": 1.0,
                    "dimension": "correctness",
                    "required_states": [STATE_BOUND],
                    "observed_states": sorted(states),
                    "binder_holo_plddt_available": binder_holo_plddt is not None,
                    "bound_iptm_available": bound_iptm is not None,
                    "interface_objective_available": interface_available,
                },
                available=evidence_complete,
                warnings=[] if evidence_complete else ["required same-protocol structure state evidence is incomplete"],
            ),
            _term(
                context,
                name="asyn_binder_holo_confidence",
                category="binder_foldability",
                score=holo_score,
                default_weight=0.20,
                details={"state": "positive", "plddt": binder_holo_plddt,
                         "target": holo_target, "source": binder_confidence["source"],
                         "node_means": binder_confidence["node_means"], "required": True},
                available=binder_holo_plddt is not None,
            ),
            _term(
                context,
                name="asyn_binder_holo_confidence_floor",
                category="binder_foldability",
                score=holo_floor_score,
                default_weight=0.15,
                details={"state": "positive", "node_plddt_min": binder_holo_floor,
                         "target": holo_floor_target,
                         "node_means": binder_confidence["node_means"], "required": True},
                available=binder_holo_floor is not None,
            ),
            _term(
                context,
                name="asyn_interface_quality",
                category="hotspot_interface",
                score=combined_interface_score,
                default_weight=0.20,
                details={
                    "state": "positive",
                    "interface_objective_score": interface_score,
                    "bound_iptm": bound_iptm,
                    "iptm_target": iptm_target,
                    "interface_details": interface_details,
                    "required": True,
                },
                available=interface_available and bound_iptm is not None,
            ),
            _term(
                context,
                name="asyn_hotspot_localization",
                category="hotspot_localization",
                score=localization_score,
                default_weight=0.45,
                details={"state": "positive", "required": True, **localization},
                available=bool(localization.get("available")),
                warnings=[] if localization.get("available") else ["residue-pair evidence is unavailable for hotspot localization"],
            ),
            _term(
                context,
                name="asyn_hotspot_contact_guard",
                category="hotspot_guardrail",
                score=1.0 if hotspot_contact_pass else 0.0,
                default_weight=0.0,
                details={
                    "state": "preserve",
                    "hard_gate": True,
                    "hard_gate_min_score": 1.0,
                    "dimension": "correctness",
                    "unique_hotspot_residue_count": hotspot_contact_count,
                    "minimum": hotspot_contact_min_unique,
                    "contacted_hotspot_residues": localization.get(
                        "contacted_hotspot_residues", []
                    ),
                },
                available=bool(localization.get("available")),
                warnings=[]
                if hotspot_contact_pass
                else ["no direct binder contact to ASYN residues 99-140"],
            ),
            _term(
                context,
                name="asyn_binder_developability",
                category="sequence_developability",
                score=float(developability.get("score", 0.0)),
                default_weight=0.15,
                details={"state": "preserve", **developability},
                available=bool(binder),
            ),
            _term(
                context,
                name="asyn_binder_hydrophobic_guard",
                category="sequence_guardrail",
                score=1.0 if developability.get("hydrophobic_pass") else 0.0,
                default_weight=0.0,
                details={
                    "state": "preserve",
                    "hard_gate": True,
                    "hard_gate_min_score": 1.0,
                    "dimension": "correctness",
                    "hydrophobic_fraction": developability.get("hydrophobic_fraction"),
                    "maximum": hydrophobic_max,
                },
                available=bool(binder),
                warnings=[] if developability.get("hydrophobic_pass") else ["binder hydrophobic fraction exceeds the case limit"],
            ),
            _term(
                context,
                name="asyn_interface_clash_guard",
                category="interface_guardrail",
                score=clash_score if clash_pass else 0.0,
                default_weight=0.0,
                details={
                    "state": "preserve",
                    "hard_gate": True,
                    "hard_gate_min_score": 0.01,
                    "dimension": "correctness",
                    "clash_count": clash_number,
                    "clash_budget": clash_budget,
                },
                available=clash_available,
                warnings=[] if clash_pass else ["interface clash evidence is missing or exceeds the budget"],
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
    """CPU-only sequence contract used by project preflight.

    This entry point intentionally makes no structure or binding claim. Formal
    search scoring goes through :class:`ASYNHotspotBinderPlugin` after structure
    evidence is attached to the candidate.
    """

    _ = compiled, design_state, masks, score_config
    sequences = _sequences(out)
    templates = template_seqs if isinstance(template_seqs, Mapping) else {}
    fixed = fixed_residues if isinstance(fixed_residues, Mapping) else {}
    reasons: list[str] = []
    for chain_id, template in sorted(templates.items(), key=lambda item: str(item[0])):
        chain = str(chain_id)
        sequence = sequences.get(chain, "")
        if not sequence:
            reasons.append(f"missing_chain:{chain}")
        elif len(sequence) != len(str(template)):
            reasons.append(f"length_mismatch:{chain}")
    for chain_id, assignments in sorted(fixed.items(), key=lambda item: str(item[0])):
        chain = str(chain_id)
        sequence = sequences.get(chain, "")
        if not isinstance(assignments, Mapping):
            continue
        for raw_position, expected in assignments.items():
            position = int(raw_position)
            actual = sequence[position] if 0 <= position < len(sequence) else None
            if actual != str(expected):
                reasons.append(f"fixed_residue_modified:{chain}:{position}")
    binder = sequences.get("B", "")
    if len(binder) != 50 or not set(binder).issubset(CANONICAL_AA):
        reasons.append("binder_sequence_invalid:B")
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
                "provider": "asyn_preflight_cpu",
                "state": "preserve",
                "name": "fixed_residue_integrity",
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


def register_asyn_hotspot_binder_plugin() -> None:
    """Register the case plugin from an explicit trusted launcher."""

    register_plugin(
        EvaluatorPluginSpec(
            name=PLUGIN_NAME,
            factory="asyn_binder_evaluator:ASYNHotspotBinderPlugin",
            weight_fields=(
                "eval_asyn_target_sequence_integrity",
                "eval_asyn_binder_sequence_validity",
                "eval_asyn_state_evidence_complete",
                "eval_asyn_binder_holo_confidence",
                "eval_asyn_binder_holo_confidence_floor",
                "eval_asyn_interface_quality",
                "eval_asyn_hotspot_localization",
                "eval_asyn_hotspot_contact_guard",
                "eval_asyn_binder_developability",
                "eval_asyn_binder_hydrophobic_guard",
                "eval_asyn_interface_clash_guard",
            ),
            config_fields={
                "holo_binder_plddt_target": PluginConfigField(kind="float", runtime_key="asyn_holo_binder_plddt_target"),
                "holo_binder_plddt_floor_target": PluginConfigField(kind="float", runtime_key="asyn_holo_binder_plddt_floor_target"),
                "iptm_target": PluginConfigField(kind="float", runtime_key="asyn_iptm_target"),
                "hydrophobic_fraction_max": PluginConfigField(kind="float", runtime_key="asyn_hydrophobic_fraction_max"),
                "interface_clash_budget": PluginConfigField(kind="float", runtime_key="asyn_interface_clash_budget"),
                "hotspot_proximity_decay_residues": PluginConfigField(kind="float", runtime_key="asyn_hotspot_proximity_decay_residues"),
                "hotspot_contact_residue_target": PluginConfigField(kind="float", runtime_key="asyn_hotspot_contact_residue_target"),
                "hotspot_contact_min_unique": PluginConfigField(kind="float", runtime_key="asyn_hotspot_contact_min_unique"),
            },
        ),
        replace=True,
    )


register_asyn_hotspot_binder_plugin()


__all__ = [
    "ASYNHotspotBinderPlugin",
    "evaluate_candidate",
    "register_asyn_hotspot_binder_plugin",
]
