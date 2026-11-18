

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from astevolve.evaluation.objectives.interface_metrics import (
    _interface_left_residue_names,
    _interface_strength,
)
from astevolve.evaluation.objectives.mechanistic import _score_mechanistic_transition
from astevolve.evaluation.objectives.region_metrics import (
    _score_conf_change,
    _score_confidence,
    _score_region_confidence,
    _score_region_confidence_floor,
)
from astevolve.evaluation.objectives.support import (
    _aa_class_members,
    _clamp01,
    _name_list,
    _safe_float,
    _state_names,
)


ObjectiveScorer = Callable[
    [Dict[str, Dict[str, Any]], Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]],
    Tuple[float, Dict[str, Any], List[str]],
]


OBJECTIVE_REGISTRY: Dict[str, ObjectiveScorer] = {}


def _strength_for_state(
    by_state: Dict[str, Dict[str, Any]],
    state_name: str,
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    return _interface_strength(by_state.get(state_name, {}), spec, compiled, design_state)


def _state_pair_names(spec: Dict[str, Any]) -> Tuple[str, str]:
    positive = str(
        spec.get("positive_state")
        or spec.get("on_state")
        or spec.get("state_a")
        or ""
    )
    negative = str(
        spec.get("negative_state")
        or spec.get("off_state")
        or spec.get("state_b")
        or ""
    )
    states = _state_names(spec)
    if (not positive or not negative) and len(states) >= 2:
        positive = positive or states[0]
        negative = negative or states[1]
    return positive, negative


def _register_objective(*names: str) -> Callable[[ObjectiveScorer], ObjectiveScorer]:


    def decorator(func: ObjectiveScorer) -> ObjectiveScorer:
        for name in names:
            key = str(name).strip().lower()
            if key:
                OBJECTIVE_REGISTRY[key] = func
        return func
    return decorator


@_register_objective("interface_delta", "delta_interface", "state_interface_delta")
def _objective_interface_delta(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    positive_state, negative_state = _state_pair_names(spec)
    if not positive_state or not negative_state:
        return 0.0, {}, ["interface_delta needs positive_state/negative_state or a two-state states list"]

    positive_spec = dict(spec)
    negative_spec = dict(spec)
    positive_pair = spec.get("positive_pair", spec.get("on_pair"))
    negative_pair = spec.get("negative_pair", spec.get("off_pair"))
    if positive_pair is not None:
        positive_spec["pair"] = positive_pair
    if negative_pair is not None:
        negative_spec["pair"] = negative_pair
    for source_key, target_key in [
        ("positive_left_region", "left_region"),
        ("positive_right_region", "right_region"),
        ("positive_contact_target", "contact_target"),
        ("positive_residue_pair_target", "residue_pair_target"),
        ("positive_coverage_target", "coverage_target"),
        ("negative_left_region", "left_region"),
        ("negative_right_region", "right_region"),
        ("negative_contact_target", "contact_target"),
        ("negative_residue_pair_target", "residue_pair_target"),
        ("negative_coverage_target", "coverage_target"),
    ]:
        if source_key not in spec:
            continue
        if source_key.startswith("positive_"):
            positive_spec[target_key] = spec[source_key]
        else:
            negative_spec[target_key] = spec[source_key]

    positive_strength, positive_details, warnings_a = _strength_for_state(
        by_state, positive_state, positive_spec, compiled, design_state
    )
    negative_strength, negative_details, warnings_b = _strength_for_state(
        by_state, negative_state, negative_spec, compiled, design_state
    )
    direction = str(spec.get("direction", "decrease")).lower()
    if direction in {"increase", "up", "higher"}:
        delta = negative_strength - positive_strength
        favorable_low = positive_strength
        favorable_high = negative_strength
    else:
        delta = positive_strength - negative_strength
        favorable_low = negative_strength
        favorable_high = positive_strength

    min_delta = float(spec.get("min_delta", 0.05))
    target_delta = max(min_delta + 1e-6, float(spec.get("target_delta", 0.35)))
    delta_score = _clamp01((delta - min_delta) / (target_delta - min_delta))
    high_score = _clamp01(favorable_high)
    low_score = 1.0 - _clamp01(favorable_low)
    score = (0.55 * delta_score) + (0.25 * high_score) + (0.20 * low_score)

    contact_a = _safe_float(positive_details.get("contact_count"), 0.0) or 0.0
    contact_b = _safe_float(negative_details.get("contact_count"), 0.0) or 0.0
    contact_delta = contact_b - contact_a if direction in {"increase", "up", "higher"} else contact_a - contact_b
    contact_delta_target = _safe_float(spec.get("contact_delta_target"), None)
    contact_delta_score = None
    if contact_delta_target is not None:
        contact_delta_score = _clamp01(contact_delta / max(1e-6, float(contact_delta_target)))
        score = 0.75 * score + 0.25 * contact_delta_score

    return _clamp01(score), {
        "states": [positive_state, negative_state],
        "direction": direction,
        "positive_state": positive_state,
        "negative_state": negative_state,
        "positive_strength": positive_strength,
        "negative_strength": negative_strength,
        "strength_delta": delta,
        "min_delta": min_delta,
        "target_delta": target_delta,
        "delta_score": delta_score,
        "contact_count_delta": contact_delta,
        "contact_delta_target": contact_delta_target,
        "contact_delta_score": contact_delta_score,
        "positive_details": positive_details,
        "negative_details": negative_details,
    }, warnings_a + warnings_b


@_register_objective("competition_interface", "interface_competition", "competitive_binding")
def _objective_competition_interface(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    state_name = (_state_names(spec) or [""])[0]
    if not state_name:
        return 0.0, {}, ["competition_interface needs state"]
    winner_spec = dict(spec.get("winner") or spec.get("on") or {})
    loser_spec = dict(spec.get("loser") or spec.get("off") or {})
    if not winner_spec:
        winner_spec = {
            "pair": spec.get("winner_pair", spec.get("on_pair")),
            "left_region": spec.get("winner_left_region", spec.get("on_left_region")),
            "right_region": spec.get("winner_right_region", spec.get("on_right_region")),
            "contact_target": spec.get("winner_contact_target", spec.get("contact_target")),
            "residue_pair_target": spec.get("winner_residue_pair_target", spec.get("residue_pair_target")),
            "coverage_target": spec.get("winner_coverage_target", spec.get("coverage_target")),
        }
    if not loser_spec:
        loser_spec = {
            "pair": spec.get("loser_pair", spec.get("off_pair")),
            "left_region": spec.get("loser_left_region", spec.get("off_left_region")),
            "right_region": spec.get("loser_right_region", spec.get("off_right_region")),
            "contact_target": spec.get("loser_contact_target", spec.get("contact_target")),
            "residue_pair_target": spec.get("loser_residue_pair_target", spec.get("residue_pair_target")),
            "coverage_target": spec.get("loser_coverage_target", spec.get("coverage_target")),
        }
    winner_spec = {**spec, **{k: v for k, v in winner_spec.items() if v is not None}}
    loser_spec = {**spec, **{k: v for k, v in loser_spec.items() if v is not None}}
    winner_strength, winner_details, warnings_a = _interface_strength(
        by_state.get(state_name, {}), winner_spec, compiled, design_state
    )
    loser_strength, loser_details, warnings_b = _interface_strength(
        by_state.get(state_name, {}), loser_spec, compiled, design_state
    )
    score = (0.55 * _clamp01(winner_strength)) + (0.45 * (1.0 - _clamp01(loser_strength)))
    return _clamp01(score), {
        "state": state_name,
        "winner_strength": winner_strength,
        "loser_strength": loser_strength,
        "winner_details": winner_details,
        "loser_details": loser_details,
    }, warnings_a + warnings_b


@_register_objective("epitope_specificity", "hotspot_specificity", "interface_specificity")
def _objective_epitope_specificity(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    state_name = (_state_names(spec) or [""])[0]
    if not state_name:
        return 0.0, {}, ["epitope_specificity needs state"]

    target_spec = dict(spec)
    if spec.get("target_region") is not None:
        target_spec["right_region"] = spec.get("target_region")
    target_strength, target_details, warnings = _interface_strength(
        by_state.get(state_name, {}), target_spec, compiled, design_state
    )
    full_contact = max(0.0, _safe_float(target_details.get("full_contact_count"), 0.0) or 0.0)
    target_contact = max(0.0, _safe_float(target_details.get("contact_count"), 0.0) or 0.0)
    off_contact = max(0.0, _safe_float(target_details.get("off_target_contact_count"), full_contact - target_contact) or 0.0)
    specificity_ratio = target_contact / max(1.0, full_contact)
    target_fraction_goal = float(spec.get("target_fraction_goal", spec.get("specificity_target", 0.45)))
    specificity_score = _clamp01(specificity_ratio / max(1e-6, target_fraction_goal))

    avoid_score = None
    avoid_details: Dict[str, Any] = {}
    avoid_region = spec.get("avoid_region", spec.get("avoid_regions"))
    if avoid_region is not None:
        avoid_spec = dict(spec)
        avoid_spec["right_region"] = avoid_region
        avoid_strength, avoid_details, avoid_warnings = _interface_strength(
            by_state.get(state_name, {}), avoid_spec, compiled, design_state
        )
        avoid_score = 1.0 - _clamp01(avoid_strength)
        warnings.extend(avoid_warnings)

    score = (0.55 * _clamp01(target_strength)) + (0.35 * specificity_score)
    score += 0.10 * (avoid_score if avoid_score is not None else (1.0 - _clamp01(off_contact / max(1.0, float(spec.get("off_target_contact_tolerance", 120.0))))))
    return _clamp01(score), {
        "state": state_name,
        "target_region": spec.get("target_region", spec.get("right_region")),
        "avoid_region": avoid_region,
        "target_strength": target_strength,
        "target_contact_count": target_contact,
        "full_contact_count": full_contact,
        "off_target_contact_count": off_contact,
        "specificity_ratio": specificity_ratio,
        "target_fraction_goal": target_fraction_goal,
        "specificity_score": specificity_score,
        "avoid_score": avoid_score,
        "target_details": target_details,
        "avoid_details": avoid_details,
    }, warnings


@_register_objective("ligand_pocket_pharmacophore", "pocket_pharmacophore", "ion_pocket_pharmacophore")
def _objective_ligand_pocket_pharmacophore(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    state_name = (_state_names(spec) or [""])[0]
    if not state_name:
        return 0.0, {}, ["ligand_pocket_pharmacophore needs state"]
    state_result = by_state.get(state_name, {})
    strength, strength_details, warnings = _interface_strength(state_result, spec, compiled, design_state)
    residues, residue_metrics, residue_warnings = _interface_left_residue_names(state_result, spec, compiled, design_state)
    warnings.extend(residue_warnings)

    classes = spec.get("desired_residue_classes", {}) or {}
    if not isinstance(classes, dict):
        classes = {}
    class_scores: Dict[str, Any] = {}
    for class_name, members in classes.items():
        allowed = set("".join(_name_list(members)).upper())
        if not allowed:
            allowed = set(_aa_class_members(str(class_name)))
        hits = sum(1 for aa in residues if aa[:1] in allowed or aa in allowed)
        min_hits = int(_safe_float((spec.get("min_class_hits", {}) or {}).get(class_name), 1) or 1) if isinstance(spec.get("min_class_hits"), dict) else 1
        class_scores[str(class_name)] = {
            "allowed": sorted(allowed),
            "hits": hits,
            "min_hits": min_hits,
            "score": _clamp01(hits / max(1, min_hits)),
        }
    chemistry_score = (
        float(sum(item["score"] for item in class_scores.values()) / len(class_scores))
        if class_scores
        else 0.5
    )
    score = (0.60 * _clamp01(strength)) + (0.40 * chemistry_score)
    return _clamp01(score), {
        "state": state_name,
        "interface_strength": strength,
        "contact_residue_count": len(residues),
        "contact_residues": residues[:40],
        "chemistry_score": chemistry_score,
        "class_scores": class_scores,
        "strength_details": strength_details,
        "residue_metrics": residue_metrics,
    }, warnings


@_register_objective("confidence", "structural_confidence")
def _objective_confidence(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    return _score_confidence(by_state, spec)


@_register_objective("region_confidence", "node_confidence", "motif_confidence")
def _objective_region_confidence(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    return _score_region_confidence(by_state, spec, design_state)


@_register_objective("region_confidence_floor", "node_confidence_floor", "motif_confidence_floor")
def _objective_region_confidence_floor(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    return _score_region_confidence_floor(by_state, spec, design_state)


@_register_objective("interface_on", "preserve_interface", "bind", "binding")
def _objective_interface_on(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    state_name = (_state_names(spec) or [""])[0]
    score, details, warnings = _interface_strength(by_state.get(state_name, {}), spec, compiled, design_state)
    return score, {"state": state_name, **details}, warnings


@_register_objective("interface_off", "disrupt_interface", "anti_bind", "anti-binding", "anti_binding")
def _objective_interface_off(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    state_name = (_state_names(spec) or [""])[0]
    score, details, warnings = _interface_strength(by_state.get(state_name, {}), spec, compiled, design_state)
    if any("could not resolve" in warning for warning in warnings):
        return 0.0, {"state": state_name, "interface_strength": score, **details}, warnings
    return 1.0 - score, {"state": state_name, "interface_strength": score, **details}, warnings


@_register_objective("conf_change", "conformational_change")
def _objective_conf_change(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    return _score_conf_change(by_state, spec, compiled, design_state)


@_register_objective("mechanistic_transition", "kinetic_path", "transition_path")
def _objective_mechanistic_transition(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    return _score_mechanistic_transition(by_state, spec, compiled, design_state)


@_register_objective("preserve_motif", "motif_on")
def _objective_motif_on(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    motif_spec = dict(spec)
    if motif_spec.get("region") or motif_spec.get("regions") or motif_spec.get("node") or motif_spec.get("nodes"):
        motif_spec.setdefault("metric", "plddt_mean")
        return _score_region_confidence(by_state, motif_spec, design_state)
    motif_spec.setdefault("metric", "node_plddt_mean")
    return _score_confidence(by_state, motif_spec)


@_register_objective("disrupt_motif", "motif_off")
def _objective_motif_off(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    score, details, warnings = _objective_motif_on(by_state, spec, compiled, design_state)
    return 1.0 - score, {"motif_confidence": score, **details}, warnings


def supported_objective_types() -> List[str]:


    return sorted(OBJECTIVE_REGISTRY)


def _score_objective(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:


    kind = str(spec.get("type") or spec.get("kind") or "").strip().lower()
    scorer = OBJECTIVE_REGISTRY.get(kind)
    if scorer is not None:
        return scorer(by_state, spec, compiled, design_state)
    return 0.0, {}, [f"unsupported objective type: {kind}"]
