

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from astevolve.core.amino_acids import AA, CHARGED, HYDROPHOBIC
from astevolve.search.config import SAConfig


def _memory_node_block(internal_memory: Optional[Dict[str, Any]], section: str, node_name: str) -> Dict[str, Any]:
    if not internal_memory:
        return {}
    block = internal_memory.get(section, {})
    if not isinstance(block, dict):
        return {}
    for key in (node_name, node_name.lower(), node_name.casefold()):
        val = block.get(key)
        if isinstance(val, dict):
            return val
    return {}


def _window_priority_map(internal_memory: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not internal_memory:
        return {}
    windows = internal_memory.get("optimization_windows", {})
    editable = windows.get("editable_windows", {}) if isinstance(windows, dict) else {}
    priority = {}
    weights = {
        "highest_priority": 2.8,
        "medium_priority": 1.6,
        "low_priority": 0.8,
    }
    for group, weight in weights.items():
        for item in editable.get(group, []) or []:
            if isinstance(item, dict) and "node" in item:
                priority[str(item["node"])] = weight

    protected = windows.get("protected_windows", {}) if isinstance(windows, dict) else {}
    for name in protected.get("strongly_protected", []) or []:
        priority[str(name)] = min(priority.get(str(name), 1.0), 0.05)
    for name in protected.get("conditionally_protected", []) or []:
        priority[str(name)] = min(priority.get(str(name), 1.0), 0.35)
    return priority


def _node_policy(cfg: SAConfig, seg: Any) -> Dict[str, Any]:
    policies = cfg.node_edit_policies or {}
    if not isinstance(policies, dict):
        return {}
    for key in (seg.name, seg.name.lower(), seg.name.casefold()):
        val = policies.get(key)
        if isinstance(val, dict):
            return val
    return {}


def _policy_float(policy: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(policy.get(key, default))
    except (TypeError, ValueError):
        return default


def _policy_int(policy: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(round(float(policy.get(key, default))))
    except (TypeError, ValueError):
        return default


def _segment_prior(
    seg: Any,
    internal_memory: Optional[Dict[str, Any]],
    node_policy: Optional[Dict[str, Any]] = None,
) -> float:


    priority = _window_priority_map(internal_memory).get(seg.name, 1.0)
    if seg.kind == "cdr":
        priority *= 1.25
    elif seg.kind == "linker":
        priority *= 0.85
    elif seg.kind == "framework":
        priority *= 0.55

    adaptive = internal_memory.get("adaptive_memory", {}) if internal_memory else {}
    motif_memory = adaptive.get("motif_memory", {}) if isinstance(adaptive, dict) else {}
    motif = {}
    for key in (seg.name, seg.name.lower(), seg.name.casefold()):
        if isinstance(motif_memory.get(key), dict):
            motif = motif_memory[key]
            break
    if motif:
        confidence = float(motif.get("confidence") or 0.0)
        support = float(motif.get("support_count") or 0.0)
        priority *= 1.0 + min(1.0, confidence) + min(0.5, 0.03 * support)

    node_stats = adaptive.get("node_level_statistics", {}) if isinstance(adaptive, dict) else {}
    stats = {}
    for key in (seg.name, seg.name.lower(), seg.name.casefold()):
        if isinstance(node_stats.get(key), dict):
            stats = node_stats[key]
            break
    if stats:
        try:
            priority *= max(0.5, min(1.75, float(stats.get("priority_multiplier", 1.0))))
        except (TypeError, ValueError):
            pass

    if isinstance(node_policy, dict) and node_policy:
        priority *= max(0.01, _policy_float(node_policy, "priority_boost", 1.0))

    return max(0.01, float(priority))


def _aa_class_members(class_name: str) -> str:
    table = {
        "aromatic": "YWHF",
        "polar_uncharged": "STNQY",
        "contextual_charge": "RKHDE",
        "flexible_small": "GSA",
        "positive": "RKH",
        "negative": "DE",
        "hydrophobic": "AILMFWVY",
        "charged": "RKHDE",
        "small": "GAS",
        "turn_loop": "GSPNDT",
        "calcium_ligand": "DENQSTG",
    }
    return table.get(class_name, "")


def _apply_design_residue_prior(
    weights: np.ndarray,
    prior: Dict[str, Any],
    policy_weight: float,
) -> np.ndarray:
    if not prior:
        return weights

    aa_index = {aa: i for i, aa in enumerate(AA)}
    try:
        confidence = float(prior.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0
    strength = max(0.0, float(policy_weight)) * max(0.0, min(1.0, confidence))
    if strength <= 0:
        return weights

    for aa in prior.get("favored_residues", []) or []:
        aa = str(aa)
        if aa in aa_index:


            weights[aa_index[aa]] *= math.exp(0.9 * strength)

    for class_name in prior.get("favored_residue_classes", []) or []:
        for aa in _aa_class_members(str(class_name)):
            if aa in aa_index:
                weights[aa_index[aa]] *= 1.0 + 1.2 * strength

    for aa in prior.get("disfavored_residues", []) or []:
        aa = str(aa)
        if aa in aa_index:
            weights[aa_index[aa]] *= math.exp(-1.1 * strength)

    for class_name in prior.get("disfavored_residue_classes", []) or []:
        for aa in _aa_class_members(str(class_name)):
            if aa in aa_index:
                weights[aa_index[aa]] *= max(0.05, 1.0 - 0.65 * strength)

    aa_weights = prior.get("aa_weights", {})
    if isinstance(aa_weights, dict):
        raw = np.zeros(len(AA), dtype=float)
        for aa, value in aa_weights.items():
            aa = str(aa)
            if aa not in aa_index:
                continue
            try:
                raw[aa_index[aa]] = max(0.0, float(value))
            except (TypeError, ValueError):
                continue
        if raw.sum() > 0:
            freq = raw / raw.sum()
            uniform = 1.0 / len(AA)
            multipliers = np.clip(freq / uniform, 0.20, 5.0)
            weights *= np.power(multipliers, 0.45 * strength)

    return weights


def _policy_residue_prior(node_policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(node_policy, dict) or not node_policy:
        return {}
    prior: Dict[str, Any] = {
        "confidence": node_policy.get("confidence", node_policy.get("policy_weight", 1.0)),
        "favored_residues": node_policy.get("favored_residues", []),
        "favored_residue_classes": node_policy.get("favored_residue_classes", []),
        "disfavored_residues": node_policy.get("disfavored_residues", []),
        "disfavored_residue_classes": node_policy.get("disfavored_residue_classes", []),
        "aa_weights": node_policy.get("aa_weights", {}),
    }
    return prior


def _aa_weights_for_segment(
    seg: Any,
    internal_memory: Optional[Dict[str, Any]],
    node_policy: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    weights = np.ones(len(AA), dtype=float)
    aa_index = {aa: i for i, aa in enumerate(AA)}

    if "C" in aa_index:
        weights[aa_index["C"]] *= 0.05

    if seg.kind == "linker":
        for aa in "GS":
            weights[aa_index[aa]] *= 6.0
        for aa in "AT":
            weights[aa_index[aa]] *= 1.8
        for aa in HYDROPHOBIC:
            weights[aa_index[aa]] *= 0.25
        for aa in CHARGED:
            weights[aa_index[aa]] *= 0.45
    elif seg.kind == "cdr":
        for aa in "YWHNQSTRDE":
            weights[aa_index[aa]] *= 2.0
        for aa in "ILMFV":
            weights[aa_index[aa]] *= 0.75
    elif seg.kind == "framework":
        for aa in "GSPNQ":
            weights[aa_index[aa]] *= 1.2
        for aa in "CWF":
            weights[aa_index[aa]] *= 0.35

    windows = internal_memory.get("optimization_windows", {}) if internal_memory else {}
    node_bias = windows.get("node_specific_bias", {}) if isinstance(windows, dict) else {}
    bias = {}
    for key in (seg.name, seg.name.lower(), seg.name.casefold()):
        if isinstance(node_bias.get(key), dict):
            bias = node_bias[key]
            break
    for class_name in bias.get("preferred_residue_classes", []) if bias else []:
        for aa in _aa_class_members(str(class_name)):
            if aa in aa_index:
                weights[aa_index[aa]] *= 1.6

    adaptive = internal_memory.get("adaptive_memory", {}) if internal_memory else {}
    motif_memory = adaptive.get("motif_memory", {}) if isinstance(adaptive, dict) else {}
    motif = {}
    for key in (seg.name, seg.name.lower(), seg.name.casefold()):
        if isinstance(motif_memory.get(key), dict):
            motif = motif_memory[key]
            break
    for aa in motif.get("enriched_residues", []) if motif else []:
        if aa in aa_index:
            weights[aa_index[aa]] *= 2.5
    confidence = float(motif.get("confidence") or 0.0) if motif else 0.0
    class_boost = 1.0 + 1.1 * min(1.0, confidence)
    for class_name in motif.get("favored_classes", []) if motif else []:
        for aa in _aa_class_members(str(class_name)):
            if aa in aa_index:
                weights[aa_index[aa]] *= class_boost

    policy_prior = _policy_residue_prior(node_policy)
    if policy_prior:
        weights = _apply_design_residue_prior(
            weights,
            policy_prior,
            _policy_float(node_policy or {}, "policy_weight", 1.0),
        )

    weights = np.maximum(weights, 1e-6)
    return weights / weights.sum()


def _position_residue_rule(node_policy: Optional[Dict[str, Any]], seg: Any, pos: int) -> Dict[str, Any]:
    if not isinstance(node_policy, dict) or not node_policy:
        return {}
    rules = node_policy.get("position_residue_rules")
    if not isinstance(rules, dict):
        return {}
    keys = [str(int(pos)), int(pos)]
    try:
        rel = int(pos) - int(getattr(seg, "start", 0))
        keys.extend([str(rel), rel])
    except Exception:
        pass
    for key in keys:
        rule = rules.get(key)
        if isinstance(rule, dict):
            return dict(rule)
    return {}


def _aa_weights_for_position(base_weights: np.ndarray, node_policy: Optional[Dict[str, Any]], seg: Any, pos: int) -> np.ndarray:
    rule = _position_residue_rule(node_policy, seg, int(pos))
    if not rule:
        return base_weights
    weights = np.array(base_weights, dtype=float, copy=True)
    weights = _apply_design_residue_prior(weights, rule, _policy_float(rule, "policy_weight", 1.2))
    weights = np.maximum(weights, 1e-6)
    return weights / weights.sum()
