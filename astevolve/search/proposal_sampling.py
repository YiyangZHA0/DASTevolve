

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from astevolve.core.amino_acids import AA
from astevolve.search.config import SAConfig
from astevolve.search.operator_registry import (
    OperatorConfigError,
    operator_supports_node_kind,
    proposal_tier_operator_weights,
    validate_operator_weights,
)
from astevolve.search.proposal_priors import _policy_float, _policy_int


MUTATION_PLAN_SCHEMA_VERSION = "ast_mutation_plan_v2"
TIER_PRIOR_WEIGHT = 0.25


def _sample_aa_from_pool(rng: np.random.Generator, pool: str, old_aa: Optional[str] = None) -> str:
    residues = [aa for aa in str(pool or "") if aa in AA]
    if not residues:
        residues = list(AA)
    for _ in range(6):
        aa = str(rng.choice(np.asarray(residues)))
        if old_aa is None or aa != old_aa:
            return aa
    return str(rng.choice(np.asarray(residues)))


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    validated = validate_operator_weights(
        weights,
        mode="node",
        context="proposal.op_weights",
        drop_zero=True,
    )
    total = float(sum(validated.values()))
    return {name: float(weight) / total for name, weight in validated.items()}


def _effective_operator_weights(
    seg: Any,
    node_policy: Dict[str, Any],
    cfg: SAConfig,
    tier: str,
) -> Tuple[Dict[str, float], Dict[str, Any]]:


    explicit_node_policy = isinstance(node_policy, dict) and "mutation_ops" in node_policy
    requested_value = node_policy.get("mutation_ops") if explicit_node_policy else cfg.mutation_ops
    requested = validate_operator_weights(
        requested_value,
        mode="node",
        context=(
            f"node_edit_policy.{seg.name}.mutation_ops"
            if explicit_node_policy
            else "sa_config.mutation_ops"
        ),
        drop_zero=True,
    )
    node_kind = str(getattr(seg, "kind", "") or "").lower()
    node_kind_filtered = {
        operator: weight
        for operator, weight in requested.items()
        if not operator_supports_node_kind(operator, node_kind)
    }
    if explicit_node_policy and node_kind_filtered:
        names = ", ".join(node_kind_filtered)
        raise OperatorConfigError(
            f"node_edit_policy.{seg.name}.mutation_ops contains operator(s) "
            f"incompatible with node kind {node_kind!r}: {names}"
        )
    policy_source = (
        "node_edit_policy.mutation_ops"
        if explicit_node_policy
        else "sa_config.mutation_ops"
    )

    policy_with_evidence = {
        operator: weight
        for operator, weight in requested.items()
        if operator not in node_kind_filtered
    }
    if not policy_with_evidence:
        raise OperatorConfigError(
            f"{policy_source} has no operator compatible with node kind {node_kind!r}"
        )
    applied_evidence_floors: Dict[str, float] = {}
    ignored_evidence_floors: Dict[str, float] = {}

    policy_normalized = _normalize_weights(policy_with_evidence)
    if tier == "frozen":
        effective: Dict[str, float] = {}
        return effective, {
            "schema_version": "operator_weight_merge.v1",
            "tier": tier,
            "node_kind": node_kind,
            "policy_source": policy_source,
            "requested": requested,
            "node_kind_filtered": node_kind_filtered,
            "policy_with_evidence": policy_with_evidence,
            "policy_normalized": policy_normalized,
            "tier_prior": {},
            "tier_prior_on_policy_support": {},
            "tier_prior_weight": 0.0,
            "applied_evidence_floors": applied_evidence_floors,
            "ignored_evidence_floors": ignored_evidence_floors,
            "merge_rule": "frozen",
            "effective": effective,
        }

    if tier == "legacy":
        tier_prior: Dict[str, float] = {}
        tier_prior_on_support: Dict[str, float] = {}
        tier_prior_weight = 0.0
        effective = dict(policy_normalized)
    else:
        tier_prior = _normalize_weights(proposal_tier_operator_weights(tier))
        overlap = {
            operator: tier_prior[operator]
            for operator in policy_normalized
            if operator in tier_prior
        }
        tier_prior_on_support = (
            _normalize_weights(overlap) if overlap else dict(policy_normalized)
        )
        tier_prior_weight = TIER_PRIOR_WEIGHT
        effective = _normalize_weights(
            {
                operator: (1.0 - tier_prior_weight) * policy_weight
                + tier_prior_weight * tier_prior_on_support.get(operator, 0.0)
                for operator, policy_weight in policy_normalized.items()
            }
        )

    return effective, {
        "schema_version": "operator_weight_merge.v1",
        "tier": tier,
        "node_kind": node_kind,
        "policy_source": policy_source,
        "requested": requested,
        "node_kind_filtered": node_kind_filtered,
        "policy_with_evidence": policy_with_evidence,
        "policy_normalized": policy_normalized,
        "tier_prior": tier_prior,
        "tier_prior_on_policy_support": tier_prior_on_support,
        "tier_prior_weight": tier_prior_weight,
        "applied_evidence_floors": applied_evidence_floors,
        "ignored_evidence_floors": ignored_evidence_floors,
        "merge_rule": "support_preserving_convex_mix_v1",
        "effective": effective,
    }


def _proposal_tier(seg: Any, node_policy: Dict[str, Any], cfg: SAConfig, rng: np.random.Generator) -> str:
    if str(cfg.proposal_engine).lower() != "contract_guided":
        return "legacy"
    action = str(node_policy.get("edit_contract_action") or "").lower()
    phase = str(node_policy.get("operator_phase") or "").lower()
    tier_mode = str(getattr(cfg, "proposal_tier_mode", "fixed_node")).lower()
    mut_rate = _policy_float(node_policy, "mutation_rate", cfg.mutation_rate)
    max_step = _policy_int(node_policy, "max_mutations_per_step", 1)
    if action == "freeze_node" or mut_rate <= 0.0 or max_step == 0:
        return "frozen"


    if action == "repair_node" or phase == "repair":
        return "repair"
    if tier_mode == "fixed_node" and phase in {"explore", "exploit"}:
        return phase
    exploit = max(0.0, float(cfg.proposal_exploit_frac))
    explore = max(0.0, float(cfg.proposal_explore_frac))
    repair = max(0.0, float(cfg.proposal_repair_frac))
    total = exploit + explore + repair
    if total > 0:
        r = float(rng.random()) * total
        if r < repair:
            return "repair"
        if r < repair + explore:
            return "explore"
    return "exploit"


def _proposal_plan(
    seg: Any,
    node_policy: Dict[str, Any],
    cfg: SAConfig,
    rng: np.random.Generator,
    mapping_action_spec: Optional[Dict[str, Any]] = None,
    forced_tier: Optional[str] = None,
) -> Dict[str, Any]:


    if forced_tier is None:
        tier = _proposal_tier(seg, node_policy, cfg, rng)
    else:
        tier = str(forced_tier).strip().lower()
        if tier not in {"exploit", "explore", "repair"}:
            raise ValueError(f"unsupported controller-forced proposal tier: {tier!r}")
        if str(cfg.proposal_engine).lower() != "contract_guided":
            raise ValueError(
                "controller-forced proposal tiers require contract_guided"
            )
    action = str(node_policy.get("edit_contract_action") or "optimize_node")
    intent = str(node_policy.get("edit_intent") or node_policy.get("role") or "")
    if not intent:
        intent = "execute the declared local edit while preserving explicit invariants"

    favored = [str(x) for x in (node_policy.get("favored_residues") or []) if str(x) in AA]
    favored_classes = [str(x) for x in (node_policy.get("favored_residue_classes") or [])]
    forbidden = [str(x) for x in (node_policy.get("disfavored_residues") or []) if str(x) in AA]

    if tier == "legacy":
        max_mut = _policy_int(node_policy, "max_mutations_per_step", cfg.exploit_max_mutations)
    elif tier == "exploit":
        max_mut = min(_policy_int(node_policy, "max_mutations_per_step", cfg.exploit_max_mutations), int(cfg.exploit_max_mutations))
    elif tier == "explore":
        max_mut = min(_policy_int(node_policy, "max_mutations_per_step", cfg.explore_max_mutations), int(cfg.explore_max_mutations))
    elif tier == "repair":
        max_mut = min(_policy_int(node_policy, "max_mutations_per_step", cfg.repair_max_mutations), int(cfg.repair_max_mutations))
    else:
        max_mut = 0

    max_mut = max(0, int(max_mut))
    min_mut = 0 if tier == "frozen" else 1
    op_weights, operator_weight_provenance = _effective_operator_weights(
        seg,
        node_policy,
        cfg,
        tier,
    )
    mapping = (
        dict(mapping_action_spec)
        if isinstance(mapping_action_spec, dict) and mapping_action_spec
        else {}
    )
    if mapping:
        expected_segment = str(mapping.get("compiled_segment_name") or "")
        if expected_segment != str(seg.name):
            raise ValueError(
                f"mapping action targets segment {expected_segment!r}, not {seg.name!r}"
            )
        mapped_operator = str(mapping.get("operator") or "")
        if mapped_operator not in op_weights:
            raise ValueError(
                f"mapping action operator {mapped_operator!r} is not enabled by "
                f"node policy {seg.name!r}"
            )
        op_weights = {mapped_operator: 1.0}
        operator_weight_provenance = {
            **operator_weight_provenance,
            "mapping_edge_restriction": {
                "edge_id": str(mapping.get("edge_id") or ""),
                "action_id": str(mapping.get("action_id") or ""),
                "operator": mapped_operator,
            },
            "effective": dict(op_weights),
        }
        raw_budget = mapping.get("budget")
        if not isinstance(raw_budget, dict):
            raise ValueError("mapping action budget must be a mapping")
        min_mut = int(raw_budget.get("min", 0))
        max_mut = int(raw_budget.get("max", 0))
    plan = {
        "schema_version": MUTATION_PLAN_SCHEMA_VERSION,
        "tier": tier,
        "tier_mode": str(
            getattr(cfg, "proposal_tier_mode", "fixed_node")
        ).lower(),
        "action": action,
        "node": seg.name,
        "node_kind": seg.kind,
        "intent": intent,
        "op_weights": op_weights,
        "operator_weight_provenance": operator_weight_provenance,
        "forced_tier_receipt": (
            {
                "schema_version": "astevolve.forced_proposal_tier.v1",
                "tier": tier,
                "authority": "compiled_candidate_wave_slot",
            }
            if forced_tier is not None
            else None
        ),
        "budget": {"min": min_mut, "max": max_mut},
        "allowed_action_budgets": dict(
            node_policy.get("allowed_action_budgets", {})
            if isinstance(node_policy.get("allowed_action_budgets"), dict)
            else {}
        ),
        "allowed_residues": favored[:16],
        "allowed_residue_classes": favored_classes[:12],
        "forbidden_residues": forbidden[:12],
        "positive_targets": node_policy.get("positive_targets", []),
        "negative_targets": node_policy.get("negative_targets", []),
    }
    if mapping:
        for key in (
            "ast_id",
            "ast_revision",
            "edge_id",
            "functional_node_id",
            "structural_node_id",
            "action_id",
            "measurement_id",
        ):
            if mapping.get(key) in (None, ""):
                raise ValueError(f"mapping action is missing required field {key!r}")
            plan[key] = mapping[key]
        plan["mapping_execution"] = "full"
        plan["budget"] = {"min": min_mut, "max": max_mut}
        plan["allowed_action_budgets"] = {
            str(mapping["operator"]): {"min": min_mut, "max": max_mut}
        }
        plan["legal_positions"] = [
            int(position) for position in mapping.get("legal_positions", [])
        ]
    return plan


def _apply_forbidden_residue_filter(weights: np.ndarray, forbidden: List[str]) -> np.ndarray:
    if not forbidden:
        return weights
    filtered = np.array(weights, dtype=float, copy=True)
    aa_index = {aa: i for i, aa in enumerate(AA)}
    for aa in forbidden:
        if aa in aa_index:
            filtered[aa_index[aa]] *= 0.03
    filtered = np.maximum(filtered, 1e-8)
    return filtered / filtered.sum()


def _node_default_motifs(seg: Any, node_policy: Optional[Dict[str, Any]]) -> List[str]:
    motifs = _policy_motifs(node_policy)
    if motifs:
        return motifs
    kind = str(getattr(seg, "kind", "") or "").lower()
    if kind == "cdr":
        return ["YYG", "GYW", "RYY", "DYY", "NSY", "STY"]
    if kind == "pocket":
        return ["DY", "EY", "YH", "DEN", "STN", "NQY"]
    return ["GS", "ST", "NQ"]


def _relative_to_abs_position(seg: Any, raw_pos: Any) -> Optional[int]:
    try:
        pos = int(raw_pos)
    except (TypeError, ValueError):
        return None
    indices = [int(x) for x in seg.indices()]
    if 0 <= pos < len(indices):
        return indices[pos]
    if pos in indices:
        return pos
    return None


def _position_sampling_probs(
    seg: Any,
    positions: List[int],
    node_policy: Optional[Dict[str, Any]],
) -> Optional[np.ndarray]:

    if not positions or not isinstance(node_policy, dict) or not node_policy:
        return None

    pos_to_idx = {int(pos): idx for idx, pos in enumerate(positions)}
    weights = np.ones(len(positions), dtype=float)

    def boost_abs(abs_pos: Optional[int], factor: Any) -> None:
        if abs_pos is None or int(abs_pos) not in pos_to_idx:
            return
        try:
            value = float(factor)
        except (TypeError, ValueError):
            value = 1.0
        weights[pos_to_idx[int(abs_pos)]] *= max(0.05, value)

    raw_position_weights = node_policy.get("position_weights", {})
    if isinstance(raw_position_weights, dict):
        for raw_pos, raw_weight in raw_position_weights.items():
            boost_abs(_relative_to_abs_position(seg, raw_pos), raw_weight)

    hotspot_weight = _policy_float(node_policy, "hotspot_weight", 2.0)
    for raw_pos in node_policy.get("hotspot_positions", []) or []:
        boost_abs(_relative_to_abs_position(seg, raw_pos), hotspot_weight)

    raw_anchor = node_policy.get("site_anchors", {})
    if isinstance(raw_anchor, dict) and seg.name in raw_anchor and isinstance(raw_anchor[seg.name], dict):
        raw_anchor = raw_anchor[seg.name]

    if isinstance(raw_anchor, dict):
        anchor_weight = _policy_float(raw_anchor, "weight", 2.0)
        for raw_pos in raw_anchor.get("relative_positions", raw_anchor.get("positions", [])) or []:
            boost_abs(_relative_to_abs_position(seg, raw_pos), anchor_weight)

        seg_indices = [int(x) for x in seg.indices()]
        for raw_range in raw_anchor.get("relative_ranges", []) or []:
            if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
                continue
            try:
                start = max(0, int(raw_range[0]))
                end = min(len(seg_indices), int(raw_range[1]))
            except (TypeError, ValueError):
                continue
            for rel_pos in range(start, max(start, end)):
                boost_abs(_relative_to_abs_position(seg, rel_pos), anchor_weight)

    weights = np.maximum(weights, 1e-8)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        return None
    probs = weights / total
    if np.allclose(probs, np.ones(len(probs)) / len(probs)):
        return None
    return probs


def _policy_abs_positions(seg: Any, node_policy: Optional[Dict[str, Any]], field: str) -> List[int]:
    if not isinstance(node_policy, dict):
        return []
    out: List[int] = []
    for raw_pos in node_policy.get(field, []) or []:
        pos = _relative_to_abs_position(seg, raw_pos)
        if pos is not None and int(pos) not in out:
            out.append(int(pos))
    return out


def _position_rule_abs_positions(seg: Any, node_policy: Optional[Dict[str, Any]]) -> List[int]:
    if not isinstance(node_policy, dict):
        return []
    rules = node_policy.get("position_residue_rules")
    if not isinstance(rules, dict):
        return []
    out: List[int] = []
    for raw_pos in rules:
        pos = _relative_to_abs_position(seg, raw_pos)
        if pos is not None and int(pos) not in out:
            out.append(int(pos))
    return out


def _policy_anchor_positions(seg: Any, node_policy: Optional[Dict[str, Any]]) -> List[int]:
    if not isinstance(node_policy, dict):
        return []
    out = _policy_abs_positions(seg, node_policy, "anchor_positions")
    raw_anchor = node_policy.get("site_anchors", {})
    if isinstance(raw_anchor, dict) and seg.name in raw_anchor and isinstance(raw_anchor[seg.name], dict):
        raw_anchor = raw_anchor[seg.name]
    if isinstance(raw_anchor, dict):
        for raw_pos in raw_anchor.get("relative_positions", raw_anchor.get("positions", [])) or []:
            pos = _relative_to_abs_position(seg, raw_pos)
            if pos is not None and int(pos) not in out:
                out.append(int(pos))
    return out


def _policy_motif_options(
    node_policy: Optional[Dict[str, Any]],
) -> List[Tuple[str, str]]:
    if not isinstance(node_policy, dict):
        return []
    motifs: List[Tuple[str, str]] = []
    seen = set()
    for field in ("graft_motifs", "motif_candidates"):
        value = node_policy.get(field)
        if isinstance(value, str):
            raw = [value]
        elif isinstance(value, list):
            raw = value
        else:
            raw = []
        for item in raw:
            motif = "".join(ch for ch in str(item).upper() if ch in AA)
            if 2 <= len(motif) <= 48 and motif not in seen:
                seen.add(motif)
                motifs.append((motif, f"node_policy.{field}"))
    return motifs


def _policy_motifs(node_policy: Optional[Dict[str, Any]]) -> List[str]:
    return [motif for motif, _source in _policy_motif_options(node_policy)]


def _sample_positions(
    positions: List[int],
    rng: np.random.Generator,
    k: int,
    probs: Optional[np.ndarray] = None,
    preferred: Optional[List[int]] = None,
) -> List[int]:
    if not positions:
        return []
    preferred = [int(p) for p in (preferred or []) if int(p) in set(positions)]
    chosen: List[int] = []
    if preferred:
        rng.shuffle(preferred)
        chosen.extend(preferred[: min(len(preferred), max(1, k))])
    remaining = [int(p) for p in positions if int(p) not in set(chosen)]
    if len(chosen) < k and remaining:
        if probs is not None and len(probs) == len(positions):
            prob_by_pos = {int(pos): float(prob) for pos, prob in zip(positions, probs)}
            rem_probs = np.asarray([prob_by_pos[int(pos)] for pos in remaining], dtype=float)
            total = float(rem_probs.sum())
            rem_probs = rem_probs / total if total > 0 else None
        else:
            rem_probs = None
        extra = rng.choice(
            np.asarray(remaining),
            size=min(len(remaining), k - len(chosen)),
            replace=False,
            p=rem_probs,
        ).tolist()
        chosen.extend(int(x) for x in extra)
    return sorted(set(chosen))


def _graft_motif_into_node(
    seq_list: List[str],
    seg: Any,
    positions: List[int],
    motif: str,
    rng: np.random.Generator,
    node_policy: Optional[Dict[str, Any]],
) -> Tuple[List[int], List[Dict[str, Any]]]:
    motif = "".join(aa for aa in str(motif).upper() if aa in AA)
    if not motif:
        return [], []

    pos_set = {int(p) for p in positions if 0 <= int(p) < len(seq_list)}
    if len(pos_set) < len(motif):
        return [], []

    def numeric_window(start: int) -> Optional[List[int]]:
        window = [int(start) + offset for offset in range(len(motif))]
        if all(pos in pos_set for pos in window):
            return window
        return None

    candidate_windows: List[List[int]] = []
    seen_windows = set()

    def add_window(window: Optional[List[int]]) -> None:
        if not window:
            return
        key = tuple(int(pos) for pos in window)
        if key not in seen_windows:
            seen_windows.add(key)
            candidate_windows.append([int(pos) for pos in window])

    anchors = _policy_anchor_positions(seg, node_policy) + _policy_abs_positions(seg, node_policy, "hotspot_positions")
    for anchor in anchors:
        add_window(numeric_window(int(anchor)))

    if not candidate_windows:
        for start in sorted(pos_set):
            add_window(numeric_window(int(start)))

    if not candidate_windows:
        indices = [int(x) for x in seg.indices() if int(x) in pos_set and 0 <= int(x) < len(seq_list)]
        if len(indices) >= len(motif):
            for offset in range(0, len(indices) - len(motif) + 1):
                window = indices[offset : offset + len(motif)]
                if all(window[i + 1] == window[i] + 1 for i in range(len(window) - 1)):
                    add_window(window)

    if not candidate_windows:
        return [], []

    window = candidate_windows[int(rng.integers(0, len(candidate_windows)))]
    changes: List[Dict[str, Any]] = []
    chosen: List[int] = []
    for pos, aa in zip(window, motif):
        old = seq_list[pos]
        if old == aa:
            continue
        seq_list[pos] = aa
        chosen.append(int(pos))
        changes.append({"position": int(pos), "from": old, "to": aa, "motif": motif})
    return chosen, changes
