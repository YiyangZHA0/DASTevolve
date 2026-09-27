

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from astevolve.core.amino_acids import AA
from astevolve.search.config import SAConfig
from astevolve.search.generation_runtime import realize_generated_substitutions
from astevolve.search.mutation_move import finalize_mutation_move
from astevolve.search.operator_registry import operator_manifest, require_operator
from astevolve.search.proposal_priors import (
    _aa_weights_for_position,
    _aa_weights_for_segment,
    _node_policy,
    _policy_float,
    _policy_int,
    _position_residue_rule,
)
from astevolve.search.proposal_sampling import (
    _apply_forbidden_residue_filter,
    _graft_motif_into_node,
    _policy_abs_positions,
    _policy_anchor_positions,
    _policy_motif_options,
    _position_sampling_probs,
    _proposal_plan,
    _sample_positions,
)
from astevolve.search.position_distribution_runtime import (
    resolve_position_distribution_policy,
)
from astevolve.search.sequence_generator import SequenceGeneratorRegistry
from astevolve.search.sequence_ops import _choose_op


def _mutate_node_seqs(
    seqs: Dict[str, str],
    seg: Any,
    designable_positions: List[int],
    rng: np.random.Generator,
    cfg: SAConfig,
    masks: Dict[str, np.ndarray],
    internal_memory: Optional[Dict[str, Any]],
    mapping_action_spec: Optional[Dict[str, Any]] = None,
    fixed_residues: Optional[Mapping[str, Mapping[int, str]]] = None,
    generation_step: int = 0,
    sequence_generator_registry: Optional[SequenceGeneratorRegistry] = None,
    forced_proposal_tier: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, Any]]:


    new = {k: list(v) for k, v in seqs.items()}
    cid = seg.chain_id
    positions = [
        i
        for i in designable_positions
        if cid in new and 0 <= i < len(new[cid]) and bool(masks[cid][i])
    ]
    mapping = (
        dict(mapping_action_spec)
        if isinstance(mapping_action_spec, dict) and mapping_action_spec
        else {}
    )
    if mapping:
        if str(mapping.get("chain_id") or "") != str(cid):
            raise ValueError("mapping action chain_id does not match selected segment")
        legal = {int(position) for position in mapping.get("legal_positions", [])}
        positions = [position for position in positions if position in legal]
        if not positions:
            raise ValueError("mapping action has no legal designable positions")
    structural_node_id = str(mapping.get("structural_node_id") or seg.name)
    distribution_policy = resolve_position_distribution_policy(cfg)
    node_distribution_rows = distribution_policy.for_node(cid, structural_node_id)
    move: Dict[str, Any] = {
        "op": None,
        "node": seg.name,
        "node_kind": seg.kind,
        "chain_id": cid,
        "positions": {cid: []},
        "segments": [(cid, seg.name, seg.spans)],
        "changes": [],
    }

    def finish(result: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, Any]]:
        finalized = finalize_mutation_move(seqs, result, move)
        if mapping and str(finalized.get("outcome")) == "executed":
            raw_budget = mapping.get("budget")
            if not isinstance(raw_budget, dict):
                raise ValueError("mapped action budget must be a mapping")
            minimum = int(raw_budget.get("min", 0))
            maximum = int(raw_budget.get("max", 0))
            count = int(
                (finalized.get("actual_delta") or {}).get(
                    "residue_change_count", 0
                )
            )
            if count < minimum or count > maximum:
                raise ValueError(
                    f"mapped action realized {count} residue changes outside "
                    f"declared budget [{minimum}, {maximum}]"
                )
        return result, finalized

    def finish_new() -> Tuple[Dict[str, str], Dict[str, Any]]:
        return finish({key: "".join(value) for key, value in new.items()})

    if not positions:
        move["reason"] = "node has no designable positions"
        return finish(seqs)

    node_policy = _node_policy(cfg, seg)
    protected = set(_policy_abs_positions(seg, node_policy, "protected_positions"))
    if protected:
        positions = [p for p in positions if int(p) not in protected]
    missing_distribution_positions = sorted(
        row.position
        for row in node_distribution_rows
        if row.position not in positions
        and not (
            row.required_mutation is not None
            and seqs[cid][row.position] == row.required_mutation
        )
    )
    if missing_distribution_positions:
        raise ValueError(
            "compiled position distribution falls outside executable node scope "
            f"{structural_node_id!r}: {missing_distribution_positions}"
        )
    satisfied_required = {
        row.position
        for row in node_distribution_rows
        if row.required_mutation is not None
        and seqs[cid][row.position] == row.required_mutation
    }
    positions = [position for position in positions if position not in satisfied_required]
    required_positions = {
        row.position
        for row in node_distribution_rows
        if row.required_mutation is not None
        and row.position not in satisfied_required
    }
    if not positions:
        move["reason"] = "all designable positions are protected"
        return finish(seqs)
    mutation_plan = _proposal_plan(
        seg,
        node_policy,
        cfg,
        rng,
        mapping_action_spec=mapping or None,
        forced_tier=forced_proposal_tier,
    )
    move["mutation_plan"] = mutation_plan
    if mapping:
        move["mapping_attribution"] = {
            key: mapping[key]
            for key in (
                "ast_id",
                "ast_revision",
                "edge_id",
                "functional_node_id",
                "structural_node_id",
                "action_id",
                "measurement_id",
            )
        }
    if str(mutation_plan.get("tier", "")).lower() == "frozen" or int(mutation_plan.get("budget", {}).get("max", 0) or 0) <= 0:
        move["op"] = "freeze_node"
        move["outcome"] = "frozen"
        return finish(seqs)
    op_weights = mutation_plan.get("op_weights")
    if not isinstance(op_weights, dict) or not op_weights:
        raise RuntimeError("Executable mutation plan has no effective operator weights")
    if node_distribution_rows:
        exact_handlers = {
            "point_substitution",
            "segment_resample",
            "site_resample",
            "segment_mutagenesis",
            "cdr_resample",
        }
        op_weights = {
            name: weight
            for name, weight in op_weights.items()
            if require_operator(name, mode="node").node_handler in exact_handlers
        }
        if not op_weights:
            raise ValueError(
                "compiled position distributions require a residue-realizing "
                f"operator at node {structural_node_id!r}"
            )
    op = _choose_op(rng, op_weights, mode="node")
    operator_spec = require_operator(op, mode="node")
    handler = operator_spec.node_handler
    move["op"] = op
    move["operator_spec"] = operator_manifest(op, mode="node")
    move["operator_selection"] = {
        "selected": op,
        "source": "mutation_plan.op_weights",
        "effective_weights": dict(op_weights),
    }
    position_probs = _position_sampling_probs(seg, positions, node_policy)
    current_fragment = seg.extract(seqs.get(cid, ""))
    weights = _aa_weights_for_segment(
        seg,
        internal_memory,
        node_policy=node_policy,
    )
    weights = _apply_forbidden_residue_filter(weights, list(mutation_plan.get("forbidden_residues", []) or []))
    if node_policy:
        move["node_policy"] = {
            "priority_boost": node_policy.get("priority_boost"),
            "mutation_rate": node_policy.get("mutation_rate"),
            "max_mutations_per_step": node_policy.get("max_mutations_per_step"),
            "edit_intent": node_policy.get("edit_intent"),
            "favored_residues": list(node_policy.get("favored_residues", []) or [])[:12],
            "favored_residue_classes": list(node_policy.get("favored_residue_classes", []) or [])[:8],
            "site_anchors": node_policy.get("site_anchors"),
            "anchor_positions": list(node_policy.get("anchor_positions", []) or [])[:16],
            "hotspot_positions": list(node_policy.get("hotspot_positions", []) or [])[:16],
            "graft_motifs": list(node_policy.get("graft_motifs", []) or [])[:6],
            "motif_candidates": list(node_policy.get("motif_candidates", []) or [])[:6],
            "operator_phase": node_policy.get("operator_phase"),
            "large_jump": node_policy.get("large_jump"),
            "secondary_structure": node_policy.get("secondary_structure"),
            "position_residue_rules": node_policy.get("position_residue_rules"),
            "mutation_ops": node_policy.get("mutation_ops"),
        }
    mutation_rate = _policy_float(node_policy, "mutation_rate", cfg.mutation_rate)
    base_k = max(1, int(round(mutation_rate * len(positions))))
    budget = mutation_plan.get("budget", {}) if isinstance(mutation_plan.get("budget"), dict) else {}
    action_budgets = mutation_plan.get("allowed_action_budgets", {})
    if isinstance(action_budgets, dict) and isinstance(action_budgets.get(op), dict):
        budget = dict(action_budgets[op])
        move["selected_action_budget"] = dict(budget)
    min_step = max(1, int(budget.get("min", 1) or 1))
    plan_max = int(budget.get("max", 0) or 0)
    max_step = plan_max if plan_max > 0 else _policy_int(node_policy, "max_mutations_per_step", 0)
    base_k = max(base_k, min_step)
    if max_step > 0:
        base_k = min(base_k, max_step)
    if max_step > 0 and len(required_positions) > max_step:
        raise ValueError(
            "required compiled position distributions exceed node mutation "
            f"budget at {structural_node_id!r}"
        )
    if handler == "cdr_resample":
        if str(getattr(seg, "kind", "") or "").lower() == "cdr":
            k = len(positions) if max_step <= 0 else min(len(positions), max_step)
            k = max(min_step, k)
            chosen = _sample_positions(positions, rng, k, position_probs)
        else:
            k = min(len(positions), max(base_k, min(8, len(positions))))
            chosen = _sample_positions(positions, rng, k, position_probs)
    elif handler == "segment_resample":
        k = min(len(positions), max(base_k, min(4, len(positions))))
        if max_step > 0:
            k = min(k, max_step)
        chosen = _sample_positions(positions, rng, k, position_probs)
    elif handler == "segment_mutagenesis":
        jump_floor = min(8, len(positions)) if bool(node_policy.get("large_jump")) else min(5, len(positions))
        k = min(len(positions), max(base_k, jump_floor))
        if max_step > 0:
            k = min(k, max_step)
        preferred = _policy_abs_positions(seg, node_policy, "hotspot_positions") + _policy_anchor_positions(seg, node_policy)
        chosen = _sample_positions(positions, rng, k, position_probs, preferred=preferred)
    elif handler == "site_resample":
        preferred = (
            _policy_abs_positions(seg, node_policy, "hotspot_positions")
            + _policy_abs_positions(seg, node_policy, "anchor_positions")
            + _policy_abs_positions(seg, node_policy, "mutable_positions")
            + _policy_anchor_positions(seg, node_policy)
        )
        k = min(len(positions), max(base_k, min(4, len(positions))))
        if max_step > 0:
            k = min(k, max_step)
        chosen = _sample_positions(positions, rng, k, position_probs, preferred=preferred)
    elif handler == "motif_graft":
        motif_options = _policy_motif_options(node_policy)
        if not motif_options:
            move["outcome"] = "rejected"
            move["reason"] = "motif_graft requires an explicit node-policy motif"
            return finish(seqs)
        motif, motif_source = motif_options[int(rng.integers(0, len(motif_options)))]
        chosen, _motif_changes = _graft_motif_into_node(
            new[cid], seg, positions, motif, rng, node_policy
        )
        move["motif"] = motif
        move["motif_source"] = motif_source
        move["positions"][cid] = [int(x) for x in chosen]
        if chosen:
            if len(chosen) < min_step or (max_step > 0 and len(chosen) > max_step):
                move["outcome"] = "rejected"
                move["reason"] = (
                    "motif_graft realized positions outside declared action budget"
                )
                return finish(seqs)
            return finish_new()
        if motif in current_fragment:
            move["outcome"] = "noop"
            move["reason"] = "motif_already_present"
        else:
            move["outcome"] = "rejected"
            move["reason"] = "motif_not_applicable"
        return finish(seqs)
    elif handler == "region_shuffle":
        k = min(len(positions), max(2, max(base_k, min(8, len(positions)))))
        if max_step > 0:
            k = min(k, max_step)
        chosen = _sample_positions(positions, rng, k, position_probs)
        old_residues = [new[cid][pos] for pos in chosen]
        shuffled = old_residues[:]
        rng.shuffle(shuffled)
        if shuffled == old_residues and len(shuffled) > 1:
            shuffled = shuffled[1:] + shuffled[:1]
        for pos, aa in zip(chosen, shuffled):
            old = new[cid][pos]
            new[cid][pos] = aa
            move["changes"].append(
                {"chain_id": cid, "position": int(pos), "from": old, "to": aa, "node": seg.name}
            )
        move["positions"][cid] = [int(x) for x in chosen]
        return finish_new()
    elif handler == "block_substitution":
        pos_set = set(positions)
        lower = max(2, min_step)
        upper = min(len(positions), max_step if max_step > 0 else 6)
        feasible_blocks = [
            list(range(start, start + length))
            for start in sorted(pos_set)
            for length in range(lower, upper + 1)
            if all(position in pos_set for position in range(start, start + length))
        ]
        if not feasible_blocks:
            move["outcome"] = "rejected"
            move["reason"] = (
                "block operator has no contiguous legal span satisfying its budget"
            )
            return finish(seqs)
        chosen = feasible_blocks[int(rng.integers(0, len(feasible_blocks)))]
    elif handler == "swap":
        if len(positions) < 2:
            move["outcome"] = "noop"
            move["reason"] = "swap requires at least two designable positions"
            return finish(seqs)
        i, j = rng.choice(np.asarray(positions), size=2, replace=False, p=position_probs)
        i, j = int(i), int(j)
        old_i, old_j = new[cid][i], new[cid][j]
        new[cid][i], new[cid][j] = old_j, old_i
        chosen = [i, j]
        move["changes"] = [
            {"chain_id": cid, "position": i, "from": old_i, "to": old_j, "node": seg.name},
            {"chain_id": cid, "position": j, "from": old_j, "to": old_i, "node": seg.name},
        ]
        move["positions"][cid] = chosen
        return finish_new()
    elif handler == "point_substitution":
        k = min(len(positions), base_k)
        chosen = sorted(rng.choice(np.asarray(positions), size=k, replace=False, p=position_probs).tolist())
    else:
        raise RuntimeError(
            f"Registry handler {handler!r} has no node execution branch"
        )

    if required_positions:
        target_count = max(len(chosen), len(required_positions), min_step)
        if max_step > 0:
            target_count = min(target_count, max_step)
        retained = [position for position in chosen if position not in required_positions]
        chosen = sorted(
            [*required_positions, *retained[: target_count - len(required_positions)]]
        )

    position_weights = {}
    for pos in chosen:
        pos_weights = _aa_weights_for_position(
            weights, node_policy, seg, int(pos)
        )
        position_weights[int(pos)] = {
            residue: float(pos_weights[index])
            for index, residue in enumerate(AA)
        }
    generated, generation_artifact = realize_generated_substitutions(
        parent_sequences=seqs,
        target_chain=cid,
        write_positions=chosen,
        operator=op,
        structural_node_id=structural_node_id,
        cfg=cfg,
        masks=masks,
        fixed_residues=fixed_residues,
        position_weights=position_weights,
        mapping_action=mapping or None,
        generation_step=generation_step,
        registry=sequence_generator_registry,
    )
    new = {chain: list(sequence) for chain, sequence in generated.items()}
    move["sequence_generation"] = generation_artifact
    actual_chosen = []
    for pos in chosen:
        old = seqs[cid][pos]
        aa = generated[cid][pos]
        if aa == old:
            continue
        actual_chosen.append(int(pos))
        change = {
            "chain_id": cid,
            "position": int(pos),
            "from": old,
            "to": aa,
            "node": seg.name,
        }
        pos_rule = _position_residue_rule(node_policy, seg, int(pos))
        if pos_rule:
            applied_rule = {
                "position": int(pos),
                "favored_residues": list(pos_rule.get("favored_residues", []) or [])[:12],
                "favored_residue_classes": list(pos_rule.get("favored_residue_classes", []) or [])[:8],
                "disfavored_residues": list(pos_rule.get("disfavored_residues", []) or [])[:12],
                "disfavored_residue_classes": list(pos_rule.get("disfavored_residue_classes", []) or [])[:8],
                "intent": pos_rule.get("intent"),
            }
            change["position_residue_rule"] = applied_rule
            move.setdefault("position_residue_rule_applications", []).append(applied_rule)
        move["changes"].append(change)

    move["positions"][cid] = actual_chosen
    return finish_new()
