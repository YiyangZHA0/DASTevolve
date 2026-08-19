

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from astevolve.search.artifact_io import _seqs_hash
from astevolve.search.candidate_validation import (
    _candidate_fast_filter,
    _required_final_mutation_coverage,
)
from astevolve.search.causal_binding import (
    bind_candidate as _bind_causal_candidate,
    bind_move as _bind_causal_move,
)
from astevolve.search.config import SAConfig
from astevolve.search.energy_reporting import fast_energy_record
from astevolve.search.generation_runtime import record_generation_attempt
from astevolve.search.node_optimizer_runtime import (
    mutate_node_candidates,
    record_node_optimization_attempt,
)
from astevolve.search.proposal_engine import _semantic_coverage_hard_enabled
from astevolve.search.reporting import (
    _mcts_backprop,
    _mcts_best_path,
    _refresh_mcts_widening_pair_state,
)
from astevolve.search.run_memory import InnerRunMemory
from engine.history_runtime import (
    register_sequence_occurrence as _register_history_sequence,
)


FastScore = Callable[
    [
        Dict[str, str],
        List[tuple[float, Any]],
        SAConfig,
        Dict[str, Any],
        Optional[InnerRunMemory],
    ],
    Tuple[Dict[str, Any], Dict[str, float], float, bool],
]
InnerStructureEvaluator = Callable[[Dict[str, Any]], Dict[str, Any]]
SequenceIdentity = Tuple[Tuple[str, str], ...]
ExpansionPair = Tuple[str, str]
PrebuiltProposal = Tuple[Mapping[str, str], Mapping[str, Any]]
_MAX_PROPOSAL_RESAMPLE_ATTEMPTS = 4

_PORTFOLIO_RECEIPT_FIELDS = (
    "compiled_portfolio_request_hash",
    "portfolio_realization_receipts",
    "portfolio_pair_receipts",
    "portfolio_realization_summary",
    "portfolio_receipt",
    "portfolio_id",
    "portfolio_role",
    "portfolio_slot_id",
    "portfolio_required",
    "matched_pair_id",
    "matched_pair_member",
    "prebuilt_exact_duplicate_receipts",
    "candidate_wave_slot_directive",
)


@dataclass
class MCTSExpansionState:


    candidate_serial: int
    best_sequences: Dict[str, str]
    best_breakdown: Dict[str, Any]
    best_progen: Dict[str, float]
    best_fast: float
    best_node_id: str
    best_selection_loss: float = float("inf")


    deferred_expansions: Set[Tuple[str, str]] = field(default_factory=set)


    proposed_sequence_identities: Dict[
        ExpansionPair, Set[SequenceIdentity]
    ] = field(default_factory=dict)

    sequence_nodes: Dict[SequenceIdentity, str] = field(default_factory=dict)


    proposal_rng_seeds: Dict[ExpansionPair, int] = field(default_factory=dict)

    # A node/action pair may temporarily return only proposals that were
    # already consumed. Rotate its optimizer seed a bounded number of times
    # before declaring the proposal space permanently exhausted.
    proposal_resample_attempts: Dict[ExpansionPair, int] = field(
        default_factory=dict
    )


def _sequence_identity(seqs: Mapping[str, str]) -> SequenceIdentity:
    return tuple(
        sorted((str(chain_id), str(sequence)) for chain_id, sequence in seqs.items())
    )


def _target_chain_collision_exclusions(
    existing_identities: Sequence[SequenceIdentity],
    *,
    parent_sequences: Mapping[str, str],
    target_chain: str,
) -> List[str]:


    parent = {str(chain): str(sequence) for chain, sequence in parent_sequences.items()}
    if target_chain not in parent:
        return []
    excluded: Set[str] = set()
    for identity in existing_identities:
        bundle = dict(identity)
        if set(bundle) != set(parent):
            continue
        if all(
            bundle[chain] == sequence
            for chain, sequence in parent.items()
            if chain != target_chain
        ):
            excluded.add(str(bundle[target_chain]))
    return sorted(excluded)


def _prebuilt_history(history: Dict[str, Any]) -> Dict[str, Any]:


    return history.setdefault(
        "mcts_prebuilt_exact",
        {
            "schema_version": "astevolve.mcts_prebuilt_exact.v1",
            "materialization_calls": 0,
            "requested": 0,
            "unique_submitted": 0,
            "duplicate_in_batch_skips": 0,
            "existing_tree_skips": 0,
            "committed": 0,
            "fast_filter_failures": 0,
            "counts_as_physical_expansion_round": False,
            "bypasses_random_proposal": True,
            "bypasses_node_optimizer": True,
            "bypasses_progressive_widening": True,
        },
    )


def _portfolio_candidate_fields(move: Mapping[str, Any]) -> Dict[str, Any]:


    return {
        field: deepcopy(move[field])
        for field in _PORTFOLIO_RECEIPT_FIELDS
        if field in move
    }


def _prebuilt_move(
    sequences: Mapping[str, str],
    parent_sequences: Mapping[str, str],
    move: Mapping[str, Any],
    *,
    proposal_index: int,
) -> Dict[str, Any]:


    copied = deepcopy(dict(move))
    copied.setdefault("op", "prebuilt_exact")
    copied.setdefault("node", "portfolio_exact")
    copied.setdefault("proposal_log_prior", 0.0)
    plan = copied.get("mutation_plan")
    if plan is None:
        copied["mutation_plan"] = {"tier": "prebuilt_exact"}
    elif not isinstance(plan, Mapping):
        raise ValueError("prebuilt proposal mutation_plan must be a mapping")
    else:
        copied["mutation_plan"] = deepcopy(dict(plan))
        copied["mutation_plan"].setdefault("tier", "prebuilt_exact")

    if "changes" not in copied:
        node_name = str(copied.get("node") or "portfolio_exact")
        copied["changes"] = [
            {
                "chain_id": chain_id,
                "position": position,
                "from": parent_sequences[chain_id][position],
                "to": sequence[position],
                "node": node_name,
            }
            for chain_id, sequence in sorted(sequences.items())
            for position in range(len(sequence))
            if parent_sequences[chain_id][position] != sequence[position]
        ]
    elif isinstance(copied["changes"], (str, bytes)) or not isinstance(
        copied["changes"], Sequence
    ):
        raise ValueError("prebuilt proposal changes must be a sequence")
    copied["prebuilt_exact"] = {
        "schema_version": "astevolve.prebuilt_exact_move.v1",
        "proposal_index": int(proposal_index),
        "materialization_phase": "before_physical_mcts_loop",
        "counts_as_physical_expansion_round": False,
    }
    return copied


def _merge_prebuilt_duplicate_receipts(
    retained: Dict[str, Any], duplicate: Mapping[str, Any]
) -> None:


    retained_request_hash = retained.get("compiled_portfolio_request_hash")
    duplicate_request_hash = duplicate.get("compiled_portfolio_request_hash")
    if (
        retained_request_hash not in (None, "")
        and duplicate_request_hash not in (None, "")
        and retained_request_hash != duplicate_request_hash
    ):
        raise ValueError(
            "duplicate exact sequence is bound to different compiled portfolio requests"
        )
    for field in ("portfolio_realization_receipts", "portfolio_pair_receipts"):
        raw = duplicate.get(field)
        if raw is None:
            continue
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise ValueError(f"prebuilt proposal {field} must be a sequence")
        target = retained.setdefault(field, [])
        if isinstance(target, (str, bytes)) or not isinstance(target, list):
            raise ValueError(f"prebuilt proposal {field} must be a list")
        for receipt in raw:
            copied = deepcopy(receipt)
            if copied not in target:
                target.append(copied)
    duplicate_receipt = {
        field: deepcopy(duplicate[field])
        for field in _PORTFOLIO_RECEIPT_FIELDS
        if field in duplicate and field != "prebuilt_exact_duplicate_receipts"
    }
    if duplicate_receipt:
        retained.setdefault("prebuilt_exact_duplicate_receipts", []).append(
            duplicate_receipt
        )


def _normalize_prebuilt_proposals(
    proposals: Sequence[PrebuiltProposal],
    *,
    parent_sequences: Mapping[str, str],
    existing_identities: Set[SequenceIdentity],
    history: Dict[str, Any],
) -> List[Tuple[Dict[str, str], Dict[str, Any]]]:


    if isinstance(proposals, (str, bytes)) or not isinstance(proposals, Sequence):
        raise ValueError("prebuilt_proposals must be a sequence of (sequences, move)")
    summary = _prebuilt_history(history)
    summary["requested"] = int(summary["requested"]) + len(proposals)
    unique: List[Tuple[Dict[str, str], Dict[str, Any]]] = []
    seen: Dict[SequenceIdentity, int] = {}
    parent_identity = _sequence_identity(parent_sequences)
    expected_chains = set(parent_sequences)
    for index, raw in enumerate(proposals):
        if (
            isinstance(raw, (str, bytes))
            or not isinstance(raw, Sequence)
            or len(raw) != 2
        ):
            raise ValueError(
                f"prebuilt proposal {index} must be a (sequences, move) pair"
            )
        raw_sequences, raw_move = raw
        if not isinstance(raw_sequences, Mapping) or not raw_sequences:
            raise ValueError(f"prebuilt proposal {index} sequences must be a mapping")
        if not isinstance(raw_move, Mapping):
            raise ValueError(f"prebuilt proposal {index} move must be a mapping")
        sequences = {
            str(chain_id): str(sequence)
            for chain_id, sequence in sorted(raw_sequences.items())
        }
        if set(sequences) != expected_chains:
            raise ValueError(
                f"prebuilt proposal {index} chain set differs from MCTS root"
            )
        if any(
            not sequence
            or len(sequence) != len(parent_sequences[chain_id])
            for chain_id, sequence in sequences.items()
        ):
            raise ValueError(
                f"prebuilt proposal {index} sequence shape differs from MCTS root"
            )
        identity = _sequence_identity(sequences)
        if identity in seen:
            summary["duplicate_in_batch_skips"] = int(
                summary["duplicate_in_batch_skips"]
            ) + 1
            _merge_prebuilt_duplicate_receipts(
                unique[seen[identity]][1], raw_move
            )
            continue
        seen[identity] = len(unique)
        if identity == parent_identity or identity in existing_identities:

            seen.pop(identity, None)
            summary["existing_tree_skips"] = int(summary["existing_tree_skips"]) + 1
            continue
        unique.append(
            (
                sequences,
                _prebuilt_move(
                    sequences,
                    parent_sequences,
                    raw_move,
                    proposal_index=index,
                ),
            )
        )
    summary["unique_submitted"] = int(summary["unique_submitted"]) + len(unique)
    return unique


def _widening_history(history: Dict[str, Any]) -> Dict[str, Any]:
    return history.setdefault(
        "mcts_progressive_widening",
        {
            "schema_version": "astevolve.mcts_progressive_widening.v2",
            "capacity_scope": "parent_and_expansion_key",
            "available_slots_minimum": 0,
            "stalled_batches": 0,
            "zero_available_slot_rounds": 0,
            "without_replacement_skips": 0,
            "duplicate_fast_score_short_circuits": 0,
            "duplicate_backprop_skips": 0,

            "deferred_parent_expansions": [],
            "exhausted_parent_expansions": [],
        },
    )


def _sync_pair_history(
    history: Dict[str, Any],
    *,
    parent_id: str,
    expansion_key: str,
    pair_state: Mapping[str, Any],
) -> None:
    widening = _widening_history(history)
    rows = widening.setdefault("pair_states", [])
    row = {
        "parent_id": str(parent_id),
        "expansion_key": str(expansion_key),
        "committed_children": int(pair_state.get("committed_children", 0) or 0),
        "base_capacity": int(pair_state.get("base_capacity", 0) or 0),
        "recovery_capacity_bonus": int(
            pair_state.get("recovery_capacity_bonus", 0) or 0
        ),
        "recovery_unlocks": int(pair_state.get("recovery_unlocks", 0) or 0),
        "capacity": int(pair_state.get("capacity", 0) or 0),
        "available_slots": int(pair_state.get("available_slots", 0) or 0),
        "capacity_exhausted": bool(pair_state.get("capacity_exhausted", False)),
        "proposal_space_exhausted": bool(
            pair_state.get("proposal_space_exhausted", False)
        ),
        "parent_visits": int(pair_state.get("parent_visits", 0) or 0),
    }
    for index, prior in enumerate(rows):
        if (
            prior.get("parent_id") == row["parent_id"]
            and prior.get("expansion_key") == row["expansion_key"]
        ):
            rows[index] = row
            break
    else:
        rows.append(row)


def _selected_siblings(
    proposed: Sequence[Tuple[Dict[str, str], Dict[str, Any]]],
    *,
    parent: Dict[str, Any],
    expansion_key: str,
    cfg: SAConfig,
) -> List[Tuple[Dict[str, str], Dict[str, Any]]]:


    selected = list(proposed)
    if not bool(getattr(cfg, "node_optimizer_enabled", False)):
        return selected
    pair_state = _refresh_mcts_widening_pair_state(
        parent, str(expansion_key), cfg
    )
    available_slots = int(pair_state.get("available_slots", 0) or 0)
    return sorted(
        selected,
        key=lambda item: (
            -int(
                bool(
                    (item[1].get("_preselection_fast_filter") or {}).get(
                        "pass", False
                    )
                )
            ),
            -float(
                (item[1].get("_preselection_fast_filter") or {}).get(
                    "search_progress", 0.0
                )
                or 0.0
            ),
            -float(item[1].get("proposal_log_prior", 0.0)),
            _seqs_hash(item[0]),
        ),
    )[:available_slots]


def _normalized_sibling_priors(
    proposed: Sequence[Tuple[Dict[str, str], Dict[str, Any]]],
) -> np.ndarray:
    logs = np.asarray(
        [float(move.get("proposal_log_prior", 0.0)) for _seqs, move in proposed],
        dtype=float,
    )
    logs = logs - float(np.max(logs))
    weights = np.exp(np.clip(logs, -80.0, 0.0))
    return weights / max(float(weights.sum()), 1e-12)


def _renormalize_child_priors(
    tree: Dict[str, Dict[str, Any]],
    parent: Dict[str, Any],
) -> None:
    total = sum(
        max(1e-12, float(tree[child_id].get("prior_raw", 1.0)))
        for child_id in parent["children"]
    )
    for child_id in parent["children"]:
        raw = max(1e-12, float(tree[child_id].get("prior_raw", 1.0)))
        tree[child_id]["prior"] = raw / total


def expand_mcts_candidate_siblings(
    state: MCTSExpansionState,
    *,
    tree: Dict[str, Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    history: Dict[str, Any],
    parent_id: str,
    expansion_key: str,
    segment: Any,
    designable_positions: List[int],
    segment_prior: float,
    selection: Mapping[str, Any],
    expansion_round: int,
    rng: np.random.Generator,
    cfg: SAConfig,
    masks: Dict[str, np.ndarray],
    internal_memory: Optional[Dict[str, Any]],
    mapping_action: Optional[Dict[str, Any]],
    fixed_residues: Optional[Mapping[str, Mapping[int, str]]],
    compiled: Dict[str, Any],
    template_seqs: Optional[Dict[str, str]],
    terms_fast: List[tuple[float, Any]],
    root_fast: float,
    reward_scale: float,
    run_memory: Optional[InnerRunMemory],
    causal_context: Optional[Mapping[str, Any]],
    score_fast: FastScore,
    inner_structure_evaluator: Optional[InnerStructureEvaluator] = None,
    prebuilt_proposals: Optional[Sequence[PrebuiltProposal]] = None,
    candidate_wave_slot_directive: Optional[Mapping[str, Any]] = None,
    candidate_evaluation_slots: Optional[int] = None,
) -> MCTSExpansionState:


    parent = tree[parent_id]
    if not np.isfinite(state.best_selection_loss):
        state.best_selection_loss = float(state.best_fast)
    optimizer_enabled = bool(getattr(cfg, "node_optimizer_enabled", False))
    prebuilt_mode = prebuilt_proposals is not None
    if prebuilt_mode and candidate_wave_slot_directive is not None:
        raise ValueError("prebuilt exact proposals cannot consume a free wave slot")
    optimizer_path = optimizer_enabled and not prebuilt_mode
    pair_key: ExpansionPair = (str(parent_id), str(expansion_key))


    for known_node_id, known_node in tree.items():
        known_sequences = known_node.get("seqs")
        if isinstance(known_sequences, Mapping) and known_sequences:
            state.sequence_nodes.setdefault(
                _sequence_identity(known_sequences), str(known_node_id)
            )

    if optimizer_path:
        pair_state = _refresh_mcts_widening_pair_state(
            parent, str(expansion_key), cfg
        )
        _sync_pair_history(
            history,
            parent_id=parent_id,
            expansion_key=expansion_key,
            pair_state=pair_state,
        )
        if bool(pair_state.get("proposal_space_exhausted", False)):
            return state
        if int(pair_state.get("available_slots", 0) or 0) == 0:
            widening = _widening_history(history)
            widening["zero_available_slot_rounds"] = int(
                widening.get("zero_available_slot_rounds", 0)
            ) + 1
            return state

    consumed = state.proposed_sequence_identities.setdefault(pair_key, set())
    if prebuilt_mode:
        proposed = _normalize_prebuilt_proposals(
            prebuilt_proposals or (),
            parent_sequences=parent["seqs"],
            existing_identities=set(state.sequence_nodes),
            history=history,
        )
    else:
        excluded_target_sequences: List[str] = []
        optimizer_seed: Optional[int] = None
        proposal_rng = rng
        if optimizer_path:
            target_chain = str(getattr(segment, "chain_id", "") or "")
            excluded_target_sequences = sorted(
                {
                    identity_map[target_chain]
                    for identity in consumed
                    for identity_map in [dict(identity)]
                    if target_chain and target_chain in identity_map
                }
            )
            if candidate_wave_slot_directive is not None:


                excluded_target_sequences = sorted(
                    set(excluded_target_sequences)
                    | set(
                        _target_chain_collision_exclusions(
                            tuple(state.sequence_nodes),
                            parent_sequences=parent["seqs"],
                            target_chain=target_chain,
                        )
                    )
                )
            if pair_key not in state.proposal_rng_seeds:
                state.proposal_rng_seeds[pair_key] = int(
                    rng.integers(0, np.iinfo(np.int64).max)
                )
            optimizer_seed = state.proposal_rng_seeds[pair_key]
            proposal_rng = np.random.default_rng(optimizer_seed)
        forced_tier = None
        if candidate_wave_slot_directive is not None:
            if not isinstance(candidate_wave_slot_directive, Mapping):
                raise ValueError("candidate wave slot directive must be a mapping")
            forced_tier = str(
                candidate_wave_slot_directive.get("forced_tier") or ""
            ).strip().lower()
            if forced_tier not in {"exploit", "explore", "repair"}:
                raise ValueError("candidate wave slot directive tier is invalid")
        raw_proposed = mutate_node_candidates(
            parent["seqs"],
            segment,
            designable_positions,
            proposal_rng,
            cfg,
            masks,
            internal_memory,
            mapping_action_spec=mapping_action,
            fixed_residues=fixed_residues,
            generation_step=expansion_round,
            excluded_target_sequences=excluded_target_sequences,
            optimizer_seed=optimizer_seed,
            forced_proposal_tier=forced_tier,
        )
        proposed = list(raw_proposed)
        if candidate_wave_slot_directive is not None:
            for _sequences, move in proposed:
                actual_tier = str(
                    (move.get("mutation_plan") or {}).get("tier") or ""
                )
                if actual_tier != forced_tier:
                    raise RuntimeError(
                        "candidate wave generator did not realize its forced tier"
                    )
                move["candidate_wave_slot_directive"] = deepcopy(
                    dict(candidate_wave_slot_directive)
                )


            wave_unique: List[Tuple[Dict[str, str], Dict[str, Any]]] = []
            wave_batch_identities: Set[SequenceIdentity] = set()
            collision_count = 0
            for item in proposed:
                identity = _sequence_identity(item[0])
                if (
                    identity in state.sequence_nodes
                    or identity in wave_batch_identities
                ):
                    collision_count += 1
                    continue
                wave_batch_identities.add(identity)
                wave_unique.append(item)
            uniqueness = history.setdefault(
                "candidate_wave_global_uniqueness",
                {
                    "schema_version": (
                        "astevolve.candidate_wave_global_uniqueness.v1"
                    ),
                    "policy": "full_sequence_bundle_without_replacement",
                    "collision_rejections": 0,
                },
            )
            uniqueness["collision_rejections"] = int(
                uniqueness.get("collision_rejections", 0)
            ) + collision_count
            proposed = wave_unique
    if optimizer_path:
        unseen: List[Tuple[Dict[str, str], Dict[str, Any]]] = []
        batch_identities: Set[SequenceIdentity] = set()
        skipped = 0
        for item in proposed:
            identity = _sequence_identity(item[0])
            if identity in consumed or identity in batch_identities:
                skipped += 1
            else:
                batch_identities.add(identity)
                unseen.append(item)
        if skipped:
            widening = _widening_history(history)
            widening["without_replacement_skips"] = int(
                widening.get("without_replacement_skips", 0)
            ) + skipped
        if not unseen:
            pair_state = _refresh_mcts_widening_pair_state(
                parent, str(expansion_key), cfg
            )
            retry_count = int(
                state.proposal_resample_attempts.get(pair_key, 0)
            ) + 1
            state.proposal_resample_attempts[pair_key] = retry_count
            if retry_count <= _MAX_PROPOSAL_RESAMPLE_ATTEMPTS:
                previous_seed = state.proposal_rng_seeds.get(pair_key)
                replacement_seed = int(
                    rng.integers(0, np.iinfo(np.int64).max)
                )
                while replacement_seed == previous_seed:
                    replacement_seed = int(
                        rng.integers(0, np.iinfo(np.int64).max)
                    )
                state.proposal_rng_seeds[pair_key] = replacement_seed
                pair_state["proposal_space_exhausted"] = False
                widening = _widening_history(history)
                widening["proposal_resample_attempts"] = int(
                    widening.get("proposal_resample_attempts", 0)
                ) + 1
                event = {
                    "parent_id": str(parent_id),
                    "expansion_key": str(expansion_key),
                    "expansion_round": int(expansion_round),
                    "reason": "no_novel_child_in_optimizer_batch",
                    "status": "optimizer_seed_resample_scheduled",
                    "optimizer_returned_count": len(proposed),
                    "resample_attempt": retry_count,
                }
                widening["deferred_parent_expansions"].append(dict(event))
                _sync_pair_history(
                    history,
                    parent_id=parent_id,
                    expansion_key=expansion_key,
                    pair_state=pair_state,
                )
                return state
            pair_state["proposal_space_exhausted"] = True
            state.deferred_expansions.add(pair_key)
            widening = _widening_history(history)
            widening["stalled_batches"] = int(
                widening.get("stalled_batches", 0)
            ) + 1
            event = {
                "parent_id": str(parent_id),
                "expansion_key": str(expansion_key),
                "expansion_round": int(expansion_round),
                "reason": "no_novel_child_in_optimizer_batch",
                "status": "proposal_space_exhausted_without_replacement",
                "optimizer_returned_count": len(proposed),
            }
            widening["deferred_parent_expansions"].append(dict(event))
            widening["exhausted_parent_expansions"].append(dict(event))
            _sync_pair_history(
                history,
                parent_id=parent_id,
                expansion_key=expansion_key,
                pair_state=pair_state,
            )
            return state
        state.proposal_resample_attempts[pair_key] = 0
        proposed = unseen
    if not prebuilt_mode:
        if optimizer_path and str(
            getattr(cfg, "sequence_prefilter_callable", "") or ""
        ).strip():
            for proposed_sequences, proposed_move in proposed:
                proposed_move["_preselection_fast_filter"] = _candidate_fast_filter(
                    proposed_sequences,
                    template_seqs,
                    fixed_residues,
                    compiled,
                    cfg,
                )
        proposed = _selected_siblings(
            proposed,
            parent=parent,
            expansion_key=str(expansion_key),
            cfg=cfg,
        )
    if candidate_evaluation_slots is not None:
        slots = max(0, int(candidate_evaluation_slots))
        proposed = proposed[:slots]
    if not proposed:
        return state
    if optimizer_path:
        consumed = state.proposed_sequence_identities.setdefault(pair_key, set())
        consumed.update(_sequence_identity(sequences) for sequences, _move in proposed)


    if not prebuilt_mode:
        round_tiers = {
            str((move.get("mutation_plan") or {}).get("tier", "unknown"))
            for _sequences, move in proposed
        }
        if len(round_tiers) != 1:
            raise RuntimeError(
                "one MCTS expansion round cannot mix proposal tiers within its "
                "optimizer sibling batch"
            )
        round_tier = next(iter(round_tiers))
        round_counts = history.setdefault("proposal_tier_round_counts", {})
        round_counts[round_tier] = int(round_counts.get(round_tier, 0)) + 1

    batch_weights = _normalized_sibling_priors(proposed)
    semantic_prefix_options: List[Tuple[float, str, Dict[str, Any]]] = []
    root_selection_loss = float(tree["root"].get("selection_loss", root_fast))

    for batch_rank, ((prop, move), batch_prior) in enumerate(
        zip(proposed, batch_weights)
    ):
        state.candidate_serial += 1
        candidate_serial = state.candidate_serial
        history["logical_candidates"] = int(history["logical_candidates"]) + 1
        tier = str((move.get("mutation_plan") or {}).get("tier", "unknown"))
        attempt_counts = history.setdefault("proposal_tier_attempt_counts", {})
        attempt_counts[tier] = int(attempt_counts.get(tier, 0)) + 1
        record_generation_attempt(history, move)
        move["selection"] = {
            **selection,
            "expansion_round": int(expansion_round),
            "candidate_rank": int(batch_rank),
            "candidate_batch_size": len(proposed),
        }
        _bind_causal_move(move, causal_context)
        _register_history_sequence(
            prop,
            role="candidate",
            context_id=f"mcts:{candidate_serial}",
            metadata={
                "search_method": "mcts",
                "step": expansion_round,
                "candidate_rank": batch_rank,
            },
        )
        sequence_claim = (
            run_memory.claim_sequence(prop) if run_memory is not None else None
        )
        proposal_identity = _sequence_identity(prop)
        transposition_target = state.sequence_nodes.get(proposal_identity)
        if (
            transposition_target is None
            and sequence_claim is not None
            and not sequence_claim.is_new
        ):
            transposition_target = sequence_claim.transposition_node_id


        if transposition_target and transposition_target in tree:
            target = tree[transposition_target]
            fast_filter = dict(target.get("fast_filter") or {"pass": True})
            prop_fast = float(target.get("fast_loss", root_fast))
            prop_break = {
                "total": float(target.get("constraint_penalty", 0.0)),
                "duplicate_reused_from_tree": str(transposition_target),
            }
            prop_progen = {
                "loglik_sum": float(target.get("progen_loglik_sum", 0.0)),
                "loglik_avg": float(target.get("progen_loglik_avg", 0.0)),
            }
            reward = float(
                target.get(
                    "reward",
                    np.tanh((float(root_fast) - prop_fast) / reward_scale),
                )
            )
            energy_record = target.get("energy")
            if not isinstance(energy_record, Mapping):
                energy_record = fast_energy_record(
                    fast_loss=prop_fast,
                    constraint_penalty=prop_break["total"],
                    progen_loglik_avg=prop_progen["loglik_avg"],
                    progen_weight=cfg.progen_weight,
                    hard_gate_pass=bool(fast_filter.get("pass", True)),
                )
            history["duplicate_sequence_attempts"] = int(
                history.get("duplicate_sequence_attempts", 0)
            ) + 1
            widening = _widening_history(history)
            widening["duplicate_fast_score_short_circuits"] = int(
                widening.get("duplicate_fast_score_short_circuits", 0)
            ) + 1
            widening["duplicate_backprop_skips"] = int(
                widening.get("duplicate_backprop_skips", 0)
            ) + 1
            if run_memory is not None:
                run_memory.record_action(
                    str(move.get("op") or "unknown"),
                    accepted=False,
                )
            duplicate_candidate = _bind_causal_candidate(
                {
                    "variant_id": f"repeat_{candidate_serial}",
                    "parent_id": parent_id,
                    "seq_hash": _seqs_hash(prop),
                    "seqs": prop,
                    "fast_loss": prop_fast,
                    "selection_loss": float(target.get("selection_loss", prop_fast)),
                    "inner_structure_loss": target.get("inner_structure_loss"),
                    "inner_structure_gate_pass": bool(target.get("inner_structure_gate_pass", True)),
                    "inner_structure_evaluation": target.get("inner_structure_evaluation"),
                    "constraint_penalty": float(prop_break["total"]),
                    "progen_loglik_avg": float(prop_progen["loglik_avg"]),
                    "progen_loglik_sum": float(prop_progen["loglik_sum"]),
                    "reward": reward,
                    "energy": dict(energy_record),
                    "fast_filter": fast_filter,
                    "move": move,
                    "duplicate_sequence": True,
                    "transposition_target": transposition_target,
                    "fast_cache_hit": False,
                    "fast_score_skipped": True,
                    "duplicate_short_circuit_stage": (
                        "before_fast_filter_and_fast_score"
                    ),
                    "semantic_final_coverage": _required_final_mutation_coverage(
                        prop,
                        template_seqs,
                        compiled,
                        cfg,
                    ),
                },
                causal_context,
            )
            candidates.append(duplicate_candidate)
            continue


        if not prebuilt_mode:
            record_node_optimization_attempt(history, move)
        fast_filter = move.pop("_preselection_fast_filter", None)
        if not isinstance(fast_filter, dict):
            fast_filter = _candidate_fast_filter(
                prop,
                template_seqs,
                fixed_residues,
                compiled,
                cfg,
            )
        if fast_filter.get("pass", True):
            prop_break, prop_progen, prop_fast, fast_cache_hit = score_fast(
                prop,
                terms_fast,
                cfg,
                compiled,
                run_memory,
            )
        else:
            search_progress = min(
                1.0,
                max(0.0, float(fast_filter.get("search_progress", 0.0) or 0.0)),
            )
            fail_penalty = max(0.05, 1.0 - search_progress)
            prop_break = {
                "total": float(parent.get("constraint_penalty", 0.0))
                + fail_penalty,
                "fast_filter": fast_filter,
            }
            prop_progen = {"loglik_sum": 0.0, "loglik_avg": 0.0}
            prop_fast = float(parent.get("fast_loss", root_fast)) + fail_penalty
            fast_cache_hit = False
        energy_record = fast_energy_record(
            fast_loss=prop_fast,
            constraint_penalty=prop_break["total"],
            progen_loglik_avg=prop_progen["loglik_avg"],
            progen_weight=cfg.progen_weight,
            hard_gate_pass=bool(fast_filter.get("pass", True)),
        )
        inner_structure_evaluation = None
        selection_loss = float(prop_fast)
        inner_structure_gate_pass = bool(fast_filter.get("pass", True))
        if inner_structure_gate_pass and inner_structure_evaluator is not None:
            provisional_candidate = {
                "variant_id": f"n{candidate_serial}",
                "parent_id": parent_id,
                "seq_hash": _seqs_hash(prop),
                "seqs": prop,
                "fast_loss": float(prop_fast),
                "constraint_penalty": float(prop_break["total"]),
                "progen_loglik_avg": float(prop_progen["loglik_avg"]),
                "progen_loglik_sum": float(prop_progen["loglik_sum"]),
                "fast_filter": fast_filter,
                "move": move,
            }
            inner_structure_evaluation = inner_structure_evaluator(provisional_candidate)
            selection_loss = float(inner_structure_evaluation["selection_loss"])
            inner_structure_gate_pass = bool(inner_structure_evaluation.get("gate_pass", False))
        energy_record["hard_gate_pass"] = bool(inner_structure_gate_pass)
        energy_record["selection_loss"] = float(selection_loss)
        energy_record["inner_structure_energy"] = (
            inner_structure_evaluation or {}
        ).get("structure_combined_energy")
        reward = float(
            np.tanh((root_selection_loss - selection_loss) / reward_scale)
        )

        if run_memory is not None:
            run_memory.record_action(
                str(move.get("op") or "unknown"),
                accepted=bool(inner_structure_gate_pass),
                reward=reward,
            )
        history["fast_cache_hits"] = int(
            history.get("fast_cache_hits", 0)
        ) + int(fast_cache_hit)

        node_id = f"n{candidate_serial}"
        prior_raw = float(segment_prior) * (
            float(batch_prior) if (optimizer_path or prebuilt_mode) else 1.0
        )
        portfolio_fields = _portfolio_candidate_fields(move)
        child = {
            "id": node_id,
            "parent": parent_id,
            "children": [],
            "depth": int(parent["depth"]) + 1,
            "visits": 0,
            "total_reward": 0.0,
            "best_reward": -1e9,
            "prior_raw": prior_raw,
            "prior": prior_raw,
            "expansion_key": str(expansion_key),
            "move": move,
            "seqs": prop,
            "fast_loss": float(prop_fast),
            "selection_loss": float(selection_loss),
            "inner_structure_loss": (inner_structure_evaluation or {}).get("structure_combined_energy"),
            "inner_structure_gate_pass": bool(inner_structure_gate_pass),
            "bootstrap_expandable": bool(fast_filter.get("search_expandable", False)),
            "inner_structure_evaluation": inner_structure_evaluation,
            "constraint_penalty": float(prop_break["total"]),
            "progen_loglik_avg": float(prop_progen["loglik_avg"]),
            "reward": reward,
            "energy": energy_record,
            "fast_filter": fast_filter,
            **portfolio_fields,
        }
        tree[node_id] = child
        parent["children"].append(node_id)
        state.sequence_nodes[proposal_identity] = node_id
        if optimizer_path:
            pair_state = _refresh_mcts_widening_pair_state(
                parent, str(expansion_key), cfg
            )
            pair_state["committed_children"] = int(
                pair_state.get("committed_children", 0)
            ) + 1
            _renormalize_child_priors(tree, parent)
        _mcts_backprop(tree, node_id, reward)
        if run_memory is not None:
            run_memory.record_transposition(
                prop,
                node_id=node_id,
                payload={"fast_loss": float(prop_fast), "reward": reward},
            )

        history["op_counts"][move["op"]] = (
            history["op_counts"].get(move["op"], 0) + 1
        )
        if prebuilt_mode:
            changes = [
                item
                for item in (move.get("changes", []) or [])
                if isinstance(item, Mapping)
            ]
            raw_owner_rows = (
                (move.get("portfolio_realization_summary") or {}).get(
                    "position_owners", []
                )
                if isinstance(
                    move.get("portfolio_realization_summary"), Mapping
                )
                else []
            )
            owner_by_position = {
                (str(row.get("chain_id") or ""), int(row.get("position", -1))): str(
                    row.get("owner_node_id") or ""
                )
                for row in raw_owner_rows
                if isinstance(row, Mapping)
                and str(row.get("chain_id") or "")
                and isinstance(row.get("position"), int)
                and not isinstance(row.get("position"), bool)
                and str(row.get("owner_node_id") or "")
            }
            mutation_nodes = {
                owner_by_position.get(
                    (str(item.get("chain_id") or ""), int(item.get("position", -1))),
                    str(item.get("node") or move.get("node") or "portfolio_exact"),
                )
                for item in changes
            } or {str(move.get("node") or "portfolio_exact")}
            for mutation_node in sorted(mutation_nodes):
                history["node_visit_counts"][mutation_node] = (
                    history["node_visit_counts"].get(mutation_node, 0) + 1
                )
                if mutation_node in history.get(
                    "semantic_designable_required_nodes", []
                ):
                    history["semantic_required_node_visits"][mutation_node] = (
                        history["semantic_required_node_visits"].get(
                            mutation_node, 0
                        )
                        + 1
                    )
                    history["semantic_required_node_mutations"][mutation_node] = (
                        history["semantic_required_node_mutations"].get(
                            mutation_node, 0
                        )
                        + sum(
                            1
                            for item in changes
                            if owner_by_position.get(
                                (
                                    str(item.get("chain_id") or ""),
                                    int(item.get("position", -1)),
                                ),
                                str(
                                    item.get("node")
                                    or move.get("node")
                                    or "portfolio_exact"
                                ),
                            )
                            == mutation_node
                        )
                    )
        else:
            history["node_visit_counts"][segment.name] = (
                history["node_visit_counts"].get(segment.name, 0) + 1
            )
            if segment.name in history.get("semantic_designable_required_nodes", []):
                history["semantic_required_node_visits"][segment.name] = (
                    history["semantic_required_node_visits"].get(segment.name, 0) + 1
                )
                history["semantic_required_node_mutations"][segment.name] = (
                    history["semantic_required_node_mutations"].get(segment.name, 0)
                    + len(move.get("changes", []) or [])
                )
        novel_counts = history.setdefault("proposal_tier_novel_counts", {})
        novel_counts[tier] = int(novel_counts.get(tier, 0)) + 1

        history["proposal_tier_counts"][tier] = (
            history["proposal_tier_counts"].get(tier, 0) + 1
        )
        if not fast_filter.get("pass", True):
            for reason in fast_filter.get("reasons", []) or ["unknown"]:
                history["fast_filter_failures"][reason] = (
                    history["fast_filter_failures"].get(reason, 0) + 1
                )
        semantic_final_coverage = _required_final_mutation_coverage(
            prop,
            template_seqs,
            compiled,
            cfg,
        )
        if (
            _semantic_coverage_hard_enabled(cfg)
            and selection.get("source") == "semantic_required_node"
            and bool(move.get("changes"))
            and fast_filter.get("pass", True)
            and inner_structure_gate_pass
        ):
            semantic_prefix_options.append(
                (float(selection_loss), node_id, semantic_final_coverage)
            )

        if cfg.history_size > 0:
            history["accepted_moves"].append(move)
            if len(history["accepted_moves"]) > cfg.history_size:
                history["accepted_moves"].pop(0)

        candidate = _bind_causal_candidate(
            {
                "variant_id": node_id,
                "parent_id": parent_id,
                "seq_hash": _seqs_hash(prop),
                "seqs": prop,
                "fast_loss": float(prop_fast),
                "constraint_penalty": float(prop_break["total"]),
                "progen_loglik_avg": float(prop_progen["loglik_avg"]),
                "progen_loglik_sum": float(prop_progen["loglik_sum"]),
                "selection_loss": float(selection_loss),
                "inner_structure_loss": (inner_structure_evaluation or {}).get("structure_combined_energy"),
                "inner_structure_gate_pass": bool(inner_structure_gate_pass),
                "bootstrap_expandable": bool(fast_filter.get("search_expandable", False)),
                "inner_structure_evaluation": inner_structure_evaluation,
                "proposal_log_prior": float(
                    move.get("proposal_log_prior", 0.0)
                ),
                "reward": reward,
                "energy": energy_record,
                "fast_filter": fast_filter,
                "move": move,
                "semantic_final_coverage": semantic_final_coverage,
                "duplicate_sequence": False,
                "fast_cache_hit": bool(fast_cache_hit),
                **portfolio_fields,
                "mcts": {
                    "depth": child["depth"],
                    "prior": child["prior"],
                    "prior_raw": child["prior_raw"],
                    "path": _mcts_best_path(tree, node_id),
                },
            },
            causal_context,
        )
        candidates.append(candidate)
        if prebuilt_mode:
            summary = _prebuilt_history(history)
            summary["committed"] = int(summary["committed"]) + 1
            if not fast_filter.get("pass", True):
                summary["fast_filter_failures"] = int(
                    summary["fast_filter_failures"]
                ) + 1

        if inner_structure_gate_pass and selection_loss < state.best_selection_loss:
            state.best_sequences = prop
            state.best_fast = float(prop_fast)
            state.best_selection_loss = float(selection_loss)
            state.best_breakdown = prop_break
            state.best_progen = prop_progen
            state.best_node_id = node_id

    if semantic_prefix_options:
        _loss, semantic_node_id, semantic_coverage = min(
            semantic_prefix_options,
            key=lambda item: item[0],
        )
        history["semantic_prefix_node_id"] = semantic_node_id
        history["semantic_prefix_coverage"] = semantic_coverage
    if optimizer_path:
        pair_state = _refresh_mcts_widening_pair_state(
            parent, str(expansion_key), cfg
        )
        _sync_pair_history(
            history,
            parent_id=parent_id,
            expansion_key=expansion_key,
            pair_state=pair_state,
        )
    return state


def materialize_prebuilt_mcts_root_children(
    state: MCTSExpansionState,
    *,
    prebuilt_proposals: Optional[Sequence[PrebuiltProposal]],
    tree: Dict[str, Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    history: Dict[str, Any],
    segment: Any,
    designable_positions: List[int],
    rng: np.random.Generator,
    cfg: SAConfig,
    masks: Dict[str, np.ndarray],
    internal_memory: Optional[Dict[str, Any]],
    fixed_residues: Optional[Mapping[str, Mapping[int, str]]],
    compiled: Dict[str, Any],
    template_seqs: Optional[Dict[str, str]],
    terms_fast: List[tuple[float, Any]],
    root_fast: float,
    reward_scale: float,
    run_memory: Optional[InnerRunMemory],
    causal_context: Optional[Mapping[str, Any]],
    score_fast: FastScore,
    inner_structure_evaluator: Optional[InnerStructureEvaluator] = None,
    candidate_evaluation_slots: Optional[int] = None,
) -> MCTSExpansionState:


    if prebuilt_proposals is None:
        return state
    summary = _prebuilt_history(history)
    summary["materialization_calls"] = int(summary["materialization_calls"]) + 1
    return expand_mcts_candidate_siblings(
        state,
        tree=tree,
        candidates=candidates,
        history=history,
        parent_id="root",
        expansion_key="portfolio:prebuilt_exact_root",
        segment=segment,
        designable_positions=designable_positions,
        segment_prior=1.0,
        selection={
            "source": "portfolio_prebuilt_exact",
            "materialization_phase": "before_physical_mcts_loop",
            "counts_as_physical_expansion_round": False,
        },
        expansion_round=-1,
        rng=rng,
        cfg=cfg,
        masks=masks,
        internal_memory=internal_memory,
        mapping_action=None,
        fixed_residues=fixed_residues,
        compiled=compiled,
        template_seqs=template_seqs,
        terms_fast=terms_fast,
        root_fast=root_fast,
        reward_scale=reward_scale,
        run_memory=run_memory,
        causal_context=causal_context,
        score_fast=score_fast,
        inner_structure_evaluator=inner_structure_evaluator,
        prebuilt_proposals=prebuilt_proposals,
        candidate_evaluation_slots=candidate_evaluation_slots,
    )


__all__ = [
    "MCTSExpansionState",
    "PrebuiltProposal",
    "expand_mcts_candidate_siblings",
    "materialize_prebuilt_mcts_root_children",
]
