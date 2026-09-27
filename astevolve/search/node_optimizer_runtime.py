

from __future__ import annotations

from functools import lru_cache
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from astevolve.core.amino_acids import AA
from astevolve.providers.masked_lm import (
    MASKED_LM_DEVICE_ENV,
    MaskedProteinLMPrior,
)
from astevolve.search.config import SAConfig
from astevolve.search.mutation_move import finalize_mutation_move
from astevolve.search.node_mutation import _mutate_node_seqs
from astevolve.search.node_sequence_optimizer import (
    DefaultNodeSequenceOptimizer,
    NodeSequenceOptimizationError,
    NodeSequenceOptimizationRequest,
    NodeSequenceOptimizer,
    SoftWeightResiduePriorScorer,
    validate_node_sequence_optimization_result,
)
from astevolve.search.operator_registry import operator_manifest, require_operator
from astevolve.search.proposal_priors import (
    _aa_weights_for_position,
    _aa_weights_for_segment,
    _node_policy,
    _position_residue_rule,
)
from astevolve.search.proposal_sampling import (
    _apply_forbidden_residue_filter,
    _policy_abs_positions,
    _proposal_plan,
)
from astevolve.search.position_distribution_runtime import (
    RuntimePositionDistribution,
    resolve_position_distribution_policy,
)
from astevolve.search.sequence_generator import SequenceGeneratorRegistry
from astevolve.search.sequence_ops import _choose_op


NODE_OPTIMIZATION_RUNTIME_VERSION = "astevolve.node_optimization_runtime.v1"
MASKED_LM_SOFT_PRIOR_ID = "masked_lm_plus_node_soft_weights_v1"

_OPTIMIZABLE_HANDLERS = frozenset(
    {
        "point_substitution",
        "segment_resample",
        "site_resample",
        "segment_mutagenesis",
        "cdr_resample",
    }
)


def _residue_set(value: Any) -> set[str]:
    if value is None:
        return set()
    raw: Sequence[Any]
    if isinstance(value, str):
        raw = list(value) if all(char in AA for char in value) else [value]
    elif isinstance(value, Sequence):
        raw = value
    else:
        raw = [value]
    return {str(item).strip().upper() for item in raw if str(item).strip().upper() in AA}


class _MaskedLMSoftWeightPriorScorer:


    scorer_id = MASKED_LM_SOFT_PRIOR_ID
    scorer_version = "1"

    def __init__(self, prior: MaskedProteinLMPrior) -> None:
        self._prior = prior
        self._request_hash: Optional[str] = None
        self._log_probs: Dict[int, Dict[str, float]] = {}
        self.model_id: Optional[str] = None
        self._ranking_cache: Dict[
            str, Tuple[Dict[int, Dict[str, float]], Optional[str]]
        ] = {}

    def _prepare(self, request: NodeSequenceOptimizationRequest) -> None:


        ranking_hash = request.ranking_hash
        if self._request_hash == ranking_hash:
            return
        cached = self._ranking_cache.get(ranking_hash)
        if cached is not None:
            self._log_probs, self.model_id = cached
            self._request_hash = ranking_hash
            return
        sequence = request.parent_sequences[request.target_chain]
        model_positions = tuple(
            position
            for position in request.legal_positions
            if position not in request.exact_residue_probabilities
        )
        marginals = self._prior.batch_masked_marginals(sequence, model_positions)
        self._log_probs = {
            item.position: dict(item.log_probabilities) for item in marginals
        }
        self.model_id = marginals[0].model_id if marginals else None
        self._request_hash = ranking_hash
        self._ranking_cache[ranking_hash] = (self._log_probs, self.model_id)


        if len(self._ranking_cache) > 128:
            oldest = next(iter(self._ranking_cache))
            self._ranking_cache.pop(oldest, None)

    def score(
        self,
        request: NodeSequenceOptimizationRequest,
        position: int,
        residue: str,
    ) -> float:
        exact = request.exact_residue_probabilities.get(position)
        if exact is not None:
            if residue not in exact:
                raise NodeSequenceOptimizationError(
                    "exact_probability_residue_not_executable",
                    f"{request.target_chain}:{position}:{residue}",
                )


            return request.temperature * math.log(float(exact[residue]))
        self._prepare(request)
        model_log_probability = float(self._log_probs[position][residue])
        soft_weight = float(
            request.soft_residue_weights.get(position, {}).get(residue, 1.0)
        )
        return model_log_probability + math.log(max(soft_weight, 1e-300))


@lru_cache(maxsize=8)
def _cached_optimizer(
    prior_model: str,
    model_path: Optional[str],
    device: str,
) -> NodeSequenceOptimizer:
    selected = str(prior_model or "heuristic").strip().lower()
    if selected in {"heuristic", "soft_weight", "soft_weights"}:
        return DefaultNodeSequenceOptimizer(SoftWeightResiduePriorScorer())
    if selected in {"masked_lm", "esm", "esm2", "masked_lm+heuristic"}:
        prior = MaskedProteinLMPrior(
            model_path=model_path,
            device=device,
            batch_size=32,
        )
        return DefaultNodeSequenceOptimizer(
            _MaskedLMSoftWeightPriorScorer(prior)
        )
    raise ValueError(
        "node_optimizer_prior_model must be one of heuristic or masked_lm; "
        f"got {prior_model!r}"
    )


def clear_node_optimizer_cache() -> None:


    _cached_optimizer.cache_clear()


def _optimizer_for_config(cfg: SAConfig) -> NodeSequenceOptimizer:
    raw_path = getattr(cfg, "node_optimizer_model_path", None)
    model_path = str(raw_path).strip() if raw_path else None
    if model_path is None:
        env_path = os.environ.get("ASTEVOLVE_MASKED_LM_MODEL_DIR", "").strip()
        model_path = env_path or None
    configured_device = str(os.environ.get(MASKED_LM_DEVICE_ENV, "") or "").strip()
    device = configured_device or str(
        getattr(cfg, "node_optimizer_device", "cuda") or "cuda"
    )
    return _cached_optimizer(
        str(getattr(cfg, "node_optimizer_prior_model", "heuristic")),
        model_path,
        device,
    )


def _allowed_residues_for_position(
    *,
    parent_residue: str,
    node_policy: Mapping[str, Any],
    position_rule: Mapping[str, Any],
    case_allowed_residues: Optional[Sequence[str]] = None,
) -> List[str]:
    allowed = set(AA)


    if case_allowed_residues is not None:
        allowed.intersection_update(case_allowed_residues)
    global_allowed = _residue_set(node_policy.get("hard_allowed_residues"))
    if global_allowed:
        allowed.intersection_update(global_allowed)
    allowed.difference_update(
        _residue_set(node_policy.get("hard_forbidden_residues"))
    )
    position_allowed = _residue_set(
        position_rule.get("hard_allowed_residues")
    )
    if position_allowed:
        allowed.intersection_update(position_allowed)
    allowed.difference_update(
        _residue_set(position_rule.get("hard_forbidden_residues"))
    )
    if not any(residue != parent_residue for residue in allowed):
        raise ValueError(
            "node optimizer hard residue rules leave no legal substitution"
        )
    return sorted(allowed)


def _node_policy_artifact(node_policy: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "priority_boost",
        "mutation_rate",
        "max_mutations_per_step",
        "edit_intent",
        "favored_residues",
        "favored_residue_classes",
        "disfavored_residues",
        "hard_allowed_residues",
        "hard_forbidden_residues",
        "site_anchors",
        "anchor_positions",
        "hotspot_positions",
        "operator_phase",
        "position_residue_rules",
        "mutation_ops",
    )
    return {key: node_policy.get(key) for key in keys if key in node_policy}


def _fallback_single_candidate(
    *,
    seqs: Dict[str, str],
    seg: Any,
    designable_positions: List[int],
    rng: np.random.Generator,
    cfg: SAConfig,
    masks: Dict[str, np.ndarray],
    internal_memory: Optional[Dict[str, Any]],
    mapping_action_spec: Optional[Dict[str, Any]],
    fixed_residues: Optional[Mapping[str, Mapping[int, str]]],
    generation_step: int,
    sequence_generator_registry: Optional[SequenceGeneratorRegistry],
    forced_proposal_tier: Optional[str],
) -> List[Tuple[Dict[str, str], Dict[str, Any]]]:
    return [
        _mutate_node_seqs(
            seqs,
            seg,
            designable_positions,
            rng,
            cfg,
            masks,
            internal_memory,
            mapping_action_spec=mapping_action_spec,
            fixed_residues=fixed_residues,
            generation_step=generation_step,
            sequence_generator_registry=sequence_generator_registry,
            forced_proposal_tier=forced_proposal_tier,
        )
    ]


def mutate_node_candidates(
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
    node_sequence_optimizer: Optional[NodeSequenceOptimizer] = None,
    excluded_target_sequences: Sequence[str] = (),
    optimizer_seed: Optional[int] = None,
    forced_proposal_tier: Optional[str] = None,
) -> List[Tuple[Dict[str, str], Dict[str, Any]]]:


    fallback_kwargs = {
        "seqs": seqs,
        "seg": seg,
        "designable_positions": designable_positions,
        "rng": rng,
        "cfg": cfg,
        "masks": masks,
        "internal_memory": internal_memory,
        "mapping_action_spec": mapping_action_spec,
        "fixed_residues": fixed_residues,
        "generation_step": generation_step,
        "sequence_generator_registry": sequence_generator_registry,
        "forced_proposal_tier": forced_proposal_tier,
    }
    if not bool(getattr(cfg, "node_optimizer_enabled", False)):
        return _fallback_single_candidate(**fallback_kwargs)

    cid = str(seg.chain_id)
    if cid not in seqs or cid not in masks:
        return _fallback_single_candidate(**fallback_kwargs)
    positions = [
        int(position)
        for position in designable_positions
        if 0 <= int(position) < len(seqs[cid])
        and bool(masks[cid][int(position)])
    ]
    mapping = (
        dict(mapping_action_spec)
        if isinstance(mapping_action_spec, Mapping) and mapping_action_spec
        else {}
    )
    structural_node_id = str(mapping.get("structural_node_id") or seg.name)
    distribution_policy = resolve_position_distribution_policy(cfg)
    node_distribution_rows = distribution_policy.for_node(
        cid, structural_node_id
    )
    if mapping:
        if str(mapping.get("chain_id") or "") != cid:
            raise ValueError("mapping action chain_id does not match selected segment")
        legal = {int(position) for position in mapping.get("legal_positions", [])}
        positions = [position for position in positions if position in legal]
    fixed = {
        int(position)
        for position in (fixed_residues or {}).get(cid, {})
    }
    positions = [position for position in positions if position not in fixed]

    node_policy = _node_policy(cfg, seg)
    protected = set(
        _policy_abs_positions(seg, node_policy, "protected_positions")
    )
    positions = [position for position in positions if position not in protected]
    missing_distribution_positions = sorted(
        row.position
        for row in node_distribution_rows
        if row.position not in positions
        and not (
            row.required_mutation is not None
            and 0 <= row.position < len(seqs[cid])
            and seqs[cid][row.position] == row.required_mutation
        )
    )
    if missing_distribution_positions:
        raise ValueError(
            "compiled position distribution falls outside executable node scope "
            f"{structural_node_id!r}: {missing_distribution_positions}"
        )


    satisfied_required_positions = {
        row.position
        for row in node_distribution_rows
        if row.required_mutation is not None
        and seqs[cid][row.position] == row.required_mutation
    }
    positions = [
        position
        for position in positions
        if position not in satisfied_required_positions
    ]
    if not positions:


        return []

    mutation_plan = _proposal_plan(
        seg,
        node_policy,
        cfg,
        rng,
        mapping_action_spec=mapping or None,
        forced_tier=forced_proposal_tier,
    )
    if (
        str(mutation_plan.get("tier", "")).lower() == "frozen"
        or int((mutation_plan.get("budget") or {}).get("max", 0) or 0) <= 0
    ):
        return _fallback_single_candidate(**fallback_kwargs)
    op_weights = mutation_plan.get("op_weights")
    if not isinstance(op_weights, Mapping) or not op_weights:
        raise RuntimeError("Executable mutation plan has no effective operator weights")
    op = _choose_op(rng, dict(op_weights), mode="node")
    operator_spec = require_operator(op, mode="node")
    if operator_spec.node_handler not in _OPTIMIZABLE_HANDLERS:
        return _fallback_single_candidate(**fallback_kwargs)

    raw_budget = mutation_plan.get("budget")
    budget = dict(raw_budget) if isinstance(raw_budget, Mapping) else {}
    action_budgets = mutation_plan.get("allowed_action_budgets")
    if isinstance(action_budgets, Mapping) and isinstance(
        action_budgets.get(op), Mapping
    ):
        budget = dict(action_budgets[op])
    minimum = max(1, int(budget.get("min", 1) or 1))
    maximum = int(budget.get("max", 0) or 0)
    if maximum <= 0:
        maximum = min(len(positions), int(cfg.exploit_max_mutations))
    maximum = min(maximum, len(positions))
    required_rows = {
        row.position: row
        for row in node_distribution_rows
        if row.required_mutation is not None
        and row.position not in satisfied_required_positions
    }
    minimum = max(minimum, len(required_rows))
    if minimum > maximum:
        raise ValueError(
            f"node optimizer mutation budget [{minimum}, {maximum}] is infeasible"
        )

    parent = seqs[cid]
    allowed_residues: Dict[int, List[str]] = {}
    residue_contract = dict(
        getattr(cfg, "residue_mutation_contract", {}) or {}
    )
    chain_contract: Optional[Mapping[int, Sequence[str]]] = None
    if residue_contract:
        if cid not in residue_contract:
            raise ValueError(
                "residue_mutation_contract does not declare target chain "
                f"{cid!r}"
            )
        chain_contract = residue_contract[cid]
        missing_positions = [
            position for position in positions if position not in chain_contract
        ]
        if missing_positions:
            raise ValueError(
                "residue_mutation_contract does not declare optimizer position(s) "
                f"for chain {cid!r}: {missing_positions}"
            )
    soft_weights: Dict[int, Dict[str, float]] = {}
    exact_probabilities: Dict[int, Dict[str, float]] = {}
    distribution_hashes: Dict[int, str] = {}
    required_mutations: Dict[int, str] = {}
    distributions_by_position: Dict[int, RuntimePositionDistribution] = {
        row.position: row for row in node_distribution_rows
    }
    base_weights = _aa_weights_for_segment(
        seg,
        internal_memory,
        node_policy=node_policy,
    )
    base_weights = _apply_forbidden_residue_filter(
        base_weights,
        list(mutation_plan.get("forbidden_residues", []) or []),
    )
    for position in positions:
        compiled_distribution = distributions_by_position.get(position)
        if compiled_distribution is not None:
            executable = compiled_distribution.effective_probability_map
            if compiled_distribution.required_mutation is not None:
                required = compiled_distribution.required_mutation
                allowed = [required]
                exact = {required: 1.0}
                required_mutations[position] = required
            else:
                allowed = sorted(executable)
                exact = {residue: float(executable[residue]) for residue in allowed}
            if chain_contract is not None and not set(allowed).issubset(
                set(chain_contract[position])
            ):
                raise ValueError(
                    "compiled position distribution widens residue mutation "
                    f"contract at {cid}:{position}"
                )
            if not any(residue != parent[position] for residue in allowed):
                raise ValueError(
                    "compiled position distribution leaves no executable "
                    f"substitution at {cid}:{position}"
                )
            allowed_residues[position] = allowed

            soft_weights[position] = dict(exact)
            exact_probabilities[position] = dict(exact)
            distribution_hashes[position] = compiled_distribution.distribution_hash
            continue
        rule = _position_residue_rule(node_policy, seg, position)
        allowed = _allowed_residues_for_position(
            parent_residue=parent[position],
            node_policy=node_policy,
            position_rule=rule,
            case_allowed_residues=(
                chain_contract[position]
                if chain_contract is not None else None
            ),
        )
        allowed_residues[position] = allowed
        position_weights = _aa_weights_for_position(
            base_weights, node_policy, seg, position
        )
        soft_weights[position] = {
            residue: float(position_weights[index])
            for index, residue in enumerate(AA)
            if residue in allowed
        }

    request = NodeSequenceOptimizationRequest.create(
        parent_sequences=seqs,
        structural_node_id=structural_node_id,
        mapping_edge_id=str(mapping.get("edge_id") or f"search::{seg.name}"),
        action_id=str(mapping.get("action_id") or f"search::{seg.name}::{op}"),
        operator=str(op),
        target_chain=cid,
        legal_positions=positions,
        allowed_residues=allowed_residues,
        mutation_budget={"min": minimum, "max": maximum},
        soft_residue_weights=soft_weights,
        exact_residue_probabilities=exact_probabilities,
        position_distribution_hashes=distribution_hashes,
        required_mutations=required_mutations,
        candidate_count=int(cfg.node_optimizer_candidate_count),
        beam_width=int(cfg.node_optimizer_beam_width),
        top_k_per_position=int(cfg.node_optimizer_top_k_per_position),
        temperature=float(cfg.node_optimizer_temperature),
        diversity_weight=float(cfg.node_optimizer_diversity_weight),
        mutation_penalty=float(cfg.node_optimizer_mutation_penalty),
        seed=(
            int(optimizer_seed)
            if optimizer_seed is not None
            else int(cfg.seed if cfg.seed is not None else 0)
            + int(generation_step)
        ),
        excluded_sequences=tuple(str(sequence) for sequence in excluded_target_sequences),
    )
    optimizer = node_sequence_optimizer or _optimizer_for_config(cfg)
    try:
        result = optimizer.optimize(request)
        result = validate_node_sequence_optimization_result(request, result)
    except NodeSequenceOptimizationError as exc:
        if exc.code == "candidate_space_exhausted":
            return []
        raise

    output: List[Tuple[Dict[str, str], Dict[str, Any]]] = []
    result_artifact = result.to_dict()
    for rank, candidate in enumerate(result.candidates):
        generated = dict(seqs)
        generated[cid] = candidate.sequence
        distribution_receipt = distribution_policy.execution_receipt(
            node_distribution_rows,
            parent_sequence=seqs[cid],
            final_sequence=candidate.sequence,
        )
        unrealized_required = [
            item
            for item in distribution_receipt["realizations"]
            if item["required_mutation"] is not None
            and item["required_realized"] is not True
        ]
        if unrealized_required:
            raise ValueError(
                "node optimizer failed required compiled position distribution: "
                f"{[(item['chain_id'], item['position']) for item in unrealized_required]}"
            )
        move: Dict[str, Any] = {
            "op": op,
            "node": seg.name,
            "node_kind": seg.kind,
            "chain_id": cid,
            "positions": {cid: [action.position for action in candidate.actions]},
            "segments": [(cid, seg.name, seg.spans)],
            "changes": [
                {
                    "chain_id": cid,
                    "position": action.position,
                    "from": action.from_residue,
                    "to": action.to_residue,
                    "node": seg.name,
                }
                for action in candidate.actions
            ],
            "mutation_plan": mutation_plan,
            "operator_spec": operator_manifest(op, mode="node"),
            "operator_selection": {
                "selected": op,
                "source": "mutation_plan.op_weights",
                "effective_weights": dict(op_weights),
            },
            "proposal_log_prior": float(candidate.log_prior),
            "node_optimization": {
                "schema_version": NODE_OPTIMIZATION_RUNTIME_VERSION,
                "validation_stage": "before_sequence_claim_and_scoring",
                "selected_optimizer_id": result.optimizer_id,
                "request": request.to_dict(),
                "result": result_artifact,
                "candidate_rank": rank,
                "candidate_id": candidate.candidate_id,
                "candidate": candidate.to_dict(),
                "position_distribution_execution": distribution_receipt,
            },
        }


        if isinstance(action_budgets, Mapping) and isinstance(
            action_budgets.get(op), Mapping
        ):
            move["selected_action_budget"] = dict(budget)
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
        if node_policy:
            move["node_policy"] = _node_policy_artifact(node_policy)
        for change in move["changes"]:
            rule = _position_residue_rule(
                node_policy, seg, int(change["position"])
            )
            if rule:
                change["position_residue_rule"] = dict(rule)
        finalized = finalize_mutation_move(seqs, generated, move)
        count = int(
            (finalized.get("actual_delta") or {}).get(
                "residue_change_count", 0
            )
        )
        if count < minimum or count > maximum:
            raise ValueError(
                f"optimized action realized {count} changes outside "
                f"declared budget [{minimum}, {maximum}]"
            )
        output.append((generated, finalized))
    if not output:
        raise RuntimeError("node optimizer returned no candidates")
    return output


def record_node_optimization_attempt(
    history: Dict[str, Any], move: Mapping[str, Any]
) -> None:


    artifact = move.get("node_optimization")
    if not isinstance(artifact, Mapping):
        return
    request = artifact.get("request")
    result = artifact.get("result")
    if not isinstance(request, Mapping) or not isinstance(result, Mapping):
        raise ValueError("node optimization artifact is incomplete")
    summary = history.setdefault(
        "node_optimization",
        {
            "schema_version": "astevolve.node_optimization_summary.v1",
            "enabled": True,
            "selected_optimizer_id": artifact.get("selected_optimizer_id"),
            "proposal_prior_role": "mcts_edge_prior_only",
            "fast_reward_includes_proposal_prior": False,
            "batches": 0,
            "generated": 0,
            "selected_for_fast_evaluation": 0,
            "request_hashes": [],
            "result_hashes": [],
            "prior_models": [],
        },
    )
    request_hash = str(request.get("request_hash") or "")
    result_hash = str(result.get("result_hash") or "")
    if result_hash not in summary["result_hashes"]:
        summary["batches"] += 1
        summary["generated"] += len(result.get("candidates") or [])
        summary["request_hashes"].append(request_hash)
        summary["result_hashes"].append(result_hash)
        prior_model = str(
            (result.get("provenance") or {}).get("prior_scorer_id") or ""
        )
        if prior_model and prior_model not in summary["prior_models"]:
            summary["prior_models"].append(prior_model)
    summary["selected_for_fast_evaluation"] += 1


__all__ = [
    "MASKED_LM_SOFT_PRIOR_ID",
    "NODE_OPTIMIZATION_RUNTIME_VERSION",
    "clear_node_optimizer_cache",
    "mutate_node_candidates",
    "record_node_optimization_attempt",
]
