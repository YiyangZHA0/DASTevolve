

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from astevolve.search.artifact_io import _seqs_hash
from astevolve.search.candidate_validation import _candidate_fast_filter
from astevolve.search.causal_binding import (
    bind_candidate as _bind_causal_candidate,
    bind_move as _bind_causal_move,
)
from astevolve.search.config import SAConfig
from astevolve.search.energy_reporting import fast_energy_record
from astevolve.search.generation_runtime import record_generation_attempt
from astevolve.search.parent_baseline import build_parent_baseline_candidate
from astevolve.search.run_memory import InnerRunMemory
from astevolve.search.sequence_ops import init_seqs, mutate_seqs
from engine.history_runtime import register_sequence_occurrence


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


def run_sa_search(
    *,
    compiled: Dict[str, Any],
    terms_fast: List[tuple[float, Any]],
    cfg: SAConfig,
    masks: Dict[str, np.ndarray],
    rng: np.random.Generator,
    template_seqs: Optional[Dict[str, str]],
    fixed_residues: Optional[Dict[str, Dict[int, str]]],
    run_memory: Optional[InnerRunMemory],
    causal_context: Optional[Mapping[str, Any]],
    score_fast: FastScore,
    inner_structure_evaluator: Optional[InnerStructureEvaluator] = None,
) -> Tuple[
    Dict[str, str],
    Dict[str, Any],
    Dict[str, float],
    float,
    Dict[str, Any],
    List[Dict[str, Any]],
    Dict[str, Any],
    str,
]:


    cur = init_seqs(
        compiled["chain_lengths"],
        rng,
        template_seqs=template_seqs,
        fixed_residues=fixed_residues,
    )
    register_sequence_occurrence(
        cur,
        role="root",
        context_id="sa:root",
        metadata={"search_method": "sa"},
    )
    cur_break, cur_progen, cur_fast, root_fast_cache_hit = score_fast(
        cur, terms_fast, cfg, compiled, run_memory
    )
    if run_memory is not None:
        run_memory.claim_sequence(cur, node_id="root")
        run_memory.record_transposition(
            cur,
            node_id="root",
            payload={"fast_loss": float(cur_fast)},
        )
    search_artifacts = {
        "method": "sa",
        "artifact_paths": {},
        "root_candidate": build_parent_baseline_candidate(
            cur,
            cur_break,
            cur_progen,
            cur_fast,
            template_seqs=template_seqs,
            fixed_residues=fixed_residues,
            compiled=compiled,
            cfg=cfg,
        ),
    }
    _bind_causal_candidate(search_artifacts["root_candidate"], causal_context)
    root_inner_structure = (
        inner_structure_evaluator(search_artifacts["root_candidate"])
        if inner_structure_evaluator is not None else None
    )
    cur_selection_loss = float(cur_fast)
    cur_structure_gate_pass = True
    if root_inner_structure is not None:
        cur_selection_loss = float(root_inner_structure["selection_loss"])
        cur_structure_gate_pass = bool(root_inner_structure.get("gate_pass", False))
        search_artifacts["root_candidate"].update({
            "selection_loss": cur_selection_loss,
            "inner_structure_loss": root_inner_structure.get("structure_combined_energy"),
            "inner_structure_gate_pass": cur_structure_gate_pass,
            "inner_structure_evaluation": root_inner_structure,
        })
        root_energy = search_artifacts["root_candidate"].get("energy")
        if isinstance(root_energy, dict):
            root_energy["hard_gate_pass"] = cur_structure_gate_pass
            root_energy["selection_loss"] = cur_selection_loss

    best = cur
    best_break = cur_break
    best_progen = cur_progen
    best_fast = cur_fast
    best_selection_loss = cur_selection_loss
    current_variant_id = "root"
    fast_selected_variant_id = "root"
    temperature = float(cfg.init_temp)
    history: Dict[str, Any] = {
        "accepted_moves": [],
        "op_counts": {},
        "fast_filter_failures": {},
        "search_method": "sa",
        "duplicate_sequence_attempts": 0,
        "fast_cache_hits": int(root_fast_cache_hit),
    }
    candidates: List[Dict[str, Any]] = []

    for step in range(cfg.iterations):
        proposal_parent_id = current_variant_id
        prop, move = mutate_seqs(
            cur,
            compiled,
            rng,
            cfg,
            masks=masks,
            fixed_residues=fixed_residues,
            generation_step=step,
        )
        record_generation_attempt(history, move)
        _bind_causal_move(move, causal_context)
        candidate_id = f"sa_{step + 1}"
        register_sequence_occurrence(
            prop,
            role="candidate",
            context_id=f"sa:{step + 1}",
            metadata={"search_method": "sa", "step": step},
        )
        claim = run_memory.claim_sequence(prop) if run_memory is not None else None

        fast_filter = _candidate_fast_filter(
            prop,
            template_seqs,
            fixed_residues,
            compiled,
            cfg,
        )
        if fast_filter.get("pass", True):
            prop_break, prop_progen, prop_fast, fast_cache_hit = score_fast(
                prop, terms_fast, cfg, compiled, run_memory
            )
        else:
            fail_penalty = 1000.0 + 100.0 * float(
                len(fast_filter.get("reasons", []) or [])
            )
            prop_break = {
                "total": float(cur_break.get("total", 0.0)) + fail_penalty,
                "fast_filter": fast_filter,
            }
            prop_progen = {"loglik_sum": 0.0, "loglik_avg": 0.0}
            prop_fast = float(cur_fast) + fail_penalty
            fast_cache_hit = False
            for reason in fast_filter.get("reasons", []) or ["unknown"]:
                history["fast_filter_failures"][reason] = (
                    history["fast_filter_failures"].get(reason, 0) + 1
                )

        inner_structure_evaluation = None
        prop_selection_loss = float(prop_fast)
        inner_structure_gate_pass = bool(fast_filter.get("pass", True))
        if inner_structure_gate_pass and inner_structure_evaluator is not None:
            inner_structure_evaluation = inner_structure_evaluator({
                "variant_id": candidate_id,
                "parent_id": proposal_parent_id,
                "seq_hash": _seqs_hash(prop),
                "seqs": prop,
                "fast_loss": float(prop_fast),
                "constraint_penalty": float(prop_break["total"]),
                "progen_loglik_avg": float(prop_progen["loglik_avg"]),
                "progen_loglik_sum": float(prop_progen["loglik_sum"]),
                "fast_filter": fast_filter,
                "move": move,
            })
            prop_selection_loss = float(inner_structure_evaluation["selection_loss"])
            inner_structure_gate_pass = bool(inner_structure_evaluation.get("gate_pass", False))

        action_reward = float(cur_selection_loss) - float(prop_selection_loss)
        accept = bool(inner_structure_gate_pass and prop_selection_loss <= cur_selection_loss)
        if inner_structure_gate_pass and not accept and temperature > 1e-8:
            accept = bool(
                rng.random()
                < float(np.exp((cur_selection_loss - prop_selection_loss) / temperature))
            )

        if accept:
            cur, cur_fast = prop, prop_fast
            cur_selection_loss = prop_selection_loss
            cur_break, cur_progen = prop_break, prop_progen
            current_variant_id = candidate_id
            history["op_counts"][move["op"]] = (
                history["op_counts"].get(move["op"], 0) + 1
            )
            if cfg.history_size > 0:
                history["accepted_moves"].append(move)
                if len(history["accepted_moves"]) > cfg.history_size:
                    history["accepted_moves"].pop(0)
            if inner_structure_gate_pass and cur_selection_loss < best_selection_loss:
                best, best_fast = cur, cur_fast
                best_selection_loss = cur_selection_loss
                best_break, best_progen = cur_break, cur_progen
                fast_selected_variant_id = candidate_id

        if run_memory is not None:
            run_memory.record_action(
                str(move.get("op") or "unknown"),
                accepted=accept,
                reward=action_reward,
            )
            if claim is not None and not claim.is_new:
                history["duplicate_sequence_attempts"] += 1
            elif claim is not None:
                run_memory.record_transposition(
                    prop,
                    node_id=candidate_id,
                    payload={"fast_loss": float(prop_fast), "accepted": accept},
                )
        history["fast_cache_hits"] += int(fast_cache_hit)

        candidates.append(
            _bind_causal_candidate(
                {
                    "variant_id": candidate_id,
                    "parent_id": proposal_parent_id,
                    "seq_hash": _seqs_hash(prop),
                    "seqs": prop,
                    "fast_loss": float(prop_fast),
                    "selection_loss": float(prop_selection_loss),
                    "inner_structure_loss": (inner_structure_evaluation or {}).get("structure_combined_energy"),
                    "inner_structure_gate_pass": bool(inner_structure_gate_pass),
                    "inner_structure_evaluation": inner_structure_evaluation,
                    "constraint_penalty": float(prop_break["total"]),
                    "progen_loglik_avg": float(prop_progen["loglik_avg"]),
                    "progen_loglik_sum": float(prop_progen["loglik_sum"]),
                    "energy": fast_energy_record(
                        fast_loss=prop_fast,
                        constraint_penalty=prop_break["total"],
                        progen_loglik_avg=prop_progen["loglik_avg"],
                        progen_weight=cfg.progen_weight,
                        hard_gate_pass=bool(inner_structure_gate_pass),
                    ),
                    "fast_filter": fast_filter,
                    "move": move,
                    "duplicate_sequence": bool(
                        claim is not None and not claim.is_new
                    ),
                    "transposition_target": (
                        claim.transposition_node_id
                        if claim is not None and not claim.is_new
                        else None
                    ),
                    "fast_cache_hit": bool(fast_cache_hit),
                },
                causal_context,
            )
        )
        temperature *= float(cfg.cooling)

    return (
        best,
        best_break,
        best_progen,
        float(best_fast),
        history,
        candidates,
        search_artifacts,
        fast_selected_variant_id,
    )


__all__ = ["run_sa_search"]
