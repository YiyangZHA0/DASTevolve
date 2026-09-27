

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


FIDELITY_UPGRADE_VERSION = "astevolve.mcts_fidelity_upgrade.v1"


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("variant_id") or "")


def _candidate_loss(candidate: Mapping[str, Any]) -> float:
    try:
        value = float(candidate.get("selection_loss", float("inf")))
    except (TypeError, ValueError):
        return float("inf")
    return value if math.isfinite(value) else float("inf")


def _candidate_node(candidate: Mapping[str, Any]) -> str:
    move = candidate.get("move")
    if not isinstance(move, Mapping):
        return "unknown"
    return str(
        move.get("node")
        or move.get("structural_node_id")
        or (move.get("selection") or {}).get("structural_node_id")
        or "unknown"
    )


def _sequence_identity(candidate: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...]:
    sequences = candidate.get("seqs")
    if not isinstance(sequences, Mapping):
        return ()
    return tuple(
        sorted((str(chain), str(sequence)) for chain, sequence in sequences.items())
    )


def _sequence_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    lhs = dict(_sequence_identity(left))
    rhs = dict(_sequence_identity(right))
    distance = 0
    for chain in sorted(set(lhs) | set(rhs)):
        a = lhs.get(chain, "")
        b = rhs.get(chain, "")
        distance += abs(len(a) - len(b))
        distance += sum(x != y for x, y in zip(a, b))
    return distance


def _eligible(candidate: Mapping[str, Any], upgraded_ids: set[str]) -> bool:
    identity = _candidate_id(candidate)
    if not identity or identity in upgraded_ids or bool(candidate.get("duplicate_sequence")):
        return False
    evaluation = candidate.get("inner_structure_evaluation")
    result = evaluation.get("result") if isinstance(evaluation, Mapping) else None
    return bool(
        isinstance(evaluation, Mapping)
        and str(evaluation.get("status") or "") == "ok"
        and isinstance(result, Mapping)
        and isinstance(result.get("inner_evaluator_report"), Mapping)
        and math.isfinite(_candidate_loss(candidate))
    )


def _completed_proxy(candidate: Mapping[str, Any]) -> bool:


    if not _candidate_id(candidate) or bool(candidate.get("duplicate_sequence")):
        return False
    evaluation = candidate.get("inner_structure_evaluation")
    result = evaluation.get("result") if isinstance(evaluation, Mapping) else None
    return bool(
        isinstance(evaluation, Mapping)
        and str(evaluation.get("status") or "") == "ok"
        and isinstance(result, Mapping)
        and isinstance(result.get("inner_evaluator_report"), Mapping)
        and math.isfinite(_candidate_loss(candidate))
    )


def select_fidelity_upgrade_cohort(
    candidates: Sequence[MutableMapping[str, Any]],
    *,
    limit: int,
    wave_start: int,
    wave_end: int,
) -> List[Tuple[MutableMapping[str, Any], str]]:


    quota = max(0, int(limit))
    if quota == 0:
        return []
    upgraded_ids = {
        _candidate_id(candidate)
        for candidate in candidates
        if isinstance(candidate.get("mcts_fidelity_upgrade"), Mapping)
        and str(candidate["mcts_fidelity_upgrade"].get("status") or "") == "ok"
    }
    eligible = [
        candidate for candidate in candidates if _eligible(candidate, upgraded_ids)
    ]
    if not eligible:
        return []

    selected: List[Tuple[MutableMapping[str, Any], str]] = []
    selected_ids: set[str] = set()

    def add(candidate: MutableMapping[str, Any] | None, lane: str) -> None:
        if candidate is None or len(selected) >= quota:
            return
        identity = _candidate_id(candidate)
        if identity and identity not in selected_ids:
            selected.append((candidate, lane))
            selected_ids.add(identity)


    chronological = [
        candidate for candidate in candidates if _completed_proxy(candidate)
    ]
    wave = [
        candidate
        for candidate in chronological[max(0, int(wave_start)) : max(0, int(wave_end))]
        if candidate in eligible
    ]
    add(
        min(wave, key=lambda row: (_candidate_loss(row), _candidate_id(row)))
        if wave
        else None,
        "wave_leader",
    )
    remaining = [row for row in eligible if _candidate_id(row) not in selected_ids]
    add(
        min(remaining, key=lambda row: (_candidate_loss(row), _candidate_id(row)))
        if remaining
        else None,
        "global_proxy_leader",
    )

    upgraded_node_counts = Counter(
        _candidate_node(candidate)
        for candidate in candidates
        if _candidate_id(candidate) in upgraded_ids
    )
    remaining = [row for row in eligible if _candidate_id(row) not in selected_ids]
    if remaining:
        node = min(
            {_candidate_node(row) for row in remaining},
            key=lambda name: (upgraded_node_counts[name], name),
        )
        add(
            min(
                (row for row in remaining if _candidate_node(row) == node),
                key=lambda row: (_candidate_loss(row), _candidate_id(row)),
            ),
            "underrepresented_node",
        )

    while len(selected) < quota:
        remaining = [row for row in eligible if _candidate_id(row) not in selected_ids]
        if not remaining:
            break
        reference = [row for row, _lane in selected]
        if reference:
            diverse = max(
                remaining,
                key=lambda row: (
                    min(_sequence_distance(row, other) for other in reference),
                    -_candidate_loss(row),
                    _candidate_id(row),
                ),
            )
            add(diverse, "sequence_diversity")
        else:
            add(
                min(remaining, key=lambda row: (_candidate_loss(row), _candidate_id(row))),
                "proxy_rank_fill",
            )
    return selected


def select_final_fidelity_cohort(
    candidates: Sequence[MutableMapping[str, Any]],
    *,
    limit: int,
) -> List[Tuple[MutableMapping[str, Any], str]]:


    ranked = sorted(
        (
            candidate
            for candidate in candidates
            if not bool(candidate.get("duplicate_sequence"))
            and isinstance(candidate.get("inner_structure_evaluation"), Mapping)
            and str(candidate["inner_structure_evaluation"].get("status") or "") == "ok"
            and isinstance(candidate["inner_structure_evaluation"].get("result"), Mapping)
            and isinstance(
                candidate["inner_structure_evaluation"]["result"].get(
                    "inner_evaluator_report"
                ),
                Mapping,
            )
            and math.isfinite(_candidate_loss(candidate))
        ),
        key=lambda row: (
            float(row.get("proxy_selection_loss", _candidate_loss(row))),
            _candidate_id(row),
        ),
    )[: max(0, int(limit))]
    return [
        (candidate, "final_global_proxy_top")
        for candidate in ranked
        if not (
            isinstance(candidate.get("mcts_fidelity_upgrade"), Mapping)
            and str(candidate["mcts_fidelity_upgrade"].get("status") or "") == "ok"
        )
    ]


def apply_reward_delta(
    tree: MutableMapping[str, MutableMapping[str, Any]],
    node_id: str,
    *,
    old_reward: float,
    new_reward: float,
) -> float:


    delta = float(new_reward) - float(old_reward)
    current: str | None = str(node_id)
    while current is not None:
        node = tree[current]
        node["total_reward"] = float(node.get("total_reward", 0.0)) + delta
        node["fidelity_reward_delta_total"] = float(
            node.get("fidelity_reward_delta_total", 0.0)
        ) + delta
        current = node.get("parent")
    return delta


def refresh_best_reward(tree: MutableMapping[str, MutableMapping[str, Any]]) -> None:


    ordered = sorted(
        tree.values(), key=lambda node: int(node.get("depth", 0)), reverse=True
    )
    for node in ordered:
        values = []
        if node.get("reward") is not None:
            values.append(float(node["reward"]))
        for child_id in node.get("children", []) or []:
            child = tree.get(str(child_id))
            if child is not None and child.get("best_reward") is not None:
                values.append(float(child["best_reward"]))
        node["best_reward"] = max(values) if values else -1e9


__all__ = [
    "FIDELITY_UPGRADE_VERSION",
    "apply_reward_delta",
    "refresh_best_reward",
    "select_fidelity_upgrade_cohort",
    "select_final_fidelity_cohort",
]
