

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


MOVE_SCHEMA_VERSION = "astevolve.mutation_move.v2"


class MoveContractError(ValueError):
    pass


def _ordered_chain_ids(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> Iterable[str]:
    seen = set()
    for chain_id in (*before.keys(), *after.keys()):
        if chain_id not in seen:
            seen.add(chain_id)
            yield str(chain_id)


def _target_nodes(move: Mapping[str, Any]) -> list[str]:
    nodes = []
    if move.get("node"):
        nodes.append(str(move["node"]))
    for segment in move.get("segments", []) or []:
        if isinstance(segment, (list, tuple)) and len(segment) >= 2 and segment[1]:
            nodes.append(str(segment[1]))
    return list(dict.fromkeys(nodes))


def _canonical_delta(
    before: Mapping[str, str],
    after: Mapping[str, str],
    *,
    node: Optional[str],
    motif: Optional[str],
    position_keys: Iterable[str],
) -> Tuple[Dict[str, list[int]], list[Dict[str, Any]], Dict[str, int]]:
    changed_by_chain: Dict[str, list[int]] = {}
    changes: list[Dict[str, Any]] = []
    length_delta_by_chain: Dict[str, int] = {}
    requested_keys = [str(key) for key in position_keys]

    for chain_id in _ordered_chain_ids(before, after):
        old_sequence = str(before.get(chain_id, ""))
        new_sequence = str(after.get(chain_id, ""))
        length_delta_by_chain[chain_id] = len(new_sequence) - len(old_sequence)
        positions = []
        for position in range(max(len(old_sequence), len(new_sequence))):
            old = old_sequence[position] if position < len(old_sequence) else None
            new = new_sequence[position] if position < len(new_sequence) else None
            if old == new:
                continue
            positions.append(position)
            change: Dict[str, Any] = {
                "chain_id": chain_id,
                "position": position,
                "from": old,
                "to": new,
            }
            if node:
                change["node"] = node
            if motif:
                change["motif"] = motif
            changes.append(change)
        if positions or chain_id in requested_keys:
            changed_by_chain[chain_id] = positions

    for chain_id in requested_keys:
        changed_by_chain.setdefault(chain_id, [])
    return changed_by_chain, changes, length_delta_by_chain


def finalize_mutation_move(
    before: Mapping[str, str],
    after: Mapping[str, str],
    move: Dict[str, Any],
) -> Dict[str, Any]:


    move = dict(move)
    attempted = deepcopy(move.get("attempted_positions", move.get("positions", {})))
    move["attempted_positions"] = attempted if isinstance(attempted, dict) else {}
    move["target_nodes"] = _target_nodes(move)
    node = str(move.get("node")) if move.get("node") else None
    if node is None and len(move["target_nodes"]) == 1:
        node = move["target_nodes"][0]
    motif = str(move.get("motif")) if move.get("motif") else None
    positions, changes, length_delta = _canonical_delta(
        before,
        after,
        node=node,
        motif=motif,
        position_keys=move["attempted_positions"].keys(),
    )
    move["schema_version"] = MOVE_SCHEMA_VERSION
    move["positions"] = positions
    move["changes"] = changes
    move["actual_delta"] = {
        "residue_change_count": len(changes),
        "length_delta_by_chain": length_delta,
    }

    has_delta = bool(changes) or any(delta != 0 for delta in length_delta.values())
    existing_outcome = str(move.get("outcome") or "")
    if has_delta:
        if existing_outcome in {"rejected", "frozen"}:
            raise MoveContractError(
                f"Move outcome {existing_outcome!r} cannot contain an actual sequence delta"
            )
        move["outcome"] = "executed"
    elif existing_outcome not in {"rejected", "frozen", "noop"}:
        move["outcome"] = "noop"

    validate_move_contract(before, after, move)
    return move


def validate_move_contract(
    before: Mapping[str, str],
    after: Mapping[str, str],
    move: Mapping[str, Any],
) -> None:


    if move.get("schema_version") != MOVE_SCHEMA_VERSION:
        raise MoveContractError("Mutation move schema_version is missing or unsupported")
    operator = move.get("op")
    spec = move.get("operator_spec")
    if isinstance(spec, Mapping) and operator != spec.get("name"):
        raise MoveContractError("move.op disagrees with operator_spec.name")
    selection = move.get("operator_selection")
    if isinstance(selection, Mapping) and operator != selection.get("selected"):
        raise MoveContractError("move.op disagrees with operator_selection.selected")

    node = str(move.get("node")) if move.get("node") else None
    target_nodes_blob = [
        str(node_name) for node_name in move.get("target_nodes", []) or []
    ]
    if node is None and len(target_nodes_blob) == 1:
        node = target_nodes_blob[0]
    motif = str(move.get("motif")) if move.get("motif") else None
    position_blob = move.get("positions", {})
    position_keys = position_blob.keys() if isinstance(position_blob, Mapping) else ()
    positions, changes, length_delta = _canonical_delta(
        before,
        after,
        node=node,
        motif=motif,
        position_keys=position_keys,
    )
    if move.get("positions") != positions:
        raise MoveContractError("move.positions disagrees with actual sequence delta")
    if move.get("changes") != changes:
        raise MoveContractError("move.changes disagrees with actual sequence delta")
    expected_delta = {
        "residue_change_count": len(changes),
        "length_delta_by_chain": length_delta,
    }
    if move.get("actual_delta") != expected_delta:
        raise MoveContractError("move.actual_delta disagrees with actual sequence delta")

    target_nodes = set(target_nodes_blob)
    changed_nodes = set(
        str(change["node"])
        for change in changes
        if isinstance(change, Mapping) and change.get("node")
    )
    if not changed_nodes.issubset(target_nodes):
        raise MoveContractError("move.changes contains a node outside target_nodes")

    has_delta = bool(changes) or any(delta != 0 for delta in length_delta.values())
    outcome = str(move.get("outcome") or "")
    if has_delta and outcome != "executed":
        raise MoveContractError("A move with an actual delta must have outcome='executed'")
    if not has_delta and outcome == "executed":
        raise MoveContractError("A move without an actual delta cannot have outcome='executed'")


__all__ = [
    "MOVE_SCHEMA_VERSION",
    "MoveContractError",
    "finalize_mutation_move",
    "validate_move_contract",
]
