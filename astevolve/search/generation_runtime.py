

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from astevolve.core.amino_acids import AA

from .config import SAConfig
from .position_distribution_runtime import resolve_position_distribution_policy
from .sequence_generator import (
    SequenceGenerationRequest,
    SequenceGeneratorRegistry,
    default_sequence_generator_registry,
)


def _string_refs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    return sorted(
        {
            str(item).strip()
            for item in values
            if str(item).strip()
        }
    )


def realize_generated_substitutions(
    *,
    parent_sequences: Mapping[str, str],
    target_chain: str,
    write_positions: Sequence[int],
    operator: str,
    structural_node_id: str,
    cfg: SAConfig,
    masks: Mapping[str, np.ndarray],
    fixed_residues: Optional[Mapping[str, Mapping[int, str]]],
    position_weights: Mapping[int, Mapping[str, float]],
    mapping_action: Optional[Mapping[str, Any]],
    generation_step: int,
    registry: Optional[SequenceGeneratorRegistry] = None,
) -> tuple[Dict[str, str], Dict[str, Any]]:


    positions = sorted({int(position) for position in write_positions})
    if not positions:
        raise ValueError("sequence generator requires at least one write position")
    mapping = dict(mapping_action) if isinstance(mapping_action, Mapping) else {}
    edge_id = str(mapping.get("edge_id") or f"search::{structural_node_id}")
    action_id = str(
        mapping.get("action_id")
        or f"search::{structural_node_id}::{operator}"
    )
    mapped_structural_id = str(
        mapping.get("structural_node_id") or structural_node_id
    )
    distribution_policy = resolve_position_distribution_policy(cfg)
    node_distribution_rows = distribution_policy.for_node(
        target_chain, mapped_structural_id
    )
    distributions_by_position = {
        row.position: row for row in node_distribution_rows
    }
    parent_target = str(parent_sequences[target_chain])
    missing_required = sorted(
        row.position
        for row in node_distribution_rows
        if row.required_mutation is not None
        and parent_target[row.position] != row.required_mutation
        and row.position not in positions
    )
    if missing_required:
        raise ValueError(
            "required compiled position distribution missing from write set "
            f"for node {mapped_structural_id!r}: {missing_required}"
        )
    state_refs = _string_refs(
        getattr(cfg, "sequence_generator_state_condition_refs", [])
    )
    functional_node_id = str(mapping.get("functional_node_id") or "").strip()
    if functional_node_id:
        state_refs = sorted(
            {*state_refs, f"functional_node:{functional_node_id}"}
        )
    structure_refs = _string_refs(
        getattr(cfg, "sequence_generator_structure_condition_refs", [])
    )
    if node_distribution_rows:
        state_refs = sorted(
            {
                *state_refs,
                f"position_distribution_set:{distribution_policy.distribution_set_hash}",
                *(
                    f"position_distribution:{row.distribution_hash}"
                    for row in node_distribution_rows
                ),
            }
        )
    residue_contract = dict(
        getattr(cfg, "residue_mutation_contract", {}) or {}
    )
    if residue_contract:
        if target_chain not in residue_contract:
            raise ValueError(
                "residue_mutation_contract does not declare target chain "
                f"{target_chain!r}"
            )
        chain_contract = residue_contract[target_chain]
        missing_positions = [
            position for position in positions if position not in chain_contract
        ]
        if missing_positions:
            raise ValueError(
                "residue_mutation_contract does not declare write position(s) "
                f"for chain {target_chain!r}: {missing_positions}"
            )
        allowed_for_positions = {
            position: list(chain_contract[position]) for position in positions
        }
    else:
        allowed_for_positions = {
            position: list(AA) for position in positions
        }
    for position in positions:
        distribution = distributions_by_position.get(position)
        if distribution is None:
            continue
        executable = distribution.effective_probability_map
        if distribution.required_mutation is not None:
            executable = {distribution.required_mutation: 1.0}
        if not set(executable).issubset(set(allowed_for_positions[position])):
            raise ValueError(
                "compiled position distribution widens residue mutation "
                f"contract at {target_chain}:{position}"
            )
        if not executable:
            raise ValueError(
                "compiled position distribution has empty executable support "
                f"at {target_chain}:{position}"
            )
        allowed_for_positions[position] = sorted(executable)
    allowed = {target_chain: allowed_for_positions}
    soft = {
        target_chain: {
            position: {
                residue: float(
                    (
                        1.0
                        if distributions_by_position[position].required_mutation
                        == residue
                        else 0.0
                    )
                    if position in distributions_by_position
                    and distributions_by_position[position].required_mutation is not None
                    else (
                        distributions_by_position[position].effective_probability_map[
                            residue
                        ]
                        if position in distributions_by_position
                        else position_weights.get(position, {}).get(residue, 1.0)
                    )
                )
                for residue in allowed_for_positions[position]
            }
            for position in positions
        }
    }
    request = SequenceGenerationRequest.create(
        parent_sequences={
            str(chain): str(sequence)
            for chain, sequence in parent_sequences.items()
        },
        mapping_edge_id=edge_id,
        action_id=action_id,
        structural_node_id=mapped_structural_id,
        operator=str(operator),
        target_chain=str(target_chain),
        write_positions=positions,

        mutation_budget={"min": len(positions), "max": len(positions)},
        mutable_masks={
            str(chain): [bool(value) for value in mask]
            for chain, mask in masks.items()
        },
        fixed_residues={
            str(chain): {
                int(position): str(residue)
                for position, residue in residues.items()
            }
            for chain, residues in (fixed_residues or {}).items()
        },
        allowed_residues=allowed,
        soft_residue_weights=soft,
        seed=int(cfg.seed if cfg.seed is not None else 0),
        step=max(0, int(generation_step)),
        structure_condition_refs=structure_refs,
        state_condition_refs=state_refs,
    )
    selected_generator = str(getattr(cfg, "sequence_generator_id", "") or "")
    if not selected_generator:
        raise ValueError("sequence_generator_id must name an explicit registry arm")
    active_registry = registry or default_sequence_generator_registry()
    result = active_registry.generate(selected_generator, request)
    final_target = result.sequences[target_chain]
    distribution_receipt = distribution_policy.execution_receipt(
        node_distribution_rows,
        parent_sequence=parent_target,
        final_sequence=final_target,
    )
    unrealized_required = [
        item
        for item in distribution_receipt["realizations"]
        if item["required_mutation"] is not None
        and item["required_realized"] is not True
    ]
    if unrealized_required:
        raise ValueError(
            "sequence generator failed required compiled position distribution: "
            f"{[(item['chain_id'], item['position']) for item in unrealized_required]}"
        )
    return result.sequences, {
        "schema_version": "astevolve.sequence_generation_runtime.v1",
        "selected_generator_id": selected_generator,
        "validation_stage": "before_sequence_claim_and_scoring",
        "request": request.to_dict(),
        "result": result.to_dict(),
        "position_distribution_execution": distribution_receipt,
    }


def record_generation_attempt(
    history: MutableMapping[str, Any], move: Mapping[str, Any]
) -> None:


    raw_many = move.get("sequence_generations")
    rows = (
        list(raw_many)
        if isinstance(raw_many, list)
        else [move.get("sequence_generation")]
    )
    for raw in rows:
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise ValueError("sequence generation artifact must be a mapping")
        result = raw.get("result")
        request = raw.get("request")
        if not isinstance(result, Mapping) or not isinstance(request, Mapping):
            raise ValueError("sequence generation artifact is incomplete")
        generator_id = str(raw.get("selected_generator_id") or "")
        summary = history.setdefault(
            "sequence_generation",
            {
                "schema_version": "astevolve.sequence_generation_summary.v1",
                "selected_generator_id": generator_id,
                "attempted": 0,
                "validated": 0,
                "request_hashes": [],
                "result_hashes": [],
            },
        )
        if summary.get("selected_generator_id") != generator_id:
            raise ValueError("one inner run cannot silently switch sequence generators")
        summary["attempted"] = int(summary.get("attempted", 0)) + 1
        summary["validated"] = int(summary.get("validated", 0)) + 1
        summary["request_hashes"].append(str(request.get("request_hash") or ""))
        summary["result_hashes"].append(str(result.get("result_hash") or ""))


__all__ = ["realize_generated_substitutions", "record_generation_attempt"]
