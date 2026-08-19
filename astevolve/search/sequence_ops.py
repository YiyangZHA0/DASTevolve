

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from astevolve.core.amino_acids import AA
from astevolve.search.config import SAConfig
from astevolve.search.generation_runtime import realize_generated_substitutions
from astevolve.search.mutation_move import finalize_mutation_move
from astevolve.search.operator_registry import (
    operator_manifest,
    require_operator,
    validate_operator_weights,
)
from astevolve.search.sequence_generator import SequenceGeneratorRegistry


ChainSequences = Dict[str, str]
ChainMasks = Mapping[str, np.ndarray]
FixedResidues = Mapping[str, Mapping[int, str]]


def _to_bool_mask(mask: Any, L: int) -> np.ndarray:


    if mask is None:
        return np.ones(L, dtype=bool)
    if isinstance(mask, list):
        mask = np.array(mask, dtype=bool)
    if isinstance(mask, np.ndarray):
        if mask.dtype != bool:
            mask = mask.astype(bool)
        if mask.size != L:
            raise ValueError(f"Mask length mismatch: {mask.size} vs {L}")
        return mask
    raise ValueError("mask must be list, np.ndarray, or None")


def random_chain_sequence(length: int, rng: np.random.Generator) -> str:


    indices = rng.integers(0, len(AA), size=length)
    return "".join(AA[index] for index in indices)


def init_seqs(
    chain_lengths: Mapping[str, int],
    rng: np.random.Generator,
    template_seqs: Optional[Mapping[str, str]] = None,
    fixed_residues: Optional[FixedResidues] = None,
) -> ChainSequences:


    sequences: ChainSequences = {}
    for chain_id, length in chain_lengths.items():
        base = list(random_chain_sequence(length, rng))

        if template_seqs and chain_id in template_seqs:
            template = template_seqs[chain_id]
            if len(template) != length:
                raise ValueError(
                    f"Template length mismatch for chain {chain_id}: "
                    f"{len(template)} vs {length}"
                )
            base = list(template)

        if fixed_residues and chain_id in fixed_residues:
            for position, amino_acid in fixed_residues[chain_id].items():
                if 0 <= position < length:
                    base[position] = amino_acid

        sequences[chain_id] = "".join(base)
    return sequences


def _choose_op(
    rng: np.random.Generator,
    op_weights: Mapping[str, float],
    *,
    mode: str = "node",
) -> str:


    normalized = validate_operator_weights(
        op_weights,
        mode=mode,
        context="executor.op_weights",
        drop_zero=True,
    )
    operators = list(normalized.keys())
    weights = np.array([normalized[key] for key in operators], dtype=float)
    weights = weights / weights.sum()
    return operators[int(rng.choice(len(operators), p=weights))]


def _mutate_swap(
    seq_list: list[str],
    designable: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    if designable.size < 2:
        return []
    first, second = rng.choice(designable, size=2, replace=False)
    seq_list[first], seq_list[second] = seq_list[second], seq_list[first]
    return [int(first), int(second)]


def _mutate_region_shuffle(
    seq_list: list[str],
    designable: np.ndarray,
    rng: np.random.Generator,
    k: int,
) -> list[int]:
    if designable.size < 2:
        return []
    k = min(max(2, k), designable.size)
    positions = list(map(int, rng.choice(designable, size=k, replace=False)))
    old_residues = [seq_list[index] for index in positions]
    shuffled = old_residues[:]
    rng.shuffle(shuffled)
    if shuffled == old_residues and len(shuffled) > 1:
        shuffled = shuffled[1:] + shuffled[:1]
    for position, amino_acid in zip(positions, shuffled):
        seq_list[position] = amino_acid
    return positions


def mutate_seqs(
    seqs: Mapping[str, str],
    compiled: Mapping[str, Any],
    rng: np.random.Generator,
    cfg: SAConfig,
    masks: ChainMasks,
    fixed_residues: Optional[FixedResidues] = None,
    generation_step: int = 0,
    sequence_generator_registry: Optional[SequenceGeneratorRegistry] = None,
) -> Tuple[ChainSequences, Dict[str, Any]]:


    new = {key: list(value) for key, value in seqs.items()}
    move: Dict[str, Any] = {"op": None, "positions": {}, "segments": []}

    def finish() -> Tuple[ChainSequences, Dict[str, Any]]:
        result = {key: "".join(value) for key, value in new.items()}
        return result, finalize_mutation_move(seqs, result, move)

    def realize(chain_id: str, positions: list[int], node_id: str) -> None:
        if not positions:
            return
        parent = {key: "".join(value) for key, value in new.items()}
        generated, artifact = realize_generated_substitutions(
            parent_sequences=parent,
            target_chain=chain_id,
            write_positions=positions,
            operator=operation,
            structural_node_id=node_id,
            cfg=cfg,
            masks=masks,
            fixed_residues=fixed_residues,
            position_weights={},
            mapping_action=None,
            generation_step=generation_step,
            registry=sequence_generator_registry,
        )
        new.update({key: list(value) for key, value in generated.items()})
        move.setdefault("sequence_generations", []).append(artifact)
        if len(move["sequence_generations"]) == 1:
            move["sequence_generation"] = artifact

    operation = _choose_op(rng, cfg.mutation_ops, mode="legacy")
    operator_spec = require_operator(operation, mode="legacy")
    handler = operator_spec.legacy_handler
    move["op"] = operation
    move["operator_spec"] = operator_manifest(operation, mode="legacy")
    move["operator_selection"] = {
        "selected": operation,
        "source": "sa_config.mutation_ops",
        "effective_weights": dict(cfg.mutation_ops),
    }

    if handler in {"segment_resample", "segment_mutagenesis"} or (
        handler == "block_substitution" and rng.random() < cfg.resample_segment_prob
    ):
        segment = rng.choice(compiled["segments"])
        chain_id = segment.chain_id
        mask = masks[chain_id]
        designable = [int(index) for index in segment.indices() if bool(mask[index])]
        if handler == "segment_mutagenesis":
            count = min(
                len(designable),
                max(4, int(round(cfg.mutation_rate * len(designable) * 2))),
            )
            selected = (
                set(
                    map(
                        int,
                        rng.choice(
                            np.asarray(designable), size=count, replace=False
                        ),
                    )
                )
                if designable
                else set()
            )
        else:
            selected = set(designable)
        positions = [index for index in designable if index in selected]
        realize(chain_id, positions, str(segment.name))
        move["segments"] = [(chain_id, segment.name, segment.spans)]
        move["positions"][chain_id] = positions
        return finish()

    for chain_id, sequence in new.items():
        designable = np.where(masks[chain_id])[0]
        if designable.size == 0:
            continue

        count = max(1, int(round(cfg.mutation_rate * designable.size)))
        if handler == "point_substitution":
            positions = list(
                map(
                    int,
                    rng.choice(
                        designable,
                        size=min(count, designable.size),
                        replace=False,
                    ),
                )
            )
        elif handler == "site_resample":
            position_count = min(
                designable.size, max(count, min(4, designable.size))
            )
            positions = list(
                map(
                    int,
                    rng.choice(designable, size=position_count, replace=False),
                )
            )
        elif handler == "block_substitution":
            block_length = int(rng.integers(2, 6))
            start = int(rng.choice(designable))
            designable_set = set(designable)
            positions = [
                index
                for index in range(start, min(len(sequence), start + block_length))
                if index in designable_set
            ]
        elif handler == "region_shuffle":
            positions = _mutate_region_shuffle(
                sequence,
                designable,
                rng,
                max(count, min(8, designable.size)),
            )
        elif handler == "swap":
            positions = _mutate_swap(sequence, designable, rng)
        else:
            raise RuntimeError(
                f"Registry handler {handler!r} has no legacy execution branch"
            )

        if handler in {
            "point_substitution",
            "site_resample",
            "block_substitution",
        }:
            realize(chain_id, positions, f"legacy::{chain_id}")

        move["positions"][chain_id] = positions

    return finish()


__all__ = [
    "ChainMasks",
    "ChainSequences",
    "FixedResidues",
    "init_seqs",
    "mutate_seqs",
    "random_chain_sequence",
]
