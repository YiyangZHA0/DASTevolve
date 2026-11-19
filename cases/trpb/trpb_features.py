"""Feature contracts for the TrpB Dual-AST v1 development arms."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {residue: index for index, residue in enumerate(AMINO_ACIDS)}
ORDERED_PAIR_COUNT = len(AMINO_ACIDS) ** 2
LEARNED_ARMS = ("onehot_ridge", "flat_no_ast", "shuffled_mapping", "full_dast")


def ordered_pair_index(left_residue: str, right_residue: str) -> int:
    """Encode a sequence-ordered amino-acid pair without symmetrizing it."""

    return AA_INDEX[left_residue] * len(AMINO_ACIDS) + AA_INDEX[right_residue]


def feature_dimension(arm: str, structural_count: int, functional_count: int) -> int:
    position_residue = 20 * 20
    if arm == "onehot_ridge":
        return position_residue + 1
    flat = position_residue + 20 + ORDERED_PAIR_COUNT + 5
    if arm == "flat_no_ast":
        return flat
    if arm not in {"full_dast", "shuffled_mapping"}:
        raise ValueError(f"unknown TrpB v1 arm: {arm}")
    return (
        flat
        + structural_count * 20
        + functional_count * 20
        + structural_count * ORDERED_PAIR_COUNT
        + functional_count * ORDERED_PAIR_COUNT
        + structural_count * 2
        + functional_count * 2
    )


def feature_contract(
    arm: str, structural_ids: Sequence[str], functional_ids: Sequence[str]
) -> dict[str, Any]:
    """Return auditable offsets for every v1 feature family."""

    structural_count = len(structural_ids)
    functional_count = len(functional_ids)
    offsets: dict[str, dict[str, Any]] = {}
    cursor = 0

    def add(name: str, width: int, labels: Sequence[str] | None = None) -> None:
        nonlocal cursor
        offsets[name] = {
            "start": cursor,
            "stop": cursor + width,
            "width": width,
        }
        if labels is not None:
            offsets[name]["labels"] = list(labels)
        cursor += width

    add("position_residue", 20 * 20)
    if arm == "onehot_ridge":
        add("mutation_count", 1)
    else:
        add("global_residue_bag", 20)
        add("global_ordered_residue_pair", ORDERED_PAIR_COUNT)
        add("global_prior_statistics", 5)
        if arm != "flat_no_ast":
            add("structural_node_residue", structural_count * 20, structural_ids)
            add("functional_node_residue", functional_count * 20, functional_ids)
            add(
                "structural_node_ordered_pair",
                structural_count * ORDERED_PAIR_COUNT,
                structural_ids,
            )
            add(
                "functional_node_ordered_pair",
                functional_count * ORDERED_PAIR_COUNT,
                functional_ids,
            )
            add("structural_node_prior_statistics", structural_count * 2, structural_ids)
            add("functional_node_prior_statistics", functional_count * 2, functional_ids)
    expected = feature_dimension(arm, structural_count, functional_count)
    if cursor != expected:
        raise AssertionError(f"feature offset mismatch: {cursor} != {expected}")
    return {"arm_id": arm, "dimension": expected, "families": offsets}


def build_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    allowed_positions: Sequence[int],
    ast: Mapping[str, Any],
    prior: Mapping[tuple[int, str], float],
) -> np.ndarray:
    """Build equal-resource flat or mapping-aware v1 design matrices."""

    position_index = {position: index for index, position in enumerate(allowed_positions)}
    structural_ids = tuple(ast["structural_ids"])
    functional_ids = tuple(ast["functional_ids"])
    structural_index = {node_id: index for index, node_id in enumerate(structural_ids)}
    functional_index = {node_id: index for index, node_id in enumerate(functional_ids)}
    structural_to_functional = ast["structural_to_functional"]
    if arm == "shuffled_mapping":
        position_to_structural = ast["shuffled_position_to_structural"]
    else:
        position_to_structural = ast["position_to_structural"]

    contract = feature_contract(arm, structural_ids, functional_ids)
    families = contract["families"]
    features = np.zeros((len(rows), contract["dimension"]), dtype=np.float32)
    position_residue_offset = families["position_residue"]["start"]

    if arm == "onehot_ridge":
        count_offset = families["mutation_count"]["start"]
        for row_index, row in enumerate(rows):
            mutations = row["mutations"]
            for mutation in mutations:
                position = int(mutation["position_1based"])
                residue = str(mutation["alt_aa"])
                features[
                    row_index,
                    position_residue_offset + position_index[position] * 20 + AA_INDEX[residue],
                ] += 1.0
            features[row_index, count_offset] = float(len(mutations))
        return features

    global_aa_offset = families["global_residue_bag"]["start"]
    global_pair_offset = families["global_ordered_residue_pair"]["start"]
    global_stats_offset = families["global_prior_statistics"]["start"]
    mapping_aware = arm in {"full_dast", "shuffled_mapping"}
    if mapping_aware:
        structural_aa_offset = families["structural_node_residue"]["start"]
        functional_aa_offset = families["functional_node_residue"]["start"]
        structural_pair_offset = families["structural_node_ordered_pair"]["start"]
        functional_pair_offset = families["functional_node_ordered_pair"]["start"]
        structural_stats_offset = families["structural_node_prior_statistics"]["start"]
        functional_stats_offset = families["functional_node_prior_statistics"]["start"]

    for row_index, row in enumerate(rows):
        mutations = sorted(
            row["mutations"], key=lambda mutation: int(mutation["position_1based"])
        )
        encoded: list[tuple[int, str, int, int, float]] = []
        deltas: list[float] = []
        for mutation in mutations:
            position = int(mutation["position_1based"])
            residue = str(mutation["alt_aa"])
            aa_index = AA_INDEX[residue]
            delta = float(prior[(position, residue)])
            structural_id = position_to_structural[position]
            structural_node = structural_index[structural_id]
            functional_node = functional_index[structural_to_functional[structural_id]]
            encoded.append((position, residue, structural_node, functional_node, delta))
            deltas.append(delta)
            features[
                row_index,
                position_residue_offset + position_index[position] * 20 + aa_index,
            ] += 1.0
            features[row_index, global_aa_offset + aa_index] += 1.0
            if mapping_aware:
                features[row_index, structural_aa_offset + structural_node * 20 + aa_index] += 1.0
                features[row_index, functional_aa_offset + functional_node * 20 + aa_index] += 1.0
                features[row_index, structural_stats_offset + structural_node * 2] += delta
                features[row_index, structural_stats_offset + structural_node * 2 + 1] += 1.0
                features[row_index, functional_stats_offset + functional_node * 2] += delta
                features[row_index, functional_stats_offset + functional_node * 2 + 1] += 1.0

        for left_index in range(len(encoded)):
            _, left_residue, left_structural, left_functional, _ = encoded[left_index]
            for right_index in range(left_index + 1, len(encoded)):
                _, right_residue, right_structural, right_functional, _ = encoded[right_index]
                pair_index = ordered_pair_index(left_residue, right_residue)
                features[row_index, global_pair_offset + pair_index] += 1.0
                if mapping_aware and left_structural == right_structural:
                    features[
                        row_index,
                        structural_pair_offset
                        + left_structural * ORDERED_PAIR_COUNT
                        + pair_index,
                    ] += 1.0
                if mapping_aware and left_functional == right_functional:
                    features[
                        row_index,
                        functional_pair_offset
                        + left_functional * ORDERED_PAIR_COUNT
                        + pair_index,
                    ] += 1.0

        features[row_index, global_stats_offset] = float(len(encoded))
        if deltas:
            features[row_index, global_stats_offset + 1] = float(sum(deltas))
            features[row_index, global_stats_offset + 2] = float(sum(deltas) / len(deltas))
            features[row_index, global_stats_offset + 3] = float(min(deltas))
            features[row_index, global_stats_offset + 4] = float(max(deltas))
    return features
