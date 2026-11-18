from __future__ import annotations

from collections import Counter
import csv
from functools import lru_cache
from math import exp, log, log2
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from astevolve.runtime.paths import data_path
from cases.cab_lys3_hewl_cdr_recovery.sabdab_vhh_prior import (
    score_sabdab_vhh_prior,
)


CDR_PARTITIONS = {
    "cdr1": ((25, 32),),
    "cdr2": ((52, 57),),
    "cdr3": ((98, 108), (109, 122)),
}
CDR_REQUIREMENTS = {
    "cdr1": {"min_nonalanine": 4, "min_unique": 4, "min_entropy": 0.45},
    "cdr2": {"min_nonalanine": 3, "min_unique": 3, "min_entropy": 0.35},
    "cdr3": {"min_nonalanine": 12, "min_unique": 6, "min_entropy": 0.50},
}
MAX_IDENTICAL_RUN = 3
MAX_SINGLE_RESIDUE_FRACTION = 0.40
MIN_GLOBAL_UNIQUE = 8
MIN_GLOBAL_ENTROPY = 0.55
MIN_SABDAB_PLAUSIBILITY = 0.40
TARGET_NONALANINE_COUNT = 22
SABDAB_RULE_POSITIONS = {
    "cdr1": tuple(range(25, 32)),
    "cdr2": tuple(range(52, 57)),
    "cdr3": tuple(range(98, 108)) + tuple(range(109, 122)),
}
DEFAULT_SABDAB_INDEX_DIR = data_path(
    "cab_lys3_hewl_cdr_recovery",
    "sabdab_vhh_reference",
    "embeddings",
    "esm2_t6_8M_UR50D",
)


def _resampled_residues(sequence: str, length: int) -> tuple[str, ...]:
    observed = "".join(
        residue for residue in str(sequence).strip().upper()
        if residue in "ACDEFGHIKLMNPQRSTVWY"
    )
    if not observed or length < 1:
        return ()
    if length == 1:
        return (observed[len(observed) // 2],)
    return tuple(
        observed[round(index * (len(observed) - 1) / (length - 1))]
        for index in range(length)
    )


@lru_cache(maxsize=4)
def sabdab_position_residue_rules(index_dir: str = "") -> Dict[str, Dict[str, Any]]:
    configured = str(
        index_dir
        or os.environ.get("ASTEVOLVE_CAB_LYS3_SABDAB_INDEX")
        or ""
    ).strip()
    root = (
        Path(configured)
        if configured
        else DEFAULT_SABDAB_INDEX_DIR
    )
    metadata_path = root / "embedding_metadata.csv"
    counts = {
        region: [Counter() for _ in positions]
        for region, positions in SABDAB_RULE_POSITIONS.items()
    }
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for region, positions in SABDAB_RULE_POSITIONS.items():
                residues = _resampled_residues(row.get(region, ""), len(positions))
                for index, residue in enumerate(residues):
                    if residue != "C":
                        counts[region][index][residue] += 1
    rules: Dict[str, Dict[str, Any]] = {}
    for region, positions in SABDAB_RULE_POSITIONS.items():
        for index, position in enumerate(positions):
            favored = [
                residue
                for residue, _count in counts[region][index].most_common(6)
            ]
            if not favored:
                raise ValueError(
                    f"SAbDab position prior is empty for {region}:{position}"
                )
            rules[str(position)] = {
                "favored_residues": favored,
                "disfavored_residues": ["C"],
                "policy_weight": 2.5,
                "intent": (
                    "Position-level soft residue prior derived from external "
                    f"SAbDab VHH {region} sequences without case-native leakage."
                ),
            }
    return rules


def build_sabdab_bootstrap_proposals(
    *,
    root_sequences: Mapping[str, str],
    config: Mapping[str, Any],
    count: int,
    seed: int,
    masks: Optional[Mapping[str, Any]] = None,
    fixed_residues: Optional[Mapping[str, Mapping[Any, str]]] = None,
    score_config: Optional[Mapping[str, Any]] = None,
) -> list[tuple[Dict[str, str], Dict[str, Any]]]:
    if not bool(config.get("sabdab_prior_enabled", True)):
        raise ValueError(
            "SAbDab bootstrap cannot run when the external embedding prior is disabled"
        )
    root_a = str(root_sequences.get("A") or "")
    if len(root_a) != 133:
        raise ValueError("CAB SAbDab bootstrap requires a 133-residue A chain")
    if masks is None or score_config is None:
        raise ValueError("CAB bootstrap requires compiled masks and mutation scope")
    scope = score_config.get("mutation_scope_contract") or {}
    active = {int(i) for i in (scope.get("active_positions_by_chain") or {}).get("A", [])}
    mask = masks.get("A")
    if mask is None or len(mask) != len(root_a):
        raise ValueError("CAB bootstrap requires a full-length A-chain mask")
    masked = {i for i, enabled in enumerate(mask) if bool(enabled)}
    if active != masked:
        raise ValueError("CAB bootstrap mask and compiled mutation scope disagree")
    fixed = {int(i): str(aa) for i, aa in (fixed_residues or {}).get("A", {}).items()}
    if any(root_a[i] != aa for i, aa in fixed.items()):
        raise ValueError("CAB bootstrap parent violates fixed residues")
    editable = active.difference(fixed)
    owner_by_position: Dict[int, str] = {}
    for node_id, node in (scope.get("active_positions_by_node") or {}).items():
        if str(node.get("chain_id")) != "A":
            continue
        for value in node.get("positions", []):
            position = int(value)
            if position in owner_by_position and owner_by_position[position] != str(node_id):
                raise ValueError("CAB bootstrap mutation ownership is ambiguous")
            owner_by_position[position] = str(node_id)
    if not editable or not editable.issubset(owner_by_position):
        raise ValueError("CAB bootstrap has no completely attributed editable scope")
    alphabet = {
        int(i): set(values)
        for i, values in (score_config.get("residue_mutation_contract") or {}).get("A", {}).items()
    }
    max_changes = int(scope.get("max_total_mutations", len(editable)))
    index_dir = str(config.get("sabdab_prior_index_dir") or "").strip()
    root = Path(index_dir) if index_dir else DEFAULT_SABDAB_INDEX_DIR
    with (root / "embedding_metadata.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 3:
        raise ValueError("SAbDab bootstrap reference table is empty")
    rules = sabdab_position_residue_rules(str(root))
    requested = max(1, int(count))
    start = int(seed) % len(rows)
    proposals: list[tuple[Dict[str, str], Dict[str, Any]]] = []
    seen: set[str] = set()
    for attempt in range(max(64, requested * 24)):
        selected_rows = {
            "cdr1": rows[(start + attempt * 5) % len(rows)],
            "cdr2": rows[(start + 3 + attempt * 11) % len(rows)],
            "cdr3": rows[(start + 7 + attempt * 17) % len(rows)],
        }
        candidate = list(root_a)
        source_refs: Dict[str, str] = {}
        for region, positions in SABDAB_RULE_POSITIONS.items():
            row = selected_rows[region]
            source_refs[region] = str(row.get("reference_id") or "")
            residues = _resampled_residues(row.get(region, ""), len(positions))
            if len(residues) != len(positions):
                break
            for position, residue in zip(positions, residues):
                if position not in editable:
                    continue
                if residue == "C":
                    residue = str(rules[str(position)]["favored_residues"][0])
                allowed = alphabet.get(position)
                if allowed is not None and residue not in allowed:
                    alternatives = [aa for aa in rules[str(position)]["favored_residues"] if aa in allowed]
                    residue = alternatives[0] if alternatives else root_a[position]
                candidate[position] = residue
        else:
            sequence = "".join(candidate)
            changed = {i for i, (before, after) in enumerate(zip(root_a, sequence)) if before != after}
            if not changed or len(changed) > max_changes:
                continue
            if sequence in seen:
                continue
            seen.add(sequence)
            seqs = dict(root_sequences)
            seqs["A"] = sequence
            gate = evaluate_pre_model_gate(seqs=seqs, config=config)
            if not bool(gate.get("pass")):
                continue
            changes = [
                {
                    "chain_id": "A",
                    "position": position,
                    "from": root_a[position],
                    "to": sequence[position],
                    "node": next(
                        region
                        for region, region_positions in SABDAB_RULE_POSITIONS.items()
                        if position in region_positions
                    ),
                }
                for position in sorted(
                    position
                    for region_positions in SABDAB_RULE_POSITIONS.values()
                    for position in region_positions
                )
                if root_a[position] != sequence[position]
            ]
            from astevolve.search.mutation_move import finalize_mutation_move

            move = finalize_mutation_move(
                root_sequences,
                seqs,
                {
                    "op": "case_bootstrap_exact",
                    "node": "case_bootstrap_exact",
                    "changes": changes,
                    "proposal_log_prior": float(
                        gate["sabdab"].get("plausibility", 0.0)
                    ),
                    "mutation_plan": {"tier": "case_bootstrap"},
                    "case_sequence_bootstrap": True,
                    "mapping_realization_summary": {
                        "position_owners": [
                            {
                                "chain_id": "A",
                                "position": int(change["position"]),
                                "owner_node_id": owner_by_position[int(change["position"])],
                            }
                            for change in changes
                        ]
                    },
                    "bootstrap_source": {
                        "kind": "external_sabdab_vhh_region_recombination",
                        "reference_ids": source_refs,
                        "native_sequence_used": False,
                        "hard_gate": gate,
                    },
                },
            )
            proposals.append(
                (
                    seqs,
                    move,
                )
            )
            if len(proposals) >= requested:
                return proposals
    raise ValueError(
        f"SAbDab bootstrap produced {len(proposals)} passing proposals, "
        f"required {requested}"
    )


def _partition_sequence(sequence: str, spans: tuple[tuple[int, int], ...]) -> str:
    return "".join(sequence[start:end] for start, end in spans)


def _longest_identical_run(sequence: str) -> int:
    best = current = 0
    previous = ""
    for residue in sequence:
        current = current + 1 if residue == previous else 1
        previous = residue
        best = max(best, current)
    return best


def _normalized_entropy(sequence: str) -> float:
    if len(sequence) < 2:
        return 0.0
    counts = Counter(sequence)
    entropy = -sum(
        (count / len(sequence)) * log2(count / len(sequence))
        for count in counts.values()
    )
    maximum = log2(min(20, len(sequence)))
    return float(entropy / maximum) if maximum > 0.0 else 0.0


def _dominant_fraction(sequence: str) -> float:
    counts = Counter(sequence)
    return float(max(counts.values()) / len(sequence)) if sequence else 1.0


def sequence_quality_summary(sequence: str) -> Dict[str, Any]:
    binder = str(sequence).strip().upper()
    partitions: Dict[str, Dict[str, Any]] = {}
    hard_failures: list[str] = []
    all_cdr = ""
    for name, spans in CDR_PARTITIONS.items():
        observed = _partition_sequence(binder, spans)
        all_cdr += observed
        requirement = CDR_REQUIREMENTS[name]
        nonalanine_count = sum(residue != "A" for residue in observed)
        unique_count = len(set(observed))
        entropy = _normalized_entropy(observed)
        longest_run = _longest_identical_run(observed)
        dominant_fraction = _dominant_fraction(observed)
        partition_pass = (
            nonalanine_count >= int(requirement["min_nonalanine"])
            and unique_count >= int(requirement["min_unique"])
            and entropy >= float(requirement["min_entropy"])
            and longest_run <= MAX_IDENTICAL_RUN
        )
        if longest_run > MAX_IDENTICAL_RUN:
            hard_failures.append(
                f"cdr_identical_run_exceeded:{name}:{longest_run}>{MAX_IDENTICAL_RUN}"
            )
        if unique_count < int(requirement["min_unique"]):
            hard_failures.append(
                f"cdr_low_complexity:{name}:unique={unique_count}<{requirement['min_unique']}"
            )
        if entropy < float(requirement["min_entropy"]):
            hard_failures.append(
                f"cdr_entropy_below_floor:{name}:{entropy:.4f}<{requirement['min_entropy']:.4f}"
            )
        if nonalanine_count < int(requirement["min_nonalanine"]):
            hard_failures.append(
                f"cdr_all_a_escape_underfilled:{name}:{nonalanine_count}<{requirement['min_nonalanine']}"
            )
        partitions[name] = {
            "sequence": observed,
            "length": len(observed),
            "nonalanine_count": nonalanine_count,
            "unique_residue_count": unique_count,
            "normalized_entropy": entropy,
            "longest_identical_run": longest_run,
            "dominant_residue_fraction": dominant_fraction,
            "requirements": dict(requirement),
            "pass": partition_pass,
        }

    global_unique = len(set(all_cdr))
    global_entropy = _normalized_entropy(all_cdr)
    dominant_fraction = _dominant_fraction(all_cdr)
    nonalanine_count = sum(residue != "A" for residue in all_cdr)
    if global_unique < MIN_GLOBAL_UNIQUE:
        hard_failures.append(
            f"cdr_global_low_complexity:unique={global_unique}<{MIN_GLOBAL_UNIQUE}"
        )
    if global_entropy < MIN_GLOBAL_ENTROPY:
        hard_failures.append(
            f"cdr_global_entropy_below_floor:{global_entropy:.4f}<{MIN_GLOBAL_ENTROPY:.4f}"
        )
    if dominant_fraction > MAX_SINGLE_RESIDUE_FRACTION:
        hard_failures.append(
            "cdr_single_residue_fraction_exceeded:"
            f"{dominant_fraction:.4f}>{MAX_SINGLE_RESIDUE_FRACTION:.4f}"
        )

    nonalanine_score = min(1.0, nonalanine_count / TARGET_NONALANINE_COUNT)
    diversity_score = min(
        1.0,
        max(0.0, (global_unique - 1) / (MIN_GLOBAL_UNIQUE - 1)),
    )
    partition_scores = []
    for name, details in partitions.items():
        requirement = CDR_REQUIREMENTS[name]
        partition_scores.append(
            min(
                1.0,
                details["nonalanine_count"] / requirement["min_nonalanine"],
                details["unique_residue_count"] / requirement["min_unique"],
                details["normalized_entropy"] / requirement["min_entropy"],
            )
        )
    partition_coverage = min(partition_scores, default=0.0)
    escape_components = (
        max(0.0, nonalanine_score),
        max(0.0, diversity_score),
        max(0.0, partition_coverage),
    )
    all_a_escape_score = (
        exp(sum(log(value) for value in escape_components) / len(escape_components))
        if all(value > 0.0 for value in escape_components)
        else 0.0
    )
    return {
        "schema_version": "cab_lys3.cdr_sequence_quality.v2",
        "pass": not hard_failures,
        "hard_failures": sorted(set(hard_failures)),
        "partitions": partitions,
        "all_cdr_sequence": all_cdr,
        "global_unique_residue_count": global_unique,
        "global_normalized_entropy": global_entropy,
        "global_dominant_residue_fraction": dominant_fraction,
        "global_nonalanine_count": nonalanine_count,
        "max_identical_run": max(
            (row["longest_identical_run"] for row in partitions.values()),
            default=0,
        ),
        "all_a_escape": {
            "score": float(all_a_escape_score),
            "nonalanine_score": float(nonalanine_score),
            "diversity_score": float(diversity_score),
            "partition_coverage": float(partition_coverage),
            "nonalanine_count": nonalanine_count,
            "target_nonalanine_count": TARGET_NONALANINE_COUNT,
            "unique_residue_count": global_unique,
            "target_unique_residue_count": MIN_GLOBAL_UNIQUE,
            "aggregation": "geometric_mean",
        },
        "thresholds": {
            "max_identical_run": MAX_IDENTICAL_RUN,
            "max_single_residue_fraction": MAX_SINGLE_RESIDUE_FRACTION,
            "min_global_unique": MIN_GLOBAL_UNIQUE,
            "min_global_entropy": MIN_GLOBAL_ENTROPY,
        },
    }


def evaluate_pre_model_gate(
    *,
    seqs: Mapping[str, str],
    config: Mapping[str, Any],
    **_: Any,
) -> Dict[str, Any]:
    sequence = str(seqs.get("A") or "")
    quality = sequence_quality_summary(sequence)
    reasons = list(quality["hard_failures"])
    sabdab_enabled = bool(config.get("sabdab_prior_enabled", True))
    sabdab_summary: Dict[str, Any] = {
        "available": False,
        "enabled": sabdab_enabled,
        "plausibility": 0.0,
        "skipped": bool(reasons) or not sabdab_enabled,
    }
    quality_pass = not reasons
    if not sabdab_enabled:
        sabdab_summary["skip_reason"] = "external_embedding_disabled"
    elif not reasons:
        try:
            sabdab_summary = score_sabdab_vhh_prior(sequence, config)
            sabdab_summary["enabled"] = True
        except Exception as exc:
            sabdab_summary = {
                "available": False,
                "enabled": True,
                "plausibility": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        plausibility = float(sabdab_summary.get("plausibility", 0.0))
        if not bool(sabdab_summary.get("available")):
            reasons.append("sabdab_plausibility_unavailable")
        elif plausibility < MIN_SABDAB_PLAUSIBILITY:
            reasons.append(
                "sabdab_plausibility_below_floor:"
                f"{plausibility:.4f}<{MIN_SABDAB_PLAUSIBILITY:.4f}"
            )
    escape_progress = float(quality["all_a_escape"]["score"])
    if quality_pass:
        if not sabdab_enabled:
            plausibility_progress = 1.0
        else:
            plausibility_progress = min(
                1.0,
                max(
                    0.0,
                    float(sabdab_summary.get("plausibility", 0.0))
                    / MIN_SABDAB_PLAUSIBILITY,
                ),
            )
        search_progress = 0.5 + 0.5 * plausibility_progress
    else:
        search_progress = 0.5 * escape_progress
    return {
        "schema_version": "cab_lys3.pre_model_sequence_gate.v1",
        "pass": not reasons,
        "reasons": sorted(set(reasons)),
        "search_expandable": True,
        "search_progress": float(search_progress),
        "quality": quality,
        "sabdab": sabdab_summary,
        "sabdab_prior_enabled": sabdab_enabled,
        "sabdab_min_plausibility": MIN_SABDAB_PLAUSIBILITY,
        "execution_stage": "before_progen_protenix",
    }


__all__ = [
    "CDR_PARTITIONS",
    "CDR_REQUIREMENTS",
    "MAX_IDENTICAL_RUN",
    "MAX_SINGLE_RESIDUE_FRACTION",
    "MIN_GLOBAL_UNIQUE",
    "MIN_GLOBAL_ENTROPY",
    "MIN_SABDAB_PLAUSIBILITY",
    "TARGET_NONALANINE_COUNT",
    "SABDAB_RULE_POSITIONS",
    "evaluate_pre_model_gate",
    "build_sabdab_bootstrap_proposals",
    "sabdab_position_residue_rules",
    "sequence_quality_summary",
]
