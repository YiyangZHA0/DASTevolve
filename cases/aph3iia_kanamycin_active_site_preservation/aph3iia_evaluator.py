"""Case-owned evaluator for APH(3')-IIa kanamycin-site preservation.

``evaluate_candidate`` is deliberately a deterministic CPU preflight boundary:
it validates sequence, fixed-residue, controller-owned amino-acid, and compiled
AST-scope contracts without pretending that structural evidence exists.

``Aph3iiaKanamycinPlugin`` is the production evaluator.  It requires both the
candidate apo state and a candidate-specific KAN/Mg complex, and fails closed
when coordinates, state provenance, or comparable ligand evidence are absent.
Its outputs are computational compatibility signals, not MIC or kinetic claims.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from astevolve.runtime.paths import data_path

from astevolve.evaluation.contracts import EvaluatorContext, ScoreTerm
from astevolve.evaluation.plugins.registry import (
    EvaluatorPluginSpec,
    PluginConfigField,
    register_plugin,
)


PLUGIN_NAME = "aph3iia_kanamycin"
CASE_ROOT = Path(__file__).resolve().parent
REFERENCE_CHAIN_A = data_path("aph3iia_kanamycin_active_site_preservation", "structures", "ast_ready", "1ND4_chainA_KAN_MG.pdb")
REFERENCE_CHAIN_B = data_path("aph3iia_kanamycin_active_site_preservation", "structures", "ast_ready", "1ND4_chainB_KAN_MG.pdb")
WT_SEQUENCE = (
    "MIEQDGLHAGSPAAWVERLFGYDWAQQTIGCSDAAVFRLSAQGRPVLFVKTDLSGALNEL"
    "QDEAARLSWLATTGVPCAAVLDVVTEAGRDWLLLGEVPGQDLLSSHLAPAEKVSIMADAM"
    "RRLHTLDPATCPFDHQAKHRIERARTRMEAGLVDQDDLDEEHQGLAPAELFARLKARMPD"
    "GEDLVVTHGDACLPNIMVENGRFSGFIDCGRLGVADRYQDIALATRDIAEELGGEWADRF"
    "LVLYGIAAPDSQRIAFYRLLDEFF"
)
CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
SEED_POSITIONS = frozenset({13, 71, 121, 177, 237})  # zero-based
OPEN_POSITIONS = SEED_POSITIONS  # backward-compatible public alias
VERIFIED_CORE = (157, 190, 195, 208)  # one-based
KAN_CONTACT_SHELL = (157, 158, 159, 160, 190, 195, 211, 226, 227, 230, 261, 262, 264)
BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})
LEVEL_ALLOWED = {
    "level1": {13: {"A", "T"}, 71: {"T", "S"}, 121: {"R", "K"}, 177: {"M", "F"}, 237: {"D", "E"}},
    "level2": {13: {"A", "T", "S"}, 71: {"T", "S", "A"}, 121: {"R", "K", "Q"}, 177: {"M", "F", "L"}, 237: {"D", "E", "N"}},
}
HARD_GATE_MIN_SCORES = {
    "aph_sequence_integrity": 1.0,
    "aph_mutation_scope_integrity": 1.0,
    "aph_global_fold_preservation": 0.5,
    "aph_verified_core_preservation": 0.5,
    "aph_kanamycin_contact_preservation": 0.9,
    "aph_kanamycin_pose_preservation": 0.5,
    "aph_mg_geometry_preservation": 0.5,
    "aph_clash_free": 1.0,
}
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
CLAIM_BOUNDARY = (
    "Computational sequence/structure compatibility only; this evaluator does "
    "not establish kanamycin MIC, catalytic rate, or preserved resistance."
)
EXCLUDED_CLAIMS = (
    "phosphate-donor geometry is not scored because no bound donor is present",
    "the crystallographic A2 interface is not a functional gate",
    "chain B is a report-only crystallographic-repeat robustness anchor",
)


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return float(max(0.0, min(1.0, number))) if math.isfinite(number) else 0.0


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _half_at_limit(value: Any, limit: Any) -> float:
    """Map a non-negative error to [0, 1], with score 0.5 at ``limit``."""

    number = _finite_float(value)
    scale = _finite_float(limit)
    if number is None or scale is None or number < 0.0 or scale <= 0.0:
        return 0.0
    ratio = number / scale
    return _clamp01(math.exp(-math.log(2.0) * ratio * ratio))


def _config_float(context: EvaluatorContext, key: str, default: float) -> float:
    value = _finite_float(context.score_config.get(key))
    return float(default if value is None else value)


def _config_int(context: EvaluatorContext, key: str, default: int) -> int:
    try:
        return int(context.score_config.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _config_bool(context: EvaluatorContext, key: str, default: bool) -> bool:
    value = context.score_config.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _weight(context: EvaluatorContext, term_name: str, default: float) -> float:
    values = context.score_config.get("evaluator_weights", {})
    if not isinstance(values, Mapping):
        return float(default)
    value = _finite_float(values.get(term_name, values.get(f"eval_{term_name}")))
    return float(default if value is None else value)


def _hard_gate_threshold(context: EvaluatorContext, term_name: str) -> float:
    """Resolve a locked absolute or WT-relative threshold without silent relaxation."""

    absolute = float(HARD_GATE_MIN_SCORES[term_name])
    calibration = context.score_config.get("aph_hard_gate_calibration", {})
    if not isinstance(calibration, Mapping) or not calibration:
        return absolute
    mode = str(calibration.get("mode") or "absolute").strip().lower()
    if mode == "absolute":
        return absolute
    if mode != "relative_wt_noninferiority":
        raise ValueError(f"unsupported APH hard-gate calibration mode: {mode}")
    seeds = calibration.get("seeds", [])
    if (
        calibration.get("locked") is not True
        or not isinstance(calibration.get("protocol_sha256"), str)
        or len(calibration["protocol_sha256"]) != 64
        or not isinstance(seeds, Sequence)
        or isinstance(seeds, (str, bytes))
        or len(set(str(seed) for seed in seeds)) < 3
    ):
        raise ValueError(
            "WT-relative APH hard gates require a locked protocol hash and at least three seeds"
        )
    wt_scores = calibration.get("wt_scores", {})
    tolerances = calibration.get("noninferiority_tolerances", {})
    if not isinstance(wt_scores, Mapping) or term_name not in wt_scores:
        raise ValueError(f"WT calibration score missing for {term_name}")
    tolerance = 0.0
    if isinstance(tolerances, Mapping):
        tolerance = float(tolerances.get(term_name, 0.0) or 0.0)
    if tolerance < 0.0:
        raise ValueError("noninferiority tolerance must be non-negative")
    return _clamp01(float(wt_scores[term_name]) - tolerance)


def _candidate_sequence(out: Mapping[str, Any]) -> str:
    for key in ("seqs", "best_seqs"):
        seqs = out.get(key)
        if isinstance(seqs, Mapping) and seqs.get("A"):
            return "".join(str(seqs["A"]).split()).upper()
    return ""


def _template_sequence(template_seqs: Optional[Mapping[str, str]]) -> str:
    templates = template_seqs if isinstance(template_seqs, Mapping) else {}
    return "".join(str(templates.get("A") or WT_SEQUENCE).split()).upper()


def _changed_positions(before: str, after: str) -> list[int]:
    if len(before) != len(after):
        return []
    return [index for index, (left, right) in enumerate(zip(before, after)) if left != right]


def _raw_policy(design_state: Mapping[str, Any]) -> Mapping[str, Any]:
    value = design_state.get("case_owned_residue_policy", {})
    return value if isinstance(value, Mapping) else {}


def _normalise_contract(value: Any) -> dict[int, set[str]]:
    if not isinstance(value, Mapping):
        return {}
    chain = value.get("A", value)
    if not isinstance(chain, Mapping):
        return {}
    result: dict[int, set[str]] = {}
    for raw_position, raw_allowed in chain.items():
        try:
            position = int(raw_position)
        except (TypeError, ValueError):
            continue
        if isinstance(raw_allowed, Mapping):
            raw_allowed = raw_allowed.get("hard_allowed_residues", [])
        if isinstance(raw_allowed, (list, tuple, set)):
            allowed = {str(item).strip().upper() for item in raw_allowed}
            if allowed and all(len(item) == 1 and item in CANONICAL_AA for item in allowed):
                result[position] = allowed
    return result


def _tier_position_rules(tier: Mapping[str, Any]) -> dict[int, set[str]]:
    """Flatten case-owned per-segment hard rules into a position contract."""

    policies = tier.get("node_edit_policies", {})
    if not isinstance(policies, Mapping):
        return {}
    result: dict[int, set[str]] = {}
    for raw_policy in policies.values():
        if not isinstance(raw_policy, Mapping):
            continue
        rules = _normalise_contract(raw_policy.get("position_residue_rules"))
        for position, allowed in rules.items():
            if position in result and result[position] != allowed:
                # An invalid duplicate is represented by an empty rule so the
                # caller fails closed without raising inside evaluator output.
                result[position] = set()
            else:
                result[position] = set(allowed)
    return result


def _resolve_residue_contract(
    score_config: Mapping[str, Any], design_state: Mapping[str, Any]
) -> tuple[str, dict[int, set[str]], dict[str, Any], list[str]]:
    """Resolve the active-Node contract and audit it against case hard rules."""

    raw = _raw_policy(design_state)
    errors: list[str] = []
    resolution: Mapping[str, Any] = {}
    for key in ("case_owned_residue_policy_resolution", "case_owned_residue_policy"):
        item = score_config.get(key)
        if isinstance(item, Mapping) and item.get("active_tier"):
            resolution = item
            break

    tiers = raw.get("tiers", {}) if isinstance(raw.get("tiers"), Mapping) else {}
    resolved_tier = str(resolution.get("active_tier") or "").strip()
    configured_tier = str(score_config.get("case_owned_residue_policy_tier") or "").strip()
    environment_tier = str(os.environ.get("ASTEVOLVE_APH_DESIGN_TIER") or "").strip()
    tier = (
        resolved_tier
        or configured_tier
        or environment_tier
        or str(raw.get("default_tier") or "frontier")
    )
    if tier not in tiers:
        errors.append(f"unsupported_residue_policy_tier:{tier or 'missing'}")
        fallback = str(raw.get("default_tier") or "")
        tier = fallback if fallback in tiers else ("level1" if "level1" in tiers else "")
    for label, declared in (("configured", configured_tier), ("environment", environment_tier)):
        if resolved_tier and declared and declared != resolved_tier:
            errors.append(f"residue_policy_tier_mismatch:{label}:{declared}:{resolved_tier}")

    declared_tier = tiers.get(tier, {}) if isinstance(tiers.get(tier), Mapping) else {}
    declared_contract = _normalise_contract(
        declared_tier.get("residue_mutation_contract")
    )
    declared_rules = _tier_position_rules(declared_tier)
    contract = _normalise_contract(resolution.get("residue_mutation_contract"))
    controller_resolved = bool(contract)
    source = "controller_active_nodes" if controller_resolved else "case_owned_tier"
    if not contract:
        contract = declared_contract
    if not contract and tier in LEVEL_ALLOWED:
        contract = {
            position: set(values)
            for position, values in LEVEL_ALLOWED[tier].items()
        }
    if not contract:
        errors.append("residue_mutation_contract_missing_or_empty")

    for position, allowed in sorted(contract.items()):
        hard_allowed = declared_rules.get(position)
        if not hard_allowed:
            errors.append(f"position_missing_case_hard_rule:A:{position}")
        elif not allowed <= hard_allowed:
            errors.append(f"controller_residue_contract_expands_case_hard_rule:A:{position}")
    if not controller_resolved and declared_contract and contract != declared_contract:
        errors.append("case_tier_contract_resolution_mismatch")

    audit = {
        "active_tier": tier,
        "authority": "case_controller",
        "resolution_source": source,
        "controller_active_node_resolution": controller_resolved,
        "declared_tier_position_count": len(declared_contract),
        "active_contract_position_count": len(contract),
        "seed_positions_zero_based": sorted(SEED_POSITIONS),
        "hard_allowed_residues_zero_based": {
            str(k): sorted(v) for k, v in sorted(contract.items())
        },
    }
    return tier, contract, audit, errors

def _fixed_violations(
    candidate: str, fixed_residues: Optional[Mapping[str, Mapping[Any, str]]]
) -> list[str]:
    fixed = fixed_residues if isinstance(fixed_residues, Mapping) else {}
    assignments = fixed.get("A", {}) if isinstance(fixed.get("A", {}), Mapping) else {}
    reasons: list[str] = []
    for raw_position, expected in assignments.items():
        try:
            position = int(raw_position)
        except (TypeError, ValueError):
            continue
        actual = candidate[position] if 0 <= position < len(candidate) else None
        if actual != str(expected).upper():
            reasons.append(f"fixed_residue_modified:A:{position}")
    return reasons


def _sequence_contract(
    out: Mapping[str, Any], *, design_state: Mapping[str, Any], score_config: Mapping[str, Any],
    fixed_residues: Optional[Mapping[str, Mapping[Any, str]]]
) -> dict[str, Any]:
    candidate = _candidate_sequence(out)
    tier, contract, policy_audit, policy_errors = _resolve_residue_contract(score_config, design_state)
    reasons = list(policy_errors)
    if not candidate:
        reasons.append("missing_chain:A")
    if len(candidate) != len(WT_SEQUENCE):
        reasons.append(f"length_mismatch:A:{len(candidate)}:{len(WT_SEQUENCE)}")
    if candidate:
        for position, residue in enumerate(candidate):
            if residue not in CANONICAL_AA:
                reasons.append(f"noncanonical_residue:A:{position}:{residue}")
        if len(candidate) == len(WT_SEQUENCE):
            for position in _changed_positions(WT_SEQUENCE, candidate):
                if position not in contract:
                    reasons.append(
                        f"mutation_outside_active_residue_contract:A:{position}"
                    )
            for position, allowed in sorted(contract.items()):
                if not 0 <= position < len(candidate):
                    reasons.append(f"residue_contract_position_out_of_bounds:A:{position}")
                    continue
                residue = candidate[position]
                if residue not in allowed:
                    reasons.append(
                        f"amino_acid_not_allowed:{tier}:A:{position}:{residue}"
                    )
            for one_based in VERIFIED_CORE:
                position = one_based - 1
                if candidate[position] != WT_SEQUENCE[position]:
                    reasons.append(f"verified_core_modified:A:{position}")
    reasons.extend(_fixed_violations(candidate, fixed_residues))
    return {
        "candidate": candidate,
        "tier": tier,
        "contract": contract,
        "policy_audit": policy_audit,
        "reasons": sorted(set(reasons)),
        "passed": not reasons,
    }


def _node_positions(value: Any) -> tuple[Optional[str], set[int]]:
    if isinstance(value, Mapping):
        chain = str(value.get("chain_id") or "A")
        raw = value.get("positions", value.get("active_positions", []))
    else:
        chain, raw = "A", value
    if not isinstance(raw, (list, tuple, set)):
        return chain, set()
    try:
        return chain, {int(item) for item in raw}
    except (TypeError, ValueError):
        return chain, set()


def _mutation_scope_contract(
    candidate: str, *, template: str, score_config: Mapping[str, Any],
    masks: Optional[Mapping[str, Any]],
    residue_contract: Optional[Mapping[int, set[str]]] = None,
    require_scope: bool = True,
) -> dict[str, Any]:
    raw = score_config.get("mutation_scope_contract", {})
    reasons: list[str] = []
    if not isinstance(raw, Mapping):
        raw = {}
    by_chain = raw.get("active_positions_by_chain", {})
    by_node = raw.get("active_positions_by_node", {})
    try:
        active = {int(item) for item in (by_chain.get("A", []) if isinstance(by_chain, Mapping) else [])}
        maximum = int(raw.get("max_total_mutations", 0))
    except (TypeError, ValueError):
        active, maximum = set(), 0
        reasons.append("mutation_scope_values_invalid")
    node_union: set[int] = set()
    node_ids: list[str] = []
    if isinstance(by_node, Mapping):
        for node_id, value in by_node.items():
            chain, positions = _node_positions(value)
            node_ids.append(str(node_id))
            if chain == "A":
                node_union.update(positions)
    mask = masks.get("A") if isinstance(masks, Mapping) else None
    mask_values = [] if mask is None else mask
    mask_active = {index for index, enabled in enumerate(mask_values) if bool(enabled)}
    available = bool(raw and active and maximum > 0 and mask is not None and by_node)
    if require_scope and not available:
        reasons.append("mutation_scope_contract_missing_or_empty")
    if available and active != mask_active:
        reasons.append("mutation_scope_mask_mismatch")
    if available and active != node_union:
        reasons.append("mutation_scope_node_mismatch")
    hard_contract_positions = set(residue_contract or {})
    if active - hard_contract_positions:
        reasons.append("mutation_scope_outside_case_residue_contract")
    raw_resolution = score_config.get("case_owned_residue_policy_resolution", {})
    has_controller_resolution = bool(
        isinstance(raw_resolution, Mapping)
        and _normalise_contract(raw_resolution.get("residue_mutation_contract"))
    )
    if has_controller_resolution and active != hard_contract_positions:
        reasons.append("mutation_scope_residue_contract_mismatch")
    step_changes = _changed_positions(template, candidate)
    absolute_changes = _changed_positions(WT_SEQUENCE, candidate)
    if len(candidate) != len(template):
        reasons.append("mutation_scope_template_length_mismatch")
    elif not set(step_changes) <= active:
        reasons.append("mutation_outside_active_ast_scope")
    if len(candidate) == len(WT_SEQUENCE) and not set(absolute_changes) <= active:
        reasons.append("retired_position_not_reverted_to_case_reference")
    if len(step_changes) > maximum:
        reasons.append("mutation_budget_exceeded")
    return {
        "passed": not reasons,
        "available": available,
        "reasons": sorted(set(reasons)),
        "active_positions_zero_based": sorted(active),
        "active_node_ids": sorted(node_ids),
        "max_total_mutations": maximum,
        "step_changed_positions_zero_based": step_changes,
        "absolute_changed_positions_zero_based": absolute_changes,
        "scope_contract": dict(raw),
    }


def _states(structure: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = structure.get("states", []) if isinstance(structure, Mapping) else []
    return {
        str(row.get("name")): row for row in (rows or [])
        if isinstance(row, Mapping) and row.get("name")
    }


def _state_path(state: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not isinstance(state, Mapping):
        return None
    nested = state.get("structure_metrics", {})
    candidates = [
        state.get("cif_path"), state.get("structure_path"), state.get("pdb_path"),
        nested.get("cif_path") if isinstance(nested, Mapping) else None,
        nested.get("structure_path") if isinstance(nested, Mapping) else None,
        nested.get("pdb_path") if isinstance(nested, Mapping) else None,
    ]
    for value in candidates:
        if value and Path(str(value)).exists():
            return str(value)
    return None


def _load_atoms(path: Optional[str]) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    try:
        from astevolve.metrics.structure import _load_structure_atoms

        return [dict(atom) for atom in _load_structure_atoms(str(path))]
    except Exception:
        return []


def _dominant_protein_chain(atoms: Sequence[Mapping[str, Any]], state: Optional[Mapping[str, Any]]) -> str:
    actual = {str(atom.get("asym") or "") for atom in atoms if str(atom.get("group") or "").upper() == "ATOM"}
    preferred: list[str] = []
    if isinstance(state, Mapping):
        for unit in state.get("entity_units", []) or []:
            if not isinstance(unit, Mapping) or str(unit.get("source_chain") or "") != "A":
                continue
            preferred.extend(str(unit.get(key) or "") for key in ("asym_id", "asym_alias", "label", "base_label"))
    counts: dict[str, int] = {}
    for atom in atoms:
        if str(atom.get("group") or "").upper() != "ATOM" or str(atom.get("atom") or "").upper() != "CA":
            continue
        chain = str(atom.get("asym") or "")
        counts[chain] = counts.get(chain, 0) + 1
    candidates = [chain for chain in preferred if chain in actual and counts.get(chain)]
    return max(candidates, key=lambda item: counts[item]) if candidates else (max(counts, key=counts.get) if counts else "")


def _protein_atoms(atoms: Sequence[Mapping[str, Any]], chain: str) -> list[dict[str, Any]]:
    return [dict(atom) for atom in atoms if str(atom.get("group") or "").upper() == "ATOM" and str(atom.get("asym") or "") == chain]


def _atom_map(atoms: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], np.ndarray]:
    result: dict[tuple[int, str], np.ndarray] = {}
    for atom in atoms:
        try:
            key = (int(atom.get("seq_id")), str(atom.get("atom") or "").strip().upper())
            xyz = np.asarray(atom.get("xyz"), dtype=float)
        except (TypeError, ValueError):
            continue
        if xyz.shape == (3,) and np.all(np.isfinite(xyz)):
            result[key] = xyz
    return result


def _fit_transform(reference: np.ndarray, candidate: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if reference.shape != candidate.shape or len(reference) < 3:
        return None
    ref_centroid = reference.mean(axis=0)
    mob_centroid = candidate.mean(axis=0)
    ref_zero = reference - ref_centroid
    mob_zero = candidate - mob_centroid
    try:
        left, _, right = np.linalg.svd(mob_zero.T @ ref_zero)
    except np.linalg.LinAlgError:
        return None
    correction = np.diag([1.0, 1.0, -1.0 if np.linalg.det(left @ right) < 0 else 1.0])
    return left @ correction @ right, mob_centroid, ref_centroid


def _aligned(xyz: np.ndarray, transform: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    rotation, mob_centroid, ref_centroid = transform
    return (xyz - mob_centroid) @ rotation + ref_centroid


def _rmsd(reference: Sequence[np.ndarray], candidate: Sequence[np.ndarray]) -> Optional[float]:
    if len(reference) != len(candidate) or not reference:
        return None
    delta = np.asarray(candidate) - np.asarray(reference)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _sequence_mismatches(protein_atoms: Sequence[Mapping[str, Any]], candidate_sequence: str) -> list[dict[str, Any]]:
    observed: dict[int, str] = {}
    for atom in protein_atoms:
        try:
            observed.setdefault(int(atom.get("seq_id")), str(atom.get("comp") or "").upper())
        except (TypeError, ValueError):
            continue
    mismatches = []
    for one_based, comp in sorted(observed.items()):
        if not 1 <= one_based <= len(candidate_sequence):
            continue
        expected = candidate_sequence[one_based - 1]
        actual = THREE_TO_ONE.get(comp)
        if actual != expected:
            mismatches.append({"position_1based": one_based, "expected": expected, "coordinate_residue": comp})
    return mismatches


def _one_instance(atoms: Sequence[Mapping[str, Any]], comp: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for atom in atoms:
        if str(atom.get("comp") or "").upper() != comp:
            continue
        try:
            key = (str(atom.get("asym") or ""), int(atom.get("seq_id")))
        except (TypeError, ValueError):
            continue
        groups.setdefault(key, []).append(dict(atom))
    return next(iter(groups.values())) if len(groups) == 1 else []


def _distance(left: Any, right: Any) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))


def _severe_clashes(
    protein_atoms: Sequence[Mapping[str, Any]], kan_atoms: Sequence[Mapping[str, Any]],
    mg_atoms: Sequence[Mapping[str, Any]], cutoff: float,
) -> dict[str, Any]:
    """Count severe nonbonded clashes using a cutoff-sized 3-D cell list."""

    rows: list[tuple[str, Mapping[str, Any]]] = []
    rows.extend(("protein", atom) for atom in protein_atoms if not str(atom.get("atom") or "").upper().startswith("H"))
    rows.extend(("KAN", atom) for atom in kan_atoms if not str(atom.get("atom") or "").upper().startswith("H"))
    rows.extend(("MG", atom) for atom in mg_atoms)
    cell_size = max(float(cutoff), 1e-8)
    buckets: dict[
        tuple[int, int, int],
        list[tuple[str, Mapping[str, Any], np.ndarray]],
    ] = {}
    count = 0
    checked_pairs = 0
    examples: list[dict[str, Any]] = []
    minimum: Optional[float] = None
    offsets = (-1, 0, 1)

    for right_kind, right in rows:
        try:
            right_xyz = np.asarray(right.get("xyz"), dtype=float)
        except (TypeError, ValueError):
            continue
        if right_xyz.shape != (3,) or not np.all(np.isfinite(right_xyz)):
            continue
        cell = tuple(int(math.floor(float(value) / cell_size)) for value in right_xyz)
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    neighbour = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    for left_kind, left, left_xyz in buckets.get(neighbour, []):
                        if left_kind == right_kind == "KAN" or left_kind == right_kind == "MG":
                            continue
                        if left_kind == right_kind == "protein":
                            try:
                                same_chain = str(left.get("asym")) == str(right.get("asym"))
                                adjacent = abs(int(left.get("seq_id")) - int(right.get("seq_id"))) <= 1
                            except (TypeError, ValueError):
                                continue
                            if same_chain and adjacent:
                                continue
                        checked_pairs += 1
                        value = float(np.linalg.norm(left_xyz - right_xyz))
                        minimum = value if minimum is None else min(minimum, value)
                        if value >= cutoff:
                            continue
                        count += 1
                        if len(examples) < 12:
                            examples.append({
                                "left": [left_kind, left.get("asym"), left.get("seq_id"), left.get("atom")],
                                "right": [right_kind, right.get("asym"), right.get("seq_id"), right.get("atom")],
                                "distance_angstrom": value,
                            })
        buckets.setdefault(cell, []).append((right_kind, right, right_xyz))

    return {
        "count": count,
        "cutoff_angstrom": cutoff,
        "minimum_checked_distance_angstrom": minimum,
        "checked_pair_count": checked_pairs,
        "algorithm": "cell_list_27_neighbors",
        "examples": examples,
    }

def structure_measurements(
    candidate_path: Optional[str], *, reference_path: str | Path = REFERENCE_CHAIN_A,
    candidate_sequence: str = WT_SEQUENCE, state: Optional[Mapping[str, Any]] = None,
    contact_cutoff: float = 4.5, clash_cutoff: float = 1.6,
) -> dict[str, Any]:
    """Return raw reference-aligned protein, KAN, Mg, and clash measurements."""

    reference_atoms = _load_atoms(str(reference_path))
    candidate_atoms = _load_atoms(candidate_path)
    reference_chain = _dominant_protein_chain(reference_atoms, None)
    candidate_chain = _dominant_protein_chain(candidate_atoms, state)
    reference_protein = _protein_atoms(reference_atoms, reference_chain)
    candidate_protein = _protein_atoms(candidate_atoms, candidate_chain)
    ref_map, cand_map = _atom_map(reference_protein), _atom_map(candidate_protein)
    reference_backbone = sorted(key for key in ref_map if 10 <= key[0] <= 264 and key[1] in BACKBONE_ATOMS)
    common_backbone = [key for key in reference_backbone if key in cand_map]
    transform = _fit_transform(
        np.asarray([ref_map[key] for key in common_backbone]),
        np.asarray([cand_map[key] for key in common_backbone]),
    ) if common_backbone else None

    def aligned_rmsd(keys: Iterable[tuple[int, str]]) -> tuple[Optional[float], int]:
        common = [key for key in keys if key in ref_map and key in cand_map]
        if transform is None:
            return None, len(common)
        return _rmsd([ref_map[key] for key in common], [_aligned(cand_map[key], transform) for key in common]), len(common)

    global_rmsd, global_count = aligned_rmsd(reference_backbone)
    core_backbone_keys = sorted(key for key in ref_map if key[0] in VERIFIED_CORE and key[1] in BACKBONE_ATOMS)
    core_heavy_keys = sorted(key for key in ref_map if key[0] in VERIFIED_CORE and not key[1].startswith("H"))
    core_backbone_rmsd, core_backbone_count = aligned_rmsd(core_backbone_keys)
    core_heavy_rmsd, core_heavy_count = aligned_rmsd(core_heavy_keys)
    mismatches = _sequence_mismatches(candidate_protein, candidate_sequence)
    backbone_coverage = global_count / max(1, len(reference_backbone))

    reference_kan = _one_instance(reference_atoms, "KAN")
    candidate_kan = _one_instance(candidate_atoms, "KAN")
    reference_mg = _one_instance(reference_atoms, "MG")
    candidate_mg = _one_instance(candidate_atoms, "MG")
    ref_kan_map = {str(atom.get("atom") or "").upper(): np.asarray(atom.get("xyz"), dtype=float) for atom in reference_kan if not str(atom.get("atom") or "").upper().startswith("H")}
    cand_kan_map = {str(atom.get("atom") or "").upper(): np.asarray(atom.get("xyz"), dtype=float) for atom in candidate_kan if not str(atom.get("atom") or "").upper().startswith("H")}
    common_kan = sorted(set(ref_kan_map) & set(cand_kan_map))
    pose_rmsd = None
    if transform is not None and common_kan:
        pose_rmsd = _rmsd([ref_kan_map[key] for key in common_kan], [_aligned(cand_kan_map[key], transform) for key in common_kan])

    contacted = []
    if candidate_kan:
        ligand_xyz = [np.asarray(atom.get("xyz"), dtype=float) for atom in candidate_kan if not str(atom.get("atom") or "").upper().startswith("H")]
        for position in KAN_CONTACT_SHELL:
            residue_xyz = [np.asarray(atom.get("xyz"), dtype=float) for atom in candidate_protein if int(atom.get("seq_id") or -1) == position and not str(atom.get("atom") or "").upper().startswith("H")]
            if residue_xyz and ligand_xyz and min(_distance(left, right) for left in residue_xyz for right in ligand_xyz) <= contact_cutoff:
                contacted.append(position)
    contact_coverage = len(contacted) / len(KAN_CONTACT_SHELL) if candidate_kan else None

    mg_distances: dict[str, Any] = {}
    mg_errors: dict[str, Any] = {}
    if len(reference_mg) == len(candidate_mg) == 1:
        ref_mg_xyz, cand_mg_xyz = reference_mg[0]["xyz"], candidate_mg[0]["xyz"]
        for position in (195, 208):
            ref_xyz = [atom["xyz"] for atom in reference_protein if int(atom.get("seq_id") or -1) == position and not str(atom.get("atom") or "").upper().startswith("H")]
            cand_xyz = [atom["xyz"] for atom in candidate_protein if int(atom.get("seq_id") or -1) == position and not str(atom.get("atom") or "").upper().startswith("H")]
            if ref_xyz and cand_xyz:
                ref_distance = min(_distance(item, ref_mg_xyz) for item in ref_xyz)
                cand_distance = min(_distance(item, cand_mg_xyz) for item in cand_xyz)
                mg_distances[str(position)] = {"reference_angstrom": ref_distance, "candidate_angstrom": cand_distance}
                mg_errors[str(position)] = abs(cand_distance - ref_distance)

    return {
        "candidate_path": candidate_path,
        "reference_path": str(reference_path),
        "reference_chain": reference_chain,
        "candidate_chain": candidate_chain,
        "protein_sequence_mismatches": mismatches,
        "global_backbone_rmsd_angstrom": global_rmsd,
        "global_backbone_atom_count": global_count,
        "global_backbone_reference_atom_count": len(reference_backbone),
        "global_backbone_coverage": backbone_coverage,
        "core_backbone_rmsd_angstrom": core_backbone_rmsd,
        "core_backbone_atom_count": core_backbone_count,
        "core_backbone_reference_atom_count": len(core_backbone_keys),
        "core_all_heavy_rmsd_angstrom": core_heavy_rmsd,
        "core_all_heavy_atom_count": core_heavy_count,
        "core_all_heavy_reference_atom_count": len(core_heavy_keys),
        "kan_heavy_atom_count": len(cand_kan_map),
        "kan_reference_heavy_atom_count": len(ref_kan_map),
        "kan_matched_heavy_atom_count": len(common_kan),
        "kan_pose_rmsd_angstrom": pose_rmsd,
        "kan_contacted_positions_1based": contacted,
        "kan_contact_coverage": contact_coverage,
        "mg_atom_count": len(candidate_mg),
        "mg_distances": mg_distances,
        "mg_distance_errors_angstrom": mg_errors,
        "mg_max_distance_error_angstrom": max(mg_errors.values()) if len(mg_errors) == 2 else None,
        "severe_clashes": _severe_clashes(candidate_protein, candidate_kan, candidate_mg, clash_cutoff),
        "protein_available": bool(transform is not None and backbone_coverage >= 0.95 and not mismatches),
        "ligand_atoms_available": bool(candidate_kan and len(candidate_mg) == 1),
    }


def _entity_evidence(state: Optional[Mapping[str, Any]]) -> tuple[bool, bool, list[dict[str, str]]]:
    rows = []
    if not isinstance(state, Mapping):
        return False, False, rows
    for item in state.get("entities", []) or []:
        if not isinstance(item, Mapping):
            continue
        label = str(item.get("label") or item.get("id") or item.get("name") or "").upper()
        kind = str(item.get("kind") or item.get("type") or "").lower()
        rows.append({"label": label, "kind": kind})
    for item in state.get("entity_units", []) or []:
        if not isinstance(item, Mapping):
            continue
        label = str(item.get("base_label") or item.get("label") or "").upper()
        kind = str(item.get("kind") or "").lower()
        rows.append({"label": label, "kind": kind})
    has_kan = any(row["label"] == "KAN" and "ligand" in row["kind"] for row in rows)
    has_mg = any(row["label"] == "MG" and "ion" in row["kind"] for row in rows)
    return has_kan, has_mg, rows


def _protein_provenance(state: Optional[Mapping[str, Any]], path: Optional[str]) -> dict[str, Any]:
    if not isinstance(state, Mapping) or not path:
        return {"accepted": False, "reason": "candidate state/path missing"}
    provider = str(state.get("provider") or "").strip().lower()
    stage = str(state.get("structure_stage") or "").strip().lower()
    provenance = state.get("protein_provenance", {})
    explicit = provenance if isinstance(provenance, Mapping) else {}
    copied = bool(
        state.get("reference_coordinates_copied")
        or state.get("reference_heteroatoms_transplanted")
        or explicit.get("reference_coordinates_copied")
        or explicit.get("reference_heteroatoms_transplanted")
    )
    accepted = (provider in {"protenix", "alphafold3", "service"} and bool(stage)) or (
        bool(explicit.get("coordinates_generated_with_candidate")) and not copied
    )
    return {"accepted": bool(accepted and not copied), "provider": provider, "structure_stage": stage or None, "explicit": dict(explicit), "reason": "" if accepted and not copied else "candidate-specific protein provenance is absent"}


def _ligand_provenance(
    state: Optional[Mapping[str, Any]], path: Optional[str], candidate: str,
    *,
    require_generated: bool,
    allow_transplant: bool,
    reference_paths: Sequence[str | Path] = (REFERENCE_CHAIN_A, REFERENCE_CHAIN_B),
) -> dict[str, Any]:
    protein = _protein_provenance(state, path)
    if not isinstance(state, Mapping) or not path:
        return {**protein, "accepted": False, "reason": "candidate complex state/path missing"}
    has_kan, has_mg, entities = _entity_evidence(state)
    raw = state.get("ligand_provenance", state.get("complex_provenance", {}))
    explicit = raw if isinstance(raw, Mapping) else {}
    copied = bool(
        state.get("reference_coordinates_copied")
        or state.get("reference_heteroatoms_transplanted")
        or explicit.get("reference_coordinates_copied")
        or explicit.get("reference_heteroatoms_transplanted")
    )
    method = str(explicit.get("method") or explicit.get("source") or "").lower()
    provider = str(state.get("provider") or "").lower()
    runtime_generated = provider in {"protenix", "alphafold3", "service"} and bool(state.get("structure_stage"))
    explicit_generated = bool(explicit.get("coordinates_generated_with_candidate")) and method in {
        "joint_prediction", "candidate_complex_prediction", "candidate_specific_docking", "experimental_candidate_complex"
    }
    declared_reference_paths = {Path(item).resolve() for item in reference_paths}
    reference_path = Path(path).resolve() in declared_reference_paths
    experimental_reference = (
        candidate == WT_SEQUENCE and reference_path and method in {"experimental_1nd4", "experimental_candidate_complex"}
    )
    blocked_word = any(token in f"{provider} {method}" for token in ("transplant", "copied_reference", "template", "unknown"))
    generated = runtime_generated or explicit_generated or experimental_reference
    accepted = bool(
        protein.get("accepted") and has_kan and has_mg and not blocked_word
        and not copied and (generated or not require_generated)
        and (not reference_path or experimental_reference)
        and not allow_transplant
    )
    reason = "" if accepted else "KAN/Mg coordinates are not an independently comparable candidate complex"
    return {
        **protein,
        "accepted": accepted,
        "reason": reason,
        "has_declared_KAN": has_kan,
        "has_declared_MG": has_mg,
        "entities": entities,
        "runtime_generated": runtime_generated,
        "explicit": dict(explicit),
        "reference_path": reference_path,
        "reference_coordinates_copied": copied,
    }


def _hard_details(
    reason: str, binding: Sequence[str], *, min_score: float = 1.0, **extra: Any
) -> dict[str, Any]:
    return {
        "dimension": "correctness",
        "state": "preserve",
        "required": True,
        "hard_gate": True,
        "hard_gate_min_score": _clamp01(min_score),
        "hard_gate_reason": reason,
        "semantic_binding": {
            "structural_nodes": list(binding),
            "failure_action": "optimize_node",
            "reason": reason,
            "priority": "high",
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "excluded_scientific_claims": list(EXCLUDED_CLAIMS),
        **extra,
    }


def _term_dict(term: ScoreTerm) -> dict[str, Any]:
    return term.to_dict()


def evaluate_candidate(
    out: Mapping[str, Any], *, compiled: Optional[Mapping[str, Any]] = None,
    design_state: Optional[Mapping[str, Any]] = None, masks: Optional[Mapping[str, Any]] = None,
    template_seqs: Optional[Mapping[str, str]] = None,
    fixed_residues: Optional[Mapping[str, Mapping[Any, str]]] = None,
    score_config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """CPU-only manifest preflight; production structural gates are not run."""

    del compiled
    state = design_state if isinstance(design_state, Mapping) else {}
    config = score_config if isinstance(score_config, Mapping) else {}
    sequence = _sequence_contract(out, design_state=state, score_config=config, fixed_residues=fixed_residues)
    scope = _mutation_scope_contract(
        sequence["candidate"], template=_template_sequence(template_seqs), score_config=config,
        masks=masks, residue_contract=sequence["contract"],
        require_scope=bool(config.get("mutation_scope_contract")),
    )
    reasons = sorted(set(sequence["reasons"] + scope["reasons"]))
    passed = not reasons
    terms = [
        ScoreTerm(
            "aph_sequence_integrity", "task_correctness", 1.0 if not sequence["reasons"] else 0.0, 0.0,
            _hard_details("aph_sequence_contract_failed", ["design_surface_sites"],
                          active_tier=sequence["tier"], residue_policy_artifact=sequence["policy_audit"], violations=sequence["reasons"]),
            warnings=[] if not sequence["reasons"] else sequence["reasons"], backend=PLUGIN_NAME, available=bool(sequence["candidate"]),
        ),
        ScoreTerm(
            "aph_mutation_scope_integrity", "semantic_constraint", 1.0 if scope["passed"] else 0.0, 0.0,
            _hard_details("aph_mutation_outside_active_ast_scope", scope["active_node_ids"], **scope),
            warnings=[] if scope["passed"] else scope["reasons"], backend=PLUGIN_NAME, available=True,
        ),
    ]
    return {
        "schema_version": "ast_evaluator_report_v1",
        "evaluation_scope": "cpu_sequence_contract_preflight",
        "normalized_score": 1.0 if passed else 0.0,
        "soft_score": 1.0 if passed else 0.0,
        "loss": 0.0 if passed else 1.0,
        "hard_gate_pass": passed,
        "disqualification_reasons": reasons,
        "gate_status": {"passed": passed, "hard_gate_pass": passed, "hard_failures": reasons, "disqualification_reasons": reasons},
        "terms": [_term_dict(term) for term in terms],
        "weakest_terms": sorted((_term_dict(term) for term in terms), key=lambda row: row["score"]),
        "warnings": ["CPU preflight does not evaluate production structural or ligand gates"],
        "residue_policy_artifact": sequence["policy_audit"],
    }


class Aph3iiaKanamycinPlugin:
    """Production two-state evaluator with provenance-gated KAN/Mg geometry."""

    name = PLUGIN_NAME

    def score_terms(self, context: EvaluatorContext) -> list[ScoreTerm]:
        sequence = _sequence_contract(
            context.out, design_state=context.design_state, score_config=context.score_config,
            fixed_residues=context.fixed_residues,
        )
        scope = _mutation_scope_contract(
            sequence["candidate"], template=_template_sequence(context.template_seqs),
            score_config=context.score_config, masks=context.masks,
            residue_contract=sequence["contract"], require_scope=True,
        )
        states = _states(context.structure)
        apo, complex_state = states.get("candidate_apo"), states.get("candidate_chainA_KAN_MG")
        apo_path, complex_path = _state_path(apo), _state_path(complex_state)
        protein_apo_provenance = _protein_provenance(apo, apo_path)
        protein_complex_provenance = _protein_provenance(complex_state, complex_path)
        require_generated = _config_bool(context, "aph_require_generated_candidate_ligand", True)
        allow_transplant = _config_bool(context, "aph_allow_reference_heteroatom_transplant", False)
        reference_a = str(context.score_config.get("aph_reference_chain_a_path") or REFERENCE_CHAIN_A)
        reference_b = str(context.score_config.get("aph_reference_chain_b_path") or REFERENCE_CHAIN_B)
        ligand_provenance = _ligand_provenance(
            complex_state, complex_path, sequence["candidate"],
            require_generated=require_generated,
            allow_transplant=allow_transplant,
            reference_paths=(reference_a, reference_b),
        )
        contact_cutoff = _config_float(context, "aph_kanamycin_contact_cutoff", 4.5)
        clash_cutoff = 1.6
        apo_measure = structure_measurements(
            apo_path, reference_path=reference_a, candidate_sequence=sequence["candidate"], state=apo,
            contact_cutoff=contact_cutoff, clash_cutoff=clash_cutoff,
        )
        complex_measure = structure_measurements(
            complex_path, reference_path=reference_a, candidate_sequence=sequence["candidate"], state=complex_state,
            contact_cutoff=contact_cutoff, clash_cutoff=clash_cutoff,
        )
        min_coverage = 0.95
        apo_state_composition = {
            "kan_heavy_atom_count": int(apo_measure.get("kan_heavy_atom_count") or 0),
            "mg_atom_count": int(apo_measure.get("mg_atom_count") or 0),
        }
        apo_composition_valid = (
            apo_state_composition["kan_heavy_atom_count"] == 0
            and apo_state_composition["mg_atom_count"] == 0
        )
        apo_composition_reason = (
            None if apo_composition_valid else "candidate_apo_contains_KAN_or_MG"
        )
        protein_available = all((
            protein_apo_provenance.get("accepted"), protein_complex_provenance.get("accepted"),
            apo_composition_valid,
            apo_measure.get("protein_available"), complex_measure.get("protein_available"),
            float(apo_measure.get("global_backbone_coverage") or 0.0) >= min_coverage,
            float(complex_measure.get("global_backbone_coverage") or 0.0) >= min_coverage,
        ))
        global_limit = _config_float(context, "aph_global_backbone_rmsd_max", 1.5)
        global_values = [
            apo_measure.get("global_backbone_rmsd_angstrom"),
            complex_measure.get("global_backbone_rmsd_angstrom"),
        ]
        global_available = bool(
            protein_available
            and global_limit > 0.0
            and all(_finite_float(value) is not None for value in global_values)
        )
        global_worst_rmsd = (
            max(float(value) for value in global_values)
            if global_available else None
        )
        global_score = _half_at_limit(global_worst_rmsd, global_limit)
        gate_thresholds = {
            name: _hard_gate_threshold(context, name)
            for name in HARD_GATE_MIN_SCORES
        }
        global_pass = bool(
            global_available
            and global_score >= gate_thresholds["aph_global_fold_preservation"]
        )
        core_backbone_limit = _config_float(context, "aph_core_backbone_rmsd_max", 0.75)
        core_heavy_limit = _config_float(context, "aph_core_all_heavy_rmsd_max", 1.25)
        core_count_available = all(
            int(item.get("core_backbone_atom_count") or 0) == 16
            and int(item.get("core_all_heavy_atom_count") or 0) == 32
            for item in (apo_measure, complex_measure)
        )
        core_components = [
            (apo_measure.get("core_backbone_rmsd_angstrom"), core_backbone_limit),
            (complex_measure.get("core_backbone_rmsd_angstrom"), core_backbone_limit),
            (apo_measure.get("core_all_heavy_rmsd_angstrom"), core_heavy_limit),
            (complex_measure.get("core_all_heavy_rmsd_angstrom"), core_heavy_limit),
        ]
        core_available = bool(
            protein_available
            and core_backbone_limit > 0.0
            and core_heavy_limit > 0.0
            and core_count_available
            and all(_finite_float(value) is not None for value, _ in core_components)
        )
        core_worst_normalized_error = (
            max(float(value) / max(1e-8, limit) for value, limit in core_components)
            if core_available else None
        )
        core_score = _half_at_limit(core_worst_normalized_error, 1.0)
        core_pass = bool(
            core_available
            and core_score >= gate_thresholds["aph_verified_core_preservation"]
        )
        ligand_available = bool(
            ligand_provenance.get("accepted")
            and complex_measure.get("ligand_atoms_available")
        )
        contact_min = _config_float(context, "aph_kanamycin_contact_coverage_min", 0.90)
        contact_value = complex_measure.get("kan_contact_coverage")
        contact_available = bool(
            ligand_available and _finite_float(contact_value) is not None
        )
        contact_score = _clamp01(contact_value) if contact_available else 0.0
        contact_pass = bool(
            contact_available
            and contact_score >= gate_thresholds["aph_kanamycin_contact_preservation"]
        )
        pose_limit = _config_float(context, "aph_kanamycin_pose_rmsd_max", 1.5)
        pose_value = complex_measure.get("kan_pose_rmsd_angstrom")
        pose_atom_coverage = (
            float(complex_measure.get("kan_matched_heavy_atom_count") or 0)
            / max(1, int(complex_measure.get("kan_reference_heavy_atom_count") or 0))
        )
        pose_available = bool(
            ligand_available
            and pose_limit > 0.0
            and pose_atom_coverage >= 0.95
            and _finite_float(pose_value) is not None
        )
        pose_score = _half_at_limit(pose_value, pose_limit) if pose_available else 0.0
        pose_pass = bool(
            pose_available
            and pose_score >= gate_thresholds["aph_kanamycin_pose_preservation"]
        )
        mg_limit = _config_float(context, "aph_mg_distance_error_max", 0.5)
        mg_value = complex_measure.get("mg_max_distance_error_angstrom")
        mg_available = bool(
            ligand_available
            and mg_limit > 0.0
            and _finite_float(mg_value) is not None
        )
        mg_score = _half_at_limit(mg_value, mg_limit) if mg_available else 0.0
        mg_pass = bool(
            mg_available
            and mg_score >= gate_thresholds["aph_mg_geometry_preservation"]
        )
        clash_max = _config_int(context, "aph_severe_clash_count_max", 0)
        apo_clashes = int((apo_measure.get("severe_clashes") or {}).get("count") or 0)
        complex_clashes = int((complex_measure.get("severe_clashes") or {}).get("count") or 0)
        clash_available = bool(protein_available and ligand_available)
        clash_excess = (
            max(0, apo_clashes - clash_max) + max(0, complex_clashes - clash_max)
        )
        clash_score = 1.0 / (1.0 + clash_excess) if clash_available else 0.0
        clash_pass = bool(
            clash_available
            and clash_score >= gate_thresholds["aph_clash_free"]
        )
        candidate = sequence["candidate"]
        mutation_capacity = max(
            1,
            int(scope.get("max_total_mutations") or 0),
            len(sequence["contract"]),
        )
        mutation_count = (
            len(_changed_positions(WT_SEQUENCE, candidate))
            if len(candidate) == len(WT_SEQUENCE)
            else mutation_capacity
        )
        seed_mutation_count = (
            sum(
                candidate[position] != WT_SEQUENCE[position]
                for position in SEED_POSITIONS
            )
            if len(candidate) == len(WT_SEQUENCE)
            else len(SEED_POSITIONS)
        )
        frontier_unlabeled_mutation_count = max(
            0, mutation_count - seed_mutation_count
        )
        label_components = []
        label_counts = {
            "wild_type": 0,
            "level1_supported": 0,
            "level2_extension": 0,
            "unsupported_seed_substitution": 0,
            "frontier_unlabeled_mutations": frontier_unlabeled_mutation_count,
        }
        if len(candidate) == len(WT_SEQUENCE):
            for position in sorted(SEED_POSITIONS):
                residue = candidate[position]
                if residue == WT_SEQUENCE[position]:
                    label_components.append(1.0)
                    label_counts["wild_type"] += 1
                elif residue in LEVEL_ALLOWED["level1"][position]:
                    label_components.append(1.0)
                    label_counts["level1_supported"] += 1
                elif residue in LEVEL_ALLOWED["level2"][position]:
                    label_components.append(0.5)
                    label_counts["level2_extension"] += 1
                else:
                    label_components.append(0.0)
                    label_counts["unsupported_seed_substitution"] += 1
        label_score = (
            float(sum(label_components) / len(label_components))
            if label_components
            else 0.0
        )
        burden_target = 2
        burden_score = _clamp01(
            1.0
            - max(0, mutation_count - burden_target)
            / max(1, mutation_capacity - burden_target)
        )
        prior = _finite_float(context.out.get("progen_loglik_avg"))
        prior_reference = _finite_float(context.score_config.get("aph_sequence_prior_reference"))
        prior_scale = max(1e-8, _finite_float(context.score_config.get("aph_sequence_prior_scale")) or 0.25)
        prior_score = None
        if prior is not None and prior_reference is not None:
            z = max(-700.0, min(700.0, (prior - prior_reference) / prior_scale))
            prior_score = 1.0 / (1.0 + math.exp(-z))
        plausibility_score = burden_score if prior_score is None else 0.75 * prior_score + 0.25 * burden_score
        diversity_score = _clamp01(mutation_count / mutation_capacity)

        repeat_measure = structure_measurements(
            complex_path, reference_path=reference_b, candidate_sequence=candidate, state=complex_state,
            contact_cutoff=contact_cutoff, clash_cutoff=clash_cutoff,
        )
        repeat_components = []
        for value, scale in (
            (repeat_measure.get("global_backbone_rmsd_angstrom"), global_limit),
            (repeat_measure.get("core_backbone_rmsd_angstrom"), core_backbone_limit),
            (repeat_measure.get("core_all_heavy_rmsd_angstrom"), core_heavy_limit),
            (repeat_measure.get("kan_pose_rmsd_angstrom"), pose_limit),
            (repeat_measure.get("mg_max_distance_error_angstrom"), mg_limit),
        ):
            if value is not None:
                repeat_components.append(math.exp(-((float(value) / max(1e-8, scale)) ** 2)))
        if repeat_measure.get("kan_contact_coverage") is not None:
            repeat_components.append(_clamp01(repeat_measure["kan_contact_coverage"]))
        repeat_available = bool(ligand_available and repeat_measure.get("protein_available") and len(repeat_components) == 6)
        repeat_score = min(repeat_components) if repeat_available else 0.0
        seed_std = _finite_float(context.out.get("seed_score_std"))
        node_summary = context.structure.get("node_summary", {}) if isinstance(context.structure, Mapping) else {}
        node_min = _finite_float(node_summary.get("node_plddt_min") if isinstance(node_summary, Mapping) else None)

        terms = [
            ScoreTerm("aph_sequence_integrity", "task_correctness", 1.0 if sequence["passed"] else 0.0, _weight(context, "aph_sequence_integrity", 0.0), _hard_details("aph_sequence_contract_failed", ["design_surface_sites"], active_tier=sequence["tier"], residue_policy_artifact=sequence["policy_audit"], violations=sequence["reasons"]), sequence["reasons"], PLUGIN_NAME, bool(candidate)),
            ScoreTerm("aph_mutation_scope_integrity", "semantic_constraint", 1.0 if scope["passed"] else 0.0, _weight(context, "aph_mutation_scope_integrity", 0.0), _hard_details("aph_mutation_outside_active_ast_scope", scope["active_node_ids"], **scope), scope["reasons"], PLUGIN_NAME, scope["available"]),
            ScoreTerm("aph_global_fold_preservation", "task_specific", global_score, _weight(context, "aph_global_fold_preservation", 0.0), _hard_details("aph_global_fold_not_preserved", ["global_observed_fold"], min_score=HARD_GATE_MIN_SCORES["aph_global_fold_preservation"], apo_state_composition=apo_state_composition, state_composition_reason=apo_composition_reason, worst_case_rmsd_angstrom=global_worst_rmsd, score_formula="exp(-ln(2)*(worst_rmsd/limit)^2)", threshold_angstrom=global_limit, minimum_backbone_coverage=min_coverage, apo_provenance=protein_apo_provenance, complex_provenance=protein_complex_provenance, report_details={"apo": apo_measure, "candidate_complex": complex_measure}), [] if global_pass else ["candidate apo/complex protein evidence is missing, mismatched, or above the RMSD threshold"], PLUGIN_NAME, global_available),
            ScoreTerm("aph_verified_core_preservation", "task_specific", core_score, _weight(context, "aph_verified_core_preservation", 0.0), _hard_details("aph_verified_core_not_preserved", ["verified_functional_core"], min_score=HARD_GATE_MIN_SCORES["aph_verified_core_preservation"], worst_case_normalized_error=core_worst_normalized_error, score_formula="exp(-ln(2)*worst_normalized_error^2)", verified_positions_1based=list(VERIFIED_CORE), backbone_threshold_angstrom=core_backbone_limit, all_heavy_threshold_angstrom=core_heavy_limit, report_details={"apo": apo_measure, "candidate_complex": complex_measure}), [] if core_pass else ["verified-core backbone/heavy evidence is incomplete or above threshold"], PLUGIN_NAME, core_available),
            ScoreTerm("aph_kanamycin_contact_preservation", "task_specific", contact_score, _weight(context, "aph_kanamycin_contact_preservation", 0.0), _hard_details("aph_kanamycin_contact_not_preserved", ["kanamycin_contact_shell"], min_score=HARD_GATE_MIN_SCORES["aph_kanamycin_contact_preservation"], score_formula="contacted_shell_fraction", contact_cutoff_angstrom=contact_cutoff, minimum_coverage=contact_min, contact_coverage=contact_value, ligand_provenance=ligand_provenance, report_details=complex_measure), [] if contact_pass else ["comparable candidate KAN contact evidence is unavailable or below threshold"], PLUGIN_NAME, contact_available),
            ScoreTerm("aph_kanamycin_pose_preservation", "task_specific", pose_score, _weight(context, "aph_kanamycin_pose_preservation", 0.0), _hard_details("aph_kanamycin_pose_not_preserved", ["kanamycin_contact_shell"], min_score=HARD_GATE_MIN_SCORES["aph_kanamycin_pose_preservation"], score_formula="exp(-ln(2)*(pose_rmsd/limit)^2)", threshold_angstrom=pose_limit, pose_rmsd_angstrom=pose_value, ligand_atom_coverage=pose_atom_coverage, ligand_provenance=ligand_provenance, report_details=complex_measure), [] if pose_pass else ["comparable candidate KAN pose is unavailable or above threshold"], PLUGIN_NAME, pose_available),
            ScoreTerm("aph_mg_geometry_preservation", "task_specific", mg_score, _weight(context, "aph_mg_geometry_preservation", 0.0), _hard_details("aph_mg_geometry_not_preserved", ["mg_coordination"], min_score=HARD_GATE_MIN_SCORES["aph_mg_geometry_preservation"], score_formula="exp(-ln(2)*(max_distance_error/limit)^2)", threshold_angstrom=mg_limit, maximum_distance_error_angstrom=mg_value, ligand_provenance=ligand_provenance, report_details=complex_measure), [] if mg_pass else ["comparable candidate Mg geometry is unavailable or above threshold"], PLUGIN_NAME, mg_available),
            ScoreTerm("aph_clash_free", "task_specific", clash_score, _weight(context, "aph_clash_free", 0.0), _hard_details("aph_severe_clash_detected", ["global_observed_fold", "kanamycin_contact_shell", "mg_coordination"], min_score=HARD_GATE_MIN_SCORES["aph_clash_free"], apo_state_composition=apo_state_composition, state_composition_reason=apo_composition_reason, clash_excess=clash_excess, score_formula="1/(1+excess_clashes)", severe_clash_definition="nonbonded heavy-atom distance <1.6 A; same residue, adjacent peptide residues, and KAN-internal pairs excluded", severe_clash_count_max=clash_max, apo_clash_count=apo_clashes, complex_clash_count=complex_clashes, clash_count=max(apo_clashes, complex_clashes), report_details={"apo": apo_measure.get("severe_clashes"), "candidate_complex": complex_measure.get("severe_clashes")}), [] if clash_pass else ["apo/complex clash evidence is unavailable or exceeds the severe-clash budget"], PLUGIN_NAME, clash_available),
            ScoreTerm("aph_kanamycin_label_compatibility", "task_specific", label_score, _weight(context, "aph_kanamycin_label_compatibility", 0.0), {"dimension": "quality", "state": "preserve", "required": False, "qualitative_counts": label_counts, "active_tier": sequence["tier"], "DMS_labeled_positions_zero_based": sorted(SEED_POSITIONS), "frontier_positions_are_unlabeled": True, "exact_DMS_values_exposed": False, "combination_status": "unmeasured_combination", "claim_boundary": CLAIM_BOUNDARY, "ignored_for_score": _weight(context, "aph_kanamycin_label_compatibility", 0.0) == 0.0}, backend=PLUGIN_NAME, available=bool(candidate)),
            ScoreTerm("aph_sequence_plausibility", "task_specific", plausibility_score, _weight(context, "aph_sequence_plausibility", 1.0), {"dimension": "quality", "state": "preserve", "required": False, "mutation_count": mutation_count, "mutation_capacity": mutation_capacity, "seed_mutation_count": seed_mutation_count, "frontier_unlabeled_mutation_count": frontier_unlabeled_mutation_count, "mutation_burden_score": burden_score, "sequence_prior_loglik_avg": prior, "matched_WT_prior_reference": prior_reference, "sequence_prior_component_available": prior_score is not None, "sequence_prior_component": prior_score, "formula": "burden only until matched-WT calibration; then 0.75*sigmoid(delta/scale)+0.25*burden", "claim_boundary": CLAIM_BOUNDARY}, backend=PLUGIN_NAME, available=bool(candidate)),
            ScoreTerm("aph_repeat_robustness", "task_specific", repeat_score, _weight(context, "aph_repeat_robustness", 0.0), {"dimension": "quality", "state": "preserve", "required": _config_bool(context, "aph_repeat_state_required", False), "reference_role": "independent crystallographic repeat; not an A2 functional requirement", "worst_case_score": repeat_score if repeat_available else None, "seed_score_std": seed_std, "node_plddt_min": node_min, "report_details": repeat_measure, "claim_boundary": CLAIM_BOUNDARY, "ignored_for_score": not repeat_available}, warnings=[] if repeat_available else ["chain-B repeat audit evidence is unavailable"], backend=PLUGIN_NAME, available=repeat_available),
            ScoreTerm("aph_sequence_diversity", "task_specific", diversity_score, _weight(context, "aph_sequence_diversity", 0.0), {"dimension": "quality", "state": "positive", "required": False, "mutated_active_positions": mutation_count, "active_mutation_capacity": mutation_capacity, "seed_mutation_count": seed_mutation_count, "frontier_unlabeled_mutation_count": frontier_unlabeled_mutation_count, "formula": "mutated_active_positions/active_mutation_capacity", "does_not_imply_functional_diversity": True, "claim_boundary": CLAIM_BOUNDARY}, backend=PLUGIN_NAME, available=bool(candidate)),
        ]
        for term in terms:
            if term.name in gate_thresholds and isinstance(term.details, dict):
                term.details["hard_gate_min_score"] = gate_thresholds[term.name]
                term.details["hard_gate_calibration_mode"] = str(
                    (context.score_config.get("aph_hard_gate_calibration", {}) or {}).get(
                        "mode", "absolute"
                    )
                )
        return terms


TERM_NAMES = (
    "aph_sequence_integrity", "aph_mutation_scope_integrity", "aph_global_fold_preservation",
    "aph_verified_core_preservation", "aph_kanamycin_contact_preservation",
    "aph_kanamycin_pose_preservation", "aph_mg_geometry_preservation", "aph_clash_free",
    "aph_kanamycin_label_compatibility", "aph_sequence_plausibility",
    "aph_repeat_robustness", "aph_sequence_diversity",
)


def register_aph3iia_plugin() -> None:
    register_plugin(
        EvaluatorPluginSpec(
            name=PLUGIN_NAME,
            factory="aph3iia_evaluator:Aph3iiaKanamycinPlugin",
            weight_fields=TERM_NAMES,
            config_fields={
                "evaluation_mode": PluginConfigField("str", "aph_evaluation_mode"),
                "reference_chain_a_path": PluginConfigField("str", "aph_reference_chain_a_path"),
                "reference_chain_b_path": PluginConfigField("str", "aph_reference_chain_b_path"),
                "resolved_scientific_span_1based": PluginConfigField("sequence", "aph_resolved_scientific_span_1based"),
                "verified_core_positions_1based": PluginConfigField("sequence", "aph_verified_core_positions_1based"),
                "kanamycin_contact_positions_1based": PluginConfigField("sequence", "aph_kanamycin_contact_positions_1based"),
                "global_backbone_rmsd_max": PluginConfigField("float", "aph_global_backbone_rmsd_max"),
                "core_backbone_rmsd_max": PluginConfigField("float", "aph_core_backbone_rmsd_max"),
                "core_all_heavy_rmsd_max": PluginConfigField("float", "aph_core_all_heavy_rmsd_max"),
                "kanamycin_contact_cutoff": PluginConfigField("float", "aph_kanamycin_contact_cutoff"),
                "kanamycin_contact_coverage_min": PluginConfigField("float", "aph_kanamycin_contact_coverage_min"),
                "kanamycin_pose_rmsd_max": PluginConfigField("float", "aph_kanamycin_pose_rmsd_max"),
                "mg_distance_error_max": PluginConfigField("float", "aph_mg_distance_error_max"),
                "severe_clash_count_max": PluginConfigField("int", "aph_severe_clash_count_max"),
                "require_generated_candidate_ligand": PluginConfigField("bool", "aph_require_generated_candidate_ligand"),
                "allow_reference_heteroatom_transplant": PluginConfigField("bool", "aph_allow_reference_heteroatom_transplant"),
                "repeat_state_required": PluginConfigField("bool", "aph_repeat_state_required"),
                "hard_gate_calibration": PluginConfigField("mapping", "aph_hard_gate_calibration"),
            },
        ),
        replace=True,
    )


__all__ = [
    "Aph3iiaKanamycinPlugin", "KAN_CONTACT_SHELL", "LEVEL_ALLOWED", "OPEN_POSITIONS",
    "SEED_POSITIONS",
    "PLUGIN_NAME", "REFERENCE_CHAIN_A", "REFERENCE_CHAIN_B", "TERM_NAMES", "VERIFIED_CORE",
    "WT_SEQUENCE", "evaluate_candidate", "register_aph3iia_plugin", "structure_measurements",
]
