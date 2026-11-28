"""Case-local blind evaluator for the TtgR apo+naltrexone GPU smoke."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

from astevolve.evaluation.contracts import EvaluatorContext, ScoreTerm
from astevolve.evaluation.plugins.registry import EvaluatorPluginSpec, register_plugin


PLUGIN_NAME = "ttgr_blind"
WT = "MVRRTKEEAQETRAQIIEAAERAFYKRGVARTTLADIAELAGVTRGAIYWHFNNKAELVQALLDSLHETHDHLARASESEDEVDPLGCMRKLLLQVFNELVLDARTRRINEILHHKCEFTDDMCEIRQQRQSAVLDCHKGITLALANAVRRGQLPGELDAERAAVAMFAYVDGLIRRWLLLPDSVDLLGDVEKWVDTGLDMLRLSPALRK"
ALLOWED = {
    66: set("EHKT"), 69: set("FHL"), 73: set("AILV"), 77: set("EQ"),
    88: set("FILMV"), 91: set("AILMTV"), 92: set("ILW"), 95: set("FILV"),
    109: set("FMN"), 112: set("AFL"), 113: set("FHY"),
}


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number)) if math.isfinite(number) else 0.0


def _weight(context: EvaluatorContext, name: str, default: float) -> float:
    values = context.score_config.get("evaluator_weights", {})
    try:
        return float(values.get(name, default)) if isinstance(values, Mapping) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _states(structure: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(item.get("name")): item for item in structure.get("states", []) or [] if isinstance(item, Mapping) and item.get("name")}


def _plddt(state: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not isinstance(state, Mapping):
        return None
    summary = state.get("structure_metrics", {})
    scalar = summary.get("scalar", {}) if isinstance(summary, Mapping) else {}
    value = scalar.get("plddt") if isinstance(scalar, Mapping) else None
    if value is None:
        confidence = state.get("confidence_metrics", {})
        value = confidence.get("plddt") if isinstance(confidence, Mapping) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _objective(structure: Mapping[str, Any], name: str) -> Tuple[float, bool, Dict[str, Any]]:
    pack = structure.get("multistate_objectives", {})
    objectives = pack.get("objectives", {}) if isinstance(pack, Mapping) else {}
    item = objectives.get(name, {}) if isinstance(objectives, Mapping) else {}
    if not isinstance(item, Mapping) or not item:
        return 0.0, False, {}
    details = item.get("details", {})
    available = not bool(item.get("warnings"))
    return _clamp01(item.get("score")), available, dict(details) if isinstance(details, Mapping) else {}


def _candidate(context: EvaluatorContext) -> str:
    sequences = context.out.get("seqs")
    if not isinstance(sequences, Mapping):
        sequences = context.out.get("best_seqs")
    return str(sequences.get("T") or "") if isinstance(sequences, Mapping) else ""


class TTGRBlindPlugin:
    name = PLUGIN_NAME

    def score_terms(self, context: EvaluatorContext) -> list[ScoreTerm]:
        candidate = _candidate(context)
        scope = context.score_config.get("mutation_scope_contract", {})
        active_by_chain = scope.get("active_positions_by_chain", {}) if isinstance(scope, Mapping) else {}
        active = {int(item) for item in active_by_chain.get("T", [])} if isinstance(active_by_chain, Mapping) else set()
        changes = [index for index, (before, after) in enumerate(zip(WT, candidate)) if before != after] if len(candidate) == len(WT) else []
        alphabet_violations = [
            {"position": index, "residue": candidate[index], "allowed": sorted(ALLOWED.get(index, set()))}
            for index in changes
            if index not in ALLOWED or candidate[index] not in ALLOWED[index]
        ]
        sequence_pass = bool(candidate) and len(candidate) == len(WT) and set(changes) <= active and len(changes) <= 4 and not alphabet_violations

        states = _states(context.structure)
        apo = states.get("candidate_apo")
        nal = states.get("candidate_plus_naltrexone")
        apo_plddt = _plddt(apo)
        nal_plddt = _plddt(nal)
        state_complete = all(
            state is not None and _plddt(state) is not None and bool(state.get("cif_path") or state.get("structure_path"))
            for state in (apo, nal)
        )
        fold_score = min(_clamp01((apo_plddt or 0.0) / 100.0), _clamp01((nal_plddt or 0.0) / 100.0))
        target_score, target_available, target_details = _objective(context.structure, "naltrexone_primary_pocket_proxy")
        apo_dimer, apo_dimer_available, apo_dimer_details = _objective(context.structure, "apo_dimer_interface")
        nal_dimer, nal_dimer_available, nal_dimer_details = _objective(context.structure, "naltrexone_dimer_interface")
        dimer_available = apo_dimer_available and nal_dimer_available
        dimer_score = min(apo_dimer, nal_dimer) if dimer_available else 0.0
        transition_score, transition_available, transition_details = _objective(context.structure, "apo_to_naltrexone_allosteric_geometry")
        burden_score = 1.0 - _clamp01(len(changes) / 4.0)

        return [
            ScoreTerm("ttgr_state_evidence_complete", "task_correctness", 1.0 if state_complete else 0.0, _weight(context, "eval_ttgr_state_evidence", 0.0), {"state": "preserve", "required": True, "hard_gate": True, "hard_gate_min_score": 1.0, "hard_gate_reason": "ttgr_apo_nal_structure_evidence_missing", "apo_plddt": apo_plddt, "nal_plddt": nal_plddt}, [] if state_complete else ["apo+naltrexone structure evidence is incomplete"], PLUGIN_NAME, state_complete),
            ScoreTerm("ttgr_sequence_contract", "semantic_constraint", 1.0 if sequence_pass else 0.0, _weight(context, "eval_ttgr_sequence_contract", 0.0), {"state": "preserve", "required": True, "hard_gate": True, "hard_gate_min_score": 1.0, "hard_gate_reason": "ttgr_sequence_contract_failed", "changes_zero_based": changes, "active_positions_zero_based": sorted(active), "alphabet_violations": alphabet_violations, "max_mutations": 4, "generator_hard_alphabet_enforcement": True, "evaluator_hard_alphabet_enforcement": True}, [] if sequence_pass else ["candidate violates active scope, mutation budget, length, or site alphabet"], PLUGIN_NAME, bool(candidate)),
            ScoreTerm("ttgr_fold_guardrail", "task_specific", fold_score, _weight(context, "eval_ttgr_fold_guardrail", 0.20), {"state": "preserve", "required": True, "hard_gate": True, "hard_gate_min_score": 0.45, "hard_gate_reason": "ttgr_fold_confidence_smoke_floor_failed", "apo_plddt": apo_plddt, "nal_plddt": nal_plddt, "calibration_status": "uncalibrated_smoke_floor"}, [], PLUGIN_NAME, apo_plddt is not None and nal_plddt is not None),
            ScoreTerm("ttgr_dimer_guardrail", "task_specific", dimer_score, _weight(context, "eval_ttgr_dimer_guardrail", 0.15), {"state": "preserve", "required": True, "diagnostic_smoke": True, "apo": apo_dimer_details, "naltrexone": nal_dimer_details}, [] if dimer_available else ["dimer interface proxy unavailable"], PLUGIN_NAME, dimer_available),
            ScoreTerm("ttgr_target_pocket_proxy", "task_specific", target_score, _weight(context, "eval_ttgr_target_pocket_proxy", 0.35), {"state": "positive", "required": True, "claim_scope": "static contact/confidence proxy only", "report_details": target_details}, [] if target_available else ["target pocket proxy unavailable"], PLUGIN_NAME, target_available),
            ScoreTerm("ttgr_counter_relative_margin_proxy", "task_specific", 0.0, _weight(context, "eval_ttgr_counter_relative_margin_proxy", 0.0), {"state": "negative", "required": False, "enabled": False, "ignored_for_score": True, "reason": "End/Tam states are staged after the apo+naltrexone smoke"}, ["counter-ligand margin is not active in this smoke"], PLUGIN_NAME, False),
            ScoreTerm("ttgr_allosteric_transition_proxy", "task_specific", transition_score, _weight(context, "eval_ttgr_allosteric_transition_proxy", 0.0), {"state": "positive", "required": False, "diagnostic_only": True, "claim_scope": "uncalibrated apo-to-holo geometric movement/stability audit; not reporter allostery and not used for candidate selection", "report_details": transition_details}, [] if transition_available else ["allosteric geometry proxy unavailable"], PLUGIN_NAME, transition_available),
            ScoreTerm("ttgr_mutation_burden", "regularization", burden_score, _weight(context, "eval_ttgr_mutation_burden", 0.10), {"state": "preserve", "mutation_count": len(changes), "maximum": 4}, [], PLUGIN_NAME, bool(candidate))
        ]


def register_ttgr_plugin() -> None:
    register_plugin(
        EvaluatorPluginSpec(
            name=PLUGIN_NAME,
            factory="cases.ttgr_naltrexone_l1_blind.ttgr_evaluator:TTGRBlindPlugin",
            weight_fields=(
                "eval_ttgr_state_evidence",
                "eval_ttgr_sequence_contract",
                "eval_ttgr_fold_guardrail",
                "eval_ttgr_dimer_guardrail",
                "eval_ttgr_target_pocket_proxy",
                "eval_ttgr_counter_relative_margin_proxy",
                "eval_ttgr_allosteric_transition_proxy",
                "eval_ttgr_mutation_burden"
            )
        ),
        replace=True
    )


__all__ = ["TTGRBlindPlugin", "register_ttgr_plugin"]
