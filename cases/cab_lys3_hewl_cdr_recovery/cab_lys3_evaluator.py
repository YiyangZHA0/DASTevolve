

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

from astevolve.evaluation.contracts import EvaluatorContext, ScoreTerm
from astevolve.evaluation.plugins.registry import (
    EvaluatorPluginSpec,
    PluginConfigField,
    register_plugin,
)
from astevolve.evaluation.support import evaluator_weight
from cases.cab_lys3_hewl_cdr_recovery.cab_lys3_structure_evaluator import (
    structure_score_terms,
)
from cases.cab_lys3_hewl_cdr_recovery.sabdab_vhh_prior import (
    REGIONS as _SABDAB_REGIONS,
    score_sabdab_vhh_prior,
)
from cases.cab_lys3_hewl_cdr_recovery.cab_lys3_sequence_gate import (
    MIN_SABDAB_PLAUSIBILITY,
    sequence_quality_summary,
)


PLUGIN_NAME = "cab_lys3_cpu"
_PREFLIGHT_PLAUSIBLE_BINDER = (
    "DVQLQASGGGSVQAGGSLRLSCAASGYTIGPYCMGWFRQAPGKEREGVAAINMGGGITYYADSVKGRF"
    "TISQDNAKNTVYLLMNSLEPEDTAIYYCAADNTIYASYYECGHGLSTGGYGYDSWGQGTQVTVSS"
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


_L0_NATIVE = {101: "I", 104: "S", 105: "Y"}
_L1_NATIVE = {
    98: "D",
    99: "S",
    100: "T",
    102: "Y",
    103: "A",
    106: "Y",
    107: "E",
    109: "G",
    110: "H",
    111: "G",
    112: "L",
    113: "S",
    114: "T",
    115: "G",
    116: "G",
    117: "Y",
    118: "G",
    119: "Y",
    120: "D",
    121: "S",
}
_L2_NATIVE = {28: "I", 29: "G", 30: "P", 52: "M", 56: "I"}
_ALL_NATIVE = {**_L0_NATIVE, **_L1_NATIVE, **_L2_NATIVE}
_CDR_ENVELOPE_POSITIONS = tuple(
    [*range(25, 32), *range(52, 57), *range(98, 108), *range(109, 122)]
)
_ALL_A_ESCAPE_TARGET = 22
_DISULFIDE_CYSTEINES = (21, 32, 95, 108)
_EDITABLE_WINDOWS = ((25, 32), (52, 57), (98, 108), (109, 122))
_HYDROPHOBIC = frozenset("AILMFWVY")
_CHARGED_POSITIVE = frozenset("KR")
_CHARGED_NEGATIVE = frozenset("DE")


def _sequences(output: Mapping[str, Any]) -> Dict[str, str]:
    raw = output.get("seqs")
    if not isinstance(raw, Mapping):
        raw = output.get("best_seqs")
    if not isinstance(raw, Mapping):
        return {}
    return {str(chain): str(sequence) for chain, sequence in raw.items()}


def preflight_passing_sequences(
    template_sequences: Mapping[str, str],
) -> Dict[str, str]:
    sequences = {
        str(chain): str(sequence) for chain, sequence in template_sequences.items()
    }
    sequences["A"] = _PREFLIGHT_PLAUSIBLE_BINDER
    return sequences


def _recovery(sequence: str, oracle: Mapping[int, str]) -> tuple[int, float]:
    matches = sum(
        int(position < len(sequence) and sequence[position] == expected)
        for position, expected in oracle.items()
    )
    return matches, float(matches / len(oracle))


def _longest_run(values: Iterable[str], allowed: Optional[frozenset[str]] = None) -> int:
    best = current = 0
    previous: Optional[str] = None
    for value in values:
        matches = value in allowed if allowed is not None else value == previous
        current = current + 1 if matches else (1 if allowed is None else 0)
        best = max(best, current)
        previous = value
    return best


def _developability(sequence: str) -> Dict[str, Any]:
    windows = [sequence[start:end] for start, end in _EDITABLE_WINDOWS]
    hydrophobic_run = max((_longest_run(window, _HYDROPHOBIC) for window in windows), default=0)
    homopolymer_run = max((_longest_run(window) for window in windows), default=0)
    editable = "".join(windows)
    net_charge = sum(value in _CHARGED_POSITIVE for value in editable) - sum(
        value in _CHARGED_NEGATIVE for value in editable
    )
    charge_scale = max(4.0, 0.35 * len(editable))
    components = {
        "hydrophobic_run": max(0.0, 1.0 - 0.25 * max(0, hydrophobic_run - 2)),
        "homopolymer_run": max(0.0, 1.0 - 0.20 * max(0, homopolymer_run - 3)),
        "charge_balance": max(0.0, 1.0 - abs(net_charge) / charge_scale),
        "no_new_cysteine": 1.0 if "C" not in editable else 0.0,
    }
    return {
        "score": float(sum(components.values()) / len(components)),
        "components": components,
        "max_hydrophobic_run": hydrophobic_run,
        "max_homopolymer_run": homopolymer_run,
        "editable_net_charge": net_charge,
    }


def evaluate_candidate(
    out: Mapping[str, Any],
    *,
    compiled: Optional[Mapping[str, Any]] = None,
    design_state: Optional[Mapping[str, Any]] = None,
    masks: Optional[Mapping[str, Any]] = None,
    template_seqs: Optional[Mapping[str, str]] = None,
    fixed_residues: Optional[Mapping[str, Mapping[Any, str]]] = None,
    score_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:


    _ = compiled, design_state
    templates = {
        str(chain): str(sequence) for chain, sequence in (template_seqs or {}).items()
    }
    sequences = _sequences(out)
    raw_masks = masks if isinstance(masks, Mapping) else {}
    fixed = fixed_residues if isinstance(fixed_residues, Mapping) else {}
    score_cfg = score_config if isinstance(score_config, Mapping) else {}
    reasons: list[str] = []

    expected_chains = set(templates)
    observed_chains = set(sequences)
    for chain in sorted(expected_chains - observed_chains):
        reasons.append(f"missing_chain:{chain}")
    for chain in sorted(observed_chains - expected_chains):
        reasons.append(f"unexpected_chain:{chain}")

    for chain, template in sorted(templates.items()):
        sequence = sequences.get(chain, "")
        if len(sequence) != len(template):
            reasons.append(f"length_mismatch:{chain}")
            continue
        mask = raw_masks.get(chain)
        if mask is None or len(mask) != len(template):
            reasons.append(f"mask_missing_or_malformed:{chain}")
            continue
        for position, (parent, candidate) in enumerate(zip(template, sequence)):
            if candidate != parent and not bool(mask[position]):
                reasons.append(f"outside_compiled_mask:{chain}:{position}")

    for chain_id, assignments in sorted(fixed.items(), key=lambda item: str(item[0])):
        chain = str(chain_id)
        sequence = sequences.get(chain, "")
        if not isinstance(assignments, Mapping):
            continue
        for raw_position, expected in sorted(
            assignments.items(), key=lambda item: int(item[0])
        ):
            position = int(raw_position)
            actual = sequence[position] if 0 <= position < len(sequence) else None
            if actual != str(expected):
                reasons.append(f"fixed_residue_modified:{chain}:{position}")

    binder = sequences.get("A", "")
    for position in _DISULFIDE_CYSTEINES:
        if position >= len(binder) or binder[position] != "C":
            reasons.append(f"disulfide_cysteine_modified:A:{position}")

    scope = score_cfg.get("mutation_scope_contract", {})
    maximum = None
    if isinstance(scope, Mapping):
        try:
            maximum = int(scope.get("max_total_mutations"))
        except (TypeError, ValueError):
            maximum = None
    parent_a = templates.get("A", "")
    if binder and len(binder) == len(parent_a):
        mutation_count = sum(left != right for left, right in zip(parent_a, binder))
        if maximum is not None and maximum > 0 and mutation_count > maximum:
            reasons.append(f"mutation_budget_exceeded:A:{mutation_count}>{maximum}")
    else:
        mutation_count = 0

    l0_matches, l0_recovery = _recovery(binder, _L0_NATIVE)
    l1_matches, l1_recovery = _recovery(binder, _L1_NATIVE)
    l2_matches, l2_recovery = _recovery(binder, _L2_NATIVE)
    all_matches, all_recovery = _recovery(binder, _ALL_NATIVE)
    sequence_quality = sequence_quality_summary(binder)
    all_a_escape_summary = dict(sequence_quality["all_a_escape"])
    all_a_nonalanine_count = int(all_a_escape_summary["nonalanine_count"])
    all_a_escape = float(all_a_escape_summary["score"])
    mutation_sparsity = max(0.0, 1.0 - mutation_count / len(_ALL_NATIVE))
    developability = _developability(binder)
    sabdab_enabled = bool(score_cfg.get("sabdab_prior_enabled", False))
    sabdab_required = bool(score_cfg.get("sabdab_prior_required", False))
    sabdab_summary: Dict[str, Any] = {
        "available": False,
        "enabled": sabdab_enabled,
        "required": sabdab_required,
    }
    if sabdab_enabled:
        try:
            sabdab_summary = score_sabdab_vhh_prior(binder, score_cfg)
            sabdab_summary.update(
                {"enabled": True, "required": sabdab_required}
            )
        except Exception as exc:
            sabdab_summary.update(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "plausibility": 0.0,
                    "ood_penalty": 1.0,
                    "regions": {},
                }
            )
    reasons.extend(sequence_quality["hard_failures"])
    if sabdab_enabled:
        sabdab_plausibility = float(sabdab_summary.get("plausibility", 0.0))
        if not bool(sabdab_summary.get("available")):
            reasons.append("sabdab_plausibility_unavailable")
        elif sabdab_plausibility < MIN_SABDAB_PLAUSIBILITY:
            reasons.append(
                "sabdab_plausibility_below_floor:"
                f"{sabdab_plausibility:.4f}<{MIN_SABDAB_PLAUSIBILITY:.4f}"
            )
    reasons = sorted(set(reasons))
    passed = not reasons
    normalized = all_recovery if passed else 0.0

    def term(
        *,
        name: str,
        state: str,
        category: str,
        score: float,
        weight_key: str,
        default_weight: float,
        details: Mapping[str, Any],
        warnings: list[str] | None = None,
    ) -> Dict[str, Any]:
        return {
            "provider": PLUGIN_NAME,
            "state": state,
            "name": name,
            "category": category,
            "score": score,
            "weight": evaluator_weight(score_cfg, weight_key, default_weight),
            "available": True,
            "details": dict(details),
            "warnings": list(warnings or []),
        }

    oracle_warning = ["Oracle sequence recovery is not an affinity prediction."]
    terms = [
        term(
            name="l0_native_recovery",
            state="positive",
            category="staged_recovery_l0",
            score=l0_recovery,
            weight_key="eval_l0_native_recovery",
            default_weight=0.0,
            details={
                "matches": l0_matches,
                "total": len(_L0_NATIVE),
                "positions_zero_based": sorted(_L0_NATIVE),
                "native_identities_withheld": True,
                "stage": "L0",
            },
            warnings=oracle_warning,
        ),
        term(
            name="l1_native_recovery",
            state="positive",
            category="staged_recovery_l1",
            score=l1_recovery,
            weight_key="eval_l1_native_recovery",
            default_weight=0.0,
            details={
                "matches": l1_matches,
                "total": len(_L1_NATIVE),
                "positions_zero_based": sorted(_L1_NATIVE),
                "native_identities_withheld": True,
                "stage": "L1",
            },
            warnings=oracle_warning,
        ),
        term(
            name="l2_native_recovery",
            state="positive",
            category="staged_recovery_l2",
            score=l2_recovery,
            weight_key="eval_l2_native_recovery",
            default_weight=0.0,
            details={
                "matches": l2_matches,
                "total": len(_L2_NATIVE),
                "positions_zero_based": sorted(_L2_NATIVE),
                "native_identities_withheld": True,
                "stage": "L2",
            },
            warnings=oracle_warning,
        ),
        term(
            name="all_nodes_native_recovery",
            state="positive",
            category="recovery_calibration",
            score=all_recovery,
            weight_key="eval_all_nodes_native_recovery",
            default_weight=1.0,
            details={
                "matches": all_matches,
                "total": len(_ALL_NATIVE),
                "positions_zero_based": sorted(_ALL_NATIVE),
                "native_identities_withheld": True,
                "aggregation": "L0+L1+L2",
            },
            warnings=oracle_warning,
        ),
        term(
            name="cdr_all_a_escape",
            state="positive",
            category="search_bootstrap_diversification",
            score=all_a_escape,
            weight_key="eval_cdr_all_a_escape",
            default_weight=0.0,
            details={
                **all_a_escape_summary,
                "all_a_position_count": len(_CDR_ENVELOPE_POSITIONS),
                "transform": "nonalanine_diversity_partition_geometric_mean",
                "hard_gate": True,
            },
            warnings=[
                "All-A escape is a search bootstrap objective, not an affinity predictor."
            ],
        ),
        term(
            name="mutation_sparsity",
            state="preserve",
            category="search_regularization",
            score=mutation_sparsity,
            weight_key="eval_mutation_sparsity",
            default_weight=0.0,
            details={
                "mutation_count": mutation_count,
                "editable_position_count": len(_ALL_NATIVE),
            },
        ),
        term(
            name="fixed_residue_integrity",
            state="preserve",
            category="hard_constraint",
            score=1.0 if passed else 0.0,
            weight_key="eval_fixed_residue_integrity",
            default_weight=0.0,
            details={"violations": reasons, "mutation_count": mutation_count},
        ),
        term(
            name="developability_score",
            state="preserve",
            category="sequence_developability",
            score=float(developability["score"]),
            weight_key="eval_developability_score",
            default_weight=0.0,
            details=developability,
            warnings=["Developability is a sequence heuristic, not an experimental assay."],
        ),
    ]
    if sabdab_enabled:
        sabdab_available = bool(sabdab_summary.get("available"))
        sabdab_error = str(sabdab_summary.get("error") or "")
        sabdab_warnings = [
            "SAbDab/ESM2 is an empirical VHH plausibility prior, not an affinity predictor."
        ]
        if sabdab_error:
            sabdab_warnings.append(sabdab_error)
        terms.append(
            term(
                name="sabdab_vhh_plausibility",
                state="preserve",
                category="sequence_empirical_prior",
                score=float(sabdab_summary.get("plausibility", 0.0)),
                weight_key="eval_sabdab_vhh_plausibility",
                default_weight=0.0,
                details={
                    "dimension": "sequence_plausibility",
                    "required": sabdab_required,
                    "hard_gate": True,
                    "hard_gate_min_score": MIN_SABDAB_PLAUSIBILITY,
                    "ood_penalty": float(sabdab_summary.get("ood_penalty", 1.0)),
                    "calibration": "official antigen-aware heldout VHH lower tail",
                    "penalty_policy": "zero above q05; linear to one at q01",
                    "top_k": sabdab_summary.get("top_k"),
                    "device": sabdab_summary.get("device"),
                    "index_dir": sabdab_summary.get("index_dir"),
                    "cache_hit": bool(sabdab_summary.get("cache_hit", False)),
                    "error": sabdab_error or None,
                },
                warnings=sabdab_warnings,
            )
        )
        terms[-1]["available"] = sabdab_available
        for region in _SABDAB_REGIONS:
            region_summary = _mapping(
                _mapping(sabdab_summary.get("regions")).get(region)
            )
            terms.append(
                term(
                    name=f"sabdab_{region}_similarity",
                    state="preserve",
                    category="sequence_empirical_prior_region",
                    score=float(
                        region_summary.get(
                            "cluster_balanced_top_k_mean_cosine",
                            0.0,
                        )
                    ),
                    weight_key=f"eval_sabdab_{region}_similarity",
                    default_weight=0.0,
                    details={
                        "dimension": "sequence_plausibility",
                        "region": region,
                        "heldout_percentile": region_summary.get(
                            "heldout_percentile"
                        ),
                        "ood_penalty": region_summary.get("ood_penalty"),
                        "plausibility": region_summary.get("plausibility"),
                        "q01": region_summary.get("q01"),
                        "q05": region_summary.get("q05"),
                        "nearest": list(region_summary.get("nearest") or []),
                        "reported_only": True,
                    },
                    warnings=sabdab_warnings[:1],
                )
            )
            terms[-1]["available"] = sabdab_available
    structure_terms, structure_summary = structure_score_terms(
        out,
        score_cfg,
        provider_name=PLUGIN_NAME,
    )
    terms.extend(structure_terms)

    l0_unlock = float(score_cfg.get("staged_l0_unlock_recovery", 1.0 / 3.0))
    l1_unlock = float(score_cfg.get("staged_l1_unlock_recovery", 0.15))
    if l0_recovery < l0_unlock:
        active_stage = "L0"
        iptm_floor = float(score_cfg.get("staged_l0_af3_iptm_floor", 0.10))
    elif l1_recovery < l1_unlock:
        active_stage = "L1"
        iptm_floor = float(score_cfg.get("staged_l1_af3_iptm_floor", 0.18))
    else:
        active_stage = "L2"
        iptm_floor = float(score_cfg.get("staged_l2_af3_iptm_floor", 0.25))
    af3_quality = _mapping(_mapping(structure_summary.get("providers")).get("alphafold3"))
    af3_components = _mapping(af3_quality.get("components"))
    af3_iptm = af3_components.get("interface_confidence")
    af3_iptm_value = float(af3_iptm) if isinstance(af3_iptm, (int, float)) else 0.0
    stage_gate_available = bool(af3_quality.get("available"))
    stage_gate_required = bool(score_cfg.get("staged_interface_gate_required", False))
    stage_gate_score = min(1.0, af3_iptm_value / iptm_floor) if iptm_floor > 0.0 else 1.0
    stage_gate_pass = bool(stage_gate_available and stage_gate_score >= 1.0)
    terms.append(
        {
            "provider": PLUGIN_NAME,
            "state": "preserve",
            "name": "af3_stage_iptm_gate",
            "category": "staged_interface_gate",
            "score": stage_gate_score,
            "weight": evaluator_weight(score_cfg, "eval_af3_stage_iptm_gate", 0.0),
            "available": stage_gate_available,
            "details": {
                "dimension": "interface_correctness",
                "required": stage_gate_required,
                "hard_gate": stage_gate_required,
                "hard_gate_min_score": 1.0,
                "active_stage": active_stage,
                "af3_iptm": af3_iptm_value,
                "af3_iptm_floor": iptm_floor,
                "l0_unlock_recovery": l0_unlock,
                "l1_unlock_recovery": l1_unlock,
            },
            "warnings": ([] if stage_gate_pass else [f"{active_stage} AF3 ipTM floor not met"]),
        }
    )
    unavailable_required = [
        str(item["name"])
        for item in terms
        if bool((item.get("details") or {}).get("required"))
        and not bool(item.get("available"))
    ]
    stage_gate_reasons = []
    if stage_gate_required and stage_gate_available and not stage_gate_pass:
        stage_gate_reasons.append(
            f"staged_af3_iptm_below_floor:{active_stage}:{af3_iptm_value:.4f}<{iptm_floor:.4f}"
        )
    direct_reasons = sorted(
        set(
            reasons
            + [f"required_evidence_unavailable:{name}" for name in unavailable_required]
            + stage_gate_reasons
        )
    )
    weighted_terms = [item for item in terms if float(item.get("weight", 0.0)) > 0.0]
    total_weight = sum(float(item.get("weight", 0.0)) for item in weighted_terms)
    soft_score = (
        sum(
            float(item.get("score", 0.0)) * float(item.get("weight", 0.0))
            for item in weighted_terms
            if bool(item.get("available"))
        )
        / total_weight
        if total_weight > 0.0
        else 0.0
    )
    direct_pass = not direct_reasons
    normalized = soft_score if direct_pass else 0.0
    return {
        "schema_version": "ast_evaluator_report_v1",
        "normalized_score": normalized,
        "soft_score": soft_score,
        "loss": 1.0 - normalized,
        "hard_gate_pass": direct_pass,
        "disqualification_reasons": direct_reasons,
        "gate_status": {
            "passed": direct_pass,
            "hard_gate_pass": direct_pass,
            "hard_failures": direct_reasons,
            "disqualification_reasons": direct_reasons,
        },
        "terms": terms,
        "weakest_terms": sorted(terms, key=lambda item: float(item["score"])),
        "stage_summary": {
            "L0": {
                "matches": l0_matches,
                "total": len(_L0_NATIVE),
                "score": l0_recovery,
            },
            "L1": {
                "matches": l1_matches,
                "total": len(_L1_NATIVE),
                "score": l1_recovery,
            },
            "L2": {
                "matches": l2_matches,
                "total": len(_L2_NATIVE),
                "score": l2_recovery,
            },
            "all": {
                "matches": all_matches,
                "total": len(_ALL_NATIVE),
                "score": all_recovery,
            },
            "all_a_escape": {
                **all_a_escape_summary,
            },
            "sequence_quality": sequence_quality,
        },
        "dual_structure_summary": structure_summary,
        "stage_control": {
            "active_stage": active_stage,
            "gate_required": stage_gate_required,
            "gate_available": stage_gate_available,
            "gate_pass": stage_gate_pass,
            "af3_iptm": af3_iptm_value,
            "af3_iptm_floor": iptm_floor,
            "l0_unlock_recovery": l0_unlock,
            "l1_unlock_recovery": l1_unlock,
        },
        "developability": developability,
        "sabdab_vhh_prior": sabdab_summary,
        "warnings": [
            "Sequence recovery is a hidden-answer calibration; AF3/Protenix "
            "terms are structure-confidence proxies, not affinity validation."
        ],
    }


class CabLys3CPUPlugin:


    name = PLUGIN_NAME

    def score_terms(self, context: EvaluatorContext) -> list[ScoreTerm]:
        report = evaluate_candidate(
            context.out,
            compiled=context.compiled,
            design_state=context.design_state,
            masks=context.masks,
            template_seqs=context.template_seqs,
            fixed_residues=context.fixed_residues,
            score_config=context.score_config,
        )
        terms: list[ScoreTerm] = []
        for raw in report["terms"]:
            details = dict(raw.get("details") or {})
            details.setdefault("state", str(raw.get("state") or ""))
            if raw["name"] == "fixed_residue_integrity":
                details.update(
                    {
                        "hard_gate": True,
                        "hard_gate_min_score": 1.0,
                        "dimension": "sequence_integrity",
                    }
                )
            elif raw["name"] == "mutation_sparsity":
                details.setdefault("dimension", "robustness")
            else:
                details.setdefault("dimension", "recovery")
            terms.append(
                ScoreTerm(
                    name=str(raw["name"]),
                    category=str(raw["category"]),
                    score=float(raw["score"]),
                    weight=float(raw.get("weight", 1.0)),
                    details=details,
                    warnings=[str(item) for item in raw.get("warnings", [])],
                    backend=self.name,
                    available=bool(raw.get("available", True)),
                )
            )
        return terms


def register_cab_lys3_plugin() -> None:
    register_plugin(
        EvaluatorPluginSpec(
            name=PLUGIN_NAME,
            factory="cases.cab_lys3_hewl_cdr_recovery.cab_lys3_evaluator:CabLys3CPUPlugin",
            config_fields={
                "dual_structure_required": PluginConfigField(
                    "bool", "dual_structure_required"
                ),
                "dual_structure_af3_weight": PluginConfigField(
                    "float", "dual_structure_af3_weight"
                ),
                "dual_structure_disagreement_weight": PluginConfigField(
                    "float", "dual_structure_disagreement_weight"
                ),
                "dual_structure_worst_case_weight": PluginConfigField(
                    "float", "dual_structure_worst_case_weight"
                ),
                "staged_interface_gate_required": PluginConfigField(
                    "bool", "staged_interface_gate_required"
                ),
                "staged_l0_af3_iptm_floor": PluginConfigField(
                    "float", "staged_l0_af3_iptm_floor"
                ),
                "staged_l1_af3_iptm_floor": PluginConfigField(
                    "float", "staged_l1_af3_iptm_floor"
                ),
                "staged_l2_af3_iptm_floor": PluginConfigField(
                    "float", "staged_l2_af3_iptm_floor"
                ),
                "staged_l0_unlock_recovery": PluginConfigField(
                    "float", "staged_l0_unlock_recovery"
                ),
                "staged_l1_unlock_recovery": PluginConfigField(
                    "float", "staged_l1_unlock_recovery"
                ),
                "sabdab_prior_enabled": PluginConfigField(
                    "bool", "sabdab_prior_enabled"
                ),
                "sabdab_prior_required": PluginConfigField(
                    "bool", "sabdab_prior_required"
                ),
                "sabdab_prior_index_dir": PluginConfigField(
                    "str", "sabdab_prior_index_dir"
                ),
                "sabdab_prior_model_dir": PluginConfigField(
                    "str", "sabdab_prior_model_dir"
                ),
                "sabdab_prior_device": PluginConfigField(
                    "str", "sabdab_prior_device"
                ),
                "sabdab_prior_top_k": PluginConfigField(
                    "int", "sabdab_prior_top_k"
                ),
            },
            weight_fields=(
                "eval_l0_native_recovery",
                "eval_l1_native_recovery",
                "eval_l2_native_recovery",
                "eval_all_nodes_native_recovery",
                "eval_cdr_all_a_escape",
                "eval_mutation_sparsity",
                "eval_fixed_residue_integrity",
                "eval_developability_score",
                "eval_sabdab_vhh_plausibility",
                "eval_sabdab_whole_vhh_similarity",
                "eval_sabdab_framework_similarity",
                "eval_sabdab_cdr1_similarity",
                "eval_sabdab_cdr2_similarity",
                "eval_sabdab_cdr3_similarity",
                "eval_af3_stage_iptm_gate",
                "eval_af3_binder_confidence",
                "eval_af3_design_region_confidence",
                "eval_af3_interface_confidence",
                "eval_af3_interface_plddt",
                "eval_af3_interface_pae_quality",
                "eval_af3_contact_quality",
                "eval_af3_structure_quality",
                "eval_protenix_binder_confidence",
                "eval_protenix_design_region_confidence",
                "eval_protenix_interface_confidence",
                "eval_protenix_interface_plddt",
                "eval_protenix_interface_pae_quality",
                "eval_protenix_contact_quality",
                "eval_protenix_structure_quality",
                "eval_dual_structure_consensus",
                "eval_dual_structure_worst_case",
                "eval_dual_structure_agreement",
                "eval_dual_structure_score",
            ),
        ),
        replace=True,
    )


register_cab_lys3_plugin()


__all__ = [
    "CabLys3CPUPlugin",
    "evaluate_candidate",
    "preflight_passing_sequences",
    "register_cab_lys3_plugin",
]
