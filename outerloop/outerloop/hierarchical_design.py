

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence

from outerloop.reasoning_audit import (
    ReasoningAuditLedger,
    build_failed_reasoning_observation,
    build_reasoning_observation,
)
from outerloop.island_capabilities import (
    requires_outside_incumbent_exploration,
)
from outerloop.island_runtime import compile_executable_island_directives


EVIDENCE_BUNDLE_VERSION = "astevolve.design_evidence_bundle.v1"
GLOBAL_STRATEGY_VERSION = "astevolve.global_protein_strategy.v1"
HIERARCHICAL_SNAPSHOT_VERSION = "astevolve.hierarchical_design_snapshot.v1"
HYPOTHESIS_LEDGER_VERSION = "astevolve.hypothesis_ledger.v1"


class HierarchicalDesignError(ValueError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _closed(value: Any, fields: set[str], label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HierarchicalDesignError(f"{label} must be a JSON object")
    actual = {str(key) for key in value}
    if actual != fields:
        raise HierarchicalDesignError(
            f"{label} must contain exactly {sorted(fields)}; "
            f"missing={sorted(fields - actual)}, unknown={sorted(actual - fields)}"
        )
    return dict(value)


def _text(value: Any, label: str, maximum: int = 2000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HierarchicalDesignError(f"{label} must be a non-empty string")
    output = value.strip()
    if len(output) > maximum:
        raise HierarchicalDesignError(f"{label} exceeds {maximum} characters")
    return output


def _strings(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = 16,
) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HierarchicalDesignError(f"{label} must be a JSON list")
    output = [_text(item, f"{label}[{index}]", 256) for index, item in enumerate(value)]
    if not minimum <= len(output) <= maximum:
        raise HierarchicalDesignError(
            f"{label} must contain {minimum} through {maximum} entries"
        )
    if len(output) != len(set(output)):
        raise HierarchicalDesignError(f"{label} contains duplicates")
    return output


def _safe_payload(value: Any, limit: int) -> Dict[str, Any]:


    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {"text": value}
    if not isinstance(value, Mapping):
        value = {"value": value}
    detached = deepcopy(dict(value))
    encoded = _canonical_json(detached).encode("utf-8")
    if len(encoded) <= limit:
        return detached
    return {
        "omitted": True,
        "reason": "controller_prompt_budget",
        "raw_size": len(encoded),
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _migration_frontier_projection(value: Any) -> Dict[str, Any]:


    raw = _safe_payload(value, 1 << 30)
    if raw.get("schema_version") != "astevolve.migration_frontier.v1":
        return raw
    source_indexing = str(raw.get("indexing") or "zero_based")

    def public_position(value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            return value
        return value + 1 if source_indexing == "zero_based" else value

    segments = []
    for item in raw.get("segments", []):
        if not isinstance(item, Mapping):
            continue
        segments.append(
            {
                "chain_id": item.get("chain_id"),
                "compiled_segment": item.get("compiled_segment"),
                "incumbent": item.get("incumbent"),
                "incumbent_node_ids": list(item.get("incumbent_node_ids") or []),
                "legal_position_count": item.get("legal_position_count"),
                "positions": [
                    {
                        "position": public_position(row.get("position")),
                        "evidence_id": row.get("evidence_id"),
                        "safety_score": row.get("safety_score"),
                        "incumbent_node_id": row.get("incumbent_node_id"),
                        "hard_allowed_residues": list(
                            row.get("hard_allowed_residues") or []
                        ),
                        "case_owned_residue_tier": row.get(
                            "case_owned_residue_tier"
                        ),
                    }
                    for row in item.get("positions", [])
                    if isinstance(row, Mapping)
                ],
            }
        )
    position_map: Dict[str, Any] = {}
    for reference, segment in dict(
        raw.get("compiled_segment_by_position") or {}
    ).items():
        fields = str(reference).rsplit(":", 1)
        if len(fields) != 2:
            continue
        try:
            position = int(fields[1])
        except ValueError:
            continue
        position_map[f"{fields[0]}:{public_position(position)}"] = segment

    return {
        "schema_version": raw["schema_version"],
        "indexing": "one_based",
        "source_indexing": source_indexing,
        "constraints": deepcopy(raw.get("constraints")),
        "incumbent_segments": list(raw.get("incumbent_segments") or []),
        "ranked_new_segments": list(raw.get("ranked_new_segments") or []),
        "segments": segments,
        "compiled_segment_by_position": position_map,
    }


def build_evidence_bundle(database: Any, *, max_chars: int = 32768) -> Dict[str, Any]:


    summary = database.get_global_summary(top_n=12)
    records: list[Dict[str, Any]] = []
    selected = summary.get("global_selection", []) if isinstance(summary, Mapping) else []
    per_record = max(1024, int(max_chars) // max(4, len(selected) + 4))
    for entry in selected:
        if not isinstance(entry, Mapping):
            continue
        program_id = str(entry.get("program_id") or "").strip()
        if not program_id:
            continue
        program = database.get(program_id) if hasattr(database, "get") else None
        metrics = getattr(program, "metrics", {}) if program is not None else {}
        payload = {
            "global_entry": deepcopy(dict(entry)),
            "measured_metrics": deepcopy(dict(metrics or {})),
        }
        records.append(
            {
                "evidence_id": f"program:{program_id}:measurements",
                "kind": "measured_candidate",
                "provenance_class": "computed_fact",
                "source_program_id": program_id,
                "payload": _safe_payload(payload, per_record),
                "payload_hash": _hash(payload),
            }
        )
        artifacts = database.get_artifacts(program_id) if hasattr(database, "get_artifacts") else {}
        if isinstance(artifacts, Mapping):
            for key in (
                "objective_vector",
                "migration_frontier",
                "ast_revision_report",
                "residue_evidence_context",
                "structure_evaluation_summary",
            ):
                if key not in artifacts:
                    continue
                raw_payload = (
                    _migration_frontier_projection(artifacts[key])
                    if key == "migration_frontier"
                    else artifacts[key]
                )
                artifact_payload = _safe_payload(
                    raw_payload,
                    max(per_record, 12 * 1024)
                    if key == "migration_frontier"
                    else per_record,
                )
                records.append(
                    {
                        "evidence_id": f"program:{program_id}:{key}",
                        "kind": key,
                        "provenance_class": (
                            "llm_proposal"
                            if key == "ast_revision_report"
                            else "accepted_observation"
                            if key == "structure_evaluation_summary"
                            else "computed_fact"
                        ),
                        "source_program_id": program_id,
                        "payload": artifact_payload,
                        "payload_hash": _hash(raw_payload),
                    }
                )
    for portfolio in summary.get("role_portfolios", []) if isinstance(summary, Mapping) else []:
        if not isinstance(portfolio, Mapping):
            continue
        island = int(portfolio.get("island", 0) or 0)
        records.append(
            {
                "evidence_id": f"island:{island}:portfolio",
                "kind": "island_role_portfolio",
                "provenance_class": "computed_fact",
                "source_program_id": "",
                "payload": _safe_payload(portfolio, per_record),
                "payload_hash": _hash(portfolio),
            }
        )
    if not records:
        records.append(
            {
                "evidence_id": "controller:no_measured_evidence",
                "kind": "no_measured_evidence",
                "provenance_class": "computed_fact",
                "source_program_id": "",
                "payload": {"available": False},
                "payload_hash": _hash({"available": False}),
            }
        )
    material = {
        "schema_version": EVIDENCE_BUNDLE_VERSION,
        "iteration": int(getattr(database, "last_iteration", 0) or 0),
        "records": records,
        "global_summary_hash": _hash(summary),
    }
    return {**material, "bundle_hash": _hash(material)}


_STRATEGY_FIELDS = {
    "schema_version",
    "strategy_epoch_id",
    "evidence_bundle_hash",
    "global_assessment",
    "global_constraints",
    "island_directives",
    "cross_island_synthesis",
    "final_selection",
    "stagnation_response",
}
_DIRECTIVE_FIELDS = {
    "island",
    "role_id",
    "objective",
    "priority_metrics",
    "architecture_targets",
    "avoid_patterns",
    "evidence_refs",
    "outside_incumbent_quota",
    "required_ablation",
}
_SYNTHESIS_FIELDS = {
    "migration_pairs",
    "preserve_discoveries",
    "conflict_resolution",
}
_SELECTION_FIELDS = {
    "hard_gate_metrics",
    "ranking_metrics",
    "robustness_metrics",
    "selection_rationale",
}


def validate_global_strategy(
    value: Any,
    *,
    epoch_id: str,
    evidence_bundle: Mapping[str, Any],
    island_roles: Sequence[Mapping[str, Any]],
    minimum_region_outside_quota: float = 0.0,
    required_hard_gate_metrics: Sequence[str] = (),
) -> Dict[str, Any]:
    raw = _closed(value, _STRATEGY_FIELDS, "global strategy")
    if raw["schema_version"] != GLOBAL_STRATEGY_VERSION:
        raise HierarchicalDesignError("unsupported global strategy schema")
    if raw["strategy_epoch_id"] != epoch_id:
        raise HierarchicalDesignError("global strategy epoch binding mismatch")
    if raw["evidence_bundle_hash"] != evidence_bundle.get("bundle_hash"):
        raise HierarchicalDesignError("global strategy evidence binding mismatch")
    known_evidence = {
        str(item.get("evidence_id"))
        for item in evidence_bundle.get("records", [])
        if isinstance(item, Mapping)
    }
    directives_value = raw["island_directives"]
    if isinstance(directives_value, (str, bytes)) or not isinstance(
        directives_value, Sequence
    ):
        raise HierarchicalDesignError("island_directives must be a JSON list")
    if len(directives_value) != len(island_roles):
        raise HierarchicalDesignError("strategy must assign every island exactly once")
    directives = []
    seen_islands: set[int] = set()
    for index, item in enumerate(directives_value):
        directive = _closed(item, _DIRECTIVE_FIELDS, f"island_directives[{index}]")
        island = directive["island"]
        if isinstance(island, bool) or not isinstance(island, int):
            raise HierarchicalDesignError("directive island must be an integer")
        if island in seen_islands or not 0 <= island < len(island_roles):
            raise HierarchicalDesignError("directive island is duplicate or out of range")
        seen_islands.add(island)
        expected_role = str(island_roles[island].get("role_id") or "")
        if directive["role_id"] != expected_role:
            raise HierarchicalDesignError("directive role does not match controller assignment")
        evidence_refs = _strings(
            directive["evidence_refs"],
            f"island_directives[{index}].evidence_refs",
            minimum=1,
            maximum=16,
        )
        if not set(evidence_refs) <= known_evidence:
            raise HierarchicalDesignError("directive cites unknown evidence")
        quota = directive["outside_incumbent_quota"]
        if (
            isinstance(quota, bool)
            or not isinstance(quota, (int, float))
            or not math.isfinite(float(quota))
            or not 0.0 <= float(quota) <= 1.0
        ):
            raise HierarchicalDesignError("outside_incumbent_quota must be in [0,1]")
        if (
            requires_outside_incumbent_exploration(island_roles[island])
            and float(quota) + 1.0e-12 < float(minimum_region_outside_quota)
        ):
            raise HierarchicalDesignError(
                f"{expected_role} outside_incumbent_quota is below the "
                "controller-owned minimum"
            )
        directives.append(
            {
                "island": island,
                "role_id": expected_role,
                "objective": _text(directive["objective"], "directive.objective"),
                "priority_metrics": _strings(
                    directive["priority_metrics"], "directive.priority_metrics", minimum=1
                ),
                "architecture_targets": _strings(
                    directive["architecture_targets"], "directive.architecture_targets", minimum=1
                ),
                "avoid_patterns": _strings(
                    directive["avoid_patterns"], "directive.avoid_patterns"
                ),
                "evidence_refs": evidence_refs,
                "outside_incumbent_quota": float(quota),
                "required_ablation": _text(
                    directive["required_ablation"], "directive.required_ablation"
                ),
            }
        )
    synthesis = _closed(
        raw["cross_island_synthesis"], _SYNTHESIS_FIELDS, "cross_island_synthesis"
    )
    selection = _closed(raw["final_selection"], _SELECTION_FIELDS, "final_selection")
    hard_gate_metrics = _strings(
        selection["hard_gate_metrics"], "hard_gate_metrics", minimum=1
    )
    missing_hard_metrics = sorted(
        set(map(str, required_hard_gate_metrics)) - set(hard_gate_metrics)
    )
    if missing_hard_metrics:
        raise HierarchicalDesignError(
            "global strategy omitted controller-owned hard gate metrics: "
            + ", ".join(missing_hard_metrics)
        )
    return {
        "schema_version": GLOBAL_STRATEGY_VERSION,
        "strategy_epoch_id": epoch_id,
        "evidence_bundle_hash": evidence_bundle["bundle_hash"],
        "global_assessment": _text(raw["global_assessment"], "global_assessment", 4000),
        "global_constraints": _strings(
            raw["global_constraints"], "global_constraints", minimum=1
        ),
        "island_directives": sorted(directives, key=lambda item: item["island"]),
        "cross_island_synthesis": {
            "migration_pairs": _strings(synthesis["migration_pairs"], "migration_pairs"),
            "preserve_discoveries": _strings(
                synthesis["preserve_discoveries"], "preserve_discoveries", minimum=1
            ),
            "conflict_resolution": _text(
                synthesis["conflict_resolution"], "conflict_resolution"
            ),
        },
        "final_selection": {
            "hard_gate_metrics": hard_gate_metrics,
            "ranking_metrics": _strings(
                selection["ranking_metrics"], "ranking_metrics", minimum=1
            ),
            "robustness_metrics": _strings(
                selection["robustness_metrics"], "robustness_metrics", minimum=1
            ),
            "selection_rationale": _text(
                selection["selection_rationale"], "selection_rationale"
            ),
        },
        "stagnation_response": _text(raw["stagnation_response"], "stagnation_response"),
    }


def deterministic_global_strategy(
    *,
    epoch_id: str,
    evidence_bundle: Mapping[str, Any],
    island_roles: Sequence[Mapping[str, Any]],
    outside_incumbent_quota: float,
    hard_gate_metrics: Sequence[str] = (
        "hard_gate_pass",
        "positive_A_plddt",
        "apo_plddt",
    ),
) -> Dict[str, Any]:
    evidence_ids = [
        str(item["evidence_id"])
        for item in evidence_bundle.get("records", [])
        if isinstance(item, Mapping) and item.get("evidence_id")
    ] or ["controller:no_measured_evidence"]
    directives = []
    for island, role in enumerate(island_roles):
        role_id = str(role.get("role_id") or f"island_{island}")
        directives.append(
            {
                "island": island,
                "role_id": role_id,
                "objective": str(role.get("focus") or "Explore feasible improvements."),
                "priority_metrics": list(role.get("soft_objectives") or ["combined_score"]),
                "architecture_targets": [
                    "revise node topology and amino-acid policy only when cited evidence supports it",
                    "retain an incumbent structural anchor while testing one interpretable change",
                ],
                "avoid_patterns": [
                    "repeating the same four residue positions without new evidence",
                    "improving selectivity by degrading absolute A-state quality",
                ],
                "evidence_refs": evidence_ids[: min(4, len(evidence_ids))],
                "outside_incumbent_quota": float(outside_incumbent_quota),
                "required_ablation": "Compare the architecture change against the selected parent.",
            }
        )
    return {
        "schema_version": GLOBAL_STRATEGY_VERSION,
        "strategy_epoch_id": epoch_id,
        "evidence_bundle_hash": evidence_bundle["bundle_hash"],
        "global_assessment": (
            "Fallback controller strategy: preserve absolute A-state/fold quality, "
            "separate interface, selectivity, region-exploration, and robustness work, "
            "then synthesize only measured improvements."
        ),
        "global_constraints": [
            "Global evaluator hard gates are immutable.",
            "Metric values and evidence identities may not be rewritten by an LLM.",
            "A/B margin is not accepted when absolute A-state quality violates its floor.",
        ],
        "island_directives": directives,
        "cross_island_synthesis": {
            "migration_pairs": [
                f"{role.get('role_id', index)}->"
                + str((role.get("complementary_roles") or ["global"])[0])
                for index, role in enumerate(island_roles)
            ],
            "preserve_discoveries": [
                "feasible absolute-quality anchors",
                "new measured segments and amino-acid policies",
                "low-variance candidates",
            ],
            "conflict_resolution": (
                "Prefer global hard-gate feasibility, then primary objective, worst-case "
                "quality, clashes, and node confidence."
            ),
        },
        "final_selection": {
            "hard_gate_metrics": list(map(str, hard_gate_metrics)),
            "ranking_metrics": ["final_energy", "selectivity_proxy_score"],
            "robustness_metrics": ["worst_case_score", "seed_score_std", "clash_count"],
            "selection_rationale": (
                "Select across all islands with global feasibility-first robust ordering."
            ),
        },
        "stagnation_response": (
            "Move one node to a cited non-incumbent segment, retain one anchor, and test "
            "an explicit amino-acid-policy ablation."
        ),
    }


class HypothesisLedger:


    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: Dict[str, Dict[str, Any]] = {}
        if self.path.is_file():
            self.load(self.path)

    def observe_generation(self, database: Any, plan: Any, observations: Sequence[Any]) -> None:
        for observation in observations:
            child = getattr(observation, "child_program", None)
            parent = getattr(observation, "parent", None)
            artifacts = getattr(observation, "artifacts", {}) or {}
            proposal = artifacts.get("structured_ast_proposal_audit")
            audit = proposal.get("audit") if isinstance(proposal, Mapping) else None
            if not isinstance(audit, Mapping) or not audit.get("hypothesis_id"):
                continue
            hypothesis_id = str(audit["hypothesis_id"])
            proposal_id = str(getattr(observation, "proposal_id", "") or "")
            if not proposal_id:
                proposal_id = str(getattr(child, "id", "") or "unidentified")
            hypothesis_instance_id = f"{hypothesis_id}@{proposal_id}"
            prediction = deepcopy(dict(audit))
            prediction_hash = _hash(prediction)
            record = self.records.setdefault(
                hypothesis_instance_id,
                {
                    "hypothesis_instance_id": hypothesis_instance_id,
                    "hypothesis_id": hypothesis_id,
                    "proposal_id": proposal_id,
                    "prediction": prediction,
                    "prediction_hash": prediction_hash,
                    "observations": [],
                },
            )
            if record["prediction_hash"] != prediction_hash:
                raise HierarchicalDesignError(
                    f"hypothesis instance {hypothesis_instance_id!r} changed prediction"
                )
            child_metrics = deepcopy(dict(getattr(child, "metrics", {}) or {}))
            parent_metrics = deepcopy(dict(getattr(parent, "metrics", {}) or {}))
            effects = []
            for expected in audit.get("expected_effects", []):
                if not isinstance(expected, Mapping):
                    continue
                name = str(expected.get("metric_id") or "")
                before = parent_metrics.get(name)
                after = child_metrics.get(name)
                observed_direction = "unmeasured"
                if all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in (before, after)
                ):
                    observed_direction = (
                        "increase" if after > before else "decrease" if after < before else "maintain"
                    )
                effects.append(
                    {
                        "metric_id": name,
                        "predicted_direction": expected.get("direction"),
                        "observed_direction": observed_direction,
                        "matched": observed_direction == expected.get("direction"),
                        "parent_value": before,
                        "child_value": after,
                    }
                )
            observation_record = {
                "generation_id": getattr(plan, "generation_id", ""),
                "proposal_id": getattr(observation, "proposal_id", ""),
                "iteration": int(getattr(observation, "iteration", 0) or 0),
                "parent_program_id": getattr(parent, "id", None),
                "child_program_id": getattr(child, "id", None),
                "hard_gate_pass": child_metrics.get("hard_gate_pass"),
                "effect_calibration": effects,
                "status": (
                    "rejected"
                    if child is None
                    else
                    "contradicted"
                    if child_metrics.get("hard_gate_pass") is False
                    else "supported"
                    if effects and all(item["matched"] for item in effects)
                    else "partial"
                    if any(item["matched"] for item in effects)
                    else "inconclusive"
                ),
                "rollback_recommended": (
                    child is None or child_metrics.get("hard_gate_pass") is False
                ),
                "error": str(getattr(observation, "error", "") or "") or None,
            }
            if observation_record not in record["observations"]:
                record["observations"].append(observation_record)
            if child is not None:
                existing = database.get_artifacts(child.id) if hasattr(database, "get_artifacts") else {}
                database.store_artifacts(
                    child.id,
                    {**(existing or {}), "hierarchical_hypothesis_observation": observation_record},
                )
        self.save()

    def snapshot(self, limit: int = 64) -> Dict[str, Any]:
        ordered = sorted(
            self.records.values(),
            key=lambda item: (
                (item.get("observations") or [{}])[-1].get("iteration", -1),
                item.get("hypothesis_id", ""),
            ),
            reverse=True,
        )[: max(1, int(limit))]
        return {
            "schema_version": HYPOTHESIS_LEDGER_VERSION,
            "records": deepcopy(ordered),
            "record_count": len(self.records),
        }

    def save(self, path: str | Path | None = None) -> None:
        _atomic_json(Path(path) if path else self.path, self.snapshot(limit=max(1, len(self.records))))

    def load(self, path: str | Path) -> None:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping) or value.get("schema_version") != HYPOTHESIS_LEDGER_VERSION:
            raise HierarchicalDesignError("invalid hypothesis ledger checkpoint")
        records = value.get("records")
        if not isinstance(records, list):
            raise HierarchicalDesignError("invalid hypothesis ledger records")
        self.records = {
            str(
                item.get("hypothesis_instance_id")
                or f"{item['hypothesis_id']}@{item.get('proposal_id', 'legacy')}"
            ): deepcopy(dict(item))
            for item in records
            if isinstance(item, Mapping) and item.get("hypothesis_id")
        }


class HierarchicalDesignRuntime:


    def __init__(self, config: Any, database_config: Any, output_dir: str | Path):
        self.config = config
        self.database_config = database_config
        self.output_dir = Path(output_dir)
        self.state_path = self.output_dir / "hierarchical_design.json"
        ledger_path = getattr(config, "hypothesis_ledger_path", None)
        self.ledger = HypothesisLedger(
            Path(ledger_path) if ledger_path else self.output_dir / "hypothesis_ledger.json"
        )
        reasoning_path = getattr(config, "reasoning_audit_path", None)
        self.reasoning_ledger = ReasoningAuditLedger(
            Path(reasoning_path) if reasoning_path else self.output_dir / "reasoning_audit.json"
        )
        self.epoch_index = 0
        self.last_refresh_iteration = -1
        self._stagnation_refresh_active = False
        self._snapshot: Dict[str, Any] = {}
        if self.state_path.is_file():
            self.load(self.state_path)

    def _roles(self) -> list[Dict[str, Any]]:
        roles = list(getattr(self.database_config, "island_roles", []) or [])
        count = int(getattr(self.database_config, "num_islands", len(roles) or 1))
        return [deepcopy(dict(roles[index % len(roles)])) for index in range(count)] if roles else [
            {"role_id": f"island_{index}", "focus": "Explore feasible improvements."}
            for index in range(count)
        ]

    def needs_refresh(self, iteration: int) -> bool:
        interval = max(
            1,
            int(getattr(self.database_config, "strategy_epoch_candidate_interval", 8)),
        )
        return not self._snapshot or int(iteration) - self.last_refresh_iteration >= interval

    def stagnation_refresh_required(self, stagnated: bool) -> bool:


        if not bool(stagnated):
            self._stagnation_refresh_active = False
            return False
        if self._stagnation_refresh_active:
            return False
        self._stagnation_refresh_active = True
        return True

    async def refresh(
        self,
        database: Any,
        llm_ensemble: Any,
        *,
        iteration: int,
        force: bool = False,
    ) -> Dict[str, Any]:
        if not force and not self.needs_refresh(iteration):
            return self.snapshot()
        evidence = build_evidence_bundle(
            database,
            max_chars=int(getattr(self.config, "evidence_max_chars", 32768)),
        )
        self.epoch_index += 1
        epoch_id = f"strategy-{self.epoch_index:06d}-{evidence['bundle_hash'][:12]}"
        roles = self._roles()
        fallback = deterministic_global_strategy(
            epoch_id=epoch_id,
            evidence_bundle=evidence,
            island_roles=roles,
            outside_incumbent_quota=float(
                getattr(self.config, "outside_incumbent_execution_quota", 0.5)
            ),
            hard_gate_metrics=list(
                getattr(
                    self.config,
                    "required_hard_gate_metrics",
                    ["hard_gate_pass", "positive_A_plddt", "apo_plddt"],
                )
            ),
        )
        strategy = validate_global_strategy(
            fallback,
            epoch_id=epoch_id,
            evidence_bundle=evidence,
            island_roles=roles,
            minimum_region_outside_quota=float(
                getattr(self.config, "outside_incumbent_execution_quota", 0.5)
            ),
            required_hard_gate_metrics=fallback["final_selection"][
                "hard_gate_metrics"
            ],
        )
        source = "deterministic_fallback"
        errors: list[str] = []
        if bool(getattr(self.config, "global_strategist_enabled", True)) and llm_ensemble is not None:
            request = {
                "task": "Synthesize one evidence-bound whole-protein evolution strategy.",
                "required_schema": GLOBAL_STRATEGY_VERSION,
                "strategy_epoch_id": epoch_id,
                "evidence_bundle": evidence,
                "controller_island_roles": roles,
                "rules": [
                    "Return one JSON object only; all fields are closed.",
                    "Cite supplied evidence IDs; never invent or rewrite measurements.",
                    "Keep global hard gates immutable and distinguish margin from absolute quality.",
                    "Assign node topology, segment exploration, amino-acid policy, ablation, and rollback-aware work.",
                    "Provide concise public scientific rationale, not hidden chain-of-thought.",
                ],
                "fallback_shape_example": fallback,
            }
            retries = max(0, int(getattr(self.config, "global_strategy_retries", 2)))
            messages = [{"role": "user", "content": _canonical_json(request)}]
            for _attempt in range(retries + 1):
                try:
                    response = await llm_ensemble.generate_with_context(
                        system_message=(
                            "You are the global protein-architecture strategist. Compare all "
                            "islands and return strict public scientific JSON only."
                        ),
                        messages=messages,
                    )
                    parsed = json.loads(str(response).strip())
                    strategy = validate_global_strategy(
                        parsed,
                        epoch_id=epoch_id,
                        evidence_bundle=evidence,
                        island_roles=roles,
                        minimum_region_outside_quota=float(
                            getattr(
                                self.config,
                                "outside_incumbent_execution_quota",
                                0.5,
                            )
                        ),
                        required_hard_gate_metrics=fallback["final_selection"][
                            "hard_gate_metrics"
                        ],
                    )
                    source = "llm_validated"
                    break
                except Exception as error:
                    errors.append(f"{type(error).__name__}: {error}"[:1000])
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Repair the same strategy from the unchanged evidence. "
                                f"Validation error: {errors[-1]}. Return JSON only."
                            ),
                        }
                    )
        self.last_refresh_iteration = int(iteration)
        self._snapshot = {
            "schema_version": HIERARCHICAL_SNAPSHOT_VERSION,
            "controller_owned": True,
            "strategy_epoch_id": epoch_id,
            "iteration": int(iteration),
            "evidence_bundle": evidence,
            "strategy_plan": strategy,
            "strategy_hash": "global_strategy_sha256:" + _hash(strategy),
            "strategy_source": source,
            "strategy_validation_errors": errors,
            "executable_island_set": (
                compile_executable_island_directives(strategy, roles)
                if bool(
                    getattr(
                        self.database_config,
                        "executable_island_directives_enabled",
                        False,
                    )
                )
                else {
                    "schema_version": "astevolve.executable_island_set.disabled.v1",
                    "enabled": False,
                }
            ),
            "governance": {
                "node_tenure": int(getattr(self.config, "node_tenure", 3)),
                "exploration_window": int(
                    getattr(self.config, "exploration_window", 4)
                ),
                "outside_incumbent_execution_quota": float(
                    getattr(
                        self.config,
                        "outside_incumbent_execution_quota",
                        0.5,
                    )
                ),
                "min_segments_per_window": int(
                    getattr(self.config, "min_segments_per_window", 2)
                ),
            },
            "hypothesis_ledger": self.ledger.snapshot(),
            "reasoning_audit": self.reasoning_ledger.snapshot(),
        }
        self.save()
        return self.snapshot()

    def observe_generation(
        self, database: Any, plan: Any, observations: Sequence[Any]
    ) -> Dict[str, Any]:
        self.ledger.observe_generation(database, plan, observations)
        if not bool(getattr(self.config, "reasoning_audit_enabled", True)):
            if self._snapshot:
                self._snapshot["hypothesis_ledger"] = self.ledger.snapshot()
                self.save()
            return {
                "status": "published",
                "hypothesis_record_count": self.ledger.snapshot()["record_count"],
                "reasoning_audit_enabled": False,
                "reasoning_record_count": self.reasoning_ledger.aggregate()[
                    "record_count"
                ],
            }
        for observation in observations:
            child = getattr(observation, "child_program", None)
            parent = getattr(observation, "parent", None)
            artifacts = getattr(observation, "artifacts", {}) or {}
            prediction = artifacts.get("sealed_reasoning_prediction")
            if parent is None or not isinstance(prediction, Mapping):
                continue
            if child is None:
                reasoning_observation = build_failed_reasoning_observation(
                    prediction,
                    parent_program=parent,
                    artifacts=artifacts,
                    error=getattr(observation, "error", None),
                    hierarchical_design=self._snapshot,
                )
            else:
                reasoning_observation = build_reasoning_observation(
                    prediction,
                    parent_program=parent,
                    child_program=child,
                    artifacts=artifacts,
                    hierarchical_design=self._snapshot,
                )
            self.reasoning_ledger.record(prediction, reasoning_observation)
            if child is not None:
                existing = database.get_artifacts(child.id) if hasattr(database, "get_artifacts") else {}
                database.store_artifacts(
                    child.id,
                    {
                        **(existing or {}),
                        "reasoning_observation": reasoning_observation,
                    },
                )
        if self._snapshot:
            self._snapshot["hypothesis_ledger"] = self.ledger.snapshot()
            self._snapshot["reasoning_audit"] = self.reasoning_ledger.snapshot()
            self.save()
        return {
            "status": "published",
            "hypothesis_record_count": self.ledger.snapshot()["record_count"],
            "reasoning_audit_enabled": True,
            "reasoning_record_count": self.reasoning_ledger.aggregate()[
                "record_count"
            ],
        }

    def snapshot(self) -> Dict[str, Any]:
        return deepcopy(self._snapshot)

    def save(self, path: str | Path | None = None) -> None:
        payload = {
            "schema_version": HIERARCHICAL_SNAPSHOT_VERSION,
            "epoch_index": self.epoch_index,
            "last_refresh_iteration": self.last_refresh_iteration,
            "stagnation_refresh_active": self._stagnation_refresh_active,
            "snapshot": self._snapshot,
        }
        _atomic_json(Path(path) if path else self.state_path, payload)

    def load(self, path: str | Path) -> None:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        legacy = {"schema_version", "epoch_index", "last_refresh_iteration", "snapshot"}
        current = legacy | {"stagnation_refresh_active"}
        if not isinstance(value, Mapping) or set(value) not in (legacy, current):
            raise HierarchicalDesignError("invalid hierarchical design checkpoint")
        if value["schema_version"] != HIERARCHICAL_SNAPSHOT_VERSION:
            raise HierarchicalDesignError("unsupported hierarchical design checkpoint")
        self.epoch_index = int(value["epoch_index"])
        self.last_refresh_iteration = int(value["last_refresh_iteration"])
        stagnation_active = value.get("stagnation_refresh_active", False)
        if not isinstance(stagnation_active, bool):
            raise HierarchicalDesignError(
                "hierarchical checkpoint stagnation state must be boolean"
            )
        self._stagnation_refresh_active = stagnation_active
        self._snapshot = deepcopy(dict(value["snapshot"]))


__all__ = [
    "EVIDENCE_BUNDLE_VERSION",
    "GLOBAL_STRATEGY_VERSION",
    "HIERARCHICAL_SNAPSHOT_VERSION",
    "HYPOTHESIS_LEDGER_VERSION",
    "HierarchicalDesignError",
    "HierarchicalDesignRuntime",
    "HypothesisLedger",
    "build_evidence_bundle",
    "deterministic_global_strategy",
    "validate_global_strategy",
]
