

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
import threading
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

from astevolve.evaluation.selection import select_feasibility_first


EVOLUTION_CANDIDATE_VERSION = "astevolve.outer_evolution_candidate.v1"
EVOLUTION_DECISION_VERSION = "astevolve.outer_evolution_decision.v1"
_MODES = frozenset({"best", "uniform", "weighted", "mixture"})
_MIXTURE_ARMS = ("elite", "weighted", "uniform")


class EvolutionPolicyError(ValueError):
    pass


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EvolutionPolicyError(f"value is not canonical JSON: {exc}") from exc


def _digest(domain: str, value: Any) -> str:
    return hashlib.sha256(f"{domain}\0{_canonical(value)}".encode("utf-8")).hexdigest()


def _required(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise EvolutionPolicyError(f"{label} must be non-empty")
    return text


def _objective(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvolutionPolicyError("objective must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise EvolutionPolicyError("objective must be a finite number")
    return result


@dataclass(frozen=True)
class EvolutionCandidate:
    candidate_id: str
    objective: float
    gate_sources_json: str
    phenotype_hash: str
    effective_contract_hash: str
    sequence_bundle_hash: str
    schema_version: str = EVOLUTION_CANDIDATE_VERSION

    @property
    def gate_sources(self) -> dict[str, Any]:
        return json.loads(self.gate_sources_json)

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        objective: float,
        gate_sources: Mapping[str, Any],
        phenotype_hash: str,
        effective_contract_hash: str,
        sequence_bundle_hash: str,
    ) -> "EvolutionCandidate":
        if not isinstance(gate_sources, Mapping) or not gate_sources:
            raise EvolutionPolicyError("gate_sources must be a non-empty mapping")
        candidate = cls(
            candidate_id=_required(candidate_id, "candidate_id"),
            objective=_objective(objective),
            gate_sources_json=_canonical(dict(gate_sources)),
            phenotype_hash=_required(phenotype_hash, "phenotype_hash"),
            effective_contract_hash=_required(
                effective_contract_hash, "effective_contract_hash"
            ),
            sequence_bundle_hash=_required(
                sequence_bundle_hash, "sequence_bundle_hash"
            ),
        )

        select_feasibility_first([candidate.selection_row()])
        return candidate

    def selection_row(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "raw_objective": self.objective,
            "gate_sources": self.gate_sources,
        }

    @property
    def feasible(self) -> bool:
        decision = select_feasibility_first([self.selection_row()])
        return bool(decision["candidates"][0]["feasible"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "objective": self.objective,
            "gate_sources": self.gate_sources,
            "phenotype_hash": self.phenotype_hash,
            "effective_contract_hash": self.effective_contract_hash,
            "sequence_bundle_hash": self.sequence_bundle_hash,
            "feasible": self.feasible,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvolutionCandidate":
        if not isinstance(value, Mapping):
            raise EvolutionPolicyError("candidate must be a mapping")
        fields = {
            "schema_version",
            "candidate_id",
            "objective",
            "gate_sources",
            "phenotype_hash",
            "effective_contract_hash",
            "sequence_bundle_hash",
            "feasible",
        }
        if set(value) != fields:
            raise EvolutionPolicyError("candidate fields are not closed")
        if value.get("schema_version") != EVOLUTION_CANDIDATE_VERSION:
            raise EvolutionPolicyError("candidate schema_version is invalid")
        result = cls.create(
            candidate_id=value.get("candidate_id"),
            objective=value.get("objective"),
            gate_sources=value.get("gate_sources"),
            phenotype_hash=value.get("phenotype_hash"),
            effective_contract_hash=value.get("effective_contract_hash"),
            sequence_bundle_hash=value.get("sequence_bundle_hash"),
        )
        if value.get("feasible") is not result.feasible:
            raise EvolutionPolicyError("candidate feasible field is inconsistent")
        return result


@dataclass(frozen=True)
class EvolutionDecision:
    kind: str
    namespace: str
    decision_index: int
    derived_seed: int
    selected_candidate_id: Optional[str]
    reason: str
    candidate_ids: tuple[str, ...]
    eligible_ids: tuple[str, ...]
    add_ids: tuple[str, ...]
    remove_ids: tuple[str, ...]
    details_json: str
    decision_hash: str
    schema_version: str = EVOLUTION_DECISION_VERSION

    @property
    def details(self) -> dict[str, Any]:
        return json.loads(self.details_json)

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        namespace: str,
        decision_index: int,
        derived_seed: int,
        selected_candidate_id: Optional[str],
        reason: str,
        candidate_ids: Iterable[str],
        eligible_ids: Iterable[str],
        add_ids: Iterable[str] = (),
        remove_ids: Iterable[str] = (),
        details: Optional[Mapping[str, Any]] = None,
    ) -> "EvolutionDecision":
        if isinstance(decision_index, bool) or not isinstance(decision_index, int) or decision_index < 0:
            raise EvolutionPolicyError("decision_index must be a non-negative integer")
        if isinstance(derived_seed, bool) or not isinstance(derived_seed, int) or derived_seed < 0:
            raise EvolutionPolicyError("derived_seed must be a non-negative integer")
        candidates = tuple(str(item) for item in candidate_ids)
        if not candidates or len(candidates) != len(set(candidates)):
            raise EvolutionPolicyError("candidate_ids must be non-empty and unique")
        eligible = tuple(str(item) for item in eligible_ids)
        if not set(eligible) <= set(candidates):
            raise EvolutionPolicyError("eligible_ids must be candidate_ids")
        selected = str(selected_candidate_id) if selected_candidate_id is not None else None
        if selected is not None and selected not in candidates:
            raise EvolutionPolicyError("selected_candidate_id is not a candidate")
        payload = {
            "kind": _required(kind, "kind"),
            "namespace": _required(namespace, "namespace"),
            "decision_index": decision_index,
            "derived_seed": derived_seed,
            "selected_candidate_id": selected,
            "reason": _required(reason, "reason"),
            "candidate_ids": list(candidates),
            "eligible_ids": list(eligible),
            "add_ids": [str(item) for item in add_ids],
            "remove_ids": [str(item) for item in remove_ids],
            "details": dict(details or {}),
        }
        digest = "outer_decision_sha256:" + _digest("outer_decision.v1", payload)
        return cls(
            kind=payload["kind"],
            namespace=payload["namespace"],
            decision_index=decision_index,
            derived_seed=derived_seed,
            selected_candidate_id=selected,
            reason=payload["reason"],
            candidate_ids=candidates,
            eligible_ids=eligible,
            add_ids=tuple(payload["add_ids"]),
            remove_ids=tuple(payload["remove_ids"]),
            details_json=_canonical(payload["details"]),
            decision_hash=digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "namespace": self.namespace,
            "decision_index": self.decision_index,
            "derived_seed": self.derived_seed,
            "selected_candidate_id": self.selected_candidate_id,
            "reason": self.reason,
            "candidate_ids": list(self.candidate_ids),
            "eligible_ids": list(self.eligible_ids),
            "add_ids": list(self.add_ids),
            "remove_ids": list(self.remove_ids),
            "details": self.details,
            "decision_hash": self.decision_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvolutionDecision":
        if not isinstance(value, Mapping):
            raise EvolutionPolicyError("decision must be a mapping")
        fields = {
            "schema_version", "kind", "namespace", "decision_index",
            "derived_seed", "selected_candidate_id", "reason", "candidate_ids",
            "eligible_ids", "add_ids", "remove_ids", "details", "decision_hash",
        }
        if set(value) != fields or value.get("schema_version") != EVOLUTION_DECISION_VERSION:
            raise EvolutionPolicyError("decision fields or schema_version are invalid")
        result = cls.create(
            kind=value.get("kind"), namespace=value.get("namespace"),
            decision_index=value.get("decision_index"), derived_seed=value.get("derived_seed"),
            selected_candidate_id=value.get("selected_candidate_id"), reason=value.get("reason"),
            candidate_ids=value.get("candidate_ids") or (), eligible_ids=value.get("eligible_ids") or (),
            add_ids=value.get("add_ids") or (), remove_ids=value.get("remove_ids") or (),
            details=value.get("details") or {},
        )
        if value.get("decision_hash") != result.decision_hash:
            raise EvolutionPolicyError("decision hash mismatch")
        return result


def derive_private_seed(
    base_seed: Optional[int], *, namespace: str, decision_index: int, candidate_ids: Iterable[str]
) -> int:
    if base_seed is not None and (isinstance(base_seed, bool) or not isinstance(base_seed, int)):
        raise EvolutionPolicyError("base_seed must be an integer or null")
    payload = {
        "base_seed": base_seed,
        "namespace": _required(namespace, "namespace"),
        "decision_index": int(decision_index),
        "candidate_ids": sorted(str(item) for item in candidate_ids),
    }
    return int(_digest("outer_private_rng.v1", payload)[:16], 16)


def _selection(candidates: Sequence[EvolutionCandidate]) -> dict[str, Any]:
    if not candidates or any(not isinstance(item, EvolutionCandidate) for item in candidates):
        raise EvolutionPolicyError("candidates must contain EvolutionCandidate values")
    return select_feasibility_first([item.selection_row() for item in candidates])


def _parent_mixture(value: Optional[Mapping[str, Any]]) -> dict[str, float]:
    raw = dict(value or {"elite": 0.1, "weighted": 0.7, "uniform": 0.2})
    if set(raw) != set(_MIXTURE_ARMS):
        raise EvolutionPolicyError(
            "parent mixture must contain exactly elite, weighted, and uniform"
        )
    weights: dict[str, float] = {}
    for arm in _MIXTURE_ARMS:
        value = raw[arm]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvolutionPolicyError("parent mixture weights must be numeric")
        weight = float(value)
        if not math.isfinite(weight) or weight < 0.0:
            raise EvolutionPolicyError(
                "parent mixture weights must be finite and non-negative"
            )
        weights[arm] = weight
    total = sum(weights.values())
    if total <= 0.0:
        raise EvolutionPolicyError("parent mixture total must be positive")
    return {arm: weights[arm] / total for arm in _MIXTURE_ARMS}


def _sampling_factors(
    value: Optional[Mapping[str, Any]], *, candidate_ids: Iterable[str]
) -> dict[str, float]:
    ids = tuple(str(item) for item in candidate_ids)
    if value is None:
        return {candidate_id: 1.0 for candidate_id in ids}
    raw = dict(value)
    unknown = sorted(set(raw) - set(ids))
    missing = sorted(set(ids) - set(raw))
    if unknown or missing:
        raise EvolutionPolicyError(
            f"candidate sampling factors must cover candidates; unknown={unknown}, missing={missing}"
        )
    factors: dict[str, float] = {}
    for candidate_id in ids:
        raw_factor = raw[candidate_id]
        if isinstance(raw_factor, bool) or not isinstance(raw_factor, (int, float)):
            raise EvolutionPolicyError("candidate sampling factors must be numeric")
        factor = float(raw_factor)
        if not math.isfinite(factor) or factor < 0.0:
            raise EvolutionPolicyError(
                "candidate sampling factors must be finite and non-negative"
            )
        factors[candidate_id] = factor
    if not any(value > 0.0 for value in factors.values()):
        raise EvolutionPolicyError("at least one candidate sampling factor must be positive")
    return factors


def decide_parent(
    candidates: Sequence[EvolutionCandidate], *, base_seed: Optional[int],
    namespace: str, decision_index: int,
    mode: Literal["best", "uniform", "weighted", "mixture"] = "weighted",
    mixture: Optional[Mapping[str, Any]] = None,
    candidate_sampling_factors: Optional[Mapping[str, Any]] = None,
) -> EvolutionDecision:
    if mode not in _MODES:
        raise EvolutionPolicyError(f"unsupported parent mode: {mode}")
    selection = _selection(candidates)
    by_id = {item.candidate_id: item for item in candidates}
    eligible = tuple(selection["eligible_ids"])
    seed = derive_private_seed(
        base_seed, namespace=namespace, decision_index=decision_index,
        candidate_ids=by_id,
    )
    rng = random.Random(seed)
    if mode != "mixture" and candidate_sampling_factors is None:
        if mode == "best":
            selected = eligible[0]
        elif mode == "uniform":
            selected = eligible[rng.randrange(len(eligible))]
        else:
            objectives = [by_id[item].objective for item in eligible]
            floor = min(objectives)
            weights = [max(value - floor, 0.0) + 1e-12 for value in objectives]
            selected = rng.choices(list(eligible), weights=weights, k=1)[0]
        return EvolutionDecision.create(
            kind="parent_selection",
            namespace=namespace,
            decision_index=decision_index,
            derived_seed=seed,
            selected_candidate_id=selected,
            reason=f"feasibility_first_{mode}",
            candidate_ids=selection["ordered_ids"],
            eligible_ids=eligible,
            details={"mode": mode, "feasibility_selection": selection},
        )
    normalized_mixture = _parent_mixture(mixture) if mode == "mixture" else None
    selected_mode = (
        rng.choices(
            list(_MIXTURE_ARMS),
            weights=[normalized_mixture[arm] for arm in _MIXTURE_ARMS],
            k=1,
        )[0]
        if normalized_mixture is not None
        else "elite"
        if mode == "best"
        else mode
    )
    factors = _sampling_factors(
        candidate_sampling_factors,
        candidate_ids=by_id,
    )
    base_weights: dict[str, float]
    effective_weights: dict[str, float]
    factor_fallback = False
    if selected_mode == "elite":
        selected = eligible[0]
        base_weights = {
            candidate_id: float(candidate_id == selected) for candidate_id in eligible
        }
        effective_weights = dict(base_weights)
    elif selected_mode == "uniform":
        base_weights = {candidate_id: 1.0 for candidate_id in eligible}
        effective_weights = {
            candidate_id: base_weights[candidate_id] * factors[candidate_id]
            for candidate_id in eligible
        }
        if not any(effective_weights.values()):
            factor_fallback = True
            effective_weights = dict(base_weights)
        selected = rng.choices(
            list(eligible),
            weights=[effective_weights[candidate_id] for candidate_id in eligible],
            k=1,
        )[0]
    else:
        objectives = [by_id[item].objective for item in eligible]
        floor = min(objectives)
        base_weights = {
            candidate_id: max(by_id[candidate_id].objective - floor, 0.0) + 1e-12
            for candidate_id in eligible
        }
        effective_weights = {
            candidate_id: base_weights[candidate_id] * factors[candidate_id]
            for candidate_id in eligible
        }
        if not any(effective_weights.values()):
            factor_fallback = True
            effective_weights = dict(base_weights)
        selected = rng.choices(
            list(eligible),
            weights=[effective_weights[candidate_id] for candidate_id in eligible],
            k=1,
        )[0]
    legacy_details = {
        "mode": mode,
        "feasibility_selection": selection,
    }
    v9_details = {
        "mode": "mixture",
        "selected_mode": selected_mode,
        "mixture": normalized_mixture,
        "candidate_sampling": {
            candidate_id: {
                "objective": by_id[candidate_id].objective,
                "base_weight": base_weights[candidate_id],
                "external_factor": factors[candidate_id],
                "effective_weight": effective_weights[candidate_id],
            }
            for candidate_id in eligible
        },
        "sampling_factor_fallback": factor_fallback,
        "feasibility_selection": selection,
    }
    return EvolutionDecision.create(
        kind="parent_selection", namespace=namespace, decision_index=decision_index,
        derived_seed=seed, selected_candidate_id=selected,
        reason=(
            f"feasibility_first_mixture_{selected_mode}"
            if mode == "mixture"
            else f"feasibility_first_{mode}"
        ), candidate_ids=selection["ordered_ids"],
        eligible_ids=eligible,
        details=v9_details if mode == "mixture" else legacy_details,
    )


def compare_replacement(
    challenger: EvolutionCandidate,
    incumbent: EvolutionCandidate,
    *,
    base_seed: Optional[int],
    namespace: str,
    decision_index: int,
    reject_duplicate_phenotype: bool = True,
    secondary_scores: Optional[Mapping[str, float]] = None,
    secondary_label: str = "",
    secondary_objective_tolerance: float = 0.0,
) -> EvolutionDecision:
    candidates = (challenger, incumbent)
    seed = derive_private_seed(
        base_seed, namespace=namespace, decision_index=decision_index,
        candidate_ids=(challenger.candidate_id, incumbent.candidate_id),
    )
    if reject_duplicate_phenotype and challenger.phenotype_hash == incumbent.phenotype_hash:
        selected = incumbent.candidate_id
        reason = "duplicate_effective_phenotype_rejected"
    elif (
        secondary_scores is not None
        and challenger.feasible
        and incumbent.feasible
        and abs(challenger.objective - incumbent.objective)
        <= max(0.0, float(secondary_objective_tolerance))
        and challenger.candidate_id in secondary_scores
        and incumbent.candidate_id in secondary_scores
        and math.isfinite(float(secondary_scores[challenger.candidate_id]))
        and math.isfinite(float(secondary_scores[incumbent.candidate_id]))
        and float(secondary_scores[challenger.candidate_id])
        != float(secondary_scores[incumbent.candidate_id])
    ):
        selected = max(
            (challenger.candidate_id, incumbent.candidate_id),
            key=lambda candidate_id: (
                float(secondary_scores[candidate_id]),
                candidate_id == incumbent.candidate_id,
            ),
        )
        reason = (
            f"secondary_{secondary_label}_preference"
            if secondary_label
            else "secondary_preference"
        )
    else:
        selection = _selection(candidates)
        selected = selection["selected_candidate_id"]

        if (
            challenger.feasible == incumbent.feasible
            and challenger.objective == incumbent.objective
        ):
            selected = incumbent.candidate_id
            reason = "objective_tie_retains_incumbent"
        else:
            reason = selection["reason"]
    replace = selected == challenger.candidate_id
    selection = _selection(candidates)
    return EvolutionDecision.create(
        kind="cell_replacement", namespace=namespace, decision_index=decision_index,
        derived_seed=seed, selected_candidate_id=selected, reason=reason,
        candidate_ids=selection["ordered_ids"], eligible_ids=selection["eligible_ids"],
        add_ids=(challenger.candidate_id,) if replace else (),
        remove_ids=(incumbent.candidate_id,) if replace else (),
        details={
            "challenger": challenger.to_dict(), "incumbent": incumbent.to_dict(),
            "replace": replace, "feasibility_selection": selection,
            "secondary_label": secondary_label or None,
            "secondary_objective_tolerance": max(
                0.0, float(secondary_objective_tolerance)
            ),
            "secondary_scores": (
                {
                    candidate_id: float(secondary_scores[candidate_id])
                    for candidate_id in (challenger.candidate_id, incumbent.candidate_id)
                    if secondary_scores is not None and candidate_id in secondary_scores
                }
                if secondary_scores is not None
                else None
            ),
        },
    )


def decide_migration(
    migrant: EvolutionCandidate,
    residents: Sequence[EvolutionCandidate],
    *,
    capacity: int,
    base_seed: Optional[int],
    namespace: str,
    decision_index: int,
    secondary_scores: Optional[Mapping[str, float]] = None,
    secondary_label: str = "",
    secondary_objective_tolerance: float = 0.0,
) -> EvolutionDecision:
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise EvolutionPolicyError("migration capacity must be positive")
    if any(item.phenotype_hash == migrant.phenotype_hash for item in residents):
        seed = derive_private_seed(
            base_seed, namespace=namespace, decision_index=decision_index,
            candidate_ids=(migrant.candidate_id, *(item.candidate_id for item in residents)),
        )
        return EvolutionDecision.create(
            kind="island_migration", namespace=namespace, decision_index=decision_index,
            derived_seed=seed, selected_candidate_id=None,
            reason="duplicate_effective_phenotype_in_target",
            candidate_ids=(migrant.candidate_id, *(item.candidate_id for item in residents)),
            eligible_ids=(), details={"shared_program_id": migrant.candidate_id},
        )
    if len(residents) < capacity:
        seed = derive_private_seed(
            base_seed, namespace=namespace, decision_index=decision_index,
            candidate_ids=(migrant.candidate_id, *(item.candidate_id for item in residents)),
        )
        return EvolutionDecision.create(
            kind="island_migration", namespace=namespace, decision_index=decision_index,
            derived_seed=seed, selected_candidate_id=migrant.candidate_id,
            reason="target_has_capacity", candidate_ids=(migrant.candidate_id, *(item.candidate_id for item in residents)),
            eligible_ids=(migrant.candidate_id,), add_ids=(migrant.candidate_id,),
            details={"shared_program_id": migrant.candidate_id, "copy_evaluation": False},
        )


    resident_selection = _selection(tuple(residents))
    global_worst_id = resident_selection["ordered_ids"][-1]
    global_worst = next(
        item for item in residents if item.candidate_id == global_worst_id
    )
    worst_id = global_worst_id
    tolerance = max(0.0, float(secondary_objective_tolerance))
    if (
        secondary_scores is not None
        and migrant.feasible
        and global_worst.feasible
        and migrant.candidate_id in secondary_scores
        and math.isfinite(float(secondary_scores[migrant.candidate_id]))
        and all(
            item.candidate_id in secondary_scores
            and math.isfinite(float(secondary_scores[item.candidate_id]))
            for item in residents
        )
    ):
        base_rank = {
            candidate_id: index
            for index, candidate_id in enumerate(resident_selection["ordered_ids"])
        }
        near_global_worst = tuple(
            item
            for item in residents
            if item.feasible == global_worst.feasible
            and abs(item.objective - global_worst.objective) <= tolerance
        )
        worst_id = min(
            (item.candidate_id for item in near_global_worst),
            key=lambda candidate_id: (
                float(secondary_scores[candidate_id]),
                -base_rank[candidate_id],
            ),
        )
    worst = next(item for item in residents if item.candidate_id == worst_id)
    comparison = compare_replacement(
        migrant, worst, base_seed=base_seed, namespace=namespace,
        decision_index=decision_index, reject_duplicate_phenotype=True,
        secondary_scores=secondary_scores,
        secondary_label=secondary_label,
        secondary_objective_tolerance=secondary_objective_tolerance,
    )
    replace = comparison.selected_candidate_id == migrant.candidate_id
    return EvolutionDecision.create(
        kind="island_migration", namespace=namespace, decision_index=decision_index,
        derived_seed=comparison.derived_seed,
        selected_candidate_id=migrant.candidate_id if replace else worst.candidate_id,
        reason=("feasibility_first_migrant_replaces_worst" if replace else "migrant_not_better_than_worst"),
        candidate_ids=comparison.candidate_ids, eligible_ids=comparison.eligible_ids,
        add_ids=(migrant.candidate_id,) if replace else (),
        remove_ids=(worst.candidate_id,) if replace else (),
        details={
            "shared_program_id": migrant.candidate_id,
            "copy_evaluation": False,
            "global_worst_resident_id": global_worst_id,
            "compared_resident_id": worst_id,
            "secondary_objective_tolerance": tolerance,
            "replacement_decision": comparison.to_dict(),
        },
    )


class AtomicPhenotypeArchive:


    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise EvolutionPolicyError("archive capacity must be positive")
        self.capacity = capacity
        self._members: dict[str, EvolutionCandidate] = {}
        self._phenotypes: dict[str, str] = {}
        self._lock = threading.RLock()

    def snapshot(self) -> tuple[EvolutionCandidate, ...]:
        with self._lock:
            return tuple(self._members[key] for key in sorted(self._members))

    def admit(
        self,
        candidate: EvolutionCandidate,
        *,
        base_seed: Optional[int],
        namespace: str,
        decision_index: int,
    ) -> EvolutionDecision:
        if not isinstance(candidate, EvolutionCandidate):
            raise EvolutionPolicyError("archive candidate must be EvolutionCandidate")
        with self._lock:
            duplicate_id = self._phenotypes.get(candidate.phenotype_hash)
            if duplicate_id is not None:
                return compare_replacement(
                    candidate,
                    self._members[duplicate_id],
                    base_seed=base_seed,
                    namespace=namespace,
                    decision_index=decision_index,
                    reject_duplicate_phenotype=True,
                )
            if len(self._members) < self.capacity:
                seed = derive_private_seed(
                    base_seed,
                    namespace=namespace,
                    decision_index=decision_index,
                    candidate_ids=(candidate.candidate_id,),
                )
                decision = EvolutionDecision.create(
                    kind="archive_admission",
                    namespace=namespace,
                    decision_index=decision_index,
                    derived_seed=seed,
                    selected_candidate_id=candidate.candidate_id,
                    reason="archive_has_capacity",
                    candidate_ids=(candidate.candidate_id,),
                    eligible_ids=(candidate.candidate_id,),
                    add_ids=(candidate.candidate_id,),
                    details={"candidate": candidate.to_dict()},
                )
            else:
                selection = _selection(tuple(self._members.values()))
                worst_id = selection["ordered_ids"][-1]
                decision = compare_replacement(
                    candidate,
                    self._members[worst_id],
                    base_seed=base_seed,
                    namespace=namespace,
                    decision_index=decision_index,
                )


            for removed in decision.remove_ids:
                prior = self._members.pop(removed, None)
                if prior is not None:
                    self._phenotypes.pop(prior.phenotype_hash, None)
            for added in decision.add_ids:
                if added != candidate.candidate_id:
                    raise EvolutionPolicyError("archive decision attempted a foreign add")
                self._members[added] = candidate
                self._phenotypes[candidate.phenotype_hash] = added
            return decision


__all__ = [
    "EVOLUTION_CANDIDATE_VERSION", "EVOLUTION_DECISION_VERSION",
    "AtomicPhenotypeArchive", "EvolutionCandidate", "EvolutionDecision", "EvolutionPolicyError",
    "compare_replacement", "decide_migration", "decide_parent",
    "derive_private_seed",
]
