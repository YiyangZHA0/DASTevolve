

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from astevolve.domain import (
    DesignStrategy,
    EvaluationReport,
    EvidenceBundle,
    EvidenceRecord,
)
from astevolve.domain.dual_ast import ExecutableDualAST


LEGACY_PROPOSAL_SCHEMA_VERSION = "astevolve.evolution.proposal.v1"
PROPOSAL_SCHEMA_VERSION = "astevolve.evolution.proposal.v2"
GENERATION_MANIFEST_SCHEMA_VERSION = "astevolve.evolution.manifest.v1"
SEALED_EVALUATION_SCHEMA_VERSION = "astevolve.evolution.sealed_evaluation.v1"
GENERATION_COMMIT_SCHEMA_VERSION = "astevolve.evolution.commit.v1"

DUAL_AST_REVISION = "dual_ast_revision"
STRATEGY_REVISION = "strategy_revision"
PROPOSAL_KINDS = frozenset({DUAL_AST_REVISION, STRATEGY_REVISION})

EVALUATION_SUCCEEDED = "succeeded"
EVALUATION_FAILED = "failed"
TERMINAL_EVALUATION_STATUSES = frozenset({EVALUATION_SUCCEEDED, EVALUATION_FAILED})

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class EvolutionContractError(ValueError):
    pass


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvolutionContractError(f"{name} must be non-empty")
    return value.strip()


def _require_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvolutionContractError(f"{name} must be a non-negative integer")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise EvolutionContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_json(value: Any) -> str:


    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EvolutionContractError(
            "evolution contracts must contain finite JSON-compatible values"
        ) from exc


def _decode_canonical_mapping(value: str, name: str) -> Dict[str, Any]:
    if not isinstance(value, str) or not value:
        raise EvolutionContractError(f"{name} must contain canonical JSON")
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise EvolutionContractError(f"{name} is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise EvolutionContractError(f"{name} must contain a JSON object")
    if _canonical_json(decoded) != value:
        raise EvolutionContractError(f"{name} must be canonical JSON")
    return decoded


def _digest(namespace: str, value: Any) -> str:
    payload = f"{namespace}\0{_canonical_json(value)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_mapping_fields(
    value: Any,
    *,
    expected: Iterable[str],
    name: str,
) -> Mapping[str, Any]:


    if not isinstance(value, Mapping):
        raise EvolutionContractError(f"{name} must be a mapping")
    expected_fields = frozenset(expected)
    actual_fields = frozenset(value.keys())
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unknown = sorted(actual_fields - expected_fields, key=str)
        raise EvolutionContractError(
            f"{name} has invalid fields; missing={missing}, unknown={unknown}"
        )
    return value


def _require_sequence(value: Any, name: str) -> Tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise EvolutionContractError(f"{name} must be a sequence")
    return tuple(value)


def _proposal_id(generation_id: str, slot: int) -> str:
    return f"{generation_id}:proposal:{slot:04d}"


def _validate_proposal_payload(kind: str, payload_json: str) -> Dict[str, Any]:
    payload = _decode_canonical_mapping(payload_json, "proposal payload")
    try:
        if kind == DUAL_AST_REVISION:
            normalized = ExecutableDualAST.from_mapping(payload).to_dict()
        elif kind == STRATEGY_REVISION:
            normalized = DesignStrategy.from_mapping(payload).to_legacy_dict()
        else:
            raise EvolutionContractError(f"unsupported proposal kind: {kind!r}")
    except EvolutionContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise EvolutionContractError(f"invalid {kind} proposal payload") from exc
    if _canonical_json(normalized) != payload_json:
        raise EvolutionContractError("proposal payload is not in canonical domain form")
    return payload


@dataclass(frozen=True)
class Proposal:


    generation_id: str
    slot: int
    proposal_id: str
    parent_id: str
    kind: str
    payload_json: str
    payload_hash: str
    provenance_json: str
    provenance_hash: str
    proposal_hash: str
    schema_version: str = PROPOSAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.verify()

    @classmethod
    def from_dual_ast(
        cls,
        *,
        generation_id: str,
        slot: int,
        parent_id: str,
        revision: ExecutableDualAST,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "Proposal":
        if not isinstance(revision, ExecutableDualAST):
            raise TypeError("revision must be ExecutableDualAST")
        return cls._create(
            generation_id=generation_id,
            slot=slot,
            parent_id=parent_id,
            kind=DUAL_AST_REVISION,
            payload=revision.to_dict(),
            provenance=provenance,
        )

    @classmethod
    def from_strategy(
        cls,
        *,
        generation_id: str,
        slot: int,
        parent_id: str,
        revision: DesignStrategy,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "Proposal":
        if not isinstance(revision, DesignStrategy):
            raise TypeError("revision must be DesignStrategy")
        return cls._create(
            generation_id=generation_id,
            slot=slot,
            parent_id=parent_id,
            kind=STRATEGY_REVISION,
            payload=revision.to_legacy_dict(),
            provenance=provenance,
        )

    @classmethod
    def _create(
        cls,
        *,
        generation_id: str,
        slot: int,
        parent_id: str,
        kind: str,
        payload: Mapping[str, Any],
        provenance: Optional[Mapping[str, Any]],
    ) -> "Proposal":
        resolved_generation = _required_text(generation_id, "generation_id")
        resolved_slot = _require_non_negative_int(slot, "slot")
        resolved_parent = _required_text(parent_id, "parent_id")
        if kind not in PROPOSAL_KINDS:
            raise EvolutionContractError(f"unsupported proposal kind: {kind!r}")
        payload_json = _canonical_json(dict(payload))
        decoded_payload = _validate_proposal_payload(kind, payload_json)
        payload_hash = _digest(
            "astevolve.evolution.proposal_payload.v1", decoded_payload
        )
        provenance_json = _canonical_json(dict(provenance or {}))
        decoded_provenance = _decode_canonical_mapping(
            provenance_json, "proposal provenance"
        )
        provenance_hash = _digest(
            "astevolve.evolution.proposal_provenance.v1", decoded_provenance
        )
        proposal_id = _proposal_id(resolved_generation, resolved_slot)
        core = {
            "schema_version": PROPOSAL_SCHEMA_VERSION,
            "generation_id": resolved_generation,
            "slot": resolved_slot,
            "proposal_id": proposal_id,
            "parent_id": resolved_parent,
            "kind": kind,
            "payload_hash": payload_hash,
            "provenance_hash": provenance_hash,
        }
        return cls(
            generation_id=resolved_generation,
            slot=resolved_slot,
            proposal_id=proposal_id,
            parent_id=resolved_parent,
            kind=kind,
            payload_json=payload_json,
            payload_hash=payload_hash,
            provenance_json=provenance_json,
            provenance_hash=provenance_hash,
            proposal_hash=_digest("astevolve.evolution.proposal.v2", core),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Proposal":
        if not isinstance(value, Mapping):
            raise EvolutionContractError("proposal must be a mapping")
        schema_version = value.get("schema_version")
        legacy = schema_version == LEGACY_PROPOSAL_SCHEMA_VERSION
        raw = _require_mapping_fields(
            value,
            expected=(
                {
                    "schema_version",
                    "generation_id",
                    "slot",
                    "proposal_id",
                    "parent_id",
                    "kind",
                    "payload",
                    "payload_hash",
                    "proposal_hash",
                }
                if legacy
                else {
                    "schema_version",
                    "generation_id",
                    "slot",
                    "proposal_id",
                    "parent_id",
                    "kind",
                    "payload",
                    "payload_hash",
                    "provenance",
                    "provenance_hash",
                    "proposal_hash",
                }
            ),
            name="proposal",
        )
        payload = raw["payload"]
        if not isinstance(payload, Mapping):
            raise EvolutionContractError("proposal payload must be a mapping")
        provenance = {} if legacy else raw["provenance"]
        if not isinstance(provenance, Mapping):
            raise EvolutionContractError("proposal provenance must be a mapping")
        provenance_json = _canonical_json(dict(provenance))
        provenance_hash = (
            _digest(
                "astevolve.evolution.proposal_provenance.v1",
                _decode_canonical_mapping(provenance_json, "proposal provenance"),
            )
            if legacy
            else raw["provenance_hash"]
        )
        return cls(
            generation_id=raw["generation_id"],
            slot=raw["slot"],
            proposal_id=raw["proposal_id"],
            parent_id=raw["parent_id"],
            kind=raw["kind"],
            payload_json=_canonical_json(dict(payload)),
            payload_hash=raw["payload_hash"],
            provenance_json=provenance_json,
            provenance_hash=provenance_hash,
            proposal_hash=raw["proposal_hash"],
            schema_version=raw["schema_version"],
        )

    def _core(self) -> Dict[str, Any]:
        core = {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "slot": self.slot,
            "proposal_id": self.proposal_id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "payload_hash": self.payload_hash,
        }
        if self.schema_version != LEGACY_PROPOSAL_SCHEMA_VERSION:
            core["provenance_hash"] = self.provenance_hash
        return core

    def verify(self) -> None:
        if self.schema_version not in {
            LEGACY_PROPOSAL_SCHEMA_VERSION,
            PROPOSAL_SCHEMA_VERSION,
        }:
            raise EvolutionContractError(
                f"unsupported proposal schema: {self.schema_version!r}"
            )
        generation_id = _required_text(self.generation_id, "generation_id")
        slot = _require_non_negative_int(self.slot, "slot")
        _required_text(self.parent_id, "parent_id")
        if self.proposal_id != _proposal_id(generation_id, slot):
            raise EvolutionContractError("proposal_id does not match generation slot")
        if self.kind not in PROPOSAL_KINDS:
            raise EvolutionContractError(f"unsupported proposal kind: {self.kind!r}")
        payload = _validate_proposal_payload(self.kind, self.payload_json)
        expected_payload_hash = _digest(
            "astevolve.evolution.proposal_payload.v1", payload
        )
        if _require_sha256(self.payload_hash, "payload_hash") != expected_payload_hash:
            raise EvolutionContractError("proposal payload hash mismatch")
        provenance = _decode_canonical_mapping(
            self.provenance_json, "proposal provenance"
        )
        expected_provenance_hash = _digest(
            "astevolve.evolution.proposal_provenance.v1", provenance
        )
        if (
            _require_sha256(self.provenance_hash, "provenance_hash")
            != expected_provenance_hash
        ):
            raise EvolutionContractError("proposal provenance hash mismatch")
        if self.schema_version == LEGACY_PROPOSAL_SCHEMA_VERSION and provenance:
            raise EvolutionContractError("legacy proposal cannot contain provenance")
        expected_proposal_hash = _digest(
            (
                "astevolve.evolution.proposal.v1"
                if self.schema_version == LEGACY_PROPOSAL_SCHEMA_VERSION
                else "astevolve.evolution.proposal.v2"
            ),
            self._core(),
        )
        if (
            _require_sha256(self.proposal_hash, "proposal_hash")
            != expected_proposal_hash
        ):
            raise EvolutionContractError("proposal seal mismatch")

    def payload(self) -> Dict[str, Any]:


        self.verify()
        return _decode_canonical_mapping(self.payload_json, "proposal payload")

    def provenance(self) -> Dict[str, Any]:


        self.verify()
        return _decode_canonical_mapping(self.provenance_json, "proposal provenance")

    def to_dict(self) -> Dict[str, Any]:
        self.verify()
        value = {
            **self._core(),
            "payload": self.payload(),
            "proposal_hash": self.proposal_hash,
        }
        if self.schema_version != LEGACY_PROPOSAL_SCHEMA_VERSION:
            value["provenance"] = self.provenance()
        return value


@dataclass(frozen=True)
class GenerationManifest:


    generation_id: str
    input_snapshot_hash: str
    proposals: Tuple[Proposal, ...]
    logical_budget: int
    manifest_hash: str
    schema_version: str = GENERATION_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.verify()

    @classmethod
    def create(
        cls,
        *,
        generation_id: str,
        input_snapshot_hash: str,
        proposals: Iterable[Proposal],
    ) -> "GenerationManifest":
        resolved_generation = _required_text(generation_id, "generation_id")
        supplied = tuple(proposals)
        if any(not isinstance(item, Proposal) for item in supplied):
            raise EvolutionContractError("manifest contains a non-Proposal value")
        ordered = tuple(sorted(supplied, key=lambda item: item.slot))
        resolved_input_hash = _require_sha256(
            input_snapshot_hash, "input_snapshot_hash"
        )
        core = {
            "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
            "generation_id": resolved_generation,
            "input_snapshot_hash": resolved_input_hash,
            "logical_budget": len(ordered),
            "proposals": [item.to_dict() for item in ordered],
        }
        return cls(
            generation_id=resolved_generation,
            input_snapshot_hash=resolved_input_hash,
            proposals=ordered,
            logical_budget=len(ordered),
            manifest_hash=_digest("astevolve.evolution.manifest.v1", core),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GenerationManifest":
        raw = _require_mapping_fields(
            value,
            expected={
                "schema_version",
                "generation_id",
                "input_snapshot_hash",
                "proposals",
                "logical_budget",
                "manifest_hash",
            },
            name="generation manifest",
        )
        proposals = tuple(
            Proposal.from_mapping(item)
            for item in _require_sequence(raw["proposals"], "manifest proposals")
        )
        return cls(
            generation_id=raw["generation_id"],
            input_snapshot_hash=raw["input_snapshot_hash"],
            proposals=proposals,
            logical_budget=raw["logical_budget"],
            manifest_hash=raw["manifest_hash"],
            schema_version=raw["schema_version"],
        )

    def _core(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "input_snapshot_hash": self.input_snapshot_hash,
            "logical_budget": self.logical_budget,
            "proposals": [item.to_dict() for item in self.proposals],
        }

    def verify(self) -> None:
        if self.schema_version != GENERATION_MANIFEST_SCHEMA_VERSION:
            raise EvolutionContractError(
                f"unsupported generation manifest schema: {self.schema_version!r}"
            )
        generation_id = _required_text(self.generation_id, "generation_id")
        _require_sha256(self.input_snapshot_hash, "input_snapshot_hash")
        if not isinstance(self.proposals, tuple) or not self.proposals:
            raise EvolutionContractError(
                "generation manifest requires reserved proposals"
            )
        _require_non_negative_int(self.logical_budget, "logical_budget")
        if self.logical_budget != len(self.proposals):
            raise EvolutionContractError(
                "logical_budget must equal the number of reserved proposals"
            )
        expected_slots = tuple(range(len(self.proposals)))
        actual_slots = tuple(item.slot for item in self.proposals)
        if actual_slots != expected_slots:
            raise EvolutionContractError(
                "manifest proposal slots must be contiguous and deterministically ordered"
            )
        proposal_ids = []
        for proposal in self.proposals:
            if not isinstance(proposal, Proposal):
                raise EvolutionContractError("manifest contains a non-Proposal value")
            proposal.verify()
            if proposal.generation_id != generation_id:
                raise EvolutionContractError("proposal belongs to another generation")
            proposal_ids.append(proposal.proposal_id)
        if len(proposal_ids) != len(set(proposal_ids)):
            raise EvolutionContractError("manifest contains duplicate proposal IDs")
        expected_hash = _digest("astevolve.evolution.manifest.v1", self._core())
        if _require_sha256(self.manifest_hash, "manifest_hash") != expected_hash:
            raise EvolutionContractError("generation manifest seal mismatch")

    @property
    def ordered_proposal_ids(self) -> Tuple[str, ...]:
        return tuple(item.proposal_id for item in self.proposals)

    def proposal(self, proposal_id: str) -> Proposal:
        for proposal in self.proposals:
            if proposal.proposal_id == proposal_id:
                return proposal
        raise KeyError(f"proposal {proposal_id!r} is not reserved")

    def to_dict(self) -> Dict[str, Any]:
        self.verify()
        return {**self._core(), "manifest_hash": self.manifest_hash}


def _validated_evidence_payload(payload_json: str) -> Dict[str, Any]:
    payload = _decode_canonical_mapping(payload_json, "evidence payload")
    if set(payload) != {"schema_version", "records"}:
        raise EvolutionContractError("evidence payload has unknown or missing fields")
    if payload["schema_version"] != "astevolve.evidence.v1":
        raise EvolutionContractError("unsupported evidence bundle schema")
    records = payload["records"]
    if not isinstance(records, list):
        raise EvolutionContractError("evidence records must be a list")
    normalized_records = []
    record_fields = {"source", "kind", "available", "value", "details", "warnings"}
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping) or set(raw) != record_fields:
            raise EvolutionContractError(f"evidence record {index} has invalid fields")
        if not isinstance(raw["available"], bool):
            raise EvolutionContractError(
                f"evidence record {index} available must be boolean"
            )
        if not isinstance(raw["details"], Mapping):
            raise EvolutionContractError(
                f"evidence record {index} details must be a mapping"
            )
        if not isinstance(raw["warnings"], list) or any(
            not isinstance(item, str) for item in raw["warnings"]
        ):
            raise EvolutionContractError(
                f"evidence record {index} warnings must be a string list"
            )
        normalized_records.append(
            EvidenceRecord(
                source=_required_text(raw["source"], "evidence source"),
                kind=_required_text(raw["kind"], "evidence kind"),
                available=raw["available"],
                value=raw["value"],
                details=dict(raw["details"]),
                warnings=tuple(raw["warnings"]),
            )
        )
    normalized = EvidenceBundle.of(normalized_records).to_dict()
    if _canonical_json(normalized) != payload_json:
        raise EvolutionContractError("evidence payload is not in canonical domain form")
    return payload


def _validated_report_payload(payload_json: str) -> Dict[str, Any]:
    payload = _decode_canonical_mapping(payload_json, "evaluation report")
    try:
        report = EvaluationReport.from_legacy(payload)
    except (TypeError, ValueError) as exc:
        raise EvolutionContractError("invalid evaluation report") from exc
    if not math.isfinite(report.normalized_score) or not math.isfinite(report.loss):
        raise EvolutionContractError("evaluation scores must be finite")
    if _canonical_json(report.to_legacy_dict()) != payload_json:
        raise EvolutionContractError(
            "evaluation report is not in canonical domain form"
        )
    return payload


@dataclass(frozen=True)
class SealedEvaluation:


    generation_id: str
    proposal_id: str
    status: str
    candidate_id: str
    report_json: str
    report_hash: str
    evidence_json: str
    evidence_hash: str
    failure_reason: str
    evaluation_hash: str
    schema_version: str = SEALED_EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.verify()

    @classmethod
    def success(
        cls,
        *,
        proposal: Proposal,
        candidate_id: str,
        report: EvaluationReport,
        evidence: EvidenceBundle,
    ) -> "SealedEvaluation":
        if not isinstance(proposal, Proposal):
            raise TypeError("proposal must be Proposal")
        if not isinstance(report, EvaluationReport):
            raise TypeError("report must be EvaluationReport")
        if not isinstance(evidence, EvidenceBundle):
            raise TypeError("evidence must be EvidenceBundle")
        proposal.verify()
        return cls._create(
            generation_id=proposal.generation_id,
            proposal_id=proposal.proposal_id,
            status=EVALUATION_SUCCEEDED,
            candidate_id=_required_text(candidate_id, "candidate_id"),
            report_payload=report.to_legacy_dict(),
            evidence_payload=evidence.to_dict(),
            failure_reason="",
        )

    @classmethod
    def failure(
        cls,
        *,
        proposal: Proposal,
        reason: str,
        evidence: Optional[EvidenceBundle] = None,
        candidate_id: str = "",
    ) -> "SealedEvaluation":
        if not isinstance(proposal, Proposal):
            raise TypeError("proposal must be Proposal")
        if evidence is not None and not isinstance(evidence, EvidenceBundle):
            raise TypeError("evidence must be EvidenceBundle or None")
        proposal.verify()
        return cls._create(
            generation_id=proposal.generation_id,
            proposal_id=proposal.proposal_id,
            status=EVALUATION_FAILED,
            candidate_id=str(candidate_id or "").strip(),
            report_payload=None,
            evidence_payload=(evidence or EvidenceBundle()).to_dict(),
            failure_reason=_required_text(reason, "failure reason"),
        )

    @classmethod
    def _create(
        cls,
        *,
        generation_id: str,
        proposal_id: str,
        status: str,
        candidate_id: str,
        report_payload: Optional[Mapping[str, Any]],
        evidence_payload: Mapping[str, Any],
        failure_reason: str,
    ) -> "SealedEvaluation":
        report_json = (
            "" if report_payload is None else _canonical_json(dict(report_payload))
        )
        evidence_json = _canonical_json(dict(evidence_payload))
        report_value = None
        report_hash = ""
        if report_json:
            report_value = _validated_report_payload(report_json)
            report_hash = _digest(
                "astevolve.evolution.evaluation_report.v1", report_value
            )
        evidence_value = _validated_evidence_payload(evidence_json)
        evidence_hash = _digest(
            "astevolve.evolution.evidence_bundle.v1", evidence_value
        )
        core = {
            "schema_version": SEALED_EVALUATION_SCHEMA_VERSION,
            "generation_id": generation_id,
            "proposal_id": proposal_id,
            "status": status,
            "candidate_id": candidate_id,
            "report_hash": report_hash,
            "evidence_hash": evidence_hash,
            "failure_reason": failure_reason,
        }
        return cls(
            generation_id=generation_id,
            proposal_id=proposal_id,
            status=status,
            candidate_id=candidate_id,
            report_json=report_json,
            report_hash=report_hash,
            evidence_json=evidence_json,
            evidence_hash=evidence_hash,
            failure_reason=failure_reason,
            evaluation_hash=_digest("astevolve.evolution.sealed_evaluation.v1", core),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SealedEvaluation":
        raw = _require_mapping_fields(
            value,
            expected={
                "schema_version",
                "generation_id",
                "proposal_id",
                "status",
                "candidate_id",
                "report",
                "report_hash",
                "evidence",
                "evidence_hash",
                "failure_reason",
                "evaluation_hash",
            },
            name="sealed evaluation",
        )
        report = raw["report"]
        if report is not None and not isinstance(report, Mapping):
            raise EvolutionContractError("evaluation report must be a mapping or null")
        evidence = raw["evidence"]
        if not isinstance(evidence, Mapping):
            raise EvolutionContractError("evaluation evidence must be a mapping")
        return cls(
            generation_id=raw["generation_id"],
            proposal_id=raw["proposal_id"],
            status=raw["status"],
            candidate_id=raw["candidate_id"],
            report_json=("" if report is None else _canonical_json(dict(report))),
            report_hash=raw["report_hash"],
            evidence_json=_canonical_json(dict(evidence)),
            evidence_hash=raw["evidence_hash"],
            failure_reason=raw["failure_reason"],
            evaluation_hash=raw["evaluation_hash"],
            schema_version=raw["schema_version"],
        )

    def _core(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "proposal_id": self.proposal_id,
            "status": self.status,
            "candidate_id": self.candidate_id,
            "report_hash": self.report_hash,
            "evidence_hash": self.evidence_hash,
            "failure_reason": self.failure_reason,
        }

    def verify(self) -> None:
        if self.schema_version != SEALED_EVALUATION_SCHEMA_VERSION:
            raise EvolutionContractError(
                f"unsupported sealed evaluation schema: {self.schema_version!r}"
            )
        _required_text(self.generation_id, "generation_id")
        _required_text(self.proposal_id, "proposal_id")
        if self.status not in TERMINAL_EVALUATION_STATUSES:
            raise EvolutionContractError("evaluation status must be terminal")
        if not isinstance(self.candidate_id, str):
            raise EvolutionContractError("candidate_id must be a string")
        if not isinstance(self.failure_reason, str):
            raise EvolutionContractError("failure_reason must be a string")
        evidence = _validated_evidence_payload(self.evidence_json)
        expected_evidence_hash = _digest(
            "astevolve.evolution.evidence_bundle.v1", evidence
        )
        if (
            _require_sha256(self.evidence_hash, "evidence_hash")
            != expected_evidence_hash
        ):
            raise EvolutionContractError("evaluation evidence hash mismatch")
        if self.status == EVALUATION_SUCCEEDED:
            _required_text(self.candidate_id, "candidate_id")
            if self.failure_reason:
                raise EvolutionContractError(
                    "successful evaluation cannot contain a failure reason"
                )
            report = _validated_report_payload(self.report_json)
            expected_report_hash = _digest(
                "astevolve.evolution.evaluation_report.v1", report
            )
            if _require_sha256(self.report_hash, "report_hash") != expected_report_hash:
                raise EvolutionContractError("evaluation report hash mismatch")
        else:
            _required_text(self.failure_reason, "failure reason")
            if self.report_json or self.report_hash:
                raise EvolutionContractError(
                    "failed evaluation cannot contain a success report"
                )
        expected_evaluation_hash = _digest(
            "astevolve.evolution.sealed_evaluation.v1", self._core()
        )
        if (
            _require_sha256(self.evaluation_hash, "evaluation_hash")
            != expected_evaluation_hash
        ):
            raise EvolutionContractError("sealed evaluation hash mismatch")

    def verify_for(self, proposal: Proposal) -> None:
        proposal.verify()
        self.verify()
        if self.generation_id != proposal.generation_id:
            raise EvolutionContractError("evaluation belongs to another generation")
        if self.proposal_id != proposal.proposal_id:
            raise EvolutionContractError("evaluation belongs to another proposal")

    def evidence(self) -> EvidenceBundle:
        self.verify()
        payload = _decode_canonical_mapping(self.evidence_json, "evidence payload")
        return EvidenceBundle.of(
            EvidenceRecord(
                source=item["source"],
                kind=item["kind"],
                available=item["available"],
                value=item["value"],
                details=dict(item["details"]),
                warnings=tuple(item["warnings"]),
            )
            for item in payload["records"]
        )

    def report(self) -> Optional[EvaluationReport]:
        self.verify()
        if not self.report_json:
            return None
        return EvaluationReport.from_legacy(
            _decode_canonical_mapping(self.report_json, "evaluation report")
        )

    def to_dict(self) -> Dict[str, Any]:
        self.verify()
        return {
            **self._core(),
            "report": (
                _decode_canonical_mapping(self.report_json, "evaluation report")
                if self.report_json
                else None
            ),
            "evidence": _decode_canonical_mapping(
                self.evidence_json, "evidence payload"
            ),
            "evaluation_hash": self.evaluation_hash,
        }


@dataclass(frozen=True)
class GenerationCommit:


    generation_id: str
    manifest_hash: str
    logical_budget_used: int
    evaluations: Tuple[SealedEvaluation, ...]
    commit_hash: str
    schema_version: str = GENERATION_COMMIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._verify_self()

    @classmethod
    def create(
        cls,
        manifest: GenerationManifest,
        evaluations: Iterable[SealedEvaluation],
    ) -> "GenerationCommit":
        manifest.verify()
        by_proposal: Dict[str, SealedEvaluation] = {}
        for evaluation in evaluations:
            if not isinstance(evaluation, SealedEvaluation):
                raise EvolutionContractError(
                    "generation outcome must be SealedEvaluation"
                )
            if evaluation.proposal_id in by_proposal:
                raise EvolutionContractError("duplicate terminal evaluation")
            by_proposal[evaluation.proposal_id] = evaluation
        expected = set(manifest.ordered_proposal_ids)
        if set(by_proposal) != expected:
            missing = sorted(expected - set(by_proposal))
            extra = sorted(set(by_proposal) - expected)
            raise EvolutionContractError(
                f"generation barrier incomplete or invalid; missing={missing}, extra={extra}"
            )
        ordered = tuple(
            by_proposal[proposal_id] for proposal_id in manifest.ordered_proposal_ids
        )
        for proposal, evaluation in zip(manifest.proposals, ordered):
            evaluation.verify_for(proposal)
        core = {
            "schema_version": GENERATION_COMMIT_SCHEMA_VERSION,
            "generation_id": manifest.generation_id,
            "manifest_hash": manifest.manifest_hash,
            "logical_budget_used": len(ordered),
            "evaluations": [item.to_dict() for item in ordered],
        }
        return cls(
            generation_id=manifest.generation_id,
            manifest_hash=manifest.manifest_hash,
            logical_budget_used=len(ordered),
            evaluations=ordered,
            commit_hash=_digest("astevolve.evolution.commit.v1", core),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GenerationCommit":
        raw = _require_mapping_fields(
            value,
            expected={
                "schema_version",
                "generation_id",
                "manifest_hash",
                "logical_budget_used",
                "evaluations",
                "commit_hash",
            },
            name="generation commit",
        )
        evaluations = tuple(
            SealedEvaluation.from_mapping(item)
            for item in _require_sequence(raw["evaluations"], "commit evaluations")
        )
        return cls(
            generation_id=raw["generation_id"],
            manifest_hash=raw["manifest_hash"],
            logical_budget_used=raw["logical_budget_used"],
            evaluations=evaluations,
            commit_hash=raw["commit_hash"],
            schema_version=raw["schema_version"],
        )

    def _core(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "manifest_hash": self.manifest_hash,
            "logical_budget_used": self.logical_budget_used,
            "evaluations": [item.to_dict() for item in self.evaluations],
        }

    def _verify_self(self) -> None:
        if self.schema_version != GENERATION_COMMIT_SCHEMA_VERSION:
            raise EvolutionContractError(
                f"unsupported generation commit schema: {self.schema_version!r}"
            )
        _required_text(self.generation_id, "generation_id")
        _require_sha256(self.manifest_hash, "manifest_hash")
        if not isinstance(self.evaluations, tuple) or not self.evaluations:
            raise EvolutionContractError("generation commit requires evaluations")
        _require_non_negative_int(self.logical_budget_used, "logical_budget_used")
        if self.logical_budget_used != len(self.evaluations):
            raise EvolutionContractError(
                "logical_budget_used must equal terminal evaluation count"
            )
        proposal_ids = []
        for evaluation in self.evaluations:
            if not isinstance(evaluation, SealedEvaluation):
                raise EvolutionContractError(
                    "generation commit contains a non-evaluation value"
                )
            evaluation.verify()
            if evaluation.generation_id != self.generation_id:
                raise EvolutionContractError(
                    "commit evaluation belongs to another generation"
                )
            proposal_ids.append(evaluation.proposal_id)
        if len(proposal_ids) != len(set(proposal_ids)):
            raise EvolutionContractError("commit contains duplicate proposal outcomes")
        expected_hash = _digest("astevolve.evolution.commit.v1", self._core())
        if _require_sha256(self.commit_hash, "commit_hash") != expected_hash:
            raise EvolutionContractError("generation commit hash mismatch")

    def verify(self, manifest: GenerationManifest) -> None:
        manifest.verify()
        self._verify_self()
        if self.generation_id != manifest.generation_id:
            raise EvolutionContractError("commit belongs to another generation")
        if self.manifest_hash != manifest.manifest_hash:
            raise EvolutionContractError("commit manifest hash mismatch")
        if self.logical_budget_used != manifest.logical_budget:
            raise EvolutionContractError("commit did not consume the reserved budget")
        if tuple(item.proposal_id for item in self.evaluations) != (
            manifest.ordered_proposal_ids
        ):
            raise EvolutionContractError("commit outcome order differs from manifest")
        for proposal, evaluation in zip(manifest.proposals, self.evaluations):
            evaluation.verify_for(proposal)

    @property
    def succeeded(self) -> int:
        return sum(item.status == EVALUATION_SUCCEEDED for item in self.evaluations)

    @property
    def failed(self) -> int:
        return sum(item.status == EVALUATION_FAILED for item in self.evaluations)

    def to_dict(self) -> Dict[str, Any]:
        self._verify_self()
        return {**self._core(), "commit_hash": self.commit_hash}


__all__ = [
    "DUAL_AST_REVISION",
    "EVALUATION_FAILED",
    "EVALUATION_SUCCEEDED",
    "EvolutionContractError",
    "GenerationCommit",
    "GenerationManifest",
    "Proposal",
    "STRATEGY_REVISION",
    "SealedEvaluation",
]
