

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Sequence


CHECK_REGISTRY_VERSION = "astevolve.case_preflight_registry.v1"
CHECK_RESULT_VERSION = "astevolve.case_preflight_check.v1"
ENGINE_CAPABILITY_REPORT_VERSION = "astevolve.engine_capability_report.v1"
CASE_READINESS_REPORT_VERSION = "astevolve.case_readiness_report.v1"
CASE_MANIFEST_VERSION = "astevolve.case_manifest.v2"
PREFLIGHT_MANIFEST_VERSION = "astevolve.case_preflight_manifest.v2"

CPU_ONLY_CLAIMS = {
    "cpu_only": True,
    "gpu_executed": False,
    "biological_success": False,
}


class CasePreflightError(ValueError):


    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)


class ProbeTimeoutError(TimeoutError):
    pass


def _fail(code: str, detail: Any = "") -> None:
    raise CasePreflightError(code, str(detail))


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
        _fail("not_canonical_json", exc)


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _digest(domain: str, value: Any) -> str:
    preimage = domain.encode("utf-8") + b"\0" + _canonical_json(value).encode("utf-8")
    return hashlib.sha256(preimage).hexdigest()


def _file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        _fail("input_unreadable", f"{path}:{exc}")


def _closed(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = {str(key) for key in value}
    unknown = sorted(observed - expected)
    missing = sorted(expected - observed)
    if unknown:
        _fail("unknown_fields", f"{label}:{','.join(unknown)}")
    if missing:
        _fail("fields_missing", f"{label}:{','.join(missing)}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("mapping_required", label)
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("text_required", label)
    return value.strip()


def _string_list(value: Any, label: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        _fail("list_required", label)
    result = [_text(item, label) for item in value]
    if len(result) != len(set(result)):
        _fail("duplicate_values", label)
    return result


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        _fail("boolean_required", label)
    return value


def _sha256_text(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        _fail("sha256_invalid", label)
    return text


@dataclass(frozen=True)
class CheckSpec:
    check_id: str
    level: str
    scope: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {
            "check_id": self.check_id,
            "level": self.level,
            "scope": self.scope,
            "description": self.description,
        }


CHECK_REGISTRY: tuple[CheckSpec, ...] = (
    CheckSpec("engine.operator_registry", "C0", "engine", "All executable operator names resolve to declared handlers."),
    CheckSpec("engine.evaluator_registry", "C0", "engine", "Evaluator bindings and plugin resolution are explicit."),
    CheckSpec("engine.memory_isolation", "C1", "engine", "Run, lineage, case, and external-knowledge boundaries are closed."),
    CheckSpec("engine.causal_identity_history", "C2", "engine", "Causal proposal fields, three identities, occurrence registry, and exact cache are live."),
    CheckSpec("engine.generator_contract", "C2", "engine", "Production and random-baseline generators share a validated hard contract."),
    CheckSpec("engine.dual_ast_runtime", "C2", "engine", "Dual-AST semantic changes alter execution while preserving the residue mask."),
    CheckSpec("engine.outer_effective_phenotype", "C2", "engine", "Outer diversity is based on effective phenotype rather than source formatting."),
    CheckSpec("engine.parent_comparison", "C2", "engine", "Parent/root comparison is feasibility-first and deterministic."),
    CheckSpec("engine.recorded_evidence_parser", "C3", "engine", "Recorded evidence distinguishes agreement, conflict, and missingness."),
    CheckSpec("case.manifest_schema", "C0", "case", "The case manifest and its preflight declaration are closed."),
    CheckSpec("case.state_schema_compile", "C0", "case", "State and strategy compile with no silent or unconsumed field."),
    CheckSpec("case.mask_explanation", "C1", "case", "Every residue is explained as mutable or frozen without overlap."),
    CheckSpec("case.operator_execution", "C2", "case", "Every declared case action resolves to an executable operator."),
    CheckSpec("case.hard_gate_boundary", "C2", "case", "The CPU evaluator passes and fails on opposite sides of a hard gate."),
    CheckSpec("case.evaluator_ast_wiring", "C2", "case", "Functional nodes, mapping edges, evaluator terms, states, and thresholds agree."),
    CheckSpec("case.recorded_evidence", "C3", "case", "The case evidence pack contains accept, conflict-abstain, and missing-abstain outcomes."),
)


def check_registry_hash() -> str:
    return _digest(
        CHECK_REGISTRY_VERSION,
        [spec.to_dict() for spec in CHECK_REGISTRY],
    )


_CHECK_RESULT_FIELDS = {
    "schema_version",
    "check_id",
    "level",
    "scope",
    "status",
    "summary",
    "evidence",
    "error_code",
    "result_hash",
}


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    level: str
    scope: str
    status: str
    summary: str
    evidence: Mapping[str, Any]
    error_code: Optional[str]
    result_hash: str
    schema_version: str = CHECK_RESULT_VERSION

    @classmethod
    def create(
        cls,
        spec: CheckSpec,
        *,
        status: str,
        summary: str,
        evidence: Optional[Mapping[str, Any]] = None,
        error_code: Optional[str] = None,
    ) -> "CheckResult":
        if status not in {"passed", "failed"}:
            _fail("check_status_invalid", status)
        if (status == "passed") != (error_code is None):
            _fail("check_error_status_mismatch", spec.check_id)
        core = {
            "schema_version": CHECK_RESULT_VERSION,
            "check_id": spec.check_id,
            "level": spec.level,
            "scope": spec.scope,
            "status": status,
            "summary": _text(summary, "check.summary"),
            "evidence": _json_copy(dict(evidence or {})),
            "error_code": error_code,
        }
        return cls(
            check_id=spec.check_id,
            level=spec.level,
            scope=spec.scope,
            status=status,
            summary=core["summary"],
            evidence=core["evidence"],
            error_code=error_code,
            result_hash=_digest(CHECK_RESULT_VERSION, core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "check_id": self.check_id,
            "level": self.level,
            "scope": self.scope,
            "status": self.status,
            "summary": self.summary,
            "evidence": _json_copy(self.evidence),
            "error_code": self.error_code,
            "result_hash": self.result_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CheckResult":
        raw = _mapping(value, "check_result")
        _closed(raw, _CHECK_RESULT_FIELDS, "check_result")
        if raw.get("schema_version") != CHECK_RESULT_VERSION:
            _fail("schema_version_invalid", "check_result")
        spec = next((item for item in CHECK_REGISTRY if item.check_id == raw.get("check_id")), None)
        if spec is None:
            _fail("check_id_unknown", raw.get("check_id"))
        if raw.get("level") != spec.level or raw.get("scope") != spec.scope:
            _fail("check_registry_mismatch", spec.check_id)
        restored = cls.create(
            spec,
            status=str(raw.get("status")),
            summary=raw.get("summary"),
            evidence=_mapping(raw.get("evidence"), "check.evidence"),
            error_code=raw.get("error_code"),
        )
        if raw.get("result_hash") != restored.result_hash:
            _fail("hash_mismatch", f"check:{spec.check_id}")
        return restored


_ENGINE_FIELDS = {
    "schema_version",
    "registry_version",
    "registry_hash",
    "checks",
    "ready",
    "failure_codes",
    "claims",
    "report_hash",
}


@dataclass(frozen=True)
class EngineCapabilityReport:
    checks: tuple[CheckResult, ...]
    ready: bool
    failure_codes: tuple[str, ...]
    claims: Mapping[str, bool]
    report_hash: str
    registry_version: str = CHECK_REGISTRY_VERSION
    registry_hash: str = ""
    schema_version: str = ENGINE_CAPABILITY_REPORT_VERSION

    @classmethod
    def create(cls, checks: Sequence[CheckResult]) -> "EngineCapabilityReport":
        expected = [spec.check_id for spec in CHECK_REGISTRY if spec.scope == "engine"]
        observed = [item.check_id for item in checks]
        if observed != expected:
            _fail("engine_check_set_mismatch", f"expected={expected},observed={observed}")
        failures = tuple(
            sorted({str(item.error_code) for item in checks if item.error_code})
        )
        core = {
            "schema_version": ENGINE_CAPABILITY_REPORT_VERSION,
            "registry_version": CHECK_REGISTRY_VERSION,
            "registry_hash": check_registry_hash(),
            "checks": [item.to_dict() for item in checks],
            "ready": not failures,
            "failure_codes": list(failures),
            "claims": dict(CPU_ONLY_CLAIMS),
        }
        return cls(
            tuple(checks),
            not failures,
            failures,
            dict(CPU_ONLY_CLAIMS),
            _digest(ENGINE_CAPABILITY_REPORT_VERSION, core),
            registry_hash=check_registry_hash(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "registry_hash": self.registry_hash,
            "checks": [item.to_dict() for item in self.checks],
            "ready": self.ready,
            "failure_codes": list(self.failure_codes),
            "claims": dict(self.claims),
            "report_hash": self.report_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EngineCapabilityReport":
        raw = _mapping(value, "engine_report")
        _closed(raw, _ENGINE_FIELDS, "engine_report")
        if raw.get("schema_version") != ENGINE_CAPABILITY_REPORT_VERSION:
            _fail("schema_version_invalid", "engine_report")
        if raw.get("registry_version") != CHECK_REGISTRY_VERSION:
            _fail("registry_version_mismatch", "engine_report")
        checks_raw = raw.get("checks")
        if not isinstance(checks_raw, list):
            _fail("list_required", "engine_report.checks")
        restored = cls.create([CheckResult.from_mapping(item) for item in checks_raw])
        if raw.get("registry_hash") != restored.registry_hash:
            _fail("registry_hash_mismatch", "engine_report")
        if raw.get("ready") is not restored.ready:
            _fail("report_consistency_error", "engine.ready")
        if raw.get("failure_codes") != list(restored.failure_codes):
            _fail("report_consistency_error", "engine.failure_codes")
        if raw.get("claims") != dict(CPU_ONLY_CLAIMS):
            _fail("claim_mismatch", "engine_report")
        if raw.get("report_hash") != restored.report_hash:
            _fail("hash_mismatch", "engine_report")
        return restored


_CASE_REPORT_FIELDS = {
    "schema_version",
    "case_id",
    "manifest_hash",
    "input_hashes",
    "required_checks",
    "engine_report",
    "case_checks",
    "failure_codes",
    "ready_for_smoke",
    "claims",
    "report_hash",
}


@dataclass(frozen=True)
class CaseReadinessReport:
    case_id: str
    manifest_hash: str
    input_hashes: Mapping[str, str]
    required_checks: tuple[str, ...]
    engine_report: EngineCapabilityReport
    case_checks: tuple[CheckResult, ...]
    failure_codes: tuple[str, ...]
    ready_for_smoke: bool
    claims: Mapping[str, bool]
    report_hash: str
    schema_version: str = CASE_READINESS_REPORT_VERSION

    @classmethod
    def create(
        cls,
        *,
        case_id: str,
        manifest_hash: str,
        input_hashes: Mapping[str, str],
        required_checks: Sequence[str],
        engine_report: EngineCapabilityReport,
        case_checks: Sequence[CheckResult],
        extra_failure_codes: Sequence[str] = (),
    ) -> "CaseReadinessReport":
        expected_case = [spec.check_id for spec in CHECK_REGISTRY if spec.scope == "case"]
        if [item.check_id for item in case_checks] != expected_case:
            _fail("case_check_set_mismatch")
        required = tuple(str(item) for item in required_checks)
        expected_all = tuple(sorted(spec.check_id for spec in CHECK_REGISTRY))
        if required != expected_all:
            _fail("required_check_set_mismatch")
        failures = tuple(
            sorted(
                {
                    *engine_report.failure_codes,
                    *(str(item.error_code) for item in case_checks if item.error_code),
                    *(str(item) for item in extra_failure_codes if str(item)),
                }
            )
        )
        ready = not failures and engine_report.ready
        clean_hashes = {
            str(name): _sha256_text(value, f"input_hashes.{name}")
            for name, value in sorted(input_hashes.items())
        }
        core = {
            "schema_version": CASE_READINESS_REPORT_VERSION,
            "case_id": _text(case_id, "case_id"),
            "manifest_hash": _sha256_text(manifest_hash, "manifest_hash"),
            "input_hashes": clean_hashes,
            "required_checks": list(required),
            "engine_report": engine_report.to_dict(),
            "case_checks": [item.to_dict() for item in case_checks],
            "failure_codes": list(failures),
            "ready_for_smoke": ready,
            "claims": dict(CPU_ONLY_CLAIMS),
        }
        return cls(
            core["case_id"],
            core["manifest_hash"],
            clean_hashes,
            required,
            engine_report,
            tuple(case_checks),
            failures,
            ready,
            dict(CPU_ONLY_CLAIMS),
            _digest(CASE_READINESS_REPORT_VERSION, core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "manifest_hash": self.manifest_hash,
            "input_hashes": dict(self.input_hashes),
            "required_checks": list(self.required_checks),
            "engine_report": self.engine_report.to_dict(),
            "case_checks": [item.to_dict() for item in self.case_checks],
            "failure_codes": list(self.failure_codes),
            "ready_for_smoke": self.ready_for_smoke,
            "claims": dict(self.claims),
            "report_hash": self.report_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CaseReadinessReport":
        raw = _mapping(value, "case_readiness_report")
        _closed(raw, _CASE_REPORT_FIELDS, "case_readiness_report")
        if raw.get("schema_version") != CASE_READINESS_REPORT_VERSION:
            _fail("schema_version_invalid", "case_readiness_report")
        sealed_core = {
            key: _json_copy(raw[key])
            for key in _CASE_REPORT_FIELDS
            if key != "report_hash"
        }
        if raw.get("report_hash") != _digest(
            CASE_READINESS_REPORT_VERSION, sealed_core
        ):
            _fail("hash_mismatch", "case_readiness_report")
        case_raw = raw.get("case_checks")
        if not isinstance(case_raw, list):
            _fail("list_required", "case_checks")
        restored = cls.create(
            case_id=raw.get("case_id"),
            manifest_hash=raw.get("manifest_hash"),
            input_hashes=_mapping(raw.get("input_hashes"), "input_hashes"),
            required_checks=_string_list(raw.get("required_checks"), "required_checks"),
            engine_report=EngineCapabilityReport.from_mapping(
                _mapping(raw.get("engine_report"), "engine_report")
            ),
            case_checks=[CheckResult.from_mapping(item) for item in case_raw],
        )
        if raw.get("failure_codes") != list(restored.failure_codes):
            _fail("report_consistency_error", "case.failure_codes")
        if raw.get("ready_for_smoke") is not restored.ready_for_smoke:
            _fail("report_consistency_error", "case.ready_for_smoke")
        if raw.get("claims") != dict(CPU_ONLY_CLAIMS):
            _fail("claim_mismatch", "case_report")
        if raw.get("report_hash") != restored.report_hash:
            _fail("hash_mismatch", "case_readiness_report")
        return restored


_MANIFEST_FIELDS = {
    "schema_version",
    "case_id",
    "name",
    "task_type",
    "lifecycle",
    "design_state_path",
    "memory_path",
    "entry_program",
    "test_evaluator_path",
    "recorded_evidence_path",
    "launcher_defaults",
    "preflight",
    "notes",
}
_MANIFEST_OPTIONAL_FIELDS = {
    "immutable_reference",


    "config_path",
    "outer_evaluator_path",
    "scientific_spec",
}
_IMMUTABLE_REFERENCE_VERSION = "astevolve.immutable_sequence_reference.v1"
_IMMUTABLE_REFERENCE_FIELDS = {
    "schema_version",
    "reference_id",
    "candidate_role",
    "chain_id",
    "sequence_path",
    "sequence_length",
    "fasta_sha256",
    "sequence_sha256",
    "sequence_bundle_hash",
    "comparison_policy",
}
_IMMUTABLE_REFERENCE_ROLE = "immutable_reference"
_IMMUTABLE_REFERENCE_COMPARISON_POLICY = (
    "fixed_for_entire_run_and_never_replaced_by_dynamic_parent"
)
_PREFLIGHT_FIELDS = {
    "schema_version",
    "required_checks",
    "check_registry_hash",
    "recorded_evidence_sha256",
    "timeout_seconds",
    "intended_stage",
    "claims",
}
_INPUT_PATH_FIELDS = {
    "design_state": "design_state_path",
    "memory": "memory_path",
    "entry_program": "entry_program",
    "test_evaluator": "test_evaluator_path",
    "recorded_evidence": "recorded_evidence_path",
}


def _validate_immutable_reference_declaration(value: Any) -> Mapping[str, Any]:


    reference = _mapping(value, "manifest.immutable_reference")
    _closed(
        reference,
        _IMMUTABLE_REFERENCE_FIELDS,
        "manifest.immutable_reference",
    )
    if reference.get("schema_version") != _IMMUTABLE_REFERENCE_VERSION:
        _fail("schema_version_invalid", "manifest.immutable_reference")
    _text(reference.get("reference_id"), "immutable_reference.reference_id")
    if reference.get("candidate_role") != _IMMUTABLE_REFERENCE_ROLE:
        _fail("immutable_reference_candidate_role_invalid")
    chain_id = _text(reference.get("chain_id"), "immutable_reference.chain_id")
    if any(character.isspace() for character in chain_id):
        _fail("immutable_reference_chain_id_invalid")
    _text(reference.get("sequence_path"), "immutable_reference.sequence_path")
    length = reference.get("sequence_length")
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        _fail("immutable_reference_sequence_length_invalid")
    _sha256_text(
        reference.get("fasta_sha256"),
        "immutable_reference.fasta_sha256",
    )
    _sha256_text(
        reference.get("sequence_sha256"),
        "immutable_reference.sequence_sha256",
    )
    sequence_bundle_hash = _text(
        reference.get("sequence_bundle_hash"),
        "immutable_reference.sequence_bundle_hash",
    )
    sequence_hash_prefix = "sequence_sha256:"
    if not sequence_bundle_hash.startswith(sequence_hash_prefix):
        _fail("immutable_reference_sequence_bundle_hash_invalid")
    _sha256_text(
        sequence_bundle_hash[len(sequence_hash_prefix):],
        "immutable_reference.sequence_bundle_hash",
    )
    if reference.get("comparison_policy") != _IMMUTABLE_REFERENCE_COMPARISON_POLICY:
        _fail("immutable_reference_comparison_policy_invalid")
    return reference


def _read_single_fasta(path: Path) -> str:


    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        _fail("immutable_reference_fasta_unreadable", f"{path}:{exc}")
    headers = 0
    sequence_parts: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            headers += 1
            if headers != 1 or len(line) == 1 or sequence_parts:
                _fail("immutable_reference_fasta_record_count_invalid", path)
            continue
        if headers != 1 or any(character.isspace() for character in line):
            _fail("immutable_reference_fasta_invalid", path)
        sequence_parts.append(line)
    if headers != 1 or not sequence_parts:
        _fail("immutable_reference_fasta_record_count_invalid", path)
    sequence = "".join(sequence_parts)
    if any(character < "A" or character > "Z" for character in sequence):
        _fail("immutable_reference_sequence_invalid", path)
    return sequence


def _resolve_immutable_reference(
    manifest_path: Path,
    declaration: Mapping[str, Any],
) -> Path:


    case_root = manifest_path.parent.resolve()
    raw_path = Path(
        _text(declaration.get("sequence_path"), "immutable_reference.sequence_path")
    ).expanduser()
    resolved = raw_path.resolve() if raw_path.is_absolute() else (case_root / raw_path).resolve()
    try:
        resolved.relative_to(case_root)
    except ValueError:
        _fail("immutable_reference_path_outside_case_root", resolved)
    if not resolved.is_file():
        _fail("input_missing", f"immutable_reference:{resolved}")
    if _file_hash(resolved) != declaration["fasta_sha256"]:
        _fail("immutable_reference_fasta_hash_mismatch", resolved)
    sequence = _read_single_fasta(resolved)
    if len(sequence) != declaration["sequence_length"]:
        _fail("immutable_reference_sequence_length_mismatch", resolved)
    if hashlib.sha256(sequence.encode("utf-8")).hexdigest() != declaration["sequence_sha256"]:
        _fail("immutable_reference_sequence_hash_mismatch", resolved)
    from engine.experiment_identity import SequenceBundleIdentity

    canonical_hash = SequenceBundleIdentity.create(
        {str(declaration["chain_id"]): sequence}
    ).sequence_bundle_hash
    if canonical_hash != declaration["sequence_bundle_hash"]:
        _fail("immutable_reference_sequence_bundle_hash_mismatch", resolved)
    return resolved


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail("manifest_schema_invalid", exc)
    raw = _mapping(value, "manifest")
    try:
        observed_fields = {str(key) for key in raw}
        unknown_fields = sorted(
            observed_fields - _MANIFEST_FIELDS - _MANIFEST_OPTIONAL_FIELDS
        )
        missing_fields = sorted(_MANIFEST_FIELDS - observed_fields)
        if unknown_fields:
            _fail("unknown_fields", f"manifest:{','.join(unknown_fields)}")
        if missing_fields:
            _fail("fields_missing", f"manifest:{','.join(missing_fields)}")
        if raw.get("schema_version") != CASE_MANIFEST_VERSION:
            _fail("schema_version_invalid", "manifest")
        for field in (
            "case_id", "name", "task_type", "design_state_path",
            "memory_path", "entry_program", "test_evaluator_path",
            "recorded_evidence_path", "notes",
        ):
            _text(raw.get(field), f"manifest.{field}")
        lifecycle = raw.get("lifecycle")
        if isinstance(lifecycle, str):
            _text(lifecycle, "manifest.lifecycle")
        else:
            lifecycle = _mapping(lifecycle, "manifest.lifecycle")
            _closed(
                lifecycle,
                {
                    "schema_version", "status", "default_eligible",
                    "smoke_eligible", "formal_eligible", "explicit_run_only",
                },
                "manifest.lifecycle",
            )
            if lifecycle.get("schema_version") != "astevolve.case_lifecycle.v1":
                _fail("schema_version_invalid", "manifest.lifecycle")
            _text(lifecycle.get("status"), "manifest.lifecycle.status")
            for field in (
                "default_eligible", "smoke_eligible", "formal_eligible",
                "explicit_run_only",
            ):
                _strict_bool(lifecycle.get(field), f"manifest.lifecycle.{field}")
        defaults = _mapping(raw.get("launcher_defaults"), "launcher_defaults")
        _closed(defaults, {"inner_iterations", "progen_weight"}, "launcher_defaults")
        iterations = defaults.get("inner_iterations")
        weight = defaults.get("progen_weight")
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
            _fail("launcher_default_invalid", "inner_iterations")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
            _fail("launcher_default_invalid", "progen_weight")
        preflight = _mapping(raw.get("preflight"), "preflight")
        _closed(preflight, _PREFLIGHT_FIELDS, "preflight")
        if preflight.get("schema_version") != PREFLIGHT_MANIFEST_VERSION:
            _fail("schema_version_invalid", "preflight")
        required = _string_list(preflight.get("required_checks"), "required_checks")
        expected = sorted(spec.check_id for spec in CHECK_REGISTRY)
        if required != expected:
            _fail("required_check_set_mismatch")
        if preflight.get("check_registry_hash") != check_registry_hash():
            _fail("registry_hash_mismatch", "manifest")
        _sha256_text(preflight.get("recorded_evidence_sha256"), "recorded_evidence_sha256")
        timeout = preflight.get("timeout_seconds")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or float(timeout) <= 0:
            _fail("timeout_invalid", timeout)
        if preflight.get("intended_stage") != "cpu_contract_smoke":
            _fail("intended_stage_invalid", preflight.get("intended_stage"))
        claims = _mapping(preflight.get("claims"), "preflight.claims")
        _closed(claims, set(CPU_ONLY_CLAIMS), "preflight.claims")
        if dict(claims) != CPU_ONLY_CLAIMS:
            _fail("claim_mismatch", "manifest")
        if "immutable_reference" in raw:
            _validate_immutable_reference_declaration(raw["immutable_reference"])
    except CasePreflightError as exc:
        if exc.code == "manifest_schema_invalid":
            raise
        _fail("manifest_schema_invalid", exc)
    return _json_copy(raw)


def _resolve_inputs(manifest_path: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, field in _INPUT_PATH_FIELDS.items():
        raw = Path(_text(manifest.get(field), field)).expanduser()
        resolved = raw.resolve() if raw.is_absolute() else (manifest_path.parent / raw).resolve()
        if not resolved.is_file():
            _fail("input_missing", f"{name}:{resolved}")
        paths[name] = resolved
    if "immutable_reference" in manifest:
        declaration = _validate_immutable_reference_declaration(
            manifest["immutable_reference"]
        )
        paths["immutable_reference"] = _resolve_immutable_reference(
            manifest_path,
            declaration,
        )
    expected_evidence = manifest["preflight"]["recorded_evidence_sha256"]
    if _file_hash(paths["recorded_evidence"]) != expected_evidence:
        _fail("recorded_evidence_hash_mismatch", paths["recorded_evidence"])
    return paths


def execute_probe_with_timeout(
    probe: Callable[[], Any], *, timeout_seconds: float
) -> Any:


    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise ValueError("timeout_seconds must be a positive finite number")
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be a positive finite number")
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise ProbeTimeoutError("hard probe timeout is unavailable on this platform")
    if threading.current_thread() is not threading.main_thread():
        raise ProbeTimeoutError("hard probe timeout requires the process main thread")

    def alarm_handler(_signum: int, _frame: Any) -> None:
        raise ProbeTimeoutError(f"probe exceeded {timeout:g} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return probe()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _load_module(path: Path, label: str) -> ModuleType:
    name = f"_astevolve_preflight_{label}_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _fail("module_load_failed", label)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@dataclass
class _ProbeContext:
    manifest_path: Path
    report_path: Path
    manifest: dict[str, Any]
    paths: dict[str, Path]
    entry_module: ModuleType
    evaluator_module: ModuleType
    strategy: dict[str, Any]
    prepared: Any
    recorded_evidence: dict[str, Any]


def _prepare_context(
    manifest_path: Path,
    report_path: Path,
    manifest: dict[str, Any],
    paths: dict[str, Path],
) -> _ProbeContext:
    from engine.case_builder import prepare_case_inputs

    entry = _load_module(paths["entry_program"], "entry")
    evaluator = _load_module(paths["test_evaluator"], "evaluator")
    strategy_factory = getattr(entry, "runtime_strategy", None) or getattr(entry, "propose_strategy", None)
    if not callable(strategy_factory):
        _fail("entry_strategy_missing")
    strategy = strategy_factory()
    if not isinstance(strategy, dict):
        _fail("entry_strategy_invalid")
    prepared = prepare_case_inputs(
        strategy,
        design_state_path=str(paths["design_state"]),
        memory_path=str(paths["memory"]),
    )
    evidence = json.loads(paths["recorded_evidence"].read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        _fail("recorded_evidence_invalid", "mapping required")
    return _ProbeContext(
        manifest_path,
        report_path,
        manifest,
        paths,
        entry,
        evaluator,
        strategy,
        prepared,
        evidence,
    )


ProbeResult = tuple[str, Mapping[str, Any]]
Probe = Callable[[_ProbeContext], ProbeResult]


def _probe_operator_registry(_context: _ProbeContext) -> ProbeResult:
    from astevolve.search.operator_registry import (
        EXECUTABLE_STATUSES,
        NODE_HANDLER_IDS,
        OPERATOR_REGISTRY,
        REGISTRY_VERSION,
        executable_operator_names,
        require_operator,
    )

    executable = sorted(executable_operator_names("node"))
    if not executable:
        raise AssertionError("operator registry contains no executable node operators")
    handlers = []
    for name in executable:
        spec = require_operator(name, mode="node")
        if spec.status not in EXECUTABLE_STATUSES or spec.node_handler not in NODE_HANDLER_IDS:
            raise AssertionError(f"operator {name} has no registered node handler")
        handlers.append(str(spec.node_handler))
    unresolved_enabled = sorted(
        name
        for name, spec in OPERATOR_REGISTRY.items()
        if spec.status in EXECUTABLE_STATUSES and spec.node_handler is None
    )
    if unresolved_enabled:
        raise AssertionError(f"enabled operators lack node handlers: {unresolved_enabled}")
    return "Executable operator registry is closed and handler-complete.", {
        "registry_version": REGISTRY_VERSION,
        "executable_operator_count": len(executable),
        "executable_operators": executable,
        "handler_ids": sorted(set(handlers)),
    }


def _probe_evaluator_registry(context: _ProbeContext) -> ProbeResult:
    prepared = context.prepared
    capabilities = {
        str(provider): {str(term) for term in terms}
        for provider, terms in prepared.design_state.get("evaluator_capabilities", {}).items()
    }
    intents = prepared.executable_node_plan.measurement_intents
    if not intents:
        raise AssertionError("no functional measurement intents were compiled")
    for intent in intents:
        if intent.term_name not in capabilities.get(intent.evaluator_id, set()):
            raise AssertionError(f"unresolved evaluator term {intent.evaluator_id}/{intent.term_name}")
        if intent.kind == "hard_constraint" and intent.gate is None:
            raise AssertionError(f"hard constraint {intent.functional_node_id} lacks gate")
        if intent.missing_policy not in {"fail", "abstain"}:
            raise AssertionError("measurement missing policy is unresolved")
    plugins = prepared.score_config.get("plugin_resolution", {})
    if plugins.get("unknown") or plugins.get("failed"):
        raise AssertionError("evaluator plugin resolution contains unknown or failed entries")
    return "Evaluator terms, gates, and missing-data policies resolve exactly.", {
        "measurement_count": len(intents),
        "bindings": sorted(f"{item.evaluator_id}/{item.term_name}" for item in intents),
        "hard_gate_count": sum(item.gate is not None for item in intents),
        "plugin_source": plugins.get("source"),
        "resolved_plugins": list(plugins.get("resolved") or []),
    }


def _probe_memory_isolation(context: _ProbeContext) -> ProbeResult:
    from astevolve.search.run_memory import InnerRunMemory
    from engine.external_knowledge_policy import build_external_knowledge_policy
    from engine.memory_policy import MemoryPolicyConfig, MemoryPolicyError, MemoryScope

    first_document = context.prepared.memory_snapshot.materialize()
    second_document = context.prepared.memory_snapshot.materialize()
    first_document["_probe_mutation"] = True
    if "_probe_mutation" in second_document:
        raise AssertionError("memory snapshot materializations share state")
    scope = MemoryScope(case_id=context.manifest["case_id"], run_id="run-a", lineage_id="lineage-a")
    wrong_scopes = (
        MemoryScope(case_id="other-case", run_id="run-a", lineage_id="lineage-a"),
        MemoryScope(case_id=context.manifest["case_id"], run_id="run-b", lineage_id="lineage-a"),
        MemoryScope(case_id=context.manifest["case_id"], run_id="run-a", lineage_id="lineage-b"),
    )
    rejected = 0
    for level, wrong in zip(("case", "run", "lineage"), wrong_scopes):
        try:
            scope.require_compatible(wrong, level=level)
        except MemoryPolicyError:
            rejected += 1
    if rejected != 3:
        raise AssertionError("one or more memory-scope leaks were accepted")
    run_one = InnerRunMemory(scope=scope, run_instance_id="one")
    run_two = InnerRunMemory(scope=scope, run_instance_id="two")
    if not run_one.claim_sequence(context.prepared.template_sequences).is_new:
        raise AssertionError("first run-local occurrence was not new")
    if run_one.claim_sequence(context.prepared.template_sequences).is_new:
        raise AssertionError("same-run duplicate was not detected")
    if not run_two.claim_sequence(context.prepared.template_sequences).is_new:
        raise AssertionError("sibling run observed another run's transient state")
    policy = build_external_knowledge_policy(
        {"external_kb_enabled": True, "external_kb_path": "disabled-external-kb-sentinel"},
        environ={"ASTEVOLVE_EXTERNAL_KB_PATH": "disabled-external-kb-sentinel"},
    )
    if policy["external_kb_used"] or policy["provider_loaded"]:
        raise AssertionError("formal policy permitted an external KB read")
    memory_policy = MemoryPolicyConfig()
    if memory_policy.may_read_adaptive_prior or memory_policy.inner_state_scope != "run":
        raise AssertionError("default memory policy is not run-scoped and read-disabled")
    return "Run memory is isolated and external knowledge remains disabled.", {
        "scope_rejections": rejected,
        "inner_state_scope": memory_policy.inner_state_scope,
        "adaptive_prior_mode": memory_policy.adaptive_prior_mode,
        "external_kb_used": policy["external_kb_used"],
        "provider_loaded": policy["provider_loaded"],
    }


def _probe_causal_identity_history(context: _ProbeContext) -> ProbeResult:
    from engine.experiment_identity import (
        CodeIdentity,
        EffectiveContractIdentity,
        EvaluatorDescriptor,
        ExactEvaluationKey,
        SequenceBundleIdentity,
    )
    from engine.experiment_registry import ExperimentRegistry

    code = CodeIdentity.create(context.paths["entry_program"].read_bytes())
    contract = EffectiveContractIdentity.create(context.prepared.effective_search_contract)
    sequence = SequenceBundleIdentity.create(context.prepared.template_sequences)
    graph_patch = context.prepared.graph_patch.to_dict()
    executable_fields = [
        row for row in graph_patch["fields"]
        if row.get("disposition") in {"compiled", "executed"} and row.get("consumer")
    ]
    actions = context.prepared.effective_mapping_schedule.active_action_specs
    if not graph_patch.get("proposal_id") or not executable_fields or not actions:
        raise AssertionError("proposal-to-inner causal attribution is incomplete")

    descriptor = EvaluatorDescriptor.create(
        tool="case_preflight_cpu",
        tool_version="1",
        model="deterministic_fixture",
        config={"mode": "contract"},
        state="positive",
        seed=0,
    )
    exact = ExactEvaluationKey.create(context.prepared.template_sequences, descriptor)
    with tempfile.TemporaryDirectory(prefix="case-preflight-history-") as directory:
        with ExperimentRegistry(Path(directory) / "registry.sqlite") as registry:
            first = registry.register_sequence_occurrence(
                context.prepared.template_sequences,
                role="preflight",
                context_id="fixture",
                scope="case-preflight",
            )
            second = registry.register_sequence_occurrence(
                context.prepared.template_sequences,
                role="preflight",
                context_id="fixture",
                scope="case-preflight",
            )
            claim = registry.claim_evaluation(
                exact,
                owner_token="preflight-owner",
                scope="case-preflight",
            )
            if not claim.acquired:
                raise AssertionError("fresh exact evaluator claim was not acquired")
            registry.complete_evaluation(
                exact,
                owner_token="preflight-owner",
                fencing_token=claim.fencing_token,
                result={"score": 1.0},
                scope="case-preflight",
            )
            cached = registry.claim_evaluation(
                exact,
                owner_token="preflight-reader",
                scope="case-preflight",
            )
            entry = registry.lookup_evaluation(exact, scope="case-preflight")
    if not first.inserted or second.inserted:
        raise AssertionError("sequence occurrence registry did not deduplicate exactly")
    if cached.acquired or cached.outcome != "cache_hit" or entry is None or entry.status != "completed":
        raise AssertionError("exact evaluator cache did not return the completed entry")
    return "Proposal causality and all three experiment identities reach durable history.", {
        "proposal_id": graph_patch["proposal_id"],
        "consumer_bound_field_count": len(executable_fields),
        "active_mapping_action_count": len(actions),
        "code_identity": code.identity_hash,
        "effective_contract_identity": contract.identity_hash,
        "sequence_identity": sequence.sequence_bundle_hash,
        "occurrence_deduplicated": True,
        "exact_cache_hit": True,
        "exact_cache_key": exact.cache_key,
    }


def _first_generation_request(context: _ProbeContext) -> Any:
    from astevolve.search.sequence_generator import CANONICAL_AMINO_ACIDS, SequenceGenerationRequest

    action = context.prepared.effective_mapping_schedule.active_action_specs[0]
    chain = action.chain_id
    position = int(action.legal_positions[0])
    parent = context.prepared.template_sequences[chain][position]
    alternative = sorted(set(CANONICAL_AMINO_ACIDS) - {parent})[0]
    return SequenceGenerationRequest.create(
        parent_sequences=context.prepared.template_sequences,
        mapping_edge_id=action.edge_id,
        action_id=action.action_id,
        structural_node_id=action.structural_node_id,
        operator=action.operator,
        target_chain=chain,
        write_positions=[position],
        mutation_budget={"min": 1, "max": 1},
        mutable_masks=context.prepared.masks,
        fixed_residues=context.prepared.fixed_residues,
        allowed_residues={chain: {position: [parent, alternative]}},
        soft_residue_weights={chain: {position: {alternative: 4.0, parent: 0.1}}},
        seed=17,
        step=0,
        structure_condition_refs=["fixture:structure"],
        state_condition_refs=["fixture:positive"],
    )


def _probe_generator_contract(context: _ProbeContext) -> ProbeResult:
    from astevolve.search.sequence_generator import (
        CONSTRAINT_AWARE_GENERATOR_ID,
        UNIFORM_RANDOM_BASELINE_ID,
        build_default_sequence_generator_registry,
    )

    request = _first_generation_request(context)
    registry = build_default_sequence_generator_registry()
    if set(registry.available()) != {CONSTRAINT_AWARE_GENERATOR_ID, UNIFORM_RANDOM_BASELINE_ID}:
        raise AssertionError("default generator registry is incomplete")
    production = registry.generate(CONSTRAINT_AWARE_GENERATOR_ID, request)
    baseline = registry.generate(UNIFORM_RANDOM_BASELINE_ID, request)
    if production.constraint_violations or baseline.constraint_violations:
        raise AssertionError("a reference generator violated the shared hard contract")
    if production.provenance["baseline"] or not baseline.provenance["baseline"]:
        raise AssertionError("random control is not provenance-distinct from production")
    if production.request_hash != request.request_hash or baseline.request_hash != request.request_hash:
        raise AssertionError("generator output is not request-bound")
    return "Production generation and the explicit random control satisfy one hard contract.", {
        "request_hash": request.request_hash,
        "production_generator": production.generator_id,
        "baseline_generator": baseline.generator_id,
        "production_condition_refs_consumed": production.provenance["condition_refs_consumed"],
        "baseline_condition_refs_consumed": baseline.provenance["condition_refs_consumed"],
    }


def _probe_dual_ast_runtime(context: _ProbeContext) -> ProbeResult:
    from engine.mapping_compiler import compile_effective_mapping_schedule, compile_executable_dual_ast

    prepared = context.prepared
    schedule = prepared.effective_mapping_schedule
    if not prepared.executable_node_plan.enabled or not schedule.execution_enabled:
        raise AssertionError("Dual AST did not compile into an executable schedule")
    if not prepared.executable_node_plan.structural_nodes or not prepared.executable_node_plan.measurement_intents:
        raise AssertionError("Dual AST is missing structural or functional semantics")
    perturbed = deepcopy(prepared.design_state["executable_dual_ast"])
    original_state = perturbed["functional_nodes"][0]["state"]
    perturbed["functional_nodes"][0]["state"] = (
        "negative" if original_state != "negative" else "positive"
    )
    compilation = compile_executable_dual_ast(
        perturbed,
        compiled=prepared.blueprint.compile(),
        template_sequences=prepared.template_sequences,
        base_masks=prepared.masks,
        base_fixed_residues=prepared.fixed_residues,
        evaluator_capabilities=prepared.design_state.get("evaluator_capabilities"),
        execution_mode="full",
    )
    changed_schedule = compile_effective_mapping_schedule(
        compilation.mapping_plan,
        node_policies=prepared.resolved_strategy.get("node_edit_policies", {}),
    )
    same_masks = compilation.node_plan.effective_masks == prepared.executable_node_plan.effective_masks
    if not same_masks or changed_schedule.schedule_hash == schedule.schedule_hash:
        raise AssertionError("functional semantics did not affect execution independently of masks")
    return "Functional semantics change execution even when editable residues are unchanged.", {
        "structural_node_count": len(prepared.executable_node_plan.structural_nodes),
        "functional_node_count": len(prepared.executable_node_plan.measurement_intents),
        "active_mapping_count": len(schedule.active_action_specs),
        "same_residue_masks": same_masks,
        "original_state": original_state,
        "perturbed_state": perturbed["functional_nodes"][0]["state"],
        "original_schedule_hash": schedule.schedule_hash,
        "perturbed_schedule_hash": changed_schedule.schedule_hash,
    }


def _load_outer_module(context: _ProbeContext, filename: str, label: str) -> ModuleType:
    project_root = context.manifest_path
    for parent in (context.paths["entry_program"].parents):
        candidate = parent / "outerloop" / "outerloop" / filename
        if candidate.is_file():
            project_root = candidate
            break
    if not project_root.is_file() or project_root.name != filename:

        candidate = Path(__file__).resolve().parents[1] / "outerloop" / "outerloop" / filename
        if not candidate.is_file():
            raise AssertionError(f"outer module missing: {filename}")
        project_root = candidate
    return _load_module(project_root, label)


def _probe_outer_effective_phenotype(context: _ProbeContext) -> ProbeResult:
    module = _load_outer_module(context, "effective_phenotype.py", "effective_phenotype")
    config = module.PhenotypeDescriptorConfig.create()
    components = tuple(config.components)
    if not components or any("code" in item or "source" in item for item in components):
        raise AssertionError("effective phenotype descriptor contains source-code identity")
    try:
        module.PhenotypeDescriptorConfig.create(components=["code_identity"])
    except module.EffectivePhenotypeError:
        rejected_code_component = True
    else:
        rejected_code_component = False
    if not rejected_code_component:
        raise AssertionError("descriptor accepted a source-code-only component")
    policy = _load_outer_module(context, "evolution_policy.py", "phenotype_policy")
    incumbent = policy.EvolutionCandidate.create(
        candidate_id="incumbent",
        objective=0.0,
        gate_sources={"hard_gate": {"passed": True, "reasons": []}},
        phenotype_hash="phenotype:same-runtime-behavior",
        effective_contract_hash="contract:incumbent",
        sequence_bundle_hash="sequence:incumbent",
    )
    source_only_edit = policy.EvolutionCandidate.create(
        candidate_id="source-only-edit",
        objective=10.0,
        gate_sources={"hard_gate": {"passed": True, "reasons": []}},
        phenotype_hash="phenotype:same-runtime-behavior",
        effective_contract_hash="contract:source-only-edit",
        sequence_bundle_hash="sequence:source-only-edit",
    )
    duplicate = policy.compare_replacement(
        source_only_edit,
        incumbent,
        base_seed=3,
        namespace="preflight-phenotype",
        decision_index=0,
    )
    if duplicate.reason != "duplicate_effective_phenotype_rejected":
        raise AssertionError("archive admitted a duplicate effective phenotype")
    return "Outer descriptor components exclude source formatting and require runtime phenotype evidence.", {
        "components": list(components),
        "code_component_rejected": True,
        "duplicate_effective_phenotype_rejected": True,
        "duplicate_decision_hash": duplicate.decision_hash,
        "config_hash": config.config_hash,
        "effective_contract_hash": context.prepared.effective_search_contract.contract_hash,
    }


def _probe_parent_comparison(context: _ProbeContext) -> ProbeResult:
    module = _load_outer_module(context, "evolution_policy.py", "evolution_policy")
    root = module.EvolutionCandidate.create(
        candidate_id="root",
        objective=-100.0,
        gate_sources={"hard_gate": {"passed": True, "reasons": []}},
        phenotype_hash="phenotype:root",
        effective_contract_hash="contract:root",
        sequence_bundle_hash="sequence:root",
    )
    mutant = module.EvolutionCandidate.create(
        candidate_id="mutant",
        objective=100.0,
        gate_sources={"hard_gate": {"passed": False, "reasons": ["fixture_failure"]}},
        phenotype_hash="phenotype:mutant",
        effective_contract_hash="contract:mutant",
        sequence_bundle_hash="sequence:mutant",
    )
    decision = module.compare_replacement(
        mutant,
        root,
        base_seed=7,
        namespace="case-preflight",
        decision_index=0,
    )
    if decision.selected_candidate_id != "root" or decision.add_ids:
        raise AssertionError("infeasible high score replaced the feasible root")
    return "Feasibility-first comparison preserves a valid root against an invalid high score.", {
        "selected_candidate_id": decision.selected_candidate_id,
        "reason": decision.reason,
        "decision_hash": decision.decision_hash,
        "root_explicitly_compared": True,
    }


def _parse_recorded_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(value, "recorded_evidence")
    _closed(
        raw,
        {"schema_version", "case_id", "source", "provenance", "records", "claims"},
        "recorded_evidence",
    )
    if raw.get("schema_version") != "astevolve.recorded_evidence_pack.v1":
        _fail("recorded_evidence_invalid", "schema_version")
    _text(raw.get("case_id"), "recorded_evidence.case_id")
    _text(raw.get("source"), "recorded_evidence.source")

    provenance = _mapping(raw.get("provenance"), "recorded_evidence.provenance")
    _closed(
        provenance,
        {
            "schema_version", "provenance_id", "artifact_kind",
            "historical_execution", "replay_mode", "source_files",
            "observed_metrics",
        },
        "recorded_evidence.provenance",
    )
    if provenance.get("schema_version") != "astevolve.recorded_tool_provenance.v1":
        _fail("recorded_evidence_invalid", "provenance schema_version")
    provenance_id = _text(provenance.get("provenance_id"), "provenance_id")
    _text(provenance.get("artifact_kind"), "artifact_kind")
    historical_execution = _text(
        provenance.get("historical_execution"), "historical_execution"
    )
    if historical_execution not in {"cpu", "gpu"}:
        _fail("recorded_evidence_invalid", "historical_execution")
    if provenance.get("replay_mode") != "frozen_values_no_tool_execution":
        _fail("recorded_evidence_invalid", "replay_mode")
    source_files = provenance.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        _fail("recorded_evidence_invalid", "source_files")
    source_file_hashes: dict[str, str] = {}
    for source_value in source_files:
        source = _mapping(source_value, "recorded_evidence.source_file")
        _closed(source, {"role", "relative_path", "sha256"}, "source_file")
        role = _text(source.get("role"), "source_file.role")
        if role in source_file_hashes:
            _fail("recorded_evidence_invalid", "duplicate source-file role")
        relative = Path(_text(source.get("relative_path"), "source_file.relative_path"))
        if relative.is_absolute() or ".." in relative.parts:
            _fail("recorded_evidence_invalid", "source path must be repository-relative")
        source_file_hashes[role] = _sha256_text(
            source.get("sha256"), f"source_file.{role}.sha256"
        )
    raw_metrics = provenance.get("observed_metrics")
    if not isinstance(raw_metrics, list) or not raw_metrics:
        _fail("recorded_evidence_invalid", "observed_metrics")
    observed_metrics: dict[tuple[str, str], float] = {}
    for metric_value in raw_metrics:
        metric = _mapping(metric_value, "recorded_evidence.observed_metric")
        _closed(metric, {"tool", "metric", "value"}, "observed_metric")
        key = (
            _text(metric.get("tool"), "observed_metric.tool"),
            _text(metric.get("metric"), "observed_metric.metric"),
        )
        value_number = metric.get("value")
        if (
            isinstance(value_number, bool)
            or not isinstance(value_number, (int, float))
            or not math.isfinite(float(value_number))
        ):
            _fail("recorded_evidence_invalid", "observed metric value")
        if key in observed_metrics:
            _fail("recorded_evidence_invalid", "duplicate observed metric")
        observed_metrics[key] = float(value_number)

    claims = _mapping(raw.get("claims"), "recorded_evidence.claims")
    expected_claims = {
        "preflight_execution_cpu_only": True,
        "historical_gpu_artifact_replayed": historical_execution == "gpu",
        "current_gpu_executed": False,
        "biological_success": False,
    }
    _closed(claims, set(expected_claims), "recorded_evidence.claims")
    if dict(claims) != expected_claims:
        _fail("recorded_evidence_invalid", "claims")
    records = raw.get("records")
    if not isinstance(records, list) or not records:
        _fail("recorded_evidence_invalid", "records")
    decisions: dict[str, int] = {}
    conflicts: list[str] = []
    missing: list[str] = []
    recorded: list[str] = []
    injected: list[str] = []
    ids: list[str] = []
    for index, record_value in enumerate(records):
        record = _mapping(record_value, f"records[{index}]")
        _closed(
            record,
            {
                "record_id", "evidence_kind", "provenance_ref", "injection",
                "evaluator_states", "expected_decision", "reason",
            },
            f"records[{index}]",
        )
        record_id = _text(record.get("record_id"), "record_id")
        if record_id in ids:
            _fail("recorded_evidence_invalid", "duplicate record_id")
        ids.append(record_id)
        decision = _text(record.get("expected_decision"), "expected_decision")
        if decision not in {"accept", "abstain"}:
            _fail("recorded_evidence_invalid", "expected_decision")
        _text(record.get("reason"), "reason")
        evidence_kind = _text(record.get("evidence_kind"), "evidence_kind")
        provenance_ref = _text(record.get("provenance_ref"), "provenance_ref")
        injection_kind: Optional[str] = None
        if evidence_kind == "recorded_tool_smoke":
            if provenance_ref != provenance_id or record.get("injection") is not None:
                _fail("recorded_evidence_invalid", "recorded provenance mismatch")
            recorded.append(record_id)
        elif evidence_kind == "deterministic_failure_injection":
            injection = _mapping(record.get("injection"), "failure_injection")
            _closed(
                injection,
                {"schema_version", "injection_id", "kind", "deterministic"},
                "failure_injection",
            )
            if injection.get("schema_version") != "astevolve.deterministic_failure_injection.v1":
                _fail("recorded_evidence_invalid", "injection schema_version")
            injection_id = _text(injection.get("injection_id"), "injection_id")
            injection_kind = _text(injection.get("kind"), "injection.kind")
            if injection_kind not in {"score_conflict", "missing_evidence"}:
                _fail("recorded_evidence_invalid", "injection kind")
            if injection.get("deterministic") is not True:
                _fail("recorded_evidence_invalid", "injection must be deterministic")
            if provenance_ref != f"failure_injection:{injection_id}":
                _fail("recorded_evidence_invalid", "injection provenance mismatch")
            injected.append(record_id)
        else:
            _fail("recorded_evidence_invalid", "evidence_kind")
        states = record.get("evaluator_states")
        if not isinstance(states, list) or not states:
            _fail("recorded_evidence_invalid", "evaluator_states")
        available_scores: list[float] = []
        unavailable = False
        state_metrics: dict[tuple[str, str], float] = {}
        for state_value in states:
            state = _mapping(state_value, "evaluator_state")
            _closed(
                state,
                {"tool", "metric", "state", "score", "available"},
                "evaluator_state",
            )
            metric_key = (
                _text(state.get("tool"), "tool"),
                _text(state.get("metric"), "metric"),
            )
            _text(state.get("state"), "state")
            available = _strict_bool(state.get("available"), "available")
            score = state.get("score")
            if available:
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                    _fail("recorded_evidence_invalid", "available score")
                available_scores.append(float(score))
                if metric_key in state_metrics:
                    _fail("recorded_evidence_invalid", "duplicate evaluator metric")
                state_metrics[metric_key] = float(score)
            elif score is not None:
                _fail("recorded_evidence_invalid", "unavailable score must be null")
            else:
                unavailable = True
        conflict = len(available_scores) >= 2 and max(available_scores) - min(available_scores) >= 0.5
        if evidence_kind == "recorded_tool_smoke" and state_metrics != observed_metrics:
            _fail("recorded_evidence_invalid", "recorded metric/provenance mismatch")
        if injection_kind == "score_conflict" and (not conflict or unavailable):
            _fail("recorded_evidence_invalid", "score-conflict injection ineffective")
        if injection_kind == "missing_evidence" and not unavailable:
            _fail("recorded_evidence_invalid", "missing-evidence injection ineffective")
        inferred = "abstain" if unavailable or conflict else "accept"
        if inferred != decision:
            _fail("recorded_evidence_invalid", f"decision mismatch:{record_id}")
        decisions[decision] = decisions.get(decision, 0) + 1
        if conflict:
            conflicts.append(record_id)
        if unavailable:
            missing.append(record_id)
    return {
        "record_count": len(records),
        "decision_counts": dict(sorted(decisions.items())),
        "conflict_records": sorted(conflicts),
        "missing_records": sorted(missing),
        "recorded_records": sorted(recorded),
        "injected_records": sorted(injected),
        "provenance_id": provenance_id,
        "historical_execution": historical_execution,
        "replay_mode": provenance["replay_mode"],
        "source_file_hashes": dict(sorted(source_file_hashes.items())),
        "observed_metrics": [
            {"tool": tool, "metric": metric, "value": metric_value}
            for (tool, metric), metric_value in sorted(observed_metrics.items())
        ],
        "historical_gpu_artifact_replayed": expected_claims[
            "historical_gpu_artifact_replayed"
        ],
        "current_gpu_executed": False,
    }


def _probe_recorded_parser(context: _ProbeContext) -> ProbeResult:
    parsed = _parse_recorded_evidence(context.recorded_evidence)
    if (
        not parsed["recorded_records"]
        or not parsed["conflict_records"]
        or not parsed["missing_records"]
    ):
        raise AssertionError("evidence lacks recorded agreement or injected failure modes")
    return "Frozen tool evidence and deterministic failure injections are provenance-distinct.", parsed


def _probe_manifest_schema(context: _ProbeContext) -> ProbeResult:
    return "Case manifest, check list, evidence hash, and claim boundary are closed.", {
        "schema_version": context.manifest["schema_version"],
        "case_id": context.manifest["case_id"],
        "required_check_count": len(context.manifest["preflight"]["required_checks"]),
        "check_registry_hash": context.manifest["preflight"]["check_registry_hash"],
        "recorded_evidence_sha256": context.manifest["preflight"]["recorded_evidence_sha256"],
        "intended_stage": context.manifest["preflight"]["intended_stage"],
    }


def _probe_state_schema_compile(context: _ProbeContext) -> ProbeResult:
    report = context.prepared.resolved_strategy.get("strategy_schema_report", {})
    summary = report.get("summary", {})
    counts = summary.get("counts", {})
    if counts.get("unknown") or counts.get("rejected"):
        raise AssertionError("strategy contains unknown or rejected fields")
    if summary.get("unclassified_count") != 0 or summary.get("applied_without_consumer"):
        raise AssertionError("strategy contains silent or unconsumed fields")
    if summary.get("requested_count") != summary.get("classified_count"):
        raise AssertionError("strategy classification is incomplete")
    return "Design state and strategy compile without silent fields.", {
        "requested_count": summary.get("requested_count"),
        "classified_count": summary.get("classified_count"),
        "unclassified_count": summary.get("unclassified_count"),
        "unknown_count": counts.get("unknown"),
        "rejected_count": counts.get("rejected"),
        "applied_without_consumer": summary.get("applied_without_consumer"),
    }


def _probe_mask_explanation(context: _ProbeContext) -> ProbeResult:
    prepared = context.prepared
    chains: dict[str, Any] = {}
    overlap_count = 0
    for chain, sequence in sorted(prepared.template_sequences.items()):
        mask = prepared.masks.get(chain)
        if mask is None or len(mask) != len(sequence):
            raise AssertionError(f"mask length mismatch for chain {chain}")
        fixed = {int(position) for position in prepared.fixed_residues.get(chain, {})}
        overlap = sorted(position for position in fixed if mask[position])
        overlap_count += len(overlap)
        chains[chain] = {
            "length": len(sequence),
            "mutable": sum(bool(item) for item in mask),
            "frozen": sum(not bool(item) for item in mask),
            "fixed_assignments": len(fixed),
            "fixed_mutable_overlap": overlap,
        }
    if overlap_count:
        raise AssertionError("fixed residues remain mutable")
    frozen_nodes = [
        node.node_id
        for node in prepared.executable_node_plan.structural_nodes
        if node.declared_kind == "frozen"
    ]
    if not frozen_nodes:
        raise AssertionError("fixture has no frozen structural node to explain")
    return "Every residue has one non-overlapping mutable/frozen classification.", {
        "chains": chains,
        "fixed_mutable_overlap_count": overlap_count,
        "frozen_structural_nodes": frozen_nodes,
    }


def _probe_case_operator_execution(context: _ProbeContext) -> ProbeResult:
    from astevolve.search.operator_registry import require_operator

    plan = context.prepared.executable_mapping_plan
    schedule = context.prepared.effective_mapping_schedule
    resolved = []
    for action in plan.action_specs:
        spec = require_operator(action.operator, mode="node")
        if (
            not action.legal_positions
            or action.budget_min < 1
            or action.budget_max > len(action.legal_positions)
        ):
            raise AssertionError(f"invalid action write contract: {action.action_id}")
        resolved.append({"action_id": action.action_id, "operator": spec.name, "handler": spec.node_handler})
    active_ids = {action.action_id for action in schedule.active_action_specs}
    measured_ids = {item.action_id for item in schedule.active_measurement_specs}
    if active_ids != measured_ids or not active_ids:
        raise AssertionError("active action/measurement coverage is not exact")
    return "Every declared action has a handler, bounded write set, and measurement.", {
        "declared_action_count": len(resolved),
        "active_action_count": len(active_ids),
        "disabled_action_count": len(schedule.disabled_actions),
        "resolved_actions": resolved,
    }


def _probe_hard_gate_boundary(context: _ProbeContext) -> ProbeResult:
    evaluate = getattr(context.evaluator_module, "evaluate_candidate", None)
    if not callable(evaluate):
        raise AssertionError("test evaluator has no evaluate_candidate")
    prepared = context.prepared
    kwargs = {
        "compiled": prepared.blueprint.compile(),
        "design_state": prepared.design_state,
        "masks": prepared.masks,
        "template_seqs": prepared.template_sequences,
        "fixed_residues": prepared.fixed_residues,
        "score_config": prepared.score_config,
    }
    passing_fixture = getattr(
        context.evaluator_module,
        "preflight_passing_sequences",
        None,
    )
    passing_sequences = (
        passing_fixture(prepared.template_sequences)
        if callable(passing_fixture)
        else dict(prepared.template_sequences)
    )
    passing = evaluate({"seqs": passing_sequences}, **kwargs)
    failing_sequences = dict(passing_sequences)
    chosen: Optional[tuple[str, int, str]] = None
    for chain, assignments in sorted(prepared.fixed_residues.items()):
        if assignments:
            position, expected = sorted(assignments.items())[0]
            replacement = "A" if expected != "A" else "C"
            sequence = list(failing_sequences[chain])
            sequence[position] = replacement
            failing_sequences[chain] = "".join(sequence)
            chosen = (chain, int(position), replacement)
            break
    if chosen is None:
        raise AssertionError("fixture has no fixed residue for the gate boundary")
    failing = evaluate({"seqs": failing_sequences}, **kwargs)
    if passing.get("hard_gate_pass") is not True or failing.get("hard_gate_pass") is not False:
        raise AssertionError("CPU evaluator did not separate both sides of the hard gate")
    if not any("fixed_residue_modified" in str(reason) for reason in failing.get("disqualification_reasons", [])):
        raise AssertionError("failed boundary lacks a fixed-residue reason")
    return "CPU evaluator deterministically exercises both sides of a hard gate.", {
        "passing_hard_gate": passing["hard_gate_pass"],
        "failing_hard_gate": failing["hard_gate_pass"],
        "passing_score": passing.get("normalized_score"),
        "failing_score": failing.get("normalized_score"),
        "perturbation": {"chain_id": chosen[0], "position": chosen[1], "replacement": chosen[2]},
        "failure_reasons": failing.get("disqualification_reasons", []),
    }


def _probe_evaluator_ast_wiring(context: _ProbeContext) -> ProbeResult:
    prepared = context.prepared
    intents = {item.functional_node_id: item for item in prepared.executable_node_plan.measurement_intents}
    schedule = prepared.effective_mapping_schedule
    if set(schedule.functional_action_coverage) != set(intents):
        raise AssertionError("functional action coverage differs from compiled intents")
    if set(schedule.functional_measurement_coverage) != set(intents):
        raise AssertionError("functional measurement coverage differs from compiled intents")
    bindings = []
    for measurement in schedule.active_measurement_specs:
        intent = intents.get(measurement.functional_node_id)
        if intent is None:
            raise AssertionError("measurement references an unknown functional node")
        expected_threshold = intent.gate["threshold"] if intent.gate else None
        fields_match = (
            measurement.evaluator_id == intent.evaluator_id
            and measurement.term_name == intent.term_name
            and measurement.state == intent.state
            and measurement.direction == intent.direction
            and measurement.threshold == expected_threshold
            and measurement.missing_policy == intent.missing_policy
        )
        if not fields_match:
            raise AssertionError(f"evaluator wiring mismatch for {measurement.action_id}")
        bindings.append(
            {
                "functional_node_id": measurement.functional_node_id,
                "structural_node_id": measurement.structural_node_id,
                "evaluator_id": measurement.evaluator_id,
                "term_name": measurement.term_name,
                "state": measurement.state,
                "threshold": measurement.threshold,
                "missing_policy": measurement.missing_policy,
            }
        )
    if not any(item["threshold"] is not None for item in bindings):
        raise AssertionError("no hard evaluator threshold reaches the active mapping schedule")
    return "Functional and structural nodes reach exact evaluator terms through active mapping edges.", {
        "binding_count": len(bindings),
        "bindings": bindings,
        "mapping_attribution_hash": schedule.mapping_attribution_hash,
        "schedule_hash": schedule.schedule_hash,
    }


def _probe_case_recorded_evidence(context: _ProbeContext) -> ProbeResult:
    parsed = _parse_recorded_evidence(context.recorded_evidence)
    expected = {"abstain": 2, "accept": 1}
    if parsed["decision_counts"] != expected:
        raise AssertionError(f"unexpected recorded decision coverage: {parsed['decision_counts']}")
    return (
        "Case evidence covers one consistency-accepted low-interface replay "
        "and two deterministic abstention injections.",
        parsed,
    )


ENGINE_PROBES: dict[str, Probe] = {
    "engine.operator_registry": _probe_operator_registry,
    "engine.evaluator_registry": _probe_evaluator_registry,
    "engine.memory_isolation": _probe_memory_isolation,
    "engine.causal_identity_history": _probe_causal_identity_history,
    "engine.generator_contract": _probe_generator_contract,
    "engine.dual_ast_runtime": _probe_dual_ast_runtime,
    "engine.outer_effective_phenotype": _probe_outer_effective_phenotype,
    "engine.parent_comparison": _probe_parent_comparison,
    "engine.recorded_evidence_parser": _probe_recorded_parser,
}

CASE_PROBES: dict[str, Probe] = {
    "case.manifest_schema": _probe_manifest_schema,
    "case.state_schema_compile": _probe_state_schema_compile,
    "case.mask_explanation": _probe_mask_explanation,
    "case.operator_execution": _probe_case_operator_execution,
    "case.hard_gate_boundary": _probe_hard_gate_boundary,
    "case.evaluator_ast_wiring": _probe_evaluator_ast_wiring,
    "case.recorded_evidence": _probe_case_recorded_evidence,
}


def _failed_result(spec: CheckSpec, code: str, detail: Any) -> CheckResult:
    return CheckResult.create(
        spec,
        status="failed",
        summary=f"{spec.check_id} failed closed.",
        evidence={"error_type": type(detail).__name__, "detail": str(detail)},
        error_code=code,
    )


def _run_probe(spec: CheckSpec, probe: Probe, context: _ProbeContext, timeout: float) -> CheckResult:
    try:
        summary, evidence = execute_probe_with_timeout(
            lambda: probe(context),
            timeout_seconds=timeout,
        )
        if not isinstance(evidence, Mapping):
            raise TypeError("probe evidence must be a mapping")
        return CheckResult.create(spec, status="passed", summary=summary, evidence=evidence)
    except ProbeTimeoutError as exc:
        return _failed_result(spec, "probe_timeout", exc)
    except Exception as exc:
        return _failed_result(spec, "probe_exception", exc)


def _atomic_write_report(path: Path, report: CaseReadinessReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _failure_report(
    *,
    manifest_path: Path,
    manifest_hash: str,
    case_id: str,
    input_hashes: Mapping[str, str],
    failure_code: str,
    detail: Any,
) -> CaseReadinessReport:
    engine_checks = [
        _failed_result(spec, failure_code, detail)
        for spec in CHECK_REGISTRY
        if spec.scope == "engine"
    ]
    case_checks = [
        _failed_result(spec, failure_code, detail)
        for spec in CHECK_REGISTRY
        if spec.scope == "case"
    ]
    return CaseReadinessReport.create(
        case_id=case_id or manifest_path.stem or "invalid-case",
        manifest_hash=manifest_hash,
        input_hashes=input_hashes,
        required_checks=sorted(spec.check_id for spec in CHECK_REGISTRY),
        engine_report=EngineCapabilityReport.create(engine_checks),
        case_checks=case_checks,
        extra_failure_codes=(failure_code,),
    )


def run_case_preflight(
    manifest_path: str | Path,
    report_path: str | Path,
    *,
    timeout_seconds: Optional[float] = None,
) -> CaseReadinessReport:


    manifest_file = Path(manifest_path).expanduser().resolve()
    report_file = Path(report_path).expanduser().resolve()
    manifest_read_error: Optional[OSError] = None
    try:
        raw_manifest = manifest_file.read_bytes()
    except OSError as exc:
        raw_manifest = b""
        manifest_read_error = exc
    raw_manifest_hash = hashlib.sha256(raw_manifest).hexdigest()
    manifest: dict[str, Any] = {}
    input_hashes: dict[str, str] = {"manifest": raw_manifest_hash}
    try:
        if manifest_read_error is not None:
            _fail("manifest_schema_invalid", manifest_read_error)
        manifest = _read_manifest(manifest_file)
        paths = _resolve_inputs(manifest_file, manifest)
        input_hashes.update({name: _file_hash(path) for name, path in paths.items()})
        declared_timeout = float(manifest["preflight"]["timeout_seconds"])
        timeout = declared_timeout if timeout_seconds is None else float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            _fail("timeout_invalid", timeout)
        context = execute_probe_with_timeout(
            lambda: _prepare_context(manifest_file, report_file, manifest, paths),
            timeout_seconds=timeout,
        )
    except ProbeTimeoutError as exc:
        report = _failure_report(
            manifest_path=manifest_file,
            manifest_hash=raw_manifest_hash,
            case_id=str(manifest.get("case_id") or "invalid-case"),
            input_hashes=input_hashes,
            failure_code="preparation_timeout",
            detail=exc,
        )
        _atomic_write_report(report_file, report)
        return report
    except Exception as exc:
        code = "manifest_schema_invalid"
        if isinstance(exc, CasePreflightError) and exc.code not in {
            "manifest_schema_invalid", "unknown_fields", "fields_missing",
            "mapping_required", "text_required", "schema_version_invalid",
            "required_check_set_mismatch", "registry_hash_mismatch",
            "claim_mismatch", "timeout_invalid", "launcher_default_invalid",
            "intended_stage_invalid", "sha256_invalid", "list_required",
            "duplicate_values", "boolean_required",
        }:
            code = "case_preparation_failed"
        report = _failure_report(
            manifest_path=manifest_file,
            manifest_hash=raw_manifest_hash,
            case_id=str(manifest.get("case_id") or "invalid-case"),
            input_hashes=input_hashes,
            failure_code=code,
            detail=exc,
        )
        _atomic_write_report(report_file, report)
        return report

    by_id = {spec.check_id: spec for spec in CHECK_REGISTRY}
    engine_checks = [
        _run_probe(by_id[check_id], probe, context, timeout)
        for check_id, probe in ENGINE_PROBES.items()
    ]
    case_checks = [
        _run_probe(by_id[check_id], probe, context, timeout)
        for check_id, probe in CASE_PROBES.items()
    ]
    report = CaseReadinessReport.create(
        case_id=manifest["case_id"],
        manifest_hash=raw_manifest_hash,
        input_hashes=input_hashes,
        required_checks=manifest["preflight"]["required_checks"],
        engine_report=EngineCapabilityReport.create(engine_checks),
        case_checks=case_checks,
    )
    _atomic_write_report(report_file, report)
    return report


def verify_case_readiness(
    report_path: str | Path,
    manifest_path: str | Path,
) -> CaseReadinessReport:


    report_file = Path(report_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    try:
        raw_report = json.loads(report_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail("report_unreadable", exc)
    report = CaseReadinessReport.from_mapping(_mapping(raw_report, "case_readiness_report"))
    current_manifest_hash = _file_hash(manifest_file)
    if current_manifest_hash != report.manifest_hash:
        _fail("stale_input", "manifest")
    if not report.ready_for_smoke:
        _fail("not_ready", ",".join(report.failure_codes))
    try:
        manifest = _read_manifest(manifest_file)
        paths = _resolve_inputs(manifest_file, manifest)
    except CasePreflightError as exc:
        _fail("stale_input", f"manifest_or_dependency:{exc}")
    if report.input_hashes.get("manifest") != current_manifest_hash:
        _fail("stale_input", "manifest_hash_binding")
    expected_names = {"manifest", *_INPUT_PATH_FIELDS}
    if "immutable_reference" in manifest:
        expected_names.add("immutable_reference")
    if set(report.input_hashes) != expected_names:
        _fail("input_hash_set_mismatch")
    for name, path in paths.items():
        if _file_hash(path) != report.input_hashes.get(name):
            _fail("stale_input", name)
    if manifest["case_id"] != report.case_id:
        _fail("stale_input", "case_id")
    if manifest["preflight"]["required_checks"] != list(report.required_checks):
        _fail("stale_input", "required_checks")
    if manifest["preflight"]["check_registry_hash"] != report.engine_report.registry_hash:
        _fail("registry_hash_mismatch", "report/manifest")
    if any(item.status != "passed" for item in (*report.engine_report.checks, *report.case_checks)):
        _fail("not_ready", "one or more checks failed")
    return report


__all__ = [
    "CASE_PROBES",
    "CHECK_REGISTRY",
    "CHECK_REGISTRY_VERSION",
    "ENGINE_PROBES",
    "CasePreflightError",
    "CaseReadinessReport",
    "CheckResult",
    "CheckSpec",
    "EngineCapabilityReport",
    "ProbeTimeoutError",
    "check_registry_hash",
    "execute_probe_with_timeout",
    "run_case_preflight",
    "verify_case_readiness",
]
