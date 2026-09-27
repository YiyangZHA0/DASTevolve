

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from engine.experiment_identity import CodeIdentity, SequenceBundleIdentity
from engine.memory_lifecycle import MemoryExecutionContext, ProposalCausalEnvelope
from engine.parent_lineage import bind_selected_parent_lineages
from outerloop.effective_phenotype import AcceptedRuntimeArtifact


class ContinuationSeedError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContinuationSeedError(f"{label} must be a non-empty string")
    return value.strip()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContinuationSeedError(f"{label} must be a mapping")
    return value


def _artifact_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ContinuationSeedError(f"{label} is invalid JSON") from error
    return _mapping(value, label)


@dataclass(frozen=True)
class ContinuationSeed:


    source_path: Path
    source_file_sha256: str
    source_program_id: str
    code: str
    code_identity: CodeIdentity
    sequence_identity: SequenceBundleIdentity
    effective_ast_hash: str
    effective_ast_revision: int
    parent_artifacts: Mapping[str, Any]
    source_metrics: Mapping[str, Any]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_program_id: str = "",
        expected_file_sha256: str = "",
    ) -> "ContinuationSeed":
        raw_source_path = Path(path).expanduser()
        if raw_source_path.is_symlink() or not raw_source_path.is_file():
            raise ContinuationSeedError(
                "continuation seed program JSON must be a regular file"
            )
        source_path = raw_source_path.resolve()
        if source_path.stat().st_size > 32 * 1024 * 1024:
            raise ContinuationSeedError("continuation seed program JSON is too large")
        observed_file_sha256 = _sha256_file(source_path)
        expected_hash = str(expected_file_sha256 or "").strip().lower()
        if expected_hash:
            if len(expected_hash) != 64 or any(
                character not in "0123456789abcdef" for character in expected_hash
            ):
                raise ContinuationSeedError("expected continuation seed SHA-256 is invalid")
            if observed_file_sha256 != expected_hash:
                raise ContinuationSeedError("continuation seed program JSON SHA-256 mismatch")
        try:
            record = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContinuationSeedError("continuation seed program JSON is unreadable") from error
        record = _mapping(record, "continuation seed program record")
        program_id = _require_text(record.get("id"), "continuation seed program id")
        requested_id = str(expected_program_id or "").strip()
        if requested_id and program_id != requested_id:
            raise ContinuationSeedError("continuation seed program id mismatch")


        code = record.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ContinuationSeedError("continuation seed code must be non-empty")
        artifacts = _artifact_mapping(record.get("artifacts_json"), "continuation seed artifacts_json")
        for required in ("best_seqs", "executable_dual_ast", "ast_revision_report"):
            if required not in artifacts:
                raise ContinuationSeedError(f"continuation seed artifacts missing {required}")
        report = _mapping(artifacts["ast_revision_report"], "continuation ast report")
        effective_ast_hash = _require_text(
            report.get("effective_ast_hash"), "continuation effective AST hash"
        )
        revision = report.get("effective_revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ContinuationSeedError("continuation effective AST revision is invalid")
        metrics = _mapping(record.get("metrics"), "continuation seed metrics")
        if metrics.get("hard_gate_pass") is not True or metrics.get("disqualified") is True:
            raise ContinuationSeedError("continuation seed was not an accepted feasible result")
        metadata = _mapping(record.get("metadata"), "continuation seed metadata")
        try:
            accepted = AcceptedRuntimeArtifact.from_mapping(
                _mapping(
                    metadata.get("accepted_runtime_artifact"),
                    "continuation accepted runtime artifact",
                )
            )
        except (TypeError, ValueError) as error:
            raise ContinuationSeedError("continuation accepted runtime artifact is invalid") from error
        code_identity = CodeIdentity.from_text(code)
        if accepted.code_identity.to_dict() != code_identity.to_dict():
            raise ContinuationSeedError("continuation code does not match accepted evidence")
        try:
            sequence_identity = SequenceBundleIdentity.create(
                _mapping(artifacts["best_seqs"], "continuation best_seqs")
            )
        except (TypeError, ValueError) as error:
            raise ContinuationSeedError("continuation best_seqs is invalid") from error
        if accepted.sequence_bundle_identity.to_dict() != sequence_identity.to_dict():
            raise ContinuationSeedError(
                "continuation best_seqs does not match accepted evidence"
            )


        try:
            bind_selected_parent_lineages(
                MemoryExecutionContext(
                    proposal_causal_envelope_json=ProposalCausalEnvelope.create(
                        parent_program_id=program_id,
                        parent_code_hash=code_identity.code_hash.split(":", 1)[1],
                    ).to_json()
                ),
                parent_program_id=program_id,
                parent_artifacts=artifacts,
            )
        except (TypeError, ValueError) as error:
            raise ContinuationSeedError("continuation sequence/AST lineage is invalid") from error
        return cls(
            source_path=source_path,
            source_file_sha256=observed_file_sha256,
            source_program_id=program_id,
            code=code,
            code_identity=code_identity,
            sequence_identity=sequence_identity,
            effective_ast_hash=effective_ast_hash,
            effective_ast_revision=revision,
            parent_artifacts=dict(artifacts),
            source_metrics=dict(metrics),
        )

    def bind_context(self, context: MemoryExecutionContext) -> MemoryExecutionContext:


        envelope = ProposalCausalEnvelope.create(
            parent_program_id=self.source_program_id,
            parent_code_hash=self.code_identity.code_hash.split(":", 1)[1],
        )
        scoped = replace(context, proposal_causal_envelope_json=envelope.to_json())
        return bind_selected_parent_lineages(
            scoped,
            parent_program_id=self.source_program_id,
            parent_artifacts=self.parent_artifacts,
        )

    def to_artifact(self) -> dict[str, Any]:


        return {
            "schema_version": "astevolve.continuation_seed.v1",
            "mode": "reevaluated_continuation_root",
            "source_program_id": self.source_program_id,
            "source_program_json_sha256": self.source_file_sha256,
            "source_code_identity": self.code_identity.to_dict(),
            "source_sequence_bundle_identity": self.sequence_identity.to_dict(),
            "source_effective_ast_hash": self.effective_ast_hash,
            "source_effective_ast_revision": self.effective_ast_revision,
            "source_metrics": dict(self.source_metrics),
        }


__all__ = ["ContinuationSeed", "ContinuationSeedError"]
