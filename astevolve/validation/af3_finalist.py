

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from astevolve.runtime.paths import model_path, tmp_root


AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
REPORT_VERSION = "dastevolve.af3_finalist_validation.v2"


class FinalistValidationError(ValueError):
    pass


@dataclass(frozen=True)
class FixedEntity:
    entity_id: str
    sequence: str


@dataclass(frozen=True)
class StateSpec:
    name: str
    role: str
    entity_ids: tuple[str, ...]


@dataclass(frozen=True)
class FinalistSpec:
    case_id: str
    binder_chain_id: str
    binder_entity_id: str
    binder_length: int
    states: tuple[StateSpec, ...]
    fixed_entities: tuple[FixedEntity, ...] = ()
    protected_spans: tuple[tuple[int, int, str], ...] = ()
    required_residues: tuple[tuple[int, str], ...] = ()
    allowed_cysteine_positions: tuple[int, ...] | None = None
    baseline_sequence_sha256: tuple[str, ...] = ()
    forbidden_sequence_sha256: tuple[str, ...] = ()
    forbidden_input_tokens: tuple[str, ...] = ("hidden", "oracle")
    interpretation_boundary: str = (
        "AlphaFold 3 structure-confidence evidence is a computational "
        "guardrail and is not an affinity, activity, or experimental claim."
    )

    def __post_init__(self) -> None:
        if self.binder_length < 1:
            raise ValueError("binder_length must be positive")
        entity_ids = {self.binder_entity_id}
        for entity in self.fixed_entities:
            _validate_sequence(entity.sequence, expected_length=0)
            if entity.entity_id in entity_ids:
                raise ValueError(f"duplicate entity id: {entity.entity_id}")
            entity_ids.add(entity.entity_id)
        if not self.states:
            raise ValueError("at least one AF3 state is required")
        for state in self.states:
            if not state.entity_ids or any(item not in entity_ids for item in state.entity_ids):
                raise ValueError(f"state {state.name!r} references an unknown entity")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    sequence: str
    source: str
    source_path: str | None = None
    source_energy: float | None = None
    source_feasible: bool | None = None
    is_baseline: bool = False

    @property
    def sequence_sha256(self) -> str:
        return hashlib.sha256(self.sequence.encode("ascii")).hexdigest()

    def public_record(self) -> dict[str, Any]:
        source_gate_status = (
            "passed"
            if self.source_feasible is True
            else "failed"
            if self.source_feasible is False
            else "unknown"
        )
        return {
            "candidate_id": self.candidate_id,
            "sequence_sha256": self.sequence_sha256,
            "length": len(self.sequence),
            "source": self.source,
            "source_path": self.source_path,
            "source_energy": self.source_energy,
            "source_feasible": self.source_feasible,
            "source_gate_status": source_gate_status,
            "is_baseline": self.is_baseline,
            "baseline_status": "baseline" if self.is_baseline else "non_baseline",
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_id(value: Any, fallback: str = "candidate") -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return (normalized or fallback)[:96]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_sequence(value: Any, *, expected_length: int) -> str:
    sequence = "".join(str(value or "").split()).upper()
    if not sequence:
        raise FinalistValidationError("candidate sequence is empty")
    invalid = sorted(set(sequence) - AA_ALPHABET)
    if invalid:
        raise FinalistValidationError(
            f"candidate sequence contains non-canonical residues: {invalid}"
        )
    if expected_length and len(sequence) != expected_length:
        raise FinalistValidationError(
            f"candidate sequence length is {len(sequence)}, expected {expected_length}"
        )
    return sequence


def _validate_case_sequence(sequence: str, spec: FinalistSpec) -> None:
    for start, end, expected in spec.protected_spans:
        if not 0 <= start < end <= len(sequence):
            raise FinalistValidationError(
                f"invalid protected span configured for {spec.case_id}: [{start},{end})"
            )
        if sequence[start:end] != expected:
            raise FinalistValidationError(
                f"candidate violates protected span [{start},{end})"
            )
    for position, expected in spec.required_residues:
        if not 0 <= position < len(sequence) or sequence[position] != expected:
            raise FinalistValidationError(
                f"candidate violates required residue {expected} at zero-based {position}"
            )
    if spec.allowed_cysteine_positions is not None:
        observed = tuple(index for index, residue in enumerate(sequence) if residue == "C")
        if observed != tuple(sorted(spec.allowed_cysteine_positions)):
            raise FinalistValidationError(
                "candidate cysteine positions do not match the case integrity contract"
            )
    if hashlib.sha256(sequence.encode("ascii")).hexdigest() in set(
        spec.forbidden_sequence_sha256
    ):
        raise FinalistValidationError("candidate is a forbidden baseline sequence")


def _is_declared_baseline(sequence: str, spec: FinalistSpec) -> bool:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest() in set(
        spec.baseline_sequence_sha256
    )


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _candidate_energy(value: Mapping[str, Any]) -> float | None:
    for container in (value, value.get("metrics"), value.get("energy")):
        if not isinstance(container, Mapping):
            continue
        for key in (
            "final_energy",
            "combined_energy",
            "scientific_final_energy",
            "total_energy",
            "design_energy",
            "combined_loss",
            "fast_loss",
        ):
            number = _finite_float(container.get(key))
            if number is not None:
                return number
    return None


def _candidate_feasible(value: Mapping[str, Any]) -> bool | None:
    metrics = value.get("metrics")
    if isinstance(metrics, Mapping):
        if bool(metrics.get("disqualified", False)):
            return False
        if "hard_gate_pass" in metrics:
            return bool(metrics.get("hard_gate_pass"))
    selection = value.get("feasibility_selection")
    if isinstance(selection, Mapping) and "feasible" in selection:
        return bool(selection.get("feasible"))
    report = value.get("inner_evaluator_report")
    if isinstance(report, Mapping) and "hard_gate_pass" in report:
        return bool(report.get("hard_gate_pass"))
    if "hard_gate_pass" in value:
        return bool(value.get("hard_gate_pass"))
    return None


def _sequence_from_mapping(
    seqs: Mapping[str, Any], *, chain_id: str, expected_length: int
) -> str | None:
    raw = seqs.get(chain_id)
    if raw is None:
        return None
    try:
        return _validate_sequence(raw, expected_length=expected_length)
    except FinalistValidationError:
        return None


def _sequence_from_bundle(
    bundle: Any, *, chain_id: str, expected_length: int
) -> str | None:
    if not isinstance(bundle, Mapping):
        return None
    chains = bundle.get("chains")
    if isinstance(chains, Mapping):
        return _sequence_from_mapping(
            chains, chain_id=chain_id, expected_length=expected_length
        )
    if not isinstance(chains, list):
        return None
    for chain in chains:
        if not isinstance(chain, Mapping) or str(chain.get("chain_id")) != chain_id:
            continue
        try:
            return _validate_sequence(
                chain.get("sequence"), expected_length=expected_length
            )
        except FinalistValidationError:
            return None
    return None


def _extract_candidates_from_json(
    value: Any,
    *,
    source_path: Path,
    spec: FinalistSpec,
) -> list[Candidate]:
    extracted: list[Candidate] = []

    def add(
        sequence: str | None,
        *,
        candidate_id: Any,
        path: str,
        energy: float | None,
        feasible: bool | None,
    ) -> None:
        if sequence is None:
            return
        try:
            _validate_case_sequence(sequence, spec)
        except FinalistValidationError:
            return
        extracted.append(
            Candidate(
                candidate_id=_safe_id(candidate_id, fallback="outer_candidate"),
                sequence=sequence,
                source="outer_output",
                source_path=f"{source_path}:{path}",
                source_energy=energy,
                source_feasible=feasible,
                is_baseline=_is_declared_baseline(sequence, spec),
            )
        )

    def walk(
        item: Any,
        path: str,
        inherited_id: Any = None,
        inherited_energy: float | None = None,
        inherited_feasible: bool | None = None,
    ) -> None:
        if isinstance(item, Mapping):
            candidate_id = (
                item.get("selection_candidate_id")
                or item.get("candidate_id")
                or item.get("variant_id")
                or item.get("proposal_id")
                or item.get("id")
                or inherited_id
                or path
            )
            energy = _candidate_energy(item)
            if energy is None:
                energy = inherited_energy
            feasible = _candidate_feasible(item)
            if feasible is None:
                feasible = inherited_feasible

            seqs = item.get("seqs")
            if isinstance(seqs, Mapping):
                add(
                    _sequence_from_mapping(
                        seqs,
                        chain_id=spec.binder_chain_id,
                        expected_length=spec.binder_length,
                    ),
                    candidate_id=candidate_id,
                    path=f"{path}.seqs",
                    energy=energy,
                    feasible=feasible,
                )
            bundle = item.get("sequence_bundle_identity")
            if isinstance(bundle, Mapping):
                add(
                    _sequence_from_bundle(
                        bundle,
                        chain_id=spec.binder_chain_id,
                        expected_length=spec.binder_length,
                    ),
                    candidate_id=candidate_id,
                    path=f"{path}.sequence_bundle_identity",
                    energy=energy,
                    feasible=feasible,
                )


            if ".tables.sequences." in path:
                add(
                    _sequence_from_mapping(
                        item,
                        chain_id=spec.binder_chain_id,
                        expected_length=spec.binder_length,
                    ),
                    candidate_id=candidate_id,
                    path=path,
                    energy=energy,
                    feasible=feasible,
                )
            raw_sequence = item.get("candidate_sequence")
            if raw_sequence is None and any(
                key in item
                for key in ("candidate_id", "variant_id", "proposal_id")
            ):
                raw_sequence = item.get("sequence")
            if raw_sequence is not None:
                try:
                    direct = _validate_sequence(
                        raw_sequence, expected_length=spec.binder_length
                    )
                except FinalistValidationError:
                    direct = None
                add(
                    direct,
                    candidate_id=candidate_id,
                    path=f"{path}.candidate_sequence",
                    energy=energy,
                    feasible=feasible,
                )
            for key, nested in item.items():
                walk(
                    nested,
                    f"{path}.{key}",
                    candidate_id,
                    energy,
                    feasible,
                )
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                walk(
                    nested,
                    f"{path}[{index}]",
                    inherited_id,
                    inherited_energy,
                    inherited_feasible,
                )

    walk(value, "$")
    return extracted


def _assert_input_path_allowed(path: Path, spec: FinalistSpec) -> None:
    lowered_parts = [part.lower() for part in path.parts]
    for token in spec.forbidden_input_tokens:
        lowered = str(token).lower()
        if lowered and any(lowered in part for part in lowered_parts):
            raise FinalistValidationError(
                f"refusing oracle-like candidate input path containing {token!r}: {path}"
            )


def _load_json(path: Path, spec: FinalistSpec) -> Any:
    _assert_input_path_allowed(path, spec)
    if not path.is_file():
        raise FileNotFoundError(f"candidate artifact does not exist: {path}")
    if path.stat().st_size > 128 * 1024 * 1024:
        raise FinalistValidationError(
            f"refusing oversized candidate artifact (>128 MiB): {path}"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalistValidationError(f"cannot read candidate artifact {path}: {exc}") from exc


def _outer_json_files(path: Path, spec: FinalistSpec) -> list[Path]:
    _assert_input_path_allowed(path, spec)
    if path.is_file():
        return [path.resolve()]
    if not path.is_dir():
        raise FileNotFoundError(f"outer output does not exist: {path}")
    patterns = (
        "**/accepted_runtime_artifact",
        "**/structure_evaluated_variants.json",
        "**/mcts_search.normalized.json",
    )
    found: list[Path] = []
    for pattern in patterns:
        found.extend(path.glob(pattern))
    files = sorted({item.resolve() for item in found if item.is_file()})
    if not files:
        raise FinalistValidationError(
            f"no accepted-runtime or structure-candidate artifacts found below {path}"
        )
    return files


def _read_outer_candidates(paths: Sequence[str], spec: FinalistSpec) -> list[Candidate]:
    candidates: list[Candidate] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        for json_path in _outer_json_files(path, spec):
            candidates.extend(
                _extract_candidates_from_json(
                    _load_json(json_path, spec),
                    source_path=json_path,
                    spec=spec,
                )
            )
    return candidates


def _declared_checkpoint_best(
    program: Any,
    *,
    program_path: Path,
    best_id: str,
    spec: FinalistSpec,
) -> Candidate:


    if not isinstance(program, Mapping):
        raise FinalistValidationError(
            f"checkpoint best program is not a JSON object: {program_path}"
        )
    sources: list[tuple[str, Any]] = []
    raw_artifacts = program.get("artifacts_json")
    if isinstance(raw_artifacts, str):
        try:
            raw_artifacts = json.loads(raw_artifacts)
        except json.JSONDecodeError as exc:
            raise FinalistValidationError(
                f"checkpoint best has invalid artifacts_json: {program_path}"
            ) from exc
    if isinstance(raw_artifacts, Mapping):
        sources.append(("$.artifacts_json.best_seqs", raw_artifacts.get("best_seqs")))
    sources.extend(
        [
            ("$.best_seqs", program.get("best_seqs")),
            ("$.seqs", program.get("seqs")),
        ]
    )
    metadata = program.get("metadata")
    if isinstance(metadata, Mapping):
        accepted = metadata.get("accepted_runtime_artifact")
        if isinstance(accepted, Mapping):
            bundle = accepted.get("sequence_bundle_identity")
            sources.append(
                (
                    "$.metadata.accepted_runtime_artifact.sequence_bundle_identity",
                    bundle,
                )
            )

    for source_path, payload in sources:
        sequence = None
        if source_path.endswith("sequence_bundle_identity"):
            sequence = _sequence_from_bundle(
                payload,
                chain_id=spec.binder_chain_id,
                expected_length=spec.binder_length,
            )
        elif isinstance(payload, Mapping):
            sequence = _sequence_from_mapping(
                payload,
                chain_id=spec.binder_chain_id,
                expected_length=spec.binder_length,
            )
        if sequence is None:
            continue
        _validate_case_sequence(sequence, spec)
        return Candidate(
            candidate_id=_safe_id(best_id),
            sequence=sequence,
            source="checkpoint_best",
            source_path=f"{program_path}:{source_path}",
            source_energy=_candidate_energy(program),
            source_feasible=_candidate_feasible(program),
            is_baseline=_is_declared_baseline(sequence, spec),
        )
    raise FinalistValidationError(
        "checkpoint best sequence is absent or violates case integrity: "
        f"{program_path}"
    )


def _read_checkpoint_candidates(paths: Sequence[str], spec: FinalistSpec) -> list[Candidate]:
    candidates: list[Candidate] = []
    for raw in paths:
        checkpoint = Path(raw).expanduser().resolve()
        _assert_input_path_allowed(checkpoint, spec)
        programs = checkpoint / "programs"
        if not checkpoint.is_dir() or not programs.is_dir():
            raise FinalistValidationError(
                f"checkpoint must contain a programs directory: {checkpoint}"
            )
        info_path = checkpoint / "best_program_info.json"
        best_id: str | None = None
        if info_path.is_file():
            info = _load_json(info_path, spec)
            if isinstance(info, Mapping) and info.get("id"):
                best_id = str(info["id"])
        if not best_id:
            raise FinalistValidationError(
                f"checkpoint lacks a declared best program id: {checkpoint}"
            )
        program_paths = sorted(programs.glob("*.json"))
        if best_id:
            program_paths.sort(key=lambda item: (item.stem != best_id, item.name))
        if not program_paths:
            raise FinalistValidationError(f"checkpoint has no program JSON: {checkpoint}")
        for program_path in program_paths:
            value = _load_json(program_path, spec)
            if program_path.stem == best_id:
                candidates.append(
                    _declared_checkpoint_best(
                        value,
                        program_path=program_path,
                        best_id=best_id,
                        spec=spec,
                    )
                )
            extracted = _extract_candidates_from_json(
                value,
                source_path=program_path,
                spec=spec,
            )
            for candidate in extracted:
                candidates.append(
                    Candidate(
                        candidate_id=candidate.candidate_id,
                        sequence=candidate.sequence,


                        source="checkpoint",
                        source_path=candidate.source_path,
                        source_energy=candidate.source_energy,
                        source_feasible=candidate.source_feasible,
                        is_baseline=candidate.is_baseline,
                    )
                )
    return candidates


def _rank_and_deduplicate(
    candidates: Iterable[Candidate], *, top_k: int
) -> list[Candidate]:
    if top_k < 1:
        raise FinalistValidationError("top-k must be positive")

    def key(candidate: Candidate) -> tuple[int, float, str]:
        feasible_rank = (
            0
            if candidate.source_feasible is True
            else 1
            if candidate.source_feasible is None
            else 2
        )
        energy = candidate.source_energy if candidate.source_energy is not None else math.inf
        return feasible_rank, energy, candidate.candidate_id

    def merge_same_sequence(left: Candidate, right: Candidate) -> Candidate:
        checkpoint = next(
            (
                item
                for item in (left, right)
                if item.source == "checkpoint_best"
            ),
            None,
        )
        if checkpoint is None:
            return left if key(left) <= key(right) else right
        other = right if checkpoint is left else left


        feasible = checkpoint.source_feasible
        if feasible is None:
            if other.source_feasible is True:
                feasible = True
            elif other.source_feasible is False:
                feasible = False
        energies = [
            value
            for value in (checkpoint.source_energy, other.source_energy)
            if value is not None
        ]
        return Candidate(
            candidate_id=checkpoint.candidate_id,
            sequence=checkpoint.sequence,
            source="checkpoint_best",
            source_path=checkpoint.source_path,
            source_energy=min(energies) if energies else None,
            source_feasible=feasible,
            is_baseline=checkpoint.is_baseline or other.is_baseline,
        )

    best_by_hash: dict[str, Candidate] = {}
    for candidate in candidates:
        current = best_by_hash.get(candidate.sequence_sha256)
        if current is None:
            best_by_hash[candidate.sequence_sha256] = candidate
        else:
            best_by_hash[candidate.sequence_sha256] = merge_same_sequence(
                current, candidate
            )
    unique = list(best_by_hash.values())
    checkpoint_best = [
        item for item in unique if item.source == "checkpoint_best"
    ]
    if len(checkpoint_best) > top_k:
        raise FinalistValidationError(
            "top-k is smaller than the number of supplied checkpoint best candidates"
        )
    mandatory_hashes = {item.sequence_sha256 for item in checkpoint_best}
    remaining = [
        item for item in unique if item.sequence_sha256 not in mandatory_hashes
    ]
    feasible = [item for item in remaining if item.source_feasible is True]
    if feasible:
        remaining = feasible
    else:
        remaining = [
            item for item in remaining if item.source_feasible is not False
        ]
        if not checkpoint_best and not remaining:
            raise FinalistValidationError(
                "all supplied finalists fail their source hard gates"
            )
    return sorted(checkpoint_best, key=key) + sorted(remaining, key=key)[
        : top_k - len(checkpoint_best)
    ]


def _collect_candidates(args: argparse.Namespace, spec: FinalistSpec) -> list[Candidate]:
    candidates: list[Candidate] = []
    for index, raw in enumerate(args.sequence, start=1):
        sequence = _validate_sequence(raw, expected_length=spec.binder_length)
        _validate_case_sequence(sequence, spec)
        candidates.append(
            Candidate(
                candidate_id=f"candidate_{index}",
                sequence=sequence,
                source="explicit_sequence",
                is_baseline=_is_declared_baseline(sequence, spec),
            )
        )
    for index, raw in enumerate(args.candidate, start=1):
        if "=" not in raw:
            raise FinalistValidationError(
                f"--candidate value #{index} must have ID=SEQ form"
            )
        raw_id, raw_sequence = raw.split("=", 1)
        sequence = _validate_sequence(raw_sequence, expected_length=spec.binder_length)
        _validate_case_sequence(sequence, spec)
        candidates.append(
            Candidate(
                candidate_id=_safe_id(raw_id, fallback=f"candidate_{index}"),
                sequence=sequence,
                source="explicit_named_sequence",
                is_baseline=_is_declared_baseline(sequence, spec),
            )
        )
    candidates.extend(_read_outer_candidates(args.outer_output, spec))
    candidates.extend(_read_checkpoint_candidates(args.checkpoint, spec))
    selected = _rank_and_deduplicate(candidates, top_k=int(args.top_k))
    if not selected:
        raise FinalistValidationError("no valid finalist sequence was recovered")
    return selected


def _conda_environment_available(name: str | None) -> bool:
    requested = str(name or "").strip()
    conda = shutil.which("conda") or os.environ.get("CONDA_EXE")
    if not requested or not conda:
        return False
    try:
        completed = subprocess.run(
            [str(conda), "env", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        payload = json.loads(completed.stdout) if completed.returncode == 0 else {}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return False
    envs = [str(Path(item).name) for item in payload.get("envs", [])]
    return requested in envs or Path(requested).expanduser().is_dir()


def _af3_capability(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from astevolve.providers.alphafold3 import describe_alphafold3_setup
        from astevolve.providers.alphafold3_runner import (
            probe_alphafold3_environment,
        )

        setup = dict(describe_alphafold3_setup())
        requested_probe = dict(
            probe_alphafold3_environment(args.af3_conda_env)
        )
    except Exception as exc:
        return {
            "provider": "alphafold3",
            "can_execute": False,
            "reason": f"capability probe failed: {type(exc).__name__}: {exc}",
        }
    model_dir = Path(args.af3_model_dir).expanduser().resolve()
    weight = model_dir / "af3.bin.zst"
    conda_ready = _conda_environment_available(args.af3_conda_env)
    setup.update(
        {
            "provider": "alphafold3",
            "requested_model_dir": str(model_dir),
            "requested_weight_path": str(weight),
            "requested_weight_available": weight.is_file(),
            "requested_conda_env": args.af3_conda_env,
            "requested_conda_env_available": conda_ready,
            "requested_environment_probe": requested_probe,
        }
    )
    setup["can_execute"] = bool(
        setup.get("source_available")
        and setup.get("source_revision_ok")
        and requested_probe.get("gpu_dependencies_ready")
        and weight.is_file()
        and conda_ready
    )
    setup["reason"] = (
        "official AF3 MSA-free runtime is ready"
        if setup["can_execute"]
        else "official AF3 MSA-free runtime is incomplete"
    )
    return setup


def _gpu_use() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    processes: list[dict[str, int]] = []
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2:
                continue
            try:
                processes.append(
                    {"pid": int(fields[0]), "used_memory_mib": int(fields[1])}
                )
            except ValueError:
                continue
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "processes": processes,
        "total_used_memory_mib": sum(item["used_memory_mib"] for item in processes),
    }


def _assert_gpu_available(args: argparse.Namespace, report: Mapping[str, Any]) -> None:
    if args.allow_busy_gpu:
        return
    if not report.get("available"):
        raise FinalistValidationError(
            "cannot verify GPU occupancy; use --allow-busy-gpu only after manual inspection"
        )
    used = int(report.get("total_used_memory_mib") or 0)
    if used > int(args.max_gpu_memory_used_mib):
        raise FinalistValidationError(
            f"GPU already uses {used} MiB; limit is {args.max_gpu_memory_used_mib} MiB"
        )


def _invoke_alphafold3(**kwargs: Any) -> Mapping[str, Any]:
    from astevolve.providers.registry import run_structure_confidence_complex

    return run_structure_confidence_complex(provider="alphafold3", **kwargs)


def _numeric_mapping(value: Any, *, limit: int = 80) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, raw in value.items():
        number = _finite_float(raw)
        if number is not None:
            result[str(key)] = number
        if len(result) >= limit:
            break
    return result


def _artifact_manifest(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    direct_keys = (
        "cif_path",
        "structure_path",
        "summary_json",
        "confidence_json",
        "input_json",
        "status_json",
    )
    paths: dict[Path, set[str]] = {}
    for key in direct_keys:
        raw = result.get(key)
        if raw:
            path = Path(str(raw)).expanduser().resolve()
            if path.is_file():
                paths.setdefault(path, set()).add(key)
    for key in ("out_dir", "alphafold3_output_dir"):
        raw = result.get(key)
        if not raw:
            continue
        directory = Path(str(raw)).expanduser().resolve()
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.is_file():
                    paths.setdefault(path.resolve(), set()).add(key)
    records = [
        {
            "roles": sorted(roles),


            "artifact_path": str(path),
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path, roles in sorted(paths.items(), key=lambda item: str(item[0]))
    ]
    return records


def _run_state(
    candidate: Candidate,
    state: StateSpec,
    *,
    fixed_by_id: Mapping[str, str],
    spec: FinalistSpec,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sequences = {spec.binder_entity_id: candidate.sequence, **fixed_by_id}
    entities = [
        {
            "type": "protein",
            "id": entity_id,
            "sequence": sequences[entity_id],
            "count": 1,
        }
        for entity_id in state.entity_ids
    ]
    prefix = f"{_safe_id(spec.case_id)}__{_safe_id(candidate.candidate_id)}__s{seed}"
    try:
        result = dict(
            _invoke_alphafold3(
                pred_name=f"{prefix}__{_safe_id(state.name)}",
                entities=entities,
                metric="plddt",
                seed=int(seed),
                model_dir=args.af3_model_dir,
                conda_env=args.af3_conda_env,
                timeout=int(args.af3_timeout),
                run_data_pipeline=False,
                db_dir=None,
                num_recycles=int(args.af3_num_recycles),
                num_diffusion_samples=int(args.af3_num_diffusion_samples),
                flash_attention_implementation=str(args.af3_flash_attention),
                gpu_device=int(args.af3_gpu_device),
            )
        )
    except Exception as exc:
        return {
            "state": state.name,
            "role": state.role,
            "provider": "alphafold3",
            "status": "exception",
            "success": False,
            "entity_sequence_sha256": {
                key: hashlib.sha256(sequences[key].encode("ascii")).hexdigest()
                for key in state.entity_ids
            },
            "metrics": {},
            "artifacts": [],
            "artifact_manifest_sha256": _canonical_hash([]),
            "error": f"{type(exc).__name__}: {exc}",
        }
    metrics = _numeric_mapping(result.get("metrics"))
    artifacts = _artifact_manifest(result)
    cif_values = {
        str(Path(str(result.get(key))).expanduser().resolve())
        for key in ("cif_path", "structure_path")
        if result.get(key)
    }
    artifact_paths = {item["path"] for item in artifacts}
    has_cif = bool(cif_values & artifact_paths)
    status = str(result.get("status") or result.get("runner_status") or "unknown")
    provider = str(result.get("provider") or "alphafold3").lower()
    status_ok = status.lower() in {"ok", "success", "completed"}
    success = bool(
        provider in {"alphafold3", "af3"}
        and status_ok
        and metrics
        and has_cif
        and artifacts
    )
    return {
        "state": state.name,
        "role": state.role,
        "provider": "alphafold3",
        "status": "ok" if success else status,
        "runner_status": status,
        "success": success,
        "entity_sequence_sha256": {
            key: hashlib.sha256(sequences[key].encode("ascii")).hexdigest()
            for key in state.entity_ids
        },
        "metrics": metrics,
        "chain_metrics": result.get("chain_metrics", {}),
        "chain_pair_metrics": result.get("chain_pair_metrics", {}),
        "artifacts": artifacts,
        "artifact_manifest_sha256": _canonical_hash(artifacts),
        "warnings": [str(item)[:1000] for item in (result.get("warnings") or [])],
    }


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _parser(spec: FinalistSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Strict, closed-book AF3 finalist validation for {spec.case_id}."
    )
    parser.add_argument("--outer-output", action="append", default=[], metavar="PATH")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="PATH")
    parser.add_argument("--sequence", action="append", default=[], metavar="SEQ")
    parser.add_argument("--candidate", action="append", default=[], metavar="ID=SEQ")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", metavar="JSON")
    parser.add_argument("--artifact-root", metavar="DIR")
    parser.add_argument(
        "--af3-model-dir",
        default=os.environ.get("ASTEVOLVE_AF3_MODEL_DIR")
        or str(model_path("alphafold3")),
    )
    parser.add_argument(
        "--af3-conda-env",
        default=os.environ.get("ASTEVOLVE_AF3_CONDA_ENV", "alphafold3"),
    )
    parser.add_argument("--af3-timeout", type=int, default=7200)
    parser.add_argument("--af3-num-recycles", type=int, default=10)
    parser.add_argument("--af3-num-diffusion-samples", type=int, default=1)
    parser.add_argument(
        "--af3-flash-attention",
        choices=("triton", "cudnn", "xla"),
        default=os.environ.get("ASTEVOLVE_AF3_FLASH_ATTENTION", "triton"),
    )
    parser.add_argument("--af3-gpu-device", type=int, default=0)
    parser.add_argument("--max-gpu-memory-used-mib", type=int, default=4096)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    return parser


def _artifact_root(args: argparse.Namespace, spec: FinalistSpec) -> Path:
    if args.artifact_root:
        return Path(args.artifact_root).expanduser().resolve()
    if args.output:
        report = Path(args.output).expanduser().resolve()
        return report.parent / f"{report.stem}_af3_artifacts"
    configured = os.environ.get("ASTEVOLVE_AF3_TMP")
    if configured:
        return Path(configured).expanduser().resolve()
    return tmp_root(f"{_safe_id(spec.case_id)}_af3_finalists")


def run(spec: FinalistSpec, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    seeds = tuple(dict.fromkeys(args.seed or [101]))
    if any(seed < 0 for seed in seeds):
        raise FinalistValidationError("seeds must be non-negative")
    candidates = _collect_candidates(args, spec)
    capability = _af3_capability(args)
    gpu = _gpu_use()
    fixed_by_id = {entity.entity_id: entity.sequence for entity in spec.fixed_entities}
    root = _artifact_root(args, spec)
    report: dict[str, Any] = {
        "schema_version": REPORT_VERSION,
        "created_at": _utc_now(),
        "case_id": spec.case_id,
        "provider": "alphafold3",
        "state_names": [state.name for state in spec.states],
        "requested": {
            "providers": ["alphafold3"],
            "top_k": int(args.top_k),
            "seeds": list(seeds),
            "execute": bool(args.execute),
            "msa_mode": "msa_free",
        },
        "oracle_boundary": {
            "closed_book": True,
            "native_or_hidden_oracle_accessed": False,
            "candidate_sources": ["explicit_outer_output", "frozen_checkpoint"],
            "fixed_public_entities": {
                entity.entity_id: {
                    "length": len(entity.sequence),
                    "sequence_sha256": hashlib.sha256(
                        entity.sequence.encode("ascii")
                    ).hexdigest(),
                }
                for entity in spec.fixed_entities
            },
        },
        "capability": capability,
        "gpu_before_execution": gpu,
        "artifact_root": str(root),
        "candidates": [candidate.public_record() for candidate in candidates],
        "selection_policy": {
            "checkpoint_best": (
                "mandatory structural validation when sequence integrity passes, "
                "including baseline or source-hard-gate-failed checkpoints"
            ),
            "additional_candidates": "feasibility_first",
            "scientific_go_is_separate": True,
        },
        "execution_plan": {
            "candidate_count": len(candidates),
            "seeds": list(seeds),
            "states_per_candidate_seed": [state.name for state in spec.states],
            "expected_af3_process_launches": len(candidates)
            * len(seeds)
            * len(spec.states),
            "scheduling": "serial",
        },
        "interpretation_boundary": spec.interpretation_boundary,
    }
    if not args.execute:
        report.update(
            {
                "status": "planned_no_inference",
                "ok": True,
                "models_invoked": False,
                "results": [],
                "all_requested_validations_successful": False,
            }
        )
        return report, 0
    if not capability.get("can_execute"):
        raise FinalistValidationError("official AlphaFold 3 runtime is unavailable")
    _assert_gpu_available(args, gpu)
    root.mkdir(parents=True, exist_ok=True)
    os.environ["ASTEVOLVE_AF3_TMP"] = str(root)

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_result: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "sequence_sha256": candidate.sequence_sha256,
            "source_gate_status": candidate.public_record()["source_gate_status"],
            "is_baseline": candidate.is_baseline,
            "providers": [],
        }
        provider_result: dict[str, Any] = {
            "provider": "alphafold3",
            "seeds": [],
        }
        for seed in seeds:
            states = [
                _run_state(
                    candidate,
                    state,
                    fixed_by_id=fixed_by_id,
                    spec=spec,
                    seed=seed,
                    args=args,
                )
                for state in spec.states
            ]
            provider_result["seeds"].append(
                {
                    "seed": seed,
                    "states": states,
                    "all_states_successful": all(
                        bool(state.get("success")) for state in states
                    ),
                }
            )
        provider_result["all_seeds_successful"] = all(
            bool(seed_result.get("all_states_successful"))
            for seed_result in provider_result["seeds"]
        )
        candidate_result["providers"].append(provider_result)
        candidate_result["all_requested_validations_successful"] = bool(
            provider_result["all_seeds_successful"]
        )
        results.append(candidate_result)
    ok = bool(results) and all(
        bool(candidate.get("all_requested_validations_successful"))
        for candidate in results
    )
    report.update(
        {
            "status": "ok" if ok else "provider_validation_failed",
            "ok": ok,
            "models_invoked": True,
            "results": results,
            "all_requested_validations_successful": ok,
        }
    )
    report["report_payload_sha256"] = _canonical_hash(
        {key: value for key, value in report.items() if key != "report_payload_sha256"}
    )
    return report, 0 if ok else 1


def run_cli(spec: FinalistSpec, argv: Sequence[str] | None = None) -> int:
    parser = _parser(spec)
    args = parser.parse_args(argv)
    try:
        report, status = run(spec, args)
    except Exception as exc:
        report = {
            "schema_version": REPORT_VERSION,
            "created_at": _utc_now(),
            "case_id": spec.case_id,
            "provider": "alphafold3",
            "status": "request_failed",
            "ok": False,
            "models_invoked": False,
            "all_requested_validations_successful": False,
            "oracle_boundary": {
                "closed_book": True,
                "native_or_hidden_oracle_accessed": False,
            },
            "error": f"{type(exc).__name__}: {exc}",
        }
        status = 2
    if args.output:
        _atomic_write(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return status


__all__ = [
    "FinalistSpec",
    "FinalistValidationError",
    "FixedEntity",
    "StateSpec",
    "run",
    "run_cli",
]
