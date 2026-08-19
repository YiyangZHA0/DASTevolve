

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, runtime_checkable


CANONICAL_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
MASKED_LM_MODEL_PATH_ENV = "ASTEVOLVE_MASKED_LM_MODEL_DIR"
MASKED_LM_DEVICE_ENV = "ASTEVOLVE_MASKED_LM_DEVICE"
MASKED_LM_RESULT_VERSION = "astevolve.masked_lm_prior.v1"


class MaskedLMError(RuntimeError):
    pass


class MaskedLMUnavailableError(MaskedLMError):
    pass


class MaskedLMBackendError(MaskedLMError):
    pass


class MaskedLMInputError(ValueError):
    pass


@runtime_checkable
class MaskedLMBackend(Protocol):


    model_id: str

    def masked_log_probabilities(
        self,
        *,
        sequence: str,
        positions: Sequence[int],
        residues: Sequence[str],
    ) -> Sequence[Mapping[str, float]]:
        ...


@dataclass(frozen=True)
class MaskedMarginal:


    model_id: str
    position: int
    parent_residue: str
    log_probabilities: Mapping[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MASKED_LM_RESULT_VERSION,
            "model_id": self.model_id,
            "position": self.position,
            "parent_residue": self.parent_residue,
            "log_probabilities": dict(self.log_probabilities),
        }


@dataclass(frozen=True)
class SubstitutionDelta:


    position: int
    parent_residue: str
    candidate_residue: str
    parent_log_probability: float
    candidate_log_probability: float
    delta: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "parent_residue": self.parent_residue,
            "candidate_residue": self.candidate_residue,
            "parent_log_probability": self.parent_log_probability,
            "candidate_log_probability": self.candidate_log_probability,
            "delta": self.delta,
        }


@dataclass(frozen=True)
class CandidateParentDelta:


    model_id: str
    parent_sequence: str
    candidate_sequence: str
    substitutions: tuple[SubstitutionDelta, ...]
    delta_sum: float
    delta_mean: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MASKED_LM_RESULT_VERSION,
            "model_id": self.model_id,
            "parent_sequence": self.parent_sequence,
            "candidate_sequence": self.candidate_sequence,
            "substitutions": [item.as_dict() for item in self.substitutions],
            "delta_sum": self.delta_sum,
            "delta_mean": self.delta_mean,
        }


class TransformersMaskedLMBackend:


    def __init__(self, model_path: str | Path, *, device: str = "cpu") -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = str(device).strip() or "cpu"
        self.model_id = f"transformers:{self.model_path}"
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def _ensure_loaded(self) -> tuple[Any, Any, Any]:
        if self.is_loaded:
            return self._torch, self._model, self._tokenizer

        with self._load_lock:
            if self.is_loaded:
                return self._torch, self._model, self._tokenizer
            if not self.model_path.is_dir():
                raise MaskedLMUnavailableError(
                    "Local masked protein-LM directory does not exist: "
                    f"{self.model_path}. Network downloads are disabled."
                )
            try:
                import torch
                from transformers import AutoModelForMaskedLM, AutoTokenizer
            except ImportError as exc:
                raise MaskedLMUnavailableError(
                    "Masked protein-LM inference requires the optional torch and "
                    "transformers dependencies. Install them in the provider "
                    "environment; no fallback score was produced."
                ) from exc

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    local_files_only=True,
                    trust_remote_code=False,
                )
                model = AutoModelForMaskedLM.from_pretrained(
                    str(self.model_path),
                    local_files_only=True,
                    trust_remote_code=False,
                )
                if tokenizer.mask_token is None or tokenizer.mask_token_id is None:
                    raise ValueError("tokenizer does not define a mask token")
                model = model.to(self.device)
                model.eval()
            except Exception as exc:
                raise MaskedLMUnavailableError(
                    "Failed to load a masked protein LM from the local directory "
                    f"{self.model_path}: {exc}. Network downloads are disabled."
                ) from exc

            self._torch = torch
            self._model = model
            self._tokenizer = tokenizer
            return torch, model, tokenizer

    @staticmethod
    def _residue_token_ids(
        tokenizer: Any,
        residues: Sequence[str],
    ) -> tuple[int, ...]:
        token_ids: list[int] = []
        for residue in residues:
            token_id = tokenizer.convert_tokens_to_ids(residue)
            if (
                not isinstance(token_id, int)
                or token_id < 0
                or token_id == tokenizer.unk_token_id
            ):
                raise MaskedLMBackendError(
                    f"Local tokenizer has no unique token for residue {residue!r}"
                )
            token_ids.append(token_id)
        if len(set(token_ids)) != len(token_ids):
            raise MaskedLMBackendError(
                "Local tokenizer maps canonical residues to duplicate token ids"
            )
        return tuple(token_ids)

    def masked_log_probabilities(
        self,
        *,
        sequence: str,
        positions: Sequence[int],
        residues: Sequence[str],
    ) -> Sequence[Mapping[str, float]]:
        torch, model, tokenizer = self._ensure_loaded()
        if not positions:
            return ()

        residue_token_ids = self._residue_token_ids(tokenizer, residues)
        masked_sequences = [
            sequence[:position] + tokenizer.mask_token + sequence[position + 1 :]
            for position in positions
        ]

        try:
            encoded = tokenizer(
                masked_sequences,
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_attention_mask=True,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            input_ids = encoded.get("input_ids")
            if input_ids is None:
                raise ValueError("tokenizer did not return input_ids")

            masked_token_positions: list[int] = []
            for row_index in range(len(masked_sequences)):
                hits = torch.nonzero(
                    input_ids[row_index] == tokenizer.mask_token_id,
                    as_tuple=False,
                ).reshape(-1)
                if int(hits.numel()) != 1:
                    raise ValueError(
                        "each encoded sequence must contain exactly one mask token"
                    )
                masked_token_positions.append(int(hits[0].item()))

            with self._inference_lock, torch.inference_mode():
                outputs = model(**encoded)
            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise ValueError("masked-LM output does not expose logits")
            selected_logits = torch.stack(
                [
                    logits[row_index, token_position]
                    for row_index, token_position in enumerate(masked_token_positions)
                ]
            )
            log_probabilities = torch.log_softmax(selected_logits, dim=-1)
        except MaskedLMError:
            raise
        except Exception as exc:
            raise MaskedLMBackendError(
                f"Local masked protein-LM inference failed: {exc}"
            ) from exc

        normalized: list[dict[str, float]] = []
        for row in log_probabilities:
            normalized.append(
                {
                    residue: float(row[token_id].detach().cpu().item())
                    for residue, token_id in zip(residues, residue_token_ids)
                }
            )
        return tuple(normalized)


def _validate_sequence(sequence: str, *, field: str) -> str:
    if not isinstance(sequence, str) or not sequence:
        raise MaskedLMInputError(f"{field} must be a non-empty protein sequence")
    normalized = sequence.strip().upper()
    if normalized != sequence:
        raise MaskedLMInputError(
            f"{field} must already be uppercase and contain no surrounding whitespace"
        )
    invalid = sorted(set(normalized).difference(CANONICAL_AMINO_ACIDS))
    if invalid:
        raise MaskedLMInputError(
            f"{field} contains non-canonical residues: {''.join(invalid)}"
        )
    return normalized


def _validate_positions(sequence: str, positions: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for position in positions:
        if isinstance(position, bool) or not isinstance(position, int):
            raise MaskedLMInputError("masked positions must be zero-based integers")
        if position < 0 or position >= len(sequence):
            raise MaskedLMInputError(
                f"masked position {position} is outside sequence length {len(sequence)}"
            )
        normalized.append(position)
    if len(set(normalized)) != len(normalized):
        raise MaskedLMInputError("masked positions must be unique")
    return tuple(normalized)


class MaskedProteinLMPrior:


    name = "masked_lm"

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        device: str | None = None,
        batch_size: int = 32,
        backend: MaskedLMBackend | None = None,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self._configured_model_path = (
            Path(model_path).expanduser() if model_path is not None else None
        )
        self.device = str(
            device if device is not None else os.environ.get(MASKED_LM_DEVICE_ENV, "cpu")
        ).strip() or "cpu"
        self.batch_size = batch_size
        self._backend: MaskedLMBackend | None = backend
        self._backend_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        backend = self._backend
        if backend is None:
            return False
        loaded = getattr(backend, "is_loaded", None)
        return bool(loaded) if loaded is not None else True

    def _ensure_backend(self) -> MaskedLMBackend:
        if self._backend is not None:
            return self._backend
        with self._backend_lock:
            if self._backend is not None:
                return self._backend
            configured = self._configured_model_path
            if configured is None:
                raw_path = os.environ.get(MASKED_LM_MODEL_PATH_ENV, "").strip()
                configured = Path(raw_path).expanduser() if raw_path else None
            if configured is None:
                raise MaskedLMUnavailableError(
                    "No local masked protein-LM is configured. Pass model_path=... "
                    f"or set {MASKED_LM_MODEL_PATH_ENV}. Network downloads and "
                    "synthetic fallback scores are disabled."
                )
            resolved = configured.resolve()
            if not resolved.is_dir():
                raise MaskedLMUnavailableError(
                    "Local masked protein-LM directory does not exist: "
                    f"{resolved}. Network downloads and synthetic fallback scores "
                    "are disabled."
                )
            self._backend = TransformersMaskedLMBackend(
                resolved,
                device=self.device,
            )
            return self._backend

    @staticmethod
    def _model_id(backend: MaskedLMBackend) -> str:
        value = str(getattr(backend, "model_id", "")).strip()
        return value or backend.__class__.__name__

    @staticmethod
    def _normalize_backend_rows(
        rows: Sequence[Mapping[str, float]],
        *,
        expected: int,
    ) -> tuple[dict[str, float], ...]:
        if len(rows) != expected:
            raise MaskedLMBackendError(
                "masked-LM backend returned "
                f"{len(rows)} rows for {expected} requested positions"
            )
        normalized: list[dict[str, float]] = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise MaskedLMBackendError(
                    f"masked-LM backend row {row_index} is not a mapping"
                )
            values: dict[str, float] = {}
            for residue in CANONICAL_AMINO_ACIDS:
                if residue not in row:
                    raise MaskedLMBackendError(
                        f"masked-LM backend row {row_index} omitted residue {residue}"
                    )
                try:
                    value = float(row[residue])
                except (TypeError, ValueError) as exc:
                    raise MaskedLMBackendError(
                        f"masked-LM backend returned a non-numeric value for {residue}"
                    ) from exc
                if not math.isfinite(value) or value > 1e-6:
                    raise MaskedLMBackendError(
                        "masked-LM backend values must be finite natural-log "
                        f"probabilities; got {value!r} for {residue}"
                    )
                values[residue] = value
            normalized.append(values)
        return tuple(normalized)

    def batch_masked_marginals(
        self,
        sequence: str,
        positions: Sequence[int],
    ) -> tuple[MaskedMarginal, ...]:


        normalized_sequence = _validate_sequence(sequence, field="sequence")
        normalized_positions = _validate_positions(normalized_sequence, positions)
        backend = self._ensure_backend()
        model_id = self._model_id(backend)
        if not normalized_positions:
            return ()

        rows: list[dict[str, float]] = []
        method = getattr(backend, "masked_log_probabilities", None)
        if not callable(method):
            raise MaskedLMBackendError(
                "masked-LM backend does not implement masked_log_probabilities"
            )
        for offset in range(0, len(normalized_positions), self.batch_size):
            chunk = normalized_positions[offset : offset + self.batch_size]
            try:
                raw_rows = method(
                    sequence=normalized_sequence,
                    positions=chunk,
                    residues=CANONICAL_AMINO_ACIDS,
                )
            except MaskedLMError:
                raise
            except Exception as exc:
                raise MaskedLMBackendError(
                    f"masked-LM backend {model_id!r} failed: {exc}"
                ) from exc
            if not isinstance(raw_rows, Sequence):
                raise MaskedLMBackendError(
                    "masked-LM backend output must be a sequence of mappings"
                )
            rows.extend(
                self._normalize_backend_rows(raw_rows, expected=len(chunk))
            )

        return tuple(
            MaskedMarginal(
                model_id=model_id,
                position=position,
                parent_residue=normalized_sequence[position],
                log_probabilities=row,
            )
            for position, row in zip(normalized_positions, rows)
        )

    def candidate_parent_deltas(
        self,
        parent_sequence: str,
        candidate_sequences: Sequence[str],
    ) -> tuple[CandidateParentDelta, ...]:


        parent = _validate_sequence(parent_sequence, field="parent_sequence")
        candidates = tuple(
            _validate_sequence(candidate, field=f"candidate_sequences[{index}]")
            for index, candidate in enumerate(candidate_sequences)
        )
        for index, candidate in enumerate(candidates):
            if len(candidate) != len(parent):
                raise MaskedLMInputError(
                    "masked marginal deltas require equal-length candidates; "
                    f"candidate_sequences[{index}] has length {len(candidate)} "
                    f"and parent has length {len(parent)}"
                )

        changed_positions = tuple(
            sorted(
                {
                    position
                    for candidate in candidates
                    for position, (parent_residue, candidate_residue) in enumerate(
                        zip(parent, candidate)
                    )
                    if parent_residue != candidate_residue
                }
            )
        )
        marginals = self.batch_masked_marginals(parent, changed_positions)
        backend = self._ensure_backend()
        model_id = self._model_id(backend)
        marginal_by_position = {item.position: item for item in marginals}

        output: list[CandidateParentDelta] = []
        for candidate in candidates:
            substitutions: list[SubstitutionDelta] = []
            for position, (parent_residue, candidate_residue) in enumerate(
                zip(parent, candidate)
            ):
                if parent_residue == candidate_residue:
                    continue
                row = marginal_by_position[position].log_probabilities
                parent_log_probability = row[parent_residue]
                candidate_log_probability = row[candidate_residue]
                substitutions.append(
                    SubstitutionDelta(
                        position=position,
                        parent_residue=parent_residue,
                        candidate_residue=candidate_residue,
                        parent_log_probability=parent_log_probability,
                        candidate_log_probability=candidate_log_probability,
                        delta=candidate_log_probability - parent_log_probability,
                    )
                )
            delta_sum = math.fsum(item.delta for item in substitutions)
            output.append(
                CandidateParentDelta(
                    model_id=model_id,
                    parent_sequence=parent,
                    candidate_sequence=candidate,
                    substitutions=tuple(substitutions),
                    delta_sum=delta_sum,
                    delta_mean=(
                        delta_sum / len(substitutions) if substitutions else 0.0
                    ),
                )
            )
        return tuple(output)

    def candidate_parent_delta(
        self,
        parent_sequence: str,
        candidate_sequence: str,
    ) -> CandidateParentDelta:


        return self.candidate_parent_deltas(
            parent_sequence,
            (candidate_sequence,),
        )[0]


__all__ = [
    "CANONICAL_AMINO_ACIDS",
    "CandidateParentDelta",
    "MASKED_LM_DEVICE_ENV",
    "MASKED_LM_MODEL_PATH_ENV",
    "MASKED_LM_RESULT_VERSION",
    "MaskedLMBackend",
    "MaskedLMBackendError",
    "MaskedLMError",
    "MaskedLMInputError",
    "MaskedLMUnavailableError",
    "MaskedMarginal",
    "MaskedProteinLMPrior",
    "SubstitutionDelta",
    "TransformersMaskedLMBackend",
]
