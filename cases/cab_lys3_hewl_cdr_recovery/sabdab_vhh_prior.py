

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, Mapping, Optional

import numpy as np

from astevolve.runtime.paths import data_path, model_root


REGIONS = ("whole_vhh", "framework", "cdr1", "cdr2", "cdr3")
DEFAULT_INDEX_DIR = data_path(
    "cab_lys3_hewl_cdr_recovery",
    "sabdab_vhh_reference",
    "embeddings",
    "esm2_t6_8M_UR50D",
)
DEFAULT_MODEL_DIR = model_root() / "esm2_t6_8M_UR50D"


class SabdabVHHPriorUnavailable(RuntimeError):
    pass


def _resolve_path(
    configured: Optional[str],
    environment_name: str,
    default: Path,
) -> Path:
    raw = str(configured or os.environ.get(environment_name) or default).strip()
    return Path(raw).expanduser().resolve()


def _read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean_pool(
    sequences: list[str],
    *,
    tokenizer: Any,
    model: Any,
    device: Any,
) -> np.ndarray:
    import torch

    encoded = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device).bool()
    residue_mask = attention_mask.clone()
    for token_id in set(tokenizer.all_special_ids):
        residue_mask &= input_ids.ne(token_id)
    with torch.inference_mode():
        hidden = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
    denom = residue_mask.sum(dim=1, keepdim=True).clamp_min(1)
    pooled = (hidden * residue_mask.unsqueeze(-1)).sum(dim=1) / denom
    pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
    return pooled.cpu().numpy().astype(np.float32, copy=False)


def calibrated_ood_penalty(value: float, *, q01: float, q05: float) -> float:


    numeric = float(value)
    lower = float(q01)
    upper = float(q05)
    if upper <= lower:
        raise ValueError("SAbDab calibration requires q05 > q01")
    if numeric >= upper:
        return 0.0
    if numeric <= lower:
        return 1.0
    return float((upper - numeric) / (upper - lower))


class SabdabVHHPrior:


    def __init__(
        self,
        *,
        index_dir: Path,
        model_dir: Path,
        device_name: str,
        top_k: int,
    ) -> None:
        if top_k < 1:
            raise ValueError("SAbDab top_k must be positive")
        self.index_dir = index_dir
        self.model_dir = model_dir
        self.device_name = str(device_name).strip().lower()
        self.top_k = int(top_k)
        self._lock = threading.RLock()
        self._score_cache: Dict[str, Dict[str, Any]] = {}
        self._model = None
        self._tokenizer = None
        self._device = None

        required = [
            self.index_dir / "embedding_metadata.csv",
            self.index_dir / "embedding_manifest.json",
            self.index_dir / "calibration.json",
            self.index_dir / "target_profile.json",
            *(self.index_dir / f"{region}.npy" for region in REGIONS),
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise SabdabVHHPriorUnavailable(
                "missing SAbDab VHH prior artifacts: " + ", ".join(missing)
            )
        if not self.model_dir.is_dir():
            raise SabdabVHHPriorUnavailable(
                f"missing local ESM-2 model directory: {self.model_dir}"
            )

        self.metadata = _read_metadata(self.index_dir / "embedding_metadata.csv")
        self.calibration = json.loads(
            (self.index_dir / "calibration.json").read_text(encoding="utf-8")
        )
        self.target_profile = json.loads(
            (self.index_dir / "target_profile.json").read_text(encoding="utf-8")
        )
        self.matrices = {
            region: np.load(
                self.index_dir / f"{region}.npy",
                mmap_mode="r",
            )
            for region in REGIONS
        }
        row_count = len(self.metadata)
        if row_count < 1 or any(
            matrix.shape != (row_count, 320)
            for matrix in self.matrices.values()
        ):
            raise SabdabVHHPriorUnavailable(
                "SAbDab VHH embedding matrices are not row-aligned 320D indices"
            )
        if int(self.calibration.get("reference_rows", -1)) != row_count:
            raise SabdabVHHPriorUnavailable(
                "SAbDab calibration/reference row count mismatch"
            )
        self.cluster_rows: Dict[str, np.ndarray] = {}
        for index, row in enumerate(self.metadata):
            cluster = row.get("cdrh3_cluster_95") or f"row:{index}"
            self.cluster_rows.setdefault(cluster, []).append(index)
        self.cluster_rows = {
            cluster: np.asarray(indices, dtype=np.int64)
            for cluster, indices in self.cluster_rows.items()
        }

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            raise SabdabVHHPriorUnavailable(
                f"ESM-2 runtime dependencies unavailable: {exc}"
            ) from exc
        requested = self.device_name
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise SabdabVHHPriorUnavailable(
                "SAbDab VHH prior requested CUDA but CUDA is unavailable"
            )
        if requested not in {"cpu", "cuda"}:
            raise SabdabVHHPriorUnavailable(
                f"unsupported SAbDab VHH prior device {requested!r}"
            )
        self._device = torch.device(requested)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self._model = AutoModel.from_pretrained(
            self.model_dir,
            local_files_only=True,
            add_pooling_layer=False,
        )
        self._model.eval().to(self._device)

    def _split(self, sequence: str) -> Dict[str, str]:
        native = str(self.target_profile["native_sequence"])
        candidate = str(sequence).strip().upper()
        if len(candidate) != len(native):
            raise ValueError(
                f"VHH prior candidate length {len(candidate)} differs from {len(native)}"
            )
        if any(residue not in "ACDEFGHIKLMNPQRSTVWY" for residue in candidate):
            raise ValueError("VHH prior candidate contains a non-canonical residue")
        indices = self.target_profile["region_sequence_indices_0based"]
        result = {"whole_vhh": candidate}
        for region in REGIONS[1:]:
            result[region] = "".join(
                candidate[int(index)] for index in indices[region]
            )
        return result

    def _cluster_balanced_score(
        self,
        region: str,
        query: np.ndarray,
    ) -> tuple[float, list[dict[str, Any]]]:
        similarities = np.asarray(self.matrices[region] @ query)
        cluster_best = [
            (
                float(np.max(similarities[indices])),
                int(indices[int(np.argmax(similarities[indices]))]),
            )
            for indices in self.cluster_rows.values()
        ]
        cluster_best.sort(key=lambda item: (-item[0], item[1]))
        selected = cluster_best[: min(self.top_k, len(cluster_best))]
        score = float(np.mean([value for value, _index in selected]))
        nearest = [
            {
                "reference_id": self.metadata[index]["reference_id"],
                "pdb_id": self.metadata[index]["pdb_id"],
                "cdrh3_cluster_95": self.metadata[index].get(
                    "cdrh3_cluster_95"
                ),
                "cosine": value,
            }
            for value, index in selected[:5]
        ]
        return score, nearest

    def score(self, sequence: str) -> Dict[str, Any]:
        candidate = str(sequence).strip().upper()
        with self._lock:
            cached = self._score_cache.get(candidate)
            if cached is not None:
                out = deepcopy(cached)
                out["cache_hit"] = True
                return out
            self._ensure_model()
            segmented = self._split(candidate)
            vectors = _mean_pool(
                [segmented[region] for region in REGIONS],
                tokenizer=self._tokenizer,
                model=self._model,
                device=self._device,
            )
            region_scores: Dict[str, Dict[str, Any]] = {}
            aggregate_penalty = 0.0
            for index, region in enumerate(REGIONS):
                cosine, nearest = self._cluster_balanced_score(
                    region,
                    vectors[index],
                )
                calibration = self.calibration["regions"][region]
                q01 = float(calibration["ood_full_penalty_below"])
                q05 = float(calibration["ood_no_penalty_at_or_above"])
                penalty = calibrated_ood_penalty(cosine, q01=q01, q05=q05)
                distribution = calibration[
                    "heldout_cluster_balanced_top_k_cosines"
                ]
                percentile = float(
                    bisect_right(distribution, cosine) / len(distribution)
                )
                weight = float(calibration.get("weight", 0.0))
                aggregate_penalty += weight * penalty
                region_scores[region] = {
                    "cluster_balanced_top_k_mean_cosine": cosine,
                    "heldout_percentile": percentile,
                    "ood_penalty": penalty,
                    "plausibility": 1.0 - penalty,
                    "weight": weight,
                    "q01": q01,
                    "q05": q05,
                    "nearest": nearest,
                }
            result = {
                "schema_version": "cab_lys3.sabdab_vhh_prior_score.v1",
                "available": True,
                "sequence": candidate,
                "plausibility": float(max(0.0, min(1.0, 1.0 - aggregate_penalty))),
                "ood_penalty": float(max(0.0, min(1.0, aggregate_penalty))),
                "top_k": self.top_k,
                "device": str(self._device),
                "index_dir": str(self.index_dir),
                "model_dir": str(self.model_dir),
                "cache_hit": False,
                "regions": region_scores,
            }
            self._score_cache[candidate] = deepcopy(result)
            return result


_PRIOR_CACHE: Dict[tuple[str, str, str, int], SabdabVHHPrior] = {}
_PRIOR_CACHE_LOCK = threading.RLock()


def get_sabdab_vhh_prior(
    *,
    index_dir: Optional[str] = None,
    model_dir: Optional[str] = None,
    device: str = "auto",
    top_k: int = 20,
) -> SabdabVHHPrior:
    resolved_index = _resolve_path(
        index_dir,
        "ASTEVOLVE_CAB_LYS3_SABDAB_INDEX",
        DEFAULT_INDEX_DIR,
    )
    resolved_model = _resolve_path(
        model_dir,
        "ASTEVOLVE_CAB_LYS3_ESM2_MODEL",
        DEFAULT_MODEL_DIR,
    )
    key = (
        str(resolved_index),
        str(resolved_model),
        str(device).strip().lower(),
        int(top_k),
    )
    with _PRIOR_CACHE_LOCK:
        prior = _PRIOR_CACHE.get(key)
        if prior is None:
            prior = SabdabVHHPrior(
                index_dir=resolved_index,
                model_dir=resolved_model,
                device_name=key[2],
                top_k=key[3],
            )
            _PRIOR_CACHE[key] = prior
        return prior


def score_sabdab_vhh_prior(
    sequence: str,
    score_config: Mapping[str, Any],
) -> Dict[str, Any]:
    prior = get_sabdab_vhh_prior(
        index_dir=str(score_config.get("sabdab_prior_index_dir") or "") or None,
        model_dir=str(score_config.get("sabdab_prior_model_dir") or "") or None,
        device=str(score_config.get("sabdab_prior_device") or "auto"),
        top_k=int(score_config.get("sabdab_prior_top_k", 20) or 20),
    )
    return prior.score(sequence)


__all__ = [
    "DEFAULT_INDEX_DIR",
    "DEFAULT_MODEL_DIR",
    "REGIONS",
    "SabdabVHHPrior",
    "SabdabVHHPriorUnavailable",
    "calibrated_ood_penalty",
    "get_sabdab_vhh_prior",
    "score_sabdab_vhh_prior",
]
