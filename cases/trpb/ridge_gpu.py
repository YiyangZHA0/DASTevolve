from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")


def _load_prior(path: Path, allowed_positions: Sequence[int]) -> dict[tuple[int, str], float]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        path,
        columns=["position_1based", "residue", "delta_from_parent"],
    )
    rows = table.to_pylist()
    lookup = {
        (int(row["position_1based"]), str(row["residue"])): float(
            row["delta_from_parent"]
        )
        for row in rows
    }
    expected = {(position, residue) for position in allowed_positions for residue in AMINO_ACIDS}
    if set(lookup) != expected:
        raise ValueError("GPU prior must contain exactly 20 residues at every allowed position")
    if any(not math.isfinite(value) for value in lookup.values()):
        raise ValueError("GPU prior contains non-finite deltas")
    return lookup


def _fit_gpu_ridge(
    torch: Any,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    dev_features: np.ndarray,
    *,
    device: Any,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    train = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    target = torch.as_tensor(train_targets, dtype=torch.float32, device=device)
    dev = torch.as_tensor(dev_features, dtype=torch.float32, device=device)
    if not train.is_cuda or not target.is_cuda or not dev.is_cuda:
        raise RuntimeError("ridge input tensors are not resident on CUDA")
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False)
    active = std > 1e-7
    active_count = int(active.sum().item())
    if active_count < 1:
        raise RuntimeError("ridge design has no active features")
    train_z = (train[:, active] - mean[active]) / std[active]
    dev_z = (dev[:, active] - mean[active]) / std[active]
    target_mean = target.mean()
    centered_target = target - target_mean
    scale = float(train_z.shape[0])
    gram = train_z.T @ train_z / scale
    gram.diagonal().add_(float(alpha))
    rhs = train_z.T @ centered_target / scale
    try:
        factor = torch.linalg.cholesky(gram)
        coefficients = torch.cholesky_solve(rhs[:, None], factor).squeeze(1)
    except Exception as exc:
        raise RuntimeError(f"CUDA ridge Cholesky solve failed: {exc}") from exc
    predictions = dev_z @ coefficients + target_mean
    if not predictions.is_cuda or not torch.isfinite(predictions).all():
        raise RuntimeError("CUDA ridge predictions are missing or non-finite")
    residual = train_z @ coefficients + target_mean - target
    report = {
        "active_features": active_count,
        "alpha": float(alpha),
        "feature_dimension": int(train.shape[1]),
        "fit_rmse": float(torch.sqrt(torch.mean(residual.square())).item()),
        "gpu_solve_verified": True,
        "train_rows": int(train.shape[0]),
    }
    active_indices = torch.nonzero(active, as_tuple=False).squeeze(1)
    weights = {
        "active_index": active_indices.detach().cpu().numpy().astype(np.int32),
        "coefficient": coefficients.detach().cpu().numpy().astype(np.float64),
        "feature_mean": mean[active].detach().cpu().numpy().astype(np.float64),
        "feature_std": std[active].detach().cpu().numpy().astype(np.float64),
        "target_mean": np.full(active_count, float(target_mean.item()), dtype=np.float64),
    }
    result = predictions.detach().cpu().numpy().astype(np.float64)
    del train, target, dev, train_z, dev_z, gram, rhs, factor, coefficients, predictions, residual
    torch.cuda.empty_cache()
    return result, report, weights
