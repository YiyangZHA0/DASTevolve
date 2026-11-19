from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

import numpy as np


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def value_hash(namespace: str, value: Any) -> str:
    return hashlib.sha256(f"{namespace}\0{canonical(value)}".encode()).hexdigest()


def normalize_ast(raw: Mapping[str, Any], allowed: Sequence[int]) -> dict[str, Any]:
    structural = []
    seen_positions: list[int] = []
    seen_ids: set[str] = set()
    for item in raw["structural_nodes"]:
        node_id = str(item["id"])
        values = sorted({int(x) for x in item["positions_1based"]})
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", node_id) or node_id in seen_ids:
            raise ValueError(f"invalid or duplicate structural node id: {node_id!r}")
        if not values:
            raise ValueError("structural nodes may not be empty")
        seen_ids.add(node_id)
        seen_positions.extend(values)
        structural.append({"id": node_id, "positions_1based": values, "ordered_positions_1based": values})
    if not 3 <= len(structural) <= 8:
        raise ValueError("structural node count must be within [3, 8]")
    if sorted(seen_positions) != sorted(int(x) for x in allowed):
        raise ValueError("structural nodes must partition the complete position allowlist")

    functional = []
    mapped: list[str] = []
    functional_ids: set[str] = set()
    for item in raw["functional_nodes"]:
        node_id = str(item["id"])
        children = [str(x) for x in item["mapped_structural_nodes"]]
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", node_id) or node_id in functional_ids:
            raise ValueError(f"invalid or duplicate functional node id: {node_id!r}")
        if not children or any(child not in seen_ids for child in children):
            raise ValueError("functional mappings must reference existing structural nodes")
        functional_ids.add(node_id)
        mapped.extend(children)
        functional.append({"id": node_id, "mapped_structural_nodes": children})
    if not 2 <= len(functional) <= 6:
        raise ValueError("functional node count must be within [2, 6]")
    if sorted(mapped) != sorted(seen_ids):
        raise ValueError("functional nodes must partition structural nodes exactly once")
    return {
        "case_id": "trpb",
        "schema_version": "dastbench.trpb.dual_ast.iterative.v1",
        "structural_ast": {"nodes": structural},
        "functional_ast": {"nodes": functional},
        "mapping_edges": [
            {"functional": f["id"], "structural": child, "type": "realizes"}
            for f in functional for child in f["mapped_structural_nodes"]
        ],
    }


def initial_ast(raw: Mapping[str, Any], allowed: Sequence[int]) -> dict[str, Any]:
    return normalize_ast(
        {
            "structural_nodes": raw["structural_ast"]["nodes"],
            "functional_nodes": raw["functional_ast"]["nodes"],
        },
        allowed,
    )


def ast_runtime(ast: Mapping[str, Any]) -> dict[str, Any]:
    structural_nodes = ast["structural_ast"]["nodes"]
    functional_nodes = ast["functional_ast"]["nodes"]
    position_to_structural = {
        int(position): str(node["id"])
        for node in structural_nodes for position in node["positions_1based"]
    }
    structural_to_functional = {
        str(child): str(node["id"])
        for node in functional_nodes for child in node["mapped_structural_nodes"]
    }
    return {
        "config": ast,
        "structural_ids": tuple(str(x["id"]) for x in structural_nodes),
        "functional_ids": tuple(str(x["id"]) for x in functional_nodes),
        "position_to_structural": position_to_structural,
        "shuffled_position_to_structural": position_to_structural,
        "structural_to_functional": structural_to_functional,
    }


def corrupt_ast(ast: Mapping[str, Any], severity: float, seed: int) -> tuple[dict[str, Any], float]:
    if severity == 0.0:
        return deepcopy(ast), 0.0
    nodes = ast["structural_ast"]["nodes"]
    positions = [int(p) for node in nodes for p in node["positions_1based"]]
    original = [str(node["id"]) for node in nodes for _p in node["positions_1based"]]
    target = int(round(severity * len(positions)))
    rng = np.random.default_rng(seed)
    best = list(original)
    best_distance = abs(target)
    for _ in range(30000):
        candidate = list(original)
        rng.shuffle(candidate)
        mismatches = sum(a != b for a, b in zip(candidate, original))
        distance = abs(mismatches - target)
        if distance < best_distance or (distance == best_distance and mismatches > sum(a != b for a, b in zip(best, original))):
            best, best_distance = candidate, distance
        if distance == 0:
            break
    grouped = {str(node["id"]): [] for node in nodes}
    for position, node_id in zip(positions, best):
        grouped[node_id].append(position)
    raw = {
        "structural_nodes": [
            {"id": str(node["id"]), "positions_1based": sorted(grouped[str(node["id"])])}
            for node in nodes
        ],
        "functional_nodes": ast["functional_ast"]["nodes"],
    }
    result = normalize_ast(raw, positions)
    achieved = sum(a != b for a, b in zip(best, original)) / len(original)
    return result, achieved


class Evaluator:
    def __init__(self, *, torch: Any, feature_module: Any, gpu_module: Any, prior: Any,
                 allowed: Sequence[int], device: Any, alpha: float) -> None:
        self.torch = torch
        self.features = feature_module
        self.gpu = gpu_module
        self.prior = prior
        self.allowed = tuple(allowed)
        self.device = device
        self.alpha = alpha
        self.fit_count = 0
        self.cache: dict[tuple[str, str, str, str], tuple[np.ndarray, dict[str, Any]]] = {}

    def predict(self, train_rows: Sequence[Mapping[str, Any]], dev_rows: Sequence[Mapping[str, Any]],
                arm: str, ast: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        ast_identity = "ast_independent" if arm in {"onehot_ridge", "flat_no_ast"} else value_hash("ast", ast)
        train_identity = value_hash("train_ids", [x["variant_id"] for x in train_rows])
        dev_identity = value_hash("dev_ids", [x["variant_id"] for x in dev_rows])
        cache_key = (arm, ast_identity, train_identity, dev_identity)
        if cache_key in self.cache:
            cached_predictions, cached_report = self.cache[cache_key]
            report = dict(cached_report)
            report["cache_hit"] = True
            return cached_predictions.copy(), report
        runtime = ast_runtime(ast)
        train_x = self.features.build_features(train_rows, arm=arm, allowed_positions=self.allowed,
                                               ast=runtime, prior=self.prior)
        dev_x = self.features.build_features(dev_rows, arm=arm, allowed_positions=self.allowed,
                                             ast=runtime, prior=self.prior)
        targets = np.asarray([x["target"] for x in train_rows], dtype=np.float32)
        pred, report, _weights = self.gpu._fit_gpu_ridge(
            self.torch, train_x, targets, dev_x, device=self.device, alpha=self.alpha
        )
        self.fit_count += 1
        del train_x, dev_x
        report["cache_hit"] = False
        self.cache[cache_key] = (pred.copy(), dict(report))
        return pred, report
