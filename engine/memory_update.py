

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import hashlib
import math
import os
import tempfile
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from astevolve.search.structure_failure_memory import (
    update_structure_failure_memory,
)

from .memory_lifecycle import (
    MemoryExecutionContext,
    MemorySnapshot,
    canonical_memory_content,
    capture_memory_snapshot,
    current_memory_execution_context,
)


AA_CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
MEMORY_UPDATE_PROPOSAL_VERSION = "astevolve.memory_update_proposal.v1"
MEMORY_COMMIT_VERSION = "astevolve.memory_commit.v1"


class MemoryProposalError(ValueError):
    pass


class StaleMemorySnapshotError(RuntimeError):
    pass


def _safe_import_yaml():
    try:
        import yaml

        return yaml
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _round(value: Any, ndigits: int = 4) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return round(out, ndigits)


def _aa_classes(aa: str) -> List[str]:
    aa = str(aa).upper()
    classes: List[str] = []
    if aa in "YWHF":
        classes.append("aromatic")
    if aa in "STNQY":
        classes.append("polar_uncharged")
    if aa in "RKHDE":
        classes.append("contextual_charge")
    if aa in "GSPNDT":
        classes.append("turn_loop")
    if aa in "GSA":
        classes.append("flexible_small")
    return classes


def _node_key(block: Dict[str, Any], node: str) -> str:
    for key in (node, node.lower(), node.casefold()):
        if key in block:
            return str(key)
    return str(node)


def _top_counter_dict(counter: Counter, limit: int) -> Dict[str, int]:
    return {str(k): int(v) for k, v in counter.most_common(max(1, int(limit))) if int(v) > 0}


def _counter_from_dict(data: Any) -> Counter:
    counter: Counter = Counter()
    if isinstance(data, dict):
        for key, value in data.items():
            count = _safe_int(value, 0)
            if count > 0:
                counter[str(key)] += count
    return counter


def _bounded_mutations(mutations: Any, limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(mutations, list):
        return out
    for item in mutations:
        if not isinstance(item, dict):
            continue
        aa_to = str(item.get("to", "")).upper()
        if aa_to not in AA_CANONICAL:
            continue
        compact = {
            "node": str(item.get("node", "")),
            "chain_id": str(item.get("chain_id", "")),
            "position": _safe_int(item.get("position"), 0),
            "from": str(item.get("from", ""))[:1],
            "to": aa_to,
        }
        reward = _round(item.get("reward"), 6)
        if reward is not None:
            compact["reward"] = reward
        fast_loss = _round(item.get("fast_loss"), 6)
        if fast_loss is not None:
            compact["fast_loss"] = fast_loss
        out.append(compact)
        if len(out) >= limit:
            break
    return out


def _seq_hash(seqs: Any) -> Optional[str]:
    if not isinstance(seqs, dict) or not seqs:
        return None
    payload = "|".join(f"{k}:{seqs[k]}" for k in sorted(seqs))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _blend(old: Any, new: float, alpha: float = 0.25) -> float:
    old_val = _safe_float(old, new)
    return float((1.0 - alpha) * old_val + alpha * new)


def _apply_update_to_document(
    base_memory: Mapping[str, Any],
    search_output: Dict[str, Any],
    *,
    max_recent_runs: int = 10,
    max_residues_per_node: int = 8,
    timestamp: Optional[str] = None,
    scope: Optional[MemoryExecutionContext] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:


    memory = deepcopy(dict(base_memory))
    timestamp = str(timestamp or (scope.logical_time if scope else "") or _now_iso())
    search_artifacts = search_output.get("search_artifacts", {}) or {}
    round_summary = search_artifacts.get("round_summary", {}) or {}
    suggestion = round_summary.get("internal_memory_update_suggestion", {}) or {}
    node_stats = round_summary.get("node_level_statistics", {}) or {}
    effective_mutations = _bounded_mutations(
        suggestion.get("effective_mutations", []),
        max(4, max_residues_per_node * 2),
    )

    metadata = memory.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["last_auto_update"] = timestamp

    adaptive = memory.setdefault("adaptive_memory", {})
    if not isinstance(adaptive, dict):
        adaptive = {}
        memory["adaptive_memory"] = adaptive
    structure_failure_update = update_structure_failure_memory(
        adaptive,
        search_output.get("structure_finalist_feedback")
        or search_artifacts.get("structure_finalist_feedback"),
        timestamp=timestamp,
    )
    motif_memory = adaptive.setdefault("motif_memory", {})
    if not isinstance(motif_memory, dict):
        motif_memory = {}
        adaptive["motif_memory"] = motif_memory
    node_level = adaptive.setdefault("node_level_statistics", {})
    if not isinstance(node_level, dict):
        node_level = {}
        adaptive["node_level_statistics"] = node_level

    mutations_by_node: Dict[str, List[Dict[str, Any]]] = {}
    for change in effective_mutations:
        node = str(change.get("node", ""))
        if node:
            mutations_by_node.setdefault(node, []).append(change)

    touched_nodes: List[str] = []
    for node, stats in node_stats.items():
        if not isinstance(stats, dict):
            continue
        node_name = str(node)
        touched_nodes.append(node_name)

        stat_key = _node_key(node_level, node_name)
        stat_block = node_level.setdefault(stat_key, {})
        if not isinstance(stat_block, dict):
            stat_block = {}
            node_level[stat_key] = stat_block

        evaluated = max(1, _safe_int(stats.get("evaluated"), 1))
        improved = max(0, _safe_int(stats.get("improved"), 0))
        success_rate = _safe_float(stats.get("success_rate"), improved / evaluated)
        mean_reward = _safe_float(stats.get("mean_reward"), 0.0)
        priority_delta = max(-0.25, min(0.35, success_rate * 0.25 + mean_reward * 0.5))
        priority_multiplier = max(0.5, min(1.75, 1.0 + priority_delta))

        stat_block["evaluated_total"] = _safe_int(stat_block.get("evaluated_total"), 0) + evaluated
        stat_block["improved_total"] = _safe_int(stat_block.get("improved_total"), 0) + improved
        stat_block["last_success_rate"] = _round(success_rate, 4)
        stat_block["last_mean_reward"] = _round(mean_reward, 6)
        stat_block["last_best_fast_loss"] = _round(stats.get("best_fast_loss"), 6)
        stat_block["priority_multiplier"] = _round(priority_multiplier, 4)
        stat_block["last_update"] = timestamp

        node_changes = mutations_by_node.get(node_name, [])
        residue_counter = _counter_from_dict(stat_block.get("top_residue_frequency", {}))
        class_counter = _counter_from_dict(stat_block.get("top_class_frequency", {}))
        for change in node_changes:
            aa_to = str(change.get("to", "")).upper()
            residue_counter[aa_to] += 1
            for class_name in _aa_classes(aa_to):
                class_counter[class_name] += 1
        stat_block["top_residue_frequency"] = _top_counter_dict(residue_counter, max_residues_per_node)
        stat_block["top_class_frequency"] = _top_counter_dict(class_counter, 5)

        if node_changes:
            mean_mut_count = len(node_changes)
            stat_block["mean_mutation_count"] = _round(
                _blend(stat_block.get("mean_mutation_count"), mean_mut_count, alpha=0.25),
                4,
            )

        node_plddt = (search_output.get("node_plddt", {}) or {}).get(node_name, {})
        if isinstance(node_plddt, dict):
            for key in ("plddt_mean", "plddt_min", "plddt_max"):
                if key in node_plddt:
                    stat_block[f"last_{key}"] = _round(node_plddt.get(key), 4)

        motif_key = _node_key(motif_memory, node_name)
        motif = motif_memory.setdefault(motif_key, {})
        if not isinstance(motif, dict):
            motif = {}
            motif_memory[motif_key] = motif
        motif.setdefault("enriched_motifs", [])
        motif.setdefault("depleted_patterns", [])
        motif["enriched_residues"] = list(stat_block.get("top_residue_frequency", {}).keys())[
            :max_residues_per_node
        ]
        motif["favored_classes"] = list(stat_block.get("top_class_frequency", {}).keys())[:5]
        motif["support_count"] = _safe_int(motif.get("support_count"), 0) + improved
        old_conf = _safe_float(motif.get("confidence"), 0.0)
        conf_delta = max(0.0, min(0.12, 0.035 * improved + 0.04 * success_rate + max(0.0, mean_reward) * 0.15))
        motif["confidence"] = _round(min(0.85, old_conf * 0.92 + conf_delta), 4)
        motif["last_update"] = timestamp

    run_id = (
        str(scope.scope_id)
        if scope is not None and scope.scope_id
        else f"run_{_safe_int(round_summary.get('created_at_unix'), int(datetime.now().timestamp()))}"
    )
    artifact_paths = search_artifacts.get("artifact_paths", {}) or {}
    round_path = artifact_paths.get("round_summary")
    if round_path and not (scope is not None and scope.scope_id):
        run_id = Path(str(round_path)).parent.name

    compact_run = {
        "run_id": run_id,
        "updated_at": timestamp,
        "search_method": search_output.get("search_method", round_summary.get("search_method", "")),
        "best_node_id": search_artifacts.get("best_node_id", round_summary.get("best_node_id")),
        "best_path": search_artifacts.get("best_path", round_summary.get("best_path", [])),
        "seq_hash": _seq_hash(search_output.get("seqs")),
        "fast_loss": _round(search_output.get("fast_loss"), 6),
        "design_energy": _round(
            search_output.get(
                "final_energy",
                search_output.get("design_energy", search_output.get("chai_combined_loss")),
            ),
            6,
        ),
        "energy_direction": str(search_output.get("energy_direction") or "minimize"),
        "constraint_penalty": _round(search_output.get("constraint_penalty"), 6),
        "progen_loglik_avg": _round(search_output.get("progen_loglik_avg"), 6),
        "chai_plddt": _round(search_output.get("chai_plddt"), 4),
        "effective_mutations": effective_mutations[:max_residues_per_node],
    }

    candidate_stats = memory.setdefault("candidate_statistics", {})
    if not isinstance(candidate_stats, dict):
        candidate_stats = {}
        memory["candidate_statistics"] = candidate_stats
    recent = candidate_stats.setdefault("recent_inner_loop_runs", [])
    if not isinstance(recent, list):
        recent = []
    recent = [item for item in recent if not isinstance(item, dict) or item.get("run_id") != run_id]
    recent.insert(0, compact_run)
    recent = recent[: max(1, int(max_recent_runs))]
    candidate_stats["recent_inner_loop_runs"] = recent
    candidate_stats["max_recent_inner_loop_runs"] = max(1, int(max_recent_runs))
    candidate_stats["max_residues_per_node"] = max(1, int(max_residues_per_node))

    best_seen = candidate_stats.get("best_seen")
    current_loss = _safe_float(
        search_output.get(
            "final_energy",
            search_output.get(
                "design_energy",
                search_output.get("chai_combined_loss", search_output.get("fast_loss")),
            ),
        ),
        1e9,
    )
    previous_loss = _safe_float(
        (
            best_seen.get("design_energy", best_seen.get("combined_loss"))
            if isinstance(best_seen, dict)
            else None
        ),
        1e9,
    )
    if current_loss <= previous_loss:
        candidate_stats["best_seen"] = {
            "run_id": run_id,
            "updated_at": timestamp,
            "seq_hash": compact_run["seq_hash"],
            "combined_loss": _round(current_loss, 6),
            "design_energy": _round(current_loss, 6),
            "energy_direction": "minimize",
            "fast_loss": compact_run["fast_loss"],
            "chai_plddt": compact_run["chai_plddt"],
        }

    run_state = memory.setdefault("run_state", {})
    if not isinstance(run_state, dict):
        run_state = {}
        memory["run_state"] = run_state
    run_state["last_inner_loop_run"] = {
        "run_id": run_id,
        "updated_at": timestamp,
        "touched_nodes": touched_nodes,
        "num_effective_mutations": len(effective_mutations),
        "memory_policy": "bounded_summary_only",
    }

    result = {
        "run_id": run_id,
        "touched_nodes": touched_nodes,
        "effective_mutations": len(effective_mutations),
        "recent_run_count": len(recent),
        "structure_failure_memory": structure_failure_update,
    }
    return memory, result


def _content_hash(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_memory_content(document).encode("utf-8")).hexdigest()


def _scope_provenance(
    snapshot: MemorySnapshot,
    search_output: Mapping[str, Any],
    scope: Optional[MemoryExecutionContext],
    target_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    return {
        "generation_id": scope.generation_id if scope else "",
        "proposal_id": scope.proposal_id if scope else "",
        "trial_id": scope.trial_id if scope else "",
        "scope_id": scope.scope_id if scope else "",
        "sequence_hash": _seq_hash(
            search_output.get("seqs") or search_output.get("best_seqs")
        ),
        "snapshot_source_path": snapshot.source_path,
        "target_path": (
            str(target_path)
            if target_path is not None
            else scope.target_path
            if scope is not None and scope.target_path
            else snapshot.source_path
        ),
    }


def _proposal_commit_id(
    snapshot: MemorySnapshot,
    document: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    identity = {
        "base_content_hash": snapshot.content_hash,
        "candidate_document_hash": _content_hash(document),
        "provenance": dict(provenance),
    }
    payload = canonical_memory_content(identity).encode("utf-8")
    return "mem_" + hashlib.sha256(payload).hexdigest()[:24]


def propose_internal_memory_update(
    snapshot: MemorySnapshot,
    search_output: Dict[str, Any],
    *,
    max_recent_runs: int = 10,
    max_residues_per_node: int = 8,
    scope: Optional[MemoryExecutionContext] = None,
    logical_time: Optional[str] = None,
    target_path: Optional[str | Path] = None,
) -> Dict[str, Any]:


    if not isinstance(snapshot, MemorySnapshot):
        raise TypeError("snapshot must be a MemorySnapshot")
    effective_scope = scope if scope is not None else current_memory_execution_context()
    timestamp = str(
        logical_time
        or (effective_scope.logical_time if effective_scope else "")
        or _now_iso()
    )
    document, summary = _apply_update_to_document(
        snapshot.materialize(),
        search_output,
        max_recent_runs=max_recent_runs,
        max_residues_per_node=max_residues_per_node,
        timestamp=timestamp,
        scope=effective_scope,
    )
    provenance = _scope_provenance(
        snapshot,
        search_output,
        effective_scope,
        target_path=target_path,
    )
    commit_id = _proposal_commit_id(snapshot, document, provenance)

    run_state = document.setdefault("run_state", {})
    if not isinstance(run_state, dict):
        run_state = {}
        document["run_state"] = run_state
    commit_ids = run_state.setdefault("memory_commit_ids", [])
    if not isinstance(commit_ids, list):
        commit_ids = []
    commit_ids = [str(item) for item in commit_ids if item]
    if commit_id not in commit_ids:
        commit_ids.append(commit_id)
    run_state["memory_commit_ids"] = commit_ids[-100:]
    run_state["last_memory_commit"] = {
        "commit_id": commit_id,
        "base_content_hash": snapshot.content_hash,
        "generation_id": provenance["generation_id"],
        "proposal_id": provenance["proposal_id"],
        "trial_id": provenance["trial_id"],
        "scope_id": provenance["scope_id"],
        "sequence_hash": provenance["sequence_hash"],
        "logical_time": timestamp,
    }

    proposed_hash = _content_hash(document)
    proposal_core = {
        "schema_version": MEMORY_UPDATE_PROPOSAL_VERSION,
        "commit_id": commit_id,
        "base_content_hash": snapshot.content_hash,
        "base_raw_hash": snapshot.raw_hash,
        "base_source_exists": snapshot.source_exists,
        "proposed_content_hash": proposed_hash,
        "target_path": provenance["target_path"],
        "logical_time": timestamp,
        "provenance": provenance,
        "summary": summary,
        "document": document,
    }
    proposal_hash_payload = canonical_memory_content(proposal_core).encode("utf-8")
    proposal = dict(proposal_core)
    proposal["proposal_hash"] = hashlib.sha256(proposal_hash_payload).hexdigest()
    return proposal


def _commit_ids(document: Mapping[str, Any]) -> List[str]:
    run_state = document.get("run_state", {})
    if not isinstance(run_state, Mapping):
        return []
    values = run_state.get("memory_commit_ids", [])
    if not isinstance(values, list):
        return []
    return [str(item) for item in values if item]


def _validate_proposal(proposal: Mapping[str, Any]) -> None:
    if proposal.get("schema_version") != MEMORY_UPDATE_PROPOSAL_VERSION:
        raise MemoryProposalError(
            f"unsupported memory proposal version: {proposal.get('schema_version')!r}"
        )
    document = proposal.get("document")
    if not isinstance(document, Mapping):
        raise MemoryProposalError("memory proposal document must be a mapping")
    actual_hash = _content_hash(document)
    if actual_hash != proposal.get("proposed_content_hash"):
        raise MemoryProposalError(
            "memory proposal document hash does not match proposed_content_hash"
        )
    if not str(proposal.get("commit_id") or ""):
        raise MemoryProposalError("memory proposal is missing commit_id")
    if str(proposal["commit_id"]) not in _commit_ids(document):
        raise MemoryProposalError("memory proposal commit_id is absent from its document")
    proposal_hash = str(proposal.get("proposal_hash") or "")
    core = {key: value for key, value in proposal.items() if key != "proposal_hash"}
    actual_proposal_hash = hashlib.sha256(
        canonical_memory_content(core).encode("utf-8")
    ).hexdigest()
    if proposal_hash != actual_proposal_hash:
        raise MemoryProposalError("memory proposal hash does not match its payload")


@contextmanager
def _exclusive_memory_lock(path: Path) -> Iterator[None]:


    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _yaml_bytes(document: Mapping[str, Any]) -> bytes:
    yaml = _safe_import_yaml()
    if yaml is None:
        raise MemoryProposalError("PyYAML is not available")
    text = yaml.safe_dump(
        dict(document),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    return text.encode("utf-8")


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:


            pass
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def commit_internal_memory_update(
    memory_path: str | Path,
    proposal: Mapping[str, Any],
) -> Dict[str, Any]:


    _validate_proposal(proposal)
    path = Path(memory_path)
    proposal_target = str(proposal.get("target_path") or "")
    if proposal_target and Path(proposal_target).resolve() != path.resolve():
        raise MemoryProposalError(
            f"memory proposal targets {proposal_target!r}, not {str(path)!r}"
        )
    commit_id = str(proposal["commit_id"])
    with _exclusive_memory_lock(path):
        current = capture_memory_snapshot(path)
        current_document = current.materialize()
        if commit_id in _commit_ids(current_document):
            return {
                "schema_version": MEMORY_COMMIT_VERSION,
                "commit_id": commit_id,
                "committed": False,
                "idempotent": True,
                "commit_count": 0,
                "reason": "already_committed",
                "path": str(path),
                "input_content_hash": str(proposal["base_content_hash"]),
                "input_raw_hash": str(proposal["base_raw_hash"]),
                "output_content_hash": current.content_hash,
                "output_raw_hash": current.raw_hash,
            }

        expected_content = str(proposal.get("base_content_hash") or "")
        expected_raw = str(proposal.get("base_raw_hash") or "")
        expected_exists = bool(proposal.get("base_source_exists", True))
        if (
            current.content_hash != expected_content
            or current.raw_hash != expected_raw
            or current.source_exists != expected_exists
        ):
            raise StaleMemorySnapshotError(
                "memory commit rejected: target no longer matches proposal base "
                f"(expected content={expected_content}, raw={expected_raw}; "
                f"found content={current.content_hash}, raw={current.raw_hash})"
            )

        document = deepcopy(dict(proposal["document"]))
        payload = _yaml_bytes(document)
        _atomic_replace(path, payload)
        output = capture_memory_snapshot(path)
        if output.content_hash != proposal["proposed_content_hash"]:
            raise MemoryProposalError(
                "atomic memory write did not preserve proposed canonical content"
            )
        return {
            "schema_version": MEMORY_COMMIT_VERSION,
            "commit_id": commit_id,
            "committed": True,
            "idempotent": False,
            "commit_count": 1,
            "reason": "committed",
            "path": str(path),
            "input_content_hash": current.content_hash,
            "input_raw_hash": current.raw_hash,
            "output_content_hash": output.content_hash,
            "output_raw_hash": output.raw_hash,
        }


def update_internal_memory(
    memory_path: Path,
    search_output: Dict[str, Any],
    *,
    max_recent_runs: int = 10,
    max_residues_per_node: int = 8,
    dry_run: bool = False,
    snapshot: Optional[MemorySnapshot] = None,
    commit_mode: Optional[str] = None,
    scope: Optional[MemoryExecutionContext] = None,
) -> Dict[str, Any]:


    path = Path(memory_path)
    base = snapshot if snapshot is not None else capture_memory_snapshot(path)
    effective_scope = scope if scope is not None else current_memory_execution_context()
    mode = str(
        commit_mode
        or (effective_scope.commit_mode if effective_scope is not None else "standalone")
    ).strip().lower()
    if mode not in {"deferred", "standalone"}:
        raise ValueError(f"unsupported memory commit mode: {mode!r}")
    proposal = propose_internal_memory_update(
        base,
        search_output,
        max_recent_runs=max_recent_runs,
        max_residues_per_node=max_residues_per_node,
        scope=effective_scope,
        target_path=path,
    )
    summary = dict(proposal.get("summary", {}))
    if dry_run or mode == "deferred":
        return {
            **summary,
            "updated": False,
            "deferred": True,
            "dry_run": bool(dry_run),
            "reason": "deferred_candidate_update",
            "path": str(path),
            "memory": deepcopy(proposal["document"]),
            "proposal": proposal,
            "input_content_hash": base.content_hash,
            "input_raw_hash": base.raw_hash,
            "proposed_content_hash": proposal["proposed_content_hash"],
        }
    committed = commit_internal_memory_update(path, proposal)
    return {
        **summary,
        "updated": bool(committed["committed"]),
        "deferred": False,
        "dry_run": False,
        "reason": committed["reason"],
        "path": str(path),
        "proposal": proposal,
        "commit": committed,
        "input_content_hash": base.content_hash,
        "input_raw_hash": base.raw_hash,
        "proposed_content_hash": proposal["proposed_content_hash"],
        "output_content_hash": committed["output_content_hash"],
        "output_raw_hash": committed["output_raw_hash"],
    }
