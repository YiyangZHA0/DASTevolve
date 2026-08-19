

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from typing import Any, Dict, Mapping, MutableMapping, Sequence

from astevolve.domain.dual_ast import ExecutableDualAST

from .memory_lifecycle import (
    MemoryExecutionContext,
    build_parent_effective_ast_lineage,
    build_parent_sequence_lineage,
)


class ParentLineageError(ValueError):
    pass


def _artifact_mapping(value: Any, label: str) -> Mapping[str, Any] | None:
    if value is None or value == "":
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ParentLineageError(f"{label} is not UTF-8 JSON") from error
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ParentLineageError(f"{label} is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise ParentLineageError(f"{label} must be a mapping")
    return value


def bind_selected_parent_lineages(
    context: MemoryExecutionContext,
    *,
    parent_program_id: str,
    parent_artifacts: Mapping[str, Any] | None,
) -> MemoryExecutionContext:


    if not isinstance(context, MemoryExecutionContext):
        raise TypeError("context must be a MemoryExecutionContext")
    artifacts = parent_artifacts if isinstance(parent_artifacts, Mapping) else {}
    sequences = _artifact_mapping(
        artifacts.get("best_seqs"), "selected parent best_seqs"
    )
    sequence_lineage_json = build_parent_sequence_lineage(parent_program_id, sequences)

    raw_ast = _artifact_mapping(
        artifacts.get("executable_dual_ast"),
        "selected parent executable_dual_ast",
    )
    raw_report = _artifact_mapping(
        artifacts.get("ast_revision_report"),
        "selected parent ast_revision_report",
    )
    node_metadata = (
        raw_report.get("node_metadata") if isinstance(raw_report, Mapping) else None
    )
    expected_ast_hash = (
        str(raw_report.get("effective_ast_hash") or "")
        if isinstance(raw_report, Mapping)
        else ""
    )
    ast_lineage_json = build_parent_effective_ast_lineage(
        parent_program_id,
        raw_ast,
        node_metadata,
        expected_effective_ast_hash=expected_ast_hash,
    )
    if not sequence_lineage_json and not ast_lineage_json:
        return context
    return replace(
        context,
        parent_sequence_lineage_json=sequence_lineage_json,
        parent_effective_ast_lineage_json=ast_lineage_json,
    )


def resolve_trusted_parent_inputs(
    execution_scope: MemoryExecutionContext | None,
    causal_envelope: Mapping[str, Any],
    strategy: Mapping[str, Any],
) -> tuple[Dict[str, Any], Any, Any, Any]:


    sequence_lineage = (
        execution_scope.parent_sequence_lineage()
        if execution_scope is not None
        else None
    )
    ast_lineage = (
        execution_scope.parent_effective_ast_lineage()
        if execution_scope is not None
        else None
    )
    causal_parent_id = str(causal_envelope.get("parent_program_id") or "")
    lineages = (
        ("sequence", sequence_lineage),
        ("effective AST", ast_lineage),
    )
    lineage_parent_ids = set()
    for label, lineage in lineages:
        if not isinstance(lineage, Mapping):
            continue
        lineage_parent_id = str(lineage.get("parent_program_id") or "")
        lineage_parent_ids.add(lineage_parent_id)
        if causal_parent_id and causal_parent_id != lineage_parent_id:
            raise ParentLineageError(
                f"trusted parent {label} lineage does not match the causal parent"
            )
    if len(lineage_parent_ids) > 1:
        raise ParentLineageError(
            "trusted parent sequence and effective AST lineages disagree"
        )

    resolved = deepcopy(dict(strategy))
    if isinstance(sequence_lineage, Mapping):
        resolved["resume_template_seqs"] = deepcopy(sequence_lineage["sequences"])
    return (
        resolved,
        resolved.get("resume_template_seqs"),
        sequence_lineage,
        ast_lineage,
    )


def _specs_by_id(values: Any) -> Dict[str, Dict[str, Any]]:
    return {str(value.node_id): value.to_dict() for value in values}


def _case_owned_structural_ids(
    state: Mapping[str, Any], base: ExecutableDualAST
) -> set[str]:
    base_ids = {node.node_id for node in base.structural_nodes}
    frozen_ids = {
        node.node_id for node in base.structural_nodes if node.kind == "frozen"
    }
    global_policy = state.get("global_ast_evolution_policy")
    if isinstance(global_policy, Mapping):
        raw_ids = global_policy.get("immutable_structural_node_ids")
        declared = (
            {str(value) for value in raw_ids}
            if isinstance(raw_ids, (list, tuple))
            else set()
        )
        return frozen_ids | declared
    legacy_policy = state.get("ast_evolution_policy")
    if isinstance(legacy_policy, Mapping):
        envelopes = legacy_policy.get("structural_node_envelopes")
        editable_ids = (
            {str(value) for value in envelopes}
            if isinstance(envelopes, Mapping)
            else set()
        )
        return frozen_ids | (base_ids - editable_ids)
    return base_ids


def install_parent_effective_ast(
    state: Mapping[str, Any],
    lineage: Mapping[str, Any] | None,
) -> Dict[str, Any]:


    updated = deepcopy(dict(state))
    if not isinstance(lineage, Mapping):
        return updated
    raw_base = updated.get("executable_dual_ast")
    if not isinstance(raw_base, Mapping):
        raise ParentLineageError(
            "parent effective AST lineage requires a case executable_dual_ast"
        )
    try:
        base = ExecutableDualAST.from_mapping(raw_base)
        inherited = ExecutableDualAST.from_mapping(lineage["executable_dual_ast"])
    except (KeyError, TypeError, ValueError) as error:
        raise ParentLineageError(
            f"parent effective AST lineage is invalid for this case: {error}"
        ) from error
    if inherited.ast_id != base.ast_id:
        raise ParentLineageError(
            "parent effective AST lineage belongs to a different case AST"
        )
    if inherited.revision < base.revision:
        raise ParentLineageError(
            "parent effective AST lineage predates the case AST revision"
        )
    if _specs_by_id(inherited.functional_nodes) != _specs_by_id(base.functional_nodes):
        raise ParentLineageError(
            "parent effective AST lineage changes case-owned functional nodes"
        )

    base_structural = _specs_by_id(base.structural_nodes)
    inherited_structural = _specs_by_id(inherited.structural_nodes)
    inherited_frozen = {
        node.node_id for node in inherited.structural_nodes if node.kind == "frozen"
    }
    base_frozen = {
        node.node_id for node in base.structural_nodes if node.kind == "frozen"
    }
    if inherited_frozen != base_frozen:
        raise ParentLineageError(
            "parent effective AST lineage changes case-owned frozen nodes"
        )
    for node_id in sorted(_case_owned_structural_ids(updated, base)):
        if inherited_structural.get(node_id) != base_structural.get(node_id):
            raise ParentLineageError(
                f"parent effective AST lineage changes immutable node {node_id!r}"
            )

    metadata = lineage.get("node_metadata")
    if not isinstance(metadata, Mapping):
        raise ParentLineageError(
            "parent effective AST lineage node_metadata must be a mapping"
        )
    inherited_editable = {
        node.node_id for node in inherited.structural_nodes if node.kind == "editable"
    }
    unknown_metadata = sorted(set(str(key) for key in metadata) - inherited_editable)
    if unknown_metadata:
        raise ParentLineageError(
            "parent effective AST lineage metadata references unknown nodes: "
            + ", ".join(unknown_metadata)
        )

    updated["executable_dual_ast"] = inherited.to_dict()
    updated["_global_ast_node_metadata"] = deepcopy(dict(metadata))
    updated["_parent_effective_ast_lineage"] = deepcopy(dict(lineage))
    return updated


def bind_ast_revision_report(
    report: Mapping[str, Any],
    lineage: Mapping[str, Any] | None,
) -> Dict[str, Any]:


    bound = deepcopy(dict(report))
    if not isinstance(lineage, Mapping):
        bound["base_ast_source"] = "case_static"
        return bound
    expected_hash = str(lineage.get("effective_ast_hash") or "")
    if str(bound.get("base_ast_hash") or "") != expected_hash:
        raise ParentLineageError(
            "AST revision report base does not match selected-parent lineage"
        )
    bound.update(
        {
            "base_ast_source": "selected_parent",
            "parent_program_id": str(lineage.get("parent_program_id") or ""),
            "parent_effective_ast_lineage_hash": str(lineage.get("lineage_hash") or ""),
        }
    )
    return bound


def apply_resume_template_sequences(
    templates: Mapping[str, str],
    resume_template_seqs: Any,
    state: MutableMapping[str, Any],
) -> Dict[str, str]:


    updated = dict(templates)
    if not isinstance(resume_template_seqs, Mapping):
        return updated
    for chain_id, sequence in resume_template_seqs.items():
        cid = str(chain_id)
        if cid not in updated:
            raise ValueError(f"resume_template_seqs contains unknown chain {cid!r}")
        normalized = str(sequence).strip().replace(" ", "").replace("\n", "")
        if len(normalized) != len(updated[cid]):
            raise ValueError(
                f"resume_template_seqs length mismatch for chain {cid}: "
                f"{len(normalized)} vs {len(updated[cid])}"
            )
        updated[cid] = normalized
    state["_resume_template_seqs_applied"] = sorted(
        str(key) for key in resume_template_seqs
    )
    return updated


def mutation_owners_from_dual_ast(raw_ast: Any) -> Dict[str, Dict[int, tuple[str, ...]]]:


    if not isinstance(raw_ast, Mapping):
        return {}
    ast = ExecutableDualAST.from_mapping(raw_ast)
    owners: Dict[str, Dict[int, list[str]]] = {}
    for node in ast.structural_nodes:
        if node.kind != "editable":
            continue
        chain = node.selector.chain_id
        for start, end in node.selector.spans:
            for position in range(start, end):
                owners.setdefault(chain, {}).setdefault(position, []).append(node.node_id)
    return {chain: {position: tuple(sorted(values)) for position, values in positions.items()} for chain, positions in owners.items()}


def apply_retired_position_policy(
    templates: Mapping[str, str],
    *,
    case_reference_templates: Mapping[str, str],
    masks: Mapping[str, Sequence[bool]],
    raw_policy: Mapping[str, Any] | None,
    current_node_owners: Mapping[str, Mapping[int, Sequence[str]]] | None = None,
    parent_node_owners: Mapping[str, Mapping[int, Sequence[str]]] | None = None,
) -> tuple[Dict[str, str], Dict[str, Any] | None]:


    updated = dict(templates)
    if not isinstance(raw_policy, Mapping):
        return updated, None
    policy = raw_policy.get("retired_position_policy")
    if policy is None:
        return updated, None
    if policy != "revert_to_case_reference":
        raise ParentLineageError(
            "unsupported case_owned_residue_policy.retired_position_policy "
            f"{policy!r}"
        )

    reversions: list[Dict[str, Any]] = []
    retained: list[Dict[str, Any]] = []
    reassigned: list[Dict[str, Any]] = []
    for chain_id, current in sorted(updated.items()):
        reference = case_reference_templates.get(chain_id)
        mask = masks.get(chain_id)
        if reference is None or mask is None:
            raise ParentLineageError(
                f"retired-position policy is missing reference/mask for chain {chain_id!r}"
            )
        if len(current) != len(reference) or len(current) != len(mask):
            raise ParentLineageError(
                f"retired-position policy length mismatch for chain {chain_id!r}"
            )
        residues = list(current)
        for position, (from_residue, to_residue, active) in enumerate(
            zip(current, reference, mask)
        ):
            if from_residue == to_residue:
                continue
            current_owners = sorted(str(value) for value in ((current_node_owners or {}).get(chain_id, {}).get(position, ()) or ()))
            parent_owners = sorted(str(value) for value in ((parent_node_owners or {}).get(chain_id, {}).get(position, ()) or ()))
            if bool(active):
                record = {"chain_id": chain_id, "position": position, "residue": from_residue, "parent_node_ids": parent_owners, "current_node_ids": current_owners}
                if parent_owners and current_owners and not set(parent_owners).intersection(current_owners):
                    record["reason"] = "mutation_transferred_to_covering_current_node"
                    reassigned.append(record)
                else:
                    record["reason"] = "mutation_retained_in_active_selector"
                    retained.append(record)
                continue
            residues[position] = to_residue
            reversions.append(
                {
                    "chain_id": chain_id,
                    "position": position,
                    "from": from_residue,
                    "to": to_residue,
                    "parent_node_ids": parent_owners,
                    "current_node_ids": current_owners,
                    "reason": "inherited_parent_residue_retired_by_current_ast",
                }
            )
        updated[chain_id] = "".join(residues)

    audit = {
        "schema_version": "astevolve.mutation_scope_reconciliation.v1",
        "policy": policy,
        "indexing": "zero_based",
        "applied": bool(reversions),
        "retained_count": len(retained),
        "reassigned_count": len(reassigned),
        "reverted_count": len(reversions),
        "retained_mutations": retained,
        "reassigned_mutations": reassigned,
        "reverted_mutations": reversions,
        "reversions": reversions,
    }
    return updated, audit


__all__ = [
    "ParentLineageError",
    "apply_retired_position_policy",
    "apply_resume_template_sequences",
    "mutation_owners_from_dual_ast",
    "bind_ast_revision_report",
    "bind_selected_parent_lineages",
    "install_parent_effective_ast",
    "resolve_trusted_parent_inputs",
]
