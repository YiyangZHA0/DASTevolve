

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from astevolve.domain.compiled import (
    CompiledCandidateSlot,
    CompiledDesignAction,
    CompiledMutationModule,
    CompiledMutationSet,
    CompiledPortfolioOptimizationRequest,
    CompiledPositionDistribution,
    CompiledResidueChange,
    CompiledSequenceSeed,
    MutationOwnershipEntry,
    MutationOwnershipLedger,
)
from engine.causal_flow import EffectiveSearchContract, canonical_json
from engine.experiment_identity import SequenceBundleIdentity
from engine.node_compiler import ExecutableNodePlan


class DesignActionCompileError(ValueError):


    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)


def _fail(code: str, detail: str = "") -> None:
    raise DesignActionCompileError(code, detail)


def _digest(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("utf-8") + b"\0" + canonical_json(value).encode("utf-8")
    ).hexdigest()


def _canonical_sequences(
    value: Mapping[str, str], *, label: str
) -> Dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        _fail("sequence_bundle_invalid", label)
    sequences: Dict[str, str] = {}
    canonical = set("ACDEFGHIKLMNPQRSTVWY")
    for raw_chain, raw_sequence in sorted(value.items()):
        chain = str(raw_chain or "").strip()
        sequence = str(raw_sequence or "").strip().upper()
        if not chain or not sequence:
            _fail("sequence_bundle_invalid", label)
        invalid = sorted(set(sequence) - canonical)
        if invalid:
            _fail(
                "sequence_residue_invalid",
                f"{label}.{chain}:{','.join(invalid)}",
            )
        sequences[chain] = sequence
    return sequences


def _require_same_shape(
    immutable: Mapping[str, str], parent: Mapping[str, str]
) -> None:
    if set(immutable) != set(parent):
        _fail("parent_chain_set_mismatch")
    for chain in immutable:
        if len(immutable[chain]) != len(parent[chain]):
            _fail(
                "parent_sequence_length_mismatch",
                f"{chain}:{len(parent[chain])}/{len(immutable[chain])}",
            )


def _parse_effective_contract(
    value: EffectiveSearchContract | Mapping[str, Any] | str,
) -> tuple[str, Mapping[str, Any] | None]:
    if isinstance(value, EffectiveSearchContract):
        parsed = EffectiveSearchContract.from_mapping(value.to_dict())
        return parsed.contract_hash, parsed.semantic
    if isinstance(value, Mapping):
        try:
            parsed = EffectiveSearchContract.from_mapping(value)
            return parsed.contract_hash, parsed.semantic
        except (TypeError, ValueError) as error:
            _fail("parent_effective_contract_invalid", str(error))
    text = str(value or "").strip()
    if not text:
        _fail("parent_effective_contract_missing")
    return text, None


def parent_mutation_owners_from_effective_contract(
    value: EffectiveSearchContract | Mapping[str, Any],
) -> Dict[Tuple[str, int], str]:


    _hash, semantic = _parse_effective_contract(value)
    if semantic is None:
        _fail("parent_owner_contract_unavailable")
    node_policies = semantic.get("node_policies")
    if not isinstance(node_policies, Mapping):
        _fail("parent_node_policies_missing")

    mapping_chains: Dict[Tuple[str, int], set[str]] = {}
    evaluator = semantic.get("evaluator_routing")
    if isinstance(evaluator, Mapping):
        plan = evaluator.get("executable_mapping_plan")
        if isinstance(plan, Mapping):
            actions = plan.get("action_specs") or plan.get("actions") or ()
            if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)):
                for raw in actions:
                    if not isinstance(raw, Mapping):
                        continue
                    node_id = str(raw.get("structural_node_id") or "")
                    chain_id = str(raw.get("chain_id") or "")
                    for position in raw.get("legal_positions") or ():
                        if node_id and chain_id:
                            mapping_chains.setdefault(
                                (node_id, int(position)), set()
                            ).add(chain_id)

    residue_contract = None
    generator = semantic.get("generator_policy")
    if isinstance(generator, Mapping):
        residue_contract = generator.get("residue_mutation_contract")
    masks = semantic.get("masks")
    owners: Dict[Tuple[str, int], str] = {}
    for segment_name, raw_policy in sorted(node_policies.items()):
        if not isinstance(raw_policy, Mapping):
            continue
        node_id = str(raw_policy.get("executable_ast_node_id") or "").strip()
        positions = raw_policy.get("ast_legal_positions")
        if not node_id or not positions:
            continue
        if isinstance(positions, (str, bytes)) or not isinstance(positions, Sequence):
            _fail("parent_legal_positions_invalid", str(segment_name))
        for raw_position in positions:
            try:
                position = int(raw_position)
            except (TypeError, ValueError):
                _fail("parent_legal_position_invalid", repr(raw_position))
            candidates = set(mapping_chains.get((node_id, position), set()))
            if isinstance(residue_contract, Mapping):
                contract_candidates = {
                    str(chain)
                    for chain, rules in residue_contract.items()
                    if isinstance(rules, Mapping)
                    and (str(position) in rules or position in rules)
                }
                candidates = (
                    candidates & contract_candidates
                    if candidates and contract_candidates
                    else candidates | contract_candidates
                )
            if isinstance(masks, Mapping):
                mask_candidates = {
                    str(chain)
                    for chain, mask in masks.items()
                    if isinstance(mask, Sequence)
                    and not isinstance(mask, (str, bytes))
                    and 0 <= position < len(mask)
                    and bool(mask[position])
                }
                candidates = (
                    candidates & mask_candidates
                    if candidates and mask_candidates
                    else candidates | mask_candidates
                )
            if len(candidates) != 1:
                _fail(
                    "parent_owner_chain_ambiguous",
                    f"{node_id}:{position}:{','.join(sorted(candidates))}",
                )
            chain = next(iter(candidates))
            key = (chain, position)
            if key in owners and owners[key] != node_id:
                _fail(
                    "parent_owner_ambiguous",
                    f"{chain}:{position}:{owners[key]},{node_id}",
                )
            owners[key] = node_id
    return owners


def _parse_design_action(value: Any) -> Any:
    try:
        from astevolve.domain.design_action import DesignAction
    except ImportError as error:
        _fail("design_action_schema_unavailable", str(error))
    if isinstance(value, DesignAction):


        return DesignAction.from_mapping(value.to_dict())
    if not isinstance(value, Mapping):
        _fail("design_action_invalid", type(value).__name__)
    try:
        if "design_action_hash" in value:
            return DesignAction.from_mapping(value)
        return DesignAction.from_payload(value)
    except Exception as error:
        code = str(getattr(error, "code", "design_action_invalid"))
        path = str(getattr(error, "path", ""))
        detail = str(getattr(error, "detail", "") or error)
        _fail(code, f"{path}:{detail}" if path else detail)


def _unsupported_nonempty_capabilities(action: Any) -> None:


    for field in (
        "mutation_modules",
        "sequence_seeds",
        "candidate_portfolio",
    ):
        if getattr(action, field, ()):
            _fail("unsupported_design_action_capability", field)


def _node_authority(
    plan: ExecutableNodePlan,
    sequences: Mapping[str, str],
) -> tuple[Dict[Tuple[str, int], str], Dict[str, Tuple[int, ...]], str]:
    if not isinstance(plan, ExecutableNodePlan):
        _fail("executable_node_plan_required")
    owners: Dict[Tuple[str, int], str] = {}
    positions_by_chain: Dict[str, list[int]] = {}
    for node in plan.structural_nodes:
        if node.declared_kind != "editable":
            if node.legal_positions:
                _fail("noneditable_node_has_legal_positions", node.node_id)
            continue
        chain = str(node.chain_id)
        if chain not in sequences:
            _fail("node_plan_unknown_chain", f"{node.node_id}:{chain}")
        for position in node.legal_positions:
            position = int(position)
            key = (chain, position)
            if position < 0 or position >= len(sequences[chain]):
                _fail("node_legal_position_out_of_bounds", f"{chain}:{position}")
            if key in owners:
                _fail(
                    "node_legal_position_ambiguous",
                    f"{chain}:{position}:{owners[key]},{node.node_id}",
                )
            owners[key] = str(node.node_id)
            positions_by_chain.setdefault(chain, []).append(position)
    active = {
        chain: tuple(sorted(positions))
        for chain, positions in sorted(positions_by_chain.items())
    }
    plan_hash = "executable_node_plan_sha256:" + _digest(
        "astevolve.executable_node_plan.v1", plan.to_dict()
    )
    return owners, active, plan_hash


def _normalize_fixed(
    value: Mapping[str, Mapping[int | str, str]] | None,
    sequences: Mapping[str, str],
    *,
    code_prefix: str,
) -> Dict[str, Dict[int, str]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        _fail(f"{code_prefix}_invalid")
    out: Dict[str, Dict[int, str]] = {}
    for raw_chain, raw_entries in value.items():
        chain = str(raw_chain)
        if chain not in sequences or not isinstance(raw_entries, Mapping):
            _fail(f"{code_prefix}_invalid", chain)
        entries: Dict[int, str] = {}
        for raw_position, raw_residue in raw_entries.items():
            try:
                position = int(raw_position)
            except (TypeError, ValueError):
                _fail(f"{code_prefix}_position_invalid", repr(raw_position))
            residue = str(raw_residue or "").upper()
            if (
                position < 0
                or position >= len(sequences[chain])
                or len(residue) != 1
                or residue not in "ACDEFGHIKLMNPQRSTVWY"
            ):
                _fail(f"{code_prefix}_invalid", f"{chain}:{position}:{residue}")
            if position in entries:
                _fail(f"{code_prefix}_position_duplicate", f"{chain}:{position}")
            entries[position] = residue
        out[chain] = dict(sorted(entries.items()))
    return dict(sorted(out.items()))


def _normalize_protected(
    value: Mapping[str, Iterable[int]] | None,
    sequences: Mapping[str, str],
) -> Dict[str, Tuple[int, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        _fail("protected_residues_invalid")
    out: Dict[str, Tuple[int, ...]] = {}
    for raw_chain, raw_positions in value.items():
        chain = str(raw_chain)
        if chain not in sequences or isinstance(raw_positions, (str, bytes)):
            _fail("protected_residues_invalid", chain)
        try:
            positions = tuple(sorted(int(item) for item in raw_positions))
        except (TypeError, ValueError):
            _fail("protected_residues_invalid", chain)
        if len(positions) != len(set(positions)):
            _fail("protected_residue_duplicate", chain)
        for position in positions:
            if position < 0 or position >= len(sequences[chain]):
                _fail("protected_residue_out_of_bounds", f"{chain}:{position}")
        out[chain] = positions
    return dict(sorted(out.items()))


def _normalize_hard_alphabet(
    value: Mapping[str, Mapping[int | str, Sequence[str]]],
    *,
    owners: Mapping[Tuple[str, int], str],
) -> Dict[str, Dict[int, Tuple[str, ...]]]:
    if not isinstance(value, Mapping):
        _fail("hard_alphabet_invalid")
    canonical = set("ACDEFGHIKLMNPQRSTVWY")
    out: Dict[str, Dict[int, Tuple[str, ...]]] = {}
    observed: set[Tuple[str, int]] = set()
    for raw_chain, raw_rules in value.items():
        chain = str(raw_chain)
        if not isinstance(raw_rules, Mapping):
            _fail("hard_alphabet_invalid", chain)
        rules: Dict[int, Tuple[str, ...]] = {}
        for raw_position, raw_residues in raw_rules.items():
            try:
                position = int(raw_position)
            except (TypeError, ValueError):
                _fail("hard_alphabet_position_invalid", repr(raw_position))
            key = (chain, position)
            if key not in owners:
                _fail("hard_alphabet_outside_active_selector", f"{chain}:{position}")
            if isinstance(raw_residues, (str, bytes)) or not isinstance(
                raw_residues, Sequence
            ):
                _fail("hard_alphabet_invalid", f"{chain}:{position}")
            alphabet = tuple(
                sorted({str(residue).strip().upper() for residue in raw_residues})
            )
            if not alphabet or any(
                len(residue) != 1 or residue not in canonical for residue in alphabet
            ):
                _fail("hard_alphabet_invalid", f"{chain}:{position}")
            rules[position] = alphabet
            observed.add(key)
        out[chain] = dict(sorted(rules.items()))
    missing = sorted(set(owners) - observed)
    if missing:
        _fail(
            "hard_alphabet_missing",
            ",".join(f"{chain}:{position}" for chain, position in missing),
        )
    return dict(sorted(out.items()))


def _parent_mutations(
    immutable: Mapping[str, str], parent: Mapping[str, str]
) -> Dict[Tuple[str, int], Tuple[str, str]]:
    return {
        (chain, position): (immutable[chain][position], parent[chain][position])
        for chain in immutable
        for position in range(len(immutable[chain]))
        if immutable[chain][position] != parent[chain][position]
    }


def _action_residue(action: Any) -> tuple[str, int]:
    residue = getattr(action, "residue", None)
    chain = str(getattr(residue, "chain_id", "") or "")
    position = getattr(residue, "zero_based_position", None)
    if not chain or isinstance(position, bool) or not isinstance(position, int):
        _fail("lineage_action_residue_invalid")
    return chain, position


def _mutations_between(
    source: Mapping[str, str],
    target: Mapping[str, str],
    owners: Mapping[Tuple[str, int], str],
    *,
    comparison_base: str,
) -> CompiledMutationSet:
    changes = []
    for chain in source:
        for position, (before, after) in enumerate(zip(source[chain], target[chain])):
            if before == after:
                continue
            changes.append(
                CompiledResidueChange.create(
                    chain_id=chain,
                    position=position,
                    from_residue=before,
                    to_residue=after,
                    owner_node_id=owners.get((chain, position)),
                )
            )
    return CompiledMutationSet.create(
        comparison_base=comparison_base, changes=changes
    )


def compile_design_action(
    design_action: Any,
    *,
    immutable_sequences: Mapping[str, str],
    immutable_sequence_bundle_hash: str,
    parent_sequences: Mapping[str, str],
    parent_effective_contract: EffectiveSearchContract | Mapping[str, Any] | str,
    trusted_case_id: str,
    trusted_parent_program_id: str,
    trusted_parent_candidate_id: str,
    trusted_parent_evolve_hash: str,
    executable_node_plan: ExecutableNodePlan,
    hard_allowed_residues: Mapping[
        str, Mapping[int | str, Sequence[str]]
    ],
    fixed_residues: Mapping[str, Mapping[int | str, str]] | None = None,
    protected_residues: Mapping[str, Iterable[int]] | None = None,
    max_total_mutations: int,
    _allow_portfolio_capabilities: bool = False,
) -> CompiledDesignAction:


    action = _parse_design_action(design_action)
    if not _allow_portfolio_capabilities:
        _unsupported_nonempty_capabilities(action)
    immutable = _canonical_sequences(
        immutable_sequences, label="immutable_reference"
    )
    parent = _canonical_sequences(parent_sequences, label="selected_parent")
    _require_same_shape(immutable, parent)

    expected_immutable_hash = SequenceBundleIdentity.create(
        immutable
    ).sequence_bundle_hash
    if str(immutable_sequence_bundle_hash or "") != expected_immutable_hash:
        _fail("immutable_sequence_bundle_hash_mismatch")
    parent_hash = SequenceBundleIdentity.create(parent).sequence_bundle_hash
    effective_contract_hash, parent_contract_semantic = _parse_effective_contract(
        parent_effective_contract
    )

    binding = action.parent
    trusted_bindings = {
        "case_id": str(trusted_case_id or ""),
        "parent_program_id": str(trusted_parent_program_id or ""),
        "parent_candidate_id": str(trusted_parent_candidate_id or ""),
        "parent_sequence_bundle_hash": parent_hash,
        "parent_effective_contract_hash": effective_contract_hash,
        "parent_evolve_hash": str(trusted_parent_evolve_hash or ""),
    }
    for field, expected in trusted_bindings.items():
        if not expected:
            _fail("trusted_parent_binding_missing", field)
        observed = str(getattr(binding, field, "") or "")
        if observed != expected:
            code = (
                "parent_effective_contract_mismatch"
                if field == "parent_effective_contract_hash"
                else "parent_binding_mismatch"
            )
            _fail(code, f"{field}:{observed}/{expected}")

    owners, active_positions, node_plan_hash = _node_authority(
        executable_node_plan, immutable
    )
    fixed = _normalize_fixed(
        fixed_residues, immutable, code_prefix="fixed_residue"
    )
    protected = _normalize_protected(protected_residues, immutable)
    for chain, positions in protected.items():
        for position in positions:
            if (chain, position) in owners:
                _fail("protected_residue_in_active_selector", f"{chain}:{position}")
    for chain, entries in fixed.items():
        for position in entries:
            if (chain, position) in owners:
                _fail("fixed_residue_in_active_selector", f"{chain}:{position}")
    alphabet = _normalize_hard_alphabet(
        hard_allowed_residues, owners=owners
    )

    inherited = _parent_mutations(immutable, parent)
    if inherited:
        if parent_contract_semantic is None:
            _fail("parent_owner_contract_unavailable")
        parent_owners = parent_mutation_owners_from_effective_contract(
            parent_effective_contract
        )
        missing_parent_owners = sorted(set(inherited) - set(parent_owners))
        if missing_parent_owners:
            _fail(
                "parent_mutation_owner_missing",
                ",".join(
                    f"{chain}:{position}"
                    for chain, position in missing_parent_owners
                ),
            )
    else:
        parent_owners = {}
    actions_by_position: Dict[Tuple[str, int], Any] = {}
    action_ids: set[str] = set()
    for lineage_action in action.lineage_actions:
        key = _action_residue(lineage_action)
        action_id = str(getattr(lineage_action, "lineage_action_id", "") or "")
        if action_id in action_ids:
            _fail("lineage_action_id_duplicate", action_id)
        action_ids.add(action_id)
        if key in actions_by_position:
            _fail("lineage_action_duplicate", f"{key[0]}:{key[1]}")
        if key not in inherited:
            _fail("lineage_action_not_parent_mutation", f"{key[0]}:{key[1]}")
        actions_by_position[key] = lineage_action

    missing = sorted(set(inherited) - set(actions_by_position))
    if missing:
        _fail(
            "lineage_action_missing",
            ",".join(f"{chain}:{position}" for chain, position in missing),
        )
    if len(actions_by_position) != len(inherited):
        _fail("ownership_accounting_incomplete")

    reconciled_lists = {chain: list(sequence) for chain, sequence in parent.items()}
    ledger_entries: list[MutationOwnershipEntry] = []
    for key in sorted(inherited):
        chain, position = key
        immutable_residue, parent_residue = inherited[key]
        lineage_action = actions_by_position[key]
        expected_parent = str(
            getattr(lineage_action, "expected_parent_residue", "") or ""
        ).upper()
        if expected_parent != parent_residue:
            _fail(
                "lineage_expected_parent_mismatch",
                f"{chain}:{position}:{expected_parent}/{parent_residue}",
            )
        kind = str(getattr(lineage_action, "action", "") or "")
        requested_node = getattr(lineage_action, "target_node_id", None)
        requested_node = (
            str(requested_node).strip()
            if requested_node not in (None, "")
            else None
        )
        effective_owner = owners.get(key)
        source_node = str(
            getattr(lineage_action, "source_node_id", "") or ""
        ).strip()
        parent_owner = parent_owners.get(key)
        if not source_node:
            _fail("lineage_source_node_required", f"{chain}:{position}")
        if source_node != parent_owner:
            _fail(
                "lineage_source_owner_mismatch",
                f"{chain}:{position}:{source_node}/{parent_owner}",
            )
        replacement = getattr(lineage_action, "replacement_residue", None)

        if kind == "revert_to_wt":
            target = immutable_residue
            ledger_owner = None
            requested_node = None
        elif kind == "retain":
            if effective_owner is None:
                _fail("retain_owner_inactive", f"{chain}:{position}")
            if (
                requested_node != effective_owner
                or source_node != requested_node
            ):
                _fail("retain_owner_mismatch", f"{chain}:{position}")
            target = parent_residue
            ledger_owner = effective_owner
        elif kind == "reassign_node":
            if (
                effective_owner is None
                or requested_node != effective_owner
                or source_node == requested_node
            ):
                _fail(
                    "reassign_owner_invalid",
                    f"{chain}:{position}:{requested_node}/{effective_owner}",
                )
            target = parent_residue
            ledger_owner = effective_owner
        elif kind == "replace":
            if effective_owner is None or requested_node != effective_owner:
                _fail(
                    "replace_owner_invalid",
                    f"{chain}:{position}:{requested_node}/{effective_owner}",
                )
            target = str(replacement or "").strip().upper()
            if len(target) != 1 or target not in "ACDEFGHIKLMNPQRSTVWY":
                _fail("replacement_residue_invalid", f"{chain}:{position}:{target}")
            if target in {parent_residue, immutable_residue}:
                _fail("replacement_residue_no_effect", f"{chain}:{position}")
            ledger_owner = effective_owner
        else:
            _fail("lineage_action_invalid", kind)

        if target != immutable_residue:
            if effective_owner is None:
                _fail("mutation_outside_active_selector", f"{chain}:{position}")
            allowed = alphabet.get(chain, {}).get(position, ())
            if target not in allowed:
                _fail(
                    "hard_alphabet_violation",
                    f"{chain}:{position}:{target}",
                )
        reconciled_lists[chain][position] = target
        ledger_entries.append(
            MutationOwnershipEntry.create(
                lineage_action_id=str(lineage_action.lineage_action_id),
                chain_id=chain,
                position=position,
                immutable_residue=immutable_residue,
                parent_residue=parent_residue,
                reconciled_residue=target,
                action=kind,
                source_node_id=source_node,
                requested_node_id=requested_node,
                effective_owner_node_id=ledger_owner,
            )
        )


    for requested_distribution in action.position_distributions:
        residue_ref = requested_distribution.residue
        chain = str(residue_ref.chain_id)
        position = int(residue_ref.zero_based_position)
        key = (chain, position)
        if key not in owners:
            _fail("position_distribution_not_active", f"{chain}:{position}")
        hard = set(alphabet[chain][position])
        enabled = set(requested_distribution.enabled_residues)
        forbidden = set(requested_distribution.forbidden_residues)
        if not enabled.issubset(hard):
            _fail(
                "position_distribution_enabled_outside_hard",
                f"{chain}:{position}:{','.join(sorted(enabled - hard))}",
            )
        if not forbidden.issubset(hard):
            _fail(
                "position_distribution_forbidden_outside_hard",
                f"{chain}:{position}:{','.join(sorted(forbidden - hard))}",
            )
        required = requested_distribution.required_mutation
        if required is None:
            continue
        if required == immutable[chain][position]:
            _fail(
                "position_distribution_required_noop",
                f"{chain}:{position}:{required}",
            )
        if required not in hard:
            _fail(
                "hard_alphabet_violation",
                f"{chain}:{position}:{required}",
            )
        current = reconciled_lists[chain][position]
        if key in inherited and current != required:
            _fail(
                "position_distribution_required_conflicts_lineage",
                f"{chain}:{position}:{current}/{required}",
            )
        reconciled_lists[chain][position] = required

    reconciled = {
        chain: "".join(residues) for chain, residues in reconciled_lists.items()
    }
    for chain, entries in fixed.items():
        for position, residue in entries.items():
            if reconciled[chain][position] != residue:
                _fail("fixed_residue_violation", f"{chain}:{position}")
    for chain, positions in protected.items():
        for position in positions:
            if reconciled[chain][position] != immutable[chain][position]:
                _fail("protected_residue_violation", f"{chain}:{position}")

    absolute = _mutations_between(
        immutable,
        reconciled,
        owners,
        comparison_base="immutable_reference",
    )
    lineage_delta = _mutations_between(
        parent,
        reconciled,
        owners,
        comparison_base="selected_parent",
    )
    outside = [
        change
        for change in absolute.changes
        if (change.chain_id, change.position) not in owners
    ]
    if outside:
        _fail(
            "mutation_outside_active_selector",
            ",".join(f"{item.chain_id}:{item.position}" for item in outside),
        )
    if (
        isinstance(max_total_mutations, bool)
        or not isinstance(max_total_mutations, int)
        or max_total_mutations < 0
    ):
        _fail("max_total_mutations_invalid")
    action_min, action_max = tuple(action.mutation_count_range)
    effective_max = min(action_max, max_total_mutations)
    if action_max > max_total_mutations:
        _fail(
            "mutation_budget_contract_mismatch",
            f"action_max={action_max},case_max={max_total_mutations}",
        )
    if absolute.mutation_count > effective_max:
        _fail(
            "absolute_mutation_budget_exceeded",
            f"{absolute.mutation_count}/{effective_max}",
        )
    minimum_hamming_distance = int(action.minimum_hamming_distance)
    if minimum_hamming_distance > len(owners):
        _fail(
            "minimum_hamming_distance_unsatisfiable",
            f"{minimum_hamming_distance}/{len(owners)}",
        )

    ledger = MutationOwnershipLedger.create(
        ledger_entries, parent_mutation_count=len(inherited)
    )
    compiled_position_distributions = tuple(
        CompiledPositionDistribution.create(
            chain_id=item.residue.chain_id,
            position=item.residue.zero_based_position,
            owner_node_id=owners[
                (item.residue.chain_id, item.residue.zero_based_position)
            ],
            immutable_residue=immutable[item.residue.chain_id][
                item.residue.zero_based_position
            ],
            reconciled_root_residue=reconciled[item.residue.chain_id][
                item.residue.zero_based_position
            ],
            hard_allowed_residues=alphabet[item.residue.chain_id][
                item.residue.zero_based_position
            ],
            enabled_residues=item.enabled_residues,
            forbidden_residues=item.forbidden_residues,
            requested_probabilities=item.probabilities,
            required_mutation=item.required_mutation,
            wt_revert_weight=item.wt_revert_weight,
            temperature=item.temperature,
        )
        for item in action.position_distributions
    )
    active_owner_mapping: Dict[str, Dict[int, str]] = {}
    for (chain, position), node_id in sorted(owners.items()):
        active_owner_mapping.setdefault(chain, {})[position] = node_id
    parent_owner_mapping: Dict[str, Dict[int, str]] = {}
    for (chain, position), node_id in sorted(parent_owners.items()):
        parent_owner_mapping.setdefault(chain, {})[position] = node_id
    return CompiledDesignAction.create(
        design_action_hash=action.design_action_hash,
        case_id=trusted_bindings["case_id"],
        parent_program_id=trusted_bindings["parent_program_id"],
        parent_candidate_id=trusted_bindings["parent_candidate_id"],
        parent_sequence_bundle_hash=parent_hash,
        parent_effective_contract_hash=effective_contract_hash,
        parent_evolve_hash=trusted_bindings["parent_evolve_hash"],
        immutable_sequence_bundle_hash=expected_immutable_hash,
        executable_node_plan_hash=node_plan_hash,
        reconciled_root_sequences=reconciled,
        active_selector_positions=active_positions,
        active_position_owners=active_owner_mapping,
        parent_position_owners=parent_owner_mapping,
        ownership_ledger=ledger,
        absolute_mutations=absolute,
        lineage_delta=lineage_delta,
        hard_allowed_residues=alphabet,
        fixed_residues=fixed,
        protected_residues=protected,
        mutation_count_range=(action_min, action_max),
        minimum_hamming_distance=minimum_hamming_distance,
        max_total_mutations=max_total_mutations,
        position_distributions=compiled_position_distributions,
    )


def _portfolio_exact_sequences(
    root: Mapping[str, str],
    changes: Sequence[CompiledResidueChange],
) -> Dict[str, str]:
    lists = {chain: list(sequence) for chain, sequence in root.items()}
    for change in changes:
        lists[change.chain_id][change.position] = change.to_residue
    return {chain: "".join(residues) for chain, residues in sorted(lists.items())}


def compile_design_action_with_portfolio(
    design_action: Any,
    *,
    immutable_sequences: Mapping[str, str],
    immutable_sequence_bundle_hash: str,
    parent_sequences: Mapping[str, str],
    parent_effective_contract: EffectiveSearchContract | Mapping[str, Any] | str,
    trusted_case_id: str,
    trusted_parent_program_id: str,
    trusted_parent_candidate_id: str,
    trusted_parent_evolve_hash: str,
    executable_node_plan: ExecutableNodePlan,
    hard_allowed_residues: Mapping[str, Mapping[int | str, Sequence[str]]],
    fixed_residues: Mapping[str, Mapping[int | str, str]] | None = None,
    protected_residues: Mapping[str, Iterable[int]] | None = None,
    max_total_mutations: int,
) -> tuple[CompiledDesignAction, CompiledPortfolioOptimizationRequest]:


    action = _parse_design_action(design_action)
    compiled = compile_design_action(
        action,
        immutable_sequences=immutable_sequences,
        immutable_sequence_bundle_hash=immutable_sequence_bundle_hash,
        parent_sequences=parent_sequences,
        parent_effective_contract=parent_effective_contract,
        trusted_case_id=trusted_case_id,
        trusted_parent_program_id=trusted_parent_program_id,
        trusted_parent_candidate_id=trusted_parent_candidate_id,
        trusted_parent_evolve_hash=trusted_parent_evolve_hash,
        executable_node_plan=executable_node_plan,
        hard_allowed_residues=hard_allowed_residues,
        fixed_residues=fixed_residues,
        protected_residues=protected_residues,
        max_total_mutations=max_total_mutations,
        _allow_portfolio_capabilities=True,
    )
    immutable = _canonical_sequences(immutable_sequences, label="immutable_reference")
    parent = _canonical_sequences(parent_sequences, label="selected_parent")
    root = dict(compiled.reconciled_root_sequences)
    owners = {
        (chain, position): node_id
        for chain, entries in compiled.active_position_owners.items()
        for position, node_id in entries.items()
    }

    compiled_modules: list[CompiledMutationModule] = []
    for module in action.mutation_modules:
        if not module.atomic:
            _fail("module_not_atomic", module.module_id)
        changes = []
        for mutation in module.mutations:
            chain = mutation.residue.chain_id
            position = mutation.residue.zero_based_position
            key = (chain, position)
            if key not in owners:
                _fail("module_mutation_outside_active_selector", f"{module.module_id}:{chain}:{position}")
            if module.operation != "matched_module_ablation" and mutation.from_residue != root[chain][position]:
                _fail(
                    "module_source_mismatch",
                    f"{module.module_id}:{chain}:{position}:{mutation.from_residue}/{root[chain][position]}",
                )
            if mutation.to_residue not in compiled.hard_allowed_residues[chain][position]:
                _fail("hard_alphabet_violation", f"{module.module_id}:{chain}:{position}:{mutation.to_residue}")
            changes.append(
                CompiledResidueChange.create(
                    chain_id=chain,
                    position=position,
                    from_residue=mutation.from_residue,
                    to_residue=mutation.to_residue,
                    owner_node_id=owners[key],
                )
            )
        try:
            compiled_module = CompiledMutationModule.create(
                module_id=module.module_id,
                operation=module.operation,
                atomic=module.atomic,
                mutations=changes,
            )
        except Exception as error:
            _fail(str(getattr(error, "code", "compiled_module_invalid")), str(error))
        if compiled_module.operation != "matched_module_ablation":
            validate_candidate_against_compiled_design_action(
                compiled,
                _portfolio_exact_sequences(root, compiled_module.mutations),
                expected_compiled_design_action_hash=(
                    compiled.compiled_design_action_hash
                ),
            )
        compiled_modules.append(compiled_module)
    modules_by_id = {item.module_id: item for item in compiled_modules}

    compiled_seeds: list[CompiledSequenceSeed] = []
    for seed in action.sequence_seeds:
        sequences = _canonical_sequences(seed.sequences, label=f"sequence_seed.{seed.seed_id}")
        _require_same_shape(root, sequences)
        delta = _mutations_between(root, sequences, owners, comparison_base="reconciled_root")
        if not delta.changes:
            _fail("sequence_seed_not_novel", seed.seed_id)
        validate_candidate_against_compiled_design_action(
            compiled,
            sequences,
            expected_compiled_design_action_hash=compiled.compiled_design_action_hash,
        )
        refinement = []
        for residue in seed.refinement_positions:
            key = (residue.chain_id, residue.zero_based_position)
            if key not in owners:
                _fail("sequence_seed_refinement_outside_active_selector", f"{seed.seed_id}:{key[0]}:{key[1]}")
            refinement.append((key[0], key[1], owners[key]))
        try:
            compiled_seeds.append(
                CompiledSequenceSeed.create(
                    seed_id=seed.seed_id,
                    sequences=sequences,
                    root_delta=delta,
                    refinement_positions=refinement,
                )
            )
        except Exception as error:
            _fail(str(getattr(error, "code", "compiled_sequence_seed_invalid")), str(error))
    seeds_by_id = {item.seed_id: item for item in compiled_seeds}

    slots: list[CompiledCandidateSlot] = []

    def exact_slot(
        *, entry: Any, suffix: str, kind: str, sequences: Mapping[str, str]
    ) -> None:
        validate_candidate_against_compiled_design_action(
            compiled,
            sequences,
            expected_compiled_design_action_hash=compiled.compiled_design_action_hash,
        )
        delta = _mutations_between(root, sequences, owners, comparison_base="reconciled_root")
        slots.append(
            CompiledCandidateSlot.create(
                slot_id=f"{entry.portfolio_id}:{suffix}",
                portfolio_id=entry.portfolio_id,
                role=entry.role,
                generation_mode=entry.generation_mode,
                realization_kind=kind,
                source_refs=entry.source_refs,
                required=entry.required,
                exact=True,
                exact_sequences=sequences,
                root_delta=delta,
            )
        )

    for entry in action.candidate_portfolio:
        mode = entry.generation_mode
        refs = set(entry.source_refs)
        if mode in {"module", "strict_sequence_seed", "matched_ablation"} and entry.count != 1:
            _fail("portfolio_exact_count_invalid", f"{entry.portfolio_id}:{entry.count}")
        if mode == "module":
            if len(refs) != 1 or not refs.issubset(modules_by_id):
                _fail("module_portfolio_reference_invalid", entry.portfolio_id)
            module = modules_by_id[next(iter(refs))]
            if module.operation == "matched_module_ablation":
                _fail("module_portfolio_reference_invalid", entry.portfolio_id)
            exact_slot(
                entry=entry,
                suffix="module",
                kind="module",
                sequences=_portfolio_exact_sequences(root, module.mutations),
            )
        elif mode == "strict_sequence_seed":
            if len(refs) != 1 or not refs.issubset(seeds_by_id):
                _fail("sequence_seed_portfolio_reference_invalid", entry.portfolio_id)
            exact_slot(
                entry=entry,
                suffix="seed",
                kind="sequence_seed",
                sequences=seeds_by_id[next(iter(refs))].sequences,
            )
        elif mode == "matched_ablation":
            if len(refs) != 2 or not refs.issubset(modules_by_id):
                _fail("matched_ablation_reference_invalid", entry.portfolio_id)
            referenced = [modules_by_id[item] for item in refs]
            treatments = [item for item in referenced if item.operation != "matched_module_ablation"]
            ablations = [item for item in referenced if item.operation == "matched_module_ablation"]
            if len(treatments) != 1 or len(ablations) != 1:
                _fail("matched_ablation_reference_invalid", entry.portfolio_id)
            treatment, ablation = treatments[0], ablations[0]
            treatment_by_position = {
                (item.chain_id, item.position): item for item in treatment.mutations
            }
            ablation_by_position = {
                (item.chain_id, item.position): item for item in ablation.mutations
            }
            if not ablation_by_position or not set(ablation_by_position) < set(treatment_by_position):
                _fail("matched_ablation_not_proper_subset", entry.portfolio_id)
            for key, change in ablation_by_position.items():
                treatment_change = treatment_by_position[key]
                if change.from_residue != treatment_change.to_residue or change.to_residue != root[key[0]][key[1]]:
                    _fail("matched_ablation_delta_invalid", f"{entry.portfolio_id}:{key[0]}:{key[1]}")
            treatment_sequences = _portfolio_exact_sequences(root, treatment.mutations)
            control_lists = {chain: list(sequence) for chain, sequence in treatment_sequences.items()}
            for chain, position in ablation_by_position:
                control_lists[chain][position] = root[chain][position]
            control_sequences = {
                chain: "".join(residues) for chain, residues in sorted(control_lists.items())
            }
            exact_slot(entry=entry, suffix="treatment", kind="matched_treatment", sequences=treatment_sequences)
            exact_slot(entry=entry, suffix="control", kind="matched_control", sequences=control_sequences)
        elif mode in {"mcts", "control"}:
            if refs:
                _fail("free_portfolio_reference_invalid", entry.portfolio_id)
            for index in range(entry.count):
                slots.append(
                    CompiledCandidateSlot.create(
                        slot_id=f"{entry.portfolio_id}:{index}",
                        portfolio_id=entry.portfolio_id,
                        role=entry.role,
                        generation_mode=mode,
                        realization_kind=mode,
                        source_refs=(),
                        required=entry.required,
                        exact=False,
                        exact_sequences={},
                        root_delta=CompiledMutationSet.create(
                            comparison_base="reconciled_root", changes=()
                        ),
                    )
                )
        else:
            _fail("generation_mode_invalid", mode)

    try:
        request = CompiledPortfolioOptimizationRequest.create(
            design_action_hash=action.design_action_hash,
            compiled_design_action_hash=compiled.compiled_design_action_hash,
            case_id=compiled.case_id,
            parent_program_id=compiled.parent_program_id,
            parent_candidate_id=compiled.parent_candidate_id,
            immutable_sequences=immutable,
            immutable_sequence_bundle_hash=compiled.immutable_sequence_bundle_hash,
            parent_sequences=parent,
            parent_sequence_bundle_hash=compiled.parent_sequence_bundle_hash,
            reconciled_root_sequences=root,
            reconciled_root_sequence_bundle_hash=compiled.reconciled_root_sequence_bundle_hash,
            mutation_modules=compiled_modules,
            sequence_seeds=compiled_seeds,
            candidate_slots=slots,
        )
    except Exception as error:
        _fail(str(getattr(error, "code", "compiled_portfolio_request_invalid")), str(error))
    return compiled, request


def validate_compiled_position_distribution_optimizer_support(
    compiled_action: CompiledDesignAction | Mapping[str, Any],
    *,
    top_k_per_position: int,
) -> None:


    if (
        isinstance(top_k_per_position, bool)
        or not isinstance(top_k_per_position, int)
        or top_k_per_position < 1
    ):
        _fail("node_optimizer_top_k_invalid", repr(top_k_per_position))
    artifact = (
        CompiledDesignAction.from_artifact(compiled_action)
        if isinstance(compiled_action, Mapping)
        else CompiledDesignAction.from_artifact(compiled_action.to_artifact())
    )
    violations = []
    for item in artifact.position_distributions:
        if item.required_mutation is not None:
            continue
        substitutions = sorted(
            residue
            for residue, probability in item.effective_probabilities.items()
            if probability > 0.0 and residue != item.reconciled_root_residue
        )
        if len(substitutions) > top_k_per_position:
            violations.append(
                f"{item.chain_id}:{item.position}:"
                f"{len(substitutions)}/{top_k_per_position}:"
                f"{','.join(substitutions)}"
            )
    if violations:
        _fail(
            "position_distribution_top_k_truncation",
            ";".join(violations),
        )


def validate_candidate_against_compiled_design_action(
    compiled: CompiledDesignAction | Mapping[str, Any],
    candidate_sequences: Mapping[str, str],
    *,
    expected_compiled_design_action_hash: str,
    allow_reconciled_root: bool = False,
) -> tuple[CompiledMutationSet, CompiledMutationSet]:


    try:
        artifact = (
            CompiledDesignAction.from_mapping(compiled)
            if isinstance(compiled, Mapping)
            else CompiledDesignAction.from_mapping(compiled.to_dict())
        )
    except Exception as error:
        _fail("compiled_design_action_invalid", str(error))
    if expected_compiled_design_action_hash != artifact.compiled_design_action_hash:
        _fail("candidate_compiled_action_hash_mismatch")
    candidate = _canonical_sequences(candidate_sequences, label="candidate")
    root = dict(artifact.reconciled_root_sequences)
    _require_same_shape(root, candidate)
    owners = {
        (chain, position): node_id
        for chain, entries in artifact.active_position_owners.items()
        for position, node_id in entries.items()
    }

    immutable_lists = {chain: list(sequence) for chain, sequence in root.items()}
    for change in artifact.absolute_mutations.changes:
        immutable_lists[change.chain_id][change.position] = change.from_residue
    immutable = {
        chain: "".join(residues) for chain, residues in immutable_lists.items()
    }
    absolute = _mutations_between(
        immutable,
        candidate,
        owners,
        comparison_base="immutable_reference",
    )
    incremental = _mutations_between(
        root,
        candidate,
        owners,
        comparison_base="reconciled_root",
    )
    is_reconciled_root = incremental.mutation_count == 0
    if not (allow_reconciled_root and is_reconciled_root):
        for distribution in artifact.position_distributions:
            observed = candidate[distribution.chain_id][distribution.position]
            required = distribution.required_mutation
            if required is not None and observed != required:
                _fail(
                    "position_distribution_required_violation",
                    f"{distribution.chain_id}:{distribution.position}:{observed}/{required}",
                )
            if observed in distribution.forbidden_residues:
                _fail(
                    "position_distribution_forbidden_violation",
                    f"{distribution.chain_id}:{distribution.position}:{observed}",
                )
            probability = distribution.effective_probabilities.get(observed)
            if probability is not None and probability <= 0.0:
                _fail(
                    "position_distribution_zero_probability_violation",
                    f"{distribution.chain_id}:{distribution.position}:{observed}",
                )
            if probability is None:
                _fail(
                    "position_distribution_disabled_residue_violation",
                    f"{distribution.chain_id}:{distribution.position}:{observed}",
                )
    for chain, entries in artifact.fixed_residues.items():
        for position, residue in entries.items():
            if candidate[chain][position] != residue:
                _fail("fixed_residue_violation", f"{chain}:{position}")
    for chain, positions in artifact.protected_residues.items():
        for position in positions:
            if candidate[chain][position] != immutable[chain][position]:
                _fail("protected_residue_violation", f"{chain}:{position}")
    for change in absolute.changes:
        key = (change.chain_id, change.position)
        if key not in owners:
            _fail("mutation_outside_active_selector", f"{key[0]}:{key[1]}")
        if change.to_residue not in artifact.hard_allowed_residues[
            change.chain_id
        ][change.position]:
            _fail(
                "hard_alphabet_violation",
                f"{change.chain_id}:{change.position}:{change.to_residue}",
            )
    minimum, maximum = artifact.mutation_count_range


    reconciled_root_exception = bool(allow_reconciled_root and is_reconciled_root)
    if (
        (not reconciled_root_exception and absolute.mutation_count < minimum)
        or absolute.mutation_count > maximum
    ):
        _fail(
            "absolute_mutation_count_out_of_range",
            f"{absolute.mutation_count}/{minimum}-{maximum}",
        )
    if absolute.mutation_count > artifact.max_total_mutations:
        _fail("absolute_mutation_budget_exceeded")
    if incremental.mutation_count > artifact.mutation_count_range[1]:
        _fail(
            "incremental_mutation_budget_exceeded",
            f"{incremental.mutation_count}/{artifact.mutation_count_range[1]}",
        )
    if not reconciled_root_exception and incremental.mutation_count == 0:
        _fail("candidate_identical_to_reconciled_root")
    if (
        incremental.mutation_count > 0
        and incremental.mutation_count < artifact.minimum_hamming_distance
    ):
        _fail(
            "minimum_hamming_distance_not_met",
            f"{incremental.mutation_count}/{artifact.minimum_hamming_distance}",
        )
    return absolute, incremental


def validate_candidate_against_compiled_action(
    compiled_action: CompiledDesignAction | Mapping[str, Any],
    candidate_sequences: Mapping[str, str],
    candidate_compiled_hash: str,
    *,
    allow_reconciled_root: bool = False,
) -> Dict[str, Any]:


    artifact = (
        CompiledDesignAction.from_artifact(compiled_action)
        if isinstance(compiled_action, Mapping)
        else CompiledDesignAction.from_artifact(compiled_action.to_artifact())
    )
    absolute, incremental = validate_candidate_against_compiled_design_action(
        artifact,
        candidate_sequences,
        expected_compiled_design_action_hash=candidate_compiled_hash,
        allow_reconciled_root=allow_reconciled_root,
    )
    candidate_hash = SequenceBundleIdentity.create(
        _canonical_sequences(candidate_sequences, label="candidate")
    ).sequence_bundle_hash
    semantic = {
        "passed": True,
        "compiled_design_action_hash": artifact.compiled_design_action_hash,
        "design_action_hash": artifact.design_action_hash,
        "candidate_sequence_bundle_hash": candidate_hash,
        "absolute_mutations": absolute.to_dict(),
        "incremental_mutations": incremental.to_dict(),
        "ownership_coverage_fraction": artifact.ownership_ledger.coverage_fraction,
        "mutation_outside_active_selector_count": 0,
        "hard_alphabet_violation_count": 0,
        "fixed_residue_violation_count": 0,
        "protected_residue_violation_count": 0,
        "absolute_budget_passed": True,
        "incremental_budget_passed": True,
    }
    schema_version = "astevolve.candidate_design_action_validation.v1"
    if artifact.position_distributions:
        candidate = _canonical_sequences(candidate_sequences, label="candidate")
        semantic.update(
            {
                "position_distribution_set_hash": (
                    artifact.position_distribution_set_hash
                ),
                "position_distribution_violation_count": 0,
                "zero_probability_violation_count": 0,
                "forbidden_residue_violation_count": 0,
                "required_mutation_violation_count": 0,
                "position_distribution_realization": [
                    {
                        "chain_id": item.chain_id,
                        "position": item.position,
                        "owner_node_id": item.owner_node_id,
                        "distribution_hash": item.distribution_hash,
                        "requested_probabilities": dict(
                            item.requested_probabilities
                        ),
                        "effective_probabilities": dict(
                            item.effective_probabilities
                        ),
                        "required_mutation": item.required_mutation,
                        "actual_residue": candidate[item.chain_id][item.position],
                        "actual_probability": item.effective_probabilities.get(
                            candidate[item.chain_id][item.position]
                        ),
                        "prior_source": "compiled_exact_override_v1",
                    }
                    for item in artifact.position_distributions
                ],
            }
        )
        schema_version = "astevolve.candidate_design_action_validation.v2"
    return {
        "schema_version": schema_version,
        **semantic,
        "validation_receipt_hash": "candidate_validation_sha256:"
        + _digest(schema_version, semantic),
    }


__all__ = [
    "DesignActionCompileError",
    "compile_design_action",
    "compile_design_action_with_portfolio",
    "parent_mutation_owners_from_effective_contract",
    "validate_compiled_position_distribution_optimizer_support",
    "validate_candidate_against_compiled_action",
    "validate_candidate_against_compiled_design_action",
]
