

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, Mapping

from engine.causal_flow import EffectiveSearchContract
from engine.experiment_identity import SequenceBundleIdentity
from outerloop.structured_ast_proposal import parent_evolve_hash
from outerloop.structured_ast_proposal import extract_current_ast_revision_plan


class DesignActionParentBindingError(ValueError):


    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise DesignActionParentBindingError(
                "parent_artifact_not_utf8", label
            ) from error
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise DesignActionParentBindingError(
                "parent_artifact_invalid_json", label
            ) from error
    if not isinstance(value, Mapping):
        raise DesignActionParentBindingError(
            "parent_artifact_mapping_required", label
        )
    return value


def build_design_action_parent_binding(
    *,
    parent_program_id: str,
    parent_code: str,
    parent_artifacts: Mapping[str, Any] | None,
) -> Dict[str, str]:


    program_id = str(parent_program_id or "").strip()
    if not program_id:
        raise DesignActionParentBindingError("parent_program_id_missing")
    artifacts = parent_artifacts if isinstance(parent_artifacts, Mapping) else {}
    sequences = _mapping(artifacts.get("best_seqs"), "best_seqs")
    try:
        sequence_identity = SequenceBundleIdentity.create(sequences)
    except (TypeError, ValueError) as error:
        raise DesignActionParentBindingError(
            "parent_sequence_identity_invalid", str(error)
        ) from error

    raw_contract = _mapping(
        artifacts.get("effective_search_contract"),
        "effective_search_contract",
    )
    try:
        contract = EffectiveSearchContract.from_mapping(raw_contract)
    except (TypeError, ValueError) as error:
        raise DesignActionParentBindingError(
            "parent_effective_contract_invalid", str(error)
        ) from error
    state_context = contract.semantic.get("state_context")
    if not isinstance(state_context, Mapping):
        raise DesignActionParentBindingError("parent_case_id_missing")
    case_id = str(state_context.get("case_id") or "").strip()
    if not case_id:
        raise DesignActionParentBindingError("parent_case_id_missing")

    search = _mapping(artifacts.get("search_artifacts"), "search_artifacts")
    decision = search.get("structure_selection_decision")
    candidate_id = ""
    if isinstance(decision, Mapping):
        candidate_id = str(decision.get("selected_candidate_id") or "").strip()
    if not candidate_id:
        candidate_id = str(search.get("best_node_id") or "").strip()
    if not candidate_id:
        raise DesignActionParentBindingError("parent_candidate_id_missing")

    return {
        "case_id": case_id,
        "parent_program_id": program_id,
        "parent_candidate_id": candidate_id,
        "parent_sequence_bundle_hash": sequence_identity.sequence_bundle_hash,
        "parent_effective_contract_hash": contract.contract_hash,
        "parent_evolve_hash": parent_evolve_hash(parent_code),
    }


def design_action_parent_prompt_projection(
    binding: Mapping[str, str],
) -> Dict[str, Any]:


    required = {
        "case_id",
        "parent_program_id",
        "parent_candidate_id",
        "parent_sequence_bundle_hash",
        "parent_effective_contract_hash",
        "parent_evolve_hash",
    }
    if not isinstance(binding, Mapping) or set(binding) != required:
        raise DesignActionParentBindingError("parent_binding_shape_invalid")
    return {
        "schema_version": "astevolve.design_action_parent_prompt.v1",
        "binding": deepcopy(dict(binding)),
        "instruction": (
            "Copy every binding field exactly; controller validation rejects "
            "missing, stale, or altered values before sequence search."
        ),
    }


def build_structured_design_action_prompt_context(
    *,
    parent_code: str,
    parent_artifacts: Mapping[str, Any] | None,
    binding: Mapping[str, str],
    require_sequence_reconciliation: bool = False,
    require_position_distributions: bool = False,
    require_portfolio_capabilities: bool = False,
    require_frozen_candidate_wave: bool = False,
) -> Dict[str, Any]:


    artifacts = parent_artifacts if isinstance(parent_artifacts, Mapping) else {}
    parent_sequences = SequenceBundleIdentity.create(
        _mapping(artifacts.get("best_seqs"), "best_seqs")
    )
    reference_artifact = _mapping(
        artifacts.get("immutable_sequence_reference"),
        "immutable_sequence_reference",
    )
    try:
        reference = SequenceBundleIdentity.from_mapping(reference_artifact)
    except (TypeError, ValueError) as error:
        raise DesignActionParentBindingError(
            "immutable_reference_invalid", str(error)
        ) from error
    if set(reference.chains) != set(parent_sequences.chains):
        raise DesignActionParentBindingError("immutable_parent_chain_mismatch")

    contract = EffectiveSearchContract.from_mapping(
        _mapping(
            artifacts.get("effective_search_contract"),
            "effective_search_contract",
        )
    )
    search_budget = contract.semantic.get("search_budget")
    if not isinstance(search_budget, Mapping):
        raise DesignActionParentBindingError(
            "parent_search_budget_missing"
        )
    max_total_mutations = search_budget.get("max_total_mutations")
    if (
        isinstance(max_total_mutations, bool)
        or not isinstance(max_total_mutations, int)
        or max_total_mutations < 0
    ):
        raise DesignActionParentBindingError(
            "parent_max_total_mutations_invalid"
        )
    owner_by_residue: Dict[tuple[str, int], set[str]] = {}
    for policy in contract.semantic.get("node_policies", {}).values():
        if not isinstance(policy, Mapping):
            continue
        node_id = str(policy.get("executable_ast_node_id") or "").strip()
        positions = policy.get("ast_legal_positions")
        if not node_id or not isinstance(positions, list):
            continue


        for raw_position in positions:
            if isinstance(raw_position, bool) or not isinstance(raw_position, int):
                raise DesignActionParentBindingError("parent_owner_position_invalid")
            candidate_chains = [
                chain_id
                for chain_id, mask in contract.semantic.get("masks", {}).items()
                if isinstance(mask, list)
                and 0 <= raw_position < len(mask)
                and bool(mask[raw_position])
            ]
            if len(candidate_chains) != 1:
                raise DesignActionParentBindingError(
                    "parent_owner_chain_ambiguous", f"{node_id}:{raw_position}"
                )
            owner_by_residue.setdefault(
                (str(candidate_chains[0]), raw_position), set()
            ).add(node_id)

    mutations = []
    for chain_id in sorted(reference.chains):
        wt = reference.chains[chain_id]
        parent = parent_sequences.chains[chain_id]
        if len(wt) != len(parent):
            raise DesignActionParentBindingError(
                "immutable_parent_length_mismatch", chain_id
            )
        for position, (wt_residue, parent_residue) in enumerate(zip(wt, parent)):
            if wt_residue == parent_residue:
                continue
            mutations.append(
                {
                    "residue": {
                        "chain_id": chain_id,
                        "residue_number": position + 1,
                    },
                    "wt_residue": wt_residue,
                    "parent_residue": parent_residue,
                    "parent_node_ids": sorted(
                        owner_by_residue.get((chain_id, position), set())
                    ),
                    "required_action_count": 1,
                    "allowed_actions": [
                        "retain",
                        "reassign_node",
                        "replace",
                        "revert_to_wt",
                    ],
                }
            )

    generator_policy = contract.semantic.get("generator_policy")
    residue_contract = (
        generator_policy.get("residue_mutation_contract")
        if isinstance(generator_policy, Mapping)
        else None
    )
    if not isinstance(residue_contract, Mapping):
        if require_position_distributions:
            raise DesignActionParentBindingError("parent_residue_contract_missing")
        residue_contract = {}
    raw_top_k = (
        generator_policy.get("node_optimizer_top_k_per_position")
        if isinstance(generator_policy, Mapping)
        else None
    )
    exact_support_cap = (
        int(raw_top_k)
        if isinstance(raw_top_k, int)
        and not isinstance(raw_top_k, bool)
        and raw_top_k > 0
        else 4
    )
    active_position_catalog = []
    for chain_id, mask in sorted(contract.semantic.get("masks", {}).items()):
        if not isinstance(mask, list) or chain_id not in reference.chains:
            continue
        chain_rules = residue_contract.get(chain_id)
        if not isinstance(chain_rules, Mapping):
            if require_position_distributions:
                raise DesignActionParentBindingError(
                    "parent_residue_contract_chain_missing", str(chain_id)
                )
            continue
        for position, editable in enumerate(mask):
            if not bool(editable):
                continue
            raw_alphabet = chain_rules.get(str(position), chain_rules.get(position))
            if not isinstance(raw_alphabet, list) or not raw_alphabet:
                raise DesignActionParentBindingError(
                    "parent_residue_contract_position_missing",
                    f"{chain_id}:{position}",
                )
            owners = sorted(owner_by_residue.get((str(chain_id), position), set()))
            if len(owners) != 1:
                raise DesignActionParentBindingError(
                    "parent_owner_position_ambiguous", f"{chain_id}:{position}"
                )
            active_position_catalog.append(
                {
                    "residue": {
                        "chain_id": str(chain_id),
                        "residue_number": position + 1,
                    },
                    "owner_node_id": owners[0],
                    "immutable_residue": reference.chains[str(chain_id)][position],
                    "parent_residue": parent_sequences.chains[str(chain_id)][position],
                    "hard_allowed_residues": list(raw_alphabet),
                    "llm_may_only_narrow_or_reweight": True,
                }
            )
    if require_sequence_reconciliation and not mutations:
        raise DesignActionParentBindingError(
            "step1_sequence_reconciliation_parent_has_no_non_wt_residue"
        )

    if require_portfolio_capabilities and require_frozen_candidate_wave:
        raise DesignActionParentBindingError(
            "design_action_acceptance_mode_conflict",
            "Step3 exact and Step4 frozen-wave modes are mutually exclusive",
        )
    portfolio_enabled = bool(
        require_portfolio_capabilities or require_frozen_candidate_wave
    )

    prospective_position_catalog = []
    frontier_evidence_by_coordinate: Dict[tuple[str, int], str] = {}
    if require_position_distributions or require_frozen_candidate_wave:
        active_coordinates = {
            (
                str(row["residue"]["chain_id"]),
                int(row["residue"]["residue_number"]) - 1,
            )
            for row in active_position_catalog
        }
        raw_frontier = artifacts.get("migration_frontier")
        if not isinstance(raw_frontier, Mapping):
            if require_frozen_candidate_wave:
                raise DesignActionParentBindingError(
                    "migration_frontier_schema_invalid"
                )
            raw_frontier = {
                "schema_version": "astevolve.migration_frontier.v1",
                "indexing": "zero_based",
                "segments": [],
            }
        frontier = dict(raw_frontier)
        if frontier.get("schema_version") != "astevolve.migration_frontier.v1":
            raise DesignActionParentBindingError("migration_frontier_schema_invalid")
        source_indexing = str(frontier.get("indexing") or "zero_based")
        if source_indexing not in {"zero_based", "one_based"}:
            raise DesignActionParentBindingError("migration_frontier_indexing_invalid")
        seen_future: set[tuple[str, int]] = set()
        for segment in frontier.get("segments", []) or []:
            if not isinstance(segment, Mapping):
                continue
            chain_id = str(segment.get("chain_id") or "")
            compiled_segment = str(segment.get("compiled_segment") or "")
            if not chain_id or chain_id not in reference.chains or not compiled_segment:
                continue
            for row in segment.get("positions", []) or []:
                if not isinstance(row, Mapping):
                    continue
                raw_position = row.get("position")
                if isinstance(raw_position, bool) or not isinstance(raw_position, int):
                    raise DesignActionParentBindingError(
                        "migration_frontier_position_invalid"
                    )
                position = raw_position - 1 if source_indexing == "one_based" else raw_position
                if not (0 <= position < len(reference.chains[chain_id])):
                    raise DesignActionParentBindingError(
                        "migration_frontier_position_invalid",
                        f"{chain_id}:{raw_position}",
                    )
                alphabet = row.get("hard_allowed_residues")
                if not isinstance(alphabet, list) or not alphabet:


                    continue
                coordinate = (chain_id, position)
                evidence_id = str(row.get("evidence_id") or "").strip()
                if evidence_id:
                    frontier_evidence_by_coordinate[coordinate] = evidence_id


                if coordinate in active_coordinates:
                    continue
                if coordinate in seen_future:
                    raise DesignActionParentBindingError(
                        "migration_frontier_position_duplicate",
                        f"{chain_id}:{position}",
                    )
                seen_future.add(coordinate)
                prospective_position_catalog.append(
                    {
                        "residue": {
                            "chain_id": chain_id,
                            "residue_number": position + 1,
                        },
                        "compiled_segment": compiled_segment,
                        "current_owner_node_id": row.get("incumbent_node_id"),
                        "immutable_residue": reference.chains[chain_id][position],
                        "parent_residue": parent_sequences.chains[chain_id][position],
                        "hard_allowed_residues": [str(item) for item in alphabet],
                        **({"evidence_id": evidence_id} if evidence_id else {}),
                        "case_owned_residue_tier": row.get(
                            "case_owned_residue_tier"
                        ),
                        "owner_assignment_source": "proposed_ast_selector",
                        "llm_may_only_narrow_or_reweight": True,
                    }
                )
        for row in active_position_catalog:
            residue = row["residue"]
            coordinate = (
                str(residue["chain_id"]),
                int(residue["residue_number"]) - 1,
            )
            evidence_id = frontier_evidence_by_coordinate.get(coordinate)
            if evidence_id:
                row["evidence_id"] = evidence_id
        if require_frozen_candidate_wave and not prospective_position_catalog:
            raise DesignActionParentBindingError(
                "step4_prospective_position_catalog_empty"
            )
    if require_position_distributions and (
        len(active_position_catalog) + len(prospective_position_catalog) < 2
    ):
        raise DesignActionParentBindingError(
            "step2_position_distribution_catalog_underfilled",
            str(len(active_position_catalog) + len(prospective_position_catalog)),
        )

    exact_module_count = 3 if require_frozen_candidate_wave else 2
    exact_portfolio_entry_count = 7 if require_frozen_candidate_wave else 3
    exact_candidate_slot_count = 8 if require_frozen_candidate_wave else 4
    exact_unique_sequence_count = 8 if require_frozen_candidate_wave else 3
    step3_capabilities = {
        "enabled": portfolio_enabled,
        "controller_mode": (
            "step4_frozen_wave"
            if require_frozen_candidate_wave
            else "step3_exact"
            if require_portfolio_capabilities
            else "disabled"
        ),
        "mutation_modules": portfolio_enabled,
        "sequence_seeds": portfolio_enabled,
        "candidate_portfolio": portfolio_enabled,
        "minimum_atomic_module_count": (
            exact_module_count if portfolio_enabled else 0
        ),
        "minimum_cross_node_atomic_module_count": (
            1 if portfolio_enabled else 0
        ),
        "minimum_sequence_seed_count": (
            1 if portfolio_enabled else 0
        ),
        "minimum_module_portfolio_count": (
            1 if portfolio_enabled else 0
        ),
        "minimum_strict_sequence_seed_portfolio_count": (
            1 if portfolio_enabled else 0
        ),
        "minimum_matched_ablation_portfolio_count": (
            1 if portfolio_enabled else 0
        ),
        "acceptance_quota_semantics": (
            "frozen_wave" if require_frozen_candidate_wave else
            "exact" if require_portfolio_capabilities else "disabled"
        ),
        "exact_atomic_module_count": (
            exact_module_count if portfolio_enabled else 0
        ),
        "exact_sequence_seed_count": (
            1 if portfolio_enabled else 0
        ),
        "exact_required_portfolio_entry_count": (
            exact_portfolio_entry_count if portfolio_enabled else 0
        ),
        "exact_compiled_candidate_slot_count": (
            exact_candidate_slot_count if portfolio_enabled else 0
        ),
        "exact_compiled_unique_sequence_count": (
            exact_unique_sequence_count if portfolio_enabled else 0
        ),
        "matched_ablation_source_semantics": (
            "exactly_one_non_ablation_treatment_module_and_exactly_one_"
            "matched_module_ablation_module; source_refs_are_an_unordered_set"
        ),
        "exact_sequence_seed_semantics": (
            "full_chain_bundle_equal_in_chain_set_and_length_to_reconciled_root"
        ),
        "required_portfolio_entries_are_fail_closed": True,
        "partial_atomic_realization_allowed": False,
    }
    step4_capabilities = {
        "enabled": bool(require_frozen_candidate_wave),
        "wave_size": 8 if require_frozen_candidate_wave else 0,
        "required_role_histogram": (
            {
                "primary": 2,
                "repair": 2,
                "matched_ablation": 2,
                "novelty": 1,
                "control": 1,
            }
            if require_frozen_candidate_wave
            else {}
        ),
        "required_portfolio_entry_shape": (
            {
                "module_primary": 1,
                "mcts_primary": 1,
                "strict_sequence_seed_repair": 1,
                "mcts_repair": 1,
                "matched_ablation": 1,
                "mcts_novelty": 1,
                "control": 1,
            }
            if require_frozen_candidate_wave
            else {}
        ),
        "free_slot_forced_tiers": (
            {
                "primary": "exploit",
                "repair": "repair",
                "novelty": "explore",
                "control": "exploit",
            }
            if require_frozen_candidate_wave
            else {}
        ),
        "require_live_node_change": bool(require_frozen_candidate_wave),
        "live_node_change_kinds": ["created", "migrated", "resized", "shifted"],
        "changed_node_min_generated_unique": (
            2 if require_frozen_candidate_wave else 0
        ),
        "changed_node_min_frozen_unique": (
            2 if require_frozen_candidate_wave else 0
        ),
        "changed_node_min_protenix_attempts": (
            1 if require_frozen_candidate_wave else 0
        ),
        "all_slots_required": True,
        "all_sequences_unique": True,
        "underfill_policy": "fail_before_any_provider",
        "protenix_mutant_quota": 8 if require_frozen_candidate_wave else 0,
        "af3_mutant_quota": 4 if require_frozen_candidate_wave else 0,
        "af3_required_roles": (
            ["matched_ablation", "matched_ablation", "primary", "repair"]
            if require_frozen_candidate_wave
            else []
        ),
    }
    current_ast_revision_plan = extract_current_ast_revision_plan(parent_code)
    allowed_residue_evidence_ids = {
        str(row.get("evidence_id"))
        for row in active_position_catalog + prospective_position_catalog
        if row.get("evidence_id")
    }
    for node in current_ast_revision_plan.get("structural_nodes", []) or []:
        if isinstance(node, Mapping):
            allowed_residue_evidence_ids.update(
                str(item) for item in node.get("evidence_refs", []) or []
            )
    for edge in current_ast_revision_plan.get("mapping_edges", []) or []:
        if isinstance(edge, Mapping):
            allowed_residue_evidence_ids.update(
                str(item) for item in edge.get("evidence_refs", []) or []
            )
    decision_record = current_ast_revision_plan.get("decision_record")
    if isinstance(decision_record, Mapping):
        allowed_residue_evidence_ids.update(
            str(item) for item in decision_record.get("evidence_refs", []) or []
        )
    context = {
        "schema_version": "astevolve.structured_design_action_context.v1",
        "parent": design_action_parent_prompt_projection(binding),
        "coordinate_system": {
            "chain_id_scheme": "runtime_chain_id",
            "residue_numbering": "one_based",
            "index_base": 1,
        },
        "current_ast_revision_plan": current_ast_revision_plan,
        "residue_evidence_contract": {
            "allowed_ast_evidence_ids": sorted(allowed_residue_evidence_ids),
            "applies_to": [
                "ast_revision_plan.structural_nodes[*].evidence_refs",
                "ast_revision_plan.mapping_edges[*].evidence_refs",
                "ast_revision_plan.decision_record.evidence_refs",
            ],
            "program_and_island_artifact_ids_forbidden": True,
            "unknown_id_policy": "reject_before_gpu",
        },
        "parent_non_wt_mutations": mutations,
        "active_position_catalog": active_position_catalog,
        "prospective_position_catalog": prospective_position_catalog,
        "lineage_contract": {
            "required_actions": len(mutations),
            "coverage": "exactly_one_action_per_parent_non_wt_residue",
            "silent_inheritance": "forbidden",
            "silent_clipping": "forbidden",
        },
        "step1_execution_capabilities": {
            "lineage_actions": True,
            "position_distributions": True,
            "mutation_modules": portfolio_enabled,
            "sequence_seeds": portfolio_enabled,
            "candidate_portfolio": portfolio_enabled,
            "nonempty_unsupported_fields_fail_code": (
                None
                if portfolio_enabled
                else "unsupported_design_action_capability"
            ),
            "required_mutation_count_range": [0, max_total_mutations],
            "required_minimum_hamming_distance": 0,
            "required_sequence_reconciliation": bool(
                require_sequence_reconciliation
            ),
            "minimum_position_distribution_count": (
                2 if require_position_distributions else 0
            ),
            "minimum_soft_position_distribution_count": (
                1 if require_position_distributions else 0
            ),
            "minimum_required_position_distribution_count": (
                1 if require_position_distributions else 0
            ),
            "max_positive_substitution_support_per_position": exact_support_cap,
            "mutation_count_semantics": (
                "absolute_non_wt_count_against_immutable_reference"
            ),
        },
        "step3_execution_capabilities": step3_capabilities,
        "step4_frozen_wave_capabilities": step4_capabilities,
    }
    if portfolio_enabled:
        context["step3_sequence_context"] = {
            "parent_sequences": dict(parent_sequences.chains),
            "immutable_reference_sequences": dict(reference.chains),
            "chain_lengths": {
                chain_id: len(sequence)
                for chain_id, sequence in sorted(parent_sequences.chains.items())
            },
            "required_exact_chain_set": sorted(parent_sequences.chains),
            "sequence_seed_must_be_novel_to_reconciled_root": True,
        }
    return context


def design_action_execution_constraints(
    context: Mapping[str, Any],
) -> Dict[str, Any]:


    if not isinstance(context, Mapping):
        raise DesignActionParentBindingError(
            "design_action_context_mapping_required"
        )
    capabilities = context.get("step1_execution_capabilities")
    if not isinstance(capabilities, Mapping):
        raise DesignActionParentBindingError(
            "design_action_execution_capabilities_missing"
        )
    for field in (
        "minimum_position_distribution_count",
        "minimum_soft_position_distribution_count",
        "minimum_required_position_distribution_count",
    ):
        value = capabilities.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DesignActionParentBindingError(
                "position_distribution_quota_invalid", field
            )
    require_position_distributions = bool(
        capabilities.get("minimum_position_distribution_count", 0)
    )
    raw_range = capabilities.get("required_mutation_count_range")
    if (
        not isinstance(raw_range, list)
        or len(raw_range) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_range)
        or raw_range[0] < 0
        or raw_range[0] > raw_range[1]
    ):
        raise DesignActionParentBindingError(
            "required_mutation_count_range_invalid"
        )
    minimum_hamming_distance = capabilities.get(
        "required_minimum_hamming_distance"
    )
    if (
        isinstance(minimum_hamming_distance, bool)
        or not isinstance(minimum_hamming_distance, int)
        or minimum_hamming_distance < 0
    ):
        raise DesignActionParentBindingError(
            "required_minimum_hamming_distance_invalid"
        )
    support_cap = capabilities.get(
        "max_positive_substitution_support_per_position"
    )
    if (
        isinstance(support_cap, bool)
        or not isinstance(support_cap, int)
        or support_cap < 1
    ):
        raise DesignActionParentBindingError(
            "position_distribution_support_cap_invalid"
        )
    site_context: Dict[str, Dict[str, Any]] = {}
    catalog = context.get("active_position_catalog")
    if not isinstance(catalog, list):
        raise DesignActionParentBindingError(
            "active_position_catalog_invalid"
        )
    for row in catalog:
        if not isinstance(row, Mapping):
            raise DesignActionParentBindingError(
                "active_position_catalog_row_invalid"
            )
        residue = row.get("residue")
        if not isinstance(residue, Mapping):
            raise DesignActionParentBindingError(
                "active_position_catalog_residue_invalid"
            )
        chain = str(residue.get("chain_id") or "")
        number = residue.get("residue_number")
        parent_residue = str(row.get("parent_residue") or "")
        immutable_residue = str(row.get("immutable_residue") or "")
        owner_node_id = str(row.get("owner_node_id") or "")
        raw_hard_alphabet = row.get("hard_allowed_residues")
        if (
            not chain
            or isinstance(number, bool)
            or not isinstance(number, int)
            or number < 1
            or len(parent_residue) != 1
            or len(immutable_residue) != 1
            or not owner_node_id
            or not isinstance(raw_hard_alphabet, list)
            or not raw_hard_alphabet
        ):
            raise DesignActionParentBindingError(
                "active_position_catalog_row_invalid"
            )
        key = f"{chain}:{number}"
        if key in site_context:
            raise DesignActionParentBindingError(
                "active_position_catalog_residue_duplicate", key
            )
        site_context[key] = {
            "parent_residue": parent_residue,
            "immutable_residue": immutable_residue,
            "owner_node_id": owner_node_id,
            "hard_allowed_residues": [
                str(residue) for residue in raw_hard_alphabet
            ],
        }
    step3 = context.get("step3_execution_capabilities")
    if not isinstance(step3, Mapping):
        raise DesignActionParentBindingError(
            "step3_execution_capabilities_missing"
        )
    require_portfolio_capabilities = bool(step3.get("enabled", False))
    step3_integer_fields = (
        "minimum_atomic_module_count",
        "minimum_cross_node_atomic_module_count",
        "minimum_sequence_seed_count",
        "minimum_module_portfolio_count",
        "minimum_strict_sequence_seed_portfolio_count",
        "minimum_matched_ablation_portfolio_count",
    )
    for field in step3_integer_fields:
        value = step3.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DesignActionParentBindingError(
                "step3_portfolio_quota_invalid", field
            )
    sequence_context: Dict[str, Any] = {}
    if require_portfolio_capabilities:
        raw_sequence_context = context.get("step3_sequence_context")
        if not isinstance(raw_sequence_context, Mapping):
            raise DesignActionParentBindingError(
                "step3_sequence_context_missing"
            )
        parent_sequence_context = raw_sequence_context.get("parent_sequences")
        immutable_sequence_context = raw_sequence_context.get(
            "immutable_reference_sequences"
        )
        if not isinstance(parent_sequence_context, Mapping) or not isinstance(
            immutable_sequence_context, Mapping
        ):
            raise DesignActionParentBindingError(
                "step3_sequence_context_invalid"
            )
        sequence_context = {
            "parent_sequences": {
                str(chain): str(sequence)
                for chain, sequence in parent_sequence_context.items()
            },
            "immutable_reference_sequences": {
                str(chain): str(sequence)
                for chain, sequence in immutable_sequence_context.items()
            },
        }
    step4 = context.get("step4_frozen_wave_capabilities")
    if not isinstance(step4, Mapping):
        raise DesignActionParentBindingError(
            "step4_frozen_wave_capabilities_missing"
        )
    require_frozen_candidate_wave = bool(step4.get("enabled", False))
    if require_frozen_candidate_wave and not require_portfolio_capabilities:
        raise DesignActionParentBindingError(
            "step4_portfolio_capability_missing"
        )
    expected_histogram = {
        "primary": 2,
        "repair": 2,
        "matched_ablation": 2,
        "novelty": 1,
        "control": 1,
    }
    role_histogram = step4.get("required_role_histogram", {})
    if require_frozen_candidate_wave and role_histogram != expected_histogram:
        raise DesignActionParentBindingError(
            "step4_role_histogram_invalid"
        )
    for field in (
        "wave_size",
        "changed_node_min_generated_unique",
        "changed_node_min_frozen_unique",
        "changed_node_min_protenix_attempts",
        "protenix_mutant_quota",
        "af3_mutant_quota",
    ):
        value = step4.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DesignActionParentBindingError(
                "step4_frozen_wave_quota_invalid", field
            )
    if require_frozen_candidate_wave and (
        int(step4.get("wave_size", 0)) != 8
        or int(step4.get("protenix_mutant_quota", 0)) != 8
        or int(step4.get("af3_mutant_quota", 0)) != 4
    ):
        raise DesignActionParentBindingError("step4_frozen_wave_quota_invalid")

    portfolio_site_context = deepcopy(site_context)
    if require_position_distributions or require_frozen_candidate_wave:
        prospective = context.get("prospective_position_catalog")
        if not isinstance(prospective, list):
            raise DesignActionParentBindingError(
                "prospective_position_catalog_invalid"
            )
        if require_frozen_candidate_wave and not prospective:
            raise DesignActionParentBindingError(
                "step4_prospective_position_catalog_empty"
            )
        for row in prospective:
            if not isinstance(row, Mapping):
                raise DesignActionParentBindingError(
                    "prospective_position_catalog_row_invalid"
                )
            residue = row.get("residue")
            if not isinstance(residue, Mapping):
                raise DesignActionParentBindingError(
                    "prospective_position_catalog_row_invalid"
                )
            chain = str(residue.get("chain_id") or "")
            number = residue.get("residue_number")
            alphabet = row.get("hard_allowed_residues")
            compiled_segment = str(row.get("compiled_segment") or "")
            if (
                not chain
                or isinstance(number, bool)
                or not isinstance(number, int)
                or number < 1
                or not compiled_segment
                or not isinstance(alphabet, list)
                or not alphabet
            ):
                raise DesignActionParentBindingError(
                    "prospective_position_catalog_row_invalid"
                )
            key = f"{chain}:{number}"
            existing = portfolio_site_context.get(key)
            prospective_row = {
                "parent_residue": str(row.get("parent_residue") or ""),
                "immutable_residue": str(row.get("immutable_residue") or ""),
                "owner_node_id": str(row.get("current_owner_node_id") or ""),
                "hard_allowed_residues": [str(item) for item in alphabet],
                "compiled_segment": compiled_segment,
                "prospective": True,
                "owner_assignment_source": "proposed_ast_selector",
            }
            if existing is not None:


                if (
                    existing["hard_allowed_residues"]
                    != prospective_row["hard_allowed_residues"]
                    or existing["parent_residue"]
                    != prospective_row["parent_residue"]
                ):
                    raise DesignActionParentBindingError(
                        "prospective_active_position_conflict", key
                    )
                existing["compiled_segment"] = compiled_segment
                existing["prospective"] = False
                existing["owner_assignment_source"] = "proposed_ast_selector"
            else:
                if (
                    len(prospective_row["parent_residue"]) != 1
                    or len(prospective_row["immutable_residue"]) != 1
                ):
                    raise DesignActionParentBindingError(
                        "prospective_position_catalog_row_invalid", key
                    )
                portfolio_site_context[key] = prospective_row
    execution_constraints = {
        "expected_mutation_count_range": tuple(raw_range),
        "expected_minimum_hamming_distance": minimum_hamming_distance,
        "require_sequence_reconciliation": bool(
            capabilities.get("required_sequence_reconciliation", False)
        ),
        "minimum_position_distribution_count": capabilities.get(
            "minimum_position_distribution_count", 0
        ),
        "minimum_soft_position_distribution_count": capabilities.get(
            "minimum_soft_position_distribution_count", 0
        ),
        "minimum_required_position_distribution_count": capabilities.get(
            "minimum_required_position_distribution_count", 0
        ),
        "max_positive_substitution_support_per_position": support_cap,


        "position_distribution_site_context": portfolio_site_context,
        "require_portfolio_capabilities": require_portfolio_capabilities,
        "minimum_atomic_module_count": step3.get(
            "minimum_atomic_module_count", 0
        ),
        "minimum_cross_node_atomic_module_count": step3.get(
            "minimum_cross_node_atomic_module_count", 0
        ),
        "minimum_sequence_seed_count": step3.get(
            "minimum_sequence_seed_count", 0
        ),
        "minimum_module_portfolio_count": step3.get(
            "minimum_module_portfolio_count", 0
        ),
        "minimum_strict_sequence_seed_portfolio_count": step3.get(
            "minimum_strict_sequence_seed_portfolio_count", 0
        ),
        "minimum_matched_ablation_portfolio_count": step3.get(
            "minimum_matched_ablation_portfolio_count", 0
        ),
        "portfolio_position_site_context": portfolio_site_context,
        "portfolio_sequence_context": sequence_context,
    }


    if require_frozen_candidate_wave:
        execution_constraints.update(
            {
                "portfolio_controller_mode": str(
                    step3.get("controller_mode") or "disabled"
                ),
                "require_frozen_candidate_wave": True,
                "step4_required_role_histogram": deepcopy(expected_histogram),
                "step4_required_portfolio_entry_shape": deepcopy(
                    step4.get("required_portfolio_entry_shape", {})
                ),
                "step4_free_slot_forced_tiers": deepcopy(
                    step4.get("free_slot_forced_tiers", {})
                ),
                "step4_require_live_node_change": bool(
                    step4.get("require_live_node_change", False)
                ),
                "step4_changed_node_min_generated_unique": int(
                    step4.get("changed_node_min_generated_unique", 0)
                ),
                "step4_changed_node_min_frozen_unique": int(
                    step4.get("changed_node_min_frozen_unique", 0)
                ),
                "step4_changed_node_min_protenix_attempts": int(
                    step4.get("changed_node_min_protenix_attempts", 0)
                ),
                "step4_current_ast_revision_plan": deepcopy(
                    context.get("current_ast_revision_plan", {})
                ),
            }
        )
    return execution_constraints


__all__ = [
    "DesignActionParentBindingError",
    "build_design_action_parent_binding",
    "build_structured_design_action_prompt_context",
    "design_action_execution_constraints",
    "design_action_parent_prompt_projection",
]
